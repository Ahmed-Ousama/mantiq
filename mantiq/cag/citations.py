import contextvars
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import torch
from transformers import AttentionInterface
from transformers.cache_utils import DynamicCache
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, AttentionMaskInterface


@dataclass
class Document:
    """One knowledge document and where its tokens sit in the cache."""
    name: Union[int, str]
    text: str
    start: int
    end: int


@dataclass
class Citation:
    """
    A document the answer draws on.
    score: how much less likely the answer becomes when the document is hidden, in nats per token.
    """
    document: Union[int, str]
    text: str
    score: float


@dataclass
class CitedAnswer:
    """
    An answer with the documents it draws on, most important first.
    supported: the model's own judgment, from 0 to 1, that the knowledge states the answer. Low
    means the answer doesn't come from the knowledge. It doesn't catch a wrong answer that some
    other part of the knowledge does state.
    confidence: from 0 to 1, how likely the answer is right, read from the model's internal state by
    its answer check (see KNOWN_ANSWER_CHECKS). It also catches wrong answers that the knowledge
    does state. None for models without an answer check.
    """
    answer: str
    citations: List[Citation] = field(default_factory=list)
    supported: float = 0.0
    confidence: Optional[float] = None


@dataclass
class CheckedAnswer:
    """
    An answer from generate_checked_response().
    confidence: the answer check's confidence in the first, fast answer (None without a check).
    rethought: whether that confidence was low and the answer was given again with thinking on.
    """
    answer: str
    confidence: Optional[float] = None
    rethought: bool = False


class KnowledgeCache(DynamicCache):
    """
    A DynamicCache that remembers where each knowledge document is, for citations, and the
    knowledge tokens and instructions it was built from, for updates.
    """

    def __init__(self, *args, documents: Optional[List[Document]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.documents = documents or []
        self.token_ids: Optional[List[int]] = None
        self.instructions: Optional[List[str]] = None


# Attention heads that point to the documents an answer comes from, found with
# NaiveCAG.calibrate_citations() on 40 HotpotQA paragraphs and the questions the model wrote about
# them, and checked on HotpotQA and SQuAD questions. Qwen3-8B, Mistral-7B and Llama-3.1-8B were
# calibrated in 4-bit, and Gemma-3-4B in float32 (it breaks in float16).
KNOWN_CITATION_HEADS = {
    "Qwen/Qwen3-0.6B": [(16, 14), (6, 7), (20, 14), (16, 15), (21, 8), (19, 4), (21, 9), (6, 6)],
    "Qwen/Qwen3-1.7B": [(16, 14), (11, 2), (20, 14), (6, 6), (19, 4), (6, 7), (16, 13), (17, 1)],
    "Qwen/Qwen3-4B": [(23, 10), (24, 5), (23, 5), (29, 11), (23, 30), (23, 4), (24, 13), (21, 10)],
    "Qwen/Qwen3-8B": [(23, 10), (23, 4), (29, 11), (23, 5), (24, 29), (24, 23), (20, 15), (19, 24)],
    "google/gemma-3-1b-it": [(17, 0), (17, 2), (11, 1), (5, 2), (5, 1), (17, 3), (11, 2), (11, 3)],
    "google/gemma-3-4b-it": [(17, 0), (17, 5), (17, 1), (29, 5), (23, 3), (23, 7), (23, 6), (23, 0)],
    "meta-llama/Llama-3.2-1B-Instruct": [(12, 2), (14, 15), (14, 14), (12, 0), (15, 20), (10, 1), (12, 13), (10, 23)],
    "meta-llama/Llama-3.2-3B-Instruct": [(15, 18), (17, 2), (15, 19), (23, 10), (15, 17), (16, 3), (15, 21), (24, 5)],
    "meta-llama/Llama-3.1-8B-Instruct": [(26, 15), (8, 11), (16, 20), (8, 1), (16, 27), (16, 26), (14, 19), (27, 6)],
    "microsoft/Phi-4-mini-instruct": [(19, 14), (21, 22), (21, 1), (19, 16), (17, 4), (19, 5), (16, 18), (19, 9)],
    "mistralai/Mistral-7B-Instruct-v0.3": [(14, 24), (17, 26), (18, 21), (19, 9), (15, 8), (18, 0), (19, 16), (15, 1)],
}

# Answer checks: a linear read-out of the model's internal state, averaged over the answer, that says
# how likely the answer is right. Trained on 2,000 HotpotQA questions without reference answers, to
# predict whether a better answer agrees: Qwen3-4B's own answer with thinking on (for Qwen3-4B and, as
# its own thinking teaches it little, for Qwen3-1.7B). Files in mantiq/cag/answer_checks/, with the
# confidence below which generate_checked_response() thinks again: it flags about as many training
# questions as the teacher disagrees on.
KNOWN_ANSWER_CHECKS = {
    "Qwen/Qwen3-4B": ("qwen3-4b.pt", 0.75),
    "Qwen/Qwen3-1.7B": ("qwen3-1.7b.pt", 0.58),
}


# sdpa attention that can also add up how much the answer attends to each document. It is
# registered as its own implementation and switched on only while citing.
COLLECTING_ATTENTION = "mantiq_collecting_sdpa"
_collector = contextvars.ContextVar("mantiq_attention_collector", default=None)


def _collecting_sdpa(module, query, key, value, attention_mask, **kwargs):
    collector = _collector.get()
    if collector is not None:
        collector.add(query, key, attention_mask, kwargs.get("scaling"))
    return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)


AttentionInterface.register(COLLECTING_ATTENTION, _collecting_sdpa)
AttentionMaskInterface.register(COLLECTING_ATTENTION, ALL_MASK_ATTENTION_FUNCTIONS["sdpa"])


class AttentionCollector:
    """
    Adds up, per layer and head, the attention from the positions that predict the answer
    tokens to each document, without keeping the attention weights.
    """

    # Answer positions scored at a time, to bound memory with long knowledge.
    _ROWS_PER_CHUNK = 16

    def __init__(self, documents: List[Document], knowledge_len: int, first_row: int):
        self.starts = torch.tensor([d.start for d in documents])
        self.ends = torch.tensor([d.end for d in documents])
        self.knowledge_len = knowledge_len
        self.first_row = first_row
        self.totals = torch.zeros(len(documents))
        self.layers: List[torch.Tensor] = []  # per layer: [heads, documents]

    def __enter__(self):
        self._token = _collector.set(self)
        return self

    def __exit__(self, *exc):
        _collector.reset(self._token)

    def add(self, query, key, attention_mask, scaling):
        batch, heads, q_len, head_dim = query.shape
        kv_heads, kv_len = key.shape[1], key.shape[2]
        groups = heads // kv_heads
        scaling = scaling or head_dim ** -0.5
        per_head = torch.zeros(heads, len(self.starts))
        # In float32: the products of half-precision queries and keys can overflow.
        key = key.float()
        # The answer is predicted from the second-to-last query position down to first_row.
        for begin in range(self.first_row, q_len - 1, self._ROWS_PER_CHUNK):
            end = min(begin + self._ROWS_PER_CHUNK, q_len - 1)
            # Group query heads by the key head they share, so the keys aren't copied per head.
            rows = query[:, :, begin:end].float().reshape(batch, kv_heads, groups * (end - begin), head_dim)
            scores = (rows @ key.transpose(-1, -2)) * scaling
            if attention_mask is None:
                positions = torch.arange(begin, end, device=query.device) + kv_len - q_len
                visible = torch.arange(kv_len, device=query.device) <= positions[:, None]
                scores = scores.masked_fill(~visible.repeat(groups, 1), float("-inf"))
            else:
                mask = attention_mask[:, :, begin:end, :kv_len].repeat(1, 1, groups, 1)
                scores = scores.masked_fill(~mask, float("-inf")) if mask.dtype == torch.bool else scores + mask
            weights = scores.softmax(dim=-1)[..., :self.knowledge_len]
            # Back to one row per query head, summed over the answer positions
            weights = weights.reshape(batch, kv_heads, groups, end - begin, -1).sum(dim=(0, 3)).reshape(heads, -1)
            cumulative = torch.cat([weights.new_zeros(heads, 1), weights.cumsum(-1)], dim=-1).cpu()
            per_head += cumulative[:, self.ends] - cumulative[:, self.starts]
        self.layers.append(per_head)
        self.totals += per_head.sum(0)

    def ranking(self, heads: Optional[List[Tuple[int, int]]] = None) -> List[int]:
        """
        Document indices, most attended first. With heads, (layer, head) pairs, only those heads
        count, each by its share of attention over the documents.
        """
        if not heads:
            return self.totals.argsort(descending=True).tolist()
        per_head = torch.stack(self.layers)
        shares = per_head / per_head.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return sum(shares[layer, head] for layer, head in heads).argsort(descending=True).tolist()
