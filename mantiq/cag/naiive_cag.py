import bisect
import os
import pickle
import time
import warnings
from collections.abc import Mapping
from typing import Iterator, Optional

import torch
from jinja2.exceptions import TemplateError
from transformers.cache_utils import DynamicCache
from ..cag.utils import load_model, GenericDatasetProcessor, DatasetParser
from .citations import (
    COLLECTING_ATTENTION,
    KNOWN_CITATION_HEADS,
    AttentionCollector,
    Citation,
    CitedAnswer,
    Document,
    KnowledgeCache,
)

KV_CACHE_FILENAME = "kv_cache.pt"
_CACHE_FORMAT = "mantiq.kv_cache.v1"
# Marks where the question goes when the rendered chat prompt is split into the
# cached knowledge prefix and the per-query suffix.
_QUESTION_PLACEHOLDER = "<<mantiq:question>>"
# Between documents in the knowledge.
_DOCUMENT_SEPARATOR = "\n\n"


class NaiveCAG:
    # Knowledge tokens read per forward pass while preparing a cache. Lower it if that runs out of memory.
    prefill_chunk_size = 1024

    def __init__(
        self,
        model_name: str,
        hf_token: Optional[str] = None,
        quantized: Optional[bool] = None,
        trust_remote_code: bool = False,
        chat_template_kwargs: Optional[dict] = None,
        model_kwargs: Optional[dict] = None,
        citation_heads: Optional[list] = None,
    ):
        """
        Initialize NaiveCAG.
        Args:
            model_name: HuggingFace model name.
            hf_token: HuggingFace API token, needed for gated models. Defaults to the
                HF_TOKEN environment variable or a cached `hf auth login`.
            quantized: Whether to load the model in 4-bit. Defaults to True when a CUDA GPU
                and bitsandbytes are available, otherwise False.
            trust_remote_code: Whether to run custom modeling code from the model repo.
                Only enable it for repos you trust.
            chat_template_kwargs: Extra variables for the tokenizer's chat template,
                e.g. {"enable_thinking": False} for Qwen3.
            model_kwargs: Extra arguments for AutoModelForCausalLM.from_pretrained,
                e.g. {"dtype": torch.bfloat16} or {"revision": "main"}.
            citation_heads: (layer, head) pairs from an earlier calibrate_citations() with this model.
                Defaults to known heads for some models (KNOWN_CITATION_HEADS); [] turns them off.
        """
        self.model_name = model_name
        self.model, self.tokenizer = load_model(model_name, hf_token, quantized, trust_remote_code, model_kwargs)
        self.processor = GenericDatasetProcessor()  # Initialize processor
        self.prompt_instruction = None
        self.chat_template_kwargs = chat_template_kwargs or {}
        if citation_heads is None:
            citation_heads = KNOWN_CITATION_HEADS.get(model_name)
        self.citation_heads = [tuple(head) for head in citation_heads] if citation_heads else None
        self._yes_no_ids = None  # token ids that start "yes" and "no", for the supported check
        # The text after the question doesn't depend on the knowledge, so a cache
        # built in another session can be queried without its instructions.
        _, self._question_suffix = self._render_prompt("", "", "")

    def prepare_kvcache(
        self,
        dataset,
        saving_path: Optional[str] = None,
        prompt_instruction: Optional[str] = None,
        answer_instruction: Optional[str] = None,
    ):
        """
        Prepare the KV cache from a question/answer dataset.
        Args:
            dataset: List of dicts with "question", "answer" and optional "context" keys.
            saving_path: Directory to save kv_cache.pt in (created if missing), or None to skip saving.
            prompt_instruction: Instruction for the system. Defaults to
                "You are an assistant for giving short answers."
            answer_instruction: Instruction for the assistant. Defaults to
                "Provide a concise and relevant answer."
        Returns:
            Tuple[KnowledgeCache, float]: KV cache and preparation time.
        """
        # Load and parse the dataset
        dataset = self.processor.parse_dataset(dataset, parser_function=DatasetParser.default_parser)

        # Each entry is a document, cited by its index
        documents = [
            f"Question: {entry['question']}\nAnswer: {entry['answer']}" +
            (f"\nContext: {entry['context']}" if entry['context'] else "") +
            "\n"
            for entry in dataset
        ]
        return self._prepare(list(enumerate(documents)), saving_path, prompt_instruction, answer_instruction)

    def prepare_kvcache_from_loader(
        self,
        loaded_data,
        saving_path: Optional[str] = None,
        prompt_instruction: Optional[str] = None,
        answer_instruction: Optional[str] = None,
    ):
        """
        Prepare and save a key-value (KV) cache using the provided loaded data and instructions.

        This function generates a KV cache based on the provided `loaded_data`, formats it using
        `prompt_instruction` and optional `answer_instruction`, and saves the result to a specified path.

        Args:
            loaded_data (str, list or dict): The knowledge text, e.g. from a loader. Pass a list of
            texts, or a dict of names to texts, to keep them as separate documents that cite() can
            point to.
            saving_path (str, optional): The directory where the KV cache will be saved (created if missing).
            Defaults to None, which skips saving.
            prompt_instruction (str, optional): Instruction or context for the system to format the prompt.
            Defaults to "You are an assistant for giving short answers."
            answer_instruction (str, optional): Instruction for generating concise responses. Defaults to
            "Provide a concise and relevant answer."
        Returns:
            Tuple[KnowledgeCache, float]:
            - `KnowledgeCache`: The prepared KV cache object, a DynamicCache that also knows its documents.
            - `float`: The time taken to prepare the KV cache, in seconds.
        """
        return self._prepare(_as_documents(loaded_data), saving_path, prompt_instruction, answer_instruction)

    def update_kvcache(self, past_key_values: KnowledgeCache, loaded_data, saving_path: Optional[str] = None):
        """
        Update a cache to new knowledge, re-reading it only from the first change onward.
        The result is the same as preparing a new cache with the same instructions, but the
        unchanged beginning is reused: put the documents that change most often last.
        Args:
            past_key_values: A cache from prepare_kvcache_from_loader or read_kv_cache.
            loaded_data: The new knowledge, in any form prepare_kvcache_from_loader takes.
            saving_path: Directory to save the updated kv_cache.pt in, or None to skip saving.
        Returns:
            Tuple[KnowledgeCache, float]: The same cache, updated in place, and the update time.
        """
        if getattr(past_key_values, "token_ids", None) is None or getattr(past_key_values, "instructions", None) is None:
            raise ValueError("This cache can't be updated. Build it with prepare_kvcache_from_loader.")
        documents = _as_documents(loaded_data)
        context = _DOCUMENT_SEPARATOR.join(text for _, text in documents)
        prompt, _ = self._render_prompt(context, *past_key_values.instructions)
        t1 = time.time()
        input_ids = self._encode_knowledge(prompt)
        old_ids, new_ids = past_key_values.token_ids, input_ids[0].tolist()
        same = 0
        while same < min(len(old_ids), len(new_ids)) and old_ids[same] == new_ids[same]:
            same += 1
        _crop_to(past_key_values, same)
        try:
            self._extend(past_key_values, input_ids[:, same:])
        except BaseException:
            # Keep a consistent cache that holds the unchanged beginning, so that updating again
            # picks up from there. Its documents can't be cited until an update succeeds.
            _crop_to(past_key_values, same)
            past_key_values.token_ids = new_ids[:same]
            past_key_values.documents = []
            raise
        past_key_values.token_ids = new_ids
        past_key_values.documents = self._locate_documents(prompt, context, documents)

        if saving_path is not None:
            self.write_kv_cache(past_key_values, os.path.join(saving_path, KV_CACHE_FILENAME))
        t2 = time.time()
        return past_key_values, t2 - t1

    def _prepare(self, documents: list, saving_path, prompt_instruction: str, answer_instruction: str):
        """documents: (name, text) pairs, joined by blank lines into the knowledge."""
        self.prompt_instruction = prompt_instruction or "You are an assistant for giving short answers."
        answer_instruction = answer_instruction or "Provide a concise and relevant answer."

        context = _DOCUMENT_SEPARATOR.join(text for _, text in documents)
        prompt, _ = self._render_prompt(context, self.prompt_instruction, answer_instruction)
        t1 = time.time()
        kv = self.preprocess_knowledge(prompt)
        kv.documents = self._locate_documents(prompt, context, documents)
        kv.instructions = [self.prompt_instruction, answer_instruction]

        if saving_path is not None:
            self.write_kv_cache(kv, os.path.join(saving_path, KV_CACHE_FILENAME))
        t2 = time.time()
        return kv, t2 - t1

    def _locate_documents(self, prompt: str, context: str, documents: list) -> list:
        """Find each document's token span in the knowledge prompt. Empty if they can't be found."""
        offset = prompt.find(context)
        if offset < 0 or not self.tokenizer.is_fast:  # only fast tokenizers report character offsets
            return []
        encoding = self.tokenizer(
            prompt, add_special_tokens=self.tokenizer.chat_template is None, return_offsets_mapping=True
        )
        token_starts = [start for start, _ in encoding["offset_mapping"]]
        located = []
        for name, text in documents:
            start = bisect.bisect_left(token_starts, offset)
            end = bisect.bisect_left(token_starts, offset + len(text))
            located.append(Document(name, text, start, end))
            offset += len(text) + len(_DOCUMENT_SEPARATOR)
        return located

    def _render_prompt(self, context: str, prompt_instruction: str, answer_instruction: str):
        """
        Render the knowledge prompt with the model's chat template.
        Returns (prefix, suffix): the prefix ends where the question goes and is what gets
        cached; the suffix closes the user turn and opens the assistant turn.
        """
        user_content = (
            "Context:\n"
            "------------------------------------------------\n"
            f"{context}\n"
            "------------------------------------------------\n\n"
            f"{answer_instruction}\n\n"
            f"Question:\n{_QUESTION_PLACEHOLDER}"
        )
        prompt = self._apply_chat_template(prompt_instruction, user_content)
        prefix, suffix = prompt.rsplit(_QUESTION_PLACEHOLDER, 1)
        return prefix, suffix

    def _apply_chat_template(self, system: str, user: str) -> str:
        if self.tokenizer.chat_template is None:
            return f"{system}\n\n{user}\n\nAnswer:\n"
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **self.chat_template_kwargs
            )
        except TemplateError:
            # Some templates (e.g. Gemma) reject the system role; fold it into the user turn.
            messages = [{"role": "user", "content": f"{system}\n\n{user}"}]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **self.chat_template_kwargs
            )

    def preprocess_knowledge(self, prompt: str) -> KnowledgeCache:
        """
        Preprocess knowledge and return KV cache.
        """
        input_ids = self._encode_knowledge(prompt)
        past_key_values = KnowledgeCache()
        self._extend(past_key_values, input_ids)
        past_key_values.token_ids = input_ids[0].tolist()
        return past_key_values

    def _encode_knowledge(self, prompt: str) -> torch.Tensor:
        input_ids = self.tokenizer.encode(
            prompt,
            add_special_tokens=self.tokenizer.chat_template is None,  # chat templates already add BOS
            return_tensors="pt",
        ).to(self._input_device())
        max_length = getattr(self.model.config, "max_position_embeddings", None)
        if max_length and input_ids.shape[-1] >= max_length:
            raise ValueError(
                f"The knowledge prompt is {input_ids.shape[-1]} tokens, but {self.model_name} has a "
                f"{max_length}-token context window. Use less knowledge or a model with a longer context."
            )
        return input_ids

    def _extend(self, past_key_values: KnowledgeCache, input_ids: torch.Tensor):
        """Run the knowledge tokens through the model in chunks, appending them to the cache."""
        with torch.no_grad():
            # Chunks bound the attention memory where it is computed in full (on CPU, and on GPUs
            # without flash attention): chunk x knowledge instead of knowledge x knowledge.
            for start in range(0, input_ids.shape[-1], self.prefill_chunk_size):
                chunk = input_ids[:, start:start + self.prefill_chunk_size]
                filled = past_key_values.get_seq_length()
                # Run the model without its output layer: only the cache is needed, and logits for
                # every knowledge token (tokens x vocabulary size) can take more memory than the model.
                self.model.base_model(
                    input_ids=chunk,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_attentions=False,
                    output_hidden_states=False
                )
                if past_key_values.get_seq_length() != filled + chunk.shape[-1]:
                    raise ValueError(f"{self.model_name} doesn't fill the cache it is given, so mantiq can't cache its knowledge.")

    def write_kv_cache(self, kv: DynamicCache, path: str):
        """
        Save the KV cache as plain tensors, so it loads with torch.load(weights_only=True).
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        token_ids = getattr(kv, "token_ids", None)
        torch.save(
            {
                "format": _CACHE_FORMAT,
                "model_name": self.model_name,
                "layers": _cache_to_layers(kv),
                "documents": [vars(document) for document in getattr(kv, "documents", [])],
                "token_ids": None if token_ids is None else torch.tensor(token_ids),
                "instructions": getattr(kv, "instructions", None),
            },
            path,
        )

    def read_kv_cache(self, path: str) -> KnowledgeCache:
        """
        Load a KV cache saved by write_kv_cache.
        """
        try:
            data = torch.load(path, map_location=self._cache_map_location(), weights_only=True)
        except pickle.UnpicklingError as e:
            raise ValueError(
                f"{path} is not a mantiq KV cache file. Caches saved by mantiq 0.1 must be rebuilt with prepare_kvcache."
            ) from e
        if not isinstance(data, dict) or data.get("format") != _CACHE_FORMAT:
            raise ValueError(f"{path} is not a mantiq KV cache file.")
        if data["model_name"] != self.model_name:
            warnings.warn(f"{path} was built with {data['model_name']}, but this instance uses {self.model_name}.")
        cache = _layers_to_cache(data["layers"])
        cache.documents = [Document(**document) for document in data.get("documents", [])]
        if data.get("token_ids") is not None:
            cache.token_ids = data["token_ids"].tolist()
        cache.instructions = data.get("instructions")
        return cache

    def generate(
        self,
        input_query: str,
        past_key_values: DynamicCache,
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate a response using the KV cache and return the generated token ids.
        The cache is cropped back to its original length afterwards, so it can be reused for the next query.
        """
        token_ids = list(self._generate_tokens(input_query, past_key_values, max_new_tokens, temperature, top_p))
        return torch.tensor([token_ids], dtype=torch.long, device=self._input_device())

    def generate_response(
        self, query: str, past_key_values, max_new_tokens: int = 100, temperature: float = 0.0, top_p: float = 1.0
    ) -> str:
        """
        Generate a response to the query.
        temperature=0 picks the most likely token at each step; a higher temperature samples,
        keeping only the most likely tokens whose probabilities add up to top_p.
        """
        output = self.generate(query, past_key_values, max_new_tokens, temperature, top_p)
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return response

    def stream_response(
        self, query: str, past_key_values, max_new_tokens: int = 100, temperature: float = 0.0, top_p: float = 1.0
    ) -> Iterator[str]:
        """
        Like generate_response, but yields the response in pieces as it is generated.
        The cache is restored when the stream ends or is closed, so finish or close it
        before sending the next query.
        """
        token_ids = []
        sent = ""
        for token_id in self._generate_tokens(query, past_key_values, max_new_tokens, temperature, top_p):
            token_ids.append(token_id)
            text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            # Hold back a multi-byte character until all of its tokens have arrived.
            if len(text) > len(sent) and not text.endswith("�"):
                yield text[len(sent):]
                sent = text
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        if len(text) > len(sent):
            yield text[len(sent):]

    @torch.no_grad()
    def _generate_tokens(
        self, input_query: str, past_key_values: DynamicCache, max_new_tokens: int, temperature: float, top_p: float
    ) -> Iterator[int]:
        device = self._input_device()
        input_ids = self.tokenizer.encode(
            input_query.strip() + self._question_suffix, add_special_tokens=False, return_tensors="pt"
        ).to(device)
        eos_token_ids = self._eos_token_ids()
        origin_len = past_key_values.get_seq_length()

        next_token = input_ids

        try:
            for _ in range(max_new_tokens):
                outputs = self.model(
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                next_token = _next_token(outputs.logits[:, -1, :], temperature, top_p).to(device)
                token_id = next_token.item()
                yield token_id

                if token_id in eos_token_ids:
                    break
        finally:
            # Drop the query and answer so the next query only sees the knowledge.
            _crop_to(past_key_values, origin_len)

    def cite(
        self, query: str, response: str, past_key_values: KnowledgeCache, top_k: int = 3, max_candidates: int = 16
    ) -> CitedAnswer:
        """
        Find the knowledge documents a response to the query is based on.
        After calibrate_citations(), the attention heads it found rank the documents in one pass,
        however many there are. Otherwise each document is hidden from the cache in turn, and the
        ones whose hiding makes the response least likely come first; with more than max_candidates
        documents, only the ones the response attends to most are hidden.
        A citation's score is how much less likely the response becomes with that document hidden,
        in nats per token. Nothing is re-encoded.
        The answer's supported score is the model's probability that the knowledge states it,
        asked after the response; it works best with models that answer without thinking first.
        """
        documents = getattr(past_key_values, "documents", None)
        if not documents:
            raise ValueError(
                "This cache doesn't know where its documents are. Build it with prepare_kvcache or "
                "prepare_kvcache_from_loader, with a model that has a fast tokenizer."
            )
        query_ids = self.tokenizer.encode(query.strip() + self._question_suffix, add_special_tokens=False)
        answer_ids = self.tokenizer.encode(response, add_special_tokens=False)
        if not answer_ids:
            return CitedAnswer(response)

        # Plain attention is a rough guide (it gets less reliable in larger models), so without
        # calibrated heads it only narrows down which documents to hide when there are too many.
        by_heads = bool(self.citation_heads)
        base, attention = self._attend(query_ids, answer_ids, past_key_values, by_heads or len(documents) > max_candidates)
        if attention is not None and by_heads:
            candidates = attention.ranking(self.citation_heads)[:top_k]
        elif attention is not None:
            candidates = attention.ranking()[:max_candidates]
        else:
            candidates = range(len(documents))

        def drop(hidden):
            return (base - self._answer_logprob(query_ids, answer_ids, past_key_values, hidden)) / len(answer_ids)

        citations = [Citation(documents[i].name, documents[i].text, drop([documents[i]])) for i in candidates]
        if attention is None or not by_heads:
            citations = sorted((c for c in citations if c.score > 0), key=lambda c: c.score, reverse=True)
        return CitedAnswer(response, citations[:top_k], self._supported(query, response, past_key_values))

    @torch.no_grad()
    def _supported(self, query: str, response: str, past_key_values: KnowledgeCache) -> float:
        """The model's probability that the knowledge states the response, asked as a follow-up question."""
        check = (
            f"{query.strip()}\n\nProposed answer: {response.strip()}\n\n"
            "Is the proposed answer stated in the context? Reply with yes or no."
        )
        input_ids = self.tokenizer.encode(check + self._question_suffix, add_special_tokens=False, return_tensors="pt")
        origin_len = past_key_values.get_seq_length()
        try:
            logits = self.model(input_ids=input_ids.to(self._input_device()), past_key_values=past_key_values,
                                use_cache=True).logits[0, -1]
        finally:
            _crop_to(past_key_values, origin_len)
        probs = logits.float().softmax(dim=-1)
        if self._yes_no_ids is None:
            yes = _first_word_tokens(self.tokenizer, ("yes", "Yes", " yes", " Yes"))
            no = _first_word_tokens(self.tokenizer, ("no", "No", " no", " No"))
            self._yes_no_ids = (sorted(yes - no), sorted(no - yes))
        yes, no = (probs[ids].sum() for ids in self._yes_no_ids)
        return (yes / (yes + no).clamp_min(1e-12)).item()

    def calibrate_citations(
        self,
        past_key_values: KnowledgeCache,
        questions: Optional[list] = None,
        num_heads: int = 8,
        num_questions: int = 20,
        max_new_tokens: int = 48,
    ) -> list:
        """
        Find the attention heads that point to the documents an answer comes from, so that cite()
        ranks documents in one pass, and more accurately. For each question the model answers from
        the cache and every document is hidden in turn; the heads that most often attend most to the
        document whose hiding hurts the answer most are kept. No correct answers are needed.
        Without questions, the model writes one about each of num_questions documents of the cache.
        Returns the heads as (layer, head) pairs and uses them from then on. Pass them as
        citation_heads to reuse them with this model later.
        """
        documents = getattr(past_key_values, "documents", None)
        if not documents or len(documents) < 2:
            raise ValueError("Calibrating needs a cache with at least 2 documents.")
        if questions is None:
            step = max(1, len(documents) // num_questions)
            questions = [self._write_question(d.text) for d in documents[::step][:num_questions]]
        agreement = shares = None
        count = 0
        for question in questions:
            response = self.generate_response(question, past_key_values, max_new_tokens=max_new_tokens)
            query_ids = self.tokenizer.encode(question.strip() + self._question_suffix, add_special_tokens=False)
            answer_ids = self.tokenizer.encode(response, add_special_tokens=False)
            if not answer_ids:
                continue
            base, attention = self._attend(query_ids, answer_ids, past_key_values, collect=True)
            if attention is None:
                raise ValueError(f"{self.model_name} doesn't support the pluggable attention that calibrating needs.")
            drops = torch.tensor([base - self._answer_logprob(query_ids, answer_ids, past_key_values, [d]) for d in documents])
            per_head = torch.stack(attention.layers)  # [layers, heads, documents]
            top = drops.argmax()
            hits = (per_head.argmax(dim=-1) == top).float()
            share = per_head[..., top] / per_head.sum(dim=-1).clamp_min(1e-9)
            agreement, shares = (hits, share) if agreement is None else (agreement + hits, shares + share)
            count += 1
        if agreement is None:
            raise ValueError("None of the questions got an answer to calibrate with.")
        # Agreement counts first; the attention share on the right document breaks ties, which are
        # common with few questions.
        score = agreement + shares / (count + 1)
        heads_per_layer = score.shape[1]
        best = score.flatten().topk(min(num_heads, score.numel())).indices.tolist()
        self.citation_heads = [(i // heads_per_layer, i % heads_per_layer) for i in best]
        return self.citation_heads

    @torch.no_grad()
    def _write_question(self, text: str, max_chars: int = 2000) -> str:
        """A question that the text answers, written by the model without the knowledge cache."""
        prompt = self._apply_chat_template(
            "You write questions.",
            f"Write one short question that the following text answers. Reply with the question only.\n\n{text[:max_chars]}",
        )
        input_ids = self.tokenizer.encode(
            prompt, add_special_tokens=self.tokenizer.chat_template is None, return_tensors="pt"
        ).to(self._input_device())
        output = self.model.generate(
            input_ids, max_new_tokens=40, do_sample=False, pad_token_id=self.tokenizer.eos_token_id
        )
        question = self.tokenizer.decode(output[0, input_ids.shape[-1]:], skip_special_tokens=True).strip()
        return question.split("\n")[0] or text[:200]

    def _attend(self, query_ids: list, answer_ids: list, past_key_values: KnowledgeCache, collect: bool):
        """
        The answer's log-probability and, if collect is set and the model allows it, an
        AttentionCollector with how much the answer attends to each document (else None).
        """
        previous = self.model.config._attn_implementation
        if collect:
            try:
                self.model.set_attn_implementation(COLLECTING_ATTENTION)
            except (ValueError, NotImplementedError):  # a model without pluggable attention
                collect = False
        try:
            with AttentionCollector(past_key_values.documents, past_key_values.get_seq_length(), len(query_ids) - 1) as attention:
                base = self._answer_logprob(query_ids, answer_ids, past_key_values)
        finally:
            if collect:
                self.model.set_attn_implementation(previous)
        if collect and previous != "sdpa":
            # Measure the answer with the model's own attention too, like the hidden passes that follow.
            base = self._answer_logprob(query_ids, answer_ids, past_key_values)
        return base, attention if collect else None

    @torch.no_grad()
    def _answer_logprob(self, query_ids: list, answer_ids: list, past_key_values: DynamicCache, hidden=()) -> float:
        """Log-probability of the answer after the query, with the hidden documents masked out of the cache."""
        origin_len = past_key_values.get_seq_length()
        device = self._input_device()
        input_ids = torch.tensor([query_ids + answer_ids], device=device)
        attention_mask = torch.ones(1, origin_len + input_ids.shape[1], dtype=torch.long, device=device)
        for document in hidden:
            attention_mask[0, document.start:document.end] = 0
        try:
            logits = self.model(
                input_ids=input_ids, past_key_values=past_key_values, attention_mask=attention_mask, use_cache=True
            ).logits
        finally:
            _crop_to(past_key_values, origin_len)
        logprobs = logits[0, len(query_ids) - 1:-1].float().log_softmax(dim=-1)
        return logprobs.gather(-1, input_ids[0, len(query_ids):, None]).sum().item()

    def _input_device(self) -> torch.device:
        return self.model.get_input_embeddings().weight.device

    def _cache_map_location(self):
        # A model sharded across devices keeps each layer's cache on the device it was saved from.
        if len(set((getattr(self.model, "hf_device_map", None) or {}).values())) > 1:
            return None
        return self._input_device()

    def _eos_token_ids(self) -> set:
        eos = self.model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else list(eos or [])
        if self.tokenizer.eos_token_id is not None:
            eos.append(self.tokenizer.eos_token_id)
        return set(eos)


# The original, misspelled name, kept so existing code keeps working.
NaiiveCAG = NaiveCAG


def _next_token(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_ids = probs.sort(dim=-1, descending=True)
        # Keep the smallest set of most likely tokens whose probabilities add up to top_p.
        sorted_probs[sorted_probs.cumsum(dim=-1) - sorted_probs >= top_p] = 0
        probs = torch.zeros_like(probs).scatter(-1, sorted_ids, sorted_probs)
    return torch.multinomial(probs, num_samples=1)


def _as_documents(loaded_data) -> list:
    """(name, text) pairs from a string, a list of strings or a dict of names to strings."""
    if isinstance(loaded_data, str):
        return [(0, loaded_data)]
    if isinstance(loaded_data, Mapping):
        # Saved caches load with weights_only=True, which only allows plain names.
        return [(name if isinstance(name, (int, str)) else str(name), text) for name, text in loaded_data.items()]
    return list(enumerate(loaded_data))


def _first_word_tokens(tokenizer, words) -> set:
    """
    The first token of each word that has letters in it, skipping pieces such as a lone leading
    space, which "yes" and "no" can share.
    """
    ids = set()
    for word in words:
        for token_id in tokenizer.encode(word, add_special_tokens=False):
            if tokenizer.decode([token_id]).strip():
                ids.add(token_id)
                break
    return ids


def _crop_to(cache: DynamicCache, length: int):
    # A negative crop removes that many tokens on every supported transformers version.
    tokens_to_remove = cache.get_seq_length() - length
    if tokens_to_remove > 0:
        cache.crop(-tokens_to_remove)


def _cache_to_layers(kv: DynamicCache) -> list:
    return [[layer.keys, layer.values] for layer in kv.layers]


def _layers_to_cache(layers: list) -> KnowledgeCache:
    cache = KnowledgeCache()
    for layer_idx, (keys, values) in enumerate(layers):
        cache.update(keys, values, layer_idx)
    return cache
