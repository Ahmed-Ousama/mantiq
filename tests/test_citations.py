import pytest
import torch
from transformers.cache_utils import DynamicCache

from mantiq.cag import CitedAnswer
from mantiq.cag.citations import COLLECTING_ATTENTION, AttentionCollector

DATASET = [
    {"question": "What are the features of this TV?", "answer": "This TV has 4K resolution and smart features.", "context": "55-inch TV."},
    {"question": "Does this refrigerator save energy?", "answer": "Yes, it has an A++ energy rating.", "context": None},
    {"question": "What makes this washing machine special?", "answer": "It has AI-powered wash cycles.", "context": None},
]
INSTRUCTION = "You are a sales agent."
QUESTION = "Does this refrigerator save energy?"
ANSWER = "Yes, it has an A++ energy rating."


def prepare(cag, data=DATASET):
    calls = []
    hook = cag.model.get_input_embeddings().register_forward_pre_hook(lambda module, args: calls.append(args[0]))
    kv, _ = cag.prepare_kvcache(data, None, INSTRUCTION) if data is DATASET else cag.prepare_kvcache_from_loader(data)
    hook.remove()
    return kv, calls[0][0].tolist()


def test_documents_are_located_in_the_cache(make_cag):
    cag = make_cag()
    kv, prompt_ids = prepare(cag)

    assert [d.name for d in kv.documents] == [0, 1, 2]
    for i, document in enumerate(kv.documents):
        span = cag.tokenizer.decode(prompt_ids[document.start:document.end])
        assert DATASET[i]["question"] in span
        assert all(other["question"] not in span for other in DATASET if other is not DATASET[i])


@pytest.mark.parametrize("data, names", [(["The TV is 4K.", "The fridge is A++."], [0, 1]),
                                         ({"tv.md": "The TV is 4K.", "fridge.md": "The fridge is A++."}, ["tv.md", "fridge.md"])])
def test_documents_from_a_list_or_dict(make_cag, data, names):
    cag = make_cag()
    kv, prompt_ids = prepare(cag, data)

    assert [d.name for d in kv.documents] == names
    for document, text in zip(kv.documents, data.values() if isinstance(data, dict) else data):
        assert cag.tokenizer.decode(prompt_ids[document.start:document.end]).strip() == text


def test_cite_returns_ranked_citations_and_restores_the_cache(make_cag):
    cag = make_cag()
    kv, _ = cag.prepare_kvcache(DATASET, None, INSTRUCTION)
    length = kv.get_seq_length()

    cited = cag.cite(QUESTION, ANSWER, kv, top_k=2)

    assert isinstance(cited, CitedAnswer) and cited.answer == ANSWER
    assert len(cited.citations) <= 2
    assert all(c.document in (0, 1, 2) and c.score > 0 for c in cited.citations)
    assert [c.score for c in cited.citations] == sorted((c.score for c in cited.citations), reverse=True)
    assert 0 <= cited.supported <= 1
    assert kv.get_seq_length() == length
    assert cag.model.config._attn_implementation == "sdpa"


def test_attention_collector_matches_eager_attention(make_cag):
    cag = make_cag()
    kv, _ = cag.prepare_kvcache(DATASET, None, INSTRUCTION)
    length = kv.get_seq_length()
    query_ids = cag.tokenizer.encode(QUESTION + cag._question_suffix, add_special_tokens=False)
    answer_ids = cag.tokenizer.encode(ANSWER, add_special_tokens=False)

    cag.model.set_attn_implementation(COLLECTING_ATTENTION)
    with AttentionCollector(kv.documents, length, len(query_ids) - 1) as collector:
        cag._answer_logprob(query_ids, answer_ids, kv)

    # The same totals from the full attention weights of eager attention
    cag.model.set_attn_implementation("eager")
    with torch.no_grad():
        out = cag.model(input_ids=torch.tensor([query_ids + answer_ids]), past_key_values=kv, output_attentions=True)
    kv.crop(-(len(query_ids) + len(answer_ids)))
    rows = slice(len(query_ids) - 1, len(query_ids) + len(answer_ids) - 1)
    weights = sum(layer[0, :, rows, :length].sum(dim=(0, 1)) for layer in out.attentions)
    expected = torch.tensor([weights[d.start:d.end].sum().item() for d in kv.documents])
    per_head = torch.stack([
        torch.stack([layer[0, :, rows, d.start:d.end].sum(dim=(1, 2)) for d in kv.documents], dim=-1)
        for layer in out.attentions
    ])

    torch.testing.assert_close(collector.totals, expected, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(torch.stack(collector.layers), per_head, rtol=1e-4, atol=1e-5)


def test_attention_collector_without_a_mask_is_causal(make_cag):
    cag = make_cag()
    kv, _ = cag.prepare_kvcache(DATASET, None, INSTRUCTION)
    torch.manual_seed(0)
    query, key = torch.randn(1, 4, 6, 8), torch.randn(1, 2, kv.get_seq_length() + 6, 8)
    kv_len, q_len = key.shape[2], query.shape[2]
    causal = (torch.arange(kv_len) <= torch.arange(q_len)[:, None] + kv_len - q_len)[None, None]

    without, explicit = (AttentionCollector(kv.documents, kv.get_seq_length(), 2) for _ in range(2))
    without.add(query, key, None, None)
    explicit.add(query, key, causal, None)

    torch.testing.assert_close(without.totals, explicit.totals)


def spy_on_attention_switches(cag, monkeypatch, refuse=False):
    switches = []
    set_attn_implementation = cag.model.set_attn_implementation

    def switch(name):
        switches.append(name)
        if refuse and name == COLLECTING_ATTENTION:
            raise ValueError("not supported")
        set_attn_implementation(name)

    monkeypatch.setattr(cag.model, "set_attn_implementation", switch)
    return switches


def test_cite_hides_every_document_when_there_are_few(make_cag, monkeypatch):
    cag = make_cag()
    kv, _ = cag.prepare_kvcache(DATASET, None, INSTRUCTION)
    switches = spy_on_attention_switches(cag, monkeypatch)

    cag.cite(QUESTION, ANSWER, kv)

    assert switches == []  # attention isn't needed to pick candidates


def test_cite_uses_attention_to_narrow_down_many_documents(make_cag, monkeypatch):
    cag = make_cag()
    kv, _ = cag.prepare_kvcache(DATASET, None, INSTRUCTION)
    switches = spy_on_attention_switches(cag, monkeypatch)

    cited = cag.cite(QUESTION, ANSWER, kv, max_candidates=1)

    assert switches == [COLLECTING_ATTENTION, "sdpa"]
    assert len(cited.citations) <= 1


def test_cite_works_without_pluggable_attention(make_cag, monkeypatch):
    cag = make_cag()
    kv, _ = cag.prepare_kvcache(DATASET, None, INSTRUCTION)
    spy_on_attention_switches(cag, monkeypatch, refuse=True)

    cited = cag.cite(QUESTION, ANSWER, kv, top_k=3, max_candidates=1)

    assert len(cited.citations) > 1  # every document was hidden instead
    assert all(c.document in (0, 1, 2) for c in cited.citations)
    assert cag.model.config._attn_implementation == "sdpa"


def test_documents_survive_saving_and_loading(make_cag, tmp_path):
    cag = make_cag()
    kv, _ = cag.prepare_kvcache(DATASET, str(tmp_path), INSTRUCTION)

    loaded = cag.read_kv_cache(str(tmp_path / "kv_cache.pt"))

    assert loaded.documents == kv.documents
    assert cag.cite(QUESTION, ANSWER, loaded) == cag.cite(QUESTION, ANSWER, kv)


def test_cite_needs_document_boundaries(make_cag):
    cag = make_cag()
    plain = DynamicCache()

    with pytest.raises(ValueError, match="where its documents are"):
        cag.cite(QUESTION, ANSWER, plain)



def count_passes(model):
    passes = []
    model.get_input_embeddings().register_forward_hook(lambda module, args, output: passes.append(1))
    return passes


def test_calibrate_citations_picks_heads(make_cag):
    cag = make_cag()
    kv, _ = cag.prepare_kvcache(DATASET, None, INSTRUCTION)
    length = kv.get_seq_length()

    heads = cag.calibrate_citations(kv, [QUESTION, "What makes this washing machine special?"], num_heads=3, max_new_tokens=4)

    layers, per_layer = cag.model.config.num_hidden_layers, cag.model.config.num_attention_heads
    assert len(set(heads)) == 3 and all(0 <= l < layers and 0 <= h < per_layer for l, h in heads)
    assert cag.citation_heads == heads
    assert kv.get_seq_length() == length


def test_cite_with_heads_ranks_by_them_in_a_few_passes(make_cag):
    cag = make_cag(citation_heads=[[0, 1], [1, 3]])
    kv, _ = cag.prepare_kvcache(DATASET, None, INSTRUCTION)
    query_ids = cag.tokenizer.encode(QUESTION + cag._question_suffix, add_special_tokens=False)
    answer_ids = cag.tokenizer.encode(ANSWER, add_special_tokens=False)
    _, attention = cag._attend(query_ids, answer_ids, kv, collect=True)
    passes = count_passes(cag.model)

    cited = cag.cite(QUESTION, ANSWER, kv, top_k=2)

    assert cag.citation_heads == [(0, 1), (1, 3)]
    assert len(passes) == 4  # one to rank, one per citation, one for the supported check
    assert [c.document for c in cited.citations] == attention.ranking(cag.citation_heads)[:2]


def test_calibrate_citations_writes_its_own_questions(make_cag, monkeypatch):
    cag = make_cag()
    kv, _ = cag.prepare_kvcache(DATASET, None, INSTRUCTION)
    written = []
    write_question = cag._write_question
    monkeypatch.setattr(cag, "_write_question", lambda text: written.append(text) or write_question(text))

    heads = cag.calibrate_citations(kv, num_heads=2, num_questions=2, max_new_tokens=4)

    assert len(written) == 2 and all(isinstance(t, str) for t in written)
    assert len(heads) == 2


def test_known_models_get_their_citation_heads(make_cag, monkeypatch):
    from mantiq.cag.citations import KNOWN_CITATION_HEADS

    monkeypatch.setitem(KNOWN_CITATION_HEADS, "known/model", [(1, 2), (0, 3)])

    assert make_cag(model_name="known/model").citation_heads == [(1, 2), (0, 3)]
    assert make_cag(model_name="known/model", citation_heads=[]).citation_heads is None
    assert make_cag(model_name="other/model").citation_heads is None


def test_supported_asks_about_the_answer_and_restores_the_cache(make_cag):
    cag = make_cag()
    kv, _ = cag.prepare_kvcache(DATASET, None, INSTRUCTION)
    length = kv.get_seq_length()
    calls = []
    cag.model.get_input_embeddings().register_forward_pre_hook(lambda module, args: calls.append(args[0]))

    supported = cag._supported(QUESTION, ANSWER, kv)

    assert 0 <= supported <= 1
    assert ANSWER in cag.tokenizer.decode(calls[-1][0])
    assert kv.get_seq_length() == length


def test_yes_and_no_skip_a_shared_space_piece():
    from mantiq.cag.naiive_cag import _first_word_tokens

    class SpacePieceTokenizer:
        pieces = {0: " ", 1: "yes", 2: "no", 3: "Yes", 4: "No"}

        def encode(self, word, add_special_tokens=False):
            ids = [k for k, v in self.pieces.items() if v == word.strip()]
            return ([0] if word.startswith(" ") else []) + ids

        def decode(self, ids):
            return "".join(self.pieces[i] for i in ids)

    tokenizer = SpacePieceTokenizer()

    assert _first_word_tokens(tokenizer, ("yes", " yes", "Yes", " Yes")) == {1, 3}
    assert _first_word_tokens(tokenizer, ("no", " no")) == {2}


def test_attention_collector_handles_half_precision_overflow():
    from mantiq.cag import Document

    documents = [Document(0, "a", 0, 4), Document(1, "b", 4, 8)]
    collector = AttentionCollector(documents, knowledge_len=8, first_row=0)
    query = torch.full((1, 2, 3, 8), 300.0).half()  # 300 * 300 * 8 overflows float16
    key = torch.full((1, 1, 11, 8), 300.0).half()

    collector.add(query, key, None, None)

    assert torch.isfinite(collector.totals).all() and collector.totals.sum() > 0


def test_document_names_that_are_not_strings_survive_saving(make_cag, tmp_path):
    from pathlib import Path

    cag = make_cag()
    cag.prepare_kvcache_from_loader({Path("tv.md"): "The TV is 4K."}, str(tmp_path))

    loaded = cag.read_kv_cache(str(tmp_path / "kv_cache.pt"))

    assert [d.name for d in loaded.documents] == ["tv.md"]


@pytest.mark.parametrize("written, expected", [
    ("<think>\nThe text is about a TV.\n</think>\n\nWhat does the TV cost?", "What does the TV cost?"),
    ("<think>\nStill thinking when the tokens ran out", "The TV costs $499."),
])
def test_self_written_questions_leave_out_thinking(make_cag, monkeypatch, written, expected):
    cag = make_cag()
    template_kwargs = []
    apply_chat_template = cag.tokenizer.apply_chat_template
    monkeypatch.setattr(cag.tokenizer, "apply_chat_template",
                        lambda *a, **k: template_kwargs.append(k) or apply_chat_template(*a, **k))
    monkeypatch.setattr(cag.tokenizer, "decode", lambda *a, **k: written)

    assert cag._write_question("The TV costs $499.") == expected
    assert template_kwargs[-1]["enable_thinking"] is False
