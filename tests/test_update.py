import pytest
import torch

from mantiq.cag import KnowledgeCache
from helpers import make_llama, make_tokenizer

DOCUMENTS = {
    "tv.md": "The TV has 4K resolution and smart features.",
    "fridge.md": "The refrigerator has an A++ energy rating.",
    "prices.md": "The TV costs $499 and the refrigerator costs $899.",
}
NEW_PRICES = {**DOCUMENTS, "prices.md": "The TV costs $449 and the refrigerator costs $899."}
QUESTION = "How much is the TV?"


def count_tokens_read(model):
    counts = []
    model.get_input_embeddings().register_forward_hook(lambda module, args, output: counts.append(args[0].shape[-1]))
    return counts


def assert_same_cache(a, b):
    assert a.token_ids == b.token_ids
    assert a.documents == b.documents
    for layer_a, layer_b in zip(a.layers, b.layers):
        torch.testing.assert_close(layer_a.keys, layer_b.keys)
        torch.testing.assert_close(layer_a.values, layer_b.values)


@pytest.mark.parametrize("new_documents", [NEW_PRICES, {**DOCUMENTS, "tv.md": "The TV has 8K resolution."}, {"tv.md": DOCUMENTS["tv.md"]}])
def test_update_matches_a_fresh_cache(make_cag, new_documents):
    cag = make_cag()
    kv, _ = cag.prepare_kvcache_from_loader(DOCUMENTS, None, "You are a sales agent.")

    cag.update_kvcache(kv, new_documents)
    fresh, _ = cag.prepare_kvcache_from_loader(new_documents, None, "You are a sales agent.")

    assert_same_cache(kv, fresh)
    assert torch.equal(cag.generate(QUESTION, kv, max_new_tokens=8), cag.generate(QUESTION, fresh, max_new_tokens=8))


def test_update_rereads_only_from_the_first_change(make_cag):
    cag = make_cag()
    kv, _ = cag.prepare_kvcache_from_loader(DOCUMENTS)
    total = kv.get_seq_length()
    counts = count_tokens_read(cag.model)

    cag.update_kvcache(kv, DOCUMENTS)
    assert counts == []  # nothing changed

    cag.update_kvcache(kv, NEW_PRICES)
    assert 0 < counts[0] < total - kv.documents[-1].start + 2  # about the last document and what follows it


def test_update_a_reloaded_cache(make_cag, tmp_path):
    cag = make_cag()
    cag.prepare_kvcache_from_loader(DOCUMENTS, str(tmp_path))
    kv = cag.read_kv_cache(str(tmp_path / "kv_cache.pt"))

    cag.update_kvcache(kv, NEW_PRICES, saving_path=str(tmp_path))
    fresh, _ = cag.prepare_kvcache_from_loader(NEW_PRICES)

    assert_same_cache(kv, fresh)
    assert_same_cache(cag.read_kv_cache(str(tmp_path / "kv_cache.pt")), fresh)


def test_update_that_does_not_fit_leaves_the_cache_alone(make_cag):
    tokenizer = make_tokenizer()
    length = make_cag(tokenizer=tokenizer).prepare_kvcache_from_loader(DOCUMENTS)[0].get_seq_length()
    cag = make_cag(tokenizer=tokenizer, model=make_llama(tokenizer, max_position_embeddings=length + 5))
    kv, _ = cag.prepare_kvcache_from_loader(DOCUMENTS)

    with pytest.raises(ValueError, match="context window"):
        cag.update_kvcache(kv, {**DOCUMENTS, "prices.md": DOCUMENTS["prices.md"] * 3})

    assert kv.get_seq_length() == length


def test_update_needs_the_knowledge_tokens(make_cag):
    cag = make_cag()

    with pytest.raises(ValueError, match="can't be updated"):
        cag.update_kvcache(KnowledgeCache(), DOCUMENTS)


def test_reading_the_knowledge_in_chunks_gives_the_same_cache(make_cag):
    cag = make_cag()
    whole, _ = cag.prepare_kvcache_from_loader(DOCUMENTS)

    cag.prefill_chunk_size = 7
    chunked, _ = cag.prepare_kvcache_from_loader(DOCUMENTS)

    assert_same_cache(chunked, whole)


def test_a_failed_update_leaves_a_cache_that_a_new_update_completes(make_cag, monkeypatch):
    cag = make_cag()
    kv, _ = cag.prepare_kvcache_from_loader(DOCUMENTS)
    extend = cag._extend

    def run_out_of_memory(cache, input_ids):
        extend(cache, input_ids[:, :1])  # part of the way, then fail
        raise RuntimeError("out of memory")

    monkeypatch.setattr(cag, "_extend", run_out_of_memory)
    with pytest.raises(RuntimeError):
        cag.update_kvcache(kv, NEW_PRICES)

    assert kv.get_seq_length() == len(kv.token_ids)
    assert kv.documents == []
    monkeypatch.setattr(cag, "_extend", extend)
    cag.update_kvcache(kv, NEW_PRICES)
    assert_same_cache(kv, cag.prepare_kvcache_from_loader(NEW_PRICES)[0])


def test_a_model_that_does_not_fill_the_cache_raises(make_cag, monkeypatch):
    cag = make_cag()
    monkeypatch.setattr(type(cag.model), "base_model", property(lambda self: lambda **kwargs: None))

    with pytest.raises(ValueError, match="doesn't fill the cache"):
        cag.prepare_kvcache_from_loader(DOCUMENTS)
