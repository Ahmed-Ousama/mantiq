import pytest
import torch

from mantiq.cag import CheckedAnswer
from mantiq.cag.citations import KNOWN_ANSWER_CHECKS, KNOWN_ANSWERABLE_CHECKS
from mantiq.cag.naiive_cag import _load_answer_check
from helpers import THINKING_TEMPLATE, make_llama, make_tokenizer

DOCUMENTS = {
    "tv.md": "The TV has 4K resolution and smart features.",
    "fridge.md": "The refrigerator has an A++ energy rating.",
}
QUESTION = "Does this refrigerator save energy?"
ANSWER = "Yes, it has an A++ energy rating."


def make_check(model, layers=(1, 2), bias=0.0):
    torch.manual_seed(0)
    return {"layers": list(layers), "weight": torch.randn(len(layers), model.config.hidden_size), "bias": bias}


def test_without_an_answer_check_there_is_no_confidence(make_cag):
    cag = make_cag()
    kv, _ = cag.prepare_kvcache_from_loader(DOCUMENTS)

    assert cag.answer_check is None
    assert cag.cite(QUESTION, ANSWER, kv).confidence is None
    checked = cag.generate_checked_response(QUESTION, kv, max_new_tokens=6)
    assert checked.confidence is None and not checked.rethought


def test_confidence_reads_the_internal_state_over_the_answer(make_cag):
    tokenizer = make_tokenizer()
    model = make_llama(tokenizer)
    check = make_check(model)
    cag = make_cag(tokenizer=tokenizer, model=model, answer_check=check)
    kv, _ = cag.prepare_kvcache_from_loader(DOCUMENTS)
    length = kv.get_seq_length()

    confidence = cag.cite(QUESTION, ANSWER, kv).confidence

    assert kv.get_seq_length() == length
    # The same, from one pass over the whole prompt without a cache.
    query_ids = tokenizer.encode(QUESTION + cag._question_suffix, add_special_tokens=False)
    answer_ids = tokenizer.encode(ANSWER, add_special_tokens=False)
    hidden = model(torch.tensor([kv.token_ids + query_ids + answer_ids]), output_hidden_states=True).hidden_states
    start = len(kv.token_ids) + len(query_ids)
    features = torch.stack([hidden[layer][0, start:].mean(0) for layer in check["layers"]])
    assert confidence == pytest.approx(torch.sigmoid((features * check["weight"]).sum() + check["bias"]).item(), abs=1e-4)


@pytest.mark.parametrize("bias, rethinks", [(100.0, False), (-100.0, True)])
def test_checked_response_thinks_again_only_when_unsure(make_cag, bias, rethinks):
    tokenizer = make_tokenizer(THINKING_TEMPLATE)
    tokenizer.add_tokens(["<think>", "</think>"])
    model = make_llama(tokenizer)
    cag = make_cag(tokenizer=tokenizer, model=model, chat_template_kwargs={"enable_thinking": False},
                   answer_check=make_check(model, bias=bias))
    kv, _ = cag.prepare_kvcache_from_loader(DOCUMENTS)
    length = kv.get_seq_length()

    checked = cag.generate_checked_response(QUESTION, kv, max_new_tokens=6, max_thinking_tokens=5)

    assert isinstance(checked, CheckedAnswer)
    assert checked.rethought is rethinks
    assert (checked.confidence < 0.5) is rethinks
    assert kv.get_seq_length() == length  # the thinking is cropped away too


def test_a_model_that_cannot_think_keeps_its_fast_answer(make_cag):
    tokenizer = make_tokenizer()
    model = make_llama(tokenizer)
    cag = make_cag(tokenizer=tokenizer, model=model, answer_check=make_check(model, bias=-100.0))
    kv, _ = cag.prepare_kvcache_from_loader(DOCUMENTS)

    checked = cag.generate_checked_response(QUESTION, kv, max_new_tokens=6)

    assert not checked.rethought and checked.confidence < 0.5
    assert checked.answer == cag.generate_response(QUESTION, kv, max_new_tokens=6).strip()


FALLBACK = "I'm not sure, let me pass you to a colleague."


def test_a_fallback_answers_what_the_knowledge_answers(make_cag, monkeypatch):
    cag = make_cag()
    kv, _ = cag.prepare_kvcache_from_loader(DOCUMENTS)
    monkeypatch.setattr(cag, "_answerable", lambda *args: 0.9)
    monkeypatch.setattr(cag, "_unsure_of_question", lambda *args: False)
    monkeypatch.setattr(cag, "_supported", lambda *args: pytest.fail("the answer was checked"))

    checked = cag.generate_checked_response(QUESTION, kv, max_new_tokens=6, fallback=FALLBACK)

    assert checked.answerable == 0.9 and not checked.declined and checked.draft is None
    assert checked.answer == cag.generate_response(QUESTION, kv, max_new_tokens=6).strip()


def test_a_fallback_is_returned_without_answering_when_the_knowledge_does_not_answer(make_cag, monkeypatch):
    cag = make_cag()
    kv, _ = cag.prepare_kvcache_from_loader(DOCUMENTS)
    monkeypatch.setattr(cag, "_answerable", lambda *args: 0.1)
    monkeypatch.setattr(cag, "generate_response", lambda *args, **kwargs: pytest.fail("the model answered"))

    checked = cag.generate_checked_response(QUESTION, kv, max_new_tokens=6, fallback=FALLBACK)

    assert checked.declined and checked.answer == FALLBACK and checked.draft is None
    assert checked.answerable == 0.1


@pytest.mark.parametrize("supported, declined", [(0.1, True), (0.9, False)])
def test_when_unsure_of_the_question_the_answer_is_checked_against_the_knowledge(make_cag, monkeypatch, supported, declined):
    cag = make_cag()
    kv, _ = cag.prepare_kvcache_from_loader(DOCUMENTS)
    monkeypatch.setattr(cag, "_answerable", lambda *args: 0.9)
    monkeypatch.setattr(cag, "_unsure_of_question", lambda *args: True)
    monkeypatch.setattr(cag, "_supported", lambda *args: supported)
    fast = cag.generate_response(QUESTION, kv, max_new_tokens=6).strip()

    checked = cag.generate_checked_response(QUESTION, kv, max_new_tokens=6, fallback=FALLBACK)

    assert checked.declined is declined
    assert checked.answer == (FALLBACK if declined else fast)
    assert checked.draft == (fast if declined else None)


def test_an_answerable_check_reads_the_internal_state_and_sets_the_thresholds(make_cag):
    tokenizer = make_tokenizer()
    model = make_llama(tokenizer)
    check = {**make_check(model), "threshold": 0.1, "unsure": 0.3}
    cag = make_cag(tokenizer=tokenizer, model=model, answerable_check=check)
    kv, _ = cag.prepare_kvcache_from_loader(DOCUMENTS)
    length = kv.get_seq_length()

    answerable = cag._answerable(QUESTION, kv)

    assert kv.get_seq_length() == length
    ask = tokenizer.encode(f"Question: {QUESTION}\n\nIs the answer to this question in the context, or can it be worked out "
                           "from it? Reply with yes or no." + cag._question_suffix, add_special_tokens=False)
    hidden = model(torch.tensor([kv.token_ids + ask]), output_hidden_states=True).hidden_states
    features = torch.stack([hidden[layer][0, -1] for layer in check["layers"]])
    assert answerable == pytest.approx(torch.sigmoid((features * check["weight"]).sum() + check["bias"]).item(), abs=1e-4)
    assert cag._unsure_of_question(QUESTION, kv, 0.2) and not cag._unsure_of_question(QUESTION, kv, 0.4)


def test_a_fallback_replaces_an_unsure_answer_the_model_cannot_think_over(make_cag, monkeypatch):
    tokenizer = make_tokenizer()
    model = make_llama(tokenizer)
    cag = make_cag(tokenizer=tokenizer, model=model, answer_check=make_check(model, bias=-100.0))
    kv, _ = cag.prepare_kvcache_from_loader(DOCUMENTS)
    monkeypatch.setattr(cag, "_answerable", lambda *args: 0.9)
    monkeypatch.setattr(cag, "_unsure_of_question", lambda *args: False)
    fast = cag.generate_response(QUESTION, kv, max_new_tokens=6).strip()

    checked = cag.generate_checked_response(QUESTION, kv, max_new_tokens=6, fallback=FALLBACK)

    assert checked.declined and checked.answer == FALLBACK and not checked.rethought
    assert checked.draft == fast


def test_a_fallback_keeps_the_answer_given_after_thinking(make_cag, monkeypatch):
    tokenizer = make_tokenizer(THINKING_TEMPLATE)
    tokenizer.add_tokens(["<think>", "</think>"])
    model = make_llama(tokenizer)
    cag = make_cag(tokenizer=tokenizer, model=model, chat_template_kwargs={"enable_thinking": False},
                   answer_check=make_check(model, bias=-100.0))
    kv, _ = cag.prepare_kvcache_from_loader(DOCUMENTS)
    monkeypatch.setattr(cag, "_answerable", lambda *args: 0.9)
    monkeypatch.setattr(cag, "_unsure_of_question", lambda *args: False)

    checked = cag.generate_checked_response(QUESTION, kv, max_new_tokens=6, max_thinking_tokens=5, fallback=FALLBACK)

    assert checked.rethought and not checked.declined and checked.answer != FALLBACK


def test_without_a_fallback_answerable_is_not_asked(make_cag, monkeypatch):
    cag = make_cag()
    kv, _ = cag.prepare_kvcache_from_loader(DOCUMENTS)
    monkeypatch.setattr(cag, "_answerable", lambda *args: pytest.fail("answerable was asked"))

    checked = cag.generate_checked_response(QUESTION, kv, max_new_tokens=6)

    assert checked.answerable is None and not checked.declined


def test_answerable_asks_about_the_question_and_restores_the_cache(make_cag):
    cag = make_cag()
    kv, _ = cag.prepare_kvcache_from_loader(DOCUMENTS)
    length = kv.get_seq_length()
    calls = []
    cag.model.get_input_embeddings().register_forward_pre_hook(lambda module, args: calls.append(args[0]))

    answerable = cag._answerable(QUESTION, kv)

    assert 0 <= answerable <= 1
    assert QUESTION in cag.tokenizer.decode(calls[-1][0])
    assert kv.get_seq_length() == length


def test_answer_checks_load_from_the_package_a_file_or_not_at_all(make_cag, tmp_path):
    for model_name, (_, threshold) in KNOWN_ANSWER_CHECKS.items():
        check = _load_answer_check(model_name)
        assert check["weight"].shape[0] == len(check["layers"])
        assert check["threshold"] == threshold
    for model_name, (_, threshold, unsure) in KNOWN_ANSWERABLE_CHECKS.items():
        check = _load_answer_check(model_name, KNOWN_ANSWERABLE_CHECKS)
        assert check["weight"].shape[0] == len(check["layers"])
        assert (check["threshold"], check["unsure"]) == (threshold, unsure)
    assert _load_answer_check("someone/model-without-a-check") is None
    assert _load_answer_check(False) is None

    tokenizer = make_tokenizer()
    model = make_llama(tokenizer)
    torch.save(make_check(model), tmp_path / "check.pt")
    cag = make_cag(tokenizer=tokenizer, model=model, answer_check=str(tmp_path / "check.pt"))
    assert cag.answer_check["layers"] == [1, 2]
    assert make_cag(model_name="Qwen/Qwen3-4B", answer_check=False).answer_check is None
