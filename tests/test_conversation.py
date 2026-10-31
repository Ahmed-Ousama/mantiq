import pytest
import torch
from transformers.cache_utils import DynamicCache

from helpers import NO_SYSTEM_TEMPLATE, make_llama, make_tokenizer

DOCUMENTS = {
    "tv.md": "The TV has 4K resolution and smart features.",
    "fridge.md": "The refrigerator has an A++ energy rating.",
    "prices.md": "The TV costs $499 and the refrigerator costs $899.",
}
FIRST = "What are the features of this TV?"
FOLLOW_UP = "Does this refrigerator save energy?"
THIRD = "What makes this washing machine special?"


@torch.no_grad()
def greedy_from_scratch(cag, ids, max_new_tokens):
    """Greedy answer to the whole prompt, read in one go into a fresh cache."""
    cache, out = DynamicCache(), []
    logits = cag.model(input_ids=torch.tensor([ids]), past_key_values=cache, use_cache=True).logits[0, -1]
    for _ in range(max_new_tokens):
        token = int(logits.argmax())
        out.append(token)
        if token in cag._eos_token_ids():
            break
        logits = cag.model(input_ids=torch.tensor([[token]]), past_key_values=cache, use_cache=True).logits[0, -1]
    return cag.tokenizer.decode(out, skip_special_tokens=True).strip()


def test_the_first_question_is_answered_like_a_single_one(make_cag):
    cag = make_cag()
    kv, _ = cag.prepare_kvcache_from_loader(DOCUMENTS)

    assert cag.chat(kv).ask(FIRST, max_new_tokens=6) == cag.generate_response(FIRST, kv, max_new_tokens=6).strip()


def test_a_follow_up_sees_the_conversation(make_cag):
    cag = make_cag()
    kv, _ = cag.prepare_kvcache_from_loader(DOCUMENTS)
    chat = cag.chat(kv)
    first = chat.ask(FIRST, max_new_tokens=6)

    follow_up = chat.ask(FOLLOW_UP, max_new_tokens=6)

    text = cag._conversation_text([(FIRST, first)], FOLLOW_UP)
    assert FIRST in text and FOLLOW_UP in text
    ids = kv.token_ids + cag.tokenizer.encode(text, add_special_tokens=False)
    assert follow_up == greedy_from_scratch(cag, ids, 6)
    assert chat.history == [{"question": FIRST, "answer": first}, {"question": FOLLOW_UP, "answer": follow_up}]


def test_a_follow_up_reads_only_what_is_new(make_cag):
    cag = make_cag()
    kv, _ = cag.prepare_kvcache_from_loader(DOCUMENTS)
    chat = cag.chat(kv)
    first = chat.ask(FIRST, max_new_tokens=6)
    reads = []
    hook = cag.model.get_input_embeddings().register_forward_hook(lambda module, args, output: reads.append(args[0].shape[-1]))

    chat.ask(FOLLOW_UP, max_new_tokens=6)

    hook.remove()
    whole = len(cag.tokenizer.encode(cag._conversation_text([(FIRST, first)], FOLLOW_UP), add_special_tokens=False))
    assert reads[0] < whole - len(cag.tokenizer.encode(FIRST, add_special_tokens=False))


def test_the_shared_cache_is_left_as_it_was(make_cag):
    cag = make_cag()
    kv, _ = cag.prepare_kvcache_from_loader(DOCUMENTS)
    length = kv.get_seq_length()
    before = cag.generate_response(THIRD, kv, max_new_tokens=6)

    chat = cag.chat(kv)
    chat.ask(FIRST, max_new_tokens=6)
    chat.ask(FOLLOW_UP, max_new_tokens=6)

    assert kv.get_seq_length() == length
    assert cag.generate_response(THIRD, kv, max_new_tokens=6) == before


def test_conversations_on_one_cache_do_not_mix(make_cag):
    cag = make_cag()
    kv, _ = cag.prepare_kvcache_from_loader(DOCUMENTS)
    alone = cag.chat(kv)
    expected = [alone.ask(FIRST, max_new_tokens=6), alone.ask(FOLLOW_UP, max_new_tokens=6)]

    a, b = cag.chat(kv), cag.chat(kv)
    got = [a.ask(FIRST, max_new_tokens=6)]
    b.ask(THIRD, max_new_tokens=6)
    got.append(a.ask(FOLLOW_UP, max_new_tokens=6))

    assert got == expected


def test_a_saved_conversation_resumes(make_cag):
    cag = make_cag()
    kv, _ = cag.prepare_kvcache_from_loader(DOCUMENTS)
    chat = cag.chat(kv)
    chat.ask(FIRST, max_new_tokens=6)
    second = chat.ask(FOLLOW_UP, max_new_tokens=6)

    resumed = cag.chat(kv, history=[(turn["question"], turn["answer"]) for turn in chat.history[:1]])

    assert resumed.ask(FOLLOW_UP, max_new_tokens=6) == second


def test_streaming_records_what_was_streamed(make_cag):
    cag = make_cag()
    kv, _ = cag.prepare_kvcache_from_loader(DOCUMENTS)
    chat = cag.chat(kv)

    streamed = "".join(chat.stream(FIRST, max_new_tokens=6))

    assert chat.history[-1]["answer"] == streamed.strip()
    assert kv.get_seq_length() == len(kv.token_ids)


def test_max_turns_keeps_only_the_latest_turns(make_cag):
    cag = make_cag()
    kv, _ = cag.prepare_kvcache_from_loader(DOCUMENTS)
    chat = cag.chat(kv, history=[(FIRST, "It is 4K."), (FOLLOW_UP, "Yes.")], max_turns=1)
    seen = []
    original = cag._conversation_text
    cag._conversation_text = lambda turns, question: seen.append(turns) or original(turns, question)

    chat.ask(THIRD, max_new_tokens=3)

    assert seen[-1] == [(FOLLOW_UP, "Yes.")]


def test_old_turns_make_way_when_the_context_window_is_full(make_cag):
    tokenizer = make_tokenizer()
    length = make_cag(tokenizer=tokenizer).prepare_kvcache_from_loader(DOCUMENTS)[0].get_seq_length()
    cag = make_cag(tokenizer=tokenizer, model=make_llama(tokenizer, max_position_embeddings=length + 60))
    kv, _ = cag.prepare_kvcache_from_loader(DOCUMENTS)
    chat = cag.chat(kv, history=[(FIRST, "It has 4K resolution and smart features.")] * 5)

    chat.ask(FOLLOW_UP, max_new_tokens=4)  # five earlier turns don't fit; the latest ones that do are kept

    assert len(chat.history) == 6
    with pytest.raises(ValueError, match="doesn't fit"):
        cag.chat(kv).ask(FOLLOW_UP * 20, max_new_tokens=4)


@pytest.mark.parametrize("template", [NO_SYSTEM_TEMPLATE, None])
def test_conversations_work_without_a_system_role_or_a_chat_template(make_cag, template):
    tokenizer = make_tokenizer(template)
    cag = make_cag(tokenizer=tokenizer, model=make_llama(tokenizer))
    kv, _ = cag.prepare_kvcache_from_loader(DOCUMENTS)
    chat = cag.chat(kv)
    first = chat.ask(FIRST, max_new_tokens=6)

    follow_up = chat.ask(FOLLOW_UP, max_new_tokens=6)

    text = cag._conversation_text([(FIRST, first)], FOLLOW_UP)
    assert text.startswith(FIRST) and FOLLOW_UP in text
    assert follow_up == greedy_from_scratch(cag, kv.token_ids + cag.tokenizer.encode(text, add_special_tokens=False), 6)


def test_reset_forgets_the_conversation(make_cag):
    cag = make_cag()
    kv, _ = cag.prepare_kvcache_from_loader(DOCUMENTS)
    chat = cag.chat(kv)
    chat.ask(FIRST, max_new_tokens=6)

    chat.reset()

    assert chat.history == []
    assert chat.ask(FOLLOW_UP, max_new_tokens=6) == cag.generate_response(FOLLOW_UP, kv, max_new_tokens=6).strip()
