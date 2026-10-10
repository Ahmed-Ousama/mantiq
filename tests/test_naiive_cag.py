import subprocess
import sys

import pytest
import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
from transformers.cache_utils import DynamicCache

from helpers import NO_SYSTEM_TEMPLATE, THINKING_TEMPLATE, make_tokenizer

DATASET = [
    {"question": "What are the features of this TV?", "answer": "This TV has 4K resolution and smart features.", "context": "55-inch TV with advanced display technology."},
    {"question": "Does this refrigerator save energy?", "answer": "Yes, it has an A++ energy rating.", "context": "Eco-friendly refrigerator with advanced cooling technology."},
    {"question": "What makes this washing machine special?", "answer": "It has AI-powered wash cycles for better cleaning.", "context": None},
]
INSTRUCTION = "You are a sales agent."
QUESTION = "Does this refrigerator save energy?"


def capture_input_ids(model):
    calls = []
    model.register_forward_pre_hook(lambda module, args, kwargs: calls.append(kwargs["input_ids"]), with_kwargs=True)
    return calls


def test_generate_leaves_cache_length_unchanged(make_cag):
    cag = make_cag()
    kv, _ = cag.prepare_kvcache(DATASET, None, INSTRUCTION)
    length = kv.get_seq_length()

    cag.generate_response("What are the features of this TV?", kv, max_new_tokens=5)
    cag.generate_response(QUESTION, kv, max_new_tokens=5)

    assert kv.get_seq_length() == length


def test_answer_does_not_depend_on_previous_queries(make_cag):
    cag = make_cag()
    kv, _ = cag.prepare_kvcache(DATASET, None, INSTRUCTION)

    first = cag.generate(QUESTION, kv, max_new_tokens=8)
    cag.generate("What are the features of this TV?", kv, max_new_tokens=8)
    again = cag.generate(QUESTION, kv, max_new_tokens=8)

    assert torch.equal(first, again)


def test_knowledge_prompt_has_a_single_bos(make_cag):
    cag = make_cag()
    calls = capture_input_ids(cag.model)

    cag.prepare_kvcache(DATASET, None, INSTRUCTION)

    assert calls[0][0].tolist().count(cag.tokenizer.bos_token_id) == 1


def test_query_prompt_continues_the_user_turn(make_cag):
    cag = make_cag()
    kv, _ = cag.prepare_kvcache(DATASET, None, INSTRUCTION)
    calls = capture_input_ids(cag.model)

    cag.generate(QUESTION, kv, max_new_tokens=1)

    assert cag.tokenizer.decode(calls[0][0]) == QUESTION + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


def test_template_without_system_role(make_cag):
    cag = make_cag(tokenizer=make_tokenizer(NO_SYSTEM_TEMPLATE))

    prefix, suffix = cag._render_prompt("Some context", INSTRUCTION, "Be brief.")

    assert prefix.startswith("<start_of_turn>user\n" + INSTRUCTION)
    assert suffix == "<end_of_turn>\n<start_of_turn>model\n"


def test_chat_template_kwargs_reach_the_template(make_cag):
    tokenizer = make_tokenizer(THINKING_TEMPLATE)

    assert make_cag(tokenizer=tokenizer)._question_suffix.endswith("<|end_header_id|>\n\n")
    assert make_cag(tokenizer=tokenizer, chat_template_kwargs={"enable_thinking": False})._question_suffix.endswith(
        "<think></think>"
    )


def test_tokenizer_without_chat_template(make_cag):
    cag = make_cag(tokenizer=make_tokenizer(chat_template=None))
    kv, _ = cag.prepare_kvcache(DATASET, None, INSTRUCTION)
    length = kv.get_seq_length()

    assert isinstance(cag.generate_response(QUESTION, kv, max_new_tokens=3), str)
    assert kv.get_seq_length() == length


@pytest.mark.parametrize("as_list", [False, True])
def test_generation_stops_at_eos(make_cag, as_list):
    cag = make_cag()
    kv, _ = cag.prepare_kvcache(DATASET, None, INSTRUCTION)
    first_token = cag.generate(QUESTION, kv, max_new_tokens=5)[0, 0].item()

    cag.model.generation_config.eos_token_id = [first_token] if as_list else first_token
    output = cag.generate(QUESTION, kv, max_new_tokens=5)

    assert output.tolist() == [[first_token]]


def test_model_without_llama_module_layout(make_cag):
    tokenizer = make_tokenizer()
    torch.manual_seed(0)
    config = GPTNeoXConfig(
        vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, max_position_embeddings=4096, eos_token_id=tokenizer.eos_token_id,
    )
    model = GPTNeoXForCausalLM(config).eval()
    assert not hasattr(model, "model")
    cag = make_cag(tokenizer=tokenizer, model=model)

    kv, _ = cag.prepare_kvcache(DATASET, None, INSTRUCTION)

    assert isinstance(cag.generate_response(QUESTION, kv, max_new_tokens=3), str)


def test_saving_creates_missing_directories(make_cag, tmp_path):
    cag = make_cag()
    saving_path = tmp_path / "cache_directory" / "nested"

    cag.prepare_kvcache(DATASET, str(saving_path), INSTRUCTION)

    assert (saving_path / "kv_cache.pt").exists()


def test_saved_cache_loads_in_a_fresh_process(make_cag, tmp_path):
    cag = make_cag()
    cag.prepare_kvcache(DATASET, str(tmp_path), INSTRUCTION)
    code = "import sys, torch; print(len(torch.load(sys.argv[1], weights_only=True)['layers']))"

    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "kv_cache.pt")], capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(cag.model.config.num_hidden_layers)


def test_reloaded_cache_answers_like_the_original(make_cag, tmp_path):
    cag = make_cag()
    kv, _ = cag.prepare_kvcache(DATASET, str(tmp_path), INSTRUCTION)
    expected = cag.generate(QUESTION, kv, max_new_tokens=8)

    # A new instance, as in a new session: prepare_kvcache is never called on it.
    other = make_cag(model=cag.model, tokenizer=cag.tokenizer)
    loaded = other.read_kv_cache(str(tmp_path / "kv_cache.pt"))

    assert loaded.get_seq_length() == kv.get_seq_length()
    assert torch.equal(other.generate(QUESTION, loaded, max_new_tokens=8), expected)


def test_reading_a_cache_from_another_model_warns(make_cag, tmp_path):
    cag = make_cag(model_name="model-a")
    cag.prepare_kvcache(DATASET, str(tmp_path), INSTRUCTION)
    other = make_cag(model=cag.model, tokenizer=cag.tokenizer, model_name="model-b")

    with pytest.warns(UserWarning, match="model-a"):
        other.read_kv_cache(str(tmp_path / "kv_cache.pt"))


@pytest.mark.parametrize("content", [{"foo": torch.zeros(1)}, DynamicCache()])
def test_reading_other_files_raises(make_cag, tmp_path, content):
    cag = make_cag()
    path = tmp_path / "other.pt"
    torch.save(content, path)

    with pytest.raises(ValueError, match="not a mantiq KV cache"):
        cag.read_kv_cache(str(path))
