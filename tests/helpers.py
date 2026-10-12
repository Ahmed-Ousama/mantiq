"""Tiny offline models and tokenizers for the tests."""

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

LLAMA3_STYLE_TEMPLATE = (
    "{{ bos_token }}{% for message in messages %}"
    "<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n{{ message['content'] | trim }}<|eot_id|>"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}"
)

# Rejects the system role, like the Gemma templates do.
NO_SYSTEM_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}"
    "{% for message in messages %}<start_of_turn>{{ message['role'] }}\n{{ message['content'] }}<end_of_turn>\n{% endfor %}"
    "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
)

# Reads an extra template variable, like the Qwen3 template does with enable_thinking.
THINKING_TEMPLATE = LLAMA3_STYLE_TEMPLATE + (
    "{% if add_generation_prompt and enable_thinking is defined and not enable_thinking %}<think></think>{% endif %}"
)

SPECIAL_TOKENS = [
    "<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>",
    "<start_of_turn>", "<end_of_turn>",
]

CORPUS = (
    "You are a sales agent. Context: Question: Answer: What are the features of this TV? "
    "This TV has 4K resolution and smart features. Does this refrigerator save energy? "
    "Yes, it has an A++ energy rating. What makes this washing machine special? "
    "It has AI-powered wash cycles for better cleaning. Provide a concise and relevant answer."
)


def make_tokenizer(chat_template=LLAMA3_STYLE_TEMPLATE):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=400, special_tokens=SPECIAL_TOKENS, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    tokenizer.train_from_iterator([CORPUS], trainer)
    # Add BOS on encode, like the Llama 3 tokenizer does.
    bos_id = tokenizer.token_to_id("<|begin_of_text|>")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<|begin_of_text|> $A", special_tokens=[("<|begin_of_text|>", bos_id)]
    )
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, bos_token="<|begin_of_text|>", eos_token="<|eot_id|>")
    fast.chat_template = chat_template
    return fast


def make_llama(tokenizer, seed=0, max_position_embeddings=4096, **config_kwargs):
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=max_position_embeddings,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        **config_kwargs,
    )
    return LlamaForCausalLM(config).eval()
