# Changelog

## 0.2.0

### Added

- `cite()` finds the documents an answer comes from, without re-reading the knowledge, and a `supported` score that flags answers that don't come from it (AUROC 0.97 on HotpotQA with Qwen3-4B, against 0.58 for the hiding-based score it replaces). `prepare_kvcache_from_loader()` accepts a list or a dict of named documents, and saved caches remember them.
- Citation heads: a few attention heads per model that look at the document an answer comes from. They are built in for 11 models across Qwen3, Llama 3.1 and 3.2, Gemma 3, Mistral and Phi-4, and `calibrate_citations()` finds them for any model, from questions the model writes itself. With them, `cite()` ranks documents in one pass: 93% top-citation accuracy on HotpotQA with Qwen3-4B (87% without), and about 3 times faster.
- `update_kvcache()` updates a cache to new knowledge, re-reading it only from the first change onward. The result is the same as a new cache.
- The knowledge is read in chunks (`prefill_chunk_size`, 1024 tokens by default), so large knowledge fits in memory on CPUs and on GPUs without flash attention. Models whose position encoding changes with the length (longrope, as in Phi-3 and Phi-4, and dynamic NTK) read knowledge past that length in one pass, so the encoding stays consistent.
- `stream_response()` yields the answer piece by piece as it is generated.
- `temperature` and `top_p` sampling for `generate_response()`, `generate()` and `stream_response()`. The default stays greedy.
- `model_kwargs` passes extra arguments to `from_pretrained`, e.g. `dtype`, `revision` or `attn_implementation`.
- `chat_template_kwargs` passes extra variables to the chat template, e.g. `{"enable_thinking": False}` for Qwen3.
- `Text` loader for plain text files. Loader type names are now case-insensitive.
- `from mantiq import NaiveCAG` and `mantiq.__version__`.
- A clear error when the knowledge doesn't fit in the model's context window.
- Examples: quick start, benchmark, citations, live updates and the citation evaluation, in `examples/`.

### Changed

- `NaiiveCAG` is now `NaiveCAG`. The old name still works.
- `hf_token` is optional; it is only needed for gated models and defaults to `HF_TOKEN` or a cached login.
- `quantized` defaults to 4-bit only when a CUDA GPU and bitsandbytes are available, so mantiq runs on CPU out of the box.
- Models load in their checkpoint's dtype (usually bfloat16) instead of float32 on every transformers version: half the memory, and faster answers even on CPU. On CPU, `model_kwargs={"dtype": torch.float32}` prepares the knowledge faster but answers slower.
- Prompts use the model's own chat template instead of a hard-coded Llama 3 one.
- Saved caches are plain tensors that load with `torch.load(weights_only=True)`. Caches saved by 0.1.0 must be rebuilt.
- `trust_remote_code` is off by default.
- Packaging moved to `pyproject.toml`. The loaders' dependencies are now the optional `loaders` extra (`azure` for Azure Document Intelligence).
- Requires Python 3.10+ and transformers 4.56+.

### Fixed

- The KV cache grew with every query, so later answers saw earlier questions and answers. It is now restored after each query.
- Saved caches could not be loaded in a new session.
- Preparing the knowledge computed logits for every knowledge token (tokens × vocabulary size), which could take more memory than the model itself and slowed preparation down.
- Generation crashed for models whose `eos_token_id` is a single integer, and for models without a Llama-style module layout.
- The knowledge prompt started with two BOS tokens.
- The JSON loader crashed when given a custom `jq_schema`, and the WebPage loader split a single URL into characters.

## 0.1.0

First release.
