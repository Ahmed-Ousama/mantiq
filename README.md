# mantiq

**Cache-Augmented Generation (CAG) for HuggingFace models.** mantiq reads your knowledge into the model's KV cache once, then answers any number of questions from it, with no retrieval pipeline and without re-reading the knowledge for every question.

[![tests](https://github.com/tactful-ai/mantiq/actions/workflows/tests.yml/badge.svg)](https://github.com/tactful-ai/mantiq/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
[![license](https://img.shields.io/badge/license-MIT-green)](https://github.com/tactful-ai/mantiq/blob/main/LICENSE)

```python
from mantiq import NaiveCAG

cag = NaiveCAG("Qwen/Qwen3-0.6B", chat_template_kwargs={"enable_thinking": False})

# Read the knowledge once...
kv_cache, _ = cag.prepare_kvcache_from_loader(open("faq.txt").read())

# ...then answer any number of questions from it.
for chunk in cag.stream_response("Do you deliver on weekends?", kv_cache):
    print(chunk, end="", flush=True)
```

## Why CAG

RAG retrieves a few chunks for every question. That means embeddings, a vector store and a chunking strategy, and an answer can only be as good as the chunks that were retrieved. When your knowledge fits in the model's context window (a product catalog, an FAQ, a policy, a manual), you can skip all of that:

- **Fast answers.** The model reads the knowledge once. Each question then only processes its own few tokens.
- **No retrieval misses.** Every answer sees all of the knowledge, not just the top-k chunks.
- **Read once, reuse anywhere.** Save the cache to disk and load it in another process or after a restart.
- **Knows its sources and stays current.** `cite()` points to the documents an answer comes from (its top citation is a supporting paragraph 93% of the time on HotpotQA with Qwen3-4B), and `update_kvcache()` re-reads only what changed.

Answering 5 questions about a 2,355-token product catalog:

| | first token | full answer |
| --- | ---: | ---: |
| Re-reading the knowledge for every question | 34.33 s | 37.21 s |
| mantiq (knowledge cached once) | **0.53 s** | **3.39 s** |
| Speedup | **65x** | **11x** |

<sub>Qwen3-0.6B in bfloat16 on a laptop CPU (Intel i7-9850H), answers of up to 32 tokens, averaged over 5 questions. Caching the knowledge took 33 s, once. The more knowledge, the bigger the gap. Measure your own setup with <code>python examples/benchmark.py</code>.</sub>

## How it works

```mermaid
flowchart LR
    K["Knowledge<br/>(Q&A pairs, PDFs, web pages...)"] --> P["Chat prompt:<br/>instructions + knowledge"]
    P -->|"run the model once"| C[("KV cache")]
    C <-->|"save / load"| F["kv_cache.pt"]
    Q["Question"] --> G["Generate"]
    C --> G
    G --> A["Answer"]
    G -.->|"crop back to the knowledge"| C
```

mantiq renders the model's own chat template with a placeholder where the question goes, runs the model over everything before it and keeps the resulting KV cache. For each question it feeds only the question and the end of the turn, generates the answer, then crops the cache back so the next question starts from the knowledge alone.

## Install

```bash
pip install mantiq

# with the document loaders (PDF, CSV, JSON, Markdown, HTML, web pages, ...)
pip install "mantiq[loaders]"
```

Or the latest version from GitHub: `pip install git+https://github.com/tactful-ai/mantiq.git`

mantiq runs on CPU out of the box. With a CUDA GPU, models are loaded in 4-bit automatically.

## Quickstart

```python
from mantiq import NaiveCAG

catalog = [
    {"question": "What are the features of the Vision 55 TV?", "answer": "It has 4K resolution, HDR10 and smart apps.", "context": "55-inch TV, $499, 2-year warranty."},
    {"question": "Do you deliver?", "answer": "Delivery is free for orders over $300, otherwise $25.", "context": "Delivery takes 2-4 business days."},
]

cag = NaiveCAG("Qwen/Qwen3-0.6B", chat_template_kwargs={"enable_thinking": False})

kv_cache, seconds = cag.prepare_kvcache(
    dataset=catalog,
    prompt_instruction="You are a friendly sales agent for an electronics store.",
    answer_instruction="Answer in one or two sentences, using only the given context.",
)

print(cag.generate_response("I want a 4K screen, what do you have?", kv_cache))
# The Vision 55 TV has a 4K resolution, HDR10, and smart apps.
print(cag.generate_response("How long does delivery take?", kv_cache))
# Delivery takes 2-4 business days.
```

The full script is in [`examples/quickstart.py`](examples/quickstart.py).

## Usage

### Knowledge from documents

```python
from mantiq.loaders import LoaderFactory

# "PDF", "CSV", "JSON", "Markdown", "HTML", "Text", "WebPage", "Directory", "MicrosoftFile" (case-insensitive)
text = LoaderFactory.get_loader("PDF", "./catalog.pdf")

kv_cache, seconds = cag.prepare_kvcache_from_loader(text, prompt_instruction="You are a support agent.")
```

`prepare_kvcache_from_loader` takes any string, so you can also pass text you already have. Extra keyword arguments go to the underlying LangChain loader, e.g. `LoaderFactory.get_loader("JSON", "data.json", jq_schema=".items[].text")`. For Azure Document Intelligence (`"MicrosoftFile"`), install `mantiq[azure]`.

### Citations: which documents an answer comes from

Pass the knowledge as separate documents, a list or a dict of names to texts, and `cite()` tells you which of them an answer draws on:

```python
kv_cache, _ = cag.prepare_kvcache_from_loader({
    "delivery.md": "Delivery takes 2-4 business days. It is free for orders over $300.",
    "returns.md": "Returns are accepted within 14 days with the receipt.",
})

answer = cag.generate_response("Can I send it back?", kv_cache)
cited = cag.cite("Can I send it back?", answer, kv_cache)

cited.citations  # [Citation(document="returns.md", text="Returns are...", score=...)]
cited.supported  # from 0 to 1: low when the answer doesn't come from the knowledge
```

`cite()` works on the cache itself, without re-reading the knowledge:

- **With citation heads**, built in for 10 models (Qwen3 0.6B to 8B, Llama 3.2 1B and 3B, Llama 3.1 8B, Gemma 3 1B and 4B, Mistral 7B v0.3) or found for any model with `calibrate_citations()`: a few attention heads in a model look at the document an answer comes from, and `cite()` ranks the documents by where they look, in one pass however many documents there are.
- **Without them**, it hides each document from the cache in turn and ranks the documents by how much less likely the answer becomes. With more than `max_candidates` documents (16 by default), only the ones the answer attends to most are hidden.

A citation's `score` is how much less likely the answer becomes with that document hidden, in nats per token. With `prepare_kvcache`, each Q&A entry is a document, cited by its index.

To find the heads for another model, calibrate once on a cache with a few dozen documents:

```python
heads = cag.calibrate_citations(kv_cache)  # 2 to 4 minutes on a T4 GPU, once per model
cag = NaiveCAG("your/model", citation_heads=heads)  # reuse them later
```

The model writes a question about 20 of the documents, answers each one from the cache, and mantiq keeps the heads that most often look at the document whose hiding hurts the answer most. No labeled data is needed; pass your own `questions` if you have them.

How well it works, without → with citation heads. [HotpotQA](https://hotpotqa.github.io/) questions come with 10 paragraphs, 2 of them labeled as supporting the answer, so picking at random finds one 20% of the time. [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) questions come with their paragraph and 9 others.

| Model | HotpotQA: top citation supporting | HotpotQA: both in the top 2 | SQuAD: top citation right |
| --- | ---: | ---: | ---: |
| Qwen3-0.6B | 69% → **74%** | 34% → 30% | 95% → **97.5%** |
| Qwen3-1.7B | 81% → **87.5%** | 30% → **40%** | 97.5% → **100%** |
| Qwen3-4B | 87% → **93%** | 43% → **56%** | 98% → **99%** |
| Qwen3-8B (4-bit) | 89% → **92.5%** | 37.5% → **45%** | 100% → 100% |
| Mistral-7B-Instruct-v0.3 (4-bit) | 86% → **91%** | 40% → **54%** | 97.5% → **100%** |
| Gemma-3-1B-it | 74% → **77.5%** | 15% → **27.5%** | 92.5% → **97.5%** |
| Gemma-3-4B-it | 86% → **92.5%** | 25% → **54%** | 100% → 100% |
| Llama-3.2-1B-Instruct | 89% → 89% | 29% → **41%** | 90% → **97.5%** |
| Llama-3.2-3B-Instruct | 95% → 94% | 49% → **54%** | 97.5% → **100%** |
| Llama-3.1-8B-Instruct (4-bit) | 94% → **96%** | 47.5% → **54%** | 100% → 100% |

With 100 HotpotQA paragraphs per question, Qwen3-4B's top citation is supporting 90% of the time with heads, against 77.5% without. The heads also work across languages: on 150 Arabic questions from [ARCD](https://huggingface.co/datasets/hsseinmz/arcd), with Arabic instructions, Qwen3-4B's built-in heads, found on English questions, put the right paragraph first 98% of the time (96% without heads). Heads also make `cite()` about 3 times faster: 0.3 s instead of 1 s per answer with 10 documents, and 1.2 s instead of 5 s with 100, for Qwen3-4B. Measured on a Kaggle T4 GPU with 80 to 150 questions per model (40 with 100 paragraphs), so differences of a few points are within noise. Treat citations as strong hints, not proof.

`supported` catches answers that don't come from the knowledge. After the answer, the model is asked whether the knowledge states it, and `supported` is its probability of yes. To measure it, each HotpotQA question was answered once over its own paragraphs and once over another question's paragraphs, where the answer can't be found:

| Model | made-up answers (from the wrong paragraphs) | wrong answers (from the right paragraphs) |
| --- | ---: | ---: |
| Qwen3-0.6B | 88% | 62% |
| Qwen3-1.7B | 95% | 57% |
| Qwen3-4B | 97% | 56% |
| Qwen3-8B (4-bit) | 90% | 70% |
| Mistral-7B-Instruct-v0.3 (4-bit) | 96% | 67% |
| Gemma-3-1B-it | 90% | 50% |
| Gemma-3-4B-it | 84% | 59% |
| Llama-3.2-1B-Instruct | 81% | 60% |
| Llama-3.2-3B-Instruct | 91% | 60% |
| Llama-3.1-8B-Instruct (4-bit) | 87% | 69% |
| Qwen3-4B, Arabic questions | 92% | 70% |

The numbers are AUROC over 100 questions per model (80 for Mistral, Gemma and Llama, 150 in Arabic): how often `supported` scores an answer from the right paragraphs above one from the wrong paragraphs (first column), or a correct answer above a wrong one (second column). 50% is guessing. So `supported` reliably flags made-up answers, but it hardly helps with a wrong answer that the knowledge does state, for example a name taken from the wrong paragraph. It works best with models that answer without thinking first, like Qwen3 with `enable_thinking` off.

Reproduce with `python examples/evaluate_citations.py results.jsonl --model Qwen/Qwen3-4B --dtype float16`, and the Arabic results with `examples/evaluate_citations_arabic.py`.

### Updating the knowledge

When the knowledge changes, update the cache instead of rebuilding it:

```python
documents["prices.md"] = "The Vision 55 TV now costs $449."
kv_cache, seconds = cag.update_kvcache(kv_cache, documents)
```

The result is the same as a new cache, but mantiq only re-reads the knowledge from the first change onward. Put the documents that change most often (prices, stock, offers) last, and an update costs a fraction of a rebuild: in [`examples/live_updates.py`](examples/live_updates.py), changing this week's offer at the end of a 964-token catalog takes 0.3 s instead of 11.8 s on a laptop CPU. It also works on a cache loaded with `read_kv_cache`, and takes `saving_path` to save the result.

### Streaming and sampling

```python
for chunk in cag.stream_response(question, kv_cache, max_new_tokens=200):
    print(chunk, end="", flush=True)

# Greedy by default; set a temperature to sample
answer = cag.generate_response(question, kv_cache, temperature=0.7, top_p=0.9)
```

The cache is restored after every answer, so one cache can answer any number of questions, one at a time. If you stop reading a stream early, close it (or let it go out of scope) before the next question.

### Saving and loading the cache

```python
kv_cache, _ = cag.prepare_kvcache(catalog, saving_path="./cache_directory")  # writes kv_cache.pt

# Later, in another process:
kv_cache = cag.read_kv_cache("./cache_directory/kv_cache.pt")
```

A cache only works with the model that built it; mantiq warns if the model names differ. It takes memory in proportion to the knowledge: about 112 KB per token for Qwen3-0.6B in bfloat16, so 10,000 tokens take about 1.1 GB.

### Loading the model

```python
cag = NaiveCAG(
    "meta-llama/Llama-3.2-1B-Instruct",
    hf_token="hf_...",          # only for gated models; defaults to HF_TOKEN or `hf auth login`
    quantized=True,              # 4-bit; defaults to True when a CUDA GPU and bitsandbytes are available
    model_kwargs={"revision": "main", "attn_implementation": "sdpa"},  # passed to from_pretrained
)
```

- Models load in their own dtype (usually bfloat16). On CPUs without native bfloat16 support, `model_kwargs={"dtype": torch.float32}` prepares knowledge faster but answers slower (3x and 1.7x on the benchmark laptop above).
- `chat_template_kwargs` passes extra variables to the chat template, e.g. `{"enable_thinking": False}` for Qwen3.
- `trust_remote_code=True` runs custom modeling code from the model repo. It is off by default; only enable it for repos you trust.
- The knowledge is read 1,024 tokens at a time. If preparing a cache still runs out of memory, lower `cag.prefill_chunk_size`.

## When to use RAG instead

- The knowledge doesn't fit in the model's context window (mantiq raises a clear error when it doesn't).
- Large parts of the knowledge change all the time. Changes to a few documents are cheap if they come last.
- You need citations you can rely on without checking them.

## Limitations

- One question at a time per cache, and each question is answered on its own, without conversation history.
- Supports models with standard attention caches (e.g. Llama, Qwen3). Hybrid linear-attention models are not supported.

## Roadmap

- Lighter document loaders without LangChain, since `langchain-community` is no longer maintained.
- Multi-turn conversations on top of the cached knowledge.
- Answering several questions at once from one cache.
- Built-in citation heads for more models.
- Catching wrong answers that the knowledge does state; `supported` only catches made-up ones.

Have an idea or a use case? [Open an issue](https://github.com/tactful-ai/mantiq/issues).

## Background

mantiq is developed by the [tactful](https://tactful.ai/) team as part of its work on conversational AI. It builds on the paper [Don't Do RAG: When Cache-Augmented Generation is All You Need for Knowledge Tasks](https://arxiv.org/abs/2412.15605) and its [original implementation](https://github.com/hhhuang/CAG).

## Contributing

Contributions are welcome. Fork the repository, create a branch, and open a pull request.

```bash
pip install -e ".[loaders,test]"
pytest
```

The tests use tiny models built on the fly, so they need no downloads and no GPU. See [`CHANGELOG.md`](CHANGELOG.md) for what changed in each release.

## Citation

If you use mantiq in your research, please cite the paper it builds on:

```bibtex
@article{chan2024dontdorag,
  title   = {Don't Do RAG: When Cache-Augmented Generation is All You Need for Knowledge Tasks},
  author  = {Chan, Brian J and Chen, Chao-Ting and Cheng, Jui-Hung and Huang, Hen-Hsen},
  journal = {arXiv preprint arXiv:2412.15605},
  year    = {2024}
}
```

## License

MIT. See [LICENSE](https://github.com/tactful-ai/mantiq/blob/main/LICENSE).
