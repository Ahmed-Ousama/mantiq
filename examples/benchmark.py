"""Compare answering from a cached knowledge KV cache (mantiq) with re-reading the
knowledge for every question, which is what putting it in the prompt costs.

    python examples/benchmark.py --model Qwen/Qwen3-0.6B --products 80
"""

import argparse
import platform
import time

import torch

from mantiq import NaiveCAG

KINDS = ["TV", "fridge", "washer", "laptop", "phone", "air conditioner", "microwave", "vacuum cleaner"]
INSTRUCTION = "You are a sales agent for an electronics store."
ANSWER_INSTRUCTION = "Answer in one sentence, using only the given context."


def make_catalog(n):
    return "\n".join(
        f"Item {100 + i}: a {KINDS[i % len(KINDS)]} priced at ${200 + 37 * i}, "
        f"with a {1 + i % 5}-year warranty, delivered in {2 + i % 4} days."
        for i in range(n)
    )


def timed_answer(cag, question, get_cache, max_new_tokens):
    """Return (time to first token, time to full answer) in seconds."""
    start = time.perf_counter()
    kv = get_cache()
    first = None
    for _ in cag.stream_response(question, kv, max_new_tokens=max_new_tokens):
        if first is None:
            first = time.perf_counter() - start
    return first, time.perf_counter() - start


def mean(values):
    return sum(values) / len(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--products", type=int, default=80, help="catalog size; more products means more knowledge tokens")
    parser.add_argument("--questions", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], help="default: mantiq's choice")
    args = parser.parse_args()

    template_kwargs = {"enable_thinking": False} if "qwen3" in args.model.lower() else None
    model_kwargs = {"dtype": getattr(torch, args.dtype)} if args.dtype else None
    cag = NaiveCAG(args.model, chat_template_kwargs=template_kwargs, model_kwargs=model_kwargs)
    knowledge = make_catalog(args.products)
    questions = [f"What is the warranty on item {100 + i * 7 % args.products}?" for i in range(args.questions)]

    def prepare():
        return cag.prepare_kvcache_from_loader(knowledge, None, INSTRUCTION, ANSWER_INSTRUCTION)

    prepare()  # warm up
    kv, prep_seconds = prepare()
    device = "CUDA " + torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU " + platform.processor()
    print(f"{args.model} ({cag.model.dtype}) on {device}")
    print(f"Knowledge: {kv.get_seq_length()} tokens, cached once in {prep_seconds:.2f}s\n")

    no_cache = [timed_answer(cag, q, lambda: prepare()[0], args.max_new_tokens) for q in questions]
    cached = [timed_answer(cag, q, lambda: kv, args.max_new_tokens) for q in questions]

    rows = [("re-read knowledge", no_cache), ("mantiq (cached)", cached)]
    print(f"{'':<20}{'first token':>14}{'full answer':>14}")
    for name, times in rows:
        print(f"{name:<20}{mean([t[0] for t in times]):>13.2f}s{mean([t[1] for t in times]):>13.2f}s")
    first_speedup = mean([t[0] for t in no_cache]) / mean([t[0] for t in cached])
    total_speedup = mean([t[1] for t in no_cache]) / mean([t[1] for t in cached])
    print(f"{'speedup':<20}{first_speedup:>13.1f}x{total_speedup:>13.1f}x")
    print(f"\n(average over {args.questions} questions, up to {args.max_new_tokens} new tokens each)")


if __name__ == "__main__":
    main()
