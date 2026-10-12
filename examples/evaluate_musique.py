"""Evaluate cite() on MuSiQue-Ans, as in the paper.

Each question comes with 20 paragraphs, 2 to 4 of them supporting the answer. The answer is
attributed with the built-in citation heads (found on HotpotQA, unchanged), by hiding each of the 20
paragraphs, and, with --at2, by AT2's published estimator for the model, given the exact text the
model saw and its answer.

    pip install pyarrow
    python examples/evaluate_musique.py qwen.jsonl --model Qwen/Qwen3-4B

    pip install "transformers==4.56.2" at2 pyarrow
    python examples/evaluate_musique.py llama.jsonl --model meta-llama/Llama-3.1-8B-Instruct --at2 madrylab/at2-llama-3.1-8b-instruct --quantized
"""

import argparse
import json
import random
import string
import time

import torch
from huggingface_hub import hf_hub_download

from mantiq import NaiveCAG

parser = argparse.ArgumentParser()
parser.add_argument("out_path")
parser.add_argument("--model", required=True)
parser.add_argument("--at2", default=None)
parser.add_argument("--n", type=int, default=150)
parser.add_argument("--dtype", default="float16")
parser.add_argument("--quantized", action="store_true")
args = parser.parse_args()

path = hf_hub_download("dgslibisey/MuSiQue", "musique_ans_v1.0_dev.jsonl", repo_type="dataset")
questions = [json.loads(line) for line in open(path)]
random.Random(0).shuffle(questions)

kwargs = {"enable_thinking": False} if "qwen3" in args.model.lower() else None
cag = NaiveCAG(args.model, quantized=args.quantized, chat_template_kwargs=kwargs, model_kwargs={"dtype": getattr(torch, args.dtype)})
heads = cag.citation_heads
if not heads:
    raise SystemExit(f"No built-in citation heads for {args.model}; calibrate them first.")
INSTRUCTION = "You answer questions using only the given context."
PLAIN = "Answer with a short phrase only."

estimator = None
if args.at2:
    from at2 import AT2Attributor, AT2ScoreEstimator
    from at2.tasks import AttributionTask

    class CachedPromptTask(AttributionTask):
        """An AT2 task over the exact text the model saw (knowledge prompt, question) and its answer."""

        def __init__(self, model, tokenizer, input_text, generation, source_char_ranges):
            super().__init__(model, tokenizer)
            self._input_text = input_text
            self._source_char_ranges = source_char_ranges
            self.set_generation(generation)

        @property
        def input_text(self):
            return self._input_text

        def _get_source_token_ranges(self):
            tokens = self.get_tokens()
            return [(tokens.char_to_token(s), tokens.char_to_token(e - 1) + 1) for s, e in self._source_char_ranges]

        @property
        def target_token_range(self):
            return self.generation_token_start, self.generation_token_end

    estimator = AT2ScoreEstimator.from_hub(args.at2)


def normalize(text):
    text = "".join(c for c in text.lower() if c not in string.punctuation)
    return " ".join(w for w in text.split() if w not in ("a", "an", "the"))


start = time.perf_counter()
results = []
methods = ("heads", "hiding", "at2") if estimator else ("heads", "hiding")
with open(args.out_path, "w") as out:
    out.write(json.dumps({"heads": heads, "model": args.model, "at2": args.at2, "dataset": "musique"}) + "\n")
    for i, row in enumerate(questions[:args.n]):
        docs = {f"{p['idx']}: {p['title']}": p["paragraph_text"] for p in row["paragraphs"]}
        question = row["question"]
        kv, _ = cag.prepare_kvcache_from_loader(docs, None, INSTRUCTION, PLAIN)
        response = cag.generate_response(question, kv, max_new_tokens=24).strip()
        gold = [name for name, p in zip(docs, row["paragraphs"]) if p["is_supporting"]]
        rec = {"question": question, "gold": gold, "hops": row["id"].split("__")[0], "response": response,
               "correct": any(normalize(a) in normalize(response) for a in [row["answer"], *row["answer_aliases"]])}

        for mode, mode_heads in (("heads", heads), ("hiding", None)):
            cag.citation_heads = mode_heads
            t = time.perf_counter()
            cited = cag.cite(question, response, kv, top_k=4, max_candidates=len(docs))  # hiding: all 20
            rec[mode] = [c.document for c in cited.citations]
            rec[f"{mode}_seconds"] = time.perf_counter() - t
        cag.citation_heads = heads

        if estimator:
            # AT2 sees the same text: the cached knowledge prompt, then the question as mantiq sends it.
            prefix, _ = cag._render_prompt("\n\n".join(docs.values()), INSTRUCTION, PLAIN)
            input_text = prefix + question.strip() + cag._question_suffix
            ranges, offset = [], prefix.index("\n\n".join(docs.values()))
            for text in docs.values():
                ranges.append((offset, offset + len(text)))
                offset += len(text) + 2
            t = time.perf_counter()
            task = CachedPromptTask(cag.model, cag.tokenizer, input_text, response, ranges)
            scores = AT2Attributor(task, estimator).get_attribution_scores()
            names = list(docs)
            rec["at2"] = [names[j] for j in scores.argsort()[::-1][:4]]
            rec["at2_seconds"] = time.perf_counter() - t

        results.append(rec)
        out.write(json.dumps(rec) + "\n")
        out.flush()
        if (i + 1) % 10 == 0 or i + 1 == args.n:
            line = f"[{i + 1}/{args.n}] {time.perf_counter() - start:.0f}s"
            for m in methods:
                top = sum(bool(r[m]) and r[m][0] in r["gold"] for r in results) / len(results)
                # the share of the supporting paragraphs found among as many top citations as there are of them
                recall = sum(len(set(r[m][:len(r["gold"])]) & set(r["gold"])) / len(r["gold"]) for r in results) / len(results)
                secs = sum(r[f"{m}_seconds"] for r in results) / len(results)
                line += f"  {m}: top={top:.2f} recall={recall:.2f} ({secs:.2f}s)"
            print(line, flush=True)
