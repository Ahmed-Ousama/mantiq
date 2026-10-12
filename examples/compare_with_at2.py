"""Compare cite() with AT2 (Cohen-Wang et al., 2025, https://arxiv.org/abs/2504.13752), as in the README.

For each HotpotQA question (10 paragraphs, 2 labeled as supporting), mantiq answers from its cache,
then the same answer is attributed to the paragraphs three ways:
  heads   cite() with heads from calibrate_citations() on questions the model writes itself
  hiding  cite() without heads: each paragraph hidden in turn
  at2     AT2's published estimator for this model, given the exact text the model saw and its answer

AT2's code targets transformers 4.56 and computes attention on a single device:

    pip install "transformers==4.56.2" at2 pyarrow
    python examples/compare_with_at2.py phi.jsonl --model microsoft/Phi-4-mini-instruct --at2 madrylab/at2-phi-4-mini-instruct
    python examples/compare_with_at2.py llama.jsonl --model meta-llama/Llama-3.1-8B-Instruct --at2 madrylab/at2-llama-3.1-8b-instruct --quantized
"""

import argparse
import json
import random
import string
import time

import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download

from at2 import AT2Attributor, AT2ScoreEstimator
from at2.tasks import AttributionTask
from mantiq import NaiveCAG

parser = argparse.ArgumentParser()
parser.add_argument("out_path")
parser.add_argument("--model", required=True)
parser.add_argument("--at2", required=True)
parser.add_argument("--n", type=int, default=150)
parser.add_argument("--dtype", default="float16")
parser.add_argument("--quantized", action="store_true")
args = parser.parse_args()

path = hf_hub_download("hotpotqa/hotpot_qa", "distractor/validation-00000-of-00001.parquet", repo_type="dataset")
hotpot = [r for r in pq.read_table(path).to_pylist() if r["answer"].lower() not in ("yes", "no")]
random.Random(0).shuffle(hotpot)

cag = NaiveCAG(args.model, quantized=args.quantized, citation_heads=[], model_kwargs={"dtype": getattr(torch, args.dtype)})
INSTRUCTION = "You answer questions using only the given context."
PLAIN = "Answer with a short phrase only."


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


def normalize(text):
    text = "".join(c for c in text.lower() if c not in string.punctuation)
    return " ".join(w for w in text.split() if w not in ("a", "an", "the"))


def hotpot_docs(row, rng):
    docs = {t: "".join(s) for t, s in zip(row["context"]["title"], row["context"]["sentences"])}
    titles = list(docs)
    rng.shuffle(titles)
    return {t: docs[t] for t in titles}


start = time.perf_counter()
calibration_docs = {}
for row in hotpot[3000:3010]:
    calibration_docs.update({t: "".join(s) for t, s in zip(row["context"]["title"], row["context"]["sentences"])})
calibration_docs = dict(list(calibration_docs.items())[:40])
kv, _ = cag.prepare_kvcache_from_loader(calibration_docs, None, INSTRUCTION, PLAIN)
heads = cag.calibrate_citations(kv)
print(f"calibrated in {time.perf_counter() - start:.0f}s: {heads}", flush=True)
estimator = AT2ScoreEstimator.from_hub(args.at2)

results = []
with open(args.out_path, "w") as out:
    out.write(json.dumps({"heads": heads, "model": args.model, "at2": args.at2}) + "\n")
    for i, row in enumerate(hotpot[:args.n]):
        docs = hotpot_docs(row, random.Random(i))
        question = row["question"]
        kv, _ = cag.prepare_kvcache_from_loader(docs, None, INSTRUCTION, PLAIN)
        response = cag.generate_response(question, kv, max_new_tokens=24).strip()
        rec = {"question": question, "gold": sorted(set(row["supporting_facts"]["title"])), "response": response,
               "correct": normalize(row["answer"]) in normalize(response)}

        for mode, mode_heads in (("heads", heads), ("hiding", None)):
            cag.citation_heads = mode_heads
            t = time.perf_counter()
            rec[mode] = [c.document for c in cag.cite(question, response, kv, top_k=2).citations]
            rec[f"{mode}_seconds"] = time.perf_counter() - t

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
        rec["at2"] = [names[j] for j in scores.argsort()[::-1][:2]]
        rec["at2_seconds"] = time.perf_counter() - t

        results.append(rec)
        out.write(json.dumps(rec) + "\n")
        out.flush()
        if (i + 1) % 10 == 0 or i + 1 == args.n:
            line = f"[{i + 1}/{args.n}] {time.perf_counter() - start:.0f}s"
            for m in ("heads", "hiding", "at2"):
                top = sum(bool(r[m]) and r[m][0] in r["gold"] for r in results) / len(results)
                both = sum(set(r[m][:2]) >= set(r["gold"]) for r in results) / len(results)
                secs = sum(r[f"{m}_seconds"] for r in results) / len(results)
                line += f"  {m}: top={top:.2f} both={both:.2f} ({secs:.2f}s)"
            print(line, flush=True)
