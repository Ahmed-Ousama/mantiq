"""Evaluate cite() with and without citation heads, as in the README.

Finds the heads with calibrate_citations() on 40 HotpotQA paragraphs and questions the model writes
itself, then cites answers to HotpotQA questions (10 paragraphs, 2 labeled as supporting, and 100
paragraphs) and SQuAD questions (their paragraph and 9 others), with the heads and without.

    pip install pyarrow
    python examples/evaluate_citations.py results.jsonl --model Qwen/Qwen3-4B --dtype float16
"""

import argparse
import json
import random
import string
import time

import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download

from mantiq import NaiveCAG

parser = argparse.ArgumentParser()
parser.add_argument("out_path")
parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
parser.add_argument("--dtype", default="float32")
parser.add_argument("--quantized", action="store_true")
parser.add_argument("--sizes", default="hotpot10:150,squad10:150,hotpot100:40")
args = parser.parse_args()
sizes = {k: int(v) for k, v in (s.split(":") for s in args.sizes.split(","))}

kwargs = {"enable_thinking": False} if "qwen3" in args.model.lower() else None
cag = NaiveCAG(args.model, quantized=args.quantized, chat_template_kwargs=kwargs,
               model_kwargs={"dtype": getattr(torch, args.dtype)})
INSTRUCTION = "You answer questions using only the given context."
PLAIN = "Answer with a short phrase only."

hotpot_path = hf_hub_download("hotpotqa/hotpot_qa", "distractor/validation-00000-of-00001.parquet", repo_type="dataset")
hotpot = [r for r in pq.read_table(hotpot_path).to_pylist() if r["answer"].lower() not in ("yes", "no")]
random.Random(0).shuffle(hotpot)
squad_path = hf_hub_download("rajpurkar/squad", "plain_text/validation-00000-of-00001.parquet", repo_type="dataset")
squad = pq.read_table(squad_path).to_pylist()
random.Random(0).shuffle(squad)
all_contexts = list({r["context"]: r["title"] for r in squad}.items())


def normalize(text):
    text = "".join(c for c in text.lower() if c not in string.punctuation)
    return " ".join(w for w in text.split() if w not in ("a", "an", "the"))


def hotpot_docs(row, total, rng):
    docs = {t: "".join(s) for t, s in zip(row["context"]["title"], row["context"]["sentences"])}
    for other in rng.sample(hotpot[2000:], 200):
        for t, s in zip(other["context"]["title"], other["context"]["sentences"]):
            if len(docs) < total and t not in docs:
                docs[t] = "".join(s)
        if len(docs) >= total:
            break
    titles = list(docs)
    rng.shuffle(titles)
    return {t: docs[t] for t in titles}


def squad_docs(row, total, rng):
    docs = {f"{row['title']} 0": row["context"]}
    for context, title in rng.sample(all_contexts, total * 3):
        if len(docs) < total and title != row["title"]:
            docs[f"{title} {len(docs)}"] = context
    names = list(docs)
    rng.shuffle(names)
    return {k: docs[k] for k in names}, f"{row['title']} 0"


# Calibrate on a knowledge base of 40 HotpotQA paragraphs that no test question uses, with
# 20 questions the model writes itself.
start = time.perf_counter()
calibration_docs = {}
for row in hotpot[3000:3010]:
    calibration_docs.update({t: "".join(s) for t, s in zip(row["context"]["title"], row["context"]["sentences"])})
calibration_docs = dict(list(calibration_docs.items())[:40])
kv, _ = cag.prepare_kvcache_from_loader(calibration_docs, None, INSTRUCTION, PLAIN)
heads = cag.calibrate_citations(kv)
calibration_seconds = time.perf_counter() - start
print(f"calibrated in {calibration_seconds:.0f}s: {heads}", flush=True)

results = []
with open(args.out_path, "w") as out:
    out.write(json.dumps({"heads": heads, "calibration_seconds": calibration_seconds}) + "\n")
    for name, count in sizes.items():
        done = []
        for i in range(count):
            rng = random.Random(i)
            if name.startswith("hotpot"):
                row = hotpot[i]
                docs, gold, answers = hotpot_docs(row, int(name[6:]), rng), set(row["supporting_facts"]["title"]), [row["answer"]]
            else:
                row = squad[i]
                docs, gold_name = squad_docs(row, int(name[5:]), rng)
                gold, answers = {gold_name}, row["answers"]["text"]
            kv, _ = cag.prepare_kvcache_from_loader(docs, None, INSTRUCTION, PLAIN)
            question = row["question"]
            response = cag.generate_response(question, kv, max_new_tokens=24).strip()
            rec = {"set": name, "question": question, "gold": sorted(gold), "response": response,
                   "correct": any(normalize(a) in normalize(response) for a in answers)}
            for mode, mode_heads in (("calibrated", heads), ("plain", None)):
                cag.citation_heads = mode_heads
                t = time.perf_counter()
                cited = cag.cite(question, response, kv, top_k=2)
                rec[mode] = [c.document for c in cited.citations]
                rec[f"{mode}_seconds"] = time.perf_counter() - t
            cag.citation_heads = heads
            done.append(rec)
            out.write(json.dumps(rec) + "\n")
            out.flush()
            if (i + 1) % 10 == 0 or i + 1 == count:
                line = f"{name} [{i + 1}/{count}] {time.perf_counter() - start:.0f}s"
                for mode in ("calibrated", "plain"):
                    top = sum(bool(r[mode]) and r[mode][0] in r["gold"] for r in done) / len(done)
                    secs = sum(r[f"{mode}_seconds"] for r in done) / len(done)
                    line += f"  {mode}: top={top:.2f} ({secs:.2f}s)"
                print(line, flush=True)
