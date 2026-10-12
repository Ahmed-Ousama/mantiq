"""Evaluate cite() and the supported score on Arabic questions, as in the README.

ARCD questions (Arabic Wikipedia) each get their paragraph and 9 from other articles, with Arabic
instructions. Compares cite() with the built-in heads, which were found on English questions, with
heads calibrated on Arabic paragraphs, and without heads; and checks supported over the right
paragraphs against another question's.

    pip install pyarrow
    python examples/evaluate_citations_arabic.py results.jsonl --model Qwen/Qwen3-4B --dtype float16
"""

import argparse
import json
import random
import re
import time

import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download

from mantiq import NaiveCAG

parser = argparse.ArgumentParser()
parser.add_argument("out_path")
parser.add_argument("--n", type=int, default=150)
parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
parser.add_argument("--dtype", default="float32")
parser.add_argument("--quantized", action="store_true")
args = parser.parse_args()

path = hf_hub_download("hsseinmz/arcd", "plain_text/validation-00000-of-00001.parquet", repo_type="dataset")
rows = pq.read_table(path).to_pylist()
titles = sorted({r["title"] for r in rows})
random.Random(0).shuffle(titles)
calibration_titles = set(titles[:15])
contexts = {r["context"]: r["title"] for r in rows}
test = [r for r in rows if r["title"] not in calibration_titles]
random.Random(0).shuffle(test)
test = test[:args.n]

kwargs = {"enable_thinking": False} if "qwen3" in args.model.lower() else None
cag = NaiveCAG(args.model, quantized=args.quantized, chat_template_kwargs=kwargs,
               model_kwargs={"dtype": getattr(torch, args.dtype)})
built_in = cag.citation_heads
INSTRUCTION = "أنت تجيب عن الأسئلة باستخدام المعلومات المعطاة فقط."
PLAIN = "أجب بعبارة قصيرة فقط."


def normalize(text):
    text = re.sub(r"[ً-ْـ]", "", text)  # diacritics and tatweel
    text = re.sub("[إأآا]", "ا", text).replace("ى", "ي").replace("ة", "ه")
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def correct(answer, golds):
    a = normalize(answer).split()
    for gold in golds:
        g = normalize(gold).split()
        if not g:
            continue
        common = sum(min(a.count(w), g.count(w)) for w in set(g))
        if 2 * common / (len(a) + len(g)) >= 0.5 or " ".join(g) in " ".join(a):
            return True
    return False


def docs_for(row, rng):
    docs = {f"{row['title']} 0": row["context"]}
    others = [(c, t) for c, t in contexts.items() if t != row["title"] and t not in calibration_titles]
    for context, title in rng.sample(others, 9):
        docs[f"{title} {len(docs)}"] = context
    names = list(docs)
    rng.shuffle(names)
    return {k: docs[k] for k in names}, f"{row['title']} 0"


def auroc(pos, neg):
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / max(1, len(pos) * len(neg))


start = time.perf_counter()
calibration_docs = {f"{t} {k}": c for k, (c, t) in enumerate(contexts.items()) if t in calibration_titles}
calibration_docs = dict(list(calibration_docs.items())[:40])
kv, _ = cag.prepare_kvcache_from_loader(calibration_docs, None, INSTRUCTION, PLAIN)
arabic_heads = cag.calibrate_citations(kv)
print(f"calibrated on {len(calibration_docs)} Arabic paragraphs in {time.perf_counter() - start:.0f}s: {arabic_heads}", flush=True)
print(f"built-in heads: {built_in}", flush=True)

results = []
with open(args.out_path, "w") as out:
    out.write(json.dumps({"arabic_heads": arabic_heads, "built_in": built_in}) + "\n")
    for i, row in enumerate(test):
        rng = random.Random(i)
        docs, gold = docs_for(row, rng)
        kv, _ = cag.prepare_kvcache_from_loader(docs, None, INSTRUCTION, PLAIN)
        question = row["question"]
        answer = cag.generate_response(question, kv, max_new_tokens=48).strip()
        rec = {"question": question, "answer": answer, "gold": gold, "correct": correct(answer, row["answers"]["text"]),
               "tokens": kv.get_seq_length()}
        for mode, heads in (("built_in", built_in), ("arabic", arabic_heads), ("plain", None)):
            cag.citation_heads = heads
            cited = cag.cite(question, answer, kv, top_k=1)
            rec[mode] = cited.citations[0].document if cited.citations else None
        rec["supported"] = cited.supported
        other_docs, _ = docs_for(test[(i + 1) % len(test)], random.Random(10_000 + i))
        kv_other, _ = cag.prepare_kvcache_from_loader(other_docs, None, INSTRUCTION, PLAIN)
        answer_other = cag.generate_response(question, kv_other, max_new_tokens=48).strip()
        rec["answer_other"] = answer_other
        rec["supported_other"] = cag._supported(question, answer_other, kv_other)
        cag.citation_heads = built_in
        results.append(rec)
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()
        if (i + 1) % 10 == 0 or i + 1 == len(test):
            line = f"[{i + 1}/{len(test)}] {time.perf_counter() - start:.0f}s tokens={rec['tokens']} correct={sum(r['correct'] for r in results) / len(results):.2f}"
            for mode in ("built_in", "arabic", "plain"):
                line += f" {mode}={sum(r[mode] == r['gold'] for r in results) / len(results):.2f}"
            line += f" supported made-up={auroc([r['supported'] for r in results], [r['supported_other'] for r in results]):.2f}"
            print(line, flush=True)
