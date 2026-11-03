"""Evaluate the fallback of generate_checked_response(), as in the README.

How often it declines questions the knowledge answers, and questions it doesn't, with the answerable
check and without it, on:

- support_test: hand-written customer service questions over three small businesses, written after the
  answerable check was trained and its thresholds were set (support_questions_test.json).
- support: the same over five other businesses (support_questions.json). The built-in check was trained
  on these too, so its numbers here say little.
- techqa: real IT support questions over IBM technotes. The answerable ones get the technote with the
  answer and 4 others; the others get the first 5 of their candidates.
- privacyqa: real questions over privacy policies; unanswerable when no sentence is relevant.
- faitheval: contexts with the sentence that answers the question removed, from 10 datasets.
- squad: SQuAD 2.0 questions over their paragraph and 9 others; the unanswerable ones were written to
  look answerable from it.
- hotpot: HotpotQA questions over their 10 paragraphs, without the ones that answer them, and over
  another question's paragraphs.
- musique: MuSiQue questions over 20 paragraphs, with and without what answers them.

The answer check's thinking is skipped: it comes after the fallback, and doesn't change what is declined.

    pip install pyarrow
    python examples/evaluate_fallback.py results.jsonl --model Qwen/Qwen3-4B --dtype float16
"""

import argparse
import json
import random
import re
import string
import tarfile
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download, list_repo_files

from mantiq import NaiveCAG

parser = argparse.ArgumentParser()
parser.add_argument("out_path")
parser.add_argument("--model", default="Qwen/Qwen3-4B")
parser.add_argument("--dtype", default="float16")
parser.add_argument("--quantized", action="store_true")
parser.add_argument("--sets", default="support_test,support,techqa,privacyqa,faitheval,squad,hotpot,musique")
parser.add_argument("--count", type=int, default=150, help="questions per kind in squad, hotpot and musique")
parser.add_argument("--max-tokens", type=int, default=12000, help="skip questions with more knowledge than this")
args = parser.parse_args()

kwargs = {"enable_thinking": False} if "qwen3" in args.model.lower() else None
cag = NaiveCAG(args.model, quantized=args.quantized, chat_template_kwargs=kwargs,
               model_kwargs=None if args.quantized else {"dtype": getattr(torch, args.dtype)})
INSTRUCTION = "You answer questions using only the given context."
PLAIN = "Answer with a short phrase only."
FALLBACK = "I'm not sure."
# A rough test of whether an answer says the knowledge doesn't have it.
REFUSAL = re.compile(r"not (mentioned|provided|stated|specified|given|available|included|found|known|clear)|no information|"
                     r"(does|do) not (mention|say|state|specify|provide|contain|include|indicate)|"
                     r"doesn't (mention|say|state|specify|provide|contain)|cannot be determined|can't be determined|"
                     r"not enough information|unknown|not in the (given )?context|no mention|غير مذكور|لم يذكر|لا توجد معلومات", re.I)


def normalize(text):
    text = "".join(c for c in text.lower() if c not in string.punctuation)
    return " ".join(w for w in text.split() if w not in ("a", "an", "the"))


def parquet(repo, name=None, split=None):
    if name is None:
        name = next(f for f in list_repo_files(repo, repo_type="dataset") if split in f and f.endswith(".parquet"))
    return pq.read_table(hf_hub_download(repo, name, repo_type="dataset")).to_pylist()


def paragraphs(row, leave_out=()):
    return {t: "".join(s) for t, s in zip(row["context"]["title"], row["context"]["sentences"]) if t not in leave_out}


def item(kind, question, docs, answerable, answers=None, instructions=(INSTRUCTION, PLAIN), tokens=24):
    return {"set": kind, "question": question, "docs": docs, "answerable": answerable, "answers": answers,
            "instructions": instructions, "tokens": tokens}


def businesses(filename, tag):
    spec = json.loads((Path(__file__).parent / filename).read_text())
    for business in spec.values():
        instructions = business["prompt"], business["answer"]
        for label, kind in ((True, f"{tag}_answerable"), (False, f"{tag}_unanswerable")):
            for q in business["answerable" if label else "unanswerable"]:
                yield item(kind, q, business["docs"], label, instructions=instructions, tokens=80)


def support():
    return businesses("support_questions.json", "support")


def support_test():
    return businesses("support_questions_test.json", "support_test")


def techqa():
    path = hf_hub_download("PrimeQA/TechQA", "TechQA.tar.gz", repo_type="dataset")
    with tarfile.open(path) as archive:
        rows = json.load(archive.extractfile("TechQA/training_and_dev/dev_Q_A.json"))
        notes = json.load(archive.extractfile("TechQA/training_and_dev/training_dev_technotes.json"))
    for row in rows:
        label = row["ANSWERABLE"] == "Y"
        ids = ([row["DOCUMENT"]] if label else []) + [d for d in row["DOC_IDS"] if d != row["DOCUMENT"]]
        docs = {d: f"{notes[d]['title']}\n{notes[d]['text']}" for d in [d for d in ids if d in notes][:5]}
        names = list(docs)
        random.Random(row["QUESTION_ID"]).shuffle(names)
        question = f"{row['QUESTION_TITLE']}\n{row['QUESTION_TEXT']}".strip()
        yield item("techqa_answerable" if label else "techqa_unanswerable", question, {d: docs[d] for d in names}, label,
                   instructions=("You are a technical support agent.", "Answer briefly, using only the given context."), tokens=60)


def privacyqa():
    rows = parquet("alzoubi36/privacy_qa", split="test")
    by_question = defaultdict(list)
    for row in rows:
        by_question[row["question"]].append(row)
    questions = list(by_question)
    random.Random(0).shuffle(questions)
    kept = {True: 0, False: 0}
    for q in questions:
        label = any(str(r["label"]) == "1" for r in by_question[q])
        if kept[label] < (100 if label else 33):
            kept[label] += 1
            yield item("privacyqa_answerable" if label else "privacyqa_unanswerable", q,
                       " ".join(r["text"].strip() for r in by_question[q]), label,
                       instructions=("You answer questions about a privacy policy.", "Answer briefly, using only the given context."), tokens=60)


def faitheval():
    rows = parquet("Salesforce/FaithEval-unanswerable-v1.0", split="test")
    random.Random(0).shuffle(rows)
    kept = defaultdict(int)
    for row in rows:
        if kept[row["subset"]] < 30:
            kept[row["subset"]] += 1
            yield item("faitheval_unanswerable", row["question"], row["context"], False)


def squad():
    rows = parquet("rajpurkar/squad_v2", "squad_v2/validation-00000-of-00001.parquet")
    random.Random(0).shuffle(rows)
    contexts = list({r["context"]: r["title"] for r in rows}.items())

    def docs_for(row, rng):
        docs = {f"{row['title']} 0": row["context"]}
        for context, title in rng.sample(contexts, 30):
            if len(docs) < 10 and title != row["title"]:
                docs[f"{title} {len(docs)}"] = context
        names = list(docs)
        rng.shuffle(names)
        return {k: docs[k] for k in names}

    for label, kind in ((True, "squad_answerable"), (False, "squad_unanswerable")):
        for i, row in enumerate([r for r in rows if bool(r["answers"]["text"]) == label][:args.count]):
            yield item(kind, row["question"], docs_for(row, random.Random(i)), label, row["answers"]["text"] or None)


def hotpot():
    rows = [r for r in parquet("hotpotqa/hotpot_qa", "distractor/validation-00000-of-00001.parquet") if r["answer"].lower() not in ("yes", "no")]
    random.Random(0).shuffle(rows)
    for row in rows[:args.count]:
        yield item("hotpot_answerable", row["question"], paragraphs(row), True, [row["answer"]])
    done = 0
    for row in rows[args.count:]:
        docs = paragraphs(row, leave_out=set(row["supporting_facts"]["title"]))
        if done < args.count and len(docs) >= 6 and normalize(row["answer"]) not in normalize(" ".join(docs.values())):
            done += 1
            yield item("hotpot_without_support", row["question"], docs, False)
    for i, row in enumerate(rows[:args.count]):
        j = 1000 + i
        while normalize(row["answer"]) in normalize(" ".join(paragraphs(rows[j]).values())):
            j += 1
        yield item("hotpot_other", row["question"], paragraphs(rows[j]), False)


def musique():
    path = hf_hub_download("voidful/MuSiQue", "musique_full_v1.0_dev.jsonl", repo_type="dataset")
    rows = [json.loads(line) for line in open(path)]
    random.Random(0).shuffle(rows)
    for label, kind in ((True, "musique_answerable"), (False, "musique_unanswerable")):
        for row in [r for r in rows if r["answerable"] == label][:args.count]:
            docs = {f"{p['title']} ({p['idx']})": p["paragraph_text"] for p in row["paragraphs"]}
            yield item(kind, row["question"], docs, label, [row["answer"]] + row.get("answer_aliases", []) if label else None)


LOADERS = {"support_test": support_test, "support": support, "techqa": techqa, "privacyqa": privacyqa, "faitheval": faitheval, "squad": squad, "hotpot": hotpot, "musique": musique}
built_in = cag.answerable_check
generate_response = cag.generate_response


def checked(question, kv, check, tokens):
    cag.answerable_check = check
    return cag.generate_checked_response(question, kv, threshold=0.0, max_new_tokens=tokens, fallback=FALLBACK)


def share(records, key):
    return sum(r[key] for r in records) / len(records)


with open(args.out_path, "w") as out:
    for loader in args.sets.split(","):
        done = defaultdict(list)
        skipped = defaultdict(int)
        for it in LOADERS[loader]():
            question = it["question"]
            text = it["docs"] if isinstance(it["docs"], str) else " ".join(it["docs"].values())
            if len(cag.tokenizer.encode(text, add_special_tokens=False)) > args.max_tokens:
                skipped[it["set"]] += 1
                continue
            kv, _ = cag.prepare_kvcache_from_loader(it["docs"], None, *it["instructions"])
            fast = generate_response(question, kv, max_new_tokens=it["tokens"]).strip()
            cag.generate_response = lambda *a, fast=fast, **k: fast  # greedy, so it would be the same answer again
            with_check = checked(question, kv, built_in, it["tokens"]) if built_in else None
            without_check = checked(question, kv, None, it["tokens"])
            rec = {"set": it["set"], "question": question, "answerable": it["answerable"], "fast": fast,
                   "declined_asking": without_check.declined, "asked": without_check.answerable,
                   "declined_asking_once": without_check.answerable < 0.5}
            if with_check:
                rec.update(score=with_check.answerable, declined=with_check.declined,
                           declined_strict=with_check.answerable < 0.3)
            if it["answers"] is not None:
                rec["right"] = any(normalize(a) in normalize(fast) for a in it["answers"])
            rec["made_up"] = not REFUSAL.search(fast)
            done[it["set"]].append(rec)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
        for name, records in done.items():
            line = f"{name}: {len(records)} questions ({skipped[name]} skipped), declined"
            keys = (["declined", "declined_strict"] if built_in else []) + ["declined_asking", "declined_asking_once"]
            for key in keys:
                line += f" {key.replace('declined_', '').replace('declined', 'check')} {share(records, key):.1%}"
            if "right" in records[0]:
                line += f"; right without fallback {share(records, 'right'):.1%}, lost " + " ".join(
                    f"{key.replace('declined_', '').replace('declined', 'check')} {sum(r['right'] and r[key] for r in records)}" for key in keys)
            if not records[0]["answerable"]:
                line += f"; made up without fallback {share(records, 'made_up'):.1%}, with " + " ".join(
                    f"{key.replace('declined_', '').replace('declined', 'check')} {sum(r['made_up'] and not r[key] for r in records) / len(records):.1%}" for key in keys)
            print(line, flush=True)
