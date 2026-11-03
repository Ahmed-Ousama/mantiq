"""Put a support bot in front of customers: six businesses, 240 questions, mantiq against RAG.

The businesses are in examples/business_bench: three in Egyptian Arabic and three in English, each with
its documents and 40 customer questions, 24 that the documents answer (with a reference answer) and 16
that they don't. Four systems answer every question with the same model:

- rag: finds the 3 sections closest to the question with an embedding model (BAAI/bge-m3) and reads only those.
- rag+checks: the same sections, through generate_checked_response() with the business's fallback.
- mantiq: reads all the documents once and answers from the cache.
- mantiq+checks: the same cache, through generate_checked_response() with the fallback.

A bigger model then judges every answer: against the reference when the documents answer the question,
and otherwise whether the answer says it doesn't know instead of making something up. The table counts,
per 100 customer questions, the right answers, the wrong or made-up answers that reach the customer, the
questions handed to a person, and the replies that only say the information isn't available.

    pip install sentence-transformers
    python examples/business_benchmark.py answers.jsonl
"""

import argparse
import gc
import json
import re
import statistics
import time
from pathlib import Path

import torch

from mantiq import NaiveCAG

parser = argparse.ArgumentParser()
parser.add_argument("out_path")
parser.add_argument("--businesses", default="dental,internet,phones,bank,saas,airline")
parser.add_argument("--model", default="Qwen/Qwen3-4B")
parser.add_argument("--dtype", default=None, help="e.g. float16 to load the model unquantized; 4-bit on a GPU otherwise")
parser.add_argument("--judge", default="Qwen/Qwen3-14B")
parser.add_argument("--embedder", default="BAAI/bge-m3")
parser.add_argument("--top-k", type=int, default=3)
parser.add_argument("--judge-only", action="store_true", help="judge the answers already in out_path")
args = parser.parse_args()
HERE = Path(__file__).parent / "business_bench"
SYSTEMS = ["rag", "rag+checks", "mantiq", "mantiq+checks"]
DONT_KNOW = "said it doesn't know"


def load(name):
    spec = json.loads((HERE / f"{name}.json").read_text())
    text = (HERE / f"{name}.md").read_text()
    spec["docs"] = {s.split("\n", 1)[0].strip(): s.strip() for s in re.split(r"^## ", text, flags=re.M)[1:]}
    return spec


def answer_everything():
    from sentence_transformers import SentenceTransformer

    dtype = {"quantized": False, "model_kwargs": {"dtype": getattr(torch, args.dtype)}} if args.dtype else {}
    cag = NaiveCAG(args.model, chat_template_kwargs={"enable_thinking": False}, **dtype)
    embedder = SentenceTransformer(args.embedder, device="cuda" if torch.cuda.is_available() else "cpu")
    cag.generate_response("Hi", cag.prepare_kvcache_from_loader("Hello.")[0], max_new_tokens=2)  # warm up

    def first_word_and_answer(question, cache):
        start, first, text = time.perf_counter(), None, ""
        for piece in cag.stream_response(question, cache, max_new_tokens=120):
            first = first if first is not None else time.perf_counter() - start
            text += piece
        return text.strip(), first, time.perf_counter() - start

    with open(args.out_path, "w") as out:
        for name in args.businesses.split(","):
            spec = load(name)
            instructions = spec["prompt"], spec["answer"]
            titles = list(spec["docs"])
            vectors = embedder.encode([spec["docs"][t] for t in titles], normalize_embeddings=True)
            kv, read_seconds = cag.prepare_kvcache_from_loader(spec["docs"], None, *instructions)
            print(f"{name}: {kv.get_seq_length()} tokens of knowledge, read once in {read_seconds:.1f} s", flush=True)
            for item in spec["questions"]:
                q = item["q"]
                base = {"business": name, "language": spec["language"], "question": q, "answerable": item["answerable"],
                        "kind": item["kind"], "reference": item.get("reference")}

                start = time.perf_counter()
                scores = vectors @ embedder.encode(q, normalize_embeddings=True)
                found = [titles[i] for i in scores.argsort()[::-1][:args.top_k]]
                rag_kv, _ = cag.prepare_kvcache_from_loader({t: spec["docs"][t] for t in found}, None, *instructions)
                setup = time.perf_counter() - start
                text, first, total = first_word_and_answer(q, rag_kv)
                records = [{"system": "rag", "answer": text, "first_word": setup + first, "seconds": setup + total, "found": found}]
                text, first, total = first_word_and_answer(q, kv)
                records.append({"system": "mantiq", "answer": text, "first_word": first, "seconds": total})
                for system, cache, extra in (("rag+checks", rag_kv, setup), ("mantiq+checks", kv, 0.0)):
                    start = time.perf_counter()
                    checked = cag.generate_checked_response(q, cache, max_new_tokens=120, fallback=spec["fallback"])
                    records.append({"system": system, "answer": checked.answer, "declined": checked.declined,
                                    "rethought": checked.rethought, "seconds": extra + time.perf_counter() - start})
                for record in records:
                    out.write(json.dumps({**base, **record}, ensure_ascii=False) + "\n")
                out.flush()
            print(f"{name}: done", flush=True)
    del cag, embedder
    gc.collect()
    torch.cuda.empty_cache()


SAME = """You check a customer service agent's answer against the right answer. The texts may be in Arabic.
The answer is right if it gives the same key fact as the right answer: the same number, the same yes or no, the
same name. It doesn't have to repeat every detail. It is wrong if it gives a different number or the opposite yes
or no, or if any sentence in it contradicts the right answer, even when another sentence agrees with it.

Examples:
Question: Can I pay for whitening in installments?
Right answer: No, installments are for treatments of 5,000 or more, and whitening costs 3,500.
Answer: Yes, you can pay for whitening in installments. It costs 3,500.
Verdict: no

Question: How long is the warranty on a zirconia crown?
Right answer: 5 years, with a checkup every 6 months.
Answer: The zirconia crown has a 5-year warranty.
Verdict: yes

Question: How much does a user save per year by paying annually?
Right answer: $60.
Answer: Paying annually saves $228 a year. The annual price is $60 less per year.
Verdict: no

Now this one:
Question: {q}
Right answer: {ref}
Answer: {resp}
Is the answer right? Reply with yes or no."""
MADE_UP = """You check whether a customer service agent answered a customer's question or not. The texts may be in
Arabic. The agent answered if it states an answer to the question as a fact: a number, a yes or no, a name, a
policy. It did not if it says the information isn't given, says it isn't sure, passes the question on, or only
repeats related facts without answering the question itself.

Examples:
Question: Do you sell gaming consoles?
Answer: No, we don't sell gaming consoles.
Verdict: yes

Question: What is your customer service number?
Answer: Our customer service number is 920000000.
Verdict: yes

Question: Is the smart watch waterproof?
Answer: The watch has a one-year warranty, which doesn't cover water damage, so be careful with water.
Verdict: no

Question: Does single sign-on work with Okta?
Answer: Okta isn't mentioned, but single sign-on through SAML is available on the Business plan.
Verdict: no

Now this one:
Question: {q}
Answer: {resp}
Did the agent answer the question? Reply with yes or no."""


def judge_everything():
    records = [json.loads(line) for line in open(args.out_path)]
    judge = NaiveCAG(args.judge, chat_template_kwargs={"enable_thinking": False}, answer_check=False)
    kv, _ = judge.prepare_kvcache_from_loader("The responses below come from a company's customer service agent.", None,
                                              "You grade the answers of a customer service agent.", "")
    for r in records:
        if r.get("declined"):
            r["verdict"] = "handed off"
            continue
        if r["answerable"] and judge._yes_probability(SAME.format(q=r["question"], ref=r["reference"], resp=r["answer"]), kv) >= 0.5:
            r["verdict"] = "right"
            continue
        made_up = judge._yes_probability(MADE_UP.format(q=r["question"], resp=r["answer"]), kv) >= 0.5
        r["verdict"] = ("wrong" if r["answerable"] else "made up") if made_up else DONT_KNOW
    with open(args.out_path, "w") as out:
        for r in records:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    return records


def table(records, title):
    print(f"\n===== {title}", flush=True)
    print("Per 100 customer questions: right answers, wrong or made-up answers, handed to a person, only said it doesn't "
          "know. Then, of the answerable questions: right / wrong / to a person; of the rest: made up / safe. Then the "
          "median time to the first word and to the full reply.")
    for system in SYSTEMS:
        rs = [r for r in records if r["system"] == system]
        if not rs:
            continue
        n = len(rs)
        count = lambda v, sub=rs: sum(r["verdict"] == v for r in sub)
        ans = [r for r in rs if r["answerable"]]
        rest = [r for r in rs if not r["answerable"]]
        first = [r["first_word"] for r in rs if "first_word" in r]
        line = (f"{system:<15}{100 * count('right') / n:>7.1f}{100 * (count('wrong') + count('made up')) / n:>18.1f}"
                f"{100 * count('handed off') / n:>13.1f}{100 * count(DONT_KNOW) / n:>14.1f}   "
                f"{count('right', ans) / len(ans):.0%} / {count('wrong', ans) / len(ans):.0%} / {count('handed off', ans) / len(ans):.0%}"
                f"  | {count('made up', rest) / len(rest):.0%} / {1 - count('made up', rest) / len(rest):.0%}  | ")
        line += (f"{statistics.median(first):.2f} s, " if first else "      ") + f"{statistics.median(r['seconds'] for r in rs):.1f} s"
        print(line, flush=True)


if not args.judge_only:
    answer_everything()
records = judge_everything()
table(records, "all six businesses")
for language, label in (("ar", "Arabic businesses"), ("en", "English businesses")):
    table([r for r in records if r["language"] == language], label)
print("\nEvery wrong or made-up answer, for reading:")
for r in records:
    if r["verdict"] in ("wrong", "made up"):
        print(f"[{r['system']}] {r['business']} | {r['question']} -> {r['answer'][:160]}".replace("\n", " "))
