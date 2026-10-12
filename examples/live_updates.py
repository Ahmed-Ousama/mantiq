"""Keep a cache up to date when the knowledge changes, without rebuilding it.

    python examples/live_updates.py
"""

import time

from mantiq import NaiveCAG

KINDS = ["TV", "fridge", "washer", "laptop", "phone", "air conditioner", "microwave", "vacuum cleaner"]

# A catalog that rarely changes, and this week's offers, which change all the time: offers go last.
documents = {
    f"item-{100 + i}": f"Item {100 + i} is a {KINDS[i % len(KINDS)]} with a {1 + i % 5}-year warranty."
    for i in range(60)
}
documents["offers.md"] = "This week's offer: 10% off all TVs."

cag = NaiveCAG("Qwen/Qwen3-0.6B", chat_template_kwargs={"enable_thinking": False})
kv_cache, seconds = cag.prepare_kvcache_from_loader(
    documents,
    prompt_instruction="You are a sales agent for an electronics store.",
    answer_instruction="Answer in one sentence, using only the given context.",
)
print(f"Built the cache ({kv_cache.get_seq_length()} tokens) in {seconds:.1f}s")

question = "What is the offer this week?"
print("Before:", cag.generate_response(question, kv_cache, max_new_tokens=40))

documents["offers.md"] = "This week's offer: free delivery on all fridges."
start = time.perf_counter()
cag.update_kvcache(kv_cache, documents)
print(f"Updated the offer in {time.perf_counter() - start:.1f}s")
print("After: ", cag.generate_response(question, kv_cache, max_new_tokens=40))
