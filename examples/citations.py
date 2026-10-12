"""Show which documents each answer comes from, and flag answers that don't come from the knowledge.

    python examples/citations.py
"""

from mantiq import NaiveCAG

DOCUMENTS = {
    "tv.md": "The Vision 55 is a 55-inch 4K TV with HDR10 and smart apps. It costs $499 with a 2-year warranty.",
    "fridge.md": "The FrostFree is a 450-liter refrigerator with an A++ energy rating. It costs $899.",
    "washer.md": "The CleanWave is a 9 kg front-load washing machine with AI-powered wash cycles. It costs $649.",
    "delivery.md": "Delivery takes 2-4 business days. It is free for orders over $300, otherwise $25.",
    "returns.md": "Returns are accepted within 14 days with the receipt. Refunds go to the original payment method.",
}

QUESTIONS = [
    "How long until my order arrives?",
    "Which fridge do you have and is it efficient?",
    "Can I send back a TV I don't like?",
    "Do you sell gaming consoles?",  # not in the knowledge
]

cag = NaiveCAG("Qwen/Qwen3-0.6B", chat_template_kwargs={"enable_thinking": False})
kv_cache, _ = cag.prepare_kvcache_from_loader(
    DOCUMENTS,
    prompt_instruction="You are a sales agent for an electronics store.",
    answer_instruction="Answer in one sentence, using only the given context.",
)

for question in QUESTIONS:
    answer = cag.generate_response(question, kv_cache, max_new_tokens=60)
    cited = cag.cite(question, answer, kv_cache, top_k=2)
    sources = ", ".join(f"{c.document} ({c.score:.2f})" for c in cited.citations) or "none"
    print(f"Q: {question}\nA: {answer}\n   sources: {sources}\n   supported: {cited.supported:.2f}\n")
