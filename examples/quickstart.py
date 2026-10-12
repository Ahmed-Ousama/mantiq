"""Answer customer questions from a small product catalog with mantiq.

    pip install mantiq
    python examples/quickstart.py

Runs on a CPU; with a CUDA GPU the model is loaded in 4-bit automatically.
"""

from mantiq import NaiveCAG

CATALOG = [
    {"question": "What are the features of the Vision 55 TV?", "answer": "It has 4K resolution, HDR10 and smart apps.", "context": "55-inch TV, $499, 2-year warranty."},
    {"question": "Does the FrostFree fridge save energy?", "answer": "Yes, it has an A++ energy rating.", "context": "450-liter refrigerator, $899, 5-year compressor warranty."},
    {"question": "What makes the CleanWave washer special?", "answer": "AI-powered wash cycles that adapt to the load.", "context": "9 kg front-load washing machine, $649."},
    {"question": "Do you deliver?", "answer": "Delivery is free for orders over $300, otherwise $25.", "context": "Delivery takes 2-4 business days."},
]

QUESTIONS = [
    "I want a 4K screen, what do you have?",
    "How long does delivery take?",
    "Which washing machine should I get?",
]

# Qwen3 thinks before answering by default; turn that off for short answers.
cag = NaiveCAG("Qwen/Qwen3-0.6B", chat_template_kwargs={"enable_thinking": False})

kv_cache, seconds = cag.prepare_kvcache(
    dataset=CATALOG,
    saving_path="./cache_directory",  # reload later with cag.read_kv_cache("./cache_directory/kv_cache.pt")
    prompt_instruction="You are a friendly sales agent for an electronics store.",
    answer_instruction="Answer in one or two sentences, using only the given context.",
)
print(f"Cached {kv_cache.get_seq_length()} tokens of knowledge in {seconds:.2f}s\n")

for question in QUESTIONS:
    print(f"Q: {question}\nA: ", end="")
    for chunk in cag.stream_response(question, kv_cache, max_new_tokens=80):
        print(chunk, end="", flush=True)
    print("\n")
