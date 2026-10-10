# Mantiq

![Mantiq Overview](images/mantiq.png)


Mantiq is a sophisticated package designed for dynamic knowledge caching and conversational AI. The package builds on the core principles of knowledge caching and conversational AI, streamlining the integration and processing of KV caches for enhanced language model performance.

## Features

- **Efficient KV Caching**: Create, save, and retrieve key-value (KV) caches of your knowledge, and reuse them across queries and sessions.
- **Customizable Prompt Instructions**: Tailor system and user instructions for specific use cases.
- **Seamless Integration**: Built on top of HuggingFace Transformers, using each model's own chat template.
- **Dynamic Processing**: Parse and utilize datasets in JSON format for knowledge preparation.
- **Document Loaders**: Load knowledge from PDF, CSV, JSON, Markdown, HTML, web pages, directories and Azure Document Intelligence.

## Installation

```bash
pip install git+https://github.com/tactful-ai/mantiq.git
```

To use the document loaders, install the `loaders` extra (or `azure` for Azure Document Intelligence):

```bash
pip install "mantiq[loaders] @ git+https://github.com/tactful-ai/mantiq.git"
```

## Usage

### Initializing mantiq

```python
from mantiq.cag import NaiiveCAG

# HuggingFace token
hf_token = "your_hf_token_here"

# Initialize NaiiveCAG
cag = NaiiveCAG(
    model_name="meta-llama/Llama-3.2-1B-Instruct",  # Replace with your model name
    hf_token=hf_token,
    quantized=True  # Enable 4-bit quantization (needs a CUDA GPU)
)

```

Optional arguments:

- `trust_remote_code=True` runs custom modeling code from the model repo. It is off by default; only enable it for repos you trust.
- `chat_template_kwargs` passes extra variables to the model's chat template, e.g. `{"enable_thinking": False}` for Qwen3.

### Preparing the KV Cache

#### Dataset Example

```python
dataset = [
    {"question": "What are the features of this TV?", "answer": "This TV has 4K resolution and smart features.", "context": "55-inch TV with advanced display technology."},
    {"question": "Does this refrigerator save energy?", "answer": "Yes, it has an A++ energy rating.", "context": "Eco-friendly refrigerator with advanced cooling technology."},
    {"question": "What makes this washing machine special?", "answer": "It has AI-powered wash cycles for better cleaning.", "context": "Front load washing machine with AI technology."}
]
```

#### KV Cache Preparation
```python
# Define instructions
prompt_instruction = "You are a Sales agent responsible for answering customer questions and convincing them to buy products."
answer_instruction = "Answer with a very short convincing response (max 50 words) to earn the customer's trust, answer only from the given context"

# Prepare KV cache
saving_path = "./cache_directory"  # Created if it doesn't exist; pass None to keep the cache in memory only
kv_cache, prep_time = cag.prepare_kvcache(
    dataset=dataset,
    saving_path=saving_path,
    prompt_instruction=prompt_instruction,
    answer_instruction=answer_instruction
)
print(f"KV cache prepared in {prep_time:.2f} seconds.")

```

### Loading Knowledge from Documents

```python
from mantiq.loaders import LoaderFactory

# Supported types: "PDF", "CSV", "JSON", "Markdown", "HTML", "WebPage", "Directory", "MicrosoftFile"
text = LoaderFactory.get_loader("PDF", "./catalog.pdf")

kv_cache, prep_time = cag.prepare_kvcache_from_loader(
    loaded_data=text,
    saving_path="./cache_directory",
    prompt_instruction=prompt_instruction,
    answer_instruction=answer_instruction
)
```

Extra keyword arguments are passed to the underlying LangChain loader, e.g. `LoaderFactory.get_loader("JSON", "data.json", jq_schema=".items[].text")`.

### Reloading KV Cache
```python
# Reload previously saved KV cache, e.g. in a new session
kv_cache = cag.read_kv_cache("./cache_directory/kv_cache.pt")

```

A saved cache only works with the model that built it; mantiq warns if the model names differ.


### Generating Responses

```python
# Customer query
input_query = "i wanna buy 4K screen, any recommendations?"

# Generate response
response = cag.generate_response(
    query=input_query,
    past_key_values=kv_cache,  # KV cache prepared earlier
    max_new_tokens=100          # Maximum number of tokens for the response
)
print(response)

#  I'd be happy to help you find a 4K screen that fits your needs. Our 55-inch 4K TV is a top-of-the-line model with crystal-clear picture and HDR for an immersive viewing experience.


```

The KV cache is restored after each query, so every query sees only the knowledge and the same cache can answer any number of queries. Run one query at a time per cache.

### Limitations

- All the knowledge must fit in the model's context window.
- Supports models with standard attention caches (e.g. Llama, Qwen3). Hybrid linear-attention models are not supported.

## Development Background

mantiq is part of a long-term development plan by the [tactful team](https://tactful.ai/) focusing on advancements in Conversational AI and Knowledge Caching (CAG) techniques. This work is inspired by the research presented in the paper "Don't Do RAG: When Cache-Augmented Generation is All You Need for Knowledge Tasks," which you can access [here](https://arxiv.org/abs/2412.15605).

The original repository that laid the foundation for CAG can be found [here](https://github.com/hhhuang/CAG).

## Contributing

We welcome contributions! Please follow the standard GitHub workflow:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Submit a pull request

To run the tests:

```bash
pip install -e ".[loaders,test]"
pytest
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

For questions or support, feel free to open an issue in the repository.

