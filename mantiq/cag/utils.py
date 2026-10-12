from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
import importlib.util
import random
import json
from typing import List, Dict, Callable, Any, Optional
import torch

def can_quantize() -> bool:
    # bitsandbytes 4-bit quantization needs a CUDA GPU.
    return torch.cuda.is_available() and importlib.util.find_spec("bitsandbytes") is not None

def load_model(
    model_name: str,
    hf_token: Optional[str] = None,
    quantized: Optional[bool] = None,
    trust_remote_code: bool = False,
    model_kwargs: Optional[dict] = None,
):
    if quantized is None:
        quantized = can_quantize()
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,              # Load model in 4-bit precision
    bnb_4bit_quant_type="nf4",      # Normalize float 4 quantization
    bnb_4bit_compute_dtype=torch.float16,  # Compute dtype for 4-bit base matrices
    bnb_4bit_use_double_quant=True  # Use nested quantization
    ) if quantized else None

    model_kwargs = model_kwargs or {}
    # The tokenizer comes from the same revision and place as the model.
    tokenizer_kwargs = {k: model_kwargs[k] for k in ("revision", "cache_dir", "local_files_only") if k in model_kwargs}
    tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=hf_token,
    use_fast= True,
    **tokenizer_kwargs
    )

    kwargs = dict(
        quantization_config=bnb_config,
        device_map="auto",          # Automatically choose best device
        trust_remote_code=trust_remote_code,  # Runs code from the model repo; only enable for repos you trust
        token=hf_token
    )
    if "torch_dtype" not in model_kwargs:  # the older name, which dtype would override
        # The checkpoint's dtype (usually bfloat16) rather than float32: half the memory, and
        # faster answers even on CPU, though float32 can prepare the knowledge faster there.
        kwargs["dtype"] = "auto"
    kwargs.update(model_kwargs)

    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    return model, tokenizer

class GenericDatasetProcessor:
    def __init__(self, random_seed: Optional[int] = None):
        self.random_seed = random_seed
        self._dataset = None
        if random_seed is not None:
            random.seed(random_seed)

    def parse_dataset(self, dataset : List , parser_function: Callable, **kwargs) -> List[Dict[str, Any]]:
        return parser_function(dataset, **kwargs)

class DatasetParser:
    @staticmethod
    def default_parser(dataset: List[Dict[str, Any]], max_entries: Optional[int] = None, key_mapping: Optional[Dict[str, str]] = None):
        key_mapping = key_mapping or {"question": "question", "answer": "answer", "context": "context"}
        return [
            {
                "question": entry.get(key_mapping["question"]),
                "answer": entry.get(key_mapping["answer"]),
                "context": entry.get(key_mapping["context"]),
            }
            for entry in dataset[:max_entries]
        ]
