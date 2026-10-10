import os
import pickle
import time
import warnings
from typing import Optional

import torch
from jinja2.exceptions import TemplateError
from transformers.cache_utils import DynamicCache
from ..cag.utils import load_model, GenericDatasetProcessor, DatasetParser

KV_CACHE_FILENAME = "kv_cache.pt"
_CACHE_FORMAT = "mantiq.kv_cache.v1"
# Marks where the question goes when the rendered chat prompt is split into the
# cached knowledge prefix and the per-query suffix.
_QUESTION_PLACEHOLDER = "<<mantiq:question>>"


class NaiiveCAG:
    def __init__(
        self,
        model_name: str,
        hf_token: str,
        quantized: bool = True,
        trust_remote_code: bool = False,
        chat_template_kwargs: Optional[dict] = None,
    ):
        """
        Initialize NaiiveCAG.
        Args:
            model_name: HuggingFace model name.
            hf_token: HuggingFace API token.
            quantized: Whether to load the model in quantized mode.
            trust_remote_code: Whether to run custom modeling code from the model repo.
                Only enable it for repos you trust.
            chat_template_kwargs: Extra variables for the tokenizer's chat template,
                e.g. {"enable_thinking": False} for Qwen3.
        """
        self.model_name = model_name
        self.model, self.tokenizer = load_model(model_name, hf_token, quantized, trust_remote_code)
        self.processor = GenericDatasetProcessor()  # Initialize processor
        self.prompt_instruction = None
        self.chat_template_kwargs = chat_template_kwargs or {}
        # The text after the question doesn't depend on the knowledge, so a cache
        # built in another session can be queried without its instructions.
        _, self._question_suffix = self._render_prompt("", "", "")

    def prepare_kvcache(self, dataset, saving_path, prompt_instruction: str, answer_instruction: str = None):
        """
        Prepare the KV cache from a question/answer dataset.
        Args:
            dataset: List of dicts with "question", "answer" and optional "context" keys.
            saving_path: Directory to save kv_cache.pt in (created if missing), or None to skip saving.
            prompt_instruction: Instruction for the system.
            answer_instruction: Instruction for the assistant.
        Returns:
            Tuple[DynamicCache, float]: KV cache and preparation time.
        """
        # Load and parse the dataset
        dataset = self.processor.parse_dataset(dataset, parser_function=DatasetParser.default_parser)

        # Combine all contexts
        documents = "\n\n".join(
            f"Question: {entry['question']}\nAnswer: {entry['answer']}" +
            (f"\nContext: {entry['context']}" if entry['context'] else "") +
            "\n"
            for entry in dataset
        )
        return self._prepare(documents, saving_path, prompt_instruction, answer_instruction)

    def prepare_kvcache_from_loader(self, loaded_data, saving_path, prompt_instruction: str, answer_instruction: str = None):
        """
        Prepare and save a key-value (KV) cache using the provided loaded data and instructions.

        This function generates a KV cache based on the provided `loaded_data`, formats it using
        `prompt_instruction` and optional `answer_instruction`, and saves the result to a specified path.

        Args:
            loaded_data (str): The dataset or context loaded from a source to be used in the KV cache.
            saving_path (str): The directory where the KV cache will be saved (created if missing),
            or None to skip saving.
            prompt_instruction (str): Instruction or context for the system to format the prompt.
            answer_instruction (str, optional): Instruction for generating concise responses. Defaults to
            "Provide a concise and relevant answer."
        Returns:
            Tuple[DynamicCache, float]:
            - `DynamicCache`: The prepared KV cache object.
            - `float`: The time taken to prepare the KV cache, in seconds.
        """
        return self._prepare(loaded_data, saving_path, prompt_instruction, answer_instruction)

    def _prepare(self, context: str, saving_path, prompt_instruction: str, answer_instruction: str):
        self.prompt_instruction = prompt_instruction or "You are an assistant for giving short answers."
        answer_instruction = answer_instruction or "Provide a concise and relevant answer."

        prompt, _ = self._render_prompt(context, self.prompt_instruction, answer_instruction)
        t1 = time.time()
        kv = self.preprocess_knowledge(prompt)

        if saving_path is not None:
            self.write_kv_cache(kv, os.path.join(saving_path, KV_CACHE_FILENAME))
        t2 = time.time()
        return kv, t2 - t1

    def _render_prompt(self, context: str, prompt_instruction: str, answer_instruction: str):
        """
        Render the knowledge prompt with the model's chat template.
        Returns (prefix, suffix): the prefix ends where the question goes and is what gets
        cached; the suffix closes the user turn and opens the assistant turn.
        """
        user_content = (
            "Context:\n"
            "------------------------------------------------\n"
            f"{context}\n"
            "------------------------------------------------\n\n"
            f"{answer_instruction}\n\n"
            f"Question:\n{_QUESTION_PLACEHOLDER}"
        )
        prompt = self._apply_chat_template(prompt_instruction, user_content)
        prefix, suffix = prompt.rsplit(_QUESTION_PLACEHOLDER, 1)
        return prefix, suffix

    def _apply_chat_template(self, system: str, user: str) -> str:
        if self.tokenizer.chat_template is None:
            return f"{system}\n\n{user}\n\nAnswer:\n"
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **self.chat_template_kwargs
            )
        except TemplateError:
            # Some templates (e.g. Gemma) reject the system role; fold it into the user turn.
            messages = [{"role": "user", "content": f"{system}\n\n{user}"}]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **self.chat_template_kwargs
            )

    def preprocess_knowledge(self, prompt: str) -> DynamicCache:
        """
        Preprocess knowledge and return KV cache.
        """
        input_ids = self.tokenizer.encode(
            prompt,
            add_special_tokens=self.tokenizer.chat_template is None,  # chat templates already add BOS
            return_tensors="pt",
        ).to(self._input_device())
        past_key_values = DynamicCache()
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False
            )
        return outputs.past_key_values

    def write_kv_cache(self, kv: DynamicCache, path: str):
        """
        Save the KV cache as plain tensors, so it loads with torch.load(weights_only=True).
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"format": _CACHE_FORMAT, "model_name": self.model_name, "layers": _cache_to_layers(kv)}, path)

    def read_kv_cache(self, path: str) -> DynamicCache:
        """
        Load a KV cache saved by write_kv_cache.
        """
        try:
            data = torch.load(path, map_location=self._cache_map_location(), weights_only=True)
        except pickle.UnpicklingError as e:
            raise ValueError(
                f"{path} is not a mantiq KV cache file. Caches saved by mantiq 0.1 must be rebuilt with prepare_kvcache."
            ) from e
        if not isinstance(data, dict) or data.get("format") != _CACHE_FORMAT:
            raise ValueError(f"{path} is not a mantiq KV cache file.")
        if data["model_name"] != self.model_name:
            warnings.warn(f"{path} was built with {data['model_name']}, but this instance uses {self.model_name}.")
        return _layers_to_cache(data["layers"])

    def generate(self, input_query: str, past_key_values: DynamicCache, max_new_tokens: int = 100) -> torch.Tensor:
        """
        Generate a response using the KV cache.
        The cache is cropped back to its original length afterwards, so it can be reused for the next query.
        """
        device = self._input_device()
        input_ids = self.tokenizer.encode(
            input_query.strip() + self._question_suffix, add_special_tokens=False, return_tensors="pt"
        ).to(device)
        eos_token_ids = self._eos_token_ids()
        origin_len = past_key_values.get_seq_length()

        output_ids = input_ids.new_empty((1, 0))
        next_token = input_ids

        try:
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    outputs = self.model(
                        input_ids=next_token,
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                    next_token_logits = outputs.logits[:, -1, :]
                    next_token = next_token_logits.argmax(dim=-1).unsqueeze(-1)
                    next_token = next_token.to(device)

                    output_ids = torch.cat([output_ids, next_token], dim=1)

                    if next_token.item() in eos_token_ids:
                        break
        finally:
            # Drop the query and answer so the next query only sees the knowledge. A negative
            # crop removes that many tokens on every supported transformers version.
            tokens_to_remove = past_key_values.get_seq_length() - origin_len
            if tokens_to_remove > 0:
                past_key_values.crop(-tokens_to_remove)
        return output_ids

    def generate_response(self, query: str, past_key_values, max_new_tokens: int = 100) -> str:
        """
        Generate a response to the query.
        """
        output = self.generate(query, past_key_values, max_new_tokens)
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return response

    def _input_device(self) -> torch.device:
        return self.model.get_input_embeddings().weight.device

    def _cache_map_location(self):
        # A model sharded across devices keeps each layer's cache on the device it was saved from.
        if len(set((getattr(self.model, "hf_device_map", None) or {}).values())) > 1:
            return None
        return self._input_device()

    def _eos_token_ids(self) -> set:
        eos = self.model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else list(eos or [])
        if self.tokenizer.eos_token_id is not None:
            eos.append(self.tokenizer.eos_token_id)
        return set(eos)


def _cache_to_layers(kv: DynamicCache) -> list:
    if hasattr(kv, "layers"):  # transformers >= 4.56
        return [[layer.keys, layer.values] for layer in kv.layers]
    return [[keys, values] for keys, values in zip(kv.key_cache, kv.value_cache)]


def _layers_to_cache(layers: list) -> DynamicCache:
    cache = DynamicCache()
    for layer_idx, (keys, values) in enumerate(layers):
        cache.update(keys, values, layer_idx)
    return cache
