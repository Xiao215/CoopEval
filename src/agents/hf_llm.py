"""Local Hugging Face model runner used by :class:`Agent` implementations."""

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging as hf_logging

from config import MODEL_WEIGHTS_DIR
from src.agents.base import LLM

# Suppress HF warnings
hf_logging.set_verbosity_error()
torch.set_float32_matmul_precision("high")


class HFInstance(LLM):
    """A class to manage a Hugging Face LLM pipeline that can be moved between CPU and GPU."""

    def __init__(self, model_name: str):
        self.model_path = MODEL_WEIGHTS_DIR / model_name

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,  # use float16 for better performance on GPU
            device_map="auto",  # automatically shards layers across GPU/CPU
            offload_folder="hf_offload",
            offload_state_dict=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.eos_token_id

        # Gemma models have some issue with cache in long generation, temporarily disable it
        self.use_cache = False if "gemma" in model_name else True

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        """Invoke the model with the given messages."""

        messages = [
            {"role": "user", "content": prompt},
        ]
        prompt_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)
        prompt_len = prompt_ids.shape[1]
        out_ids = self.model.generate(
            input_ids=prompt_ids,
            use_cache=self.use_cache,
            **kwargs,
        )
        gen_id = out_ids[:, prompt_len:]
        response = self.tokenizer.batch_decode(
            gen_id,
            skip_special_tokens=True,
        )[0]
        return response.strip()
