from abc import ABC, abstractmethod
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging as hf_logging

from config import MODEL_WEIGHTS_DIR

# Suppress HF warnings
hf_logging.set_verbosity_error()
torch.set_float32_matmul_precision("high")

torch.manual_seed(42)


class HFInstance:
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
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
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


class LLMManager:
    """A class to manage a Hugging Face LLM pipeline that can be moved between CPU and GPU."""

    def __init__(self) -> None:
        self.llms = dict()

    def get_llm(self, model_name: str) -> HFInstance:
        """Get an LLM instance for the given model name."""
        if model_name not in self.llms:
            self.llms[model_name] = HFInstance(model_name)
        return self.llms[model_name]


class Agent(ABC):
    """
    Abstract base class for an LLM-based agent.
    """

    llm_manager = LLMManager()

    def __init__(self, llm_config: dict) -> None:
        self.model_type = llm_config["model"]
        self.kwargs = llm_config.get("kwargs", {})
        self.pipeline = type(self).llm_manager.get_llm(self.model_type)

    @abstractmethod
    def chat(
        self,
        messages: str,
    ) -> str:
        """Chat with the agent using the provided messages."""
        raise NotImplementedError

    def invoke(self, messages: str) -> str:
        """Invoke the agent using the provided messages. No prompting added."""
        response = self.pipeline.invoke(messages, **self.kwargs)
        return response

    def __str__(self):
        return self.name

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the agent."""
        raise NotImplementedError


class IOAgent(Agent):
    """Input/Output Agent.
    This agent is designed to be the most basic llm agent. Given a message, answer it.
    """

    def chat(
        self,
        messages: str,
    ) -> str:
        """Chat with the agent using the provided messages."""
        messages += (
            "\nPlease ONLY provide the output to the above question."
            "DO NOT provide any additional text or explanation.\n"
        )
        response = self.pipeline.invoke(messages, **self.kwargs)
        return response

    @property
    def name(self) -> str:
        """Return the name of the agent."""
        return f"{self.model_type}(IO)"


class CoTAgent(Agent):
    """Chain-of-Thought Agent.

    This agent wraps the prompt to ask the LLM to think step-by-step.
    """

    def chat(
        self,
        messages: str,
    ) -> str:
        """Chat with the agent using the provided messages."""
        messages += """
        Think about the question step by step, break it down into small steps, explain your reasoning, and then provide the final answer.
        """
        response = self.pipeline.invoke(messages, **self.kwargs)
        return response

    @property
    def name(self) -> str:
        """Return the name of the agent."""
        return f"{self.model_type}(CoT)"
