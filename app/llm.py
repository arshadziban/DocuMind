# huggingface causal lm wrapper with gpu acceleration

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import LLM_MODEL_NAME, LLM_MAX_NEW_TOKENS, LLM_TEMPERATURE, DEVICE


class LLM:
    # loads a causal language model and generates text from prompts

    def __init__(self, model_name: str = LLM_MODEL_NAME):
        print(f"Loading LLM on [{DEVICE.upper()}]...")

        self.device = DEVICE
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # some models don't ship with a pad token — use eos as fallback
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )

        if self.device == "cpu":
            self.model.to(self.device)

        print(f"LLM '{model_name}' loaded successfully.")

    def generate(self, prompt: str, max_tokens: int = LLM_MAX_NEW_TOKENS) -> str:
        # generate a response, returning only the newly generated text
        inputs = self.tokenizer(prompt, return_tensors="pt")

        if self.device == "cuda":
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=LLM_TEMPERATURE,
                do_sample=True,
            )

        # strip the input tokens so we only return the model's answer
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
