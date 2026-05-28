from __future__ import annotations

import math
from contextlib import nullcontext

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .labels import LABELS
from .prompts import SYSTEM_PROMPT, direct_prompt, neutral_aware_prompt, reasoning_label_prompt


class LabelScoringLLM:
    def __init__(
        self,
        model_name: str,
        dtype: str = "bfloat16",
        enable_thinking: bool = False,
        max_length: int = 512,
        prompt_mode: str = "direct",
        adapter_path: str | None = None,
    ):
        self.model_name = model_name
        self.enable_thinking = enable_thinking
        self.max_length = max_length
        if prompt_mode not in {"direct", "reasoning", "neutral_aware"}:
            raise ValueError(f"Unsupported prompt_mode: {prompt_mode!r}")
        self.prompt_mode = prompt_mode
        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=False,
        )
        if adapter_path is not None:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.label_token_ids = [
            self.tokenizer(label, add_special_tokens=False).input_ids
            for label in LABELS
        ]
        self.single_token_labels = all(len(token_ids) == 1 for token_ids in self.label_token_ids)

    def predict_proba(self, texts: list[str], batch_size: int = 8) -> np.ndarray:
        if self.single_token_labels:
            probs = []
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                probs.append(self._score_single_token_batch(batch))
            return np.concatenate(probs, axis=0)

        probs = [self._score_one(text) for text in texts]
        return np.asarray(probs, dtype=np.float32)

    def _score_single_token_batch(self, sentences: list[str]) -> np.ndarray:
        prompt_texts = [self._format_chat(self._build_prompt(sentence)) for sentence in sentences]
        inputs = self.tokenizer(
            prompt_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.model.device)
        label_ids = torch.tensor(
            [token_ids[0] for token_ids in self.label_token_ids],
            device=self.model.device,
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits.float()
            last_positions = inputs["attention_mask"].sum(dim=1) - 1
            batch_indices = torch.arange(logits.shape[0], device=self.model.device)
            label_logits = logits[batch_indices, last_positions][:, label_ids]
            probs = torch.softmax(label_logits, dim=-1).detach().cpu().numpy()
        return probs.astype(np.float32)

    def _score_one(self, sentence: str) -> np.ndarray:
        prompt = self._build_prompt(sentence)
        prompt_text = self._format_chat(prompt)
        if self.single_token_labels:
            return self._score_single_token_labels(prompt_text)

        scores = []
        for label in LABELS:
            scores.append(self._completion_logprob(prompt_text, label))
        return _softmax(scores)

    def _score_single_token_labels(self, prompt_text: str) -> np.ndarray:
        input_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids.to(self.model.device)
        label_ids = torch.tensor(
            [token_ids[0] for token_ids in self.label_token_ids],
            device=self.model.device,
        )
        with torch.no_grad():
            next_token_logits = self.model(input_ids=input_ids).logits[0, -1, :].float()
            label_logits = next_token_logits[label_ids]
            probs = torch.softmax(label_logits, dim=-1).detach().cpu().numpy()
        return probs.astype(np.float32)

    def _build_prompt(self, sentence: str) -> str:
        if self.prompt_mode == "reasoning":
            return reasoning_label_prompt(sentence)
        if self.prompt_mode == "neutral_aware":
            return neutral_aware_prompt(sentence)
        return direct_prompt(sentence)

    def _format_chat(self, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                enable_thinking=self.enable_thinking,
                **kwargs,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 180,
        max_input_tokens: int = 1536,
        disable_adapter: bool = True,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        try:
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                enable_thinking=self.enable_thinking,
                **kwargs,
            )
        except TypeError:
            prompt_text = self.tokenizer.apply_chat_template(messages, **kwargs)

        inputs = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=max_input_tokens,
            return_tensors="pt",
        ).to(self.model.device)

        adapter_context = (
            self.model.disable_adapter()
            if disable_adapter and isinstance(self.model, PeftModel)
            else nullcontext()
        )
        with adapter_context:
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )
        generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def _completion_logprob(self, prompt_text: str, completion: str) -> float:
        prompt_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids.to(self.model.device)
        completion_ids = self.tokenizer(
            completion,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.to(self.model.device)

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        with torch.no_grad():
            logits = self.model(input_ids=input_ids).logits

        prompt_len = prompt_ids.shape[1]
        total_logprob = 0.0
        token_count = completion_ids.shape[1]
        for i in range(token_count):
            token_position = prompt_len + i
            next_token_logits = logits[0, token_position - 1, :]
            log_probs = torch.log_softmax(next_token_logits, dim=-1)
            token_id = completion_ids[0, i]
            total_logprob += float(log_probs[token_id].detach().cpu())

        return total_logprob / max(1, token_count)


def _softmax(scores: list[float]) -> np.ndarray:
    max_score = max(scores)
    exps = [math.exp(score - max_score) for score in scores]
    denom = sum(exps)
    return np.asarray([value / denom for value in exps], dtype=np.float32)
