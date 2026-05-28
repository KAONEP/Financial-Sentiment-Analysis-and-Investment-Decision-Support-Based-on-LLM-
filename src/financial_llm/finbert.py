from __future__ import annotations

import os

import numpy as np
import torch

os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .labels import LABELS


class FinBertSentiment:
    def __init__(self, model_name: str = "ProsusAI/finbert", device: str | None = None):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            use_safetensors=False,
        )
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        self.id2label = {
            int(idx): str(label).lower()
            for idx, label in self.model.config.id2label.items()
        }

    def predict_proba(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        outputs = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            outputs.append(self._reorder_probs(probs))
        return np.concatenate(outputs, axis=0)

    def _reorder_probs(self, probs: np.ndarray) -> np.ndarray:
        reordered = np.zeros((probs.shape[0], len(LABELS)), dtype=np.float32)
        for model_idx, model_label in self.id2label.items():
            if model_label not in LABELS:
                continue
            target_idx = LABELS.index(model_label)
            reordered[:, target_idx] = probs[:, model_idx]
        row_sums = reordered.sum(axis=1, keepdims=True)
        if np.any(row_sums == 0):
            raise ValueError(f"Could not map model labels {self.id2label} to {LABELS}")
        return reordered / row_sums
