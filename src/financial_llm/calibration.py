from __future__ import annotations

import numpy as np


# Learned on the validation split for the selected neutral-aware r8 attention+MLP
# LoRA adapter. Temperature scaling changes probability sharpness, not argmax.
DEFAULT_LORA_TEMPERATURE = 1.365


def temperature_scale_probabilities(probs: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    values = np.asarray(probs, dtype=np.float64)
    was_vector = values.ndim == 1
    if was_vector:
        values = values.reshape(1, -1)

    clipped = np.clip(values, 1e-12, 1.0)
    logits = np.log(clipped) / float(temperature)
    logits = logits - logits.max(axis=1, keepdims=True)
    scaled = np.exp(logits)
    scaled = scaled / scaled.sum(axis=1, keepdims=True)

    if was_vector:
        return scaled[0]
    return scaled


def calibrate_lora_probabilities(
    probs: np.ndarray,
    temperature: float = DEFAULT_LORA_TEMPERATURE,
) -> np.ndarray:
    return temperature_scale_probabilities(probs, temperature)
