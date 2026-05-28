from __future__ import annotations

import numpy as np


def threshold_fusion(
    llm_probs: np.ndarray,
    finbert_probs: np.ndarray,
    threshold: float,
) -> np.ndarray:
    llm_conf = llm_probs.max(axis=1)
    use_llm = llm_conf >= threshold
    fused = finbert_probs.copy()
    fused[use_llm] = llm_probs[use_llm]
    return fused


def weighted_fusion(llm_probs: np.ndarray, finbert_probs: np.ndarray) -> np.ndarray:
    alpha = llm_probs.max(axis=1, keepdims=True)
    fused = alpha * llm_probs + (1.0 - alpha) * finbert_probs
    return fused / fused.sum(axis=1, keepdims=True)

