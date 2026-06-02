from __future__ import annotations

import numpy as np


STACKING_COEF = np.asarray(
    [
        [2.2311336015060133, -1.9188413009160274, -0.9378915198162648, 1.8097661891538577, -0.7227668027333514, -1.712599125909112],
        [-0.8062325764657692, 2.1752129317615747, -0.6630809579201464, 0.00010295590681207341, 0.7932491455716224, -0.0874526526850634],
        [-1.4249010250402463, -0.25637163084555403, 1.6009724777364103, -1.8098691450606645, -0.07048234283827776, 1.8000517785941728],
    ],
    dtype=np.float64,
)
STACKING_INTERCEPT = np.asarray(
    [-0.838091543521199, 0.9592323115052873, -0.1211407679841017],
    dtype=np.float64,
)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


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


def logistic_stacking_fusion(llm_probs: np.ndarray, finbert_probs: np.ndarray) -> np.ndarray:
    features = np.hstack([finbert_probs, llm_probs]).astype(np.float64)
    logits = features @ STACKING_COEF.T + STACKING_INTERCEPT
    return _softmax(logits)
