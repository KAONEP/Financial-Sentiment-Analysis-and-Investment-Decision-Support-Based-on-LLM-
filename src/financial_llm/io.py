from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .labels import LABELS


def load_split(
    path: str | Path,
    limit: int | None = None,
    sample_per_class: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    if sample_per_class is not None:
        parts = []
        for _, group in df.groupby("label"):
            n = min(sample_per_class, len(group))
            parts.append(group.sample(n=n, random_state=seed))
        df = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if limit is not None:
        df = df.head(limit).copy()
    return df


def probs_to_frame(df: pd.DataFrame, probs: np.ndarray, model_name: str) -> pd.DataFrame:
    out = df[["example_id", "sentence", "label"]].copy()
    for idx, label in enumerate(LABELS):
        out[f"prob_{label}"] = probs[:, idx]
    pred_idx = probs.argmax(axis=1)
    out["prediction"] = [LABELS[idx] for idx in pred_idx]
    out["confidence"] = probs.max(axis=1)
    out["model"] = model_name
    return out


def write_json(data: dict, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2), encoding="utf-8")
