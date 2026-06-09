from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from financial_llm.config import ensure_dir
from financial_llm.fusion import logistic_stacking_fusion
from financial_llm.labels import LABELS


PROB_COLS = [f"prob_{label}" for label in LABELS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the fixed final logistic-stacking fusion layer on paired model predictions."
    )
    parser.add_argument("--finbert-predictions", required=True)
    parser.add_argument("--lora-predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_predictions(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"example_id", "sentence", "label", "prediction", "confidence", *PROB_COLS}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return df


def align_predictions(finbert: pd.DataFrame, lora: pd.DataFrame) -> pd.DataFrame:
    left = finbert[["example_id", "sentence", "label", "prediction", "confidence", *PROB_COLS]]
    right = lora[["example_id", "prediction", "confidence", *PROB_COLS]]
    merged = left.merge(
        right,
        on="example_id",
        suffixes=("_finbert", "_lora"),
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("No aligned examples were found.")
    return merged


def probs_from(df: pd.DataFrame, suffix: str) -> np.ndarray:
    return df[[f"{col}_{suffix}" for col in PROB_COLS]].to_numpy(dtype=np.float64)


def labels_from(df: pd.DataFrame) -> np.ndarray:
    label_to_id = {label: idx for idx, label in enumerate(LABELS)}
    return df["label"].map(label_to_id).to_numpy(dtype=np.int64)


def metric_dict(y_true: np.ndarray, probs: np.ndarray) -> dict:
    preds = np.argmax(probs, axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "macro_f1": float(f1_score(y_true, preds, average="macro")),
        "weighted_f1": float(f1_score(y_true, preds, average="weighted")),
        "log_loss": float(log_loss(y_true, probs, labels=list(range(len(LABELS))))),
        "classification_report": classification_report(
            y_true,
            preds,
            target_names=LABELS,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, preds, labels=list(range(len(LABELS)))).tolist(),
        "labels": LABELS,
    }


def write_predictions(path: Path, base: pd.DataFrame, probs: np.ndarray) -> None:
    preds = np.argmax(probs, axis=1)
    out = base[["example_id", "sentence", "label"]].copy()
    for idx, label in enumerate(LABELS):
        out[f"prob_{label}"] = probs[:, idx]
    out["prediction"] = [LABELS[idx] for idx in preds]
    out["confidence"] = probs[np.arange(len(probs)), preds]
    out["model"] = "fixed_logistic_stacking"
    out.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    aligned = align_predictions(
        load_predictions(args.finbert_predictions),
        load_predictions(args.lora_predictions),
    )
    y_true = labels_from(aligned)
    probs = logistic_stacking_fusion(probs_from(aligned, "lora"), probs_from(aligned, "finbert"))
    write_predictions(output_dir / "predictions.csv", aligned, probs)

    metrics = metric_dict(y_true, probs)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
