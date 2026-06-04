from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from financial_llm.config import ensure_dir
from financial_llm.labels import LABELS


PROB_COLS = [f"prob_{label}" for label in LABELS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the final logistic-stacking fusion layer."
    )
    parser.add_argument("--finbert-val", default="outputs/runs/finbert/val/predictions.csv")
    parser.add_argument("--finbert-test", default="outputs/runs/finbert/test/predictions.csv")
    parser.add_argument(
        "--lora-val",
        default="outputs/runs/Qwen__Qwen3-4B/lora/neutral_aware_lora_r8_full_raw_seed42/neutral_aware/val/predictions.csv",
    )
    parser.add_argument(
        "--lora-test",
        default="outputs/runs/Qwen__Qwen3-4B/lora/neutral_aware_lora_r8_full_raw_seed42/neutral_aware/test/predictions.csv",
    )
    parser.add_argument("--output-dir", default="outputs/analysis/learned_fusion_final")
    parser.add_argument("--random-state", type=int, default=42)
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
    cols = [f"{col}_{suffix}" for col in PROB_COLS]
    return df[cols].to_numpy(dtype=np.float64)


def labels_from(df: pd.DataFrame) -> np.ndarray:
    label_to_id = {label: idx for idx, label in enumerate(LABELS)}
    return df["label"].map(label_to_id).to_numpy(dtype=np.int64)


def stack_features(finbert_probs: np.ndarray, lora_probs: np.ndarray) -> np.ndarray:
    return np.hstack([finbert_probs, lora_probs]).astype(np.float64)


def metric_dict(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    preds = np.argmax(probs, axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "macro_f1": float(f1_score(y_true, preds, average="macro")),
        "weighted_f1": float(f1_score(y_true, preds, average="weighted")),
        "log_loss": float(log_loss(y_true, probs, labels=list(range(len(LABELS))))),
    }


def error_taxonomy(y_true: np.ndarray, preds: np.ndarray) -> dict[str, int]:
    neutral_idx = LABELS.index("neutral")
    negative_idx = LABELS.index("negative")
    positive_idx = LABELS.index("positive")
    directional = {negative_idx, positive_idx}
    neutral_false_direction = int(np.sum((y_true == neutral_idx) & (preds != neutral_idx)))
    missed_directional = int(np.sum((y_true != neutral_idx) & (preds == neutral_idx)))
    polarity_flip = int(
        np.sum(
            np.isin(y_true, list(directional))
            & np.isin(preds, list(directional))
            & (y_true != preds)
        )
    )
    total_errors = int(np.sum(y_true != preds))
    return {
        "total_errors": total_errors,
        "neutral_false_direction": neutral_false_direction,
        "missed_directional_sentiment": missed_directional,
        "polarity_flip": polarity_flip,
    }


def select_model(x_val: np.ndarray, y_val: np.ndarray, random_state: int) -> tuple[LogisticRegression, pd.DataFrame]:
    rows = []
    best_key: tuple[float, float, float, float] | None = None
    best_model: LogisticRegression | None = None
    for class_weight in [None, "balanced"]:
        for c_value in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]:
            model = LogisticRegression(
                C=c_value,
                max_iter=5000,
                random_state=random_state,
                class_weight=class_weight,
                solver="lbfgs",
            )
            model.fit(x_val, y_val)
            probs = model.predict_proba(x_val)
            metrics = metric_dict(y_val, probs)
            row = {
                "C": float(c_value),
                "class_weight": "none" if class_weight is None else str(class_weight),
                **{f"val_{key}": value for key, value in metrics.items()},
            }
            rows.append(row)
            key = (
                metrics["macro_f1"],
                metrics["accuracy"],
                metrics["weighted_f1"],
                -metrics["log_loss"],
            )
            if best_key is None or key > best_key:
                best_key = key
                best_model = model
    if best_model is None:
        raise ValueError("No logistic stacking model was fitted.")
    return best_model, pd.DataFrame(rows)


def write_predictions(path: Path, base: pd.DataFrame, probs: np.ndarray) -> None:
    preds = np.argmax(probs, axis=1)
    out = base[["example_id", "sentence", "label"]].copy()
    for idx, label in enumerate(LABELS):
        out[f"prob_{label}"] = probs[:, idx]
    out["prediction"] = [LABELS[idx] for idx in preds]
    out["confidence"] = probs[np.arange(len(probs)), preds]
    out["model"] = "logistic_stacking_no_margin"
    out.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)

    val = align_predictions(load_predictions(args.finbert_val), load_predictions(args.lora_val))
    test = align_predictions(load_predictions(args.finbert_test), load_predictions(args.lora_test))
    y_val = labels_from(val)
    y_test = labels_from(test)

    x_val = stack_features(probs_from(val, "finbert"), probs_from(val, "lora"))
    x_test = stack_features(probs_from(test, "finbert"), probs_from(test, "lora"))
    model, search = select_model(x_val, y_val, args.random_state)

    val_probs = model.predict_proba(x_val)
    test_probs = model.predict_proba(x_test)
    test_preds = np.argmax(test_probs, axis=1)

    search.to_csv(output_dir / "logistic_stacking_search.csv", index=False)
    write_predictions(output_dir / "test_predictions.csv", test, test_probs)

    summary = {
        "feature_names": [
            "finbert_negative",
            "finbert_neutral",
            "finbert_positive",
            "lora_negative",
            "lora_neutral",
            "lora_positive",
        ],
        "C": float(model.C),
        "class_weight": "none" if model.class_weight is None else str(model.class_weight),
        "coef": model.coef_.tolist(),
        "intercept": model.intercept_.tolist(),
        "validation": metric_dict(y_val, val_probs),
        "test": metric_dict(y_test, test_probs),
        "test_error_taxonomy": error_taxonomy(y_test, test_preds),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
