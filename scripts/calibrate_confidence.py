from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from financial_llm.labels import LABELS
from financial_llm.metrics import compute_classification_metrics


LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--val-predictions",
        default=(
            "outputs/runs/Qwen__Qwen3-4B/lora/neutral_aware_lora_r8_full_raw_seed42/"
            "neutral_aware/val/predictions.csv"
        ),
    )
    parser.add_argument(
        "--test-predictions",
        default=(
            "outputs/runs/Qwen__Qwen3-4B/lora/neutral_aware_lora_r8_full_raw_seed42/"
            "neutral_aware/test/predictions.csv"
        ),
    )
    parser.add_argument("--output-dir", default="outputs/analysis/calibration/neutral_aware_lora_r8")
    parser.add_argument("--model-name", default="neutral-aware Qwen3-4B LoRA r8")
    parser.add_argument("--bins", type=int, default=10)
    return parser.parse_args()


def read_predictions(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"example_id", "sentence", "label", "prediction", *[f"prob_{label}" for label in LABELS]}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return df


def get_probs(df: pd.DataFrame) -> np.ndarray:
    probs = df[[f"prob_{label}" for label in LABELS]].to_numpy(dtype=np.float64)
    row_sums = probs.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Probability rows must have positive sums.")
    return probs / row_sums


def get_y(df: pd.DataFrame) -> np.ndarray:
    return df["label"].map(LABEL2ID).to_numpy(dtype=np.int64)


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def temperature_scale(probs: np.ndarray, temperature: float) -> np.ndarray:
    clipped = np.clip(probs, 1e-12, 1.0)
    logits = np.log(clipped)
    return softmax(logits / temperature)


def nll(probs: np.ndarray, y: np.ndarray) -> float:
    clipped = np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)
    return float(-np.log(clipped).mean())


def brier_score(probs: np.ndarray, y: np.ndarray) -> float:
    target = np.zeros_like(probs)
    target[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((probs - target) ** 2, axis=1)))


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int) -> float:
    pred = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    correct = pred == y
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        if not mask.any():
            continue
        ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def confidence_bins(probs: np.ndarray, y: np.ndarray, n_bins: int) -> pd.DataFrame:
    pred = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    correct = pred == y
    rows = []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        if not mask.any():
            rows.append(
                {
                    "bin_low": low,
                    "bin_high": high,
                    "count": 0,
                    "accuracy": np.nan,
                    "avg_confidence": np.nan,
                }
            )
            continue
        rows.append(
            {
                "bin_low": low,
                "bin_high": high,
                "count": int(mask.sum()),
                "accuracy": float(correct[mask].mean()),
                "avg_confidence": float(confidence[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def find_temperature(probs: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    coarse = np.concatenate(
        [
            np.linspace(0.20, 1.00, 81),
            np.linspace(1.02, 5.00, 200),
            np.linspace(5.10, 10.00, 50),
        ]
    )
    best_t = min(coarse, key=lambda t: nll(temperature_scale(probs, float(t)), y))
    low = max(0.05, float(best_t) - 0.20)
    high = min(20.0, float(best_t) + 0.20)
    fine = np.linspace(low, high, 401)
    final_t = min(fine, key=lambda t: nll(temperature_scale(probs, float(t)), y))
    return float(final_t), nll(temperature_scale(probs, float(final_t)), y)


def classification_metrics_from_probs(probs: np.ndarray, labels: list[str]) -> dict:
    pred = [LABELS[idx] for idx in probs.argmax(axis=1)]
    return compute_classification_metrics(labels, pred)


def probability_metrics(probs: np.ndarray, df: pd.DataFrame, n_bins: int) -> dict:
    y = get_y(df)
    class_metrics = classification_metrics_from_probs(probs, df["label"].tolist())
    return {
        "accuracy": class_metrics["accuracy"],
        "macro_f1": class_metrics["macro_f1"],
        "weighted_f1": class_metrics["weighted_f1"],
        "nll": nll(probs, y),
        "brier_score": brier_score(probs, y),
        "ece": expected_calibration_error(probs, y, n_bins=n_bins),
        "avg_confidence": float(probs.max(axis=1).mean()),
    }


def write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    val_df = read_predictions(args.val_predictions)
    test_df = read_predictions(args.test_predictions)
    val_probs = get_probs(val_df)
    test_probs = get_probs(test_df)
    val_y = get_y(val_df)

    temperature, val_nll = find_temperature(val_probs, val_y)
    calibrated_test_probs = temperature_scale(test_probs, temperature)

    summary = {
        "model": args.model_name,
        "temperature": temperature,
        "validation_nll_after_temperature_search": val_nll,
        "raw_test": probability_metrics(test_probs, test_df, n_bins=args.bins),
        "calibrated_test": probability_metrics(calibrated_test_probs, test_df, n_bins=args.bins),
        "labels": LABELS,
    }
    write_json(summary, output_dir / "metrics.json")
    confidence_bins(test_probs, get_y(test_df), args.bins).to_csv(output_dir / "raw_bins.csv", index=False)
    confidence_bins(calibrated_test_probs, get_y(test_df), args.bins).to_csv(
        output_dir / "calibrated_bins.csv",
        index=False,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
