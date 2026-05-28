from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from financial_llm.fusion import threshold_fusion, weighted_fusion
from financial_llm.labels import LABELS
from financial_llm.metrics import compute_classification_metrics


LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finbert-val", default="outputs/runs/finbert/val/predictions.csv")
    parser.add_argument("--finbert-test", default="outputs/runs/finbert/test/predictions.csv")
    parser.add_argument(
        "--lora-val",
        default="outputs/runs/Qwen__Qwen3-4B/lora/train_frac100_raw/direct/val/predictions.csv",
    )
    parser.add_argument(
        "--lora-test",
        default="outputs/runs/Qwen__Qwen3-4B/lora/train_frac100_raw/direct/test/predictions.csv",
    )
    parser.add_argument("--output-dir", default="outputs/analysis/calibration")
    parser.add_argument("--fusion-output-dir", default="outputs/runs/fusion/finbert_qwen3_lora100_raw_calibrated")
    parser.add_argument("--report-path", default="reports/confidence_calibration.md")
    parser.add_argument("--threshold-start", type=float, default=0.50)
    parser.add_argument("--threshold-end", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument(
        "--reference-weighted-brier",
        type=float,
        default=None,
        help="Optional uncalibrated weighted-fusion test Brier score used only in the report interpretation.",
    )
    parser.add_argument(
        "--run-label",
        default="LoRA",
        help="Human-readable label for the LoRA run in the generated report.",
    )
    return parser.parse_args()


def get_probs(df: pd.DataFrame) -> np.ndarray:
    return df[[f"prob_{label}" for label in LABELS]].to_numpy(dtype=np.float64)


def get_y(df: pd.DataFrame) -> np.ndarray:
    return df["label"].map(LABEL2ID).to_numpy(dtype=np.int64)


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


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


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = pred == y
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (conf > low) & (conf <= high)
        if not mask.any():
            continue
        ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def confidence_bins(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = pred == y
    rows = []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (conf > low) & (conf <= high)
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
                "avg_confidence": float(conf[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def classification_metrics(probs: np.ndarray, labels: list[str]) -> dict:
    pred = [LABELS[idx] for idx in probs.argmax(axis=1)]
    return compute_classification_metrics(labels, pred)


def probability_metrics(probs: np.ndarray, y: np.ndarray, labels: list[str]) -> dict:
    class_metrics = classification_metrics(probs, labels)
    return {
        "accuracy": class_metrics["accuracy"],
        "macro_f1": class_metrics["macro_f1"],
        "weighted_f1": class_metrics["weighted_f1"],
        "nll": nll(probs, y),
        "brier_score": brier_score(probs, y),
        "ece_10_bins": expected_calibration_error(probs, y),
        "avg_confidence": float(probs.max(axis=1).mean()),
    }


def find_temperature(probs: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    coarse = np.concatenate(
        [
            np.linspace(0.20, 1.00, 81),
            np.linspace(1.02, 5.00, 200),
            np.linspace(5.10, 10.00, 50),
        ]
    )
    coarse_scores = [(float(t), nll(temperature_scale(probs, float(t)), y)) for t in coarse]
    best_t, _ = min(coarse_scores, key=lambda item: item[1])
    low = max(0.05, best_t - 0.20)
    high = min(20.0, best_t + 0.20)
    fine = np.linspace(low, high, 401)
    fine_scores = [(float(t), nll(temperature_scale(probs, float(t)), y)) for t in fine]
    best_t, best_nll = min(fine_scores, key=lambda item: item[1])
    return float(best_t), float(best_nll)


def write_predictions(base_df: pd.DataFrame, probs: np.ndarray, model_name: str, path: Path) -> None:
    out = base_df[["example_id", "sentence", "label"]].copy()
    for idx, label in enumerate(LABELS):
        out[f"prob_{label}"] = probs[:, idx]
    pred_idx = probs.argmax(axis=1)
    out["prediction"] = [LABELS[idx] for idx in pred_idx]
    out["confidence"] = probs.max(axis=1)
    out["model"] = model_name
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def threshold_values(start: float, end: float, step: float) -> list[float]:
    count = int(round((end - start) / step))
    return [round(start + i * step, 4) for i in range(count + 1)]


def tune_threshold(
    val_lora_probs: np.ndarray,
    val_finbert_probs: np.ndarray,
    labels: list[str],
    start: float,
    end: float,
    step: float,
) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in threshold_values(start, end, step):
        probs = threshold_fusion(val_lora_probs, val_finbert_probs, threshold)
        metrics = classification_metrics(probs, labels)
        rows.append(
            {
                "threshold": threshold,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
            }
        )
    search = pd.DataFrame(rows).sort_values(["macro_f1", "accuracy"], ascending=False).reset_index(drop=True)
    return float(search.loc[0, "threshold"]), search


def evaluate_and_save(
    base_df: pd.DataFrame,
    probs: np.ndarray,
    output_dir: Path,
    model_name: str,
    extra: dict | None = None,
) -> dict:
    labels = base_df["label"].tolist()
    y = get_y(base_df)
    metrics = compute_classification_metrics(labels, [LABELS[idx] for idx in probs.argmax(axis=1)])
    metrics.update(
        {
            "nll": nll(probs, y),
            "brier_score": brier_score(probs, y),
            "ece_10_bins": expected_calibration_error(probs, y),
            "avg_confidence": float(probs.max(axis=1).mean()),
        }
    )
    if extra:
        metrics.update(extra)
    write_predictions(base_df, probs, model_name, output_dir / "predictions.csv")
    write_json(metrics, output_dir / "metrics.json")
    return metrics


def plot_reliability(
    rows: list[tuple[str, np.ndarray, np.ndarray]],
    output_path: Path,
    n_bins: int = 10,
) -> None:
    fig, axes = plt.subplots(1, len(rows), figsize=(5 * len(rows), 4.5), sharey=True)
    if len(rows) == 1:
        axes = [axes]
    for ax, (title, probs, y) in zip(axes, rows):
        bins = confidence_bins(probs, y, n_bins=n_bins)
        non_empty = bins[bins["count"] > 0]
        ax.plot([0, 1], [0, 1], linestyle="--", color="#666666", linewidth=1)
        ax.bar(
            non_empty["avg_confidence"],
            non_empty["accuracy"],
            width=0.075,
            alpha=0.72,
            color="#4C78A8",
            edgecolor="#333333",
        )
        ax.set_title(title)
        ax.set_xlabel("Average confidence")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Accuracy")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def table(rows: list[dict], digits: int = 4) -> str:
    if not rows:
        return "No rows."
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        rendered = []
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                rendered.append(f"{value:.{digits}f}")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def write_report(
    report_path: Path,
    temperature_rows: list[dict],
    metric_rows: list[dict],
    fusion_rows: list[dict],
    interpretation: str,
    run_label: str,
) -> None:
    report = f"""# Confidence Calibration Analysis

## Purpose

This report evaluates whether the confidence scores used for confidence-based fusion are reliable for `{run_label}`. Temperature scaling is fitted on the validation split and evaluated on the test split.

## Method

For each model, probabilities are converted back to log-probability scores:

```text
logits = log(p)
p_calibrated = softmax(logits / T)
```

The temperature `T` is selected on the validation split by minimizing negative log-likelihood. The test split is used only for final evaluation.

## Selected Temperatures

{table(temperature_rows)}

## Individual Model Calibration

{table(metric_rows)}

## Calibrated Fusion Results

{table(fusion_rows)}

## Interpretation

{interpretation}
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    fusion_output_dir = Path(args.fusion_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fusion_output_dir.mkdir(parents=True, exist_ok=True)

    finbert_val = pd.read_csv(args.finbert_val)
    finbert_test = pd.read_csv(args.finbert_test)
    lora_val = pd.read_csv(args.lora_val)
    lora_test = pd.read_csv(args.lora_test)

    finbert_val_probs = get_probs(finbert_val)
    finbert_test_probs = get_probs(finbert_test)
    lora_val_probs = get_probs(lora_val)
    lora_test_probs = get_probs(lora_test)

    y_val = get_y(finbert_val)
    y_test = get_y(finbert_test)
    val_labels = finbert_val["label"].tolist()
    test_labels = finbert_test["label"].tolist()

    finbert_t, finbert_val_nll = find_temperature(finbert_val_probs, y_val)
    lora_t, lora_val_nll = find_temperature(lora_val_probs, y_val)

    finbert_val_cal = temperature_scale(finbert_val_probs, finbert_t)
    finbert_test_cal = temperature_scale(finbert_test_probs, finbert_t)
    lora_val_cal = temperature_scale(lora_val_probs, lora_t)
    lora_test_cal = temperature_scale(lora_test_probs, lora_t)

    write_predictions(finbert_val, finbert_val_cal, "ProsusAI/finbert+temperature", output_dir / "finbert_val_calibrated_predictions.csv")
    write_predictions(finbert_test, finbert_test_cal, "ProsusAI/finbert+temperature", output_dir / "finbert_test_calibrated_predictions.csv")
    write_predictions(lora_val, lora_val_cal, "Qwen/Qwen3-4B+LoRA+temperature", output_dir / "lora_val_calibrated_predictions.csv")
    write_predictions(lora_test, lora_test_cal, "Qwen/Qwen3-4B+LoRA+temperature", output_dir / "lora_test_calibrated_predictions.csv")

    temperature_rows = [
        {
            "model": "FinBERT",
            "temperature": finbert_t,
            "validation_nll_after": finbert_val_nll,
        },
        {
            "model": "LoRA",
            "temperature": lora_t,
            "validation_nll_after": lora_val_nll,
        },
    ]

    metric_rows = []
    metric_specs = [
        ("FinBERT", "val", "uncalibrated", finbert_val_probs, y_val, val_labels),
        ("FinBERT", "val", "calibrated", finbert_val_cal, y_val, val_labels),
        ("FinBERT", "test", "uncalibrated", finbert_test_probs, y_test, test_labels),
        ("FinBERT", "test", "calibrated", finbert_test_cal, y_test, test_labels),
        ("LoRA", "val", "uncalibrated", lora_val_probs, y_val, val_labels),
        ("LoRA", "val", "calibrated", lora_val_cal, y_val, val_labels),
        ("LoRA", "test", "uncalibrated", lora_test_probs, y_test, test_labels),
        ("LoRA", "test", "calibrated", lora_test_cal, y_test, test_labels),
    ]
    for model, split, condition, probs, y, labels in metric_specs:
        metrics = probability_metrics(probs, y, labels)
        metric_rows.append(
            {
                "model": model,
                "split": split,
                "condition": condition,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "nll": metrics["nll"],
                "brier": metrics["brier_score"],
                "ece": metrics["ece_10_bins"],
                "avg_conf": metrics["avg_confidence"],
            }
        )

    pd.DataFrame(temperature_rows).to_csv(output_dir / "temperature_summary.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(output_dir / "calibration_metrics.csv", index=False)

    bin_frames = []
    for model, split, condition, probs, y, _ in metric_specs:
        bins = confidence_bins(probs, y)
        bins.insert(0, "condition", condition)
        bins.insert(0, "split", split)
        bins.insert(0, "model", model)
        bin_frames.append(bins)
    pd.concat(bin_frames, ignore_index=True).to_csv(output_dir / "confidence_bins.csv", index=False)

    plot_reliability(
        [
            ("FinBERT uncalibrated", finbert_test_probs, y_test),
            ("FinBERT calibrated", finbert_test_cal, y_test),
            ("LoRA uncalibrated", lora_test_probs, y_test),
            ("LoRA calibrated", lora_test_cal, y_test),
        ],
        output_dir / "reliability_test.png",
    )

    selected_threshold, threshold_search = tune_threshold(
        lora_val_cal,
        finbert_val_cal,
        val_labels,
        args.threshold_start,
        args.threshold_end,
        args.threshold_step,
    )
    threshold_search.to_csv(fusion_output_dir / "threshold_search.csv", index=False)

    val_threshold_probs = threshold_fusion(lora_val_cal, finbert_val_cal, selected_threshold)
    test_threshold_probs = threshold_fusion(lora_test_cal, finbert_test_cal, selected_threshold)
    val_weighted_probs = weighted_fusion(lora_val_cal, finbert_val_cal)
    test_weighted_probs = weighted_fusion(lora_test_cal, finbert_test_cal)

    threshold_extra = {
        "method": "threshold_calibrated",
        "threshold": selected_threshold,
        "finbert_temperature": finbert_t,
        "lora_temperature": lora_t,
    }
    weighted_extra = {
        "method": "weighted_calibrated",
        "finbert_temperature": finbert_t,
        "lora_temperature": lora_t,
    }
    threshold_val_metrics = evaluate_and_save(
        finbert_val,
        val_threshold_probs,
        fusion_output_dir / "threshold" / "val",
        "fusion_threshold_calibrated",
        threshold_extra,
    )
    threshold_test_metrics = evaluate_and_save(
        finbert_test,
        test_threshold_probs,
        fusion_output_dir / "threshold" / "test",
        "fusion_threshold_calibrated",
        threshold_extra,
    )
    weighted_val_metrics = evaluate_and_save(
        finbert_val,
        val_weighted_probs,
        fusion_output_dir / "weighted" / "val",
        "fusion_weighted_calibrated",
        weighted_extra,
    )
    weighted_test_metrics = evaluate_and_save(
        finbert_test,
        test_weighted_probs,
        fusion_output_dir / "weighted" / "test",
        "fusion_weighted_calibrated",
        weighted_extra,
    )

    fusion_rows = [
        {
            "method": "threshold calibrated",
            "split": "val",
            "accuracy": threshold_val_metrics["accuracy"],
            "macro_f1": threshold_val_metrics["macro_f1"],
            "weighted_f1": threshold_val_metrics["weighted_f1"],
            "ece": threshold_val_metrics["ece_10_bins"],
            "brier": threshold_val_metrics["brier_score"],
        },
        {
            "method": "threshold calibrated",
            "split": "test",
            "accuracy": threshold_test_metrics["accuracy"],
            "macro_f1": threshold_test_metrics["macro_f1"],
            "weighted_f1": threshold_test_metrics["weighted_f1"],
            "ece": threshold_test_metrics["ece_10_bins"],
            "brier": threshold_test_metrics["brier_score"],
        },
        {
            "method": "weighted calibrated",
            "split": "val",
            "accuracy": weighted_val_metrics["accuracy"],
            "macro_f1": weighted_val_metrics["macro_f1"],
            "weighted_f1": weighted_val_metrics["weighted_f1"],
            "ece": weighted_val_metrics["ece_10_bins"],
            "brier": weighted_val_metrics["brier_score"],
        },
        {
            "method": "weighted calibrated",
            "split": "test",
            "accuracy": weighted_test_metrics["accuracy"],
            "macro_f1": weighted_test_metrics["macro_f1"],
            "weighted_f1": weighted_test_metrics["weighted_f1"],
            "ece": weighted_test_metrics["ece_10_bins"],
            "brier": weighted_test_metrics["brier_score"],
        },
    ]
    pd.DataFrame(fusion_rows).to_csv(fusion_output_dir / "summary.csv", index=False)

    summary = {
        "finbert_temperature": finbert_t,
        "lora_temperature": lora_t,
        "selected_calibrated_threshold": selected_threshold,
        "threshold_test": {
            "accuracy": threshold_test_metrics["accuracy"],
            "macro_f1": threshold_test_metrics["macro_f1"],
            "weighted_f1": threshold_test_metrics["weighted_f1"],
            "ece_10_bins": threshold_test_metrics["ece_10_bins"],
            "brier_score": threshold_test_metrics["brier_score"],
        },
        "weighted_test": {
            "accuracy": weighted_test_metrics["accuracy"],
            "macro_f1": weighted_test_metrics["macro_f1"],
            "weighted_f1": weighted_test_metrics["weighted_f1"],
            "ece_10_bins": weighted_test_metrics["ece_10_bins"],
            "brier_score": weighted_test_metrics["brier_score"],
        },
    }
    write_json(summary, fusion_output_dir / "summary.json")

    uncal_lora_test = probability_metrics(lora_test_probs, y_test, test_labels)
    cal_lora_test = probability_metrics(lora_test_cal, y_test, test_labels)
    reference_sentence = ""
    if args.reference_weighted_brier is not None:
        reference_sentence = (
            "Calibrated weighted fusion should be compared with the previous uncalibrated weighted fusion: "
            f"the previous weighted-fusion Brier score was {args.reference_weighted_brier:.4f}, while the calibrated "
            f"weighted-fusion Brier score is {weighted_test_metrics['brier_score']:.4f}. "
        )
    interpretation = (
        "Temperature scaling does not change the class predictions of individual models, "
        "but it changes probability sharpness. In this run, LoRA required a temperature "
        f"of {lora_t:.4f}, which confirms that the uncalibrated LoRA probabilities were too sharp. "
        f"LoRA test ECE changed from {uncal_lora_test['ece_10_bins']:.4f} to {cal_lora_test['ece_10_bins']:.4f}. "
        f"{reference_sentence}"
        "If macro-F1 decreases slightly but probability metrics improve, the report should frame calibration as a "
        "reliability improvement rather than a pure accuracy optimization."
    )
    write_report(Path(args.report_path), temperature_rows, metric_rows, fusion_rows, interpretation, args.run_label)

    print(json.dumps(summary, indent=2))
    print(f"Saved calibration outputs to: {output_dir}")
    print(f"Saved calibrated fusion outputs to: {fusion_output_dir}")
    print(f"Saved report to: {args.report_path}")


if __name__ == "__main__":
    main()
