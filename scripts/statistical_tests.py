from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from financial_llm.labels import LABELS


DEFAULT_MODELS = {
    "FinBERT": "outputs/runs/finbert/test/predictions.csv",
    "LoRA_r8": "outputs/runs/Qwen__Qwen3-4B/lora/ablation_lora_r8_full_raw_seed42/direct/test/predictions.csv",
    "Fusion_r8_threshold": "outputs/runs/fusion/finbert_qwen3_lora_r8_full_raw_seed42/threshold/test/predictions.csv",
    "Fusion_r8_threshold_calibrated": (
        "outputs/runs/fusion/finbert_qwen3_lora_r8_full_raw_seed42_calibrated/threshold/test/predictions.csv"
    ),
}

DEFAULT_COMPARISONS = [
    ("FinBERT", "LoRA_r8"),
    ("FinBERT", "Fusion_r8_threshold"),
    ("FinBERT", "Fusion_r8_threshold_calibrated"),
    ("LoRA_r8", "Fusion_r8_threshold_calibrated"),
    ("Fusion_r8_threshold", "Fusion_r8_threshold_calibrated"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/analysis/statistical_tests")
    parser.add_argument("--report-path", default="reports/statistical_significance.md")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_predictions(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"example_id", "label", "prediction"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return df[["example_id", "label", "prediction"]].copy()


def align_predictions(model_paths: dict[str, str]) -> pd.DataFrame:
    base_name = next(iter(model_paths))
    base = read_predictions(model_paths[base_name]).rename(columns={"prediction": base_name})
    for name, path in list(model_paths.items())[1:]:
        df = read_predictions(path).rename(columns={"prediction": name, "label": f"label_{name}"})
        base = base.merge(df, on="example_id", how="inner")
        if not (base["label"] == base[f"label_{name}"]).all():
            raise ValueError(f"Labels do not align for {name}")
        base = base.drop(columns=[f"label_{name}"])
    return base


def metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)),
    }


def exact_mcnemar_p_value(b: int, c: int) -> float:
    """Two-sided exact McNemar test using the binomial distribution under p=0.5."""
    n = b + c
    if n == 0:
        return 1.0
    observed = min(b, c)
    tail = sum(math.comb(n, k) * (0.5**n) for k in range(observed + 1))
    return float(min(1.0, 2.0 * tail))


def bootstrap_metric_diffs(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = {"accuracy": [], "macro_f1": [], "weighted_f1": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        a = metric_values(y_true[idx], pred_a[idx])
        b = metric_values(y_true[idx], pred_b[idx])
        for metric in diffs:
            diffs[metric].append(b[metric] - a[metric])

    out = {}
    for metric, values in diffs.items():
        arr = np.asarray(values, dtype=np.float64)
        lower, upper = np.quantile(arr, [0.025, 0.975])
        negative_or_zero = float(np.mean(arr <= 0))
        positive_or_zero = float(np.mean(arr >= 0))
        out[metric] = {
            "diff_mean": float(arr.mean()),
            "ci_low": float(lower),
            "ci_high": float(upper),
            "bootstrap_two_sided_p": float(min(1.0, 2.0 * min(negative_or_zero, positive_or_zero))),
        }
    return out


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
    model_rows: list[dict],
    comparison_rows: list[dict],
    ci_rows: list[dict],
    n_boot: int,
) -> None:
    report = f"""# Paired Statistical Comparison

## Purpose

This report checks whether the main test-set improvements are supported by paired evaluation rather than only by point estimates. All comparisons use the same 727 Financial PhraseBank test examples.

## Methods

- Exact McNemar test is applied to paired correctness outcomes.
- Paired bootstrap with {n_boot} resamples estimates 95% confidence intervals for metric differences.
- Differences are reported as `model_b - model_a`.

## Model Point Estimates

{table(model_rows)}

## Paired Correctness and McNemar Test

{table(comparison_rows)}

## Bootstrap Metric Differences

{table(ci_rows)}

## Interpretation

The key comparison is `FinBERT` versus `Fusion_r8_threshold_calibrated`. If the bootstrap confidence interval for macro-F1 is entirely above zero and the paired correctness test is small, the improvement is more defensible than a single metric table. If a comparison crosses zero, it should be described as suggestive rather than statistically secure.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    merged = align_predictions(DEFAULT_MODELS)
    merged.to_csv(output_dir / "aligned_predictions.csv", index=False)

    y_true = merged["label"].to_numpy()
    model_rows = []
    for name in DEFAULT_MODELS:
        metrics = metric_values(y_true, merged[name].to_numpy())
        model_rows.append({"model": name, **metrics})

    comparison_rows = []
    ci_rows = []
    details = []
    for idx, (model_a, model_b) in enumerate(DEFAULT_COMPARISONS):
        pred_a = merged[model_a].to_numpy()
        pred_b = merged[model_b].to_numpy()
        correct_a = pred_a == y_true
        correct_b = pred_b == y_true
        a_only = int((correct_a & ~correct_b).sum())
        b_only = int((~correct_a & correct_b).sum())
        p_value = exact_mcnemar_p_value(a_only, b_only)

        metrics_a = metric_values(y_true, pred_a)
        metrics_b = metric_values(y_true, pred_b)
        comparison_rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "a_correct_b_wrong": a_only,
                "a_wrong_b_correct": b_only,
                "accuracy_diff": metrics_b["accuracy"] - metrics_a["accuracy"],
                "macro_f1_diff": metrics_b["macro_f1"] - metrics_a["macro_f1"],
                "mcnemar_p": p_value,
            }
        )

        boot = bootstrap_metric_diffs(y_true, pred_a, pred_b, args.bootstrap_samples, args.seed + idx)
        for metric, stats in boot.items():
            ci_rows.append(
                {
                    "model_a": model_a,
                    "model_b": model_b,
                    "metric": metric,
                    "diff_mean": stats["diff_mean"],
                    "ci_low": stats["ci_low"],
                    "ci_high": stats["ci_high"],
                    "bootstrap_p": stats["bootstrap_two_sided_p"],
                }
            )
        details.append({"model_a": model_a, "model_b": model_b, "bootstrap": boot})

    pd.DataFrame(model_rows).to_csv(output_dir / "model_point_estimates.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(output_dir / "paired_mcnemar.csv", index=False)
    pd.DataFrame(ci_rows).to_csv(output_dir / "bootstrap_metric_diffs.csv", index=False)
    (output_dir / "bootstrap_details.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    write_report(Path(args.report_path), model_rows, comparison_rows, ci_rows, args.bootstrap_samples)

    print(json.dumps({"models": model_rows, "comparisons": comparison_rows}, indent=2))
    print(f"Saved statistical outputs to: {output_dir}")
    print(f"Saved report to: {args.report_path}")


if __name__ == "__main__":
    main()
