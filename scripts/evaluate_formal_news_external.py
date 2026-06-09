from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from financial_llm.config import ensure_dir
from financial_llm.labels import LABELS


DEFAULT_PREDICTIONS = [
    (
        "FinBERT",
        "outputs/runs/external/nosible_financial_sentiment_full/finbert/test/predictions.csv",
    ),
    (
        "neutral-aware LoRA r8",
        "outputs/runs/external/nosible_financial_sentiment_full/neutral_aware_lora_r8/test/predictions.csv",
    ),
    (
        "fixed learned stacking fusion",
        "outputs/analysis/external_formal_news/nosible_financial_sentiment_full/learned_stacking/predictions.csv",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze full NOSIBLE formal-news external robustness results."
    )
    parser.add_argument("--data-file", default="data/external/nosible_financial_sentiment_full/test.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/external_formal_news/nosible_financial_sentiment_full")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prediction",
        action="append",
        default=None,
        help="Prediction input in the form name=path. Can be passed multiple times.",
    )
    return parser.parse_args()


def parse_prediction_specs(values: list[str] | None) -> list[tuple[str, str]]:
    if not values:
        return DEFAULT_PREDICTIONS
    specs = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Prediction spec must be name=path, got: {value}")
        name, path = value.split("=", 1)
        specs.append((name.strip(), path.strip()))
    return specs


def read_predictions(name: str, path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"example_id", "label", "prediction", "confidence"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return df[["example_id", "label", "prediction", "confidence"]].rename(
        columns={
            "prediction": f"prediction_{name}",
            "confidence": f"confidence_{name}",
            "label": f"label_{name}",
        }
    )


def metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)),
    }


def exact_mcnemar_p_value(a_only: int, b_only: int) -> float:
    n = a_only + b_only
    if n == 0:
        return 1.0
    return float(binomtest(min(a_only, b_only), n=n, p=0.5, alternative="two-sided").pvalue)


def bootstrap_diffs(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    samples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values = {"accuracy": [], "macro_f1": [], "weighted_f1": []}
    for _ in range(samples):
        idx = rng.integers(0, n, size=n)
        metrics_a = metric_values(y_true[idx], pred_a[idx])
        metrics_b = metric_values(y_true[idx], pred_b[idx])
        for metric in values:
            values[metric].append(metrics_b[metric] - metrics_a[metric])

    out = {}
    for metric, diffs in values.items():
        arr = np.asarray(diffs, dtype=np.float64)
        low, high = np.quantile(arr, [0.025, 0.975])
        out[metric] = {
            "diff_mean": float(arr.mean()),
            "ci_low": float(low),
            "ci_high": float(high),
        }
    return out


def add_length_bin(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["word_count"] = out["sentence"].astype(str).str.split().str.len()
    out["length_bin"] = pd.cut(
        out["word_count"],
        bins=[0, 50, 150, 300, 10_000],
        labels=["<=50 words", "51-150 words", "151-300 words", ">300 words"],
        include_lowest=True,
    ).astype(str)
    return out


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        rendered = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                rendered.append(f"{value:.4f}")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    specs = parse_prediction_specs(args.prediction)

    base = pd.read_csv(args.data_file)
    required = {"example_id", "sentence", "label"}
    missing = required.difference(base.columns)
    if missing:
        raise ValueError(f"{args.data_file} is missing columns: {sorted(missing)}")
    base = add_length_bin(base)

    merged = base.copy()
    names = []
    for name, path in specs:
        names.append(name)
        pred = read_predictions(name, path)
        merged = merged.merge(pred, on="example_id", how="inner", validate="one_to_one")
        if not (merged["label"] == merged[f"label_{name}"]).all():
            raise ValueError(f"Labels do not align for {name}")
        merged = merged.drop(columns=[f"label_{name}"])

    y_true = merged["label"].to_numpy()

    overall_rows = []
    length_rows = []
    source_rows = []
    confusion = {}
    for name in names:
        pred_col = f"prediction_{name}"
        metrics = metric_values(y_true, merged[pred_col].to_numpy())
        overall_rows.append({"model": name, "n": len(merged), **metrics})
        confusion[name] = confusion_matrix(y_true, merged[pred_col], labels=LABELS).tolist()

        for length_bin, group in merged.groupby("length_bin", sort=False):
            length_metrics = metric_values(group["label"].to_numpy(), group[pred_col].to_numpy())
            length_rows.append({"model": name, "length_bin": length_bin, "n": len(group), **length_metrics})

        if "netloc" in merged.columns:
            top_sources = merged["netloc"].value_counts().head(20).index
            for source, group in merged[merged["netloc"].isin(top_sources)].groupby("netloc"):
                source_metrics = metric_values(group["label"].to_numpy(), group[pred_col].to_numpy())
                source_rows.append({"model": name, "source": source, "n": len(group), **source_metrics})

    pair_rows = []
    pair_details = []
    for idx, model_a in enumerate(names):
        for model_b in names[idx + 1 :]:
            pred_a = merged[f"prediction_{model_a}"].to_numpy()
            pred_b = merged[f"prediction_{model_b}"].to_numpy()
            correct_a = pred_a == y_true
            correct_b = pred_b == y_true
            a_only = int((correct_a & ~correct_b).sum())
            b_only = int((~correct_a & correct_b).sum())
            metrics_a = metric_values(y_true, pred_a)
            metrics_b = metric_values(y_true, pred_b)
            boot = bootstrap_diffs(y_true, pred_a, pred_b, args.bootstrap_samples, args.seed + idx)
            pair_rows.append(
                {
                    "model_a": model_a,
                    "model_b": model_b,
                    "a_correct_b_wrong": a_only,
                    "a_wrong_b_correct": b_only,
                    "accuracy_diff_b_minus_a": metrics_b["accuracy"] - metrics_a["accuracy"],
                    "macro_f1_diff_b_minus_a": metrics_b["macro_f1"] - metrics_a["macro_f1"],
                    "mcnemar_p": exact_mcnemar_p_value(a_only, b_only),
                }
            )
            pair_details.append({"model_a": model_a, "model_b": model_b, "bootstrap": boot})

    pd.DataFrame(overall_rows).to_csv(output_dir / "overall_metrics.csv", index=False)
    pd.DataFrame(length_rows).to_csv(output_dir / "length_bin_metrics.csv", index=False)
    pd.DataFrame(source_rows).to_csv(output_dir / "top_source_metrics.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(output_dir / "paired_comparisons.csv", index=False)
    merged.to_csv(output_dir / "aligned_predictions.csv", index=False)

    summary = {
        "dataset": "NOSIBLE/financial-sentiment",
        "dataset_url": "https://huggingface.co/datasets/NOSIBLE/financial-sentiment",
        "n": int(len(merged)),
        "label_counts": merged["label"].value_counts().to_dict(),
        "word_count_summary": {
            "mean": float(merged["word_count"].mean()),
            "median": float(merged["word_count"].median()),
            "max": int(merged["word_count"].max()),
        },
        "overall": overall_rows,
        "confusion_matrices": confusion,
        "paired": pair_rows,
        "paired_bootstrap": pair_details,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = f"""# Formal-News External Robustness: NOSIBLE Financial Sentiment

## Dataset

This external check uses `NOSIBLE/financial-sentiment`, a Hugging Face financial sentiment dataset with text, label, source-domain, and URL fields. The dataset is used only for evaluation. No model is retrained on this dataset, and the fixed learned stacking layer remains the one selected on Financial PhraseBank validation predictions.

Rows evaluated: {len(merged)}

Label counts:

{markdown_table([{"label": label, "count": int(count)} for label, count in merged["label"].value_counts().items()], ["label", "count"])}

## Overall Metrics

{markdown_table(overall_rows, ["model", "n", "accuracy", "macro_f1", "weighted_f1"])}

## Length-Bin Metrics

{markdown_table(length_rows, ["model", "length_bin", "n", "accuracy", "macro_f1", "weighted_f1"])}

## Paired Model Comparisons

Differences are reported as `model_b - model_a`. McNemar tests compare paired correctness outcomes on the same examples.

{markdown_table(pair_rows, ["model_a", "model_b", "a_correct_b_wrong", "a_wrong_b_correct", "accuracy_diff_b_minus_a", "macro_f1_diff_b_minus_a", "mcnemar_p"])}

## Interpretation

This check is closer to the target system setting than the archived Twitter diagnostic because the inputs are formal financial news snippets and article excerpts. If neutral-aware LoRA remains stronger than FinBERT on this dataset, it supports the claim that LoRA improves formal financial sentiment transfer beyond the Financial PhraseBank split. If the fixed learned stacking layer does not transfer as well as standalone LoRA, it should be framed as an in-domain fusion layer selected on Financial PhraseBank validation rather than a universally portable fusion rule.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary["overall"], indent=2))
    print(f"Saved analysis to: {output_dir}")


if __name__ == "__main__":
    main()
