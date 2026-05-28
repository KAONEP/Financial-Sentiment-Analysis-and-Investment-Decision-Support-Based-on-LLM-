from __future__ import annotations

import argparse
import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from financial_llm.labels import LABELS


AGREEMENT_FILES = {
    "50agree_test": None,
    "66agree_test": "FinancialPhraseBank-v1.0/Sentences_66Agree.txt",
    "75agree_test": "FinancialPhraseBank-v1.0/Sentences_75Agree.txt",
    "allagree_test": "FinancialPhraseBank-v1.0/Sentences_AllAgree.txt",
}

MODEL_PATHS = {
    "Strict_BERT": "outputs/runs/supervised/bert_base_uncased_strict/test/predictions.csv",
    "FinBERT": "outputs/runs/finbert/test/predictions.csv",
    "LoRA_r16": "outputs/runs/Qwen__Qwen3-4B/lora/train_frac100_raw/direct/test/predictions.csv",
    "LoRA_r8": "outputs/runs/Qwen__Qwen3-4B/lora/ablation_lora_r8_full_raw_seed42/direct/test/predictions.csv",
    "Fusion_r16_threshold": "outputs/runs/fusion/finbert_qwen3_lora100_raw/threshold/test/predictions.csv",
    "Fusion_r16_weighted": "outputs/runs/fusion/finbert_qwen3_lora100_raw/weighted/test/predictions.csv",
    "Fusion_r16_threshold_calibrated": (
        "outputs/runs/fusion/finbert_qwen3_lora100_raw_calibrated/threshold/test/predictions.csv"
    ),
    "Fusion_r16_weighted_calibrated": (
        "outputs/runs/fusion/finbert_qwen3_lora100_raw_calibrated/weighted/test/predictions.csv"
    ),
    "Fusion_r8_threshold": "outputs/runs/fusion/finbert_qwen3_lora_r8_full_raw_seed42/threshold/test/predictions.csv",
    "Fusion_r8_weighted": "outputs/runs/fusion/finbert_qwen3_lora_r8_full_raw_seed42/weighted/test/predictions.csv",
    "Fusion_r8_threshold_calibrated": (
        "outputs/runs/fusion/finbert_qwen3_lora_r8_full_raw_seed42_calibrated/threshold/test/predictions.csv"
    ),
    "Fusion_r8_weighted_calibrated": (
        "outputs/runs/fusion/finbert_qwen3_lora_r8_full_raw_seed42_calibrated/weighted/test/predictions.csv"
    ),
}

COMPARISONS = [
    ("FinBERT", "Strict_BERT"),
    ("FinBERT", "LoRA_r16"),
    ("FinBERT", "LoRA_r8"),
    ("FinBERT", "Fusion_r16_weighted"),
    ("FinBERT", "Fusion_r8_threshold_calibrated"),
    ("LoRA_r16", "LoRA_r8"),
    ("Fusion_r16_weighted", "Fusion_r8_threshold_calibrated"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", default="data/processed/sentences_50agree/test.csv")
    parser.add_argument("--phrasebank-zip", default="data/raw/financial_phrasebank/data/FinancialPhraseBank-v1.0.zip")
    parser.add_argument("--output-dir", default="outputs/analysis/agreement_robustness")
    parser.add_argument("--report-path", default="reports/agreement_robustness.md")
    return parser.parse_args()


def read_phrasebank_subset(zip_path: str | Path, archive_member: str) -> set[tuple[str, str]]:
    rows = set()
    with zipfile.ZipFile(zip_path) as archive:
        content = archive.read(archive_member).decode("latin-1")
    for line in content.splitlines():
        if not line.strip():
            continue
        sentence, label = line.rsplit("@", 1)
        label = label.strip().lower()
        if label in LABELS:
            rows.add((sentence.strip(), label))
    return rows


def read_predictions(path: str | Path, model_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"example_id", "sentence", "label", "prediction"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return df[["example_id", "sentence", "label", "prediction"]].rename(columns={"prediction": model_name})


def align_predictions() -> pd.DataFrame:
    names = list(MODEL_PATHS)
    merged = read_predictions(MODEL_PATHS[names[0]], names[0])
    for name in names[1:]:
        df = read_predictions(MODEL_PATHS[name], name).rename(
            columns={"sentence": f"sentence_{name}", "label": f"label_{name}"}
        )
        merged = merged.merge(df, on="example_id", how="inner")
        if not (merged["sentence"] == merged[f"sentence_{name}"]).all():
            raise ValueError(f"Sentences do not align for {name}")
        if not (merged["label"] == merged[f"label_{name}"]).all():
            raise ValueError(f"Labels do not align for {name}")
        merged = merged.drop(columns=[f"sentence_{name}", f"label_{name}"])
    return merged


def metrics_for(df: pd.DataFrame, prediction_col: str) -> dict[str, float]:
    y_true = df["label"].tolist()
    y_pred = df[prediction_col].tolist()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)),
    }


def exact_mcnemar_p_value(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    observed = min(b, c)
    tail = sum(math.comb(n, k) * (0.5**n) for k in range(observed + 1))
    return float(min(1.0, 2.0 * tail))


def paired_comparison(df: pd.DataFrame, model_a: str, model_b: str) -> dict[str, float | int | str]:
    correct_a = df[model_a] == df["label"]
    correct_b = df[model_b] == df["label"]
    a_only = int((correct_a & ~correct_b).sum())
    b_only = int((~correct_a & correct_b).sum())
    metrics_a = metrics_for(df, model_a)
    metrics_b = metrics_for(df, model_b)
    return {
        "model_a": model_a,
        "model_b": model_b,
        "a_correct_b_wrong": a_only,
        "a_wrong_b_correct": b_only,
        "accuracy_diff": metrics_b["accuracy"] - metrics_a["accuracy"],
        "macro_f1_diff": metrics_b["macro_f1"] - metrics_a["macro_f1"],
        "mcnemar_p": exact_mcnemar_p_value(a_only, b_only),
    }


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


def pivot_metric_table(metric_rows: list[dict], metric: str) -> list[dict]:
    df = pd.DataFrame(metric_rows)
    pivot = df.pivot(index="subset", columns="model", values=metric).reset_index()
    ordered_columns = ["subset", *[model for model in MODEL_PATHS if model in pivot.columns]]
    pivot = pivot[ordered_columns]
    return pivot.to_dict(orient="records")


def best_model_rows(metric_rows: list[dict]) -> list[dict]:
    rows = []
    df = pd.DataFrame(metric_rows)
    for subset, group in df.groupby("subset", sort=False):
        best_acc = group.sort_values(["accuracy", "macro_f1"], ascending=False).iloc[0]
        best_macro = group.sort_values(["macro_f1", "accuracy"], ascending=False).iloc[0]
        rows.append(
            {
                "subset": subset,
                "best_accuracy_model": best_acc["model"],
                "best_accuracy": float(best_acc["accuracy"]),
                "best_macro_f1_model": best_macro["model"],
                "best_macro_f1": float(best_macro["macro_f1"]),
            }
        )
    return rows


def write_report(
    report_path: Path,
    subset_rows: list[dict],
    metric_rows: list[dict],
    comparison_rows: list[dict],
) -> None:
    model_rows = [
        {"model": model, "prediction_file": path}
        for model, path in MODEL_PATHS.items()
    ]
    macro_rows = pivot_metric_table(metric_rows, "macro_f1")
    best_rows = best_model_rows(metric_rows)
    report = f"""# Financial PhraseBank Agreement Robustness

## Purpose

This report evaluates whether the current model ranking is stable on higher-annotator-agreement subsets of Financial PhraseBank.

The check is leakage-safe: each higher-agreement subset is formed by intersecting the original fixed `sentences_50agree` test split with the corresponding official higher-agreement file. No training examples are added to evaluation.

```text
test_66agree = current_test_split intersect Sentences_66Agree.txt
test_75agree = current_test_split intersect Sentences_75Agree.txt
test_allagree = current_test_split intersect Sentences_AllAgree.txt
```

## Compared Models

{table(model_rows)}

## Subset Sizes

{table(subset_rows)}

## Macro-F1 by Agreement Level

{table(macro_rows)}

## Best Model by Subset

{table(best_rows)}

## Full Model Metrics

{table(metric_rows)}

## Paired Comparisons

{table(comparison_rows)}

## Interpretation

The scores increase as annotation agreement becomes stricter, which is expected because higher-agreement examples are less ambiguous. The selected r8 fusion and the earlier r16 fusion settings remain stronger than FinBERT across all agreement levels, so the main fusion-over-FinBERT conclusion is robust to label-noise reduction.

The r8 configuration is more competitive than the original r16 setting across the agreement levels, especially when used with threshold fusion. On `75agree_test` and `allagree_test`, standalone `LoRA_r8` is extremely strong and can slightly exceed fusion in macro-F1 while matching or nearly matching fusion in accuracy. This suggests that the clearest high-agreement examples leave less room for FinBERT-LoRA complementarity. Therefore, the final report should state that fusion is the strongest overall method on the full and 66-agreement test sets, remains tied or competitive on higher-agreement samples, and consistently improves over FinBERT.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(args.test_file)
    merged = align_predictions()
    if len(merged) != len(test_df):
        raise ValueError(f"Prediction/test size mismatch: predictions={len(merged)}, test={len(test_df)}")

    agreement_sets = {
        name: None if archive_member is None else read_phrasebank_subset(args.phrasebank_zip, archive_member)
        for name, archive_member in AGREEMENT_FILES.items()
    }

    subset_rows = []
    metric_rows = []
    comparison_rows = []
    summary: dict[str, dict] = {}

    for subset_name, agreement_set in agreement_sets.items():
        if agreement_set is None:
            subset = merged.copy()
        else:
            mask = [(sentence, label) in agreement_set for sentence, label in zip(merged["sentence"], merged["label"])]
            subset = merged.loc[mask].copy()

        label_counts = subset["label"].value_counts().to_dict()
        subset_rows.append(
            {
                "subset": subset_name,
                "n": len(subset),
                "negative": int(label_counts.get("negative", 0)),
                "neutral": int(label_counts.get("neutral", 0)),
                "positive": int(label_counts.get("positive", 0)),
            }
        )
        subset.to_csv(output_dir / f"{subset_name}_aligned_predictions.csv", index=False)

        summary[subset_name] = {"n": len(subset), "label_counts": label_counts, "metrics": {}}
        for model_name in MODEL_PATHS:
            model_metrics = metrics_for(subset, model_name)
            summary[subset_name]["metrics"][model_name] = model_metrics
            metric_rows.append(
                {
                    "subset": subset_name,
                    "model": model_name,
                    "accuracy": model_metrics["accuracy"],
                    "macro_f1": model_metrics["macro_f1"],
                    "weighted_f1": model_metrics["weighted_f1"],
                }
            )

        for model_a, model_b in COMPARISONS:
            comparison = paired_comparison(subset, model_a, model_b)
            comparison_rows.append({"subset": subset_name, **comparison})

    pd.DataFrame(subset_rows).to_csv(output_dir / "subset_sizes.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(output_dir / "metrics_by_subset.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(output_dir / "paired_comparisons.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(Path(args.report_path), subset_rows, metric_rows, comparison_rows)

    print(json.dumps({"subsets": subset_rows, "metrics": metric_rows, "comparisons": comparison_rows}, indent=2))
    print(f"Saved agreement robustness outputs to: {output_dir}")
    print(f"Saved report to: {args.report_path}")


if __name__ == "__main__":
    main()
