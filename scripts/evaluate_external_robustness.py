from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from financial_llm.config import ensure_dir
from financial_llm.fusion import threshold_fusion, weighted_fusion
from financial_llm.io import write_json
from financial_llm.labels import LABELS
from financial_llm.metrics import compute_classification_metrics


MODEL_PATHS = {
    "Strict_BERT": "outputs/runs/external/twitter_financial_news_sentiment/strict_bert/validation/predictions.csv",
    "FinBERT": "outputs/runs/external/twitter_financial_news_sentiment/finbert/validation/predictions.csv",
    "LoRA_r16": "outputs/runs/external/twitter_financial_news_sentiment/lora_r16/validation/predictions.csv",
    "LoRA_r8": "outputs/runs/external/twitter_financial_news_sentiment/lora_r8/validation/predictions.csv",
}

FUSION_CONFIGS = {
    "Fusion_r16_threshold_fixed": {
        "llm": "LoRA_r16",
        "method": "threshold",
        "threshold": 0.86,
    },
    "Fusion_r16_weighted": {
        "llm": "LoRA_r16",
        "method": "weighted",
    },
    "Fusion_r16_threshold_calibrated_fixed": {
        "llm": "LoRA_r16",
        "method": "threshold",
        "threshold": 0.74,
        "finbert_temperature": 0.958,
        "llm_temperature": 1.547,
    },
    "Fusion_r16_weighted_calibrated": {
        "llm": "LoRA_r16",
        "method": "weighted",
        "finbert_temperature": 0.958,
        "llm_temperature": 1.547,
    },
    "Fusion_r8_threshold_fixed": {
        "llm": "LoRA_r8",
        "method": "threshold",
        "threshold": 0.85,
    },
    "Fusion_r8_weighted": {
        "llm": "LoRA_r8",
        "method": "weighted",
    },
    "Fusion_r8_threshold_calibrated_fixed": {
        "llm": "LoRA_r8",
        "method": "threshold",
        "threshold": 0.78,
        "finbert_temperature": 0.958,
        "llm_temperature": 1.365,
    },
    "Fusion_r8_weighted_calibrated": {
        "llm": "LoRA_r8",
        "method": "weighted",
        "finbert_temperature": 0.958,
        "llm_temperature": 1.365,
    },
}

COMPARISONS = [
    ("FinBERT", "LoRA_r8"),
    ("FinBERT", "Fusion_r8_weighted"),
    ("FinBERT", "Fusion_r8_threshold_fixed"),
    ("LoRA_r16", "LoRA_r8"),
    ("Fusion_r16_weighted", "Fusion_r8_weighted"),
    ("LoRA_r8", "Fusion_r8_weighted"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/analysis/external_robustness/twitter_financial_news_sentiment")
    parser.add_argument("--report-path", default="reports/external_robustness_twitter.md")
    return parser.parse_args()


def get_probs(df: pd.DataFrame) -> np.ndarray:
    return df[[f"prob_{label}" for label in LABELS]].to_numpy(dtype=np.float64)


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def temperature_scale(probs: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probs, 1e-12, 1.0))
    return softmax(logits / temperature)


def load_predictions() -> dict[str, pd.DataFrame]:
    frames = {}
    for name, path in MODEL_PATHS.items():
        frames[name] = pd.read_csv(path)
    base_ids = frames["FinBERT"]["example_id"].tolist()
    for name, df in frames.items():
        if df["example_id"].tolist() != base_ids:
            raise ValueError(f"Example ordering mismatch for {name}")
    return frames


def make_prediction_frame(base: pd.DataFrame, probs: np.ndarray, model_name: str) -> pd.DataFrame:
    out = base[["example_id", "sentence", "label"]].copy()
    for idx, label in enumerate(LABELS):
        out[f"prob_{label}"] = probs[:, idx]
    pred_idx = probs.argmax(axis=1)
    out["prediction"] = [LABELS[idx] for idx in pred_idx]
    out["confidence"] = probs.max(axis=1)
    out["model"] = model_name
    return out


def evaluate_frame(df: pd.DataFrame) -> dict:
    return compute_classification_metrics(df["label"].tolist(), df["prediction"].tolist())


def exact_mcnemar_p_value(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    observed = min(b, c)
    tail = sum(math.comb(n, k) * (0.5**n) for k in range(observed + 1))
    return float(min(1.0, 2.0 * tail))


def paired_comparison(frames: dict[str, pd.DataFrame], model_a: str, model_b: str) -> dict:
    df_a = frames[model_a]
    df_b = frames[model_b]
    correct_a = df_a["prediction"] == df_a["label"]
    correct_b = df_b["prediction"] == df_b["label"]
    a_only = int((correct_a & ~correct_b).sum())
    b_only = int((~correct_a & correct_b).sum())
    metrics_a = evaluate_frame(df_a)
    metrics_b = evaluate_frame(df_b)
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


def write_report(
    report_path: Path,
    rows: list[dict],
    comparison_rows: list[dict],
    source_rows: list[dict],
    label_rows: list[dict],
) -> None:
    report = f"""# External Robustness: Twitter Financial News Sentiment

## Purpose

This report evaluates strict out-of-domain transfer from Financial PhraseBank to Twitter Financial News Sentiment. No model is retrained and no fusion threshold is tuned on the external dataset.

Dataset source:

```text
zeroshot/twitter-financial-news-sentiment
https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment
```

The label mapping is:

```text
Bearish -> negative
Bullish -> positive
Neutral -> neutral
```

## Evaluation Split

{table(label_rows)}

## Source Predictions

{table(source_rows)}

## Results

{table(rows)}

## Paired Comparisons

{table(comparison_rows)}

## Interpretation

This is a strict OOD check. All models drop relative to Financial PhraseBank because tweets contain ticker symbols, short headlines, URLs, and market-specific wording that differ from PhraseBank sentences.

The strongest result is `Fusion_r8_weighted`, but its advantage over `LoRA_r8` is negligible: the paired comparison shows only a +0.0004 macro-F1 difference and McNemar p=0.7465. Therefore, the external conclusion should not overclaim fusion improvement over LoRA.

The robust external finding is that LoRA adaptation transfers better than FinBERT on this dataset: `LoRA_r8` improves over FinBERT by +0.1213 macro-F1, and the paired correctness difference is highly significant. The r8 adapter also clearly improves over the original r16 adapter.

Fixed-threshold fusion is weaker than LoRA r8 and weighted fusion on this external dataset. This suggests that thresholds selected on Financial PhraseBank validation do not transfer perfectly under distribution shift. In the final report, this should be framed as a limitation of confidence-threshold transfer rather than a failure of LoRA.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    frames = load_predictions()
    base = frames["FinBERT"][["example_id", "sentence", "label"]].copy()
    label_counts = base["label"].value_counts().to_dict()
    label_rows = [
        {
            "split": "validation",
            "n": len(base),
            "negative": int(label_counts.get("negative", 0)),
            "neutral": int(label_counts.get("neutral", 0)),
            "positive": int(label_counts.get("positive", 0)),
        }
    ]

    source_rows = []
    result_rows = []
    for name, df in frames.items():
        metrics = evaluate_frame(df)
        source_rows.append(
            {
                "model": name,
                "prediction_file": MODEL_PATHS[name],
            }
        )
        result_rows.append(
            {
                "model": name,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
            }
        )

    finbert_probs = get_probs(frames["FinBERT"])
    for fusion_name, cfg in FUSION_CONFIGS.items():
        llm_probs = get_probs(frames[cfg["llm"]])
        used_finbert_probs = finbert_probs
        used_llm_probs = llm_probs
        if "finbert_temperature" in cfg:
            used_finbert_probs = temperature_scale(finbert_probs, float(cfg["finbert_temperature"]))
            used_llm_probs = temperature_scale(llm_probs, float(cfg["llm_temperature"]))
        if cfg["method"] == "threshold":
            probs = threshold_fusion(used_llm_probs, used_finbert_probs, float(cfg["threshold"]))
        elif cfg["method"] == "weighted":
            probs = weighted_fusion(used_llm_probs, used_finbert_probs)
        else:
            raise ValueError(f"Unsupported fusion method: {cfg['method']}")

        pred_df = make_prediction_frame(base, probs, fusion_name)
        metrics = evaluate_frame(pred_df)
        out_dir = ensure_dir(output_dir / fusion_name)
        pred_df.to_csv(out_dir / "predictions.csv", index=False)
        write_json(metrics, out_dir / "metrics.json")
        frames[fusion_name] = pred_df
        result_rows.append(
            {
                "model": fusion_name,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
            }
        )

    result_df = pd.DataFrame(result_rows).sort_values(["macro_f1", "accuracy"], ascending=False)
    comparison_rows = [paired_comparison(frames, model_a, model_b) for model_a, model_b in COMPARISONS]
    pd.DataFrame(comparison_rows).to_csv(output_dir / "paired_comparisons.csv", index=False)
    result_df.to_csv(output_dir / "summary.csv", index=False)
    write_json({"results": result_rows, "comparisons": comparison_rows}, output_dir / "summary.json")
    write_report(Path(args.report_path), result_df.to_dict(orient="records"), comparison_rows, source_rows, label_rows)
    print(json.dumps({"results": result_df.to_dict(orient="records"), "comparisons": comparison_rows}, indent=2))
    print(f"Saved external robustness outputs to: {output_dir}")
    print(f"Saved report to: {args.report_path}")


if __name__ == "__main__":
    main()
