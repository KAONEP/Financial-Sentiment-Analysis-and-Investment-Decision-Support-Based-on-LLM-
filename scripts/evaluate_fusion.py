from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finbert-predictions", required=True)
    parser.add_argument("--llm-predictions", required=True)
    parser.add_argument("--method", default="threshold", choices=["threshold", "weighted"])
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--output-dir", default="outputs/runs/fusion")
    return parser.parse_args()


def get_probs(df: pd.DataFrame) -> np.ndarray:
    return df[[f"prob_{label}" for label in LABELS]].to_numpy(dtype=np.float32)


def main() -> None:
    args = parse_args()
    finbert = pd.read_csv(args.finbert_predictions)
    llm = pd.read_csv(args.llm_predictions)

    merged = finbert[["example_id", "sentence", "label"]].merge(
        llm[["example_id"]],
        on="example_id",
        how="inner",
    )
    finbert = finbert.set_index("example_id").loc[merged["example_id"]].reset_index()
    llm = llm.set_index("example_id").loc[merged["example_id"]].reset_index()

    finbert_probs = get_probs(finbert)
    llm_probs = get_probs(llm)
    if args.method == "threshold":
        fused_probs = threshold_fusion(llm_probs, finbert_probs, threshold=args.threshold)
    else:
        fused_probs = weighted_fusion(llm_probs, finbert_probs)

    pred_idx = fused_probs.argmax(axis=1)
    predictions = [LABELS[idx] for idx in pred_idx]
    out = merged.copy()
    for idx, label in enumerate(LABELS):
        out[f"prob_{label}"] = fused_probs[:, idx]
    out["prediction"] = predictions
    out["confidence"] = fused_probs.max(axis=1)

    metrics = compute_classification_metrics(out["label"].tolist(), out["prediction"].tolist())
    out_dir = ensure_dir(args.output_dir)
    out.to_csv(out_dir / "predictions.csv", index=False)
    write_json(metrics, out_dir / "metrics.json")
    print(metrics)


if __name__ == "__main__":
    main()

