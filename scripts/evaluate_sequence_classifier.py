from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from financial_llm.config import ensure_dir
from financial_llm.io import load_split, probs_to_frame, write_json
from financial_llm.labels import LABELS
from financial_llm.metrics import compute_classification_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_split(args.data_file)
    model_name = args.model_name or args.model_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    id2label = {int(idx): str(label).lower() for idx, label in model.config.id2label.items()}
    outputs = []
    for start in range(0, len(df), args.batch_size):
        batch = df["sentence"].iloc[start : start + args.batch_size].tolist()
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        reordered = np.zeros((probs.shape[0], len(LABELS)), dtype=np.float32)
        for model_idx, model_label in id2label.items():
            if model_label not in LABELS:
                continue
            reordered[:, LABELS.index(model_label)] = probs[:, model_idx]
        row_sums = reordered.sum(axis=1, keepdims=True)
        if np.any(row_sums == 0):
            raise ValueError(f"Could not map model labels {id2label} to {LABELS}")
        outputs.append(reordered / row_sums)

    probs = np.concatenate(outputs, axis=0)
    pred_df = probs_to_frame(df, probs, model_name=model_name)
    metrics = compute_classification_metrics(pred_df["label"].tolist(), pred_df["prediction"].tolist())
    out_dir = ensure_dir(args.output_dir)
    pred_df.to_csv(out_dir / "predictions.csv", index=False)
    write_json(metrics, out_dir / "metrics.json")
    print(metrics)


if __name__ == "__main__":
    main()
