from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from financial_llm.config import ensure_dir


LABEL_MAP = {
    0: "negative",  # Bearish
    1: "positive",  # Bullish
    2: "neutral",  # Neutral
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="zeroshot/twitter-financial-news-sentiment")
    parser.add_argument("--output-dir", default="data/external/twitter_financial_news_sentiment")
    return parser.parse_args()


def convert_split(split_df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    df = split_df.copy()
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"Expected text and label columns, got: {list(df.columns)}")
    out = pd.DataFrame(
        {
            "example_id": [f"tfns_{split_name}_{idx:05d}" for idx in range(len(df))],
            "sentence": df["text"].astype(str),
            "label": df["label"].map(lambda value: LABEL_MAP[int(value)]),
            "source_label": df["label"].astype(int),
        }
    )
    return out


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    dataset = load_dataset(args.dataset_name)

    manifest = {
        "dataset_name": args.dataset_name,
        "task": "external financial sentiment robustness",
        "label_mapping": {
            "LABEL_0": "negative / Bearish",
            "LABEL_1": "positive / Bullish",
            "LABEL_2": "neutral / Neutral",
        },
        "splits": {},
    }

    for split_name, split in dataset.items():
        df = convert_split(split.to_pandas(), split_name)
        df.to_csv(out_dir / f"{split_name}.csv", index=False)
        manifest["splits"][split_name] = {
            "rows": len(df),
            "label_counts": df["label"].value_counts().to_dict(),
            "path": str(out_dir / f"{split_name}.csv"),
        }

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
