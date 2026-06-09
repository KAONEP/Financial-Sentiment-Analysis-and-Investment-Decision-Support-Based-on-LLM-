from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from financial_llm.config import ensure_dir


LABELS = ["negative", "neutral", "positive"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="NOSIBLE/financial-sentiment")
    parser.add_argument("--output-dir", default="data/external/nosible_financial_sentiment_full")
    parser.add_argument(
        "--source-parquet",
        default=None,
        help="Optional local NOSIBLE parquet file. If omitted, the script first checks the Hugging Face cache.",
    )
    parser.add_argument("--sample-per-class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalize_label(value: object) -> str:
    text = str(value).strip().lower()
    if text in LABELS:
        return text
    if text in {"0", "negative", "neg"}:
        return "negative"
    if text in {"1", "neutral", "neu"}:
        return "neutral"
    if text in {"2", "positive", "pos"}:
        return "positive"
    raise ValueError(f"Unsupported NOSIBLE label: {value!r}")


def find_cached_parquet(dataset_name: str) -> Path | None:
    if dataset_name != "NOSIBLE/financial-sentiment":
        return None
    candidates = []
    home = Path(os.environ.get("USERPROFILE") or Path.home())
    cache_root = Path(os.environ.get("HF_HOME", home / ".cache" / "huggingface"))
    hub_root = cache_root / "hub" / "datasets--NOSIBLE--financial-sentiment" / "snapshots"
    if hub_root.exists():
        candidates.extend(hub_root.glob("*/data.parquet"))
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def load_source_frame(args: argparse.Namespace) -> pd.DataFrame:
    source_parquet = Path(args.source_parquet) if args.source_parquet else find_cached_parquet(args.dataset_name)
    if source_parquet is not None and source_parquet.exists():
        return pd.read_parquet(source_parquet)

    from datasets import load_dataset

    dataset = load_dataset(args.dataset_name, split="train")
    return dataset.to_pandas()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    df = load_source_frame(args)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"Expected text and label columns, got: {list(df.columns)}")

    df = df.copy()
    df["label"] = df["label"].map(normalize_label)
    df["sentence"] = df["text"].astype(str).str.strip()
    df = df[df["sentence"].astype(bool)].reset_index(drop=True)

    if args.sample_per_class is None:
        sampled = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    else:
        sampled_parts = []
        for _, group in df.groupby("label"):
            sampled_parts.append(
                group.sample(n=min(args.sample_per_class, len(group)), random_state=args.seed)
            )
        sampled = pd.concat(sampled_parts).reset_index(drop=True)
        sampled = sampled.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    sampled["example_id"] = [f"nosible_{idx:06d}" for idx in range(len(sampled))]

    keep_cols = ["example_id", "sentence", "label"]
    for optional_col in ["url", "netloc"]:
        if optional_col in sampled.columns:
            keep_cols.append(optional_col)
    output = sampled[keep_cols]
    output.to_csv(out_dir / "test.csv", index=False)

    manifest = {
        "dataset_name": args.dataset_name,
        "source_url": "https://huggingface.co/datasets/NOSIBLE/financial-sentiment",
        "source_license": "ODC-BY, according to the Hugging Face dataset card",
        "purpose": "formal financial news external robustness check",
        "sample_per_class": args.sample_per_class,
        "seed": args.seed,
        "rows": len(output),
        "label_counts": output["label"].value_counts().to_dict(),
        "word_count_summary": {
            "mean": float(output["sentence"].str.split().str.len().mean()),
            "median": float(output["sentence"].str.split().str.len().median()),
            "max": int(output["sentence"].str.split().str.len().max()),
        },
        "top_sources": output["netloc"].value_counts().head(20).to_dict()
        if "netloc" in output.columns
        else {},
        "path": str(out_dir / "test.csv"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
