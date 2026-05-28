from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from financial_llm.config import ensure_dir, load_config
from financial_llm.finbert import FinBertSentiment
from financial_llm.io import load_split, probs_to_frame, write_json
from financial_llm.metrics import compute_classification_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-per-class", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_cfg = config["data"]
    model_cfg = config["models"]["finbert"]

    if args.data_file is None:
        split_path = Path(data_cfg["output_dir"]) / data_cfg["dataset_config"] / f"{args.split}.csv"
    else:
        split_path = Path(args.data_file)
    df = load_split(
        split_path,
        limit=args.limit,
        sample_per_class=args.sample_per_class,
        seed=int(config["project"]["seed"]),
    )

    model = FinBertSentiment(model_cfg["model_name"])
    probs = model.predict_proba(df["sentence"].tolist(), batch_size=int(model_cfg["batch_size"]))
    pred_df = probs_to_frame(df, probs, model_name=model_cfg["model_name"])
    metrics = compute_classification_metrics(pred_df["label"].tolist(), pred_df["prediction"].tolist())

    split_name = args.split
    if args.sample_per_class is not None:
        split_name = f"{split_name}_perclass{args.sample_per_class}"
    if args.limit is not None:
        split_name = f"{split_name}_limit{args.limit}"
    if args.output_dir is None:
        out_dir = Path(config["outputs"]["run_dir"]) / "finbert" / split_name
    else:
        out_dir = Path(args.output_dir)
    out_dir = ensure_dir(out_dir)
    pred_df.to_csv(out_dir / "predictions.csv", index=False)
    write_json(metrics, out_dir / "metrics.json")
    print(metrics)


if __name__ == "__main__":
    main()
