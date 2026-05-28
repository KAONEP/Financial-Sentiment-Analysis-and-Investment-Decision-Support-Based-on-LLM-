from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from financial_llm.config import ensure_dir, load_config
from financial_llm.io import load_split, probs_to_frame, write_json
from financial_llm.llm_classifier import LabelScoringLLM
from financial_llm.metrics import compute_classification_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-per-class", type=int, default=None)
    parser.add_argument("--prompt-mode", default="direct", choices=["direct", "reasoning"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_cfg = config["data"]
    model_cfg = config["models"]["llm"]

    split_path = Path(data_cfg["output_dir"]) / data_cfg["dataset_config"] / f"{args.split}.csv"
    df = load_split(
        split_path,
        limit=args.limit,
        sample_per_class=args.sample_per_class,
        seed=int(config["project"]["seed"]),
    )

    model = LabelScoringLLM(
        model_name=model_cfg["model_name"],
        dtype=model_cfg.get("dtype", "bfloat16"),
        enable_thinking=bool(model_cfg.get("enable_thinking", False)),
        max_length=int(model_cfg.get("max_seq_length", 512)),
        prompt_mode=args.prompt_mode,
    )
    probs = model.predict_proba(
        df["sentence"].tolist(),
        batch_size=int(model_cfg.get("batch_size", 8)),
    )
    pred_df = probs_to_frame(df, probs, model_name=model_cfg["model_name"])
    metrics = compute_classification_metrics(pred_df["label"].tolist(), pred_df["prediction"].tolist())

    safe_model_name = model_cfg["model_name"].replace("/", "__")
    prompt_dir = "zero_shot" if args.prompt_mode == "direct" else f"zero_shot_{args.prompt_mode}"
    split_name = args.split
    if args.sample_per_class is not None:
        split_name = f"{split_name}_perclass{args.sample_per_class}"
    if args.limit is not None:
        split_name = f"{split_name}_limit{args.limit}"
    out_dir = ensure_dir(Path(config["outputs"]["run_dir"]) / safe_model_name / prompt_dir / split_name)
    pred_df.to_csv(out_dir / "predictions.csv", index=False)
    write_json(metrics, out_dir / "metrics.json")
    print(metrics)


if __name__ == "__main__":
    main()
