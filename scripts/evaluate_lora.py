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
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-per-class", type=int, default=None)
    parser.add_argument("--prompt-mode", default="direct", choices=["direct", "reasoning", "neutral_aware"])
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def adapter_run_name(adapter_path: Path) -> str:
    if adapter_path.name == "final_adapter":
        return adapter_path.parent.name
    return adapter_path.name


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_cfg = config["data"]
    model_cfg = config["models"]["llm"]

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

    adapter_path = Path(args.adapter_path)
    model = LabelScoringLLM(
        model_name=model_cfg["model_name"],
        dtype=model_cfg.get("dtype", "bfloat16"),
        enable_thinking=bool(model_cfg.get("enable_thinking", False)),
        max_length=int(model_cfg.get("max_seq_length", 512)),
        prompt_mode=args.prompt_mode,
        adapter_path=str(adapter_path),
    )
    probs = model.predict_proba(
        df["sentence"].tolist(),
        batch_size=int(model_cfg.get("batch_size", 8)),
    )
    pred_df = probs_to_frame(df, probs, model_name=f"{model_cfg['model_name']}+LoRA")
    metrics = compute_classification_metrics(pred_df["label"].tolist(), pred_df["prediction"].tolist())

    safe_model_name = model_cfg["model_name"].replace("/", "__")
    run_name = adapter_run_name(adapter_path)
    prompt_dir = "direct" if args.prompt_mode == "direct" else args.prompt_mode
    split_name = args.split
    if args.sample_per_class is not None:
        split_name = f"{split_name}_perclass{args.sample_per_class}"
    if args.limit is not None:
        split_name = f"{split_name}_limit{args.limit}"

    if args.output_dir is None:
        out_dir = (
            Path(config["outputs"]["run_dir"])
            / safe_model_name
            / "lora"
            / run_name
            / prompt_dir
            / split_name
        )
    else:
        out_dir = Path(args.output_dir)
    out_dir = ensure_dir(out_dir)
    pred_df.to_csv(out_dir / "predictions.csv", index=False)
    write_json(metrics, out_dir / "metrics.json")
    print(metrics)


if __name__ == "__main__":
    main()
