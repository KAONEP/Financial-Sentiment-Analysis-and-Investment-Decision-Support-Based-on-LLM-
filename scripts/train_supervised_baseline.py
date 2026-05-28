from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from financial_llm.config import ensure_dir, load_config
from financial_llm.io import write_json
from financial_llm.labels import ID2LABEL, LABEL2ID, LABELS
from financial_llm.metrics import compute_classification_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=float, default=4.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs/supervised")
    parser.add_argument("--run-output-dir", default="outputs/runs/supervised")
    return parser.parse_args()


def load_split(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["label"].isin(LABELS)].reset_index(drop=True)
    return df


def tokenize_dataset(df: pd.DataFrame, tokenizer, max_length: int) -> Dataset:
    rows = []
    for item in df.to_dict(orient="records"):
        rows.append(
            {
                "example_id": item["example_id"],
                "sentence": item["sentence"],
                "label_name": item["label"],
                "labels": LABEL2ID[item["label"]],
            }
        )
    dataset = Dataset.from_list(rows)

    def tokenize(batch: dict) -> dict:
        return tokenizer(
            batch["sentence"],
            truncation=True,
            max_length=max_length,
        )

    return dataset.map(tokenize, batched=True)


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    y_true = [ID2LABEL[int(label)] for label in labels]
    y_pred = [ID2LABEL[int(pred)] for pred in preds]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0),
    }


def predict_frame(df: pd.DataFrame, logits: np.ndarray, model_name: str) -> pd.DataFrame:
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    pred_idx = probs.argmax(axis=1)
    out = df[["example_id", "sentence", "label"]].copy()
    for idx, label in enumerate(LABELS):
        out[f"prob_{label}"] = probs[:, idx]
    out["prediction"] = [ID2LABEL[int(idx)] for idx in pred_idx]
    out["confidence"] = probs.max(axis=1)
    out["model"] = model_name
    return out


def save_predictions(
    trainer: Trainer,
    df: pd.DataFrame,
    dataset: Dataset,
    output_dir: Path,
    model_name: str,
) -> dict:
    pred = trainer.predict(dataset)
    pred_df = predict_frame(df, pred.predictions, model_name=model_name)
    metrics = compute_classification_metrics(pred_df["label"].tolist(), pred_df["prediction"].tolist())
    output_dir = ensure_dir(output_dir)
    pred_df.to_csv(output_dir / "predictions.csv", index=False)
    write_json(metrics, output_dir / "metrics.json")
    return metrics


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(args.seed if args.seed is not None else config["project"]["seed"])
    set_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    data_dir = Path(config["data"]["output_dir"]) / config["data"]["dataset_config"]
    train_df = load_split(data_dir / "train.csv")
    val_df = load_split(data_dir / "val.csv")
    test_df = load_split(data_dir / "test.csv")

    run_name = args.run_name or args.model_name.replace("/", "__").replace("-", "_")
    model_output_dir = Path(args.output_dir) / run_name
    run_output_dir = Path(args.run_output_dir) / run_name

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_ds = tokenize_dataset(train_df, tokenizer, args.max_length)
    val_ds = tokenize_dataset(val_df, tokenizer, args.max_length)
    test_ds = tokenize_dataset(test_df, tokenizer, args.max_length)
    columns_to_remove = ["example_id", "sentence", "label_name"]
    train_ds = train_ds.remove_columns(columns_to_remove)
    val_ds = val_ds.remove_columns(columns_to_remove)
    test_ds = test_ds.remove_columns(columns_to_remove)

    training_args = TrainingArguments(
        output_dir=str(model_output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        warmup_ratio=0.06,
        logging_steps=20,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to=[],
        seed=seed,
        data_seed=seed,
        bf16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(str(model_output_dir / "final_model"))
    tokenizer.save_pretrained(str(model_output_dir / "final_model"))

    val_metrics = save_predictions(trainer, val_df, val_ds, run_output_dir / "val", args.model_name)
    test_metrics = save_predictions(trainer, test_df, test_ds, run_output_dir / "test", args.model_name)

    summary = {
        "model_name": args.model_name,
        "run_name": run_name,
        "train_examples": len(train_df),
        "val_examples": len(val_df),
        "test_examples": len(test_df),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "seed": seed,
        "val": {
            "accuracy": val_metrics["accuracy"],
            "macro_f1": val_metrics["macro_f1"],
            "weighted_f1": val_metrics["weighted_f1"],
        },
        "test": {
            "accuracy": test_metrics["accuracy"],
            "macro_f1": test_metrics["macro_f1"],
            "weighted_f1": test_metrics["weighted_f1"],
        },
    }
    write_json(summary, run_output_dir / "summary.json")
    print(summary)


if __name__ == "__main__":
    main()
