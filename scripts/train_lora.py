from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from financial_llm.config import load_config
from financial_llm.prompts import SYSTEM_PROMPT, direct_prompt, neutral_aware_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--val-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lora-dropout", type=float, default=None)
    parser.add_argument(
        "--target-modules",
        default=None,
        help="Comma-separated LoRA target modules, for example q_proj,k_proj,v_proj,o_proj.",
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--prompt-mode", default="direct", choices=["direct", "neutral_aware"])
    parser.add_argument("--label-smoothing-factor", type=float, default=0.0)
    return parser.parse_args()


def apply_chat_template(tokenizer, messages: list[dict], enable_thinking: bool, add_generation_prompt: bool) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def parse_target_modules(value: str | None, fallback: list[str]) -> list[str]:
    if value is None:
        return fallback
    modules = [item.strip() for item in value.split(",") if item.strip()]
    if not modules:
        raise ValueError("--target-modules must contain at least one module name.")
    return modules


def build_tokenized_dataset(
    df: pd.DataFrame,
    tokenizer,
    enable_thinking: bool,
    max_length: int,
    prompt_mode: str,
) -> Dataset:
    rows = []
    for item in df.to_dict(orient="records"):
        if prompt_mode == "neutral_aware":
            user_prompt = neutral_aware_prompt(item["sentence"])
        else:
            user_prompt = direct_prompt(item["sentence"])
        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = apply_chat_template(
            tokenizer,
            prompt_messages,
            enable_thinking=enable_thinking,
            add_generation_prompt=True,
        )
        full_text = prompt_text + item["label"] + tokenizer.eos_token

        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        ).input_ids
        encoded = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )

        input_ids = encoded.input_ids
        attention_mask = encoded.attention_mask
        labels = input_ids.copy()
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len

        pad_len = max_length - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len
            labels = labels + [-100] * pad_len

        rows.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        )
    return Dataset.from_list(rows)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    llm_cfg = config["models"]["llm"]
    lora_cfg = config["lora"]
    seed = int(args.seed if args.seed is not None else config["project"]["seed"])
    epochs = float(args.epochs if args.epochs is not None else lora_cfg["epochs"])
    learning_rate = float(args.learning_rate if args.learning_rate is not None else lora_cfg["learning_rate"])
    lora_r = int(args.lora_r if args.lora_r is not None else lora_cfg["r"])
    lora_alpha = int(args.lora_alpha if args.lora_alpha is not None else lora_cfg["alpha"])
    lora_dropout = float(args.lora_dropout if args.lora_dropout is not None else lora_cfg["dropout"])
    target_modules = parse_target_modules(args.target_modules, list(lora_cfg["target_modules"]))
    set_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_path = Path(args.train_file)
    if args.val_file is None:
        data_cfg = config["data"]
        val_path = Path(data_cfg["output_dir"]) / data_cfg["dataset_config"] / "val.csv"
    else:
        val_path = Path(args.val_file)

    run_name = args.run_name or train_path.stem
    output_dir = Path(args.output_dir or lora_cfg["output_dir"]) / run_name

    tokenizer = AutoTokenizer.from_pretrained(llm_cfg["model_name"], trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        llm_cfg["model_name"],
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=False,
    )
    model.config.use_cache = False
    if bool(lora_cfg.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    if args.max_train_samples is not None:
        train_df = train_df.sample(
            n=min(args.max_train_samples, len(train_df)),
            random_state=seed,
        ).reset_index(drop=True)
    if args.max_val_samples is not None:
        val_df = val_df.sample(
            n=min(args.max_val_samples, len(val_df)),
            random_state=seed,
        ).reset_index(drop=True)

    max_length = int(llm_cfg["max_seq_length"])
    train_ds = build_tokenized_dataset(
        train_df,
        tokenizer=tokenizer,
        enable_thinking=bool(llm_cfg.get("enable_thinking", False)),
        max_length=max_length,
        prompt_mode=args.prompt_mode,
    )
    val_ds = build_tokenized_dataset(
        val_df,
        tokenizer=tokenizer,
        enable_thinking=bool(llm_cfg.get("enable_thinking", False)),
        max_length=max_length,
        prompt_mode=args.prompt_mode,
    )

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)
    if bool(lora_cfg.get("gradient_checkpointing", True)) and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.print_trainable_parameters()
    trainable_params, total_params = model.get_nb_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=int(lora_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=int(lora_cfg["gradient_accumulation_steps"]),
        learning_rate=learning_rate,
        warmup_ratio=float(lora_cfg["warmup_ratio"]),
        weight_decay=float(lora_cfg["weight_decay"]),
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        gradient_checkpointing=bool(lora_cfg.get("gradient_checkpointing", True)),
        remove_unused_columns=False,
        save_total_limit=3,
        report_to=[],
        seed=seed,
        data_seed=seed,
        label_smoothing_factor=float(args.label_smoothing_factor),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )
    train_output = trainer.train()
    trainer.save_model(str(output_dir / "final_adapter"))
    tokenizer.save_pretrained(str(output_dir / "final_adapter"))
    run_config = {
        "run_name": run_name,
        "seed": seed,
        "train_file": str(train_path),
        "val_file": str(val_path),
        "train_examples": len(train_df),
        "val_examples": len(val_df),
        "base_model": llm_cfg["model_name"],
        "epochs": epochs,
        "learning_rate": learning_rate,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "target_modules": target_modules,
        "prompt_mode": args.prompt_mode,
        "label_smoothing_factor": float(args.label_smoothing_factor),
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_percent": 100.0 * trainable_params / total_params,
        "train_metrics": train_output.metrics,
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
