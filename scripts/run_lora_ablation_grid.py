from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "data" / "processed" / "sentences_50agree" / "train_frac100_raw.csv"
OUTPUT_ROOT = ROOT / "outputs" / "analysis" / "lora_hyperparameter_tuning"
LORA_ROOT = ROOT / "outputs" / "lora"
RUNS_ROOT = ROOT / "outputs" / "runs" / "Qwen__Qwen3-4B" / "lora"

ATTENTION_MODULES = "q_proj,k_proj,v_proj,o_proj"
MLP_MODULES = "gate_proj,up_proj,down_proj"
BOTH_MODULES = f"{ATTENTION_MODULES},{MLP_MODULES}"


@dataclass(frozen=True)
class Experiment:
    run_name: str
    group: str
    seed: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    learning_rate: float
    target_modules: str
    notes: str


EXPERIMENTS = [
    Experiment(
        "neutral_aware_lora_r8_full_raw_seed42",
        "center",
        42,
        8,
        16,
        0.05,
        1e-4,
        BOTH_MODULES,
        "Existing final center point.",
    ),
    Experiment("neutral_aware_lora_r4_both_dropout005_lr1e4_seed42", "rank", 42, 4, 8, 0.05, 1e-4, BOTH_MODULES, "Rank sweep."),
    Experiment("neutral_aware_lora_r16_both_dropout005_lr1e4_seed42", "rank", 42, 16, 32, 0.05, 1e-4, BOTH_MODULES, "Rank sweep."),
    Experiment("neutral_aware_lora_r8_attn_dropout005_lr1e4_seed42", "target_modules", 42, 8, 16, 0.05, 1e-4, ATTENTION_MODULES, "Attention-only LoRA."),
    Experiment("neutral_aware_lora_r8_mlp_dropout005_lr1e4_seed42", "target_modules", 42, 8, 16, 0.05, 1e-4, MLP_MODULES, "MLP-only LoRA."),
    Experiment("neutral_aware_lora_r8_both_dropout000_lr1e4_seed42", "dropout", 42, 8, 16, 0.0, 1e-4, BOTH_MODULES, "Dropout sweep."),
    Experiment("neutral_aware_lora_r8_both_dropout010_lr1e4_seed42", "dropout", 42, 8, 16, 0.10, 1e-4, BOTH_MODULES, "Dropout sweep."),
    Experiment("neutral_aware_lora_r8_both_dropout005_lr5e5_seed42", "learning_rate", 42, 8, 16, 0.05, 5e-5, BOTH_MODULES, "Learning-rate sweep."),
    Experiment("neutral_aware_lora_r8_both_dropout005_lr2e4_seed42", "learning_rate", 42, 8, 16, 0.05, 2e-4, BOTH_MODULES, "Learning-rate sweep."),
    Experiment("neutral_aware_lora_r8_both_dropout005_lr1e4_seed123", "seed", 123, 8, 16, 0.05, 1e-4, BOTH_MODULES, "Final-setting stability."),
    Experiment("neutral_aware_lora_r8_both_dropout005_lr1e4_seed2026", "seed", 2026, 8, 16, 0.05, 1e-4, BOTH_MODULES, "Final-setting stability."),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--prompt-mode", default="neutral_aware", choices=["neutral_aware"])
    parser.add_argument("--start-at", default=None, help="Skip experiments until this run_name is reached.")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args()


def metrics_path(run_name: str, split: str) -> Path:
    return RUNS_ROOT / run_name / "neutral_aware" / split / "metrics.json"


def adapter_path(run_name: str) -> Path:
    return LORA_ROOT / run_name / "final_adapter"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def append_status(row: dict) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_ROOT / "status.csv"
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "timestamp",
                "run_name",
                "group",
                "stage",
                "status",
                "returncode",
                "elapsed_sec",
                "log_path",
                "message",
            ],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def log_status(exp: Experiment, stage: str, status: str, returncode: int | None, elapsed: float, log_path: Path | None, message: str) -> None:
    append_status(
        {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "run_name": exp.run_name,
            "group": exp.group,
            "stage": stage,
            "status": status,
            "returncode": "" if returncode is None else returncode,
            "elapsed_sec": f"{elapsed:.1f}",
            "log_path": "" if log_path is None else str(log_path),
            "message": message,
        }
    )


def run_command(exp: Experiment, stage: str, command: list[str]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT_ROOT / f"{exp.run_name}_{stage}.log"
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write("$ " + " ".join(command) + "\n\n")
        log_handle.flush()
        process = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    elapsed = time.time() - started
    if process.returncode != 0:
        log_status(exp, stage, "failed", process.returncode, elapsed, log_path, "Command failed.")
        raise RuntimeError(f"{exp.run_name} {stage} failed; see {log_path}")
    log_status(exp, stage, "completed", process.returncode, elapsed, log_path, "Command completed.")


def train_command(exp: Experiment, config: str, prompt_mode: str) -> list[str]:
    return [
        sys.executable,
        "scripts/train_lora.py",
        "--config",
        config,
        "--train-file",
        str(TRAIN_FILE),
        "--run-name",
        exp.run_name,
        "--seed",
        str(exp.seed),
        "--prompt-mode",
        prompt_mode,
        "--lora-r",
        str(exp.lora_r),
        "--lora-alpha",
        str(exp.lora_alpha),
        "--lora-dropout",
        str(exp.lora_dropout),
        "--learning-rate",
        str(exp.learning_rate),
        "--target-modules",
        exp.target_modules,
        "--auto-resume",
    ]


def eval_command(exp: Experiment, config: str, prompt_mode: str, split: str) -> list[str]:
    return [
        sys.executable,
        "scripts/evaluate_lora.py",
        "--config",
        config,
        "--adapter-path",
        str(adapter_path(exp.run_name)),
        "--split",
        split,
        "--prompt-mode",
        prompt_mode,
    ]


def experiment_complete(exp: Experiment) -> bool:
    return adapter_path(exp.run_name).exists() and metrics_path(exp.run_name, "val").exists() and metrics_path(exp.run_name, "test").exists()


def run_experiment(exp: Experiment, config: str, prompt_mode: str, dry_run: bool) -> None:
    if experiment_complete(exp):
        log_status(exp, "all", "skipped", None, 0.0, None, "Adapter and val/test metrics already exist.")
        return

    commands: list[tuple[str, list[str]]] = []
    if not adapter_path(exp.run_name).exists():
        commands.append(("train", train_command(exp, config, prompt_mode)))
    for split in ["val", "test"]:
        if not metrics_path(exp.run_name, split).exists():
            commands.append((f"eval_{split}", eval_command(exp, config, prompt_mode, split)))

    if dry_run:
        for stage, command in commands:
            print(exp.run_name, stage, " ".join(command))
        return

    for stage, command in commands:
        run_command(exp, stage, command)


def row_from_experiment(exp: Experiment) -> dict:
    row = asdict(exp)
    run_config_path = LORA_ROOT / exp.run_name / "run_config.json"
    if run_config_path.exists():
        run_config = load_json(run_config_path)
        row["trainable_params"] = run_config.get("trainable_params")
        row["trainable_percent"] = run_config.get("trainable_percent")
    for split in ["val", "test"]:
        path = metrics_path(exp.run_name, split)
        if not path.exists():
            continue
        metrics = load_json(path)
        row[f"{split}_accuracy"] = metrics.get("accuracy")
        row[f"{split}_macro_f1"] = metrics.get("macro_f1")
        row[f"{split}_weighted_f1"] = metrics.get("weighted_f1")
    row["complete"] = experiment_complete(exp)
    return row


def write_summary() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = [row_from_experiment(exp) for exp in EXPERIMENTS]
    df_path = OUTPUT_ROOT / "summary.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with df_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    complete_rows = [row for row in rows if row.get("complete")]
    best = sorted(
        complete_rows,
        key=lambda item: (
            item.get("val_macro_f1") if item.get("val_macro_f1") is not None else -1,
            item.get("test_macro_f1") if item.get("test_macro_f1") is not None else -1,
        ),
        reverse=True,
    )
    report_lines = [
        "# LoRA Hyperparameter Tuning Summary",
        "",
        "This report summarizes the controlled neutral-aware LoRA ablation grid.",
        "",
        "## Design",
        "",
        "- Dataset: Financial PhraseBank `sentences_50agree`, full raw training split.",
        "- Prompt: neutral-aware.",
        "- Base model: `Qwen/Qwen3-4B`.",
        "- Selection metric: validation macro-F1, with test results reported after each run.",
        "- Factors: rank, target modules, dropout, learning rate, and seed stability.",
        "",
        "## Best Completed Runs By Validation Macro-F1",
        "",
        "| Rank | Run | Group | Val Macro-F1 | Test Macro-F1 | Test Accuracy |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for rank, row in enumerate(best[:10], start=1):
        report_lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    str(row["run_name"]),
                    str(row["group"]),
                    _fmt(row.get("val_macro_f1")),
                    _fmt(row.get("test_macro_f1")),
                    _fmt(row.get("test_accuracy")),
                ]
            )
            + " |"
        )
    report_lines.extend(
        [
            "",
            f"Full CSV: `{df_path}`",
        ]
    )
    (OUTPUT_ROOT / "summary.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Wrote {df_path}")
    print(f"Wrote {OUTPUT_ROOT / 'summary.md'}")


def _fmt(value) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.4f}"


def main() -> None:
    args = parse_args()
    if args.summary_only:
        write_summary()
        return

    seen_start = args.start_at is None
    launched = 0
    for exp in EXPERIMENTS:
        if not seen_start:
            seen_start = exp.run_name == args.start_at
        if not seen_start:
            continue
        if args.max_runs is not None and launched >= args.max_runs:
            break
        run_experiment(exp, args.config, args.prompt_mode, args.dry_run)
        launched += 1
        write_summary()
    write_summary()


if __name__ == "__main__":
    main()
