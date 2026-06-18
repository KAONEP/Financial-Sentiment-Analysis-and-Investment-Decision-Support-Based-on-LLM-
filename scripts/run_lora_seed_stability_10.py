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
OUTPUT_ROOT = ROOT / "outputs" / "analysis" / "lora_seed_stability_10"
LORA_ROOT = ROOT / "outputs" / "lora"
RUNS_ROOT = ROOT / "outputs" / "runs" / "Qwen__Qwen3-4B" / "lora"

ATTENTION_MODULES = "q_proj,k_proj,v_proj,o_proj"
MLP_MODULES = "gate_proj,up_proj,down_proj"
BOTH_MODULES = f"{ATTENTION_MODULES},{MLP_MODULES}"
SEEDS = [7, 13, 21, 42, 100, 123, 2024, 2026, 3407, 31415]


@dataclass(frozen=True)
class Candidate:
    name: str
    label: str
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    learning_rate: float
    target_modules: str
    rationale: str


@dataclass(frozen=True)
class Experiment:
    candidate_name: str
    candidate_label: str
    run_name: str
    seed: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    learning_rate: float
    target_modules: str
    rationale: str


CANDIDATES = [
    Candidate(
        name="r8_both_dropout005_lr1e4",
        label="final_attention_mlp",
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        learning_rate=1e-4,
        target_modules=BOTH_MODULES,
        rationale="Deployed final adapter family: rank-8 LoRA on attention and MLP projections.",
    ),
    Candidate(
        name="r8_mlp_dropout005_lr1e4",
        label="mlp_only_competitor",
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        learning_rate=1e-4,
        target_modules=MLP_MODULES,
        rationale="Parameter-efficient target-module competitor for the final attention+MLP adapter.",
    ),
]


RUN_NAME_ALIASES = {
    ("r8_both_dropout005_lr1e4", 42): "neutral_aware_lora_r8_full_raw_seed42",
    ("r8_both_dropout005_lr1e4", 123): "neutral_aware_lora_r8_both_dropout005_lr1e4_seed123",
    ("r8_both_dropout005_lr1e4", 2026): "neutral_aware_lora_r8_both_dropout005_lr1e4_seed2026",
    ("r8_mlp_dropout005_lr1e4", 42): "neutral_aware_lora_r8_mlp_dropout005_lr1e4_seed42",
    ("r8_mlp_dropout005_lr1e4", 123): "neutral_aware_lora_r8_mlp_dropout005_lr1e4_seed123",
    ("r8_mlp_dropout005_lr1e4", 2026): "neutral_aware_lora_r8_mlp_dropout005_lr1e4_seed2026",
}


def default_run_name(candidate: Candidate, seed: int) -> str:
    return f"neutral_aware_lora_{candidate.name}_seed{seed}"


def experiments() -> list[Experiment]:
    rows: list[Experiment] = []
    for candidate in CANDIDATES:
        for seed in SEEDS:
            run_name = RUN_NAME_ALIASES.get((candidate.name, seed), default_run_name(candidate, seed))
            rows.append(
                Experiment(
                    candidate_name=candidate.name,
                    candidate_label=candidate.label,
                    run_name=run_name,
                    seed=seed,
                    lora_r=candidate.lora_r,
                    lora_alpha=candidate.lora_alpha,
                    lora_dropout=candidate.lora_dropout,
                    learning_rate=candidate.learning_rate,
                    target_modules=candidate.target_modules,
                    rationale=candidate.rationale,
                )
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 10-seed stability check for the two final LoRA candidates.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--prompt-mode", default="neutral_aware", choices=["neutral_aware"])
    parser.add_argument("--start-at", default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args()


def adapter_path(run_name: str) -> Path:
    return LORA_ROOT / run_name / "final_adapter"


def metrics_path(run_name: str, split: str) -> Path:
    return RUNS_ROOT / run_name / "neutral_aware" / split / "metrics.json"


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
                "candidate_name",
                "candidate_label",
                "run_name",
                "seed",
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


def log_status(
    exp: Experiment,
    stage: str,
    status: str,
    returncode: int | None,
    elapsed: float,
    log_path: Path | None,
    message: str,
) -> None:
    append_status(
        {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "candidate_name": exp.candidate_name,
            "candidate_label": exp.candidate_label,
            "run_name": exp.run_name,
            "seed": exp.seed,
            "stage": stage,
            "status": status,
            "returncode": "" if returncode is None else returncode,
            "elapsed_sec": f"{elapsed:.1f}",
            "log_path": "" if log_path is None else str(log_path),
            "message": message,
        }
    )


def run_command_once(exp: Experiment, stage: str, command: list[str], attempt: int, max_attempts: int) -> tuple[int, float, Path]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    suffix = "" if max_attempts == 1 else f"_attempt{attempt}"
    log_path = OUTPUT_ROOT / f"{exp.run_name}_{stage}{suffix}.log"
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write("$ " + " ".join(command) + "\n\n")
        if max_attempts > 1:
            log_handle.write(f"Attempt {attempt}/{max_attempts}\n\n")
        log_handle.flush()
        process = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    elapsed = time.time() - started
    return process.returncode, elapsed, log_path


def run_command(exp: Experiment, stage: str, command: list[str]) -> None:
    max_attempts = 3 if stage.startswith("eval_") else 1
    last_log_path: Path | None = None
    for attempt in range(1, max_attempts + 1):
        returncode, elapsed, log_path = run_command_once(exp, stage, command, attempt, max_attempts)
        last_log_path = log_path
        if returncode == 0:
            log_status(exp, stage, "completed", returncode, elapsed, log_path, "Command completed.")
            return
        status = "retrying" if attempt < max_attempts else "failed"
        message = "Command failed; retrying after a short cooldown." if attempt < max_attempts else "Command failed."
        log_status(exp, stage, status, returncode, elapsed, log_path, message)
        if attempt < max_attempts:
            time.sleep(30)
    raise RuntimeError(f"{exp.run_name} {stage} failed; see {last_log_path}")


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


def summarize(rows: list[dict]) -> list[dict]:
    summaries = []
    for candidate_name in sorted({row["candidate_name"] for row in rows}):
        group = [row for row in rows if row["candidate_name"] == candidate_name]
        complete = [row for row in group if row.get("complete")]
        item = {
            "candidate_name": candidate_name,
            "candidate_label": group[0]["candidate_label"],
            "runs_completed": len(complete),
            "runs_expected": len(group),
            "complete": len(complete) == len(group),
            "rationale": group[0]["rationale"],
        }
        for metric in ["val_macro_f1", "test_macro_f1", "test_accuracy"]:
            values = [float(row[metric]) for row in complete if row.get(metric) not in {None, ""}]
            if values:
                mean = sum(values) / len(values)
                variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
                item[f"{metric}_mean"] = mean
                item[f"{metric}_std"] = variance**0.5
                item[f"{metric}_min"] = min(values)
                item[f"{metric}_max"] = max(values)
        summaries.append(item)
    return sorted(summaries, key=lambda item: item.get("test_macro_f1_mean", -1), reverse=True)


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.4f}"


def write_summary() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = [row_from_experiment(exp) for exp in experiments()]
    summary_rows = summarize(rows)
    write_csv(OUTPUT_ROOT / "runs.csv", rows)
    write_csv(OUTPUT_ROOT / "candidates.csv", summary_rows)
    rel_output_root = OUTPUT_ROOT.relative_to(ROOT)

    lines = [
        "# LoRA 10-Seed Stability Check",
        "",
        "This protocol compares only the deployed final LoRA family and the MLP-only competitor.",
        "Existing completed seed runs are reused. Missing seeds are trained and evaluated on validation and test splits.",
        "",
        "## Candidate Summary",
        "",
        "| Candidate | Label | Complete | Val Macro-F1 Mean | Val Macro-F1 Std | Test Macro-F1 Mean | Test Macro-F1 Std | Test Accuracy Mean |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["candidate_name"]),
                    str(row["candidate_label"]),
                    f"{row['runs_completed']}/{row['runs_expected']}",
                    fmt(row.get("val_macro_f1_mean")),
                    fmt(row.get("val_macro_f1_std")),
                    fmt(row.get("test_macro_f1_mean")),
                    fmt(row.get("test_macro_f1_std")),
                    fmt(row.get("test_accuracy_mean")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- Per-run CSV: `{rel_output_root / 'runs.csv'}`",
            f"- Candidate CSV: `{rel_output_root / 'candidates.csv'}`",
            f"- Status CSV: `{rel_output_root / 'status.csv'}`",
        ]
    )
    (OUTPUT_ROOT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_ROOT / 'runs.csv'}")
    print(f"Wrote {OUTPUT_ROOT / 'candidates.csv'}")
    print(f"Wrote {OUTPUT_ROOT / 'summary.md'}")


def main() -> None:
    args = parse_args()
    if args.summary_only:
        write_summary()
        return

    seen_start = args.start_at is None
    launched = 0
    for exp in experiments():
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
