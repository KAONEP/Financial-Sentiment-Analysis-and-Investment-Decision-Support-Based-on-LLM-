from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split

from .config import ensure_dir
from .labels import LABELS


PHRASEBANK_FILES = {
    "sentences_50agree": "FinancialPhraseBank-v1.0/Sentences_50Agree.txt",
    "sentences_66agree": "FinancialPhraseBank-v1.0/Sentences_66Agree.txt",
    "sentences_75agree": "FinancialPhraseBank-v1.0/Sentences_75Agree.txt",
    "sentences_allagree": "FinancialPhraseBank-v1.0/Sentences_AllAgree.txt",
}


def load_phrasebank(dataset_name: str, dataset_config: str, raw_dir: str | Path) -> pd.DataFrame:
    if dataset_config not in PHRASEBANK_FILES:
        raise ValueError(f"Unsupported dataset_config: {dataset_config!r}")

    zip_path = hf_hub_download(
        repo_id=dataset_name,
        filename="data/FinancialPhraseBank-v1.0.zip",
        repo_type="dataset",
        local_dir=str(ensure_dir(raw_dir)),
    )
    target_file = PHRASEBANK_FILES[dataset_config]
    rows = []
    with zipfile.ZipFile(zip_path) as archive:
        content = archive.read(target_file).decode("latin-1")
    for line in content.splitlines():
        if not line.strip():
            continue
        sentence, label = line.rsplit("@", 1)
        rows.append({"sentence": sentence.strip(), "label": label.strip().lower()})

    df = pd.DataFrame(rows)
    df = df[df["label"].isin(LABELS)].reset_index(drop=True)
    df.insert(0, "example_id", [f"fpb_{i:05d}" for i in range(len(df))])
    return df


def make_splits(
    df: pd.DataFrame,
    train_size: float,
    val_size: float,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split sizes must sum to 1.0, got {total}")

    train_df, temp_df = train_test_split(
        df,
        train_size=train_size,
        random_state=seed,
        stratify=df["label"],
    )
    val_ratio_in_temp = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_ratio_in_temp,
        random_state=seed,
        stratify=temp_df["label"],
    )
    return (
        train_df.sort_values("example_id").reset_index(drop=True),
        val_df.sort_values("example_id").reset_index(drop=True),
        test_df.sort_values("example_id").reset_index(drop=True),
    )


def sample_raw(train_df: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    if fraction == 1.0:
        return train_df.copy().reset_index(drop=True)
    parts = []
    for _, group in train_df.groupby("label"):
        n = max(1, int(round(len(group) * fraction)))
        parts.append(group.sample(n=n, random_state=seed))
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def sample_balanced(train_df: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    min_count = train_df["label"].value_counts().min()
    per_class = max(1, int(round(min_count * fraction)))
    parts = []
    for label in LABELS:
        group = train_df[train_df["label"] == label]
        parts.append(group.sample(n=per_class, random_state=seed))
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def write_split(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8")


def prepare_dataset(config: dict) -> dict:
    seed = int(config["project"]["seed"])
    data_cfg = config["data"]
    dataset_config = data_cfg["dataset_config"]
    out_dir = ensure_dir(Path(data_cfg["output_dir"]) / dataset_config)

    df = load_phrasebank(
        data_cfg["dataset_name"],
        dataset_config,
        raw_dir=data_cfg.get("raw_dir", "data/raw/financial_phrasebank"),
    )
    train_df, val_df, test_df = make_splits(
        df,
        train_size=float(data_cfg["train_size"]),
        val_size=float(data_cfg["val_size"]),
        test_size=float(data_cfg["test_size"]),
        seed=seed,
    )

    write_split(train_df, out_dir / "train.csv")
    write_split(val_df, out_dir / "val.csv")
    write_split(test_df, out_dir / "test.csv")

    generated = {
        "dataset_name": data_cfg["dataset_name"],
        "dataset_config": dataset_config,
        "seed": seed,
        "splits": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "label_counts": {
            "full": df["label"].value_counts().to_dict(),
            "train": train_df["label"].value_counts().to_dict(),
            "val": val_df["label"].value_counts().to_dict(),
            "test": test_df["label"].value_counts().to_dict(),
        },
        "train_variants": {},
    }

    for fraction in data_cfg["fractions"]:
        fraction = float(fraction)
        fraction_key = int(round(fraction * 100))
        raw = sample_raw(train_df, fraction=fraction, seed=seed)
        balanced = sample_balanced(train_df, fraction=fraction, seed=seed)

        raw_name = f"train_frac{fraction_key}_raw.csv"
        balanced_name = f"train_frac{fraction_key}_balanced.csv"
        write_split(raw, out_dir / raw_name)
        write_split(balanced, out_dir / balanced_name)

        generated["train_variants"][raw_name] = raw["label"].value_counts().to_dict()
        generated["train_variants"][balanced_name] = balanced["label"].value_counts().to_dict()

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(generated, indent=2), encoding="utf-8")
    return generated
