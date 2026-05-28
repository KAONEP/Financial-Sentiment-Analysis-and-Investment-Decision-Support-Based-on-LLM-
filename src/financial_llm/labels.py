LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}


def normalize_label(label: str) -> str:
    value = str(label).strip().lower()
    if value.startswith("label_"):
        return value
    if value in LABEL2ID:
        return value
    raise ValueError(f"Unknown label: {label!r}")

