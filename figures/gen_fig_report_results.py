from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FIG_DIR = Path(__file__).resolve().parent


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linestyle": "-",
    }
)


COLORS = {
    "deep_teal": "#264653",
    "teal": "#2A9D8F",
    "gold": "#E9C46A",
    "orange": "#F4A261",
    "coral": "#E76F51",
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "gray": "#B0BEC5",
    "dark_gray": "#4B5563",
}


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG_DIR / f"{name}.pdf")
    fig.savefig(FIG_DIR / f"{name}.png")
    plt.close(fig)


def add_bar_labels(ax, bars, fmt: str = "{:.4f}", dx: float = 0.003) -> None:
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + dx,
            bar.get_y() + bar.get_height() / 2,
            fmt.format(width),
            va="center",
            ha="left",
            fontsize=8,
            color=COLORS["dark_gray"],
        )


def fig_main_performance() -> None:
    methods = [
        "Qwen direct",
        "Qwen reasoning",
        "Strict BERT",
        "FinBERT ref.",
        "LoRA seed42",
    ]
    macro_f1 = np.array([0.7711, 0.7961, 0.8215, 0.8650, 0.8813])
    colors = [COLORS["gray"], COLORS["gray"], COLORS["sky"], COLORS["blue"], COLORS["coral"]]

    fig, ax = plt.subplots(figsize=(6.75, 2.75))
    y = np.arange(len(methods))
    bars = ax.barh(y, macro_f1, color=colors, edgecolor="white", linewidth=0.7, height=0.58)
    ax.set_yticks(y)
    ax.set_yticklabels(methods)
    ax.invert_yaxis()
    ax.set_xlim(0.74, 0.90)
    ax.set_xlabel("Macro-F1 on Financial PhraseBank")
    ax.set_title("Main Financial PhraseBank Test Performance")
    add_bar_labels(ax, bars)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    save(fig, "fig_main_performance_macro_f1")


def fig_data_size_balance() -> None:
    fractions = np.array([20, 50, 100])
    raw = np.array([0.8231, 0.8574, 0.8789])
    balanced = np.array([0.8142, 0.8517, 0.8555])

    fig, ax = plt.subplots(figsize=(5.4, 3.1))
    ax.plot(
        fractions,
        raw,
        marker="o",
        color=COLORS["coral"],
        linewidth=2.0,
        label="Raw label distribution",
    )
    ax.plot(
        fractions,
        balanced,
        marker="s",
        color=COLORS["blue"],
        linewidth=2.0,
        label="Balanced undersampling",
    )
    ax.set_xticks(fractions)
    ax.set_ylim(0.80, 0.89)
    ax.set_xlabel("Training data used (%)")
    ax.set_ylabel("Macro-F1")
    ax.set_title("LoRA Data Size and Label Balance")
    ax.legend(loc="lower right")
    for x, y in zip(fractions, raw):
        ax.text(x, y + 0.003, f"{y:.4f}", ha="center", fontsize=8, color=COLORS["coral"])
    for x, y in zip(fractions, balanced):
        ax.text(x, y - 0.006, f"{y:.4f}", ha="center", fontsize=8, color=COLORS["blue"])
    save(fig, "fig_lora_data_size_balance")


def fig_seed_stability() -> None:
    labels = ["attention+MLP", "MLP-only"]
    means = np.array([0.8727, 0.8702])
    stds = np.array([0.0087, 0.0101])

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    x = np.arange(len(labels))
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=5,
        color=[COLORS["coral"], COLORS["blue"]],
        edgecolor="white",
        linewidth=0.7,
        width=0.55,
    )
    ax.axhline(0.8650, color=COLORS["dark_gray"], linestyle="--", linewidth=1.2, label="FinBERT Macro-F1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.84, 0.89)
    ax.set_ylabel("Test Macro-F1 mean +/- std")
    ax.set_title("10-Seed LoRA Stability Check")
    ax.legend(loc="lower right")
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + 0.011,
            f"{mean:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=COLORS["dark_gray"],
        )
    save(fig, "fig_lora_10seed_stability")


def fig_external_robustness() -> None:
    methods = ["FinBERT ref.", "LoRA seed42", "MLP-only"]
    accuracy = np.array([0.7255, 0.7830, 0.7804])
    macro_f1 = np.array([0.7289, 0.7817, 0.7769])

    fig, ax = plt.subplots(figsize=(5.8, 3.1))
    x = np.arange(len(methods))
    width = 0.34
    ax.bar(x - width / 2, accuracy, width, label="Accuracy", color=COLORS["sky"], edgecolor="white")
    ax.bar(x + width / 2, macro_f1, width, label="Macro-F1", color=COLORS["coral"], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0.70, 0.80)
    ax.set_ylabel("Score on NOSIBLE full set")
    ax.set_title("External Formal-News Robustness")
    ax.legend(loc="upper left")
    for xpos, value in zip(x - width / 2, accuracy):
        ax.text(xpos, value + 0.004, f"{value:.4f}", ha="center", fontsize=8, rotation=90)
    for xpos, value in zip(x + width / 2, macro_f1):
        ax.text(xpos, value + 0.004, f"{value:.4f}", ha="center", fontsize=8, rotation=90)
    save(fig, "fig_external_robustness")


def draw_box(ax, xy, width, height, text, color) -> None:
    box = plt.Rectangle(
        xy,
        width,
        height,
        facecolor=color,
        edgecolor="white",
        linewidth=1.2,
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=9,
        color="#1F2937",
        zorder=3,
        wrap=True,
    )


def draw_arrow(ax, start, end) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", lw=1.4, color=COLORS["dark_gray"]),
        zorder=4,
    )


def fig_system_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 2.2))
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.2)

    labels = [
        "Text or URL\ninput",
        "Article\nextraction",
        "Window\nchunking",
        "LoRA label\nscoring",
        "Probability\naggregation",
        "Sentiment,\nevidence,\ninsight",
    ]
    colors = ["#E8EDF2", "#E8EDF2", "#F5F0E8", "#E8F2EE", "#E8F2EE", "#FCE7DD"]
    xs = [0.2, 1.8, 3.4, 5.0, 6.6, 8.2]
    for x, label, color in zip(xs, labels, colors):
        draw_box(ax, (x, 0.65), 1.25, 0.9, label, color)
    for x in xs[:-1]:
        draw_arrow(ax, (x + 1.25, 1.1), (x + 1.55, 1.1))

    ax.text(
        5.65,
        1.93,
        "Deployed classifier: neutral-aware Qwen3-4B LoRA r8 (seed42)",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=COLORS["deep_teal"],
    )
    ax.text(
        5.65,
        0.25,
        "FinBERT is retained as a reference baseline; it does not change the final deployed label.",
        ha="center",
        va="center",
        fontsize=8.5,
        color=COLORS["dark_gray"],
    )
    save(fig, "fig_system_pipeline")


def fig_progress_summary() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.0))

    # Panel A: main Macro-F1 comparison.
    ax = axes[0]
    methods = ["Qwen\nDirect", "Qwen\nReason.", "Strict\nBERT", "FinBERT", "LoRA\nseed42"]
    macro_f1 = np.array([0.7711, 0.7961, 0.8215, 0.8650, 0.8813])
    colors = [COLORS["gray"], COLORS["gray"], COLORS["sky"], COLORS["blue"], COLORS["coral"]]
    x = np.arange(len(methods))
    ax.bar(x, macro_f1, color=colors, edgecolor="white", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=8)
    ax.set_ylim(0.74, 0.90)
    ax.set_ylabel("Macro-F1")
    ax.set_title("(a) Main Test")
    for xpos, value in zip(x, macro_f1):
        ax.text(xpos, value + 0.004, f"{value:.3f}", ha="center", fontsize=7, rotation=90)

    # Panel B: data size and label balance.
    ax = axes[1]
    fractions = np.array([20, 50, 100])
    raw = np.array([0.8231, 0.8574, 0.8789])
    balanced = np.array([0.8142, 0.8517, 0.8555])
    ax.plot(fractions, raw, marker="o", color=COLORS["coral"], linewidth=2.0, label="Raw")
    ax.plot(fractions, balanced, marker="s", color=COLORS["blue"], linewidth=2.0, label="Balanced")
    ax.set_xticks(fractions)
    ax.set_ylim(0.80, 0.89)
    ax.set_xlabel("Training data (%)")
    ax.set_title("(b) Data Conditions")
    ax.legend(loc="lower right", fontsize=7)

    # Panel C: 10-seed stability.
    ax = axes[2]
    labels = ["attention\n+MLP", "MLP-only"]
    means = np.array([0.8727, 0.8702])
    stds = np.array([0.0087, 0.0101])
    x = np.arange(len(labels))
    ax.bar(
        x,
        means,
        yerr=stds,
        capsize=4,
        color=[COLORS["coral"], COLORS["blue"]],
        edgecolor="white",
        linewidth=0.7,
        width=0.56,
    )
    ax.axhline(0.8650, color=COLORS["dark_gray"], linestyle="--", linewidth=1.1)
    ax.text(1.0, 0.866, "FinBERT", fontsize=7, ha="center", va="bottom", color=COLORS["dark_gray"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0.84, 0.89)
    ax.set_title("(c) 10-Seed Stability")

    fig.suptitle("Summary of Main LoRA Experiments", fontsize=12, fontweight="bold", y=1.04)
    fig.tight_layout()
    save(fig, "fig_progress_summary")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_main_performance()
    fig_data_size_balance()
    fig_seed_stability()
    fig_external_robustness()
    fig_system_pipeline()
    fig_progress_summary()
    print(f"Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
