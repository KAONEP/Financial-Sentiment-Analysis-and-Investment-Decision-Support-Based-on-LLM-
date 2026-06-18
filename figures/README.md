# Report Figures

This directory contains publication-style figures used by the project report.

Regenerate all figures with:

```bash
python figures/gen_fig_report_results.py
```

Each figure is saved as both PDF and PNG:

- PDF files are intended for LaTeX or paper drafts.
- PNG files are intended for GitHub preview and quick sharing.

## Figures

| File | Purpose |
|---|---|
| `fig_main_performance_macro_f1.*` | Main Financial PhraseBank Macro-F1 comparison. |
| `fig_lora_data_size_balance.*` | LoRA data-size and label-balance study. |
| `fig_lora_10seed_stability.*` | Final 10-seed LoRA stability check. |
| `fig_external_robustness.*` | NOSIBLE external formal-news robustness. |
| `fig_system_pipeline.*` | Streamlit prototype and inference pipeline. |
| `fig_progress_summary.*` | Compact three-panel figure for progress reporting. |
