# Financial Sentiment Analysis and Investment Decision Support

This repository implements a research prototype for **financial sentiment analysis with LLM fine-tuning and confidence-based fusion**. Given financial news text or a news URL, the system returns a sentiment label, a confidence score, supporting excerpts, a model decision trace, and a content-specific investment-support insight.

The project combines:

- `ProsusAI/finbert` as a financial-domain reference model.
- `Qwen/Qwen3-4B` as the open-weight LLM backbone.
- A neutral-aware LoRA adapter trained on Financial PhraseBank.
- Confidence-based fusion between FinBERT and the LoRA-adapted LLM.
- A Streamlit interface for text and URL-based financial news analysis.

## Repository Structure

```text
app/                 Streamlit application
src/financial_llm/   reusable model, prompt, fusion, metric, and system code
scripts/             data preparation, training, evaluation, calibration, and robustness scripts
configs/             experiment configuration
adapters/            final LoRA adapter used by the system
reports/report.md    consolidated research report
```

## Model Configuration

The repository includes the final LoRA adapter used by the application:

```text
adapters/neutral_aware_lora_r8_full_raw_seed42
```

The full Qwen3-4B and FinBERT base models are loaded from Hugging Face when the application or evaluation scripts are run.

| Component | Setting |
|---|---|
| FinBERT | `ProsusAI/finbert` |
| LLM backbone | `Qwen/Qwen3-4B` |
| LoRA adapter | `adapters/neutral_aware_lora_r8_full_raw_seed42` |
| Prompt mode | `neutral_aware` |
| LoRA rank | `8` |
| LoRA alpha | `16` |
| LoRA dropout | `0.05` |
| Target modules | attention and MLP projections |
| Training data | Financial PhraseBank `sentences_50agree`, full raw training split |
| Formal-news fusion | LoRA confidence threshold `0.91`, neutral margin `0.48` |
| General fusion | weighted fusion, neutral margin `0.49` |

## Installation

Python 3.10 or 3.11 is recommended. A CUDA-capable GPU is recommended for running Qwen3-4B locally.

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
python -m streamlit run app/streamlit_app.py
```

The app supports pasted financial news text and URL input. For long articles, it uses overlapping text windows, confidence-weighted aggregation, and optional multi-scale sensitivity checking. The investment-support insight is generated from the final sentiment result and supporting evidence excerpts, and is constrained to avoid direct buy, sell, hold, trading, or price-target recommendations.

## Reproduce Experiments

Prepare Financial PhraseBank splits:

```bash
python scripts/prepare_data.py --config configs/experiment.yaml
```

Run core baselines:

```bash
python scripts/run_finbert_baseline.py --config configs/experiment.yaml --split test
python scripts/run_llm_zero_shot.py --config configs/experiment.yaml --split test
python scripts/train_supervised_baseline.py --config configs/experiment.yaml
```

Train the final LoRA-style configuration:

```bash
python scripts/train_lora.py ^
  --config configs/experiment.yaml ^
  --train-file data/processed/sentences_50agree/train_frac100_raw.csv ^
  --prompt-mode neutral_aware ^
  --lora-r 8 ^
  --lora-alpha 16 ^
  --run-name neutral_aware_lora_r8_full_raw_seed42
```

On Linux or macOS, replace `^` with `\`.

Additional scripts support fusion evaluation, calibration, statistical testing, agreement robustness, and external robustness:

```text
scripts/evaluate_fusion.py
scripts/calibrate_confidence.py
scripts/statistical_tests.py
scripts/evaluate_agreement_robustness.py
scripts/prepare_external_datasets.py
scripts/evaluate_external_robustness.py
```

## Results

Financial PhraseBank `sentences_50agree`, fixed test split:

| Method | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| Qwen3-4B direct prompt | 0.7950 | 0.7711 | 0.7858 |
| Qwen3-4B reasoning prompt | 0.8019 | 0.7961 | 0.7996 |
| strict BERT supervised baseline | 0.8418 | 0.8215 | 0.8406 |
| FinBERT reference | 0.8776 | 0.8650 | 0.8792 |
| Qwen3-4B LoRA 100% raw (r16) | 0.8803 | 0.8789 | 0.8813 |
| neutral-aware Qwen3-4B LoRA r8 | 0.8858 | 0.8813 | 0.8848 |
| neutral-aware threshold + neutral-margin fusion | 0.9161 | 0.9101 | 0.9161 |

External Twitter Financial News Sentiment:

| Method | Accuracy | Macro-F1 |
|---|---:|---:|
| FinBERT | 0.7253 | 0.6682 |
| LoRA r8 | 0.8137 | 0.7895 |
| neutral-aware LoRA r8 | 0.8229 | 0.7975 |

In a separate cross-domain diagnostic, Twitter validation is split into a calibration half and a held-out evaluation half. On the held-out half, the cross-domain selected weighted fusion reaches accuracy 0.8409 and macro-F1 0.8065.

The main finding is that LoRA makes Qwen3-4B much better aligned with Financial PhraseBank than zero-shot prompting. Confidence-based fusion with explicit neutral-boundary handling gives the best in-domain result, while external evaluation shows that fixed threshold fusion is less portable under distribution shift. For this reason, the system separates formal-news and general-use modes.

## Research Report

The consolidated research report is available at:

```text
reports/report.md
```

It describes the dataset, baselines, LoRA training method, fusion strategy, calibration analysis, statistical significance tests, robustness checks, model-understanding analysis, system implementation, limitations, and references.

## Limitations

- Financial PhraseBank is small and sentence-level, while real financial articles are longer and more complex.
- `ProsusAI/finbert` is used as an off-the-shelf reference model and is not treated as a leakage-free supervised baseline for Financial PhraseBank.
- The selected fusion threshold and neutral margin are validation-selected and may not transfer directly across domains.
- Confidence is based on maximum softmax probability and should be interpreted as a model gating signal rather than a guaranteed correctness probability.
- Long-article support is implemented through window aggregation; document-level quantitative validation remains future work.
- The system is a research decision-support prototype and does not provide financial advice.

## License And Data Notes

Financial PhraseBank has non-commercial licensing constraints. Check the dataset license before redistributing data or using the system commercially.

The included LoRA adapter is provided as a research artifact for this project. The base models are downloaded from their original Hugging Face repositories and are subject to their own licenses.
