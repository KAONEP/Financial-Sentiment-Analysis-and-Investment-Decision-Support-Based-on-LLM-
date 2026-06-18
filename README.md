# Financial Sentiment Analysis and Investment Decision Support

This repository implements a research prototype for **financial sentiment analysis with a LoRA-tuned open-weight LLM**. Given financial news text or a news URL, the system returns a sentiment label, confidence score, supporting excerpts, model evidence, and a content-specific investment-support insight.

The project combines:

- `Qwen/Qwen3-4B` as the open-weight LLM backbone.
- A neutral-aware LoRA adapter trained on Financial PhraseBank.
- `ProsusAI/finbert` as a financial-domain reference model.
- Sentence-level evaluation on Financial PhraseBank.
- External formal-news robustness evaluation on `NOSIBLE/financial-sentiment`.
- LoRA model-selection checks covering rank, target modules, dropout, learning rate, and 10-seed stability.
- A Streamlit interface for text and URL-based financial news analysis.

## Repository Structure

```text
app/                 Streamlit application
src/financial_llm/   reusable model, prompt, metric, and system code
scripts/             data preparation, training, evaluation, and robustness scripts
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
| Final classifier | neutral-aware Qwen3-4B LoRA |
| LLM backbone | `Qwen/Qwen3-4B` |
| LoRA adapter | `adapters/neutral_aware_lora_r8_full_raw_seed42` |
| Prompt mode | `neutral_aware` |
| LoRA rank | `8` |
| LoRA alpha | `16` |
| LoRA dropout | `0.05` |
| Target modules | attention and MLP projections |
| Training data | Financial PhraseBank `sentences_50agree`, full raw training split |
| Reference model | `ProsusAI/finbert` |

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

Prepare and evaluate the full NOSIBLE external formal-news set:

```bash
python scripts/prepare_nosible_external.py
python scripts/run_finbert_baseline.py --config configs/experiment.yaml --data-file data/external/nosible_financial_sentiment_full/test.csv --split test --output-dir outputs/runs/external/nosible_financial_sentiment_full/finbert/test
python scripts/evaluate_lora.py --config configs/experiment.yaml --adapter-path adapters/neutral_aware_lora_r8_full_raw_seed42 --data-file data/external/nosible_financial_sentiment_full/test.csv --split test --prompt-mode neutral_aware --output-dir outputs/runs/external/nosible_financial_sentiment_full/neutral_aware_lora_r8/test
python scripts/evaluate_formal_news_external.py
```

Additional scripts support agreement robustness, paired statistical testing, and baseline evaluation:

```text
scripts/calibrate_confidence.py
scripts/evaluate_agreement_robustness.py
scripts/statistical_tests.py
scripts/evaluate_sequence_classifier.py
scripts/run_lora_ablation_grid.py
scripts/run_lora_seed_stability_10.py
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
| deployed Qwen3-4B LoRA r8 (seed42) | 0.8858 | 0.8813 | 0.8848 |

The LoRA row reports the concrete seed42 checkpoint included in this repository and used by the Streamlit prototype. The separate 10-seed stability check below reports configuration-level robustness for the same rank-8 attention+MLP LoRA design.

External formal-news robustness on `NOSIBLE/financial-sentiment`, full 100,000-example evaluation set:

| Method | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| FinBERT reference | 0.7255 | 0.7289 | 0.7259 |
| deployed Qwen3-4B LoRA r8 (seed42) | 0.7830 | 0.7817 | 0.7827 |
| MLP-only neutral-aware Qwen3-4B LoRA r8 | 0.7804 | 0.7769 | 0.7798 |

Final 10-seed stability check on Financial PhraseBank:

| Candidate family | Seeds | Val Macro-F1 mean +/- std | Test Macro-F1 mean +/- std | Test Accuracy mean |
|---|---:|---:|---:|---:|
| r8 attention+MLP, dropout 0.05, lr 1e-4 | 10 | 0.8667 +/- 0.0081 | 0.8727 +/- 0.0087 | 0.8770 |
| r8 MLP-only, dropout 0.05, lr 1e-4 | 10 | 0.8658 +/- 0.0072 | 0.8702 +/- 0.0101 | 0.8761 |

The main finding is that LoRA makes Qwen3-4B much better aligned with investor-perspective financial sentiment labels than direct prompting or reasoning prompting, and that this improvement transfers to a large formal-news external dataset. The final deployed adapter remains the neutral-aware rank-8 attention+MLP LoRA model. In the final 10-seed check, it is slightly stronger and slightly more stable than the MLP-only competitor on Financial PhraseBank, and it also performs better on the external formal-news evaluation.

## Research Report

The consolidated research report is available at:

```text
reports/report.md
```

It describes the dataset, baselines, LoRA training method, data-size and label-balance experiments, LoRA ablations, multi-seed final model selection, calibration analysis, statistical testing, higher-agreement robustness, NOSIBLE external robustness, model-understanding analysis, system implementation, limitations, and references.

## Limitations

- Financial PhraseBank is small and sentence-level, while real financial articles are longer and more complex.
- `ProsusAI/finbert` is used as an off-the-shelf reference model and is not treated as a leakage-free supervised baseline for Financial PhraseBank.
- The LoRA model is trained with a maximum sequence length of 384, so long-article support relies on window aggregation.
- Confidence is based on maximum softmax probability and should be interpreted as a model score rather than a guaranteed correctness probability.
- The system is a research decision-support prototype and does not provide financial advice.

## License And Data Notes

Financial PhraseBank has non-commercial licensing constraints. Check the dataset license before redistributing data or using the system commercially.

The NOSIBLE external dataset is loaded from its original Hugging Face dataset repository and is not redistributed in this repository.

The included LoRA adapter is provided as a research artifact for this project. The base models are downloaded from their original Hugging Face repositories and are subject to their own licenses.
