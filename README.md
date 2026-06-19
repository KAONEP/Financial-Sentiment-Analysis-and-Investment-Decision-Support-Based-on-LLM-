# Financial Sentiment Analysis and Investment Decision Support

This repository implements a research prototype for **financial sentiment analysis with a LoRA-tuned open-weight LLM**. Given financial news text or a news URL, the system returns a sentiment label, confidence score, supporting excerpts, model evidence, and a content-specific investment-support insight.

The project combines:

- `Qwen/Qwen3-4B` as the open-weight LLM backbone.
- A neutral-aware LoRA adapter trained on Financial PhraseBank.
- `ProsusAI/finbert` as a financial-domain reference model.
- Sentence-level evaluation on Financial PhraseBank.
- External formal-news robustness evaluation on `NOSIBLE/financial-sentiment`.
- LoRA model-selection checks covering rank, target modules, dropout, learning rate, and 10-seed stability.
- Confidence reliability analysis with ECE, Brier score, NLL, and temperature scaling.
- A Streamlit interface for text and URL-based financial news analysis.

## Repository Structure

```text
app/                 Streamlit application
src/financial_llm/   reusable model, prompt, metric, and system code
scripts/             data preparation, training, evaluation, and robustness scripts
configs/             experiment configuration
adapters/            final LoRA adapter used by the system
figures/             report figures and figure-generation script
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

The app supports pasted financial news text and URL input. For long articles, it uses overlapping text windows, calibrated-confidence-weighted aggregation, and optional multi-scale sensitivity checking. The investment-support insight is generated from the final sentiment result and supporting evidence excerpts, and is constrained to avoid direct buy, sell, hold, trading, or price-target recommendations.

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

Baseline comparison on Financial PhraseBank `sentences_50agree`, fixed test split:

| Method | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| Qwen3-4B direct prompt | 0.7950 | 0.7711 | 0.7858 |
| Qwen3-4B reasoning prompt | 0.8019 | 0.7961 | 0.7996 |
| strict BERT supervised baseline | 0.8418 | 0.8215 | 0.8406 |
| FinBERT reference | 0.8776 | 0.8650 | 0.8792 |

LoRA data-size and label-balance check:

| Condition | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| LoRA 20% balanced | 0.8074 | 0.8142 | 0.8088 |
| LoRA 20% raw | 0.8308 | 0.8231 | 0.8264 |
| LoRA 50% raw | 0.8624 | 0.8574 | 0.8616 |
| LoRA 50% balanced | 0.8501 | 0.8517 | 0.8507 |
| LoRA 75% raw | 0.8707 | 0.8642 | 0.8665 |
| LoRA 75% balanced | 0.8583 | 0.8522 | 0.8581 |
| LoRA 100% raw | 0.8803 | 0.8789 | 0.8813 |
| LoRA 100% balanced | 0.8624 | 0.8555 | 0.8635 |

The full raw training split is kept for later LoRA experiments because it performs best on the naturally neutral-heavy test distribution.

Single-seed LoRA screening on the same Financial PhraseBank split:

| Model setting | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| Qwen3-4B LoRA 100% raw r16 attention+MLP | 0.8803 | 0.8789 | 0.8813 |
| Neutral-aware LoRA r8 attention+MLP (seed42) | 0.8858 | 0.8813 | 0.8848 |
| Neutral-aware LoRA r8 MLP-only (seed42) | 0.8900 | 0.8844 | 0.8887 |

This screening step identifies attention+MLP and MLP-only as the two strongest LoRA settings. Since MLP-only is slightly higher on this one seed, the two settings are then compared across 10 random seeds before choosing the system model.

Final 10-seed LoRA comparison on Financial PhraseBank:

| Model variant | Seeds | Val Macro-F1 mean +/- std | Test Macro-F1 mean +/- std | Test Accuracy mean |
|---|---:|---:|---:|---:|
| LoRA r8 attention+MLP | 10 | 0.8667 +/- 0.0081 | 0.8727 +/- 0.0087 | 0.8770 |
| LoRA r8 MLP-only | 10 | 0.8658 +/- 0.0072 | 0.8702 +/- 0.0101 | 0.8761 |

External formal-news robustness on `NOSIBLE/financial-sentiment`, full 100,000-example evaluation set:

| Method | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| FinBERT reference | 0.7255 | 0.7289 | 0.7259 |
| Selected LoRA r8 attention+MLP (seed42) | 0.7830 | 0.7817 | 0.7827 |

Confidence reliability check on Financial PhraseBank test split:

| Model | Condition | NLL | Brier | ECE |
|---|---|---:|---:|---:|
| FinBERT | uncalibrated | 0.3400 | 0.1888 | 0.0189 |
| FinBERT | temperature-scaled | 0.3394 | 0.1891 | 0.0179 |
| LoRA r8 attention+MLP | uncalibrated | 0.2825 | 0.1673 | 0.0412 |
| LoRA r8 attention+MLP | temperature-scaled | 0.2696 | 0.1618 | 0.0151 |

The Streamlit prototype uses the temperature-scaled LoRA probabilities for displayed confidence, evidence weighting, and long-article aggregation. Temperature scaling is learned on the validation split and changes probability sharpness without changing the predicted label.

The main finding is that LoRA makes Qwen3-4B much better aligned with investor-perspective financial sentiment labels than direct prompting or reasoning prompting. Although MLP-only is slightly better in the seed42 PhraseBank comparison, attention+MLP is slightly stronger in the 10-seed comparison. The final system therefore uses the neutral-aware rank-8 attention+MLP LoRA model, and this selected model is then evaluated on the external formal-news dataset.

## Figures

The report figures are available in `figures/` as both PDF and PNG files.

![Main Financial PhraseBank performance](figures/fig_main_performance_macro_f1.png)

![LoRA data-size and label-balance study](figures/fig_lora_data_size_balance.png)

![10-seed LoRA stability check](figures/fig_lora_10seed_stability.png)

![External formal-news robustness](figures/fig_external_robustness.png)

![System pipeline](figures/fig_system_pipeline.png)

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
- Confidence is based on validation-calibrated LoRA probabilities and should still be interpreted as a model score rather than a guaranteed correctness probability.
- The system is a research decision-support prototype and does not provide financial advice.

## License And Data Notes

Financial PhraseBank has non-commercial licensing constraints. Check the dataset license before redistributing data or using the system commercially.

The NOSIBLE external dataset is loaded from its original Hugging Face dataset repository and is not redistributed in this repository.

The included LoRA adapter is provided as a research artifact for this project. The base models are downloaded from their original Hugging Face repositories and are subject to their own licenses.
