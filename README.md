# Financial Sentiment Analysis and Investment Decision Support

This repository implements a research prototype for **financial sentiment analysis with LLM fine-tuning and confidence-based fusion**. Given financial news text or a news URL, the system returns a sentiment label, a confidence score, supporting excerpts, a model decision trace, and a content-specific investment-support insight.

The project combines:

- `ProsusAI/finbert` as a financial-domain reference model.
- `Qwen/Qwen3-4B` as the open-weight LLM backbone.
- A neutral-aware LoRA adapter trained on Financial PhraseBank.
- Learned probability fusion between FinBERT and the LoRA-adapted LLM.
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
| Deployed fusion | validation-trained logistic stacking over FinBERT and LoRA class probabilities |

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

Train and evaluate the final learned stacking fusion layer after FinBERT and LoRA predictions are available:

```bash
python scripts/train_stacking_fusion.py
```

Additional scripts support calibration, statistical testing, agreement robustness, and external robustness:

```text
scripts/train_stacking_fusion.py
scripts/calibrate_confidence.py
scripts/statistical_tests.py
scripts/evaluate_agreement_robustness.py
scripts/prepare_nosible_external.py
scripts/evaluate_stacking_fusion.py
scripts/evaluate_formal_news_external.py
```

## Results

The main target domain is formal financial news in the style of Financial PhraseBank. `NOSIBLE/financial-sentiment` is used as the main external formal-news robustness check. Twitter Financial News Sentiment is treated as an archived exploratory diagnostic because its short social-media style is outside the primary deployment setting.

Financial PhraseBank `sentences_50agree`, fixed test split:

| Method | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| Qwen3-4B direct prompt | 0.7950 | 0.7711 | 0.7858 |
| Qwen3-4B reasoning prompt | 0.8019 | 0.7961 | 0.7996 |
| strict BERT supervised baseline | 0.8418 | 0.8215 | 0.8406 |
| FinBERT reference | 0.8776 | 0.8650 | 0.8792 |
| Qwen3-4B LoRA 100% raw (r16) | 0.8803 | 0.8789 | 0.8813 |
| neutral-aware Qwen3-4B LoRA r8 | 0.8858 | 0.8813 | 0.8848 |
| learned logistic stacking fusion | 0.9161 | 0.9096 | 0.9162 |

External formal-news robustness on `NOSIBLE/financial-sentiment`, full 100,000-example evaluation set:

| Method | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| FinBERT | 0.7255 | 0.7289 | 0.7259 |
| neutral-aware LoRA r8 | 0.7830 | 0.7817 | 0.7827 |
| learned logistic stacking fusion | 0.7435 | 0.7405 | 0.7436 |

The main finding is that LoRA makes Qwen3-4B much better aligned with Financial PhraseBank than zero-shot prompting, and this improvement transfers to a large formal-news external dataset. The deployed system uses learned logistic stacking because it improves the in-domain Financial PhraseBank result and is easy to explain as a validation-trained linear combiner over the two models' probability vectors.

The NOSIBLE result also gives an important limitation: the fixed stacking layer selected on Financial PhraseBank validation improves over FinBERT externally, but it does not transfer as well as standalone neutral-aware LoRA. The correct claim is therefore not that fusion is universally best, but that LoRA provides the strongest external transfer while learned stacking is the best in-domain system choice.

## Research Report

The consolidated research report is available at:

```text
reports/report.md
```

It describes the dataset, baselines, LoRA training method, fusion strategy, calibration analysis, statistical significance tests, robustness checks, model-understanding analysis, system implementation, limitations, and references.

## Limitations

- Financial PhraseBank is small and sentence-level, while real financial articles are longer and more complex.
- `ProsusAI/finbert` is used as an off-the-shelf reference model and is not treated as a leakage-free supervised baseline for Financial PhraseBank.
- The deployed learned fusion layer is selected on Financial PhraseBank validation data and does not transfer better than standalone LoRA on the full NOSIBLE formal-news external set.
- Confidence is based on maximum softmax probability and should be interpreted as a model gating signal rather than a guaranteed correctness probability.
- Long-article support is implemented through window aggregation; document-level quantitative validation remains future work.
- The system is a research decision-support prototype and does not provide financial advice.

## License And Data Notes

Financial PhraseBank has non-commercial licensing constraints. Check the dataset license before redistributing data or using the system commercially.

The NOSIBLE external dataset is loaded from its original Hugging Face dataset repository and is not redistributed in this repository.

The included LoRA adapter is provided as a research artifact for this project. The base models are downloaded from their original Hugging Face repositories and are subject to their own licenses.
