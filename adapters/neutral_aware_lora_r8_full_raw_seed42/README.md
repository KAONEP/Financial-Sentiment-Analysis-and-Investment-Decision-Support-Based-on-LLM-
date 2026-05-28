---
base_model: Qwen/Qwen3-4B
library_name: peft
pipeline_tag: text-classification
tags:
  - financial-sentiment-analysis
  - lora
  - peft
  - qwen3
  - financial-phrasebank
---

# Neutral-Aware Qwen3-4B LoRA Adapter

This folder contains the LoRA adapter used by the Streamlit system in this repository.

## Model

| Item | Value |
|---|---|
| Base model | `Qwen/Qwen3-4B` |
| Adapter type | LoRA |
| Task | Financial sentiment classification |
| Labels | `negative`, `neutral`, `positive` |
| Training data | Financial PhraseBank `sentences_50agree`, full raw training split |
| Prompt mode | `neutral_aware` |

## LoRA Configuration

| Parameter | Value |
|---|---:|
| Rank `r` | 8 |
| Alpha | 16 |
| Dropout | 0.05 |

Target modules:

```text
q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

## Usage

The adapter is loaded by the app from:

```text
adapters/neutral_aware_lora_r8_full_raw_seed42
```

The base model is not included in this folder. It is downloaded from Hugging Face when the model is loaded.

## Limitations

- This adapter is trained for sentence-level Financial PhraseBank-style sentiment classification.
- Long-article support in the app uses window-level inference and aggregation.
- Outputs should be treated as investment decision support, not financial advice.
- The adapter depends on the license and usage terms of the base model and dataset.
