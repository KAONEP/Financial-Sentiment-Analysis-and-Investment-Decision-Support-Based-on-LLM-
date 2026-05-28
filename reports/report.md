# Financial Sentiment Analysis and Investment Decision Support Based on LLM Fine-Tuning and Confidence-Based Fusion

## Abstract

Financial news sentiment analysis differs from general sentiment analysis because labels are often defined by likely investor impact rather than surface tone. This study examines whether parameter-efficient adaptation of an open-weight large language model improves financial sentiment classification, and whether fusion with a financial-domain encoder makes predictions more stable. Using Financial PhraseBank as the main dataset, we compare FinBERT, a strict supervised BERT baseline, zero-shot Qwen3-4B prompting, simple reasoning prompting, and Qwen3-4B with LoRA fine-tuning. We then evaluate LoRA under different training data sizes and label-balance conditions, and analyze rank/module ablations, calibration, statistical significance, higher-agreement robustness, and external robustness on Twitter Financial News Sentiment. The final system combines FinBERT, neutral-aware Qwen3-4B LoRA, confidence-based fusion, URL extraction, long-article window aggregation, supporting excerpts, and LLM-generated investment-support insights. Results show that LoRA makes Qwen3-4B much better aligned with Financial PhraseBank than zero-shot prompting. Confidence-based fusion with a neutral-margin correction gives the best in-domain result, while external evaluation shows that fixed threshold fusion is less portable under distribution shift. The system therefore separates formal-news and general modes.

## 1. Introduction

Financial sentiment analysis supports investors, analysts, and decision-support systems. Unlike general sentiment analysis, the label of a financial sentence is not determined only by whether the wording sounds favorable or unfavorable. A product launch, acquisition, agreement, or management statement may still be neutral if it does not imply a clear financial impact. Financial sentiment classification therefore needs a task-specific interpretation from the investor's perspective.

Recent large language models can perform sentiment classification through prompting, but zero-shot predictions do not necessarily match the label definitions used in financial datasets. Domain-specific models such as FinBERT also remain competitive on financial sentiment tasks. This leads to a practical research question: can lightweight fine-tuning make an open-weight LLM competitive with financial-domain models, and can the two model families be combined for more stable predictions?

We study this question through both experiments and a working prototype. The research component evaluates LoRA fine-tuning, simple reasoning prompts, confidence-based fusion, calibration, robustness, and model understanding. The system component packages the selected pipeline into an interactive application that accepts financial news text or URLs and returns sentiment, confidence, supporting excerpts, and investment-support insights.

The main contributions are:

1. A systematic evaluation of Qwen3-4B LoRA fine-tuning on Financial PhraseBank under different data-size and label-balance conditions.
2. A confidence-based fusion method combining FinBERT and LoRA-adapted Qwen3-4B.
3. Robustness and reliability analyses including multi-seed checks, calibration, statistical testing, higher-agreement subsets, and external Twitter Financial News evaluation.
4. Model understanding analyses based on error taxonomy, probability shifts, counterfactual probes, and hidden-state separability.
5. A Streamlit-based financial sentiment and investment-support system with text/URL input and long-article processing.

## 2. Related Work

Financial PhraseBank is a standard benchmark for investor-perspective financial sentiment analysis. Malo et al. introduced the dataset to study semantic orientation in economic texts, with labels reflecting whether a sentence is positive, neutral, or negative from the perspective of an investor.

FinBERT adapts pre-trained language models to financial text. It is a useful reference model because it is trained for financial language. In this study, `ProsusAI/finbert` is used as an off-the-shelf financial-domain reference baseline. Because this checkpoint was fine-tuned with Financial PhraseBank, however, it is not treated as a leakage-free supervised baseline for the current random split.

Large language models have recently been applied to financial NLP through systems such as BloombergGPT, FinGPT, PIXIU/FLARE, and instruction-tuned financial LLMs. This line of work shows the value of LLMs in finance, but also the need for task alignment and careful evaluation. Financial LLM benchmarks commonly report accuracy and F1-style metrics; macro-F1 is especially relevant when class distributions are imbalanced. Recent work on financial sentiment reasoning also suggests that reasoning or chain-of-thought prompting does not automatically improve classification and may introduce overthinking or unstable predictions.

LoRA is a parameter-efficient fine-tuning method that freezes the base model and trains low-rank adapter matrices. It fits this setting because Qwen3-4B is expensive to fully fine-tune on a 16 GB GPU, while LoRA can adapt the model with far fewer trainable parameters.

Confidence calibration is also relevant. Modern neural networks can be overconfident, so the maximum softmax probability should be treated as a confidence proxy rather than a guarantee of correctness. We use confidence for fusion and analyze calibration with ECE, Brier score, and temperature scaling.

## 3. Methodology

### 3.1 Dataset

The main dataset is Financial PhraseBank, configuration `sentences_50agree`, obtained through `takala/financial_phrasebank`. The label set contains three classes:

```text
negative, neutral, positive
```

The dataset is split into fixed stratified train, validation, and test sets:

```text
train: 70%
validation: 15%
test: 15%
```

The resulting split contains 3,392 training examples, 727 validation examples, and 727 test examples. In the full `sentences_50agree` configuration, the label distribution is neutral-heavy, with 604 negative, 2,879 neutral, and 1,363 positive examples. This motivates reporting macro-F1 in addition to accuracy and weighted-F1.

To study the effect of training data size and label balance, the project creates LoRA training subsets at 20%, 50%, and 100% of the training split. For each size, both the raw original label distribution and balanced undersampling are tested.

External robustness is evaluated on Twitter Financial News Sentiment. Labels are mapped as:

```text
Bearish -> negative
Neutral -> neutral
Bullish -> positive
```

### 3.2 Baselines

We compare the following systems:

| Model | Purpose |
|---|---|
| `ProsusAI/finbert` | Off-the-shelf financial-domain reference baseline |
| `bert-base-uncased` | Strict supervised baseline trained only on the current split |
| Qwen3-4B direct prompt | Zero-shot open-weight LLM baseline |
| Qwen3-4B reasoning prompt | Tests whether simple investor-impact reasoning helps |
| Qwen3-4B LoRA | Main adapted LLM |

### 3.3 LLM Label Scoring

The LLM classifier is implemented as a label scorer rather than a free-form generator. Given an input text `x`, the prompt asks the model to answer with one of the three labels. The system extracts the next-token logits for `negative`, `neutral`, and `positive`, and then applies softmax:

```text
p_L(y | x) = exp(z_y) / sum_{k in Y} exp(z_k)
```

The predicted label and confidence are:

```text
y_hat = argmax_y p_L(y | x)
confidence = max_y p_L(y | x)
```

This design gives reproducible probabilities for evaluation, fusion, and calibration.

### 3.4 LoRA Fine-Tuning

The base LLM is `Qwen/Qwen3-4B`. LoRA freezes the base model weights and adds trainable low-rank update matrices:

```text
W' = W + Delta W
Delta W = (alpha / r) B A
```

The final system uses the neutral-aware rank-8 LoRA adapter:

```text
rank r: 8
alpha: 16
dropout: 0.05
target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
learning rate: 1e-4
epochs: 3
max sequence length: 384
prompt mode: neutral_aware
```

The training objective is causal language modeling over the label token. The prompt part is masked, so the loss is applied only to the label target.

### 3.5 Neutral-Aware Prompt

Error analysis showed that many remaining mistakes were neutral-boundary errors. Therefore, a neutral-aware prompt was introduced. It explicitly states that positive and negative labels should be used only when there is a clear beneficial or harmful financial implication, while factual corporate events without clear impact should be neutral.

### 3.6 Confidence-Based Fusion

We use two fusion modes.

Formal financial news mode uses threshold fusion:

```text
if confidence_LoRA >= 0.91:
    p_fused = p_LoRA
else:
    p_fused = p_FinBERT
```

It then applies a neutral-margin rule:

```text
if y_base in {negative, positive}
and p_fused(y_base) - p_fused(neutral) < 0.48:
    y_final = neutral
```

General mode uses weighted fusion:

```text
alpha = confidence_LoRA
p_fused = alpha p_LoRA + (1 - alpha) p_FinBERT
```

It then applies a neutral-margin rule with margin 0.49. Formal-news mode prioritizes in-domain Financial PhraseBank performance, while general mode is intended to be more conservative under distribution shift. The deployed system uses the neutral-aware rank-8 LoRA adapter with formal-news threshold 0.91 and neutral margin 0.48.

The numeric thresholds are treated as validation-selected hyperparameters rather than fixed assumptions. The LoRA confidence threshold is selected by grid search on validation predictions over 0.50 to 0.95 with step size 0.01, using macro-F1 as the selection metric. The formal-news neutral margin is selected by a second validation-set grid search over 0.00 to 0.50 with step size 0.01 after fixing the fusion base method. The test split is used only once for final reporting. The general-mode margin of 0.49 comes from a cross-domain diagnostic that combines PhraseBank validation with a held-out Twitter calibration split and selects the setting that maximizes worst-case macro-F1. This separation is used to reduce test-set overfitting risk.

Calibration is evaluated separately from the deployed scoring rule. Temperature scaling is fitted on the validation split by minimizing negative log-likelihood, and calibrated fusion results are reported in the calibration experiment. The Streamlit prototype currently uses the raw label-scoring probabilities for operational fusion, so its confidence should be interpreted as a gating score rather than a fully calibrated correctness probability.

### 3.7 Long-Article Inference

Financial PhraseBank is sentence-level, while real URL articles can be much longer. The system therefore uses window-based inference for long articles:

1. Extract article text with `trafilatura`.
2. Split text into overlapping windows.
3. Apply FinBERT, LoRA, and fusion to each window.
4. Aggregate window probabilities into a document probability.
5. Display key supporting excerpts.

The document-level probability is:

```text
p_doc = sum_i w_i p_i / sum_i w_i
```

where the window weight is:

```text
w_i = max(confidence_i, max(p_i(negative), p_i(positive)))
```

The app provides Fast, Balanced, and Robust research speed settings. Balanced mode uses 120-word and 220-word windows by default.

This document-level processing is an engineering extension of a sentence-level classifier. The current quantitative evaluation is still sentence-level, because Financial PhraseBank does not provide document-level labels. The window aggregation and multi-scale sensitivity check are therefore presented as system safeguards, while quantitative document-level validation is left for future work.

## 4. Experimental Setup

Experiments are run on a CUDA-capable 16 GB consumer GPU. Qwen3-4B is loaded in bfloat16, and gradient checkpointing is used during LoRA training.

The evaluation follows common practice in financial sentiment classification by reporting both overall and class-sensitive metrics. The main metrics are:

```text
accuracy
macro-F1
weighted-F1
per-class precision, recall, F1
confusion matrix
```

Additional reliability and robustness analyses include:

```text
ECE
Brier score
temperature scaling
multi-seed stability
paired bootstrap and McNemar testing
higher-agreement Financial PhraseBank subsets
external Twitter Financial News evaluation
error taxonomy
counterfactual probes
hidden-state separability analysis
```

Macro-F1 is emphasized because Financial PhraseBank is class-imbalanced and neutral-heavy. Accuracy and weighted-F1 are still reported for comparability with benchmark-style results. The validation split is used for selecting fusion thresholds, neutral margins, and calibration parameters; the test split is reserved for final reporting.

## 5. Results

### 5.1 Baseline Results

| Method | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| Qwen3-4B direct | 0.7950 | 0.7711 | 0.7858 |
| Qwen3-4B reasoning | 0.8019 | 0.7961 | 0.7996 |
| Strict BERT supervised | 0.8418 | 0.8215 | 0.8406 |
| FinBERT reference | 0.8776 | 0.8650 | 0.8792 |

The reasoning prompt improves Qwen3-4B over direct prompting, but zero-shot prompting remains below FinBERT and the strict supervised BERT baseline. This motivates LoRA fine-tuning.

### 5.2 LoRA Data Size And Label Balance

| Condition | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| LoRA 20% balanced | 0.8074 | 0.8142 | 0.8088 |
| LoRA 20% raw | 0.8308 | 0.8231 | 0.8264 |
| LoRA 50% raw | 0.8624 | 0.8574 | 0.8616 |
| LoRA 50% balanced | 0.8501 | 0.8517 | 0.8507 |
| LoRA 100% raw | 0.8803 | 0.8789 | 0.8813 |
| LoRA 100% balanced | 0.8624 | 0.8555 | 0.8635 |

The raw training distribution improves consistently from 20% to 50% to 100%. Balanced undersampling improves some minority-class behavior but reduces overall stability on the naturally neutral-heavy test set.

These data-size and label-balance runs use the original r16 attention+MLP LoRA configuration. The later ablation study compares this setting with lighter variants and motivates the final r8 adapter used by the system.

### 5.3 LoRA Ablation

| Run | Adapter params | Test Accuracy | Test Macro-F1 |
|---|---:|---:|---:|
| r16 attention+MLP | 33,030,144 | 0.8803 | 0.8789 |
| r8 attention+MLP | 16,515,072 | 0.8831 | 0.8813 |
| r16 attention-only | 11,796,480 | 0.8721 | 0.8713 |

Rank-8 attention+MLP is selected because it gives a good efficiency-performance trade-off, especially in fusion. Attention-only adaptation is lighter but underperforms attention+MLP.

### 5.4 Fusion And Calibration

| Method | Test Accuracy | Test Macro-F1 | Test Weighted-F1 | ECE | Brier |
|---|---:|---:|---:|---:|---:|
| LoRA r8 | 0.8831 | 0.8813 | 0.8828 | 0.0412 | 0.1673 |
| r8 calibrated threshold fusion | 0.9051 | 0.9002 | 0.9056 | 0.0251 | 0.1545 |
| r8 calibrated weighted fusion | 0.8886 | 0.8811 | 0.8888 | 0.0233 | 0.1488 |

Calibrated threshold fusion gives the best single-seed classification result. Weighted calibrated fusion has a better Brier score but lower macro-F1. Paired testing shows that moving from FinBERT to calibrated r8 threshold fusion improves macro-F1 by +0.0352, with bootstrap 95% CI [0.0163, 0.0550] and McNemar p=0.0012.

### 5.5 Neutral-Boundary Improvement And Final In-Domain Setting

The error taxonomy showed that many remaining errors were neutral-boundary mistakes. A neutral-aware prompt and a neutral-margin fusion rule were therefore tested.

| Method | Test Accuracy | Test Macro-F1 | Test Weighted-F1 | Errors | Neutral false direction | Missed directional |
|---|---:|---:|---:|---:|---:|---:|
| LoRA r8 original | 0.8831 | 0.8813 | 0.8828 | 85 | 41 | 44 |
| LoRA r8 neutral-aware trained | 0.8858 | 0.8813 | 0.8848 | 83 | 33 | 50 |
| Previous calibrated r8 threshold fusion | 0.9051 | 0.9002 | 0.9056 | 69 | 47 | 21 |
| Neutral-aware threshold + neutral margin | 0.9161 | 0.9101 | 0.9161 | 61 | 33 | 27 |

This final in-domain setting is used in the formal-news mode of the Streamlit system. The improvement comes from combining neutral-aware LoRA training with a margin rule that prevents weak directional predictions from overriding neutral when the probability gap is small.

### 5.6 Agreement Robustness

| Subset | FinBERT | LoRA r8 | r8 calibrated threshold fusion |
|---|---:|---:|---:|
| 50agree test | 0.8650 | 0.8813 | 0.9002 |
| 66agree test | 0.8936 | 0.9214 | 0.9284 |
| 75agree test | 0.9217 | 0.9634 | 0.9620 |
| allagree test | 0.9388 | 0.9911 | 0.9872 |

Scores increase as annotation agreement becomes stricter, which is expected because high-agreement samples are less ambiguous. Fusion remains competitive, while standalone LoRA performs very well on the clearest examples.

### 5.7 External Robustness

On Twitter Financial News Sentiment, LoRA transfers much better than FinBERT:

| Method | Accuracy | Macro-F1 |
|---|---:|---:|
| FinBERT | 0.7253 | 0.6682 |
| LoRA r8 | 0.8137 | 0.7895 |
| r8 weighted fusion | 0.8153 | 0.7899 |
| neutral-aware LoRA r8 | 0.8229 | 0.7975 |

However, fixed threshold fusion does not transfer reliably. A separate cross-domain diagnostic splits Twitter validation into a calibration half and a held-out evaluation half. Under that setting, a cross-domain selected weighted fusion reaches accuracy 0.8409 and macro-F1 0.8065 on the held-out half, while sacrificing some PhraseBank performance. This supports using separate formal-news and general modes.

### 5.8 Evaluation Coverage

The experimental design covers the evaluation axes commonly expected in financial sentiment and financial LLM studies:

| Evaluation Aspect | Implementation In This Project |
|---|---|
| Standard classification metrics | Accuracy, macro-F1, weighted-F1, per-class reports, confusion matrices |
| Financial-domain reference baseline | `ProsusAI/finbert` |
| Leakage-aware supervised baseline | `bert-base-uncased` trained only on the current split |
| Open-weight LLM baseline | Qwen3-4B direct and reasoning prompts |
| Lightweight LLM adaptation | Qwen3-4B LoRA fine-tuning |
| Data-size analysis | 20%, 50%, and 100% training subsets |
| Label-balance analysis | Raw distribution versus balanced undersampling |
| LoRA ablation | Rank and target-module comparisons |
| Confidence reliability | ECE, Brier score, and temperature scaling |
| Statistical testing | McNemar test and paired bootstrap confidence intervals |
| In-domain robustness | Higher-agreement Financial PhraseBank subsets |
| External robustness | Twitter Financial News Sentiment |
| Model understanding | Error taxonomy, probability shifts, counterfactual probes, and hidden-state separability |

This coverage supports the project's positioning as a research-oriented system prototype. The main remaining gap is document-level quantitative evaluation: the deployed URL pipeline handles long articles through window aggregation, but the supervised evaluation is still sentence-level because Financial PhraseBank is sentence-level.

## 6. Model Understanding

The main remaining errors are neutral-boundary errors rather than simple positive-versus-negative polarity flips. In-domain fusion errors are dominated by neutral false direction and missed directional sentiment. This means the model usually understands basic polarity, but still struggles to decide whether a factual financial event should be considered investor-directional under the dataset definition.

Probability-shift analysis shows that LoRA reduces missed directional sentiment. Compared with zero-shot Qwen3-4B, LoRA increases the mean true-label probability and improves positive examples in particular. Neutral-aware training reduces neutral false direction errors, while fusion recovers some directional cases missed by the neutral-aware LoRA model.

The hidden-state analysis gives representation-level evidence. On a balanced subset of 90 PhraseBank test examples, a linear probe over the final hidden state improves from macro-F1 0.7739 for base Qwen3-4B to 0.8768 with the neutral-aware LoRA adapter. The silhouette score also improves from 0.2936 to 0.3596. These results suggest that LoRA changes the internal representation geometry so that financial sentiment labels become more linearly separable.

## 7. System Implementation

The final prototype is implemented with Streamlit. It accepts either pasted financial text or a URL. For URL input, the system fetches the page with a `requests.Session` that ignores broken local proxy environment variables, then extracts readable article text with `trafilatura`.

The system pipeline is:

```text
input text or URL
-> text extraction and cleaning
-> short-text or long-article inference
-> FinBERT probabilities
-> Qwen3-4B neutral-aware LoRA probabilities
-> confidence-based fusion
-> final sentiment and confidence
-> supporting excerpts
-> LLM-generated investment-support insight
```

For long articles, the app uses overlapping windows and document-level aggregation. It also checks whether different window sizes produce different document labels. If they do, the result is marked as sensitive and should be interpreted with more caution.

The investment-support insight is generated by Qwen3-4B using the final sentiment, probabilities, decision trace, and evidence excerpts. The LoRA adapter is disabled during this generation step so that the base model's general language-generation ability is used. The prompt explicitly forbids buy, sell, hold, trading, and price-target instructions.

## 8. Discussion

The experiments support several conclusions. First, LoRA adaptation is needed because zero-shot Qwen3-4B is not sufficiently aligned with investor-perspective sentiment labels. Second, the raw training distribution is more stable than balanced undersampling because neutral examples are central to the Financial PhraseBank label scheme. Third, rank-8 attention+MLP LoRA offers a good efficiency-performance trade-off and works especially well when combined with fusion.

The best in-domain result comes from neutral-aware confidence-based fusion with a neutral-margin correction. This suggests that FinBERT and LoRA provide complementary information, and that explicit handling of the neutral boundary matters for Financial PhraseBank. External robustness results also show that fixed threshold and margin rules are not universally portable. The system therefore separates formal-news mode from general mode.

The model-understanding analyses show that the remaining difficulty is not basic sentiment polarity but neutral-boundary ambiguity. This is a meaningful limitation because many financial news items report factual events whose investor impact is debatable or delayed.

## 9. Limitations

This study has several limitations. Financial PhraseBank is small and mostly sentence-level, while real financial articles are longer and more complex. The off-the-shelf FinBERT reference baseline is not leakage-free with respect to Financial PhraseBank, although the project includes a strict supervised BERT baseline to address this concern. The fusion threshold and neutral margin are selected on validation data, which reduces direct test-set overfitting but does not guarantee cross-domain portability. The confidence score is maximum softmax probability, which is useful as a gating signal but is not equivalent to true correctness probability; calibration results are therefore reported separately. The long-article system has not yet been quantitatively validated on a document-level labeled dataset. Finally, the hidden-state analysis provides diagnostic evidence, not full mechanistic interpretability.

## 10. Conclusion

This study shows that LoRA fine-tuning can adapt an open-weight LLM to investor-perspective financial sentiment analysis. Qwen3-4B LoRA improves clearly over zero-shot prompting and outperforms a strict supervised BERT baseline on the fixed Financial PhraseBank test split. Confidence-based fusion with FinBERT and neutral-margin correction further improves in-domain stability and gives the best Financial PhraseBank result in this study. Robustness and model-understanding analyses show that the main remaining challenges are neutral-boundary ambiguity and threshold portability under distribution shift. The final Streamlit system applies the selected model pipeline to financial news text and URL analysis, returning sentiment, confidence, supporting excerpts, and investment-support insight.

## References

Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained Language Models.

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.

Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized LLMs.

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. ICML.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models.

Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014). Good Debt or Bad Debt: Detecting Semantic Orientations in Economic Texts. Journal of the Association for Information Science and Technology.

Qwen Team. (2025). Qwen3 Technical Report.

Vamvourellis, D., & Mehta, D. (2025). Reasoning or Overthinking: Evaluating Large Language Models on Financial Sentiment Analysis.

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., & Zhou, D. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. NeurIPS.

Wu, S., Irsoy, O., Lu, S., Dabravolski, V., Dredze, M., Gehrmann, S., Kambadur, P., Rosenberg, D., & Mann, G. (2023). BloombergGPT: A Large Language Model for Finance.

Xie, Q., Han, W., Zhang, X., Lai, Y., Peng, M., Lopez-Lira, A., & Huang, J. (2023). PIXIU: A Large Language Model, Instruction Data and Evaluation Benchmark for Finance.

Yang, H., Liu, X.-Y., & Wang, C. D. (2023). FinGPT: Open-Source Financial Large Language Models.

Zhang, B., Yang, H., & Liu, X.-Y. (2023). Instruct-FinGPT: Financial Sentiment Analysis by Instruction Tuning of General-Purpose Large Language Models.
