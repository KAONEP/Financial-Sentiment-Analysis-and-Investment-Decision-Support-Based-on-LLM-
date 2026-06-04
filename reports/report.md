# Financial Sentiment Analysis and Investment Decision Support Based on LLM Fine-Tuning and Confidence-Based Fusion

## Abstract

Financial news sentiment analysis differs from general sentiment analysis because labels are often defined by likely investor impact rather than surface tone. This study examines whether parameter-efficient adaptation of an open-weight large language model improves financial sentiment classification, and whether fusion with a financial-domain encoder makes predictions more stable. Using Financial PhraseBank as the main dataset, we compare FinBERT, a strict supervised BERT baseline, zero-shot Qwen3-4B prompting, simple reasoning prompting, and Qwen3-4B with LoRA fine-tuning. We then evaluate LoRA under different training data sizes and label-balance conditions, and analyze rank/module ablations, calibration, statistical significance, higher-agreement robustness, and external robustness on Twitter Financial News Sentiment. The final system combines FinBERT, neutral-aware Qwen3-4B LoRA, learned probability fusion, URL extraction, long-article window aggregation, supporting excerpts, and LLM-generated investment-support insights. Results show that LoRA makes Qwen3-4B much better aligned with Financial PhraseBank than zero-shot prompting. The final learned stacking fusion improves the in-domain result while remaining interpretable as a validation-trained linear combiner over FinBERT and LoRA probability vectors. External evaluation shows that this fusion layer should be understood as an in-domain formal-news method rather than a universal cross-domain solution.

## 1. Introduction

Financial sentiment analysis supports investors, analysts, and decision-support systems. Unlike general sentiment analysis, the label of a financial sentence is not determined only by whether the wording sounds favorable or unfavorable. A product launch, acquisition, agreement, or management statement may still be neutral if it does not imply a clear financial impact. Financial sentiment classification therefore needs a task-specific interpretation from the investor's perspective.

Recent large language models can perform sentiment classification through prompting, but zero-shot predictions do not necessarily match the label definitions used in financial datasets. Domain-specific models such as FinBERT also remain competitive on financial sentiment tasks. This leads to a practical research question: can lightweight fine-tuning make an open-weight LLM competitive with financial-domain models, and can the two model families be combined for more stable predictions?

We study this question through both experiments and a working prototype. The research component evaluates LoRA fine-tuning, simple reasoning prompts, confidence-based fusion, calibration, robustness, and model understanding. The system component packages the selected pipeline into an interactive application that accepts financial news text or URLs and returns sentiment, confidence, supporting excerpts, and investment-support insights.

The main contributions are:

1. A systematic evaluation of Qwen3-4B LoRA fine-tuning on Financial PhraseBank under different data-size and label-balance conditions.
2. A learned logistic-stacking fusion method combining FinBERT and LoRA-adapted Qwen3-4B probabilities.
3. Robustness and reliability analyses including multi-seed checks, calibration, statistical testing, higher-agreement subsets, and external Twitter Financial News evaluation.
4. Model understanding analyses based on error taxonomy, probability shifts, counterfactual probes, and hidden-state separability.
5. An interactive financial sentiment and investment-support prototype with text and URL input, long-article processing, and evidence-based explanations.

## 2. Related Work

Financial PhraseBank is a standard benchmark for investor-perspective financial sentiment analysis. Malo et al. introduced the dataset to study semantic orientation in economic texts, with labels reflecting whether a sentence is positive, neutral, or negative from the perspective of an investor.

FinBERT adapts pre-trained language models to financial text. It is a useful reference model because it is trained for financial language. In this study, ProsusAI FinBERT is used as an off-the-shelf financial-domain reference baseline. Because this checkpoint was fine-tuned with Financial PhraseBank, however, it is not treated as a leakage-free supervised baseline for the current random split.

Large language models have recently been applied to financial NLP through systems such as BloombergGPT, FinGPT, PIXIU/FLARE, and instruction-tuned financial LLMs. This line of work shows the value of LLMs in finance, but also the need for task alignment and careful evaluation. Financial LLM benchmarks commonly report accuracy and F1-style metrics; macro-F1 is especially relevant when class distributions are imbalanced. Recent work on financial sentiment reasoning also suggests that reasoning or chain-of-thought prompting does not automatically improve classification and may introduce overthinking or unstable predictions.

LoRA is a parameter-efficient fine-tuning method that freezes the base model and trains low-rank adapter matrices. It fits this setting because Qwen3-4B is expensive to fully fine-tune on a 16 GB GPU, while LoRA can adapt the model with far fewer trainable parameters.

Confidence calibration is also relevant. Modern neural networks can be overconfident, so the maximum softmax probability should be treated as a confidence proxy rather than a guarantee of correctness. We use confidence for fusion and analyze calibration with ECE, Brier score, and temperature scaling.

## 3. Methodology

### 3.1 Dataset

The main dataset is Financial PhraseBank, using the commonly adopted `sentences_50agree` configuration. The task is three-class classification with negative, neutral, and positive labels. The dataset is divided into fixed stratified training, validation, and test splits with a 70/15/15 ratio.

The resulting split contains 3,392 training examples, 727 validation examples, and 727 test examples. In the full `sentences_50agree` configuration, the label distribution is neutral-heavy, with 604 negative, 2,879 neutral, and 1,363 positive examples. This motivates reporting macro-F1 in addition to accuracy and weighted-F1.

To study the effect of training data size and label balance, the project creates LoRA training subsets at 20%, 50%, and 100% of the training split. For each size, both the raw original label distribution and balanced undersampling are tested.

External robustness is evaluated on Twitter Financial News Sentiment. Its Bearish, Neutral, and Bullish labels are mapped to negative, neutral, and positive, respectively.

### 3.2 Baselines

We compare the following systems:

| Model | Purpose |
|---|---|
| ProsusAI FinBERT | Off-the-shelf financial-domain reference baseline |
| BERT-base | Strict supervised baseline trained only on the current split |
| Qwen3-4B direct prompt | Zero-shot open-weight LLM baseline |
| Qwen3-4B reasoning prompt | Tests whether simple investor-impact reasoning helps |
| Qwen3-4B LoRA | Main adapted LLM |

### 3.3 LLM Label Scoring

The LLM classifier is implemented as a label scorer rather than a free-form generator. Given an input text \(x\), the prompt asks the model to answer with one of the three labels. The next-token logits for the three candidate labels are normalized with softmax to obtain \(p_L(y \mid x)\). The predicted label is \(\hat{y}=\arg\max_y p_L(y \mid x)\), and confidence is defined as \(\max_y p_L(y \mid x)\).

This design gives reproducible probabilities for evaluation, fusion, and calibration.

### 3.4 LoRA Fine-Tuning

The base LLM is Qwen3-4B. LoRA freezes the base model weights and adds trainable low-rank update matrices, \(W' = W + \Delta W\), where \(\Delta W = (\alpha/r)BA\). The final system uses a neutral-aware rank-8 adapter with alpha 16, dropout 0.05, and adaptation applied to both attention and MLP projection modules. The model is trained for three epochs with a learning rate of \(1\times10^{-4}\), a maximum sequence length of 384, and a neutral-aware prompt.

The training objective is causal language modeling over the label token. The prompt part is masked, so the loss is applied only to the label target.

### 3.5 Neutral-Aware Prompt

Error analysis showed that many remaining mistakes were neutral-boundary errors. Therefore, a neutral-aware prompt was introduced. It explicitly states that positive and negative labels should be used only when there is a clear beneficial or harmful financial implication, while factual corporate events without clear impact should be neutral.

### 3.6 Learned Probability Fusion

The final system uses a validation-trained logistic stacking layer. Its input features are the three FinBERT class probabilities and the three LoRA-LLM class probabilities. The final probability vector is computed as \(p_{\mathrm{final}}=\mathrm{softmax}(Wf+b)\), where \(f\) is the concatenated probability vector from the two models.

The stacking layer is trained on the validation predictions and evaluated once on the held-out test split. It does not assume that low LoRA confidence automatically makes FinBERT better. Instead, it learns how the two probability vectors jointly map to the final label.

Calibration is evaluated separately from the deployed scoring rule. Temperature scaling is fitted on the validation split by minimizing negative log-likelihood for reliability analysis. The prototype currently uses raw model probabilities and the learned stacking output, so its confidence should be interpreted as a model score rather than a fully calibrated correctness probability.

### 3.7 Long-Article Inference

Financial PhraseBank is sentence-level, while real financial articles can be much longer. The system therefore uses window-based inference for long articles:

1. Extract readable article text.
2. Split the article into overlapping windows.
3. Apply FinBERT, LoRA, and fusion to each window.
4. Aggregate window probabilities into a document probability.
5. Display key supporting excerpts.

The document-level probability is computed as a weighted average of window-level probabilities, \(p_{\mathrm{doc}}=\sum_i w_i p_i / \sum_i w_i\). The window weight combines confidence and directional sentiment strength, so highly informative windows have greater influence on the document-level result.

This document-level processing is an engineering extension of a sentence-level classifier. The current quantitative evaluation is still sentence-level, because Financial PhraseBank does not provide document-level labels. The window aggregation and multi-scale sensitivity check are therefore presented as system safeguards, while quantitative document-level validation is left for future work.

## 4. Experimental Setup

Experiments are run on a CUDA-capable 16 GB consumer GPU. Qwen3-4B is loaded in bfloat16, and gradient checkpointing is used during LoRA training.

The evaluation follows common practice in financial sentiment classification by reporting both overall and class-sensitive metrics, including accuracy, macro-F1, weighted-F1, per-class precision and recall, and confusion matrices. Additional reliability and robustness analyses include ECE, Brier score, temperature scaling, multi-seed stability, paired bootstrap, McNemar testing, higher-agreement Financial PhraseBank subsets, external Twitter Financial News evaluation, error taxonomy, counterfactual probes, and hidden-state separability analysis.

Macro-F1 is emphasized because Financial PhraseBank is class-imbalanced and neutral-heavy. Accuracy and weighted-F1 are still reported for comparability with benchmark-style results. The validation split is used for selecting learned-stacking parameters and calibration parameters; the test split is reserved for final reporting.

## 5. Results

The results should be read with a clear domain distinction. Financial PhraseBank is the main benchmark and target style for the prototype: formal financial statements and news sentences labeled from an investor-impact perspective. Twitter Financial News Sentiment is used only as an external robustness check. PhraseBank therefore determines the main model-selection story, while Twitter tests how far that story transfers.

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

Calibration analysis is used to understand confidence reliability rather than to define a separate deployed model. The selected LoRA r8 model has ECE 0.0412 and Brier score 0.1673 on the test split. The final fusion layer is trained on validation-set probability vectors from FinBERT and neutral-aware LoRA, then evaluated once on the held-out test split.

### 5.5 Neutral-Aware Training And Final Fusion

The error taxonomy showed that many remaining errors were neutral-boundary mistakes. A neutral-aware prompt was therefore introduced before the final fusion experiment.

| Method | Test Accuracy | Test Macro-F1 | Test Weighted-F1 | Errors | Neutral false direction | Missed directional |
|---|---:|---:|---:|---:|---:|---:|
| LoRA r8 original | 0.8831 | 0.8813 | 0.8828 | 85 | 41 | 44 |
| LoRA r8 neutral-aware trained | 0.8858 | 0.8813 | 0.8848 | 83 | 33 | 50 |
| Learned logistic stacking fusion | 0.9161 | 0.9096 | 0.9162 | 61 | 32 | 29 |

The deployed prototype uses learned logistic stacking. It improves over both FinBERT and standalone LoRA on the in-domain test split and has a clear interpretation as a learned linear combiner over FinBERT and LoRA class probabilities.

### 5.6 Agreement Robustness

| Subset | FinBERT | LoRA r8 |
|---|---:|---:|
| 50agree test | 0.8650 | 0.8813 |
| 66agree test | 0.8936 | 0.9214 |
| 75agree test | 0.9217 | 0.9634 |
| allagree test | 0.9388 | 0.9911 |

Scores increase as annotation agreement becomes stricter, which is expected because high-agreement samples are less ambiguous. Standalone LoRA performs very well on the clearest examples, supporting the conclusion that LoRA improves the LLM's alignment with investor-perspective labels.

### 5.7 External Robustness

On Twitter Financial News Sentiment, LoRA transfers much better than FinBERT:

| Method | Accuracy | Macro-F1 |
|---|---:|---:|
| FinBERT | 0.7253 | 0.6682 |
| LoRA r8 | 0.8137 | 0.7895 |
| neutral-aware LoRA r8 | 0.8229 | 0.7975 |
| learned logistic stacking fusion | 0.8099 | 0.7605 |

However, learned stacking does not transfer better than standalone neutral-aware LoRA to this Twitter distribution. This does not invalidate the in-domain fusion result, because Twitter Financial News Sentiment contains short ticker-oriented posts and social-media phrasing rather than formal news sentences. It does mean that the deployed fusion method should not be described as a universal cross-domain solution.

### 5.8 Evaluation Coverage

The experimental design covers the evaluation axes commonly expected in financial sentiment and financial LLM studies:

| Evaluation Aspect | Implementation In This Project |
|---|---|
| Standard classification metrics | Accuracy, macro-F1, weighted-F1, per-class reports, confusion matrices |
| Financial-domain reference baseline | ProsusAI FinBERT |
| Leakage-aware supervised baseline | BERT-base trained only on the current split |
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

This coverage supports the project's positioning as a research-oriented system prototype. The main remaining gap is document-level quantitative evaluation: the system handles long articles through window aggregation, but the supervised evaluation is still sentence-level because Financial PhraseBank is sentence-level.

## 6. Model Understanding

The main remaining errors are neutral-boundary errors rather than simple positive-versus-negative polarity flips. In-domain fusion errors are dominated by neutral false direction and missed directional sentiment. This means the model usually understands basic polarity, but still struggles to decide whether a factual financial event should be considered investor-directional under the dataset definition.

Probability-shift analysis shows that LoRA reduces missed directional sentiment. Compared with zero-shot Qwen3-4B, LoRA increases the mean true-label probability and improves positive examples in particular. Neutral-aware training reduces neutral false direction errors, while fusion recovers some directional cases missed by the neutral-aware LoRA model.

The hidden-state analysis gives representation-level evidence. On a balanced subset of 90 PhraseBank test examples, a linear probe over the final hidden state improves from macro-F1 0.7739 for base Qwen3-4B to 0.8768 with the neutral-aware LoRA adapter. The silhouette score also improves from 0.2936 to 0.3596. These results suggest that LoRA changes the internal representation geometry so that financial sentiment labels become more linearly separable.

## 7. System Implementation

The final prototype is an interactive web application that accepts either pasted financial text or a news URL. For URL input, the system extracts readable article text and applies the same sentiment pipeline used for pasted text.

The system first cleans the input, then applies short-text or long-article inference depending on input length. FinBERT and the neutral-aware LoRA model each produce class probabilities, which are combined by the learned stacking layer. The final output includes the sentiment label, confidence score, supporting excerpts, and investment-support insight.

For long articles, the system uses overlapping windows and document-level aggregation. It also checks whether different window sizes produce different document labels. If they do, the result is marked as sensitive and should be interpreted with more caution.

The investment-support insight is generated by Qwen3-4B using the final sentiment, probabilities, decision trace, and evidence excerpts. The LoRA adapter is disabled during this generation step so that the base model's general language-generation ability is used. The prompt explicitly forbids buy, sell, hold, trading, and price-target instructions.

## 8. Discussion

The experiments support several conclusions. First, LoRA adaptation is needed because zero-shot Qwen3-4B is not sufficiently aligned with investor-perspective sentiment labels. Second, the raw training distribution is more stable than balanced undersampling because neutral examples are central to the Financial PhraseBank label scheme. Third, rank-8 attention+MLP LoRA offers a good efficiency-performance trade-off and works especially well when combined with fusion.

The deployed system uses learned logistic stacking because it improves the in-domain result while keeping the fusion rule simple: the final probabilities are produced by a validation-trained linear model over the two probability vectors.

The Twitter check is kept because it prevents overclaiming. It shows that any fusion layer selected on PhraseBank validation must be treated as in-domain rather than universally robust. Optimizing further on Twitter would require a different experimental question, such as multi-domain calibration or domain-adaptive fusion, and would broaden the project beyond the current sub-theme.

The model-understanding analyses show that the remaining difficulty is not basic sentiment polarity but neutral-boundary ambiguity. This is a meaningful limitation because many financial news items report factual events whose investor impact is debatable or delayed.

## 9. Limitations

This study has several limitations. Financial PhraseBank is small and mostly sentence-level, while real financial articles are longer and more complex. The off-the-shelf FinBERT reference baseline is not leakage-free with respect to Financial PhraseBank, although the project includes a strict supervised BERT baseline to address this concern. The learned stacking layer is selected on validation data, which reduces direct test-set overfitting but does not guarantee cross-domain portability. The confidence score is derived from model probabilities and should not be interpreted as a fully calibrated correctness probability; calibration results are therefore reported separately. The long-article system has not yet been quantitatively validated on a document-level labeled dataset. Finally, the hidden-state analysis provides diagnostic evidence, not full mechanistic interpretability.

## 10. Conclusion

This study shows that LoRA fine-tuning can adapt an open-weight LLM to investor-perspective financial sentiment analysis. Qwen3-4B LoRA improves clearly over zero-shot prompting and outperforms a strict supervised BERT baseline on the fixed Financial PhraseBank test split. Learned logistic stacking combines FinBERT and LoRA probabilities into the final deployed fusion method. Robustness and model-understanding analyses show that the main remaining challenges are neutral-boundary ambiguity and fusion portability under distribution shift. The final prototype applies the selected model pipeline to financial news text and URL analysis, returning sentiment, confidence, supporting excerpts, and investment-support insight.

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
