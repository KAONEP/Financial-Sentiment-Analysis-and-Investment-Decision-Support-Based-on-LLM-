from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np

from .fusion import threshold_fusion, weighted_fusion
from .labels import LABELS


FORMAL_NEWS_THRESHOLD = 0.91
FORMAL_NEWS_MARGIN = 0.48
GENERAL_MODE_MARGIN = 0.49
DEFAULT_CHUNK_WORDS = 120
DEFAULT_CHUNK_OVERLAP = 25
DEFAULT_CHUNK_WORD_SCALES = (80, 120, 160, 220)
DEFAULT_LONG_TEXT_WORD_THRESHOLD = 160
SUPPORT_SYSTEM_PROMPT = (
    "You are a cautious financial analysis assistant. Write decision-support insight "
    "based only on the provided article evidence and model outputs. Do not give direct "
    "buy, sell, hold, or price-target instructions."
)


@dataclass(frozen=True)
class SentimentModelOutput:
    model_name: str
    probabilities: np.ndarray
    prediction: str
    confidence: float


@dataclass(frozen=True)
class SentimentDecision:
    mode: str
    label: str
    confidence: float
    probabilities: np.ndarray
    base_label: str
    base_confidence: float
    neutral_margin_applied: bool
    neutral_margin_gap: float | None
    trace: str


@dataclass(frozen=True)
class SentimentSystemResult:
    finbert: SentimentModelOutput
    llm: SentimentModelOutput
    decision: SentimentDecision
    explanation: str
    investment_support: str


@dataclass(frozen=True)
class ChunkSentimentResult:
    chunk_id: int
    text: str
    word_count: int
    finbert: SentimentModelOutput
    llm: SentimentModelOutput
    decision: SentimentDecision
    evidence_score: float
    chunk_words: int | None = None


@dataclass(frozen=True)
class ChunkScaleSummary:
    chunk_words: int
    overlap_words: int
    chunk_count: int
    label: str
    confidence: float
    probabilities: np.ndarray


@dataclass(frozen=True)
class DocumentSentimentResult:
    mode: str
    label: str
    confidence: float
    probabilities: np.ndarray
    chunks: list[ChunkSentimentResult]
    top_evidence: list[ChunkSentimentResult]
    explanation: str
    investment_support: str
    scale_summaries: list[ChunkScaleSummary] | None = None
    stable_across_scales: bool | None = None


def prediction_from_probs(probs: np.ndarray, model_name: str) -> SentimentModelOutput:
    idx = int(np.argmax(probs))
    return SentimentModelOutput(
        model_name=model_name,
        probabilities=np.asarray(probs, dtype=np.float64),
        prediction=LABELS[idx],
        confidence=float(probs[idx]),
    )


def apply_neutral_margin(probs: np.ndarray, margin: float) -> tuple[str, float, bool, float | None]:
    probs = np.asarray(probs, dtype=np.float64)
    best_idx = int(np.argmax(probs))
    base_label = LABELS[best_idx]
    if base_label == "neutral":
        return "neutral", float(probs[best_idx]), False, None

    neutral_idx = LABELS.index("neutral")
    gap = float(probs[best_idx] - probs[neutral_idx])
    if gap < margin:
        return "neutral", float(probs[neutral_idx]), True, gap
    return base_label, float(probs[best_idx]), False, gap


def formal_news_fusion(
    finbert_probs: np.ndarray,
    llm_probs: np.ndarray,
    threshold: float = FORMAL_NEWS_THRESHOLD,
    margin: float = FORMAL_NEWS_MARGIN,
) -> SentimentDecision:
    fused_probs = threshold_fusion(
        llm_probs.reshape(1, -1),
        finbert_probs.reshape(1, -1),
        threshold=threshold,
    )[0]
    llm_conf = float(np.max(llm_probs))
    source = "LoRA" if llm_conf >= threshold else "FinBERT"
    base_idx = int(np.argmax(fused_probs))
    base_label = LABELS[base_idx]
    base_confidence = float(fused_probs[base_idx])
    label, confidence, margin_applied, gap = apply_neutral_margin(fused_probs, margin)
    trace = (
        f"Formal-news mode used {source} because LoRA confidence "
        f"{llm_conf:.3f} {'>=' if llm_conf >= threshold else '<'} threshold {threshold:.2f}."
    )
    if margin_applied:
        trace += f" The base label {base_label} was changed to neutral because its margin over neutral was {gap:.3f}, below {margin:.2f}."
    return SentimentDecision(
        mode="formal_news",
        label=label,
        confidence=confidence,
        probabilities=fused_probs,
        base_label=base_label,
        base_confidence=base_confidence,
        neutral_margin_applied=margin_applied,
        neutral_margin_gap=gap,
        trace=trace,
    )


def general_mode_fusion(
    finbert_probs: np.ndarray,
    llm_probs: np.ndarray,
    margin: float = GENERAL_MODE_MARGIN,
) -> SentimentDecision:
    fused_probs = weighted_fusion(llm_probs.reshape(1, -1), finbert_probs.reshape(1, -1))[0]
    llm_conf = float(np.max(llm_probs))
    base_idx = int(np.argmax(fused_probs))
    base_label = LABELS[base_idx]
    base_confidence = float(fused_probs[base_idx])
    label, confidence, margin_applied, gap = apply_neutral_margin(fused_probs, margin)
    trace = f"General mode used weighted fusion with LoRA confidence as alpha ({llm_conf:.3f})."
    if margin_applied:
        trace += f" The base label {base_label} was changed to neutral because its margin over neutral was {gap:.3f}, below {margin:.2f}."
    return SentimentDecision(
        mode="general",
        label=label,
        confidence=confidence,
        probabilities=fused_probs,
        base_label=base_label,
        base_confidence=base_confidence,
        neutral_margin_applied=margin_applied,
        neutral_margin_gap=gap,
        trace=trace,
    )


def build_explanation(
    text: str,
    finbert: SentimentModelOutput,
    llm: SentimentModelOutput,
    decision: SentimentDecision,
) -> str:
    agreement = "agree" if finbert.prediction == llm.prediction else "disagree"
    uncertainty = ""
    if decision.confidence < 0.55:
        uncertainty = " The final confidence is low, so the result should be treated as uncertain."
    elif decision.neutral_margin_applied:
        uncertainty = " The neutral-margin rule indicates that the directional evidence is close to neutral."
    length_note = ""
    if len(text.split()) > 160:
        length_note = " The input is relatively long, so the model focuses on the first tokenized segment allowed by the configured maximum length."
    return (
        f"FinBERT predicts {finbert.prediction} ({finbert.confidence:.3f}) and "
        f"LoRA predicts {llm.prediction} ({llm.confidence:.3f}); the models {agreement}. "
        f"{decision.trace}{uncertainty}{length_note}"
    )


def build_investment_support(label: str, confidence: float) -> str:
    if confidence < 0.55:
        return (
            "Use this as weak evidence only. The sentiment boundary is uncertain, so the news should be checked against "
            "company fundamentals, valuation, market context, and follow-up disclosures."
        )
    if label == "positive":
        return (
            "The text supports a constructive interpretation from an investor perspective. It may indicate favorable "
            "business momentum, but it should not be treated as a standalone buy signal."
        )
    if label == "negative":
        return (
            "The text points to potentially adverse information from an investor perspective. It may justify closer "
            "risk review, but it should not be treated as a standalone sell signal."
        )
    return (
        "The text is best treated as contextual information rather than a strong directional signal. It may still matter "
        "when combined with earnings, guidance, valuation, or broader market conditions."
    )


def _format_probs(probs: np.ndarray) -> str:
    return ", ".join(f"{label}={float(probs[idx]):.3f}" for idx, label in enumerate(LABELS))


def _truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " ..."


def generate_llm_investment_support_for_text(text: str, result: SentimentSystemResult, llm_model) -> str:
    evidence = _truncate_words(text, 260)
    user_prompt = f"""Write a concise investment-support insight in 3-5 sentences.

Model outputs:
- Final sentiment: {result.decision.label}
- Final confidence: {result.decision.confidence:.3f}
- Final probabilities: {_format_probs(result.decision.probabilities)}
- FinBERT: {result.finbert.prediction} ({result.finbert.confidence:.3f})
- Qwen3-4B LoRA: {result.llm.prediction} ({result.llm.confidence:.3f})
- Decision trace: {result.decision.trace}

Evidence text:
{evidence}

Requirements:
- Refer to specific information from the evidence text.
- Explain why the sentiment is positive, neutral, or negative from an investor perspective.
- Mention uncertainty if the confidence is moderate or if the evidence is mixed.
- Do not recommend buying, selling, holding, or trading the security.
"""
    try:
        generated = llm_model.generate_response(
            SUPPORT_SYSTEM_PROMPT,
            user_prompt,
            max_new_tokens=190,
            max_input_tokens=1536,
            disable_adapter=True,
        )
        return generated or build_investment_support(result.decision.label, result.decision.confidence)
    except Exception:
        return build_investment_support(result.decision.label, result.decision.confidence)


def generate_llm_investment_support_for_document(result: DocumentSentimentResult, llm_model) -> str:
    chunk_lines = []
    for chunk in result.top_evidence:
        chunk_lines.append(
            f"Chunk {chunk.chunk_id + 1}: label={chunk.decision.label}, "
            f"confidence={chunk.decision.confidence:.3f}, "
            f"FinBERT={chunk.finbert.prediction} ({chunk.finbert.confidence:.3f}), "
            f"LoRA={chunk.llm.prediction} ({chunk.llm.confidence:.3f})\n"
            f"Evidence: {_truncate_words(chunk.text, 120)}"
        )
    evidence = "\n\n".join(chunk_lines)
    user_prompt = f"""Write a concise investment-support insight for the full article in 4-6 sentences.

Document-level model outputs:
- Final sentiment: {result.label}
- Final confidence: {result.confidence:.3f}
- Final probabilities: {_format_probs(result.probabilities)}
- Number of analyzed chunks: {len(result.chunks)}
- Mode: {result.mode}

Top evidence chunks:
{evidence}

Requirements:
- Base the insight on the evidence chunks and document-level sentiment.
- Refer to concrete events, signals, or claims from the evidence.
- Explain why the document-level sentiment is positive, neutral, or negative from an investor perspective.
- Mention if the article contains mixed or forward-looking evidence.
- Do not recommend buying, selling, holding, or trading the security.
"""
    try:
        generated = llm_model.generate_response(
            SUPPORT_SYSTEM_PROMPT,
            user_prompt,
            max_new_tokens=230,
            max_input_tokens=1800,
            disable_adapter=True,
        )
        return generated or build_investment_support(result.label, result.confidence)
    except Exception:
        return build_investment_support(result.label, result.confidence)


def split_text_into_chunks(
    text: str,
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    overlap_words: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    chunks: list[str] = []
    current: list[str] = []

    def flush_current() -> None:
        nonlocal current
        if current:
            chunks.append(" ".join(current).strip())
            current = []

    for paragraph in paragraphs:
        words = paragraph.split()
        if not words:
            continue
        if len(words) > chunk_words:
            flush_current()
            step = max(1, chunk_words - overlap_words)
            for start in range(0, len(words), step):
                window = words[start : start + chunk_words]
                if window:
                    chunks.append(" ".join(window).strip())
                if start + chunk_words >= len(words):
                    break
            continue
        if len(current) + len(words) > chunk_words:
            flush_current()
        current.extend(words)
    flush_current()

    if not chunks and text.strip():
        words = text.split()
        step = max(1, chunk_words - overlap_words)
        for start in range(0, len(words), step):
            window = words[start : start + chunk_words]
            if window:
                chunks.append(" ".join(window).strip())
            if start + chunk_words >= len(words):
                break
    return chunks


def should_use_document_mode(text: str, threshold_words: int = DEFAULT_LONG_TEXT_WORD_THRESHOLD) -> bool:
    return len(text.split()) > threshold_words


def decide_from_probabilities(finbert_probs: np.ndarray, llm_probs: np.ndarray, mode: str) -> SentimentDecision:
    if mode == "formal_news":
        return formal_news_fusion(finbert_probs, llm_probs)
    if mode == "general":
        return general_mode_fusion(finbert_probs, llm_probs)
    raise ValueError(f"Unsupported mode: {mode}")


def analyze_with_models(text: str, finbert_model, llm_model, mode: str) -> SentimentSystemResult:
    finbert_probs = finbert_model.predict_proba([text], batch_size=1)[0]
    llm_probs = llm_model.predict_proba([text], batch_size=1)[0]
    finbert = prediction_from_probs(finbert_probs, "ProsusAI/finbert")
    llm = prediction_from_probs(llm_probs, "Qwen3-4B neutral-aware LoRA r8")
    decision = decide_from_probabilities(finbert.probabilities, llm.probabilities, mode)
    explanation = build_explanation(text, finbert, llm, decision)
    support = build_investment_support(decision.label, decision.confidence)
    return SentimentSystemResult(
        finbert=finbert,
        llm=llm,
        decision=decision,
        explanation=explanation,
        investment_support=support,
    )


def chunk_evidence_score(decision: SentimentDecision) -> float:
    neutral_idx = LABELS.index("neutral")
    if decision.label == "neutral":
        directional = float(max(decision.probabilities[LABELS.index("negative")], decision.probabilities[LABELS.index("positive")]))
        return float(max(decision.confidence, directional))
    return float(decision.confidence - decision.probabilities[neutral_idx])


def aggregate_chunk_probabilities(chunks: list[ChunkSentimentResult]) -> np.ndarray:
    if not chunks:
        raise ValueError("Cannot aggregate an empty chunk list.")
    probs = np.asarray([chunk.decision.probabilities for chunk in chunks], dtype=np.float64)
    confidences = np.asarray([chunk.decision.confidence for chunk in chunks], dtype=np.float64)
    directional_strength = np.asarray(
        [
            max(chunk.decision.probabilities[LABELS.index("negative")], chunk.decision.probabilities[LABELS.index("positive")])
            for chunk in chunks
        ],
        dtype=np.float64,
    )
    weights = np.maximum(confidences, directional_strength)
    weights = np.clip(weights, 1e-6, None)
    aggregated = np.average(probs, axis=0, weights=weights)
    return aggregated / aggregated.sum()


def build_document_explanation(result: DocumentSentimentResult, mode: str) -> str:
    counts = {label: sum(1 for chunk in result.chunks if chunk.decision.label == label) for label in LABELS}
    evidence_ids = ", ".join(str(chunk.chunk_id + 1) for chunk in result.top_evidence)
    explanation = (
        f"The article was split into {len(result.chunks)} chunks and each chunk was analyzed with FinBERT, LoRA, and fusion. "
        f"Chunk labels were negative={counts['negative']}, neutral={counts['neutral']}, positive={counts['positive']}. "
        f"The document-level probabilities are confidence-weighted across chunks, using mode `{mode}`. "
        f"The strongest evidence chunks are: {evidence_ids}."
    )
    if result.scale_summaries:
        scale_parts = [
            f"{summary.chunk_words}w -> {summary.label} ({summary.confidence:.3f}, {summary.chunk_count} chunks)"
            for summary in result.scale_summaries
        ]
        stability = "stable" if result.stable_across_scales else "unstable"
        explanation += (
            f" Robust multi-scale chunking is {stability} across configured chunk sizes: "
            f"{'; '.join(scale_parts)}."
        )
    return explanation


def _analyze_chunk_texts(
    chunk_texts: list[str],
    finbert_model,
    llm_model,
    mode: str,
    chunk_words: int | None = None,
    start_chunk_id: int = 0,
) -> list[ChunkSentimentResult]:
    finbert_probs = finbert_model.predict_proba(chunk_texts, batch_size=8)
    llm_probs = llm_model.predict_proba(chunk_texts, batch_size=1)

    chunks: list[ChunkSentimentResult] = []
    for idx, chunk_text in enumerate(chunk_texts):
        finbert = prediction_from_probs(finbert_probs[idx], "ProsusAI/finbert")
        llm = prediction_from_probs(llm_probs[idx], "Qwen3-4B neutral-aware LoRA r8")
        decision = decide_from_probabilities(finbert.probabilities, llm.probabilities, mode)
        chunks.append(
            ChunkSentimentResult(
                chunk_id=start_chunk_id + idx,
                text=chunk_text,
                word_count=len(chunk_text.split()),
                finbert=finbert,
                llm=llm,
                decision=decision,
                evidence_score=chunk_evidence_score(decision),
                chunk_words=chunk_words,
            )
        )
    return chunks


def analyze_document_with_models(
    text: str,
    finbert_model,
    llm_model,
    mode: str,
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    overlap_words: int = DEFAULT_CHUNK_OVERLAP,
    top_k: int = 3,
    robust_multiscale: bool = True,
    chunk_word_scales: tuple[int, ...] = DEFAULT_CHUNK_WORD_SCALES,
) -> DocumentSentimentResult:
    scales = tuple(sorted(set(chunk_word_scales))) if robust_multiscale else (chunk_words,)
    chunks: list[ChunkSentimentResult] = []
    scale_summaries: list[ChunkScaleSummary] = []
    scale_probs: list[np.ndarray] = []
    scale_weights: list[float] = []

    for scale in scales:
        chunk_texts = split_text_into_chunks(text, chunk_words=scale, overlap_words=overlap_words)
        if not chunk_texts:
            continue
        scale_chunks = _analyze_chunk_texts(
            chunk_texts,
            finbert_model,
            llm_model,
            mode,
            chunk_words=scale,
            start_chunk_id=len(chunks),
        )
        chunks.extend(scale_chunks)
        current_probs = aggregate_chunk_probabilities(scale_chunks)
        current_label = LABELS[int(np.argmax(current_probs))]
        current_confidence = float(np.max(current_probs))
        scale_probs.append(current_probs)
        scale_weights.append(current_confidence)
        scale_summaries.append(
            ChunkScaleSummary(
                chunk_words=scale,
                overlap_words=overlap_words,
                chunk_count=len(scale_chunks),
                label=current_label,
                confidence=current_confidence,
                probabilities=current_probs,
            )
        )

    if not chunks or not scale_probs:
        raise ValueError("No readable text chunks were produced.")

    if robust_multiscale:
        weights = np.asarray(scale_weights, dtype=np.float64)
        weights = np.clip(weights, 1e-6, None)
        document_probs = np.average(np.asarray(scale_probs, dtype=np.float64), axis=0, weights=weights)
        document_probs = document_probs / document_probs.sum()
    else:
        document_probs = scale_probs[0]
    label = LABELS[int(np.argmax(document_probs))]
    confidence = float(np.max(document_probs))
    stable_across_scales = len({summary.label for summary in scale_summaries}) == 1
    top_evidence = sorted(chunks, key=lambda chunk: chunk.evidence_score, reverse=True)[:top_k]
    placeholder = DocumentSentimentResult(
        mode=mode,
        label=label,
        confidence=confidence,
        probabilities=document_probs,
        chunks=chunks,
        top_evidence=top_evidence,
        explanation="",
        investment_support=build_investment_support(label, confidence),
        scale_summaries=scale_summaries,
        stable_across_scales=stable_across_scales,
    )
    explanation = build_document_explanation(placeholder, mode=mode)
    return DocumentSentimentResult(
        mode=mode,
        label=label,
        confidence=confidence,
        probabilities=document_probs,
        chunks=chunks,
        top_evidence=top_evidence,
        explanation=explanation,
        investment_support=build_investment_support(label, confidence),
        scale_summaries=scale_summaries,
        stable_across_scales=stable_across_scales,
    )
