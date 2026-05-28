from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import trafilatura

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from financial_llm.finbert import FinBertSentiment
from financial_llm.labels import LABELS
from financial_llm.llm_classifier import LabelScoringLLM
from financial_llm.system import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_WORDS,
    DEFAULT_CHUNK_WORD_SCALES,
    analyze_document_with_models,
    analyze_with_models,
    generate_llm_investment_support_for_document,
    generate_llm_investment_support_for_text,
    should_use_document_mode,
)


BASE_DIR = Path(__file__).resolve().parents[1]
ADAPTER_PATH = BASE_DIR / "adapters" / "neutral_aware_lora_r8_full_raw_seed42"
MODE_OPTIONS = {
    "Formal financial news": "formal_news",
    "General / cross-domain": "general",
}
CHUNK_STRATEGY_OPTIONS = {
    "Balanced (recommended)": {
        "robust_multiscale": True,
        "scales": (120, 220),
        "caption": "Runs 120 and 220 word windows. This is the default speed/stability trade-off.",
    },
    "Fast": {
        "robust_multiscale": False,
        "scales": (),
        "caption": "Runs one configurable chunk size. Fastest, but no chunk-size stability check.",
    },
    "Robust research": {
        "robust_multiscale": True,
        "scales": DEFAULT_CHUNK_WORD_SCALES,
        "caption": "Runs 80, 120, 160, and 220 word windows. Best for research checks, slowest.",
    },
}


@st.cache_resource(show_spinner="Loading FinBERT...")
def load_finbert() -> FinBertSentiment:
    return FinBertSentiment("ProsusAI/finbert")


@st.cache_resource(show_spinner="Loading Qwen3-4B LoRA adapter...")
def load_lora_classifier() -> LabelScoringLLM:
    if not ADAPTER_PATH.exists():
        raise FileNotFoundError(f"LoRA adapter not found: {ADAPTER_PATH}")
    return LabelScoringLLM(
        model_name="Qwen/Qwen3-4B",
        dtype="bfloat16",
        enable_thinking=False,
        max_length=384,
        prompt_mode="neutral_aware",
        adapter_path=str(ADAPTER_PATH),
    )


def extract_url_text(url: str) -> str:
    session = requests.Session()
    session.trust_env = False
    response = session.get(
        url,
        timeout=20,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/126.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    response.raise_for_status()
    extracted = trafilatura.extract(response.text, include_comments=False, include_tables=False)
    if not extracted:
        raise ValueError("Could not extract readable article text from this URL.")
    return clean_text(extracted)


def clean_text(text: str) -> str:
    return "\n".join(line.strip() for line in text.splitlines() if line.strip()).strip()


def probability_frame(result) -> pd.DataFrame:
    rows = []
    for source, output in [
        ("FinBERT", result.finbert),
        ("Qwen3-4B LoRA", result.llm),
        ("Final fusion", result.decision),
    ]:
        probs = output.probabilities
        rows.append(
            {
                "source": source,
                "negative": float(probs[LABELS.index("negative")]),
                "neutral": float(probs[LABELS.index("neutral")]),
                "positive": float(probs[LABELS.index("positive")]),
                "prediction": output.prediction if source != "Final fusion" else result.decision.label,
                "confidence": output.confidence if source != "Final fusion" else result.decision.confidence,
            }
        )
    return pd.DataFrame(rows)


def document_probability_frame(result) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source": "Document aggregation",
                "negative": float(result.probabilities[LABELS.index("negative")]),
                "neutral": float(result.probabilities[LABELS.index("neutral")]),
                "positive": float(result.probabilities[LABELS.index("positive")]),
                "prediction": result.label,
                "confidence": result.confidence,
            }
        ]
    )


def chunk_frame(result) -> pd.DataFrame:
    rows = []
    for chunk in result.chunks:
        rows.append(
            {
                "chunk": chunk.chunk_id + 1,
                "words": chunk.word_count,
                "final_label": chunk.decision.label,
                "final_confidence": chunk.decision.confidence,
                "scale_words": chunk.chunk_words,
                "finbert": chunk.finbert.prediction,
                "finbert_confidence": chunk.finbert.confidence,
                "lora": chunk.llm.prediction,
                "lora_confidence": chunk.llm.confidence,
                "evidence_score": chunk.evidence_score,
                "excerpt": chunk.text[:280] + ("..." if len(chunk.text) > 280 else ""),
            }
        )
    return pd.DataFrame(rows)


def scale_summary_frame(result) -> pd.DataFrame:
    if not result.scale_summaries:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "window_words": summary.chunk_words,
                "overlap_words": summary.overlap_words,
                "windows": summary.chunk_count,
                "label": summary.label,
                "confidence": summary.confidence,
                "negative": float(summary.probabilities[LABELS.index("negative")]),
                "neutral": float(summary.probabilities[LABELS.index("neutral")]),
                "positive": float(summary.probabilities[LABELS.index("positive")]),
            }
            for summary in result.scale_summaries
        ]
    )


def render_result(result, investment_support: str | None = None) -> None:
    metric_cols = st.columns(4)
    metric_cols[0].metric("Final sentiment", result.decision.label)
    metric_cols[1].metric("Final confidence", f"{result.decision.confidence:.3f}")
    metric_cols[2].metric("FinBERT", f"{result.finbert.prediction} ({result.finbert.confidence:.3f})")
    metric_cols[3].metric("LoRA", f"{result.llm.prediction} ({result.llm.confidence:.3f})")

    st.subheader("Investment support insight")
    st.write(investment_support or result.investment_support)
    if investment_support:
        st.caption("Generated by Qwen3-4B from the model outputs and evidence text. This is decision support, not trading advice.")

    probs = probability_frame(result)
    with st.expander("Research details"):
        prob_tab, trace_tab = st.tabs(["Model probabilities", "Decision trace"])
        with prob_tab:
            st.dataframe(probs, use_container_width=True, hide_index=True)
            chart_df = probs.set_index("source")[["negative", "neutral", "positive"]]
            st.bar_chart(chart_df)
        with trace_tab:
            st.write(result.explanation)
            st.write(f"Mode: `{result.decision.mode}`")
            st.write(f"Base fusion label before neutral-margin: `{result.decision.base_label}`")
            st.write(f"Base fusion confidence: `{result.decision.base_confidence:.3f}`")
            st.write(f"Neutral-margin applied: `{result.decision.neutral_margin_applied}`")
            if result.decision.neutral_margin_gap is not None:
                st.write(f"Directional minus neutral margin: `{result.decision.neutral_margin_gap:.3f}`")
            st.caption(
                "This system provides decision-support evidence, not trading instructions. "
                "The final confidence is the fused probability assigned to the displayed final label."
            )


def render_document_result(result, investment_support: str | None = None) -> None:
    metric_cols = st.columns(5)
    metric_cols[0].metric("Document sentiment", result.label)
    metric_cols[1].metric("Document confidence", f"{result.confidence:.3f}")
    metric_cols[2].metric("Chunks analyzed", str(len(result.chunks)))
    metric_cols[3].metric("Mode", result.mode)
    if result.stable_across_scales is None:
        metric_cols[4].metric("Chunk stability", "single-scale")
    else:
        metric_cols[4].metric("Chunk stability", "stable" if result.stable_across_scales else "sensitive")

    if result.stable_across_scales is False:
        st.warning(
            "Chunk-size sensitivity detected. Different chunk sizes produced different document-level labels, "
            "so this result should be treated as less stable."
        )

    st.subheader("Investment support insight")
    st.write(investment_support or result.investment_support)
    if investment_support:
        st.caption("Generated by Qwen3-4B from the document-level result and top evidence chunks. This is decision support, not trading advice.")

    probs = document_probability_frame(result)

    with st.expander("Key supporting excerpts", expanded=True):
        st.caption("Most influential article excerpts selected from the model evidence.")
        for idx, chunk in enumerate(result.top_evidence, start=1):
            st.markdown(
                f"**Excerpt {idx}** · {chunk.decision.label} "
                f"({chunk.decision.confidence:.3f})"
            )
            st.write(chunk.text)
            st.caption(
                f"Window: {chunk.chunk_words or 'single'} words | "
                f"FinBERT: {chunk.finbert.prediction} ({chunk.finbert.confidence:.3f}) | "
                f"LoRA: {chunk.llm.prediction} ({chunk.llm.confidence:.3f}) | "
                f"Evidence score: {chunk.evidence_score:.3f}"
            )
            if idx < len(result.top_evidence):
                st.divider()

    with st.expander("Research details"):
        prob_tab, scale_tab, chunk_tab, trace_tab = st.tabs(
            ["Document probabilities", "Window stability", "All excerpt windows", "Aggregation trace"]
        )
        with prob_tab:
            st.dataframe(probs, use_container_width=True, hide_index=True)
            st.bar_chart(probs.set_index("source")[["negative", "neutral", "positive"]])
        with scale_tab:
            scale_df = scale_summary_frame(result)
            if scale_df.empty:
                st.caption("Single-window mode was used, so no cross-window stability table is available.")
            else:
                st.dataframe(scale_df, use_container_width=True, hide_index=True)
        with chunk_tab:
            st.dataframe(chunk_frame(result), use_container_width=True, hide_index=True)
        with trace_tab:
            st.write(result.explanation)


def main() -> None:
    st.set_page_config(page_title="Financial LLM Sentiment", layout="wide")
    st.title("Financial Sentiment Analysis and Investment Support")

    with st.sidebar:
        st.header("Inference mode")
        mode_label = st.radio(
            "Choose mode",
            list(MODE_OPTIONS.keys()),
            index=0,
            help=(
                "Formal financial news uses the in-domain PhraseBank-best threshold + neutral-margin rule. "
                "General mode uses cross-domain weighted fusion selected with Twitter robustness."
            ),
        )
        mode = MODE_OPTIONS[mode_label]
        st.divider()
        document_mode = st.toggle(
            "Auto chunk long articles",
            value=True,
            help="When enabled, long text is split into chunks and chunk probabilities are aggregated.",
        )
        generate_support = st.toggle(
            "Generate LLM investment insight",
            value=True,
            help="Use Qwen3-4B to write a content-specific investment-support insight from model outputs and evidence chunks.",
        )
        with st.expander("Advanced settings"):
            chunk_strategy_label = st.selectbox(
                "Long-article speed",
                list(CHUNK_STRATEGY_OPTIONS.keys()),
                index=0,
                help="Choose the speed/stability trade-off for URL articles and long pasted text.",
            )
            chunk_strategy = CHUNK_STRATEGY_OPTIONS[chunk_strategy_label]
            robust_multiscale = bool(chunk_strategy["robust_multiscale"])
            chunk_word_scales = tuple(chunk_strategy["scales"])
            st.caption(chunk_strategy["caption"])
            if robust_multiscale:
                chunk_words = DEFAULT_CHUNK_WORDS
            else:
                chunk_words = st.number_input("Chunk words", min_value=60, max_value=220, value=DEFAULT_CHUNK_WORDS, step=10)
            chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=80, value=DEFAULT_CHUNK_OVERLAP, step=5)
        st.divider()
        st.caption("Final model: FinBERT + Qwen3-4B neutral-aware LoRA r8.")
        st.caption("LoRA adapter: adapters/neutral_aware_lora_r8_full_raw_seed42")

    input_mode = st.radio("Input type", ["Text", "URL"], horizontal=True)
    text = ""
    if input_mode == "Text":
        text = st.text_area("Financial news text", height=230, placeholder="Paste an English financial news sentence or article excerpt.")
    else:
        url = st.text_input("Financial news URL", placeholder="https://...")
        if url:
            try:
                text = extract_url_text(url)
                st.text_area("Extracted text", value=text, height=230)
            except Exception as exc:
                st.error(str(exc))

    text = clean_text(text)
    use_document_analysis = bool(text) and document_mode and should_use_document_mode(text)
    if use_document_analysis:
        st.info("Long article detected. The system will analyze chunks and aggregate document-level sentiment.")

    analyze = st.button("Analyze sentiment", type="primary", disabled=not bool(text))
    if analyze:
        try:
            finbert = load_finbert()
            lora = load_lora_classifier()
            if use_document_analysis:
                with st.spinner("Running chunk-level FinBERT, Qwen3-4B LoRA, and document aggregation..."):
                    result = analyze_document_with_models(
                        text,
                        finbert,
                        lora,
                        mode,
                        chunk_words=int(chunk_words),
                        overlap_words=int(chunk_overlap),
                        robust_multiscale=robust_multiscale,
                        chunk_word_scales=chunk_word_scales,
                    )
                support = None
                if generate_support:
                    with st.spinner("Generating content-specific investment support insight..."):
                        support = generate_llm_investment_support_for_document(result, lora)
                render_document_result(result, support)
            else:
                with st.spinner("Running FinBERT, Qwen3-4B LoRA, and fusion..."):
                    result = analyze_with_models(text, finbert, lora, mode)
                support = None
                if generate_support:
                    with st.spinner("Generating content-specific investment support insight..."):
                        support = generate_llm_investment_support_for_text(text, result, lora)
                render_result(result, support)
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")


if __name__ == "__main__":
    main()
