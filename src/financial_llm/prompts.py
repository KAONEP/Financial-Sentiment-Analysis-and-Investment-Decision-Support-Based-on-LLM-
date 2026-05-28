from __future__ import annotations


SYSTEM_PROMPT = (
    "You are a financial sentiment analysis assistant. "
    "Classify the sentiment from the perspective of a retail investor."
)


def direct_prompt(sentence: str) -> str:
    return (
        "Classify the financial sentiment of the following text as exactly one of "
        "negative, neutral, or positive.\n\n"
        f"Text: {sentence}\n\n"
        "Answer with only one label.\n"
        "Answer:"
    )


def neutral_aware_prompt(sentence: str) -> str:
    return (
        "Classify the financial sentiment of the following text as exactly one of "
        "negative, neutral, or positive from the perspective of a retail investor.\n\n"
        "Use positive only when the sentence states a clear beneficial financial implication, "
        "such as higher revenue, higher profit, improved guidance, a favorable rating action, "
        "or another explicitly positive investor impact.\n"
        "Use negative only when the sentence states a clear harmful financial implication, "
        "such as lower revenue, losses, layoffs, reduced guidance, a downgrade, or another "
        "explicitly negative investor impact.\n"
        "Use neutral when the sentence only reports a factual corporate event, agreement, "
        "product launch, acquisition, order, market listing, or management statement without "
        "a clear positive or negative financial impact.\n\n"
        f"Text: {sentence}\n\n"
        "Answer with only one label.\n"
        "Answer:"
    )


def reasoning_label_prompt(sentence: str) -> str:
    return (
        "Classify the financial sentiment of the following text as exactly one of "
        "negative, neutral, or positive.\n\n"
        "Before choosing the label, consider the likely impact from the perspective "
        "of a retail investor: whether the event is beneficial, harmful, or not clearly directional. "
        "Do not write the reasoning. Answer with only one label.\n\n"
        f"Text: {sentence}\n\n"
        "Answer:"
    )


def reasoning_prompt(sentence: str) -> str:
    return (
        "Analyze the financial news text from the perspective of a retail investor. "
        "Consider whether the described event is likely beneficial, harmful, or not clearly directional. "
        "Then provide the final sentiment label as exactly one of negative, neutral, or positive.\n\n"
        f"Text: {sentence}\n\n"
        "Return format:\n"
        "Reason: <brief reason>\n"
        "Label: <negative|neutral|positive>\n"
    )


def investment_support_prompt(text: str, sentiment: str, confidence: float) -> str:
    return (
        "Write a brief investment decision support insight. "
        "Do not give direct trading instructions. "
        "Mention uncertainty when confidence is low.\n\n"
        f"News text: {text}\n"
        f"Predicted sentiment: {sentiment}\n"
        f"Confidence: {confidence:.3f}\n"
    )
