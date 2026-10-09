"""Quality classification (handout problem 2.7).

The model is a fastText supervised classifier trained by
``scripts/train_quality_classifier.py`` (positives: pages at URLs linked from English
Wikipedia; negatives: random Common Crawl pages). The binary lives in the shared assets
directory (``get_shared_assets_path() / "classifiers" / "quality_classifier.bin"``) and
is never committed to git; the training script is the reproducible artifact.
"""

from __future__ import annotations

from cs336_data.classifiers import get_model

_QUALITY_MODEL = "quality_classifier.bin"
_WIKI = "wiki"
_CC = "cc"


def classify_quality(text: str) -> tuple[str, float]:
    """Top quality prediction as ``(label, probability)`` — "wiki" (high) or "cc" (low).

    fastText's predict() rejects newlines, so documents are flattened to one line first.
    """
    labels, probs = get_model(_QUALITY_MODEL).predict(text.replace("\n", " "))
    return labels[0].removeprefix("__label__"), float(probs[0])


def _probability_of(text: str, label: str) -> float:
    """fastText's probability for a specific class (1 - p when it is not the top label)."""
    predicted, probability = classify_quality(text)
    return probability if predicted == label else 1.0 - probability


def fetch_qualityScores_wiki(url: str, text: str) -> tuple[str, float]:
    """Quality score of ``text`` as a high-quality page: ``("wiki", P(wiki))``.

    ``url`` is accepted to match the handout's page-scoring framing; the classifier is
    text-only.
    """
    return (_WIKI, _probability_of(text, _WIKI))


def fetch_qualityScores_cc(url: str, text: str) -> tuple[str, float]:
    """Quality score of ``text`` as a low-quality page: ``("cc", P(cc))``.

    ``url`` is accepted to match the handout's page-scoring framing; the classifier is
    text-only.
    """
    return (_CC, _probability_of(text, _CC))
