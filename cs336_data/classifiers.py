"""Shared fastText model loading for the model-backed classifiers.

Model binaries live under ``get_shared_assets_path() / "classifiers"`` and are
fetched by ``scripts/download_data.py --offline-only``. They range from ~130 MB
(lid.176.bin) to ~1 GB (the Dolma Jigsaw models), so each model is loaded
lazily on first use and cached at module level rather than per call.
"""

from __future__ import annotations

import fasttext

from cs336_data.common import get_shared_assets_path

_MODEL_CACHE: dict[str, fasttext.FastText._FastText] = {}

_LID_MODEL = "lid.176.bin"
_NSFW_MODEL = "dolma_fasttext_nsfw_jigsaw_model.bin"
_TOXIC_MODEL = "dolma_fasttext_hatespeech_jigsaw_model.bin"


def get_model(filename: str) -> fasttext.FastText._FastText:
    """Load the named classifier from the shared assets directory, once per process."""
    if filename not in _MODEL_CACHE:
        model_path = get_shared_assets_path() / "classifiers" / filename
        _MODEL_CACHE[filename] = fasttext.load_model(str(model_path))
    return _MODEL_CACHE[filename]


def _predict_top(text: str, model_filename: str) -> tuple[str, float]:
    """Top fastText prediction as (bare label, probability).

    fastText's predict() rejects newlines ("processes one line at a time"),
    so multi-line documents are flattened to one line first.
    """
    labels, probs = get_model(model_filename).predict(text.replace("\n", " "))
    return labels[0].removeprefix("__label__"), float(probs[0])


def identify_language(text: str) -> tuple[str, float]:
    """Predict the language of `text` as an ISO code (e.g. "en", "zh") with confidence."""
    return _predict_top(text, _LID_MODEL)


def detect_nsfw(text: str) -> tuple[str, float]:
    """Predict whether `text` is NSFW: "nsfw" or "non-nsfw", with confidence."""
    return _predict_top(text, _NSFW_MODEL)


def detect_toxicity(text: str) -> tuple[str, float]:
    """Predict whether `text` is toxic speech: "toxic" or "non-toxic", with confidence."""
    return _predict_top(text, _TOXIC_MODEL)
