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


def get_model(filename: str) -> fasttext.FastText._FastText:
    """Load the named classifier from the shared assets directory, once per process."""
    if filename not in _MODEL_CACHE:
        model_path = get_shared_assets_path() / "classifiers" / filename
        _MODEL_CACHE[filename] = fasttext.load_model(str(model_path))
    return _MODEL_CACHE[filename]
