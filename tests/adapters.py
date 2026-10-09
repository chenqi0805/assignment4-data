from __future__ import annotations

import os
from typing import Any

from cs336_data.classifiers import detect_nsfw, detect_toxicity, identify_language
from cs336_data.dedup import exact_line_deduplication
from cs336_data.extraction import extract_text_from_html_bytes
from cs336_data.gopher import all_gopher_rules_pass
from cs336_data.minhash import minhash_deduplication
from cs336_data.pii import mask_emails, mask_ips, mask_phone_numbers


def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    return extract_text_from_html_bytes(html_bytes)


def run_identify_language(text: str) -> tuple[Any, float]:
    return identify_language(text)


def run_mask_emails(text: str) -> tuple[str, int]:
    return mask_emails(text)


def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    return mask_phone_numbers(text)


def run_mask_ips(text: str) -> tuple[str, int]:
    return mask_ips(text)


def run_classify_nsfw(text: str) -> tuple[Any, float]:
    return detect_nsfw(text)


def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    return detect_toxicity(text)


def run_classify_quality(text: str) -> tuple[Any, float]:
    raise NotImplementedError


def run_gopher_quality_filter(text: str) -> bool:
    return all_gopher_rules_pass(text)


def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    exact_line_deduplication(input_files, output_directory)


def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    minhash_deduplication(
        input_files,
        output_directory,
        num_hashes=num_hashes,
        num_bands=num_bands,
        ngrams=ngrams,
        jaccard_threshold=jaccard_threshold,
    )
