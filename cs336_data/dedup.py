"""Exact deduplication (problem 3.1).

Drop-all semantics at two granularities: any element that appears more than
once — a whole document, or a line across a corpus of files — is removed at
EVERY occurrence, not keep-first. Elements reduced to nothing still produce
empty output files when the output is one-file-per-input.

Line-level dedup keys lines by their MurmurHash3 digest so the corpus-wide
frequency table never materializes the lines themselves (handout 3.1).
"""

import os
from collections import Counter
from pathlib import Path

import mmh3
from xopen import xopen


def deduplicate_document_indices(docs: list[str]) -> list[int]:
    """Indices of the documents that survive drop-all exact deduplication.

    A document survives when it is non-empty and appears exactly once; the
    returned indices keep the original relative order. Index-preserving so
    callers can recover provenance for the survivors.
    """
    counts = Counter(docs)
    return [idx for idx, doc in enumerate(docs) if doc and counts[doc] == 1]


def deduplicate_documents_exact(docs: list[str]) -> list[str]:
    """Drop every document appearing more than once and every empty document.

    Survivors keep their original relative order.
    """
    return [docs[idx] for idx in deduplicate_document_indices(docs)]


def _line_key(line: str) -> int:
    # Iterate-lines content: the trailing newline is the terminator, not part
    # of the text being compared.
    content = line.removesuffix("\n")
    return mmh3.hash128(content)


def exact_line_deduplication(input_files: list[os.PathLike], output_directory: os.PathLike) -> None:
    """Rewrite each input into output_directory (same basename), keeping only
    the lines that occur exactly once across ALL input files."""
    paths = [Path(p) for p in input_files]
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts: Counter[int] = Counter()
    for path in paths:
        with xopen(path, "rt", encoding="utf-8") as f:
            for line in f:
                counts[_line_key(line)] += 1

    # Re-read rather than holding every file's lines in memory: the count
    # table is the only corpus-sized state, and it stores hashes. Lines keep
    # their terminators, so a file's trailing newline survives when its last
    # line survives, and a fully-emptied file is written as 0 bytes.
    for path in paths:
        kept: list[str] = []
        with xopen(path, "rt", encoding="utf-8") as f:
            for line in f:
                if counts[_line_key(line)] == 1:
                    kept.append(line)
        with xopen(output_dir / path.name, "wt", encoding="utf-8") as f:
            f.writelines(kept)
