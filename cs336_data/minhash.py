"""MinHash + LSH document deduplication (problem 3.2).

Documents are shingled into word n-grams after the handout's normalization
(NFD unicode, accents stripped, lowercased, punctuation removed, whitespace
squeezed). Each of the ``num_hashes`` signature rows is the minimum, over
n-grams, of MurmurHash3 of the UTF-8 (8-bit) encoded n-gram string with
``seed=i`` for the i-th hash. Consecutive signature rows form bands; documents
sharing any band are candidates, verified with the true Jaccard similarity of
their shingle sets. Verified pairs cluster transitively and one document per
cluster survives, matching the handout ("randomly retain one").
"""

import os
import random
import string
import unicodedata
from collections import defaultdict
from pathlib import Path

import mmh3
import numpy as np
from xopen import xopen

DEFAULT_NGRAMS = 5

# Seeded so runs are reproducible; the handout asks for a random survivor per
# cluster, not a random result across runs.
_RETENTION_SEED = 0

# RefinedWeb normalization, as prescribed by handout 3.2.
_PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().translate(_PUNCTUATION_TABLE)
    return " ".join(text.split())


def _ngrams(doc: str, ngrams: int) -> list[str]:
    """Whitespace-joined word n-grams of the normalized document."""
    words = normalize(doc).split()
    return [" ".join(words[i : i + ngrams]) for i in range(max(0, len(words) - ngrams + 1))]


def _shingle_set(doc: str, ngrams: int) -> set[int]:
    return {mmh3.hash(gram, signed=False) for gram in _ngrams(doc, ngrams)}


def _signature_from_grams(n: int, grams: list[str]) -> np.ndarray:
    signature = np.empty(n, dtype=np.uint64)
    for i in range(n):
        signature[i] = min(mmh3.hash(gram, seed=i, signed=False) for gram in grams)
    return signature


def signature(n: int, doc: str, ngrams: int = DEFAULT_NGRAMS) -> np.ndarray:
    """MinHash signature of one document: uint64, one min-hash per n-gram window.

    A document too short to yield any n-gram gets an all-zero signature.
    """
    return _signature_from_grams(n, _ngrams(doc, ngrams))


def buckets(signature: np.ndarray, num_buckets: int) -> list[list[int]]:
    """Split a signature into num_buckets contiguous bands of rows."""
    rows = len(signature) // num_buckets
    return [signature[i * rows : (i + 1) * rows].tolist() for i in range(num_buckets)]


def parameters(num_hashes: int, num_bands: int) -> tuple[int, float]:
    """Rows per band, and the banding scheme's approximate Jaccard threshold.

    The threshold is the inflection point (1/b)^(1/r) of the LSH S-curve;
    callers with an explicit threshold (the adapter contract) pass their own.
    Divisibility of num_hashes by num_bands is assumed per the handout.
    """
    rows = num_hashes // num_bands
    return rows, (1.0 / num_bands) ** (1.0 / rows)


def jaccard_similarity(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


class _UnionFind:
    def __init__(self, size: int):
        self._parent = list(range(size))

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        self._parent[self.find(a)] = self.find(b)


def _candidate_pairs(signatures: list[np.ndarray], num_buckets: int) -> set[tuple[int, int]]:
    """All pairs sharing at least one identical band."""
    per_doc_bands = [buckets(signature, num_buckets) for signature in signatures]
    candidates: set[tuple[int, int]] = set()
    for band in range(num_buckets):
        groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for doc_idx, bands in enumerate(per_doc_bands):
            groups[tuple(bands[band])].append(doc_idx)
        for members in groups.values():
            candidates.update(
                (members[i], members[j]) for i in range(len(members)) for j in range(i + 1, len(members))
            )
    return candidates


def _kept_indices(
    docs: list[str], num_hashes: int, num_buckets: int, ngrams: int, threshold: float
) -> set[int]:
    shingles = [_shingle_set(doc, ngrams) for doc in docs]
    signatures = [_signature_from_grams(num_hashes, _ngrams(doc, ngrams)) for doc in docs]

    components = _UnionFind(len(docs))
    for a, b in sorted(_candidate_pairs(signatures, num_buckets)):
        if jaccard_similarity(shingles[a], shingles[b]) >= threshold:
            components.union(a, b)

    clusters: dict[int, list[int]] = defaultdict(list)
    for doc_idx in range(len(docs)):
        clusters[components.find(doc_idx)].append(doc_idx)

    rng = random.Random(_RETENTION_SEED)
    return {rng.choice(members) for members in clusters.values()}


def deduplicate_documents_minhash(
    docs: list[str],
    num_hashes: int,
    num_buckets: int,
    ngrams: int = DEFAULT_NGRAMS,
    threshold: float | None = None,
) -> list[str]:
    """Union of the documents that survive LSH MinHash deduplication.

    Without an explicit threshold, the banding scheme's approximate threshold
    (see parameters) is used. Survivors keep their original relative order.
    """
    if threshold is None:
        _, threshold = parameters(num_hashes, num_buckets)
    kept = _kept_indices(docs, num_hashes, num_buckets, ngrams, threshold)
    return [doc for doc_idx, doc in enumerate(docs) if doc_idx in kept]


def minhash_deduplication(
    input_files: list[os.PathLike],
    output_directory: os.PathLike,
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
) -> None:
    """Write each surviving document, full original contents, under the same basename."""
    paths = [Path(p) for p in input_files]
    texts = []
    for path in paths:
        with xopen(path, "rt", encoding="utf-8") as f:
            texts.append(f.read())

    kept = _kept_indices(texts, num_hashes, num_bands, ngrams, jaccard_threshold)

    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    for doc_idx, path in enumerate(paths):
        if doc_idx in kept:
            with xopen(output_dir / path.name, "wt", encoding="utf-8") as f:
                f.write(texts[doc_idx])
