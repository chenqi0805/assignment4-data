"""The section-4 filtering pipeline (handout problem filter_data).

One Common Crawl WET record flows through ordered stages; each stage either
removes the document (recording which stage removed it) or keeps it, possibly
modified (PII masking rewrites addresses in place). Survivors are written back
as WET records in their original relative order, so the output file is the
input with low-quality documents dropped and addresses masked.

Stage funnel (the per-filter kept counts the handout asks the script to report):

  extract   decode the WET payload (WET is pre-extracted plaintext, so this is
            the decode half of the problem-2.2 extraction primitive)
  english   lid.176.bin P(English) >= 0.7 — the same criterion that produced
            the English-filtered corpus, re-applied per record. Mirrors
            ``cs336_data.wet_files.is_english``; not imported from there
            because ``wet_files`` transitively imports ``modal_utils``, which
            refuses to import until SUNET_ID is set (see that module).
  pii       mask emails/phones/IPs — modifies, never removes
  nsfw      Dolma NSFW classifier predicts non-nsfw
  toxic     Dolma hatespeech classifier predicts non-toxic
  gopher    every Gopher-Rules predicate passes (problem 2.6)
  quality   quality classifier (problem 2.7) assigns P(wiki) >= threshold
  exact     exact-document deduplication: drop documents seen more than once
  minhash   LSH near-duplicate deduplication: one survivor per near-dup cluster

Exact and MinHash deduplication are corpus-level, so they run once per input
file after the per-record stages. At full scale the handout's parallelism unit
is the WET file, and dedup runs over each file's survivors.

Accounting: every document ends in exactly one terminal event — removed at a
stage, kept (survives MinHash dedup), or a per-record error — so the funnel
rows sum to the input record count exactly.
"""

from __future__ import annotations

import gzip
import random
from collections import deque
from collections.abc import Iterator
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

from fastwarc.warc import ArchiveIterator, WarcRecordType
from warcio.warcwriter import WARCWriter

from cs336_data.classifiers import detect_nsfw, detect_toxicity, identify_language
from cs336_data.dedup import deduplicate_document_indices
from cs336_data.gopher import all_gopher_rules_pass
from cs336_data.minhash import deduplicate_document_indices as minhash_kept_indices
from cs336_data.pii import mask_emails, mask_ips, mask_phone_numbers
from cs336_data.quality import fetch_qualityScores_wiki

# Stages that can remove a document, in pipeline order, minus the two handled
# outside the ordered loop: "extract" (decode/emptiness, in the reader) and
# "record_error" (recovered per-record exceptions). "pii" can only raise, not
# remove, but errors are accounted with the same ordering.
REMOVAL_STAGES = ("english", "nsfw", "toxic", "gopher", "quality")
_ERROR_ACCOUNTED_STAGES = ("english", "pii", "nsfw", "toxic", "gopher", "quality")

ENGLISH_PROBABILITY_THRESHOLD = 0.7  # same criterion as cs336_data.wet_files.is_english
DEFAULT_QUALITY_THRESHOLD = 0.5  # P(wiki) floor; with two classes this equals "label is wiki"

# Records are filtered in chunks so model predictions parallelize across a
# process pool; the in-flight window bounds resident memory.
CHUNK_SIZE = 256

# WET payloads are UTF-8 with a sporadic byte-order mark.
_BOM = "﻿"


def is_english(text: str, *, threshold: float = ENGLISH_PROBABILITY_THRESHOLD) -> bool:
    """English per lid.176.bin with the corpus-building threshold (mirrors wet_files.is_english)."""
    language, probability = identify_language(text)
    return language == "en" and probability >= threshold


class StageError(Exception):
    """A per-record filter raised; ``stage`` names where, for funnel accounting."""

    def __init__(self, stage: str, error: Exception) -> None:
        super().__init__(f"{stage}: {error!r}")
        self.stage = stage
        self.error = error


def _run(stage: str, fn, *args):
    try:
        return fn(*args)
    except Exception as error:  # retagged, never swallowed
        raise StageError(stage, error) from error


@dataclass
class DocOutcome:
    """Result of running one document through the per-record filters."""

    uri: str
    date: str
    text: str  # masked text when the document survived (or was modified)
    removed_by: str | None  # a REMOVAL_STAGES member, "extract", "record_error", or None
    email_hits: int = 0
    phone_hits: int = 0
    ip_hits: int = 0
    text_before_mask: str | None = None  # set only when masking changed something
    error: str | None = None  # repr of the exception, for record_error outcomes
    error_stage: str | None = None  # stage the exception occurred at, for record_error outcomes


@dataclass
class FilterStats:
    """Funnel counts for one or more WET files.

    Kept fields are cumulative (documents that completed the stage); removed
    fields are exact per-stage event counts. Every document contributes to
    exactly one terminal event, so
    ``total == removed_extract + removed_english + ... + removed_minhash
    + record_errors + minhash_dedup``.
    """

    total: int = 0  # WET conversion records seen
    extracted: int = 0  # completed the decode step
    english: int = 0
    nsfw: int = 0
    toxic: int = 0
    gopher: int = 0
    quality: int = 0
    exact_dedup: int = 0  # set after the corpus-level stages, not per record
    minhash_dedup: int = 0  # == documents finally kept
    removed_extract: int = 0
    removed_english: int = 0
    removed_nsfw: int = 0
    removed_toxic: int = 0
    removed_gopher: int = 0
    removed_quality: int = 0
    removed_exact: int = 0
    removed_minhash: int = 0
    record_errors: int = 0
    errors_by_stage: dict[str, int] = field(default_factory=dict)
    email_docs: int = 0
    phone_docs: int = 0
    ip_docs: int = 0
    email_hits: int = 0
    phone_hits: int = 0
    ip_hits: int = 0
    chars_in: int = 0
    chars_out: int = 0  # set by filter_wet_file after dedup, not per record
    files: int = 0

    def merge_document(self, outcome: DocOutcome) -> None:
        self.total += 1
        if outcome.removed_by != "extract":  # every other outcome completed the decode step
            self.extracted += 1
        if outcome.email_hits:
            self.email_docs += 1
            self.email_hits += outcome.email_hits
        if outcome.phone_hits:
            self.phone_docs += 1
            self.phone_hits += outcome.phone_hits
        if outcome.ip_hits:
            self.ip_docs += 1
            self.ip_hits += outcome.ip_hits

        if outcome.removed_by is None:
            for stage in REMOVAL_STAGES:
                setattr(self, stage, getattr(self, stage) + 1)
            return
        if outcome.removed_by == "extract":
            self.removed_extract += 1
            return
        if outcome.removed_by == "record_error":
            self.record_errors += 1
            error_stage = outcome.error_stage or "unknown"
            self.errors_by_stage[error_stage] = self.errors_by_stage.get(error_stage, 0) + 1
            if error_stage in _ERROR_ACCOUNTED_STAGES:  # completed stages strictly before the failure
                for prior in _ERROR_ACCOUNTED_STAGES[: _ERROR_ACCOUNTED_STAGES.index(error_stage)]:
                    if prior in REMOVAL_STAGES:
                        setattr(self, prior, getattr(self, prior) + 1)
            return
        removed_field = f"removed_{outcome.removed_by}"
        setattr(self, removed_field, getattr(self, removed_field) + 1)
        for prior in REMOVAL_STAGES[: REMOVAL_STAGES.index(outcome.removed_by)]:
            setattr(self, prior, getattr(self, prior) + 1)

    def finalize_dedup(self, entered: int, exact_kept: int, minhash_kept: int) -> None:
        """Record the corpus-level dedup events for one file's survivors."""
        self.exact_dedup = exact_kept
        self.minhash_dedup = minhash_kept
        self.removed_exact = entered - exact_kept
        self.removed_minhash = exact_kept - minhash_kept

    def merge(self, other: FilterStats) -> FilterStats:
        for field_name in FilterStats.__dataclass_fields__:
            if field_name == "errors_by_stage":
                continue
            setattr(self, field_name, getattr(self, field_name) + getattr(other, field_name))
        for stage, count in other.errors_by_stage.items():
            self.errors_by_stage[stage] = self.errors_by_stage.get(stage, 0) + count
        return self


@dataclass
class OutcomeSampler:
    """Deterministic reservoir samples of kept/removed/masked documents.

    Fed every DocOutcome in pipeline order, this retains fixed-size random
    samples for the handout's inspect_filtered_data deliverables: examples
    that survived (with a quality judgment), examples a specific filter
    removed, and before/after pairs for PII-masked documents.
    """

    kept: int = 0
    removed: int = 0
    masked: int = 0
    seed: int = 336
    _seen_kept: int = field(default=0, init=False, repr=False)
    _seen_removed: int = field(default=0, init=False, repr=False)
    _seen_masked: int = field(default=0, init=False, repr=False)
    _kept_samples: list[DocOutcome] = field(default_factory=list, init=False, repr=False)
    _removed_samples: list[DocOutcome] = field(default_factory=list, init=False, repr=False)
    _masked_samples: list[DocOutcome] = field(default_factory=list, init=False, repr=False)
    _random: random.Random = field(default_factory=lambda: random.Random(), init=False, repr=False)

    def __post_init__(self) -> None:
        self._random = random.Random(self.seed)

    @property
    def kept_samples(self) -> list[DocOutcome]:
        return self._kept_samples

    @property
    def removed_samples(self) -> list[DocOutcome]:
        return self._removed_samples

    @property
    def masked_samples(self) -> list[DocOutcome]:
        return self._masked_samples

    def visit(self, outcome: DocOutcome) -> None:
        if outcome.text_before_mask is not None:  # modified docs, kept or removed
            self._reservoir(self._masked_samples, self.masked, "_seen_masked", outcome)
        if outcome.removed_by is not None:
            self._reservoir(self._removed_samples, self.removed, "_seen_removed", outcome)
        else:
            self._reservoir(self._kept_samples, self.kept, "_seen_kept", outcome)

    def _reservoir(self, samples: list[DocOutcome], capacity: int, seen_field: str, outcome: DocOutcome) -> None:
        """Algorithm R: the sample is uniform over everything visited so far."""
        if capacity <= 0:
            return
        seen = getattr(self, seen_field)
        setattr(self, seen_field, seen + 1)
        if len(samples) < capacity:
            samples.append(outcome)
        elif (swap := self._random.randint(0, seen)) < capacity:
            samples[swap] = outcome


def filter_document(uri: str, date: str, text: str, *, quality_threshold: float = DEFAULT_QUALITY_THRESHOLD) -> DocOutcome:
    """Run one document through the per-record filters.

    Returns the outcome with ``removed_by`` naming the first stage that
    removed the document, or ``None`` when it survived every stage. The
    surviving text is the PII-masked text. Exceptions propagate as
    ``StageError`` so the caller can account for the failing stage.
    """
    if not _run("english", is_english, text):
        return DocOutcome(uri=uri, date=date, text=text, removed_by="english")

    text_before_mask = text
    masked, email_hits = _run("pii", mask_emails, text)
    masked, phone_hits = _run("pii", mask_phone_numbers, masked)
    masked, ip_hits = _run("pii", mask_ips, masked)
    changed = (email_hits, phone_hits, ip_hits) != (0, 0, 0)

    def outcome(removed_by: str | None) -> DocOutcome:
        return DocOutcome(
            uri=uri,
            date=date,
            text=masked,
            removed_by=removed_by,
            email_hits=email_hits,
            phone_hits=phone_hits,
            ip_hits=ip_hits,
            text_before_mask=text_before_mask if changed else None,
        )

    if _run("nsfw", detect_nsfw, masked)[0] != "non-nsfw":
        return outcome("nsfw")
    if _run("toxic", detect_toxicity, masked)[0] != "non-toxic":
        return outcome("toxic")
    if not _run("gopher", all_gopher_rules_pass, masked):
        return outcome("gopher")
    _label, wiki_probability = _run("quality", fetch_qualityScores_wiki, uri, masked)
    if wiki_probability < quality_threshold:
        return outcome("quality")
    return outcome(None)


def _filter_chunk(chunk: list[tuple[str, str, str]], quality_threshold: float) -> list[DocOutcome]:
    """Pool worker: filter one chunk, converting per-record errors to outcomes."""
    outcomes = []
    for uri, date, text in chunk:
        if not text.strip():
            outcomes.append(DocOutcome(uri=uri, date=date, text=text, removed_by="extract"))
            continue
        try:
            outcomes.append(filter_document(uri, date, text, quality_threshold=quality_threshold))
        except StageError as error:  # one poison record must not kill a corpus run
            outcomes.append(
                DocOutcome(
                    uri=uri,
                    date=date,
                    text=text,
                    removed_by="record_error",
                    error=repr(error.error),
                    error_stage=error.stage,
                )
            )
    return outcomes


def read_wet_documents(path: Path, *, max_records: int | None = None) -> Iterator[tuple[str, str, str]]:
    """Yield ``(uri, date, text)`` for every conversion record of a WET file, in order.

    The WET payload is pre-extracted plaintext, so extraction here is the
    decode step: UTF-8 with malformed bytes replaced and byte-order marks
    stripped. Empty payloads are yielded too and removed at the extract
    stage, so the caller's record counts stay complete.
    """
    seen = 0
    with gzip.open(path, "rb") as stream:
        for record in ArchiveIterator(stream, record_types=WarcRecordType.conversion):
            uri = record.headers.get("WARC-Target-URI", "")
            date = record.headers.get("WARC-Date", "")
            text = record.reader.read().decode("utf-8", errors="replace").lstrip(_BOM)
            yield uri, date, text
            seen += 1
            if max_records is not None and seen >= max_records:
                return


def _chunks(iterable: Iterator[tuple[str, str, str]], size: int) -> Iterator[list[tuple[str, str, str]]]:
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def filter_wet_file(
    input_path: Path,
    output_path: Path | None,
    *,
    num_hashes: int = 100,
    num_bands: int = 10,
    ngrams: int = 5,
    jaccard_threshold: float = 0.8,
    quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
    workers: int = 1,
    max_records: int | None = None,
    sampler: OutcomeSampler | None = None,
) -> FilterStats:
    """Filter one WET file and optionally write the survivors as a WET file.

    Survivors are written in original record order with their PII-masked
    text. Deduplication (exact, then MinHash/LSH) runs over the per-record
    survivors. Each fastText model costs roughly its file size in RAM per
    worker process (~2.3 GB for the full set), so ``workers`` should stay
    small on memory-constrained machines; the default is sequential.
    """
    stats = FilterStats(files=1)

    def handle(outcomes: list[DocOutcome]) -> list[DocOutcome]:
        survivors = []
        for outcome in outcomes:
            stats.merge_document(outcome)
            if sampler is not None:
                sampler.visit(outcome)
            if outcome.removed_by is None:
                survivors.append(outcome)
        return survivors

    def chars_of(chunk: list[tuple[str, str, str]]) -> int:
        return sum(len(text) for _, _, text in chunk)

    chunk_iter = _chunks(read_wet_documents(input_path, max_records=max_records), CHUNK_SIZE)
    if workers <= 1:
        survivors: list[DocOutcome] = []
        for chunk in chunk_iter:
            stats.chars_in += chars_of(chunk)
            survivors.extend(handle(_filter_chunk(chunk, quality_threshold)))
    else:
        survivors = []
        pool = ProcessPoolExecutor(max_workers=workers)
        try:
            futures: deque[Future[list[DocOutcome]]] = deque()
            for chunk in chunk_iter:
                stats.chars_in += chars_of(chunk)
                futures.append(pool.submit(_filter_chunk, chunk, quality_threshold))
                if len(futures) > 2 * workers:  # bounded in-flight window
                    survivors.extend(handle(futures.popleft().result()))
            while futures:
                survivors.extend(handle(futures.popleft().result()))
        finally:
            pool.shutdown()

    # Corpus-level stages over this file's survivors, index-preserving so the
    # original record order (and provenance headers) survive the write.
    texts = [outcome.text for outcome in survivors]
    exact_kept = deduplicate_document_indices(texts)
    minhash_kept = minhash_kept_indices(
        [texts[i] for i in exact_kept], num_hashes, num_bands, ngrams=ngrams, threshold=jaccard_threshold
    )
    final = [survivors[exact_kept[i]] for i in sorted(minhash_kept)]
    stats.finalize_dedup(entered=len(survivors), exact_kept=len(exact_kept), minhash_kept=len(final))
    stats.chars_out = sum(len(outcome.text) for outcome in final)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(output_path, "wb") as stream:
            writer = WARCWriter(stream, gzip=False)
            for outcome in final:
                record = writer.create_warc_record(
                    outcome.uri,
                    "conversion",
                    payload=BytesIO(outcome.text.encode("utf-8")),
                    warc_headers_dict={"WARC-Date": outcome.date, "Content-Type": "text/plain"},
                )
                writer.write_record(record)

    return stats


def funnel_rows(stats: FilterStats) -> list[tuple[str, str, str]]:
    """The writeup's per-filter table: (stage, kept, removed/notes) rows."""
    masking = (
        f"{stats.email_docs} docs masked ({stats.email_hits} emails, "
        f"{stats.phone_hits} phones, {stats.ip_hits} IPs)"
    )
    error_note = ", ".join(f"{count} at {stage}" for stage, count in sorted(stats.errors_by_stage.items()))
    return [
        ("input WET conversion records", str(stats.total), ""),
        ("extract (decode plaintext)", str(stats.extracted), f"{stats.removed_extract} empty"),
        ("english (lid.176.bin P(en) >= 0.7)", str(stats.english), f"{stats.removed_english} removed"),
        ("pii masking", str(stats.english), masking),
        ("nsfw (dolma jigsaw)", str(stats.nsfw), f"{stats.removed_nsfw} removed"),
        ("toxic (dolma jigsaw)", str(stats.toxic), f"{stats.removed_toxic} removed"),
        ("gopher rules (2.6)", str(stats.gopher), f"{stats.removed_gopher} removed"),
        (f"quality (P(wiki) >= {DEFAULT_QUALITY_THRESHOLD})", str(stats.quality), f"{stats.removed_quality} removed"),
        ("exact document dedup", str(stats.exact_dedup), f"{stats.removed_exact} removed"),
        ("minhash lsh dedup (final kept)", str(stats.minhash_dedup), f"{stats.removed_minhash} removed"),
        ("record errors (dropped)", str(stats.record_errors), error_note),
        ("characters in -> out", str(stats.chars_out), f"{stats.chars_in} in"),
    ]


def format_funnel_table(stats: FilterStats) -> str:
    """Markdown funnel table over one or more FilterStats."""
    header = "| stage | kept | removed / notes |\n|---|---|---|\n"
    return header + "".join(f"| {stage} | {kept} | {notes} |\n" for stage, kept, notes in funnel_rows(stats))


def format_funnel_text(stats: FilterStats) -> str:
    """Plain-text funnel table for terminal runs."""
    rows = funnel_rows(stats)
    width = max(len(stage) for stage, _, _ in rows)
    lines = [f"{'stage'.ljust(width)}  {'kept':>10}  removed / notes"]
    for stage, kept, notes in rows:
        lines.append(f"{stage.ljust(width)}  {kept.rjust(10)}  {notes}")
    return "\n".join(lines)
