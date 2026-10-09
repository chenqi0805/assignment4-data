"""Train the quality classifier of handout problem 2.7 on bounded, locally reproducible data.

The handout recipe: positive examples are pages at URLs linked from English Wikipedia
(the ``extracted_urls`` list the full pipeline builds from every multistream dump shard);
negative examples are random Common Crawl pages. Both are impractical to produce locally
in full (the URL list needs every dump shard; Common Crawl is petabytes), so this script
implements the same recipe over a bounded sample:

  urls      fetch a SUBSET of the enwiki multistream dump and extract external URLs
            from ``<ref>`` bodies (the same regex the Modal path uses in download_data.py)
  scrape    fetch the subsampled URLs and extract their text (positives)
  negatives extract text from one Common Crawl example WARC plus small fixture docs
  train     dedupe + split both classes, train fastText supervised, save
            quality_classifier.bin into the shared classifiers directory, and report
            held-out accuracy plus the assignment's two fixture sanity checks

All intermediate data lives under ``get_shared_assets_path() / "quality_data"`` and the
model under ``.../classifiers/quality_classifier.bin``. Both are gitignored (repo policy:
model binaries are never committed); this script is the committed, reproducible artifact.

Note: language filtering uses ``cs336_data.classifiers.identify_language`` directly with
the same English criterion as ``cs336_data.wet_files.is_english`` (lid.176.bin, prob
>= 0.7) rather than importing ``wet_files``, which pulls in the Modal ``modal_utils``
import guard that requires SUNET_ID to be set.
"""

from __future__ import annotations

import argparse
import bz2
import gzip
import http.client
import json
import random
import re
import shutil
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import fasttext
from warcio.archiveiterator import ArchiveIterator

from cs336_data.classifiers import identify_language
from cs336_data.common import get_shared_assets_path
from cs336_data.extraction import extract_text_from_html_bytes

SEED = 336
ENGLISH_PROBABILITY_THRESHOLD = 0.7  # same criterion as wet_files.is_english
MIN_TEXT_CHARS = 300
MAX_TEXT_CHARS = 20_000  # training docs are truncated; predict() sees full text
WIKI_LABEL = "wiki"
CC_LABEL = "cc"

# Same URL/ref regexes as scripts/download_data.py (staff-provided, known to match the dump).
URL_RE = r"\b(?:https?|telnet|gopher|file|wais|ftp):[\w/#~:.?+=&%@!\-.:?\\-]+?(?=[.:?\-]*(?:[^\w/#~:.?+=&%@!\-.:?\-]|$))"
REF_RE = r"&lt;ref&gt(.*)&lt;/ref&gt;"

USER_AGENT = "Mozilla/5.0 (compatible; cs336-a4-quality-corpus)"


@dataclass
class Document:
    url: str
    text: str
    source: str  # "wiki" or "cc"


def quality_data_path() -> Path:
    path = get_shared_assets_path() / "quality_data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def quality_model_path() -> Path:
    path = get_shared_assets_path() / "classifiers"
    path.mkdir(parents=True, exist_ok=True)
    return path / "quality_classifier.bin"


def is_english(text: str) -> bool:
    language, probability = identify_language(text)
    return language == "en" and probability >= ENGLISH_PROBABILITY_THRESHOLD


def _write_documents(path: Path, docs: list[Document]) -> None:
    with open(path, "w") as f:
        f.writelines(json.dumps({"url": doc.url, "text": doc.text, "source": doc.source}) + "\n" for doc in docs)


def _read_documents(path: Path) -> list[Document]:
    with open(path) as f:
        return [Document(url=d["url"], text=d["text"], source=d["source"]) for d in map(json.loads, f)]


def extract_wiki_urls(shard_path: Path) -> list[str]:
    """External URLs from ``<ref>`` bodies of one multistream dump shard, deduped in file order."""
    url_re = re.compile(URL_RE)
    ref_re = re.compile(REF_RE)
    urls: list[str] = []
    seen: set[str] = set()
    with bz2.open(shard_path, "rt", errors="ignore") as f:
        for line in f:
            if ref := ref_re.search(line):
                for url in url_re.findall(ref.group(0)):
                    if url not in seen:
                        seen.add(url)
                        urls.append(url)
    return urls


def _fetch(url: str, timeout: float) -> bytes | None:
    """Download the body of `url`, returning None for non-HTML or failed responses."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get("Content-Type", "")
            if "text/html" not in content_type and "text/plain" not in content_type:
                return None
            return response.read(3_000_000)
    except (OSError, ValueError, http.client.HTTPException):
        return None


def _download(url: str, path: Path) -> None:
    """urlretrieve with the script's User-Agent — Wikimedia rejects urllib's default UA (HTTP 403)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response, open(path, "wb") as f:
        shutil.copyfileobj(response, f)


def stage_urls(args: argparse.Namespace) -> Path:
    """Extract external URLs from a bounded subset of the enwiki multistream dump."""
    shard_path = Path(args.shard_path)
    if not shard_path.exists():
        print(f"[urls] downloading {args.shard_url}", flush=True)
        _download(args.shard_url, shard_path)

    urls = extract_wiki_urls(shard_path)
    random.Random(SEED).shuffle(urls)
    sampled = urls[: args.max_urls]
    out = quality_data_path() / "positive_urls.txt"
    out.write_text("\n".join(sampled) + "\n")
    print(f"[urls] {len(urls)} unique refs in shard; wrote {len(sampled)} to {out}", flush=True)
    return out


def stage_scrape(args: argparse.Namespace) -> Path:
    """Scrape the subsampled URL list; keep extracted, English pages as positive examples."""
    urls = [u for u in (quality_data_path() / "positive_urls.txt").read_text().splitlines() if u]
    print(f"[scrape] fetching {len(urls)} urls with {args.workers} workers", flush=True)

    # Network calls run in a thread pool; extraction and langid stay on the main thread
    # because the fastText predict path is not documented as thread-safe.
    bodies: list[tuple[str, bytes | None]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_fetch, url, args.timeout): url for url in urls}
        for i, future in enumerate(as_completed(futures), 1):
            bodies.append((futures[future], future.result()))
            if i % 1000 == 0:
                print(f"[scrape] fetched {i}/{len(urls)}", flush=True)

    docs: list[Document] = []
    skipped = {"fetch_failed_or_non_html": 0, "no_text": 0, "too_short": 0, "not_english": 0, "duplicate": 0, "extract_error": 0}
    seen: set[str] = set()
    for url, body in bodies:
        # Scraping arbitrary web content: one bad page must not kill the corpus run.
        # Failures are counted, not silenced.
        text: str | None = None
        extract_failed = False
        try:
            text = extract_text_from_html_bytes(body) if body else None
        except Exception:  # noqa: BLE001
            extract_failed = True
            skipped["extract_error"] += 1
        if text is None or not text.strip():
            if extract_failed:
                pass  # already counted as extract_error
            elif body is None:
                skipped["fetch_failed_or_non_html"] += 1
            else:
                skipped["no_text"] += 1
            continue
        if len(text) < MIN_TEXT_CHARS:
            skipped["too_short"] += 1
            continue
        if not is_english(text):
            skipped["not_english"] += 1
            continue
        key = " ".join(text.split())[:1000]
        if key in seen:
            skipped["duplicate"] += 1
            continue
        seen.add(key)
        docs.append(Document(url=url, text=text[:MAX_TEXT_CHARS], source=WIKI_LABEL))

    out = quality_data_path() / "positives.jsonl"
    _write_documents(out, docs)
    print(f"[scrape] kept {len(docs)} positives; skipped {skipped} -> {out}", flush=True)
    return out


def _iter_warc_texts(warc_path: Path, limit: int) -> list[tuple[str, str]]:
    """(url, text) pairs from HTTP 200 HTML responses of a Common Crawl WARC."""
    pairs: list[tuple[str, str]] = []
    extract_errors = 0
    with gzip.open(warc_path, "rb") as stream:
        for rec in ArchiveIterator(stream):
            if rec.rec_type != "response" or rec.http_headers is None:
                continue
            # get_statuscode() returns the code as a string ("200"), so compare as int.
            if int(rec.http_headers.get_statuscode() or 0) != 200:
                continue
            content_type = rec.http_headers.get_header("Content-Type") or ""
            if "text/html" not in content_type:
                continue
            # One malformed record must not kill the corpus run; count and move on.
            text: str | None = None
            try:
                text = extract_text_from_html_bytes(rec.content_stream().read())
            except Exception:  # noqa: BLE001
                extract_errors += 1
            if text and len(text.strip()) > 0:
                pairs.append((rec.rec_headers.get_header("WARC-Target-URI") or "", text))
            if len(pairs) >= limit:
                break
    if extract_errors:
        print(f"[_iter_warc_texts] {extract_errors} records failed extraction", flush=True)
    return pairs


def stage_negatives(args: argparse.Namespace) -> Path:
    """Extract negative examples from one Common Crawl example WARC plus small fixture docs."""
    random_gen = random.Random(SEED)
    pairs = _iter_warc_texts(Path(args.warc_path), limit=args.warc_scan_limit)
    random_gen.shuffle(pairs)

    docs: list[Document] = []
    skipped = {"no_text": 0, "too_short": 0, "not_english": 0, "duplicate": 0, "extract_error": 0}
    seen: set[str] = set()
    for url, text in pairs:
        if len(docs) >= args.num_negatives:
            break
        if len(text) < MIN_TEXT_CHARS:
            skipped["too_short"] += 1
            continue
        if not is_english(text):
            skipped["not_english"] += 1
            continue
        key = " ".join(text.split())[:1000]
        if key in seen:
            skipped["duplicate"] += 1
            continue
        seen.add(key)
        docs.append(Document(url=url, text=text[:MAX_TEXT_CHARS], source=CC_LABEL))

    # Small in-repo fixture docs as extra negatives (the forum-boilerplate domain of low_quality_cc.txt).
    fixtures_root = Path(args.fixtures_path)
    fixture_paths = [
        fixtures_root / "low_quality_cc.txt",
        fixtures_root / "moby_extracted.txt",
        *(fixtures_root / "documents_with_line_duplicates").glob("*.txt"),
    ]
    for fixture_path in sorted(fixture_paths, key=str):
        text = fixture_path.read_text()
        if len(text) >= MIN_TEXT_CHARS:
            docs.append(Document(url=f"fixture:{fixture_path.name}", text=text[:MAX_TEXT_CHARS], source=CC_LABEL))

    out = quality_data_path() / "negatives.jsonl"
    _write_documents(out, docs)
    print(f"[negatives] kept {len(docs)} negatives; skipped {skipped} -> {out}", flush=True)
    return out


def stage_train(args: argparse.Namespace) -> Path:
    """Train fastText on positives+negatives, save the model, and report evaluation."""
    positives = _read_documents(quality_data_path() / "positives.jsonl")
    negatives = _read_documents(quality_data_path() / "negatives.jsonl")
    random_gen = random.Random(SEED)

    corpus = [(doc, WIKI_LABEL) for doc in positives] + [(doc, CC_LABEL) for doc in negatives]
    random_gen.shuffle(corpus)
    split = int(len(corpus) * (1 - args.valid_fraction))
    train_corpus, valid_corpus = corpus[:split], corpus[split:]
    print(f"[train] corpus: {len(positives)} wiki / {len(negatives)} cc; train {len(train_corpus)}, valid {len(valid_corpus)}", flush=True)

    def write_split(path: Path, docs: list[tuple[Document, str]]) -> None:
        with open(path, "w") as f:
            for doc, label in docs:
                text = " ".join(doc.text.split())
                f.write(f"__label__{label} {text}\n")

    data_dir = quality_data_path()
    train_path, valid_path = data_dir / "train.txt", data_dir / "valid.txt"
    write_split(train_path, train_corpus)
    write_split(valid_path, valid_corpus)

    model = fasttext.train_supervised(
        input=str(train_path),
        epoch=args.epochs,
        lr=args.lr,
        dim=args.dim,
        wordNgrams=args.word_ngrams,
        minCount=args.min_count,
        bucket=args.bucket,
        seed=SEED,
        verbose=2,
    )

    n, precision, _recall = model.test(str(valid_path))
    print(f"[train] held-out: n={n} accuracy={precision:.4f}", flush=True)

    predictions: dict[tuple[str, str], int] = {}
    for doc, true_label in valid_corpus:
        text = " ".join(doc.text.split())
        predicted_label = model.predict(text)[0][0].removeprefix("__label__")
        predictions[(true_label, predicted_label)] = predictions.get((true_label, predicted_label), 0) + 1
    print(f"[train] confusion (true, predicted): {dict(sorted(predictions.items()))}", flush=True)

    model_path = quality_model_path()
    model.save_model(str(model_path))
    print(f"[train] saved {model_path} ({model_path.stat().st_size / 1e6:.1f} MB)", flush=True)

    # The assignment's sanity checks: the two fixture documents must classify correctly.
    fixtures_path = Path(args.fixtures_path)
    for name, expected in [("low_quality_cc.txt", CC_LABEL), ("high_quality_wiki_reference.txt", WIKI_LABEL)]:
        text = " ".join((fixtures_path / name).read_text().split())
        labels, probs = model.predict(text)
        label = labels[0].removeprefix("__label__")
        mark = "PASS" if label == expected else "FAIL"
        print(f"[gate] {mark}: {name} -> {label} ({probs[0]:.4f}), expected {expected}", flush=True)

    return model_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="stage", required=True)

    default_shard = "enwiki-20260501-pages-articles-multistream1.xml-p1p41242.bz2"
    p_urls = subparsers.add_parser("urls", help="extract + subsample external URLs from a wiki dump shard")
    p_urls.add_argument("--shard-url", default=f"https://dumps.wikimedia.org/enwiki/20260501/{default_shard}")
    p_urls.add_argument("--shard-path", default=str(get_shared_assets_path() / "wiki_shards" / default_shard))
    p_urls.add_argument("--max-urls", type=int, default=12_000)

    p_scrape = subparsers.add_parser("scrape", help="scrape the subsampled URLs for positive pages")
    p_scrape.add_argument("--workers", type=int, default=32)
    p_scrape.add_argument("--timeout", type=float, default=8.0)

    p_neg = subparsers.add_parser("negatives", help="extract negatives from the example WARC + fixtures")
    p_neg.add_argument("--warc-path", default=str(get_shared_assets_path() / "CC" / "example.warc.gz"))
    p_neg.add_argument("--warc-scan-limit", type=int, default=200_000, help="max WARC records to scan")
    p_neg.add_argument("--num-negatives", type=int, default=12_000)
    p_neg.add_argument("--fixtures-path", default="tests/fixtures")

    p_train = subparsers.add_parser("train", help="train fastText and save quality_classifier.bin")
    p_train.add_argument("--epochs", type=int, default=15)
    p_train.add_argument("--lr", type=float, default=0.5)
    p_train.add_argument("--dim", type=int, default=50)
    p_train.add_argument("--word-ngrams", type=int, default=2)
    p_train.add_argument("--min-count", type=int, default=2)
    p_train.add_argument("--bucket", type=int, default=500_000)
    p_train.add_argument("--valid-fraction", type=float, default=0.05)
    p_train.add_argument("--fixtures-path", default="tests/fixtures")

    args = parser.parse_args()
    if args.stage == "urls":
        stage_urls(args)
    elif args.stage == "scrape":
        stage_scrape(args)
    elif args.stage == "negatives":
        stage_negatives(args)
    elif args.stage == "train":
        stage_train(args)
    else:  # unreachable with required=True
        parser.error(f"unknown stage {args.stage}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
