"""Tokenize filtered documents for LM training (handout problem tokenize_data).

GPT-2 via ``transformers.AutoTokenizer.from_pretrained("gpt2")``; one
``<|endoftext|>`` (the EOS token) appended after each document; ``np.uint16``
serialization written with ``ids_array.tofile`` — the handout's exact recipe
(the training config consumes GPT-2 tokens with vocab 50257, which fits
uint16).

Before writing anything, every training document is checked against the
Paloma C4-100 validation set (handout section 5.4 contamination rule): the
validation token stream is decoded back to text and any training document
whose normalized text appears there is a hard error that blocks the output —
validation data may inform filter design but must never end up in the
training data.

Local:

    uv run python scripts/tokenize_data.py \
        --input data/filtered --output data/tokens/train.bin

Modal (single container; inputs are the filtered WETs on the data volume):

    uv run modal run scripts/tokenize_data.py --modal \
        --input /root/data/filtered --output /root/data/tokens/train.bin
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from cs336_data.common import get_shared_assets_path
from cs336_data.pipeline import read_wet_documents

_PALOMA_FILENAME = "tokenized_paloma_c4_100_domains_validation.bin"
_DEFAULT_CONFIG = Path("configs/pipeline.yaml")


def _load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    import yaml

    return yaml.safe_load(path.read_text()) or {}


def _paloma_docs(paloma_bin: Path, tokenizer) -> set[str]:
    """Normalized texts of the Paloma C4-100 validation set.

    The file is the uint16 GPT-2 token stream of the validation corpus with
    one EOS after each document, so segments between EOS tokens decode back
    to the validation documents.
    """
    tokens = np.fromfile(paloma_bin, dtype=np.uint16)
    eos_id = tokenizer.eos_token_id
    docs: set[str] = set()
    start = 0
    for end in np.where(tokens == eos_id)[0]:
        if end > start:
            text = tokenizer.decode(tokens[start:end])
            docs.add(" ".join(text.lower().split()))
        start = end + 1
    return docs


def _normalized(text: str) -> str:
    return " ".join(text.lower().split())


def _iter_documents(path: Path) -> Iterator[str]:
    """Documents from filtered WET files (``*.gz``) or plain-text line files."""
    files = sorted(p for p in path.iterdir() if p.name.endswith(".gz")) if path.is_dir() else [path]
    if not files:
        raise SystemExit(f"no input files under {path}")
    for file_path in files:
        if file_path.name.endswith(".gz"):
            for _uri, _date, text in read_wet_documents(file_path):
                yield text
        else:
            for line in file_path.open(encoding="utf-8"):
                if line.strip():
                    yield line.rstrip("\n")


def tokenize_documents(docs: Iterator[str], tokenizer, eos_id: int) -> Iterator[np.ndarray]:
    """One uint16 array per document, EOS-terminated (handout p.14-15 recipe)."""
    for doc in docs:
        ids = tokenizer.encode(doc)
        ids.append(eos_id)
        yield np.asarray(ids, dtype=np.uint16)


def run(input_path: Path, output_path: Path, paloma_bin: Path | None, tokenizer_name: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    eos_id = tokenizer.eos_token_id

    if paloma_bin is not None:
        if not paloma_bin.exists():
            raise SystemExit(
                f"Paloma validation tokens not found at {paloma_bin} — run "
                "`uv run python scripts/download_data.py --offline-only` first. The contamination "
                "guard (handout 5.4) requires them; pass --paloma-bin to point at another copy."
            )
        paloma = _paloma_docs(paloma_bin, tokenizer)
        overlaps = []
        checked = 0
        for doc in _iter_documents(input_path):
            checked += 1
            if _normalized(doc) in paloma:
                overlaps.append(doc)
        if overlaps:
            raise SystemExit(
                f"CONTAMINATION GUARD: {len(overlaps)}/{checked} training documents appear verbatim in "
                "the Paloma C4-100 validation set (handout 5.4). Validation data must never be "
                "trained on — remove these documents upstream and re-run. First offenders: "
                + "; ".join(f"{doc[:80]}..." for doc in overlaps[:3])
            )
        print(f"[tokenize] contamination check: {checked:,} documents, {len(overlaps)} Paloma overlaps")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_docs = 0
    total_tokens = 0
    started = time.monotonic()
    with output_path.open("wb") as out:
        # Streaming tofile: identical bytes to a single ids_array.tofile, so
        # corpora larger than RAM still serialize as one uint16 token file.
        for ids_array in tokenize_documents(_iter_documents(input_path), tokenizer, eos_id):
            ids_array.tofile(out)
            total_docs += 1
            total_tokens += len(ids_array)
    elapsed = time.monotonic() - started
    print(
        f"[tokenize] {total_docs:,} documents -> {total_tokens:,} tokens "
        f"({elapsed:.1f}s) -> {output_path}"
    )


def _parse_args() -> argparse.Namespace:
    config = _load_config(_DEFAULT_CONFIG)
    defaults = config.get("tokenize_data", {})
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--modal", action="store_true", help="run against Modal volume paths")
    parser.add_argument("--input", required=True, help="filtered WET directory/file (or plain-text line file)")
    parser.add_argument("--output", required=True, help="output uint16 token file")
    parser.add_argument(
        "--paloma-bin",
        default=str(get_shared_assets_path() / _PALOMA_FILENAME),
        help=f"Paloma C4-100 validation tokens (default: get_shared_assets_path()/{_PALOMA_FILENAME})",
    )
    parser.add_argument("--no-contamination-check", action="store_true", help="skip the Paloma guard (not advised)")
    parser.add_argument("--tokenizer", default=defaults.get("tokenizer", "gpt2"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paloma_bin = Path(args.paloma_bin) if args.paloma_bin and not args.no_contamination_check else None
    run(Path(args.input), Path(args.output), paloma_bin, args.tokenizer)


if __name__ == "__main__":
    main()
