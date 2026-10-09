"""Filter Common Crawl WET files into LM-ready documents (handout problem filter_data).

Each input WET file flows through the ordered filters in
``cs336_data.pipeline`` (extract -> english -> PII masking -> NSFW -> toxic ->
Gopher -> quality classifier -> exact + MinHash dedup). Survivors keep their
original record order and are written as WET records with the input's
basename. The script reports the number of documents kept by each filter —
locally as a printed funnel table plus a JSON sidecar, and on Modal as an
aggregated table over all mapped files.

Local mode (no Modal import anywhere on this path):

    uv run python scripts/filter_data.py \
        --input local-shared-data/CC/example.warc.wet.gz \
        --output-dir data/filtered --samples 5

Modal mode (handout section 4.2: parallel across WET files):

    uv run modal run scripts/filter_data.py --modal \
        --input-dir /shared-data/english-wet-data \
        --output-dir /root/data/filtered --n-files 2500
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

from cs336_data.pipeline import (
    DEFAULT_QUALITY_THRESHOLD,
    FilterStats,
    OutcomeSampler,
    filter_wet_file,
    format_funnel_text,
)

_DEFAULT_CONFIG = Path("configs/pipeline.yaml")
_SAMPLE_EXCERPT_CHARS = 1200


def _load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    import yaml

    return yaml.safe_load(path.read_text()) or {}


def _input_files(paths: list[Path], n_files: int | None = None) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(p for p in path.iterdir() if p.name.endswith(".gz")))
        else:
            files.append(path)
    if n_files is not None:
        files = files[:n_files]
    if not files:
        raise SystemExit("no input WET files found")
    return files


def _excerpt(text: str, limit: int) -> str:
    collapsed = " ".join(text.split())
    return collapsed[:limit] + ("..." if len(collapsed) > limit else "")


def _sample_records(sampler: OutcomeSampler, limit: int) -> dict:
    return {
        "kept": [
            {"uri": o.uri, "removed_by": None, "chars": len(o.text), "excerpt": _excerpt(o.text, limit)}
            for o in sampler.kept_samples
        ],
        "removed": [
            {"uri": o.uri, "removed_by": o.removed_by, "chars": len(o.text), "excerpt": _excerpt(o.text, limit)}
            for o in sampler.removed_samples
        ],
        "masked": [
            {
                "uri": o.uri,
                "email_hits": o.email_hits,
                "phone_hits": o.phone_hits,
                "ip_hits": o.ip_hits,
                "before": _excerpt(o.text_before_mask or "", limit),
                "after": _excerpt(o.text, limit),
            }
            for o in sampler.masked_samples
        ],
    }


def _write_stats(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def run_local(args: argparse.Namespace, params: dict) -> None:
    """Sequential/pooled local run over the given WET files."""
    files = _input_files([Path(p) for p in args.input])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sampler = (
        OutcomeSampler(kept=args.samples, removed=args.samples, masked=args.samples) if args.samples else None
    )
    aggregate = FilterStats()
    per_file: dict[str, dict] = {}

    for input_path in files:
        output_path = output_dir / input_path.name
        started = time.monotonic()
        stats = filter_wet_file(
            input_path,
            output_path,
            sampler=sampler,
            max_records=args.max_records,
            **params,
        )
        elapsed = time.monotonic() - started
        aggregate.merge(stats)
        per_file[input_path.name] = dataclasses.asdict(stats)
        print(
            f"[filter] {input_path.name}: {stats.total} records -> {stats.minhash_dedup} kept "
            f"({elapsed:.1f}s, {stats.chars_out:,} chars out)"
        )

    print("\n" + format_funnel_text(aggregate))

    sidecar = {
        "params": params,
        "files": [p.name for p in files],
        "aggregate": dataclasses.asdict(aggregate),
        "per_file": per_file,
        "samples": _sample_records(sampler, _SAMPLE_EXCERPT_CHARS) if sampler is not None else None,
    }
    sidecar_path = output_dir / "filter_stats.json"
    _write_stats(sidecar_path, sidecar)
    print(f"\n[filter] stats sidecar: {sidecar_path}")


def run_modal(args: argparse.Namespace, params: dict) -> None:
    """Fan filter_wet_file out across Modal containers, one per WET file."""
    from cs336_data.modal_utils import VOLUME_MOUNTS, app, build_image

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    @app.function(
        image=build_image(),
        volumes=VOLUME_MOUNTS,
        timeout=2 * 60 * 60,
        max_containers=128,
    )
    def filter_one(remote_input: str) -> dict:
        stats = filter_wet_file(Path(remote_input), output_dir / Path(remote_input).name, **params)
        return dataclasses.asdict(stats)

    files = _input_files([input_dir], n_files=args.n_files)
    with app.run():
        stats_dicts = list(filter_one.map(str(p) for p in files))

    aggregate = FilterStats()
    per_file: dict[str, dict] = {}
    for path, stats_dict in zip(files, stats_dicts, strict=True):
        aggregate.merge(FilterStats(**stats_dict))
        per_file[path.name] = stats_dict
    print("\n" + format_funnel_text(aggregate))
    sidecar_path = Path("data") / "filter_stats.modal.json"
    _write_stats(
        sidecar_path,
        {
            "params": params,
            "files": [p.name for p in files],
            "aggregate": dataclasses.asdict(aggregate),
            "per_file": per_file,
        },
    )
    print(f"\n[filter] stats sidecar: {sidecar_path}")


def _parse_args() -> argparse.Namespace:
    config = _load_config(_DEFAULT_CONFIG)
    defaults = config.get("filter_data", {})
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--modal", action="store_true", help="run on Modal (handout 4.2) instead of locally")
    parser.add_argument("--input", action="append", help="input WET file or directory (repeatable; local mode)")
    parser.add_argument("--input-dir", help="directory of WET files on the Modal volume (modal mode)")
    parser.add_argument("--output-dir", required=True, help="directory for filtered WET files")
    parser.add_argument("--n-files", type=int, default=None, help="cap the number of input files")
    parser.add_argument("--max-records", type=int, default=None, help="cap records per file (smoke tests)")
    parser.add_argument(
        "--samples", type=int, default=0, help="reservoir-sample this many kept/removed/masked docs (local mode)"
    )
    parser.add_argument("--num-hashes", type=int, default=defaults.get("num_hashes", 100))
    parser.add_argument("--num-bands", type=int, default=defaults.get("num_bands", 10))
    parser.add_argument("--ngrams", type=int, default=defaults.get("ngrams", 5))
    parser.add_argument("--jaccard-threshold", type=float, default=defaults.get("jaccard_threshold", 0.8))
    parser.add_argument(
        "--quality-threshold", type=float, default=defaults.get("quality_threshold", DEFAULT_QUALITY_THRESHOLD)
    )
    parser.add_argument("--workers", type=int, default=defaults.get("workers", 1))
    args = parser.parse_args()
    if args.modal and not args.input_dir:
        parser.error("--modal requires --input-dir")
    if not args.modal and not args.input:
        parser.error("local mode requires --input")
    return args


def main() -> None:
    args = _parse_args()
    params = {
        "num_hashes": args.num_hashes,
        "num_bands": args.num_bands,
        "ngrams": args.ngrams,
        "jaccard_threshold": args.jaccard_threshold,
        "quality_threshold": args.quality_threshold,
        "workers": args.workers,
    }
    if args.modal:
        run_modal(args, params)
    else:
        run_local(args, params)


if __name__ == "__main__":
    main()
