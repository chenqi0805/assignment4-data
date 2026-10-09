"""Inspect filtered data (handout problem inspect_filtered_data).

Produces the writeup's evidence: the per-filter keep-count funnel over N input
WET files, random kept documents with quality-judgment slots, documents each
filter removed with the removing stage named, and before/after pairs for
PII-masked documents. The report goes to a markdown file and the funnel table
to stdout.

Three modes:

  aggregate sidecars written by filter_data.py (no recompute; local mode only):

    uv run python scripts/inspect.py --stats-dir data/filtered --output-report writeup/inspect_report.md

  re-run the pipeline over input WET files, collecting fresh samples:

    uv run python scripts/inspect.py --inputs local-shared-data/CC/example.warc.wet.gz \
        --output-report writeup/inspect_report.md --samples 5

  Modal (re-runs the pipeline across containers, counts only):

    uv run modal run scripts/inspect.py --modal \
        --input-dir /shared-data/english-wet-data --n-files 10 \
        --output-report /root/data/inspect_report.md
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from cs336_data.pipeline import (
    DEFAULT_QUALITY_THRESHOLD,
    FilterStats,
    OutcomeSampler,
    filter_wet_file,
    format_funnel_table,
    format_funnel_text,
)

_DEFAULT_CONFIG = Path("configs/pipeline.yaml")
_EXCERPT_CHARS = 1200


def _load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    import yaml

    return yaml.safe_load(path.read_text()) or {}


def _excerpt(text: str, limit: int = _EXCERPT_CHARS) -> str:
    collapsed = " ".join(text.split())
    return collapsed[:limit] + ("..." if len(collapsed) > limit else "")


def _collect_local_stats_files(stats_dir: Path) -> list[Path]:
    sidecars = sorted(stats_dir.rglob("filter_stats.json"))
    if not sidecars:
        raise SystemExit(f"no filter_stats.json sidecars under {stats_dir}")
    return sidecars


def _aggregate_sidecars(sidecars: list[Path]) -> tuple[FilterStats, dict]:
    aggregate = FilterStats()
    merged: dict = {"kept": [], "removed": [], "masked": []}
    for sidecar_path in sidecars:
        payload = json.loads(sidecar_path.read_text())
        aggregate.merge(FilterStats(**payload["aggregate"]))
        samples = payload.get("samples") or {}
        for bucket in ("kept", "removed", "masked"):
            merged[bucket].extend(samples.get(bucket, []))
    return aggregate, merged


def _render_report(
    title: str,
    aggregate: FilterStats,
    samples: dict | None,
    elapsed_seconds: float | None,
) -> str:
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    runtime = f" Pipeline runtime: {elapsed_seconds:.1f}s." if elapsed_seconds is not None else ""
    lines = [
        f"# Inspect report — {title}",
        "",
        f"_Generated {now} by `scripts/inspect.py`._{runtime}",
        "",
        "## Per-filter keep counts",
        "",
        format_funnel_table(aggregate),
    ]

    if samples:
        lines += ["## Kept samples (quality judgment slots for the writeup)", ""]
        for i, sample in enumerate(samples.get("kept", []), 1):
            lines += [
                f"### Kept {i}: {sample['uri']} ({sample['chars']:,} chars)",
                "",
                "```text",
                sample["excerpt"],
                "```",
                "",
            ]
        lines += ["## Removed samples (by filter)", ""]
        for i, sample in enumerate(samples.get("removed", []), 1):
            lines += [
                f"### Removed {i}: {sample['uri']} — removed by `{sample['removed_by']}`",
                "",
                "```text",
                sample["excerpt"],
                "```",
                "",
            ]
        lines += ["## PII-masked samples (before / after)", ""]
        for i, sample in enumerate(samples.get("masked", []), 1):
            hits = f"{sample['email_hits']} emails, {sample['phone_hits']} phones, {sample['ip_hits']} IPs"
            lines += [
                f"### Masked {i}: {sample['uri']} ({hits})",
                "",
                "**Before:**",
                "",
                "```text",
                sample["before"],
                "```",
                "",
                "**After:**",
                "",
                "```text",
                sample["after"],
                "```",
                "",
            ]
    return "\n".join(lines)


def _input_files(paths: list[Path], n_files: int | None) -> list[Path]:
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


def run_local(args: argparse.Namespace, params: dict) -> None:
    if args.stats_dir:
        sidecars = _collect_local_stats_files(Path(args.stats_dir))
        aggregate, samples = _aggregate_sidecars(sidecars)
        report = _render_report(
            f"{len(sidecars)} filter_stats.json sidecar(s) under {args.stats_dir}", aggregate, samples, None
        )
    else:
        files = _input_files([Path(p) for p in args.inputs], args.n_files)
        sampler = OutcomeSampler(kept=args.samples, removed=args.samples, masked=args.samples, seed=args.seed)
        aggregate = FilterStats(files=len(files))
        started = time.monotonic()
        for input_path in files:
            # Counts only: no filtered output is written by the inspect run.
            stats = filter_wet_file(
                input_path,
                None,
                sampler=sampler,
                max_records=args.max_records,
                **params,
            )
            aggregate.merge(stats)
        elapsed = time.monotonic() - started
        samples = {
            "kept": [
                {"uri": o.uri, "chars": len(o.text), "excerpt": _excerpt(o.text)} for o in sampler.kept_samples
            ],
            "removed": [
                {"uri": o.uri, "removed_by": o.removed_by, "chars": len(o.text), "excerpt": _excerpt(o.text)}
                for o in sampler.removed_samples
            ],
            "masked": [
                {
                    "uri": o.uri,
                    "email_hits": o.email_hits,
                    "phone_hits": o.phone_hits,
                    "ip_hits": o.ip_hits,
                    "before": _excerpt(o.text_before_mask or ""),
                    "after": _excerpt(o.text),
                }
                for o in sampler.masked_samples
            ],
        }
        report = _render_report(
            f"{len(files)} input file(s): {', '.join(p.name for p in files)}",
            aggregate,
            samples,
            elapsed,
        )

    print("\n" + format_funnel_text(aggregate))
    report_path = Path(args.output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report + "\n")
    print(f"\n[inspect] report: {report_path}")


def run_modal(args: argparse.Namespace, params: dict) -> None:
    """Aggregate per-filter counts across Modal containers; samples stay local-side."""
    from cs336_data.modal_utils import VOLUME_MOUNTS, app, build_image

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    @app.function(
        image=build_image(),
        volumes=VOLUME_MOUNTS,
        timeout=2 * 60 * 60,
        max_containers=128,
    )
    def inspect_one(remote_input: str) -> dict:
        stats = filter_wet_file(Path(remote_input), output_dir / Path(remote_input).name, **params)
        return dataclasses.asdict(stats)

    files = _input_files([input_dir], args.n_files)
    with app.run():
        stats_dicts = list(inspect_one.map(str(p) for p in files))

    aggregate = FilterStats()
    for stats_dict in stats_dicts:
        aggregate.merge(FilterStats(**stats_dict))
    report_path = Path(args.output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        _render_report(f"Modal run over {len(files)} file(s) in {input_dir}", aggregate, None, None) + "\n"
    )
    print("\n" + format_funnel_text(aggregate))
    print(f"\n[inspect] report: {report_path}")


def _parse_args() -> argparse.Namespace:
    config = _load_config(_DEFAULT_CONFIG)
    defaults = config.get("inspect", {})
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--modal", action="store_true", help="run on Modal (handout 4.2) instead of locally")
    parser.add_argument(
        "--stats-dir",
        help="aggregate the filter_stats.json sidecars in this directory instead of re-running the pipeline",
    )
    parser.add_argument("--inputs", action="append", help="input WET file or directory (repeatable; local mode)")
    parser.add_argument("--input-dir", help="directory of WET files on the Modal volume (modal mode)")
    parser.add_argument("--output-report", required=True, help="path of the markdown report to write")
    parser.add_argument("--n-files", type=int, default=None, help="cap the number of input files")
    parser.add_argument("--max-records", type=int, default=None, help="cap records per file (smoke tests)")
    parser.add_argument(
        "--samples", type=int, default=defaults.get("samples", 5), help="sample size per bucket (local mode)"
    )
    parser.add_argument("--seed", type=int, default=defaults.get("seed", 336), help="reservoir sample seed")
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
    if not args.modal and not args.stats_dir and not args.inputs:
        parser.error("local mode requires --stats-dir or --inputs")
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
