---
name: local-dev
description: How to get a working local dev environment for chenqi0805/assignment4-data (CS336 assignment 4, Python 3.12 + uv + pytest, no services)
---

# Local dev — chenqi0805/assignment4-data

Durable record of the LOCAL-DEV onboarding run (2026-10-09).

## What this repo is

Python assignment repo (CS336 @ Stanford, assignment 4: data). No server, no database,
no ports, no Docker/Compose, no required env vars. Local dev = a synced uv venv plus a
runnable pytest suite. Modal is used only for cloud data download / 8×B200 training —
out of scope for offline local dev.

## Bootstrap (works on a fresh sandbox)

1. uv is **not preinstalled**: `curl -LsSf https://astral.sh/uv/install.sh | sh`
   → installs to `~/.local/bin`; then `source $HOME/.local/bin/env`.
2. `uv sync` — downloads managed Python 3.12 + all deps (torch 2.11 + CUDA wheels ≈ GBs).
   **Slow: run detached under tmux**, e.g.
   `tmux new-session -d -s uvsync 'source $HOME/.local/bin/env && uv sync > /tmp/uvsync.log 2>&1; echo "EXIT_CODE=$?" >> /tmp/uvsync.log'`
   and poll `/tmp/uvsync.log`. Verified exit 0.

## Verify (primary flow)

- `uv run pytest ./tests --collect-only -q` → **21 tests collected**.
- `uv run pytest ./tests` → runs in ~1s. Baseline as shipped: **21 failed, all
  `NotImplementedError`** — `tests/adapters.py` and `cs336_data/` are deliberate student
  TODO stubs. This is the expected state, NOT a broken environment. Per `AGENTS.md`,
  agents must not implement these stubs.
- `uv run python -c "import cs336_basics, cs336_data"` → OK.
- `uvx ruff check cs336_basics cs336_data tests scripts` → 11 pre-existing style findings
  in shipped code (I001, F401, PLR0402, RUF059, PLC0414, PLW1508); upstream baseline,
  leave untouched.
- Quick import sanity: torch 2.11.0+cu130, transformers 4.51.3, fasttext, resiliparse,
  nltk, tiktoken, polars, hydra all import fine; CUDA unavailable in sandbox (expected).

## Gotchas learned the hard way

- Always `source $HOME/.local/bin/env` (or use full path `~/.local/bin/uv`) in fresh shells.
- Foreground sleeps ≥ 60s are blocked — poll tmux logs instead of sleeping.
- `pytest` has `addopts = "-s"` and `log_cli = true` in `pyproject.toml` — verbose output is normal.
- No lock files, no port conflicts, no services to check — "health check" = pytest
  collection + module imports.

## Verdict

`dev_stack_healthy: true` — environment installs, suite runs end-to-end, failures are
assignment TODOs by design.
