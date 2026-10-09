# obvious.md — chenqi0805/assignment4-data

Agent guidance for CS336 Spring 2026 Assignment 4: Data (data filtering/deduplication for LM training).

## Critical context

- This is a **course assignment repo**. `cs336_data/` and `tests/adapters.py` contain
  deliberate `NotImplementedError` stubs for students to implement. Per `AGENTS.md` /
  `CLAUDE.md`: **do not implement assignment TODOs, do not write solution code, do not
  edit student code**. Agent work here is environment setup, code review, debugging
  guidance, and explanation only.
- The test suite failing with `NotImplementedError` (21/21 as of onboarding) is the
  **expected shipped baseline**, not a broken environment.

## Stack

| Component | Version |
|---|---|
| Python | 3.12.15 (uv-managed, `.python-version`) |
| Package manager | uv 0.12.24 (`uv.lock`, `[tool.uv] package = true`) |
| Test runner | pytest (pytest-timeout plugin) |
| Linter | ruff (config in `pyproject.toml`, line-length 120) |
| Heavy deps | torch 2.11.0+cu130, transformers, fasttext, resiliparse, fastwarc, nltk, tiktoken, polars, hydra-core |

No web server, database, or message broker. No ports to open. Modal is used only for
cloud data download / GPU training — not needed for offline local development.
No `.env` / required env vars for the offline test flow.

## Commands

```sh
source $HOME/.local/bin/env   # if uv is not on PATH (fresh sandbox: install via astral.sh/uv/install.sh)
uv sync                       # install all deps into .venv (downloads Python 3.12 + torch on first run; slow — run under tmux)
uv run pytest ./tests         # full test suite (~1s; currently 21 failed / NotImplementedError = expected)
uv run pytest ./tests --collect-only -q   # quick health check: 21 tests collected
uv run python -c "import cs336_basics, cs336_data"        # module import check
uvx ruff check cs336_basics cs336_data tests scripts      # lint (11 pre-existing style findings in shipped code)
./test_and_make_submission.sh                              # run tests + build submission zip (requires `zip`)
uv run scripts/download_data.py --offline-only             # download offline fixtures (network; writes local-shared-data/)
```

## Codebase map

See [codebase-map.md](./codebase-map.md).

## Local verification (how to check your work)

1. `uv sync` exits 0.
2. `uv run pytest ./tests --collect-only -q` reports 21 tests collected.
3. `uv run pytest ./tests` runs the suite. Failures are expected until the assignment
   stubs are implemented; new failures beyond the `NotImplementedError` baseline (e.g.
   `ImportError`, collection errors, fixture path errors) indicate a broken environment.
4. `uv run python -c "import cs336_basics, cs336_data"` prints the OK message.

### Validation Summary (onboarding run, 2026-10-09)

- `uv sync`: exit 0; venv at `.venv` with Python 3.12.15.
- Imports: `torch 2.11.0+cu130`, `transformers 4.51.3`, `fasttext`, `resiliparse`,
  `nltk`, `tiktoken`, `polars`, `hydra` — all OK. CUDA unavailable in sandbox (expected).
- pytest: 21 collected, 21 failed with `NotImplementedError` (expected baseline;
  student TODOs unimplemented by design).
- `cs336_basics` / `cs336_data` import OK.
- ruff: 11 pre-existing style findings in shipped code (I001/F401/PLR0402/etc.), left untouched.
- Primary flow exercised: full pytest suite end-to-end.
- `dev_stack_healthy: true`

## Sandbox snapshot

- snapshotId: `czb4n6oqbn9julhii2ao:default`
- Captured: 2026-10-09T17:45:37Z
- State: fresh checkout + uv 0.12.24 installed + `uv sync` completed (warm `.venv`).

## Gotchas

- `uv` is **not preinstalled** on the sandbox; it lives in `~/.local/bin` — `source $HOME/.local/bin/env` first.
- `uv sync` pulls ~GB of torch/CUDA wheels; run it detached (tmux), not in a single short command.
- `pytest` config sets `addopts = "-s"` and `log_cli = true` — output is verbose.
- Test fixtures live in `tests/fixtures/`; data scripts write to `local-shared-data/` when running locally (both gitignored).
- `scripts/train.py` and `scripts/download_data.py` are Modal entrypoints — they need Modal auth and are out of scope for offline local dev.
