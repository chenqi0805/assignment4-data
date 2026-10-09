# Codebase map — chenqi0805/assignment4-data

Depth 2. Course assignment repo: data filtering/deduplication pipeline for LM training.

| Path | Type | Purpose |
|---|---|---|
| `cs336_basics/` | python pkg | Staff implementation of the LM + training loop from assignment 1 (`model.py`, `data.py`, `optimizer.py`, `ddp_utils.py`, `train_config.py`). Do not modify — leaderboard submissions must use it exactly. |
| `cs336_data/` | python pkg | **Student work lives here.** `common.py`, `wet_files.py` (WET file handling, `is_english` TODO), `modal_utils.py` (Modal shared volume). Currently near-empty by design. |
| `tests/` | python pkg | pytest suite (21 tests). `adapters.py` = `NotImplementedError` stubs mapping tests → student functions. `common.py` = `FIXTURES_PATH`. `fixtures/` = HTML/quality/dedup fixtures. |
| `scripts/` | scripts | `download_data.py` (Modal/non-modal data download), `train.py` (Modal GPU training entrypoint), `generate_with_gpt2_tok.py` (GPT-2 tokenization). |
| `configs/` | config | Hydra config: `config.yaml` + `experiment/your_data.yaml` (paths/model/training groups). |
| `pyproject.toml` / `uv.lock` | manifest | uv-managed deps; pytest + ruff config. |
| `test_and_make_submission.sh` | script | Runs tests then zips the repo into the submission archive. |
| `cs336_assignment4_data.pdf` | doc | Assignment handout (full spec). |
| `AGENTS.md` / `CLAUDE.md` | doc | AI assistant policy: teaching-assistant role, no solution generation, no assignment TODO implementation. |
