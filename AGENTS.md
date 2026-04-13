# AGENTS.md

## Current Scope (what exists now)
- This repo has two parallel tracks:
  - **Python/UV project scaffold** at repo root (`pyproject.toml`, `uv.lock`, `.venv`, `main.py`).
  - **LaTeX report workflow** in `report_tex/`.
- `SPEC.md` defines the intended sentiment-analysis experiment, but implementation code is still minimal (`main.py` only prints a greeting).

## High-Value Files
- `pyproject.toml` — project metadata, Python version requirement, and dependency declaration.
- `uv.lock` — locked dependency graph; includes exact wheel/index resolution.
- `.python-version` — interpreter pin (`3.11`).
- `main.py` — current Python entry file.
- `report_tex/justfile` — TeX build/watch/clean commands.
- `SPEC.md` — project objective/datasets/models/metrics/deliverables.

## Commands That Are Actually Defined

### Python / UV (repo root)
- Sync environment from lockfile:
  - `uv sync`
- Run current entrypoint:
  - `uv run python main.py`
- Lockfile-aware execution is expected because `uv.lock` is present.

### LaTeX report (`report_tex/`)
- Build PDF:
  - `cd report_tex && just build`
- Watch/rebuild on changes:
  - `cd report_tex && just watch`
- Clean TeX artifacts:
  - `cd report_tex && just clean`

## Dependency/Tooling Gotchas
- `pyproject.toml` sets `requires-python = ">=3.11"`; `.python-version` is `3.11`.
- A custom UV index is configured:
  - `[[tool.uv.index]] url = "https://download.pytorch.org/whl/xpu"`
- `uv.lock` resolves `torch/torchaudio/torchvision` as **XPU builds** (`+xpu`) and pulls many Intel runtime packages (`intel-opencl-rt`, `intel-sycl-rt`, `onemkl-*`, etc.).
  - If dependency behavior differs from standard CUDA/CPU expectations, check this index first.
- No lint/test/tool task runner is configured yet (no `pytest` config, no Makefile, no CI workflow observed).

## Architecture / Control Flow (current)
- Runtime architecture is not implemented yet.
- Present executable flow:
  1. `uv run python main.py`
  2. `main()` prints a placeholder message.
- Documentation/report flow:
  1. Write/edit manuscript in `report_tex/lastname_firstname_title.tex`
  2. Compile via `just` + `tectonic`

## Conventions Observed
- Python package/project name: `sentiment-analysis` (`pyproject.toml`).
- Root uses UV-managed env/lock (`.venv`, `uv.lock`) rather than ad-hoc `pip` files.
- Report uses ACL-style TeX assets in:
  - `report_tex/tex_styles/`
  - `report_tex/bib_styles/`

## What Not to Assume
- Do not assume training/inference/eval pipelines already exist.
- Do not assume tests/lint/typecheck commands exist yet.
- Do not assume default PyPI-only dependency resolution; XPU index is explicitly configured.
