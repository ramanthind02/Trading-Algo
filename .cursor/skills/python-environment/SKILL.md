---
name: python-environment
description: >-
  Run Python, pytest, and module entrypoints in this repository using the shared
  repo-root venv. Covers Windows PowerShell (explicit .venv\Scripts\python.exe),
  Linux/macOS activation, and fixing "pytest not found" from the wrong interpreter.
  Use when executing tests, run_is, validation scripts, or proposing shell commands.
---

# Python environment (Trading-Algo)

## Principles

- **One venv** at the repository root only (`venv` or `.venv`). Do not create nested venvs in worktrees.
- **Wrong interpreter** → missing packages or "`pytest` unavailable". Always use the venv’s `python.exe` / `python`.
- Prefer **`python -m pytest`** over bare `pytest` so PATH activation is optional.

## Windows PowerShell (repo root)

Use the venv interpreter explicitly (match your folder name: `.venv` or `venv`):

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit-tests\feature_research\test_permutation_pipeline.py -v
.\.venv\Scripts\python.exe -m feature_research.in_sample.run_is
```

Multi-file test slice example:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit-tests\feature_research\test_permutation_pipeline.py tests\unit-tests\feature_research\test_research_data.py tests\feature_research\validation\engine\test_run_walkforward_permutation.py tests\feature_research\test_pipeline_3d_selected_bin.py -v
```

You may use an absolute path to `...\Scripts\python.exe` instead of `.\.venv\...` if the working directory is not the repo root.

## Linux / macOS / WSL

```bash
source venv/bin/activate   # or: source .venv/bin/activate
python -m pytest tests/ -v
python -m feature_research.in_sample.run_is
```

## IDE

- Cursor/VS Code: `python.defaultInterpreterPath` should point at the venv interpreter (see repo `.vscode/settings.json`; on Unix use `venv/bin/python` or `.venv/bin/python`).

## Canonical reference

- Always-applied rule: `.cursor/rules/005-python-environment.mdc`
- Project docs: `AGENTS.md`, `CLAUDE.md`
