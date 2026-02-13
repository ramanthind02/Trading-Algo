# API doc scope

## Goal
Generate API docs across the entire repo into `docs/api/`.

## Include
- All first-party packages under:
  - `src/`
  - `app/` (if present)
  - `packages/` (if monorepo)
  - `libs/` (if present)
- Any CLI entry points under:
  - `scripts/`
  - `bin/`
  - `tools/`

## Exclude (unless explicitly referenced by included code)
- `**/.venv/**`
- `**/venv/**`
- `**/__pycache__/**`
- `**/.git/**`
- `**/node_modules/**`
- `**/dist/**`
- `**/build/**`
- `**/.pytest_cache/**`
- `**/.mypy_cache/**`
- `**/.ruff_cache/**`
- `**/.cache/**`
- `**/site-packages/**`
- `**/data/**` (raw datasets, large files)
- `**/logs/**`
- `**/outputs/**`
- `**/artifacts/**`

## “Public API” definition
Document:
- Public symbols: not prefixed with `_`
- Re-exports from `__init__.py`
- Cross-module surfaces: imported by other packages
- Primary configuration objects
- Main entrypoints (CLI/main functions)

Skip:
- Private helpers (`_name`)
- Purely internal utilities not referenced outside their module
- Tests and fixtures (unless they define reusable helpers that are part of a supported dev API)

## Output policy
- Create one doc per *package/module cluster* (avoid 1 file per tiny module unless needed).
- Prefer stable groupings (e.g., `trading.data`, `trading.signals`, etc.).
- Write/update:
  - `docs/api/_inventory.md`
  - `docs/api/index.md`
  - `docs/api/<group>.md`