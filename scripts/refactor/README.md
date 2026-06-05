# WP-8 restructure tooling (`scripts/refactor/`)

Execution tooling for the approved **"consolidate flat top-level"** restructure.
Encodes `docs/refactor/nautilus/wp8_move_map.md` as data and drives the moves +
import codemod per batch, **non-destructively** until the operator opts in.

> **Nothing here runs automatically.** Every step is operator-invoked, one batch
> at a time, behind the parity gate. `--dry-run` is the default everywhere;
> `--apply` is required to change anything.

## Files

| File | Role |
|---|---|
| `move_map.py` | The move-map as data: `IMPORT_RENAMES` (dotted-prefix rules, longest-first) and `PATH_MOVES` (old->new, tagged by batch) + `BATCH_ORDER`. No I/O. |
| `rewrite_imports.py` | libcst codemod: Pass A (imports), Pass B (dynamic string args of `import_module`/`patch`/`setattr`/...), Pass C (`_taxonomy.py` dict values, R15 only). `--dry-run`/`--apply`. |
| `run_moves.py` | `git mv` per batch in leaf-first order. `--batch <name>` / `--all`, `--dry-run`/`--apply`. |

## Prerequisites

* **libcst** — required for `rewrite_imports.py --apply`. If missing, `--apply`
  refuses to run; `--dry-run` still produces counts via a regex fallback.
  Install: `.\.venv\Scripts\python.exe -m pip install libcst` (not done here).
* **Phase 1 cull (B0)** must be complete first (dead `base_models` stubs,
  `metrics/` shell, `utils/simulation/`, repurpose `research/` stub). The map
  assumes those are gone.
* **WP-1 parity harness green** before starting.

## Optional flags (default OFF)

* `nodes/ -> signals/` (R15, batch B12): pass `--rename-nodes` to
  `rewrite_imports.py` and run `run_moves.py --batch B12`.
* `prop_firms/ -> research/prop_firms/` (R4, batch B11): `--rename-prop-firms` /
  `--batch B11`. Default: keep `prop_firms/` top-level.

## Operator sequence — PER BATCH

Run from the repo root with the venv interpreter. For each batch in
`BATCH_ORDER` (B1, B2, B3, ... — see `run_moves.py --list`):

```powershell
# 0. Clean tree + green gate before starting the batch.
git status                       # must be clean

# 1. Preview, then perform the git mv for THIS batch only.
.\.venv\Scripts\python.exe scripts\refactor\run_moves.py --batch B2 --dry-run
.\.venv\Scripts\python.exe scripts\refactor\run_moves.py --batch B2 --apply

# 2. Rewrite imports across the whole repo (the codemod is idempotent and
#    only touches prefixes whose package has actually moved).
.\.venv\Scripts\python.exe scripts\refactor\rewrite_imports.py --dry-run
.\.venv\Scripts\python.exe scripts\refactor\rewrite_imports.py --apply

# 3. Fixups: add __init__.py for any newly-created package root
#    (research/, features/, features/extraction/, lib/, tools/, scripts/dev/, ...).
#    For B6 also fix setup_cython.py name=/sources= and recompile (L6).

# 4. Gate. feature_research parity is the available net today:
.\.venv\Scripts\python.exe -m pytest tests\parity -k feature_research

# 5. Proceed or revert:
#    GREEN -> commit the batch, move to the next.
#    RED   -> `git mv` is reversible; revert this batch only (git checkout / git mv back),
#             re-run step 4, fix, retry. Never carry a red gate into the next batch.
```

For the optional final batch (B12, nodes->signals) add `--rename-nodes` to the
`rewrite_imports.py` calls and run `run_moves.py --batch B12`.

## Gate note (important)

The **full** WP-8 gate is WP-1 parity (feature + portfolio cases) + full
`pytest`. The **portfolio** parity case is currently **blocked on TLT** data, so
the reliably-runnable net per batch is:

```
pytest tests\parity -k feature_research
```

Treat `feature_research` parity + `pytest` as the per-batch gate until the
portfolio/TLT case is unblocked, then add it back to step 4.

## Batch order (leaf-first; highest fan-in last; optional last)

`run_moves.py --list` prints the live list. Summary:

| Batch | Scope |
|---|---|
| B0 | Phase 1 cull (prereq — not a move batch) |
| B1 | `scripts/_*.py` -> `scripts/dev/` (kept probes only) |
| B2 | `feature_extraction/` -> `features/extraction/` |
| B3 | `feature_selection/{base_models,eda,validation}` -> `features/{models,eda,validation}` |
| B4 | `utils/{research_workspace,dev}` -> `tools/` |
| B5 | `utils/evaluation/` -> `research/evaluation/` |
| B6 | `utils/compute/` -> `lib/compute/` (+ cython rebuild, L6) |
| B7 | `utils/cache/` -> `lib/cache/` |
| B8 | `utils/core/` + loose `utils/*.py` -> `lib/core/` |
| B9 | `portfolio_research/` -> `research/portfolio/` |
| B10 | `feature_research/` -> `research/feature/` |
| B11 | OPTIONAL `prop_firms/` -> `research/prop_firms/` |
| B12 | OPTIONAL FINAL `nodes/` -> `signals/` |
| B13 | Mirror `tests/` + Phase-3 docs repoint (not a move batch) |
