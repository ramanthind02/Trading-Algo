# Feature Research Phase Refactor Design

**Goal:** Hard-cutover refactor of `feature_research/` into phase-first packages (`in_sample/`, `walkforward/`, `oos/`) while preserving behavior and keeping the shared walkforward engine under `feature_research/walkforward/*`.

**Non-goals:**
- No backward-compatibility shims (no `feature_research.rule_based` / `feature_research.continuous_binning` re-exports).
- No algorithm/metric/behavior changes beyond what is required for the move (imports, module paths, entrypoint locations).

## Target Package Layout

End state (hard cutover):

- `feature_research/in_sample/rule_based/`
  - in-sample config/data loading/EDA pipeline and any in-sample permutation helpers
- `feature_research/in_sample/continuous_binning/`
  - in-sample config/data loading/EDA pipeline, binning analysis modules, and run scripts
- `feature_research/walkforward/`
  - shared engine (must remain): `config.py`, `runner.py`, `io.py`, `metrics.py`, `visualization.py`, `portfolio_evaluator.py`, `top_k_selection.py`, `stable_region_selection.py`
- `feature_research/walkforward/rule_based/`
  - rule-based walkforward entrypoints (thin orchestration over engine + in-sample config)
- `feature_research/walkforward/continuous_binning/`
  - continuous-binning walkforward entrypoints (thin orchestration over engine + in-sample config)
- `feature_research/oos/rule_based/`
  - strict OOS entrypoints (thin orchestration over engine + in-sample config)
- `feature_research/oos/continuous_binning/`
  - strict OOS entrypoints (thin orchestration over engine + in-sample config)

Canonical imports after cutover:

- `feature_research.in_sample.rule_based.*`
- `feature_research.in_sample.continuous_binning.*`
- `feature_research.walkforward.*` (engine)
- `feature_research.walkforward.rule_based.*` (entrypoints)
- `feature_research.walkforward.continuous_binning.*` (entrypoints)
- `feature_research.oos.rule_based.*` (entrypoints)
- `feature_research.oos.continuous_binning.*` (entrypoints)

## Migration Map (Old -> New)

Rule-based:
- `feature_research/rule_based/config.py` -> `feature_research/in_sample/rule_based/config.py`
- `feature_research/rule_based/data_loader.py` -> `feature_research/in_sample/rule_based/data_loader.py`
- `feature_research/rule_based/pipeline.py` -> `feature_research/in_sample/rule_based/pipeline.py`
- `feature_research/rule_based/run_eda.py` -> `feature_research/in_sample/rule_based/run_eda.py`
- `feature_research/rule_based/run_walkforward.py` -> `feature_research/walkforward/rule_based/run_walkforward.py`
- `feature_research/rule_based/param_sensitivity.ipynb` -> `feature_research/in_sample/rule_based/param_sensitivity.ipynb`

Continuous-binning:
- `feature_research/continuous_binning/config.py` -> `feature_research/in_sample/continuous_binning/config.py`
- `feature_research/continuous_binning/data_loader.py` -> `feature_research/in_sample/continuous_binning/data_loader.py`
- `feature_research/continuous_binning/pipeline.py` -> `feature_research/in_sample/continuous_binning/pipeline.py`
- `feature_research/continuous_binning/run_eda.py` -> `feature_research/in_sample/continuous_binning/run_eda.py`
- `feature_research/continuous_binning/binning_analysis.py` -> `feature_research/in_sample/continuous_binning/binning_analysis.py`
- `feature_research/continuous_binning/run_binning_analysis.py` -> `feature_research/in_sample/continuous_binning/run_binning_analysis.py`
- `feature_research/continuous_binning/run_walkforward.py` -> `feature_research/walkforward/continuous_binning/run_walkforward.py`
- `feature_research/continuous_binning/param_sensitivity.ipynb` -> `feature_research/in_sample/continuous_binning/param_sensitivity.ipynb`

New OOS entrypoints:
- `feature_research/oos/rule_based/run_oos.py`
- `feature_research/oos/continuous_binning/run_oos.py`

## Entrypoints and Import Safety

The `run_*.py` scripts currently perform `sys.path` injection based on `Path(__file__).parents[N]`. After nesting under `in_sample/` and `walkforward/`, this depth changes.

Design choice:
- Keep the scripts runnable both as modules (`python -m ...`) and as direct scripts (`python path/to/run_*.py`).
- Update the root-path shim to compute the repo root robustly (search upward for `pyproject.toml` or `feature_research/` package boundary) instead of relying on a fixed `parents[N]`.

Constraint:
- The shared walkforward engine under `feature_research/walkforward/*` must not import the phase/type entrypoints to avoid circular imports.

## Testing and Verification

Primary gates for this refactor:
- `python -m compileall feature_research`
- `pytest -q tests/feature_research -k import` (new smoke test module + updated existing tests)
- `pytest -q tests/feature_research` (best effort; keep these green after refactor)
- `python -c "import feature_research; import feature_research.walkforward.runner"`
- Repo-wide grep confirms no legacy imports remain:
  - `git grep -n "feature_research\.rule_based"`
  - `git grep -n "feature_research\.continuous_binning"`

Note: full-suite `pytest` is currently not a reliable global gate in this environment due to missing optional dependencies/fixtures unrelated to this refactor. The refactor will focus on the `feature_research` test set plus compile/import verification.
