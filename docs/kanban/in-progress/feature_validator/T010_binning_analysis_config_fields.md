# T010 — Binning Analysis Config Fields

## Goal
Expose explicit binning-analysis parameters on the continuous-binning research config so downstream analysis code can rely on a stable, typed contract.

## Context / References
- `feature_research/continuous_binning/config.py`
- `tests/unit-tests/feature_validator/test_binning_analysis_config.py`

## Scope
In scope:
- Add a typed binning-analysis config dataclass.
- Attach the binning-analysis config to `ResearchConfig` and populate defaults in `load_config()`.
- Add a unit test covering the new config fields.

Out of scope:
- Any changes to binning models, selection logic, or analysis pipelines.

## Interfaces (must match)
- Modify: `feature_research/continuous_binning/config.py`
  - Add `BinningAnalysisConfig` dataclass.
  - Add `binning_params: BinningAnalysisConfig` to `ResearchConfig`.
  - Populate defaults in `load_config()`.
- Add: `tests/unit-tests/feature_validator/test_binning_analysis_config.py`

## Data Contracts
- `BinningAnalysisConfig` fields:
  - `n_bins: int`
  - `selection_metric: str`
  - `strategy: str`
  - `metric_threshold: float`
  - `t_threshold: float`
  - `min_region_width: int`
  - `max_regions: int`
  - `direction_filter: str`

## Dependencies
- `feature_research/continuous_binning/config.py`

## Invariants / Constraints
- Frozen dataclasses and full type hints for the config surface.
- Defaults remain researcher-editable via `load_config()`.

## Acceptance tests
1. `pytest tests/unit-tests/feature_validator/test_binning_analysis_config.py::test_binning_analysis_config_fields -v`
2. `python -m py_compile feature_research/continuous_binning/config.py`

## Definition of done
- [ ] `BinningAnalysisConfig` added and wired into `ResearchConfig`.
- [ ] Unit test added under `tests/unit-tests/feature_validator/`.
- [ ] `pytest tests/unit-tests/feature_validator/test_binning_analysis_config.py::test_binning_analysis_config_fields -v` passes.
- [ ] `python -m py_compile feature_research/continuous_binning/config.py` passes.

## Notes
- Keep changes limited to config/test surface; analysis logic will be added in follow-on tasks.
