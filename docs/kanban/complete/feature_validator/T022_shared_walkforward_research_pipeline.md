# T022 - Shared Walkforward Research Pipeline

## Goal
Define the config+metrics foundation for shared walkforward research by implementing deterministic configuration modeling and objective metric resolution used by follow-on pipeline tasks.

## Context / References
- `docs/library/Feature_selection/Walkforward/walkforward.md`
- `docs/library/Feature_selection/feature_validator.md`
- `utils/walkforward.py`
- `docs/plans/2026-02-17-walkforward-shared-research-design.md`
- `feature_selection/walkforward/walkforward_model.py`

## Scope
In scope:
- Production code changes are limited to `feature_research/walkforward/config.py` and `feature_research/walkforward/metrics.py` (create file if missing, otherwise modify in place).
- Implement walkforward research configuration modeling/validation in `feature_research/walkforward/config.py`.
- Implement objective metric resolution in `feature_research/walkforward/metrics.py`.
- Add and run related unit tests for the config+metrics slice only.

Out of scope:
- Runner orchestration in `feature_research/walkforward/runner.py`.
- Artifact writing and output layout in `feature_research/walkforward/io.py`.
- Plot generation in `feature_research/walkforward/visualization.py`.
- Integration wiring in `feature_research/rule_based/pipeline.py` and `feature_research/continuous_binning/pipeline.py`.
- Feature-type adapters that bind rule-based or continuous candidate generation to the shared core.
- Changes to Stage 1/2 permutation logic, base model internals, or vault schemas.
- Production execution sizing, deployment pipelines, or live trading behavior.
- New feature families beyond follow-on tasks.

## Interfaces (must match)
- In `feature_research/walkforward/config.py` (add if missing, otherwise modify existing file)
  - `@dataclass(frozen=True)` config object signature:
    - `class WalkforwardResearchConfig:`
      - `enabled: bool`
      - `train_start: datetime`
      - `train_end: datetime`
      - `test_step: int`
      - `num_steps: int`
      - `top_k: int`
      - `objective_metric_name: str`
      - `min_fold_samples: int`
      - `output_root: Path`
- In `feature_research/walkforward/metrics.py` (add if missing, otherwise modify existing file)
  - `def resolve_objective_metric(metric_name: str) -> Callable[[pd.Series], float]`
  - Unknown metric names raise `ValueError` with the metric name included.

## Data Contracts
- `WalkforwardResearchConfig` field contract (`feature_research/walkforward/config.py`):
  - `enabled: bool = False`.
  - `train_start: datetime` is required.
  - `train_end: datetime` is required and must be strictly greater than `train_start`.
  - `test_step: int = 252`, `>= 1`.
  - `num_steps: int = 10`, `>= 1`.
  - `top_k: int = 3`, `>= 1`.
  - `objective_metric_name: str = "sharpe"`, allowed values exactly `"sharpe" | "sortino" | "mean_return"`.
  - `min_fold_samples: int = 10`, `>= 10`.
  - `output_root: Path = Path("feature_research/shared_results")`, must be non-empty.
- Metric resolver contract (`feature_research/walkforward/metrics.py`):
  - Input: `metric_name: str` (case-sensitive), accepted values exactly `"sharpe"`, `"sortino"`, `"mean_return"`.
  - Output: `Callable[[pd.Series], float]` that consumes a returns series and produces a scalar objective value.
  - Error behavior: unsupported names raise `ValueError` and include the provided metric name in the error message.

## Dependencies
- Allowed write/touch targets:
  - `feature_research/walkforward/config.py`
  - `feature_research/walkforward/metrics.py`
  - `tests/feature_research/walkforward/test_config.py`
  - `tests/feature_research/walkforward/test_metrics.py`
  - `docs/api/feature_selection.md`
- Allowed read/reference dependencies:
  - `docs/library/Feature_selection/Walkforward/walkforward.md`
  - `docs/library/Feature_selection/feature_validator.md`
  - `docs/plans/2026-02-17-walkforward-shared-research-design.md`
  - `utils/walkforward.py`
  - `feature_selection/walkforward/walkforward_model.py`
  - `feature_research/walkforward/runner.py` (read-only, out-of-scope for edits)
  - `feature_research/walkforward/io.py` (read-only, out-of-scope for edits)
  - `feature_research/walkforward/visualization.py` (read-only, out-of-scope for edits)
  - `feature_research/rule_based/pipeline.py` (read-only, out-of-scope for edits)
  - `feature_research/continuous_binning/pipeline.py` (read-only, out-of-scope for edits)

## Required config semantics (must match)
- Supported `objective_metric_name` values are exactly: `"sharpe"`, `"sortino"`, `"mean_return"`.
- `resolve_objective_metric(...)` must accept only those exact values (case-sensitive) and raise `ValueError` otherwise.
- `WalkforwardResearchConfig` defaults and validation boundaries:
  - `enabled`: default `False`.
  - `train_start`: required `datetime`; no default.
  - `train_end`: required `datetime`; no default; must be strictly greater than `train_start`.
  - `test_step`: default `252`; integer, `>= 1`.
  - `num_steps`: default `10`; integer, `>= 1`.
  - `top_k`: default `3`; integer, `>= 1`.
  - `objective_metric_name`: default `"sharpe"`; must be one of the supported names above.
  - `min_fold_samples`: default `10`; integer, `>= 10`.
  - `output_root`: default `Path("feature_research/shared_results")`; must be a non-empty `Path` value.

## Invariants / Constraints
- Deterministic behavior: identical config values and metric name resolve to identical config state and objective callable selection.
- Config validation must reject invalid boundary/sizing inputs with explicit errors.
- Metric resolution must accept supported names and raise `ValueError` for unsupported names.
- Task is strictly bounded to `feature_research/walkforward/config.py` and `feature_research/walkforward/metrics.py`, plus related unit tests.

## Acceptance tests
1. `source venv/bin/activate && pytest tests/feature_research/walkforward/test_config.py -q` - config validation and deterministic defaults.
2. `source venv/bin/activate && pytest tests/feature_research/walkforward/test_metrics.py -q` - objective metric resolution and unknown-metric error handling.
3. `source venv/bin/activate && python -c "from pathlib import Path; import re; text = Path('docs/api/feature_selection.md').read_text(encoding='utf-8'); assert re.search(r'class WalkforwardResearchConfig\b', text), 'Missing WalkforwardResearchConfig class doc'; required_fields = ['enabled: bool', 'train_start: datetime', 'train_end: datetime', 'test_step: int', 'num_steps: int', 'top_k: int', 'objective_metric_name: str', 'min_fold_samples: int', 'output_root: Path']; missing_fields = [field for field in required_fields if field not in text]; assert not missing_fields, f'Missing config fields in docs: {missing_fields}'; assert re.search(r'def resolve_objective_metric\(metric_name: str\) -> Callable\[\[pd\\.Series\], float\]', text), 'Missing resolve_objective_metric signature doc'"` - docs/api page documents the full config field set and exact metric resolver signature.

## Definition of done
- [ ] Only `feature_research/walkforward/config.py` and `feature_research/walkforward/metrics.py` are changed in production scope.
- [ ] `docs/api/feature_selection.md` is updated to document `WalkforwardResearchConfig` and `resolve_objective_metric(...)` interface changes.
- [ ] Unit tests listed above are added and pass.
- [ ] Verification commands executed:
  - `source venv/bin/activate && pytest tests/feature_research/walkforward/test_config.py -q`
  - `source venv/bin/activate && pytest tests/feature_research/walkforward/test_metrics.py -q`
  - `source venv/bin/activate && python -c "from pathlib import Path; import re; text = Path('docs/api/feature_selection.md').read_text(encoding='utf-8'); assert re.search(r'class WalkforwardResearchConfig\b', text), 'Missing WalkforwardResearchConfig class doc'; required_fields = ['enabled: bool', 'train_start: datetime', 'train_end: datetime', 'test_step: int', 'num_steps: int', 'top_k: int', 'objective_metric_name: str', 'min_fold_samples: int', 'output_root: Path']; missing_fields = [field for field in required_fields if field not in text]; assert not missing_fields, f'Missing config fields in docs: {missing_fields}'; assert re.search(r'def resolve_objective_metric\(metric_name: str\) -> Callable\[\[pd\\.Series\], float\]', text), 'Missing resolve_objective_metric signature doc'"`
- [ ] Task remains scoped to the config+metrics slice and excludes runner/io/visualization and adapter integrations.

## Notes
- Keep adapters thin; domain-specific extraction stays in each feature package while fold logic stays shared.
- Follow-on task (new ticket): shared fold runner orchestration in `feature_research/walkforward/runner.py`.
- Follow-on task (new ticket): shared artifact writer in `feature_research/walkforward/io.py`.
- Follow-on task (new ticket): shared walkforward visualizations in `feature_research/walkforward/visualization.py`.
- Follow-on task (new ticket): rule-based adapter integration into `feature_research/rule_based/pipeline.py`.
- Follow-on task (new ticket): continuous adapter integration into `feature_research/continuous_binning/pipeline.py`.

## Result
- Implemented in: uncommitted changes on branch `feature/shared-walkforward-research`
- Tests: `pytest tests/feature_research/walkforward/test_config.py -q` and `pytest tests/feature_research/walkforward/test_metrics.py -q` ✅
- Notes: Added docs API entries in `docs/api/feature_selection.md`; output_root now enforces `Path` type.
