# T011 — Multi-Timeframe Feature Research Support (H1/H4/D/W/M)

## Goal
Wire timeframe-awareness end-to-end in feature research so a single selected research timeframe controls annualization, defaults, cache auxiliaries, extraction parameters, and tearsheet/report behavior.

## Context / References
- `feature_research/config.py`
- `feature_extraction/feature_extractor.py`
- `utils/cache/cache_manager.py`
- `metrics/plotting/graphing/quantstats_reports.py`
- `feature_research/walkforward/runner.py`
- `feature_research/walkforward/io.py`
- `docs/api/utils.md`
- `docs/api/data_pipeline.md`
- `docs/api/metrics.md`

## Scope
In scope:
- Add `TimeFrame.H1/H4` and `bars_per_year`.
- Make research config/defaults/objective presets timeframe-aware.
- Make cache-required ATR/EWSD specs timeframe-aware.
- Remove ATR `"252"` coupling in forward-return normalization.
- Thread timeframe into walkforward tearsheets and aggregate annualized metrics.

Out of scope:
- Unrelated annualization hardcodes outside feature-research pipeline.
- Multi-timeframe-in-one-run research (still one timeframe per run).

## Interfaces (must match)
- Modify: `utils/core/enums.py` — `TimeFrame` adds `H1`, `H4`, and `bars_per_year`.
- Modify: `feature_research/config.py` — timeframe-aware presets/defaults; `BaseResearchConfig.timeframe`.
- Modify: `utils/cache/cache_manager.py` — `get_auxiliary_specs_for_timeframe`; `populate_cache(..., timeframe=...)`.
- Modify: `metrics/plotting/graphing/quantstats_reports.py` — `generate_tearsheet(..., timeframe=...)`.
- Modify: `feature_research/walkforward/runner.py` and `feature_research/walkforward/io.py` — pass/store timeframe and annualize by `bars_per_year`.

## Data Contracts
- `timeframes` remains single active timeframe in research configs (`[tf]`).
- ATR auxiliary period = `tf.bars_per_year`.
- EWSD auxiliary `long_run_window` = `10 * tf.bars_per_year`.
- For H1/H4 tearsheets, returns are resampled to daily by additive sum (no compounding).

## Dependencies
- `utils.core.enums`
- `feature_research.config`
- `feature_extraction.feature_extractor`
- `utils.cache.cache_manager`
- `metrics.plotting.graphing.quantstats_reports`
- `feature_research.walkforward.runner`
- `feature_research.walkforward.io`

## Invariants / Constraints
- Deterministic outputs for same inputs.
- No lookahead changes in feature-target alignment logic.
- Backward compatibility via default `TimeFrame.D` and retained `REQUIRED_AUXILIARY_SPECS` alias.

## Acceptance tests
1. `source venv/bin/activate && pytest tests/feature_research/test_config.py -q`
2. `source venv/bin/activate && pytest tests/feature_research/walkforward/test_runner.py tests/feature_research/walkforward/test_io.py -q`
3. `source venv/bin/activate && pytest tests/unit-tests/feature_extraction/test_feature_extractor_timeframe.py tests/unit-tests/utils/test_cache_manager.py tests/unit-tests/metrics/test_quantstats_reports_timeframe.py -q`

### Integration Test Data Contract (required when integration tests are in scope)
- Data source path: `data/ohlc_data` (existing integration suites)
- Tickers: researcher-config driven (single timeframe per run)
- Timeframe: configurable (`H1|H4|D|W|M`)
- Date range: `feature_research/config.py` configured windows
- Bias node spec: configured per in-sample defaults
- Cache mode: `use_cache/populate_cache` from shared config, with cache aux specs scaled by timeframe

## Definition of done
- [ ] Timeframe-aware behavior implemented across config/cache/extraction/tearsheet/walkforward aggregate metrics
- [ ] Unit tests added/updated for enum/config/cache/extractor/tearsheet threading
- [ ] API docs updated for changed interfaces
- [ ] Targeted pytest commands pass

## Notes
- EWSD parameters are intentionally aligned between cache population and extraction to avoid cache-key mismatches.
