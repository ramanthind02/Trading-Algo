"""Integration test: VaultForecastEngine against the REAL central cache + vault.

Per the repo testing boundary, this drives the real pipeline entrypoints
(``build_global_portfolio_from_ensemble_dirs`` → ``fit_from_cache`` →
``predict_from_cache``) over repo-backed cache data. It adapts to data
availability: it skips cleanly when the vault or daily cache coverage is absent,
asserts warmup structure when coverage is thin, and asserts real targets when
enough history is present.
"""
from __future__ import annotations

import math

import pytest

from cache.runtime.central_cache_errors import ArtifactMissingError
from deployment.live.forecast_engine import ForecastEngineConfig, VaultForecastEngine

# Cache-not-populated shows up as either a ValueError (no coverage window) or an
# ArtifactMissingError (a required ticker's candles aren't loaded) — both mean
# "no usable cache for the vault here", so both skip cleanly per this module's intent.
_NO_COVERAGE = (ValueError, ArtifactMissingError)

# A low warmup threshold so the full fit/predict path actually exercises whenever
# the repo cache holds a reasonable daily history for the vault's tickers.
_WARMUP_MIN_BARS = 60


@pytest.fixture(scope="module")
def loaded_engine() -> VaultForecastEngine:
    engine = VaultForecastEngine(
        ForecastEngineConfig(vault_root="vault", warmup_min_bars=_WARMUP_MIN_BARS)
    )
    try:
        engine.load()
    except ValueError as exc:  # no ensembles in the vault on this checkout
        pytest.skip(f"vault has no loadable ensembles: {exc}")
    return engine


def test_required_tickers_discovered(loaded_engine: VaultForecastEngine) -> None:
    assert loaded_engine.required_tickers, "expected at least one required ticker"


def test_warmup_status_structure(loaded_engine: VaultForecastEngine) -> None:
    try:
        ws = loaded_engine.warmup_status()
    except _NO_COVERAGE as exc:  # no daily coverage / candles not loaded in the cache
        pytest.skip(f"no daily cache coverage for vault tickers: {exc}")
    assert ws.min_bars == _WARMUP_MIN_BARS
    assert ws.per_ticker, "expected per-ticker warmup rows"
    assert all(t.bars >= 0 for t in ws.per_ticker)


def test_evaluate_produces_targets_when_warm(loaded_engine: VaultForecastEngine) -> None:
    try:
        result = loaded_engine.evaluate()
    except _NO_COVERAGE as exc:
        pytest.skip(f"no daily cache coverage for vault tickers: {exc}")

    if not result.ready:
        pytest.skip(f"cache below warmup threshold; pending: {result.warmup.not_ready()}")

    assert result.targets, "a warm portfolio must emit at least one target"
    assert result.as_of is not None
    for ticker, fraction in result.targets.items():
        assert isinstance(ticker, str)
        assert math.isfinite(fraction), f"{ticker} target must be finite, got {fraction}"
