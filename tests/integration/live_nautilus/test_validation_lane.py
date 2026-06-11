"""Integration test for the final-validation lane's on-the-fly causal signal.

The lane's architectural guarantee is that the REAL forecast engine generates the
signal causally — bounded to the simulation clock — so signal + execution share
one clock and the run is lookahead-free. This asserts that guarantee end-to-end on
the real vault + central cache:

* ``evaluate(as_of=T)`` returns a forecast whose data is dated <= T (never the
  full-cache end), and
* later ``as_of`` never regresses the forecast date (monotone), and
* the targets are real (at least one non-zero position).

It is CI-safe: it SKIPS (never fails) when the MT5 data store is absent or the
central cache is not yet populated for the vault's required tickers (the live data
pipeline / the lane's own preflight populates it). The pure ``as_of`` bound logic
is covered fast + unconditionally in
``tests/unit-tests/deployment/live_nautilus/test_forecast_engine_units.py``.
"""
from __future__ import annotations

import pandas as pd
import pytest

from cache.runtime.central_cache_errors import ArtifactMissingError
from data_platform.nautilus.ingest import mt5_data_root
from deployment.live.forecast_engine import ForecastEngineConfig, VaultForecastEngine
from lib.core import research_feed

pytestmark = pytest.mark.skipif(
    not mt5_data_root().exists(), reason="data/mt5_data not present"
)

_AS_OF_EARLY = pd.Timestamp("2024-10-02")
_AS_OF_LATE = pd.Timestamp("2024-10-08")


@pytest.fixture(scope="module")
def _engine() -> VaultForecastEngine:
    research_feed.set_research_feed("cfd")
    eng = VaultForecastEngine(
        ForecastEngineConfig(vault_root="vault", prediction_daily_max_bars=750)
    ).load()
    # Skip cleanly if the central cache has not been populated for the vault's
    # required tickers (CI / fresh checkout, or another test rebound the shared
    # CentralCacheStore singleton) — do NOT run the slow preflight here.
    try:
        result = eng.evaluate(as_of=_AS_OF_LATE)
    except (ValueError, ArtifactMissingError) as exc:
        pytest.skip(f"central cache not populated for vault tickers: {exc}")
    if not result.ready:
        pytest.skip(
            "central cache not populated for vault tickers "
            f"(pending warmup: {result.warmup.not_ready()}); "
            "run research.validation.validation_lane.ensure_central_cache_coverage first"
        )
    return eng


def test_forecast_is_causally_bounded_to_as_of(_engine: VaultForecastEngine) -> None:
    """The forecast date never exceeds ``as_of`` — the lookahead-free guarantee."""
    result = _engine.evaluate(as_of=_AS_OF_EARLY)
    assert result.ready
    assert result.as_of is not None
    assert pd.Timestamp(result.as_of) <= _AS_OF_EARLY


def test_forecast_date_is_monotone_in_as_of(_engine: VaultForecastEngine) -> None:
    early = _engine.evaluate(as_of=_AS_OF_EARLY)
    late = _engine.evaluate(as_of=_AS_OF_LATE)
    assert early.ready and late.ready
    assert pd.Timestamp(early.as_of) <= pd.Timestamp(late.as_of) <= _AS_OF_LATE


def test_forecast_produces_real_targets(_engine: VaultForecastEngine) -> None:
    """The on-the-fly signal is non-trivial (at least one non-zero position)."""
    result = _engine.evaluate(as_of=_AS_OF_LATE)
    assert result.targets, "forecast returned no targets"
    assert any(abs(v) > 0.0 for v in result.targets.values()), (
        f"all targets are flat: {dict(result.targets)}"
    )


def test_lane_end_to_end_produces_fills(_engine: VaultForecastEngine) -> None:
    """Full lane: real strategy + on-the-fly signal -> decisions -> fills -> equity.

    Regression guard for the whole pipeline, incl. the synth-quote real-UTC
    conversion (without it the decision timer never fires and there are no fills).
    Reuses ``_engine`` so it shares the cache-warm skip gate. populate=False: the
    cache is already warm here, so no slow preflight.
    """
    from research.validation import ValidationConfig, run_validation_backtest

    res = run_validation_backtest(
        ValidationConfig(
            start=pd.Timestamp("2024-10-01", tz="UTC"),
            end=pd.Timestamp("2024-10-09", tz="UTC"),
            tickers=("ES", "NQ", "GC", "CL", "SI"),
            quote_stride_min=5,
            populate_central_cache=False,
        )
    )
    assert res.resolved_tickers, "no tickers resolved"
    assert res.n_decisions > 0, "lane produced no fills — decision timer never fired (tz?)"
    assert len(res.equity_curve) > 1, "no equity samples recorded"
