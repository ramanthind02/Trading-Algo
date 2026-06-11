"""Adapter wiring for the conditional-returns endpoint (no heavy extraction / real run output).

Loads a real saved spec, stubs the best-combo grid + the extraction-backed analysis, and asserts
the analysis request and JSON shape are assembled correctly.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from frontend.api import conditional_returns
from frontend.api import results as results_module
import research.feature.binning.conditional_returns as cr_module
from lib.core.enums import Direction, Ticker, TimeFrame

SPEC_ID = "es_double7s_mr"  # module=double7s, tickers ES/NQ, D, futures, long, windows=None


def _grid_with_best(params: dict) -> dict:
    return {
        "headline": None,
        "grid": {
            "available": True,
            "param_keys": list(params),
            "rows": [
                {"label": "a", "params": {"short_period": 5, "ma_period": 200}, "is_best": False,
                 "nw_sharpe": 0.3, "t_stat": 1.0},
                {"label": "b", "params": params, "is_best": True, "nw_sharpe": 0.9, "t_stat": 2.5},
            ],
        },
        "plateau": {"available": False},
        "equity": {"available": False},
    }


def _canned_result() -> cr_module.ConditionalReturnsResult:
    return cr_module.ConditionalReturnsResult(
        available=True,
        reason=None,
        bins=[
            cr_module.BinStat(
                bin_index=0, label="< 30", lo=None, hi=30.0, regime=None, ticker=None,
                n_obs=42, mean_return=0.12, sharpe=0.8, sortino=1.1, t_stat=2.1,
                hit_rate=0.55, cumulative=1.4, instrument_mean_return=0.05,
            )
        ],
        condition_label="rsi(lookback=14)",
        strategy_label="double7s(...)",
        regime_label=None,
        bin_mode="quantile",
        n_bins=5,
        tickers=["ES", "NQ"],
    )


def test_adapter_builds_request_and_json(monkeypatch: pytest.MonkeyPatch) -> None:
    best = {"short_period": 7, "ma_period": 200}
    monkeypatch.setattr(results_module, "build_results", lambda *_a, **_k: _grid_with_best(best))

    captured: dict = {}

    def fake_analyze(req: cr_module.ConditionalReturnsRequest) -> cr_module.ConditionalReturnsResult:
        captured["req"] = req
        return _canned_result()

    monkeypatch.setattr(cr_module, "analyze_conditional_returns", fake_analyze)

    run = {"spec_id": SPEC_ID, "phase": "exploration", "reports_dir": "x", "viz_dir": "y"}
    body = {"indicator": {"module": "rsi", "params": {"lookback": 14}}, "bin_mode": "quantile", "n_bins": 5}

    out = conditional_returns.build_conditional_returns(run, body)

    req = captured["req"]
    assert req.strategy.module_name == "double7s"
    assert req.strategy.params == best
    assert req.condition.module_name == "rsi" and req.condition.params == {"lookback": 14}
    assert req.tickers == (Ticker.ES, Ticker.NQ)
    assert req.timeframe == TimeFrame.D
    # DAILY exploration analysis loads from the faithful ratio futures feed (matches the executor).
    assert req.data_feed == "futures_ratio"
    assert req.direction == Direction.LONG
    assert req.bin_mode == "quantile" and req.n_bins == 5
    assert req.regime is None
    # exploration phase → train_start .. validation_end of the default daily windows.
    assert req.start == datetime(2000, 1, 1)
    assert req.end == datetime(2022, 12, 31)

    assert out["available"] is True
    assert out["strategy"]["params"] == best
    assert out["strategy"]["module"] == "double7s"
    assert out["condition"]["module"] == "rsi"
    assert out["timeframe"] == "D"
    assert out["regime"] is None
    assert len(out["bins"]) == 1 and out["bins"][0]["label"] == "< 30"


def test_adapter_passes_regime_and_validation_window(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(results_module, "build_results", lambda *_a, **_k: _grid_with_best({"short_period": 9, "ma_period": 200}))
    captured: dict = {}

    def fake_analyze(req: cr_module.ConditionalReturnsRequest) -> cr_module.ConditionalReturnsResult:
        captured["req"] = req
        return _canned_result()

    monkeypatch.setattr(cr_module, "analyze_conditional_returns", fake_analyze)

    run = {"spec_id": SPEC_ID, "phase": "validation", "reports_dir": "x", "viz_dir": "y"}
    body = {
        "indicator": {"module": "rsi", "params": {"lookback": 14}},
        "bin_mode": "fixed",
        "edges": [30, 70],
        "regime": {"module": "adx", "params": {"period": 14}, "threshold": 25, "above": True},
    }

    conditional_returns.build_conditional_returns(run, body)
    req = captured["req"]
    assert req.bin_mode == "fixed" and req.edges == (30.0, 70.0)
    assert req.regime is not None
    assert req.regime.indicator.module_name == "adx" and req.regime.threshold == 25.0 and req.regime.above is True
    # validation phase → the locked test window (start 2023-01-01; end is "now").
    assert req.start == datetime(2023, 1, 1)
    assert req.end.year >= 2023
