"""Unit tests for conditional-returns binning (synthetic extraction, no repo data)."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

import research.feature.binning.conditional_returns as cr
from lib.core.enums import Direction, Ticker, TimeFrame


def _install_fake_extraction(monkeypatch: pytest.MonkeyPatch, *, n: int = 1000) -> None:
    """Patch the extractor so 'rsi'/'double7s'/'adx' return deterministic synthetic frames.

    Strategy is long (signal=1) only when RSI < 50; forward returns get a positive drift when
    RSI < 30 — so the lowest RSI bucket should carry the strongest mean return.
    """

    idx = pd.date_range("2010-01-01", periods=n, freq="D", tz="UTC")
    idx.name = "datetime"
    rng = np.random.default_rng(7)
    rsi = rng.uniform(0.0, 100.0, n)
    adx = rng.uniform(0.0, 60.0, n)
    signal = (rsi < 50.0).astype(float)
    log_return = rng.normal(0.0, 0.01, n) + np.where(rsi < 30.0, 0.006, 0.0)

    def fake_extract(bias_spec, ticker, start, end, **_kw):  # noqa: ANN001
        module = bias_spec["module_name"]
        if module == "rsi":
            feats = pd.DataFrame(
                {
                    "rsi_signal_D_lookback_14": rsi,
                    "ewsd_ewsd_D_period_252": rng.uniform(1.0, 2.0, n),  # must be ignored
                    "ticker": "ES",
                },
                index=idx,
            )
        elif module == "double7s":
            feats = pd.DataFrame({"double7s_signal_D_short_period_7": signal, "ticker": "ES"}, index=idx)
        elif module == "adx":
            feats = pd.DataFrame({"adx_signal_D_period_14": adx, "ticker": "ES"}, index=idx)
        else:
            raise ValueError(f"unexpected module {module}")
        targets = pd.DataFrame({"log_return": log_return, "ticker": "ES"}, index=idx)
        return feats, targets

    monkeypatch.setattr(cr, "extract_features_for_bias_node", fake_extract)
    monkeypatch.setattr(cr, "_set_feed", lambda _feed: None)


def _base_request(**overrides) -> cr.ConditionalReturnsRequest:
    defaults: dict = dict(
        tickers=(Ticker.ES,),
        timeframe=TimeFrame.D,
        start=datetime(2010, 1, 1),
        end=datetime(2013, 12, 31),
        data_feed="futures",
        direction=Direction.LONG,
        strategy=cr.IndicatorSpec("double7s", {"short_period": 7}),
        condition=cr.IndicatorSpec("rsi", {"lookback": 14}),
    )
    defaults.update(overrides)
    return cr.ConditionalReturnsRequest(**defaults)


def test_quantile_buckets_only_active_bars(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_extraction(monkeypatch)
    res = cr.analyze_conditional_returns(_base_request(bin_mode="quantile", n_bins=5))

    assert res.available and res.reason is None
    assert len(res.bins) == 5
    assert all(b.regime is None and b.ticker is None and b.n_obs > 0 for b in res.bins)
    # The strategy is active only when RSI < 50, so every bucket edge stays under ~50.
    assert all((b.hi is None) or (b.hi <= 51.0) for b in res.bins)


def test_low_rsi_bucket_has_higher_mean_return(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_extraction(monkeypatch)
    res = cr.analyze_conditional_returns(_base_request(bin_mode="quantile", n_bins=5))

    by_index = {b.bin_index: b for b in res.bins}
    assert by_index[0].mean_return > by_index[4].mean_return  # drift concentrated in low RSI


def test_fixed_edges_label_and_regime_crosscut(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_extraction(monkeypatch)
    res = cr.analyze_conditional_returns(
        _base_request(
            bin_mode="fixed",
            edges=(30.0,),
            regime=cr.RegimeSpec(cr.IndicatorSpec("adx", {"period": 14}), threshold=25.0, above=True),
        )
    )

    assert res.available
    assert {b.regime for b in res.bins} == {"on", "off"}
    assert res.regime_label is not None and "adx" in res.regime_label
    assert any(b.label.startswith("< 30") for b in res.bins)


def test_per_ticker_tags_each_bucket(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_extraction(monkeypatch)
    res = cr.analyze_conditional_returns(_base_request(bin_mode="quantile", n_bins=4, per_ticker=True))

    assert res.available
    assert all(b.ticker == "ES" for b in res.bins)


def test_fixed_mode_requires_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_extraction(monkeypatch)
    res = cr.analyze_conditional_returns(_base_request(bin_mode="fixed", edges=()))

    assert not res.available
    assert res.reason and "edge" in res.reason.lower()
