"""Data-backed regression tests for the RATIO repoint.

These exercise the real persisted ``D_{T}_ratio.parquet`` store and the two
repointed %-math consumers (EWSD sigma denominator, portfolio IDM returns), plus
the prop-firm config loader fix and the scoped reference-series validator. Each
test skips cleanly when the required repo data is absent so the suite stays green
on a fresh checkout without the data tree.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    return next(
        (p for p in here.parents if (p / ".git").exists() or (p / "AGENTS.md").exists()),
        here.parents[2],
    )


OHLC = _repo_root() / "data" / "ohlc_data"
STK = _repo_root() / "data" / "stock_data"
ETF_FOR = {"ES": "SPY", "NQ": "QQQ", "GC": "GLD", "SI": "SLV", "CL": "USO"}


def _close(path: Path) -> pd.Series:
    s = pd.read_parquet(path)["close"].astype("float64")
    s.index = pd.to_datetime(s.index).normalize()
    return s.sort_index()


def _logret(s: pd.Series) -> pd.Series:
    s = s.where(s > 0)
    return np.log(s).diff()


def _corr_to_etf(series: pd.Series, etf: pd.Series) -> float:
    j = pd.concat([_logret(series).rename("a"), _logret(etf).rename("b")], axis=1).dropna()
    return float(j["a"].corr(j["b"])) if len(j) >= 60 else float("nan")


def _etf(sym: str) -> pd.Series:
    p = STK / sym[0] / sym / f"D_CAP_{sym}.parquet"
    return _close(p) if p.exists() else pd.Series(dtype=float)


def _require_ratio(ticker: str) -> Path:
    p = OHLC / ticker / f"D_{ticker}_ratio.parquet"
    if not p.exists():
        pytest.skip(f"ratio store absent for {ticker}: {p}")
    return p


# ── ratio store invariants ───────────────────────────────────────────────────
@pytest.mark.parametrize("ticker", ["ES", "NQ", "GC", "SI"])
def test_ratio_has_no_negative_prices(ticker: str) -> None:
    """No ratio series introduces a negative (CL is the documented exception)."""
    ratio = _close(_require_ratio(ticker))
    assert float(ratio.min()) >= 0.0, f"{ticker} ratio min < 0"


def test_cl_ratio_negative_is_genuine_source_print() -> None:
    """CL ratio min<0 is inherited from the genuine WTI Apr-2020 unadjusted print."""
    ratio = _close(_require_ratio("CL"))
    unadj_path = OHLC / "CL" / "D_CL_unadj.parquet"
    if not unadj_path.exists():
        pytest.skip("CL unadjusted absent")
    unadj = _close(unadj_path)
    # the negative is present in the *source*, so it is not a method failure
    assert float(unadj.min()) < 0.0
    assert float(ratio.min()) < 0.0


@pytest.mark.parametrize("ticker", ["ES", "NQ", "GC", "SI", "CL"])
def test_ratio_corr_to_etf_at_least_additive(ticker: str) -> None:
    """Ratio %-return corr-to-ETF must be >= additive (the whole point)."""
    etf = _etf(ETF_FOR[ticker])
    if etf.empty:
        pytest.skip(f"ETF {ETF_FOR[ticker]} absent")
    add = _close(OHLC / ticker / f"D_{ticker}.parquet")
    ratio = _close(_require_ratio(ticker))
    add_corr = _corr_to_etf(add, etf)
    ratio_corr = _corr_to_etf(ratio, etf)
    assert not np.isnan(ratio_corr)
    assert ratio_corr >= add_corr - 0.01, (
        f"{ticker}: ratio_corr {ratio_corr:.3f} < additive {add_corr:.3f}"
    )


# ── EWSD repoint ─────────────────────────────────────────────────────────────
def test_ewsd_sigma_denominator_reads_ratio() -> None:
    """The EWSD sigma-denominator loader returns the RATIO series when present."""
    _require_ratio("ES")
    from lib.core.enums import Ticker
    from nodes.volatility.ewsd.ewsd import _load_unadj_close_series

    loaded = _load_unadj_close_series(Ticker.ES)
    assert loaded is not None and not loaded.empty
    ratio = _close(OHLC / "ES" / "D_ES_ratio.parquet")
    # min ties out to the ratio file (NOT the unadjusted/additive file)
    assert np.isclose(float(loaded.min()), float(ratio.min()), rtol=1e-4)


# ── portfolio IDM returns repoint ────────────────────────────────────────────
def test_portfolio_returns_uses_ratio_close() -> None:
    """_ticker_return_series uses ratio-close %-returns when candles carry a ticker."""
    _require_ratio("ES")
    from ensemble.portfolio_impl.portfolio_returns import _ticker_return_series

    ratio = _close(OHLC / "ES" / "D_ES_ratio.parquet")
    # candles carry the ADDITIVE close; the function should ignore it and use ratio
    additive = _close(OHLC / "ES" / "D_ES.parquet")
    candles = pd.DataFrame({
        "datetime": additive.index,
        "close": additive.to_numpy(),
        "ticker": "ES",
    })
    out = _ticker_return_series(candles)
    expected = ratio.reindex(pd.DatetimeIndex(additive.index).normalize()).pct_change().dropna()
    # compare on the overlap
    j = pd.concat([out.rename("o"), expected.rename("e")], axis=1).dropna()
    assert len(j) > 100
    assert np.allclose(j["o"].to_numpy(), j["e"].to_numpy(), atol=1e-9)


def test_portfolio_returns_falls_back_without_ticker() -> None:
    """No ticker column -> falls back to additive candle close (current behaviour)."""
    from ensemble.portfolio_impl.portfolio_returns import _ticker_return_series

    dates = pd.bdate_range("2020-01-01", periods=10)
    candles = pd.DataFrame({"datetime": dates, "close": np.arange(10) + 100.0})
    out = _ticker_return_series(candles)
    assert not out.empty
    assert np.allclose(out.to_numpy(), (np.arange(10) + 100.0)[1:] / (np.arange(10) + 100.0)[:-1] - 1)


# ── scoped reference-series validator ────────────────────────────────────────
def test_validate_consumed_series_passes_for_traded_universe() -> None:
    _require_ratio("ES")
    from data_platform.data_quality.reference_series import validate_consumed_series

    findings = validate_consumed_series(["ES", "NQ", "GC"])
    fails = [f for f in findings if f.status == "FAIL"]
    assert not fails, f"unexpected FAILs: {[(f.series, f.detail) for f in fails]}"


def test_validate_consumed_series_cl_is_warn_not_fail() -> None:
    _require_ratio("CL")
    from data_platform.data_quality.reference_series import validate_consumed_series

    findings = validate_consumed_series(["CL"])
    # genuine-source negative -> WARN, never FAIL
    assert not [f for f in findings if f.status == "FAIL"]
    assert any(f.status == "WARN" for f in findings)


# ── prop-firm config loader fix ──────────────────────────────────────────────
def test_prop_firm_loader_succeeds() -> None:
    """The replace(oos, val_end=...) fix lets the prop loader build without TypeError."""
    from research.portfolio.config import load_prop_firm_portfolio_research_config

    cfg = load_prop_firm_portfolio_research_config()
    assert [t.name for t in cfg.tickers] == ["ES", "NQ", "GC"]
    # test_end property reflects the updated val_end
    assert cfg.oos_window.test_end == cfg.oos_window.val_end
