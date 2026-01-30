import numpy as np
import pandas as pd

from utils.fast_volatility import compute_ewsd_annualized_from_closes
from ensemble.portfolio_tester import (
    calculate_log_returns_from_candles,
    calculate_baseline_returns,
)


def test_compute_ewsd_annualized_from_closes_flat_series() -> None:
    """Flat price series should fall back to the default 20% annual volatility."""
    closes = np.full(100, 100.0, dtype=np.float64)
    vol = compute_ewsd_annualized_from_closes(closes)
    assert np.isfinite(vol)
    # exact default from implementation
    assert vol == pytest.approx(0.20, rel=1e-12)


def test_compute_ewsd_annualized_from_closes_non_trivial() -> None:
    """Non-trivial random series should produce a positive volatility."""
    rng = np.random.default_rng(123)
    prices = 100.0 * np.exp(rng.normal(0.0, 0.01, size=252).cumsum())
    vol = compute_ewsd_annualized_from_closes(prices)
    assert np.isfinite(vol)
    assert vol > 0.0


def test_portfolio_tester_log_and_baseline_returns_shape() -> None:
    """Smoke-test log-return and baseline helpers on simple multi-ticker data."""
    dates = pd.date_range("2020-01-01", periods=10, freq="B")
    rows = []
    for ticker in ("ES", "NQ"):
        price = 100.0
        for dt in dates:
            price *= 1.01
            rows.append(
                {
                    "datetime": dt,
                    "open": price / 1.001,
                    "high": price * 1.002,
                    "low": price * 0.998,
                    "close": price,
                    "volume": 1_000_000.0,
                    "ticker": ticker,
                }
            )

    candles = pd.DataFrame(rows)

    log_ret = calculate_log_returns_from_candles(candles)
    assert isinstance(log_ret, pd.Series)
    assert not log_ret.empty

    baseline_eq = calculate_baseline_returns(candles, equal_weight=True)
    baseline_single = calculate_baseline_returns(candles, equal_weight=False)

    assert isinstance(baseline_eq, pd.Series)
    assert isinstance(baseline_single, pd.Series)
    assert not baseline_eq.empty
    assert not baseline_single.empty

