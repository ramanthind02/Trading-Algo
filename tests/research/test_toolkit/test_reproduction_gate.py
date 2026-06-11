"""Reproduction gate: toolkit must reproduce LAFO experiment numbers.

Gate achieved: COMPONENT PARITY
  (1) toolkit.bars.read_m1 + filter_session(RTH) + resample_bars("M15") produces
      a frame with identical OHLC/spread/sec/day values to engine.load_m1 for the
      same bounded symbol/period.
  (2) toolkit.metrics.summary on the same trade series produces the same Sharpe as
      engine.metrics (to 2-decimal rounding agreement), and the standalone
      toolkit.metrics.daily_sharpe formula matches exactly.
  (3) The CSV-documented headline number (Sharpe 1.01 for M15 thr=2.5% revert,
      2018-2026) is verified as present in the CSV with its parameter key columns.

Why component parity and not full-number reproduction:
  - The task requires "keep reads bounded (one or two years)".
  - The CSV value (1.01) covers 2018-2026; a 2-year bounded simulation gives a
    different Sharpe.
  - Component parity demonstrates that toolkit.bars and toolkit.metrics implement
    the SAME formulas as the engine — the only missing piece is the engine's
    trading logic (KAMA indicators + simulate()), which is intentionally NOT part
    of the toolkit (the toolkit provides primitives, not strategy engines).

All M1 reads in this file are bounded to 2018-2019 (two year-partitions only).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[3]
_NDX_M1_DIR = _REPO_ROOT / "data" / "mt5_data" / "NDX" / "bars_M1"
_LAFO_OUTPUTS = _REPO_ROOT / "research" / "experiments" / "lafo_kama_mr" / "outputs"

# Skip the whole module when NDX M1 data is absent (CI without data mount)
pytestmark = pytest.mark.skipif(
    not (_NDX_M1_DIR / "year=2018").exists(),
    reason="NDX M1 data (data/mt5_data/NDX/bars_M1/year=2018) not available locally",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_toolkit_m15_rth(year_end: int = 2019) -> pd.DataFrame:
    """Load NDX M15 RTH via toolkit pipeline, bounded to 2018–year_end."""
    from research.toolkit.bars import filter_session, read_m1, resample_bars
    from research.toolkit.sessions import Session

    m1 = read_m1("NDX", year_end=year_end)
    rth = filter_session(m1, Session.RTH)
    return resample_bars(rth, "M15")


def _load_engine_m15_rth(year_end: int = 2019) -> pd.DataFrame:
    """Load NDX M15 RTH via lafo engine, bounded to 2018–year_end."""
    import os

    # engine.load_m1 uses glob with relative paths → must run from repo root
    old_cwd = os.getcwd()
    try:
        os.chdir(_REPO_ROOT)
        from research.experiments.lafo_kama_mr.engine import load_m1, resample

        m1_engine = load_m1("NDX", 2018, year_end, "rth")
        return resample(m1_engine, "M15")
    finally:
        os.chdir(old_cwd)


# ---------------------------------------------------------------------------
# Test 1: CSV row exists and documents Sharpe = 1.01
# ---------------------------------------------------------------------------

def test_csv_headline_row_is_documented() -> None:
    """The deepen_A CSV must contain the M15 thr=2.5% sm=3.5 revert row with sharpe=1.01.

    This is the headline number from FINDINGS.md §3: 'Best cell: M15, thr 2.5%,
    long, revert exit → frictionless Sharpe 1.01'.
    """
    csv_path = _LAFO_OUTPUTS / "deepen_A_long_plateau.csv"
    assert csv_path.exists(), f"LAFO outputs CSV not found: {csv_path}"

    df = pd.read_csv(csv_path)
    row = df[
        (df["tf"] == "M15")
        & (np.abs(df["thr"] - 0.025) < 1e-9)
        & (np.abs(df["sm"] - 3.5) < 1e-9)
        & (df["exit"] == "revert")
    ]
    assert len(row) == 1, f"Expected exactly 1 matching row; got {len(row)}"
    csv_sharpe = float(row["sharpe"].iloc[0])
    assert abs(csv_sharpe - 1.01) < 1e-9, (
        f"CSV Sharpe for M15 thr=2.5% sm=3.5 revert = {csv_sharpe}, expected 1.01"
    )
    assert int(row["trades"].iloc[0]) == 146, (
        f"CSV trade count = {int(row['trades'].iloc[0])}, expected 146"
    )


# ---------------------------------------------------------------------------
# Test 2: frame parity — toolkit bars == engine bars (bounded 2018-2019)
# ---------------------------------------------------------------------------

def test_m15_rth_frame_parity_toolkit_vs_engine() -> None:
    """toolkit bars pipeline produces identical OHLC/spread/sec/day to engine.load_m1.

    Bounded to 2018-2019 (2 year-partitions).  Asserts:
      - identical number of rows
      - identical datetime index
      - OHLC values match to float64 precision
      - spread values match to float64 precision
      - sec (broker seconds-of-day) match exactly
    """
    m15_tk = _load_toolkit_m15_rth(year_end=2019)
    m15_eng = _load_engine_m15_rth(year_end=2019)

    assert len(m15_tk) == len(m15_eng), (
        f"Row count mismatch: toolkit={len(m15_tk)}, engine={len(m15_eng)}"
    )

    # datetime
    pd.testing.assert_series_equal(
        m15_tk["datetime"].reset_index(drop=True),
        m15_eng["datetime"].reset_index(drop=True),
        check_names=False,
        obj="datetime column",
    )

    # OHLC (float64 bit-exact when both use the same source parquet + same dtypes)
    for col in ("open", "high", "low", "close"):
        np.testing.assert_array_equal(
            m15_tk[col].to_numpy(),
            m15_eng[col].to_numpy(),
            err_msg=f"Column '{col}' mismatch between toolkit and engine",
        )

    # spread
    np.testing.assert_array_equal(
        m15_tk["spread"].to_numpy(),
        m15_eng["spread"].to_numpy(),
        err_msg="spread column mismatch",
    )

    # sec (broker seconds-of-day)
    np.testing.assert_array_equal(
        m15_tk["sec"].to_numpy(),
        m15_eng["sec"].to_numpy(),
        err_msg="sec column mismatch",
    )


# ---------------------------------------------------------------------------
# Test 3: metrics formula parity — toolkit vs engine.metrics on same trades
# ---------------------------------------------------------------------------

def test_metrics_formula_parity_on_bounded_simulation() -> None:
    """toolkit.metrics.summary matches engine.metrics on the same bounded trades.

    Runs the lafo engine simulation (KAMA indicators + event replay) on the
    toolkit-loaded M15 RTH bars (2018-2019), then computes metrics with BOTH
    engine.metrics and toolkit.metrics.summary and asserts agreement.

    This is the core reproduction gate: it proves toolkit.metrics implements
    the same daily-grid Sharpe formula as the experiment engines.
    """
    import os

    old_cwd = os.getcwd()
    try:
        os.chdir(_REPO_ROOT)

        from research.experiments.lafo_kama_mr import engine

        from research.toolkit.metrics import (
            daily_grid_returns,
            daily_sharpe,
            summary,
        )

        # ---- load via toolkit -------------------------------------------------
        m15_tk = _load_toolkit_m15_rth(year_end=2019)

        # ---- add KAMA/ATR indicators (engine function, not in toolkit) ---------
        p = engine.Params(
            tf="M15",
            session="rth",
            entry_mode="rel",
            threshold=0.025,
            exit_mode="revert",
            stop_atr_mult=3.5,
            longs=True,
            shorts=False,
        )
        bars_with_ind = engine.add_indicators(
            m15_tk, p.kama_period, p.kama_fast, p.kama_slow, p.atr_period, p.z_lookback
        )

        # ---- simulate (engine function, not in toolkit) -----------------------
        trades = engine.simulate(bars_with_ind, p)
        session_days = np.sort(bars_with_ind["day"].unique())

        # ---- engine metrics ---------------------------------------------------
        m_engine = engine.metrics(trades, session_days)

        # ---- toolkit metrics --------------------------------------------------
        m_toolkit = summary(trades, session_days)

        # Core comparison: Sharpe (rounded to 2 dp, matching engine.metrics rounding)
        assert abs(m_toolkit["sharpe"] - m_engine["sharpe"]) < 1e-9, (
            f"Sharpe mismatch: toolkit={m_toolkit['sharpe']}, "
            f"engine={m_engine['sharpe']}"
        )

        # Also verify the standalone daily_sharpe function produces the same value
        arr = daily_grid_returns(trades, session_days, "ret")
        sh_raw = daily_sharpe(arr)
        # engine.metrics rounds to 2 dp; round our raw value the same way
        assert abs(round(sh_raw, 2) - m_engine["sharpe"]) < 1e-9, (
            f"daily_sharpe raw={sh_raw:.6f}, rounded={round(sh_raw,2)}, "
            f"engine={m_engine['sharpe']}"
        )

        # Secondary sanity: trade count, PF, win% all agree
        assert m_toolkit["trades"] == m_engine["trades"], (
            f"Trade count: toolkit={m_toolkit['trades']}, engine={m_engine['trades']}"
        )

        if m_engine["PF"] != float("inf"):
            assert abs(m_toolkit["PF"] - m_engine["PF"]) < 1e-9, (
                f"PF: toolkit={m_toolkit['PF']}, engine={m_engine['PF']}"
            )

        assert abs(m_toolkit["win%"] - m_engine["win%"]) < 1e-9, (
            f"win%: toolkit={m_toolkit['win%']}, engine={m_engine['win%']}"
        )

    finally:
        os.chdir(old_cwd)
