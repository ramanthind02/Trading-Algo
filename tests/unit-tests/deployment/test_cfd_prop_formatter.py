"""Smoke tests for the ``cfd_prop`` branch of the live-forecast formatters.

These exercise :func:`scripts.enigma_live_forecast.format_console_output`
and :func:`scripts.enigma_live_forecast.format_telegram_message` for the
new CFD prop-firm profile so accidental regressions in the preview path
are caught without needing a TWS / MT5 connection.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.enigma_live_forecast import (
    format_console_output,
    format_telegram_message,
)


def _sample_cfd_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": ["ES", "NQ", "GC", "SI"],
        "forecast": [+8.1, -3.2, +1.4, +0.0],
        "position_fraction": [+0.10, -0.05, +0.02, +0.0],
        "position_pct": [+10.0, -5.0, +2.0, +0.0],
    })


def test_format_console_output_cfd_prop_renders_without_capital() -> None:
    df = _sample_cfd_frame()
    out = format_console_output(df, df, capital=None, profile="cfd_prop")
    # No NaNs, no Python errors, contains all four tickers and the per-account note.
    assert "CFD PROP" in out
    for tkr in ("ES", "NQ", "GC", "SI"):
        assert tkr in out
    assert "per-account" in out.lower()
    # Must NOT print a global TOTAL $ line for cfd_prop.
    assert "TOTAL" not in out


def test_format_telegram_message_cfd_prop_uses_target_alloc_block() -> None:
    df = _sample_cfd_frame()
    out = format_telegram_message(df, capital=None, profile="cfd_prop")
    assert "*ENIGMA ALGOS FORECAST -- CFD PROP*" in out
    assert "TARGET ALLOCATIONS" in out
    assert "per-MT5-account" in out
    for tkr in ("ES", "NQ", "GC", "SI"):
        assert tkr in out
    # Must NOT include the futures or ETF blocks.
    assert "micro futures" not in out
    assert "fractional shares" not in out


def test_format_console_output_futures_prop_still_works() -> None:
    """Back-compat: the existing prop/futures_prop branch is unchanged."""
    df = pd.DataFrame({
        "ticker": ["ES"],
        "forecast": [+5.0],
        "position_pct": [+10.0],
        "target_dollars": [+5000.0],
        "contracts_fractional": [+0.5],
        "contracts_whole": [1],
        "contract_symbol": ["MES"],
        "futures_price": [5000.0],
    })
    out = format_console_output(df, df, capital=50000.0, profile="futures_prop")
    assert "PROP (MICRO FUTURES)" in out
    assert "ES" in out
    # Same with the legacy alias name.
    legacy = format_console_output(df, df, capital=50000.0, profile="prop")
    assert "PROP (MICRO FUTURES)" in legacy


def test_format_telegram_message_futures_prop_alias_back_compat() -> None:
    df = pd.DataFrame({
        "ticker": ["ES"],
        "forecast": [+5.0],
        "target_dollars": [+5000.0],
        "contracts_fractional": [+0.5],
        "contracts_whole": [1],
        "contract_symbol": ["MES"],
        "futures_price": [5000.0],
    })
    legacy = format_telegram_message(df, capital=50000.0, profile="prop")
    new = format_telegram_message(df, capital=50000.0, profile="futures_prop")
    # Both branches render the same positions block.
    assert "POSITIONS" in legacy
    assert "POSITIONS" in new


def test_cfd_prop_config_skeleton_loads_and_declares_supported_tickers() -> None:
    """The bundled config exists and declares only the 4 supported CFD tickers."""
    cfg_path = Path(__file__).resolve().parents[3] / "configs" / "live_forecast_config_cfd_prop.json"
    assert cfg_path.exists(), f"config skeleton missing: {cfg_path}"
    with open(cfg_path) as f:
        cfg = json.load(f)
    assert cfg["portfolio"]["vault_root"] == "vault_cfd_prop"
    assert set(cfg["tradeable_tickers"]) == {"ES", "NQ", "GC", "SI"}
    # MT5 symbol mapping present for all four.
    for tkr in ("ES", "NQ", "GC", "SI"):
        assert "mt5_symbol" in cfg["instruments"][tkr], f"missing mt5_symbol for {tkr}"
    # CFD-specific block exists with safe defaults.
    assert cfg["mt5"]["sizing_basis"] in ("equity", "balance")
    assert cfg["mt5"]["abort_if_unmanaged_position"] is True
    # Default-on-timeout is "cancel" in v1 (safest); user can flip to "approve".
    assert cfg["execution"]["default_on_timeout"] == "cancel"
