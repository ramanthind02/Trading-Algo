"""Issue 4 — the feature-research VALIDATION lane honours the spec's ExecutionSpec.

``research.feature.pipelines._shared._select_phase_pnl_engine`` builds the realistic
multi-ticker Nautilus lane for the validation/OOS phases. These tests assert it reads
the three ``ResearchConfig`` execution fields (set from the spec by
``research.spec.adapter.to_feature_config``) into the engine, while the DEFAULT config
reproduces today's proven behavior — the rollover swap-avoidance overlay + MARKET orders
— so the parity gate and canonical configs are unaffected.

Guarded behind ``importorskip("nautilus_trader")``: building the engine constructs a
``NautilusPnLEngine`` (a nautilus_trader import). No data is loaded — only the engine's
configured policies are inspected (``.base`` is the wrapped single-instrument engine).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from research.feature.pipelines._shared import _select_phase_pnl_engine


def _config(**overrides) -> SimpleNamespace:
    """A minimal stand-in for ResearchConfig.

    ``_select_phase_pnl_engine`` reads every field via ``getattr(config, ..., default)``,
    so a namespace exposing only what a test overrides exercises the real defaults
    (which mirror ResearchConfig's defaults) for the rest.
    """
    base = dict(realistic_phases=("validation", "oos"))
    base.update(overrides)
    return SimpleNamespace(**base)


def test_default_config_keeps_rollover_overlay_and_market_orders():
    """Today's behavior: overnight default → ROLLOVER_FLATTEN_REENTER + MARKET + cross_after."""
    pytest.importorskip("nautilus_trader")
    from research.portfolio.pnl.nautilus_engine import (
        ExecutionPolicy,
        ExecutionWindowPolicy,
    )

    engine = _select_phase_pnl_engine("validation", _config())
    base = engine.base  # MultiTickerNautilusPnLEngine wraps the single-instrument engine
    assert base.window_policy is ExecutionWindowPolicy.ROLLOVER_FLATTEN_REENTER
    assert base.execution_policy is ExecutionPolicy.MARKET_ON_OPEN
    # cross_after default cutoff is enabled (NOT pure-passive carry).
    assert base.cross_after.is_enabled()
    # The proven defaults are preserved verbatim.
    assert base.measure_spread is True
    assert base.rollover_minute == 0
    assert base.rollover_half_width_min == 15


def test_intraday_holding_yields_intraday_open_to_close():
    pytest.importorskip("nautilus_trader")
    from research.portfolio.pnl.nautilus_engine import ExecutionWindowPolicy

    engine = _select_phase_pnl_engine(
        "validation", _config(execution_holding="intraday")
    )
    assert engine.base.window_policy is ExecutionWindowPolicy.INTRADAY_OPEN_TO_CLOSE


def test_limit_at_touch_entry_yields_limit_execution_policy():
    pytest.importorskip("nautilus_trader")
    from research.portfolio.pnl.nautilus_engine import ExecutionPolicy

    engine = _select_phase_pnl_engine(
        "validation", _config(execution_entry_policy="limit_at_touch")
    )
    assert engine.base.execution_policy is ExecutionPolicy.LIMIT_AT_TOUCH


def test_carry_unfilled_limit_yields_pure_passive_cross_after():
    pytest.importorskip("nautilus_trader")

    engine = _select_phase_pnl_engine(
        "validation", _config(execution_unfilled_limit="carry")
    )
    assert engine.base.cross_after.session_fraction == pytest.approx(1.0)
    assert not engine.base.cross_after.is_enabled()


def test_overnight_does_not_select_close_to_close():
    """Regression guard: overnight must keep the rollover overlay, NOT switch to
    CLOSE_TO_CLOSE (which would drop the swap-avoidance overlay and break the
    validation defaults)."""
    pytest.importorskip("nautilus_trader")
    from research.portfolio.pnl.nautilus_engine import ExecutionWindowPolicy

    engine = _select_phase_pnl_engine(
        "validation", _config(execution_holding="overnight")
    )
    assert engine.base.window_policy is not ExecutionWindowPolicy.CLOSE_TO_CLOSE
    assert engine.base.window_policy is ExecutionWindowPolicy.ROLLOVER_FLATTEN_REENTER


def test_non_realistic_phase_returns_vectorized_lane_none():
    """A phase outside realistic_phases (and no force) → None (frozen vectorized lane)."""
    engine = _select_phase_pnl_engine("validation", _config(realistic_phases=()))
    assert engine is None
