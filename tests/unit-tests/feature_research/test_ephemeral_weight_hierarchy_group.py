"""Tests for ticker-driven vault weight_hierarchy_group selection."""

from __future__ import annotations

from feature_research.config import ephemeral_weight_hierarchy_group_for_tickers
from utils.core.enums import Ticker


def test_cl_only_tickers_default_to_crude_oil_mr() -> None:
    assert (
        ephemeral_weight_hierarchy_group_for_tickers(
            (Ticker.CL,),
            configured_group="mean_reversion_indices",
        )
        == "crude_oil_mr"
    )


def test_cl_only_preserves_explicit_cl_breakout_sleeve() -> None:
    assert (
        ephemeral_weight_hierarchy_group_for_tickers(
            (Ticker.CL,),
            configured_group="cl_breakout",
        )
        == "cl_breakout"
    )


def test_gc_only_preserves_explicit_gc_breakout_sleeve() -> None:
    assert (
        ephemeral_weight_hierarchy_group_for_tickers(
            (Ticker.GC,),
            configured_group="gc_breakout",
        )
        == "gc_breakout"
    )


def test_equity_only_tickers_keep_mean_reversion_indices() -> None:
    assert (
        ephemeral_weight_hierarchy_group_for_tickers(
            (Ticker.ES, Ticker.NQ),
            configured_group="mean_reversion_indices",
        )
        == "mean_reversion_indices"
    )


def test_mixed_tickers_keep_configured_group() -> None:
    assert (
        ephemeral_weight_hierarchy_group_for_tickers(
            (Ticker.ES, Ticker.GC),
            configured_group="mean_reversion_indices",
        )
        == "mean_reversion_indices"
    )
