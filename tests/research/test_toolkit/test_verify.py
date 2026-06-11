"""Unit tests for research.toolkit.verify."""
from __future__ import annotations

import pandas as pd
import pytest

from research.toolkit.verify import assert_no_lookahead


def _clean_trades(exit_shift: float = 0.0) -> pd.DataFrame:
    """Three synthetic trades where net_pts == (exit-entry)*dir - cost_pts.

    ``exit_shift`` deliberately corrupts the exit price to trigger the assert.
    """
    # Trade 1: long, entry=100, exit=105, cost=1 → net = (105-100)*1 - 1 = 4
    # Trade 2: short, entry=200, exit=195, cost=1 → net = (200-195)*1 - 1 = 4
    # Trade 3: long, entry=150, exit=155, cost=1 → net = (155-150)*1 - 1 = 4
    return pd.DataFrame(
        {
            "dir":      [1, -1, 1],
            "entry":    [100.0, 200.0, 150.0],
            "exit":     [105.0 + exit_shift, 195.0, 155.0],
            "cost_pts": [1.0, 1.0, 1.0],
            "net_pts":  [4.0, 4.0, 4.0],
        }
    )


def test_passes_on_clean_trades() -> None:
    assert_no_lookahead(_clean_trades())


def test_passes_on_empty_frame() -> None:
    empty = pd.DataFrame(
        columns=["dir", "entry", "exit", "cost_pts", "net_pts"]
    ).astype(float)
    assert_no_lookahead(empty)


def test_fails_on_shifted_exit() -> None:
    """Deliberately shifting exit by 0.1 introduces a 0.1-unit mismatch."""
    with pytest.raises(AssertionError, match="mismatch"):
        assert_no_lookahead(_clean_trades(exit_shift=0.1))


def test_custom_tolerance_tight_fails_on_tiny_shift() -> None:
    """Even a 1e-8 shift exceeds a 1e-10 tolerance."""
    with pytest.raises(AssertionError):
        assert_no_lookahead(_clean_trades(exit_shift=1e-8), tol=1e-10)


def test_custom_tolerance_loose_passes_on_tiny_shift() -> None:
    """A 1e-10 shift is within the default 1e-6 tolerance."""
    assert_no_lookahead(_clean_trades(exit_shift=1e-10))
