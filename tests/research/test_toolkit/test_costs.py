"""Unit tests for research.toolkit.costs."""
from __future__ import annotations

import numpy as np
import pytest

from research.toolkit.costs import (
    FALLBACK_POINT,
    point_size,
    recorded_spread_cost,
    round_trip_cost,
)


# ---------------------------------------------------------------------------
# round_trip_cost
# ---------------------------------------------------------------------------

class TestRoundTripCost:
    def test_spread_only(self) -> None:
        assert abs(round_trip_cost(0.9) - 0.9) < 1e-12

    def test_spread_and_slip(self) -> None:
        # spread=0.9, slip=1.0 per side → 0.9 + 2×1.0 = 2.9
        assert abs(round_trip_cost(0.9, 1.0) - 2.9) < 1e-12

    def test_positive_swap_adds_to_cost(self) -> None:
        # Long pays swap → adds to cost
        assert abs(round_trip_cost(0.9, 0.0, swap_pts=0.5) - 1.4) < 1e-12

    def test_negative_swap_reduces_cost(self) -> None:
        # Short earns carry → reduces cost
        assert abs(round_trip_cost(0.9, 0.0, swap_pts=-0.5) - 0.4) < 1e-12

    def test_full_combination(self) -> None:
        # spread=1.0, slip=0.5, swap=0.25 → 1.0 + 2×0.5 + 0.25 = 2.25
        assert abs(round_trip_cost(1.0, 0.5, swap_pts=0.25) - 2.25) < 1e-12

    def test_zero_all_is_zero(self) -> None:
        assert round_trip_cost(0.0) == 0.0


# ---------------------------------------------------------------------------
# point_size
# ---------------------------------------------------------------------------

class TestPointSize:
    def test_ndx_is_0_1(self) -> None:
        assert point_size("NDX") == 0.1

    def test_sp500_is_0_1(self) -> None:
        assert point_size("SP500") == 0.1

    def test_xauusd_is_0_01(self) -> None:
        assert point_size("XAUUSD") == 0.01

    def test_xagusd_is_0_001(self) -> None:
        assert point_size("XAGUSD") == 0.001

    def test_usdjpy_is_0_001(self) -> None:
        assert point_size("USDJPY") == 0.001

    def test_gbpjpy_is_0_001(self) -> None:
        assert point_size("GBPJPY") == 0.001

    def test_unknown_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="UNKNOWN"):
            point_size("UNKNOWN")

    def test_wrong_catalog_values_not_used(self) -> None:
        """NDX and SP500 MUST be 0.1, NOT the catalog's wrong 0.01."""
        assert FALLBACK_POINT["NDX"] == 0.1, (
            "NDX POINT must be 0.1 (live-probed from 54.2M ticks). "
            "catalog.parquet says 0.01 — that is WRONG and would under-charge spread 10×."
        )
        assert FALLBACK_POINT["SP500"] == 0.1, (
            "SP500 POINT must be 0.1 (live-probed). "
            "catalog.parquet says 0.01 — that is WRONG."
        )


# ---------------------------------------------------------------------------
# recorded_spread_cost
# ---------------------------------------------------------------------------

class TestRecordedSpreadCost:
    def test_ndx_spread_no_slip(self) -> None:
        # MT5 integer spread = 9 points, NDX POINT=0.1 → cost = 9×0.1 = 0.9
        spread_col = np.array([9.0])
        costs = recorded_spread_cost(spread_col, "NDX", slip_pts=0.0)
        np.testing.assert_allclose(costs, [0.9], atol=1e-12)

    def test_ndx_spread_with_slip(self) -> None:
        # spread=9 pts, slip=0.5 per side → 9×0.1 + 2×0.5 = 0.9 + 1.0 = 1.9
        spread_col = np.array([9.0])
        costs = recorded_spread_cost(spread_col, "NDX", slip_pts=0.5)
        np.testing.assert_allclose(costs, [1.9], atol=1e-12)

    def test_multi_bar_array(self) -> None:
        # spreads = [9, 10], POINT=0.1, slip=0.5 → [1.9, 2.0]
        spread_col = np.array([9.0, 10.0])
        costs = recorded_spread_cost(spread_col, "NDX", slip_pts=0.5)
        np.testing.assert_allclose(costs, [1.9, 2.0], atol=1e-12)

    def test_xauusd_point(self) -> None:
        # spread=5 MT5 pts, XAUUSD POINT=0.01 → 5×0.01 = 0.05
        spread_col = np.array([5.0])
        costs = recorded_spread_cost(spread_col, "XAUUSD", slip_pts=0.0)
        np.testing.assert_allclose(costs, [0.05], atol=1e-12)

    def test_integer_input_cast_to_float64(self) -> None:
        spread_col = np.array([9, 10], dtype=np.int32)
        costs = recorded_spread_cost(spread_col, "NDX")
        assert costs.dtype == np.float64
