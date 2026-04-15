"""Unit tests for frozen-signal walkforward evaluator."""
from __future__ import annotations

import pandas as pd
import pytest

from feature_research.core_helpers import combo_key
from utils.evaluation.walkforward.evaluators import build_signed_signal_walkforward_evaluator


def test_signed_signal_evaluator_returns_signal_times_target() -> None:
    index = pd.date_range("2020-01-01", periods=5, freq="D")
    signal = pd.Series([1.0, -1.0, 0.5, 0.0, 2.0], index=index, name="signal")
    target = pd.Series([0.01, 0.02, -0.03, 0.04, 0.05], index=index, name="target")
    combo_signal_target = {
        combo_key({"lookback": 5}): pd.DataFrame({"signal": signal, "target": target}),
    }

    evaluator = build_signed_signal_walkforward_evaluator(combo_signal_target)
    out = evaluator(pd.DataFrame({"close": target}, index=index), target, {"lookback": 5})

    pd.testing.assert_series_equal(out, signal.mul(target))


def test_signed_signal_evaluator_rejects_selected_bin() -> None:
    index = pd.date_range("2020-01-01", periods=3, freq="D")
    signal = pd.Series([1.0, 1.0, 1.0], index=index, name="signal")
    target = pd.Series([0.01, 0.01, 0.01], index=index, name="target")
    combo_signal_target = {
        combo_key({"lookback": 5}): pd.DataFrame({"signal": signal, "target": target}),
    }

    evaluator = build_signed_signal_walkforward_evaluator(combo_signal_target)

    with pytest.raises(ValueError, match="selected_bin"):
        evaluator(
            pd.DataFrame({"close": target}, index=index),
            target,
            {"lookback": 5, "selected_bin": 1},
        )


def test_signed_signal_evaluator_uses_precomputed_returns_column() -> None:
    index = pd.date_range("2020-01-01", periods=4, freq="D")
    returns = pd.Series([0.01, -0.02, 0.03, 0.04], index=index, name="returns")
    combo_signal_target = {
        combo_key({"lookback": 7}): pd.DataFrame({"returns": returns}),
    }

    evaluator = build_signed_signal_walkforward_evaluator(combo_signal_target)
    out = evaluator(pd.DataFrame({"close": returns}, index=index), returns, {"lookback": 7})

    pd.testing.assert_series_equal(out, returns)
