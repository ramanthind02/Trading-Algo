from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import cast

import pandas as pd

from feature_research.config import ResearchConfig
from utils.evaluation.walkforward.research_data import load_signed_signal_research_data


def test_load_signed_signal_research_data_adds_returns_column_with_parallel_workers(
    monkeypatch,
) -> None:
    index = pd.date_range("2020-01-01", periods=4, freq="D")
    combos = [{"lookback": 3}, {"lookback": 5}]

    def _fake_load_features_for_combo(
        single_combo_spec: dict[str, object],
        config: object,
        candles_override: pd.DataFrame | None = None,
    ) -> tuple[pd.Series, pd.Series, str, pd.Series]:
        del config, candles_override
        lookback = int(single_combo_spec["params"]["lookback"])
        signal = pd.Series([float(lookback)] * len(index), index=index, name=f"signal_{lookback}")
        target = pd.Series([0.01, -0.02, 0.03, 0.04], index=index, name="target")
        ticker_s = pd.Series(["ES"] * len(index), index=index, dtype=str)
        return signal, target, signal.name, ticker_s

    monkeypatch.setattr(
        "utils.evaluation.walkforward.research_data.load_features_for_combo",
        _fake_load_features_for_combo,
    )

    config = cast(
        ResearchConfig,
        SimpleNamespace(
            n_jobs=2,
            start=datetime(2020, 1, 1),
            end=datetime(2020, 1, 4),
        ),
    )
    expanded_specs = [
        {"module_name": "rsi", "params": combo, "timeframes": []}
        for combo in combos
    ]

    result = load_signed_signal_research_data(config, expanded_specs)

    assert result.successful_param_grid == combos
    assert result.reference_index is not None
    assert result.reference_target_series is not None
    for combo_data in result.combo_signal_target.values():
        assert "returns" in combo_data.columns
        assert "ticker" in combo_data.columns
        expected = combo_data["signal"] * combo_data["target"]
        pd.testing.assert_series_equal(combo_data["returns"], expected.rename("returns"))
