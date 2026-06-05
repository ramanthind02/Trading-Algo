from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from research.feature.binning.config import BinningResearchConfig
from research.feature.binning.pipeline import _attach_investigation_strategy_returns
from research.feature.binning.transforms import summarize_bins_by_metrics
from lib.core.enums import Ticker, TimeFrame


def _panel_frame() -> pd.DataFrame:
    index = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"])
    return pd.DataFrame(
        {
            "feature": [0.1, 0.2, 0.3, 0.4],
            "target": [0.01, -0.02, 0.03, 0.0],
            "ticker": ["ES", "ES", "ES", "ES"],
            "bin_index": [0, 0, 1, 1],
            "param_combo_label": ["period_14"] * 4,
        },
        index=index,
    )


def test_summarize_bins_uses_strategy_return_on_active_days_only() -> None:
    panel = _panel_frame()
    panel["strategy_signal"] = [1.0, 0.0, 1.0, 1.0]
    panel["strategy_return"] = panel["strategy_signal"] * panel["target"]

    metrics = summarize_bins_by_metrics(
        panel,
        timeframe=TimeFrame.D,
        return_col="strategy_return",
        active_signal_only=True,
    )

    bin0 = metrics.loc[metrics["bin_index"] == 0].iloc[0]
    assert bin0["n_obs"] == 1
    assert bin0["mean_return"] == pytest.approx(0.01)

    bin1 = metrics.loc[metrics["bin_index"] == 1].iloc[0]
    assert bin1["n_obs"] == 2
    assert bin1["mean_return"] == pytest.approx((0.03 + 0.0) / 2)


def test_attach_investigation_strategy_returns_merges_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = BinningResearchConfig(
        enabled=True,
        tickers=[Ticker.ES],
        start=datetime(2020, 1, 1),
        end=datetime(2020, 2, 1),
        timeframe=TimeFrame.D,
        bias_spec={"module_name": "atr", "timeframes": [TimeFrame.D], "params": {"period": 14}},
        target_col="log_return_ewsd",
        reports_dir=Path("."),
        strategy_bias_spec={
            "module_name": "cumulative_rsi_signal",
            "timeframes": [TimeFrame.D],
            "params": {"lookback": 3},
        },
    )
    features = pd.DataFrame(
        {"cumrsi": [1.0, -1.0], "ticker": ["ES", "ES"]},
        index=pd.to_datetime(["2020-01-02", "2020-01-03"]),
    )

    monkeypatch.setattr(
        "research.feature.binning.pipeline.extract_features_for_bias_node",
        lambda **_kwargs: (features, pd.DataFrame()),
    )

    attached = _attach_investigation_strategy_returns(_panel_frame().iloc[:2], config)

    assert "strategy_signal" in attached.columns
    assert "strategy_return" in attached.columns
    assert attached["strategy_return"].iloc[0] == pytest.approx(0.01)
    assert attached["strategy_return"].iloc[1] == pytest.approx(0.02)
