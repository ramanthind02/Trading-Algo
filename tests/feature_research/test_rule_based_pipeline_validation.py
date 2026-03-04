from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import cast

import pandas as pd

from feature_research.config import FeatureType, OOSWindowConfig
from feature_research.in_sample.config import ResearchConfig
from feature_research.pipeline import run_validation_pipeline
from utils.core.enums import Ticker, TimeFrame
from utils.evaluation.walkforward.runner import WalkforwardRunReport


def _build_config(tmp_path: Path) -> ResearchConfig:
    return ResearchConfig(
        feature_type=FeatureType.RULE_BASED,
        tickers=[Ticker.ES],
        start=datetime(2020, 1, 1),
        end=datetime(2020, 4, 29),
        bias_spec={
            "module_name": "rsi_signal",
            "timeframes": [TimeFrame.D],
            "params": {
                "rsi_period": [2, 3],
                "oversold": 25.0,
                "overbought": 65.0,
                "strategy_mode": "long",
                "exit_policy": "threshold_or_bars",
                "exit_bars": 5,
            },
        },
        target_col="log_return",
        strategy="long",
        use_cache=True,
        populate_cache=False,
        reports_dir=tmp_path / "reports",
        validation_window=OOSWindowConfig(
            train_start=datetime(2020, 1, 1),
            train_end=datetime(2020, 2, 10),
            test_start=datetime(2020, 2, 11),
            test_end=datetime(2020, 3, 10),
        ),
        top_k=1,
        objective_metric_name="mean_return",
        output_root=tmp_path / "shared_results",
    )


def _series_for_combo(params: dict[str, object]) -> tuple[pd.Series, pd.Series, str]:
    index = pd.date_range("2020-01-01", periods=120, freq="D")
    period = int(cast(int, params["rsi_period"]))
    feature = pd.Series([float(period)] * len(index), index=index, name=f"feature_{period}")
    target = pd.Series([0.01] * len(index), index=index, name="target")
    return feature, target, feature.name


def _mock_candles_for_config() -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=120, freq="D")
    return pd.DataFrame(
        {
            "datetime": index,
            "open": [100.0] * len(index),
            "high": [101.0] * len(index),
            "low": [99.0] * len(index),
            "close": [100.5] * len(index),
            "volume": [1000.0] * len(index),
            "ticker": ["ES"] * len(index),
            "timeframe": [TimeFrame.D] * len(index),
        },
        index=index,
    )


def test_run_rule_based_validation_pipeline_returns_report_and_writes_artifacts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _build_config(tmp_path)

    monkeypatch.setattr(
        "feature_research.pipelines.validation.populate_cache_if_needed",
        lambda _config: None,
    )
    monkeypatch.setattr(
        "feature_research.pipelines.validation.expand_bias_specs",
        lambda _bias_spec: [
            {
                "module_name": "rsi_signal",
                "timeframes": [TimeFrame.D],
                "params": {
                    "rsi_period": 2,
                    "oversold": 25.0,
                    "overbought": 65.0,
                    "strategy_mode": "long",
                    "exit_policy": "threshold_or_bars",
                    "exit_bars": 5,
                },
            },
            {
                "module_name": "rsi_signal",
                "timeframes": [TimeFrame.D],
                "params": {
                    "rsi_period": 3,
                    "oversold": 25.0,
                    "overbought": 65.0,
                    "strategy_mode": "long",
                    "exit_policy": "threshold_or_bars",
                    "exit_bars": 5,
                },
            },
        ],
    )
    monkeypatch.setattr(
        "utils.evaluation.walkforward.research_data.load_features_for_combo",
        lambda single_spec, _config, **_kwargs: _series_for_combo(single_spec["params"]),
    )
    monkeypatch.setattr(
        "feature_research.pipelines.validation.load_portfolio_candles",
        lambda _config: _mock_candles_for_config(),
    )

    report = run_validation_pipeline(config, tmp_path / "validation_out")

    assert isinstance(report, WalkforwardRunReport)
    validation_dir = (
        config.output_root / "rule_based" / config.bias_spec["module_name"] / "validation"
    )
    assert validation_dir.exists()
    assert (validation_dir / "tables" / "selection_summary.csv").exists()
    assert (validation_dir / "tables" / "fold_scores.csv").exists()
    assert (validation_dir / "tables" / "folds.csv").exists()
