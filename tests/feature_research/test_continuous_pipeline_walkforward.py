from __future__ import annotations

import dataclasses
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import matplotlib.pyplot as plt
import pandas as pd

from feature_research.in_sample.config import ResearchConfig
from feature_research.pipeline import run_eda_pipeline
from feature_research.walkforward.config import (
    WalkforwardResearchConfig,
    WalkforwardSelectionMethod,
)
from feature_research.walkforward.runner import WalkforwardRunReport
from utils.core.enums import Ticker, TimeFrame


def _build_config(
    tmp_path: Path,
    *,
    walkforward_enabled: bool,
    selection_method: WalkforwardSelectionMethod = WalkforwardSelectionMethod.TOP_K,
) -> ResearchConfig:
    walkforward = WalkforwardResearchConfig(
        train_start=datetime(2020, 1, 1),
        train_end=datetime(2020, 2, 10),
        enabled=walkforward_enabled,
        test_step=20,
        num_steps=2,
        top_k=2,
        objective_metric_name="mean_return",
        min_fold_samples=10,
        output_root=tmp_path / "shared_results",
        selection_method=selection_method,
    )
    return ResearchConfig(
        tickers=[Ticker.ES],
        start=datetime(2020, 1, 1),
        end=datetime(2020, 4, 29),
        bias_spec={
            "module_name": "rsi",
            "timeframes": [TimeFrame.D],
            "params": {"lookback": [2, 3]},
        },
        target_col="log_return",
        strategy="long-short",
        use_cache=True,
        populate_cache=False,
        reports_dir=tmp_path / "reports",
        walkforward=walkforward,
        walkforward_selection_method=selection_method,
    )


def _mock_eda_report() -> SimpleNamespace:
    return SimpleNamespace(
        common_stats=SimpleNamespace(correlation_analysis=SimpleNamespace(pearson=0.1)),
        continuous_stats=SimpleNamespace(
            quintile_spread=SimpleNamespace(spread=0.3),
            monotonicity_test=SimpleNamespace(kendall_tau=0.2),
            decile_analysis=SimpleNamespace(overall_trend="up"),
        ),
        diagnostics=SimpleNamespace(is_viable=True, red_flags=[], warnings=[]),
    )


def _series_for_combo(
    params: dict[str, object],
    *,
    tz: str | None = None,
) -> tuple[pd.Series, pd.Series, str]:
    index = pd.date_range("2020-01-01", periods=120, freq="D", tz=tz)
    lookback = int(cast(int, params["lookback"]))
    feature = pd.Series([float(lookback)] * len(index), index=index, name=f"feature_{lookback}")
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


def test_walkforward_disabled_skips_shared_runner(monkeypatch, tmp_path: Path) -> None:
    config = _build_config(tmp_path, walkforward_enabled=False)

    monkeypatch.setattr(
        "feature_research.pipelines.in_sample.populate_cache_if_needed",
        lambda _config: None,
    )
    monkeypatch.setattr(
        "feature_research.pipelines.in_sample.expand_bias_specs",
        lambda _bias_spec: [
            {
                "module_name": "rsi",
                "timeframes": [TimeFrame.D],
                "params": {"lookback": 2},
            }
        ],
    )
    monkeypatch.setattr(
        "feature_research.pipelines.in_sample.load_features_for_combo",
        lambda single_spec, _config: _series_for_combo(single_spec["params"]),
    )
    monkeypatch.setattr(
        "feature_research.pipelines.in_sample.run_eda_for_continuous_feature",
        lambda *_args, **_kwargs: _mock_eda_report(),
    )
    monkeypatch.setattr(
        "feature_research.pipelines.in_sample.save_eda_report",
        lambda report, output_dir, overwrite: output_dir,
    )

    calls = {"runner": 0, "stability": 0, "timeline": 0, "writer": 0}
    monkeypatch.setattr(
        "feature_research.pipelines.in_sample.run_walkforward_research",
        lambda *_args, **_kwargs: calls.__setitem__("runner", calls["runner"] + 1),
        raising=False,
    )
    monkeypatch.setattr(
        "feature_research.pipelines.in_sample.plot_selection_stability",
        lambda *_args, **_kwargs: calls.__setitem__("stability", calls["stability"] + 1),
        raising=False,
    )
    monkeypatch.setattr(
        "feature_research.pipelines.in_sample.plot_fold_timeline",
        lambda *_args, **_kwargs: calls.__setitem__("timeline", calls["timeline"] + 1),
        raising=False,
    )
    monkeypatch.setattr(
        "feature_research.pipelines.in_sample.write_walkforward_artifacts",
        lambda *_args, **_kwargs: calls.__setitem__("writer", calls["writer"] + 1),
        raising=False,
    )

    run_eda_pipeline(config=config, output_dir=tmp_path / "continuous_reports")

    assert calls == {"runner": 0, "stability": 0, "timeline": 0, "writer": 0}


def test_run_continuous_walkforward_pipeline_returns_report_and_writes_artifacts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from feature_research.pipeline import run_continuous_walkforward_pipeline

    config = _build_config(tmp_path, walkforward_enabled=True)

    monkeypatch.setattr(
        "feature_research.pipelines.walkforward.populate_cache_if_needed",
        lambda _config: None,
    )
    monkeypatch.setattr(
        "feature_research.pipelines.walkforward.expand_bias_specs",
        lambda _bias_spec: [
            {"module_name": "rsi", "timeframes": [TimeFrame.D], "params": {"lookback": 2}},
            {"module_name": "rsi", "timeframes": [TimeFrame.D], "params": {"lookback": 3}},
        ],
    )
    monkeypatch.setattr(
        "feature_research.walkforward.research_data.load_features_for_combo",
        lambda single_spec, _config, **_kwargs: _series_for_combo(single_spec["params"]),
    )
    monkeypatch.setattr(
        "feature_research.pipelines.walkforward.load_portfolio_candles",
        lambda _config: _mock_candles_for_config(),
    )

    report = run_continuous_walkforward_pipeline(config, tmp_path / "wf_out")

    assert isinstance(report, WalkforwardRunReport)
    walkforward_dir = (
        config.walkforward.output_root / "continuous" / config.bias_spec["module_name"] / "walkforward"
    )
    assert walkforward_dir.exists()
    assert (walkforward_dir / "tables" / "selection_summary.csv").exists()
    assert (walkforward_dir / "tables" / "fold_scores.csv").exists()
    assert (walkforward_dir / "tables" / "folds.csv").exists()


def test_run_continuous_walkforward_pipeline_raises_if_no_combos_load(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from feature_research.pipeline import run_continuous_walkforward_pipeline
    import pytest

    config = _build_config(tmp_path, walkforward_enabled=True)

    monkeypatch.setattr(
        "feature_research.pipelines.walkforward.populate_cache_if_needed",
        lambda _config: None,
    )
    monkeypatch.setattr(
        "feature_research.pipelines.walkforward.expand_bias_specs",
        lambda _bias_spec: [
            {"module_name": "rsi", "timeframes": [TimeFrame.D], "params": {"lookback": 2}},
        ],
    )
    monkeypatch.setattr(
        "feature_research.walkforward.research_data.load_features_for_combo",
        lambda single_spec, _config, **_kwargs: None,
    )
    monkeypatch.setattr(
        "feature_research.pipelines.walkforward.load_portfolio_candles",
        lambda _config: _mock_candles_for_config(),
    )

    with pytest.raises(ValueError, match="No param combos loaded successfully"):
        run_continuous_walkforward_pipeline(config, tmp_path / "wf_out")


def test_run_continuous_walkforward_pipeline_raises_if_no_folds(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from feature_research.pipeline import run_continuous_walkforward_pipeline
    import pytest

    config = _build_config(tmp_path, walkforward_enabled=True)
    empty_fold_config = dataclasses.replace(
        config,
        walkforward=dataclasses.replace(
            config.walkforward,
            train_start=datetime(2025, 1, 1),
            train_end=datetime(2025, 2, 1),
        ),
    )

    monkeypatch.setattr(
        "feature_research.pipelines.walkforward.populate_cache_if_needed",
        lambda _config: None,
    )
    monkeypatch.setattr(
        "feature_research.pipelines.walkforward.expand_bias_specs",
        lambda _bias_spec: [
            {"module_name": "rsi", "timeframes": [TimeFrame.D], "params": {"lookback": 2}},
        ],
    )
    monkeypatch.setattr(
        "feature_research.walkforward.research_data.load_features_for_combo",
        lambda single_spec, _config, **_kwargs: _series_for_combo(single_spec["params"]),
    )
    monkeypatch.setattr(
        "feature_research.pipelines.walkforward.load_portfolio_candles",
        lambda _config: _mock_candles_for_config(),
    )

    with pytest.raises(ValueError, match="No walkforward folds were generated"):
        run_continuous_walkforward_pipeline(empty_fold_config, tmp_path / "wf_out")
