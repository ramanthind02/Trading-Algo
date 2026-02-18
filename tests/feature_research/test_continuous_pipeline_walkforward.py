from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import matplotlib.pyplot as plt
import pandas as pd

from feature_research.continuous_binning.config import ResearchConfig
from feature_research.continuous_binning.pipeline import run_continuous_eda_pipeline
from feature_research.walkforward.config import WalkforwardResearchConfig
from feature_research.walkforward.runner import WalkforwardRunReport
from utils.enums import Ticker, TimeFrame


def _build_config(tmp_path: Path, *, walkforward_enabled: bool) -> ResearchConfig:
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
    )


def _mock_eda_report() -> SimpleNamespace:
    return SimpleNamespace(
        common_stats=SimpleNamespace(correlation_analysis=SimpleNamespace(pearson=0.1)),
        continuous_stats=SimpleNamespace(
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


def test_walkforward_disabled_skips_shared_runner(monkeypatch, tmp_path: Path) -> None:
    config = _build_config(tmp_path, walkforward_enabled=False)

    monkeypatch.setattr(
        "feature_research.continuous_binning.pipeline.populate_cache_if_needed",
        lambda _config: None,
    )
    monkeypatch.setattr(
        "feature_research.continuous_binning.pipeline.expand_bias_specs",
        lambda _bias_spec: [
            {
                "module_name": "rsi",
                "timeframes": [TimeFrame.D],
                "params": {"lookback": 2},
            }
        ],
    )
    monkeypatch.setattr(
        "feature_research.continuous_binning.pipeline.load_features_for_combo",
        lambda single_spec, _config: _series_for_combo(single_spec["params"]),
    )
    monkeypatch.setattr(
        "feature_research.continuous_binning.pipeline.run_eda_for_continuous_feature",
        lambda *_args, **_kwargs: _mock_eda_report(),
    )
    monkeypatch.setattr(
        "feature_research.continuous_binning.pipeline.save_eda_report",
        lambda report, output_dir, overwrite: output_dir,
    )

    calls = {"runner": 0, "stability": 0, "timeline": 0, "writer": 0}
    monkeypatch.setattr(
        "feature_research.continuous_binning.pipeline.run_walkforward_research",
        lambda *_args, **_kwargs: calls.__setitem__("runner", calls["runner"] + 1),
        raising=False,
    )
    monkeypatch.setattr(
        "feature_research.continuous_binning.pipeline.plot_selection_stability",
        lambda *_args, **_kwargs: calls.__setitem__("stability", calls["stability"] + 1),
        raising=False,
    )
    monkeypatch.setattr(
        "feature_research.continuous_binning.pipeline.plot_fold_timeline",
        lambda *_args, **_kwargs: calls.__setitem__("timeline", calls["timeline"] + 1),
        raising=False,
    )
    monkeypatch.setattr(
        "feature_research.continuous_binning.pipeline.write_walkforward_artifacts",
        lambda *_args, **_kwargs: calls.__setitem__("writer", calls["writer"] + 1),
        raising=False,
    )

    run_continuous_eda_pipeline(config=config, output_dir=tmp_path / "continuous_reports")

    assert calls == {"runner": 0, "stability": 0, "timeline": 0, "writer": 0}


def test_walkforward_enabled_writes_selected_feature_artifacts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _build_config(tmp_path, walkforward_enabled=True)

    monkeypatch.setattr(
        "feature_research.continuous_binning.pipeline.populate_cache_if_needed",
        lambda _config: None,
    )
    monkeypatch.setattr(
        "feature_research.continuous_binning.pipeline.expand_bias_specs",
        lambda _bias_spec: [
            {
                "module_name": "rsi",
                "timeframes": [TimeFrame.D],
                "params": {"lookback": 2},
            },
            {
                "module_name": "rsi",
                "timeframes": [TimeFrame.D],
                "params": {"lookback": 3},
            },
        ],
    )
    monkeypatch.setattr(
        "feature_research.continuous_binning.pipeline.load_features_for_combo",
        lambda single_spec, _config: _series_for_combo(single_spec["params"], tz="UTC"),
    )
    monkeypatch.setattr(
        "feature_research.continuous_binning.pipeline.run_eda_for_continuous_feature",
        lambda *_args, **_kwargs: _mock_eda_report(),
    )
    monkeypatch.setattr(
        "feature_research.continuous_binning.pipeline.save_eda_report",
        lambda report, output_dir, overwrite: output_dir,
    )

    captured_inputs: dict[str, object] = {}

    def _mock_run_walkforward_research(
        candles_df: pd.DataFrame,
        target: pd.Series,
        feature_type: str,
        module_name: str,
        config: WalkforwardResearchConfig,
        param_grid: list[dict[str, object]],
        evaluate_param_combo,
    ) -> WalkforwardRunReport:
        captured_inputs["candles_df"] = candles_df
        captured_inputs["target"] = target
        captured_inputs["feature_type"] = feature_type
        captured_inputs["module_name"] = module_name
        captured_inputs["param_grid"] = param_grid
        captured_inputs["config"] = config
        captured_inputs["evaluate_param_combo"] = evaluate_param_combo

        return WalkforwardRunReport(
            folds_df=pd.DataFrame(
                [
                    {
                        "fold_id": 0,
                        "train_start": pd.Timestamp("2020-01-01"),
                        "train_end": pd.Timestamp("2020-01-31"),
                        "test_start": pd.Timestamp("2020-02-01"),
                        "test_end": pd.Timestamp("2020-02-20"),
                        "train_samples": 31,
                        "test_samples": 20,
                    }
                ]
            ),
            fold_scores_df=pd.DataFrame(
                [
                    {
                        "fold_id": 0,
                        "param_label": "lookback=2",
                        "raw_objective": 0.1,
                        "smoothed_objective": 0.1,
                        "rank": 1,
                        "selected_feature": True,
                    }
                ]
            ),
            selection_summary_df=pd.DataFrame(
                [
                    {
                        "fold_id": 0,
                        "selected_feature": "lookback=2",
                        "selected_raw_objective": 0.1,
                        "selected_smoothed_objective": 0.1,
                        "top_k_features": "[\"lookback=2\",\"lookback=3\"]",
                    }
                ]
            ),
        )

    monkeypatch.setattr(
        "feature_research.continuous_binning.pipeline.run_walkforward_research",
        _mock_run_walkforward_research,
        raising=False,
    )

    def _mock_plot_selection_stability(selection_summary_df: pd.DataFrame, top_k: int):
        _ = top_k
        figure, _ = plt.subplots(figsize=(6, 2))
        return figure, selection_summary_df

    def _mock_plot_fold_timeline(folds_df: pd.DataFrame):
        figure, _ = plt.subplots(figsize=(6, 2))
        return figure, folds_df

    monkeypatch.setattr(
        "feature_research.continuous_binning.pipeline.plot_selection_stability",
        _mock_plot_selection_stability,
        raising=False,
    )
    monkeypatch.setattr(
        "feature_research.continuous_binning.pipeline.plot_fold_timeline",
        _mock_plot_fold_timeline,
        raising=False,
    )

    run_continuous_eda_pipeline(config=config, output_dir=tmp_path / "continuous_reports")

    assert captured_inputs["feature_type"] == "continuous"
    assert captured_inputs["module_name"] == "rsi"

    run_target = cast(pd.Series, captured_inputs["target"])
    run_candles = cast(pd.DataFrame, captured_inputs["candles_df"])
    assert isinstance(run_target.index, pd.DatetimeIndex)
    assert isinstance(run_candles.index, pd.DatetimeIndex)
    assert run_target.index.tz is None
    assert run_candles.index.tz is None

    walkforward_dir = (
        config.walkforward.output_root / "continuous" / config.bias_spec["module_name"] / "walkforward"
    )
    assert walkforward_dir.exists()
    assert "continuous/rsi/walkforward" in walkforward_dir.as_posix()

    selection_summary_csv = walkforward_dir / "selection_summary.csv"
    assert selection_summary_csv.exists()
    selection_summary_df = pd.read_csv(selection_summary_csv)
    assert "selected_feature" in selection_summary_df.columns
