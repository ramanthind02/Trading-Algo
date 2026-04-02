"""Unit tests for multi-timeframe orchestration in portfolio_research.run_portfolio_test."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

rpt = importlib.import_module("portfolio_research.run_portfolio_test")
pipeline = importlib.import_module("portfolio_research.pipelines.portfolio_test")
from portfolio_research.config import ResearchWindow
from utils.core.enums import Ticker, TimeFrame


# ---------------------------------------------------------------------------
# Shared GlobalPortfolio mock
# ---------------------------------------------------------------------------

class _DummyGlobalPortfolio:
    """Minimal GlobalPortfolio stub that returns capped synthetic positions."""

    created: list[object] = []

    def __init__(
        self,
        tf_portfolios,
        max_position_pct: float = 2.0,
        **kwargs,  # noqa: ANN003
    ):
        self.tf_portfolios = tf_portfolios
        self.max_position_pct = max_position_pct
        self.fit_queries: list[object] = []
        self.predict_queries: list[object] = []
        self.is_fitted = False
        _DummyGlobalPortfolio.created.append(self)

    def fit_from_cache(self, query, instrument_returns):  # noqa: ANN001
        self.fit_queries.append((query, instrument_returns.copy()))
        self.is_fitted = True
        return self

    def save_to_vault(self, fit_start, fit_end, vault_root: str = "vault"):  # noqa: ANN001
        return f"dummy::{fit_start:%Y%m%d}::{fit_end:%Y%m%d}"

    def predict_from_cache(self, query):  # noqa: ANN001
        self.predict_queries.append(query)
        datetimes = pd.date_range(query.start, query.end, freq="D")
        if len(datetimes) == 0:
            return pd.DataFrame(
                columns=["ticker", "datetime", "forecast_score", "position_fraction"]
            )
        ticker = query.tickers[0] if query.tickers else "ES"
        n = len(datetimes)
        return pd.DataFrame(
            {
                "ticker": [ticker] * n,
                "datetime": datetimes.tolist(),
                "forecast_score": [1.0] * n,
                "position_fraction": [self.max_position_pct * 2.0] * n,
            }
        )


@dataclass
class _DummyConfig:
    tickers: list[Ticker]
    timeframe: TimeFrame
    start: datetime
    end: datetime
    use_cache: bool
    ensemble_dirs: dict[str, str]
    target_volatility: float
    weight_layer_method: str
    weight_layer_kwargs: dict[str, float]
    max_position_pct: float
    baseline_mode: str
    output_root: Path
    oos_window: None = None
    train_window: ResearchWindow = field(
        default_factory=lambda: ResearchWindow(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 2),
        )
    )
    validation_window: ResearchWindow = field(
        default_factory=lambda: ResearchWindow(
            start=datetime(2024, 1, 3),
            end=datetime(2024, 1, 4),
        )
    )
    test_window: ResearchWindow = field(
        default_factory=lambda: ResearchWindow(
            start=datetime(2024, 1, 4),
            end=datetime(2024, 1, 5),
        )
    )

    def walkforward_train_test_bounds(self) -> tuple[datetime, datetime, datetime, datetime]:
        return (
            datetime(2024, 1, 1),
            datetime(2024, 1, 2),
            datetime(2024, 1, 3),
            datetime(2024, 1, 5),
        )


class _DummyEnsemble:
    def __init__(self, base_tf: TimeFrame):
        self.base_tf = base_tf
        self.use_cache = False
        self.base_models = {"model": SimpleNamespace(use_cache=False)}


class _DummyWeightLayer:
    def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
        self.args = args
        self.kwargs = kwargs


class _DummyPortfolio:
    created_timeframes: list[TimeFrame] = []

    def __init__(
        self,
        ensembles,
        trading_timeframe,
        target_volatility,
        max_position_pct,
        use_cache,
        **kwargs,
    ):
        self.ensembles = ensembles
        self.trading_timeframe = trading_timeframe
        self.target_volatility = target_volatility
        self.max_position_pct = max_position_pct
        self.use_cache = use_cache
        self.fit_queries: list[object] = []
        self.predict_queries: list[object] = []
        _DummyPortfolio.created_timeframes.append(trading_timeframe)

    def fit_from_cache(self, query):  # noqa: ANN001
        self.fit_queries.append(query)
        return self

    def predict_from_cache(
        self,
        query,  # noqa: ANN001
        return_ensemble_predictions: bool = False,
        return_base_model_predictions: bool = False,
    ):
        self.predict_queries.append(query)
        datetimes = pd.date_range(query.start, query.end, freq="D")
        ticker = query.tickers[0] if query.tickers else "ES"
        positions = pd.DataFrame(
            {
                "ticker": [ticker] * len(datetimes),
                "datetime": datetimes.tolist(),
                "position_fraction": [self.max_position_pct] * len(datetimes),
                "forecast_score": [self.max_position_pct] * len(datetimes),
            }
        )
        if return_ensemble_predictions or return_base_model_predictions:
            result = {"portfolio": positions}
            if return_ensemble_predictions:
                result["ensembles"] = {
                    "ensemble_0": positions[["ticker", "datetime", "position_fraction"]].copy()
                }
            if return_base_model_predictions:
                result["base_models"] = {
                    "ensemble_0::model_a": positions[
                        ["ticker", "datetime", "position_fraction"]
                    ].copy()
                }
            return result
        return positions


class _DummyTester:
    def __init__(self, portfolio, baseline_mode: str = "equal_weight"):
        self.portfolio = portfolio
        self.baseline_mode = baseline_mode
        self.positions_df: pd.DataFrame | None = None
        self.ensemble_predictions: dict[str, pd.DataFrame] | None = None
        self.base_model_predictions: dict[str, pd.DataFrame] | None = None

    def fit(self, candles_df: pd.DataFrame):
        return self

    def fit_from_cache(self, query):
        self.portfolio.fit_from_cache(query)
        return self

    def predict(
        self,
        candles_df: pd.DataFrame,
        return_ensemble_predictions: bool = False,
        return_base_model_predictions: bool = False,
    ):
        ticker = candles_df["ticker"].iloc[0]
        datetimes = pd.to_datetime(candles_df["datetime"]).sort_values().drop_duplicates()
        tf = self.portfolio.trading_timeframe

        if tf == TimeFrame.D:
            position_values = [2.0] * len(datetimes)
        else:
            position_values = [2.0] * len(datetimes)

        positions = pd.DataFrame(
            {
                "ticker": [ticker] * len(datetimes),
                "datetime": datetimes.to_list(),
                "position_fraction": position_values,
                "forecast_score": position_values,
            }
        )
        self.positions_df = positions
        self.ensemble_predictions = {"ensemble_0": positions[["ticker", "datetime", "position_fraction"]].copy()}
        self.base_model_predictions = {
            "ensemble_0::model_a": positions[["ticker", "datetime", "position_fraction"]].copy()
        }

        if return_ensemble_predictions or return_base_model_predictions:
            result = {"portfolio": positions}
            if return_ensemble_predictions:
                result["ensembles"] = self.ensemble_predictions
            if return_base_model_predictions:
                result["base_models"] = self.base_model_predictions
            return result
        return positions

    def predict_from_cache(
        self,
        query,
        return_ensemble_predictions: bool = False,
        return_base_model_predictions: bool = False,
    ):
        result = self.portfolio.predict_from_cache(
            query,
            return_ensemble_predictions=return_ensemble_predictions,
            return_base_model_predictions=return_base_model_predictions,
        )
        if isinstance(result, dict):
            self.positions_df = result.get("portfolio")
            self.ensemble_predictions = result.get("ensembles")
            self.base_model_predictions = result.get("base_models")
        else:
            self.positions_df = result
            self.ensemble_predictions = None
            self.base_model_predictions = None
        return result

    def calculate_strategy_returns(
        self,
        candles_df: pd.DataFrame,
        positions_df: pd.DataFrame | None = None,
    ) -> pd.Series:
        idx = pd.to_datetime(candles_df["datetime"]).sort_values().drop_duplicates()
        return pd.Series(0.001, index=idx, name="strategy_return")

    def calculate_baseline_returns(self, candles_df: pd.DataFrame) -> pd.Series:
        idx = pd.to_datetime(candles_df["datetime"]).sort_values().drop_duplicates()
        return pd.Series(0.0005, index=idx, name="baseline_return")


def _make_candles(timeframe: TimeFrame, datetimes: list[str]) -> pd.DataFrame:
    rows = []
    for idx, dt in enumerate(pd.to_datetime(datetimes)):
        rows.append(
            {
                "datetime": dt,
                "open": 100.0 + idx,
                "high": 101.0 + idx,
                "low": 99.0 + idx,
                "close": 100.5 + idx,
                "volume": 1000 + idx,
                "ticker": Ticker.ES,
                "timeframe": timeframe,
            }
        )
    return pd.DataFrame(rows)


def test_run_portfolio_test_multi_timeframe_combines_caps_and_prefixes_outputs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _DummyPortfolio.created_timeframes = []
    _DummyGlobalPortfolio.created = []
    migrate_calls: list[str] = []
    bootstrap_calls: list[dict[str, object]] = []
    preflight_calls: list[tuple[tuple[str, ...], datetime, datetime, str]] = []

    config = _DummyConfig(
        tickers=[Ticker.ES],
        timeframe=TimeFrame.D,
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 5),
        use_cache=True,
        ensemble_dirs={
            "daily": "vault/D/daily_strategy",
            "weekly": "vault/W/weekly_strategy",
        },
        target_volatility=0.15,
        weight_layer_method="equal_signal",
        weight_layer_kwargs={"fdm_max": 2.5},
        max_position_pct=3.5,
        baseline_mode="equal_weight",
        output_root=tmp_path / "results",
    )

    daily_candles = _make_candles(
        TimeFrame.D,
        ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
    )
    weekly_candles = _make_candles(TimeFrame.W, ["2024-01-01", "2024-01-03", "2024-01-05"])

    def _mock_load_candles(_config, timeframe, start=None, end=None):  # noqa: ANN001
        if timeframe == TimeFrame.D:
            return daily_candles.copy()
        if timeframe == TimeFrame.W:
            return weekly_candles.copy()
        raise AssertionError(f"Unexpected timeframe: {timeframe}")

    def _mock_load_ensemble_from_vault(path: str, refit: bool, target_volatility: float):
        if "/W/" in path:
            return _DummyEnsemble(TimeFrame.W)
        return _DummyEnsemble(TimeFrame.D)

    strategy_position_calls: list[pd.DataFrame] = []

    def _mock_calculate_strategy_returns_from_positions(
        positions_df: pd.DataFrame,
        candles_df: pd.DataFrame,
        strategy: str = "long",
    ) -> pd.Series:
        strategy_position_calls.append(positions_df.copy())
        idx = pd.to_datetime(candles_df["datetime"]).sort_values().drop_duplicates()
        return pd.Series(0.001, index=idx, name="strategy_return")

    def _mock_calculate_baseline_returns(
        candles_df: pd.DataFrame,
        equal_weight: bool = True,
    ) -> pd.Series:
        idx = pd.to_datetime(candles_df["datetime"]).sort_values().drop_duplicates()
        return pd.Series(0.0005, index=idx, name="baseline_return")

    aggregation_counter = {"n": 0}

    def _mock_aggregate_intraday_returns_to_daily(returns: pd.Series) -> pd.Series:
        aggregation_counter["n"] += 1
        return returns

    generated_output_files: list[str] = []

    def _mock_generate_tearsheet(
        strategy_returns: pd.Series,
        baseline_returns: pd.Series,
        feature_name: str,
        output_file: str,
        mode: str,
    ) -> None:
        generated_output_files.append(output_file)

    def _mock_migrate_legacy_feature_members_schema(ensemble_dir: str) -> None:
        migrate_calls.append(ensemble_dir)

    def _mock_get_ensemble_tickers(ensemble_dir: str) -> list[Ticker]:
        if ensemble_dir == "vault/W/weekly_strategy":
            return [Ticker.ES, Ticker.NQ]
        return [Ticker.ES]

    def _mock_get_bias_node_specs(ensemble_dir: str) -> list[dict[str, object]]:
        if ensemble_dir == "vault/W/weekly_strategy":
            return [{"timeframes": [TimeFrame.W], "params": {"cross_tickers": ["GC"]}}]
        return [{"timeframes": [TimeFrame.D], "params": {}}]

    def _mock_bootstrap_source_candles(**kwargs: object) -> dict[str, object]:
        bootstrap_calls.append(kwargs)
        return {"success": 3, "failed": 0, "total": 3}

    def _mock_ensure_vault_cache_coverage(
        *,
        vault_ensemble_dirs: tuple[str, ...],
        start_date: datetime,
        end_date: datetime,
        refresh_mode: str = "missing_stale_only",
    ) -> dict[str, int | bool]:
        preflight_calls.append((vault_ensemble_dirs, start_date, end_date, refresh_mode))
        return {"success": 2, "total": 2}

    monkeypatch.setattr(pipeline, "_load_candles", _mock_load_candles)
    monkeypatch.setattr(pipeline, "load_ensemble_from_vault", _mock_load_ensemble_from_vault)
    monkeypatch.setattr(
        pipeline,
        "migrate_legacy_feature_members_schema",
        _mock_migrate_legacy_feature_members_schema,
    )
    monkeypatch.setattr(pipeline, "get_ensemble_tickers", _mock_get_ensemble_tickers)
    monkeypatch.setattr(pipeline, "get_bias_node_specs", _mock_get_bias_node_specs)
    monkeypatch.setattr(pipeline, "bootstrap_source_candles", _mock_bootstrap_source_candles)
    monkeypatch.setattr(
        pipeline,
        "ensure_vault_cache_coverage",
        _mock_ensure_vault_cache_coverage,
    )
    monkeypatch.setattr(pipeline, "WeightLayer", _DummyWeightLayer)
    monkeypatch.setattr(pipeline, "Portfolio", _DummyPortfolio)
    monkeypatch.setattr(pipeline, "GlobalPortfolio", _DummyGlobalPortfolio)
    monkeypatch.setattr(pipeline, "PortfolioTester", _DummyTester)
    monkeypatch.setattr(pipeline, "calculate_strategy_returns_from_positions", _mock_calculate_strategy_returns_from_positions)
    monkeypatch.setattr(pipeline, "calculate_baseline_returns", _mock_calculate_baseline_returns)
    monkeypatch.setattr(pipeline, "aggregate_intraday_returns_to_daily", _mock_aggregate_intraday_returns_to_daily)
    monkeypatch.setattr(pipeline, "generate_tearsheet", _mock_generate_tearsheet)
    monkeypatch.setattr(pipeline, "materialize_global_portfolio_predictions", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "_build_instrument_returns", lambda candles: pd.DataFrame())

    rpt.run_portfolio_test(config)

    # Three phases (Train, Validation, Test), each with D and W portfolios.
    assert _DummyPortfolio.created_timeframes.count(TimeFrame.D) == 3
    assert _DummyPortfolio.created_timeframes.count(TimeFrame.W) == 3
    assert migrate_calls == ["vault/D/daily_strategy", "vault/W/weekly_strategy"]
    assert bootstrap_calls == [
        {
            "tickers": (Ticker.ES, Ticker.NQ, Ticker.GC),
            "timeframes": (TimeFrame.D, TimeFrame.W),
            "start_date": datetime(2024, 1, 1),
            "end_date": datetime(2024, 1, 5),
        }
    ]
    assert preflight_calls == [
        (
            ("vault/D/daily_strategy", "vault/W/weekly_strategy"),
            datetime(2024, 1, 1),
            datetime(2024, 1, 5),
            "missing_stale_only",
        )
    ]

    # Multi-TF path: GlobalPortfolio created once per phase (3 phases).
    assert len(_DummyGlobalPortfolio.created) == 3, (
        f"Expected 3 GlobalPortfolio instances (one per phase), got {len(_DummyGlobalPortfolio.created)}"
    )

    generated_names = {Path(path).name for path in generated_output_files}
    generated_paths = [Path(p) for p in generated_output_files]
    assert "Portfolio_Train_window_tearsheet.html" in generated_names
    assert "daily_Portfolio_Train_window_tearsheet.html" in generated_names
    assert "weekly_Portfolio_Train_window_tearsheet.html" in generated_names

    # Component tearsheets live in timeframe subfolders (daily/, weekly/) with un-prefixed filenames.
    assert "ensemble_0_tearsheet.html" in generated_names
    assert "ensemble_0_model_a_tearsheet.html" in generated_names
    assert any("portfolio" in p.parts for p in generated_paths), "Expected portfolio/ subfolder"
    assert any("daily" in p.parts for p in generated_paths), "Expected daily/ subfolder"
    assert any("weekly" in p.parts for p in generated_paths), "Expected weekly/ subfolder"

    assert aggregation_counter["n"] >= 6

    capped_combined_frames = [
        frame
        for frame in strategy_position_calls
        if not frame.empty and frame["position_fraction"].abs().max() <= config.max_position_pct
    ]
    assert capped_combined_frames, "Expected at least one combined capped position frame"
    assert all(
        frame["position_fraction"].abs().max() <= config.max_position_pct
        for frame in capped_combined_frames
    )


def test_run_portfolio_test_single_timeframe_still_materializes_global_portfolio(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _DummyPortfolio.created_timeframes = []
    _DummyGlobalPortfolio.created = []
    migrate_calls: list[str] = []
    bootstrap_calls: list[dict[str, object]] = []
    preflight_calls: list[tuple[tuple[str, ...], datetime, datetime, str]] = []

    config = _DummyConfig(
        tickers=[Ticker.ES],
        timeframe=TimeFrame.D,
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 5),
        use_cache=True,
        ensemble_dirs={"daily": "vault/D/daily_strategy"},
        target_volatility=0.15,
        weight_layer_method="equal_signal",
        weight_layer_kwargs={"fdm_max": 2.5},
        max_position_pct=3.5,
        baseline_mode="equal_weight",
        output_root=tmp_path / "results",
    )

    daily_candles = _make_candles(
        TimeFrame.D,
        ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
    )

    def _mock_load_candles(_config, timeframe, start=None, end=None):  # noqa: ANN001
        if timeframe == TimeFrame.D:
            return daily_candles.copy()
        raise AssertionError(f"Unexpected timeframe: {timeframe}")

    def _mock_load_ensemble_from_vault(path: str, refit: bool, target_volatility: float):
        return _DummyEnsemble(TimeFrame.D)

    def _mock_calculate_strategy_returns_from_positions(
        positions_df: pd.DataFrame,
        candles_df: pd.DataFrame,
        strategy: str = "long",
    ) -> pd.Series:
        idx = pd.to_datetime(candles_df["datetime"]).sort_values().drop_duplicates()
        return pd.Series(0.001, index=idx, name="strategy_return")

    def _mock_calculate_baseline_returns(
        candles_df: pd.DataFrame,
        equal_weight: bool = True,
    ) -> pd.Series:
        idx = pd.to_datetime(candles_df["datetime"]).sort_values().drop_duplicates()
        return pd.Series(0.0005, index=idx, name="baseline_return")

    def _mock_migrate_legacy_feature_members_schema(ensemble_dir: str) -> None:
        migrate_calls.append(ensemble_dir)

    def _mock_get_ensemble_tickers(_ensemble_dir: str) -> list[Ticker]:
        return [Ticker.ES]

    def _mock_get_bias_node_specs(_ensemble_dir: str) -> list[dict[str, object]]:
        return [{"timeframes": [TimeFrame.D], "params": {}}]

    def _mock_bootstrap_source_candles(**kwargs: object) -> dict[str, object]:
        bootstrap_calls.append(kwargs)
        return {"success": 1, "failed": 0, "total": 1}

    def _mock_ensure_vault_cache_coverage(
        *,
        vault_ensemble_dirs: tuple[str, ...],
        start_date: datetime,
        end_date: datetime,
        refresh_mode: str = "missing_stale_only",
    ) -> dict[str, int | bool]:
        preflight_calls.append((vault_ensemble_dirs, start_date, end_date, refresh_mode))
        return {"success": 1, "total": 1}

    monkeypatch.setattr(pipeline, "_load_candles", _mock_load_candles)
    monkeypatch.setattr(pipeline, "load_ensemble_from_vault", _mock_load_ensemble_from_vault)
    monkeypatch.setattr(
        pipeline,
        "migrate_legacy_feature_members_schema",
        _mock_migrate_legacy_feature_members_schema,
    )
    monkeypatch.setattr(pipeline, "get_ensemble_tickers", _mock_get_ensemble_tickers)
    monkeypatch.setattr(pipeline, "get_bias_node_specs", _mock_get_bias_node_specs)
    monkeypatch.setattr(pipeline, "bootstrap_source_candles", _mock_bootstrap_source_candles)
    monkeypatch.setattr(
        pipeline,
        "ensure_vault_cache_coverage",
        _mock_ensure_vault_cache_coverage,
    )
    monkeypatch.setattr(pipeline, "WeightLayer", _DummyWeightLayer)
    monkeypatch.setattr(pipeline, "Portfolio", _DummyPortfolio)
    monkeypatch.setattr(pipeline, "GlobalPortfolio", _DummyGlobalPortfolio)
    monkeypatch.setattr(pipeline, "PortfolioTester", _DummyTester)
    monkeypatch.setattr(pipeline, "calculate_strategy_returns_from_positions", _mock_calculate_strategy_returns_from_positions)
    monkeypatch.setattr(pipeline, "calculate_baseline_returns", _mock_calculate_baseline_returns)
    monkeypatch.setattr(pipeline, "aggregate_intraday_returns_to_daily", lambda returns: returns)
    monkeypatch.setattr(pipeline, "generate_tearsheet", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "materialize_global_portfolio_predictions", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "_build_instrument_returns", lambda candles: pd.DataFrame())

    rpt.run_portfolio_test(config)

    assert len(_DummyGlobalPortfolio.created) == 3
    assert migrate_calls == ["vault/D/daily_strategy"]
    assert bootstrap_calls == [
        {
            "tickers": (Ticker.ES,),
            "timeframes": (TimeFrame.D,),
            "start_date": datetime(2024, 1, 1),
            "end_date": datetime(2024, 1, 5),
        }
    ]
    assert preflight_calls == [
        (
            ("vault/D/daily_strategy",),
            datetime(2024, 1, 1),
            datetime(2024, 1, 5),
            "missing_stale_only",
        )
    ]
    assert not (config.output_root / "train" / "global_weight_layer").exists()
    assert not (config.output_root / "validation" / "global_weight_layer").exists()
    assert not (config.output_root / "test" / "global_weight_layer").exists()
