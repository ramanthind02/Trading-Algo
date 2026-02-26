from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import Any

import pandas as pd

import feature_extraction.feature_extractor as feature_extractor
from feature_research.config import FeatureType
from feature_research.in_sample import data_loader as in_sample_data_loader
from utils.core.enums import TimeFrame, Ticker


def _aligned_feature_target() -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    features = pd.DataFrame({"feat": [1.0, 2.0, 3.0], "ticker": ["ES", "ES", "ES"]}, index=idx)
    targets = pd.DataFrame({"log_return": [0.1, 0.2, 0.3], "ticker": ["ES", "ES", "ES"]}, index=idx)
    return features, targets


def _sample_override_candles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC"),
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "ticker": ["ES", "ES", "ES"],
        }
    )


def _sample_multi_ticker_override_candles() -> pd.DataFrame:
    base_dt = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "datetime": list(base_dt) + list(base_dt),
            "open": [100.0, 101.0, 102.0, 200.0, 201.0, 202.0],
            "high": [101.0, 102.0, 103.0, 201.0, 202.0, 203.0],
            "low": [99.0, 100.0, 101.0, 199.0, 200.0, 201.0],
            "close": [100.5, 101.5, 102.5, 200.5, 201.5, 202.5],
            "ticker": ["ES", "ES", "ES", "NQ", "NQ", "NQ"],
        }
    )


def test_extract_features_for_bias_node_forwards_candles_override(monkeypatch: Any) -> None:
    override = _sample_override_candles()
    captured: dict[str, Any] = {}

    def _fake_extract_features_with_forward_returns(**kwargs: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
        captured.update(kwargs)
        return _aligned_feature_target()

    monkeypatch.setattr(
        feature_extractor,
        "extract_features_with_forward_returns",
        _fake_extract_features_with_forward_returns,
    )

    bias_spec = {
        "module_name": "rsi",
        "timeframes": [TimeFrame.D],
        "params": {"lookback": 14},
    }

    feature_extractor.extract_features_for_bias_node(
        bias_spec=bias_spec,
        ticker=Ticker.ES,
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 10),
        candles_override=override,
    )

    assert captured["candles_override"] is override


def test_extract_features_with_forward_returns_uses_override_instead_of_loading(monkeypatch: Any) -> None:
    override = _sample_override_candles()
    idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")

    def _fake_extract_features(**kwargs: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
        module_name = kwargs["module_name"]
        if module_name == "atr":
            return pd.DataFrame({"atr_252": [0.2, 0.2, 0.2], "ticker": ["ES", "ES", "ES"]}, index=idx), pd.DataFrame()
        if module_name == "ewsd":
            return pd.DataFrame({"ewsd_63": [1.0, 1.0, 1.0], "ticker": ["ES", "ES", "ES"]}, index=idx), pd.DataFrame()
        return pd.DataFrame({"feat": [1.0, 2.0, 3.0], "ticker": ["ES", "ES", "ES"]}, index=idx), pd.DataFrame()

    seen: dict[str, Any] = {}

    def _fake_compute_forward_returns(
        candles_df: pd.DataFrame,
        features_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        seen["candles_df"] = candles_df
        return pd.DataFrame(
            {
                "raw_return": [0.01, 0.01, 0.01],
                "log_return": [0.01, 0.01, 0.01],
                "log_return_atr": [0.05, 0.05, 0.05],
                "log_return_ewsd": [1.0, 1.0, 1.0],
                "ticker": ["ES", "ES", "ES"],
            },
            index=idx,
        )

    def _fail_if_called(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("load_data_multi_ticker should not be called when candles_override is provided")

    monkeypatch.setattr(feature_extractor, "extract_features", _fake_extract_features)
    monkeypatch.setattr(feature_extractor, "compute_forward_returns", _fake_compute_forward_returns)
    monkeypatch.setattr(feature_extractor.helpers, "load_data_multi_ticker", _fail_if_called)

    feature_extractor.extract_features_with_forward_returns(
        module_name="rsi",
        params={"lookback": 14},
        ticker=Ticker.ES,
        candles_override=override,
    )

    assert list(seen["candles_df"]["ticker"].unique()) == ["ES"]


def test_extract_features_with_forward_returns_normalizes_override_datetimes_to_utc(monkeypatch: Any) -> None:
    override = _sample_override_candles()
    override["datetime"] = pd.date_range("2024-01-01", periods=3, freq="D")
    idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")

    def _fake_extract_features(**kwargs: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
        module_name = kwargs["module_name"]
        if module_name == "atr":
            return pd.DataFrame({"atr_252": [0.2, 0.2, 0.2], "ticker": ["ES", "ES", "ES"]}, index=idx), pd.DataFrame()
        if module_name == "ewsd":
            return pd.DataFrame({"ewsd_63": [1.0, 1.0, 1.0], "ticker": ["ES", "ES", "ES"]}, index=idx), pd.DataFrame()
        return pd.DataFrame({"feat": [1.0, 2.0, 3.0], "ticker": ["ES", "ES", "ES"]}, index=idx), pd.DataFrame()

    seen: dict[str, Any] = {}

    def _fake_compute_forward_returns(
        candles_df: pd.DataFrame,
        features_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        seen["tz"] = candles_df["datetime"].dt.tz
        return pd.DataFrame(
            {
                "raw_return": [0.01, 0.01, 0.01],
                "log_return": [0.01, 0.01, 0.01],
                "log_return_atr": [0.05, 0.05, 0.05],
                "log_return_ewsd": [1.0, 1.0, 1.0],
                "ticker": ["ES", "ES", "ES"],
            },
            index=idx,
        )

    monkeypatch.setattr(feature_extractor, "extract_features", _fake_extract_features)
    monkeypatch.setattr(feature_extractor, "compute_forward_returns", _fake_compute_forward_returns)

    feature_extractor.extract_features_with_forward_returns(
        module_name="rsi",
        params={"lookback": 14},
        ticker=Ticker.ES,
        candles_override=override,
    )

    assert str(seen["tz"]) == "UTC"


def test_extract_features_with_forward_returns_multi_ticker_aligns_on_datetime_ticker(monkeypatch: Any) -> None:
    """Primary key is (datetime, ticker); no millisecond offsets. All tickers align on bar datetime."""
    override = _sample_multi_ticker_override_candles()
    # Bar datetimes (no offset): same timestamp for ES and NQ per bar
    bar_dts = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-01-01", tz="UTC"),
            pd.Timestamp("2024-01-02", tz="UTC"),
            pd.Timestamp("2024-01-03", tz="UTC"),
            pd.Timestamp("2024-01-01", tz="UTC"),
            pd.Timestamp("2024-01-02", tz="UTC"),
            pd.Timestamp("2024-01-03", tz="UTC"),
        ]
    )

    def _fake_extract_features(**kwargs: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
        module_name = kwargs["module_name"]
        if module_name == "atr":
            return pd.DataFrame(
                {"atr_252": [0.2] * 6, "ticker": ["ES", "ES", "ES", "NQ", "NQ", "NQ"]},
                index=bar_dts,
            ), pd.DataFrame()
        if module_name == "ewsd":
            return pd.DataFrame(
                {"ewsd_63": [1.0] * 6, "ticker": ["ES", "ES", "ES", "NQ", "NQ", "NQ"]},
                index=bar_dts,
            ), pd.DataFrame()
        return pd.DataFrame(
            {"feat": [1.0, 2.0, 3.0, 11.0, 12.0, 13.0], "ticker": ["ES", "ES", "ES", "NQ", "NQ", "NQ"]},
            index=bar_dts,
        ), pd.DataFrame()

    seen: dict[str, Any] = {}

    def _fake_compute_forward_returns(
        candles_df: pd.DataFrame,
        features_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        seen["candles_df"] = candles_df
        # Return targets keyed by same (datetime, ticker) as features
        return pd.DataFrame(
            {
                "raw_return": [0.01] * 6,
                "log_return": [0.01] * 6,
                "log_return_atr": [0.05] * 6,
                "log_return_ewsd": [1.0] * 6,
                "ticker": ["ES", "ES", "ES", "NQ", "NQ", "NQ"],
            },
            index=bar_dts,
        )

    monkeypatch.setattr(feature_extractor, "extract_features", _fake_extract_features)
    monkeypatch.setattr(feature_extractor, "compute_forward_returns", _fake_compute_forward_returns)

    feat_df, tgt_df = feature_extractor.extract_features_with_forward_returns(
        module_name="rsi",
        params={"lookback": 14},
        ticker=[Ticker.ES, Ticker.NQ],
        candles_override=override,
    )

    # Candles passed to compute_forward_returns use bar datetime (no offset)
    nq_dts = seen["candles_df"].loc[seen["candles_df"]["ticker"] == "NQ", "datetime"]
    assert nq_dts.iloc[0] == pd.Timestamp("2024-01-01", tz="UTC")
    # Result has both tickers
    assert set(feat_df["ticker"].unique()) == {"ES", "NQ"}
    assert len(feat_df) == 6


def test_extract_features_with_forward_returns_rejects_invalid_override_schema(monkeypatch: Any) -> None:
    invalid_override = pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC"),
            "open": [100.0, 101.0, 102.0],
            "close": [100.5, 101.5, 102.5],
            "ticker": ["ES", "ES", "ES"],
        }
    )

    def _fail_if_called(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("extract_features should not be called when override schema is invalid")

    monkeypatch.setattr(feature_extractor, "extract_features", _fail_if_called)

    try:
        feature_extractor.extract_features_with_forward_returns(
            module_name="rsi",
            params={"lookback": 14},
            ticker=Ticker.ES,
            candles_override=invalid_override,
        )
    except ValueError as exc:
        assert "candles_override missing required columns" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid candles_override schema")


def test_extract_features_with_forward_returns_errors_when_ticker_is_dropped_after_alignment(monkeypatch: Any) -> None:
    override = _sample_multi_ticker_override_candles()
    idx = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-01-01", tz="UTC"),
            pd.Timestamp("2024-01-02", tz="UTC"),
            pd.Timestamp("2024-01-01 00:00:00.001", tz="UTC"),
            pd.Timestamp("2024-01-02 00:00:00.001", tz="UTC"),
        ]
    )

    def _fake_extract_features(**kwargs: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
        module_name = kwargs["module_name"]
        if module_name == "atr":
            return pd.DataFrame(
                {"atr_252": [0.2] * 4, "ticker": ["ES", "ES", "NQ", "NQ"]},
                index=idx,
            ), pd.DataFrame()
        if module_name == "ewsd":
            return pd.DataFrame(
                {"ewsd_63": [1.0] * 4, "ticker": ["ES", "ES", "NQ", "NQ"]},
                index=idx,
            ), pd.DataFrame()
        return pd.DataFrame(
            {"feat": [1.0, 2.0, 11.0, 12.0], "ticker": ["ES", "ES", "NQ", "NQ"]},
            index=idx,
        ), pd.DataFrame()

    def _fake_compute_forward_returns(
        candles_df: pd.DataFrame,
        features_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "raw_return": [0.01, 0.01],
                "log_return": [0.01, 0.01],
                "log_return_atr": [0.05, 0.05],
                "log_return_ewsd": [1.0, 1.0],
                "ticker": ["ES", "ES"],
            },
            index=idx[:2],
        )

    monkeypatch.setattr(feature_extractor, "extract_features", _fake_extract_features)
    monkeypatch.setattr(feature_extractor, "compute_forward_returns", _fake_compute_forward_returns)

    try:
        feature_extractor.extract_features_with_forward_returns(
            module_name="rsi",
            params={"lookback": 14},
            ticker=[Ticker.ES, Ticker.NQ],
            candles_override=override,
        )
    except ValueError as exc:
        assert "Tickers dropped during feature-target alignment" in str(exc)
        assert "NQ" in str(exc)
    else:
        raise AssertionError("Expected ValueError when alignment drops a ticker")


def test_continuous_loader_forwards_candles_override(monkeypatch: Any) -> None:
    override = _sample_override_candles()
    captured: dict[str, Any] = {}

    def _fake_extract_features_for_bias_node(**kwargs: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
        captured.update(kwargs)
        return _aligned_feature_target()

    monkeypatch.setattr(
        in_sample_data_loader,
        "extract_features_for_bias_node",
        _fake_extract_features_for_bias_node,
    )

    config = SimpleNamespace(
        tickers=[Ticker.ES],
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 10),
        target_col="log_return",
        use_cache=False,
        feature_type=FeatureType.CONTINUOUS,
    )

    result = in_sample_data_loader.load_features_for_combo(
        single_combo_spec={"module_name": "rsi", "params": {"lookback": 14}, "timeframes": [TimeFrame.D]},
        config=config,
        candles_override=override,
    )

    assert result is not None
    assert captured["candles_override"] is override


def test_rule_based_loader_default_behavior_without_override(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def _fake_extract_features_for_bias_node(**kwargs: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
        captured.update(kwargs)
        return _aligned_feature_target()

    monkeypatch.setattr(
        in_sample_data_loader,
        "extract_features_for_bias_node",
        _fake_extract_features_for_bias_node,
    )

    config = SimpleNamespace(
        tickers=[Ticker.ES],
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 10),
        target_col="log_return",
        use_cache=False,
        feature_type=FeatureType.RULE_BASED,
    )

    result = in_sample_data_loader.load_features_for_combo(
        single_combo_spec={"module_name": "rsi", "params": {"lookback": 14}, "timeframes": [TimeFrame.D]},
        config=config,
    )

    assert result is not None
    assert captured["candles_override"] is None


def test_rule_based_loader_forwards_candles_override(monkeypatch: Any) -> None:
    override = _sample_override_candles()
    captured: dict[str, Any] = {}

    def _fake_extract_features_for_bias_node(**kwargs: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
        captured.update(kwargs)
        return _aligned_feature_target()

    monkeypatch.setattr(
        in_sample_data_loader,
        "extract_features_for_bias_node",
        _fake_extract_features_for_bias_node,
    )

    config = SimpleNamespace(
        tickers=[Ticker.ES],
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 10),
        target_col="log_return",
        use_cache=False,
        feature_type=FeatureType.RULE_BASED,
    )

    result = in_sample_data_loader.load_features_for_combo(
        single_combo_spec={"module_name": "rsi", "params": {"lookback": 14}, "timeframes": [TimeFrame.D]},
        config=config,
        candles_override=override,
    )

    assert result is not None
    assert captured["candles_override"] is override
