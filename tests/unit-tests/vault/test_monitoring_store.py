"""Unit tests for ensemble/monitoring_store.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ensemble.monitoring_store import (
    append_monitoring_data,
    initialize_monitoring,
    load_monitoring_data,
)
from ensemble.vault_manager import create_ensemble_directory
from feature_selection.base_models import ContinuousBinningModel
from feature_selection.base_models import RuleBasedModel
from feature_selection.base_models.feature_base_model import BaseModel
from utils.core.enums import Direction, Ticker, TimeFrame


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_signals(n: int = 50, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series(rng.choice([0.5, 1.0, 1.5], size=n).astype(float), index=idx, name="signal")


def _make_targets(n: int = 50, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series(rng.normal(0.001, 0.01, size=n), index=idx, name="target")


def _make_fitted_base_model() -> BaseModel:
    bias_node_spec = {
        "module_name": "rsi",
        "timeframes": [TimeFrame.D],
        "params": {"lookback": 2},
    }
    model = BaseModel(
        feature_config={"bias_node_spec": bias_node_spec},
        tickers=[Ticker.ES],
        binning_model=ContinuousBinningModel(n_bins=3, strategy="long"),
        use_cache=False,
    )
    idx = pd.date_range("2020-01-01", periods=150, freq="D")
    candles = pd.DataFrame(
        {
            "datetime": idx,
            "open": 100.0 + np.arange(len(idx)) * 0.1,
            "high": 101.0 + np.arange(len(idx)) * 0.1,
            "low": 99.0 + np.arange(len(idx)) * 0.1,
            "close": 100.2 + np.arange(len(idx)) * 0.1,
            "volume": 1_000_000.0,
            "ticker": Ticker.ES,
            "timeframe": TimeFrame.D,
        }
    )
    rng = np.random.default_rng(42)
    target = pd.Series(0.01 + rng.normal(0, 0.002, len(idx)), index=idx)
    model.fit(candles, target)
    return model


def _make_unfitted_base_model() -> BaseModel:
    bias_node_spec = {
        "module_name": "buy_hold",
        "timeframes": [TimeFrame.D],
        "params": {},
    }
    return BaseModel(
        feature_config={"bias_node_spec": bias_node_spec},
        tickers=[Ticker.ES],
        binning_model=RuleBasedModel(strategy="long"),
        use_cache=False,
    )


# ---------------------------------------------------------------------------
# initialize_monitoring
# ---------------------------------------------------------------------------

def test_initialize_monitoring_creates_parquet_with_correct_schema(tmp_path) -> None:
    ensemble_dir = create_ensemble_directory(
        timeframe=TimeFrame.D,
        ensemble_name="test",
        direction=Direction.LONG,
        vault_root=str(tmp_path / "vault"),
    )
    signals = _make_signals()
    targets = _make_targets()

    path = initialize_monitoring(ensemble_dir, "rsi_signal_D", "continuous_binning_3", signals, targets)

    assert path.exists()
    df = pd.read_parquet(path, engine="fastparquet")
    assert set(df.columns) == {"signal", "target", "period"}
    assert df.index.name == "date"
    assert (df["period"] == "IS").all()
    assert len(df) == len(signals)


def test_initialize_monitoring_default_period_is_IS(tmp_path) -> None:
    ensemble_dir = create_ensemble_directory(
        timeframe=TimeFrame.D,
        ensemble_name="test",
        direction=Direction.LONG,
        vault_root=str(tmp_path / "vault"),
    )
    path = initialize_monitoring(
        ensemble_dir, "rsi_signal_D", "continuous_binning_3", _make_signals(), _make_targets()
    )
    df = pd.read_parquet(path, engine="fastparquet")
    assert (df["period"] == "IS").all()


def test_initialize_monitoring_raises_if_file_already_exists(tmp_path) -> None:
    ensemble_dir = create_ensemble_directory(
        timeframe=TimeFrame.D,
        ensemble_name="test",
        direction=Direction.LONG,
        vault_root=str(tmp_path / "vault"),
    )
    initialize_monitoring(
        ensemble_dir, "rsi_signal_D", "continuous_binning_3", _make_signals(), _make_targets()
    )
    with pytest.raises(FileExistsError, match="already exists"):
        initialize_monitoring(
            ensemble_dir, "rsi_signal_D", "continuous_binning_3", _make_signals(), _make_targets()
        )


def test_initialize_monitoring_creates_monitoring_subdirectory(tmp_path) -> None:
    ensemble_dir = create_ensemble_directory(
        timeframe=TimeFrame.D,
        ensemble_name="test",
        direction=Direction.LONG,
        vault_root=str(tmp_path / "vault"),
    )
    path = initialize_monitoring(
        ensemble_dir, "rsi_signal_D", "continuous_binning_3", _make_signals(), _make_targets()
    )
    assert path.parent.name == "monitoring"
    assert path.name == "rsi_signal_D__continuous_binning_3.parquet"


# ---------------------------------------------------------------------------
# append_monitoring_data
# ---------------------------------------------------------------------------

def test_append_monitoring_data_adds_rows_to_existing_file(tmp_path) -> None:
    ensemble_dir = create_ensemble_directory(
        timeframe=TimeFrame.D,
        ensemble_name="test",
        direction=Direction.LONG,
        vault_root=str(tmp_path / "vault"),
    )
    initialize_monitoring(
        ensemble_dir, "rsi_signal_D", "continuous_binning_3", _make_signals(50), _make_targets(50)
    )
    live_signals = _make_signals(20, seed=10)
    live_signals.index = pd.date_range("2020-03-01", periods=20, freq="D")
    live_targets = _make_targets(20, seed=11)
    live_targets.index = live_signals.index

    append_monitoring_data(
        ensemble_dir, "rsi_signal_D", "continuous_binning_3", live_signals, live_targets, period="LIVE"
    )

    df = load_monitoring_data(ensemble_dir, "rsi_signal_D", "continuous_binning_3")
    assert len(df) == 70
    assert set(df["period"].unique()) == {"IS", "LIVE"}


def test_append_monitoring_data_deduplicates_on_date(tmp_path) -> None:
    ensemble_dir = create_ensemble_directory(
        timeframe=TimeFrame.D,
        ensemble_name="test",
        direction=Direction.LONG,
        vault_root=str(tmp_path / "vault"),
    )
    signals = _make_signals(30)
    targets = _make_targets(30)
    initialize_monitoring(ensemble_dir, "rsi_signal_D", "continuous_binning_3", signals, targets)

    # Append overlapping dates — last value should win
    append_monitoring_data(
        ensemble_dir, "rsi_signal_D", "continuous_binning_3", signals, targets, period="LIVE"
    )
    df = load_monitoring_data(ensemble_dir, "rsi_signal_D", "continuous_binning_3")
    assert len(df) == 30  # no duplicate rows
    assert (df["period"] == "LIVE").all()  # last write wins


def test_append_monitoring_data_creates_file_if_missing(tmp_path) -> None:
    ensemble_dir = create_ensemble_directory(
        timeframe=TimeFrame.D,
        ensemble_name="test",
        direction=Direction.LONG,
        vault_root=str(tmp_path / "vault"),
    )
    signals = _make_signals()
    targets = _make_targets()

    # No initialize_monitoring call — append should create file
    append_monitoring_data(
        ensemble_dir, "rsi_signal_D", "continuous_binning_3", signals, targets, period="OOS"
    )
    df = load_monitoring_data(ensemble_dir, "rsi_signal_D", "continuous_binning_3")
    assert len(df) == len(signals)
    assert (df["period"] == "OOS").all()


# ---------------------------------------------------------------------------
# load_monitoring_data
# ---------------------------------------------------------------------------

def test_load_monitoring_data_returns_all_periods_by_default(tmp_path) -> None:
    ensemble_dir = create_ensemble_directory(
        timeframe=TimeFrame.D,
        ensemble_name="test",
        direction=Direction.LONG,
        vault_root=str(tmp_path / "vault"),
    )
    initialize_monitoring(
        ensemble_dir, "rsi_signal_D", "continuous_binning_3", _make_signals(30), _make_targets(30)
    )
    live_signals = _make_signals(10, seed=5)
    live_signals.index = pd.date_range("2020-04-01", periods=10, freq="D")
    live_targets = _make_targets(10, seed=6)
    live_targets.index = live_signals.index
    append_monitoring_data(
        ensemble_dir, "rsi_signal_D", "continuous_binning_3", live_signals, live_targets, period="LIVE"
    )

    df = load_monitoring_data(ensemble_dir, "rsi_signal_D", "continuous_binning_3")
    assert len(df) == 40
    assert set(df["period"].unique()) == {"IS", "LIVE"}


def test_load_monitoring_data_filters_by_period(tmp_path) -> None:
    ensemble_dir = create_ensemble_directory(
        timeframe=TimeFrame.D,
        ensemble_name="test",
        direction=Direction.LONG,
        vault_root=str(tmp_path / "vault"),
    )
    initialize_monitoring(
        ensemble_dir, "rsi_signal_D", "continuous_binning_3", _make_signals(30), _make_targets(30)
    )
    live_signals = _make_signals(10, seed=5)
    live_signals.index = pd.date_range("2020-04-01", periods=10, freq="D")
    live_targets = _make_targets(10, seed=6)
    live_targets.index = live_signals.index
    append_monitoring_data(
        ensemble_dir, "rsi_signal_D", "continuous_binning_3", live_signals, live_targets, period="LIVE"
    )

    is_df = load_monitoring_data(ensemble_dir, "rsi_signal_D", "continuous_binning_3", period="IS")
    assert len(is_df) == 30
    assert (is_df["period"] == "IS").all()


def test_load_monitoring_data_raises_if_file_missing(tmp_path) -> None:
    ensemble_dir = create_ensemble_directory(
        timeframe=TimeFrame.D,
        ensemble_name="test",
        direction=Direction.LONG,
        vault_root=str(tmp_path / "vault"),
    )
    with pytest.raises(FileNotFoundError, match="No monitoring data found"):
        load_monitoring_data(ensemble_dir, "rsi_signal_D", "continuous_binning_3")


# ---------------------------------------------------------------------------
# save_to_vault integration
# ---------------------------------------------------------------------------

def test_save_to_vault_fitted_model_auto_creates_monitoring(tmp_path) -> None:
    ensemble_dir = create_ensemble_directory(
        timeframe=TimeFrame.D,
        ensemble_name="rsi_test",
        direction=Direction.LONG,
        vault_root=str(tmp_path / "vault"),
    )
    model = _make_fitted_base_model()
    model_id = model.save_to_vault(ensemble_dir=ensemble_dir)

    monitoring_dir = Path(ensemble_dir) / "monitoring"
    # resolve relative to vault root
    from ensemble.vault_manager import _resolve_ensemble_path
    resolved = _resolve_ensemble_path(ensemble_dir) / "monitoring"
    parquet_files = list(resolved.glob("*.parquet"))
    assert len(parquet_files) == 1
    assert model_id in parquet_files[0].name

    df = pd.read_parquet(parquet_files[0], engine="fastparquet")
    assert set(df.columns) == {"signal", "target", "period"}
    assert (df["period"] == "IS").all()
    assert len(df) > 0


def test_save_to_vault_unfitted_model_skips_monitoring(tmp_path) -> None:
    ensemble_dir = create_ensemble_directory(
        timeframe=TimeFrame.D,
        ensemble_name="buy_hold_test",
        direction=Direction.LONG,
        vault_root=str(tmp_path / "vault"),
    )
    model = _make_unfitted_base_model()
    model.save_to_vault(ensemble_dir=ensemble_dir)

    from ensemble.vault_manager import _resolve_ensemble_path
    resolved = _resolve_ensemble_path(ensemble_dir) / "monitoring"
    assert not resolved.exists() or len(list(resolved.glob("*.parquet"))) == 0


def test_save_to_vault_monitoring_preexists_is_silenced(tmp_path) -> None:
    # If the monitoring file was seeded beforehand, save_to_vault should not
    # raise — the FileExistsError from initialize_monitoring is silenced inside
    # the hook so the caller never sees it.
    from ensemble.vault_manager import generate_model_id, _resolve_ensemble_path

    ensemble_dir = create_ensemble_directory(
        timeframe=TimeFrame.D,
        ensemble_name="rsi_test",
        direction=Direction.LONG,
        vault_root=str(tmp_path / "vault"),
    )
    model = _make_fitted_base_model()
    bm = model.binning_model
    signal_vector = bm.get_fitted_vector(strategy=bm.strategy)

    # Pre-seed the monitoring file with the model_id that save_to_vault will generate
    binning_params = dict(bm.get_params())
    binning_params.pop("strategy", None)
    model_id = generate_model_id(bm.model_type, binning_params, bias_node_params={"lookback": 2})
    initialize_monitoring(ensemble_dir, "rsi_signal_D", model_id, signal_vector, bm._training_target_data)

    # save_to_vault must NOT propagate the FileExistsError from monitoring
    try:
        model.save_to_vault(ensemble_dir=ensemble_dir)
    except FileExistsError as exc:
        pytest.fail(f"FileExistsError leaked from monitoring hook: {exc}")
