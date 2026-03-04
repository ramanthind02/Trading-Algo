"""Integration test for vault schema load/consolidation and Portfolio auto-load."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import utils.core.helpers as helpers
from ensemble.portfolio import Portfolio
from ensemble.vault_manager import (
    add_feature_to_ensemble,
    consolidate_feature_files,
    create_ensemble_directory,
    load_feature_base_models,
)
from feature_selection.base_models import ContinuousBinningModel
from feature_selection.base_models.feature_base_model import BaseModel
from utils.core.enums import Direction, TimeFrame, Ticker

USE_CACHE = True


def _base_model_for_rsi(tickers: list[Ticker]) -> BaseModel:
    return BaseModel(
        feature_config={
            "bias_node_spec": {
                "module_name": "rsi",
                "timeframes": [TimeFrame.D],
                "params": {"lookback": 2},
            }
        },
        tickers=tickers,
        binning_model=ContinuousBinningModel(n_bins=6, strategy="long"),
        use_cache=USE_CACHE,
    )


def _write_legacy_feature(path: Path, lookback: int, model_id: str) -> None:
    payload = {
        "feature_column": f"rsi_signal_D_lookback_{lookback}",
        "bias_node_spec": {
            "module_name": "rsi",
            "timeframes": ["D"],
            "params": {"lookback": lookback},
        },
        "tickers": ["ES", "NQ", "RTY", "YM"],
        "base_models": [
            {
                "model_id": model_id,
                "model_name": f"rsi_signal_D_lookback_{lookback}::{model_id}",
                "model_type": "QuantileBinningModel",
                "strategy": "long",
                "constructor_params": {
                    "n_bins": 10,
                    "bin_counts": [10, 8, 5, 3],
                    "selection_metric": "t_stat",
                },
                "is_fitted": False,
                "fitted_params": None,
            }
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def test_vault_portfolio_autoload_integration(tmp_path) -> None:
    tickers = [Ticker.ES, Ticker.NQ, Ticker.RTY, Ticker.YM]
    start = pd.Timestamp("2018-01-01")
    end = pd.Timestamp("2021-12-31")

    project_root = Path(__file__).resolve().parents[2]
    data_root = project_root / "data" / "ohlc_data"
    if not data_root.exists():
        pytest.skip("Missing data/ohlc_data for integration test")

    try:
        candles = helpers.load_data_multi_ticker(
            tickers=tickers,
            timeframe=TimeFrame.D,
            start=start.to_pydatetime(),
            end=end.to_pydatetime(),
        )
    except Exception as exc:
        pytest.skip(f"Unable to load persisted candles for integration test: {exc}")
    if candles.empty:
        pytest.skip("Persisted candle dataset empty for requested range")

    vault_root = tmp_path / "vault"
    ensemble_a = create_ensemble_directory(
        timeframe=TimeFrame.D,
        ensemble_name="mean-reversion_indices",
        direction=Direction.LONG,
        tickers=tickers,
        vault_root=str(vault_root),
    )
    ensemble_b = create_ensemble_directory(
        timeframe=TimeFrame.D,
        ensemble_name="trend_indices",
        direction=Direction.LONG,
        tickers=tickers,
        vault_root=str(vault_root),
    )

    base_model = _base_model_for_rsi(tickers)
    add_feature_to_ensemble(
        feature_name="rsi_signal_D",
        bias_node_spec={"module_name": "rsi", "timeframes": [TimeFrame.D]},
        bias_node_params={"lookback": 2},
        base_model=base_model,
        ensemble_dir=ensemble_a,
        tickers=tickers,
        members=[
            {
                "member_name": "cb10",
                "binning_model_type": "continuous_binning",
                "binning_model_params": {
                    "n_bins": 10,
                    "bin_counts": [10, 8, 5, 3],
                    "bin_index_min": 0,
                    "bin_index_max": None,
                    "selection_metric": "t_stat",
                    "normalize_by": "ewsd",
                },
                "requires_fit": True,
                "is_fitted": False,
                "fitted_params": None,
            }
        ],
    )
    # Second ensemble for portfolio auto-load coverage.
    add_feature_to_ensemble(
        feature_name="rsi_signal_D",
        bias_node_spec={"module_name": "rsi", "timeframes": [TimeFrame.D]},
        bias_node_params={"lookback": 5},
        base_model=_base_model_for_rsi(tickers),
        ensemble_dir=ensemble_b,
        tickers=tickers,
    )

    loaded_models = load_feature_base_models(
        feature_name="rsi_signal_D",
        ensemble_dir=ensemble_a,
    )
    assert loaded_models
    first_model = next(iter(loaded_models.values()))
    assert first_model.members
    assert first_model.members[0][1].is_fitted_ is False

    legacy_ensemble = create_ensemble_directory(
        timeframe=TimeFrame.D,
        ensemble_name="legacy_fixture",
        direction=Direction.LONG,
        tickers=tickers,
        vault_root=str(vault_root),
    )
    features_dir = Path(legacy_ensemble) / "features"
    _write_legacy_feature(features_dir / "rsi_signal_D_lookback_2.json", 2, "legacy_q10_2")
    _write_legacy_feature(features_dir / "rsi_signal_D_lookback_5.json", 5, "legacy_q10_5")

    consolidated = consolidate_feature_files(legacy_ensemble)
    canonical_path = features_dir / "rsi_signal_D.json"
    assert str(canonical_path) in consolidated
    with open(canonical_path, "r") as handle:
        merged = json.load(handle)
    assert len(merged["base_models"]) == 2

    portfolio_all = Portfolio(ensembles=None, vault_root=str(vault_root))
    assert len(portfolio_all.ensembles) >= 2
    portfolio_filtered = Portfolio(
        ensembles=None,
        vault_root=str(vault_root),
        ensemble_names=["mean-reversion_indices_long"],
    )
    assert len(portfolio_filtered.ensembles) == 1
