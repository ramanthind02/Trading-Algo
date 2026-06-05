"""Unit tests for portfolio_research.weight_layer_export."""

from __future__ import annotations

import pandas as pd
import pytest

from research.portfolio.weight_layer_export import weight_layer_diagnostics_to_dataframe


def test_unfitted_diagnostics_yields_empty_typed_frame() -> None:
    df = weight_layer_diagnostics_to_dataframe(
        {"is_fitted": False},
        phase="Train",
        fit_start=pd.Timestamp("2020-01-01"),
        fit_end=pd.Timestamp("2020-12-31"),
        predict_start=pd.Timestamp("2020-01-01"),
        predict_end=pd.Timestamp("2020-12-31"),
    )
    assert df.empty
    assert "stream_or_model_id" in df.columns
    assert "stream_source_model_name" in df.columns


def test_global_streams_parsed_without_hardcoded_model_names() -> None:
    diagnostics = {
        "is_fitted": True,
        "weight_method": "hierarchy_equal",
        "fdm_max": 2.0,
        "tickers": {
            "__GLOBAL__": {
                "fdm": 1.25,
                "weights": {
                    "ES::D::model_a__D::ensemble_0": 0.6,
                    "NQ::M::model_b__M::ensemble_1": 0.4,
                },
                "mean_signal_correlation": 0.11,
                "mean_cluster_correlation": 0.22,
                "cluster_assignments": {
                    "ES::D::model_a__D::ensemble_0": "cluster_1",
                    "NQ::M::model_b__M::ensemble_1": "cluster_2",
                },
            }
        },
    }
    df = weight_layer_diagnostics_to_dataframe(
        diagnostics,
        phase="Validation",
        fit_start=pd.Timestamp("2019-06-01"),
        fit_end=pd.Timestamp("2019-12-31"),
        predict_start=pd.Timestamp("2020-01-01"),
        predict_end=pd.Timestamp("2020-06-30"),
    )
    assert len(df) == 2
    assert set(df["stream_instrument_ticker"].tolist()) == {"ES", "NQ"}
    assert set(df["stream_signal_timeframe"].tolist()) == {"D", "M"}
    assert "model_a__D::ensemble_0" in df["stream_source_model_name"].tolist()
    assert "model_b__M::ensemble_1" in df["stream_source_model_name"].tolist()
    assert df["phase"].tolist() == ["Validation", "Validation"]
    assert df["stream_weight"].sum() == pytest.approx(1.0)


def test_malformed_stream_id_leaves_decode_columns_null() -> None:
    diagnostics = {
        "is_fitted": True,
        "weight_method": "equal_signal",
        "fdm_max": 2.0,
        "tickers": {
            "__GLOBAL__": {
                "fdm": 1.0,
                "weights": {"not_a_triple_colon_id": 1.0},
                "mean_signal_correlation": 0.0,
                "mean_cluster_correlation": 0.0,
                "cluster_assignments": {},
            }
        },
    }
    df = weight_layer_diagnostics_to_dataframe(
        diagnostics,
        phase="Test",
        fit_start=pd.Timestamp("2021-01-01"),
        fit_end=pd.Timestamp("2021-12-31"),
        predict_start=pd.Timestamp("2022-01-01"),
        predict_end=pd.Timestamp("2022-12-31"),
    )
    assert len(df) == 1
    assert pd.isna(df.loc[0, "stream_instrument_ticker"])
    assert pd.isna(df.loc[0, "stream_signal_timeframe"])
    assert pd.isna(df.loc[0, "stream_source_model_name"])
