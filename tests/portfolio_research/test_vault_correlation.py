"""Unit tests for vault vs research OOS correlation helpers (portfolio_research)."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from research.portfolio.correlation_export import write_vault_correlation_visualization_long
from research.portfolio.vault_correlation import (
    build_vault_correlation_long_rows,
    candidate_returns_long_frame,
    rows_for_vault_member_correlations,
)
from lib.core.enums import TimeFrame, Ticker
from lib.core.helpers import build_feature_column_name
from research.evaluation.walkforward.selected_params_codec import serialize_selected_params


def _minimal_signed_feature_json(stem: str, *, tf: str = "D") -> str:
    return f"""{{
  "feature_name": "{stem}",
  "bias_node_spec": {{
    "module_name": "buy_hold",
    "timeframes": ["{tf}"],
    "params": {{}}
  }},
  "tickers": ["ES"],
  "base_models": [{{
    "model_id": "m1",
    "model_name": "{stem}::m1",
    "strategy": "long",
    "model_type": "signed_signal",
    "feature_column": "{stem}",
    "bias_node_spec": {{
      "module_name": "buy_hold",
      "timeframes": ["{tf}"],
      "params": {{}}
    }}
  }}]
}}"""


def test_write_vault_correlation_visualization_long_schema(tmp_path: Path) -> None:
    rows = [
        {
            "fold_id": 0,
            "window_kind": "extended_train_val_test",
            "research_param_combo_label": "p=1",
            "research_feature_name": "f_ES",
            "vault_member_id": "D/e/f",
            "vault_ensemble_path": "D/e",
            "vault_feature_name": "vf_ES",
            "ticker": "ES",
            "metric_name": "pearson_return_corr",
            "metric_value": 0.5,
            "n_obs": 10,
        }
    ]
    out = write_vault_correlation_visualization_long(rows, tmp_path / "vault_correlation_long")
    assert out.exists()
    df = pd.read_csv(out)
    assert list(df.columns) == [
        "fold_id",
        "window_kind",
        "research_param_combo_label",
        "research_feature_name",
        "vault_member_id",
        "vault_ensemble_path",
        "vault_feature_name",
        "ticker",
        "metric_name",
        "metric_value",
        "n_obs",
    ]


def test_candidate_returns_long_frame_matches_combo_key() -> None:
    idx = pd.DatetimeIndex(pd.date_range("2020-01-01", periods=3, freq="D"))
    params = {"x": 1}
    key = __import__(
        "research.feature.core_helpers", fromlist=["combo_key"]
    ).combo_key(params)
    paired = pd.DataFrame(
        {
            "signal": [1.0, -1.0, 0.0],
            "target": [0.01, 0.02, 0.03],
            "ticker": ["ES", "ES", "ES"],
        },
        index=idx,
    )
    combo_map = {key: paired}
    out = candidate_returns_long_frame(
        combo_map,
        params,
        extended_start=datetime(2020, 1, 1),
        extended_end=datetime(2020, 1, 3),
    )
    assert len(out) == 3
    assert list(out.columns) == ["datetime", "ticker", "research_return"]
    assert out["research_return"].tolist() == pytest.approx([0.01, -0.02, 0.0])


def test_rows_for_vault_member_empty_returns_no_rows() -> None:
    rows = rows_for_vault_member_correlations(
        fold_id=0,
        window_kind="extended_train_val_test",
        research_param_combo_label="a",
        research_feature_name_base="rf",
        vault_member_id="D/e/features/f",
        vault_ensemble_path="D/e",
        vault_feature_name_base="vf",
        candidate_df=pd.DataFrame({"datetime": [], "ticker": [], "research_return": []}),
        vault_df=pd.DataFrame(),
    )
    assert rows == []


def test_build_vault_correlation_long_rows_with_mocks(tmp_path: Path) -> None:
    feat_dir = tmp_path / "vault" / "D" / "demo_ens" / "features"
    feat_dir.mkdir(parents=True)
    stem = "demo_signal_D"
    (feat_dir / f"{stem}.json").write_text(_minimal_signed_feature_json(stem), encoding="utf-8")

    params = {"rsi_period": 14}
    idx = pd.DatetimeIndex(pd.date_range("2020-01-01", periods=5, freq="D"))
    targets = [0.02, -0.01, 0.03, -0.015, 0.01]
    paired = pd.DataFrame(
        {
            "signal": [1.0] * 5,
            "target": targets,
            "ticker": ["ES"] * 5,
        },
        index=idx,
    )
    from research.feature.core_helpers import combo_key as ck

    combo_signal_target = {ck(params): paired}
    summary = pd.DataFrame(
        {
            "fold_id": [0],
            "selected_params_json": [serialize_selected_params(params)],
        }
    )

    idx2 = pd.DatetimeIndex(pd.date_range("2020-01-01", periods=5, freq="D"))
    vault_targets = targets
    features_df = pd.DataFrame(
        {"buy_hold": [1.0] * 5, "ticker": ["ES"] * 5},
        index=idx2,
    )
    targets_df = pd.DataFrame(
        {"log_return_ewsd": vault_targets, "ticker": ["ES"] * 5},
        index=idx2,
    )

    eval_spec = {
        "module_name": "rsi_signal",
        "timeframes": [TimeFrame.D],
        "params": params,
    }
    feat_name = build_feature_column_name("rsi_signal", "signal", TimeFrame.D, params)

    with patch(
        "research.portfolio.vault_correlation.extract_features_for_bias_node",
        return_value=(features_df, targets_df),
    ), patch(
        "research.portfolio.vault_correlation.populate_cache_if_needed",
        autospec=True,
    ):
        rows = build_vault_correlation_long_rows(
            combo_signal_target=combo_signal_target,
            selection_summary_df=summary,
            module_name="rsi_signal",
            research_timeframe=TimeFrame.D,
            research_eval_bias_spec=eval_spec,
            target_col="log_return_ewsd",
            extended_start=datetime(2020, 1, 1),
            extended_end=datetime(2020, 1, 7),
            vault_root=tmp_path / "vault",
        )

    assert rows
    pearson_rows = [r for r in rows if r.get("metric_name") == "pearson_return_corr"]
    assert pearson_rows
    assert pearson_rows[0]["ticker"] == "ES"
    assert pearson_rows[0]["research_feature_name"] == f"{feat_name}_ES"
    assert pearson_rows[0]["vault_feature_name"] == f"{stem}_ES"
    assert pearson_rows[0]["vault_ensemble_path"] == "D/demo_ens"
    assert float(pearson_rows[0]["metric_value"]) == pytest.approx(1.0, abs=1e-9)
    metric_names = {r["metric_name"] for r in rows}
    assert "pearson_drawdown_corr" in metric_names
    assert "spearman_drawdown_corr" in metric_names


def test_build_vault_correlation_skips_mismatched_timeframe(tmp_path: Path) -> None:
    feat_dir = tmp_path / "vault" / "W" / "weekly_ens" / "features"
    feat_dir.mkdir(parents=True)
    stem = "demo_signal_W"
    (feat_dir / f"{stem}.json").write_text(_minimal_signed_feature_json(stem, tf="W"), encoding="utf-8")

    params = {"rsi_period": 14}
    idx = pd.DatetimeIndex(pd.date_range("2020-01-01", periods=3, freq="D"))
    paired = pd.DataFrame(
        {
            "signal": [1.0] * 3,
            "target": [0.01] * 3,
            "ticker": ["ES"] * 3,
        },
        index=idx,
    )
    from research.feature.core_helpers import combo_key as ck

    combo_signal_target = {ck(params): paired}
    summary = pd.DataFrame(
        {
            "fold_id": [0],
            "selected_params_json": [serialize_selected_params(params)],
        }
    )
    eval_spec = {
        "module_name": "rsi_signal",
        "timeframes": [TimeFrame.D],
        "params": params,
    }

    rows = build_vault_correlation_long_rows(
        combo_signal_target=combo_signal_target,
        selection_summary_df=summary,
        module_name="rsi_signal",
        research_timeframe=TimeFrame.D,
        research_eval_bias_spec=eval_spec,
        target_col="log_return_ewsd",
        extended_start=datetime(2020, 1, 1),
        extended_end=datetime(2020, 1, 5),
        vault_root=tmp_path / "vault",
    )
    assert rows == []


def test_build_feature_column_name_flattens_nested_params_for_safe_paths() -> None:
    """Nested dict params (e.g. filter_gate) must not use str(dict); Windows rejects : ' {{}} in filenames."""
    params = {
        "filter_module": "sma_above_filter",
        "filter_params": {"period": 200},
        "signal_module": "consec_momentum",
        "signal_params": {"lookback": 40, "consecutive_bars": 3, "is_buy": True},
    }
    name = build_feature_column_name("filter_gate", "signal", TimeFrame.D, params)
    assert "{" not in name and "'" not in name and ":" not in name
    assert "period_200" in name
    assert "consec_momentum" in name
