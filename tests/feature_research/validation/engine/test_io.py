from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from research.evaluation.walkforward.io import (
    _build_selected_params_detailed,
    resolve_walkforward_output_dir,
    write_walkforward_artifacts,
)
from research.evaluation.walkforward.runner import WalkforwardRunReport
from research.evaluation.walkforward.selected_params_codec import serialize_selected_params


def _build_report() -> WalkforwardRunReport:
    folds_df = pd.DataFrame(
        {
            "fold_id": [0, 1],
            "train_start": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01")],
            "train_end": [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29")],
            "test_start": [pd.Timestamp("2020-02-01"), pd.Timestamp("2020-03-01")],
            "test_end": [pd.Timestamp("2020-02-29"), pd.Timestamp("2020-03-31")],
            "train_samples": [31, 29],
            "test_samples": [29, 31],
        }
    )
    fold_scores_df = pd.DataFrame(
        {
            "fold_id": [0, 0, 1, 1],
            "param_label": ["x=1", "x=2", "x=1", "x=2"],
            "raw_objective": [0.4, 0.6, 0.5, 0.7],
            "oos_objective": [0.3, 0.8, 0.4, 0.9],
            "smoothed_objective": [0.45, 0.65, 0.55, 0.75],
            "rank": [2, 1, 2, 1],
            "selected_feature": [False, True, False, True],
        }
    )
    selection_summary_df = pd.DataFrame(
        {
            "fold_id": [0, 1],
            "selected_feature": ["x=2", "x=2"],
            "selected_raw_objective": [0.6, 0.7],
            "selected_smoothed_objective": [0.65, 0.75],
            "selected_params_json": [
                serialize_selected_params({"x": 2}),
                serialize_selected_params({"x": 2}),
            ],
        }
    )
    return WalkforwardRunReport(
        folds_df=folds_df,
        fold_scores_df=fold_scores_df,
        selection_summary_df=selection_summary_df,
        portfolio_results_df=pd.DataFrame(
            columns=["fold_id", "oos_portfolio_sharpe", "n_params_selected", "error"]
        ),
    )


def test_write_walkforward_artifacts_writes_required_files_and_columns(tmp_path: Path) -> None:
    report = _build_report()
    paths = write_walkforward_artifacts(
        report=report,
        feature_type="continuous",
        module_name="rsi",
        root_dir=tmp_path,
    )

    assert paths.output_dir == tmp_path / "continuous" / "rsi" / "walkforward"
    assert paths.tearsheets_dir == paths.output_dir / "tearsheets"
    assert paths.report_json.exists()

    payload = json.loads(paths.report_json.read_text(encoding="utf-8"))
    assert payload["feature_type"] == "continuous"
    assert payload["module_name"] == "rsi"
    assert payload["output_dir"] == str(paths.output_dir)
    assert payload["tearsheets_dir"] == str(paths.tearsheets_dir)
    assert "tearsheet_files" in payload
    assert isinstance(payload["tearsheet_files"], list)
    assert "research_context" in payload
    assert "objective_metric_name" in payload
    assert "timeframe" in payload


def test_write_walkforward_artifacts_is_deterministic_for_same_inputs(tmp_path: Path) -> None:
    report = _build_report()
    first_paths = write_walkforward_artifacts(
        report=report,
        feature_type="continuous",
        module_name="rsi",
        root_dir=tmp_path,
    )
    first_json_bytes = first_paths.report_json.read_bytes()

    second_paths = write_walkforward_artifacts(
        report=report,
        feature_type="continuous",
        module_name="rsi",
        root_dir=tmp_path,
    )

    assert first_json_bytes == second_paths.report_json.read_bytes()


def test_write_walkforward_artifacts_normalizes_metadata_identifiers_to_match_output_path(
    tmp_path: Path,
) -> None:
    report = _build_report()
    whitespace_paths = write_walkforward_artifacts(
        report=report,
        feature_type="  signed_signal  ",
        module_name="  rsi  ",
        root_dir=tmp_path,
    )
    whitespace_json_bytes = whitespace_paths.report_json.read_bytes()
    whitespace_payload = json.loads(whitespace_json_bytes.decode("utf-8"))

    trimmed_paths = write_walkforward_artifacts(
        report=report,
        feature_type="signed_signal",
        module_name="rsi",
        root_dir=tmp_path,
    )

    assert whitespace_paths.output_dir == tmp_path / "signed_signal" / "rsi" / "walkforward"
    assert whitespace_payload["feature_type"] == "signed_signal"
    assert whitespace_payload["module_name"] == "rsi"
    assert whitespace_payload["output_dir"] == str(whitespace_paths.output_dir)
    assert whitespace_json_bytes == trimmed_paths.report_json.read_bytes()


def test_resolve_walkforward_output_dir_returns_expected_layout(tmp_path: Path) -> None:
    output_dir = resolve_walkforward_output_dir(
        feature_type="signed_signal",
        module_name="atr_breakout",
        root_dir=tmp_path,
    )

    assert output_dir == tmp_path / "signed_signal" / "atr_breakout" / "walkforward"


@pytest.mark.parametrize("feature_type,module_name", [("", "x"), ("x", ""), ("   ", "x"), ("x", "   ")])
def test_write_walkforward_artifacts_rejects_blank_identifiers(
    tmp_path: Path,
    feature_type: str,
    module_name: str,
) -> None:
    report = _build_report()
    with pytest.raises(ValueError):
        write_walkforward_artifacts(
            report=report,
            feature_type=feature_type,
            module_name=module_name,
            root_dir=tmp_path,
        )


def test_build_selected_params_detailed_uses_selected_feature_rows() -> None:
    """Rows with ``selected_feature`` True appear in the detailed export."""
    report = _build_report()
    detailed = _build_selected_params_detailed(report, {})
    assert set(detailed["param_label"]) == {"x=2"}


def test_write_walkforward_artifacts_includes_research_context_and_last_fold_test_end(
    tmp_path: Path,
) -> None:
    report = _build_report()
    paths = write_walkforward_artifacts(
        report=report,
        feature_type="signed_signal",
        module_name="rsi",
        root_dir=tmp_path,
        research_context={"custom_key": "custom_value"},
    )
    payload = json.loads(paths.report_json.read_text(encoding="utf-8"))
    ctx = payload["research_context"]
    assert ctx["custom_key"] == "custom_value"
    assert ctx["last_fold_test_end"] == "2020-03-31"
