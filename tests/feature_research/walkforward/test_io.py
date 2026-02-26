from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from feature_research.walkforward.io import (
    resolve_walkforward_output_dir,
    write_walkforward_artifacts,
)
from feature_research.walkforward.runner import WalkforwardRunReport


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
            "top_k_features": ['["x=2","x=1"]', '["x=2","x=1"]'],
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


def _build_enhanced_report() -> WalkforwardRunReport:
    """WalkforwardRunReport with enhanced selection columns in fold_scores_df."""
    folds_df = pd.DataFrame(
        {
            "fold_id": [0],
            "train_start": [pd.Timestamp("2020-01-01")],
            "train_end": [pd.Timestamp("2020-01-31")],
            "test_start": [pd.Timestamp("2020-02-01")],
            "test_end": [pd.Timestamp("2020-02-29")],
            "train_samples": [31],
            "test_samples": [29],
        }
    )
    fold_scores_df = pd.DataFrame(
        {
            "fold_id": [0, 0, 0],
            "param_label": ["x=1", "x=2", "x=3"],
            "raw_objective": [0.4, 0.6, 0.5],
            "oos_objective": [0.3, 0.7, 0.65],
            "smoothed_objective": [0.45, 0.65, 0.62],
            "rank": [3, 1, 2],
            "selected_feature": [False, True, False],
            "trade_frequency": [0.6, 0.7, 0.65],
            "selected_in_top_k": [False, True, True],
        }
    )
    selection_summary_df = pd.DataFrame(
        {
            "fold_id": [0],
            "selected_feature": ["x=2"],
            "selected_raw_objective": [0.6],
            "selected_smoothed_objective": [0.65],
            "top_k_features": ['["x=2","x=1"]'],
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


def _build_figure() -> Figure:
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.plot([0, 1], [0, 1])
    return fig


def test_write_walkforward_artifacts_writes_required_files_and_columns(tmp_path: Path) -> None:
    report = _build_report()
    stability_figure = _build_figure()
    timeline_figure = _build_figure()

    try:
        paths = write_walkforward_artifacts(
            report=report,
            walkforward_stability_figure=stability_figure,
            fold_timeline_figure=timeline_figure,
            feature_type="continuous",
            module_name="rsi",
            root_dir=tmp_path,
        )
    finally:
        plt.close(stability_figure)
        plt.close(timeline_figure)

    assert paths.output_dir == tmp_path / "continuous" / "rsi" / "walkforward"
    assert paths.folds_csv.exists()
    assert paths.fold_scores_csv.exists()
    assert paths.selection_summary_csv.exists()
    assert paths.selected_params_detailed_csv.exists()
    assert paths.oos_metrics_csv.exists()
    assert paths.report_json.exists()
    assert paths.walkforward_stability_png.exists()
    assert paths.fold_timeline_png.exists()
    assert paths.summary_md.exists()
    assert paths.summary_html.exists()
    assert paths.tables_report_html.exists()

    folds_df = pd.read_csv(paths.folds_csv)
    assert folds_df.columns.tolist() == [
        "fold_id",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
        "train_samples",
        "test_samples",
    ]

    fold_scores_df = pd.read_csv(paths.fold_scores_csv)
    assert fold_scores_df.columns.tolist() == [
        "fold_id",
        "param_label",
        "raw_objective",
        "oos_objective",
        "smoothed_objective",
        "rank",
        "selected_feature",
    ]

    selection_summary_df = pd.read_csv(paths.selection_summary_csv)
    assert selection_summary_df.columns.tolist() == [
        "fold_id",
        "selected_feature",
        "selected_raw_objective",
        "selected_smoothed_objective",
        "top_k_features",
    ]

    detailed_df = pd.read_csv(paths.selected_params_detailed_csv)
    assert "param_x" in detailed_df.columns

    oos_metrics_df = pd.read_csv(paths.oos_metrics_csv)
    assert oos_metrics_df.columns.tolist() == ["metric", "value"]


def test_write_walkforward_artifacts_is_deterministic_for_same_inputs(tmp_path: Path) -> None:
    report = _build_report()
    stability_figure_first = _build_figure()
    timeline_figure_first = _build_figure()
    stability_figure_second = _build_figure()
    timeline_figure_second = _build_figure()

    try:
        first_paths = write_walkforward_artifacts(
            report=report,
            walkforward_stability_figure=stability_figure_first,
            fold_timeline_figure=timeline_figure_first,
            feature_type="continuous",
            module_name="rsi",
            root_dir=tmp_path,
        )
        first_json_bytes = first_paths.report_json.read_bytes()
        first_folds_csv = first_paths.folds_csv.read_text(encoding="utf-8")
        first_fold_scores_csv = first_paths.fold_scores_csv.read_text(encoding="utf-8")
        first_summary_csv = first_paths.selection_summary_csv.read_text(encoding="utf-8")
        first_selected_params_csv = first_paths.selected_params_detailed_csv.read_text(encoding="utf-8")
        first_oos_metrics_csv = first_paths.oos_metrics_csv.read_text(encoding="utf-8")
        first_summary_md = first_paths.summary_md.read_text(encoding="utf-8")
        first_summary_html = first_paths.summary_html.read_text(encoding="utf-8")
        first_tables_report_html = first_paths.tables_report_html.read_text(encoding="utf-8")

        second_paths = write_walkforward_artifacts(
            report=report,
            walkforward_stability_figure=stability_figure_second,
            fold_timeline_figure=timeline_figure_second,
            feature_type="continuous",
            module_name="rsi",
            root_dir=tmp_path,
        )
    finally:
        plt.close(stability_figure_first)
        plt.close(timeline_figure_first)
        plt.close(stability_figure_second)
        plt.close(timeline_figure_second)

    assert first_json_bytes == second_paths.report_json.read_bytes()
    assert first_folds_csv == second_paths.folds_csv.read_text(encoding="utf-8")
    assert first_fold_scores_csv == second_paths.fold_scores_csv.read_text(encoding="utf-8")
    assert first_summary_csv == second_paths.selection_summary_csv.read_text(encoding="utf-8")
    assert first_selected_params_csv == second_paths.selected_params_detailed_csv.read_text(encoding="utf-8")
    assert first_oos_metrics_csv == second_paths.oos_metrics_csv.read_text(encoding="utf-8")
    assert first_summary_md == second_paths.summary_md.read_text(encoding="utf-8")
    assert first_summary_html == second_paths.summary_html.read_text(encoding="utf-8")
    assert first_tables_report_html == second_paths.tables_report_html.read_text(encoding="utf-8")


def test_write_walkforward_artifacts_normalizes_metadata_identifiers_to_match_output_path(
    tmp_path: Path,
) -> None:
    report = _build_report()
    stability_figure_with_whitespace = _build_figure()
    timeline_figure_with_whitespace = _build_figure()
    stability_figure_trimmed = _build_figure()
    timeline_figure_trimmed = _build_figure()

    try:
        whitespace_paths = write_walkforward_artifacts(
            report=report,
            walkforward_stability_figure=stability_figure_with_whitespace,
            fold_timeline_figure=timeline_figure_with_whitespace,
            feature_type="  continuous  ",
            module_name="  rsi  ",
            root_dir=tmp_path,
        )
        whitespace_json_bytes = whitespace_paths.report_json.read_bytes()
        whitespace_payload = json.loads(whitespace_json_bytes.decode("utf-8"))

        trimmed_paths = write_walkforward_artifacts(
            report=report,
            walkforward_stability_figure=stability_figure_trimmed,
            fold_timeline_figure=timeline_figure_trimmed,
            feature_type="continuous",
            module_name="rsi",
            root_dir=tmp_path,
        )
    finally:
        plt.close(stability_figure_with_whitespace)
        plt.close(timeline_figure_with_whitespace)
        plt.close(stability_figure_trimmed)
        plt.close(timeline_figure_trimmed)

    assert whitespace_paths.output_dir == tmp_path / "continuous" / "rsi" / "walkforward"
    assert whitespace_payload["feature_type"] == "continuous"
    assert whitespace_payload["module_name"] == "rsi"
    assert whitespace_payload["output_dir"] == str(whitespace_paths.output_dir)
    assert whitespace_json_bytes == trimmed_paths.report_json.read_bytes()


def test_resolve_walkforward_output_dir_returns_expected_layout(tmp_path: Path) -> None:
    output_dir = resolve_walkforward_output_dir(
        feature_type="rule_based",
        module_name="atr_breakout",
        root_dir=tmp_path,
    )

    assert output_dir == tmp_path / "rule_based" / "atr_breakout" / "walkforward"


@pytest.mark.parametrize("feature_type,module_name", [("", "x"), ("x", ""), ("   ", "x"), ("x", "   ")])
def test_write_walkforward_artifacts_rejects_blank_identifiers(
    tmp_path: Path,
    feature_type: str,
    module_name: str,
) -> None:
    report = _build_report()
    stability_figure = _build_figure()
    timeline_figure = _build_figure()

    try:
        with pytest.raises(ValueError):
            write_walkforward_artifacts(
                report=report,
                walkforward_stability_figure=stability_figure,
                fold_timeline_figure=timeline_figure,
                feature_type=feature_type,
                module_name=module_name,
                root_dir=tmp_path,
            )
    finally:
        plt.close(stability_figure)
        plt.close(timeline_figure)


def test_write_walkforward_artifacts_includes_enhanced_columns_when_present(
    tmp_path: Path,
) -> None:
    report = _build_enhanced_report()
    stability_figure = _build_figure()
    timeline_figure = _build_figure()
    try:
        paths = write_walkforward_artifacts(
            report=report,
            walkforward_stability_figure=stability_figure,
            fold_timeline_figure=timeline_figure,
            feature_type="continuous",
            module_name="rsi",
            root_dir=tmp_path,
        )
    finally:
        plt.close(stability_figure)
        plt.close(timeline_figure)

    written = pd.read_csv(paths.fold_scores_csv)
    assert written.columns.tolist() == [
        "fold_id",
        "param_label",
        "raw_objective",
        "oos_objective",
        "smoothed_objective",
        "rank",
        "selected_feature",
        "trade_frequency",
        "selected_in_top_k",
    ]


def test_write_walkforward_artifacts_legacy_report_omits_enhanced_columns(
    tmp_path: Path,
) -> None:
    report = _build_report()
    stability_figure = _build_figure()
    timeline_figure = _build_figure()
    try:
        paths = write_walkforward_artifacts(
            report=report,
            walkforward_stability_figure=stability_figure,
            fold_timeline_figure=timeline_figure,
            feature_type="continuous",
            module_name="rsi",
            root_dir=tmp_path,
        )
    finally:
        plt.close(stability_figure)
        plt.close(timeline_figure)

    written = pd.read_csv(paths.fold_scores_csv)
    assert "trade_frequency" not in written.columns
    assert "selected_in_top_k" not in written.columns
    assert written.columns.tolist() == [
        "fold_id",
        "param_label",
        "raw_objective",
        "oos_objective",
        "smoothed_objective",
        "rank",
        "selected_feature",
    ]


def test_write_walkforward_artifacts_selected_params_detailed_uses_selected_in_top_k_rows(
    tmp_path: Path,
) -> None:
    report = _build_enhanced_report()
    stability_figure = _build_figure()
    timeline_figure = _build_figure()
    try:
        paths = write_walkforward_artifacts(
            report=report,
            walkforward_stability_figure=stability_figure,
            fold_timeline_figure=timeline_figure,
            feature_type="continuous",
            module_name="rsi",
            root_dir=tmp_path,
        )
    finally:
        plt.close(stability_figure)
        plt.close(timeline_figure)

    selected_params = pd.read_csv(paths.selected_params_detailed_csv)
    assert set(selected_params["param_label"]) == {"x=2", "x=3"}


def test_write_walkforward_artifacts_includes_portfolio_simulation_section(
    tmp_path: Path,
) -> None:
    report = _build_report()
    report = WalkforwardRunReport(
        folds_df=report.folds_df,
        fold_scores_df=report.fold_scores_df,
        selection_summary_df=report.selection_summary_df,
        portfolio_results_df=pd.DataFrame(
            [
                {
                    "fold_id": 0,
                    "oos_portfolio_sharpe": 0.42,
                    "n_params_selected": 2,
                    "error": "",
                }
            ]
        ),
    )
    stability_figure = _build_figure()
    timeline_figure = _build_figure()
    try:
        paths = write_walkforward_artifacts(
            report=report,
            walkforward_stability_figure=stability_figure,
            fold_timeline_figure=timeline_figure,
            feature_type="continuous",
            module_name="rsi",
            root_dir=tmp_path,
        )
    finally:
        plt.close(stability_figure)
        plt.close(timeline_figure)

    summary_md = paths.summary_md.read_text(encoding="utf-8")
    assert "Portfolio Simulation (Stage 2)" in summary_md
    oos_metrics_df = pd.read_csv(paths.oos_metrics_csv)
    assert "mean_oos_portfolio_sharpe" in set(oos_metrics_df["metric"])
