from __future__ import annotations

from pathlib import Path

import pandas as pd

from research.feature.shared import FeatureResearchPhase
from research.feature.ui.artifact_catalog import (
    PANEL_KIND_PRIMARY,
    PANEL_KIND_RAW,
    RAW_DATA_SECTION_ID,
    WorkspaceArtifactCategory,
    build_phase_report_sections,
    categorize_artifact_path,
    discover_artifact_records,
    ensure_phase_visualization_reports,
    group_artifact_records,
)


def test_categorize_artifact_path_maps_doc_referenced_outputs() -> None:
    assert (
        categorize_artifact_path(Path("reports/tearsheets/validation_ensemble_tearsheet.html"))
        is WorkspaceArtifactCategory.QUANTSTATS_TEARSHEET
    )
    assert (
        categorize_artifact_path(Path("visualization/matplotlib/param_sensitivity.png"))
        is WorkspaceArtifactCategory.MATPLOTLIB_REPORT
    )
    assert (
        categorize_artifact_path(Path("visualization/permutation_vector_shuffle.csv"))
        is WorkspaceArtifactCategory.PERMUTATION
    )
    assert (
        categorize_artifact_path(Path("reports/permutation_summary.md"))
        is WorkspaceArtifactCategory.PERMUTATION
    )
    assert (
        categorize_artifact_path(Path("visualization/param_sensitivity.csv"))
        is WorkspaceArtifactCategory.PARAMETER_SENSITIVITY
    )
    assert (
        categorize_artifact_path(Path("visualization/perturbation_report.json"))
        is WorkspaceArtifactCategory.PARAMETER_SENSITIVITY
    )
    assert (
        categorize_artifact_path(Path("visualization/equity_curve.csv"))
        is WorkspaceArtifactCategory.VISUALIZATION_DATA
    )
    assert (
        categorize_artifact_path(Path("reports/robustness_report.json"))
        is WorkspaceArtifactCategory.ROBUSTNESS
    )
    assert (
        categorize_artifact_path(Path("visualization/validation_robustness_report.json"))
        is WorkspaceArtifactCategory.ROBUSTNESS
    )
    assert (
        categorize_artifact_path(Path("visualization/portfolio_addition_report.json"))
        is WorkspaceArtifactCategory.ROBUSTNESS
    )
    assert (
        categorize_artifact_path(Path("visualization/validation_rank_correlation.csv"))
        is WorkspaceArtifactCategory.ROBUSTNESS
    )
    assert (
        categorize_artifact_path(Path("reports/matplotlib/cusum_stability.png"))
        is WorkspaceArtifactCategory.MATPLOTLIB_REPORT
    )
    assert (
        categorize_artifact_path(Path("reports/robustness_rolling_cusum.csv"))
        is WorkspaceArtifactCategory.ROBUSTNESS
    )
    assert (
        categorize_artifact_path(Path("validation/report.json"))
        is WorkspaceArtifactCategory.WALKFORWARD
    )


def test_discover_and_group_artifact_records(tmp_path: Path) -> None:
    exploration_dir = tmp_path / "exploration"
    viz_dir = tmp_path / "visualization" / "matplotlib"
    exploration_dir.mkdir(parents=True)
    viz_dir.mkdir(parents=True)

    (exploration_dir / "robustness_report.json").write_text("{}", encoding="utf-8")
    (exploration_dir / "perturbation_report.json").write_text("{}", encoding="utf-8")
    (exploration_dir / "permutation_summary.csv").write_text("param_combo\na", encoding="utf-8")
    pd.DataFrame({"param_combo_label": ["a"], "t_stat": [1.2]}).to_csv(
        tmp_path / "visualization" / "param_sensitivity.csv",
        index=False,
    )
    (viz_dir / "filter_gate_comparison.png").write_bytes(b"png")

    records = discover_artifact_records(
        (exploration_dir, tmp_path / "visualization", viz_dir),
        phase=FeatureResearchPhase.EXPLORATION,
        relative_to=tmp_path,
    )
    grouped = group_artifact_records(records, FeatureResearchPhase.EXPLORATION)

    categories = {group["category"] for group in grouped}
    assert WorkspaceArtifactCategory.MATPLOTLIB_REPORT.value in categories
    assert WorkspaceArtifactCategory.PARAMETER_SENSITIVITY.value in categories
    assert WorkspaceArtifactCategory.PERMUTATION.value in categories
    assert WorkspaceArtifactCategory.ROBUSTNESS.value in categories

    sections, default_section_id = build_phase_report_sections(
        grouped,
        FeatureResearchPhase.EXPLORATION,
    )
    assert default_section_id == WorkspaceArtifactCategory.MATPLOTLIB_REPORT.value
    sensitivity = next(section for section in sections if section["id"] == "parameter_sensitivity")
    assert any(
        panel["relative_path"].endswith("perturbation_report.json")
        for panel in sensitivity["panels"]
    )
    assert not any(
        panel["view_type"] == "embed" and "param_sensitivity" in panel["relative_path"]
        for panel in sensitivity["panels"]
    )


def test_build_phase_report_sections_separates_raw_panels() -> None:
    grouped = [
        {
            "category": WorkspaceArtifactCategory.ROBUSTNESS.value,
            "label": "Robustness Summaries",
            "description": "Robustness outputs.",
            "priority": 5,
            "artifacts": [
                {
                    "relative_path": "reports/matplotlib/cusum_stability.png",
                    "kind": "image",
                    "category": WorkspaceArtifactCategory.ROBUSTNESS.value,
                },
                {
                    "relative_path": "reports/robustness_rolling_cusum.csv",
                    "kind": "table",
                    "category": WorkspaceArtifactCategory.ROBUSTNESS.value,
                },
                {
                    "relative_path": "reports/robustness_report.json",
                    "kind": "json",
                    "category": WorkspaceArtifactCategory.ROBUSTNESS.value,
                },
            ],
        },
        {
            "category": WorkspaceArtifactCategory.PERMUTATION.value,
            "label": "Permutation Testing",
            "description": "Permutation outputs.",
            "priority": 3,
            "artifacts": [
                {
                    "relative_path": "reports/permutation_summary.md",
                    "kind": "markdown",
                    "category": WorkspaceArtifactCategory.PERMUTATION.value,
                },
                {
                    "relative_path": "viz/permutation_vector_shuffle.csv",
                    "kind": "table",
                    "category": WorkspaceArtifactCategory.PERMUTATION.value,
                },
            ],
        },
    ]
    sections, default_section_id = build_phase_report_sections(
        grouped,
        FeatureResearchPhase.EXPLORATION,
    )
    assert default_section_id == WorkspaceArtifactCategory.ROBUSTNESS.value
    assert default_section_id != RAW_DATA_SECTION_ID

    robustness = next(section for section in sections if section["id"] == "robustness")
    assert all(panel["panel_kind"] == PANEL_KIND_PRIMARY for panel in robustness["panels"])
    assert any(path.endswith("robustness_report.json") for path in (p["relative_path"] for p in robustness["panels"]))
    assert robustness["raw_panel_count"] == 1

    permutation = next(section for section in sections if section["id"] == "permutation")
    assert permutation["panels"][0]["panel_kind"] == PANEL_KIND_PRIMARY
    assert permutation["raw_panel_count"] == 1

    raw_section = next(section for section in sections if section["id"] == RAW_DATA_SECTION_ID)
    assert raw_section["is_raw_data"] is True
    assert len(raw_section["panels"]) == 2
    assert all(panel["panel_kind"] == PANEL_KIND_RAW for panel in raw_section["panels"])


def test_permutation_summary_markdown_stays_primary() -> None:
    grouped = [
        {
            "category": WorkspaceArtifactCategory.PERMUTATION.value,
            "label": "Permutation Testing",
            "description": "Permutation outputs.",
            "priority": 3,
            "artifacts": [
                {
                    "relative_path": "reports/permutation_summary.md",
                    "kind": "markdown",
                    "category": WorkspaceArtifactCategory.PERMUTATION.value,
                },
                {
                    "relative_path": "viz/permutation_vector_shuffle.csv",
                    "kind": "table",
                    "category": WorkspaceArtifactCategory.PERMUTATION.value,
                },
            ],
        }
    ]
    sections, _ = build_phase_report_sections(grouped, FeatureResearchPhase.EXPLORATION)
    permutation = next(section for section in sections if section["id"] == "permutation")
    assert permutation["panels"][0]["view_type"] == "markdown"
    assert permutation["panels"][0]["panel_kind"] == PANEL_KIND_PRIMARY


def test_build_phase_report_sections_assigns_unique_embed_panel_ids() -> None:
    grouped = [
        {
            "category": WorkspaceArtifactCategory.MATPLOTLIB_REPORT.value,
            "label": "Matplotlib Reports",
            "description": "PNG charts.",
            "priority": 1,
            "artifacts": [
                {
                    "relative_path": "viz/matplotlib/param_sensitivity.png",
                    "kind": "image",
                    "category": WorkspaceArtifactCategory.MATPLOTLIB_REPORT.value,
                },
                {
                    "relative_path": "viz/matplotlib/equity_curve.png",
                    "kind": "image",
                    "category": WorkspaceArtifactCategory.MATPLOTLIB_REPORT.value,
                },
                {
                    "relative_path": "viz/matplotlib/param_sensitivity_by_ticker.png",
                    "kind": "image",
                    "category": WorkspaceArtifactCategory.MATPLOTLIB_REPORT.value,
                },
            ],
        }
    ]
    sections, _default_section_id = build_phase_report_sections(
        grouped,
        FeatureResearchPhase.EXPLORATION,
    )
    matplotlib = next(section for section in sections if section["id"] == "matplotlib_report")
    panel_ids = [panel["panel_id"] for panel in matplotlib["panels"]]
    assert len(panel_ids) == len(set(panel_ids))
    assert all(panel["view_type"] == "embed" for panel in matplotlib["panels"])


def test_validation_phase_includes_robustness_category() -> None:
    from research.feature.ui.artifact_catalog import category_groups_for_phase

    categories = {group.category for group in category_groups_for_phase(FeatureResearchPhase.VALIDATION)}
    assert WorkspaceArtifactCategory.ROBUSTNESS in categories
    assert WorkspaceArtifactCategory.PERMUTATION not in categories


def test_build_phase_report_sections_prioritizes_tearsheets_for_validation() -> None:
    grouped = [
        {
            "category": WorkspaceArtifactCategory.QUANTSTATS_TEARSHEET.value,
            "label": "QuantStats Tearsheets",
            "description": "HTML tearsheets.",
            "priority": 0,
            "artifacts": [
                {
                    "relative_path": "out/tearsheets/walkforward_ensemble_tearsheet.html",
                    "kind": "html",
                    "category": WorkspaceArtifactCategory.QUANTSTATS_TEARSHEET.value,
                }
            ],
        }
    ]
    sections, default_section_id = build_phase_report_sections(
        grouped,
        FeatureResearchPhase.VALIDATION,
    )
    assert default_section_id == WorkspaceArtifactCategory.QUANTSTATS_TEARSHEET.value
    assert sections[0]["panels"][0]["view_type"] == "embed"


def test_ensure_phase_visualization_reports_generates_matplotlib_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_dir = tmp_path / "visualization"
    input_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "research_display_label": "A: Baseline",
                "gate_type": "A",
                "sharpe": 0.4,
                "n_nonzero_signal": 120,
            },
            {
                "research_display_label": "C: Vol gate",
                "gate_type": "C",
                "sharpe": 0.8,
                "n_nonzero_signal": 90,
            },
        ]
    ).to_csv(input_dir / "filter_exploration_summary.csv", index=False)

    monkeypatch.setattr(
        "research.feature.ui.artifact_catalog.canonical_in_sample_visualization_dir",
        lambda: input_dir,
    )
    config = type(
        "ConfigStub",
        (),
        {"output_root": tmp_path, "feature_type": type("FT", (), {"value": "signed_signal"})()},
    )()

    generated = ensure_phase_visualization_reports(
        config,
        FeatureResearchPhase.EXPLORATION,
    )

    assert generated
    assert (input_dir / "matplotlib" / "filter_gate_comparison.png").exists()
    assert (input_dir / "matplotlib" / "matplotlib_manifest.json").exists()
