from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from research.feature.shared import FeatureResearchPhase
from research.feature.ui.artifact_catalog import RAW_DATA_SECTION_ID
from research.feature.ui.workspace import build_workspace_view, load_artifact_preview


def test_load_artifact_preview_builds_table_and_chart(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("research.feature.ui.workspace._REPO_ROOT", tmp_path)
    artifact = tmp_path / "reports" / "equity_curve_validation_only.csv"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "datetime": ["2024-01-01", "2024-01-02"],
            "cumulative_strategy_return": [0.0, 0.12],
        }
    ).to_csv(artifact, index=False)

    preview = load_artifact_preview("reports/equity_curve_validation_only.csv")

    assert preview["kind"] == "table"
    assert preview["row_count"] == 2
    assert preview["chart"]["label"] == "cumulative_strategy_return"


def test_build_workspace_view_discovers_phase_artifacts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    exploration_dir = tmp_path / "exploration"
    validation_dir = tmp_path / "validation"
    oos_dir = tmp_path / "oos"
    exploration_dir.mkdir()
    validation_dir.mkdir()
    oos_dir.mkdir()

    (exploration_dir / "robustness_report.json").write_text(
        json.dumps(
            {
                "feature_name": "stacked_sma_long_only",
                "n_combinations": 6,
                "n_effective": {"n_effective": 2.5},
            }
        ),
        encoding="utf-8",
    )
    (validation_dir / "report.json").write_text(
        json.dumps(
            {
                "module_name": "stacked_sma_long_only",
                "objective_metric_name": "t_stat",
                "tearsheet_files": ["validation_ensemble_tearsheet.html"],
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "datetime": ["2024-01-01"],
            "cumulative_strategy_return": [0.15],
        }
    ).to_csv(validation_dir / "equity_curve_validation_only.csv", index=False)

    monkeypatch.setattr(
        "research.feature.ui.workspace.resolve_ui_config",
        lambda config, request: (config, None),
    )
    monkeypatch.setattr(
        "research.feature.ui.workspace.phase_discovery_roots",
        lambda config, phase: {
            FeatureResearchPhase.EXPLORATION: (exploration_dir,),
            FeatureResearchPhase.VALIDATION: (validation_dir,),
            FeatureResearchPhase.PORTFOLIO_ADDITION: (oos_dir,),
        }[phase],
    )
    monkeypatch.setattr(
        "research.feature.ui.workspace._vault_commit_view",
        lambda config, request: {"enabled": False},
    )
    monkeypatch.setattr("research.feature.ui.workspace._REPO_ROOT", tmp_path)

    workspace = build_workspace_view(
        config=SimpleNamespace(),
        request=SimpleNamespace(phase=FeatureResearchPhase.EXPLORATION),
        current_job=None,
    )

    by_phase = {phase["phase"]: phase for phase in workspace["phases"]}
    assert by_phase["exploration"]["artifact_counts"]["json"] == 1
    assert by_phase["validation"]["artifact_counts"]["json"] == 1
    exploration = by_phase["exploration"]
    assert exploration["report_sections"]
    robustness_section = next(
        section for section in exploration["report_sections"] if section["id"] == "robustness"
    )
    assert any(
        panel["relative_path"].endswith("robustness_report.json")
        for panel in robustness_section["panels"]
    )
    assert exploration["default_section_id"] == "robustness"
    assert any(card["label"] == "Effective combos" for card in exploration["summary_cards"])
    assert any(card["label"] == "Latest cumulative return" for card in by_phase["validation"]["summary_cards"])
