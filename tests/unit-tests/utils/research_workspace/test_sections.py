from __future__ import annotations

from pathlib import Path

from research.workspace.models import CategoryGroup, PanelPolicy
from research.workspace.sections import build_phase_report_sections


def test_build_phase_report_sections_splits_primary_and_raw() -> None:
    grouped = [
        {
            "category": "robustness",
            "label": "Robustness",
            "description": "Summary outputs",
            "priority": 0,
            "artifacts": [
                {
                    "relative_path": "out/validation_robustness_report.json",
                    "kind": "json",
                    "category": "robustness",
                },
                {
                    "relative_path": "out/validation_z_cusum_sr.csv",
                    "kind": "table",
                    "category": "robustness",
                },
            ],
        }
    ]
    policy = PanelPolicy(
        summary_json_names=frozenset({"validation_robustness_report.json"}),
        always_visible_categories=frozenset({"robustness"}),
    )
    sections, default_id = build_phase_report_sections(
        grouped,
        policy=policy,
        default_section_priority=("robustness",),
    )
    assert default_id == "robustness"
    assert len(sections) == 2
    primary = sections[0]
    raw = sections[1]
    assert primary["id"] == "robustness"
    assert len(primary["panels"]) == 1
    assert primary["panels"][0]["view_type"] == "json"
    assert raw["id"] == "raw_data"
    assert len(raw["panels"]) == 1
    assert raw["panels"][0]["view_type"] == "table"
