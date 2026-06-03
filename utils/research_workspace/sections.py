"""Report section assembly for research workspace dashboards."""
from __future__ import annotations

from utils.research_workspace.constants import RAW_DATA_SECTION_ID
from utils.research_workspace.models import PanelPolicy
from utils.research_workspace.panels import artifact_panel, artifact_panel_sort_key


def build_raw_data_section(raw_panels: list[dict[str, object]]) -> dict[str, object]:
    ordered = sorted(raw_panels, key=lambda panel: (str(panel["category"]), str(panel["title"])))
    return {
        "id": RAW_DATA_SECTION_ID,
        "label": "Raw data",
        "description": (
            "CSV and JSON exports for download or spot checks. "
            "Plots, tearsheets, and summaries stay in the sections above."
        ),
        "category": RAW_DATA_SECTION_ID,
        "priority": 999,
        "panels": ordered,
        "raw_panel_count": len(ordered),
        "primary_panel_id": None,
        "is_raw_data": True,
    }


def report_section(
    group: dict[str, object],
    policy: PanelPolicy,
) -> tuple[dict[str, object] | None, list[dict[str, object]]]:
    artifacts = list(group["artifacts"])
    ordered = sorted(artifacts, key=lambda item: artifact_panel_sort_key(item, policy))
    panels = [artifact_panel(artifact, index=index, policy=policy) for index, artifact in enumerate(ordered)]
    primary_panels = [panel for panel in panels if panel["panel_kind"] == "primary"]
    raw_panels = [panel for panel in panels if panel["panel_kind"] == "raw"]
    category = str(group["category"])
    if not primary_panels and category not in policy.always_visible_categories:
        return None, raw_panels
    featured = next((panel for panel in primary_panels if panel.get("is_primary")), None)
    return (
        {
            "id": category,
            "label": str(group["label"]),
            "description": str(group["description"]),
            "category": category,
            "priority": int(group["priority"]),
            "panels": primary_panels,
            "raw_panel_count": len(raw_panels),
            "primary_panel_id": None if featured is None else str(featured["panel_id"]),
            "is_raw_data": False,
        },
        raw_panels,
    )


def default_section_id(
    sections: list[dict[str, object]],
    preferred_order: tuple[str, ...],
) -> str | None:
    if not sections:
        return None
    by_id = {
        str(section["id"]): section
        for section in sections
        if str(section["id"]) != RAW_DATA_SECTION_ID
    }
    for section_id in preferred_order:
        if section_id in by_id:
            return section_id
    first = next(iter(by_id.values()), None)
    return None if first is None else str(first["id"])


def build_phase_report_sections(
    grouped_artifacts: list[dict[str, object]],
    *,
    policy: PanelPolicy,
    default_section_priority: tuple[str, ...],
) -> tuple[list[dict[str, object]], str | None]:
    """Build ordered report sections with primary panels and a dedicated raw-data section."""

    raw_panels: list[dict[str, object]] = []
    sections: list[dict[str, object]] = []
    for group in grouped_artifacts:
        if not group.get("artifacts"):
            continue
        section, section_raw = report_section(group, policy)
        raw_panels.extend(section_raw)
        if section is not None:
            sections.append(section)
    if raw_panels:
        sections.append(build_raw_data_section(raw_panels))
    default_id = default_section_id(
        [section for section in sections if section["id"] != RAW_DATA_SECTION_ID],
        default_section_priority,
    )
    return sections, default_id
