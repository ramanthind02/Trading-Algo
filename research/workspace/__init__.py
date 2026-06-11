"""Shared research workspace artifact discovery, sections, and preview."""
from research.workspace.constants import (
    PANEL_KIND_PRIMARY,
    PANEL_KIND_RAW,
    RAW_DATA_SECTION_ID,
    SUPPORTED_SUFFIXES,
)
from research.workspace.discovery import (
    discover_artifact_records,
    group_artifact_records,
)
from research.workspace.models import CategoryGroup, PanelPolicy
from research.workspace.preview import load_artifact_preview, resolve_artifact_path
from research.workspace.sections import build_phase_report_sections

__all__ = [
    "CategoryGroup",
    "PANEL_KIND_PRIMARY",
    "PANEL_KIND_RAW",
    "PanelPolicy",
    "RAW_DATA_SECTION_ID",
    "SUPPORTED_SUFFIXES",
    "build_phase_report_sections",
    "discover_artifact_records",
    "group_artifact_records",
    "load_artifact_preview",
    "resolve_artifact_path",
]
