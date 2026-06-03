"""Policy types for research workspace artifact layout."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class CategoryGroup:
    """One categorized bucket of workspace artifacts."""

    category: str
    label: str
    description: str
    priority: int


@dataclass(frozen=True)
class PanelPolicy:
    """Hooks that customize panel classification and titles per research app."""

    summary_json_names: frozenset[str] = frozenset()
    always_visible_categories: frozenset[str] = frozenset()
    is_featured_panel: Callable[[Path, str], bool] | None = None
    friendly_panel_title: Callable[[Path], str] | None = None
    panel_id_for_artifact: Callable[[dict[str, object], int], str] | None = None
