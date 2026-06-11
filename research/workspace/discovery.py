"""Filesystem discovery for research workspace artifacts."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from research.workspace.constants import SUPPORTED_SUFFIXES


def artifact_kind(suffix: str) -> str:
    if suffix == ".csv":
        return "table"
    if suffix == ".json":
        return "json"
    if suffix == ".md":
        return "markdown"
    if suffix == ".html":
        return "html"
    if suffix == ".png":
        return "image"
    return "text"


def preview_kind(suffix: str) -> str:
    if suffix in {".csv", ".json", ".md", ".txt"}:
        return "inline"
    return "embed"


def dedupe_artifact_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    seen: set[str] = set()
    unique: list[dict[str, object]] = []
    for record in records:
        path = str(record["relative_path"])
        if path in seen:
            continue
        seen.add(path)
        unique.append(record)
    return unique


def artifact_record(
    path: Path,
    *,
    phase: str,
    root: Path,
    relative_to: Path,
    categorize: Callable[[Path], str],
    category_label: Callable[[str], str] | None = None,
) -> dict[str, object]:
    category = categorize(path)
    suffix = path.suffix.lower()
    label_fn = category_label or (lambda value: value.replace("_", " ").title())
    resolved = path.resolve()
    try:
        rel = resolved.relative_to(relative_to.resolve()).as_posix()
    except ValueError:
        rel = str(resolved)
    return {
        "phase": phase,
        "name": path.name,
        "relative_path": rel,
        "kind": artifact_kind(suffix),
        "category": category,
        "category_label": label_fn(category),
        "size_bytes": path.stat().st_size,
        "modified_at": path.stat().st_mtime,
        "group": root.name,
        "preview_kind": preview_kind(suffix),
    }


def discover_artifact_records(
    roots: tuple[Path, ...],
    *,
    phase: str,
    relative_to: Path,
    categorize: Callable[[Path], str],
    category_label: Callable[[str], str] | None = None,
) -> list[dict[str, object]]:
    """Discover supported artifacts under ``roots`` and attach category metadata."""

    records: list[dict[str, object]] = []
    for root in roots:
        if root.is_file():
            if root.suffix.lower() in SUPPORTED_SUFFIXES:
                records.append(
                    artifact_record(
                        root,
                        phase=phase,
                        root=root.parent,
                        relative_to=relative_to,
                        categorize=categorize,
                        category_label=category_label,
                    )
                )
            continue
        if not root.exists():
            continue
        paths = [
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        ]
        records.extend(
            artifact_record(
                path,
                phase=phase,
                root=root,
                relative_to=relative_to,
                categorize=categorize,
                category_label=category_label,
            )
            for path in sorted(paths, key=lambda item: item.stat().st_mtime, reverse=True)
        )
    return dedupe_artifact_records(records)


def group_artifact_records(
    artifacts: list[dict[str, object]],
    category_groups: tuple["CategoryGroup", ...],
) -> list[dict[str, object]]:
    """Bucket flat artifact records into ordered category groups."""

    by_category: dict[str, list[dict[str, object]]] = {
        group.category: [] for group in category_groups
    }
    for artifact in artifacts:
        category = str(artifact["category"])
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(artifact)

    grouped: list[dict[str, object]] = []
    for group in category_groups:
        items = by_category.get(group.category, [])
        if not items:
            continue
        grouped.append(
            {
                "category": group.category,
                "label": group.label,
                "description": group.description,
                "priority": group.priority,
                "artifacts": items,
            }
        )
    return grouped
