"""List + preview run artifacts (CSV/JSON/Markdown/PNG) under the allowed results roots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from frontend.api.paths import repo_relative, resolve_artifact_path

# Suffix -> artifact kind the UI renders differently (table / chart-data / text / image).
_KIND_BY_SUFFIX: dict[str, str] = {
    ".csv": "csv",
    ".json": "json",
    ".md": "markdown",
    ".txt": "text",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".html": "html",
}

_PREVIEW_ROW_CAP = 500


def _kind(path: Path) -> str:
    return _KIND_BY_SUFFIX.get(path.suffix.lower(), "binary")


def _entry(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "name": path.name,
        "path": repo_relative(path),
        "kind": _kind(path),
        "size": stat.st_size,
        "mtime": stat.st_mtime,
    }


def list_dir(root: Path) -> list[dict[str, Any]]:
    """Flat list of files under ``root`` (recursive), sorted by path. Empty if missing."""

    if not root.exists():
        return []
    files = (p for p in root.rglob("*") if p.is_file())
    return sorted((_entry(p) for p in files), key=lambda e: e["path"])


def list_for_run(reports_dir: str | None, viz_dir: str | None) -> dict[str, Any]:
    """Artifacts for a run: per-spec reports + the shared in-sample visualization folder."""

    groups: list[dict[str, Any]] = []
    if reports_dir:
        groups.append(
            {
                "label": "Run reports (per-spec)",
                "root": reports_dir,
                "files": list_dir(Path(reports_dir)),
            }
        )
    if viz_dir:
        groups.append(
            {
                "label": "Exploration visualization (latest run)",
                "root": viz_dir,
                "files": list_dir(Path(viz_dir)),
            }
        )
    return {"groups": groups}


def preview(relative_path: str) -> dict[str, Any]:
    """Return a renderable preview of one artifact (table rows / parsed JSON / text / image ref)."""

    path = resolve_artifact_path(relative_path)
    kind = _kind(path)
    if kind == "csv":
        return _preview_csv(path)
    if kind == "json":
        return {"kind": "json", "path": relative_path, "data": json.loads(path.read_text(encoding="utf-8"))}
    if kind in ("markdown", "text"):
        return {"kind": kind, "path": relative_path, "text": path.read_text(encoding="utf-8")}
    if kind in ("image", "html"):
        return {"kind": kind, "path": relative_path, "raw_url": f"/api/artifacts/raw?path={relative_path}"}
    return {"kind": "binary", "path": relative_path}


def _preview_csv(path: Path) -> dict[str, Any]:
    import pandas as pd

    frame = pd.read_csv(path)
    total_rows = int(frame.shape[0])
    head = frame.head(_PREVIEW_ROW_CAP)
    # NaN -> None so the JSON is valid (NaN is not valid JSON).
    records = head.where(head.notna(), None).to_dict(orient="records")
    return {
        "kind": "csv",
        "path": repo_relative(path),
        "columns": [str(col) for col in frame.columns],
        "rows": records,
        "total_rows": total_rows,
        "truncated": total_rows > _PREVIEW_ROW_CAP,
    }
