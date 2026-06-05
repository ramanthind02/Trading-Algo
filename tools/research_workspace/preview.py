"""Artifact preview loaders for research workspace UIs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def resolve_artifact_path(repo_root: Path, relative_path: str) -> Path:
    """Resolve a repo-relative (or absolute) artifact path."""

    candidate = Path(relative_path)
    if not candidate.is_absolute():
        candidate = (repo_root / relative_path).resolve()
    else:
        candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(relative_path)
    if not candidate.is_absolute():
        try:
            candidate.relative_to(repo_root.resolve())
        except ValueError as exc:
            raise ValueError("Artifact path is outside the repository.") from exc
    return candidate


def relative_to_repo(repo_root: Path, path: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def csv_preview(path: Path) -> dict[str, object]:
    frame = pd.read_csv(path)
    preview = frame.head(100)
    payload: dict[str, Any] = {
        "kind": "table",
        "columns": list(preview.columns),
        "rows": preview.fillna("").astype(str).to_dict("records"),
        "row_count": int(frame.shape[0]),
    }
    if {"datetime", "cumulative_strategy_return"}.issubset(frame.columns):
        chart_frame = frame[["datetime", "cumulative_strategy_return"]].dropna().tail(500)
        payload["chart"] = {
            "x": chart_frame["datetime"].astype(str).tolist(),
            "y": chart_frame["cumulative_strategy_return"].astype(float).tolist(),
            "label": "cumulative_strategy_return",
        }
    return payload


def json_preview(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {"kind": "json", "data": payload}


def load_artifact_preview(repo_root: Path, relative_path: str) -> dict[str, object]:
    """Return a lightweight preview payload for one workspace artifact."""

    path = resolve_artifact_path(repo_root, relative_path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return csv_preview(path)
    if suffix == ".json":
        return json_preview(path)
    if suffix in {".md", ".txt"}:
        return {"kind": "text", "text": path.read_text(encoding="utf-8")}
    if suffix in {".html", ".png"}:
        return {"kind": "embed", "path": relative_to_repo(repo_root, path)}
    return {"kind": "unsupported", "message": f"Preview not supported for {suffix} files."}
