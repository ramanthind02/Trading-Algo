from __future__ import annotations

import os
from pathlib import Path
from typing import Literal


def win32_extended_path(path: Path) -> str:
    """Return an absolute path string usable with Win32 APIs beyond ``MAX_PATH`` (``\\\\?\\`` prefix)."""
    if os.name != "nt":
        return str(path)
    resolved = path.resolve()
    s = str(resolved)
    if s.startswith("\\\\?\\"):
        return s
    if s.startswith("\\\\"):
        return "\\\\?\\UNC\\" + s[2:]
    return "\\\\?\\" + s


def project_root() -> Path:
    resolved = Path(__file__).resolve()
    return next(
        (
            parent
            for parent in resolved.parents
            if (parent / ".git").exists() or (parent / "AGENTS.md").exists()
        ),
        resolved.parents[2],
    )


def resolve_relative_path(
    path_value: str | Path,
    *,
    prefer_existing_candidate: bool = False,
    include_project_root: bool = True,
    project_root_fallback: Literal["candidate", "project_candidate"] = "project_candidate",
) -> Path:
    """Resolve a path against cwd and optionally the repo root without changing fallbacks."""
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate

    if prefer_existing_candidate and candidate.exists():
        return candidate

    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists():
        return cwd_candidate

    if not include_project_root:
        return candidate

    project_candidate = project_root() / candidate
    if project_candidate.exists():
        return project_candidate

    if project_root_fallback == "candidate":
        return candidate
    return project_candidate


def default_runtime_root() -> Path:
    return project_root() / ".cache" / "trading_algo"


def default_central_cache_dir() -> Path:
    return default_runtime_root() / "central_cache"


def default_candle_cache_dir() -> Path:
    return default_central_cache_dir() / "candles"


def default_live_artifact_cache_dir() -> Path:
    return default_central_cache_dir() / "artifacts" / "live"


def default_research_artifact_cache_dir() -> Path:
    return default_central_cache_dir() / "artifacts" / "research"


def default_materialized_cache_dir() -> Path:
    return default_central_cache_dir() / "materialized"


def default_live_materialized_cache_dir() -> Path:
    return default_materialized_cache_dir() / "live"


def default_research_materialized_cache_dir() -> Path:
    return default_materialized_cache_dir() / "research"


def default_source_candle_dir() -> Path:
    return project_root() / "data" / "ohlc_data"
