from __future__ import annotations

from pathlib import Path


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


def default_source_candle_dir() -> Path:
    return project_root() / "data" / "ohlc_data"
