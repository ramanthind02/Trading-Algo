from __future__ import annotations

from pathlib import Path

import pytest

from utils.cache.runtime.central_cache_errors import ArtifactMissingError


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def skip_if_no_data() -> None:
    candle_dir = project_root() / "data" / "ohlc_data"
    if not candle_dir.exists():
        pytest.skip(f"Missing persisted candle directory: {candle_dir}")


def skip_if_missing_data_prereq(exc: Exception) -> None:
    message = str(exc)
    if isinstance(exc, FileNotFoundError):
        pytest.skip(f"Missing persisted data prerequisite: {message}")
    if isinstance(exc, ArtifactMissingError) or "Artifact is missing" in message:
        pytest.skip(f"Missing cached artifact prerequisite: {message}")
    if isinstance(exc, ValueError) and (
        "Unable to load feature/target data" in message
        or "Feature extraction returned no data" in message
        or "Artifact is not loaded" in message
    ):
        pytest.skip(f"Missing data prerequisite for permutation suite: {message}")
