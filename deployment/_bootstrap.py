from __future__ import annotations

from pathlib import Path

from lib.core.repo_bootstrap import ensure_repo_root_on_syspath


def ensure_project_root_on_path() -> Path:
    return ensure_repo_root_on_syspath(Path(__file__).resolve())
