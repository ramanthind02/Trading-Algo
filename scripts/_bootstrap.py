from __future__ import annotations

import sys
from pathlib import Path

# Insert repo root before any ``utils`` import so ``python scripts/foo.py`` works
# when the interpreter's initial sys.path entry is the ``scripts/`` directory.
_repo_root = Path(__file__).resolve().parent.parent
_root_str = str(_repo_root)
if _root_str not in sys.path:
    sys.path.insert(0, _root_str)

from utils.repo_bootstrap import ensure_repo_root_on_syspath


def ensure_project_root_on_path() -> Path:
    """Add the repository root to sys.path and return it."""
    return ensure_repo_root_on_syspath(Path(__file__).resolve())
