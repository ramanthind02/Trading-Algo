"""Entry-point bootstrap shim.

The real logic now lives in :mod:`lib.core.runtime_bootstrap` so that library
code (e.g. ``data_platform`` providers) can prepare the runtime without importing
this entry-point module. Scripts keep doing ``import scripts._bootstrap`` for its
import-time side effects (UTF-8 console + ``.env`` load).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repo root is importable BEFORE importing ``lib.core`` — handles
# ``python scripts/foo.py`` (where sys.path[0] is the scripts/ dir) in any
# environment without the editable install.
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from lib.core.repo_bootstrap import ensure_repo_root_on_syspath  # noqa: E402
from lib.core.runtime_bootstrap import bootstrap_runtime  # noqa: E402

# Preserve historical import-time side effects (UTF-8 streams + .env load).
bootstrap_runtime(_repo_root)


def ensure_project_root_on_path() -> Path:
    """Add the repository root to sys.path and return it (backward-compatible)."""
    return ensure_repo_root_on_syspath(Path(__file__).resolve())
