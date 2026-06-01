from __future__ import annotations

import os
import sys
from pathlib import Path

# Insert repo root before any ``utils`` import so ``python scripts/foo.py`` works
# when the interpreter's initial sys.path entry is the ``scripts/`` directory.
_repo_root = Path(__file__).resolve().parent.parent
_root_str = str(_repo_root)
if _root_str not in sys.path:
    sys.path.insert(0, _root_str)

# Best-effort .env load. We prefer python-dotenv (handles quoting, multi-line,
# variable expansion) but fall back to a tiny manual parser if it isn't
# installed so existing shell-env workflows still work unchanged.
def _load_env_file() -> None:
    env_path = _repo_root / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv  # type: ignore[import-not-found]
    except ImportError:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            # Never clobber values already set in the real environment.
            if key and key not in os.environ:
                os.environ[key] = val
        return
    # python-dotenv path: load without overriding pre-set env vars.
    load_dotenv(dotenv_path=env_path, override=False)


_load_env_file()

from utils.repo_bootstrap import ensure_repo_root_on_syspath


def ensure_project_root_on_path() -> Path:
    """Add the repository root to sys.path and return it."""
    return ensure_repo_root_on_syspath(Path(__file__).resolve())
