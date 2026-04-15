from __future__ import annotations

import os
from pathlib import Path

from utils.cache.runtime.cache_paths import project_root

_VAULT_ROOT_ENV_VAR = "TRADING_ALGO_VAULT_ROOT"
_DEFAULT_VAULT_DIRNAME = "vault"


def default_vault_root() -> Path:
    configured = os.getenv(_VAULT_ROOT_ENV_VAR, "").strip()
    if configured:
        return _normalize_vault_root(Path(configured))
    return project_root() / _DEFAULT_VAULT_DIRNAME


def resolve_vault_root(vault_root: str | Path | None = None) -> Path:
    if vault_root is not None:
        return _normalize_vault_root(Path(vault_root))
    return default_vault_root()


def _normalize_vault_root(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded
    return project_root() / expanded
