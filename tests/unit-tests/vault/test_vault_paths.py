from __future__ import annotations

from utils.cache.runtime.cache_paths import project_root
from utils.vault_paths import default_vault_root, resolve_vault_root


def test_resolve_vault_root_anchors_relative_paths_to_repo_root() -> None:
    resolved = resolve_vault_root("custom_vault")
    assert resolved == project_root() / "custom_vault"


def test_default_vault_root_uses_repo_relative_env_override(
    monkeypatch,
) -> None:
    monkeypatch.setenv("TRADING_ALGO_VAULT_ROOT", "env_vault")
    assert default_vault_root() == project_root() / "env_vault"


def test_resolve_vault_root_preserves_absolute_paths() -> None:
    absolute_path = project_root() / ".tmp_vault_absolute"
    assert resolve_vault_root(absolute_path) == absolute_path
