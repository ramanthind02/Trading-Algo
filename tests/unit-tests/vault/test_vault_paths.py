from __future__ import annotations

from cache.runtime.cache_paths import project_root
from lib.core.vault_paths import (
    default_vault_discovery_dirnames,
    default_vault_root,
    resolve_vault_personal,
    resolve_vault_prop,
    resolve_vault_root,
    resolve_vault_root_for_profile,
    vault_discovery_dirnames_for_profile,
    vault_root_from_repo_relative_ensemble,
)


def test_resolve_vault_root_anchors_relative_paths_to_repo_root() -> None:
    resolved = resolve_vault_root("custom_vault")
    assert resolved == project_root() / "custom_vault"


def test_default_vault_root_uses_repo_relative_env_override(
    monkeypatch,
) -> None:
    monkeypatch.setenv("TRADING_ALGO_VAULT_ROOT", "env_vault")
    monkeypatch.delenv("TRADING_ALGO_VAULT_PROP", raising=False)
    assert default_vault_root() == project_root() / "env_vault"
    assert resolve_vault_prop() == project_root() / "env_vault"


def test_resolve_vault_prop_prefers_vault_prop_env_over_legacy_root(monkeypatch) -> None:
    monkeypatch.setenv("TRADING_ALGO_VAULT_PROP", "prop_named")
    monkeypatch.setenv("TRADING_ALGO_VAULT_ROOT", "legacy_should_lose")
    assert resolve_vault_prop() == project_root() / "prop_named"


def test_resolve_vault_personal_default_and_env(monkeypatch) -> None:
    monkeypatch.delenv("TRADING_ALGO_VAULT_PERSONAL", raising=False)
    assert resolve_vault_personal() == project_root() / "vault_personal"
    monkeypatch.setenv("TRADING_ALGO_VAULT_PERSONAL", "my_personal_vault")
    assert resolve_vault_personal() == project_root() / "my_personal_vault"


def test_resolve_vault_root_for_profile_dispatches() -> None:
    assert resolve_vault_root_for_profile("prop") == resolve_vault_prop()
    assert resolve_vault_root_for_profile("personal") == resolve_vault_personal()


def test_vault_root_from_repo_relative_ensemble_first_component(tmp_path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    assert vault_root_from_repo_relative_ensemble(repo, "vault_personal/D/a") == repo / "vault_personal"


def test_default_vault_discovery_dirnames_includes_standard_names(monkeypatch) -> None:
    monkeypatch.delenv("TRADING_ALGO_VAULT_PROP", raising=False)
    monkeypatch.delenv("TRADING_ALGO_VAULT_PERSONAL", raising=False)
    monkeypatch.delenv("TRADING_ALGO_VAULT_ROOT", raising=False)
    names = default_vault_discovery_dirnames()
    assert "vault" in names
    assert "vault_personal" in names


def test_vault_discovery_dirnames_for_profile_returns_single_root(monkeypatch) -> None:
    monkeypatch.delenv("TRADING_ALGO_VAULT_PROP", raising=False)
    monkeypatch.delenv("TRADING_ALGO_VAULT_PERSONAL", raising=False)
    monkeypatch.delenv("TRADING_ALGO_VAULT_ROOT", raising=False)
    assert vault_discovery_dirnames_for_profile("prop") == ("vault",)
    assert vault_discovery_dirnames_for_profile("personal") == ("vault_personal",)


def test_resolve_vault_root_preserves_absolute_paths() -> None:
    absolute_path = project_root() / ".tmp_vault_absolute"
    assert resolve_vault_root(absolute_path) == absolute_path
