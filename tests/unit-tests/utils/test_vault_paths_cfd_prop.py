"""Tests for the new ``cfd_prop`` vault profile resolution in
:mod:`utils.vault_paths`.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from lib.core import vault_paths


@pytest.fixture(autouse=True)
def _clear_vault_env(monkeypatch):
    """Strip any vault env vars from the test process so defaults are exercised."""
    for var in (
        "TRADING_ALGO_VAULT_ROOT",
        "TRADING_ALGO_VAULT_PROP",
        "TRADING_ALGO_VAULT_PERSONAL",
        "TRADING_ALGO_VAULT_CFD_PROP",
    ):
        monkeypatch.delenv(var, raising=False)


def test_resolve_vault_cfd_prop_default_is_under_repo_root() -> None:
    p = vault_paths.resolve_vault_cfd_prop()
    assert p.name == "vault_cfd_prop"


def test_resolve_vault_cfd_prop_env_override(monkeypatch, tmp_path) -> None:
    target = tmp_path / "custom_cfd_vault"
    target.mkdir()
    monkeypatch.setenv("TRADING_ALGO_VAULT_CFD_PROP", str(target))
    assert vault_paths.resolve_vault_cfd_prop() == target


def test_resolve_vault_root_for_profile_cfd_prop() -> None:
    p = vault_paths.resolve_vault_root_for_profile("cfd_prop")
    assert p.name == "vault_cfd_prop"


def test_resolve_vault_root_for_profile_futures_prop_matches_prop() -> None:
    # "futures_prop" is the new canonical name; "prop" is the legacy alias.
    p_legacy = vault_paths.resolve_vault_root_for_profile("prop")
    p_canonical = vault_paths.resolve_vault_root_for_profile("futures_prop")
    assert p_legacy == p_canonical


def test_default_vault_discovery_dirnames_includes_cfd_prop() -> None:
    names = vault_paths.default_vault_discovery_dirnames()
    assert "vault_cfd_prop" in names
    assert "vault" in names
    assert "vault_personal" in names


def test_vault_profile_literal_includes_new_values() -> None:
    """The VaultProfile Literal should now include cfd_prop and futures_prop."""
    # The Literal can't be inspected as a set directly, but we can check
    # that calling resolve_vault_root_for_profile with each accepted value
    # does not raise.
    for profile in ("prop", "futures_prop", "personal", "cfd_prop"):
        vault_paths.resolve_vault_root_for_profile(profile)  # type: ignore[arg-type]
