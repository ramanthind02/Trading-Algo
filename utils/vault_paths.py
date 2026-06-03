from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from utils.cache.runtime.cache_paths import project_root

_VAULT_ROOT_ENV_VAR = "TRADING_ALGO_VAULT_ROOT"
_VAULT_PROP_ENV_VAR = "TRADING_ALGO_VAULT_PROP"
_VAULT_PERSONAL_ENV_VAR = "TRADING_ALGO_VAULT_PERSONAL"
_VAULT_CFD_PROP_ENV_VAR = "TRADING_ALGO_VAULT_CFD_PROP"
_DEFAULT_VAULT_DIRNAME = "vault"
_DEFAULT_VAULT_PERSONAL_DIRNAME = "vault_personal"
_DEFAULT_VAULT_CFD_PROP_DIRNAME = "vault_cfd_prop"

# "prop" is the legacy alias for the futures prop-firm profile; "futures_prop"
# is the canonical name introduced alongside the CFD prop profile.
VaultProfile = Literal["prop", "futures_prop", "personal", "cfd_prop"]


def _env_path(env_var: str) -> Path | None:
    raw = os.getenv(env_var, "").strip()
    if not raw:
        return None
    return _normalize_vault_root(Path(raw))


def resolve_vault_prop() -> Path:
    """Prop-firm vault root: ``TRADING_ALGO_VAULT_PROP``, else legacy ``TRADING_ALGO_VAULT_ROOT``, else ``<repo>/vault``."""
    explicit = _env_path(_VAULT_PROP_ENV_VAR)
    if explicit is not None:
        return explicit
    legacy = _env_path(_VAULT_ROOT_ENV_VAR)
    if legacy is not None:
        return legacy
    return project_root() / _DEFAULT_VAULT_DIRNAME


def resolve_vault_personal() -> Path:
    """Personal vault root: ``TRADING_ALGO_VAULT_PERSONAL`` else ``<repo>/vault_personal``."""
    explicit = _env_path(_VAULT_PERSONAL_ENV_VAR)
    if explicit is not None:
        return explicit
    return project_root() / _DEFAULT_VAULT_PERSONAL_DIRNAME


def resolve_vault_cfd_prop() -> Path:
    """CFD prop-firm vault root: ``TRADING_ALGO_VAULT_CFD_PROP`` else ``<repo>/vault_cfd_prop``."""
    explicit = _env_path(_VAULT_CFD_PROP_ENV_VAR)
    if explicit is not None:
        return explicit
    return project_root() / _DEFAULT_VAULT_CFD_PROP_DIRNAME


def resolve_vault_root_for_profile(profile: VaultProfile) -> Path:
    if profile in ("prop", "futures_prop"):
        return resolve_vault_prop()
    if profile == "cfd_prop":
        return resolve_vault_cfd_prop()
    return resolve_vault_personal()


def vault_root_from_repo_relative_ensemble(repo_root: str | Path, ensemble_path_rel: str) -> Path:
    """First path component of a repo-relative ensemble path is the vault top-level directory."""
    parts = Path(ensemble_path_rel).parts
    if not parts:
        raise ValueError("ensemble_path_rel must be non-empty")
    return Path(repo_root).resolve() / parts[0]


def default_vault_root() -> Path:
    """Default / primary vault root (prop firm). Same as :func:`resolve_vault_prop`."""
    return resolve_vault_prop()


def resolve_vault_root(vault_root: str | Path | None = None) -> Path:
    if vault_root is not None:
        return _normalize_vault_root(Path(vault_root))
    return default_vault_root()


def _normalize_vault_root(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded
    return project_root() / expanded


def default_vault_discovery_dirnames() -> tuple[str, ...]:
    """Repo-relative top-level directory names to scan for ensembles (prop + personal defaults).

    Uses the first path component of :func:`resolve_vault_prop` / :func:`resolve_vault_personal`
    when those roots lie under the repo; always appends ``vault`` and ``vault_personal`` if
    missing so discovery still sees standard layouts when env points outside the repo.

    Prefer :func:`vault_discovery_dirnames_for_profile` when a caller must stay inside one vault.
    """
    repo = project_root().resolve()
    out: list[str] = []
    for p in (resolve_vault_prop(), resolve_vault_personal(), resolve_vault_cfd_prop()):
        try:
            rel = p.resolve().relative_to(repo)
        except ValueError:
            continue
        if rel.parts:
            name = rel.parts[0]
            if name not in out:
                out.append(name)
    for fallback in (
        _DEFAULT_VAULT_DIRNAME,
        _DEFAULT_VAULT_PERSONAL_DIRNAME,
        _DEFAULT_VAULT_CFD_PROP_DIRNAME,
    ):
        if fallback not in out:
            out.append(fallback)
    return tuple(out)


def _repo_relative_vault_top_name(vault_root: Path) -> str | None:
    repo = project_root().resolve()
    try:
        rel = vault_root.resolve().relative_to(repo)
    except ValueError:
        return None
    return rel.parts[0] if rel.parts else None


def vault_discovery_dirnames_for_profile(profile: VaultProfile) -> tuple[str, ...]:
    """Repo-relative top-level directory for one vault profile (prop or personal)."""
    top = _repo_relative_vault_top_name(resolve_vault_root_for_profile(profile))
    if top is not None:
        return (top,)
    return (_DEFAULT_VAULT_DIRNAME if profile == "prop" else _DEFAULT_VAULT_PERSONAL_DIRNAME,)


def vault_discovery_dirnames_for_root(vault_root: str | Path) -> tuple[str, ...]:
    """Repo-relative top-level directory for an explicit vault root path."""
    top = _repo_relative_vault_top_name(_normalize_vault_root(Path(vault_root)))
    if top is not None:
        return (top,)
    return (_normalize_vault_root(Path(vault_root)).name,)


def resolve_portfolio_vault_discovery_dirnames(
    *,
    vault_profile: VaultProfile | None = "prop",
    vault_root: str | Path | None = None,
) -> tuple[str, ...]:
    """Resolve which repo-relative vault top-level folder(s) to scan for ensembles."""
    if vault_root is not None:
        return vault_discovery_dirnames_for_root(vault_root)
    return vault_discovery_dirnames_for_profile(vault_profile or "prop")
