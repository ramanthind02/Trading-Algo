"""Save a frozen signed-signal ``bias_node_spec`` from research config to the vault.

Settings live in :func:`feature_research.config.load_config` under ``ResearchConfig.vault_save``
(:class:`~feature_research.config.VaultSaveConfig`): ensemble target, optional ``tickers``
override, ``vault_profile`` (prop vs personal when ``vault_root`` is unset), ``dry_run``, etc.

Always materializes :attr:`~feature_research.config.ResearchConfig.eval_bias_spec` (scalar
hyperparameters: ``evaluation_defaults`` when set, else in-sample defaults). List-valued params
in that spec are rejected — freeze one combo in config before saving.

Examples
--------
Configure ``vault_save`` in ``feature_research/config.py``, then from the repo root::

    python -m feature_research.save_feature_to_vault

Override dry-run for a one-off write without editing config::

    python -m feature_research.save_feature_to_vault --write

Windows (explicit venv interpreter)::

    .\\.venv\\Scripts\\python.exe -m feature_research.save_feature_to_vault

Or pass the script path (repo root is prepended to ``sys.path`` before imports)::

    .\\.venv\\Scripts\\python.exe feature_research\\save_feature_to_vault.py
"""

from __future__ import annotations

from pathlib import Path

from utils.repo_bootstrap import ensure_repo_root_on_syspath

ensure_repo_root_on_syspath(Path(__file__).resolve())

import argparse
import dataclasses
import json
from collections.abc import Mapping, Sequence
from typing import Any

from ensemble.vault.manager import create_ensemble_directory, get_ensemble_path
from ensemble.vault_manager import initialize_vault
from feature_research.config import (
    VaultSaveConfig,
    load_config,
    vault_save_effective_vault_root,
)
from feature_selection.base_models.feature_base_model import BaseModel
from utils.core.enums import DirectionInput, Ticker, TimeFrame, coerce_direction


# Params that are legitimately list-valued at a frozen combo (not exploration grids).
_STRUCTURAL_LIST_PARAM_KEYS: frozenset[str] = frozenset({"cross_tickers"})


def _params_contain_grid(params: object) -> bool:
    if not isinstance(params, dict):
        return False
    return any(
        isinstance(v, list) and k not in _STRUCTURAL_LIST_PARAM_KEYS
        for k, v in params.items()
    )


def _normalize_bias_spec_for_model(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy suitable for ``BaseModel`` (eval / frozen combo only)."""
    spec: dict[str, Any] = dict(raw)
    raw_tfs = spec.get("timeframes", [])
    if not isinstance(raw_tfs, list) or not raw_tfs:
        raise ValueError("bias_spec.timeframes must be a non-empty list.")
    spec["timeframes"] = [
        tf.name if isinstance(tf, TimeFrame) else str(tf) for tf in raw_tfs
    ]
    params = spec.get("params", {})
    if not isinstance(params, dict):
        raise ValueError("bias_spec.params must be a dict when present.")
    if _params_contain_grid(params):
        raise ValueError(
            "eval_bias_spec.params contains list-valued grid parameters. "
            "Freeze one combo in evaluation_defaults (or in-sample defaults used as eval fallback)."
        )
    spec["params"] = dict(params)
    return spec


def _resolve_ensemble_dir(
    *,
    vault_save: VaultSaveConfig,
    timeframe: TimeFrame,
    direction: DirectionInput,
    tickers: list[Ticker],
    dry_run: bool,
) -> str:
    if vault_save.existing_ensemble_dir is not None:
        resolved = str(Path(vault_save.existing_ensemble_dir))
        if not dry_run and not Path(resolved).is_dir():
            raise FileNotFoundError(f"ensemble_dir does not exist: {resolved}")
        return resolved
    name = (vault_save.ensemble_name or "").strip()
    if not name:
        raise ValueError("vault_save.ensemble_name is required when existing_ensemble_dir is unset.")
    root = vault_save_effective_vault_root(vault_save)
    group = vault_save.weight_hierarchy_group
    if dry_run:
        return get_ensemble_path(
            timeframe,
            name,
            direction,
            vault_root=str(root),
            weight_hierarchy_group=group,
        )
    return create_ensemble_directory(
        timeframe,
        name,
        direction,
        tickers=tickers,
        vault_root=str(root),
        weight_hierarchy_group=group,
    )


def _build_override_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Persist a signed-signal feature using ResearchConfig.vault_save from "
            "feature_research.config.load_config()."
        ),
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--write",
        action="store_true",
        help="Write to the vault (overrides vault_save.dry_run=False).",
    )
    group.add_argument(
        "--dry-run",
        action="store_true",
        help="Print only (overrides vault_save.dry_run=True).",
    )
    parser.add_argument(
        "--vault-profile",
        choices=("prop", "personal"),
        default=None,
        help=(
            "When vault_save.vault_root is unset, use this profile's default root "
            "(prop=firm vault, personal=personal vault). Overrides vault_save.vault_profile."
        ),
    )
    return parser


def _effective_dry_run(*, vault_save: VaultSaveConfig, args: argparse.Namespace) -> bool:
    if args.write:
        return False
    if args.dry_run:
        return True
    return vault_save.dry_run


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_override_parser().parse_args(list(argv) if argv is not None else None)

    research = load_config()
    vault_save = research.vault_save
    if args.vault_profile is not None and vault_save is not None:
        vault_save = dataclasses.replace(vault_save, vault_profile=args.vault_profile)
    if vault_save is None:
        print(
            "Set vault_save=VaultSaveConfig(...) on ResearchConfig in "
            "feature_research.config.load_config().",
            file=sys.stderr,
        )
        return 2

    try:
        bias_spec = _normalize_bias_spec_for_model(dict(research.eval_bias_spec))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    tickers = (
        list(vault_save.tickers)
        if vault_save.tickers is not None
        else list(research.tickers)
    )
    direction = coerce_direction(vault_save.direction, field_name="direction")

    first_tf_name = bias_spec["timeframes"][0]
    timeframe = (
        TimeFrame[first_tf_name]
        if isinstance(first_tf_name, str)
        else first_tf_name
    )

    dry_run = _effective_dry_run(vault_save=vault_save, args=args)

    if vault_save.init_vault and not dry_run:
        initialize_vault(str(vault_save_effective_vault_root(vault_save)))

    ensemble_dir = _resolve_ensemble_dir(
        vault_save=vault_save,
        timeframe=timeframe,
        direction=direction,
        tickers=tickers,
        dry_run=dry_run,
    )

    feature_config: dict[str, Any] = {
        "bias_node_spec": bias_spec,
        "strategy": direction,
    }

    print("Vault root:", str(vault_save_effective_vault_root(vault_save)))
    print("Ensemble dir:", ensemble_dir)
    print("Bias spec:")
    print(json.dumps(bias_spec, indent=2, default=str))

    if dry_run:
        print("Dry-run: no file written.")
        return 0

    model = BaseModel(feature_config, tickers=tickers)
    model_id = model.save_to_vault(ensemble_dir, tickers=tickers)
    print("Saved model_id:", model_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
