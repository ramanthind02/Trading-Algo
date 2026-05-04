#!/usr/bin/env python3
"""Emit a draft ``hierarchy_equal`` JSON from vault feature tags and base model names.

Run from the repository root::

    .\\.venv\\Scripts\\python.exe scripts\\emit_weight_hierarchy_json.py -o hierarchy_draft.json --stats

Uses each feature's ``weight_hierarchy_group`` (or infers ``.../<group>/...`` from a nested path),
``tickers``, ``bias_node_spec.timeframes``, and ``base_models[0].model_name`` to build global
``stream_id`` strings ``ticker::timeframe::model_name``.

Pass ``--vault-root`` for a non-default tree (e.g. ``vault_personal``); otherwise the root is
``resolve_vault_root(None)`` (prop: ``TRADING_ALGO_VAULT_PROP`` / legacy ``TRADING_ALGO_VAULT_ROOT`` / ``<repo>/vault``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.vault_paths import resolve_vault_root

from ensemble.vault.constants import VAULT_WEIGHT_HIERARCHY_GROUP_DIR_NAMES
from ensemble.vault.hierarchy_spec import (
    build_hierarchy_spec_from_vault,
)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a hierarchy_equal JSON spec: root → five manual buckets → leaves as "
            "global stream_id strings (ticker::timeframe::model_name)."
        )
    )
    parser.add_argument(
        "--vault-root",
        type=Path,
        default=None,
        help=(
            "Vault root path or repo-relative dirname (default: resolve_vault_root(None) — "
            "TRADING_ALGO_VAULT_PROP, else TRADING_ALGO_VAULT_ROOT, else <repo>/vault). "
            "Use vault_personal or an absolute path for the personal tree."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write JSON to this path (UTF-8). Default: print to stdout",
    )
    parser.add_argument(
        "--strict-group",
        action="store_true",
        help="Fail if a feature has no weight_hierarchy_group and is not under a nested group folder",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indent (default: 2; use 0 for compact)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print per-bucket stream counts to stderr",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    vault = resolve_vault_root(args.vault_root)
    spec, by_group = build_hierarchy_spec_from_vault(vault, strict_group=args.strict_group)
    if args.stats:
        for name in sorted(VAULT_WEIGHT_HIERARCHY_GROUP_DIR_NAMES):
            n = len(by_group.get(name, ()))
            print(f"{name}: {n} streams", file=sys.stderr)
    text = json.dumps(spec, indent=args.indent if args.indent > 0 else None, sort_keys=False)
    text = text if args.indent > 0 else text + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text if text.endswith("\n") else text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
