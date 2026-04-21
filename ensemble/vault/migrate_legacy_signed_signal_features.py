"""Migrate vault ``**/features/*.json`` files from binning-era base model keys to signed-signal schema.

Run from repo root::

    python -m ensemble.vault.migrate_legacy_signed_signal_features

Dry run::

    python -m ensemble.vault.migrate_legacy_signed_signal_features --dry-run

Custom vault root::

    python -m ensemble.vault.migrate_legacy_signed_signal_features --vault-root path/to/vault
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _prepend_repo_root_to_syspath() -> Path:
    start = Path(__file__).resolve()
    for parent in (start.parent, *start.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            root_str = str(parent)
            if root_str not in sys.path:
                sys.path.insert(0, root_str)
            return parent
    raise RuntimeError(
        "Could not locate repository root (no pyproject.toml or .git above this file)."
    )


_REPO_ROOT = _prepend_repo_root_to_syspath()

from ensemble.vault.feature_files import migrate_legacy_signed_signal_feature_files_under_vault  # noqa: E402
from utils.repo_bootstrap import require_repo_root  # noqa: E402
from utils.vault_paths import resolve_vault_prop  # noqa: E402


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vault-root",
        type=Path,
        default=None,
        help="Directory containing ensemble vault layout (default: resolve_vault_prop()).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate migrations only; do not write files.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _ = require_repo_root(Path(__file__).resolve())
    args = _parse_args(argv)
    vault_root = (args.vault_root or resolve_vault_prop()).resolve()
    if not vault_root.is_dir():
        print(f"Vault root is not a directory: {vault_root}", file=sys.stderr)
        return 1
    updated = migrate_legacy_signed_signal_feature_files_under_vault(
        vault_root,
        dry_run=bool(args.dry_run),
    )
    action = "Would update" if args.dry_run else "Updated"
    print(f"{action} {len(updated)} feature file(s) under {vault_root}")
    for path in updated:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
