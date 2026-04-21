"""One-off migration: move vault ensembles under weight-hierarchy group folders.

Run from repo root::

    .\\.venv\\Scripts\\python.exe scripts\\migrate_vault_nested_groups.py

Prop vault by default (:func:`utils.vault_paths.resolve_vault_prop`). Override::

    python scripts\\migrate_vault_nested_groups.py --vault-root vault_personal
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Repo root = parent of scripts/
REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils.vault_paths import resolve_vault_prop  # noqa: E402

# (timeframe, ensemble_leaf_name, group_dir_name)
MOVES: list[tuple[str, str, str]] = [
    ("D", "mr_indices_long", "mean_reversion_indices"),
    ("D", "mr_indices_long_long", "mean_reversion_indices"),
    ("D", "rsi_signal_long", "mean_reversion_indices"),
    ("D", "williamsr_signal_long", "mean_reversion_indices"),
    ("D", "casey_percent_c_signal_long", "mean_reversion_indices"),
    ("D", "zscore_rsi_signal_r14_z100_os-2_ob2_exitThrBars5_long", "mean_reversion_indices"),
    ("D", "seasonal_indices_long", "seasonal"),
    ("D", "seasonal_bonds_long_short", "seasonal"),
    ("D", "rebalancing_es_tlt_long", "es_tlt"),
    ("D", "rebalancing_tlt_es_long", "es_tlt"),
    ("D", "algomatic_momentum_signal_long_defaults_long", "momentum"),
    ("D", "filter_sma200_above_consec_momentum_lb40_cb3_buy_long", "momentum"),
    ("D", "gc_sma_above_filter_d_period_200_long", "momentum_gc"),
    ("D", "sma_regime_long_short_long_short", "momentum"),
    ("D", "filter_gate_sma200_rsi_signal_long_long", "momentum"),
    ("M", "buy_hold_long", "buy_hold"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vault-root",
        type=Path,
        default=None,
        help="Vault directory (default: resolve_vault_prop()).",
    )
    args = parser.parse_args()
    vault = (args.vault_root or resolve_vault_prop()).resolve()

    for tf, name, group in MOVES:
        src = vault / tf / name
        if not src.is_dir():
            try:
                rel = src.relative_to(REPO)
            except ValueError:
                rel = src
            print(f"SKIP (missing): {rel}", file=sys.stderr)
            continue
        dst = vault / tf / group / name
        if dst.exists():
            try:
                rel = dst.relative_to(REPO)
            except ValueError:
                rel = dst
            print(f"SKIP (exists): {rel}", file=sys.stderr)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        try:
            srel, drel = src.relative_to(REPO), dst.relative_to(REPO)
        except ValueError:
            srel, drel = src, dst
        print(f"OK: {srel} -> {drel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
