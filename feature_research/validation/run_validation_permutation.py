"""Validation permutation test (vector-shuffle or candle-shuffle, single fold from config.validation_window).

This is a thin wrapper script that delegates to the shared permutation logic in
feature_research.validation.permutation_helpers.run_permutation_for_phase.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_repo_hint = Path(__file__).resolve().parents[2]
if str(_repo_hint) not in sys.path:
    sys.path.insert(0, str(_repo_hint))

from feature_research.bootstrap import ensure_repo_root_on_syspath

ensure_repo_root_on_syspath(Path(__file__).resolve())

from feature_research.in_sample.config import load_config
from feature_research.validation.permutation_helpers import run_permutation_for_phase


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validation permutation test (vector-shuffle or candle-shuffle, single fold from config.validation_window)."
    )
    parser.add_argument("--nreps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = load_config()
    if config.validation_window is None:
        print("Error: config.validation_window is not set. Set it in feature_research.config.load_config().")
        return 1

    return run_permutation_for_phase(phase="validation", config=config, args=args)


if __name__ == "__main__":
    sys.exit(main())
