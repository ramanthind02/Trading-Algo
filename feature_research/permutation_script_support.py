"""Shared CLI bootstrap for OOS and validation permutation scripts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

from feature_research.bootstrap import ensure_repo_root_on_syspath

Phase = Literal["oos", "validation"]


def parse_permutation_phase_args(*, phase: Phase) -> argparse.Namespace:
    if phase == "oos":
        parser = argparse.ArgumentParser(
            description="OOS permutation test (vector-shuffle null, single fold from config.oos_window)."
        )
        parser.add_argument(
            "--nreps",
            type=int,
            default=None,
            help="Number of replicates (default: config.permutation.nreps).",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=None,
            help="Random seed (default from config.permutation.random_seed).",
        )
        parser.add_argument(
            "--n-jobs",
            type=int,
            default=8,
            help="Parallel jobs for vector-shuffle replicates (default from config or 1). -1 = all CPUs.",
        )
        parser.add_argument(
            "--output-dir",
            type=Path,
            default=None,
            help="Directory for report and null distribution (default: output_root/.../oos/permutation/).",
        )
        return parser.parse_args()

    parser = argparse.ArgumentParser(
        description=(
            "Validation permutation test (vector-shuffle null, single fold from config.validation_window)."
        )
    )
    parser.add_argument("--nreps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def run_permutation_phase_main(*, phase: Phase, script_path: Path) -> int:
    ensure_repo_root_on_syspath(script_path.resolve())
    args = parse_permutation_phase_args(phase=phase)

    from feature_research.config import load_config
    from feature_research.validation.phase_permutation import run_permutation_for_phase

    config = load_config()
    if phase == "oos":
        if config.oos_window is None:
            print("Error: config.oos_window is not set. Set it in feature_research.config.load_config().")
            return 1
    elif config.validation_window is None:
        print(
            "Error: config.validation_window is not set. Set it in feature_research.config.load_config()."
        )
        return 1

    return run_permutation_for_phase(phase=phase, config=config, args=args)
