"""Validation permutation test (vector-shuffle null, single fold from config.validation_window)."""

from __future__ import annotations

import sys
from pathlib import Path

from feature_research.permutation_script_support import run_permutation_phase_main


def main() -> int:
    return run_permutation_phase_main(phase="validation", script_path=Path(__file__))


if __name__ == "__main__":
    sys.exit(main())
