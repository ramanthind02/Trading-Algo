"""OOS permutation test (vector-shuffle null, single fold from config.oos_window)."""

from __future__ import annotations

import sys
from pathlib import Path

from feature_research.permutation_script_support import run_permutation_phase_main


def main() -> int:
    return run_permutation_phase_main(phase="oos", script_path=Path(__file__))


if __name__ == "__main__":
    sys.exit(main())
