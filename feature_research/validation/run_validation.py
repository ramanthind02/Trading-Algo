"""Unified validation entrypoint for both continuous and rule-based research.

Run with:
    source /home/raman/repos/Trading-Algo/venv/bin/activate
    python feature_research/validation/run_validation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_repo_hint = Path(__file__).resolve().parents[2]
if str(_repo_hint) not in sys.path:
    sys.path.insert(0, str(_repo_hint))

from feature_research.bootstrap import ensure_repo_root_on_syspath

ensure_repo_root_on_syspath(Path(__file__).resolve())

from feature_research.in_sample.config import load_config
from feature_research.pipeline import run_validation_pipeline


def main() -> None:
    config = load_config()
    report = run_validation_pipeline(config, config.reports_dir / "validation")
    print(
        f"Validation complete. {len(report.folds_df)} fold(s). "
        f"Artifacts written to {config.output_root}"
    )


if __name__ == "__main__":
    main()
