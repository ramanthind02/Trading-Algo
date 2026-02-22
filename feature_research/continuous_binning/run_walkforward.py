"""Entry point for the continuous binning walkforward research pipeline.

Runs walkforward-only analysis (no EDA) with enhanced selection enabled.

Usage
-----
    source /home/raman/repos/Trading-Algo/venv/bin/activate
    python feature_research/continuous_binning/run_walkforward.py

Artifacts are written to the ``output_root`` defined in ``config.walkforward``
(default: ``feature_research/shared_results/continuous/{module_name}/walkforward/``).

To customise tickers, dates, bias_spec, or walkforward parameters, edit
``feature_research/continuous_binning/config.py``.

To disable enhanced selection, set ``use_enhanced_selection=False`` in the
``dataclasses.replace`` call below, or override ``load_config()`` directly.
"""
import dataclasses
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from feature_research.continuous_binning.config import load_config
from feature_research.continuous_binning.pipeline import run_continuous_walkforward_pipeline


if __name__ == "__main__":
    config = load_config()

    walkforward = dataclasses.replace(
        config.walkforward,
        enabled=True,
        use_enhanced_selection=True,
    )
    config = dataclasses.replace(config, walkforward=walkforward)

    report = run_continuous_walkforward_pipeline(config, config.reports_dir / "walkforward")
    n_folds = len(report.folds_df)
    print(
        f"Walkforward complete. {n_folds} folds. "
        f"Artifacts written to {config.walkforward.output_root}"
    )
