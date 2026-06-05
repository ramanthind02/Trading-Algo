"""Unified validation entrypoint for the signed-signal trading pipeline.

Run with (repo root as cwd):

    Linux/macOS:  python -m feature_research validation
                  # or:  python feature_research/validation/run_validation.py

    Windows:      .\\.venv\\Scripts\\python.exe -m feature_research validation
                  # or:  .\\.venv\\Scripts\\python.exe feature_research\\validation\\run_validation.py

Direct script execution prepends the repo root to ``sys.path`` (stdlib only) so
``feature_research`` imports work without ``PYTHONPATH``.
"""
from __future__ import annotations

from pathlib import Path

from lib.core.repo_bootstrap import ensure_repo_root_on_syspath

ensure_repo_root_on_syspath(Path(__file__).resolve())

from research.feature.config import load_config
from research.feature.ui.artifact_catalog import ensure_phase_visualization_reports
from research.feature.ui.workspace_manifest import write_validation_manifest
from research.feature.shared import FeatureResearchPhase
from research.feature.validation import run_validation_pipeline


def main() -> None:
    config = load_config()
    # Use default output path: output_root/feature_type/module_name/validation
    report = run_validation_pipeline(config, output_dir=None)
    write_validation_manifest(config)
    generated = ensure_phase_visualization_reports(config, FeatureResearchPhase.VALIDATION)
    print(
        f"Validation complete. {len(report.folds_df)} fold(s). "
        f"Artifacts written to {config.output_root}"
    )
    if generated:
        print(f"Generated {len(generated)} Matplotlib plot(s) for the workspace.")


if __name__ == "__main__":
    main()
