"""CLI entrypoint for research-only continuous-feature binning exports."""
from __future__ import annotations

from feature_research.binning.config import load_binning_research_config
from feature_research.binning.pipeline import run_continuous_binning_phase


def main() -> int:
    config = load_binning_research_config()
    summary = run_continuous_binning_phase(config)
    print(f"Continuous binning research complete. bar_level_rows={summary['bar_level_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
