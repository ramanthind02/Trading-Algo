"""CLI for portfolio research UI phases."""
from __future__ import annotations

import argparse

from research.portfolio.config import load_config
from research.portfolio.ui.contracts import UiRunAction
from research.portfolio.ui.planner import build_phase_plan, build_ui_defaults, build_ui_request
from research.portfolio.ui.runner import execute_phase_request, render_phase_plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Portfolio research workspace phase runner")
    parser.add_argument(
        "--phase",
        default="full_pipeline",
        help="Phase to run (default: full_pipeline = all stages)",
    )
    parser.add_argument("--tickers", default="")
    parser.add_argument("--fit-mode", default="")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    config = load_config()
    defaults = build_ui_defaults(config)
    request = build_ui_request(
        phase_name=args.phase,
        tickers_text=args.tickers or None,
        fallback_tickers=defaults.default_tickers,
        fit_mode_name=args.fit_mode or None,
    )
    plan = build_phase_plan(config, request)
    print(render_phase_plan(plan))
    if args.execute:
        raise SystemExit(execute_phase_request(config, request))
    _ = UiRunAction.PREVIEW


if __name__ == "__main__":
    main()
