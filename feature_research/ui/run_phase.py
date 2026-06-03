"""Small CLI wrapper used by the lightweight feature_research UI."""
from __future__ import annotations

import argparse
from collections.abc import Sequence

from feature_research.config import load_config
from feature_research.shared import FeatureResearchPhase
from feature_research.ui.contracts import UiRunAction
from feature_research.ui.planner import build_phase_plan, build_ui_defaults, build_ui_request
from feature_research.ui.runner import execute_phase_request, render_phase_plan


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Preview or execute a lightweight feature_research phase run "
            "while keeping feature_research/config.py as the source of truth."
        )
    )
    parser.add_argument(
        "--phase",
        choices=[phase.value for phase in FeatureResearchPhase],
        required=True,
        help="Top-level feature_research phase to run.",
    )
    parser.add_argument(
        "--tickers",
        default="",
        help="Comma-separated research ticker symbols. Leave blank to use the config default.",
    )
    parser.add_argument(
        "--portfolio-tickers",
        default="",
        help=(
            "Comma-separated tickers for portfolio addition gate baselines. "
            "Leave blank to use portfolio_source.tickers from feature_research/config.py "
            "(e.g. ES,NQ,GC while researching GC only)."
        ),
    )
    parser.add_argument("--train-start", default=None, help="Optional YYYY-MM-DD train start override.")
    parser.add_argument("--train-end", default=None, help="Optional YYYY-MM-DD train end override.")
    parser.add_argument("--val-start", default=None, help="Optional YYYY-MM-DD validation start override.")
    parser.add_argument("--val-end", default=None, help="Optional YYYY-MM-DD validation end override.")
    parser.add_argument("--test-start", default=None, help="Deprecated alias for --val-start.")
    parser.add_argument("--test-end", default=None, help="Deprecated alias for --val-end.")
    parser.add_argument(
        "--action",
        choices=[action.value for action in UiRunAction],
        default=UiRunAction.PREVIEW.value,
        help="Preview the plan or execute the requested phase.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    config = load_config()
    defaults = build_ui_defaults(config)
    request = build_ui_request(
        phase_name=args.phase,
        tickers_text=args.tickers,
        fallback_tickers=defaults.default_tickers,
        portfolio_tickers_text=args.portfolio_tickers or None,
        train_start=args.train_start,
        train_end=args.train_end,
        val_start=args.val_start,
        val_end=args.val_end,
        test_start=args.test_start,
        test_end=args.test_end,
    )
    plan = build_phase_plan(config, request)
    print(render_phase_plan(plan))
    if UiRunAction(args.action) is UiRunAction.PREVIEW:
        return 0
    return execute_phase_request(config, request)


if __name__ == "__main__":
    raise SystemExit(main())
