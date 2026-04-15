"""Vault portfolio phases -> prop_firms multi-account simulator + Markdown/HTML reports.

Configuration for purchase caps, payout policy, vol multipliers, return-engine date window,
and optional target-volatility scaling lives in ``prop_firms.report_config`` (same as
``prop_firms/run_lucid_portfolio_report.py`` / ``run_apex_portfolio_report.py``).

Usage (repo root, Windows)::

    .\\.venv\\Scripts\\python.exe portfolio_research/run_portfolio_prop_firm.py --provider lucid
    .\\.venv\\Scripts\\python.exe portfolio_research/run_portfolio_prop_firm.py --provider apex --phases validation test

Portfolio windows and ensembles come from ``portfolio_research.config.load_config()``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _prepend_repo_root_to_syspath() -> None:
    start = Path(__file__).resolve()
    for parent in (start.parent, *start.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            root = str(parent)
            if root not in sys.path:
                sys.path.insert(0, root)
            return
    raise RuntimeError(
        "Could not locate repository root (no pyproject.toml or .git above this file)."
    )


_prepend_repo_root_to_syspath()

from utils.repo_bootstrap import ensure_repo_root_on_syspath

ensure_repo_root_on_syspath(Path(__file__).resolve())

from prop_firms.report_config import load_apex_report_config, load_report_config

from portfolio_research.config import load_config
from portfolio_research.portfolio_prop_firm_reports import (
    run_portfolio_prop_firm_portfolio_reports,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run portfolio research phase(s) through the prop_firms portfolio simulator "
            "using report_config (Lucid or Apex)."
        )
    )
    parser.add_argument(
        "--provider",
        choices=("lucid", "apex"),
        required=True,
        help="Which report_config loader and portfolio simulator to use",
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=("train", "validation", "test"),
        default=["validation", "test"],
        metavar="PHASE",
        help="Portfolio phases to run (default: validation test)",
    )
    parser.add_argument(
        "--emit-tearsheets",
        action="store_true",
        help="Also write QuantStats HTML for each portfolio phase (slower)",
    )
    parser.add_argument(
        "--report-output-subdir",
        default="vault_portfolio",
        help="Subfolder under report_config.output_dir for report artifacts",
    )
    args = parser.parse_args()

    portfolio_config = load_config()
    report_config = (
        load_report_config()
        if args.provider == "lucid"
        else load_apex_report_config()
    )
    artifacts = run_portfolio_prop_firm_portfolio_reports(
        portfolio_config,
        report_config,
        provider=args.provider,
        phases=tuple(args.phases),
        emit_tearsheets=args.emit_tearsheets,
        report_output_subdir=args.report_output_subdir,
    )

    print("=" * 70)
    print(f"PORTFOLIO -> {args.provider.upper()} PROP FIRM REPORTS")
    print("=" * 70)
    for phase, art in artifacts.items():
        print(f"\n--- Phase: {phase} ---")
        print(f"Markdown: {art.report_markdown_path}")
        print(f"HTML:     {art.report_html_path}")
        if art.daily_timeline_csv_path is not None:
            print(f"Timeline CSV: {art.daily_timeline_csv_path}")
        if art.events_csv_path is not None:
            print(f"Events CSV:   {art.events_csv_path}")


if __name__ == "__main__":
    main()
