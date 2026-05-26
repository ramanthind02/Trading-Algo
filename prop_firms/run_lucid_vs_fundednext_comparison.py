"""Compare Lucid 50K vs FundedNext 50K CFD on vault portfolio returns."""

from __future__ import annotations

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

from prop_firms.comparison import (
    load_comparison_config,
    run_prop_firm_comparison,
    write_prop_firm_comparison_reports,
)


def main() -> None:
    config = load_comparison_config()
    result = run_prop_firm_comparison(comparison_config=config)
    summary_path = write_prop_firm_comparison_reports(
        result=result,
        comparison_config=config,
    )

    lucid_net = result.lucid.summary.net_cashflow
    fundednext_net = result.fundednext.summary.net_cashflow
    winner = (
        "Lucid"
        if lucid_net > fundednext_net
        else "FundedNext"
        if fundednext_net > lucid_net
        else "Tie"
    )

    print("=" * 70)
    print("LUCID 50K vs FUNDEDNEXT 50K CFD — PORTFOLIO RETURN COMPARISON")
    print("=" * 70)
    print(
        f"Return window: {result.return_window_start.date()} → "
        f"{result.return_window_end.date()} ({len(result.returns)} days @ 10% vol)"
    )
    print(f"Lucid net cashflow:      ${lucid_net:,.2f}")
    print(f"FundedNext net cashflow: ${fundednext_net:,.2f}")
    print(f"Winner: {winner} (${abs(lucid_net - fundednext_net):,.2f} delta)")
    print(f"Summary report: {summary_path}")


if __name__ == "__main__":
    main()
