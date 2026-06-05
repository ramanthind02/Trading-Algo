"""Deprecated entrypoint — prop-firm reports run inside the portfolio test pipeline."""
from __future__ import annotations

import sys


def main() -> None:
    print(
        "portfolio_research/run_portfolio_prop_firm.py is deprecated.\n"
        "Prop-firm simulation is integrated into the portfolio test pipeline.\n\n"
        "Run instead:\n"
        "  python -m portfolio_research.run_portfolio_test\n"
        "  python -m portfolio_research.ui.runner   # UI / full pipeline\n\n"
        "FundedNext reports are written under "
        "{output_root}/{train|validation|test}/prop_firm/fundednext/ "
        "when PortfolioResearchConfig.prop_firm_report.enabled is True "
        "(default in load_config()).",
        file=sys.stderr,
    )
    sys.exit(2)


if __name__ == "__main__":
    main()
