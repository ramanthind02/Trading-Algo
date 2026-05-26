from __future__ import annotations

from pathlib import Path

from prop_firms.base.config_loader import load_account_definitions
from prop_firms.base.models import AccountDefinition


DEFAULT_FUNDEDNEXT_CONFIG_PATH = Path(__file__).with_name("config.json")


def load_fundednext_accounts(
    config_path: Path | None = None,
) -> dict[str, AccountDefinition]:
    """Load all FundedNext account definitions from config."""

    resolved_path = (
        DEFAULT_FUNDEDNEXT_CONFIG_PATH if config_path is None else config_path
    )
    return load_account_definitions(resolved_path)


def load_fundednext_account(
    account_code: str,
    config_path: Path | None = None,
) -> AccountDefinition:
    """Load one FundedNext account definition by account code."""

    accounts = load_fundednext_accounts(config_path=config_path)
    if account_code not in accounts:
        raise KeyError(
            f"Unknown FundedNext account '{account_code}'. "
            f"Available accounts: {sorted(accounts)}"
        )
    return accounts[account_code]


def create_fundednext_simulator(
    config_path: Path | None = None,
) -> "FundedNextRuleEngine":
    """Factory for the FundedNext single-account simulator."""

    from prop_firms.fundednext.rules import FundedNextRuleEngine

    return FundedNextRuleEngine(
        account_definitions=load_fundednext_accounts(config_path=config_path)
    )


def create_fundednext_portfolio_simulator(
    config_path: Path | None = None,
) -> "FundedNextPortfolioSimulator":
    """Factory for the FundedNext multi-account portfolio simulator."""

    from prop_firms.fundednext.portfolio_simulator import FundedNextPortfolioSimulator

    return FundedNextPortfolioSimulator(
        account_definitions=load_fundednext_accounts(config_path=config_path)
    )
