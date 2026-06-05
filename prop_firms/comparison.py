from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from prop_firms.base.portfolio_models import (
    PortfolioPayoutPolicyConfig,
    PortfolioPayoutPolicyMode,
    PortfolioSimulationConfig,
    PortfolioSimulationResult,
    PurchasePolicyConfig,
    ReturnEngineConfig,
)
from prop_firms.base.return_engine import build_return_series, normalize_return_series
from prop_firms.fundednext.provider import create_fundednext_portfolio_simulator
from prop_firms.lucid.provider import create_lucid_portfolio_simulator
from prop_firms.reporting import generate_portfolio_report
from research.portfolio.config import PortfolioResearchConfig, load_prop_firm_portfolio_research_config
from research.portfolio.pipelines.portfolio_test import (
    run_portfolio_research_cache_preflight,
    run_single_phase_for_prop_firm,
)
from research.portfolio.prop_firm_bridge import (
    align_portfolio_returns_with_report_engine,
    build_prop_firm_returns,
)


_PROP_FIRMS_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PropFirmComparisonConfig:
    """Settings for Lucid vs FundedNext portfolio-backed comparison."""

    years: int = 5
    target_annual_volatility: float = 0.10
    account_cap: int = 6
    lucid_simulation: PortfolioSimulationConfig | None = None
    fundednext_simulation: PortfolioSimulationConfig | None = None
    output_dir: Path = _PROP_FIRMS_DIR / "results" / "provider_comparison"
    report_stem: str = "lucid_vs_fundednext_50k"


@dataclass(frozen=True)
class PropFirmComparisonResult:
    """Side-by-side portfolio simulation outcomes."""

    returns: pd.Series
    return_window_start: pd.Timestamp
    return_window_end: pd.Timestamp
    lucid: PortfolioSimulationResult
    fundednext: PortfolioSimulationResult


def load_comparison_config() -> PropFirmComparisonConfig:
    """Editable defaults for the Lucid 50K vs FundedNext 50K CFD comparison."""

    return_engine = ReturnEngineConfig(
        target_annual_volatility=0.10,
        target_sharpe=1.0,
        annualization_factor=252.0,
        start_date=None,
        end_date=None,
        random_seed=42,
    )
    purchase_policy = PurchasePolicyConfig(
        funded_account_cap=6,
        challenge_account_cap=6,
        challenges_per_purchase_window=1,
    )
    lucid_simulation = PortfolioSimulationConfig(
        account_code="50000",
        purchase_policy=purchase_policy,
        payout_policy=PortfolioPayoutPolicyConfig(
            mode=PortfolioPayoutPolicyMode.BUFFER,
            buffer_amount=1500.0,
            withdrawal_fraction=1.0,
        ),
        return_engine=return_engine,
        challenge_vol_multiplier=1.0,
        funded_vol_multiplier=1.0,
    )
    fundednext_simulation = PortfolioSimulationConfig(
        account_code="50000",
        purchase_policy=purchase_policy,
        payout_policy=PortfolioPayoutPolicyConfig(
            mode=PortfolioPayoutPolicyMode.CFD_LADDER,
            cfd_buffer_pct=0.03,
            cfd_profit_step_pct=0.02,
            cfd_payout_amount_pct=0.02,
        ),
        return_engine=return_engine,
        challenge_vol_multiplier=1.0,
        funded_vol_multiplier=1.0,
    )
    return PropFirmComparisonConfig(
        lucid_simulation=lucid_simulation,
        fundednext_simulation=fundednext_simulation,
    )


def build_portfolio_returns_for_comparison(
    portfolio_config: PortfolioResearchConfig,
    *,
    years: int,
) -> pd.Series:
    """Load validation+test portfolio returns and trim to the last N calendar years."""

    run_portfolio_research_cache_preflight(portfolio_config)
    phase_returns = [
        build_prop_firm_returns(
            run_single_phase_for_prop_firm(
                portfolio_config,
                phase,
                emit_tearsheets=False,
                run_preflight=False,
            )
        )
        for phase in ("validation", "test")
    ]
    combined = pd.concat(phase_returns)
    combined = combined[~combined.index.duplicated(keep="last")].sort_index(kind="stable")
    normalized = normalize_return_series(combined.astype(float))
    cutoff = pd.Timestamp(date.today() - timedelta(days=365 * years))
    trimmed = normalized[normalized.index >= cutoff]
    if trimmed.empty:
        raise ValueError(
            f"No portfolio returns remain after trimming to the last {years} years "
            f"(cutoff={cutoff.date().isoformat()})"
        )
    return trimmed


def run_prop_firm_comparison(
    *,
    portfolio_config: PortfolioResearchConfig | None = None,
    comparison_config: PropFirmComparisonConfig | None = None,
) -> PropFirmComparisonResult:
    """Run Lucid 50K and FundedNext 50K CFD simulators on the same return stream."""

    resolved_comparison = (
        load_comparison_config() if comparison_config is None else comparison_config
    )
    resolved_portfolio = (
        load_prop_firm_portfolio_research_config()
        if portfolio_config is None
        else portfolio_config
    )
    if resolved_comparison.lucid_simulation is None:
        raise ValueError("lucid_simulation must be set")
    if resolved_comparison.fundednext_simulation is None:
        raise ValueError("fundednext_simulation must be set")

    raw_returns = build_portfolio_returns_for_comparison(
        resolved_portfolio,
        years=resolved_comparison.years,
    )
    cutoff = raw_returns.index.min()
    end_date = raw_returns.index.max()
    lucid_engine = ReturnEngineConfig(
        target_annual_volatility=resolved_comparison.target_annual_volatility,
        target_sharpe=resolved_comparison.lucid_simulation.return_engine.target_sharpe,
        annualization_factor=resolved_comparison.lucid_simulation.return_engine.annualization_factor,
        start_date=cutoff.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        random_seed=resolved_comparison.lucid_simulation.return_engine.random_seed,
    )
    aligned_returns = build_return_series(
        config=lucid_engine,
        external_returns=raw_returns,
    )
    lucid_simulation = PortfolioSimulationConfig(
        account_code=resolved_comparison.lucid_simulation.account_code,
        challenge_vol_multiplier=resolved_comparison.lucid_simulation.challenge_vol_multiplier,
        funded_vol_multiplier=resolved_comparison.lucid_simulation.funded_vol_multiplier,
        purchase_policy=resolved_comparison.lucid_simulation.purchase_policy,
        payout_policy=resolved_comparison.lucid_simulation.payout_policy,
        return_engine=lucid_engine,
        max_payouts_per_funded_account=resolved_comparison.lucid_simulation.max_payouts_per_funded_account,
    )
    fundednext_simulation = PortfolioSimulationConfig(
        account_code=resolved_comparison.fundednext_simulation.account_code,
        challenge_vol_multiplier=resolved_comparison.fundednext_simulation.challenge_vol_multiplier,
        funded_vol_multiplier=resolved_comparison.fundednext_simulation.funded_vol_multiplier,
        purchase_policy=resolved_comparison.fundednext_simulation.purchase_policy,
        payout_policy=resolved_comparison.fundednext_simulation.payout_policy,
        return_engine=lucid_engine,
        max_payouts_per_funded_account=resolved_comparison.fundednext_simulation.max_payouts_per_funded_account,
    )
    lucid_result = create_lucid_portfolio_simulator().simulate(
        returns=aligned_returns,
        config=lucid_simulation,
    )
    fundednext_result = create_fundednext_portfolio_simulator().simulate(
        returns=aligned_returns,
        config=fundednext_simulation,
    )
    return PropFirmComparisonResult(
        returns=aligned_returns,
        return_window_start=aligned_returns.index.min(),
        return_window_end=aligned_returns.index.max(),
        lucid=lucid_result,
        fundednext=fundednext_result,
    )


def write_prop_firm_comparison_reports(
    result: PropFirmComparisonResult,
    comparison_config: PropFirmComparisonConfig,
) -> Path:
    """Write per-provider reports plus a side-by-side Markdown summary."""

    output_dir = comparison_config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    lucid_artifacts = generate_portfolio_report(
        result=result.lucid,
        output_dir=output_dir,
        report_stem=f"{comparison_config.report_stem}_lucid",
        save_csvs=True,
    )
    fundednext_artifacts = generate_portfolio_report(
        result=result.fundednext,
        output_dir=output_dir,
        report_stem=f"{comparison_config.report_stem}_fundednext",
        save_csvs=True,
    )
    summary_path = output_dir / f"{comparison_config.report_stem}.md"
    summary_path.write_text(
        _build_comparison_markdown(
            result=result,
            lucid_report=lucid_artifacts.report_markdown_path,
            fundednext_report=fundednext_artifacts.report_markdown_path,
        ),
        encoding="utf-8",
    )
    return summary_path


def _build_comparison_markdown(
    *,
    result: PropFirmComparisonResult,
    lucid_report: Path,
    fundednext_report: Path,
) -> str:
    lucid = result.lucid.summary
    fundednext = result.fundednext.summary
    winner = (
        "Lucid"
        if lucid.net_cashflow > fundednext.net_cashflow
        else "FundedNext"
        if fundednext.net_cashflow > lucid.net_cashflow
        else "Tie"
    )
    delta = lucid.net_cashflow - fundednext.net_cashflow
    lines = [
        "# Lucid 50K vs FundedNext Stellar 2-Step CFD (50K)",
        "",
        "## Return Source",
        f"- Portfolio returns: vault-backed validation + test phases",
        f"- Window: `{result.return_window_start.date()}` → `{result.return_window_end.date()}`",
        f"- Trading days: `{len(result.returns)}`",
        f"- Target annual volatility: `10%`",
        "",
        "## Payout Policies",
        "- **Lucid:** native rules with `BUFFER` policy and `$1,500` cushion (~3% of $50K)",
        "- **FundedNext:** `CFD_LADDER` — build 3% buffer, then withdraw 2% of nominal account size at each additional 2% profit step (first at 5%)",
        "",
        "## Headline Comparison",
        "| Metric | Lucid 50K | FundedNext 50K CFD |",
        "| --- | ---: | ---: |",
        f"| Net cashflow | `${lucid.net_cashflow:,.2f}` | `${fundednext.net_cashflow:,.2f}` |",
        f"| Trader payouts | `${lucid.total_trader_payouts:,.2f}` | `${fundednext.total_trader_payouts:,.2f}` |",
        f"| Fee refunds | `${lucid.total_fee_refunds:,.2f}` | `${fundednext.total_fee_refunds:,.2f}` |",
        f"| Challenge costs | `${lucid.total_challenge_costs:,.2f}` | `${fundednext.total_challenge_costs:,.2f}` |",
        f"| Funded accounts created | `{lucid.funded_accounts_created}` | `{fundednext.funded_accounts_created}` |",
        f"| Funded accounts closed | `{lucid.funded_accounts_closed}` | `{fundednext.funded_accounts_closed}` |",
        f"| Challenges purchased | `{lucid.challenges_purchased}` | `{fundednext.challenges_purchased}` |",
        f"| Challenges failed | `{lucid.challenges_failed}` | `{fundednext.challenges_failed}` |",
        f"| Payouts per funded account | `{lucid.payouts_per_funded_account:.2f}` | `{fundednext.payouts_per_funded_account:.2f}` |",
        "",
        f"**Winner on net cashflow:** {winner} (`${abs(delta):,.2f}` difference)",
        "",
        "## Detail Reports",
        f"- Lucid: `{lucid_report}`",
        f"- FundedNext: `{fundednext_report}`",
    ]
    return "\n".join(lines)
