from __future__ import annotations

from dataclasses import dataclass, field, replace

import optuna
import pandas as pd

from prop_firms.base import build_return_series, compute_portfolio_statistics
from prop_firms.base.portfolio_models import (
    PortfolioBatchStatistics,
    PortfolioPayoutPolicyConfig,
    PortfolioPayoutPolicyMode,
    PortfolioSimulationConfig,
    PurchasePolicyConfig,
    ReturnEngineConfig,
)
from prop_firms.lucid import LucidPortfolioSimulator, create_lucid_portfolio_simulator


@dataclass(frozen=True)
class FloatSearchRange:
    """Continuous search range used by the optimizer."""

    low: float
    high: float

    def __post_init__(self) -> None:
        if self.low <= 0.0:
            raise ValueError("FloatSearchRange.low must be > 0")
        if self.high <= self.low:
            raise ValueError("FloatSearchRange.high must be > low")


@dataclass(frozen=True)
class IntSearchRange:
    """Integer search range used by the optimizer."""

    low: int
    high: int

    def __post_init__(self) -> None:
        if self.low < 1:
            raise ValueError("IntSearchRange.low must be >= 1")
        if self.high < self.low:
            raise ValueError("IntSearchRange.high must be >= low")


@dataclass(frozen=True)
class LucidHyperoptConfig:
    """Top-level config for Lucid Monte Carlo hyperparameter optimization."""

    account_code: str = "25000"
    funded_account_cap: int = 20
    challenge_account_cap: int = 20
    target_sharpe: float = 2.0
    monte_carlo_runs: int = 100
    n_trials: int = 50
    n_jobs: int = 1
    target_annual_volatility: float = 0.10
    annualization_factor: float = 252.0
    start_date: str | None = "2021-01-01"
    end_date: str | None = "2025-12-31"
    sampler_seed: int = 42
    random_seed_start: int = 10_000
    challenge_vol_multiplier_range: FloatSearchRange = field(
        default_factory=lambda: FloatSearchRange(low=0.4, high=2.0)
    )
    funded_vol_multiplier_range: FloatSearchRange = field(
        default_factory=lambda: FloatSearchRange(low=0.2, high=1.5)
    )
    payout_buffer_range: FloatSearchRange = field(
        default_factory=lambda: FloatSearchRange(low=100.0, high=5_000.0)
    )
    withdrawal_fraction_range: FloatSearchRange = field(
        default_factory=lambda: FloatSearchRange(low=0.25, high=1.0)
    )
    challenges_per_purchase_window_range: IntSearchRange = field(
        default_factory=lambda: IntSearchRange(low=1, high=5)
    )
    max_payouts_per_funded_account_choices: tuple[int | None, ...] = (
        None,
        3,
        4,
        5,
        6,
    )
    payout_modes: tuple[PortfolioPayoutPolicyMode, ...] = (
        PortfolioPayoutPolicyMode.AGGRESSIVE,
        PortfolioPayoutPolicyMode.BUFFER,
        PortfolioPayoutPolicyMode.FRACTIONAL,
    )

    def __post_init__(self) -> None:
        if self.funded_account_cap < 1:
            raise ValueError("funded_account_cap must be >= 1")
        if self.challenge_account_cap < 1:
            raise ValueError("challenge_account_cap must be >= 1")
        if self.target_annual_volatility <= 0.0:
            raise ValueError("target_annual_volatility must be > 0")
        if self.annualization_factor <= 0.0:
            raise ValueError("annualization_factor must be > 0")
        if self.monte_carlo_runs < 1:
            raise ValueError("monte_carlo_runs must be >= 1")
        if self.n_trials < 1:
            raise ValueError("n_trials must be >= 1")
        if self.n_jobs < 1:
            raise ValueError("n_jobs must be >= 1")
        if self.sampler_seed < 0:
            raise ValueError("sampler_seed must be >= 0")
        if self.random_seed_start < 0:
            raise ValueError("random_seed_start must be >= 0")
        if not self.max_payouts_per_funded_account_choices:
            raise ValueError("max_payouts_per_funded_account_choices must be non-empty")
        if not self.payout_modes:
            raise ValueError("payout_modes must be non-empty")
        if self.start_date is not None:
            pd.Timestamp(self.start_date)
        if self.end_date is not None:
            pd.Timestamp(self.end_date)


@dataclass(frozen=True)
class MonteCarloRunSummary:
    """Single seed-level run summary inside one optimization trial."""

    trial_number: int
    random_seed: int
    net_cashflow: float
    trader_payouts: float
    gross_payouts: float
    funded_accounts_created: int
    challenges_purchased: int
    challenges_failed: int
    days_to_first_payout: int | None

    def to_record(self) -> dict[str, object]:
        return {
            "trial_number": self.trial_number,
            "random_seed": self.random_seed,
            "net_cashflow": self.net_cashflow,
            "trader_payouts": self.trader_payouts,
            "gross_payouts": self.gross_payouts,
            "funded_accounts_created": self.funded_accounts_created,
            "challenges_purchased": self.challenges_purchased,
            "challenges_failed": self.challenges_failed,
            "days_to_first_payout": self.days_to_first_payout,
        }


@dataclass(frozen=True)
class LucidHyperoptTrialResult:
    """Aggregate result for one optimizer trial."""

    trial_number: int
    objective_value: float
    simulation_config: PortfolioSimulationConfig
    batch_statistics: PortfolioBatchStatistics
    monte_carlo_seeds: tuple[int, ...]

    def to_record(self) -> dict[str, object]:
        return {
            "trial_number": self.trial_number,
            "objective_value": self.objective_value,
            **_portfolio_config_to_record(self.simulation_config),
            **self.batch_statistics.to_record(),
        }


@dataclass(frozen=True)
class LucidHyperoptResult:
    """Full optimizer output including all trial and seed summaries."""

    config: LucidHyperoptConfig
    best_trial: LucidHyperoptTrialResult
    trials: tuple[LucidHyperoptTrialResult, ...]
    seed_runs: tuple[MonteCarloRunSummary, ...]
    trials_frame: pd.DataFrame
    seed_runs_frame: pd.DataFrame


def build_monte_carlo_seeds(config: LucidHyperoptConfig) -> tuple[int, ...]:
    """Build the deterministic seed set used for every trial."""

    return tuple(config.random_seed_start + idx for idx in range(config.monte_carlo_runs))


def build_base_return_engine_config(config: LucidHyperoptConfig) -> ReturnEngineConfig:
    """Build the shared synthetic-return template for optimizer trials."""

    return ReturnEngineConfig(
        target_annual_volatility=config.target_annual_volatility,
        target_sharpe=config.target_sharpe,
        annualization_factor=config.annualization_factor,
        start_date=config.start_date,
        end_date=config.end_date,
        random_seed=config.random_seed_start,
    )


def evaluate_portfolio_config(
    simulator: LucidPortfolioSimulator,
    config: LucidHyperoptConfig,
    portfolio_config: PortfolioSimulationConfig,
    trial_number: int,
) -> tuple[LucidHyperoptTrialResult, tuple[MonteCarloRunSummary, ...]]:
    """Evaluate one portfolio config across many synthetic seeded paths."""

    monte_carlo_seeds = build_monte_carlo_seeds(config)
    base_return_engine = build_base_return_engine_config(config)
    results = tuple(
        simulator.simulate(
            returns=build_return_series(
                config=replace(base_return_engine, random_seed=random_seed)
            ),
            config=portfolio_config,
        )
        for random_seed in monte_carlo_seeds
    )
    batch_statistics = compute_portfolio_statistics(list(results))
    trial_result = LucidHyperoptTrialResult(
        trial_number=trial_number,
        objective_value=batch_statistics.expected_net_cashflow,
        simulation_config=portfolio_config,
        batch_statistics=batch_statistics,
        monte_carlo_seeds=monte_carlo_seeds,
    )
    run_summaries = tuple(
        MonteCarloRunSummary(
            trial_number=trial_number,
            random_seed=random_seed,
            net_cashflow=result.summary.net_cashflow,
            trader_payouts=result.summary.total_trader_payouts,
            gross_payouts=result.summary.total_gross_payouts,
            funded_accounts_created=result.summary.funded_accounts_created,
            challenges_purchased=result.summary.challenges_purchased,
            challenges_failed=result.summary.challenges_failed,
            days_to_first_payout=result.summary.days_to_first_payout,
        )
        for random_seed, result in zip(monte_carlo_seeds, results, strict=True)
    )
    return trial_result, run_summaries


def sample_portfolio_config(
    trial: optuna.trial.Trial,
    config: LucidHyperoptConfig,
) -> PortfolioSimulationConfig:
    """Sample one candidate portfolio config from the optimizer search space."""

    payout_mode = PortfolioPayoutPolicyMode(
        trial.suggest_categorical(
            "payout_mode",
            [mode.value for mode in config.payout_modes],
        )
    )
    payout_policy = _sample_payout_policy(
        trial=trial,
        config=config,
        payout_mode=payout_mode,
    )
    max_payouts_per_funded_account = _sample_max_payouts_per_funded_account(
        trial=trial,
        config=config,
    )
    challenges_per_purchase_window = trial.suggest_int(
        "challenges_per_purchase_window",
        config.challenges_per_purchase_window_range.low,
        config.challenges_per_purchase_window_range.high,
    )
    challenge_vol_multiplier = trial.suggest_float(
        "challenge_vol_multiplier",
        config.challenge_vol_multiplier_range.low,
        config.challenge_vol_multiplier_range.high,
    )
    funded_vol_multiplier = trial.suggest_float(
        "funded_vol_multiplier",
        config.funded_vol_multiplier_range.low,
        config.funded_vol_multiplier_range.high,
    )
    return PortfolioSimulationConfig(
        account_code=config.account_code,
        challenge_vol_multiplier=challenge_vol_multiplier,
        funded_vol_multiplier=funded_vol_multiplier,
        purchase_policy=PurchasePolicyConfig(
            funded_account_cap=config.funded_account_cap,
            challenge_account_cap=config.challenge_account_cap,
            challenges_per_purchase_window=challenges_per_purchase_window,
        ),
        payout_policy=payout_policy,
        return_engine=build_base_return_engine_config(config),
        max_payouts_per_funded_account=max_payouts_per_funded_account,
    )


def optimize_lucid_hyperparameters(
    config: LucidHyperoptConfig,
    simulator: LucidPortfolioSimulator | None = None,
) -> LucidHyperoptResult:
    """Run the Optuna study and return normalized trial outputs."""

    resolved_simulator = (
        create_lucid_portfolio_simulator() if simulator is None else simulator
    )
    sampler = optuna.samplers.TPESampler(seed=config.sampler_seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    trial_results: dict[int, LucidHyperoptTrialResult] = {}
    seed_runs_by_trial: dict[int, tuple[MonteCarloRunSummary, ...]] = {}

    def objective(trial: optuna.trial.Trial) -> float:
        portfolio_config = sample_portfolio_config(trial=trial, config=config)
        trial_result, run_summaries = evaluate_portfolio_config(
            simulator=resolved_simulator,
            config=config,
            portfolio_config=portfolio_config,
            trial_number=trial.number,
        )
        _attach_trial_metrics(trial=trial, trial_result=trial_result)
        trial_results[trial.number] = trial_result
        seed_runs_by_trial[trial.number] = run_summaries
        return trial_result.objective_value

    study.optimize(objective, n_trials=config.n_trials, n_jobs=config.n_jobs)
    ordered_trials = tuple(
        trial_results[trial_number] for trial_number in sorted(trial_results)
    )
    ordered_seed_runs = tuple(
        run_summary
        for trial_number in sorted(seed_runs_by_trial)
        for run_summary in seed_runs_by_trial[trial_number]
    )
    trials_frame = pd.DataFrame(
        trial_result.to_record() for trial_result in ordered_trials
    ).sort_values("objective_value", ascending=False, kind="stable")
    seed_runs_frame = pd.DataFrame(
        run_summary.to_record() for run_summary in ordered_seed_runs
    ).sort_values(["trial_number", "random_seed"], kind="stable")
    best_trial = trial_results[study.best_trial.number]
    return LucidHyperoptResult(
        config=config,
        best_trial=best_trial,
        trials=ordered_trials,
        seed_runs=ordered_seed_runs,
        trials_frame=trials_frame.reset_index(drop=True),
        seed_runs_frame=seed_runs_frame.reset_index(drop=True),
    )


def run_lucid_hyperopt(
    config: LucidHyperoptConfig,
    simulator: LucidPortfolioSimulator | None = None,
) -> LucidHyperoptResult:
    """Backward-compatible alias for the Lucid hyperparameter optimizer."""

    return optimize_lucid_hyperparameters(config=config, simulator=simulator)


def _sample_payout_policy(
    trial: optuna.trial.Trial,
    config: LucidHyperoptConfig,
    payout_mode: PortfolioPayoutPolicyMode,
) -> PortfolioPayoutPolicyConfig:
    match payout_mode:
        case PortfolioPayoutPolicyMode.AGGRESSIVE:
            return PortfolioPayoutPolicyConfig(mode=payout_mode)
        case PortfolioPayoutPolicyMode.BUFFER:
            return PortfolioPayoutPolicyConfig(
                mode=payout_mode,
                buffer_amount=trial.suggest_float(
                    "buffer_amount",
                    config.payout_buffer_range.low,
                    config.payout_buffer_range.high,
                ),
            )
        case PortfolioPayoutPolicyMode.FRACTIONAL:
            return PortfolioPayoutPolicyConfig(
                mode=payout_mode,
                withdrawal_fraction=trial.suggest_float(
                    "withdrawal_fraction",
                    config.withdrawal_fraction_range.low,
                    config.withdrawal_fraction_range.high,
                ),
            )
    raise ValueError(f"Unsupported payout mode: {payout_mode}")


def _sample_max_payouts_per_funded_account(
    trial: optuna.trial.Trial,
    config: LucidHyperoptConfig,
) -> int | None:
    choice_labels = tuple(
        "provider_default" if choice is None else str(choice)
        for choice in config.max_payouts_per_funded_account_choices
    )
    selected_label = trial.suggest_categorical(
        "max_payouts_per_funded_account",
        list(choice_labels),
    )
    return None if selected_label == "provider_default" else int(selected_label)


def _attach_trial_metrics(
    trial: optuna.trial.Trial,
    trial_result: LucidHyperoptTrialResult,
) -> None:
    stats = trial_result.batch_statistics
    trial.set_user_attr("expected_net_cashflow", stats.expected_net_cashflow)
    trial.set_user_attr(
        "probability_negative_net_cashflow",
        stats.probability_negative_net_cashflow,
    )
    trial.set_user_attr(
        "expected_total_trader_payouts",
        stats.expected_total_trader_payouts,
    )
    trial.set_user_attr(
        "expected_funded_accounts_created",
        stats.expected_funded_accounts_created,
    )
    trial.set_user_attr(
        "expected_days_to_first_payout",
        stats.expected_days_to_first_payout,
    )


def _portfolio_config_to_record(
    config: PortfolioSimulationConfig,
) -> dict[str, object]:
    return {
        "account_code": config.account_code,
        "challenge_vol_multiplier": config.challenge_vol_multiplier,
        "funded_vol_multiplier": config.funded_vol_multiplier,
        "funded_account_cap": config.purchase_policy.funded_account_cap,
        "challenge_account_cap": config.purchase_policy.challenge_account_cap,
        "challenges_per_purchase_window": (
            config.purchase_policy.challenges_per_purchase_window
        ),
        "payout_mode": config.payout_policy.mode.value,
        "buffer_amount": config.payout_policy.buffer_amount,
        "withdrawal_fraction": config.payout_policy.withdrawal_fraction,
        "target_sharpe": config.return_engine.target_sharpe,
        "target_annual_volatility": (
            config.return_engine.target_annual_volatility
        ),
        "start_date": config.return_engine.start_date,
        "end_date": config.return_engine.end_date,
        "max_payouts_per_funded_account": config.max_payouts_per_funded_account,
    }
