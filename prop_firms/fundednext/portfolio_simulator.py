from __future__ import annotations

from dataclasses import replace

import pandas as pd

from prop_firms.base.enums import AccountPhase, BreachReason
from prop_firms.base.models import AccountDefinition, EvaluationRules, PayoutCycleRule
from prop_firms.base.portfolio_models import (
    AccountLifecycleState,
    PortfolioAccountState,
    PortfolioDailySnapshot,
    PortfolioEvent,
    PortfolioEventType,
    PortfolioPayoutPolicyConfig,
    PortfolioPayoutPolicyMode,
    PortfolioSimulationConfig,
    PortfolioSimulationResult,
    PortfolioSimulationSummary,
)
from prop_firms.base.return_engine import normalize_return_series
from prop_firms.base.cfd_ladder_payout import resolve_cfd_ladder_payout
from prop_firms.fundednext.drawdown import compute_static_max_loss_floor
from prop_firms.fundednext.rules import _apply_daily_loss_limit


class FundedNextPortfolioSimulator:
    """Daily event-driven multi-account simulator for FundedNext Stellar 2-Step CFD."""

    def __init__(self, account_definitions: dict[str, AccountDefinition]) -> None:
        self._account_definitions = account_definitions

    def simulate(
        self,
        returns: pd.Series,
        config: PortfolioSimulationConfig,
    ) -> PortfolioSimulationResult:
        account_definition = self._get_account_definition(config.account_code)
        normalized_returns = normalize_return_series(returns)
        challenge_returns = normalized_returns * config.challenge_vol_multiplier
        funded_returns = normalized_returns * config.funded_vol_multiplier
        max_payouts = (
            account_definition.funded.payout_rule.max_requests_per_account
            if config.max_payouts_per_funded_account is None
            else config.max_payouts_per_funded_account
        )

        active_accounts: list[PortfolioAccountState] = []
        closed_accounts: list[PortfolioAccountState] = []
        events: list[PortfolioEvent] = [
            PortfolioEvent(
                trading_day=normalized_returns.index[0],
                event_type=PortfolioEventType.SIMULATION_STARTED,
                message=(
                    "Started FundedNext Stellar 2-Step CFD portfolio simulation for "
                    f"{account_definition.account_code}"
                ),
            )
        ]
        snapshots: list[PortfolioDailySnapshot] = []
        cumulative_net_cashflow = 0.0
        next_account_id = 1

        for idx, (trading_day, base_daily_return) in enumerate(normalized_returns.items()):
            challenge_daily_return = float(challenge_returns.loc[trading_day])
            funded_daily_return = float(funded_returns.loc[trading_day])
            day_events: list[PortfolioEvent] = []
            purchased_challenges = 0
            challenge_failures = 0
            challenge_passes = 0
            funded_closures = 0
            gross_payouts = 0.0
            trader_payouts = 0.0
            fee_refunds = 0.0
            challenge_costs = 0.0
            activation_costs = 0.0
            reset_costs = 0.0

            if _is_month_start(trading_day=trading_day, index=normalized_returns.index, idx=idx):
                active_funded_count = sum(
                    1
                    for account in active_accounts
                    if account.lifecycle_state == AccountLifecycleState.FUNDED_ACTIVE
                )
                active_challenge_count = sum(
                    1
                    for account in active_accounts
                    if account.lifecycle_state == AccountLifecycleState.CHALLENGE_ACTIVE
                )
                if (
                    active_funded_count < config.purchase_policy.funded_account_cap
                    and active_challenge_count < config.purchase_policy.challenge_account_cap
                ):
                    while (
                        purchased_challenges
                        < config.purchase_policy.challenges_per_purchase_window
                        and active_funded_count < config.purchase_policy.funded_account_cap
                        and active_challenge_count < config.purchase_policy.challenge_account_cap
                    ):
                        account_id = f"fundednext_{next_account_id}"
                        next_account_id += 1
                        new_account = _open_challenge_account(
                            account_id=account_id,
                            trading_day=trading_day,
                            account_definition=account_definition,
                        )
                        active_accounts.append(new_account)
                        purchased_challenges += 1
                        active_challenge_count += 1
                        challenge_costs += account_definition.fees.challenge_fee
                        day_events.append(
                            PortfolioEvent(
                                trading_day=trading_day,
                                event_type=PortfolioEventType.CHALLENGE_PURCHASED,
                                account_id=account_id,
                                lifecycle_state=AccountLifecycleState.CHALLENGE_ACTIVE,
                                message=f"Purchased FundedNext challenge {account_id}",
                                amount=account_definition.fees.challenge_fee,
                            )
                        )
                else:
                    day_events.append(
                        PortfolioEvent(
                            trading_day=trading_day,
                            event_type=PortfolioEventType.PURCHASE_SKIPPED,
                            message=(
                                "Skipped month-start challenge purchase due to "
                                "funded/challenge slot cap"
                            ),
                        )
                    )

            challenge_accounts = [
                account
                for account in active_accounts
                if account.lifecycle_state == AccountLifecycleState.CHALLENGE_ACTIVE
            ]
            funded_accounts = [
                account
                for account in active_accounts
                if account.lifecycle_state == AccountLifecycleState.FUNDED_ACTIVE
            ]
            next_active_accounts: list[PortfolioAccountState] = []

            for account in funded_accounts:
                (
                    updated_account,
                    account_events,
                    account_gross_payouts,
                    account_trader_payouts,
                    account_fee_refunds,
                    funded_closed_flag,
                ) = _advance_funded_account(
                    account=account,
                    trading_day=trading_day,
                    daily_return=funded_daily_return,
                    payout_policy=config.payout_policy,
                    account_definition=account_definition,
                    max_payouts=max_payouts,
                )
                gross_payouts += account_gross_payouts
                trader_payouts += account_trader_payouts
                fee_refunds += account_fee_refunds
                funded_closures += int(funded_closed_flag)
                day_events.extend(account_events)
                if updated_account.lifecycle_state == AccountLifecycleState.FUNDED_ACTIVE:
                    next_active_accounts.append(updated_account)
                else:
                    closed_accounts.append(updated_account)

            funded_slots_remaining = max(
                config.purchase_policy.funded_account_cap
                - sum(
                    1
                    for account in next_active_accounts
                    if account.lifecycle_state == AccountLifecycleState.FUNDED_ACTIVE
                ),
                0,
            )
            for account in challenge_accounts:
                (
                    updated_account,
                    account_events,
                    passed_flag,
                    failed_flag,
                ) = _advance_challenge_account(
                    account=account,
                    trading_day=trading_day,
                    daily_return=challenge_daily_return,
                    account_definition=account_definition,
                )
                day_events.extend(account_events)
                if updated_account.lifecycle_state == AccountLifecycleState.CHALLENGE_ACTIVE:
                    next_active_accounts.append(updated_account)
                    challenge_failures += int(failed_flag)
                    continue
                if updated_account.lifecycle_state == AccountLifecycleState.FUNDED_ACTIVE:
                    if funded_slots_remaining > 0:
                        next_active_accounts.append(updated_account)
                        funded_slots_remaining -= 1
                        challenge_passes += int(passed_flag)
                        day_events.extend(
                            _build_challenge_pass_events(
                                account_id=updated_account.account_id,
                                trading_day=trading_day,
                            )
                        )
                    else:
                        closed_accounts.append(
                            _discard_passed_challenge(
                                account=updated_account,
                                trading_day=trading_day,
                            )
                        )
                        day_events.append(
                            PortfolioEvent(
                                trading_day=trading_day,
                                event_type=PortfolioEventType.CHALLENGE_DISCARDED,
                                account_id=updated_account.account_id,
                                lifecycle_state=AccountLifecycleState.CHALLENGE_DISCARDED,
                                message=(
                                    f"Challenge {updated_account.account_id} was discarded "
                                    "because funded slots were full"
                                ),
                            )
                        )
                    continue
                challenge_failures += int(failed_flag)
                closed_accounts.append(updated_account)

            active_accounts = next_active_accounts
            net_cashflow = (
                trader_payouts + fee_refunds - challenge_costs - activation_costs - reset_costs
            )
            cumulative_net_cashflow += net_cashflow
            snapshots.append(
                PortfolioDailySnapshot(
                    trading_day=trading_day,
                    daily_return=float(base_daily_return),
                    active_challenges=sum(
                        1
                        for account in active_accounts
                        if account.lifecycle_state == AccountLifecycleState.CHALLENGE_ACTIVE
                    ),
                    active_funded=sum(
                        1
                        for account in active_accounts
                        if account.lifecycle_state == AccountLifecycleState.FUNDED_ACTIVE
                    ),
                    purchased_challenges=purchased_challenges,
                    challenge_failures=challenge_failures,
                    challenge_passes=challenge_passes,
                    funded_closures=funded_closures,
                    gross_payouts=gross_payouts,
                    trader_payouts=trader_payouts,
                    challenge_costs=challenge_costs,
                    activation_costs=activation_costs,
                    reset_costs=reset_costs,
                    fee_refunds=fee_refunds,
                    net_cashflow=net_cashflow,
                    cumulative_net_cashflow=cumulative_net_cashflow,
                )
            )
            events.extend(day_events)

        all_accounts = (*active_accounts, *closed_accounts)
        summary = _build_portfolio_summary(
            snapshots=snapshots,
            accounts=list(all_accounts),
            account_definition=account_definition,
        )
        events.append(
            PortfolioEvent(
                trading_day=normalized_returns.index[-1],
                event_type=PortfolioEventType.SIMULATION_COMPLETED,
                message=(
                    "Completed FundedNext portfolio simulation for "
                    f"{account_definition.account_code}"
                ),
                amount=summary.net_cashflow,
            )
        )
        return PortfolioSimulationResult(
            daily_timeline=pd.DataFrame(snapshot.to_record() for snapshot in snapshots),
            events=pd.DataFrame(event.to_record() for event in events),
            account_summaries=pd.DataFrame(account.to_record() for account in all_accounts),
            summary=summary,
            realized_returns=normalized_returns,
            account_definition=account_definition,
        )

    def _get_account_definition(self, account_code: str) -> AccountDefinition:
        if account_code not in self._account_definitions:
            raise KeyError(
                f"Unknown FundedNext account '{account_code}'. "
                f"Available accounts: {sorted(self._account_definitions)}"
            )
        return self._account_definitions[account_code]


def _open_challenge_account(
    account_id: str,
    trading_day: pd.Timestamp,
    account_definition: AccountDefinition,
) -> PortfolioAccountState:
    initial_balance = account_definition.account_size
    rules = account_definition.evaluation
    return PortfolioAccountState(
        account_id=account_id,
        account_code=account_definition.account_code,
        lifecycle_state=AccountLifecycleState.CHALLENGE_ACTIVE,
        phase=AccountPhase.EVALUATION,
        start_day=trading_day,
        balance=initial_balance,
        closing_high_balance=initial_balance,
        max_loss_floor=compute_static_max_loss_floor(
            account_definition.account_size,
            rules,
        ),
        largest_daily_profit=0.0,
        evaluation_trading_days=0,
        payout_cycle_profit=0.0,
        payout_cycle_largest_profit=0.0,
        payout_profit_days=0,
        payout_count=0,
        total_gross_payouts=0.0,
        total_trader_payouts=0.0,
        challenge_fee_paid=account_definition.fees.challenge_fee,
        activation_fee_paid=0.0,
        reset_fee_paid=0.0,
        scale_mini_limit=rules.max_contract_limit.mini_contracts,
        scale_micro_limit=rules.max_contract_limit.micro_contracts,
    )


def _advance_challenge_account(
    account: PortfolioAccountState,
    trading_day: pd.Timestamp,
    daily_return: float,
    account_definition: AccountDefinition,
) -> tuple[PortfolioAccountState, tuple[PortfolioEvent, ...], bool, bool]:
    rules = _resolve_challenge_rules(account_definition, account.phase)
    raw_balance = account.balance * (1.0 + daily_return)
    balance, _ = (
        (raw_balance, False)
        if raw_balance <= account.max_loss_floor
        else _apply_daily_loss_limit(
            starting_balance=account.balance,
            raw_balance=raw_balance,
            daily_loss_limit=rules.daily_loss_limit,
        )
    )
    daily_profit = balance - account.balance
    largest_daily_profit = max(account.largest_daily_profit, max(daily_profit, 0.0))
    trading_days = account.evaluation_trading_days + 1

    if balance <= account.max_loss_floor:
        failed_account = replace(
            account,
            lifecycle_state=AccountLifecycleState.CHALLENGE_FAILED,
            balance=balance,
            closing_high_balance=max(account.closing_high_balance, balance),
            max_loss_floor=compute_static_max_loss_floor(
                account_definition.account_size,
                rules,
            ),
            largest_daily_profit=largest_daily_profit,
            evaluation_trading_days=trading_days,
            close_day=trading_day,
            breach_reason=BreachReason.MAX_LOSS_LIMIT,
        )
        event = PortfolioEvent(
            trading_day=trading_day,
            event_type=PortfolioEventType.CHALLENGE_FAILED,
            account_id=account.account_id,
            lifecycle_state=AccountLifecycleState.CHALLENGE_FAILED,
            message=f"Challenge {account.account_id} failed at static max-loss floor",
        )
        return failed_account, (event,), False, True

    total_profit = balance - account_definition.account_size
    passed = (
        total_profit >= rules.profit_target_amount
        and trading_days >= rules.minimum_trading_days
    )
    if not passed:
        active_account = replace(
            account,
            balance=balance,
            closing_high_balance=max(account.closing_high_balance, balance),
            max_loss_floor=compute_static_max_loss_floor(
                account_definition.account_size,
                rules,
            ),
            largest_daily_profit=largest_daily_profit,
            evaluation_trading_days=trading_days,
        )
        return active_account, tuple(), False, False

    if account.phase == AccountPhase.EVALUATION:
        verification_rules = account_definition.verification
        if verification_rules is None:
            raise ValueError("verification rules are required for Stellar 2-Step")
        opening_balance = account_definition.account_size
        verification_account = replace(
            account,
            phase=AccountPhase.VERIFICATION,
            balance=opening_balance,
            closing_high_balance=opening_balance,
            max_loss_floor=compute_static_max_loss_floor(
                account_definition.account_size,
                verification_rules,
            ),
            largest_daily_profit=0.0,
            evaluation_trading_days=0,
            scale_mini_limit=verification_rules.max_contract_limit.mini_contracts,
            scale_micro_limit=verification_rules.max_contract_limit.micro_contracts,
        )
        event = PortfolioEvent(
            trading_day=trading_day,
            event_type=PortfolioEventType.CHALLENGE_PASSED,
            account_id=account.account_id,
            lifecycle_state=AccountLifecycleState.CHALLENGE_ACTIVE,
            message=f"Challenge {account.account_id} passed phase 1",
        )
        return verification_account, (event,), False, False

    opening_funded_balance = account_definition.account_size
    funded_rules = account_definition.funded
    funded_account = replace(
        account,
        lifecycle_state=AccountLifecycleState.FUNDED_ACTIVE,
        phase=AccountPhase.FUNDED,
        balance=opening_funded_balance,
        closing_high_balance=opening_funded_balance,
        max_loss_floor=compute_static_max_loss_floor(
            account_definition.account_size,
            funded_rules,
        ),
        evaluation_trading_days=trading_days,
        payout_cycle_profit=0.0,
        payout_cycle_largest_profit=0.0,
        payout_profit_days=0,
        pass_day=trading_day,
        funded_start_day=trading_day,
        payout_cycle_start_day=trading_day,
        previous_cycle_had_payout=False,
        scale_mini_limit=funded_rules.max_contract_limit.mini_contracts,
        scale_micro_limit=funded_rules.max_contract_limit.micro_contracts,
    )
    return funded_account, tuple(), True, False


def _build_challenge_pass_events(
    account_id: str,
    trading_day: pd.Timestamp,
) -> tuple[PortfolioEvent, PortfolioEvent]:
    return (
        PortfolioEvent(
            trading_day=trading_day,
            event_type=PortfolioEventType.CHALLENGE_PASSED,
            account_id=account_id,
            lifecycle_state=AccountLifecycleState.FUNDED_ACTIVE,
            message=f"Challenge {account_id} passed both phases",
        ),
        PortfolioEvent(
            trading_day=trading_day,
            event_type=PortfolioEventType.FUNDED_STARTED,
            account_id=account_id,
            lifecycle_state=AccountLifecycleState.FUNDED_ACTIVE,
            message=f"FundedNext funded account {account_id} activated",
            amount=0.0,
        ),
    )


def _discard_passed_challenge(
    account: PortfolioAccountState,
    trading_day: pd.Timestamp,
) -> PortfolioAccountState:
    return replace(
        account,
        lifecycle_state=AccountLifecycleState.CHALLENGE_DISCARDED,
        phase=AccountPhase.EVALUATION,
        pass_day=None,
        funded_start_day=None,
        payout_cycle_start_day=None,
        close_day=trading_day,
        breach_reason=None,
    )


def _advance_funded_account(
    account: PortfolioAccountState,
    trading_day: pd.Timestamp,
    daily_return: float,
    payout_policy: PortfolioPayoutPolicyConfig,
    account_definition: AccountDefinition,
    max_payouts: int,
) -> tuple[PortfolioAccountState, tuple[PortfolioEvent, ...], float, float, float, bool]:
    rules = account_definition.funded
    raw_balance = account.balance * (1.0 + daily_return)
    balance, _ = (
        (raw_balance, False)
        if raw_balance <= account.max_loss_floor
        else _apply_daily_loss_limit(
            starting_balance=account.balance,
            raw_balance=raw_balance,
            daily_loss_limit=rules.daily_loss_limit,
        )
    )
    daily_profit = balance - account.balance
    payout_cycle_profit = account.payout_cycle_profit + daily_profit
    payout_cycle_largest_profit = max(
        account.payout_cycle_largest_profit,
        max(daily_profit, 0.0),
    )
    payout_profit_days = account.payout_profit_days + int(
        daily_profit >= rules.payout_rule.profit_day_rule.minimum_profit_amount
    )

    if balance <= account.max_loss_floor:
        closed_account = replace(
            account,
            lifecycle_state=AccountLifecycleState.FUNDED_CLOSED,
            balance=balance,
            closing_high_balance=max(account.closing_high_balance, balance),
            max_loss_floor=compute_static_max_loss_floor(
                account_definition.account_size,
                rules,
            ),
            payout_cycle_profit=payout_cycle_profit,
            payout_cycle_largest_profit=payout_cycle_largest_profit,
            payout_profit_days=payout_profit_days,
            close_day=trading_day,
            breach_reason=BreachReason.MAX_LOSS_LIMIT,
        )
        event = PortfolioEvent(
            trading_day=trading_day,
            event_type=PortfolioEventType.FUNDED_CLOSED,
            account_id=account.account_id,
            lifecycle_state=AccountLifecycleState.FUNDED_CLOSED,
            message=f"Funded account {account.account_id} closed from drawdown breach",
        )
        return closed_account, (event,), 0.0, 0.0, 0.0, True

    requested_amount = _resolve_payout_amount(
        account=account,
        trading_day=trading_day,
        payout_policy=payout_policy,
        account_balance=balance,
        payout_cycle_profit=payout_cycle_profit,
        account_definition=account_definition,
    )
    if requested_amount is None:
        active_account = replace(
            account,
            balance=balance,
            closing_high_balance=max(account.closing_high_balance, balance),
            max_loss_floor=compute_static_max_loss_floor(
                account_definition.account_size,
                rules,
            ),
            payout_cycle_profit=payout_cycle_profit,
            payout_cycle_largest_profit=payout_cycle_largest_profit,
            payout_profit_days=payout_profit_days,
        )
        return active_account, tuple(), 0.0, 0.0, 0.0, False

    ladder_decision = (
        resolve_cfd_ladder_payout(
            account_size=account_definition.account_size,
            account_balance=balance,
            payout_policy=payout_policy,
            milestones_already_taken=account.payout_milestones_taken,
        )
        if payout_policy.mode == PortfolioPayoutPolicyMode.CFD_LADDER
        else None
    )
    trader_amount = requested_amount * rules.payout_rule.trader_profit_split
    fee_refund = (
        account_definition.fees.challenge_fee
        if account_definition.fees.refundable_on_first_payout
        and not account.fee_refund_received
        else 0.0
    )
    payout_count = account.payout_count + 1
    post_payout_balance = balance - requested_amount
    lifecycle_state = (
        AccountLifecycleState.FUNDED_CLOSED
        if payout_count >= max_payouts
        else AccountLifecycleState.FUNDED_ACTIVE
    )
    close_day = trading_day if lifecycle_state == AccountLifecycleState.FUNDED_CLOSED else None
    updated_account = replace(
        account,
        lifecycle_state=lifecycle_state,
        balance=post_payout_balance,
        closing_high_balance=max(account.closing_high_balance, post_payout_balance),
        max_loss_floor=compute_static_max_loss_floor(
            account_definition.account_size,
            rules,
        ),
        payout_cycle_profit=0.0,
        payout_cycle_largest_profit=0.0,
        payout_profit_days=0,
        payout_count=payout_count,
        total_gross_payouts=account.total_gross_payouts + requested_amount,
        total_trader_payouts=account.total_trader_payouts + trader_amount,
        last_payout_day=trading_day,
        payout_cycle_start_day=trading_day,
        previous_cycle_had_payout=True,
        payout_milestones_taken=(
            ladder_decision.milestones_taken
            if ladder_decision is not None
            else account.payout_milestones_taken
        ),
        fee_refund_received=account.fee_refund_received or fee_refund > 0.0,
        close_day=close_day,
        breach_reason=(
            BreachReason.PAYOUT_LIMIT
            if lifecycle_state == AccountLifecycleState.FUNDED_CLOSED
            else None
        ),
    )
    payout_event = PortfolioEvent(
        trading_day=trading_day,
        event_type=PortfolioEventType.PAYOUT_REQUESTED,
        account_id=account.account_id,
        lifecycle_state=lifecycle_state,
        message=f"Account {account.account_id} requested payout of {requested_amount:.2f}",
        amount=requested_amount,
    )
    events: tuple[PortfolioEvent, ...] = (payout_event,)
    if fee_refund > 0.0:
        events = (
            payout_event,
            PortfolioEvent(
                trading_day=trading_day,
                event_type=PortfolioEventType.FEE_REFUNDED,
                account_id=account.account_id,
                lifecycle_state=lifecycle_state,
                message=f"Refundable challenge fee of {fee_refund:.2f}",
                amount=fee_refund,
            ),
        )
    if lifecycle_state == AccountLifecycleState.FUNDED_CLOSED:
        events = (
            *events,
            PortfolioEvent(
                trading_day=trading_day,
                event_type=PortfolioEventType.FUNDED_CLOSED,
                account_id=account.account_id,
                lifecycle_state=AccountLifecycleState.FUNDED_CLOSED,
                message=f"Funded account {account.account_id} closed after max payouts",
            ),
        )
    return updated_account, events, requested_amount, trader_amount, fee_refund, (
        lifecycle_state == AccountLifecycleState.FUNDED_CLOSED
    )


def _resolve_payout_amount(
    account: PortfolioAccountState,
    trading_day: pd.Timestamp,
    payout_policy: PortfolioPayoutPolicyConfig,
    account_balance: float,
    payout_cycle_profit: float,
    account_definition: AccountDefinition,
) -> float | None:
    payout_rule = account_definition.funded.payout_rule
    if payout_policy.mode == PortfolioPayoutPolicyMode.CFD_LADDER:
        ladder_decision = resolve_cfd_ladder_payout(
            account_size=account_definition.account_size,
            account_balance=account_balance,
            payout_policy=payout_policy,
            milestones_already_taken=account.payout_milestones_taken,
        )
        if ladder_decision is None:
            return None
        return min(
            ladder_decision.gross_amount,
            payout_rule.payout_cap_for_request(account.payout_count + 1),
            max(account_balance - account_definition.account_size, 0.0),
        )

    if payout_cycle_profit <= 0.0:
        return None
    if not _is_calendar_payout_eligible(
        account=account,
        trading_day=trading_day,
        payout_cycle_rule=payout_rule.payout_cycle_rule,
    ):
        return None

    gross_cap = min(
        payout_cycle_profit * payout_rule.max_profit_share,
        payout_rule.payout_cap_for_request(account.payout_count + 1),
        max(account_balance - account_definition.account_size, 0.0),
    )
    if gross_cap < payout_rule.min_request_amount:
        return None

    match payout_policy.mode:
        case PortfolioPayoutPolicyMode.AGGRESSIVE:
            candidate = gross_cap
        case PortfolioPayoutPolicyMode.BUFFER:
            candidate = min(
                gross_cap,
                max(
                    account_balance
                    - (account_definition.account_size + payout_policy.buffer_amount),
                    0.0,
                ),
            )
        case PortfolioPayoutPolicyMode.FRACTIONAL:
            candidate = gross_cap * payout_policy.withdrawal_fraction
        case _:
            raise ValueError(f"Unsupported payout policy mode: {payout_policy.mode}")

    if candidate < payout_rule.min_request_amount:
        return None
    return candidate


def _is_calendar_payout_eligible(
    account: PortfolioAccountState,
    trading_day: pd.Timestamp,
    payout_cycle_rule: PayoutCycleRule | None,
) -> bool:
    if payout_cycle_rule is None:
        return True
    cycle_start = account.payout_cycle_start_day or account.funded_start_day
    if cycle_start is None:
        return False
    required_days = (
        payout_cycle_rule.first_cycle_calendar_days
        if account.payout_count == 0 or not account.previous_cycle_had_payout
        else payout_cycle_rule.subsequent_cycle_calendar_days
    )
    elapsed_days = (trading_day.normalize() - cycle_start.normalize()).days
    return elapsed_days >= required_days


def _resolve_challenge_rules(
    account_definition: AccountDefinition,
    phase: AccountPhase,
) -> EvaluationRules:
    if phase == AccountPhase.EVALUATION:
        return account_definition.evaluation
    if phase == AccountPhase.VERIFICATION:
        verification_rules = account_definition.verification
        if verification_rules is None:
            raise ValueError("verification rules are required for phase 2")
        return verification_rules
    raise ValueError(f"Unsupported challenge phase: {phase.value}")


def _is_month_start(
    trading_day: pd.Timestamp,
    index: pd.DatetimeIndex,
    idx: int,
) -> bool:
    if idx == 0:
        return True
    previous_day = index[idx - 1]
    return (trading_day.year, trading_day.month) != (
        previous_day.year,
        previous_day.month,
    )


def _build_portfolio_summary(
    snapshots: list[PortfolioDailySnapshot],
    accounts: list[PortfolioAccountState],
    account_definition: AccountDefinition,
) -> PortfolioSimulationSummary:
    first_payout_day = next(
        (
            snapshot.trading_day
            for snapshot in snapshots
            if snapshot.trader_payouts > 0.0
        ),
        None,
    )
    start_day = snapshots[0].trading_day if snapshots else None
    funded_accounts = [account for account in accounts if account.pass_day is not None]
    funded_closed_accounts = [
        account
        for account in accounts
        if account.lifecycle_state == AccountLifecycleState.FUNDED_CLOSED
    ]
    funded_closed_from_drawdown = [
        account
        for account in funded_closed_accounts
        if account.breach_reason == BreachReason.MAX_LOSS_LIMIT
    ]
    funded_closed_from_max_payouts = [
        account
        for account in funded_closed_accounts
        if account.breach_reason == BreachReason.PAYOUT_LIMIT
    ]
    challenge_failed_accounts = [
        account
        for account in accounts
        if account.lifecycle_state == AccountLifecycleState.CHALLENGE_FAILED
    ]
    total_challenge_costs = sum(account.challenge_fee_paid for account in accounts)
    total_activation_costs = sum(account.activation_fee_paid for account in accounts)
    total_reset_costs = sum(account.reset_fee_paid for account in accounts)
    total_fee_refunds = sum(
        account_definition.fees.challenge_fee
        for account in accounts
        if account.fee_refund_received
    )
    total_gross_payouts = sum(account.total_gross_payouts for account in accounts)
    total_trader_payouts = sum(account.total_trader_payouts for account in accounts)
    net_cashflow = (
        total_trader_payouts
        + total_fee_refunds
        - total_challenge_costs
        - total_activation_costs
        - total_reset_costs
    )
    funded_account_count = len(funded_accounts)
    return PortfolioSimulationSummary(
        account_code=account_definition.account_code,
        total_days=len(snapshots),
        challenges_purchased=len(accounts),
        challenges_failed=len(challenge_failed_accounts),
        funded_accounts_created=funded_account_count,
        funded_accounts_closed=len(funded_closed_accounts),
        total_gross_payouts=total_gross_payouts,
        total_trader_payouts=total_trader_payouts,
        total_challenge_costs=total_challenge_costs,
        total_activation_costs=total_activation_costs,
        total_reset_costs=total_reset_costs,
        total_fee_refunds=total_fee_refunds,
        net_cashflow=net_cashflow,
        first_payout_day=first_payout_day,
        days_to_first_payout=(
            None
            if first_payout_day is None or start_day is None
            else int((first_payout_day - start_day).days)
        ),
        average_active_funded_accounts=(
            0.0
            if not snapshots
            else sum(snapshot.active_funded for snapshot in snapshots) / len(snapshots)
        ),
        average_active_challenges=(
            0.0
            if not snapshots
            else sum(snapshot.active_challenges for snapshot in snapshots) / len(snapshots)
        ),
        payouts_per_funded_account=(
            0.0
            if funded_account_count == 0
            else sum(account.payout_count for account in funded_accounts)
            / funded_account_count
        ),
        negative_net_cashflow_probability=1.0 if net_cashflow < 0.0 else 0.0,
        funded_closed_from_drawdown=len(funded_closed_from_drawdown),
        funded_closed_from_max_payouts=len(funded_closed_from_max_payouts),
    )
