from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from prop_firms.base.enums import (
    AccountPhase,
    AccountStatus,
    BreachReason,
    EventType,
    PayoutPolicy,
    ResetPolicy,
)
from prop_firms.base.exposure import ExposureCheck, evaluate_exposure
from prop_firms.base.models import (
    AccountDefinition,
    ContractLimit,
    DailySnapshot,
    DayInput,
    EvaluationRules,
    PayoutCycleRule,
    SimulationEvent,
    SimulationRequest,
    SimulationSummary,
)
from prop_firms.base.simulator import BasePropFirmEngine, DayProcessingResult
from prop_firms.fundednext.drawdown import compute_static_max_loss_floor


@dataclass(frozen=True)
class FundedNextState:
    """Provider-specific account state for FundedNext Stellar 2-Step CFD accounts."""

    phase: AccountPhase
    status: AccountStatus
    balance: float
    closing_high_balance: float
    max_loss_floor: float
    current_daily_loss_limit: float | None
    daily_loss_limit_hits: int
    largest_daily_profit: float
    payout_cycle_profit: float
    payout_cycle_largest_profit: float
    payout_profit_days: int
    payout_count: int
    total_payout_amount: float
    fee_refund_amount: float
    challenge_fee_paid: float
    reset_fee_paid: float
    resets_used: int
    evaluation_start_day: pd.Timestamp
    evaluation_trading_days: int
    evaluation_pass_day: pd.Timestamp | None
    funded_start_day: pd.Timestamp | None
    payout_cycle_start_day: pd.Timestamp | None
    previous_cycle_had_payout: bool
    fee_refund_received: bool
    breach_reason: BreachReason | None
    scale_contract_limit: ContractLimit


class FundedNextRuleEngine(BasePropFirmEngine[FundedNextState]):
    """FundedNext Stellar 2-Step CFD implementation."""

    def __init__(self, account_definitions: dict[str, AccountDefinition]) -> None:
        self._account_definitions = account_definitions

    def get_account_definition(self, account_code: str) -> AccountDefinition:
        if account_code not in self._account_definitions:
            raise KeyError(
                f"Unknown FundedNext account '{account_code}'. "
                f"Available accounts: {sorted(self._account_definitions)}"
            )
        return self._account_definitions[account_code]

    def build_initial_state(
        self,
        request: SimulationRequest,
        account_definition: AccountDefinition,
    ) -> FundedNextState:
        initial_balance = (
            account_definition.account_size
            if request.starting_balance_override is None
            else request.starting_balance_override
        )
        evaluation_start_day = _normalize_timestamp(request.returns.index.min())
        rules = account_definition.evaluation
        return FundedNextState(
            phase=AccountPhase.EVALUATION,
            status=AccountStatus.ACTIVE,
            balance=initial_balance,
            closing_high_balance=initial_balance,
            max_loss_floor=compute_static_max_loss_floor(
                account_definition.account_size,
                rules,
            ),
            current_daily_loss_limit=rules.daily_loss_limit,
            daily_loss_limit_hits=0,
            largest_daily_profit=0.0,
            payout_cycle_profit=0.0,
            payout_cycle_largest_profit=0.0,
            payout_profit_days=0,
            payout_count=0,
            total_payout_amount=0.0,
            fee_refund_amount=0.0,
            challenge_fee_paid=account_definition.fees.challenge_fee,
            reset_fee_paid=0.0,
            resets_used=0,
            evaluation_start_day=evaluation_start_day,
            evaluation_trading_days=0,
            evaluation_pass_day=None,
            funded_start_day=None,
            payout_cycle_start_day=None,
            previous_cycle_had_payout=False,
            fee_refund_received=False,
            breach_reason=None,
            scale_contract_limit=rules.max_contract_limit,
        )

    def process_day(
        self,
        state: FundedNextState,
        day_input: DayInput,
        request: SimulationRequest,
        account_definition: AccountDefinition,
    ) -> DayProcessingResult[FundedNextState]:
        if state.phase == AccountPhase.FUNDED:
            return self._process_funded_day(
                state=state,
                day_input=day_input,
                request=request,
                account_definition=account_definition,
            )
        return self._process_challenge_day(
            state=state,
            day_input=day_input,
            account_definition=account_definition,
        )

    def can_reset(
        self,
        state: FundedNextState,
        request: SimulationRequest,
        account_definition: AccountDefinition,
    ) -> bool:
        del account_definition
        return (
            request.reset_policy == ResetPolicy.AUTO_RESET
            and state.phase != AccountPhase.FUNDED
            and state.status == AccountStatus.BREACHED
            and state.resets_used < request.max_resets
        )

    def apply_reset(
        self,
        state: FundedNextState,
        request: SimulationRequest,
        trading_day: pd.Timestamp,
        account_definition: AccountDefinition,
    ) -> tuple[FundedNextState, SimulationEvent]:
        del request
        rules = account_definition.evaluation
        opening_balance = account_definition.account_size
        reset_state = FundedNextState(
            phase=AccountPhase.EVALUATION,
            status=AccountStatus.ACTIVE,
            balance=opening_balance,
            closing_high_balance=opening_balance,
            max_loss_floor=compute_static_max_loss_floor(
                account_definition.account_size,
                rules,
            ),
            current_daily_loss_limit=rules.daily_loss_limit,
            daily_loss_limit_hits=state.daily_loss_limit_hits,
            largest_daily_profit=0.0,
            payout_cycle_profit=0.0,
            payout_cycle_largest_profit=0.0,
            payout_profit_days=0,
            payout_count=0,
            total_payout_amount=state.total_payout_amount,
            fee_refund_amount=state.fee_refund_amount,
            challenge_fee_paid=state.challenge_fee_paid,
            reset_fee_paid=state.reset_fee_paid + account_definition.fees.reset_fee,
            resets_used=state.resets_used + 1,
            evaluation_start_day=trading_day,
            evaluation_trading_days=0,
            evaluation_pass_day=None,
            funded_start_day=None,
            payout_cycle_start_day=None,
            previous_cycle_had_payout=False,
            fee_refund_received=state.fee_refund_received,
            breach_reason=None,
            scale_contract_limit=rules.max_contract_limit,
        )
        return (
            reset_state,
            SimulationEvent(
                trading_day=trading_day,
                event_type=EventType.RESET_APPLIED,
                phase=AccountPhase.EVALUATION,
                message=(
                    f"Reset FundedNext challenge {account_definition.account_code} "
                    f"to phase 1"
                ),
                amount=account_definition.fees.reset_fee,
            ),
        )

    def build_summary(
        self,
        state: FundedNextState,
        request: SimulationRequest,
        account_definition: AccountDefinition,
    ) -> SimulationSummary:
        del request
        gross_profit = (
            (state.balance - account_definition.account_size)
            if state.phase != AccountPhase.FUNDED
            else (state.balance - account_definition.account_size)
            + state.total_payout_amount
        )
        return SimulationSummary(
            provider_name=account_definition.provider_name,
            account_code=account_definition.account_code,
            final_phase=state.phase,
            final_status=state.status,
            final_balance=state.balance,
            total_profit=gross_profit,
            challenge_fee_paid=state.challenge_fee_paid,
            reset_fee_paid=state.reset_fee_paid,
            fee_refund_amount=state.fee_refund_amount,
            total_payout_amount=state.total_payout_amount,
            payout_count=state.payout_count,
            resets_used=state.resets_used,
            evaluation_pass_day=state.evaluation_pass_day,
            breach_reason=state.breach_reason,
        )

    def _process_challenge_day(
        self,
        state: FundedNextState,
        day_input: DayInput,
        account_definition: AccountDefinition,
    ) -> DayProcessingResult[FundedNextState]:
        rules = _resolve_challenge_rules(account_definition, state.phase)
        raw_balance = state.balance * (1.0 + day_input.daily_return)
        balance, daily_loss_limit_hit = (
            (raw_balance, False)
            if raw_balance <= state.max_loss_floor
            else _apply_daily_loss_limit(
                starting_balance=state.balance,
                raw_balance=raw_balance,
                daily_loss_limit=rules.daily_loss_limit,
            )
        )
        daily_profit = balance - state.balance
        largest_daily_profit = max(state.largest_daily_profit, max(daily_profit, 0.0))
        exposure = evaluate_exposure(
            day_input=day_input,
            contract_limit=rules.max_contract_limit,
            account_balance=balance,
            max_leverage=rules.max_leverage,
        )
        trading_days = state.evaluation_trading_days + 1
        day_state = replace(
            state,
            balance=balance,
            current_daily_loss_limit=rules.daily_loss_limit,
            daily_loss_limit_hits=state.daily_loss_limit_hits + int(daily_loss_limit_hit),
            largest_daily_profit=largest_daily_profit,
            evaluation_trading_days=trading_days,
            breach_reason=None,
        )
        daily_events = _daily_loss_limit_events(
            trading_day=day_input.trading_day,
            phase=state.phase,
            daily_loss_limit_hit=daily_loss_limit_hit,
            daily_loss_limit=rules.daily_loss_limit,
        )
        maybe_breached_state = _maybe_build_breach_state(state=day_state, exposure=exposure)
        if maybe_breached_state is not None:
            snapshot = _build_snapshot(
                state=maybe_breached_state,
                day_input=day_input,
                account_definition=account_definition,
                exposure=exposure,
                daily_loss_limit_hit=daily_loss_limit_hit,
            )
            return DayProcessingResult(
                state=maybe_breached_state,
                snapshot=snapshot,
                events=daily_events
                + (
                    _breach_event(
                        trading_day=day_input.trading_day,
                        phase=state.phase,
                        breach_reason=maybe_breached_state.breach_reason,
                    ),
                ),
            )

        next_state = replace(
            day_state,
            closing_high_balance=max(state.closing_high_balance, balance),
            max_loss_floor=compute_static_max_loss_floor(
                account_definition.account_size,
                rules,
            ),
        )
        total_profit = balance - account_definition.account_size
        passed = (
            total_profit >= rules.profit_target_amount
            and trading_days >= rules.minimum_trading_days
        )
        snapshot = _build_snapshot(
            state=next_state,
            day_input=day_input,
            account_definition=account_definition,
            exposure=exposure,
            daily_loss_limit_hit=daily_loss_limit_hit,
        )
        if not passed:
            return DayProcessingResult(
                state=next_state,
                snapshot=snapshot,
                events=daily_events,
            )

        if state.phase == AccountPhase.EVALUATION:
            verification_rules = account_definition.verification
            if verification_rules is None:
                raise ValueError(
                    "FundedNext Stellar 2-Step requires verification rules in config"
                )
            verification_state = _build_verification_state(
                account_definition=account_definition,
                previous_state=next_state,
                trading_day=day_input.trading_day,
            )
            return DayProcessingResult(
                state=verification_state,
                snapshot=snapshot,
                events=daily_events
                + (
                    SimulationEvent(
                        trading_day=day_input.trading_day,
                        event_type=EventType.EVALUATION_PASSED,
                        phase=AccountPhase.EVALUATION,
                        message=(
                            f"Passed FundedNext phase 1 for "
                            f"{account_definition.account_code}"
                        ),
                        amount=balance,
                    ),
                ),
            )

        funded_state = _build_funded_state(
            account_definition=account_definition,
            previous_state=next_state,
            trading_day=day_input.trading_day,
        )
        return DayProcessingResult(
            state=funded_state,
            snapshot=snapshot,
            events=daily_events
            + (
                SimulationEvent(
                    trading_day=day_input.trading_day,
                    event_type=EventType.EVALUATION_PASSED,
                    phase=AccountPhase.VERIFICATION,
                    message=(
                        f"Passed FundedNext phase 2 for "
                        f"{account_definition.account_code}"
                    ),
                    amount=balance,
                ),
                SimulationEvent(
                    trading_day=day_input.trading_day,
                    event_type=EventType.FUNDED_STARTED,
                    phase=AccountPhase.FUNDED,
                    message=(
                        f"Started FundedNext funded account "
                        f"{account_definition.account_code} at nominal balance"
                    ),
                    amount=account_definition.account_size,
                ),
            ),
        )

    def _process_funded_day(
        self,
        state: FundedNextState,
        day_input: DayInput,
        request: SimulationRequest,
        account_definition: AccountDefinition,
    ) -> DayProcessingResult[FundedNextState]:
        rules = account_definition.funded
        raw_balance = state.balance * (1.0 + day_input.daily_return)
        balance, daily_loss_limit_hit = (
            (raw_balance, False)
            if raw_balance <= state.max_loss_floor
            else _apply_daily_loss_limit(
                starting_balance=state.balance,
                raw_balance=raw_balance,
                daily_loss_limit=rules.daily_loss_limit,
            )
        )
        daily_profit = balance - state.balance
        payout_cycle_profit = state.payout_cycle_profit + daily_profit
        payout_cycle_largest_profit = max(
            state.payout_cycle_largest_profit,
            max(daily_profit, 0.0),
        )
        payout_profit_days = state.payout_profit_days + int(
            daily_profit >= rules.payout_rule.profit_day_rule.minimum_profit_amount
        )
        exposure = evaluate_exposure(
            day_input=day_input,
            contract_limit=state.scale_contract_limit,
            account_balance=balance,
            max_leverage=rules.max_leverage,
        )
        day_state = replace(
            state,
            balance=balance,
            daily_loss_limit_hits=state.daily_loss_limit_hits + int(daily_loss_limit_hit),
            payout_cycle_profit=payout_cycle_profit,
            payout_cycle_largest_profit=payout_cycle_largest_profit,
            payout_profit_days=payout_profit_days,
            breach_reason=None,
        )
        daily_events = _daily_loss_limit_events(
            trading_day=day_input.trading_day,
            phase=AccountPhase.FUNDED,
            daily_loss_limit_hit=daily_loss_limit_hit,
            daily_loss_limit=rules.daily_loss_limit,
        )
        maybe_breached_state = _maybe_build_breach_state(state=day_state, exposure=exposure)
        if maybe_breached_state is not None:
            snapshot = _build_snapshot(
                state=maybe_breached_state,
                day_input=day_input,
                account_definition=account_definition,
                exposure=exposure,
                daily_loss_limit_hit=daily_loss_limit_hit,
            )
            return DayProcessingResult(
                state=maybe_breached_state,
                snapshot=snapshot,
                events=daily_events
                + (
                    _breach_event(
                        trading_day=day_input.trading_day,
                        phase=AccountPhase.FUNDED,
                        breach_reason=maybe_breached_state.breach_reason,
                    ),
                ),
            )

        post_payout_events, post_payout_state = _maybe_request_payout(
            state=day_state,
            request=request,
            trading_day=day_input.trading_day,
            account_definition=account_definition,
        )
        next_state = replace(
            post_payout_state,
            closing_high_balance=max(state.closing_high_balance, post_payout_state.balance),
            max_loss_floor=compute_static_max_loss_floor(
                account_definition.account_size,
                rules,
            ),
            scale_contract_limit=rules.max_contract_limit,
        )
        snapshot = _build_snapshot(
            state=next_state,
            day_input=day_input,
            account_definition=account_definition,
            exposure=exposure,
            daily_loss_limit_hit=daily_loss_limit_hit,
        )
        return DayProcessingResult(
            state=next_state,
            snapshot=snapshot,
            events=daily_events + post_payout_events,
        )


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


def _build_verification_state(
    account_definition: AccountDefinition,
    previous_state: FundedNextState,
    trading_day: pd.Timestamp,
) -> FundedNextState:
    verification_rules = account_definition.verification
    if verification_rules is None:
        raise ValueError("verification rules are required for phase 2")
    opening_balance = account_definition.account_size
    return FundedNextState(
        phase=AccountPhase.VERIFICATION,
        status=AccountStatus.ACTIVE,
        balance=opening_balance,
        closing_high_balance=opening_balance,
        max_loss_floor=compute_static_max_loss_floor(
            account_definition.account_size,
            verification_rules,
        ),
        current_daily_loss_limit=verification_rules.daily_loss_limit,
        daily_loss_limit_hits=previous_state.daily_loss_limit_hits,
        largest_daily_profit=0.0,
        payout_cycle_profit=0.0,
        payout_cycle_largest_profit=0.0,
        payout_profit_days=0,
        payout_count=0,
        total_payout_amount=previous_state.total_payout_amount,
        fee_refund_amount=previous_state.fee_refund_amount,
        challenge_fee_paid=previous_state.challenge_fee_paid,
        reset_fee_paid=previous_state.reset_fee_paid,
        resets_used=previous_state.resets_used,
        evaluation_start_day=previous_state.evaluation_start_day,
        evaluation_trading_days=0,
        evaluation_pass_day=None,
        funded_start_day=None,
        payout_cycle_start_day=None,
        previous_cycle_had_payout=False,
        fee_refund_received=previous_state.fee_refund_received,
        breach_reason=None,
        scale_contract_limit=verification_rules.max_contract_limit,
    )


def _build_funded_state(
    account_definition: AccountDefinition,
    previous_state: FundedNextState,
    trading_day: pd.Timestamp,
) -> FundedNextState:
    opening_balance = account_definition.account_size
    funded_rules = account_definition.funded
    return FundedNextState(
        phase=AccountPhase.FUNDED,
        status=AccountStatus.ACTIVE,
        balance=opening_balance,
        closing_high_balance=opening_balance,
        max_loss_floor=compute_static_max_loss_floor(
            account_definition.account_size,
            funded_rules,
        ),
        current_daily_loss_limit=funded_rules.daily_loss_limit,
        daily_loss_limit_hits=previous_state.daily_loss_limit_hits,
        largest_daily_profit=previous_state.largest_daily_profit,
        payout_cycle_profit=0.0,
        payout_cycle_largest_profit=0.0,
        payout_profit_days=0,
        payout_count=0,
        total_payout_amount=previous_state.total_payout_amount,
        fee_refund_amount=previous_state.fee_refund_amount,
        challenge_fee_paid=previous_state.challenge_fee_paid,
        reset_fee_paid=previous_state.reset_fee_paid,
        resets_used=previous_state.resets_used,
        evaluation_start_day=previous_state.evaluation_start_day,
        evaluation_trading_days=previous_state.evaluation_trading_days,
        evaluation_pass_day=trading_day,
        funded_start_day=trading_day,
        payout_cycle_start_day=trading_day,
        previous_cycle_had_payout=False,
        fee_refund_received=previous_state.fee_refund_received,
        breach_reason=None,
        scale_contract_limit=funded_rules.max_contract_limit,
    )


def _maybe_request_payout(
    state: FundedNextState,
    request: SimulationRequest,
    trading_day: pd.Timestamp,
    account_definition: AccountDefinition,
) -> tuple[tuple[SimulationEvent, ...], FundedNextState]:
    payout_rule = account_definition.funded.payout_rule
    if request.payout_policy == PayoutPolicy.NO_REQUESTS:
        return tuple(), state
    if state.payout_count >= payout_rule.max_requests_per_account:
        return tuple(), state
    if not _is_calendar_payout_eligible(
        state=state,
        trading_day=trading_day,
        payout_cycle_rule=payout_rule.payout_cycle_rule,
    ):
        return tuple(), state
    if state.payout_cycle_profit <= 0.0:
        return tuple(), state

    gross_payout_cap = min(
        state.payout_cycle_profit * payout_rule.max_profit_share,
        payout_rule.payout_cap_for_request(state.payout_count + 1),
        max(state.balance - account_definition.account_size, 0.0),
    )
    if gross_payout_cap < payout_rule.min_request_amount:
        return tuple(), state

    requested_amount = _resolve_requested_payout_amount(
        request=request,
        payout_cap=gross_payout_cap,
        minimum_amount=payout_rule.min_request_amount,
    )
    if requested_amount is None:
        return tuple(), state

    fee_refund = (
        account_definition.fees.challenge_fee
        if account_definition.fees.refundable_on_first_payout
        and not state.fee_refund_received
        else 0.0
    )
    next_balance = state.balance - requested_amount
    payout_count = state.payout_count + 1
    next_state = replace(
        state,
        balance=next_balance,
        payout_cycle_profit=0.0,
        payout_cycle_largest_profit=0.0,
        payout_profit_days=0,
        payout_count=payout_count,
        total_payout_amount=state.total_payout_amount + requested_amount,
        fee_refund_amount=state.fee_refund_amount + fee_refund,
        payout_cycle_start_day=trading_day,
        previous_cycle_had_payout=True,
        fee_refund_received=state.fee_refund_received or fee_refund > 0.0,
    )
    events: list[SimulationEvent] = [
        SimulationEvent(
            trading_day=trading_day,
            event_type=EventType.PAYOUT_ELIGIBLE,
            phase=AccountPhase.FUNDED,
            message="Account is eligible for performance reward",
            amount=gross_payout_cap,
        ),
        SimulationEvent(
            trading_day=trading_day,
            event_type=EventType.PAYOUT_REQUESTED,
            phase=AccountPhase.FUNDED,
            message=f"Requested performance reward of {requested_amount:.2f}",
            amount=requested_amount,
        ),
    ]
    if fee_refund > 0.0:
        events.append(
            SimulationEvent(
                trading_day=trading_day,
                event_type=EventType.FEE_REFUNDED,
                phase=AccountPhase.FUNDED,
                message=f"Refundable challenge fee of {fee_refund:.2f}",
                amount=fee_refund,
            )
        )
    return tuple(events), next_state


def _is_calendar_payout_eligible(
    state: FundedNextState,
    trading_day: pd.Timestamp,
    payout_cycle_rule: PayoutCycleRule | None,
) -> bool:
    if payout_cycle_rule is None:
        return True
    cycle_start = state.payout_cycle_start_day or state.funded_start_day
    if cycle_start is None:
        return False
    required_days = (
        payout_cycle_rule.first_cycle_calendar_days
        if state.payout_count == 0 or not state.previous_cycle_had_payout
        else payout_cycle_rule.subsequent_cycle_calendar_days
    )
    elapsed_days = (trading_day.normalize() - cycle_start.normalize()).days
    return elapsed_days >= required_days


def _apply_daily_loss_limit(
    starting_balance: float,
    raw_balance: float,
    daily_loss_limit: float | None,
) -> tuple[float, bool]:
    if daily_loss_limit is None:
        return raw_balance, False
    session_floor = starting_balance - daily_loss_limit
    return max(raw_balance, session_floor), raw_balance < session_floor


def _daily_loss_limit_events(
    trading_day: pd.Timestamp,
    phase: AccountPhase,
    daily_loss_limit_hit: bool,
    daily_loss_limit: float | None,
) -> tuple[SimulationEvent, ...]:
    if not daily_loss_limit_hit or daily_loss_limit is None:
        return tuple()
    return (
        SimulationEvent(
            trading_day=trading_day,
            event_type=EventType.DAILY_LOSS_LIMIT_HIT,
            phase=phase,
            message=(
                "Daily loss limit triggered; session stopped at "
                f"{daily_loss_limit:.2f}"
            ),
            amount=daily_loss_limit,
        ),
    )


def _maybe_build_breach_state(
    state: FundedNextState,
    exposure: ExposureCheck,
) -> FundedNextState | None:
    if exposure.contract_limit_breached:
        return replace(
            state,
            status=AccountStatus.BREACHED,
            breach_reason=BreachReason.CONTRACT_LIMIT,
        )
    if exposure.leverage_limit_breached:
        return replace(
            state,
            status=AccountStatus.BREACHED,
            breach_reason=BreachReason.LEVERAGE_LIMIT,
        )
    if state.balance <= state.max_loss_floor:
        return replace(
            state,
            status=AccountStatus.BREACHED,
            breach_reason=BreachReason.MAX_LOSS_LIMIT,
        )
    return None


def _build_snapshot(
    state: FundedNextState,
    day_input: DayInput,
    account_definition: AccountDefinition,
    exposure: ExposureCheck,
    daily_loss_limit_hit: bool = False,
) -> DailySnapshot:
    active_rules = (
        account_definition.funded
        if state.phase == AccountPhase.FUNDED
        else _resolve_challenge_rules(account_definition, state.phase)
    )
    profit_target_balance = (
        account_definition.account_size + active_rules.profit_target_amount
        if state.phase != AccountPhase.FUNDED
        else account_definition.account_size
    )
    return DailySnapshot(
        trading_day=day_input.trading_day,
        phase=state.phase,
        status=state.status,
        balance=state.balance,
        net_profit=state.balance - account_definition.account_size,
        daily_return=day_input.daily_return,
        closing_high_balance=state.closing_high_balance,
        max_loss_floor=state.max_loss_floor,
        profit_target_balance=profit_target_balance,
        largest_daily_profit=state.largest_daily_profit,
        consistency_ratio=None,
        consistency_cap_ratio=None,
        payout_cycle_profit=state.payout_cycle_profit,
        payout_cycle_largest_profit=state.payout_cycle_largest_profit,
        payout_profit_days=state.payout_profit_days,
        payouts_taken=state.payout_count,
        daily_loss_limit=state.current_daily_loss_limit,
        daily_loss_limit_hit=int(daily_loss_limit_hit),
        scale_mini_limit=state.scale_contract_limit.mini_contracts,
        scale_micro_limit=state.scale_contract_limit.micro_contracts,
        mini_contracts=exposure.mini_contracts,
        micro_contracts=exposure.micro_contracts,
        mini_equivalent_contracts=exposure.mini_equivalent_contracts,
        leverage=exposure.leverage,
        contract_limit_breached=int(exposure.contract_limit_breached),
        leverage_limit_breached=int(exposure.leverage_limit_breached),
    )


def _breach_event(
    trading_day: pd.Timestamp,
    phase: AccountPhase,
    breach_reason: BreachReason | None,
) -> SimulationEvent:
    resolved_reason = (
        BreachReason.DATA_VALIDATION if breach_reason is None else breach_reason
    )
    return SimulationEvent(
        trading_day=trading_day,
        event_type=EventType.BREACH,
        phase=phase,
        message=f"Breach due to {resolved_reason.value}",
        breach_reason=resolved_reason,
    )


def _resolve_requested_payout_amount(
    request: SimulationRequest,
    payout_cap: float,
    minimum_amount: float,
) -> float | None:
    if request.payout_policy == PayoutPolicy.AUTO_MAX:
        return payout_cap
    if request.payout_policy == PayoutPolicy.AUTO_MINIMUM:
        return minimum_amount
    if request.payout_policy == PayoutPolicy.FIXED_AMOUNT:
        if request.payout_amount is None or request.payout_amount > payout_cap:
            return None
        return request.payout_amount
    return None


def _normalize_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp if timestamp.tzinfo is None else timestamp.tz_localize(None)
