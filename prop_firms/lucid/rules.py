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
    EodTrailingDrawdownRule,
    FundedRules,
    SimulationEvent,
    SimulationRequest,
    SimulationSummary,
)
from prop_firms.base.simulator import BasePropFirmEngine, DayProcessingResult


@dataclass(frozen=True)
class LucidState:
    """Provider-specific account state for LucidFlex."""

    phase: AccountPhase
    status: AccountStatus
    balance: float
    closing_high_balance: float
    max_loss_floor: float
    largest_daily_profit: float
    payout_cycle_profit: float
    payout_cycle_largest_profit: float
    payout_profit_days: int
    payout_count: int
    total_payout_amount: float
    challenge_fee_paid: float
    reset_fee_paid: float
    resets_used: int
    evaluation_trading_days: int
    evaluation_pass_day: pd.Timestamp | None
    breach_reason: BreachReason | None
    scale_contract_limit: ContractLimit


class LucidRuleEngine(BasePropFirmEngine[LucidState]):
    """LucidFlex implementation over the shared simulator shell."""

    def __init__(self, account_definitions: dict[str, AccountDefinition]) -> None:
        self._account_definitions = account_definitions

    def get_account_definition(self, account_code: str) -> AccountDefinition:
        if account_code not in self._account_definitions:
            raise KeyError(
                f"Unknown Lucid account '{account_code}'. "
                f"Available accounts: {sorted(self._account_definitions)}"
            )
        return self._account_definitions[account_code]

    def build_initial_state(
        self,
        request: SimulationRequest,
        account_definition: AccountDefinition,
    ) -> LucidState:
        initial_balance = (
            account_definition.account_size
            if request.starting_balance_override is None
            else request.starting_balance_override
        )
        initial_floor = _compute_max_loss_floor(
            drawdown_rule=account_definition.evaluation.drawdown_rule,
            closing_high_balance=initial_balance,
            payout_count=0,
            account_size=account_definition.account_size,
        )
        return LucidState(
            phase=AccountPhase.EVALUATION,
            status=AccountStatus.ACTIVE,
            balance=initial_balance,
            closing_high_balance=initial_balance,
            max_loss_floor=initial_floor,
            largest_daily_profit=0.0,
            payout_cycle_profit=0.0,
            payout_cycle_largest_profit=0.0,
            payout_profit_days=0,
            payout_count=0,
            total_payout_amount=0.0,
            challenge_fee_paid=account_definition.fees.challenge_fee,
            reset_fee_paid=0.0,
            resets_used=0,
            evaluation_trading_days=0,
            evaluation_pass_day=None,
            breach_reason=None,
            scale_contract_limit=account_definition.evaluation.max_contract_limit,
        )

    def process_day(
        self,
        state: LucidState,
        day_input: DayInput,
        request: SimulationRequest,
        account_definition: AccountDefinition,
    ) -> DayProcessingResult[LucidState]:
        if state.phase == AccountPhase.EVALUATION:
            return self._process_evaluation_day(
                state=state,
                day_input=day_input,
                account_definition=account_definition,
            )
        return self._process_funded_day(
            state=state,
            day_input=day_input,
            request=request,
            account_definition=account_definition,
        )

    def can_reset(
        self,
        state: LucidState,
        request: SimulationRequest,
        account_definition: AccountDefinition,
    ) -> bool:
        return (
            request.reset_policy == ResetPolicy.AUTO_RESET
            and state.phase == AccountPhase.EVALUATION
            and state.status == AccountStatus.BREACHED
            and state.resets_used < request.max_resets
            and account_definition.fees.reset_fee >= 0.0
        )

    def apply_reset(
        self,
        state: LucidState,
        request: SimulationRequest,
        trading_day: pd.Timestamp,
        account_definition: AccountDefinition,
    ) -> tuple[LucidState, SimulationEvent]:
        reset_state = self.build_initial_state(request=request, account_definition=account_definition)
        next_state = replace(
            reset_state,
            challenge_fee_paid=state.challenge_fee_paid,
            reset_fee_paid=state.reset_fee_paid + account_definition.fees.reset_fee,
            resets_used=state.resets_used + 1,
        )
        reset_event = SimulationEvent(
            trading_day=trading_day,
            event_type=EventType.RESET_APPLIED,
            phase=AccountPhase.EVALUATION,
            message=(
                f"Applied Lucid evaluation reset #{next_state.resets_used} "
                f"for {account_definition.account_code}"
            ),
            amount=account_definition.fees.reset_fee,
        )
        return next_state, reset_event

    def build_summary(
        self,
        state: LucidState,
        request: SimulationRequest,
        account_definition: AccountDefinition,
    ) -> SimulationSummary:
        gross_profit = (
            (state.balance - account_definition.account_size)
            if state.phase == AccountPhase.EVALUATION
            else (state.balance - account_definition.account_size) + state.total_payout_amount
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
            total_payout_amount=state.total_payout_amount,
            payout_count=state.payout_count,
            resets_used=state.resets_used,
            evaluation_pass_day=state.evaluation_pass_day,
            breach_reason=state.breach_reason,
        )

    def _process_evaluation_day(
        self,
        state: LucidState,
        day_input: DayInput,
        account_definition: AccountDefinition,
    ) -> DayProcessingResult[LucidState]:
        rules = account_definition.evaluation
        balance = state.balance * (1.0 + day_input.daily_return)
        daily_profit = balance - state.balance
        closing_high_balance = max(state.closing_high_balance, balance)
        max_loss_floor = _compute_max_loss_floor(
            drawdown_rule=rules.drawdown_rule,
            closing_high_balance=closing_high_balance,
            payout_count=state.payout_count,
            account_size=account_definition.account_size,
        )
        largest_daily_profit = max(state.largest_daily_profit, max(daily_profit, 0.0))
        exposure = evaluate_exposure(
            day_input=day_input,
            contract_limit=rules.max_contract_limit,
            account_balance=balance,
            max_leverage=rules.max_leverage,
        )

        breached_state = _maybe_build_breach_state(
            state=replace(
                state,
                balance=balance,
                closing_high_balance=closing_high_balance,
                max_loss_floor=max_loss_floor,
                largest_daily_profit=largest_daily_profit,
                evaluation_trading_days=state.evaluation_trading_days + 1,
                breach_reason=None,
            ),
            exposure=exposure,
        )
        if breached_state is not None:
            snapshot = _build_snapshot(
                state=breached_state,
                day_input=day_input,
                account_definition=account_definition,
                exposure=exposure,
            )
            return DayProcessingResult(
                state=breached_state,
                snapshot=snapshot,
                events=(
                    _breach_event(
                        trading_day=day_input.trading_day,
                        phase=AccountPhase.EVALUATION,
                        breach_reason=breached_state.breach_reason,
                    ),
                ),
            )

        evaluation_state = replace(
            state,
            balance=balance,
            closing_high_balance=closing_high_balance,
            max_loss_floor=max_loss_floor,
            largest_daily_profit=largest_daily_profit,
            evaluation_trading_days=state.evaluation_trading_days + 1,
            breach_reason=None,
        )
        total_profit = balance - account_definition.account_size
        consistency_ratio = _compute_consistency_ratio(
            largest_daily_profit=largest_daily_profit,
            total_profit=total_profit,
        )
        passed = (
            total_profit >= rules.profit_target_amount
            and evaluation_state.evaluation_trading_days >= rules.minimum_trading_days
            and (
                rules.consistency_rule is None
                or rules.consistency_rule.is_satisfied(
                    largest_day_profit=largest_daily_profit,
                    total_profit=total_profit,
                )
            )
        )
        if not passed:
            snapshot = _build_snapshot(
                state=evaluation_state,
                day_input=day_input,
                account_definition=account_definition,
                exposure=exposure,
                consistency_ratio=consistency_ratio,
                consistency_cap_ratio=(
                    None
                    if rules.consistency_rule is None
                    else (
                        rules.consistency_rule.max_ratio
                        * rules.consistency_rule.cushion_multiplier
                    )
                ),
            )
            return DayProcessingResult(
                state=evaluation_state,
                snapshot=snapshot,
                events=tuple(),
            )

        funded_state = _build_funded_state(
            account_definition=account_definition,
            previous_state=evaluation_state,
            trading_day=day_input.trading_day,
        )
        snapshot = _build_snapshot(
            state=evaluation_state,
            day_input=day_input,
            account_definition=account_definition,
            exposure=exposure,
            consistency_ratio=consistency_ratio,
            consistency_cap_ratio=(
                None
                if rules.consistency_rule is None
                else (
                    rules.consistency_rule.max_ratio
                    * rules.consistency_rule.cushion_multiplier
                )
            ),
        )
        events = (
            SimulationEvent(
                trading_day=day_input.trading_day,
                event_type=EventType.EVALUATION_PASSED,
                phase=AccountPhase.EVALUATION,
                message=f"Passed Lucid evaluation for {account_definition.account_code}",
                amount=balance,
            ),
            SimulationEvent(
                trading_day=day_input.trading_day,
                event_type=EventType.FUNDED_STARTED,
                phase=AccountPhase.FUNDED,
                message=(
                    f"Started Lucid funded account {account_definition.account_code} "
                    "at nominal balance"
                ),
                amount=account_definition.account_size,
            ),
        )
        return DayProcessingResult(
            state=funded_state,
            snapshot=snapshot,
            events=events,
        )

    def _process_funded_day(
        self,
        state: LucidState,
        day_input: DayInput,
        request: SimulationRequest,
        account_definition: AccountDefinition,
    ) -> DayProcessingResult[LucidState]:
        rules = account_definition.funded
        balance = state.balance * (1.0 + day_input.daily_return)
        daily_profit = balance - state.balance
        closing_high_balance = max(state.closing_high_balance, balance)
        max_loss_floor = _compute_max_loss_floor(
            drawdown_rule=rules.drawdown_rule,
            closing_high_balance=closing_high_balance,
            payout_count=state.payout_count,
            account_size=account_definition.account_size,
        )
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

        base_state = replace(
            state,
            balance=balance,
            closing_high_balance=closing_high_balance,
            max_loss_floor=max_loss_floor,
            payout_cycle_profit=payout_cycle_profit,
            payout_cycle_largest_profit=payout_cycle_largest_profit,
            payout_profit_days=payout_profit_days,
            breach_reason=None,
        )
        breached_state = _maybe_build_breach_state(
            state=base_state,
            exposure=exposure,
        )
        if breached_state is not None:
            snapshot = _build_snapshot(
                state=breached_state,
                day_input=day_input,
                account_definition=account_definition,
                exposure=exposure,
            )
            return DayProcessingResult(
                state=breached_state,
                snapshot=snapshot,
                events=(
                    _breach_event(
                        trading_day=day_input.trading_day,
                        phase=AccountPhase.FUNDED,
                        breach_reason=breached_state.breach_reason,
                    ),
                ),
            )

        payout_events, post_payout_state = _maybe_request_payout(
            state=base_state,
            request=request,
            trading_day=day_input.trading_day,
            account_definition=account_definition,
        )
        next_scale = _resolve_scaling_limit(
            funded_rules=account_definition.funded,
            balance=post_payout_state.balance,
            account_size=account_definition.account_size,
        )
        scale_events = _scale_change_events(
            trading_day=day_input.trading_day,
            previous_limit=post_payout_state.scale_contract_limit,
            next_limit=next_scale,
        )
        next_status = (
            AccountStatus.COMPLETED
            if post_payout_state.payout_count
            >= rules.payout_rule.max_requests_per_account
            else AccountStatus.ACTIVE
        )
        next_state = replace(
            post_payout_state,
            status=next_status,
            scale_contract_limit=next_scale,
        )
        snapshot = _build_snapshot(
            state=next_state,
            day_input=day_input,
            account_definition=account_definition,
            exposure=exposure,
        )
        return DayProcessingResult(
            state=next_state,
            snapshot=snapshot,
            events=payout_events + scale_events,
        )


def _build_funded_state(
    account_definition: AccountDefinition,
    previous_state: LucidState,
    trading_day: pd.Timestamp,
) -> LucidState:
    opening_balance = account_definition.account_size
    initial_scale = _resolve_scaling_limit(
        funded_rules=account_definition.funded,
        balance=opening_balance,
        account_size=account_definition.account_size,
    )
    return LucidState(
        phase=AccountPhase.FUNDED,
        status=AccountStatus.ACTIVE,
        balance=opening_balance,
        closing_high_balance=opening_balance,
        max_loss_floor=_compute_max_loss_floor(
            drawdown_rule=account_definition.funded.drawdown_rule,
            closing_high_balance=opening_balance,
            payout_count=0,
            account_size=account_definition.account_size,
        ),
        largest_daily_profit=previous_state.largest_daily_profit,
        payout_cycle_profit=0.0,
        payout_cycle_largest_profit=0.0,
        payout_profit_days=0,
        payout_count=0,
        total_payout_amount=0.0,
        challenge_fee_paid=previous_state.challenge_fee_paid,
        reset_fee_paid=previous_state.reset_fee_paid,
        resets_used=previous_state.resets_used,
        evaluation_trading_days=previous_state.evaluation_trading_days,
        evaluation_pass_day=trading_day,
        breach_reason=None,
        scale_contract_limit=initial_scale,
    )


def _maybe_build_breach_state(
    state: LucidState,
    exposure: ExposureCheck,
) -> LucidState | None:
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
    state: LucidState,
    day_input: DayInput,
    account_definition: AccountDefinition,
    exposure: ExposureCheck,
    consistency_ratio: float | None = None,
    consistency_cap_ratio: float | None = None,
) -> DailySnapshot:
    profit_target_balance = (
        account_definition.account_size
        + account_definition.evaluation.profit_target_amount
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
        consistency_ratio=consistency_ratio,
        consistency_cap_ratio=consistency_cap_ratio,
        payout_cycle_profit=state.payout_cycle_profit,
        payout_cycle_largest_profit=state.payout_cycle_largest_profit,
        payout_profit_days=state.payout_profit_days,
        payouts_taken=state.payout_count,
        daily_loss_limit=None,
        daily_loss_limit_hit=0,
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


def _compute_consistency_ratio(
    largest_daily_profit: float,
    total_profit: float,
) -> float | None:
    if total_profit <= 0.0:
        return None
    return largest_daily_profit / total_profit


def _compute_max_loss_floor(
    drawdown_rule: EodTrailingDrawdownRule,
    closing_high_balance: float,
    payout_count: int,
    account_size: float,
) -> float:
    starting_floor = account_size - drawdown_rule.loss_limit_amount
    if payout_count > 0 or closing_high_balance >= drawdown_rule.initial_trail_balance:
        return drawdown_rule.locked_mll_balance
    trailing_floor = min(
        closing_high_balance - drawdown_rule.loss_limit_amount,
        drawdown_rule.locked_mll_balance,
    )
    return max(starting_floor, trailing_floor)


def _resolve_scaling_limit(
    funded_rules: FundedRules,
    balance: float,
    account_size: float,
) -> ContractLimit:
    profit = max(balance - account_size, 0.0)
    matching_tiers = (
        tier.contract_limit
        for tier in funded_rules.scaling_plan
        if tier.matches(profit)
    )
    return next(matching_tiers, funded_rules.max_contract_limit)


def _maybe_request_payout(
    state: LucidState,
    request: SimulationRequest,
    trading_day: pd.Timestamp,
    account_definition: AccountDefinition,
) -> tuple[tuple[SimulationEvent, ...], LucidState]:
    payout_rule = account_definition.funded.payout_rule
    if request.payout_policy == PayoutPolicy.NO_REQUESTS:
        return tuple(), state
    if state.payout_count >= payout_rule.max_requests_per_account:
        return tuple(), state

    eligible = (
        state.payout_profit_days >= payout_rule.profit_day_rule.required_days
        and state.payout_cycle_profit > 0.0
        and (
            payout_rule.consistency_rule is None
            or payout_rule.consistency_rule.is_satisfied(
                largest_day_profit=state.payout_cycle_largest_profit,
                total_profit=state.payout_cycle_profit,
            )
        )
    )
    if not eligible:
        return tuple(), state

    gross_payout_cap = min(
        state.payout_cycle_profit * payout_rule.max_profit_share,
        payout_rule.payout_cap_for_request(state.payout_count + 1),
        max(
            state.balance - _resolve_protected_balance(account_definition),
            0.0,
        ),
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

    trader_amount = requested_amount * payout_rule.trader_profit_split
    next_balance = state.balance - requested_amount
    next_state = replace(
        state,
        balance=next_balance,
        closing_high_balance=max(state.closing_high_balance, next_balance),
        max_loss_floor=account_definition.funded.drawdown_rule.locked_mll_balance,
        payout_cycle_profit=0.0,
        payout_cycle_largest_profit=0.0,
        payout_profit_days=0,
        payout_count=state.payout_count + 1,
        total_payout_amount=state.total_payout_amount + requested_amount,
        breach_reason=None,
    )
    eligible_event = SimulationEvent(
        trading_day=trading_day,
        event_type=EventType.PAYOUT_ELIGIBLE,
        phase=AccountPhase.FUNDED,
        message="Account is eligible for payout",
        amount=gross_payout_cap,
    )
    payout_event = SimulationEvent(
        trading_day=trading_day,
        event_type=EventType.PAYOUT_REQUESTED,
        phase=AccountPhase.FUNDED,
        message=(
            f"Requested payout of {requested_amount:.2f}; "
            f"trader receives {trader_amount:.2f} after split"
        ),
        amount=requested_amount,
    )
    return (eligible_event, payout_event), next_state


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


def _resolve_protected_balance(account_definition: AccountDefinition) -> float:
    payout_rule = account_definition.funded.payout_rule
    protected_balance = payout_rule.protected_balance
    if protected_balance is not None:
        return protected_balance
    return account_definition.funded.drawdown_rule.locked_mll_balance


def _scale_change_events(
    trading_day: pd.Timestamp,
    previous_limit: ContractLimit,
    next_limit: ContractLimit,
) -> tuple[SimulationEvent, ...]:
    if previous_limit == next_limit:
        return tuple()
    return (
        SimulationEvent(
            trading_day=trading_day,
            event_type=EventType.SCALE_CHANGED,
            phase=AccountPhase.FUNDED,
            message=(
                "Scaling tier changed to "
                f"{next_limit.mini_contracts} mini / {next_limit.micro_contracts} micro"
            ),
        ),
    )
