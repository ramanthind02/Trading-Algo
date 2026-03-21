from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import pandas as pd

from prop_firms.base.enums import AccountPhase, AccountStatus, EventType
from prop_firms.base.exposure import build_day_inputs, normalize_held_contracts
from prop_firms.base.models import (
    AccountDefinition,
    DailySnapshot,
    DayInput,
    SimulationEvent,
    SimulationRequest,
    SimulationResult,
    SimulationSummary,
)


StateT = TypeVar("StateT")


@dataclass(frozen=True)
class DayProcessingResult(Generic[StateT]):
    """Container returned by provider-specific daily processing hooks."""

    state: StateT
    snapshot: DailySnapshot
    events: tuple[SimulationEvent, ...]


class BasePropFirmEngine(ABC, Generic[StateT]):
    """Shared simulator shell for provider-specific rule engines."""

    def simulate(self, request: SimulationRequest) -> SimulationResult:
        normalized_returns = _normalize_returns(request.returns)
        held_contracts = normalize_held_contracts(
            held_contracts=request.held_contracts,
            returns_index=normalized_returns.index,
        )
        day_inputs = build_day_inputs(normalized_returns, held_contracts)
        account_definition = self.get_account_definition(request.account_code)
        state = self.build_initial_state(request, account_definition)

        snapshots: list[DailySnapshot] = []
        events: list[SimulationEvent] = [
            self.build_start_event(day_inputs=day_inputs, account_definition=account_definition)
        ]

        for day_input in day_inputs:
            processed = self.process_day(
                state=state,
                day_input=day_input,
                request=request,
                account_definition=account_definition,
            )
            state = processed.state
            snapshots.append(processed.snapshot)
            events.extend(processed.events)

            if processed.snapshot.status == AccountStatus.BREACHED:
                if self.can_reset(state=state, request=request, account_definition=account_definition):
                    state, reset_event = self.apply_reset(
                        state=state,
                        request=request,
                        trading_day=day_input.trading_day,
                        account_definition=account_definition,
                    )
                    events.append(reset_event)
                    continue
                break

            if processed.snapshot.status == AccountStatus.COMPLETED:
                break

        summary = self.build_summary(
            state=state,
            request=request,
            account_definition=account_definition,
        )
        completion_day = (
            snapshots[-1].trading_day
            if snapshots
            else normalized_returns.index[-1]
        )
        events.append(
            self.build_completion_event(
                summary=summary,
                account_definition=account_definition,
                trading_day=completion_day,
            )
        )

        return SimulationResult(
            timeline=_build_timeline_frame(snapshots),
            events=_build_events_frame(events),
            summary=summary,
            account_definition=account_definition,
        )

    @abstractmethod
    def get_account_definition(self, account_code: str) -> AccountDefinition:
        """Resolve the requested account definition."""

    @abstractmethod
    def build_initial_state(
        self,
        request: SimulationRequest,
        account_definition: AccountDefinition,
    ) -> StateT:
        """Create the provider-specific starting state."""

    @abstractmethod
    def process_day(
        self,
        state: StateT,
        day_input: DayInput,
        request: SimulationRequest,
        account_definition: AccountDefinition,
    ) -> DayProcessingResult[StateT]:
        """Advance the provider state by one day."""

    @abstractmethod
    def can_reset(
        self,
        state: StateT,
        request: SimulationRequest,
        account_definition: AccountDefinition,
    ) -> bool:
        """Return whether a breached evaluation account can reset."""

    @abstractmethod
    def apply_reset(
        self,
        state: StateT,
        request: SimulationRequest,
        trading_day: pd.Timestamp,
        account_definition: AccountDefinition,
    ) -> tuple[StateT, SimulationEvent]:
        """Apply a provider-specific reset after an evaluation breach."""

    @abstractmethod
    def build_summary(
        self,
        state: StateT,
        request: SimulationRequest,
        account_definition: AccountDefinition,
    ) -> SimulationSummary:
        """Build the terminal summary."""

    def build_start_event(
        self,
        day_inputs: tuple[DayInput, ...],
        account_definition: AccountDefinition,
    ) -> SimulationEvent:
        first_day = day_inputs[0].trading_day
        return SimulationEvent(
            trading_day=first_day,
            event_type=EventType.SIMULATION_STARTED,
            phase=AccountPhase.EVALUATION,
            message=(
                f"Started {account_definition.provider_name} "
                f"{account_definition.program_name} {account_definition.account_code}"
            ),
        )

    def build_completion_event(
        self,
        summary: SimulationSummary,
        account_definition: AccountDefinition,
        trading_day: pd.Timestamp,
    ) -> SimulationEvent:
        return SimulationEvent(
            trading_day=trading_day,
            event_type=EventType.SIMULATION_COMPLETED,
            phase=summary.final_phase,
            message=(
                f"Completed {account_definition.provider_name} "
                f"{account_definition.program_name} {account_definition.account_code}"
            ),
            amount=summary.final_balance,
            breach_reason=summary.breach_reason,
        )


def _normalize_returns(returns: pd.Series) -> pd.Series:
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("returns must use a DatetimeIndex")
    if returns.empty:
        raise ValueError("returns must be non-empty")
    if returns.isna().any():
        raise ValueError("returns cannot contain NaN values")

    normalized = returns.copy()
    normalized.index = pd.to_datetime(normalized.index).tz_localize(None)
    normalized = normalized.sort_index(kind="stable")
    return normalized.astype(float)


def _build_timeline_frame(snapshots: list[DailySnapshot]) -> pd.DataFrame:
    if not snapshots:
        return pd.DataFrame()
    return pd.DataFrame(snapshot.to_record() for snapshot in snapshots)


def _build_events_frame(events: list[SimulationEvent]) -> pd.DataFrame:
    if not events:
        return pd.DataFrame()
    return pd.DataFrame(event.to_record() for event in events)
