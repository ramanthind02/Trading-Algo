"""Tests for the new ``default_on_timeout`` and ``POLL_UNHEALTHY`` behaviour
in :func:`execution.approval_flow.request_approval`.

These cover the safety inversion needed by the CFD prop-firm flow:
"no answer in N min = execute" should still fail-closed if the Telegram
polling path is unhealthy (so a Telegram outage never becomes an unintended
approval).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pytest

from execution.approval_flow import (
    APPROVED_DECISIONS,
    ApprovalDecision,
    new_run_id,
    request_approval,
)


@dataclass
class _FakeNotifier:
    updates_queue: List[List[Dict]] = field(default_factory=list)
    sent_message_id: int = 1001
    sent_payloads: List[Dict] = field(default_factory=list)
    edited: List[Dict] = field(default_factory=list)
    answered: List[Dict] = field(default_factory=list)
    send_should_fail: bool = False
    healthy: bool = True
    health_probe_calls: int = 0

    def send_with_inline_keyboard(self, text: str, buttons: List[Dict]) -> Optional[int]:
        self.sent_payloads.append({"text": text, "buttons": buttons})
        return None if self.send_should_fail else self.sent_message_id

    def edit_message_text(self, message_id: int, text: str) -> bool:
        self.edited.append({"message_id": message_id, "text": text})
        return True

    def answer_callback_query(self, callback_query_id: str, text: str = "") -> bool:
        self.answered.append({"id": callback_query_id, "text": text})
        return True

    def get_updates(self, offset=None, timeout=25, allowed_updates=None) -> List[Dict]:
        if not self.updates_queue:
            return []
        return self.updates_queue.pop(0)

    def probe_health(self) -> bool:
        self.health_probe_calls += 1
        return self.healthy


def test_default_on_timeout_cancelled_is_default() -> None:
    """Back-compat: omitting default_on_timeout preserves TIMED_OUT behaviour."""
    notifier = _FakeNotifier()
    out = request_approval(
        notifier,
        message_text="t",
        run_id="R",
        authorized_user_ids=[111],
        timeout_seconds=1,
        poll_chunk_seconds=1,
    )
    assert out.decision == ApprovalDecision.TIMED_OUT
    assert notifier.health_probe_calls == 0


def test_default_on_timeout_approve_with_healthy_poll_yields_timeout_approval() -> None:
    notifier = _FakeNotifier(healthy=True)
    out = request_approval(
        notifier,
        message_text="t",
        run_id="R",
        authorized_user_ids=[111],
        timeout_seconds=1,
        poll_chunk_seconds=1,
        default_on_timeout=ApprovalDecision.APPROVED,
    )
    assert out.decision == ApprovalDecision.APPROVED_BY_TIMEOUT
    assert out.decision in APPROVED_DECISIONS
    assert notifier.health_probe_calls == 1
    assert "auto-approved" in notifier.edited[-1]["text"]


def test_default_on_timeout_approve_with_unhealthy_poll_fails_closed() -> None:
    """The critical safety test: an unhealthy Telegram path must NOT auto-approve."""
    notifier = _FakeNotifier(healthy=False)
    out = request_approval(
        notifier,
        message_text="t",
        run_id="R",
        authorized_user_ids=[111],
        timeout_seconds=1,
        poll_chunk_seconds=1,
        default_on_timeout=ApprovalDecision.APPROVED,
    )
    assert out.decision == ApprovalDecision.POLL_UNHEALTHY
    assert out.decision not in APPROVED_DECISIONS
    assert notifier.health_probe_calls == 1
    assert "Poll unhealthy" in notifier.edited[-1]["text"]


def test_user_approval_still_records_as_approved_not_by_timeout() -> None:
    """When a user explicitly clicks approve, decision must be APPROVED (not APPROVED_BY_TIMEOUT)."""
    run_id = new_run_id()
    notifier = _FakeNotifier(
        updates_queue=[[
            {
                "update_id": 1,
                "callback_query": {
                    "id": "cb1",
                    "from": {"id": 111},
                    "data": f"approve:{run_id}",
                },
            },
        ]],
        healthy=True,
    )
    out = request_approval(
        notifier,
        message_text="t",
        run_id=run_id,
        authorized_user_ids=[111],
        timeout_seconds=2,
        poll_chunk_seconds=1,
        default_on_timeout=ApprovalDecision.APPROVED,
    )
    assert out.decision == ApprovalDecision.APPROVED
    assert out.decision != ApprovalDecision.APPROVED_BY_TIMEOUT
    # The probe is only used on timeout, not user-click.
    assert notifier.health_probe_calls == 0


def test_default_on_timeout_invalid_value_raises() -> None:
    notifier = _FakeNotifier()
    with pytest.raises(ValueError):
        request_approval(
            notifier,
            message_text="t",
            run_id="R",
            authorized_user_ids=[111],
            timeout_seconds=1,
            poll_chunk_seconds=1,
            default_on_timeout=ApprovalDecision.POLL_UNHEALTHY,
        )


def test_approved_decisions_set_contains_both_approval_outcomes() -> None:
    assert ApprovalDecision.APPROVED in APPROVED_DECISIONS
    assert ApprovalDecision.APPROVED_BY_TIMEOUT in APPROVED_DECISIONS
    assert ApprovalDecision.CANCELLED not in APPROVED_DECISIONS
    assert ApprovalDecision.TIMED_OUT not in APPROVED_DECISIONS
    assert ApprovalDecision.POLL_UNHEALTHY not in APPROVED_DECISIONS
    assert ApprovalDecision.NOT_REQUESTED not in APPROVED_DECISIONS


def test_custom_poll_health_probe_overrides_notifier_probe() -> None:
    notifier = _FakeNotifier(healthy=True)
    out = request_approval(
        notifier,
        message_text="t",
        run_id="R",
        authorized_user_ids=[111],
        timeout_seconds=1,
        poll_chunk_seconds=1,
        default_on_timeout=ApprovalDecision.APPROVED,
        poll_health_probe=lambda: False,  # explicit override says unhealthy
    )
    assert out.decision == ApprovalDecision.POLL_UNHEALTHY
    # Custom probe used; notifier's own probe should NOT have been called.
    assert notifier.health_probe_calls == 0
