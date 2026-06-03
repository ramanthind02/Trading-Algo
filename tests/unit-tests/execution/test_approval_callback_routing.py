"""Tests for :func:`execution.approval_flow.request_batch_approval` -- the
multi-target approval UX used by the CFD prop-firm execution flow.

Covers:
- approve_all short-circuits the window.
- cancel_all returns immediately with every account cancelled.
- cancel_one marks a single account cancelled but continues polling.
- Timeout default_on_timeout=cancel → non-cancelled accounts TIMED_OUT.
- Timeout default_on_timeout=approve + healthy poll → APPROVED_BY_TIMEOUT.
- Timeout default_on_timeout=approve + unhealthy poll → POLL_UNHEALTHY for all.
- Stale callbacks (different run_id) ignored.
- Empty whitelist fail-closes the batch.
- approve_all preserves prior cancel_one decisions (does NOT override them).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pytest

from execution.approval_flow import (
    APPROVED_DECISIONS,
    ApprovalDecision,
    BatchApprovalOutcome,
    new_run_id,
    request_batch_approval,
)


@dataclass
class _FakeRowNotifier:
    """Captures send_with_inline_keyboard_rows + standard methods."""

    updates_queue: List[List[Dict]] = field(default_factory=list)
    sent_message_id: int = 1001
    sent_rows_payloads: List[Dict] = field(default_factory=list)
    edited: List[Dict] = field(default_factory=list)
    answered: List[Dict] = field(default_factory=list)
    send_should_fail: bool = False
    healthy: bool = True

    def send_with_inline_keyboard_rows(
        self, text: str, button_rows: List[List[Dict]]
    ) -> Optional[int]:
        self.sent_rows_payloads.append({"text": text, "button_rows": button_rows})
        return None if self.send_should_fail else self.sent_message_id

    def send_with_inline_keyboard(self, text: str, buttons: List[Dict]) -> Optional[int]:
        # Should not be called when rows variant is available, but keep the
        # method present for the fall-back path test.
        self.sent_rows_payloads.append({"text": text, "buttons_flat": buttons})
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
        return self.healthy


def _cb(update_id: int, cb_id: str, data: str, user_id: int = 111) -> Dict:
    return {
        "update_id": update_id,
        "callback_query": {
            "id": cb_id,
            "from": {"id": user_id},
            "data": data,
        },
    }


def test_message_layout_has_three_distinct_row_types() -> None:
    """Approve-all + cancel-all + N per-account rows."""
    run_id = new_run_id()
    notifier = _FakeRowNotifier()
    request_batch_approval(
        notifier,
        message_text="batch test",
        run_id=run_id,
        account_labels=["a", "b", "c"],
        authorized_user_ids=[111],
        timeout_seconds=1,
        poll_chunk_seconds=1,
    )
    rows = notifier.sent_rows_payloads[0]["button_rows"]
    assert rows[0][0]["callback_data"] == f"approve_all:{run_id}"
    assert rows[1][0]["callback_data"] == f"cancel_all:{run_id}"
    # Three per-account cancel rows follow.
    assert {rows[2][0]["callback_data"], rows[3][0]["callback_data"], rows[4][0]["callback_data"]} == {
        f"cancel_one:{run_id}:a",
        f"cancel_one:{run_id}:b",
        f"cancel_one:{run_id}:c",
    }


def test_approve_all_short_circuits_and_marks_every_account_approved() -> None:
    run_id = new_run_id()
    notifier = _FakeRowNotifier(updates_queue=[[_cb(1, "x", f"approve_all:{run_id}")]])
    out = request_batch_approval(
        notifier,
        message_text="batch",
        run_id=run_id,
        account_labels=["acct_a", "acct_b"],
        authorized_user_ids=[111],
        timeout_seconds=5,
        poll_chunk_seconds=1,
    )
    assert out.global_decision == ApprovalDecision.APPROVED
    assert out.per_account == {
        "acct_a": ApprovalDecision.APPROVED,
        "acct_b": ApprovalDecision.APPROVED,
    }
    assert out.approver_telegram_id == 111
    assert all(d in APPROVED_DECISIONS for d in out.per_account.values())


def test_cancel_all_returns_immediately_with_everything_cancelled() -> None:
    run_id = new_run_id()
    notifier = _FakeRowNotifier(updates_queue=[[_cb(1, "x", f"cancel_all:{run_id}")]])
    out = request_batch_approval(
        notifier,
        message_text="batch",
        run_id=run_id,
        account_labels=["a", "b"],
        authorized_user_ids=[111],
        timeout_seconds=5,
        poll_chunk_seconds=1,
    )
    assert out.global_decision == ApprovalDecision.CANCELLED
    assert all(d == ApprovalDecision.CANCELLED for d in out.per_account.values())


def test_cancel_one_then_timeout_only_affects_that_account() -> None:
    run_id = new_run_id()
    notifier = _FakeRowNotifier(updates_queue=[[_cb(1, "x", f"cancel_one:{run_id}:acct_a")]])
    out = request_batch_approval(
        notifier,
        message_text="batch",
        run_id=run_id,
        account_labels=["acct_a", "acct_b"],
        authorized_user_ids=[111],
        timeout_seconds=1,
        poll_chunk_seconds=1,
    )
    # Timeout default = cancel
    assert out.global_decision == ApprovalDecision.TIMED_OUT
    assert out.per_account["acct_a"] == ApprovalDecision.CANCELLED
    assert out.per_account["acct_b"] == ApprovalDecision.TIMED_OUT


def test_cancel_one_then_approve_all_preserves_cancellation() -> None:
    """approve_all must NOT override an explicit cancel_one decision."""
    run_id = new_run_id()
    notifier = _FakeRowNotifier(updates_queue=[
        [_cb(1, "x", f"cancel_one:{run_id}:acct_a")],  # round 1: cancel one
        [_cb(2, "y", f"approve_all:{run_id}")],         # round 2: approve all
    ])
    out = request_batch_approval(
        notifier,
        message_text="batch",
        run_id=run_id,
        account_labels=["acct_a", "acct_b", "acct_c"],
        authorized_user_ids=[111],
        timeout_seconds=5,
        poll_chunk_seconds=1,
    )
    assert out.global_decision == ApprovalDecision.APPROVED
    assert out.per_account["acct_a"] == ApprovalDecision.CANCELLED   # preserved
    assert out.per_account["acct_b"] == ApprovalDecision.APPROVED
    assert out.per_account["acct_c"] == ApprovalDecision.APPROVED


def test_unknown_account_label_in_cancel_one_is_ignored() -> None:
    run_id = new_run_id()
    notifier = _FakeRowNotifier(updates_queue=[
        [_cb(1, "x", f"cancel_one:{run_id}:does_not_exist")],
    ])
    out = request_batch_approval(
        notifier,
        message_text="batch",
        run_id=run_id,
        account_labels=["acct_a"],
        authorized_user_ids=[111],
        timeout_seconds=1,
        poll_chunk_seconds=1,
    )
    # Timeout (no valid cancel happened) with default = cancel
    assert out.per_account == {"acct_a": ApprovalDecision.TIMED_OUT}
    assert notifier.answered[-1]["text"] == "Unknown account."


def test_timeout_default_approve_healthy_marks_uncancelled_as_approved_by_timeout() -> None:
    run_id = new_run_id()
    notifier = _FakeRowNotifier(
        updates_queue=[[_cb(1, "x", f"cancel_one:{run_id}:acct_a")]],
        healthy=True,
    )
    out = request_batch_approval(
        notifier,
        message_text="batch",
        run_id=run_id,
        account_labels=["acct_a", "acct_b"],
        authorized_user_ids=[111],
        timeout_seconds=1,
        poll_chunk_seconds=1,
        default_on_timeout=ApprovalDecision.APPROVED,
    )
    assert out.global_decision == ApprovalDecision.APPROVED_BY_TIMEOUT
    assert out.per_account["acct_a"] == ApprovalDecision.CANCELLED
    assert out.per_account["acct_b"] == ApprovalDecision.APPROVED_BY_TIMEOUT


def test_timeout_default_approve_unhealthy_fails_closed_for_all_accounts() -> None:
    run_id = new_run_id()
    notifier = _FakeRowNotifier(
        updates_queue=[[_cb(1, "x", f"cancel_one:{run_id}:acct_a")]],
        healthy=False,
    )
    out = request_batch_approval(
        notifier,
        message_text="batch",
        run_id=run_id,
        account_labels=["acct_a", "acct_b"],
        authorized_user_ids=[111],
        timeout_seconds=1,
        poll_chunk_seconds=1,
        default_on_timeout=ApprovalDecision.APPROVED,
    )
    assert out.global_decision == ApprovalDecision.POLL_UNHEALTHY
    # Critical: ALL accounts get POLL_UNHEALTHY, including the previously-
    # cancelled one (executor sees no APPROVED decisions anywhere).
    assert out.per_account == {
        "acct_a": ApprovalDecision.POLL_UNHEALTHY,
        "acct_b": ApprovalDecision.POLL_UNHEALTHY,
    }
    assert all(d not in APPROVED_DECISIONS for d in out.per_account.values())


def test_stale_run_id_callback_ignored() -> None:
    notifier = _FakeRowNotifier(updates_queue=[[
        _cb(1, "x", "approve_all:STALE_RUN"),
    ]])
    out = request_batch_approval(
        notifier,
        message_text="batch",
        run_id="REAL_RUN",
        account_labels=["a"],
        authorized_user_ids=[111],
        timeout_seconds=1,
        poll_chunk_seconds=1,
    )
    assert out.global_decision == ApprovalDecision.TIMED_OUT
    assert notifier.answered[-1]["text"] == "Stale or unrecognised."


def test_unauthorized_user_ignored() -> None:
    run_id = new_run_id()
    notifier = _FakeRowNotifier(updates_queue=[[
        _cb(1, "x", f"approve_all:{run_id}", user_id=999),
    ]])
    out = request_batch_approval(
        notifier,
        message_text="batch",
        run_id=run_id,
        account_labels=["a"],
        authorized_user_ids=[111],
        timeout_seconds=1,
        poll_chunk_seconds=1,
    )
    assert out.global_decision == ApprovalDecision.TIMED_OUT
    assert notifier.answered[-1]["text"] == "Not authorized."


def test_allow_any_approver_accepts_unlisted_user() -> None:
    run_id = new_run_id()
    notifier = _FakeRowNotifier(updates_queue=[[
        _cb(1, "x", f"approve_all:{run_id}", user_id=999),
    ]])
    out = request_batch_approval(
        notifier,
        message_text="batch",
        run_id=run_id,
        account_labels=["a"],
        authorized_user_ids=[],
        timeout_seconds=2,
        poll_chunk_seconds=1,
        allow_any_approver=True,
    )
    assert out.global_decision == ApprovalDecision.APPROVED
    assert out.per_account["a"] == ApprovalDecision.APPROVED
    assert out.approver_telegram_id == 999


def test_empty_whitelist_without_allow_any_fails_closed_without_sending() -> None:
    notifier = _FakeRowNotifier()
    out = request_batch_approval(
        notifier,
        message_text="batch",
        run_id="R",
        account_labels=["a", "b"],
        authorized_user_ids=[],
        timeout_seconds=1,
        poll_chunk_seconds=1,
    )
    assert out.global_decision == ApprovalDecision.CANCELLED
    assert all(d == ApprovalDecision.CANCELLED for d in out.per_account.values())
    assert notifier.sent_rows_payloads == []


def test_empty_account_labels_raises() -> None:
    notifier = _FakeRowNotifier()
    with pytest.raises(ValueError):
        request_batch_approval(
            notifier,
            message_text="batch",
            run_id="R",
            account_labels=[],
            authorized_user_ids=[111],
            timeout_seconds=1,
        )


def test_send_failure_fail_closes_batch() -> None:
    notifier = _FakeRowNotifier(send_should_fail=True)
    out = request_batch_approval(
        notifier,
        message_text="batch",
        run_id="R",
        account_labels=["a"],
        authorized_user_ids=[111],
        timeout_seconds=1,
        poll_chunk_seconds=1,
    )
    assert out.global_decision == ApprovalDecision.CANCELLED
    assert out.per_account == {"a": ApprovalDecision.CANCELLED}


def test_fallback_when_notifier_lacks_rows_method() -> None:
    """If the notifier only exposes ``send_with_inline_keyboard`` we flatten."""

    @dataclass
    class _OldNotifier:
        sent_payloads: List[Dict] = field(default_factory=list)
        updates_queue: List[List[Dict]] = field(default_factory=list)

        def send_with_inline_keyboard(self, text, buttons):
            self.sent_payloads.append({"text": text, "buttons": buttons})
            return 1001

        def edit_message_text(self, message_id, text):
            return True

        def answer_callback_query(self, cb_id, text=""):
            return True

        def get_updates(self, offset=None, timeout=25, allowed_updates=None):
            return self.updates_queue.pop(0) if self.updates_queue else []

        def probe_health(self):
            return True

    notifier = _OldNotifier()
    out = request_batch_approval(
        notifier,
        message_text="batch",
        run_id="R",
        account_labels=["a"],
        authorized_user_ids=[111],
        timeout_seconds=1,
        poll_chunk_seconds=1,
    )
    assert out.global_decision == ApprovalDecision.TIMED_OUT
    # Flattened path used (single send_with_inline_keyboard call).
    assert len(notifier.sent_payloads) == 1
    flat_buttons = notifier.sent_payloads[0]["buttons"]
    # Approve_all and cancel_all both present in the flat list.
    callbacks = {b["callback_data"] for b in flat_buttons}
    assert "approve_all:R" in callbacks
    assert "cancel_all:R" in callbacks
    assert "cancel_one:R:a" in callbacks
