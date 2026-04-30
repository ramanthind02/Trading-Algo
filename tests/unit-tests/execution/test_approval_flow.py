"""Unit tests for execution.approval_flow.request_approval.

All Telegram API calls go through a fake notifier, so these tests run
offline and deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pytest

from execution.approval_flow import (
    ApprovalDecision,
    new_run_id,
    request_approval,
)


@dataclass
class _FakeNotifier:
    """Mimics the subset of TelegramNotifier used by approval_flow."""

    updates_queue: List[List[Dict]] = field(default_factory=list)
    sent_message_id: int = 1001
    sent_payloads: List[Dict] = field(default_factory=list)
    edited: List[Dict] = field(default_factory=list)
    answered: List[Dict] = field(default_factory=list)
    send_should_fail: bool = False

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


def _callback(update_id: int, cb_id: str, data: str, user_id: int) -> Dict:
    return {
        "update_id": update_id,
        "callback_query": {
            "id": cb_id,
            "from": {"id": user_id},
            "data": data,
        },
    }


def test_approve_happy_path() -> None:
    run_id = new_run_id()
    notifier = _FakeNotifier(updates_queue=[[_callback(1, "cb1", f"approve:{run_id}", 111)]])

    out = request_approval(
        notifier,
        message_text="test",
        run_id=run_id,
        authorized_user_ids=[111],
        timeout_seconds=1,
        poll_chunk_seconds=1,
    )

    assert out.decision == ApprovalDecision.APPROVED
    assert out.approver_telegram_id == 111
    assert notifier.answered[0]["id"] == "cb1"
    assert "Approved" in notifier.edited[-1]["text"]


def test_cancel_happy_path() -> None:
    run_id = new_run_id()
    notifier = _FakeNotifier(updates_queue=[[_callback(1, "cb1", f"cancel:{run_id}", 111)]])

    out = request_approval(
        notifier,
        message_text="test",
        run_id=run_id,
        authorized_user_ids=[111],
        timeout_seconds=1,
        poll_chunk_seconds=1,
    )

    assert out.decision == ApprovalDecision.CANCELLED


def test_unauthorized_user_ignored_then_timeout() -> None:
    run_id = new_run_id()
    notifier = _FakeNotifier(updates_queue=[
        [_callback(1, "cb1", f"approve:{run_id}", 999)],   # wrong user
    ])

    out = request_approval(
        notifier,
        message_text="test",
        run_id=run_id,
        authorized_user_ids=[111],
        timeout_seconds=1,
        poll_chunk_seconds=1,
    )

    assert out.decision == ApprovalDecision.TIMED_OUT
    assert notifier.answered[0]["text"] == "Not authorized."


def test_stale_run_id_ignored_then_timeout() -> None:
    notifier = _FakeNotifier(updates_queue=[
        [_callback(1, "cb1", "approve:OLDRUN", 111)],
    ])

    out = request_approval(
        notifier,
        message_text="test",
        run_id="NEWRUN",
        authorized_user_ids=[111],
        timeout_seconds=1,
        poll_chunk_seconds=1,
    )

    assert out.decision == ApprovalDecision.TIMED_OUT
    assert notifier.answered[0]["text"] == "Stale run."


def test_timeout_edits_message() -> None:
    notifier = _FakeNotifier(updates_queue=[])
    out = request_approval(
        notifier,
        message_text="test",
        run_id="R",
        authorized_user_ids=[111],
        timeout_seconds=1,
        poll_chunk_seconds=1,
    )
    assert out.decision == ApprovalDecision.TIMED_OUT
    assert "Timed out" in notifier.edited[-1]["text"]


def test_empty_authorized_list_refuses() -> None:
    notifier = _FakeNotifier()
    out = request_approval(
        notifier,
        message_text="test",
        run_id="R",
        authorized_user_ids=[],
        timeout_seconds=1,
    )
    assert out.decision == ApprovalDecision.CANCELLED
    assert notifier.sent_payloads == []  # never sent


def test_send_failure_treated_as_cancel() -> None:
    notifier = _FakeNotifier(send_should_fail=True)
    out = request_approval(
        notifier,
        message_text="test",
        run_id="R",
        authorized_user_ids=[111],
        timeout_seconds=1,
    )
    assert out.decision == ApprovalDecision.CANCELLED


def test_run_ids_are_unique() -> None:
    ids = {new_run_id() for _ in range(100)}
    assert len(ids) == 100


def test_allow_any_approver_accepts_unlisted_user() -> None:
    run_id = new_run_id()
    notifier = _FakeNotifier(updates_queue=[[_callback(1, "cb1", f"approve:{run_id}", 999)]])

    out = request_approval(
        notifier,
        message_text="test",
        run_id=run_id,
        authorized_user_ids=[],
        timeout_seconds=1,
        poll_chunk_seconds=1,
        allow_any_approver=True,
    )

    assert out.decision == ApprovalDecision.APPROVED
    assert out.approver_telegram_id == 999


def test_allow_any_approver_with_empty_whitelist_still_sends() -> None:
    notifier = _FakeNotifier(updates_queue=[])
    out = request_approval(
        notifier,
        message_text="test",
        run_id="R",
        authorized_user_ids=[],
        timeout_seconds=1,
        poll_chunk_seconds=1,
        allow_any_approver=True,
    )
    assert out.decision == ApprovalDecision.TIMED_OUT
    assert len(notifier.sent_payloads) == 1  # message WAS sent (unlike default refuse-to-send)


def test_approve_after_noise_update() -> None:
    """First batch has an irrelevant update; second batch has the approve."""
    run_id = new_run_id()
    notifier = _FakeNotifier(updates_queue=[
        [{"update_id": 1, "message": {}}],
        [_callback(2, "cb2", f"approve:{run_id}", 111)],
    ])
    out = request_approval(
        notifier,
        message_text="test",
        run_id=run_id,
        authorized_user_ids=[111],
        timeout_seconds=2,
        poll_chunk_seconds=1,
    )
    assert out.decision == ApprovalDecision.APPROVED
