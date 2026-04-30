"""
Telegram-based human-in-the-loop approval via inline keyboard + long-poll.

Flow
----
1. Post a message with [Approve] / [Cancel] buttons tagged with a per-run UUID.
2. Long-poll ``getUpdates`` for ``callback_query`` updates up to the timeout.
3. Accept only a callback whose payload matches this run's UUID AND comes from
   a whitelisted Telegram user. Everything else is ignored (logged).
4. On decision, ack the callback, edit the original message to show outcome,
   return the decision.

Deliberately polling-only: no webhook, no public URL, no tunnel. The script
is both sender and listener for the lifetime of a single run.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from deployment.telegram_notifier import TelegramNotifier


logger = logging.getLogger(__name__)


class ApprovalDecision(str, Enum):
    APPROVED = "approved"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    NOT_REQUESTED = "not_requested"


@dataclass(frozen=True)
class ApprovalOutcome:
    decision: ApprovalDecision
    approver_telegram_id: Optional[int] = None


def new_run_id() -> str:
    """Opaque UUID4 used to match callbacks to this specific run."""
    return uuid.uuid4().hex[:12]


def request_approval(
    notifier: TelegramNotifier,
    *,
    message_text: str,
    run_id: str,
    authorized_user_ids: List[int],
    timeout_seconds: int,
    poll_chunk_seconds: int = 25,
    allow_any_approver: bool = False,
) -> ApprovalOutcome:
    """Post the approval message and poll for a matching callback_query.

    Returns as soon as a valid approve/cancel callback arrives, or after
    ``timeout_seconds`` elapse.

    Whitelist behaviour
    -------------------
    By default, only ``authorized_user_ids`` may approve. An empty list
    fails closed (refuses to send the message) so we never auto-trade with
    no whitelist by accident. To intentionally accept any approver, pass
    ``allow_any_approver=True`` — the approver's user id is still recorded
    on the outcome and audited.
    """
    if not allow_any_approver and not authorized_user_ids:
        logger.error("No authorized user IDs configured; refusing to request approval.")
        return ApprovalOutcome(ApprovalDecision.CANCELLED)

    message_id = notifier.send_with_inline_keyboard(
        text=message_text,
        buttons=[
            {"text": "✅ Approve", "callback_data": f"approve:{run_id}"},
            {"text": "❌ Cancel", "callback_data": f"cancel:{run_id}"},
        ],
    )
    if message_id is None:
        logger.error("Approval message send failed; aborting batch.")
        return ApprovalOutcome(ApprovalDecision.CANCELLED)

    deadline = time.monotonic() + timeout_seconds
    offset: Optional[int] = None
    authorized_set = set(authorized_user_ids)

    while time.monotonic() < deadline:
        remaining = max(1, int(deadline - time.monotonic()))
        chunk = min(poll_chunk_seconds, remaining)
        updates = notifier.get_updates(
            offset=offset,
            timeout=chunk,
            allowed_updates=["callback_query"],
        )
        for update in updates:
            offset = int(update["update_id"]) + 1
            cb = update.get("callback_query")
            if not cb:
                continue
            user_id = int(cb.get("from", {}).get("id", 0))
            data = str(cb.get("data", ""))
            cb_id = str(cb.get("id", ""))

            if not allow_any_approver and user_id not in authorized_set:
                logger.warning(f"Ignoring callback from unauthorized user {user_id}")
                notifier.answer_callback_query(cb_id, "Not authorized.")
                continue
            if not data.endswith(f":{run_id}"):
                logger.info(f"Ignoring callback for stale run_id: {data}")
                notifier.answer_callback_query(cb_id, "Stale run.")
                continue

            if data.startswith("approve:"):
                notifier.answer_callback_query(cb_id, "Approved — placing orders.")
                notifier.edit_message_text(message_id, f"{message_text}\n\n✅ *Approved* by user {user_id}")
                return ApprovalOutcome(ApprovalDecision.APPROVED, approver_telegram_id=user_id)

            if data.startswith("cancel:"):
                notifier.answer_callback_query(cb_id, "Cancelled.")
                notifier.edit_message_text(message_id, f"{message_text}\n\n❌ *Cancelled* by user {user_id}")
                return ApprovalOutcome(ApprovalDecision.CANCELLED, approver_telegram_id=user_id)

    notifier.edit_message_text(message_id, f"{message_text}\n\n⏱ *Timed out* — no orders placed.")
    return ApprovalOutcome(ApprovalDecision.TIMED_OUT)
