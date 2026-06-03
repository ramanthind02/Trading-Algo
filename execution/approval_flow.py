"""
Telegram-based human-in-the-loop approval via inline keyboard + long-poll.

Two entry points:

- :func:`request_approval` -- single-target approve/cancel (the legacy IB
  personal-account flow). Accepts a ``default_on_timeout`` argument and a
  ``poll_health_probe`` callable so callers can opt into "execute on timeout"
  while still failing closed when the Telegram polling path is unhealthy.

- :func:`request_batch_approval` -- multi-target approval for CFD prop-firm
  execution where one forecast fans out to N MT5 accounts. Sends one message
  with ``[✅ Approve all] [❌ Cancel all]`` plus per-account
  ``[🚫 Cancel <label>]`` rows. Returns a :class:`BatchApprovalOutcome`
  describing the decision per account.

Flow (single-target)
--------------------
1. Post a message with [Approve] / [Cancel] buttons tagged with a per-run UUID.
2. Long-poll ``getUpdates`` for ``callback_query`` updates up to the timeout.
3. Accept only a callback whose payload matches this run's UUID AND comes from
   a whitelisted Telegram user. Everything else is ignored (logged).
4. On decision, ack the callback, edit the original message to show outcome,
   return the decision.
5. On timeout: consult ``default_on_timeout`` (default ``CANCELLED`` for safety).
   If the caller opted into ``APPROVED`` on timeout, the health probe is run
   first; an unhealthy probe yields ``POLL_UNHEALTHY`` instead of approving.

Deliberately polling-only: no webhook, no public URL, no tunnel. The script
is both sender and listener for the lifetime of a single run.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Mapping, Optional, Sequence


logger = logging.getLogger(__name__)


class ApprovalDecision(str, Enum):
    # Legacy value -- user explicitly clicked APPROVE. Existing IB personal
    # callers compare against this directly; do not rename.
    APPROVED = "approved"
    # New: timer expired with default_on_timeout="approve" AND polling was
    # healthy. Audit logs MUST distinguish this from APPROVED so an unintended
    # execution is obvious in retrospect.
    APPROVED_BY_TIMEOUT = "approved_by_timeout"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    # New: the polling path was detected unhealthy (Telegram API unreachable,
    # bot token invalid, etc.) so we refuse to treat a timeout as approval.
    # Always fail-closed regardless of default_on_timeout.
    POLL_UNHEALTHY = "poll_unhealthy"
    NOT_REQUESTED = "not_requested"


# Decisions that the executor should treat as "go ahead and place orders".
APPROVED_DECISIONS = frozenset({
    ApprovalDecision.APPROVED,
    ApprovalDecision.APPROVED_BY_TIMEOUT,
})


@dataclass(frozen=True)
class ApprovalOutcome:
    decision: ApprovalDecision
    approver_telegram_id: Optional[int] = None


@dataclass(frozen=True)
class BatchApprovalOutcome:
    """Outcome for a multi-target approval (one message, N accounts).

    ``per_account`` maps account label to its individual decision.
    ``global_decision`` summarises the batch:

    - ``CANCELLED`` if user clicked "Cancel all" (every account is cancelled).
    - ``APPROVED`` if user clicked "Approve all" (every non-individually-
      cancelled account becomes APPROVED).
    - ``APPROVED_BY_TIMEOUT`` if the window expired with healthy polling and
      ``default_on_timeout=approve`` (non-cancelled accounts are
      APPROVED_BY_TIMEOUT, individually-cancelled accounts stay CANCELLED).
    - ``TIMED_OUT`` if the window expired with healthy polling and
      ``default_on_timeout=cancel`` (every non-cancelled account is TIMED_OUT).
    - ``POLL_UNHEALTHY`` if polling was unhealthy at decision time -- every
      account is POLL_UNHEALTHY regardless of default.
    """

    global_decision: ApprovalDecision
    per_account: Dict[str, ApprovalDecision]
    approver_telegram_id: Optional[int] = None


def new_run_id() -> str:
    """Opaque UUID4 used to match callbacks to this specific run."""
    return uuid.uuid4().hex[:12]


PollHealthProbe = Callable[[], bool]


def _default_poll_health_probe(notifier) -> bool:
    """Best-effort 'is the bot reachable?' check.

    Calls ``notifier.probe_health()`` if available, else falls back to True
    (back-compat for callers that don't supply a health probe).
    """
    probe = getattr(notifier, "probe_health", None)
    if probe is None:
        return True
    try:
        return bool(probe())
    except Exception:
        logger.exception("Poll health probe raised; treating as unhealthy.")
        return False


def request_approval(
    notifier,
    *,
    message_text: str,
    run_id: str,
    authorized_user_ids: List[int],
    timeout_seconds: int,
    poll_chunk_seconds: int = 25,
    allow_any_approver: bool = False,
    default_on_timeout: ApprovalDecision = ApprovalDecision.CANCELLED,
    poll_health_probe: Optional[PollHealthProbe] = None,
) -> ApprovalOutcome:
    """Post the approval message and poll for a matching callback_query.

    Returns as soon as a valid approve/cancel callback arrives, or after
    ``timeout_seconds`` elapse.

    ``default_on_timeout`` (NEW)
        - ``CANCELLED`` (default, conservative): timer expires → ``TIMED_OUT``.
        - ``APPROVED``: timer expires → ``APPROVED_BY_TIMEOUT`` *only if* the
          poll-health probe says the Telegram path was healthy. Otherwise the
          decision is ``POLL_UNHEALTHY`` and the executor must NOT proceed.
          Any other value is rejected with ``ValueError``.

    ``poll_health_probe`` (NEW)
        Optional callable returning ``True`` if Telegram is reachable. Used
        only when timing out with ``default_on_timeout=APPROVED``. Defaults
        to ``notifier.probe_health()`` if the notifier exposes one, else
        always-True.

    Whitelist behaviour
    -------------------
    By default, only ``authorized_user_ids`` may approve. An empty list
    fails closed (refuses to send the message) so we never auto-trade with
    no whitelist by accident. To intentionally accept any approver, pass
    ``allow_any_approver=True`` -- the approver's user id is still recorded
    on the outcome and audited.
    """
    if default_on_timeout not in (ApprovalDecision.CANCELLED, ApprovalDecision.APPROVED):
        raise ValueError(
            f"default_on_timeout must be CANCELLED or APPROVED, got {default_on_timeout!r}"
        )

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

    # Timeout reached -- resolve decision per default_on_timeout, with a
    # health probe gating the APPROVED-on-timeout case so a Telegram outage
    # never becomes an unintended approval.
    if default_on_timeout == ApprovalDecision.APPROVED:
        probe = poll_health_probe or (lambda: _default_poll_health_probe(notifier))
        healthy = bool(probe())
        if not healthy:
            notifier.edit_message_text(
                message_id,
                f"{message_text}\n\n🚫 *Poll unhealthy* — refusing to auto-approve. No orders placed.",
            )
            return ApprovalOutcome(ApprovalDecision.POLL_UNHEALTHY)
        notifier.edit_message_text(
            message_id,
            f"{message_text}\n\n⏱ *Timed out* — auto-approved (no answer received).",
        )
        return ApprovalOutcome(ApprovalDecision.APPROVED_BY_TIMEOUT)

    notifier.edit_message_text(message_id, f"{message_text}\n\n⏱ *Timed out* — no orders placed.")
    return ApprovalOutcome(ApprovalDecision.TIMED_OUT)


# ---------------------------------------------------------------------------
# Batch approval (multi-account, used by CFD prop-firm execution)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _BatchPayload:
    """Pre-parsed callback for a batch approval."""

    kind: str        # "approve_all", "cancel_all", "cancel_one"
    account_label: Optional[str] = None


def _parse_batch_callback(data: str, run_id: str) -> Optional[_BatchPayload]:
    """Parse a batch callback payload, or return None if it doesn't match this run."""
    # Formats:
    #   approve_all:<run_id>
    #   cancel_all:<run_id>
    #   cancel_one:<run_id>:<account_label>
    parts = data.split(":", 2)
    if len(parts) < 2:
        return None
    kind, cb_run_id = parts[0], parts[1]
    if cb_run_id != run_id:
        return None
    if kind == "approve_all":
        return _BatchPayload(kind="approve_all")
    if kind == "cancel_all":
        return _BatchPayload(kind="cancel_all")
    if kind == "cancel_one" and len(parts) == 3 and parts[2]:
        return _BatchPayload(kind="cancel_one", account_label=parts[2])
    return None


def request_batch_approval(
    notifier,
    *,
    message_text: str,
    run_id: str,
    account_labels: Sequence[str],
    authorized_user_ids: List[int],
    timeout_seconds: int,
    poll_chunk_seconds: int = 25,
    allow_any_approver: bool = False,
    default_on_timeout: ApprovalDecision = ApprovalDecision.CANCELLED,
    poll_health_probe: Optional[PollHealthProbe] = None,
) -> BatchApprovalOutcome:
    """Multi-target approval for the CFD prop-firm execution flow.

    One message with three rows of inline buttons:

      [✅ Approve all]
      [❌ Cancel all]
      [🚫 Cancel <label_1>] [🚫 Cancel <label_2>] ...

    Returns a :class:`BatchApprovalOutcome` describing the decision per
    account label. The orchestrator should only execute accounts whose
    per-account decision is in :data:`APPROVED_DECISIONS`.

    Behaviour
    ---------
    - ``Cancel all`` → returns immediately with every account cancelled.
    - ``Approve all`` → returns immediately; accounts that were individually
      cancelled before "Approve all" stay cancelled.
    - ``Cancel <label>`` → marks that account cancelled and keeps polling.
    - Timeout with ``default_on_timeout=CANCELLED`` → non-cancelled accounts
      get ``TIMED_OUT``.
    - Timeout with ``default_on_timeout=APPROVED`` AND healthy poll →
      non-cancelled accounts get ``APPROVED_BY_TIMEOUT``.
    - Timeout with ``default_on_timeout=APPROVED`` AND unhealthy poll →
      every account gets ``POLL_UNHEALTHY`` (fail-closed).
    - Empty whitelist + ``allow_any_approver=False`` → fail closed
      (every account ``CANCELLED``, no message sent).
    """
    if default_on_timeout not in (ApprovalDecision.CANCELLED, ApprovalDecision.APPROVED):
        raise ValueError(
            f"default_on_timeout must be CANCELLED or APPROVED, got {default_on_timeout!r}"
        )
    if not account_labels:
        raise ValueError("account_labels must be non-empty")

    if not allow_any_approver and not authorized_user_ids:
        logger.error("No authorized user IDs configured; refusing batch approval.")
        return BatchApprovalOutcome(
            global_decision=ApprovalDecision.CANCELLED,
            per_account={label: ApprovalDecision.CANCELLED for label in account_labels},
        )

    # Build the keyboard: row 1 approve_all, row 2 cancel_all, then one row
    # per account for cancel_one.
    rows: List[List[Dict[str, str]]] = [
        [{"text": "✅ Approve all", "callback_data": f"approve_all:{run_id}"}],
        [{"text": "❌ Cancel all", "callback_data": f"cancel_all:{run_id}"}],
    ]
    for label in account_labels:
        rows.append([{
            "text": f"🚫 Cancel {label}",
            "callback_data": f"cancel_one:{run_id}:{label}",
        }])

    # The notifier interface today accepts a flat list of buttons (one per
    # row). For richer layouts we send rows directly via a small extension
    # method when available; otherwise we fall back to flat.
    message_id = _send_with_rows(notifier, message_text, rows)
    if message_id is None:
        logger.error("Batch approval message send failed; aborting batch.")
        return BatchApprovalOutcome(
            global_decision=ApprovalDecision.CANCELLED,
            per_account={label: ApprovalDecision.CANCELLED for label in account_labels},
        )

    deadline = time.monotonic() + timeout_seconds
    offset: Optional[int] = None
    authorized_set = set(authorized_user_ids)
    label_set = set(account_labels)
    per_account: Dict[str, Optional[ApprovalDecision]] = {label: None for label in account_labels}
    last_approver: Optional[int] = None

    def _outcome_terminal(global_decision: ApprovalDecision) -> BatchApprovalOutcome:
        finalized: Dict[str, ApprovalDecision] = {}
        for label in account_labels:
            existing = per_account[label]
            if existing == ApprovalDecision.CANCELLED:
                finalized[label] = ApprovalDecision.CANCELLED
            else:
                finalized[label] = global_decision
        return BatchApprovalOutcome(
            global_decision=global_decision,
            per_account=finalized,
            approver_telegram_id=last_approver,
        )

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
                logger.warning(f"Ignoring batch callback from unauthorized user {user_id}")
                notifier.answer_callback_query(cb_id, "Not authorized.")
                continue

            parsed = _parse_batch_callback(data, run_id)
            if parsed is None:
                logger.info(f"Ignoring stale or malformed batch callback: {data}")
                notifier.answer_callback_query(cb_id, "Stale or unrecognised.")
                continue

            last_approver = user_id

            if parsed.kind == "cancel_all":
                notifier.answer_callback_query(cb_id, "All cancelled.")
                notifier.edit_message_text(
                    message_id,
                    f"{message_text}\n\n❌ *All cancelled* by user {user_id}",
                )
                for label in account_labels:
                    per_account[label] = ApprovalDecision.CANCELLED
                return BatchApprovalOutcome(
                    global_decision=ApprovalDecision.CANCELLED,
                    per_account={l: ApprovalDecision.CANCELLED for l in account_labels},
                    approver_telegram_id=user_id,
                )

            if parsed.kind == "approve_all":
                notifier.answer_callback_query(cb_id, "Approved — placing orders.")
                # Build per-account: cancelled stays cancelled, others APPROVED.
                finalized: Dict[str, ApprovalDecision] = {}
                for label in account_labels:
                    if per_account[label] == ApprovalDecision.CANCELLED:
                        finalized[label] = ApprovalDecision.CANCELLED
                    else:
                        finalized[label] = ApprovalDecision.APPROVED
                cancelled_count = sum(1 for d in finalized.values() if d == ApprovalDecision.CANCELLED)
                approved_count = len(finalized) - cancelled_count
                notifier.edit_message_text(
                    message_id,
                    f"{message_text}\n\n✅ *Approved* by user {user_id} "
                    f"({approved_count} executing, {cancelled_count} cancelled)",
                )
                return BatchApprovalOutcome(
                    global_decision=ApprovalDecision.APPROVED,
                    per_account=finalized,
                    approver_telegram_id=user_id,
                )

            if parsed.kind == "cancel_one":
                label = parsed.account_label or ""
                if label not in label_set:
                    notifier.answer_callback_query(cb_id, "Unknown account.")
                    continue
                per_account[label] = ApprovalDecision.CANCELLED
                notifier.answer_callback_query(cb_id, f"Cancelled {label}.")
                # Keep polling -- user may still approve the rest or cancel others.
                continue

    # Timeout reached.
    if default_on_timeout == ApprovalDecision.APPROVED:
        probe = poll_health_probe or (lambda: _default_poll_health_probe(notifier))
        if not bool(probe()):
            notifier.edit_message_text(
                message_id,
                f"{message_text}\n\n🚫 *Poll unhealthy* — refusing to auto-approve. No orders placed.",
            )
            return BatchApprovalOutcome(
                global_decision=ApprovalDecision.POLL_UNHEALTHY,
                per_account={label: ApprovalDecision.POLL_UNHEALTHY for label in account_labels},
                approver_telegram_id=last_approver,
            )
        notifier.edit_message_text(
            message_id,
            f"{message_text}\n\n⏱ *Timed out* — auto-approved (no answer received).",
        )
        return _outcome_terminal(ApprovalDecision.APPROVED_BY_TIMEOUT)

    notifier.edit_message_text(
        message_id,
        f"{message_text}\n\n⏱ *Timed out* — no orders placed.",
    )
    return _outcome_terminal(ApprovalDecision.TIMED_OUT)


def _send_with_rows(
    notifier,
    text: str,
    rows: List[List[Dict[str, str]]],
) -> Optional[int]:
    """Send a message with explicit button rows.

    Prefers ``notifier.send_with_inline_keyboard_rows`` if implemented;
    otherwise falls back to ``send_with_inline_keyboard`` (which historically
    placed one button per row -- we flatten while preserving row 1=approve,
    row 2=cancel ordering).
    """
    sender_rows = getattr(notifier, "send_with_inline_keyboard_rows", None)
    if callable(sender_rows):
        return sender_rows(text=text, button_rows=rows)
    flat = [btn for row in rows for btn in row]
    return notifier.send_with_inline_keyboard(text=text, buttons=flat)
