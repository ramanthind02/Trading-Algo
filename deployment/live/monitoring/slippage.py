"""Slippage tracking: live submit-capture joined to broker deal history.

For every magic-tagged fill we record how the actual fill price compared to the live
touch (ask for a BUY, bid for a SELL) and to the mid — the numbers that matter for
execution quality:

* ``spread_bps``        = ``(ask - bid) / mid * 1e4`` at submit
* ``slip_vs_touch_bps`` = how much WORSE than the visible touch we filled
                          (signed; ``+`` = worse, e.g. paid above the ask)
* ``slip_vs_mid_bps``   = implementation shortfall vs mid (≈ half-spread + slippage)

Two halves, joined by ``client_order_id`` (the live node sets it as the MT5 order
comment, so it lands on the deal):

* **Expected (spread + touch)** — captured LIVE by the node at order submit via
  :func:`record_submit` (the node's ``symbol_info_tick`` bid/ask). This is the ONLY
  source: these CFDs serve NO historical ticks (``copy_ticks_range`` returns empty),
  so the spread cannot be reconstructed after the fact.
* **Actual fill** — read from the broker's own ``history_deals_get`` (the source of
  truth, magic-filtered) by :func:`read_fills`, which joins each deal to its submit.

All times are **broker wall-clock** (``deal.time`` is broker epoch — never the host
clock). A deal with no matching submit (manual / pre-capture) keeps its fill price but
leaves the slippage fields ``None``.

CONTENTION WARNING
------------------
:func:`read_fills` opens its OWN MT5 connection (one ``history_deals_get`` — light, no
tick queries). A live vault node holds the same terminal, so still prefer to run the
reader when the node is STOPPED or idle; the CLI refuses against a live node unless
forced. ``record_submit`` runs INSIDE the node (no extra connection).
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

from data_platform.providers.mt5 import brokers
from deployment.live.monitoring.live_state import live_state_dir

_EPOCH = datetime(1970, 1, 1)  # naive; broker deal/tick time is broker wall-clock as epoch

CSV_FIELDS = (
    "ticket", "broker", "canonical", "symbol", "side", "entry", "volume",
    "broker_time", "fill_px", "bid", "ask", "mid", "spread_bps",
    "expected_touch_px", "slip_vs_touch_bps", "slip_vs_mid_bps",
    "commission", "swap", "order", "client_order_id",
)


@dataclass(frozen=True)
class SlipRow:
    ticket: int
    broker: str
    canonical: str
    symbol: str
    side: str            # BUY | SELL
    entry: str           # IN | OUT | INOUT | ?
    volume: float
    broker_time: str     # ISO broker wall-clock
    fill_px: float
    bid: float | None
    ask: float | None
    mid: float | None
    spread_bps: float | None
    expected_touch_px: float | None
    slip_vs_touch_bps: float | None
    slip_vs_mid_bps: float | None
    commission: float
    swap: float
    order: int
    client_order_id: str = ""


def _broker_dt(epoch: int | float) -> datetime:
    return _EPOCH + timedelta(seconds=int(epoch))


def _canonical_for(broker: str, symbol: str) -> str:
    """Reverse-map a native broker symbol to the repo-canonical ticker (best effort)."""
    for canonical in ("ES", "NQ", "GC", "CL", "SI", "TLT"):
        try:
            if brokers.resolve(broker, canonical) == symbol:
                return canonical
        except (ValueError, KeyError):
            continue
    return symbol


def _entry_label(entry: int) -> str:
    return {0: "IN", 1: "OUT", 2: "INOUT"}.get(int(entry), "?")


# ── live submit-capture (the only source of CFD spread / expected fill) ───────

def submits_jsonl_path(broker: str) -> Path:
    return Path(live_state_dir(broker)).parent / "slippage" / "submits.jsonl"


def record_submit(broker: str, *, canonical: str, side: str, qty: float,
                  bid: float, ask: float, client_order_id: str, broker_time_iso: str,
                  intent: str | None = None,
                  target_fraction: float | None = None) -> None:
    """Append the LIVE quote observed at order submit.

    The node is the ONLY place the bid/ask is observable for these CFDs (the brokers
    serve no historical ticks, so post-hoc reconstruction is impossible). Keyed by
    ``client_order_id`` so the deal reader can join it to the actual fill price.
    Append-only JSONL; the caller wraps this so a capture failure never breaks trading.

    ``intent`` (``'EXIT'`` or ``'ENTRY'``) and ``target_fraction`` (the position
    fraction being targeted) are included in the record when provided; absent kwargs
    leave the legacy ``{client_order_id, canonical, side, qty, bid, ask, broker_time}``
    shape unchanged for backward compatibility.
    """
    path = submits_jsonl_path(broker)
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "client_order_id": client_order_id, "canonical": canonical, "side": side,
        "qty": qty, "bid": bid, "ask": ask, "broker_time": broker_time_iso,
    }
    if intent is not None:
        rec["intent"] = intent
    if target_fraction is not None:
        rec["target_fraction"] = target_fraction
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def load_submits(broker: str) -> dict[str, dict]:
    """``{client_order_id -> submit record}`` from the submit-log (empty if absent)."""
    path = submits_jsonl_path(broker)
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if isinstance(r, dict) and r.get("client_order_id"):
            out[str(r["client_order_id"])] = r
    return out


def read_fills(broker: str, since: datetime, until: datetime | None = None, *, mt5=None) -> list[SlipRow]:
    """Read magic-tagged trade fills in ``[since, until]`` (broker wall-clock) and
    compute slippage for each. Opens + closes its own MT5 connection.
    """
    if mt5 is None:
        import MetaTrader5 as mt5  # noqa: PLC0415
    path = brokers.terminal_path(broker)
    magic = int(brokers.execution_rules(broker).magic_number)
    submits = load_submits(broker)  # client_order_id -> live quote captured at submit
    if not mt5.initialize(path=str(path)):
        raise RuntimeError(f"MT5 initialize failed for {broker}: {mt5.last_error()}")
    try:
        until = until or (_broker_dt(int(mt5.symbol_info_tick("EURUSD").time)) + timedelta(minutes=5))
        deals = mt5.history_deals_get(since, until)
        if deals is None:
            raise RuntimeError(f"history_deals_get returned None: {mt5.last_error()}")
        rows: list[SlipRow] = []
        for d in sorted(deals, key=lambda x: x.time):
            if getattr(d, "magic", None) != magic or d.type not in (mt5.DEAL_TYPE_BUY, mt5.DEAL_TYPE_SELL):
                continue
            side = "BUY" if d.type == mt5.DEAL_TYPE_BUY else "SELL"
            # Join the deal to the live quote captured at submit (deal.comment == client_order_id).
            # CFDs serve no historical ticks, so this is the only spread/expected source; a deal
            # with no matching submit (manual / pre-capture) leaves the slippage fields None.
            coid = str(getattr(d, "comment", "") or "")
            sub = submits.get(coid)
            bid = ask = mid = spread_bps = touch = slip_touch = slip_mid = None
            if sub is not None:
                bid, ask = sub.get("bid"), sub.get("ask")
                if bid and ask and bid > 0 and ask > 0:
                    mid = (bid + ask) / 2.0
                    spread_bps = (ask - bid) / mid * 1e4
                    touch = ask if side == "BUY" else bid
                    slip_touch = ((d.price - touch) if side == "BUY" else (touch - d.price)) / mid * 1e4
                    slip_mid = ((d.price - mid) if side == "BUY" else (mid - d.price)) / mid * 1e4
            rows.append(SlipRow(
                ticket=int(d.ticket), broker=broker, canonical=_canonical_for(broker, d.symbol),
                symbol=d.symbol, side=side, entry=_entry_label(getattr(d, "entry", -1)),
                volume=float(d.volume), broker_time=_broker_dt(d.time).isoformat(),
                fill_px=float(d.price), bid=bid, ask=ask, mid=mid, spread_bps=spread_bps,
                expected_touch_px=touch, slip_vs_touch_bps=slip_touch, slip_vs_mid_bps=slip_mid,
                commission=float(getattr(d, "commission", 0.0)), swap=float(getattr(d, "swap", 0.0)),
                order=int(getattr(d, "order", 0)), client_order_id=coid,
            ))
        return rows
    finally:
        mt5.shutdown()


def slippage_csv_path(broker: str) -> Path:
    # Sibling of live_state/ under the broker root (NOT the central_cache signal namespace).
    return Path(live_state_dir(broker)).parent / "slippage" / "slippage.csv"


def append_rows(broker: str, rows: list[SlipRow]) -> int:
    """Append new rows (idempotent by deal ticket). Returns the count actually added."""
    path = slippage_csv_path(broker)
    path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[int] = set()
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            seen = {int(r["ticket"]) for r in csv.DictReader(f) if r.get("ticket")}
    new = [r for r in rows if r.ticket not in seen]
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            w.writeheader()
        for r in new:
            w.writerow({k: asdict(r)[k] for k in CSV_FIELDS})
    return len(new)


def load_rows(broker: str) -> list[dict]:
    path = slippage_csv_path(broker)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def summarize(rows: list[dict]) -> dict:
    """Per-canonical + overall slippage/spread stats from CSV rows."""
    def stats(vals: list[float]) -> dict:
        vals = sorted(v for v in vals if v is not None)
        if not vals:
            return {"n": 0}
        n = len(vals)
        mean = sum(vals) / n
        median = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
        return {"n": n, "mean": round(mean, 2), "median": round(median, 2),
                "min": round(vals[0], 2), "max": round(vals[-1], 2)}

    out: dict = {"overall": {}, "by_canonical": {}}
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r.get("canonical", "?"), []).append(r)
    for metric in ("spread_bps", "slip_vs_touch_bps", "slip_vs_mid_bps"):
        out["overall"][metric] = stats([_fnum(r.get(metric)) for r in rows])
    for c, rs in sorted(groups.items()):
        out["by_canonical"][c] = {
            m: stats([_fnum(r.get(m)) for r in rs])
            for m in ("spread_bps", "slip_vs_touch_bps", "slip_vs_mid_bps")
        }
    return out


__all__ = [
    "CSV_FIELDS", "SlipRow", "submits_jsonl_path", "record_submit", "load_submits",
    "read_fills", "slippage_csv_path", "append_rows", "load_rows", "summarize",
]
