"""Rollover-window monitor for the live FTMO + Darwinex demo nodes.

Polls each node's published live_state snapshot (the node is the sole owner of its
MT5 terminal, so we read its JSON, never touch MT5) and emits ONE line whenever
something a human would act on changes: strategy state (IDLE / AWAITING_EXIT /
AWAITING_ENTRY / HALTED), the open-position set, the halt flag, or a command ack.
Also flags node liveness (snapshot going stale = node down) and prints a phase-aware
heartbeat so silence is never ambiguous.

Broker wall-clock is host_utc - 4h (the data-client's measured broker->UTC offset;
the host system clock is ~7h fast and must NOT be trusted directly). Decisions in the
node itself use a fresh EURUSD tick; this label is only for the human countdown.
"""
import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

REPO = Path(r"c:\Users\raman\Documents\repos\Trading-Algo")
BROKERS = ("ftmo", "darwinex")
OFFSET_H = 4          # host_utc - 4h = broker EEST wall-clock (data-client offset 14400s)
POLL_S = 20
HEARTBEAT_S = 1200    # 20 min liveness/phase ping
STALE_S = 60


def snap(b):
    p = REPO / "data" / "broker_cache" / b / "live_state" / "snapshot.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def pos_str(j):
    ps = j.get("positions", []) if j else []
    return ",".join(f"{x.get('canonical')}={x.get('net_qty')}" for x in ps) or "FLAT"


def now_utc_naive():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def broker_now():
    return now_utc_naive() - timedelta(hours=OFFSET_H)


def phase(bn):
    h, m = bn.hour, bn.minute
    if h == 23 and m >= 45:
        return "EXIT-WINDOW (23:45-00:00)"
    if h == 0:
        return "DEAD-ZONE (00:00-01:00, hold flat)"
    if 1 <= h < 4:
        return "ENTRY-BAND (reopen+settle)"
    # countdown to next 23:45
    target = bn.replace(hour=23, minute=45, second=0, microsecond=0)
    if bn >= target:
        target = target + timedelta(days=1)
    mins = (target - bn).total_seconds() / 60.0
    return f"IDLE; ~{mins:.0f}min to 23:45 EXIT"


def key(j):
    if j is None:
        return None
    lcr = (j.get("last_command_result") or {}).get("id")
    ps = tuple(sorted((x.get("canonical"), round(float(x.get("net_qty", 0)), 4))
                      for x in j.get("positions", [])))
    return (j.get("strategy_state"), bool(j.get("halted")), ps, lcr)


def emit(msg):
    print(f"[broker {broker_now():%a %H:%M:%S}] {msg}", flush=True)


prev = {b: "INIT" for b in BROKERS}
prev_stale = {b: False for b in BROKERS}
last_hb = 0.0

emit("MONITOR START — watching ftmo + darwinex (EXIT 23:45 / dead-zone 00:00-01:00 / reopen ~01:00 broker)")

while True:
    nu = now_utc_naive()
    for b in BROKERS:
        j = snap(b)
        # liveness
        stale = True
        if j is not None:
            try:
                ts = datetime.fromisoformat(j["ts"])
                ts = ts.replace(tzinfo=None) if ts.tzinfo is None else ts.astimezone(timezone.utc).replace(tzinfo=None)
                stale = (nu - ts).total_seconds() > STALE_S
            except Exception:
                stale = True
        if stale and not prev_stale[b]:
            emit(f"{b.upper()} !! snapshot STALE >{STALE_S}s — node may be DOWN")
            prev_stale[b] = True
        elif not stale and prev_stale[b]:
            emit(f"{b.upper()} OK snapshot fresh again")
            prev_stale[b] = False
        # state / position / halt / command change
        k = key(j)
        if k is not None and k != prev[b]:
            state, halted, _ps, _lcr = k
            cr = j.get("last_command_result")
            extra = f" | cmd={cr.get('action')}:{cr.get('status')}" if cr else ""
            mark = "  <<< STATE CHANGE" if prev[b] != "INIT" else ""
            emit(f"{b.upper()} state={state} halted={halted} pos[{pos_str(j)}]{extra}{mark}")
            prev[b] = k
    t = time.time()
    if t - last_hb > HEARTBEAT_S:
        bn = broker_now()
        st = " ".join(f"{b}={(snap(b) or {}).get('strategy_state','??')}" for b in BROKERS)
        emit(f"alive — {phase(bn)} — {st}")
        last_hb = t
    time.sleep(POLL_S)
