# Live Forecast — Complete Testing Plan

> ⚠️ Slated for rewrite under the NautilusTrader migration (WP-4 live execution). See docs/refactor/nautilus/.

End-to-end checkout for the prop-firm forecast, personal forecast (with
intraday daily-candle synthesis), and personal IB auto-execution. Run
top-to-bottom; each stage gates the next.

**Test config in place:**
- Account ID: `DUO799747` (paper)
- Whitelist: `allow_any_approver: true` (channel access = trust boundary)
- Capital: $1,000 personal / $50,000 prop

**Time references throughout this doc are PT (your local).** ET equivalents
in parentheses.

---

## Stage 0 — Prerequisites (do once, before any test)

**Hardware/software state:**
- [ ] TWS or IB Gateway running, **paper account** logged in (port 7497).
- [ ] Top-bar of TWS shows `DUO799747`.
- [ ] API enabled in TWS: *Edit → Global Configuration → API → Settings →
      "Enable ActiveX and Socket Clients"* and Socket port = `7497`.
- [ ] **Bypass order precautions for API orders**: *Edit → Global
      Configuration → API → Precautions → "Bypass Order Precautions for
      API Orders"*. Without this, TWS pops up a confirmation dialog on the
      first API order and silently rejects it (status = `Inactive`, no
      error code). Alternatively, click **Yes** on the popup the first
      time it appears — the setting persists across sessions.
- [ ] Trusted IPs include `127.0.0.1`.
- [ ] Internet up; can reach `api.telegram.org`.

**Code state:**
```bash
# Branch
git status                                    # clean, on abhi/ib-auto-execution

# Unit tests all green
.venv/Scripts/python.exe -m pytest tests/unit-tests/execution/ -q
# expected: 69 passed
```

If any unit test fails, **stop**. Don't proceed until they all pass.

---

## Stage 1 — Smoke (anytime, no Telegram, no orders, ~2 min)

Verifies the pipeline runs end-to-end. Works any time of day.

### 1a. Prop dry-run

```bash
.venv/Scripts/python.exe -m scripts.enigma_prop_forecast --dry-run --port 7497
```

**Verify in console output:**
- `Profile: prop`, `Vault: vault`, `Capital: $50,000.00`, `Port: 7497 (Paper)`
- `Required tickers:` includes `ES, GC, NQ` (others fine but those three are required).
- `Successfully fetched data for: ES, NQ, GC, RTY, TLT` (no `Failed`).
- A "Telegram message would be:" preview at the end with whole+fractional contracts.

### 1b. Personal dry-run (also confirms partial daily candle)

```bash
.venv/Scripts/python.exe -m scripts.enigma_personal_forecast --dry-run --port 7497
```

**Verify additionally:**
- A line indicating the partial daily candle synthesis happened (look for
  `Synthesiz` or `partial` / `15-min` in the log).
- Today's date appears as the last datetime in the candles for at least one
  ticker.
- "Telegram message would be:" preview shows ETFs (SPY, QQQ, GLD, IWM, DIA, TLT).

**If 1a or 1b fails:** Telegram path is irrelevant — fix the pipeline first.
Most common cause is TWS not running, port mismatch, or API permissions.

---

## Stage 2 — Telegram delivery (real messages, no orders)

### 2a. Prop → Enigma Signals - Prop Firms

Anytime. Prop pulls daily bars only.

```bash
.venv/Scripts/python.exe -m scripts.enigma_prop_forecast --port 7497
```

**Verify on Telegram (Enigma Signals - Prop Firms channel):**
- Message arrives within ~30s of script completion.
- Header reads `Prop Firms (futures)`.
- Signal strength + per-ticker fractional & rounded micro contract counts.
- Total notional looks plausible for $50K capital.

### 2b. Personal → Enigma Signals - Personal Account

**Best run during equity market hours** (06:30–13:00 PT / 09:30–16:00 ET) so
the 15-min synthesis has real data. Outside hours it still runs but the
"today" candle reflects last bar before close, which is fine for a smoke
check but won't exercise live-data behavior.

```bash
.venv/Scripts/python.exe -m scripts.enigma_personal_forecast --port 7497
```

**Verify on Telegram (Enigma Signals - Personal Account channel):**
- Header reads `Personal Account (ETFs)`.
- ETF rows for SPY, QQQ, GLD, IWM, DIA, TLT.
- Fractional shares are tiny (e.g. 0.0xxx) given $1K capital.

---

## Stage 3 — Scheduled-task wiring (one-time, then verify)

### 3a. Register both tasks (run once, as Administrator)

In **elevated** Command Prompt:

```cmd
cd C:\Users\raman\Documents\repos\Trading-Algo
deploy\setup_scheduled_task.bat
```

Verify the two tasks now exist:

```cmd
schtasks /query /tn "TradingAlgo\PropForecast" /fo LIST
schtasks /query /tn "TradingAlgo\PersonalForecast" /fo LIST
```

Each should report `Status: Ready` and `Next Run Time:` matching:
- PropForecast: today/tomorrow at **3:00 PM PT** (= 6:00 PM ET).
- PersonalForecast: today/tomorrow at **12:45 PM PT** (= 3:45 PM ET).

### 3b. Trigger each task manually (proves the scheduler env works without waiting)

```cmd
schtasks /run /tn "TradingAlgo\PropForecast"
```

In another terminal, watch the log:

```bash
tail -f deploy/forecast.log
```

You should see the same end-to-end run as Stage 2a (real Telegram sent).
Then repeat for personal:

```cmd
schtasks /run /tn "TradingAlgo\PersonalForecast"
```

**Why this matters:** the scheduler runs the task with its own working
directory and minimal environment. Most schedule-related failures
(missing venv, wrong cwd, log-file-not-writable) only show up here, not
in Stage 1/2.

### 3c. Wait for the natural fire time (final confirmation)

- Leave machine on, TWS logged in, no manual intervention.
- At 12:45 PM PT and 3:00 PM PT respectively, the tasks fire on their own.
- Verify Telegram messages arrive in both channels at the expected times.
- Tail `deploy/forecast.log` afterward to confirm clean exit.

---

## Stage 4 — IB auto-execution dry-run (no orders, market hours required)

**Run between 06:35–12:55 PT (09:35–15:55 ET).** The market-hours preflight
gate refuses to proceed otherwise.

```bash
.venv/Scripts/python.exe -m scripts.enigma_personal_forecast \
    --port 7497 --execute --dry-run-execute
```

**Verify in console output:**
- `Auto-execute: connecting trade client (client_id=3)...`
- `Managed accounts: ['DUO799747']` ← **must match config exactly**
- `Current positions: (none)` (or whatever you have)
- `Order intents: N` with one row per ETF that needs trading.
- `Preflight: all checks passed ✓`
- `--dry-run-execute set: not sending approval request, not placing orders.`

**Verify on disk:**
- `logs/execution/{date}_{run_id}.jsonl` exists with `run_start`, `intents`,
  `preflight (passed: true)`, and `run_end (status: dry_run)` records.
- No lock file written (only real placements write the lock).

**Active preflight gates (in order):**
1. **Lock file** — refuses if today's run already locked. Override: `--allow-rerun`.
2. **Live-port-requires-flag** — refuses port 7496 unless `--live` is set.
3. **Account match** — refuses if config `ib_account_id` ∉ IB `managedAccounts`.
4. **Market hours** — refuses outside 09:35–15:55 ET (IB rejects fractional shares OOH anyway).
5. **Max orders per run** — refuses batches of >10 intents (catches sizing-bug runaways).

That's it. Position-size and notional caps were removed — IB's own buying-power check is the real financial backstop, and the human-in-the-loop Telegram approval is the second one.

**Common preflight failures and what they mean:**
| Error contains | Meaning | Fix |
|----------------|---------|-----|
| `Expected account DUO799747 not in IB managedAccounts` | TWS logged into different account | log in to paper, retry |
| `outside trading window` | Run outside 09:35–15:55 ET | wait, or run Stage 1/2 instead |
| `port 7496 (LIVE) without --live` | accidentally on live port | switch back to 7497 |
| `Lock file exists` | Today's run already executed | pass `--allow-rerun` if you really want to re-execute |

> **Tip — visible intents on $1K capital.** With current sizing (~9.6%
> per ETF on a buy-hold signal), $1K capital produces ~$96 per intent.
> If you want bigger paper trades for a more thorough test, append
> `--capital 10000` (does NOT change real account size, only the sizing math).

---

## Stage 5 — IB auto-execution live (real paper orders + Telegram approval)

**Same time window as Stage 4** (market hours).

### 5a. Recommended first run: single-ticker isolation

To minimize the blast radius of the very first paper trade, temporarily edit
`vault_personal/` so only one ensemble is loaded — or set capital low enough
that only one ETF crosses the dead-band:

```bash
# Run the full flow, with approval, on $10K test sizing
.venv/Scripts/python.exe -m scripts.enigma_personal_forecast \
    --port 7497 --execute --approve-via-telegram --capital 10000
```

**Expected sequence:**
1. Console: `Auto-execute: connecting trade client...` → `Preflight: all checks passed ✓`.
2. Telegram (Personal channel): a message titled **🔔 EXECUTION APPROVAL**
   with a `run` ID, the order list, and `[✅ Approve] [❌ Cancel]` buttons.
3. **Tap ✅ Approve.** Telegram immediately edits the message to
   `✅ Approved by user <id>`.
4. Console begins printing `[1/N] SPY: PendingSubmit ...` → `Filled` per order.
5. After the last fill, Telegram receives a final **Execution complete** message
   with a per-order summary.

**Verify:**
- TWS: `Trade Log` tab shows the orders just placed against `DUO799747`.
- TWS: `Portfolio` tab shows the new positions.
- `logs/execution/{date}_{run_id}.jsonl` contains: `run_start`, `intents`,
  `preflight (passed)`, `approval (decision: approved)`, one `order_placed`
  + one `order_result` per intent, `run_end (status: done)`.
- `logs/execution/positions_latest.json` updated with new closing positions.
- `logs/execution/executed_{date}.lock` exists (prevents re-run today).

### 5b. Second run same-day (proves the lock)

```bash
.venv/Scripts/python.exe -m scripts.enigma_personal_forecast \
    --port 7497 --execute --approve-via-telegram --capital 10000
```

Should fail preflight at gate #1: `Lock file exists ... pass --allow-rerun`.
That's correct behavior.

### 5d. Cancel-path test

Run again, but tap **❌ Cancel** instead of Approve.

Expected: Telegram edits message to `❌ Cancelled by user <id>`, console
exits cleanly, no orders placed, audit log shows `approval (decision: cancelled)`,
**no lock file written** (only completed placements lock).

### 5e. Timeout-path test

Run again, do nothing for 10 minutes (config: `approval_timeout_seconds: 600`).

Expected: Telegram edits message to `⏱ Timed out — no orders placed`, console
exits cleanly, audit shows `approval (decision: timed_out)`, no lock.

---

---

## Quick-reference command table

| Goal | Command |
|------|---------|
| Smoke prop | `.venv/Scripts/python.exe -m scripts.enigma_prop_forecast --dry-run --port 7497` |
| Smoke personal | `.venv/Scripts/python.exe -m scripts.enigma_personal_forecast --dry-run --port 7497` |
| Real Telegram (prop) | `.venv/Scripts/python.exe -m scripts.enigma_prop_forecast --port 7497` |
| Real Telegram (personal) | `.venv/Scripts/python.exe -m scripts.enigma_personal_forecast --port 7497` |
| Trigger scheduled task | `schtasks /run /tn "TradingAlgo\PropForecast"` |
| Auto-execute dry-run | `... enigma_personal_forecast --port 7497 --execute --dry-run-execute` |
| Auto-execute live (paper) | `... enigma_personal_forecast --port 7497 --execute --approve-via-telegram` |
| Force-rerun today | append `--allow-rerun` |
| Test sizing on $10K | append `--capital 10000` |
| Watch logs | `tail -f deploy/forecast.log` |
| View today's audit | `ls logs/execution/$(date +%Y-%m-%d)*` |

---

## Recommended ordering — first day of testing

If today is a normal trading weekday and you want to checkout the whole
system in one session, run in this order:

1. **Stage 0** (prerequisites, ~5 min, anytime).
2. **Stage 1a + 1b** (smoke, ~5 min, anytime).
3. **Stage 2a + 2b** (real Telegram, ~5 min, ideally during market hours).
4. **Stage 3a** (register tasks, one-time, ~2 min).
5. **Stage 3b** (manual trigger of both, ~5 min, market hours).
6. **Stage 4** (auto-execute dry-run, ~3 min, market hours).
7. **Stage 5a** (first paper trade with approval, ~3 min, market hours).
8. Wait for **Stage 3c** to fire on its own at 12:45 PM PT and 3:00 PM PT.

That's ~25 minutes of active testing.

> _Verified against commit a07b6bf on 2026-06-04 (docs Phase A)._
