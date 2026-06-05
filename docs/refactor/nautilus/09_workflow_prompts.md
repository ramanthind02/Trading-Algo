# WP-9 — Migration Workflow Prompts (Orchestrator Operating Contract)

> The reusable, WP-agnostic orchestration contract for the NautilusTrader migration.
> Block A is the operating contract (inherit it for every WP). Block B is the per-WP
> template. The migration is **gated and sequential**, so parallelism belongs in the
> read-only **discovery** and adversarial **verification** phases — never across the
> mutating edits that share one parity surface.

## Block A — Operating Contract (applies to every WP)

You are the ORCHESTRATOR. You spawn/coordinate subagents; you own every gate decision.
Read `00_overview.md` (master contract) + the relevant WP file IN FULL before planning.

**Non-negotiable invariants (00_overview §2):**
- **I1/I2** `feature_research` + `portfolio_research` near-identical before/after EVERY change. Proof = the WP-1 parity harness, not judgement.
- **I3** Deployment keeps working; live execution is migrated, not broken; cut over only after a paper-trade gate.
- **I4** `quantfoundry_core` stays the AUTHORITATIVE metrics/gating source; Nautilus analyzers are additive reference only.
- **I5** No cull/delete until the replacement passes its gate (WP-5 ledger, per-row guard).
- A change that would violate I1–I5 is OUT OF SCOPE — stop and escalate; never weaken a gate.

**FROZEN — the alpha IP (READ-ONLY; never edit to make something else work):**
`nodes/`, `features/` (was feature_selection/ — base_models→models, eda, validation),
`ensemble/` (incl. `weight_layer.py`, `weight_hierarchy.py`, `portfolio_impl/`),
`features/extraction/feature_extractor.py` math, and the `quantfoundry_core` integration
points. If the only way to land a change is to touch FROZEN → STOP and escalate.

**Parity gate (run BEFORE and AFTER any research-path change):**
- verify: `.\.venv\Scripts\python.exe -m pytest tests/parity`  (both lanes; must be `2 passed`)
- regen: `... -m pytest tests/parity -m regen` is **FORBIDDEN** without an explicit, reviewed,
  intentional behaviour change. Never silently regen to make a gate pass.
- Tolerance: exact (rtol=1e-8). Data-layer-class changes must be byte-identical.

**Codegraph-first:** confirm live call sites + blast radius with `codegraph_explore` /
`codegraph_callers` / `codegraph_impact` before reading/editing. Treat an explore result as a Read.

**STALE-PATH WARNING:** the WP docs (01–08) were written BEFORE the WP-8 reorg. They use the
OLD layout (`feature_research/`, `portfolio_research/`, `data_platform/`, `utils/cache/`).
The LIVE tree uses `research/` and `lib/` — e.g. `research.feature.in_sample.run_is`,
`research.portfolio.pipelines.portfolio_test`, `lib/cache/runtime/cache_manager.py`,
`features/extraction/feature_extractor.py`, `lib/plotting`, `lib/metrics`. `data_platform/`
itself was NOT moved (stays top-level). VERIFY every path/symbol against the codegraph index;
trust the index over the doc; record the drift in the handback.

**Environment (Windows):** `.\.venv\Scripts\python.exe` directly; never create a venv; never
run `pytest` off PATH — use `python -m pytest`.

**State file (durable):** persist `docs/refactor/nautilus/_workflow_state/{WP}.md`; append
after each phase: unit | status | parity(before/after) | frozen-touched? | evidence | notes.
This lets the run resume after a gate or a /clear.

**Concurrency tuning:** discovery is read-only and cheap → fan out WIDE. Verification runs the
cache-backed parity pytest → cap parity-running verifiers at ~4 (cache contention). Never
parallelize MUTATING edits on the same parity surface; serialize them through the gate.

## Block B — Per-WP Template (four phases)

**PARITY EXPECTATION:** `{EXACT (rtol=1e-8) | ADDITIVE/opt-in, flag-OFF parity-green | no research-path change → paper gate}`.
**ACCEPTANCE GATE:** restate the exact pass criteria from the WP file.

- **PHASE 1 — DISCOVERY (fan out WIDE, READ-ONLY).** Explorer subagents use codegraph to
  produce a manifest: exact LIVE files/symbols touched (resolve stale-path drift), blast
  radius + FROZEN adjacency, the precise parity surface + before/after command, the
  contract/shape that must not change. Output a manifest + ordered edit units, each with its
  gate-evidence requirement. CHECKPOINT: surface the manifest; in autonomous/loop mode, persist
  it to the ledger and proceed only into ADDITIVE, non-FROZEN, parity-gated edits.
- **PHASE 2 — IMPLEMENT (SERIALIZE; one worktree; additive where the WP says so).** Capture
  BEFORE parity. Implement edit units in order. Honour "additive only / no deletion". Never edit FROZEN.
- **PHASE 3 — ADVERSARIAL VERIFICATION (fan out, ≤4 parity-runners).** Reviewer subagents try
  to FALSIFY "this WP is parity-safe / paper-gate-ready and touches no FROZEN module" — hunt a
  concrete counterexample (parity diff > tol, a frozen edit, WP-specific traps). Iterate to convergence.
- **PHASE 4 — GATE DECISION (orchestrator only).** Advance ONLY if AFTER==BEFORE within tol,
  acceptance gate met, no FROZEN changed, no reviewer falsified. Else report failing evidence
  and STOP — do NOT regen to pass. Hand back edits + parity numbers + gate evidence + path-drift + runtime.

### Gate flavours
- **Exact-parity WP (WP-2 data, WP-8 cleanup):** additive, byte-identical. Phase-3 traps:
  open-vs-close timestamp mismatch, float32→64 upcast, back-adjustment/continuous-future drift,
  calendar/holiday filtering differences.
- **Gated-cutover WP (WP-4 live):** no research-path parity; paper-trade gate instead. Phase-3
  traps: order-lifecycle divergence vs current ib/mt5, broken deployment forecast path, any
  silent live cutover before the paper gate.

## Autonomous/loop mode addendum

When running unattended (user away), the Phase-1 "surface and STOP" checkpoint becomes
"persist manifest to the ledger and CONTINUE into additive, parity-gated work." The hard
invariants I1–I5 + the parity gate remain the real safety net and are enforced regardless.
STOP and leave a handback in the ledger on: any parity diff > tol, any need to touch FROZEN,
any missing prerequisite (e.g. `nautilus_trader` not installed), or a gate that won't pass.
Never regen, never delete, never weaken a gate to keep the loop moving.
