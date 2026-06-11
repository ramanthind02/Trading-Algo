# Agent harness — the Claude Code wiring (superseded, reference only)

> **Superseded (2026-06-07).** The day-to-day research workflow is now the **frontend app**
> (`frontend/`): build/edit a `StrategySpec` in the Spec Builder UI or have an agent write the
> JSON in `research/specs/`, then launch and inspect results from the app. The `/research` skill
> and subagents below are **left in place but no longer the recommended path**.
>
> See [[Strategy_research/user_guide]] for the current workflow.

How the research pipeline was operated through Claude Code: one **skill** as the orchestrator,
three **subagents** for isolation, **hooks** as guardrails, and deterministic **scripts** the
agents call.

> [!important]
> **Manual, in-session — no scheduling.** No Routines, cron, GitHub/API triggers, Channels, desktop
> scheduled tasks, or `/loop`. A run starts when you type `/research <brief>` and ends when you read
> the memo. The two stopping points are **you at the keyboard**, not numerical gates.

---

## What this is NOT

- **Not scheduled / autonomous.** Dropped on purpose.
- **Not gated.** No "reject if PSR < 0.95 / p > 0.05." Metrics are evidence in the memo; *you*
  decide (see [[Strategy_research/research_report]] § Philosophy). Hooks enforce *structural*
  constraints (don't touch canonical configs / the vault), never accept/reject.
- **Not a re-implementation.** It reuses the existing pipeline + `NautilusPnLEngine`. The only new
  code is the spec, the adapter, and the `analyze` step — fixed, tested, *called* by the agents,
  never regenerated per run ("code computes, the agent narrates").

---

## The four primitives

### 1. A `research` skill — the orchestrator

`.claude/skills/research/SKILL.md`, invoked manually as `/research <path-to-brief>`. It is a thin
loader, not a rulebook: it reads [[Strategy_research/README]] + the five docs as the methodology,
runs the phase sequence, enforces the two manual checkpoints, and delegates the heavy work to
subagents so the main context stays clean. The methodology stays in the docs (one source of truth);
the skill only orchestrates.

### 2. Three subagents — context isolation

Defined in `.claude/agents/*.md` (YAML frontmatter: `name`, `description`, `tools`, `model`). The
reason they matter here is concrete: an executor run emits hundreds of lines (cache population,
per-combo backtests, Nautilus fills). Isolating that means the main session **and the analyst** see
only the structured bundle, not the noise.

| Subagent | Model | Does | Tools |
|----------|-------|------|-------|
| `strategy-designer` | opus | brief → `StrategySpec` + hypothesis; studies existing nodes/sleeves for reuse, house pattern, redundancy (per [[Strategy_research/strategy_engineering]]) | Read, Grep, Glob, codegraph, Write *(spec/hypothesis only)* |
| `backtest-executor` | sonnet | spec → adapter → pipeline → Nautilus → `analyze`; tails logs; returns the bundle path | Bash, Read, Write *(run outputs)* |
| `research-analyst` | opus | bundle → memo via the six lenses (per [[Strategy_research/research_report]]); reads **computed** metrics, never estimates | Read, Write *(memo only)* |

Designer and analyst are reasoning-heavy (opus); the executor is mechanical (sonnet/haiku — saves
cost). The orchestrator only ever sees each subagent's structured return.

### 3. Hooks — structural guardrails, not delivery

In `.claude/settings.json`, a `PreToolUse` hook on `Edit|Write` that **denies** writes to:

- `research/feature/config.py`, `research/portfolio/config.py` — the adapter must build **fresh**
  config objects; the canonical configs are never mutated.
- `vault/`, `vault_personal/`, `vault_cfd_prop/` — **promotion is human-only**; the agent can never
  auto-promote. Only the explicit promote step (below) lifts this.

This turns two of the README's "hard constraints" from hope into mechanical enforcement. No `Stop`
/ delivery hooks — it's a manual, in-session workflow; you read the memo directly.

### 4. Scripts — deterministic code the agents call

Version-controlled, tested, *not* regenerated per run. The actual locations (now built):

- `research/spec/strategy_spec.py` — `StrategySpec` dataclass + construction-time validation.
- `research/spec/adapter.py` — `to_feature_config`, `apply_vol_scaling` (spec → existing configs).
- `research/spec/serialization.py` — `load_spec` / `save_spec` / `spec_from_dict` / `spec_to_dict`.
- `research/specs/*.json` — shared round-trippable JSON spec files (agent and UI share these).
- The **`analyze`** step / bundle.json was not built as a standalone script; results are consumed
  directly from the CSV/JSON artifacts via the frontend results views.

The executor calls these; it does not re-derive metrics or statistics.

---

## Workspace per run

```
research/agent_experiments/{YYYY-MM-DD}_{slug}/
  brief.md         # copy of the input brief
  hypothesis.md    # designer output — rationale, category, parsimony justification
  spec.py          # the StrategySpec instance
  run/             # pipeline + Nautilus outputs + manifest.json
  bundle.json      # analyze output — the agent-readable metrics bundle
  memo.md          # analyst output — the deliverable
```

An invented bias node goes in `nodes/experimental/{slug}.py` so the pipeline can import it.

---

## The manual flow

```
You: /research research/briefs/turnaround_tuesday.md
      │
      ▼  research skill (loads methodology, orchestrates)
      ├─▶ strategy-designer → spec.py + hypothesis.md
      │     ↳ CHECKPOINT 1: skill shows you the spec → eyeball before spending compute
      ├─▶ backtest-executor → run/ + bundle.json   (verbose output stays here)
      └─▶ research-analyst → memo.md
      ▼
You read memo.md → approve / reject
      ↳ if approved: explicit /research-promote <experiment>  → the ONE human-gated vault write
```

**Two checkpoints, both you at the keyboard:**

1. **After design** — review the proposed spec before paying for the run; catch a bad construct or
   an over-large grid early.
2. **After the memo** — read the verdict and decide. Promotion is a *separate* explicit command
   (`/research-promote`, a slash command), the only thing permitted to write to the vault.

Neither checkpoint is a numerical gate — they are human decisions, consistent with
[[Strategy_research/research_report]].

---

## Current status

All underlying scripts (#1, #2 above) are built. The `/research` skill + subagents are wired and
functional (`.claude/skills/research/`). However, the **frontend app is now the primary workflow**
— it provides a live-validated spec form, one-click run launch, and Plotly results views without
requiring a Claude Code session. The harness here remains available for agent-driven research but
is no longer the recommended day-to-day path.

---

## Optional / out of scope

- **Worktrees + parallel variants** — only for a few *genuinely distinct* hypotheses at once; in
  tension with parsimony (do not turn it into param-grid farming). Skip until wanted.
- **`claude agents` view, push notifications, remote control** — only relevant for backgrounded
  long runs; unnecessary for manual, in-session work.
- **Scheduling of any kind** — explicitly excluded.

> _Authored 2026-06-06. Updated 2026-06-07: marked superseded by the frontend app._

> _Verified against current code via CodeGraph on 2026-06-07._
