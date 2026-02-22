# Trading-Algo: OpenCode Workflow Guide

Use this repo with OpenCode via the launcher script. It picks the right preset (OpenAI vs Zen) so you do not have to remember env vars.

## Start Here (Copy/Paste)

Default (cheap + quiet):

```bash
./scripts/oc --provider openai --preset execute
```

Research (web tools allowed for research roles):

```bash
./scripts/oc --provider openai --preset research
```

Rate-limit fallback (Zen models):

```bash
./scripts/oc --provider zen --preset execute
```

RAG toggle (optional):

```bash
./scripts/oc --provider openai --preset execute --rag off
```

## Daily Loop

- Research: ask `@orchestrator` to delegate to `explorer` (repo mapping), `librarian` (external retrieval), `oracle` (architecture/debugging).
- Plan: produce a brief with exact file paths + invariants + `python -m pytest ...` commands.
- Execute: use `--preset execute` and implement from the brief.

## 0) What Is Actually Running (The Two Layers)

Layer A: omos (global, your machine)

- Defines 6 role agents: orchestrator / explorer / oracle / librarian / designer / fixer.
- Defines presets (unlimited named profiles) that map:
  - which model each agent uses
  - which tools (MCP servers) each agent is allowed to call
  - which skills each agent is allowed to invoke

Layer B: This repo’s local RAG + memory hook (project, per-repo)

- Runs on every message via OpenCode `MessageSend` hook.
- Injects:
  - Codebase RAG snippets (from `~/.codex/memory_db/`)
  - Cognitive facts/preferences (from `~/.codex/cognitive_memory.db`)
- The hook is preset-aware (you can keep injection small during execution).

## 0.1) Why You Only See build / plan / orchestrator In Autocomplete

OpenCode has built-in primary agents (in your install you’ll see `build` and `plan`).

omos also defines role profiles (oracle/explorer/librarian/fixer/designer) in `~/.config/opencode/oh-my-opencode-slim.json`, but those roles are typically invoked by delegation ("use explorer"), not selected as top-level chat agents.

So the mental model is:

- You select a primary agent you can talk to directly (commonly `@build`, `@plan`, `@orchestrator`).
- You ask it to delegate to omos roles when you need specialized work.

To confirm what OpenCode considers "primary agents" on your machine:

```bash
opencode agent list
```

## 1) One-Time Setup Checklist

Copy/paste this into OpenCode (or do it yourself in a shell). The goal is “agents respond + retrieval works + tests runnable”.

Prompt to Orchestrator:

```text
Verify my OpenCode environment for this repo.

Checklist:
1) Confirm omos is installed and `ping all agents` works.
2) Confirm auth is present for OpenAI + OpenCode Zen.
3) Confirm the repo MessageSend hook is installed and runs (no errors).
4) Confirm the shared venv is used (`source venv/bin/activate`) and `python -m pytest --version` works.

If any step fails, give me the smallest fix and how to verify it.
```

Helpful manual commands (if you prefer):

```bash
opencode auth list
opencode models openai --verbose
opencode models opencode --verbose
```

## 2) The Only Two Questions You Ask Before Any Task

Question A: Is this “research” or “execution” right now?

- Research (chatty): you want exploration, comparisons, background tasks, and external retrieval.
- Execute (batchy): you know what to change and want low overhead.

Question B: Do I want retrieval injection right now?

- Usually yes, but minimal during execution.
- If it’s getting in the way, turn it down or off.

## 3) Presets (Unlimited Modes)

You can have as many presets as you want. This repo currently uses:

- `research`: external tools allowed (MCP) + richer local injection
- `execute`: tools off + minimal local injection
- `dynamic`: treated like execute for injection (low overhead)

How to launch: use `./scripts/oc`.

If you want more than these, add presets in `~/.config/opencode/oh-my-opencode-slim.json(.jsonc)`.

## 3.1) Current OSOS Model Mapping (Primary + Rate-Limit Backup)

Primary models are the default mapping. Backup models are used only if we hit rate limits with the OpenAI subscription; in that case we switch over to the Zen API models by changing presets/config in `~/.config/opencode/oh-my-opencode-slim.json(.jsonc)`.

- `orchestrator`: `openai/gpt-5.2` (backup: `opencode/kimi-2.5`)
- `oracle`: `openai/gpt-5.2-codex` (backup: `opencode/kimi-k2-thinking`)
- `explorer`: `openai/gpt-5.1-codex-mini` (backup: `opencode/minimax-2.5`)
- `librarian`: `openai/gpt-5.1-codex-mini` (backup: `opencode/gemini-3-flash`)
- `fixer`: `openai/gpt-5.1-codex-max` (backup: `opencode/qwen3-coder-480b`)

### Launcher Script (Provider Flag)

Instead of manually setting `OH_MY_OPENCODE_SLIM_PRESET`, use the repo launcher:

```bash
./scripts/oc --provider openai --preset execute
./scripts/oc --provider zen --preset execute
./scripts/oc --provider zen --preset research
```

This maps to `OH_MY_OPENCODE_SLIM_PRESET` for you (for example: `execute` -> `execute_zen`).

Escape hatch (advanced): you can still set `OH_MY_OPENCODE_SLIM_PRESET` directly, but the launcher should be the default.

## 4) Local RAG/Memory Controls (Token Guardrails)

The hook logic is `.opencode/hooks/rag_retriever.py`.

Defaults:

- `research` preset: inject more (k=5 code + k=5 facts)
- `execute`/`dynamic`/unset: inject less (k=1 code + k=1 fact + size cap)

Hard overrides (use any time):

```bash
./scripts/oc --rag off
./scripts/oc --rag execute
./scripts/oc --rag research
```

Fine tuning knobs:

- `RAG_K_CODE`, `RAG_K_FACTS`
- `RAG_MAX_CHARS` (cap total injection)
- `RAG_CHUNK_MAX_CHARS`, `RAG_FACT_MAX_CHARS` (cap each item)

If injection feels “too chatty” while you ship, do this:

```bash
RAG_MAX_CHARS=2500 ./scripts/oc --preset execute
```

## 4.1) Why Execute Injects Less (Builders Still Get Context)

In implementation you often have many turns (read files, edit, run tests, fix, repeat). If you inject 5 code chunks + 5 facts on every message, you pay that cost every turn, and you increase the chance the model follows irrelevant retrieved chunks instead of your plan.

Execute mode is designed to keep the plan/spec as the dominant context, while still providing a small orientation injection.

How builders get the rest of the context in execute mode:

- They `Read` the specific files named in the plan (highest-signal context).
- They `/recall <query>` when they need extra repo knowledge (naming conventions, schemas, invariants).
- They `Grep` when they need to locate symbols.

If you want to make this explicit, include a “Context Bootstrap” in your plan:

```text
Context Bootstrap (do this first):
- Read: <file A>
- Read: <file B>
- /recall <query 1>
- /recall <query 2>
```

If a task truly benefits from more automatic injection, you can bump it temporarily:

```bash
OH_MY_OPENCODE_SLIM_PRESET=execute RAG_K_CODE=3 RAG_K_FACTS=2 RAG_MAX_CHARS=8000 opencode
```

## 5) Memory Commands You Actually Use

Your memory system is documented in `docs/library/RAG_memory_system.md`.

Use these in OpenCode chat:

- `/remember <fact>`: stores a high-value preference/decision (Cognitive Memory)
- `/recall <query>`: shows what RAG + cognitive memory return
- `/mem`: shows stats

Good `/remember` examples for this repo:

```text
/remember For all new domain models, prefer @dataclass(frozen=True).
/remember Do not change vault artifact schemas without updating docs + readers.
/remember Prefer python -m pytest over bare pytest.
/remember Use functional core, imperative shell (pure logic in pure functions; IO at boundaries).
```

Use `/recall` when you suspect the agent is missing something:

```text
/recall bias node naming convention
/recall risk scaling formula
/recall ensemble weight layer schema
```

## 6) Which Agent For What (Practical)

- Orchestrator: your default. Delegates and keeps the plan coherent.
- Explorer: repo reconnaissance (where is X implemented? what files matter?).
- Librarian: external retrieval (papers, docs, library APIs).
- Oracle: big architectural decisions, deep debugging, “this might be wrong” checks.
- Fixer: fast implementation from a clear brief.
- Designer: UI/UX (rare in this repo).

## 7) Superpowers Skills (Used On-Demand)

You likely have the Superpowers skills installed as skills, but not the always-on plugin. That’s good for token cost.

Use this rule:

- If you’re about to design a new feature: invoke `brainstorming`, then `writing-plans`.
- If something is failing: invoke `systematic-debugging`.
- Before you claim done: invoke `verification-before-completion`.

Copy/paste prompts:

Feature design:

```text
Invoke the Superpowers `brainstorming` skill.
Goal: <your goal>.
Constraints: keep token use low; prefer local RAG (/recall) over web.
Ask one question at a time.
Then present 2-3 approaches and a recommended design.
```

Implementation plan (after design approval):

```text
Invoke the Superpowers `writing-plans` skill.
Create an implementation plan with:
- exact files to edit
- test commands (use python -m pytest)
- verification steps
Save the plan under docs/plans/.
```

Debugging:

```text
Invoke the Superpowers `systematic-debugging` skill.
Reproduce the failure, gather evidence (logs, failing tests), then propose the smallest fix.
```

Verification:

```text
Invoke `verification-before-completion`.
Run the smallest relevant test set first, then the broader suite if needed.
Do not claim success without command output evidence.
```

## 8) The Core Workflow Pattern (Research -> Brief -> Execute)

This pattern keeps you fast and cheap.

Step 1: Research preset

```bash
./scripts/oc --preset research
```

Step 2: Ask orchestrator to produce a brief by delegating

```text
Task: <what you want to change>.

Do this:
1) Have explorer identify the exact files, functions, and data flow involved.
2) Use /recall for any repo-specific facts (naming conventions, schema, patterns).
3) If truly needed, have librarian retrieve external references.
4) Produce a 10-line brief:
   - what exists today
   - what we will change
   - what tests will prove it
Then hand that brief to fixer.
```

Step 3: Switch to execute preset and implement

```bash
./scripts/oc --preset execute
```

```text
Use the brief below as the source of truth.
Implement it with minimal tool usage.
Run the listed tests via python -m pytest.

<paste brief>
```

## 9) Copy/Paste Prompts (One Set)

Use this for most tasks:

```text
Task: <what you want to change>.

1) Explorer: identify exact files/functions and the call/data flow.
2) /recall: any repo-specific invariants/schemas that matter.
3) Librarian: external references only if needed.
4) Oracle: highlight risks/invariants; what must not silently change.

Return a 10-line brief:
- what exists today
- what we will change
- exact files to edit
- tests to run (python -m pytest ...)
```

## 10) Cartography (Codemaps)

Use cartography when the agents keep re-reading the repo or when you start a new large feature.

Prompt:

```text
Run the `cartography` skill for this repo.
Goal: generate/update codemap.md summaries so future tasks can reference them.
After running, point me to the key codemap.md files for nodes/ensemble/execution/deployment.
```

## 11) Background Tasks (Parallelize Research)

Use this when you’re about to go down a rabbit hole.

Prompt:

```text
Start two background tasks.

Task A (explorer): locate where <topic> lives in this repo and summarize the call/data flow.
Task B (librarian): gather external references for <topic> (only if needed).

Poll until both complete. Then distill into a 10-line brief for fixer.
```

## 12) Tmux Integration (See What Agents Are Doing)

If you want live panes:

1) Enable tmux in your omos config `~/.config/opencode/oh-my-opencode-slim.json(.jsonc)`.
2) Launch OpenCode inside tmux and ensure the port matches `OPENCODE_PORT`.

Prompt to Orchestrator:

```text
Help me enable tmux integration for omos.

I want:
- layout: main-vertical
- a large main pane

Give me the exact config snippet to add to ~/.config/opencode/oh-my-opencode-slim.jsonc and the exact launch command.
```

## 13) Prompt Overrides (Make The Agents Behave Like You)

Recommended: append-only overrides.

Prompt:

```text
Create append-only prompt overrides for orchestrator and librarian.

Goals:
- In execute/dynamic: do not use MCP tools unless explicitly requested.
- Prefer /recall (local) before websearch.
- Always produce a short brief before handing work to fixer.

Give me the exact file paths and contents.
```

Paths:

- `~/.config/opencode/oh-my-opencode-slim/orchestrator_append.md`
- `~/.config/opencode/oh-my-opencode-slim/librarian_append.md`

## 14) Verification Commands (Trading Repo Defaults)

When you run tests, prefer:

```bash
source venv/bin/activate
python -m pytest <path> -q
```

If you changed cross-layer behavior (nodes -> ensemble -> execution), include the relevant integration tests under `tests/integration/`.

## 15) Troubleshooting

- Retrieval seems wrong/stale:
  - `/mem`, `/recall <topic>`, check `~/.codex/memory_errors.log`
  - reindex after big uncommitted changes: `python3 utils/index_repo.py`
- Tools not available in execute mode:
  - switch to research preset or enable MCPs in your preset config
- Too many tokens:
  - `./scripts/oc --preset execute`
  - `RAG_MAX_CHARS=2500` or `RAG_MODE=off`
