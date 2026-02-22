# Deprecated

This doc is kept for backward links.

Use the canonical workflow guide instead:

- `docs/library/opencode_trading_repo_workflow_guide.md`

---

# OpenCode + omos + Local RAG/Memory (Deep Workflow Guide)

This guide explains what is installed, how the pieces fit together, and gives practical recipes for using:

- omos (oh-my-opencode-slim) multi-agent orchestration
- presets (unlimited "modes")
- MCP + skills allow/deny lists
- background tasks
- tmux integration
- cartography codemaps
- prompt overrides
- this repo's local RAG + cognitive memory injection hook

If you only want the minimum commands, see `docs/library/opencode_multi_agent_workflow.md`.

## The Mental Model (Two Layers)

There are two independent systems shaping what the model sees.

1) omos (global, per-user)

- Lives in `~/.config/opencode/`.
- Defines 6 role agents and how they behave:
  - `orchestrator` (delegation + coordination)
  - `explorer` (codebase reconnaissance)
  - `oracle` (architecture + hard debugging)
  - `librarian` (external retrieval)
  - `designer` (UI/UX)
  - `fixer` (fast implementation)
- A "preset" is just a named mapping of (agent -> model + tool permissions).

2) This repo's local RAG/memory hook (project, per-repo)

- Lives under `.opencode/hooks/`.
- Runs on every message via OpenCode `MessageSend` hook.
- Injects local context:
  - code chunks from ChromaDB ("Codebase RAG")
  - facts/preferences from OpenMemory ("Cognitive Memory")

Key idea:

- omos controls which agent/model/tools are used.
- the repo hook controls how much local retrieval context is injected.

## Where Things Live

Global (your machine):

- OpenCode plugins: `~/.config/opencode/opencode.jsonc`
- omos config (presets/models/tools): `~/.config/opencode/oh-my-opencode-slim.json` (or `.jsonc`)
- omos prompt overrides (optional): `~/.config/opencode/oh-my-opencode-slim/`

Project (this repo):

- Hook config: `.opencode/hooks/hooks.json`
- Hook runner: `.opencode/hooks/query-hook.sh`
- Hook logic: `.opencode/hooks/rag_retriever.py`
- RAG/memory architecture doc: `docs/library/RAG_memory_system.md`

Memory storage (local):

- ChromaDB: `~/.codex/memory_db/`
- Cognitive memory (SQLite): `~/.codex/cognitive_memory.db`
- Hook errors: `~/.codex/memory_errors.log`

## Presets Are Unlimited (Not Just "2 Modes")

omos supports any number of presets. "research" and "execute" are just two named presets we created to make the cost/performance tradeoff explicit.

How preset selection works:

- If you launch with `OH_MY_OPENCODE_SLIM_PRESET=<name>`, that preset is used.
- Otherwise, the `preset` field inside `~/.config/opencode/oh-my-opencode-slim.json(.jsonc)` is used.

Examples:

```bash
OH_MY_OPENCODE_SLIM_PRESET=research opencode
OH_MY_OPENCODE_SLIM_PRESET=execute opencode
OH_MY_OPENCODE_SLIM_PRESET=premium opencode
```

## Our Current Presets (What They Actually Change)

This setup uses three presets:

- `dynamic`: safe default; low overhead; MCP tools disabled
- `execute`: low overhead; MCP tools disabled
- `research`: enables MCP tools for retrieval-focused agents

The difference is mostly tool permissions (MCP servers). Models can also differ if you choose.

In this setup:

- `research` preset enables:
  - orchestrator: `mcps: ["websearch"]`
  - librarian: `mcps: ["websearch","context7","grep_app"]`
- `execute` and `dynamic` presets keep `mcps: []` for all agents.

Why this matters:

- It prevents accidental web/search/tool calls during batch execution.
- It keeps "chatty" phases from silently ballooning cost.

## How The Local RAG/Memory Hook Tracks Presets

Your hook is always invoked, but how much it injects is preset-aware.

Behavior lives in `.opencode/hooks/rag_retriever.py`.

Defaults:

- If `OH_MY_OPENCODE_SLIM_PRESET=research`:
  - inject more: `k_code=5`, `k_facts=5` (no size cap by default)
- If `OH_MY_OPENCODE_SLIM_PRESET` is `execute`, `dynamic`, or unset:
  - inject less: `k_code=1`, `k_facts=1`, `RAG_MAX_CHARS=4000`

Hard overrides (useful in the moment):

- `RAG_MODE=off|execute|research`
- `RAG_K_CODE`, `RAG_K_FACTS`
- `RAG_MAX_CHARS` (caps total injected chars)
- `RAG_CHUNK_MAX_CHARS`, `RAG_FACT_MAX_CHARS`

Example: disable injection completely for a long “do exactly this” implementation run:

```bash
RAG_MODE=off OH_MY_OPENCODE_SLIM_PRESET=execute opencode
```

Example: keep research preset, but cap how much gets injected per message:

```bash
RAG_MODE=research RAG_MAX_CHARS=6000 OH_MY_OPENCODE_SLIM_PRESET=research opencode
```

## Provider Mixing (OpenAI + OpenCode Zen)

You can mix providers per agent.

- OpenAI models are `openai/*`
- OpenCode Zen-backed models are `opencode/*`

Check auth:

```bash
opencode auth list
```

List models:

```bash
opencode models openai --verbose
opencode models opencode --verbose
```

Edit mapping:

- `~/.config/opencode/oh-my-opencode-slim.json`

This setup currently uses:

- `orchestrator`: `openai/gpt-5.2` (backup: `opencode/kimi-2.5`)
- `oracle`: `openai/gpt-5.2-codex` (backup: `opencode/kimi-k2-thinking`)
- `explorer`: `openai/gpt-5.1-codex-mini` (backup: `opencode/minimax-2.5`)
- `librarian`: `openai/gpt-5.1-codex-mini` (backup: `opencode/gemini-3-flash`)
- `fixer`: `openai/gpt-5.1-codex-max` (backup: `opencode/qwen3-coder-480b`)

Backups are used only if we hit rate limits with our OpenAI subscription; at that point we switch over to the Zen API models.

Rationale:

- coding-heavy roles benefit from OpenAI coding models
- research roles benefit from cheap/fast long-context models

## Skills + MCP Permissions (How To Actually Use Them)

omos can control what each agent is allowed to do.

Two knobs:

- `skills`: allow/deny which skills the agent can use
- `mcps`: allow/deny which MCP servers the agent can call

Rules are allowlist-based. Patterns support:

- `[]` (none)
- `["*"]` (allow all)
- `["websearch","context7"]` (allow specific)
- `["*","!websearch"]` (allow all except one)
- `["!*"]` (deny all)

Why this matters:

- It makes "execute" truly tool-quiet.
- It prevents tool spam and reduces token burn.

## Background Tasks (Parallel Work Without Chat Spam)

omos supports running background tasks and then polling the output. This is useful when you want multiple independent investigations in parallel.

Typical pattern:

1) In `research` preset, ask the orchestrator to spin up background tasks:

- task A: librarian gathers external facts
- task B: explorer maps internal code

2) Poll results.
3) Ask orchestrator to distill into a 10-line brief.
4) Switch to `execute` preset and hand the brief to fixer.

Example prompt (what you type to the orchestrator in OpenCode chat):

```text
Run two background tasks.

Task 1 (librarian): find relevant docs/known patterns for <topic>.
Task 2 (explorer): scan this repo for where <topic> is implemented and return file paths + key functions.

Then poll both tasks and produce a short implementation brief for fixer.
```

If you need to stop one:

```text
Cancel the background task that is doing <thing>.
```


## Cartography (Codemaps) For Lower Tokens Over Time

Cartography is a bundled omos skill that generates `codemap.md` summaries and tracks changes under `.slim/`.

When to use it:

- first time on a new repo
- after a large refactor
- when you notice the agents keep re-reading the same directories

How to use it (practical approach):

1) In research preset, ask orchestrator:

```text
Run the cartography skill for this repo.
Generate/update codemaps so future questions can use those summaries instead of rereading many files.
```

2) After it runs, confirm:

- `.slim/` exists (change tracking)
- `codemap.md` files exist/updated

Then, in future tasks you can say:

```text
Before searching the whole repo, consult the relevant codemap.md.
```

## Prompt Overrides (Make Agents Behave "Your Way")

You can customize each agent prompt without forking omos by adding files under:

- `~/.config/opencode/oh-my-opencode-slim/`

Two file forms:

- `{agent}.md` replaces the prompt
- `{agent}_append.md` appends (recommended)

Recommended: start with appends so you don't lose upstream improvements.

Example: `~/.config/opencode/oh-my-opencode-slim/orchestrator_append.md`

```md
When OH_MY_OPENCODE_SLIM_PRESET is execute or dynamic:
- Do not call MCP tools unless the user explicitly requests it.
- Prefer using the repo's local retrieval via /recall if more context is needed.
- Keep output to: plan -> commands -> files changed.

When OH_MY_OPENCODE_SLIM_PRESET is research:
- Use librarian for external lookup and explorer for repo mapping.
- Distill findings into a short brief before handing work to fixer.
```

Example: `~/.config/opencode/oh-my-opencode-slim/librarian_append.md`

```md
- Always return sources/links.
- Prefer 3-7 bullet results with a short recommendation.
- If the question is repo-specific, ask explorer to locate the exact files and symbols before web searching.
```

## Recommended Daily Recipes (Concrete Examples)

### Recipe 1: "I need to understand a subsystem"

1) Launch research preset:

```bash
OH_MY_OPENCODE_SLIM_PRESET=research opencode
```

2) Ask orchestrator:

```text
Have explorer map where <subsystem> lives in this repo (files + key functions).
Have librarian retrieve external best practices/known pitfalls.
Then produce a 10-line brief: current architecture + recommended changes.
```

3) Once you have a brief, switch to execute preset:

```bash
OH_MY_OPENCODE_SLIM_PRESET=execute opencode
```

4) Hand brief to fixer:

```text
Implement the brief. Keep tool usage minimal. Use /recall only if stuck.
```

### Recipe 2: "I just want it done (low tokens)"

```bash
RAG_MODE=execute OH_MY_OPENCODE_SLIM_PRESET=execute opencode
```

Prompt:

```text
Edit these files: <paths>.
Acceptance criteria:
1) ...
2) ...
Run tests: ...
```

If you need context, explicitly call:

```text
/recall <query>
```

### Recipe 3: "This might be a rabbit hole"

Use background tasks in research preset:

```text
Start two background tasks:
1) explorer: find all references to <symbol> and summarize call graph.
2) librarian: find external references about <error/pattern>.
Poll results and produce a brief.
```

## Troubleshooting

- Agents don't respond:
  - run `ping all agents`
  - check auth: `opencode auth list`
- Web tools not working in research preset:
  - ensure `research` preset has MCPs enabled in `~/.config/opencode/oh-my-opencode-slim.json`
- Retrieval injection seems wrong:
  - check `~/.codex/memory_errors.log`
  - run `/mem` and `/recall <topic>`
  - reindex after large changes: `python3 utils/index_repo.py`
- Designer tooling (`agent-browser`) install fails:
  - keep `designer.skills: []` (recommended here)
