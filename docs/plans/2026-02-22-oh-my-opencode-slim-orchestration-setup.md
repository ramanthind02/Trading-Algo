# oh-my-opencode-slim Multi-Agent Orchestration + Local RAG/Memory

**REQUIRED SUB-SKILL:** executing-plans

## Goal

Install and configure the full oh-my-opencode-slim multi-agent suite for OpenCode, and integrate it with this repo's always-on local RAG + cognitive memory system in a way that reduces average token usage (chatty research vs batchy execution).

## Architecture

This repo already injects retrieval context on every user message via a project hook:

- OpenCode `MessageSend` hook -> `.opencode/hooks/query-hook.sh` -> `.opencode/hooks/rag_retriever.py`
- `rag_retriever.py` pulls:
  - Codebase RAG via `utils.memory.memory_service.MemoryService` (ChromaDB)
  - Cognitive Memory via `utils.memory.cognitive_memory.CognitiveMemory` (OpenMemory)

oh-my-opencode-slim adds a multi-agent orchestration layer and presets:

- Global config: `~/.config/opencode/oh-my-opencode-slim.json(.jsonc)`
- Preset switching: `OH_MY_OPENCODE_SLIM_PRESET`
- It wires roles (orchestrator/explorer/oracle/librarian/designer/fixer) to models, plus optional MCP tools and skills.

Integration point:

- Update `.opencode/hooks/rag_retriever.py` to adapt retrieval size based on `OH_MY_OPENCODE_SLIM_PRESET` (and/or `RAG_MODE`).

## Tech Stack

- OpenCode CLI (local)
- bun + bunx (installer runtime for oh-my-opencode-slim)
- Local memory stack already in this repo: ChromaDB + OpenMemory + sentence-transformers embeddings (see `docs/library/RAG_memory_system.md`)

## Plan

### 1) Install bun (if missing)

Commands (Linux):

```bash
curl -fsSL https://bun.sh/install | bash
export BUN_INSTALL="$HOME/.bun"
export PATH="$BUN_INSTALL/bin:$PATH"
bun --version
```

Expected outcome:

- `bun --version` prints successfully in the current shell.

### 2) Back up OpenCode config

Commands:

```bash
mkdir -p "$HOME/.config/opencode/backups"
cp -a "$HOME/.config/opencode/opencode.jsonc" "$HOME/.config/opencode/backups/opencode.jsonc.$(date +%Y%m%d_%H%M%S)"
test -f "$HOME/.config/opencode/oh-my-opencode-slim.json" && cp -a "$HOME/.config/opencode/oh-my-opencode-slim.json" "$HOME/.config/opencode/backups/oh-my-opencode-slim.json.$(date +%Y%m%d_%H%M%S)" || true
test -f "$HOME/.config/opencode/oh-my-opencode-slim.jsonc" && cp -a "$HOME/.config/opencode/oh-my-opencode-slim.jsonc" "$HOME/.config/opencode/backups/oh-my-opencode-slim.jsonc.$(date +%Y%m%d_%H%M%S)" || true
```

Expected outcome:

- Backups exist under `~/.config/opencode/backups/`.

### 3) Install oh-my-opencode-slim (non-interactive, cost-optimized defaults)

Commands:

```bash
bunx oh-my-opencode-slim@latest install \
  --no-tui \
  --opencode-free=yes \
  --opencode-free-model=auto \
  --tmux=no \
  --skills=yes
```

Expected outcome:

- `~/.config/opencode/oh-my-opencode-slim.json(.jsonc)` created/updated.
- `~/.config/opencode/opencode.jsonc` updated to include the oh-my plugin.

### 4) Verify multi-agent suite is live

Commands:

```bash
opencode
```

Then in OpenCode:

- `ping all agents`

Expected outcome:

- All configured agents respond (or a clear auth/model error indicates which provider/model needs configuration).

### 5) Integrate with repo RAG hook: dynamic retrieval sizing

Change:

- Update `.opencode/hooks/rag_retriever.py` to:
  - detect `OH_MY_OPENCODE_SLIM_PRESET`
  - use smaller `k` and optional truncation when preset indicates execution
  - allow overrides via `RAG_MODE`, `RAG_K_CODE`, `RAG_K_FACTS`, `RAG_MAX_CHARS`

Expected outcome:

- In research preset: current behavior (rich context).
- In execute preset: reduced injected context (lower token cost).

### 6) Add docs for workflow usage

Add:

- `docs/library/opencode_multi_agent_workflow.md`

Include:

- How to pick between chatty research vs batch execution
- How to switch `OH_MY_OPENCODE_SLIM_PRESET`
- How the local hook reacts to presets
- Recommended defaults and troubleshooting (`/recall`, `/mem`, checking `~/.codex/memory_errors.log`)

### 7) Verification

Run:

```bash
source venv/bin/activate
pytest tests/validators/test_index_repo.py -q
pytest tests/integration/test_memory_pipeline.py -q
```

Expected outcome:

- Tests pass.
