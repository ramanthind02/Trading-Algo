# Workflow & Memory System

> [!note] See also
> [[index]] for full library map.

## Quick Start

```bash
# Default (cheap, low injection)
./scripts/oc --provider openai --preset execute

# Research (web tools + richer RAG injection)
./scripts/oc --provider openai --preset research

# Rate-limit fallback
./scripts/oc --provider zen --preset execute

# Disable RAG
./scripts/oc --preset execute --rag off
```

Venv: always `source /home/raman/repos/Trading-Algo/venv/bin/activate`

## Daily Research Loop

1. **Research** (`--preset research`) — delegate to agents: explorer maps files, librarian fetches external refs, oracle flags risks
2. **Brief** — orchestrator produces a 10-line plan: what exists, what changes, exact files, test commands
3. **Execute** (`--preset execute`) — fixer implements from the brief; run `python -m pytest <path> -q`

> [!tip] Rule of thumb
> Switch to execute as soon as you have a brief. Keep the brief as the dominant context.

## Agent Presets

| Preset | RAG injection | Tools (MCP) | Use when |
|--------|--------------|-------------|----------|
| `research` | Rich (k=5 code + k=5 facts) | Allowed | Exploration, architecture decisions |
| `execute` | Minimal (k=1 + size cap) | Off | Implementation from a clear brief |
| `dynamic` | Same as execute | Off | Mixed sessions |

Agents: `orchestrator` (default), `explorer` (repo recon), `oracle` (arch/debug), `librarian` (external refs), `fixer` (implementation).

## Memory Commands

| Command | What it does |
|---------|-------------|
| `/remember <fact>` | Stores a preference/decision in Cognitive Memory (SQLite) |
| `/recall <query>` | Searches both ChromaDB (code) + Cognitive Memory; shows results |
| `/mem` | Shows storage stats for both engines |
| `/mem clear` | Clears both stores (use when pivoting architectures) |

Good `/remember` examples:
```
/remember prefer @dataclass(frozen=True) for all domain models
/remember do not change vault artifact schemas without updating docs
/remember use python -m pytest, not bare pytest
/remember functional core, imperative shell — pure logic in pure functions
```

## RAG Architecture

- **ChromaDB** (`~/.codex/memory_db/`) — codebase chunks; auto-updated on git commit
- **Cognitive Memory** (`~/.codex/cognitive_memory.db`) — facts, preferences, decisions
- Hook: `.opencode/hooks/rag_retriever.py` runs on every message

RAG env knobs:
```bash
RAG_MODE=off|execute|research
RAG_K_CODE=<int>        # chunks from ChromaDB
RAG_K_FACTS=<int>       # facts from Cognitive Memory
RAG_MAX_CHARS=<int>     # hard cap on total injection
```

> [!warning] Stale retrieval
> After big uncommitted changes run `python3 utils/index_repo.py` to force reindex.
> Check errors at `~/.codex/memory_errors.log`.

## Superpowers Skills (On-Demand)

- New feature design → invoke `brainstorming`, then `writing-plans`
- Something failing → invoke `systematic-debugging`
- Before claiming done → invoke `verification-before-completion`

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Retrieval wrong/stale | `/recall <topic>`, reindex with `python3 utils/index_repo.py` |
| Too many tokens | `--preset execute` or `RAG_MAX_CHARS=2500` |
| Tools unavailable in execute | Switch to `--preset research` |
| Hook errors | Check `~/.codex/memory_errors.log` |
