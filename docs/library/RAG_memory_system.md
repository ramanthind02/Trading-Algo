# Hybrid RAG & Cognitive Memory System

A dual-layered persistent memory pipeline for OpenCode sessions. It combines codebase-aware retrieval with high-fidelity cognitive facts, reducing token costs and improving response context by storing semantic memory locally.

## Overview

This system provides persistent memory across OpenCode sessions using a hybrid architecture:

1.  **Codebase RAG (ChromaDB)**: Optimized for large-scale repository indexing. It stores chunks of source code, documentation, and logs. It stays in sync with your code via Git hooks.
2.  **Cognitive Memory (OpenMemory)**: Optimized for facts, preferences, and high-level decisions. It uses a structured SQLite store to maintain cross-session continuity for your instructions and preferences.

### Core Technologies
- **ChromaDB** — Local vector database for codebase indexing
- **OpenMemory** — Local SQLite-backed cognitive fact storage
- **sentence-transformers** — Open-source embeddings (`all-MiniLM-L6-v2`)
- **langchain** — Text chunking and orchestration

## Benefits

| Benefit | Description |
|---------|-------------|
| **Zero-Configuration Retrieval** | No more manual `@` file references; relevant context is injected automatically. |
| **Token Reduction** | Retrieve only relevant chunks instead of sending full files or entire conversation history. |
| **Local Privacy & Cost** | 100% local storage and open-source embeddings; no API costs or data leakage. |
| **Cognitive Continuity** | The agent remembers your "style" and "decisions" across different projects and sessions. |
| **Hybrid Synergy** | Combines precise code facts ("How is X implemented?") with persistent intent ("How do *I* want X implemented?"). |

## Architecture

```
User Query ──┬──→ [Recall Orchestrator (MessageSend Hook)]
             │           │
             │           ├──→ [Cognitive Memory (SQLite)] ──┐
             │           │      (Preferences, Decisions)     │
             │           │                                   │
             │           └──→ [Codebase RAG (ChromaDB)] ────┤
             │                  (Files, Docs, Code)          │
             │                                               ↓
             └────────────────────────────────────────→ [Enriched Prompt]
                                                        (Agent sees all)

[/remember] ──→ [Cognitive Memory Store]
[Commit]   ──→ [Incremental Indexer] ──→ [Codebase RAG Store]
```

## Installation & Setup

### 1. Dependencies
Ensure the following are in your `requirements.txt`:
```bash
chromadb>=0.4.0
openmemory-py>=1.3.2
langchain>=0.1.0
langchain-community>=0.0.10
langchain-huggingface>=0.0.1
sentence-transformers>=2.2.0
nltk>=3.8.0
```

### 2. Activate Automation
Install the Git hooks to enable automatic indexing and retrieval:
```bash
./scripts/install-hooks.sh
```

## Usage in OpenCode

### Manual Commands

| Command | Usage | Description |
|---------|-------|-------------|
| `/remember <text>` | `/remember use Pydantic V2` | Stores a high-priority fact in **Cognitive Memory**. |
| `/recall <query>` | `/recall risk scaling` | Searches **Both** stores and displays labeled results. |
| `/mem` | `/mem` | Shows statistics for both storage engines. |

### Automatic Agent Behavior
You don't need to do anything special to benefit from the memory system. On **every message** you send, the `MessageSend` hook can retrieve relevant code snippets and cognitive facts and inject them into the agent's hidden context.

This repo supports token-aware retrieval sizing:

- In execution-focused flows (default), injection is minimal.
- In research flows, injection is richer.

You can explicitly control it via environment variables:

- `RAG_MODE=off|execute|research`
- `RAG_K_CODE=<int>`
- `RAG_K_FACTS=<int>`
- `RAG_MAX_CHARS=<int>`
- `RAG_CHUNK_MAX_CHARS=<int>`
- `RAG_FACT_MAX_CHARS=<int>`

**Result:** You can ask "How should I implement the new bias node?" and the agent will know both the current node architecture (from RAG) and your preference for functional patterns (from Cognitive Memory).

## Workflow Examples

### 1. Setting Project Direction
**User:** `/remember we are moving away from manual weight files to dynamic portfolio optimization.`
**Effect:** Future answers about weighting will automatically prioritize dynamic optimization over file-based methods, even if old files still exist in the repo.

### 2. Deep Technical Query
**User:** *"How does the RSI bias node handle outliers?"*
**RAG Action:** Automatically pulls relevant sections from `utils/rsi_helpers.py` and `nodes/rsi_node.py`.
**Agent Response:** *"The RSI node uses a clipping mechanism in `rsi_helpers.py` that bounds the raw RSI between 0 and 100, and our specific implementation in `nodes/rsi_node.py` further scales this into a -1 to +1 range for the ensemble..."*

### 3. Cross-Session Continuity
**User (Session A):** `/remember always use frozen dataclasses for domain models.`
**(User starts New Session B)**
**User (Session B):** *"Create a model for a new execution order."*
**Agent:** *"Sure, I'll create a frozen dataclass as per your preference stored in memory..."*

## Cognitive Memory Best Practices

The Cognitive Memory layer is for **high-value instructions** that define your relationship with the agent.

### Detailed Examples of What to /remember:

| Category | Example Command | Purpose |
|----------|-----------------|---------|
| **Coding Style** | `/remember I prefer snake_case for all local variables.` | Ensures consistent naming without repetitive instructions. |
| **Tooling** | `/remember we use pytest-asyncio for all network-related tests.` | Avoids the agent suggesting standard `unittest`. |
| **Architecture** | `/remember all new nodes must inherit from BaseBiasNode.` | Enforces inheritance patterns. |
| **Contextual Facts** | `/remember the 'vault' directory is our source of truth for validated features.` | Prevents the agent from looking in temporary or draft folders. |
| **Workflow State** | `/remember we are currently refactoring the weight layer; ignore execution/ for now.` | Keeps the agent focused on the active task. |
| **Personal Preference** | `/remember keep explanations concise and focus on code snippets.` | Adjusts the agent's "personality" and verbosity. |

## Tips for Better Retrieval

1.  **Be Specific with /remember**: Instead of `/remember handle errors`, use `/remember use the custom 'AppError' class for all domain-level exceptions`.
2.  **Regular Maintenance**: Run `python3 utils/index_repo.py` after significant refactors if you haven't committed yet.
3.  **Search via /recall**: If you're unsure if the agent "knows" something, try `/recall <topic>`. It will show you exactly what the RAG is seeing.
4.  **Clear When Pivoting**: If you start a radically different architectural approach, use `/mem clear` to avoid old context polluting new designs.

## Maintenance & Indexing

### Automatic Sync
The system automatically updates the Codebase RAG index whenever you commit. It only re-indexes the files you changed (incremental indexing).

### Manual Re-indexing
If you've made large changes without committing, or want to force a refresh:
```bash
python3 utils/index_repo.py
```

### Clearing Memory
To clear both stores (useful when starting a completely new architectural direction):
```bash
/mem clear
```
*Note: RAG is cleared immediately; Cognitive Memory may require manual deletion of `~/.codex/cognitive_memory.db` for a factory reset.*

## Storage Locations
- **Codebase RAG (ChromaDB):** `~/.codex/memory_db/`
- **Cognitive Memory (SQLite):** `~/.codex/cognitive_memory.db`
- **Error Logs:** `~/.codex/memory_errors.log`

## Troubleshooting

- **First run is slow**: The system downloads the embedding model (~90MB) on the first use.
- **Agent doesn't seem to remember**: 
  1. Check stats with `/mem` to ensure documents are stored.
  2. Try `/recall <topic>` to see if the search returns what you expect.
  3. Re-run `./scripts/install-hooks.sh` to ensure hooks are active.
- **Hook errors**: If retrieval fails, it fails **silently** to avoid breaking your session. Check `~/.codex/memory_errors.log` for details.
