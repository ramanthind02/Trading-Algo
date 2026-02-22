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
You don't need to do anything special to benefit from the memory system. On **every message** you send:
1. The `MessageSend` hook triggers.
2. It silently retrieves the top 5 relevant code snippets from RAG.
3. It silently retrieves the top 5 relevant facts from Cognitive Memory.
4. It injects them into the agent's hidden context.

**Result:** You can ask "How should I implement the new bias node?" and the agent will know both the current node architecture (from RAG) and your preference for functional patterns (from Cognitive Memory).

## Cognitive Memory Best Practices

The Cognitive Memory layer is for **high-value instructions** that define your relationship with the agent.

### What to /remember:
- **Coding Style**: "I prefer `snake_case` for all local variables."
- **Architecture**: "All new nodes must inherit from `BaseBiasNode`."
- **Contextual Facts**: "The 'vault' directory is our source of truth for validated features."
- **Workflow State**: "We are currently refactoring the weight layer; ignore the execution folder for now."

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
