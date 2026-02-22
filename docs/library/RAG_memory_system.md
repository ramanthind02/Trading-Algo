# Hybrid RAG & Cognitive Memory System

A dual-layered persistent memory pipeline for OpenCode sessions. It combines codebase-aware retrieval with high-fidelity cognitive facts, reducing token costs and improving response context by storing semantic memory locally.

## Overview

This system provides persistent memory across OpenCode sessions using a hybrid architecture:

1.  **Codebase RAG (ChromaDB)**: Optimized for large-scale repository indexing. It stores chunks of source code, documentation, and logs.
2.  **Cognitive Memory (OpenMemory)**: Optimized for facts, preferences, and high-level decisions. It uses a graph-like structure (stored in SQLite) to maintain cross-session continuity.

### Core Technologies
- **ChromaDB** — Local vector database for codebase indexing
- **OpenMemory** — Local SQLite-backed cognitive fact storage
- **sentence-transformers** — Open-source embeddings (`all-MiniLM-L6-v2`)
- **langchain** — Text chunking and orchestration

## Benefits

| Benefit | Description |
|---------|-------------|
| **Token Reduction** | Retrieve only relevant context instead of sending full history |
| **Local Storage** | No cloud costs, all data stays on your machine |
| **Semantic Search** | Find related concepts even without exact keyword matches |
| **Hybrid Retrieval** | Combine precise codebase facts with broader cognitive context |
| **Session Continuity** | Remember specific preferences and design decisions indefinitely |

## Architecture

```
User Query ──┬──→ [Recall Orchestrator]
             │           │
             │           ├──→ [Cognitive Memory (SQLite)] ──┐
             │           │                                   │
             │           └──→ [Codebase RAG (ChromaDB)] ────┤
             │                                               ↓
             └────────────────────────────────────────→ [Combined Prompt]

[/remember] ──→ [Cognitive Memory Store (OpenMemory)]
[Commit]   ──→ [Auto-Indexer] ──→ [ChromaDB Store]
```

## Installation

### Dependencies

The required packages are in `requirements.txt`:

```bash
chromadb>=0.4.0
openmemory-py>=1.3.2
langchain>=0.1.0
langchain-community>=0.0.10
sentence-transformers>=2.2.0
nltk>=3.8.0
```

### First-Time Setup

The first time you use memory commands, it will automatically:
1. Download the embedding model (~90MB, cached after first use)
2. Create the ChromaDB storage at `~/.codex/memory_db/`
3. Initialize the Cognitive Memory SQLite database at `~/.codex/cognitive_memory.db`

## Usage

### OpenCode Commands

| Command | Description | Target Store |
|---------|-------------|--------------|
| `/remember <text>` | Store a fact or preference | **Cognitive Memory** |
| `/recall <query>` | Search both memory stores | **Both** |
| `/mem` | Show storage statistics | **Both** |

### Behavior Details

- **`/remember`**: Directs all manual input to **Cognitive Memory**. This is intended for explicit instructions like "We prefer using Pydantic for models" or "Task 5 is complete".
- **`/recall`**: Performs a parallel search across both the codebase index (ChromaDB) and the cognitive fact store (OpenMemory), presenting results from both layers to the agent.

### Python API

```python
from utils.memory_service import MemoryService
from utils.cognitive_memory import CognitiveMemory
from utils.memory_commands import handle_remember, handle_recall

# Initialize services
rag = MemoryService()
cog = CognitiveMemory()

# Manual storage (Cognitive)
handle_remember("Use functional patterns for data nodes")

# Retrieval (Hybrid)
context = handle_recall("How do we implement nodes?")
print(context)
```

## Local Cognitive Memory

The Cognitive Memory layer (powered by `OpenMemory`) is specifically designed for high-value persistent facts that should not be lost when the RAG index is cleared or rebuilt.

### What to store here:
- **Researcher Preferences**: "I prefer `match/case` over nested `if` statements."
- **Project Decisions**: "We decided to use SQLite for local state instead of JSON files."
- **Workflow State**: "Task 4 of the memory plan is currently in progress."
- **Domain Knowledge**: "The 'ES' ticker represents the S&P 500 E-mini futures."

## Automation & Indexing (Codebase RAG)

The Codebase RAG system includes automation to keep the local memory synchronized with the repository.

### 1. Automatic Indexing (post-commit)

A git `post-commit` hook automatically re-indexes changed files after every commit. This ensures your RAG memory always reflects the current state of the repository.

- **Hook location:** `.git/hooks/post-commit` (installed via `scripts/install-hooks.sh`)
- **Behavior:** Runs `utils/index_repo.py` in the background after a commit.

### 2. Manual Indexing

You can manually trigger a full or incremental repository index:

```bash
# From the project root
python3 utils/index_repo.py
```

## Storage Locations

- **Codebase RAG (ChromaDB):** `~/.codex/memory_db/`
- **Cognitive Memory (SQLite):** `~/.codex/cognitive_memory.db`

## Files

| File | Purpose |
|------|---------|
| `utils/memory_service.py` | Core RAG (ChromaDB) management |
| `utils/cognitive_memory.py` | Core Cognitive (OpenMemory) management |
| `utils/memory_commands.py` | Hybrid CLI command handlers |
| `utils/index_repo.py` | Repository indexing utility |
| `scripts/install-hooks.sh` | Hook installation script |
| `tests/validators/test_memory_service.py` | RAG unit tests |
| `tests/validators/test_cognitive_memory.py` | Cognitive unit tests |
| `tests/integration/test_memory_pipeline.py` | E2E Integration test |

## Related

- **Supermemory** — Cloud-hosted long-term memory (adds automatic session continuity)
- **OpenMemory** — The underlying engine for local cognitive facts
