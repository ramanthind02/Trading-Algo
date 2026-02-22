# Local RAG Memory System

A self-hosted Retrieval-Augmented Generation pipeline for OpenCode sessions, reducing token costs and improving response context by storing semantic memory locally.

## Overview

This system provides persistent memory across OpenCode sessions using:
- **ChromaDB** — Local vector database for storing embeddings
- **sentence-transformers** — Open-source embeddings (all-MiniLM-L6-v2)
- **langchain** — Text chunking and orchestration

## Benefits

| Benefit | Description |
|---------|-------------|
| **Token Reduction** | Retrieve only relevant context instead of sending full history |
| **Local Storage** | No cloud costs, all data stays on your machine |
| **Semantic Search** | Find related concepts even without exact keyword matches |
| **Session Continuity** | Remember decisions and context across sessions |

## Architecture

```
User Query → Embed Query → ChromaDB → Top-K Results → Prompt
     ↑                                      ↓
Memory Store → Chunk → Embed → Vector DB ←
```

## Installation

### Dependencies

The required packages are in `requirements.txt`:

```bash
chromadb>=0.4.0
langchain>=0.1.0
langchain-community>=0.0.10
sentence-transformers>=2.2.0
nltk>=3.8.0
```

### First-Time Setup

The first time you use memory commands, it will automatically:
1. Download the embedding model (~90MB, cached after first use)
2. Create the ChromaDB storage at `~/.codex/memory_db/`

## Usage

### OpenCode Commands

| Command | Description |
|---------|-------------|
| `/remember <text>` | Store text to persistent memory |
| `/recall <query>` | Retrieve relevant context |
| `/mem` | Show memory statistics |

### Python API

```python
from utils.memory_service import MemoryService
from utils.memory_commands import handle_remember, handle_recall, handle_mem_stats

# Initialize (uses default ~/.codex/memory_db/)
service = MemoryService()

# Store content
service.store("We use ChromaDB for vector storage", metadata={"source": "design"})

# Retrieve similar content
results = service.retrieve("Which vector DB did we choose?", k=5)
for r in results:
    print(r["text"])

# Or use convenience handlers
handle_remember("Important context to remember")
handle_recall("Find context about X")
handle_mem_stats()
```

## Automation & Indexing

The system includes automation to keep the local memory synchronized with the codebase and to provide context-aware assistance.

### 1. Automatic Indexing (post-commit)

A git `post-commit` hook automatically re-indexes changed files after every commit. This ensures your local memory always reflects the current state of the repository.

- **Hook location:** `.git/hooks/post-commit` (installed via `scripts/install-hooks.sh`)
- **Behavior:** Runs `utils/index_repo.py` in the background after a commit.
- **Incremental:** Only indexes new or modified files.

### 2. Manual Indexing

You can manually trigger a full or incremental repository index using the `RepoIndexer` utility.

```bash
# From the project root
python3 utils/index_repo.py
```

This scans the repository for supported file types (`.py`, `.md`, `.txt`, `.json`, etc.) and stores their content in ChromaDB.

### 3. Automatic Context Retrieval (MessageSend hook)

The system integrates with OpenCode via a `MessageSend` hook to automatically retrieve relevant context from memory for your queries.

- **Hook script:** `.opencode/hooks/query-hook.sh`
- **Retrieval script:** `.opencode/hooks/rag_retriever.py`
- **Behavior:** On every message, the hook retrieves the top-k relevant chunks from memory and presents them to the model as context.

### 4. Setup Automation

Install the git hooks and configure the system with one command:

```bash
# From the project root
./scripts/install-hooks.sh
```

## Storage Location

- **Default:** `~/.codex/memory_db/`
- **Custom:** Pass `persist_directory` to `MemoryService()`

The storage is persistent — data survives restarts and session ends.

## How It Works

### 1. Chunking
Text is split into 500-character chunks with 50-character overlap to preserve context at boundaries.

### 2. Embedding
Each chunk is converted to a 384-dimensional vector using `all-MiniLM-L6-v2`.

### 3. Storage
Vectors + original text + metadata are stored in ChromaDB collection.

### 4. Retrieval
Query is embedded, cosine similarity search finds top-k most similar chunks.

## Best Practices

### What to Remember
- Design decisions and reasoning
- Architecture choices and trade-offs
- Important context about the codebase
- Bug fixes and their root causes

### What NOT to Remember
- Routine implementation details (the code already has these)
- Temporary debugging notes
- Information that's easily re-derived

### Query Tips
- Use natural language queries
- Focus on concepts, not just keywords
- Example: "how did we handle authentication" rather than "auth"

## Troubleshooting

### First Run Slow
First use downloads the embedding model (~90MB). Subsequent runs are fast.

### No Results
- Try different query phrasing
- Check memory has content: `/mem`
- Verify documents were stored: `/recall test` should find "test" if stored

### Empty Memory
The system starts empty. Use `/remember` to populate with useful context.

## Files

| File | Purpose |
|------|---------|
| `utils/memory_service.py` | Core MemoryService class |
| `utils/memory_commands.py` | CLI command handlers |
| `utils/index_repo.py` | Repository indexing utility |
| `scripts/install-hooks.sh` | Git hook installation script |
| `.opencode/hooks/query-hook.sh` | Context retrieval hook |
| `.opencode/hooks/rag_retriever.py` | Hook-specific retrieval logic |
| `.opencode/command/remember.md` | OpenCode /remember command |
| `.opencode/command/recall.md` | OpenCode /recall command |
| `.opencode/command/mem.md` | OpenCode /mem command |
| `tests/validators/test_memory_service.py` | Unit tests |
| `tests/validators/test_memory_commands.py` | Unit tests |
| `tests/integration/test_memory_pipeline.py` | Integration test |

## Related

- **Supermemory** — Cloud-hosted long-term memory (adds automatic session continuity)
- **opencode-skillful** — Lazy-load skills to reduce token bloat