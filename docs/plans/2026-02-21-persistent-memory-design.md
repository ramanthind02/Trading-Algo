# Persistent Memory Design — Local RAG for OpenCode

## Goal

Create a local Retrieval-Augmented Generation (RAG) pipeline to store and retrieve semantic memory across OpenCode sessions, reducing token costs and improving response context.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Your OpenCode Sessions                       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Memory Service (Python)                        │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────────┐  │
│  │   Chunker   │───▶│   Embedder   │───▶│  ChromaDB Store   │  │
│  │  (textsplit)│    │(sentence-    │    │ (local persist)   │  │
│  │             │    │ transformers)│    │                   │  │
│  └─────────────┘    └──────────────┘    └───────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                           │
                    Query + Retrieve
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              Retrieved Context → OpenCode Prompt                │
└─────────────────────────────────────────────────────────────────┘
```

**Design choice:** Using `sentence-transformers` (open-source) instead of OpenAI embeddings — keeps everything local, no API costs.

## Components

### 1. Memory Service (`memory_service.py`)

- **Storage class** — wraps ChromaDB client with collection management
- **Index method** — chunks text, embeds, stores in Chroma
- **Retrieve method** — embeds query, returns top-k results with metadata

### 2. Text Chunking

- **Chunk size:** 500-1000 characters (configurable)
- **Overlap:** 50-100 characters for context continuity
- **Library:** `langchain.text_splitter` or `nltk` for sentence boundaries

### 3. Embeddings

- **Model:** `all-MiniLM-L6-v2` (fast, ~384 dimensions)
- **Alternative:** `all-mpnet-base-v2` (better quality, ~768 dimensions)
- **Installed via:** `sentence-transformers`

### 4. Storage

- **Location:** `~/.codex/memory_db/` (persistent)
- **Collection:** Single collection with metadata filtering (source, date, type)

### 5. OpenCode Integration

- **Manual trigger:** Use `/remember` command to store important context
- **Auto-retrieve:** `/recall <query>` to fetch relevant memory
- **Future:** Could hook into session context if OpenCode supports custom context injection

## Commands

| Command | Description |
|---------|-------------|
| `/remember <text>` | Store text to memory |
| `/recall <query>` | Retrieve relevant context |
| `/mem stats` | Show memory stats (count, size) |
| `/mem clear` | Clear all memory (with confirmation) |

## Data Flow

1. **Store:** You call `/remember "important design decision about..."` → chunked → embedded → stored in Chroma
2. **Retrieve:** You call `/recall "how did we handle X?"` → query embedded → top-5 results returned with source/metadata → displayed in context

## Error Handling

- **Empty results:** Return "No relevant memory found" instead of error
- **ChromaDB not installed:** Show clear install instructions
- **Embedder download:** First use triggers model download (cached after)

## Dependencies

```
chromadb
langchain
langchain-community
sentence-transformers
nltk
```

## Files

- `memory_service.py` — Core service class
- `commands.py` — CLI commands for /remember, /recall, /mem
- `__main__.py` — Entry point for `python -m memory`
