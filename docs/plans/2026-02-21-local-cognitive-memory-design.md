# Local Cognitive Memory Design — OpenMemory Integration

## Goal

Integrate `OpenMemory` alongside the existing local RAG system to provide persistent "cognitive" memory (preferences, decisions, events) across OpenCode sessions, keeping all data local.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     OpenCode Agent                          │
└──────────────┬──────────────────────────────┬───────────────┘
               │                              │
        Automatic Retrieval            Manual Commands
        (MessageSend Hook)           (/remember, /recall)
               │                              │
               ▼                              ▼
┌──────────────────────────────┐      ┌───────────────────────┐
│      Query Hook Script       │      │    Memory Commands    │
└──────────────┬───────────────┘      └───────────┬───────────┘
               │                                  │
               ▼                                  ▼
┌──────────────────────────────┐      ┌───────────────────────┐
│    RAG Retriever (Python)    │      │ Unified Memory Manager│
└──────┬───────────────┬───────┘      └──────┬──────────┬─────┘
       │               │                     │          │
       ▼               ▼                     ▼          ▼
┌─────────────┐ ┌─────────────┐      ┌─────────────┐ ┌─────────────┐
│  ChromaDB   │ │ OpenMemory  │      │  ChromaDB   │ │ OpenMemory  │
│ (Repo Index)│ │ (Episodic)  │      │ (Explicit)  │ │ (Cognitive) │
└─────────────┘ └─────────────┘      └─────────────┘ └─────────────┘
```

**Design Choice:** Hybrid approach using ChromaDB for high-volume repository indexing and OpenMemory for structured cognitive recall.

## Components

### 1. Cognitive Memory Wrapper (`utils/cognitive_memory.py`)

- **Service Class**: `CognitiveMemory` wrapping the `openmemory-py` SDK.
- **Persistence**: Local SQLite at `~/.codex/cognitive_memory.db`.
- **Functionality**:
  - `add(text, type, tags)`: Stores facts/preferences with metadata.
  - `search(query, k)`: Semantic search for cognitive facts.
  - `delete(memory_id)`: Removes entries.

### 2. Unified CLI Commands (`utils/memory_commands.py`)

- **`/remember <text>`**:
  - Automatically identifies if the input is a preference/decision.
  - Default: Stores to OpenMemory.
- **`/recall <query>`**:
  - Performs parallel search in ChromaDB and OpenMemory.
  - Returns combined results labeled by source.
- **`/mem stats`**:
  - Merges statistics from both local storage systems.

### 3. Automated Retrieval Hook (`.opencode/hooks/rag_retriever.py`)

- **Enhanced Logic**:
  - On every user message, the Python retriever queries **both** databases.
  - Results are injected into the agent's prompt in two distinct blocks:
    - `<RAG-MEMORY-CONTEXT>`: Repository-level code snippets.
    - `<COGNITIVE-MEMORY-CONTEXT>`: Past decisions, preferences, and facts.

## Data Flow

1. **Storage**:
   - User: "/remember I prefer using async/await for all API calls."
   - Backend: `CognitiveMemory.add()` stores the preference in SQLite.

2. **Recall**:
   - User: "Help me write a client for the Norgate API."
   - Hook:
     - `ChromaDB` returns existing Norgate client code.
     - `OpenMemory` returns "User prefers async/await".
   - Prompt Injection: Agent receives both and implements an async client automatically.

## Error Handling

- **Independent Failures**: If OpenMemory is unavailable (e.g., DB lock), the system proceeds with ChromaDB results only (and vice versa).
- **Silent Logging**: Hook errors are captured in `~/.codex/memory_errors.log` instead of cluttering the chat.

## Benefits

| Benefit | Description |
|---------|-------------|
| **Local Privacy** | Both systems are 100% local; no cognitive data leaves the machine. |
| **Better Signal** | Distinguishes between "what is in the code" and "what the user wants". |
| **No Context Drift** | Agent remembers preferences across sessions without being reminded. |
| **Low Latency** | Optimized local SQLite and ChromaDB lookups. |

## Dependencies

- `openmemory-py`
- `sqlite3` (built-in)
- Existing RAG dependencies (chromadb, langchain, etc.)
