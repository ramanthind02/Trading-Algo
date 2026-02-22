# RAG Automation Design — Automatic Context Injection in OpenCode

## Goal

Integrate local RAG memory into OpenCode's workflow so relevant context is automatically retrieved and injected before each agent response — without manual `@` file references.

## Architecture

```
User Query → OpenCode Hook → Python RAG Script → Context Injection → Agent Response
```

**Design choice:** Silent context injection (agent sees context, user doesn't).

## Components

### 1. RAG Hook Script

- **Location:** `.opencode/hooks/query-hook.sh`
- **Trigger:** Runs on each user message (MessageSend event)
- **Behavior:**
  1. Receives user query via environment variable or stdin
  2. Calls Python script to run RAG retrieval
  3. Formats results as JSON with `additionalContext`
  4. Injects context silently to the agent

### 2. Hook Registration

- **File:** `.opencode/hooks/hooks.json`
- **Registers:** New hook for query/MessageSend events
- **Pattern:** Runs on every message (not just session start)

### 3. Python RAG Script

- **Location:** `.opencode/hooks/rag_retriever.py`
- **Responsibilities:**
  - Embeds the query using existing MemoryService
  - Searches ChromaDB for top-k results
  - Formats results for injection

### 4. Memory Index (existing)

- Reuses existing `utils/memory_service.py`
- ChromaDB at `~/.codex/memory_db/`

## Data Flow

```
1. User types message
2. OpenCode triggers MessageSend hook
3. Hook runs rag_retriever.py with user query
4. Python script embeds query → searches ChromaDB → returns results
5. Hook formats results as JSON with additionalContext
6. Context injected silently into agent's context
7. Agent processes query + context
8. Agent responds (user doesn't see the injected context)
```

## Configuration

### Hook Configuration (hooks.json)

```json
{
  "hooks": {
    "MessageSend": [
      {
        "matcher": "*",
        "hooks": [
          {
            "type": "command",
            "command": "${REPO_ROOT}/.opencode/hooks/query-hook.sh",
            "async": false
          }
        ]
      }
    ]
  }
}
```

### Environment Variables

- `QUERY_TEXT` — the user's message
- `OPENCODE_REPO_ROOT` — repository root for finding scripts

## Error Handling

- **RAG fails:** Return empty context, don't break agent
- **Empty results:** Return empty context, continue normally
- **Timeout:** If retrieval takes >2s, skip and continue

## Files

- Create: `.opencode/hooks/query-hook.sh` — Main hook script
- Create: `.opencode/hooks/rag_retriever.py` — Python RAG integration
- Modify: `.opencode/hooks/hooks.json` — Register new hook
- Update: `docs/library/RAG_memory_system.md` — Document automation