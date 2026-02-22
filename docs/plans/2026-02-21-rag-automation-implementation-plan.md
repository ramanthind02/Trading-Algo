# RAG Automation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Automate RAG context injection into OpenCode by creating a `MessageSend` hook that performs semantic retrieval before each agent response.

**Architecture:** A shell script hook (`query-hook.sh`) triggered by OpenCode on every message, which calls a Python script (`rag_retriever.py`) to query the local ChromaDB and returns `additionalContext` in JSON format.

**Tech Stack:** Bash, Python, ChromaDB, sentence-transformers

---

## Tasks

### Task 1: Python RAG Retriever Script

**Files:**
- Create: `.opencode/hooks/rag_retriever.py`
- Test: `tests/validators/test_rag_retriever.py`

**Step 1: Write the failing test**

```python
import pytest
import json
from unittest.mock import MagicMock, patch

@patch('utils.memory_service.MemoryService')
def test_rag_retriever_output(mock_service_class):
    from opencode.hooks.rag_retriever import run_retrieval
    
    mock_service = MagicMock()
    mock_service.retrieve.return_value = [
        {"text": "Relevant snippet 1", "distance": 0.1, "metadata": {}},
        {"text": "Relevant snippet 2", "distance": 0.2, "metadata": {}}
    ]
    mock_service_class.return_value = mock_service
    
    result = run_retrieval("test query")
    data = json.loads(result)
    
    assert "hookSpecificOutput" in data
    assert "Relevant snippet 1" in data["hookSpecificOutput"]["additionalContext"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/validators/test_rag_retriever.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

Create `.opencode/hooks/rag_retriever.py`:

```python
import sys
import os
import json
from pathlib import Path

# Add project root to sys.path
REPO_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(REPO_ROOT))

from utils.memory_service import MemoryService

def run_retrieval(query: str) -> str:
    if not query:
        return json.dumps({"hookSpecificOutput": {"hookEventName": "MessageSend", "additionalContext": ""}})
    
    try:
        service = MemoryService()
        results = service.retrieve(query, k=5)
        
        if not results:
            return json.dumps({"hookSpecificOutput": {"hookEventName": "MessageSend", "additionalContext": ""}})
        
        context_parts = ["<RAG-MEMORY-CONTEXT>"]
        for i, r in enumerate(results, 1):
            context_parts.append(f"Result {i}:\n{r['text']}")
        context_parts.append("</RAG-MEMORY-CONTEXT>")
        
        context = "\n\n".join(context_parts)
        
        return json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "MessageSend",
                "additionalContext": context
            }
        })
    except Exception as e:
        # Silently fail to not break the session
        return json.dumps({"hookSpecificOutput": {"hookEventName": "MessageSend", "additionalContext": ""}})

if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else ""
    print(run_retrieval(query))
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/validators/test_rag_retriever.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add .opencode/hooks/rag_retriever.py tests/validators/test_rag_retriever.py
git commit -m "feat: add Python RAG retriever script for hooks"
```

---

### Task 2: Shell Hook Script

**Files:**
- Create: `.opencode/hooks/query-hook.sh`

**Step 1: Create the script**

```bash
#!/usr/bin/env bash
# MessageSend hook for RAG automation

set -euo pipefail

# Determine script and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Activate venv if it exists
if [ -f "${REPO_ROOT}/venv/bin/activate" ]; then
    source "${REPO_ROOT}/venv/bin/activate"
fi

# The user's query is provided in the QUERY_TEXT environment variable by OpenCode
# Fallback to empty if not set
QUERY="${QUERY_TEXT:-}"

# Run the Python retriever
# Use absolute path to python from venv if available
PYTHON_EXEC="${REPO_ROOT}/venv/bin/python3"
if [ ! -f "$PYTHON_EXEC" ]; then
    PYTHON_EXEC="python3"
fi

"$PYTHON_EXEC" "${SCRIPT_DIR}/rag_retriever.py" "$QUERY"
```

**Step 2: Make it executable**

Run: `chmod +x .opencode/hooks/query-hook.sh`

**Step 3: Test manually**

Run: `QUERY_TEXT="how does risk management work" .opencode/hooks/query-hook.sh`
Expected: JSON output with `additionalContext` (even if empty)

**Step 4: Commit**

```bash
git add .opencode/hooks/query-hook.sh
git commit -m "feat: add shell hook script for RAG automation"
```

---

### Task 3: Hook Registration

**Files:**
- Create: `.opencode/hooks/hooks.json`

**Step 1: Create the registration file**

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

**Step 2: Commit**

```bash
git add .opencode/hooks/hooks.json
git commit -m "feat: register MessageSend hook for RAG automation"
```

---

### Task 4: Documentation Update

**Files:**
- Modify: `docs/library/RAG_memory_system.md`

**Step 1: Add Automation section**

```markdown
## Automation

The RAG system is automatically integrated into OpenCode via a `MessageSend` hook. 

### How it Works
1. When you send a message, OpenCode triggers `.opencode/hooks/query-hook.sh`.
2. The hook runs `rag_retriever.py` which performs a semantic search on your query.
3. Relevant snippets are injected silently into the agent's context under `<RAG-MEMORY-CONTEXT>`.
4. This allows the agent to have project-specific context without you manually using `@` references.

### Configuration
The hook is registered in `.opencode/hooks/hooks.json`. It runs on every message by default.
```

**Step 2: Commit**

```bash
git add docs/library/RAG_memory_system.md
git commit -m "docs: update RAG system docs with automation details"
```
