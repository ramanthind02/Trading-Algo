# Local Cognitive Memory Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Integrate `OpenMemory` for cognitive/conversational memory alongside the existing ChromaDB RAG system.

**Architecture:** A hybrid memory layer where `OpenMemory` handles facts/preferences and `ChromaDB` handles repository indexing. Unified commands and automatic hook-based retrieval.

**Tech Stack:** Python, OpenMemory (SDK), SQLite, ChromaDB.

---

## Tasks

### Task 1: Install OpenMemory SDK

**Files:**
- Modify: `requirements.txt`

**Step 1: Add dependency to requirements**

```txt
openmemory-py>=0.1.0
```

**Step 2: Install dependency**

Run: `source venv/bin/activate && pip install openmemory-py`
Expected: SDK installed successfully.

**Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: add openmemory-py dependency"
```

---

### Task 2: Cognitive Memory Wrapper

**Files:**
- Create: `utils/cognitive_memory.py`
- Test: `tests/validators/test_cognitive_memory.py`

**Step 1: Write the failing test**

```python
import pytest
import asyncio
from utils.memory.cognitive_memory import CognitiveMemory

@pytest.mark.asyncio
async def test_cognitive_memory_add_search(tmp_path):
    db_path = str(tmp_path / "test_cognitive.db")
    memory = CognitiveMemory(db_path=db_path)
    
    await memory.add("User prefers async code", type="preference", tags=["coding-style"])
    results = await memory.search("coding style", k=1)
    
    assert len(results) > 0
    assert "async" in results[0]["content"]
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/validators/test_cognitive_memory.py -v`
Expected: FAIL with "ModuleNotFoundError".

**Step 3: Write implementation**

Create `utils/cognitive_memory.py`:

```python
import asyncio
import pathlib
from typing import List, Dict, Any, Optional
from openmemory.client import Memory

class CognitiveMemory:
    def __init__(self, db_path: str = "~/.codex/cognitive_memory.db"):
        resolved = pathlib.Path(db_path).expanduser()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        # OpenMemory uses SQLite by default via the Python SDK
        self.client = Memory(db_path=str(resolved))

    async def add(self, content: str, user_id: str = "default", type: str = "fact", tags: List[str] = None):
        if not content or not content.strip():
            raise ValueError("Content cannot be empty")
        await self.client.add(content, user_id=user_id, metadata={"type": type, "tags": tags or []})

    async def search(self, query: str, user_id: str = "default", k: int = 5) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            return []
        return await self.client.search(query, user_id=user_id, limit=k)

    async def stats(self) -> Dict[str, Any]:
        # Implementation depends on SDK capabilities, fallback to generic for now
        return {"engine": "OpenMemory", "storage": "SQLite"}
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/validators/test_cognitive_memory.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add utils/cognitive_memory.py tests/validators/test_cognitive_memory.py
git commit -m "feat: add CognitiveMemory wrapper for OpenMemory SDK"
```

---

### Task 3: Unified Memory Commands

**Files:**
- Modify: `utils/memory_commands.py`
- Test: Update `tests/validators/test_memory_commands.py`

**Step 1: Update `handle_remember`**

Modify `utils/memory_commands.py` to use `CognitiveMemory` for general facts. Add an async wrapper or use `asyncio.run`.

**Step 2: Update `handle_recall`**

Modify `utils/memory_commands.py` to search both `MemoryService` (ChromaDB) and `CognitiveMemory` (OpenMemory).

**Step 3: Verify with tests**

Run: `PYTHONPATH=. pytest tests/validators/test_memory_commands.py -v`
Expected: PASS.

**Step 4: Commit**

```bash
git add utils/memory_commands.py
git commit -m "feat: unify memory commands for ChromaDB and OpenMemory"
```

---

### Task 4: Dual-Database Retrieval Hook

**Files:**
- Modify: `.opencode/hooks/rag_retriever.py`

**Step 1: Update `run_retrieval`**

Enhance the Python retriever to query both databases and inject two separate blocks: `<RAG-MEMORY-CONTEXT>` and `<COGNITIVE-MEMORY-CONTEXT>`.

**Step 2: Manual Verification**

Run: `QUERY_TEXT="pydantic models" bash .opencode/hooks/query-hook.sh`
Expected: JSON containing both context blocks.

**Step 3: Commit**

```bash
git add .opencode/hooks/rag_retriever.py
git commit -m "feat: update hook to retrieve from both RAG and Cognitive memory"
```

---

### Task 5: Comprehensive Documentation

**Files:**
- Modify: `docs/library/RAG_memory_system.md`

**Step 1: Update content**

- Add "Local Cognitive Memory" section.
- Explain the split: ChromaDB (code) vs OpenMemory (facts).
- Update usage examples for combined results.

**Step 2: Commit**

```bash
git add docs/library/RAG_memory_system.md
git commit -m "docs: update memory system docs with OpenMemory details"
```
