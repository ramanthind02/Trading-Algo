# Cognitive Memory Wrapper Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a robust `CognitiveMemory` wrapper around the `openmemory` SDK with SQLite persistence.

**Architecture:** A thin wrapper class `CognitiveMemory` that configures the `openmemory` environment (DB URL) and delegates to its `Memory` client. It uses `sqlite:///` for local persistence in `~/.codex/cognitive_memory.db`.

**Tech Stack:** Python, `openmemory` SDK (v1.3.2), `sqlite`, `pytest`, `pytest-asyncio`.

---

### Task 1: Environment Setup & Directory Creation

**Files:**
- Create: `utils/` (if not exists)
- Create: `tests/validators/` (if not exists)

**Step 1: Create directories**

Run: `mkdir -p utils tests/validators`
Expected: Directories created.

**Step 2: Verify `pytest-asyncio` installation**

Run: `source venv/bin/activate && pip list | grep pytest-asyncio`
Expected: `pytest-asyncio` in list. If not, run `pip install pytest-asyncio`.

---

### Task 2: Implement CognitiveMemory Wrapper (TDD)

**Files:**
- Create: `utils/cognitive_memory.py`
- Create: `tests/validators/test_cognitive_memory.py`

**Step 1: Write the failing test for initialization**

```python
import pytest
import pathlib
from utils.memory.cognitive_memory import CognitiveMemory

def test_cognitive_memory_init(tmp_path):
    db_path = str(tmp_path / "test.db")
    memory = CognitiveMemory(db_path=db_path)
    assert pathlib.Path(db_path).exists()
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/validators/test_cognitive_memory.py -k test_cognitive_memory_init -v`
Expected: FAIL (ModuleNotFound or NameError)

**Step 3: Write minimal implementation for `__init__`**

```python
import pathlib
from openmemory.client import Memory
from openmemory.core.config import env
from openmemory.core.db import db

class CognitiveMemory:
    def __init__(self, db_path: str = "~/.codex/cognitive_memory.db"):
        resolved = pathlib.Path(db_path).expanduser()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        
        db_url = f"sqlite:///{resolved}"
        env.database_url = db_url
        db.connect()
        
        # Ensure the file is created by the DB connection
        resolved.touch(exist_ok=True)
        
        self.client = Memory()
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/validators/test_cognitive_memory.py -k test_cognitive_memory_init -v`
Expected: PASS

**Step 5: Write failing test for `add` and `search`**

```python
@pytest.mark.asyncio
async def test_cognitive_memory_add_search(tmp_path):
    db_path = str(tmp_path / "test_add.db")
    memory = CognitiveMemory(db_path=db_path)
    
    await memory.add("test content", doc_type="fact", tags=["test"])
    results = await memory.search("test")
    
    assert len(results) > 0
    assert results[0]["content"] == "test content"
```

**Step 6: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/validators/test_cognitive_memory.py -k test_cognitive_memory_add_search -v`
Expected: FAIL (AttributeError: 'CognitiveMemory' object has no attribute 'add')

**Step 7: Implement `add` and `search`**

```python
from typing import List, Dict, Any, Optional

# Inside CognitiveMemory class:
    async def add(self, text: str, user_id: str = "default", doc_type: str = "fact", tags: Optional[List[str]] = None):
        if not text or not text.strip():
            raise ValueError("text cannot be empty")
        metadata = {"type": doc_type, "tags": tags or []}
        return await self.client.add(content=text, user_id=user_id, metadata=metadata)

    async def search(self, query: str, user_id: str = "default", k: int = 5) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            return []
        return await self.client.search(query, user_id=user_id, limit=k)
```

**Step 8: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/validators/test_cognitive_memory.py -k test_cognitive_memory_add_search -v`
Expected: PASS

**Step 9: Write failing test for `delete`**

```python
@pytest.mark.asyncio
async def test_cognitive_memory_delete(tmp_path):
    db_path = str(tmp_path / "test_delete.db")
    memory = CognitiveMemory(db_path=db_path)
    
    await memory.add("to be deleted")
    results = await memory.search("deleted")
    memory_id = results[0]["id"]
    
    await memory.delete(memory_id)
    results_after = await memory.search("deleted")
    assert len(results_after) == 0
```

**Step 10: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/validators/test_cognitive_memory.py -k test_cognitive_memory_delete -v`
Expected: FAIL (AttributeError)

**Step 11: Implement `delete`**

```python
    async def delete(self, memory_id: str):
        if not memory_id:
            raise ValueError("memory_id cannot be empty")
        return await self.client.delete(memory_id)
```

**Step 12: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/validators/test_cognitive_memory.py -k test_cognitive_memory_delete -v`
Expected: PASS

**Step 13: Implement `stats` and final cleanup**

```python
    async def stats(self) -> Dict[str, Any]:
        return {
            "engine": "OpenMemory",
            "storage": "SQLite",
            "database_url": env.database_url
        }
```

**Step 14: Run full test suite**

Run: `PYTHONPATH=. pytest tests/validators/test_cognitive_memory.py -v`
Expected: ALL PASS

**Step 15: Commit**

```bash
git add utils/cognitive_memory.py tests/validators/test_cognitive_memory.py
git commit -m "feat: implement robust CognitiveMemory wrapper for OpenMemory SDK"
```
