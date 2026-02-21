# Persistent Memory Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a local RAG pipeline using ChromaDB and sentence-transformers to store/retrieve semantic memory for OpenCode sessions.

**Architecture:** Python service with ChromaDB persistence, sentence-transformers embeddings, langchain chunking. Exposed via CLI commands (`/remember`, `/recall`, `/mem`).

**Tech Stack:** Python, ChromaDB, langchain, langchain-community, sentence-transformers, nltk

---

## Dependencies

### Task 1: Install Python Dependencies

**Files:**
- Modify: `requirements.txt` (or create if not exists)

**Step 1: Add dependencies to requirements**

```txt
chromadb>=0.4.0
langchain>=0.1.0
langchain-community>=0.0.10
sentence-transformers>=2.2.0
nltk>=3.8.0
```

**Step 2: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: Dependencies installed successfully

**Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: add memory dependencies"
```

---

## Core Service

### Task 2: Memory Service Class

**Files:**
- Create: `utils/memory_service.py`

**Step 1: Write the failing test**

Create `tests/validators/test_memory_service.py`:

```python
import pytest
from utils.memory_service import MemoryService


def test_memory_service_init():
    service = MemoryService(persist_directory="/tmp/test_memory")
    assert service is not None
    assert service.collection is not None


def test_store_and_retrieve():
    service = MemoryService(persist_directory="/tmp/test_memory_retrieve")
    test_text = "This is a test about Python trading algorithms"
    service.store(test_text, metadata={"source": "test"})
    results = service.retrieve("Python trading", k=1)
    assert len(results) > 0
    assert "Python" in results[0]["text"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/validators/test_memory_service.py::test_memory_service_init -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'utils.memory_service'"

**Step 3: Write minimal implementation**

Create `utils/memory_service.py`:

```python
from __future__ import annotations
import chromadb
from chromadb.config import Settings
from typing import Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings


class MemoryService:
    def __init__(self, persist_directory: str = "~/.codex/memory_db"):
        self.persist_directory = persist_directory.replace("~", str(__import__("pathlib").Path.home()))
        
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection("memories")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
        )
    
    def store(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        chunks = self.text_splitter.split_text(text)
        chunk_ids = [f"chunk_{i}_{hash(chunk) % 100000}" for i, chunk in enumerate(chunks)]
        embeddings = self.embeddings.embed_documents(chunks)
        
        metadatas = [metadata or {} for _ in chunks]
        
        self.collection.add(
            ids=chunk_ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )
    
    def retrieve(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        query_embedding = self.embeddings.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        output = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                output.append({
                    "text": doc,
                    "distance": results["distances"][0][i] if "distances" in results else None,
                    "metadata": results["metadatas"][0][i] if "metadatas" in results else {}
                })
        
        return output
    
    def stats(self) -> dict[str, Any]:
        return {
            "count": self.collection.count(),
            "persist_directory": self.persist_directory
        }
    
    def clear(self) -> None:
        self.client.delete_collection("memories")
        self.collection = self.client.get_or_create_collection("memories")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/validators/test_memory_service.py::test_memory_service_init -v`
Expected: PASS

**Step 5: Commit**

```bash
git add utils/memory_service.py tests/validators/test_memory_service.py
git commit -m "feat: add MemoryService class for RAG storage"
```

---

## CLI Commands

### Task 3: CLI Commands Module

**Files:**
- Create: `utils/memory_commands.py`

**Step 1: Write the failing test**

Create `tests/validators/test_memory_commands.py`:

```python
import pytest
from utils.memory_commands import (
    handle_remember,
    handle_recall,
    handle_mem_stats,
    parse_memory_command,
)


def test_parse_remember():
    result = parse_memory_command("remember this is a test")
    assert result["command"] == "remember"
    assert result["text"] == "this is a test"


def test_parse_recall():
    result = parse_memory_command("recall trading algorithm")
    assert result["command"] == "recall"
    assert result["query"] == "trading algorithm"


def test_parse_stats():
    result = parse_memory_command("stats")
    assert result["command"] == "stats"


def test_parse_clear():
    result = parse_memory_command("clear")
    assert result["command"] == "clear"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/validators/test_memory_commands.py::test_parse_remember -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Create `utils/memory_commands.py`:

```python
from __future__ import annotations
from typing import Any
from utils.memory_service import MemoryService


def parse_memory_command(input_str: str) -> dict[str, Any]:
    parts = input_str.strip().split(maxsplit=1)
    cmd = parts[0].lower() if parts else ""
    
    if cmd == "remember":
        return {"command": "remember", "text": parts[1] if len(parts) > 1 else ""}
    elif cmd == "recall":
        return {"command": "recall", "query": parts[1] if len(parts) > 1 else ""}
    elif cmd in ("stats", "stat"):
        return {"command": "stats"}
    elif cmd == "clear":
        return {"command": "clear"}
    else:
        return {"command": "unknown"}


def handle_remember(text: str, service: MemoryService | None = None) -> str:
    if not text:
        return "Usage: /remember <text to store>"
    if service is None:
        service = MemoryService()
    service.store(text, metadata={"source": "manual"})
    return f"Stored: {text[:50]}..." if len(text) > 50 else f"Stored: {text}"


def handle_recall(query: str, service: MemoryService | None = None) -> str:
    if not query:
        return "Usage: /recall <query>"
    if service is None:
        service = MemoryService()
    results = service.retrieve(query, k=5)
    if not results:
        return "No relevant memory found."
    
    output = f"Found {len(results)} relevant memories:\n\n"
    for i, r in enumerate(results, 1):
        output += f"{i}. {r['text'][:200]}"
        if len(r['text']) > 200:
            output += "..."
        output += f"\n   (distance: {r['distance']:.3f})" if r['distance"] else ""
        output += "\n\n"
    return output


def handle_mem_stats(service: MemoryService | None = None) -> str:
    if service is None:
        service = MemoryService()
    stats = service.stats()
    return f"Memory Stats:\n  Documents: {stats['count']}\n  Storage: {stats['persist_directory']}"


def handle_mem_clear(service: MemoryService | None = None) -> str:
    if service is None:
        service = MemoryService()
    service.clear()
    return "Memory cleared."
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/validators/test_memory_commands.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add utils/memory_commands.py tests/validators/test_memory_commands.py
git commit -m "feat: add memory CLI commands"
```

---

## OpenCode Integration

### Task 4: OpenCode Custom Command

**Files:**
- Modify: `.opencode/command/remember.md`
- Modify: `.opencode/command/recall.md`
- Modify: `.opencode/command/mem.md`

**Step 1: Create remember command**

Create `.opencode/command/remember.md`:

```markdown
---
description: Store text to persistent memory
---

Call the memory store command with the following text:

```python
from utils.memory_commands import handle_remember
result = handle_remember("""$ARGUMENTS""")
print(result)
```

**Step 2: Create recall command**

Create `.opencode/command/recall.md`:

```markdown
---
description: Retrieve relevant memory
---

Call the memory recall command with the following query:

```python
from utils.memory_commands import handle_recall
result = handle_recall("""$ARGUMENTS""")
print(result)
```

**Step 3: Create mem stats command**

Create `.opencode/command/mem.md`:

```markdown
---
description: Show memory statistics
---

Call the memory stats command:

```python
from utils.memory_commands import handle_mem_stats
result = handle_mem_stats()
print(result)
```

**Step 4: Commit**

```bash
git add .opencode/command/
git commit -m "feat: add OpenCode memory commands"
```

---

## Integration Test

### Task 5: End-to-End Integration Test

**Files:**
- Create: `tests/integration/test_memory_pipeline.py`

**Step 1: Write integration test**

```python
import pytest
from utils.memory_service import MemoryService
from utils.memory_commands import handle_remember, handle_recall, handle_mem_stats


def test_full_memory_pipeline(tmp_path):
    service = MemoryService(persist_directory=str(tmp_path / "test_db"))
    
    handle_remember("We decided to use ChromaDB for vector storage", service=service)
    handle_remember("The embedding model is all-MiniLM-L6-v2", service=service)
    
    stats = handle_mem_stats(service=service)
    assert "Documents: 2" in stats
    
    results = handle_recall("Which vector DB did we choose?", service=service)
    assert "ChromaDB" in results
```

**Step 2: Run integration test**

Run: `pytest tests/integration/test_memory_pipeline.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/integration/test_memory_pipeline.py
git commit -m "test: add memory integration test"
```

---

## Summary

**Completed:**
- Task 1: Dependencies installed
- Task 2: MemoryService class with store/retrieve/stats
- Task 3: CLI commands with parse functions
- Task 4: OpenCode custom commands
- Task 5: Integration test

**Usage after setup:**
- `/remember <text>` — Store text to memory
- `/recall <query>` — Retrieve relevant context
- `/mem` — Show memory stats
