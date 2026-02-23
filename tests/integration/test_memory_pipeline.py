import pytest
import tempfile
import asyncio
from pathlib import Path
from utils.memory.memory_service import MemoryService
from utils.memory.cognitive_memory import CognitiveMemory
from utils.memory.memory_commands import handle_remember, handle_recall, handle_mem_stats


@pytest.fixture
def memory_service(tmp_path):
    """Fixture that provides a MemoryService with automatic teardown."""
    service = MemoryService(persist_directory=str(tmp_path / "test_db"))
    yield service


@pytest.fixture
def cognitive_memory(tmp_path):
    """Fixture that provides a CognitiveMemory instance with automatic teardown."""
    db_path = str(tmp_path / "test_cognitive.db")
    memory = CognitiveMemory(db_path=db_path)
    yield memory


def test_full_memory_pipeline(memory_service, cognitive_memory):
    """Validates that the full memory pipeline works: storing, retrieving, and stats."""
    rag_service = memory_service
    cog_service = cognitive_memory
    
    # Store in RAG (manual)
    rag_service.store("We decided to use ChromaDB for vector storage", metadata={"source": "design"})
    
    # Store in Cognitive (via handler)
    handle_remember("The user prefers using pydantic for data models", cog_service=cog_service)
    
    # Check stats
    stats = handle_mem_stats(rag_service=rag_service, cog_service=cog_service)
    assert "CODEBASE RAG" in stats
    assert "COGNITIVE MEMORY" in stats
    
    # Recall (searching both)
    results = handle_recall("vector storage", rag_service=rag_service, cog_service=cog_service)
    assert "CODEBASE RAG" in results
    assert "ChromaDB" in results
    
    results_cog = handle_recall("pydantic", rag_service=rag_service, cog_service=cog_service)
    assert "COGNITIVE MEMORY" in results_cog
    assert "pydantic" in results_cog
