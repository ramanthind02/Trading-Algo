import pytest
import tempfile
from pathlib import Path
from utils.memory_service import MemoryService
from utils.memory_commands import handle_remember, handle_recall, handle_mem_stats


@pytest.fixture
def memory_service(tmp_path):
    """Fixture that provides a MemoryService with automatic teardown."""
    service = MemoryService(persist_directory=str(tmp_path / "test_db"))
    yield service


def test_full_memory_pipeline(memory_service):
    """Validates that the full memory pipeline works: storing, retrieving, and stats."""
    service = memory_service
    
    handle_remember("We decided to use ChromaDB for vector storage", service=service)
    handle_remember("The embedding model is all-MiniLM-L6-v2", service=service)
    
    stats = handle_mem_stats(service=service)
    assert "Documents: 2" in stats
    
    results = handle_recall("Which vector DB did we choose?", service=service)
    assert "ChromaDB" in results