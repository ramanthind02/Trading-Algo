import pytest
import tempfile
from pathlib import Path
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