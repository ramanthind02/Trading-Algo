import pytest
import pathlib
import asyncio
from utils.memory.cognitive_memory import CognitiveMemory
from openmemory.memory import hsg

def test_cognitive_memory_init(tmp_path):
    db_path = str(tmp_path / "test.db")
    memory = CognitiveMemory(db_path=db_path)
    assert pathlib.Path(db_path).exists()

@pytest.mark.asyncio
async def test_cognitive_memory_add_search(tmp_path):
    db_path = str(tmp_path / "test_add.db")
    memory = CognitiveMemory(db_path=db_path)
    
    await memory.add("test content", doc_type="fact", tags=["test"])
    results = await memory.search("test")
    
    assert len(results) > 0
    assert results[0]["content"] == "test content"
    assert "test" in results[0]["tags"]
    assert results[0]["metadata"]["type"] == "fact"

@pytest.mark.asyncio
async def test_cognitive_memory_delete(tmp_path):
    db_path = str(tmp_path / "test_delete.db")
    memory = CognitiveMemory(db_path=db_path)
    
    await memory.add("to be deleted")
    results = await memory.search("deleted")
    memory_id = results[0]["id"]
    
    await memory.delete(memory_id)
    # Clear cache because SDK caches queries for 60s
    hsg.cache.clear()
    
    results_after = await memory.search("deleted")
    assert len(results_after) == 0

@pytest.mark.asyncio
async def test_cognitive_memory_empty_query():
    memory = CognitiveMemory()
    results = await memory.search("")
    assert results == []

@pytest.mark.asyncio
async def test_cognitive_memory_stats():
    memory = CognitiveMemory()
    stats = await memory.stats()
    assert stats["engine"] == "OpenMemory"
    assert "database_url" in stats
