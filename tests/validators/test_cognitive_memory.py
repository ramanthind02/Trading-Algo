import pytest
import asyncio
from utils.cognitive_memory import CognitiveMemory

@pytest.mark.asyncio
async def test_cognitive_memory_add_search(tmp_path):
    db_path = str(tmp_path / "test_cognitive.db")
    memory = CognitiveMemory(db_path=db_path)
    
    await memory.add("User prefers async code", type="preference", tags=["coding-style"])
    results = await memory.search("coding style", k=1)
    
    assert len(results) > 0
    # OpenMemory search results might have 'content' or 'text' depending on version, 
    # check the SDK docs or trial/error. The design plan says 'content'.
    assert "async" in results[0]["content"]
