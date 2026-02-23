import pytest
from utils.memory.memory_service import MemoryService


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
