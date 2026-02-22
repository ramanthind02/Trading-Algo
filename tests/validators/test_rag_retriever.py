import pytest
import json
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
import sys
from pathlib import Path

# Add project root to sys.path to allow imports from .opencode.hooks
REPO_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(REPO_ROOT))

@patch('utils.cognitive_memory.CognitiveMemory')
@patch('utils.memory_service.MemoryService')
def test_rag_retriever_dual_output(mock_rag_class, mock_cog_class):
    import importlib.util
    
    # Correctly load the module from its path
    spec = importlib.util.spec_from_file_location(
        "rag_retriever", 
        str(REPO_ROOT / ".opencode" / "hooks" / "rag_retriever.py")
    )
    if spec is None or spec.loader is None:
        raise ImportError("Could not load rag_retriever module")
        
    rag_retriever = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rag_retriever)
    run_retrieval = rag_retriever.run_retrieval
    
    # Mock RAG (MemoryService)
    mock_rag = MagicMock()
    mock_rag.retrieve.return_value = [
        {"text": "RAG result 1", "distance": 0.1, "metadata": {}}
    ]
    mock_rag_class.return_value = mock_rag
    
    # Mock Cognitive (CognitiveMemory)
    mock_cog = MagicMock()
    mock_cog.search = AsyncMock(return_value=[
        {"content": "Cognitive fact 1", "score": 0.9}
    ])
    mock_cog_class.return_value = mock_cog
    
    result = run_retrieval("test query")
    data = json.loads(result)
    
    context = data["hookSpecificOutput"]["additionalContext"]
    
    assert "hookSpecificOutput" in data
    assert "<RAG-MEMORY-CONTEXT>" in context
    assert "RAG result 1" in context
    assert "<COGNITIVE-MEMORY-CONTEXT>" in context
    assert "Cognitive fact 1" in context

@patch('utils.cognitive_memory.CognitiveMemory')
@patch('utils.memory_service.MemoryService')
def test_rag_retriever_graceful_failure(mock_rag_class, mock_cog_class):
    import importlib.util
    
    spec = importlib.util.spec_from_file_location(
        "rag_retriever", 
        str(REPO_ROOT / ".opencode" / "hooks" / "rag_retriever.py")
    )
    if spec is None or spec.loader is None:
        raise ImportError("Could not load rag_retriever module")

    rag_retriever = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rag_retriever)
    run_retrieval = rag_retriever.run_retrieval

    # RAG fails, Cog succeeds
    mock_rag = MagicMock()
    mock_rag.retrieve.side_effect = Exception("RAG failure")
    mock_rag_class.return_value = mock_rag
    
    mock_cog = MagicMock()
    mock_cog.search = AsyncMock(return_value=[
        {"content": "Cognitive fact 1", "score": 0.9}
    ])
    mock_cog_class.return_value = mock_cog
    
    result = run_retrieval("test query")
    data = json.loads(result)
    context = data["hookSpecificOutput"]["additionalContext"]
    
    assert "<RAG-MEMORY-CONTEXT>" not in context
    assert "<COGNITIVE-MEMORY-CONTEXT>" in context
    assert "Cognitive fact 1" in context
