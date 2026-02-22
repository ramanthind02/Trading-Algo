import pytest
import json
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Add project root to sys.path to allow imports from .opencode.hooks
REPO_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(REPO_ROOT))

@patch('utils.memory_service.MemoryService')
def test_rag_retriever_output(mock_service_class):
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
    
    mock_service = MagicMock()
    mock_service.retrieve.return_value = [
        {"text": "Relevant snippet 1", "distance": 0.1, "metadata": {}},
        {"text": "Relevant snippet 2", "distance": 0.2, "metadata": {}}
    ]
    mock_service_class.return_value = mock_service
    
    result = run_retrieval("test query")
    data = json.loads(result)
    
    assert "hookSpecificOutput" in data
    assert "Relevant snippet 1" in data["hookSpecificOutput"]["additionalContext"]
