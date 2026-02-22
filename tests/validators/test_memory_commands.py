import pytest
import asyncio
from unittest.mock import MagicMock, patch
from utils.memory_commands import (
    handle_remember,
    handle_recall,
    handle_mem_stats,
    handle_mem_clear,
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


@patch("utils.memory_commands.asyncio.run")
def test_handle_remember_success(mock_run):
    mock_cog = MagicMock()
    result = handle_remember("test memory", cog_service=mock_cog)
    assert "Stored in cognitive memory: test memory" in result
    mock_run.assert_called_once()


def test_handle_remember_empty_text():
    result = handle_remember("")
    assert result == "Usage: /remember <text to store>"


@patch("utils.memory_commands.asyncio.run")
def test_handle_remember_long_text(mock_run):
    mock_cog = MagicMock()
    long_text = "x" * 100
    result = handle_remember(long_text, cog_service=mock_cog)
    assert "Stored in cognitive memory:" in result
    assert "..." in result


@patch("utils.memory_commands.asyncio.run")
def test_handle_remember_error(mock_run):
    mock_cog = MagicMock()
    mock_run.side_effect = Exception("Storage failed")
    result = handle_remember("test", cog_service=mock_cog)
    assert "Error" in result


@patch("utils.memory_commands.asyncio.run")
def test_handle_recall_success(mock_run):
    mock_cog = MagicMock()
    mock_rag = MagicMock()
    
    # Mock search results for Cognitive Memory
    mock_run.return_value = [{"content": "cognitive memory result"}]
    
    # Mock retrieve results for RAG Memory
    mock_rag.retrieve.return_value = [
        {"text": "rag memory result", "metadata": {"file_path": "test.py"}}
    ]
    
    result = handle_recall("search query", rag_service=mock_rag, cog_service=mock_cog)
    
    assert "--- COGNITIVE MEMORY ---" in result
    assert "cognitive memory result" in result
    assert "--- CODEBASE RAG ---" in result
    assert "rag memory result" in result
    assert "test.py" in result


def test_handle_recall_empty_query():
    result = handle_recall("")
    assert result == "Usage: /recall <query>"


@patch("utils.memory_commands.asyncio.run")
def test_handle_recall_no_results(mock_run):
    mock_cog = MagicMock()
    mock_rag = MagicMock()
    
    mock_run.return_value = []
    mock_rag.retrieve.return_value = []
    
    result = handle_recall("query", rag_service=mock_rag, cog_service=mock_cog)
    assert result == "No relevant memory found in either store."


@patch("utils.memory_commands.asyncio.run")
def test_handle_mem_stats_success(mock_run):
    mock_cog = MagicMock()
    mock_rag = MagicMock()
    
    mock_rag.stats.return_value = {"count": 42, "persist_directory": "/data/rag"}
    mock_run.return_value = {"engine": "OpenMemory", "database_url": "sqlite:///test.db"}
    
    result = handle_mem_stats(rag_service=mock_rag, cog_service=mock_cog)
    
    assert "Memory Statistics" in result
    assert "CODEBASE RAG (ChromaDB)" in result
    assert "42" in result
    assert "COGNITIVE MEMORY (OpenMemory)" in result
    assert "sqlite:///test.db" in result


def test_handle_mem_clear_success():
    mock_cog = MagicMock()
    mock_rag = MagicMock()
    
    result = handle_mem_clear(rag_service=mock_rag, cog_service=mock_cog)
    assert "Memory stores cleared" in result
    mock_rag.clear.assert_called_once()
