import pytest
from unittest.mock import MagicMock
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


def test_handle_remember_success():
    mock_service = MagicMock()
    result = handle_remember("test memory", service=mock_service)
    assert result == "Stored: test memory"
    mock_service.store.assert_called_once()


def test_handle_remember_empty_text():
    result = handle_remember("")
    assert result == "Usage: /remember <text to store>"


def test_handle_remember_long_text():
    mock_service = MagicMock()
    long_text = "x" * 100
    result = handle_remember(long_text, service=mock_service)
    assert "Stored:" in result
    assert "..." in result


def test_handle_remember_error():
    mock_service = MagicMock()
    mock_service.store.side_effect = Exception("Storage failed")
    result = handle_remember("test", service=mock_service)
    assert "Error" in result


def test_handle_recall_success():
    mock_service = MagicMock()
    mock_service.retrieve.return_value = [
        {"text": "first memory", "distance": 0.5},
        {"text": "second memory", "distance": 0.3},
    ]
    result = handle_recall("search query", service=mock_service)
    assert "Found 2 relevant memories" in result
    mock_service.retrieve.assert_called_once_with("search query", k=5)


def test_handle_recall_empty_query():
    result = handle_recall("")
    assert result == "Usage: /recall <query>"


def test_handle_recall_no_results():
    mock_service = MagicMock()
    mock_service.retrieve.return_value = []
    result = handle_recall("query", service=mock_service)
    assert result == "No relevant memory found."


def test_handle_recall_error():
    mock_service = MagicMock()
    mock_service.retrieve.side_effect = Exception("Retrieval failed")
    result = handle_recall("query", service=mock_service)
    assert "Error" in result


def test_handle_mem_stats_success():
    mock_service = MagicMock()
    mock_service.stats.return_value = {"count": 42, "persist_directory": "/data/memory"}
    result = handle_mem_stats(service=mock_service)
    assert "Memory Stats" in result
    assert "42" in result


def test_handle_mem_stats_error():
    mock_service = MagicMock()
    mock_service.stats.side_effect = Exception("Stats failed")
    result = handle_mem_stats(service=mock_service)
    assert "Error" in result


def test_handle_mem_clear_success():
    mock_service = MagicMock()
    result = handle_mem_clear(service=mock_service)
    assert result == "Memory cleared."
    mock_service.clear.assert_called_once()


def test_handle_mem_clear_error():
    mock_service = MagicMock()
    mock_service.clear.side_effect = Exception("Clear failed")
    result = handle_mem_clear(service=mock_service)
    assert "Error" in result
