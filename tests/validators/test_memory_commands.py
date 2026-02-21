import pytest
from utils.memory_commands import (
    handle_remember,
    handle_recall,
    handle_mem_stats,
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
