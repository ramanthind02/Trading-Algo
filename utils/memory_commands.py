from __future__ import annotations
from typing import Any
from utils.memory_service import MemoryService


def parse_memory_command(input_str: str) -> dict[str, Any]:
    parts = input_str.strip().split(maxsplit=1)
    cmd = parts[0].lower() if parts else ""
    
    if cmd == "remember":
        return {"command": "remember", "text": parts[1] if len(parts) > 1 else ""}
    elif cmd == "recall":
        return {"command": "recall", "query": parts[1] if len(parts) > 1 else ""}
    elif cmd in ("stats", "stat"):
        return {"command": "stats"}
    elif cmd == "clear":
        return {"command": "clear"}
    else:
        return {"command": "unknown"}


def handle_remember(text: str, service: MemoryService | None = None) -> str:
    if not text:
        return "Usage: /remember <text to store>"
    if service is None:
        service = MemoryService()
    service.store(text, metadata={"source": "manual"})
    return f"Stored: {text[:50]}..." if len(text) > 50 else f"Stored: {text}"


def handle_recall(query: str, service: MemoryService | None = None) -> str:
    if not query:
        return "Usage: /recall <query>"
    if service is None:
        service = MemoryService()
    results = service.retrieve(query, k=5)
    if not results:
        return "No relevant memory found."
    
    output = f"Found {len(results)} relevant memories:\n\n"
    for i, r in enumerate(results, 1):
        output += f"{i}. {r['text'][:200]}"
        if len(r['text']) > 200:
            output += "..."
        output += f"\n   (distance: {r['distance']:.3f})" if r['distance'] else ""
        output += "\n\n"
    return output


def handle_mem_stats(service: MemoryService | None = None) -> str:
    if service is None:
        service = MemoryService()
    stats = service.stats()
    return f"Memory Stats:\n  Documents: {stats['count']}\n  Storage: {stats['persist_directory']}"


def handle_mem_clear(service: MemoryService | None = None) -> str:
    if service is None:
        service = MemoryService()
    service.clear()
    return "Memory cleared."
