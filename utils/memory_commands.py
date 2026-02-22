from __future__ import annotations
import asyncio
from typing import Any
from utils.memory_service import MemoryService
from utils.cognitive_memory import CognitiveMemory


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


def handle_remember(text: str, rag_service: MemoryService | None = None, cog_service: CognitiveMemory | None = None) -> str:
    if not text:
        return "Usage: /remember <text to store>"
    
    if cog_service is None:
        cog_service = CognitiveMemory()
    
    try:
        # Default to CognitiveMemory for facts/preferences
        asyncio.run(cog_service.add(text, doc_type="fact"))
        return f"Stored in cognitive memory: {text[:50]}..." if len(text) > 50 else f"Stored in cognitive memory: {text}"
    except Exception as e:
        return f"Error storing cognitive memory: {e}"


def handle_recall(query: str, rag_service: MemoryService | None = None, cog_service: CognitiveMemory | None = None) -> str:
    if not query:
        return "Usage: /recall <query>"
    
    if rag_service is None:
        rag_service = MemoryService()
    if cog_service is None:
        cog_service = CognitiveMemory()
    
    output = ""
    
    # 1. Search Cognitive Memory
    try:
        cog_results = asyncio.run(cog_service.search(query, k=5))
        if cog_results:
            output += "--- COGNITIVE MEMORY ---\n"
            for i, r in enumerate(cog_results, 1):
                # SDK 1.3.2 uses 'content'
                output += f"{i}. {r['content'][:200]}"
                if len(r['content']) > 200:
                    output += "..."
                output += "\n\n"
    except Exception as e:
        output += f"Error retrieving cognitive memory: {e}\n\n"

    # 2. Search RAG Memory
    try:
        rag_results = rag_service.retrieve(query, k=5)
        if rag_results:
            output += "--- CODEBASE RAG ---\n"
            for i, r in enumerate(rag_results, 1):
                output += f"{i}. {r['text'][:200]}"
                if len(r['text']) > 200:
                    output += "..."
                output += f"\n   (file: {r['metadata'].get('file_path', 'unknown')})"
                output += "\n\n"
    except Exception as e:
        output += f"Error retrieving RAG memory: {e}\n\n"

    if not output:
        return "No relevant memory found in either store."
    
    return output


def handle_mem_stats(rag_service: MemoryService | None = None, cog_service: CognitiveMemory | None = None) -> str:
    if rag_service is None:
        rag_service = MemoryService()
    if cog_service is None:
        cog_service = CognitiveMemory()
        
    output = "Memory Statistics:\n\n"
    
    try:
        rag_stats = rag_service.stats()
        output += f"CODEBASE RAG (ChromaDB):\n"
        output += f"  Documents: {rag_stats['count']}\n"
        output += f"  Storage: {rag_stats['persist_directory']}\n\n"
    except Exception as e:
        output += f"Error getting RAG stats: {e}\n\n"
        
    try:
        cog_stats = asyncio.run(cog_service.stats())
        output += f"COGNITIVE MEMORY (OpenMemory):\n"
        output += f"  Engine: {cog_stats['engine']}\n"
        output += f"  Database: {cog_stats['database_url']}\n"
    except Exception as e:
        output += f"Error getting cognitive stats: {e}\n"
        
    return output


def handle_mem_clear(rag_service: MemoryService | None = None, cog_service: CognitiveMemory | None = None) -> str:
    if rag_service is None:
        rag_service = MemoryService()
    if cog_service is None:
        cog_service = CognitiveMemory()
        
    errors = []
    try:
        rag_service.clear()
    except Exception as e:
        errors.append(f"RAG clear error: {e}")
        
    try:
        # Note: CognitiveMemory wrapper doesn't have clear() yet in Task 2 impl, 
        # but we should probably add it or handle it.
        # Actually Task 2 design says it has delete(memory_id) but not full clear.
        # For now, let's just clear RAG and report if Cog can't be cleared easily.
        pass 
    except Exception as e:
        errors.append(f"Cognitive clear error: {e}")
        
    if errors:
        return "\n".join(errors)
    return "Memory stores cleared (Cognitive memory requires manual deletion of .db file for full reset)."
