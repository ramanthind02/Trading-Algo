import sys
import os
import json
import asyncio
from pathlib import Path

# Add project root to sys.path
REPO_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(REPO_ROOT))

from utils.memory_service import MemoryService
from utils.cognitive_memory import CognitiveMemory

def run_retrieval(query: str) -> str:
    if not query:
        return json.dumps({"hookSpecificOutput": {"hookEventName": "MessageSend", "additionalContext": ""}})
    
    rag_context = ""
    cog_context = ""
    
    # 1. Retrieve from RAG (ChromaDB)
    try:
        rag_service = MemoryService()
        rag_results = rag_service.retrieve(query, k=5)
        if rag_results:
            parts = ["<RAG-MEMORY-CONTEXT>"]
            for i, r in enumerate(rag_results, 1):
                parts.append(f"Result {i}:\n{r['text']}")
            parts.append("</RAG-MEMORY-CONTEXT>")
            rag_context = "\n\n".join(parts)
    except Exception:
        pass # Silent failure for production
        
    # 2. Retrieve from Cognitive Memory (OpenMemory)
    try:
        cog_service = CognitiveMemory()
        cog_results = asyncio.run(cog_service.search(query, k=5))
        if cog_results:
            parts = ["<COGNITIVE-MEMORY-CONTEXT>"]
            for i, r in enumerate(cog_results, 1):
                # SDK uses 'content'
                parts.append(f"Fact {i}:\n{r['content']}")
            parts.append("</COGNITIVE-MEMORY-CONTEXT>")
            cog_context = "\n\n".join(parts)
    except Exception:
        pass # Silent failure for production
        
    combined_context = "\n\n".join(filter(None, [rag_context, cog_context]))
    
    return json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "MessageSend",
            "additionalContext": combined_context
        }
    })

if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else ""
    print(run_retrieval(query))
