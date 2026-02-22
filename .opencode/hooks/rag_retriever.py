import sys
import os
import json
from pathlib import Path

# Add project root to sys.path
REPO_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(REPO_ROOT))

from utils.memory_service import MemoryService

def run_retrieval(query: str) -> str:
    if not query:
        return json.dumps({"hookSpecificOutput": {"hookEventName": "MessageSend", "additionalContext": ""}})
    
    try:
        # Use existing memory service
        service = MemoryService()
        results = service.retrieve(query, k=5)
        
        if not results:
            return json.dumps({"hookSpecificOutput": {"hookEventName": "MessageSend", "additionalContext": ""}})
        
        context_parts = ["<RAG-MEMORY-CONTEXT>"]
        for i, r in enumerate(results, 1):
            context_parts.append(f"Result {i}:\n{r['text']}")
        context_parts.append("</RAG-MEMORY-CONTEXT>")
        
        context = "\n\n".join(context_parts)
        
        return json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "MessageSend",
                "additionalContext": context
            }
        })
    except Exception as e:
        # Silently fail to not break the session in production
        return json.dumps({"hookSpecificOutput": {"hookEventName": "MessageSend", "additionalContext": ""}})

if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else ""
    print(run_retrieval(query))
