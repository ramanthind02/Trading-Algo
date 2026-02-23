import sys
import os
import json
import asyncio
from pathlib import Path

# Add project root to sys.path
REPO_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(REPO_ROOT))

from utils.memory.memory_service import MemoryService
from utils.memory.cognitive_memory import CognitiveMemory


def _as_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _as_str_env(name: str, default: str = "") -> str:
    raw = os.getenv(name)
    return default if raw is None else raw


def _detect_mode() -> str:
    """Return retrieval mode.

    Modes:
    - off: no retrieval injection
    - execute: minimal injection
    - research: fuller injection

    Precedence:
    - RAG_MODE explicitly
    - else derived from OH_MY_OPENCODE_SLIM_PRESET
    - else research
    """

    explicit = _as_str_env("RAG_MODE").strip().lower()
    if explicit in {"off", "disable", "disabled", "none", "0", "false"}:
        return "off"
    if explicit in {"execute", "exec", "batch", "ship"}:
        return "execute"
    if explicit in {"research", "chatty", "brainstorm"}:
        return "research"
    if explicit:
        # Unknown mode: fall back to execute (lower overhead).
        return "execute"

    preset = _as_str_env("OH_MY_OPENCODE_SLIM_PRESET").strip().lower()
    if preset in {"dynamic", "default"}:
        return "execute"
    if preset in {"execute", "exec", "batch", "ship"} or preset.startswith("exec"):
        return "execute"
    if preset in {"research", "chatty", "brainstorm"}:
        return "research"
    # If no preset is provided, prefer lower overhead by default.
    return "execute"


def _truncate(s: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(s) <= max_chars:
        return s
    return s[:max_chars]


def _get_retrieval_params() -> tuple[int, int, int, int, int, bool]:
    mode = _detect_mode()

    if mode == "off":
        default_k_code = 0
        default_k_facts = 0
        default_max_chars = 0
    elif mode == "execute":
        default_k_code = 1
        default_k_facts = 1
        # Keep injection bounded by default in execute mode.
        default_max_chars = 4000
    else:
        default_k_code = 5
        default_k_facts = 5
        default_max_chars = 0

    k_code = max(0, _as_int_env("RAG_K_CODE", default_k_code))
    k_facts = max(0, _as_int_env("RAG_K_FACTS", default_k_facts))
    max_chars = max(0, _as_int_env("RAG_MAX_CHARS", default_max_chars))
    chunk_max_chars = max(0, _as_int_env("RAG_CHUNK_MAX_CHARS", 0))
    fact_max_chars = max(0, _as_int_env("RAG_FACT_MAX_CHARS", 0))
    debug = _as_str_env("RAG_DEBUG").strip().lower() in {"1", "true", "yes", "on"}
    return k_code, k_facts, max_chars, chunk_max_chars, fact_max_chars, debug

def run_retrieval(query: str) -> str:
    if not query:
        return json.dumps({"hookSpecificOutput": {"hookEventName": "MessageSend", "additionalContext": ""}})

    k_code, k_facts, max_chars, chunk_max_chars, fact_max_chars, debug = _get_retrieval_params()
    if k_code == 0 and k_facts == 0:
        return json.dumps({"hookSpecificOutput": {"hookEventName": "MessageSend", "additionalContext": ""}})

    rag_context = ""
    cog_context = ""

    # 1. Retrieve from RAG (ChromaDB)
    if k_code > 0:
        try:
            rag_service = MemoryService()
            rag_results = rag_service.retrieve(query, k=k_code)
            if rag_results:
                parts = ["<RAG-MEMORY-CONTEXT>"]
                if debug:
                    parts.append(f"mode={_detect_mode()} k={k_code}")
                for i, r in enumerate(rag_results, 1):
                    text = r.get("text", "")
                    if chunk_max_chars > 0:
                        text = _truncate(text, chunk_max_chars)
                    parts.append(f"Result {i}:\n{text}")
                parts.append("</RAG-MEMORY-CONTEXT>")
                rag_context = "\n\n".join(parts)
        except Exception:
            pass  # Silent failure for production

    # 2. Retrieve from Cognitive Memory (OpenMemory)
    if k_facts > 0:
        try:
            cog_service = CognitiveMemory()
            cog_results = asyncio.run(cog_service.search(query, k=k_facts))
            if cog_results:
                parts = ["<COGNITIVE-MEMORY-CONTEXT>"]
                if debug:
                    parts.append(f"mode={_detect_mode()} k={k_facts}")
                for i, r in enumerate(cog_results, 1):
                    # SDK uses 'content'
                    text = r.get("content", "")
                    if fact_max_chars > 0:
                        text = _truncate(text, fact_max_chars)
                    parts.append(f"Fact {i}:\n{text}")
                parts.append("</COGNITIVE-MEMORY-CONTEXT>")
                cog_context = "\n\n".join(parts)
        except Exception:
            pass  # Silent failure for production

    combined_context = "\n\n".join(filter(None, [rag_context, cog_context]))

    if max_chars > 0:
        combined_context = _truncate(combined_context, max_chars)
    
    return json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "MessageSend",
            "additionalContext": combined_context
        }
    })

if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else ""
    print(run_retrieval(query))
