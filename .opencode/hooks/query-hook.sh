#!/usr/bin/env bash
# MessageSend hook for RAG automation

set -euo pipefail

# Determine script and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Activate venv if it exists
if [ -f "${REPO_ROOT}/venv/bin/activate" ]; then
    source "${REPO_ROOT}/venv/bin/activate"
fi

# The user's query is provided in the QUERY_TEXT environment variable by OpenCode
# Fallback to empty if not set
QUERY="${QUERY_TEXT:-}"

# Run the Python retriever
# Use absolute path to python from venv if available
PYTHON_EXEC="${REPO_ROOT}/venv/bin/python3"
if [ ! -f "$PYTHON_EXEC" ]; then
    PYTHON_EXEC="python3"
fi

"$PYTHON_EXEC" "${SCRIPT_DIR}/rag_retriever.py" "$QUERY"
