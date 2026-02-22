#!/usr/bin/env bash
# MessageSend hook for RAG automation

set -euo pipefail

# Find the project root (containing .git directory) by searching upwards
FIND_ROOT() {
    local dir="$PWD"
    while [[ "$dir" != "/" ]]; do
        if [[ -d "$dir/.git" ]]; then
            echo "$dir"
            return 0
        fi
        dir="$(dirname "$dir")"
    done
    return 1
}

PROJECT_ROOT=$(FIND_ROOT)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

# Use shared project venv as per AGENTS.md
VENV_PATH="${PROJECT_ROOT}/venv"

# Activate venv if it exists
if [ -f "${VENV_PATH}/bin/activate" ]; then
    source "${VENV_PATH}/bin/activate"
fi

# The user's query is provided in the QUERY_TEXT environment variable by OpenCode
# Fallback to the first argument if QUERY_TEXT is not set
QUERY="${QUERY_TEXT:-${1:-}}"

# Run the Python retriever with warning suppression
PYTHON_EXEC="${VENV_PATH}/bin/python3"
if [ ! -f "$PYTHON_EXEC" ]; then
    PYTHON_EXEC="python3"
fi

"$PYTHON_EXEC" -W ignore "${SCRIPT_DIR}/rag_retriever.py" "$QUERY"
