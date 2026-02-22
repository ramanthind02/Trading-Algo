#!/usr/bin/env bash
# MessageSend hook for RAG automation

set -euo pipefail

# Determine script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

# Find the project root (containing .git) by searching upwards
# to support shared venv in parent directories (worktrees)
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

PROJECT_ROOT=$(FIND_ROOT || echo "$(cd "${SCRIPT_DIR}/../.." && pwd)")
VENV_PATH="${PROJECT_ROOT}/venv"

# Activate venv if it exists
if [ -f "${VENV_PATH}/bin/activate" ]; then
    source "${VENV_PATH}/bin/activate"
fi

# The user's query is provided in the QUERY_TEXT environment variable by OpenCode
# or passed as the first argument
QUERY="${QUERY_TEXT:-${1:-}}"

# Run the Python retriever
# Use absolute path to python from venv if available
PYTHON_EXEC="${VENV_PATH}/bin/python3"
if [ ! -f "$PYTHON_EXEC" ]; then
    PYTHON_EXEC="python3"
fi

# Suppress warnings with -W ignore to avoid polluting stdout/stderr
"$PYTHON_EXEC" -W ignore "${SCRIPT_DIR}/rag_retriever.py" "$QUERY"
