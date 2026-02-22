#!/usr/bin/env bash
# Install git hooks for RAG automation

set -euo pipefail

# Find the project root
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
HOOKS_DIR="${PROJECT_ROOT}/.git/hooks"

echo "Installing post-commit hook to ${HOOKS_DIR}..."

cat <<EOF > "${HOOKS_DIR}/post-commit"
#!/usr/bin/env bash
# Git post-commit hook for RAG indexing
set -euo pipefail
# Find project root (main or worktree)
PROJECT_ROOT="\$(git rev-parse --show-toplevel)"
# Find main root for shared venv
MAIN_ROOT="\$(git rev-parse --git-common-dir)"
MAIN_ROOT="\$(cd "\${MAIN_ROOT}/.." && pwd)"

if [ -f "\${MAIN_ROOT}/venv/bin/activate" ]; then
    source "\${MAIN_ROOT}/venv/bin/activate"
fi
# Run indexing in background to not block the commit
python3 "\${PROJECT_ROOT}/utils/index_repo.py" > /dev/null 2>&1 &
EOF

chmod +x "${HOOKS_DIR}/post-commit"
echo "Done."
