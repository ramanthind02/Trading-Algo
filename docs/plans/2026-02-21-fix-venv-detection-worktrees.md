# Fix Venv Detection in Git Worktrees Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ensure `query-hook.sh` finds the main project root (with the shared venv) rather than stopping at a worktree root.

**Architecture:** Update `FIND_ROOT` in `.opencode/hooks/query-hook.sh` to use `[[ -d "$dir/.git" ]]` instead of `[[ -e "$dir/.git" ]]`. In git worktrees, `.git` is a file pointing to the main repo; in the main repo, it is a directory.

**Tech Stack:** Bash

---

### Task 1: Update FIND_ROOT logic

**Files:**
- Modify: `.opencode/hooks/query-hook.sh`

**Step 1: Write minimal implementation**

Change line 14:
`if [[ -e "$dir/.git" ]]; then`
to:
`if [[ -d "$dir/.git" ]]; then`

**Step 2: Commit**

```bash
git add .opencode/hooks/query-hook.sh
git commit -m "fix: make venv detection robust for git worktrees"
```

### Task 2: Verify the fix

**Step 1: Run the script manually**

Run: `QUERY_TEXT="test" .opencode/hooks/query-hook.sh`
Expected: A valid JSON output containing `hookSpecificOutput` and `additionalContext`. (Note: This assumes the venv is correctly found in the current directory or parent directory where `.git` is a directory).

**Step 2: Verify output format**

The output should be a single JSON object.

