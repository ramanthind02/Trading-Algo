# OpenCode Launcher Scripts Design (OpenAI vs Zen)

**Problem:** Switching between OpenAI and Zen model sets (especially when rate-limited) currently requires manual environment variables and remembering preset names like `execute_zen`.

**Goal:** Make provider selection easy at chat start via simple flags, without editing global config each time.

## Proposed UX

Add a repo-local launcher script:

- `scripts/oc --provider openai --preset execute`
- `scripts/oc --provider zen --preset execute`

Defaults:

- `--provider openai`
- `--preset execute`

Optional convenience:

- `--rag off|execute|research` sets `RAG_MODE`

All remaining arguments are passed through to the real `opencode` binary.

## Mapping

The script maps the (provider,preset) choice to `OH_MY_OPENCODE_SLIM_PRESET`:

- provider=openai:
  - dynamic -> `dynamic`
  - research -> `research`
  - execute -> `execute`
- provider=zen:
  - dynamic -> `dynamic_zen`
  - research -> `research_zen`
  - execute -> `execute_zen`

This leverages the presets already configured in `~/.config/opencode/oh-my-opencode-slim.json`.

## Non-Goals

- Do not attempt to detect rate-limit errors and auto-restart.
- Do not modify or overwrite the user-global omos config from the repo.

## Success Criteria

- Users can launch OpenCode with one command and a provider flag.
- The wrapper is safe (no destructive actions) and transparent (prints help, passes args).
- Documentation points users at `scripts/oc` rather than manual env vars.
