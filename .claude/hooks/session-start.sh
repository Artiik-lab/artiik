#!/bin/bash
# SessionStart hook for Claude Code on the web: sets up the Python dev
# environment in python/ so tests, ruff and pyright work in cloud sessions.
set -euo pipefail

# Only run in Claude Code on the web (remote) sessions.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel)}/python"

# uv ships with the web image; install it if a custom image lacks it.
uv_cmd=(uv)
if ! command -v uv >/dev/null 2>&1; then
  python3 -m pip install --quiet --user uv
  uv_cmd=(python3 -m uv)
fi

# Idempotent: builds python/.venv from uv.lock, or does nothing if it's current.
"${uv_cmd[@]}" sync --locked >&2

# pyright runs on Node.js. Check it starts, so a broken setup fails here
# instead of in the middle of a session.
"${uv_cmd[@]}" run pyright --version >/dev/null
