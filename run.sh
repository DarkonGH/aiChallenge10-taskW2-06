#!/usr/bin/env bash
# Build and start the CLI agent. Needs LLM_BASE_URL and OPENAI_API_KEY.
set -euo pipefail
cd "$(dirname "$0")"

# Fall back to ~/.env when the caller has not exported the credentials.
if [ -z "${LLM_BASE_URL:-}" ] || [ -z "${OPENAI_API_KEY:-}" ]; then
  if [ -f "$HOME/.env" ]; then
    set -a
    # shellcheck disable=SC1090
    . "$HOME/.env"
    set +a
  fi
fi

go build -o task2-06 .
exec ./task2-06 "$@"
