#!/bin/bash
# PreToolUse: block hand-edits to the lockfile and .env.
f=$(jq -r '.tool_input.file_path // empty')
case "$f" in
  */poetry.lock|*/.env) echo "Do not edit $f directly (use poetry add / ask the user)." >&2; exit 2 ;;
esac
exit 0
