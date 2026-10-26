#!/bin/bash
# PostToolUse: format edited Python files with black.
f=$(jq -r '.tool_input.file_path // empty')
case "$f" in
  *.py) "$CLAUDE_PROJECT_DIR/.claude/hooks/run.sh" black -q --line-length 99 "$f" >/dev/null 2>&1 ;;
esac
exit 0
