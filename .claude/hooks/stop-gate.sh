#!/bin/bash
# Stop: block completion while black or mypy fail (only if .py files changed).
input=$(cat)
[ "$(jq -r '.stop_hook_active // false' <<<"$input")" = "true" ] && exit 0
cd "$CLAUDE_PROJECT_DIR" || exit 0
git status --porcelain | grep -q '\.py$' || exit 0
command -v poetry >/dev/null 2>&1 || command -v mypy >/dev/null 2>&1 || exit 0  # tools unavailable here
run="$CLAUDE_PROJECT_DIR/.claude/hooks/run.sh"
out=$( { "$run" black --check . && "$run" mypy eflips --explicit-package-bases --strict; } 2>&1 )
if [ $? -ne 0 ]; then
  jq -n --arg r "black/mypy failed. Fix the code:
$(tail -40 <<<"$out")" '{decision:"block", reason:$r}'
fi
exit 0
