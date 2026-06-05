#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=~/eflips-x
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SENSITIVITY_DIR="$SCRIPT_DIR/data/TCO/sensitivity_analysis"
EXAMPLE_SCRIPT="$SCRIPT_DIR/eflips/x/transition_plan/example.py"

for filepath in "$SENSITIVITY_DIR"/*.json; do
    filename="$(basename "$filepath")"
    echo "=== Running with tco_params: $filename ==="
    poetry run python "$EXAMPLE_SCRIPT" --tco_params "$filename"
done

echo "=== All sensitivity analysis runs complete ==="
