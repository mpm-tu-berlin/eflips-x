#!/bin/bash
# Run a Python tool via poetry if available, else directly.
if command -v poetry >/dev/null 2>&1; then exec poetry run "$@"; else exec "$@"; fi
