#!/usr/bin/env bash
# NEWSLYSER // Prototype Launcher script (Port 8889)
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "=== NEWSLYSER // BLOOMBERG TERMINAL WORKSTATION ==="
echo "Starting server on http://127.0.0.1:8889..."
python3 -m uvicorn server:app --host 127.0.0.1 --port 8889
