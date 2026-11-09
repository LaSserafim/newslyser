#!/usr/bin/env bash
# NEWSLYSER // Launcher script
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "=== NEWSLYSER // CPI QUANTITATIVE ENGINE ==="
echo "Checking data pipeline..."
python3 data_pipeline.py

echo "Starting server on http://127.0.0.1:8888..."
python3 -m uvicorn app:app --host 127.0.0.1 --port 8888
