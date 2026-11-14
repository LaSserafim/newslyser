#!/usr/bin/env bash
set -e
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "=== NEWSLYSER // DETERMINISTIC QUANTITATIVE TEST SUITE ==="
python3 -m unittest discover -s tests -p "test_*.py" -v
echo "=== ALL TESTS VERIFIED PASSING ==="
