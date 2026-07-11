#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
echo "Starting AlfaScript..."
python3.11 -m uvicorn src.app:app --host 127.0.0.1 --port 8000 --reload
