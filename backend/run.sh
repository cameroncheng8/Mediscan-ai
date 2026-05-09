#!/usr/bin/env bash
# One-shot: create venv, install deps, run API (stable defaults — no uvloop, no reload).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
if [[ ! -x "$PY" ]]; then
  echo "Need python3 on PATH (or set PYTHON=/path/to/python3)." >&2
  exit 1
fi

if [[ ! -d .venv ]]; then
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install -q --upgrade pip setuptools wheel
pip install -r requirements.txt

export KERAS_BACKEND=tensorflow
export TF_CPP_MIN_LOG_LEVEL=2

echo "Starting MediScan on http://127.0.0.1:8000  (Ctrl+C to stop)"
exec python -m uvicorn main:app --host 127.0.0.1 --port 8000 --loop asyncio
