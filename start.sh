#!/usr/bin/env bash
# Runs the FastAPI backend and the Next.js frontend at the same time, in one
# terminal. Press Ctrl+C once to stop both.
#
# Usage:
#   ./start.sh
#
# Written for portability (macOS's default bash 3.2 + BSD tools included) —
# avoids GNU-only flags.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="python3"
command -v python3 >/dev/null 2>&1 || PYTHON_BIN="python"

# ---------- pre-flight checks ----------

if ! "$PYTHON_BIN" -c "import uvicorn" >/dev/null 2>&1; then
  echo "Python dependencies aren't installed yet. Run this first:"
  echo "    pip install -r requirements.txt"
  exit 1
fi

if [ ! -f "model/model.pkl" ]; then
  echo "No trained model found yet. Run this first:"
  echo "    python src/train.py"
  exit 1
fi

if [ ! -d "frontend/node_modules" ]; then
  echo "Installing frontend dependencies (first run only)..."
  (cd frontend && npm install)
fi

# ---------- cleanup ----------
# Kills everything in this script's process group (backend + frontend +
# their subprocesses) when the script exits for any reason, including Ctrl+C.
CLEANED_UP=0
cleanup() {
  if [ "$CLEANED_UP" -eq 1 ]; then
    return
  fi
  CLEANED_UP=1
  echo ""
  echo "Stopping backend and frontend..."
  kill 0 2>/dev/null || true
}
trap cleanup EXIT

# ---------- start both ----------

echo "Starting backend  (FastAPI)  → http://localhost:8000"
"$PYTHON_BIN" -m uvicorn api.main:app --reload --port 8000 &

echo "Starting frontend (Next.js) → http://localhost:3000"
(cd frontend && npm run dev) &

echo ""
echo "Both are starting up — give the frontend a few seconds on first run."
echo "Press Ctrl+C to stop both."
echo ""

wait
