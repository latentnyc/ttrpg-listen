#!/usr/bin/env bash
# TTRPG Listen - macOS shutdown script
# Uses the .pid file written by the app at startup; falls back to pgrep.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/.pid"

echo "============================================"
echo "  TTRPG Listen - Shutting down..."
echo "============================================"
echo

send_and_wait() {
    local pid="$1"
    local label="$2"
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "  $label ($pid) is not running."
        return 0
    fi

    echo "  Sending SIGTERM to $label ($pid)..."
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 15); do
        sleep 1
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "  Process exited cleanly."
            return 0
        fi
    done

    echo "  [WARN] Process did not exit in 15s, forcing..."
    kill -9 "$pid" 2>/dev/null || true
    return 0
}

# ---- Strategy 1: saved PID ----
if [ -f "$PID_FILE" ]; then
    APP_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [ -n "$APP_PID" ]; then
        echo "  Found PID file: $APP_PID"
        send_and_wait "$APP_PID" "ttrpglisten"
        rm -f "$PID_FILE"
        echo
        echo "  Shutdown complete."
        exit 0
    fi
    rm -f "$PID_FILE"
fi

# ---- Strategy 2: pgrep ----
echo "  Searching for ttrpglisten processes..."
PIDS="$(pgrep -f 'python.*-m ttrpglisten' 2>/dev/null || true)"
if [ -z "$PIDS" ]; then
    PIDS="$(pgrep -f 'ttrpglisten' 2>/dev/null || true)"
fi

if [ -z "$PIDS" ]; then
    echo "  No running TTRPG Listen process found."
    echo
    echo "  Shutdown complete."
    exit 0
fi

for pid in $PIDS; do
    send_and_wait "$pid" "ttrpglisten"
done

echo
echo "  Shutdown complete."
