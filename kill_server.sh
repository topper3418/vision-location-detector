#!/bin/bash
# Script to kill the vision location detector server

echo "Stopping vision location detector server..."

# Find and kill processes running src.main
PIDS=$(pgrep -f "python.*src.main")

if [ -z "$PIDS" ]; then
    echo "No server process found running."
    exit 0
fi

echo "Found server process(es): $PIDS"

# Kill processes
for PID in $PIDS; do
    echo "Killing process $PID..."
    kill -TERM $PID 2>/dev/null
done

# Wait a moment
sleep 1

# Check if still running and force kill if needed
PIDS=$(pgrep -f "python.*src.main")
if [ ! -z "$PIDS" ]; then
    echo "Force killing remaining processes..."
    for PID in $PIDS; do
        kill -9 $PID 2>/dev/null
    done
fi

echo "Server stopped."
