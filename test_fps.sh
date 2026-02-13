#!/bin/bash
# Quick FPS test script

echo "Starting stream in background..."
timeout 10 curl -s http://localhost:8080/stream > /dev/null 2>&1 &
STREAM_PID=$!

sleep 2
echo "Checking FPS..."
curl -s http://localhost:8080/fps | python3 -m json.tool

sleep 3
echo "Checking FPS again..."
curl -s http://localhost:8080/fps | python3 -m json.tool

kill $STREAM_PID 2>/dev/null
echo "Done."
