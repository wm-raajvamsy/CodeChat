#!/usr/bin/env sh
set -e

# 1) Start Python backend in background
python /app/backend/app.py > /var/log/backend.log 2>&1 &

# 2) Wait for host Ollama to be reachable
echo "Waiting for Ollama on host..."
# On Windows/macOS use host.docker.internal; on Linux, use host networking instead
OLLAMA_HOST=${OLLAMA_HOST:-host.docker.internal}
until nc -z "$OLLAMA_HOST" 11434; do
  sleep 1
done
echo "Ollama is reachable at http://$OLLAMA_HOST:11434"

# 3) Serve React static build in foreground on port 3000
serve -s /app/backend/static -l 3000
