#!/bin/bash
set -e
echo "🚀 Starting FastAPI server..."
exec uvicorn api_server:app --host 0.0.0.0 --port ${PORT:-8000} --timeout-keep-alive 300
