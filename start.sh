#!/bin/bash
set -e

echo "🔨 Building frontend..."
cd frontend
npm ci --prefer-offline 2>/dev/null || npm install
npm run build
cd ..

echo "🚀 Starting FastAPI server..."
exec uvicorn api_server:app --host 0.0.0.0 --port ${PORT:-8000}
