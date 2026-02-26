#!/bin/bash
set -e

# Ensure data directory exists (for Railway Volume mount or local dev)
DATA_DIR="${JURISPRUDENCIA_DB_PATH%/*}"
if [ -n "$DATA_DIR" ] && [ "$DATA_DIR" != "$JURISPRUDENCIA_DB_PATH" ]; then
    mkdir -p "$DATA_DIR"
fi

# Check if jurisprudência DB is available
if [ -n "$JURISPRUDENCIA_DB_PATH" ] && [ -f "$JURISPRUDENCIA_DB_PATH" ]; then
    DB_SIZE=$(du -h "$JURISPRUDENCIA_DB_PATH" | cut -f1)
    echo "📚 Jurisprudência DB encontrado: $JURISPRUDENCIA_DB_PATH ($DB_SIZE)"
else
    echo "⚠️ Jurisprudência DB não encontrado. Use POST /api/admin/upload-jurisprudencia para enviar."
fi

echo "🚀 Starting FastAPI server..."
exec uvicorn api_server:app --host 0.0.0.0 --port ${PORT:-8000} --timeout-keep-alive 300
