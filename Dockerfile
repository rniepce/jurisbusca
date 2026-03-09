FROM python:3.11-slim

# Install Node.js + system deps for PyMuPDF, Ghostscript, and Tesseract OCR
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    ghostscript \
    tesseract-ocr \
    tesseract-ocr-por \
    libgl1 \
    libglib2.0-0 \
    libgomp1 && \
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Frontend env vars (public anon keys, safe to embed)
ARG VITE_SUPABASE_URL=https://egrjeeiwqnqaeopgvinl.supabase.co
ARG VITE_SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImVncmplZWl3cW5xYWVvcGd2aW5sIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzIyMTY2OTIsImV4cCI6MjA4Nzc5MjY5Mn0.C7rFJmummGDSWWZwyBq3WNvMDaDvgeZwrODhPDR4fgI
ENV VITE_SUPABASE_URL=$VITE_SUPABASE_URL
ENV VITE_SUPABASE_ANON_KEY=$VITE_SUPABASE_ANON_KEY

# Build frontend
RUN cd frontend && npm ci && npm run build

# Expose port
EXPOSE ${PORT:-8000}

# Start the server
CMD ["sh", "-c", "uvicorn api_server:app --host 0.0.0.0 --port ${PORT:-8000}"]
