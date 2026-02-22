FROM python:3.11-slim

# Install Node.js + system deps for PaddleOCR/OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    ghostscript \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 && \
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Build frontend
RUN cd frontend && npm ci && npm run build

# Expose port
EXPOSE ${PORT:-8000}

# Start the server
CMD ["sh", "-c", "uvicorn api_server:app --host 0.0.0.0 --port ${PORT:-8000}"]
