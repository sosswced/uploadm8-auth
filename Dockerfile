# UploadM8 Worker Dockerfile
# This is for the WORKER service (worker.py) which needs FFmpeg
# The API service (app.py) can run without Docker

FROM python:3.11-slim

# Install FFmpeg and other system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Run the worker (not the web server)
CMD ["python", "worker.py"]
