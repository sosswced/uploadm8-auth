# UploadM8 Worker Dockerfile — Supercharged Build
# Includes: FFmpeg, Playwright/Chromium, TensorFlow/YAMNet, rembg/ONNX, Google Fonts

FROM python:3.11-slim-bookworm

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    # FFmpeg + audio processing
    ffmpeg \
    libsndfile1 \
    # Database client
    libpq-dev \
    gcc \
    # Playwright / Chromium dependencies
    libnss3 \
    libatk-bridge2.0-0 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libgbm1 \
    libxss1 \
    libasound2 \
    libpango-1.0-0 \
    libcairo2 \
    libcups2 \
    libdbus-1-3 \
    libgtk-3-0 \
    libnspr4 \
    libatspi2.0-0 \
    libx11-xcb1 \
    libxfixes3 \
    libfontconfig1 \
    fonts-liberation \
    # rembg / ONNX Runtime
    libgomp1 \
    # General utilities
    curl \
    unzip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── Google Fonts (Inter + Bebas Neue for Playwright thumbnails) ───────────────
RUN mkdir -p /usr/share/fonts/truetype/google && \
    curl -sL "https://github.com/google/fonts/raw/main/ofl/bebasneue/BebasNeue-Regular.ttf" \
         -o /usr/share/fonts/truetype/google/BebasNeue-Regular.ttf && \
    curl -sL "https://github.com/google/fonts/raw/main/ofl/inter/Inter%5Bslnt%2Cwght%5D.ttf" \
         -o /usr/share/fonts/truetype/google/Inter.ttf && \
    fc-cache -fv

WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Playwright Chromium browser install ───────────────────────────────────────
ENV PLAYWRIGHT_BROWSERS_PATH=/app/.playwright
RUN playwright install chromium --with-deps 2>/dev/null || playwright install chromium

# ── rembg model pre-download (isnet-general-use ~176MB) ──────────────────────
# ORT_THREADS limits ONNX Runtime CPU threads per session (Render-friendly)
ENV U2NET_HOME=/app/models/.u2net
ENV ORT_THREADS=2
RUN mkdir -p $U2NET_HOME && \
    python -c "from rembg import new_session; new_session('isnet-general-use'); print('rembg model ready')"

# ── YAMNet model pre-download via TF Hub (~20MB) ─────────────────────────────
ENV TFHUB_CACHE_DIR=/app/tfhub_cache
RUN python -c "import tensorflow_hub as hub; hub.load('https://tfhub.dev/google/yamnet/1'); print('YAMNet ready')"

# ── Copy application code ──────────────────────────────────────────────────────
COPY . .

# ── Run the worker ─────────────────────────────────────────────────────────────
CMD ["python", "worker.py"]