# Production API (no --reload). Use behind reverse proxy / Cloudflare.
# Worker runs separately: python worker.py  (or Dockerfile CMD)

uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${WEB_CONCURRENCY:-2}
