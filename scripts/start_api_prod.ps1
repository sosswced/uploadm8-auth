# Production API (no --reload). Use behind reverse proxy / Cloudflare.
# Worker runs separately: python worker.py

$port = if ($env:PORT) { $env:PORT } else { "8000" }
$workers = if ($env:WEB_CONCURRENCY) { $env:WEB_CONCURRENCY } else { "2" }
& python -m uvicorn app:app --host 0.0.0.0 --port $port --workers $workers
