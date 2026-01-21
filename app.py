# app.py — Render auth service (minimal: health + TikTok/Google callback routes)
# Deploy goal: stop 404s and confirm portals hit your service successfully.
# Later: wire token exchange + DB vault + encryption + device flow.

import os
import secrets
from typing import Optional

import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse

app = FastAPI(title="UploadM8 Auth Service")

BASE_URL = os.getenv("BASE_URL", "https://auth.uploadm8.com").rstrip("/")

# TikTok (Web Login Kit)
TIKTOK_CLIENT_KEY = os.getenv("TIKTOK_CLIENT_KEY", "")
TIKTOK_CLIENT_SECRET = os.getenv("TIKTOK_CLIENT_SECRET", "")
TIKTOK_REDIRECT_URI = os.getenv("TIKTOK_REDIRECT_URI", f"{BASE_URL}/oauth/tiktok/callback")
# Scopes you want ready (draft/inbox now; direct post later)
TIKTOK_SCOPE = os.getenv("TIKTOK_SCOPE", "user.info.basic,video.upload,video.publish")

# Google (Web OAuth client)
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", f"{BASE_URL}/oauth/google/callback")
GOOGLE_SCOPE = os.getenv("GOOGLE_SCOPE", "https://www.googleapis.com/auth/youtube.upload")

# If you later add sessions/device flow, store state server-side (DB).
# For now, we return state so you can confirm callback works end-to-end.
STATE_SECRET = os.getenv("OAUTH_STATE_SECRET", "")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "configured": True,
        "base_url": BASE_URL,
    }


# -------------------------
# TikTok: start + callback
# -------------------------
@app.get("/oauth/tiktok/start")
def tiktok_start():
    """
    Redirects user to TikTok authorization endpoint.
    This is optional for now; the critical piece is that /callback exists and returns 200.
    """
    if not TIKTOK_CLIENT_KEY:
        return JSONResponse({"ok": False, "error": "Missing TIKTOK_CLIENT_KEY env var"}, status_code=500)

    # Minimal state; for production store in DB tied to device/session.
    state = secrets.token_urlsafe(24)

    # TikTok v2 auth endpoint (commonly used pattern). If your existing code uses a different base,
    # keep it consistent across your stack.
    auth_url = (
        "https://www.tiktok.com/v2/auth/authorize/"
        f"?client_key={TIKTOK_CLIENT_KEY}"
        "&response_type=code"
        f"&scope={TIKTOK_SCOPE}"
        f"&redirect_uri={TIKTOK_REDIRECT_URI}"
        f"&state={state}"
    )
    return RedirectResponse(auth_url)


@app.get("/oauth/tiktok/callback")
async def tiktok_callback(request: Request):
    """
    TikTok redirects here with ?code=...&state=... or ?error=...
    """
    qp = dict(request.query_params)

    # Stop the 404s and prove the callback is reachable over HTTPS.
    # Token exchange will be wired in next step once you confirm portal redirect works.
    return JSONResponse(
        {
            "ok": True,
            "platform": "tiktok",
            "route": "/oauth/tiktok/callback",
            "query_params_received": qp,
            "next": "If you see this on real redirect, portal + HTTPS callback are correct.",
        }
    )


# -------------------------
# Google: start + callback
# -------------------------
@app.get("/oauth/google/start")
def google_start():
    """
    Redirects user to Google OAuth consent screen.
    """
    if not GOOGLE_CLIENT_ID:
        return JSONResponse({"ok": False, "error": "Missing GOOGLE_CLIENT_ID env var"}, status_code=500)

    state = secrets.token_urlsafe(24)

    # Google OAuth v2 authorization endpoint
    auth_url = (
        "https://accounts.google.com/o/oauth2/v2/auth"
        f"?client_id={GOOGLE_CLIENT_ID}"
        f"&redirect_uri={GOOGLE_REDIRECT_URI}"
        "&response_type=code"
        f"&scope={GOOGLE_SCOPE}"
        "&access_type=offline"
        "&prompt=consent"
        f"&state={state}"
    )
    return RedirectResponse(auth_url)


@app.get("/oauth/google/callback")
async def google_callback(request: Request):
    """
    Google redirects here with ?code=...&state=... or ?error=...
    """
    qp = dict(request.query_params)

    return JSONResponse(
        {
            "ok": True,
            "platform": "google",
            "route": "/oauth/google/callback",
            "query_params_received": qp,
            "next": "If you see this on real redirect, Google portal + HTTPS callback are correct.",
        }
    )


# Render runs: uvicorn app:app --host 0.0.0.0 --port $PORT
# This file is compatible with that start command.
