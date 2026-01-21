#!/usr/bin/env python3
"""
UploadM8 Central Auth Service (Render)

This version is hardened for Render's Python 3.13 runtime by using psycopg (psycopg3),
not psycopg2 (which can crash with missing symbols on 3.13).

NOTE:
- This file intentionally keeps OAuth callback routes "alive" (200 JSON) even if you
  haven't wired full OAuth token exchange yet, so portal redirect validation works.
- You can layer username/password and Mailgun reset later without breaking routes.
"""

from __future__ import annotations

import os
import json
import time
import base64
import secrets
import hashlib
from typing import Any, Dict, Optional
from urllib.parse import urlencode

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse

# psycopg3 (works with Python 3.13)
import psycopg
from psycopg.rows import dict_row

APP_NAME = "uploadm8-auth"
DEFAULT_META_VERSION = os.getenv("META_VERSION", "v23.0")

# ---------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "")
BASE_URL = os.getenv("BASE_URL", "https://auth.uploadm8.com").rstrip("/")
JWT_SECRET = os.getenv("JWT_SECRET", "")
TOKEN_ENC_KEYS = os.getenv("TOKEN_ENC_KEYS", "")

# Platform creds (keep optional for boot)
META_APP_ID = os.getenv("META_APP_ID", "")
META_APP_SECRET = os.getenv("META_APP_SECRET", "")

TIKTOK_CLIENT_KEY = os.getenv("TIKTOK_CLIENT_KEY", "")
TIKTOK_CLIENT_SECRET = os.getenv("TIKTOK_CLIENT_SECRET", "")

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")

# ---------------------------------------------------------------------
# Minimal DB helpers
# ---------------------------------------------------------------------
SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS audit_log (
  id BIGSERIAL PRIMARY KEY,
  event TEXT NOT NULL,
  user_id UUID NULL,
  meta JSONB NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS password_resets (
  token_hash TEXT PRIMARY KEY,
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  expires_at TIMESTAMPTZ NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS device_codes (
  device_code TEXT PRIMARY KEY,
  user_code TEXT UNIQUE NOT NULL,
  user_id UUID NULL REFERENCES users(id) ON DELETE SET NULL,
  status TEXT NOT NULL,
  expires_at TIMESTAMPTZ NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS token_vault (
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  platform TEXT NOT NULL,
  blob JSONB NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (user_id, platform)
);

CREATE TABLE IF NOT EXISTS user_settings (
  fb_user_id TEXT PRIMARY KEY,
  selected_page_id TEXT,
  selected_page_name TEXT,
  updated_at TIMESTAMPTZ NOT NULL
);
"""

def db_connect():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set")
    # dict_row gives us dict-like rows
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)

def init_db():
    if not DATABASE_URL:
        # Allow service to boot without DB for route validation (but report configured=false)
        return
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
        conn.commit()

def audit(event: str, user_id: Optional[str] = None, meta: Optional[Dict[str, Any]] = None):
    if not DATABASE_URL:
        return
    try:
        with db_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO audit_log(event, user_id, meta) VALUES (%s, %s, %s)",
                    (event, user_id, json.dumps(meta or {})),
                )
            conn.commit()
    except Exception:
        # Never crash auth service because of logging
        pass

# ---------------------------------------------------------------------
# App
# ---------------------------------------------------------------------
app = FastAPI(title="UploadM8 Auth Service", version="2.0.0")

@app.on_event("startup")
def _startup():
    # Initialize schema if DB is available
    init_db()

@app.get("/health")
def health():
    return {
        "status": "ok",
        "configured": bool(DATABASE_URL),
        "base_url": BASE_URL,
        "meta_version": DEFAULT_META_VERSION,
    }

# ---------------------------------------------------------------------
# OAuth routes (alive now; full exchange can be added later)
# ---------------------------------------------------------------------
@app.get("/oauth/tiktok/start")
def oauth_tiktok_start():
    """
    Redirects to TikTok authorize URL. Requires TIKTOK_CLIENT_KEY set.
    """
    if not TIKTOK_CLIENT_KEY:
        raise HTTPException(status_code=500, detail="TIKTOK_CLIENT_KEY missing")
    redirect_uri = f"{BASE_URL}/oauth/tiktok/callback"
    state = secrets.token_urlsafe(16)
    params = {
        "client_key": TIKTOK_CLIENT_KEY,
        "response_type": "code",
        "scope": "user.info.basic,video.upload,video.publish",
        "redirect_uri": redirect_uri,
        "state": state,
    }
    url = "https://www.tiktok.com/v2/auth/authorize/?" + urlencode(params)
    audit("oauth_tiktok_start", meta={"redirect_uri": redirect_uri})
    return RedirectResponse(url)

@app.get("/oauth/tiktok/callback")
def oauth_tiktok_callback(request: Request):
    """
    Callback endpoint to satisfy TikTok redirect URI validation.
    Later: exchange code -> tokens, store in token_vault.
    """
    code = request.query_params.get("code")
    state = request.query_params.get("state")
    error = request.query_params.get("error")
    audit("oauth_tiktok_callback", meta={"has_code": bool(code), "error": error})
    return JSONResponse({"ok": True, "callback": "ready", "platform": "tiktok", "has_code": bool(code), "state": state, "error": error})

@app.get("/oauth/google/start")
def oauth_google_start():
    """
    Redirects to Google OAuth consent screen. Requires GOOGLE_CLIENT_ID set.
    """
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=500, detail="GOOGLE_CLIENT_ID missing")
    redirect_uri = f"{BASE_URL}/oauth/google/callback"
    state = secrets.token_urlsafe(16)
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "https://www.googleapis.com/auth/youtube.upload",
        "access_type": "offline",
        "include_granted_scopes": "true",
        "prompt": "consent",
        "state": state,
    }
    url = "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(params)
    audit("oauth_google_start", meta={"redirect_uri": redirect_uri})
    return RedirectResponse(url)

@app.get("/oauth/google/callback")
def oauth_google_callback(request: Request):
    """
    Callback endpoint to satisfy Google redirect URI.
    Later: exchange code -> tokens, store in token_vault.
    """
    code = request.query_params.get("code")
    state = request.query_params.get("state")
    error = request.query_params.get("error")
    audit("oauth_google_callback", meta={"has_code": bool(code), "error": error})
    return JSONResponse({"ok": True, "callback": "ready", "platform": "google", "has_code": bool(code), "state": state, "error": error})

@app.get("/oauth/meta/start")
def oauth_meta_start():
    """
    Redirects to Meta OAuth dialog. Requires META_APP_ID set.
    """
    if not META_APP_ID:
        raise HTTPException(status_code=500, detail="META_APP_ID missing")
    redirect_uri = f"{BASE_URL}/oauth/meta/callback"
    state = secrets.token_urlsafe(16)
    params = {
        "client_id": META_APP_ID,
        "redirect_uri": redirect_uri,
        "state": state,
        "response_type": "code",
        "scope": "public_profile,pages_show_list,pages_read_engagement,pages_manage_posts,instagram_basic,instagram_content_publish",
    }
    url = "https://www.facebook.com/v23.0/dialog/oauth?" + urlencode(params)
    audit("oauth_meta_start", meta={"redirect_uri": redirect_uri})
    return RedirectResponse(url)

@app.get("/oauth/meta/callback")
def oauth_meta_callback(request: Request):
    code = request.query_params.get("code")
    state = request.query_params.get("state")
    error = request.query_params.get("error")
    audit("oauth_meta_callback", meta={"has_code": bool(code), "error": error})
    return JSONResponse({"ok": True, "callback": "ready", "platform": "meta", "has_code": bool(code), "state": state, "error": error})

# ---------------------------------------------------------------------
# Device-link placeholders (alive now; can be expanded later)
# ---------------------------------------------------------------------
DEVICE_PAGE_HTML = """<!doctype html>
<html>
<head><meta charset="utf-8"><title>UploadM8 Device Link</title></head>
<body style="font-family: Arial, sans-serif; max-width: 720px; margin: 40px auto;">
  <h2>UploadM8 Device Link</h2>
  <p>Enter the code shown in your desktop app to link this browser session.</p>
  <form method="POST" action="/device/claim">
    <label>Device Code</label><br/>
    <input name="device_code" style="font-size: 18px; padding: 10px; width: 320px;" value="__PREFILL__" />
    <div style="margin-top: 12px;">
      <button type="submit" style="font-size: 16px; padding: 10px 14px;">Link</button>
    </div>
  </form>
</body>
</html>
"""

@app.get("/device")
def device_page(device_code: str = ""):
    html = DEVICE_PAGE_HTML.replace("__PREFILL__", device_code or "")
    return HTMLResponse(html)

@app.post("/device/claim")
async def device_claim(request: Request):
    form = await request.form()
    device_code = (form.get("device_code") or "").strip()
    if not device_code:
        raise HTTPException(status_code=400, detail="device_code required")
    audit("device_claim", meta={"device_code": device_code})
    # Alive response for now
    return JSONResponse({"ok": True, "device_code": device_code})

