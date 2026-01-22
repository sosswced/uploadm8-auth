# app.py  (Render Auth Service - Google + TikTok routes + Device Link)
import os
import time
import json
import base64
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from jose import jwt
import psycopg

APP = FastAPI()

# ----------------------------
# ENV
# ----------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "")
BASE_URL = os.getenv("BASE_URL", "https://auth.uploadm8.com").rstrip("/")
JWT_SECRET = os.getenv("JWT_SECRET", "")
JWT_ISSUER = os.getenv("JWT_ISSUER", "uploadm8-auth")

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")

TIKTOK_CLIENT_KEY = os.getenv("TIKTOK_CLIENT_KEY", "")
TIKTOK_CLIENT_SECRET = os.getenv("TIKTOK_CLIENT_SECRET", "")

# Google endpoints
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"

# TikTok endpoints (OAuth token)
TIKTOK_TOKEN_URL = "https://open.tiktokapis.com/v2/oauth/token/"

# ----------------------------
# DB
# ----------------------------
SCHEMA_SQL = r"""
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS device_codes (
  device_code TEXT PRIMARY KEY,
  user_code TEXT UNIQUE NOT NULL,
  status TEXT NOT NULL,               -- PENDING or CLAIMED
  jwt TEXT,
  expires_at TIMESTAMPTZ NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS token_vault (
  subject TEXT NOT NULL,              -- who owns the tokens (device user)
  platform TEXT NOT NULL,             -- 'google' or 'tiktok'
  blob JSONB NOT NULL,                -- raw token json for now (MVP)
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (subject, platform)
);

CREATE TABLE IF NOT EXISTS audit_log (
  id BIGSERIAL PRIMARY KEY,
  event TEXT NOT NULL,
  subject TEXT NULL,
  meta JSONB NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

def db():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not set")
    return psycopg.connect(DATABASE_URL)

def init_db():
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
        conn.commit()

def now_utc():
    return datetime.now(timezone.utc)

def audit(event: str, subject: Optional[str] = None, meta: Optional[dict] = None):
    try:
        with db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO audit_log(event, subject, meta) VALUES (%s,%s,%s)",
                    (event, subject, json.dumps(meta or {})),
                )
            conn.commit()
    except Exception:
        # audit must never break auth
        pass

# ----------------------------
# JWT
# ----------------------------
def require_jwt_secret():
    if not JWT_SECRET or len(JWT_SECRET) < 16:
        raise RuntimeError("JWT_SECRET not set or too short (need 16+ chars)")

def mint_jwt(subject: str, minutes: int = 60 * 24 * 30) -> str:
    require_jwt_secret()
    exp = now_utc() + timedelta(minutes=minutes)
    payload = {
        "iss": JWT_ISSUER,
        "sub": subject,
        "exp": int(exp.timestamp()),
        "iat": int(now_utc().timestamp()),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

def get_subject_from_auth(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = auth.split(" ", 1)[1].strip()
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"], issuer=JWT_ISSUER)
        return payload["sub"]
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

# ----------------------------
# Startup
# ----------------------------
@APP.on_event("startup")
def _startup():
    init_db()
    audit("startup", meta={"base_url": BASE_URL})

# ----------------------------
# Health
# ----------------------------
@APP.get("/health")
def health():
    return {"status": "ok", "base_url": BASE_URL}

# ----------------------------
# Device Link (MVP)
# ----------------------------
def gen_user_code() -> str:
    # short human code like ABCD-EF
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    a = "".join(secrets.choice(alphabet) for _ in range(4))
    b = "".join(secrets.choice(alphabet) for _ in range(2))
    return f"{a}-{b}"

@APP.post("/device/code")
def device_code_create():
    device_code = secrets.token_urlsafe(32)
    user_code = gen_user_code()
    expires_at = now_utc() + timedelta(minutes=15)

    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO device_codes(device_code, user_code, status, expires_at) VALUES (%s,%s,%s,%s)",
                (device_code, user_code, "PENDING", expires_at),
            )
        conn.commit()

    verification_uri = f"{BASE_URL}/device"
    verification_uri_complete = f"{BASE_URL}/device?user_code={user_code}"
    audit("device_code_created", meta={"user_code": user_code})

    return {
        "device_code": device_code,
        "user_code": user_code,
        "verification_uri": verification_uri,
        "verification_uri_complete": verification_uri_complete,
        "expires_in": 15 * 60,
        "interval": 2,
    }

@APP.get("/device/poll")
def device_poll(device_code: str):
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT status, jwt, expires_at FROM device_codes WHERE device_code=%s",
                (device_code,),
            )
            row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Unknown device_code")

    status, jwt_token, expires_at = row
    if now_utc() > expires_at:
        raise HTTPException(status_code=400, detail="device_code expired")

    if status != "CLAIMED" or not jwt_token:
        return {"status": status}

    return {"status": "CLAIMED", "auth_jwt": jwt_token}

@APP.get("/device", response_class=HTMLResponse)
def device_page(user_code: Optional[str] = None):
    # Simple claim page
    return f"""
    <html>
      <head><title>UploadM8 Device Link</title></head>
      <body style="font-family: Arial; max-width: 720px; margin: 40px auto;">
        <h2>UploadM8 Device Link</h2>
        <p>Enter the code shown in your desktop app.</p>
        <form method="post" action="/device/claim">
          <input name="user_code" value="{user_code or ''}" placeholder="ABCD-EF"
                 style="font-size: 18px; padding: 8px; width: 220px;" />
          <button type="submit" style="font-size: 18px; padding: 8px 14px;">Continue</button>
        </form>
      </body>
    </html>
    """

@APP.post("/device/claim")
async def device_claim(request: Request):
    form = await request.form()
    user_code = (form.get("user_code") or "").strip().upper()

    if not user_code:
        raise HTTPException(status_code=400, detail="Missing user_code")

    # Subject for this device-user (MVP): stable hash of user_code
    subject = "dev_" + hashlib.sha256(user_code.encode("utf-8")).hexdigest()[:24]
    jwt_token = mint_jwt(subject)

    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE device_codes SET status='CLAIMED', jwt=%s WHERE user_code=%s AND status='PENDING'",
                (jwt_token, user_code),
            )
            if cur.rowcount != 1:
                raise HTTPException(status_code=404, detail="Invalid or already used code")
        conn.commit()

    audit("device_claimed", subject=subject)
    return HTMLResponse(f"""
    <html>
      <body style="font-family: Arial; max-width: 720px; margin: 40px auto;">
        <h2>Device Linked</h2>
        <p>You can close this tab and return to UploadM8 desktop.</p>
        <p><a href="/connect">Connect Platforms</a></p>
      </body>
    </html>
    """)

# ----------------------------
# Connect portal (simple)
# ----------------------------
@APP.get("/connect", response_class=HTMLResponse)
def connect_home(request: Request):
    # Must be logged in via Bearer token for real; MVP: let user paste token
    return """
    <html>
      <body style="font-family: Arial; max-width: 860px; margin: 40px auto;">
        <h2>UploadM8: Connect Platforms</h2>
        <p>This is the MVP connect hub. In production this will be a real authenticated UI.</p>
        <ol>
          <li>Use the desktop app to device-link (it stores AUTH_JWT locally).</li>
          <li>Then the desktop can open these links in your browser automatically.</li>
        </ol>
        <p>Google: <code>/oauth/google/start</code></p>
        <p>TikTok: <code>/oauth/tiktok/start</code></p>
      </body>
    </html>
    """

# ----------------------------
# OAuth helpers
# ----------------------------
def save_tokens(subject: str, platform: str, blob: Dict[str, Any]):
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO token_vault(subject, platform, blob)
                VALUES (%s,%s,%s)
                ON CONFLICT (subject, platform)
                DO UPDATE SET blob=EXCLUDED.blob, updated_at=now()
                """,
                (subject, platform, json.dumps(blob)),
            )
        conn.commit()
    audit("tokens_saved", subject=subject, meta={"platform": platform})

def load_tokens(subject: str, platform: str) -> Optional[dict]:
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT blob FROM token_vault WHERE subject=%s AND platform=%s",
                (subject, platform),
            )
            row = cur.fetchone()
    return row[0] if row else None

# ----------------------------
# Google OAuth (central)
# ----------------------------
@APP.get("/oauth/google/start")
def google_start(request: Request):
    subject = get_subject_from_auth(request)

    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        raise HTTPException(status_code=500, detail="Google env not configured")

    redirect_uri = f"{BASE_URL}/oauth/google/callback"
    state = secrets.token_urlsafe(16)

    # store state in audit only (MVP)
    audit("google_start", subject=subject, meta={"state": state})

    params = {
        "response_type": "code",
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "scope": "https://www.googleapis.com/auth/youtube.upload",
        "access_type": "offline",
        "prompt": "consent",
        "state": f"{subject}.{state}",
    }
    url = requests.Request("GET", GOOGLE_AUTH_URL, params=params).prepare().url
    return RedirectResponse(url)

@APP.get("/oauth/google/callback")
def google_callback(code: str, state: str):
    # state = "{subject}.{nonce}"
    if "." not in state:
        raise HTTPException(status_code=400, detail="Invalid state")
    subject, _nonce = state.split(".", 1)

    redirect_uri = f"{BASE_URL}/oauth/google/callback"
    data = {
        "code": code,
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
    }

    r = requests.post(GOOGLE_TOKEN_URL, data=data, timeout=30)
    if r.status_code != 200:
        audit("google_token_error", subject=subject, meta={"status": r.status_code, "body": r.text[:500]})
        raise HTTPException(status_code=400, detail=f"Google token exchange failed: {r.text[:300]}")

    tok = r.json()
    save_tokens(subject, "google", tok)

    return HTMLResponse("""
    <html><body style="font-family: Arial; margin: 40px;">
      <h2>Google connected</h2>
      <p>You can close this tab and return to UploadM8.</p>
    </body></html>
    """)

@APP.get("/vault/google/access")
def google_access(request: Request):
    subject = get_subject_from_auth(request)
    tok = load_tokens(subject, "google")
    if not tok:
        raise HTTPException(status_code=404, detail="Google not connected")

    # If token still valid, return it; else refresh
    access_token = tok.get("access_token")
    expires_in = tok.get("expires_in", 0)
    obtained_at = tok.get("_obtained_at")  # we may add later

    # MVP: always refresh if refresh_token exists to guarantee freshness
    refresh_token = tok.get("refresh_token")
    if not refresh_token:
        # Some Google responses omit refresh_token if already granted.
        # In that case we just return current access_token.
        return {"access_token": access_token}

    data = {
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }
    r = requests.post(GOOGLE_TOKEN_URL, data=data, timeout=30)
    if r.status_code != 200:
        audit("google_refresh_error", subject=subject, meta={"status": r.status_code, "body": r.text[:500]})
        raise HTTPException(status_code=400, detail="Google refresh failed")

    new_tok = r.json()
    # keep refresh_token
    new_tok["refresh_token"] = refresh_token
    save_tokens(subject, "google", new_tok)
    return {"access_token": new_tok.get("access_token")}

# ----------------------------
# TikTok OAuth (central)
# ----------------------------
@APP.get("/oauth/tiktok/start")
def tiktok_start(request: Request):
    subject = get_subject_from_auth(request)

    if not TIKTOK_CLIENT_KEY or not TIKTOK_CLIENT_SECRET:
        raise HTTPException(status_code=500, detail="TikTok env not configured")

    redirect_uri = f"{BASE_URL}/oauth/tiktok/callback"
    state = secrets.token_urlsafe(16)
    audit("tiktok_start", subject=subject, meta={"state": state})

    params = {
        "client_key": TIKTOK_CLIENT_KEY,
        "response_type": "code",
        "scope": "user.info.basic,video.upload",  # add video.publish later
        "redirect_uri": redirect_uri,
        "state": f"{subject}.{state}",
    }
    url = "https://www.tiktok.com/v2/auth/authorize/?" + requests.compat.urlencode(params)
    return RedirectResponse(url)

@APP.get("/oauth/tiktok/callback")
def tiktok_callback(code: str, state: str):
    if "." not in state:
        raise HTTPException(status_code=400, detail="Invalid state")
    subject, _nonce = state.split(".", 1)

    redirect_uri = f"{BASE_URL}/oauth/tiktok/callback"

    payload = {
        "client_key": TIKTOK_CLIENT_KEY,
        "client_secret": TIKTOK_CLIENT_SECRET,
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": redirect_uri,
    }

    r = requests.post(TIKTOK_TOKEN_URL, json=payload, timeout=30)
    if r.status_code != 200:
        audit("tiktok_token_error", subject=subject, meta={"status": r.status_code, "body": r.text[:500]})
        raise HTTPException(status_code=400, detail=f"TikTok token exchange failed: {r.text[:300]}")

    tok = r.json()
    # TikTok wraps tokens sometimes
    save_tokens(subject, "tiktok", tok)
    return HTMLResponse("""
    <html><body style="font-family: Arial; margin: 40px;">
      <h2>TikTok connected</h2>
      <p>You can close this tab and return to UploadM8.</p>
    </body></html>
    """)

@APP.get("/vault/tiktok/access")
def tiktok_access(request: Request):
    subject = get_subject_from_auth(request)
    tok = load_tokens(subject, "tiktok")
    if not tok:
        raise HTTPException(status_code=404, detail="TikTok not connected")

    # MVP: return access_token if present.
    # Later: implement refresh using refresh_token + correct TikTok refresh grant.
    data = tok.get("data") if isinstance(tok, dict) else None
    access_token = None
    if isinstance(data, dict):
        access_token = data.get("access_token")
    if not access_token:
        access_token = tok.get("access_token")

    if not access_token:
        raise HTTPException(status_code=400, detail="TikTok token missing access_token")
    return {"access_token": access_token}
