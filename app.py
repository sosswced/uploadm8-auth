#!/usr/bin/env python3
"""
UploadM8 Central Auth Service (Render)

Render start command (recommended):
  uvicorn app:app --host 0.0.0.0 --port $PORT

CRITICAL: this module must expose a FastAPI instance named `app`.

Desktop contract endpoints (auth_manager.py uses these):
  POST /device/code
  GET  /device/status?device_code=...
  POST /device/jwt
  GET  /api/token/{platform}              platform: google|tiktok|meta
  GET  /api/meta/pages
  POST /api/meta/select_page
  GET  /api/meta/page_token

OAuth routes (human UI + redirects):
  GET  /device?device_code=...
  POST /device/claim
  GET  /oauth/google/start?device_code=...
  GET  /oauth/google/callback
  GET  /oauth/tiktok/start?device_code=...
  GET  /oauth/tiktok/callback
  GET  /oauth/meta/start?device_code=...
  GET  /oauth/meta/callback
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

import psycopg
import requests
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from jose import jwt

# -----------------------------
# Env / config
# -----------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
BASE_URL = os.getenv("BASE_URL", "").strip().rstrip("/")  # e.g. https://auth.uploadm8.com
JWT_SECRET = os.getenv("JWT_SECRET", "").strip()
TOKEN_ENC_KEYS = os.getenv("TOKEN_ENC_KEYS", "").strip()  # "v1:<b64 32 bytes>,v2:<b64 32 bytes>"

META_APP_ID = os.getenv("META_APP_ID", "").strip()
META_APP_SECRET = os.getenv("META_APP_SECRET", "").strip()
META_VERSION = os.getenv("META_VERSION", "v23.0").strip()

TIKTOK_CLIENT_KEY = os.getenv("TIKTOK_CLIENT_KEY", "").strip()
TIKTOK_CLIENT_SECRET = os.getenv("TIKTOK_CLIENT_SECRET", "").strip()

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "").strip()
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "").strip()

DEVICE_CODE_TTL_MIN = int(os.getenv("DEVICE_CODE_TTL_MIN", "20"))
OAUTH_STATE_TTL_MIN = int(os.getenv("OAUTH_STATE_TTL_MIN", "20"))

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL missing in Render environment")
if not BASE_URL:
    raise RuntimeError("BASE_URL missing (must be your Render public URL, e.g. https://auth.uploadm8.com)")
if not JWT_SECRET:
    raise RuntimeError("JWT_SECRET missing in Render environment")
if not TOKEN_ENC_KEYS:
    raise RuntimeError("TOKEN_ENC_KEYS missing in Render environment")

# -----------------------------
# Helpers
# -----------------------------
def utcnow() -> datetime:
    return datetime.now(timezone.utc)

def b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")

def b64d(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)

# -----------------------------
# Encryption (AES-GCM + rotation)
# TOKEN_ENC_KEYS: "kid1:<b64 32 bytes>,kid2:<b64 32 bytes>"
# last kid = active for new writes; any kid can decrypt
# -----------------------------
@dataclass
class KeyRing:
    keys: Dict[str, bytes]
    active_kid: str

def load_keyring() -> KeyRing:
    parts = [p.strip() for p in TOKEN_ENC_KEYS.split(",") if p.strip()]
    if not parts:
        raise RuntimeError("TOKEN_ENC_KEYS empty")
    keys: Dict[str, bytes] = {}
    for part in parts:
        if ":" not in part:
            raise RuntimeError("TOKEN_ENC_KEYS invalid; expected kid:base64")
        kid, b64 = part.split(":", 1)
        raw = b64d(b64)
        if len(raw) != 32:
            raise RuntimeError(f"TOKEN_ENC_KEYS {kid} must decode to 32 bytes")
        keys[kid] = raw
    active_kid = parts[-1].split(":", 1)[0]
    return KeyRing(keys=keys, active_kid=active_kid)

KEYRING = load_keyring()

def encrypt_json(subject: str, platform: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    kid = KEYRING.active_kid
    key = KEYRING.keys[kid]
    aes = AESGCM(key)
    nonce = secrets.token_bytes(12)
    aad = f"{subject}:{platform}".encode("utf-8")
    pt = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ct = aes.encrypt(nonce, pt, aad)
    return {"kid": kid, "nonce": b64e(nonce), "ct": b64e(ct)}

def decrypt_json(subject: str, platform: str, blob: Dict[str, Any]) -> Dict[str, Any]:
    kid = str(blob.get("kid", ""))
    if kid not in KEYRING.keys:
        raise RuntimeError("Unknown kid in token blob")
    key = KEYRING.keys[kid]
    aes = AESGCM(key)
    nonce = b64d(str(blob["nonce"]))
    ct = b64d(str(blob["ct"]))
    aad = f"{subject}:{platform}".encode("utf-8")
    pt = aes.decrypt(nonce, ct, aad)
    return json.loads(pt.decode("utf-8"))

# -----------------------------
# JWT
# -----------------------------
def issue_jwt(subject: str) -> str:
    now = int(time.time())
    payload = {"sub": subject, "iat": now, "exp": now + 60 * 60 * 24 * 30}  # 30 days
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

def require_subject(req: Request) -> str:
    auth = req.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = auth.split(" ", 1)[1].strip()
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        sub = str(payload.get("sub", "")).strip()
        if not sub:
            raise ValueError("Missing sub")
        return sub
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid Bearer token")

# -----------------------------
# DB
# -----------------------------
def db() -> psycopg.Connection:
    return psycopg.connect(DATABASE_URL, autocommit=True)

SCHEMA_SQL = r"""
CREATE TABLE IF NOT EXISTS device_codes (
  device_code TEXT PRIMARY KEY,
  user_code TEXT UNIQUE NOT NULL,
  status TEXT NOT NULL,            -- PENDING / CLAIMED
  expires_at TIMESTAMPTZ NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS oauth_states (
  state TEXT PRIMARY KEY,
  device_code TEXT NOT NULL REFERENCES device_codes(device_code) ON DELETE CASCADE,
  platform TEXT NOT NULL,
  pkce_verifier TEXT NULL,
  expires_at TIMESTAMPTZ NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS token_vault (
  subject TEXT NOT NULL,
  platform TEXT NOT NULL,          -- meta | tiktok | google | meta_page
  blob JSONB NOT NULL,             -- encrypted token blob
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

def init_db() -> None:
    with db() as conn:
        conn.execute(SCHEMA_SQL)

def log_event(event: str, subject: Optional[str] = None, meta: Optional[Dict[str, Any]] = None) -> None:
    try:
        with db() as conn:
            conn.execute(
                "INSERT INTO audit_log(event, subject, meta) VALUES (%s, %s, %s)",
                (event, subject, json.dumps(meta) if meta else None),
            )
    except Exception:
        pass

def get_device_row(device_code: str) -> Optional[Dict[str, Any]]:
    with db() as conn:
        row = conn.execute(
            "SELECT device_code, user_code, status, expires_at FROM device_codes WHERE device_code=%s",
            (device_code,),
        ).fetchone()
    if not row:
        return None
    return {"device_code": row[0], "user_code": row[1], "status": row[2], "expires_at": row[3]}

def is_device_claimed(device_code: str) -> bool:
    row = get_device_row(device_code)
    if not row:
        return False
    if row["expires_at"] < utcnow():
        return False
    return row["status"] == "CLAIMED"

def upsert_vault(subject: str, platform: str, blob: Dict[str, Any]) -> None:
    with db() as conn:
        conn.execute(
            """
            INSERT INTO token_vault(subject, platform, blob, updated_at)
            VALUES (%s, %s, %s, now())
            ON CONFLICT (subject, platform)
            DO UPDATE SET blob=EXCLUDED.blob, updated_at=now()
            """,
            (subject, platform, json.dumps(blob)),
        )

def get_vault_blob(subject: str, platform: str) -> Optional[Dict[str, Any]]:
    with db() as conn:
        row = conn.execute(
            "SELECT blob FROM token_vault WHERE subject=%s AND platform=%s",
            (subject, platform),
        ).fetchone()
    if not row:
        return None
    return row[0]

# -----------------------------
# OAuth state + PKCE
# -----------------------------
def pkce_pair() -> Tuple[str, str]:
    verifier = b64e(secrets.token_bytes(32))
    challenge = b64e(hashlib.sha256(verifier.encode("utf-8")).digest())
    return verifier, challenge

def create_oauth_state(device_code: str, platform: str, pkce_verifier: Optional[str]) -> str:
    state = secrets.token_urlsafe(24)
    expires_at = utcnow() + timedelta(minutes=OAUTH_STATE_TTL_MIN)
    with db() as conn:
        conn.execute(
            "INSERT INTO oauth_states(state, device_code, platform, pkce_verifier, expires_at) VALUES (%s,%s,%s,%s,%s)",
            (state, device_code, platform, pkce_verifier, expires_at),
        )
    return state

def pop_oauth_state(state: str) -> Optional[Dict[str, Any]]:
    with db() as conn:
        row = conn.execute(
            "SELECT state, device_code, platform, pkce_verifier, expires_at FROM oauth_states WHERE state=%s",
            (state,),
        ).fetchone()
        if not row:
            return None
        conn.execute("DELETE FROM oauth_states WHERE state=%s", (state,))
    data = {"state": row[0], "device_code": row[1], "platform": row[2], "pkce_verifier": row[3], "expires_at": row[4]}
    if data["expires_at"] < utcnow():
        return None
    return data

# -----------------------------
# FastAPI app
# -----------------------------
init_db()
app = FastAPI(title="UploadM8 Auth", version="2.0")

@app.get("/health")
def health():
    return {"status": "ok", "base_url": BASE_URL, "meta_version": META_VERSION}

# ===== Device link =====
@app.post("/device/code")
def device_code():
    device_code = secrets.token_urlsafe(32)
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    user_code = "".join(secrets.choice(alphabet) for _ in range(8))
    expires_at = utcnow() + timedelta(minutes=DEVICE_CODE_TTL_MIN)
    with db() as conn:
        conn.execute(
            "INSERT INTO device_codes(device_code, user_code, status, expires_at) VALUES (%s,%s,%s,%s)",
            (device_code, user_code, "PENDING", expires_at),
        )
    verify_url = f"{BASE_URL}/device?device_code={device_code}"
    log_event("device_code_issued", subject=device_code, meta={"user_code": user_code})
    return {
        "ok": True,
        "device_code": device_code,
        "user_code": user_code,
        "verification_uri": f"{BASE_URL}/device",
        "verification_uri_complete": verify_url,
        "expires_in": DEVICE_CODE_TTL_MIN * 60,
        "interval": 3,
    }

@app.get("/device", response_class=HTMLResponse)
def device_page(device_code: str):
    row = get_device_row(device_code)
    if not row:
        return HTMLResponse("<h3>Invalid device_code</h3>", status_code=404)
    if row["expires_at"] < utcnow():
        return HTMLResponse("<h3>Device code expired</h3>", status_code=400)

    if row["status"] != "CLAIMED":
        body = f"""
        <h2>UploadM8 Device Link</h2>
        <p>Code: <b>{row['user_code']}</b></p>
        <form method="post" action="/device/claim">
          <input type="hidden" name="device_code" value="{device_code}" />
          <button type="submit">Link this device</button>
        </form>
        <p>After linking, return to the desktop app.</p>
        """
        return HTMLResponse(body)

    body = f"""
    <h2>UploadM8 Device Linked</h2>
    <p>Device <b>{row['user_code']}</b> is linked.</p>
    <h3>Connect platforms</h3>
    <ul>
      <li><a href="/oauth/google/start?device_code={device_code}">Connect Google/YouTube</a></li>
      <li><a href="/oauth/tiktok/start?device_code={device_code}">Connect TikTok</a></li>
      <li><a href="/oauth/meta/start?device_code={device_code}">Connect Facebook/Instagram</a></li>
    </ul>
    <p>You can close this tab after connecting.</p>
    """
    return HTMLResponse(body)

@app.post("/device/claim")
async def device_claim(req: Request):
    form = await req.form()
    device_code = str(form.get("device_code", "")).strip()
    row = get_device_row(device_code)
    if not row or row["expires_at"] < utcnow():
        raise HTTPException(status_code=400, detail="Invalid or expired device_code")
    with db() as conn:
        conn.execute("UPDATE device_codes SET status='CLAIMED' WHERE device_code=%s", (device_code,))
    log_event("device_claimed", subject=device_code)
    return RedirectResponse(url=f"/device?device_code={device_code}", status_code=302)

@app.get("/device/status")
def device_status(device_code: str):
    row = get_device_row(device_code)
    if not row:
        raise HTTPException(status_code=404, detail="Unknown device_code")
    if row["expires_at"] < utcnow():
        raise HTTPException(status_code=400, detail="Device code expired")
    return {"ok": True, "device_code": device_code, "status": row["status"]}

@app.post("/device/jwt")
def device_jwt(payload: Dict[str, Any]):
    device_code = str(payload.get("device_code", "")).strip()
    row = get_device_row(device_code)
    if not row or row["expires_at"] < utcnow():
        raise HTTPException(status_code=400, detail="Invalid or expired device_code")
    if row["status"] != "CLAIMED":
        raise HTTPException(status_code=400, detail="Device not claimed yet")
    token = issue_jwt(device_code)
    log_event("device_jwt_issued", subject=device_code)
    return {"ok": True, "jwt": token}

# ===== Token endpoints for desktop =====
@app.get("/api/token/{platform}")
def api_token(platform: str, req: Request):
    subject = require_subject(req)
    platform = platform.lower()
    if platform not in ("google", "tiktok", "meta"):
        raise HTTPException(status_code=400, detail="Invalid platform")

    blob = get_vault_blob(subject, platform)
    if not blob:
        return JSONResponse(
            status_code=404,
            content={"ok": False, "missing": True, "platform": platform, "connect_url": f"{BASE_URL}/device?device_code={subject}"},
        )

    data = decrypt_json(subject, platform, blob)

    # Legacy client expects "token" for these
    if platform in ("tiktok", "meta"):
        return {"ok": True, "token": data.get("access_token")}

    return {"ok": True, "token": data.get("access_token"), "payload": data}

# ===== Meta helper endpoints =====
@app.get("/api/meta/pages")
def meta_pages(req: Request):
    subject = require_subject(req)
    blob = get_vault_blob(subject, "meta")
    if not blob:
        raise HTTPException(status_code=404, detail="Meta not connected")

    tok = decrypt_json(subject, "meta", blob).get("access_token")
    if not tok:
        raise HTTPException(status_code=400, detail="Meta token missing")

    url = f"https://graph.facebook.com/{META_VERSION}/me/accounts"
    r = requests.get(url, params={"access_token": tok}, timeout=30)
    if r.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Graph error: {r.status_code} {r.text[:500]}")
    pages = [{"id": p.get("id"), "name": p.get("name")} for p in r.json().get("data", [])]
    return {"ok": True, "pages": pages}

@app.post("/api/meta/select_page")
def meta_select_page(payload: Dict[str, Any], req: Request):
    subject = require_subject(req)
    page_id = str(payload.get("page_id", "")).strip()
    if not page_id:
        raise HTTPException(status_code=400, detail="page_id required")

    blob = get_vault_blob(subject, "meta")
    if not blob:
        raise HTTPException(status_code=404, detail="Meta not connected")

    user_token = decrypt_json(subject, "meta", blob).get("access_token")

    # Pull page token + IG user id
    url = f"https://graph.facebook.com/{META_VERSION}/{page_id}"
    r = requests.get(
        url,
        params={"fields": "name,access_token,instagram_business_account", "access_token": user_token},
        timeout=30,
    )
    if r.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Graph error: {r.status_code} {r.text[:500]}")
    page = r.json()
    page_token = page.get("access_token")
    if not page_token:
        raise HTTPException(status_code=400, detail="Could not retrieve page access_token (check scopes/review).")

    ig_user_id = None
    if isinstance(page.get("instagram_business_account"), dict):
        ig_user_id = page["instagram_business_account"].get("id")

    page_payload = {
        "page_id": page_id,
        "page_name": page.get("name"),
        "page_access_token": page_token,
        "ig_user_id": ig_user_id,
    }
    upsert_vault(subject, "meta_page", encrypt_json(subject, "meta_page", page_payload))
    log_event("meta_page_selected", subject=subject, meta={"page_id": page_id, "ig_user_id": ig_user_id})
    return {"ok": True, **page_payload}

@app.get("/api/meta/page_token")
def meta_page_token(req: Request):
    subject = require_subject(req)
    blob = get_vault_blob(subject, "meta_page")
    if not blob:
        raise HTTPException(status_code=404, detail="No page selected")
    data = decrypt_json(subject, "meta_page", blob)
    return {"ok": True, **data}

# ===== OAuth flows =====
@app.get("/oauth/google/start")
def google_start(device_code: str):
    if not is_device_claimed(device_code):
        raise HTTPException(status_code=400, detail="Device not linked yet")
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        raise HTTPException(status_code=500, detail="GOOGLE_CLIENT_ID/SECRET not configured on server")

    redirect_uri = f"{BASE_URL}/oauth/google/callback"
    state = create_oauth_state(device_code, "google", pkce_verifier=None)
    scope = "https://www.googleapis.com/auth/youtube.upload"

    auth_url = (
        "https://accounts.google.com/o/oauth2/v2/auth"
        f"?response_type=code&client_id={requests.utils.quote(GOOGLE_CLIENT_ID)}"
        f"&redirect_uri={requests.utils.quote(redirect_uri, safe='')}"
        f"&scope={requests.utils.quote(scope)}"
        f"&access_type=offline&prompt=consent"
        f"&state={requests.utils.quote(state)}"
    )
    return RedirectResponse(auth_url, status_code=302)

@app.get("/oauth/google/callback", response_class=HTMLResponse)
def google_callback(code: str = "", state: str = "", error: str = ""):
    if error:
        return HTMLResponse(f"<h3>Google OAuth error</h3><pre>{error}</pre>", status_code=400)
    st = pop_oauth_state(state)
    if not st or st["platform"] != "google":
        return HTMLResponse("<h3>Invalid/expired state</h3>", status_code=400)

    device_code = st["device_code"]
    redirect_uri = f"{BASE_URL}/oauth/google/callback"

    r = requests.post(
        "https://oauth2.googleapis.com/token",
        data={
            "code": code,
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        },
        timeout=30,
    )
    if r.status_code != 200:
        log_event("google_token_exchange_failed", subject=device_code, meta={"status": r.status_code, "body": r.text[:500]})
        return HTMLResponse(f"<h3>Token exchange failed</h3><pre>{r.status_code} {r.text}</pre>", status_code=400)

    tok = r.json()
    payload = {
        "access_token": tok.get("access_token"),
        "refresh_token": tok.get("refresh_token"),
        "expires_in": tok.get("expires_in"),
        "token_type": tok.get("token_type"),
        "scope": tok.get("scope"),
        "obtained_at": int(time.time()),
    }
    upsert_vault(device_code, "google", encrypt_json(device_code, "google", payload))
    log_event("google_connected", subject=device_code, meta={"scope": payload.get("scope")})
    return HTMLResponse("<h2>Google/YouTube connected.</h2><p>You can close this tab.</p>")

@app.get("/oauth/tiktok/start")
def tiktok_start(device_code: str):
    if not is_device_claimed(device_code):
        raise HTTPException(status_code=400, detail="Device not linked yet")
    if not TIKTOK_CLIENT_KEY or not TIKTOK_CLIENT_SECRET:
        raise HTTPException(status_code=500, detail="TIKTOK_CLIENT_KEY/SECRET not configured on server")

    redirect_uri = f"{BASE_URL}/oauth/tiktok/callback"
    verifier, challenge = pkce_pair()
    state = create_oauth_state(device_code, "tiktok", pkce_verifier=verifier)
    scope = "user.info.basic,video.upload,video.publish"

    auth_url = (
        "https://www.tiktok.com/v2/auth/authorize/"
        f"?client_key={requests.utils.quote(TIKTOK_CLIENT_KEY)}"
        f"&response_type=code"
        f"&scope={requests.utils.quote(scope)}"
        f"&redirect_uri={requests.utils.quote(redirect_uri, safe='')}"
        f"&state={requests.utils.quote(state)}"
        f"&code_challenge={requests.utils.quote(challenge)}"
        f"&code_challenge_method=S256"
        f"&prompt=consent"
    )
    return RedirectResponse(auth_url, status_code=302)

@app.get("/oauth/tiktok/callback", response_class=HTMLResponse)
def tiktok_callback(code: str = "", state: str = "", error: str = ""):
    if error:
        return HTMLResponse(f"<h3>TikTok OAuth error</h3><pre>{error}</pre>", status_code=400)
    st = pop_oauth_state(state)
    if not st or st["platform"] != "tiktok":
        return HTMLResponse("<h3>Invalid/expired state</h3>", status_code=400)

    device_code = st["device_code"]
    verifier = st.get("pkce_verifier") or ""
    redirect_uri = f"{BASE_URL}/oauth/tiktok/callback"

    r = requests.post(
        "https://open.tiktokapis.com/v2/oauth/token/",
        data={
            "client_key": TIKTOK_CLIENT_KEY,
            "client_secret": TIKTOK_CLIENT_SECRET,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": redirect_uri,
            "code_verifier": verifier,
        },
        timeout=30,
    )
    if r.status_code != 200:
        log_event("tiktok_token_exchange_failed", subject=device_code, meta={"status": r.status_code, "body": r.text[:500]})
        return HTMLResponse(f"<h3>Token exchange failed</h3><pre>{r.status_code} {r.text}</pre>", status_code=400)

    data = r.json()
    tok = data.get("data") if isinstance(data, dict) else None
    if not isinstance(tok, dict):
        tok = data

    payload = {
        "access_token": tok.get("access_token"),
        "refresh_token": tok.get("refresh_token"),
        "expires_in": tok.get("expires_in"),
        "token_type": tok.get("token_type"),
        "scope": tok.get("scope"),
        "open_id": tok.get("open_id"),
        "obtained_at": int(time.time()),
    }
    upsert_vault(device_code, "tiktok", encrypt_json(device_code, "tiktok", payload))
    log_event("tiktok_connected", subject=device_code, meta={"scope": payload.get("scope")})
    return HTMLResponse("<h2>TikTok connected.</h2><p>You can close this tab.</p>")

@app.get("/oauth/meta/start")
def meta_start(device_code: str):
    if not is_device_claimed(device_code):
        raise HTTPException(status_code=400, detail="Device not linked yet")
    if not META_APP_ID or not META_APP_SECRET:
        raise HTTPException(status_code=500, detail="META_APP_ID/SECRET not configured on server")

    redirect_uri = f"{BASE_URL}/oauth/meta/callback"
    state = create_oauth_state(device_code, "meta", pkce_verifier=None)

    scope = ",".join([
        "public_profile",
        "pages_show_list",
        "pages_read_engagement",
        "pages_manage_posts",
        "instagram_basic",
        "instagram_content_publish",
        "business_management",
    ])

    dialog = f"https://www.facebook.com/{META_VERSION}/dialog/oauth"
    auth_url = (
        f"{dialog}?client_id={requests.utils.quote(META_APP_ID)}"
        f"&redirect_uri={requests.utils.quote(redirect_uri, safe='')}"
        f"&state={requests.utils.quote(state)}"
        f"&response_type=code"
        f"&scope={requests.utils.quote(scope)}"
    )
    return RedirectResponse(auth_url, status_code=302)

@app.get("/oauth/meta/callback", response_class=HTMLResponse)
def meta_callback(code: str = "", state: str = "", error: str = "", error_description: str = ""):
    if error:
        return HTMLResponse(f"<h3>Meta OAuth error</h3><pre>{error}: {error_description}</pre>", status_code=400)
    st = pop_oauth_state(state)
    if not st or st["platform"] != "meta":
        return HTMLResponse("<h3>Invalid/expired state</h3>", status_code=400)

    device_code = st["device_code"]
    redirect_uri = f"{BASE_URL}/oauth/meta/callback"

    r = requests.get(
        f"https://graph.facebook.com/{META_VERSION}/oauth/access_token",
        params={
            "client_id": META_APP_ID,
            "client_secret": META_APP_SECRET,
            "redirect_uri": redirect_uri,
            "code": code,
        },
        timeout=30,
    )
    if r.status_code != 200:
        log_event("meta_token_exchange_failed", subject=device_code, meta={"status": r.status_code, "body": r.text[:500]})
        return HTMLResponse(f"<h3>Token exchange failed</h3><pre>{r.status_code} {r.text}</pre>", status_code=400)

    tok = r.json()
    payload = {
        "access_token": tok.get("access_token"),
        "token_type": tok.get("token_type"),
        "expires_in": tok.get("expires_in"),
        "obtained_at": int(time.time()),
    }
    upsert_vault(device_code, "meta", encrypt_json(device_code, "meta", payload))
    log_event("meta_connected", subject=device_code)
    return HTMLResponse("<h2>Facebook/Instagram connected.</h2><p>Next: pick your Facebook Page in the desktop app.</p>")
