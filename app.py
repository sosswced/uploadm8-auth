import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlencode

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

import psycopg
from psycopg.rows import dict_row


# ============================================================
# Core config
# ============================================================

BASE_URL = os.getenv("BASE_URL", "").rstrip("/")
DATABASE_URL = os.getenv("DATABASE_URL", "")
JWT_SECRET = os.getenv("JWT_SECRET", "")
TOKEN_ENC_KEYS = os.getenv("TOKEN_ENC_KEYS", "")  # format: v1:<b64>,v2:<b64> (last = active)
META_APP_ID = os.getenv("META_APP_ID", "")
META_APP_SECRET = os.getenv("META_APP_SECRET", "")
TIKTOK_CLIENT_KEY = os.getenv("TIKTOK_CLIENT_KEY", "")
TIKTOK_CLIENT_SECRET = os.getenv("TIKTOK_CLIENT_SECRET", "")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")

# FastAPI must expose variable named `app` for Render command `uvicorn app:app`
app = FastAPI()


# ============================================================
# Helpers: DB
# ============================================================

def db_conn():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL missing")
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS device_codes (
  device_code TEXT PRIMARY KEY,
  user_code TEXT UNIQUE NOT NULL,
  status TEXT NOT NULL,
  expires_at BIGINT NOT NULL,
  created_at BIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS oauth_state (
  state TEXT PRIMARY KEY,
  device_code TEXT NOT NULL,
  platform TEXT NOT NULL,
  payload JSONB NOT NULL,
  expires_at BIGINT NOT NULL,
  created_at BIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS token_vault (
  device_code TEXT NOT NULL,
  platform TEXT NOT NULL,
  blob JSONB NOT NULL,
  updated_at BIGINT NOT NULL,
  PRIMARY KEY (device_code, platform)
);

CREATE TABLE IF NOT EXISTS audit_log (
  id BIGSERIAL PRIMARY KEY,
  event TEXT NOT NULL,
  meta JSONB NULL,
  created_at BIGINT NOT NULL
);
"""


@app.on_event("startup")
def _startup():
    # Initialize schema (idempotent)
    if not DATABASE_URL:
        # Allow boot without DB for troubleshooting; endpoints will error if used.
        return
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
        conn.commit()


# ============================================================
# Helpers: Encryption (AES-GCM) with key rotation
# ============================================================

@dataclass
class KeyRing:
    active_kid: str
    keys: dict  # kid -> raw bytes


def parse_keyring(raw: str) -> KeyRing:
    """
    TOKEN_ENC_KEYS format:
      v1:<base64-32-bytes>,v2:<base64-32-bytes>
    The LAST entry is the active key for new encryptions.
    """
    if not raw.strip():
        raise RuntimeError("TOKEN_ENC_KEYS missing/blank")

    items = [x.strip() for x in raw.split(",") if x.strip()]
    keys = {}
    for item in items:
        if ":" not in item:
            raise RuntimeError("TOKEN_ENC_KEYS entry missing ':'")
        kid, b64 = item.split(":", 1)
        key = base64.b64decode(b64.encode("utf-8"))
        if len(key) != 32:
            raise RuntimeError(f"Key {kid} must be 32 bytes after base64 decode")
        keys[kid] = key

    active_kid = items[-1].split(":", 1)[0]
    return KeyRing(active_kid=active_kid, keys=keys)


def encrypt_json(obj: dict) -> dict:
    kr = parse_keyring(TOKEN_ENC_KEYS)
    key = kr.keys[kr.active_kid]
    aes = AESGCM(key)
    nonce = secrets.token_bytes(12)
    pt = json.dumps(obj).encode("utf-8")
    ct = aes.encrypt(nonce, pt, None)
    return {
        "kid": kr.active_kid,
        "nonce": base64.b64encode(nonce).decode("utf-8"),
        "ct": base64.b64encode(ct).decode("utf-8"),
    }


def decrypt_json(blob: dict) -> dict:
    kr = parse_keyring(TOKEN_ENC_KEYS)
    kid = blob.get("kid")
    if kid not in kr.keys:
        raise RuntimeError(f"Unknown kid: {kid}")
    key = kr.keys[kid]
    aes = AESGCM(key)
    nonce = base64.b64decode(blob["nonce"].encode("utf-8"))
    ct = base64.b64decode(blob["ct"].encode("utf-8"))
    pt = aes.decrypt(nonce, ct, None)
    return json.loads(pt.decode("utf-8"))


# ============================================================
# Helpers: Minimal JWT (HMAC-SHA256) for device sessions
# ============================================================

def b64url(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode("utf-8").rstrip("=")


def sign_jwt(payload: dict) -> str:
    if not JWT_SECRET:
        raise RuntimeError("JWT_SECRET missing")
    header = {"alg": "HS256", "typ": "JWT"}
    h = b64url(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    p = b64url(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    msg = f"{h}.{p}".encode("utf-8")
    sig = hmac.new(JWT_SECRET.encode("utf-8"), msg, hashlib.sha256).digest()
    return f"{h}.{p}.{b64url(sig)}"


def verify_jwt(token: str) -> dict:
    try:
        h, p, s = token.split(".")
        msg = f"{h}.{p}".encode("utf-8")
        sig = base64.urlsafe_b64decode(s + "==")
        exp_sig = hmac.new(JWT_SECRET.encode("utf-8"), msg, hashlib.sha256).digest()
        if not hmac.compare_digest(sig, exp_sig):
            raise ValueError("bad signature")
        payload = json.loads(base64.urlsafe_b64decode(p + "==").decode("utf-8"))
        if int(payload.get("exp", 0)) < int(time.time()):
            raise ValueError("expired")
        return payload
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")


def bearer_device_code(req: Request) -> str:
    auth = req.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = auth.split(" ", 1)[1].strip()
    payload = verify_jwt(token)
    dc = payload.get("device_code")
    if not dc:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    return dc


# ============================================================
# Health
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "configured": bool(DATABASE_URL and BASE_URL and JWT_SECRET and TOKEN_ENC_KEYS),
        "base_url": BASE_URL or None,
    }


# ============================================================
# Device-link flow (MVP)
# Desktop calls /device/code, user approves via /device, desktop polls /device/poll
# ============================================================

def gen_user_code() -> str:
    # readable code like ABCD-EFGH
    a = secrets.token_hex(2).upper()
    b = secrets.token_hex(2).upper()
    return f"{a}-{b}"


@app.post("/device/code")
def device_code_create():
    if not DATABASE_URL:
        raise HTTPException(status_code=500, detail="DATABASE_URL missing")
    device_code = secrets.token_urlsafe(24)
    user_code = gen_user_code()
    now = int(time.time())
    exp = now + 15 * 60

    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO device_codes(device_code,user_code,status,expires_at,created_at) "
                "VALUES(%s,%s,%s,%s,%s)",
                (device_code, user_code, "PENDING", exp, now),
            )
        conn.commit()

    return {
        "device_code": device_code,
        "user_code": user_code,
        "verification_uri": f"{BASE_URL}/device",
        "expires_in": 900,
    }


@app.get("/device", response_class=HTMLResponse)
def device_page():
    # dead-simple approval UI (no accounts yet)
    html = f"""
    <html><body style="font-family: Arial; padding: 24px;">
      <h2>UploadM8 Device Link</h2>
      <p>Enter the code shown in your desktop app.</p>
      <form method="post" action="/device/claim">
        <input name="user_code" placeholder="ABCD-EFGH" style="font-size:18px; padding:8px;" />
        <button type="submit" style="font-size:18px; padding:8px 16px;">Approve</button>
      </form>
      <p style="margin-top:18px; color:#666;">This approves the desktop session so it can connect platforms.</p>
    </body></html>
    """
    return HTMLResponse(html)


@app.post("/device/claim")
async def device_claim(request: Request):
    if not DATABASE_URL:
        raise HTTPException(status_code=500, detail="DATABASE_URL missing")

    form = await request.form()
    user_code = (form.get("user_code") or "").strip().upper()

    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT device_code, expires_at, status FROM device_codes WHERE user_code=%s", (user_code,))
            row = cur.fetchone()
            if not row:
                return HTMLResponse("<h3>Invalid code.</h3>", status_code=400)
            if int(row["expires_at"]) < int(time.time()):
                return HTMLResponse("<h3>Code expired.</h3>", status_code=400)

            cur.execute("UPDATE device_codes SET status='CLAIMED' WHERE user_code=%s", (user_code,))
        conn.commit()

    # redirect to success page
    return HTMLResponse("<h3>Approved. You can close this tab and return to the desktop app.</h3>")


@app.post("/device/poll")
def device_poll(payload: dict):
    """
    Desktop sends: { "device_code": "..." }
    If approved, return { status: "CLAIMED", jwt: "..." }
    """
    if not DATABASE_URL:
        raise HTTPException(status_code=500, detail="DATABASE_URL missing")

    device_code = (payload.get("device_code") or "").strip()
    if not device_code:
        raise HTTPException(status_code=400, detail="Missing device_code")

    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT status, expires_at FROM device_codes WHERE device_code=%s", (device_code,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Unknown device_code")
            if int(row["expires_at"]) < int(time.time()):
                raise HTTPException(status_code=400, detail="Expired device_code")

    if row["status"] != "CLAIMED":
        return {"status": "PENDING"}

    jwt = sign_jwt({"device_code": device_code, "exp": int(time.time()) + 7 * 24 * 3600})
    return {"status": "CLAIMED", "jwt": jwt}


# ============================================================
# OAuth state helpers
# ============================================================

def save_oauth_state(state: str, device_code: str, platform: str, payload: dict, ttl_sec: int = 900):
    now = int(time.time())
    exp = now + ttl_sec
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO oauth_state(state,device_code,platform,payload,expires_at,created_at) "
                "VALUES(%s,%s,%s,%s,%s,%s)",
                (state, device_code, platform, json.dumps(payload), exp, now),
            )
        conn.commit()


def pop_oauth_state(state: str) -> dict:
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT device_code, platform, payload, expires_at FROM oauth_state WHERE state=%s", (state,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=400, detail="Invalid state")
            if int(row["expires_at"]) < int(time.time()):
                raise HTTPException(status_code=400, detail="State expired")
            cur.execute("DELETE FROM oauth_state WHERE state=%s", (state,))
        conn.commit()
    return row


def store_token(device_code: str, platform: str, token_obj: dict):
    blob = encrypt_json(token_obj)
    now = int(time.time())
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO token_vault(device_code,platform,blob,updated_at) "
                "VALUES(%s,%s,%s,%s) "
                "ON CONFLICT (device_code,platform) DO UPDATE SET blob=EXCLUDED.blob, updated_at=EXCLUDED.updated_at",
                (device_code, platform, json.dumps(blob), now),
            )
        conn.commit()


def read_token(device_code: str, platform: str) -> Optional[dict]:
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT blob FROM token_vault WHERE device_code=%s AND platform=%s", (device_code, platform))
            row = cur.fetchone()
            if not row:
                return None
    blob = row["blob"]
    if isinstance(blob, str):
        blob = json.loads(blob)
    return decrypt_json(blob)


# ============================================================
# Google OAuth (YouTube upload scope)
# ============================================================

@app.get("/oauth/google/start")
def google_start(request: Request):
    device_code = bearer_device_code(request)
    if not (GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET):
        raise HTTPException(status_code=500, detail="Google env missing")

    state = secrets.token_urlsafe(18)
    save_oauth_state(state, device_code, "google", {})

    redirect_uri = f"{BASE_URL}/oauth/google/callback"
    params = {
        "response_type": "code",
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "scope": "https://www.googleapis.com/auth/youtube.upload",
        "access_type": "offline",
        "prompt": "consent",
        "state": state,
    }
    url = "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(params)
    return RedirectResponse(url)


@app.get("/oauth/google/callback")
def google_callback(code: str = "", state: str = "", error: str = ""):
    if error:
        return JSONResponse({"ok": False, "error": error}, status_code=400)
    if not code or not state:
        return JSONResponse({"ok": False, "error": "Missing code/state"}, status_code=400)

    st = pop_oauth_state(state)
    device_code = st["device_code"]

    redirect_uri = f"{BASE_URL}/oauth/google/callback"
    token_url = "https://oauth2.googleapis.com/token"
    data = {
        "code": code,
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
    }
    r = requests.post(token_url, data=data, timeout=60)
    payload = r.json()
    if "access_token" not in payload:
        return JSONResponse({"ok": False, "payload": payload}, status_code=400)

    store_token(device_code, "google", payload)
    return HTMLResponse("<h3>Google connected. You can close this tab and return to the desktop app.</h3>")


# ============================================================
# TikTok OAuth (push_by_file: your PC bytes upload)
# ============================================================

def pkce_pair():
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("utf-8")).digest()
    challenge = base64.urlsafe_b64encode(digest).decode("utf-8").rstrip("=")
    return verifier, challenge


@app.get("/oauth/tiktok/start")
def tiktok_start(request: Request):
    device_code = bearer_device_code(request)
    if not (TIKTOK_CLIENT_KEY and TIKTOK_CLIENT_SECRET):
        raise HTTPException(status_code=500, detail="TikTok env missing")

    verifier, challenge = pkce_pair()
    state = secrets.token_urlsafe(18)
    save_oauth_state(state, device_code, "tiktok", {"verifier": verifier})

    redirect_uri = f"{BASE_URL}/oauth/tiktok/callback"
    params = {
        "client_key": TIKTOK_CLIENT_KEY,
        "response_type": "code",
        "scope": "user.info.basic,video.upload",  # draft/inbox (push_by_file)
        "redirect_uri": redirect_uri,
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    url = "https://www.tiktok.com/v2/auth/authorize/?" + urlencode(params, safe=":/,")
    return RedirectResponse(url)


@app.get("/oauth/tiktok/callback")
def tiktok_callback(code: str = "", state: str = "", error: str = "", error_description: str = ""):
    if error:
        return JSONResponse({"ok": False, "error": error, "desc": error_description}, status_code=400)
    if not code or not state:
        return JSONResponse({"ok": False, "error": "Missing code/state"}, status_code=400)

    st = pop_oauth_state(state)
    device_code = st["device_code"]
    payload = st["payload"]
    verifier = payload.get("verifier")
    if not verifier:
        return JSONResponse({"ok": False, "error": "PKCE verifier missing"}, status_code=400)

    token_url = "https://open.tiktokapis.com/v2/oauth/token/"
    redirect_uri = f"{BASE_URL}/oauth/tiktok/callback"
    data = {
        "client_key": TIKTOK_CLIENT_KEY,
        "client_secret": TIKTOK_CLIENT_SECRET,
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": redirect_uri,
        "code_verifier": verifier,
    }
    r = requests.post(token_url, data=data, timeout=60)
    tok = r.json()
    if "access_token" not in tok:
        return JSONResponse({"ok": False, "payload": tok}, status_code=400)

    store_token(device_code, "tiktok", tok)
    return HTMLResponse("<h3>TikTok connected. You can close this tab and return to the desktop app.</h3>")


# ============================================================
# Vault read (Desktop uses this after auth to get tokens)
# ============================================================

@app.get("/vault/token/{platform}")
def vault_get(platform: str, request: Request):
    platform = platform.lower().strip()
    if platform not in ("google", "tiktok", "meta"):
        raise HTTPException(status_code=400, detail="Unsupported platform")

    device_code = bearer_device_code(request)
    tok = read_token(device_code, platform)
    if not tok:
        raise HTTPException(status_code=404, detail="No token stored")

    # Return raw token JSON (desktop can extract access_token/refresh_token)
    return {"ok": True, "platform": platform, "token": tok}
