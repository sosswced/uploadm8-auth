"""
UploadM8 Central Auth Service (Render)

Capabilities (MVP, production-safe baseline):
- Email/password auth (register/login)
- Forgot password + reset via Mailgun SMTP (no UI dependency)
- Device-link flow (desktop EXE friendly)
- Central OAuth callbacks for TikTok + Google + Meta
- Encrypted token vault w/ key rotation (AES-256-GCM)
- Audit log table + event stamping
- Health endpoints + config echo (non-secret)

This file is designed to run on Render with:
  uvicorn app:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import smtplib
import string
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
import psycopg2.extras
import requests
import jwt
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field


# ----------------------------
# Environment / Config
# ----------------------------

BASE_URL = (os.getenv("BASE_URL", "https://auth.uploadm8.com") or "").rstrip("/")
DATABASE_URL = os.getenv("DATABASE_URL", "")
JWT_SECRET = os.getenv("JWT_SECRET", "")
JWT_ISSUER = os.getenv("JWT_ISSUER", "uploadm8-auth")
JWT_TTL_MIN = int(os.getenv("JWT_TTL_MIN", "43200"))  # 30 days default

# Token encryption keys: "v1:<b64-32bytes>,v2:<b64-32bytes>"
TOKEN_ENC_KEYS = os.getenv("TOKEN_ENC_KEYS", "")

# Mailgun SMTP
MAILGUN_SMTP_HOST = os.getenv("MAILGUN_SMTP_HOST", "smtp.mailgun.org")
MAILGUN_SMTP_PORT = int(os.getenv("MAILGUN_SMTP_PORT", "587"))
MAILGUN_SMTP_USER = os.getenv("MAILGUN_SMTP_USER", "")
MAILGUN_SMTP_PASS = os.getenv("MAILGUN_SMTP_PASS", "")
MAIL_FROM = os.getenv("MAIL_FROM", "UploadM8 <noreply@uploadm8.com>")
RESET_URL_PATH = os.getenv("RESET_URL_PATH", "/reset-password")  # your Carrd page path (optional)
RESET_TOKEN_TTL_MIN = int(os.getenv("RESET_TOKEN_TTL_MIN", "30"))

# OAuth client creds
META_APP_ID = os.getenv("META_APP_ID", "")
META_APP_SECRET = os.getenv("META_APP_SECRET", "")

TIKTOK_CLIENT_KEY = os.getenv("TIKTOK_CLIENT_KEY", "")
TIKTOK_CLIENT_SECRET = os.getenv("TIKTOK_CLIENT_SECRET", "")

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")

# Redirect URIs
META_REDIRECT_URI = os.getenv("META_REDIRECT_URI", f"{BASE_URL}/oauth/meta/callback")
TIKTOK_REDIRECT_URI = os.getenv("TIKTOK_REDIRECT_URI", f"{BASE_URL}/oauth/tiktok/callback")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", f"{BASE_URL}/oauth/google/callback")

# Scopes (edit in Render env as needed)
META_SCOPE = os.getenv("META_SCOPE", "public_profile,pages_show_list")
TIKTOK_SCOPE = os.getenv("TIKTOK_SCOPE", "user.info.basic,video.upload,video.publish")
GOOGLE_SCOPE = os.getenv("GOOGLE_SCOPE", "https://www.googleapis.com/auth/youtube.upload")

# Strong password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ----------------------------
# Crypto: AES-GCM + key rotation
# ----------------------------

@dataclass
class EncKeyRing:
    keys: Dict[str, bytes]          # kid -> raw bytes
    active_kid: str

    @staticmethod
    def from_env(raw: str) -> "EncKeyRing":
        items = [x.strip() for x in (raw or "").split(",") if x.strip()]
        if not items:
            raise ValueError("TOKEN_ENC_KEYS is empty. Provide at least one key: v1:<base64-32-bytes>")
        keys: Dict[str, bytes] = {}
        order: List[str] = []
        for item in items:
            if ":" not in item:
                raise ValueError("Bad TOKEN_ENC_KEYS format. Expected kid:base64")
            kid, b64 = item.split(":", 1)
            kid = kid.strip()
            b64 = b64.strip()
            try:
                raw_key = base64.b64decode(b64)
            except Exception as e:
                raise ValueError(f"Bad base64 for {kid}: {e}") from e
            if len(raw_key) != 32:
                raise ValueError(f"Key {kid} must be 32 bytes after base64 decode (got {len(raw_key)})")
            keys[kid] = raw_key
            order.append(kid)
        return EncKeyRing(keys=keys, active_kid=order[-1])

    def encrypt_json(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        import json as _json
        key = self.keys[self.active_kid]
        aes = AESGCM(key)
        nonce = secrets.token_bytes(12)
        pt = _json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ct = aes.encrypt(nonce, pt, None)
        return {
            "kid": self.active_kid,
            "nonce": base64.b64encode(nonce).decode("utf-8"),
            "ct": base64.b64encode(ct).decode("utf-8"),
        }

    def decrypt_json(self, blob: Dict[str, Any]) -> Dict[str, Any]:
        import json as _json
        kid = blob.get("kid")
        if not kid or kid not in self.keys:
            raise ValueError("Unknown kid in token blob")
        key = self.keys[kid]
        aes = AESGCM(key)
        nonce = base64.b64decode(blob.get("nonce", ""))
        ct = base64.b64decode(blob.get("ct", ""))
        pt = aes.decrypt(nonce, ct, None)
        return _json.loads(pt.decode("utf-8"))


def _require_env() -> Tuple[EncKeyRing, None]:
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is missing")
    if not JWT_SECRET or len(JWT_SECRET) < 32:
        raise RuntimeError("JWT_SECRET missing/too short (recommend 32+ chars)")
    ring = EncKeyRing.from_env(TOKEN_ENC_KEYS)
    return ring, None


# ----------------------------
# DB helpers
# ----------------------------

def db_conn():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL missing")
    return psycopg2.connect(DATABASE_URL, sslmode="require")


SCHEMA_SQL = r"""
CREATE EXTENSION IF NOT EXISTS pgcrypto;

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

CREATE TABLE IF NOT EXISTS oauth_states (
  state TEXT PRIMARY KEY,
  device_code TEXT NOT NULL REFERENCES device_codes(device_code) ON DELETE CASCADE,
  platform TEXT NOT NULL,
  user_id UUID NULL REFERENCES users(id) ON DELETE SET NULL,
  expires_at TIMESTAMPTZ NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS meta_selected (
  user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
  page_id TEXT NOT NULL,
  page_name TEXT NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS token_vault (
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  platform TEXT NOT NULL,
  blob JSONB NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (user_id, platform)
);

CREATE TABLE IF NOT EXISTS audit_log (
  id BIGSERIAL PRIMARY KEY,
  event TEXT NOT NULL,
  user_id UUID NULL,
  meta JSONB NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Optional: legacy compatibility (Meta pages selection)
CREATE TABLE IF NOT EXISTS user_settings (
  fb_user_id TEXT PRIMARY KEY,
  selected_page_id TEXT,
  selected_page_name TEXT,
  updated_at TIMESTAMPTZ NOT NULL
);
"""


def init_db():
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
        conn.commit()


def audit(event: str, user_id: Optional[str] = None, meta: Optional[dict] = None) -> None:
    try:
        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO audit_log(event, user_id, meta) VALUES (%s, %s, %s)",
                    (event, user_id, psycopg2.extras.Json(meta) if meta is not None else None),
                )
            conn.commit()
    except Exception:
        # don't crash auth path for audit failures
        return


# ----------------------------
# JWT Auth
# ----------------------------

def make_jwt(user_id: str) -> str:
    now = int(time.time())
    payload = {
        "iss": JWT_ISSUER,
        "sub": user_id,
        "iat": now,
        "exp": now + (JWT_TTL_MIN * 60),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def require_user(request: Request) -> str:
    authz = request.headers.get("authorization", "")
    if not authz.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authz.split(" ", 1)[1].strip()
    try:
        data = jwt.decode(token, JWT_SECRET, algorithms=["HS256"], issuer=JWT_ISSUER)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")
    return str(data.get("sub"))


# ----------------------------
# Mailgun SMTP
# ----------------------------

def send_mail(to_email: str, subject: str, body: str) -> None:
    if not (MAILGUN_SMTP_USER and MAILGUN_SMTP_PASS):
        # MVP: if SMTP not configured, log to Render so you can copy-paste reset link
        audit("mail_not_configured", None, {"to": to_email, "subject": subject, "body": body[:500]})
        return

    msg = EmailMessage()
    msg["From"] = MAIL_FROM
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP(MAILGUN_SMTP_HOST, MAILGUN_SMTP_PORT) as s:
        s.starttls()
        s.login(MAILGUN_SMTP_USER, MAILGUN_SMTP_PASS)
        s.send_message(msg)


def hash_token(token: str) -> str:
    # stable hash for storage
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


# ----------------------------
# FastAPI app
# ----------------------------

app = FastAPI(title="UploadM8 Auth Service")

@app.on_event("startup")
def _startup():
    _require_env()
    init_db()
    audit("startup_ok", None, {"base_url": BASE_URL})


@app.get("/health")
def health():
    return {
        "status": "ok",
        "configured": bool(DATABASE_URL and JWT_SECRET and TOKEN_ENC_KEYS),
        "base_url": BASE_URL,
    }


# ----------------------------
# Auth: register/login/reset
# ----------------------------

class RegisterIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=256)

class LoginIn(BaseModel):
    email: EmailStr
    password: str

class ForgotIn(BaseModel):
    email: EmailStr

class ResetIn(BaseModel):
    token: str
    new_password: str = Field(min_length=8, max_length=256)

@app.post("/auth/register")
def register(payload: RegisterIn):
    email = payload.email.lower().strip()
    pw_hash = pwd_context.hash(payload.password)

    with db_conn() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    "INSERT INTO users(email, password_hash) VALUES (%s, %s) RETURNING id",
                    (email, pw_hash),
                )
                user_id = cur.fetchone()[0]
            except psycopg2.Error:
                raise HTTPException(status_code=400, detail="Email already registered")
        conn.commit()

    audit("register", str(user_id), {"email": email})
    return {"ok": True, "token": make_jwt(str(user_id))}

@app.post("/auth/login")
def login(payload: LoginIn):
    email = payload.email.lower().strip()
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, password_hash FROM users WHERE email=%s", (email,))
            row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    user_id, pw_hash = row
    if not pwd_context.verify(payload.password, pw_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    audit("login", str(user_id), {"email": email})
    return {"ok": True, "token": make_jwt(str(user_id))}

@app.post("/auth/forgot")
def forgot(payload: ForgotIn):
    email = payload.email.lower().strip()
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM users WHERE email=%s", (email,))
            row = cur.fetchone()
    # Always return ok to avoid user enumeration
    if not row:
        return {"ok": True}

    user_id = str(row[0])
    token = secrets.token_urlsafe(32)
    th = hash_token(token)
    exp = _utcnow() + timedelta(minutes=RESET_TOKEN_TTL_MIN)

    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO password_resets(token_hash, user_id, expires_at) VALUES (%s, %s, %s)",
                (th, user_id, exp),
            )
        conn.commit()

    # Reset link: either your Carrd page, or direct API call.
    reset_link = f"{BASE_URL}{RESET_URL_PATH}?token={token}"
    body = f"""UploadM8 password reset

Someone requested a password reset for this email.

Reset link (expires in {RESET_TOKEN_TTL_MIN} minutes):
{reset_link}

If you didn't request this, ignore this email.
"""
    send_mail(email, "UploadM8 password reset", body)
    audit("forgot_password", user_id, {"email": email})
    return {"ok": True}

@app.post("/auth/reset")
def reset_password(payload: ResetIn):
    th = hash_token(payload.token)
    new_hash = pwd_context.hash(payload.new_password)

    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_id, expires_at FROM password_resets WHERE token_hash=%s",
                (th,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=400, detail="Invalid or expired token")
            user_id, expires_at = row
            if expires_at < _utcnow():
                raise HTTPException(status_code=400, detail="Invalid or expired token")
            cur.execute("UPDATE users SET password_hash=%s WHERE id=%s", (new_hash, user_id))
            cur.execute("DELETE FROM password_resets WHERE token_hash=%s", (th,))
        conn.commit()

    audit("password_reset", str(user_id), None)
    return {"ok": True}


# ----------------------------
# Device-link flow (desktop)
# ----------------------------

def _gen_user_code() -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return "".join(secrets.choice(alphabet) for _ in range(8))

@app.post("/device/code")
def device_code():
    device_code = secrets.token_urlsafe(24)
    user_code = _gen_user_code()
    exp = _utcnow() + timedelta(minutes=30)

    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO device_codes(device_code, user_code, status, expires_at) VALUES (%s, %s, %s, %s)",
                (device_code, user_code, "PENDING", exp),
            )
        conn.commit()

    audit("device_code_created", None, {"user_code": user_code})
    return {
        "ok": True,
        "device_code": device_code,
        "user_code": user_code,
        "verification_url": f"{BASE_URL}/device",
        "expires_at": exp.isoformat(),
    }

class DeviceClaimIn(BaseModel):
    user_code: str = Field(min_length=4, max_length=32)

@app.post("/device/claim")
def device_claim(payload: DeviceClaimIn, user_id: str = Depends(require_user)):
    code = payload.user_code.strip().upper()
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT device_code, status, expires_at FROM device_codes WHERE user_code=%s",
                (code,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=400, detail="Invalid code")
            device_code, status, expires_at = row
            if expires_at < _utcnow():
                raise HTTPException(status_code=400, detail="Code expired")
            cur.execute(
                "UPDATE device_codes SET user_id=%s, status='CLAIMED' WHERE user_code=%s",
                (user_id, code),
            )
        conn.commit()

    audit("device_claimed", user_id, {"user_code": code})
    return {"ok": True}

@app.get("/device/status")
def device_status(device_code: str):
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_id, status, expires_at FROM device_codes WHERE device_code=%s",
                (device_code,),
            )
            row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Not found")
    user_id, status, expires_at = row
    if expires_at < _utcnow():
        raise HTTPException(status_code=400, detail="Expired")
    if status != "CLAIMED" or not user_id:
        return {"ok": True, "status": "PENDING"}
    # return desktop JWT
    token = make_jwt(str(user_id))
    return {"ok": True, "status": "CLAIMED", "token": token}

@app.get("/device", response_class=HTMLResponse)
def device_page(request: Request):
    # Simple device-link UI: login/register -> claim code -> connect platforms.
    code_prefill = (request.query_params.get("code") or "").strip().upper()
    DEVICE_HTML_TEMPLATE = """
    <html>
      <head>
        <title>UploadM8 Device Link</title>
        <meta name="viewport" content="width=device-width,initial-scale=1" />
      </head>
      <body style="font-family: Arial; max-width: 860px; margin: 30px auto; padding: 0 16px;">
        <h2>UploadM8 — Link your desktop app</h2>
        <ol>
          <li>Enter your email + password and click <b>Login</b> (or <b>Register</b> if new).</li>
          <li>Paste the 8‑character code shown in your desktop app and click <b>Link Device</b>.</li>
          <li>Click <b>Connect</b> for the platform(s) you want, complete login, then return to your desktop app.</li>
        </ol>

        <hr/>

        <h3>1) Account</h3>
        <div>
          <label>Email</label><br/>
          <input id="email" style="width:100%;padding:10px;" placeholder="you@domain.com" />
        </div>
        <div style="margin-top:10px;">
          <label>Password</label><br/>
          <input id="password" type="password" style="width:100%;padding:10px;" placeholder="••••••••" />
        </div>
        <div style="margin-top:12px;">
          <button onclick="doRegister()" style="padding:10px 14px;">Register</button>
          <button onclick="doLogin()" style="padding:10px 14px;margin-left:8px;">Login</button>
          <span id="auth_status" style="margin-left:12px;"></span>
        </div>

        <h3 style="margin-top:24px;">2) Link Device</h3>
        <div>
          <label>Code from desktop app</label><br/>
          <input id="user_code" style="width:220px;padding:10px;letter-spacing:2px;text-transform:uppercase" value="__CODE__" />
          <button onclick="doClaim()" style="padding:10px 14px;margin-left:8px;">Link Device</button>
          <span id="claim_status" style="margin-left:12px;"></span>
        </div>

        <h3 style="margin-top:24px;">3) Connect Platforms</h3>
        <p id="connect_hint">Link your device first.</p>
        <div>
          <button id="btn_tiktok" onclick="doConnect('tiktok')" disabled style="padding:10px 14px;">Connect TikTok</button>
          <button id="btn_google" onclick="doConnect('google')" disabled style="padding:10px 14px;margin-left:8px;">Connect Google/YouTube</button>
          <button id="btn_meta" onclick="doConnect('meta')" disabled style="padding:10px 14px;margin-left:8px;">Connect Meta (FB/IG)</button>
        </div>

        <hr style="margin-top:28px;"/>
        <p style="font-size:13px;color:#666;">
          If you forget your password: call <code>POST /auth/forgot</code> with your email. A reset link is emailed (Mailgun SMTP) or printed to logs if SMTP isn't configured yet.
        </p>

        <script>
          function getToken(){ return localStorage.getItem("uploadm8_jwt") || ""; }
          function setToken(t){ localStorage.setItem("uploadm8_jwt", t); }
          function authHeader(){
            const t = getToken();
            return t ? {"authorization":"Bearer " + t} : {};
          }
          async function post(path, body, auth=false){
            const headers = {"content-type":"application/json", ...(auth?authHeader():{})};
            const r = await fetch(path, {method:"POST", headers, body: JSON.stringify(body||{})});
            const text = await r.text();
            let j = {};
            try { j = JSON.parse(text); } catch(e) {}
            if(!r.ok){ throw new Error(text || ("HTTP " + r.status)); }
            return j;
          }
          async function doRegister(){
            try{
              const email = document.getElementById("email").value.trim();
              const password = document.getElementById("password").value;
              const j = await post("/auth/register",{email,password});
              setToken(j.token);
              document.getElementById("auth_status").innerText = "Logged in ✅";
            }catch(e){
              document.getElementById("auth_status").innerText = "Register failed ❌";
              alert(e);
            }
          }
          async function doLogin(){
            try{
              const email = document.getElementById("email").value.trim();
              const password = document.getElementById("password").value;
              const j = await post("/auth/login",{email,password});
              setToken(j.token);
              document.getElementById("auth_status").innerText = "Logged in ✅";
            }catch(e){
              document.getElementById("auth_status").innerText = "Login failed ❌";
              alert(e);
            }
          }
          let deviceCode = "";
          async function doClaim(){
            try{
              const user_code = document.getElementById("user_code").value.trim().toUpperCase();
              const j = await post("/device/claim",{user_code}, true);
              deviceCode = j.device_code;
              document.getElementById("claim_status").innerText = "Linked ✅";
              document.getElementById("connect_hint").innerText = "Now connect the platforms you want:";
              document.getElementById("btn_tiktok").disabled = false;
              document.getElementById("btn_google").disabled = false;
              document.getElementById("btn_meta").disabled = false;
            }catch(e){
              document.getElementById("claim_status").innerText = "Link failed ❌";
              alert(e);
            }
          }
          async function doConnect(platform){
            try{
              if(!deviceCode){ alert("Link your device first."); return; }
              const j = await post("/oauth/" + platform + "/start",{device_code: deviceCode}, true);
              window.location.href = j.auth_url;
            }catch(e){
              alert(e);
            }
          }
        </script>
      </body>
    </html>
    """
    html = DEVICE_HTML_TEMPLATE.replace("__CODE__", code_prefill)
    return HTMLResponse(html)
    return HTMLResponse(html)



# ----------------------------
# OAuth: Start + Callback
# Associates OAuth with a device_code using state
# ----------------------------

def _create_state(device_code: str, platform: str, user_id: str) -> str:
    state = secrets.token_urlsafe(24)
    exp = _utcnow() + timedelta(minutes=15)
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO oauth_states(state, device_code, platform, user_id, expires_at) VALUES (%s,%s,%s,%s,%s)",
                (state, device_code, platform, user_id, exp),
            )
        conn.commit()
    return state

def _consume_state(state: str, expected_platform: str) -> Tuple[str, str]:
    # returns (device_code, user_id)
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT device_code, user_id, platform, expires_at FROM oauth_states WHERE state=%s",
                (state,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=400, detail="Invalid state")
            device_code, user_id, platform, exp = row
            if exp < _utcnow():
                raise HTTPException(status_code=400, detail="State expired")
            if platform != expected_platform:
                raise HTTPException(status_code=400, detail="State platform mismatch")
            cur.execute("DELETE FROM oauth_states WHERE state=%s", (state,))
        conn.commit()
    return str(device_code), str(user_id)

class OAuthStartIn(BaseModel):
    device_code: str

@app.post("/oauth/tiktok/start")
def oauth_tiktok_start(payload: OAuthStartIn, user_id: str = Depends(require_user)):
    if not TIKTOK_CLIENT_KEY:
        raise HTTPException(status_code=500, detail="TikTok client not configured")
    state = _create_state(payload.device_code, "tiktok", user_id)
    auth_url = (
        "https://www.tiktok.com/v2/auth/authorize/"
        f"?client_key={TIKTOK_CLIENT_KEY}"
        "&response_type=code"
        f"&scope={TIKTOK_SCOPE}"
        f"&redirect_uri={TIKTOK_REDIRECT_URI}"
        f"&state={state}"
    )
    return {"ok": True, "auth_url": auth_url}

@app.get("/oauth/tiktok/callback")
def oauth_tiktok_callback(code: Optional[str] = None, state: Optional[str] = None, error: Optional[str] = None):
    if error:
        return JSONResponse({"ok": False, "platform": "tiktok", "error": error}, status_code=400)
    if not (code and state):
        return {"ok": True, "platform": "tiktok", "callback": "ready"}
    _, user_id = _consume_state(state, "tiktok")

    # Exchange code -> token
    # TikTok v2 token endpoint
    token_url = "https://open.tiktokapis.com/v2/oauth/token/"
    data = {
        "client_key": TIKTOK_CLIENT_KEY,
        "client_secret": TIKTOK_CLIENT_SECRET,
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": TIKTOK_REDIRECT_URI,
    }
    r = requests.post(token_url, data=data, timeout=30)
    if r.status_code >= 400:
        audit("tiktok_token_exchange_failed", user_id, {"status": r.status_code, "body": r.text[:800]})
        return JSONResponse({"ok": False, "platform": "tiktok", "status": r.status_code, "body": r.text[:500]}, status_code=400)

    token_payload = r.json()
    ring, _ = _require_env()
    enc = ring.encrypt_json(token_payload)

    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO token_vault(user_id, platform, blob)
                VALUES (%s, %s, %s)
                ON CONFLICT (user_id, platform) DO UPDATE SET blob=EXCLUDED.blob, updated_at=now()
                """,
                (user_id, "tiktok", psycopg2.extras.Json(enc)),
            )
        conn.commit()

    audit("tiktok_connected", user_id, {"scope": TIKTOK_SCOPE})
    return RedirectResponse(f"{BASE_URL}/oauth/success?platform=tiktok")

@app.post("/oauth/google/start")
def oauth_google_start(payload: OAuthStartIn, user_id: str = Depends(require_user)):
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=500, detail="Google client not configured")
    state = _create_state(payload.device_code, "google", user_id)
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
    return {"ok": True, "auth_url": auth_url}

@app.get("/oauth/google/callback")
def oauth_google_callback(code: Optional[str] = None, state: Optional[str] = None, error: Optional[str] = None):
    if error:
        return JSONResponse({"ok": False, "platform": "google", "error": error}, status_code=400)
    if not (code and state):
        return {"ok": True, "platform": "google", "callback": "ready"}
    _, user_id = _consume_state(state, "google")

    # Exchange code -> token
    token_url = "https://oauth2.googleapis.com/token"
    data = {
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": GOOGLE_REDIRECT_URI,
    }
    r = requests.post(token_url, data=data, timeout=30)
    if r.status_code >= 400:
        audit("google_token_exchange_failed", user_id, {"status": r.status_code, "body": r.text[:800]})
        return JSONResponse({"ok": False, "platform": "google", "status": r.status_code, "body": r.text[:500]}, status_code=400)
    token_payload = r.json()
    ring, _ = _require_env()
    enc = ring.encrypt_json(token_payload)

    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO token_vault(user_id, platform, blob)
                VALUES (%s, %s, %s)
                ON CONFLICT (user_id, platform) DO UPDATE SET blob=EXCLUDED.blob, updated_at=now()
                """,
                (user_id, "google", psycopg2.extras.Json(enc)),
            )
        conn.commit()

    audit("google_connected", user_id, {"scope": GOOGLE_SCOPE})
    return RedirectResponse(f"{BASE_URL}/oauth/success?platform=google")

@app.post("/oauth/meta/start")
def oauth_meta_start(payload: OAuthStartIn, user_id: str = Depends(require_user)):
    if not META_APP_ID:
        raise HTTPException(status_code=500, detail="Meta app not configured")
    state = _create_state(payload.device_code, "meta", user_id)
    auth_url = (
        "https://www.facebook.com/v23.0/dialog/oauth"
        f"?client_id={META_APP_ID}"
        f"&redirect_uri={META_REDIRECT_URI}"
        f"&scope={META_SCOPE}"
        f"&state={state}"
        "&response_type=code"
    )
    return {"ok": True, "auth_url": auth_url}

@app.get("/oauth/meta/callback")
def oauth_meta_callback(code: Optional[str] = None, state: Optional[str] = None, error: Optional[str] = None):
    if error:
        return JSONResponse({"ok": False, "platform": "meta", "error": error}, status_code=400)
    if not (code and state):
        return {"ok": True, "platform": "meta", "callback": "ready"}
    _, user_id = _consume_state(state, "meta")

    token_url = "https://graph.facebook.com/v23.0/oauth/access_token"
    params = {
        "client_id": META_APP_ID,
        "client_secret": META_APP_SECRET,
        "redirect_uri": META_REDIRECT_URI,
        "code": code,
    }
    r = requests.get(token_url, params=params, timeout=30)
    if r.status_code >= 400:
        audit("meta_token_exchange_failed", user_id, {"status": r.status_code, "body": r.text[:800]})
        return JSONResponse({"ok": False, "platform": "meta", "status": r.status_code, "body": r.text[:500]}, status_code=400)
    token_payload = r.json()
    ring, _ = _require_env()
    enc = ring.encrypt_json(token_payload)

    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO token_vault(user_id, platform, blob)
                VALUES (%s, %s, %s)
                ON CONFLICT (user_id, platform) DO UPDATE SET blob=EXCLUDED.blob, updated_at=now()
                """,
                (user_id, "meta", psycopg2.extras.Json(enc)),
            )
        conn.commit()

    audit("meta_connected", user_id, {"scope": META_SCOPE})
    return RedirectResponse(f"{BASE_URL}/oauth/success?platform=meta")


# ----------------------------
# API: return decrypted tokens to desktop EXE
# ----------------------------

@app.get("/api/token/{platform}")
def api_get_token(platform: str, user_id: str = Depends(require_user)):
    platform = platform.lower().strip()
    if platform not in ("tiktok", "google", "meta"):
        raise HTTPException(status_code=400, detail="Unsupported platform")

    with db_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT blob FROM token_vault WHERE user_id=%s AND platform=%s",
                (user_id, platform),
            )
            row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="No token on file for this platform")

    ring, _ = _require_env()
    token_payload = ring.decrypt_json(row["blob"])
    # Return minimal secrets needed; desktop should store in-memory, not print to logs.
    return {"ok": True, "platform": platform, "token": token_payload}


@app.get("/api/meta/pages")
def api_meta_pages(user_id: str = Depends(require_user)):
    # Returns pages the user can manage (id, name, access_token)
    # Requires Meta user token in vault.
    with db_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT blob FROM token_vault WHERE user_id=%s AND platform='meta'", (user_id,))
            row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Meta not connected")

    ring, _ = _require_env()
    user_token = ring.decrypt_json(row["blob"]).get("access_token")
    if not user_token:
        raise HTTPException(status_code=500, detail="Meta token missing access_token")

    url = "https://graph.facebook.com/v23.0/me/accounts"
    params = {"access_token": user_token, "fields": "id,name,access_token"}
    r = requests.get(url, params=params, timeout=30)
    if r.status_code >= 400:
        audit("meta_pages_failed", user_id, {"status": r.status_code, "body": r.text[:800]})
        raise HTTPException(status_code=400, detail="Failed to fetch pages")
    data = r.json()
    pages = data.get("data", []) or []
    return {"ok": True, "pages": pages}

class SelectPageIn(BaseModel):
    page_id: str

@app.post("/api/meta/select_page")
def api_meta_select_page(payload: SelectPageIn, user_id: str = Depends(require_user)):
    # Looks up the page token and stores it encrypted for future use.
    pages = api_meta_pages(user_id=user_id)["pages"]
    chosen = None
    for p in pages:
        if str(p.get("id")) == str(payload.page_id):
            chosen = p
            break
    if not chosen:
        raise HTTPException(status_code=400, detail="Page not found in /me/accounts")

    ring, _ = _require_env()
    enc_page = ring.encrypt_json({"page_id": chosen["id"], "page_name": chosen.get("name",""), "access_token": chosen.get("access_token","")})

    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO token_vault(user_id, platform, blob)
                VALUES (%s, %s, %s)
                ON CONFLICT (user_id, platform) DO UPDATE SET blob=EXCLUDED.blob, updated_at=now()
                """,
                (user_id, "meta_page", psycopg2.extras.Json(enc_page)),
            )
            cur.execute(
                """
                INSERT INTO meta_selected(user_id, page_id, page_name)
                VALUES (%s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE SET page_id=EXCLUDED.page_id, page_name=EXCLUDED.page_name, updated_at=now()
                """,
                (user_id, chosen["id"], chosen.get("name","")),
            )
        conn.commit()

    audit("meta_page_selected", user_id, {"page_id": chosen["id"], "page_name": chosen.get("name","")})
    return {"ok": True, "page_id": chosen["id"], "page_name": chosen.get("name","")}

@app.get("/api/meta/page_token")
def api_meta_page_token(user_id: str = Depends(require_user)):
    with db_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT blob FROM token_vault WHERE user_id=%s AND platform='meta_page'", (user_id,))
            row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="No page selected")
    ring, _ = _require_env()
    page_payload = ring.decrypt_json(row["blob"])
    return {"ok": True, "token": page_payload}

@app.get("/oauth/success", response_class=HTMLResponse)
def oauth_success(platform: str = ""):
    platform = (platform or "").strip()
    return HTMLResponse(
        f"<html><body style='font-family:Arial;max-width:720px;margin:40px auto;'>"
        f"<h2>Connected: {platform}</h2>"
        f"<p>You can close this tab and return to UploadM8.</p>"
        f"</body></html>"
    )
