import os
import secrets
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

import requests
import psycopg
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse, JSONResponse, HTMLResponse
from cryptography.fernet import Fernet

app = FastAPI()

META_APP_ID = os.getenv("META_APP_ID", "")
META_APP_SECRET = os.getenv("META_APP_SECRET", "")
BASE_URL = os.getenv("BASE_URL", "https://auth.uploadm8.com").rstrip("/")
META_API_VERSION = os.getenv("META_API_VERSION", "v23.0")

DATABASE_URL = os.getenv("DATABASE_URL", "")
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", "")

fernet = Fernet(ENCRYPTION_KEY.encode() if isinstance(ENCRYPTION_KEY, str) else ENCRYPTION_KEY)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS oauth_states (
  state TEXT PRIMARY KEY,
  provider TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS oauth_connections (
  id BIGSERIAL PRIMARY KEY,
  provider TEXT NOT NULL,
  fb_user_id TEXT,
  fb_name TEXT,
  access_token_enc TEXT NOT NULL,
  token_type TEXT,
  expires_at TIMESTAMPTZ,
  granted_scopes TEXT,
  created_at TIMESTAMPTZ NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_oauth_connections_provider ON oauth_connections(provider);
CREATE INDEX IF NOT EXISTS idx_oauth_connections_fb_user_id ON oauth_connections(fb_user_id);
"""

def db_conn():
    # Render Postgres typically supports SSL; require keeps it safe across internal/external URLs.
    return psycopg.connect(DATABASE_URL, sslmode="require")

def init_db():
    with db_conn() as conn:
        conn.execute(SCHEMA_SQL)
        conn.commit()

@app.on_event("startup")
def on_startup():
    try:
        init_db()
    except Exception:
        # keep service running; /health will show configured=false
        pass

def save_state(state: str, provider: str):
    with db_conn() as conn:
        conn.execute(
            "INSERT INTO oauth_states (state, provider, created_at) VALUES (%s, %s, %s)",
            (state, provider, datetime.now(timezone.utc)),
        )
        conn.commit()

def pop_state(state: str, provider: str) -> bool:
    with db_conn() as conn:
        cur = conn.execute(
            "DELETE FROM oauth_states WHERE state=%s AND provider=%s RETURNING state",
            (state, provider),
        )
        row = cur.fetchone()
        conn.commit()
        return row is not None

def upsert_connection(provider: str, fb_user_id: str, fb_name: str, access_token: str,
                      token_type: str, expires_in: int, scopes: str):
    token_enc = fernet.encrypt(access_token.encode("utf-8")).decode("utf-8")
    expires_at = (datetime.now(timezone.utc) + timedelta(seconds=int(expires_in))) if expires_in else None
    now = datetime.now(timezone.utc)

    with db_conn() as conn:
        conn.execute(
            """
            INSERT INTO oauth_connections
            (provider, fb_user_id, fb_name, access_token_enc, token_type, expires_at, granted_scopes, created_at, updated_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (provider, fb_user_id, fb_name, token_enc, token_type, expires_at, scopes, now, now),
        )
        conn.commit()

def build_dialog_oauth_url(provider: str, scopes: list[str], redirect_path: str):
    state = secrets.token_urlsafe(24)
    save_state(state, provider)

    redirect_uri = f"{BASE_URL}{redirect_path}"
    params = {
        "client_id": META_APP_ID,
        "redirect_uri": redirect_uri,
        "state": state,
        "response_type": "code",
        "scope": ",".join(scopes),
    }
    return f"https://www.facebook.com/{META_API_VERSION}/dialog/oauth?{urlencode(params)}"

def exchange_code_for_short_token(code: str, redirect_uri: str) -> dict:
    url = f"https://graph.facebook.com/{META_API_VERSION}/oauth/access_token"
    params = {
        "client_id": META_APP_ID,
        "client_secret": META_APP_SECRET,
        "redirect_uri": redirect_uri,
        "code": code,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def exchange_for_long_lived_token(short_token: str) -> dict:
    url = f"https://graph.facebook.com/{META_API_VERSION}/oauth/access_token"
    params = {
        "grant_type": "fb_exchange_token",
        "client_id": META_APP_ID,
        "client_secret": META_APP_SECRET,
        "fb_exchange_token": short_token,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def get_me(access_token: str) -> dict:
    url = f"https://graph.facebook.com/{META_API_VERSION}/me"
    params = {"fields": "id,name", "access_token": access_token}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

@app.get("/health")
def health():
    configured = bool(META_APP_ID and META_APP_SECRET and DATABASE_URL and ENCRYPTION_KEY)
    return {"status": "ok", "configured": configured, "base_url": BASE_URL, "meta_version": META_API_VERSION}

@app.get("/connect/facebook")
def connect_facebook():
    scopes = ["public_profile", "email", "pages_show_list", "business_management"]
    url = build_dialog_oauth_url("facebook", scopes, "/facebook/callback")
    return RedirectResponse(url)

@app.get("/facebook/callback")
def facebook_callback(request: Request):
    if request.query_params.get("error"):
        return JSONResponse(
            {"ok": False, "error": request.query_params.get("error"),
             "desc": request.query_params.get("error_description")},
            status_code=400
        )

    code = request.query_params.get("code")
    state = request.query_params.get("state")
    if not code or not state:
        return JSONResponse({"ok": False, "error": "missing_code_or_state"}, status_code=400)

    if not pop_state(state, "facebook"):
        return JSONResponse({"ok": False, "error": "invalid_state"}, status_code=400)

    redirect_uri = f"{BASE_URL}/facebook/callback"  # must exactly match Meta Valid OAuth Redirect URIs
    short = exchange_code_for_short_token(code, redirect_uri)
    short_token = short.get("access_token")
    token_type = short.get("token_type", "bearer")
    short_expires = int(short.get("expires_in", 0))

    long_data = exchange_for_long_lived_token(short_token)
    long_token = long_data.get("access_token")
    long_expires = int(long_data.get("expires_in", 0))

    me = get_me(long_token)

    upsert_connection(
        provider="facebook",
        fb_user_id=me.get("id"),
        fb_name=me.get("name"),
        access_token=long_token,
        token_type=token_type,
        expires_in=long_expires or short_expires,
        scopes=request.query_params.get("granted_scopes", ""),
    )

    return HTMLResponse(f"<h3>Facebook connected</h3><p>User: {me.get('name')} ({me.get('id')})</p>")

@app.get("/connect/instagram")
def connect_instagram():
    scopes = [
        "public_profile", "email", "pages_show_list", "business_management",
        "instagram_basic", "instagram_content_publish", "pages_read_engagement",
    ]
    url = build_dialog_oauth_url("instagram", scopes, "/instagram/callback")
    return RedirectResponse(url)

@app.get("/instagram/callback")
def instagram_callback(request: Request):
    if request.query_params.get("error"):
        return JSONResponse(
            {"ok": False, "error": request.query_params.get("error"),
             "desc": request.query_params.get("error_description")},
            status_code=400
        )

    code = request.query_params.get("code")
    state = request.query_params.get("state")
    if not code or not state:
        return JSONResponse({"ok": False, "error": "missing_code_or_state"}, status_code=400)

    if not pop_state(state, "instagram"):
        return JSONResponse({"ok": False, "error": "invalid_state"}, status_code=400)

    redirect_uri = f"{BASE_URL}/instagram/callback"  # must exactly match Meta Valid OAuth Redirect URIs
    short = exchange_code_for_short_token(code, redirect_uri)
    short_token = short.get("access_token")
    token_type = short.get("token_type", "bearer")
    short_expires = int(short.get("expires_in", 0))

    long_data = exchange_for_long_lived_token(short_token)
    long_token = long_data.get("access_token")
    long_expires = int(long_data.get("expires_in", 0))

    me = get_me(long_token)

    upsert_connection(
        provider="instagram",
        fb_user_id=me.get("id"),
        fb_name=me.get("name"),
        access_token=long_token,
        token_type=token_type,
        expires_in=long_expires or short_expires,
        scopes=request.query_params.get("granted_scopes", ""),
    )

    return HTMLResponse(f"<h3>Instagram connected</h3><p>User: {me.get('name')} ({me.get('id')})</p>")
