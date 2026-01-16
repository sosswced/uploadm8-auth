import os
import time
import json
import secrets
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse, HTMLResponse, JSONResponse

import psycopg
from cryptography.fernet import Fernet


# =============================================================================
# Config
# =============================================================================

META_APP_ID = os.getenv("META_APP_ID", "").strip()
META_APP_SECRET = os.getenv("META_APP_SECRET", "").strip()
BASE_URL = os.getenv("BASE_URL", "").strip().rstrip("/")  # e.g. https://auth.uploadm8.com
META_API_VERSION = os.getenv("META_API_VERSION", "v23.0").strip()  # e.g. v23.0
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", "").strip()

REQUEST_TIMEOUT = 25  # seconds

configured = all([META_APP_ID, META_APP_SECRET, BASE_URL, META_API_VERSION, DATABASE_URL, ENCRYPTION_KEY])

fernet: Optional[Fernet] = None
if ENCRYPTION_KEY:
    try:
        fernet = Fernet(ENCRYPTION_KEY.encode("utf-8"))
    except Exception:
        fernet = None
        configured = False


# =============================================================================
# DB Schema
# =============================================================================

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS oauth_connections (
  provider TEXT NOT NULL,
  fb_user_id TEXT NOT NULL,
  access_token_enc TEXT NOT NULL,
  token_type TEXT,
  expires_at TIMESTAMPTZ,
  scope TEXT,
  created_at TIMESTAMPTZ NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL,
  PRIMARY KEY (provider, fb_user_id)
);

-- Route A: user-selected default Page target (NEW TABLE)
CREATE TABLE IF NOT EXISTS user_settings (
  fb_user_id TEXT PRIMARY KEY,
  selected_page_id TEXT,
  selected_page_name TEXT,
  updated_at TIMESTAMPTZ NOT NULL
);
"""


def db_connect():
    # psycopg3 uses libpq-style URL; Render provides compatible DATABASE_URL
    return psycopg.connect(DATABASE_URL, autocommit=True)


def init_db():
    if not DATABASE_URL:
        return
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)


def utcnow():
    return datetime.now(timezone.utc)


def encrypt_token(token: str) -> str:
    if not fernet:
        raise RuntimeError("ENCRYPTION_KEY not configured correctly")
    return fernet.encrypt(token.encode("utf-8")).decode("utf-8")


def decrypt_token(token_enc: str) -> str:
    if not fernet:
        raise RuntimeError("ENCRYPTION_KEY not configured correctly")
    return fernet.decrypt(token_enc.encode("utf-8")).decode("utf-8")


# =============================================================================
# Meta helpers
# =============================================================================

def meta_graph_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    url = f"https://graph.facebook.com/{META_API_VERSION}/{path.lstrip('/')}"
    r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    try:
        data = r.json()
    except Exception:
        raise HTTPException(status_code=502, detail=f"Meta GET failed (non-JSON): {r.status_code}")
    if r.status_code >= 400 or "error" in data:
        raise HTTPException(status_code=502, detail={"meta_error": data})
    return data


def meta_graph_post(path: str, data: Dict[str, Any]) -> Dict[str, Any]:
    url = f"https://graph.facebook.com/{META_API_VERSION}/{path.lstrip('/')}"
    r = requests.post(url, data=data, timeout=REQUEST_TIMEOUT)
    try:
        out = r.json()
    except Exception:
        raise HTTPException(status_code=502, detail=f"Meta POST failed (non-JSON): {r.status_code}")
    if r.status_code >= 400 or "error" in out:
        raise HTTPException(status_code=502, detail={"meta_error": out})
    return out


def exchange_code_for_token(code: str, redirect_uri: str) -> Dict[str, Any]:
    # Standard OAuth token exchange against Graph API
    token_url = f"https://graph.facebook.com/{META_API_VERSION}/oauth/access_token"
    params = {
        "client_id": META_APP_ID,
        "client_secret": META_APP_SECRET,
        "redirect_uri": redirect_uri,
        "code": code,
    }
    r = requests.get(token_url, params=params, timeout=REQUEST_TIMEOUT)
    try:
        data = r.json()
    except Exception:
        raise HTTPException(status_code=502, detail=f"Token exchange failed (non-JSON): {r.status_code}")
    if r.status_code >= 400 or "error" in data:
        raise HTTPException(status_code=502, detail={"meta_error": data})
    return data


def get_fb_user_profile(access_token: str) -> Dict[str, Any]:
    # Basic identity read
    return meta_graph_get("me", {"fields": "id,name", "access_token": access_token})


# =============================================================================
# Persistence helpers
# =============================================================================

def upsert_connection(provider: str, fb_user_id: str, access_token: str, token_type: str = "", scope: str = "", expires_in: Optional[int] = None):
    token_enc = encrypt_token(access_token)
    now = utcnow()

    expires_at = None
    if expires_in is not None:
        expires_at = now.timestamp() + int(expires_in)
        expires_at = datetime.fromtimestamp(expires_at, tz=timezone.utc)

    sql = """
    INSERT INTO oauth_connections (provider, fb_user_id, access_token_enc, token_type, expires_at, scope, created_at, updated_at)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (provider, fb_user_id)
    DO UPDATE SET
      access_token_enc = EXCLUDED.access_token_enc,
      token_type = EXCLUDED.token_type,
      expires_at = EXCLUDED.expires_at,
      scope = EXCLUDED.scope,
      updated_at = EXCLUDED.updated_at;
    """
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (provider, fb_user_id, token_enc, token_type, expires_at, scope, now, now))


def get_connection_token(provider: str, fb_user_id: str) -> str:
    sql = "SELECT access_token_enc FROM oauth_connections WHERE provider=%s AND fb_user_id=%s;"
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (provider, fb_user_id))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail=f"No OAuth connection found for provider={provider} fb_user_id={fb_user_id}")
            return decrypt_token(row[0])


def upsert_selected_page(fb_user_id: str, page_id: str, page_name: str):
    now = utcnow()
    sql = """
    INSERT INTO user_settings (fb_user_id, selected_page_id, selected_page_name, updated_at)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (fb_user_id)
    DO UPDATE SET
      selected_page_id = EXCLUDED.selected_page_id,
      selected_page_name = EXCLUDED.selected_page_name,
      updated_at = EXCLUDED.updated_at;
    """
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (fb_user_id, page_id, page_name, now))


def get_selected_page(fb_user_id: str) -> Dict[str, Optional[str]]:
    sql = "SELECT selected_page_id, selected_page_name, updated_at FROM user_settings WHERE fb_user_id=%s;"
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (fb_user_id,))
            row = cur.fetchone()
            if not row:
                return {"selected_page_id": None, "selected_page_name": None, "updated_at": None}
            return {
                "selected_page_id": row[0],
                "selected_page_name": row[1],
                "updated_at": row[2].isoformat() if row[2] else None
            }


def get_page_access_token_from_me_accounts(user_access_token: str, target_page_id: str) -> str:
    """
    Pull the Page access token from /me/accounts response and match on page_id.
    /me/accounts can return page access tokens when permissions are granted. :contentReference[oaicite:3]{index=3}
    """
    data = meta_graph_get(
        "me/accounts",
        {
            "fields": "id,name,access_token",
            "access_token": user_access_token
        }
    )
    pages = data.get("data", []) or []
    for p in pages:
        if str(p.get("id")) == str(target_page_id):
            tok = p.get("access_token")
            if not tok:
                break
            return tok
    raise HTTPException(status_code=400, detail="Selected page not found in /me/accounts or no page access_token returned. Verify permissions and that you manage the Page.")


# =============================================================================
# FastAPI app
# =============================================================================

app = FastAPI(title="UploadM8 Auth", version="1.0.0")


@app.on_event("startup")
def _startup():
    if configured:
        init_db()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "configured": configured,
        "base_url": BASE_URL,
        "meta_version": META_API_VERSION,
    }


# =============================================================================
# OAuth Connect + Callback (Facebook / Instagram)
# NOTE: Keep redirect URIs exactly matched to Meta dashboard.
# =============================================================================

@app.get("/connect/facebook")
def connect_facebook():
    if not configured:
        raise HTTPException(status_code=500, detail="Service not configured")

    redirect_uri = f"{BASE_URL}/facebook/callback"
    state = secrets.token_urlsafe(24)
    scope = "public_profile,pages_show_list,publish_video,pages_read_engagement,pages_manage_posts"
    auth_url = (
        f"https://www.facebook.com/{META_API_VERSION}/dialog/oauth"
        f"?client_id={META_APP_ID}"
        f"&redirect_uri={redirect_uri}"
        f"&state={state}"
        f"&response_type=code"
        f"&scope={scope}"
    )
    return RedirectResponse(auth_url)


@app.get("/facebook/callback")
def facebook_callback(code: Optional[str] = None, state: Optional[str] = None, error: Optional[str] = None, error_description: Optional[str] = None):
    if error:
        raise HTTPException(status_code=400, detail={"error": error, "error_description": error_description})

    if not code:
        raise HTTPException(status_code=400, detail="Missing code")

    redirect_uri = f"{BASE_URL}/facebook/callback"
    token_data = exchange_code_for_token(code, redirect_uri)
    access_token = token_data.get("access_token")
    token_type = token_data.get("token_type", "")
    expires_in = token_data.get("expires_in")

    if not access_token:
        raise HTTPException(status_code=502, detail={"meta_error": token_data})

    profile = get_fb_user_profile(access_token)
    fb_user_id = str(profile.get("id", ""))

    upsert_connection(
        provider="facebook",
        fb_user_id=fb_user_id,
        access_token=access_token,
        token_type=token_type,
        scope="",
        expires_in=expires_in
    )

    return HTMLResponse(f"Facebook connected<br/>User: {profile.get('name','')} ({fb_user_id})")


@app.get("/connect/instagram")
def connect_instagram():
    """
    For your current setup you already have this working and returning the same user id.
    Keep it consistent with your configured redirect URI.
    """
    if not configured:
        raise HTTPException(status_code=500, detail="Service not configured")

    redirect_uri = f"{BASE_URL}/instagram/callback"
    state = secrets.token_urlsafe(24)

    # If you later need IG-specific scopes, add them here.
    scope = "public_profile"
    auth_url = (
        f"https://www.facebook.com/{META_API_VERSION}/dialog/oauth"
        f"?client_id={META_APP_ID}"
        f"&redirect_uri={redirect_uri}"
        f"&state={state}"
        f"&response_type=code"
        f"&scope={scope}"
    )
    return RedirectResponse(auth_url)


@app.get("/instagram/callback")
def instagram_callback(code: Optional[str] = None, state: Optional[str] = None, error: Optional[str] = None, error_description: Optional[str] = None):
    if error:
        raise HTTPException(status_code=400, detail={"error": error, "error_description": error_description})

    if not code:
        raise HTTPException(status_code=400, detail="Missing code")

    redirect_uri = f"{BASE_URL}/instagram/callback"
    token_data = exchange_code_for_token(code, redirect_uri)
    access_token = token_data.get("access_token")
    token_type = token_data.get("token_type", "")
    expires_in = token_data.get("expires_in")

    if not access_token:
        raise HTTPException(status_code=502, detail={"meta_error": token_data})

    profile = get_fb_user_profile(access_token)
    fb_user_id = str(profile.get("id", ""))

    upsert_connection(
        provider="instagram",
        fb_user_id=fb_user_id,
        access_token=access_token,
        token_type=token_type,
        scope="",
        expires_in=expires_in
    )

    return HTMLResponse(f"Instagram connected<br/>User: {profile.get('name','')} ({fb_user_id})")


# =============================================================================
# Phase 4: Pages + Selection + Reels Publish
# =============================================================================

@app.get("/facebook/pages")
def facebook_pages(fb_user_id: str = Query(..., description="Facebook user id (string)")):
    """
    Returns pages you manage from /me/accounts. :contentReference[oaicite:4]{index=4}
    """
    user_token = get_connection_token("facebook", fb_user_id)
    data = meta_graph_get(
        "me/accounts",
        {
            "fields": "id,name,access_token",
            "access_token": user_token
        }
    )
    pages = data.get("data", []) or []
    # return minimal info (do NOT return page access tokens)
    return {"fb_user_id": fb_user_id, "pages": [{"id": p.get("id"), "name": p.get("name")} for p in pages]}


@app.post("/facebook/page/select")
def facebook_page_select(payload: Dict[str, Any]):
    """
    Body:
      { "fb_user_id": "...", "page_id": "...", "page_name": "..." }
    Persists default target Page for publishing.
    """
    fb_user_id = str(payload.get("fb_user_id", "")).strip()
    page_id = str(payload.get("page_id", "")).strip()
    page_name = str(payload.get("page_name", "")).strip()

    if not fb_user_id or not page_id:
        raise HTTPException(status_code=400, detail="fb_user_id and page_id are required")

    # Optional: validate the page belongs to the user by checking /me/accounts
    user_token = get_connection_token("facebook", fb_user_id)
    pages = meta_graph_get(
        "me/accounts",
        {"fields": "id,name", "access_token": user_token}
    ).get("data", []) or []

    if not any(str(p.get("id")) == page_id for p in pages):
        raise HTTPException(status_code=400, detail="page_id is not in /me/accounts for this user")

    upsert_selected_page(fb_user_id, page_id, page_name)
    return {"ok": True, "fb_user_id": fb_user_id, "selected_page_id": page_id, "selected_page_name": page_name}


@app.get("/facebook/page/selected")
def facebook_page_selected(fb_user_id: str = Query(...)):
    return {"fb_user_id": fb_user_id, **get_selected_page(fb_user_id)}


@app.post("/facebook/reels/start")
def facebook_reels_start(payload: Dict[str, Any]):
    """
    Starts a Reels publishing session on the selected Page.
    Returns { video_id, upload_url } from /{page_id}/video_reels start phase. :contentReference[oaicite:5]{index=5}

    Body:
      {
        "fb_user_id": "...",
        "file_size": 12345678
      }
    """
    fb_user_id = str(payload.get("fb_user_id", "")).strip()
    file_size = payload.get("file_size")

    if not fb_user_id:
        raise HTTPException(status_code=400, detail="fb_user_id is required")

    sel = get_selected_page(fb_user_id)
    page_id = sel.get("selected_page_id")
    if not page_id:
        raise HTTPException(status_code=400, detail="No selected_page_id. Call /facebook/page/select first.")

    # Use Page access token for publishing
    user_token = get_connection_token("facebook", fb_user_id)
    page_token = get_page_access_token_from_me_accounts(user_token, page_id)

    post_data = {
        "access_token": page_token,
        "upload_phase": "START",
    }

    # Some upload flows require file_size (helps Meta validate upload). :contentReference[oaicite:6]{index=6}
    if file_size is not None:
        try:
            post_data["file_size"] = str(int(file_size))
        except Exception:
            raise HTTPException(status_code=400, detail="file_size must be an integer byte count")

    out = meta_graph_post(f"{page_id}/video_reels", post_data)

    video_id = out.get("video_id") or out.get("id")
    upload_url = out.get("upload_url")

    if not video_id or not upload_url:
        # return full response for debugging
        raise HTTPException(status_code=502, detail={"meta_response": out})

    return {"fb_user_id": fb_user_id, "page_id": page_id, "video_id": video_id, "upload_url": upload_url}


@app.post("/facebook/reels/finish")
def facebook_reels_finish(payload: Dict[str, Any]):
    """
    Finishes upload and publishes the Reel using FINISH phase. :contentReference[oaicite:7]{index=7}

    Body:
      {
        "fb_user_id": "...",
        "video_id": "...",
        "description": "...",
        "title": "..."   (optional),
        "publish": true  (default true)
      }
    """
    fb_user_id = str(payload.get("fb_user_id", "")).strip()
    video_id = str(payload.get("video_id", "")).strip()
    description = str(payload.get("description", "")).strip()
    title = str(payload.get("title", "")).strip()
    publish = payload.get("publish", True)

    if not fb_user_id or not video_id:
        raise HTTPException(status_code=400, detail="fb_user_id and video_id are required")

    sel = get_selected_page(fb_user_id)
    page_id = sel.get("selected_page_id")
    if not page_id:
        raise HTTPException(status_code=400, detail="No selected_page_id. Call /facebook/page/select first.")

    user_token = get_connection_token("facebook", fb_user_id)
    page_token = get_page_access_token_from_me_accounts(user_token, page_id)

    post_data = {
        "access_token": page_token,
        "upload_phase": "FINISH",
        "video_id": video_id,
        # Meta docs allow publishing state control here. :contentReference[oaicite:8]{index=8}
        "video_state": "PUBLISHED" if publish else "DRAFT",
    }

    if description:
        post_data["description"] = description
    if title:
        post_data["title"] = title

    out = meta_graph_post(f"{page_id}/video_reels", post_data)

    return {"ok": True, "fb_user_id": fb_user_id, "page_id": page_id, "meta_response": out}
