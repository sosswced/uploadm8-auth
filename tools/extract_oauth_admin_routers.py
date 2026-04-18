"""
One-off: build routers/oauth.py and routers/admin.py from app.py.
Run from repo root: python tools/extract_oauth_admin_routers.py
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"
text = APP.read_text(encoding="utf-8")
lines = text.splitlines(keepends=True)


def grab(a: int, b: int) -> str:
    return "".join(lines[a - 1 : b])


# --- OAuth: routes only (config + helpers stay in app.py) ---
# Line numbers 1-based from current app.py
OAUTH_START, OAUTH_END = 9481, 10046
oauth_body = grab(OAUTH_START, OAUTH_END)
oauth_body = oauth_body.replace("@app.", "@router.")
oauth_body = oauth_body.replace("Depends(get_current_user)", "Depends(_oauth_user)")

# Namespace app-owned symbols (word boundaries)
_oauth_subs = [
    "OAUTH_CONFIG",
    "db_pool",
    "_oauth_state_set",
    "_oauth_state_pop",
    "get_oauth_redirect_uri",
    "_tiktok_pkce_verifier_and_challenge",
    "_sanitize_oauth_parent_origin",
    "TIKTOK_CLIENT_KEY",
    "TIKTOK_CLIENT_SECRET",
    "YOUTUBE_CLIENT_ID",
    "YOUTUBE_CLIENT_SECRET",
    "INSTAGRAM_CLIENT_ID",
    "INSTAGRAM_CLIENT_SECRET",
    "FACEBOOK_CLIENT_ID",
    "FACEBOOK_CLIENT_SECRET",
    "FRONTEND_URL",
    "encrypt_blob",
    "_mirror_oauth_profile_image_to_r2",
    "log_system_event",
]
for name in _oauth_subs:
    oauth_body = re.sub(rf"\b{name}\b", f"m.{name}", oauth_body)

oauth_body = oauth_body.replace("await m.log_system_event", "await m.log_system_event")  # no-op
oauth_body = re.sub(r"\bm\.m\.", "m.", oauth_body)

OAUTH_HEADER = '''"""OAuth routes (/api/oauth/*). Config, Redis state, and helpers remain in app.py."""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import secrets
from typing import Optional
from urllib.parse import quote, urlencode, urlparse

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import HTMLResponse

from core.time_utils import now_utc as _now_utc
from services.meta_oauth import (
    fetch_granted_permissions,
    meta_facebook_oauth_scope,
    meta_instagram_oauth_scope,
    meta_oauth_mode,
)
from stages.entitlements import can_user_connect_platform

logger = logging.getLogger("uploadm8-api")

router = APIRouter(tags=["oauth"])


async def _oauth_user(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    import app as m

    return await m.get_current_user(request, authorization)


'''

# Insert import app as m
oauth_body = oauth_body.replace(
    "async def oauth_start(\n",
    "async def oauth_start(\n",
)
oauth_body = oauth_body.replace(
    "async def oauth_start(\n    platform: str,\n    parent_origin: Optional[str] = Query(None),\n    force_login: bool = Query(False, description=\"Force account chooser/reauth where provider supports it\"),\n    reconnect_account_id: Optional[str] = Query(None, description=\"Existing platform_tokens.id to reconnect\"),\n    user: dict = Depends(_oauth_user),\n):\n    \"\"\"Start OAuth flow for a platform\"\"\"",
    "async def oauth_start(\n    platform: str,\n    parent_origin: Optional[str] = Query(None),\n    force_login: bool = Query(False, description=\"Force account chooser/reauth where provider supports it\"),\n    reconnect_account_id: Optional[str] = Query(None, description=\"Existing platform_tokens.id to reconnect\"),\n    user: dict = Depends(_oauth_user),\n):\n    \"\"\"Start OAuth flow for a platform\"\"\"\n    import app as m",
)
oauth_body = oauth_body.replace(
    "async def oauth_callback(platform: str, code: str = Query(None), state: str = Query(None), error: str = Query(None)):\n    \"\"\"Handle OAuth callback",
    "async def oauth_callback(platform: str, code: str = Query(None), state: str = Query(None), error: str = Query(None)):\n    \"\"\"Handle OAuth callback",
)
# meta_oauth_config_public needs no m.
oauth_body = oauth_body.replace(
    "async def oauth_callback(platform: str, code: str = Query(None), state: str = Query(None), error: str = Query(None)):\n    \"\"\"Handle OAuth callback - returns HTML that communicates with parent window\"\"\"\n    post_target = m.FRONTEND_URL.rstrip(\"/\")",
    "async def oauth_callback(platform: str, code: str = Query(None), state: str = Query(None), error: str = Query(None)):\n    \"\"\"Handle OAuth callback - returns HTML that communicates with parent window\"\"\"\n    import app as m\n    post_target = m.FRONTEND_URL.rstrip(\"/\")",
)

# oauth_start: add import after docstring if previous replace missed (multiline mismatch)
if "\n    import app as m\n" not in oauth_body.split("async def oauth_start", 1)[1][:800]:
    oauth_body = oauth_body.replace(
        "):\n    \"\"\"Start OAuth flow for a platform\"\"\"\n    if platform not in m.OAUTH_CONFIG",
        "):\n    \"\"\"Start OAuth flow for a platform\"\"\"\n    import app as m\n    if platform not in m.OAUTH_CONFIG",
    )

(ROOT / "routers" / "oauth.py").write_text(OAUTH_HEADER + oauth_body, encoding="utf-8")
print("wrote routers/oauth.py")

# --- Admin: split @app routes ---
pat = re.compile(
    r"^@app\.(get|post|put|delete|patch)\(\"([^\"]+)\"",
    re.MULTILINE,
)
matches = list(pat.finditer(text))
blocks: list[tuple[str, str]] = []
for i, m in enumerate(matches):
    path = m.group(2)
    start = m.start()
    end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
    block = text[start:end]
    if "/api/admin/" in path or path.startswith("/api/admin"):
        blocks.append((path, block))

admin_raw = "\n".join(b for _, b in blocks)

admin_raw = admin_raw.replace("@app.", "@router.")
admin_raw = admin_raw.replace("Depends(require_admin)", "Depends(_admin_user)")
admin_raw = admin_raw.replace("Depends(require_master_admin)", "Depends(_master_admin_user)")
admin_raw = re.sub(r"^(\s*)require_admin\(user\)", r"\1await m.require_admin(user)", admin_raw, flags=re.MULTILINE)

# Large substitution list: app module globals used across admin handlers
_ADMIN_M_SUBS = [
    "db_pool",
    "redis_client",
    "logger",
    "MIGRATIONS_LATEST_VERSION",
    "MIGRATIONS_CRITICAL_VERSIONS",
    "run_migrations",
    "FRONTEND_URL",
    "BASE_URL",
    "admin_settings_cache",
    "ENC_KEYS",
    "CURRENT_KEY_ID",
    "R2_BUCKET_NAME",
    "R2_ACCOUNT_ID",
    "R2_ACCESS_KEY_ID",
    "R2_SECRET_ACCESS_KEY",
    "R2_ENDPOINT_URL",
    "DATABASE_URL",
    "STRIPE_SECRET_KEY",
    "OPENAI_API_KEY",
    "ADMIN_DISCORD_WEBHOOK_URL",
    "SIGNUP_DISCORD_WEBHOOK_URL",
    "MRR_DISCORD_WEBHOOK_URL",
    "COMMUNITY_DISCORD_WEBHOOK_URL",
    "log_admin_audit",
    "log_system_event",
    "invalidate_me_api_cache",
    "encrypt_blob",
    "decrypt_blob",
    "hash_password",
    "verify_password",
    "generate_presigned_download_url",
    "get_s3_client",
    "client_ip",
    "api_problem",
    "_execute_account_deletion",
    "_pikzels_v2_admin_call",
    "_pikzels_v2_get_response",
    "_marketing_ai_fortune500_plan",
    "generate_trill_content",
    "install_rate_limit_middleware",
    "assert_user_update_set_clauses",
    "assert_set_fragments_columns",
    "USERS_UPDATE_COLUMNS_ADMIN",
    "USERS_UPDATE_COLUMNS_ME",
    "assert_wallet_balance_column",
    "run_migrations",
    "daily_refill",
    "credit_wallet",
    "ledger_entry",
    "get_wallet",
    "reserve_tokens",
    "spend_tokens",
    "notify_revenue_event",
    "get_notif_settings",
    "get_admin_webhook",
    "mask_email",
    "discord_notify",
    "_discord_notify_service",
    "send_announcement_email",
    "send_admin_wallet_topup_email",
    "send_admin_tier_switch_email",
    "send_admin_account_status_email",
    "send_admin_reset_password_email",
    "send_friends_family_welcome_email",
    "send_agency_welcome_email",
    "send_master_admin_welcome_email",
    "send_admin_email_change_notice_to_old_email",
    "send_user_email_change_notice_to_old_email",
    "_notify_mrr_service",
    "_notify_topup_service",
    "_parse_user_preferences_json",
    "_parse_users_preferences",
    "_strip_legacy_thumbnail_engine_keys",
    "touch_last_active",
    "_last_active_touch_interval_sec",
    "_should_touch_last_active",
    "SUCCESSFUL_STATUS_SQL_IN",
    "SUCCESSFUL_UPLOAD_STATUSES",
    "metric_defs",
    "get_plan",
    "_tier_is_upgrade",
    "normalize_tier",
    "get_entitlements_for_tier",
    "get_entitlements_from_user",
    "TIER_CONFIG",
    "ENTITLEMENT_KEYS",
    "check_queue_depth",
    "PRIORITY_QUEUE_CLASSES",
    "TIER_SLUGS",
    "get_tiers_for_api",
    "get_tier_display_name",
    "get_next_public_upgrade_tier",
    "entitlements_to_dict",
    "expand_hashtag_items",
    "estimate_pikzels_v2_call_cost",
    "format_library_rows",
    "generate_recreate_variants",
    "extract_youtube_video_id",
    "fetch_youtube_title",
    "estimate_studio_cost",
    "calculate_smart_schedule",
    "get_existing_scheduled_days",
    "ANALYTICS_OVERVIEW_PLATFORMS",
    "facebook_page_feed_reel_engagement_rollups",
    "instagram_account_degraded_live",
    "list_analytics_platform_query_values",
    "resolve_analytics_platform_filter",
    "upload_metrics",
    "kpi_collector",
    "build_wallet_marketing_payload",
    "redact_url",
    "_req_id",
    "_sha256_hex",
    "create_access_jwt",
    "create_refresh_token",
    "verify_access_jwt",
    "_bearer_from_request",
    "_attach_auth_cookies",
    "_clear_auth_cookies",
    "_cookie_secure",
    "_cookie_domain",
    "EMAIL_CRON_INTERVAL_SECONDS",
    "_run_trial_ending_reminders_once",
    "_run_monthly_user_kpi_digests_once",
    "parse_enc_keys",
    "init_enc_keys",
    "apply_asyncpg_json_codecs",
    "_init_asyncpg_codecs",
    "_load_uploads_columns",
    "_pick_cols",
    "_maybe_reconcile_stale_processing_on_read",
    "_safe_json",
    "haversine_distance_km",
    "_cors_reflect_origin",
    "_csp_connect_src_directive",
    "_rate_limit_request_bypass",
    "_json_429",
    "_parse_int_env",
    "_parse_trusted_rl_ips",
    "_client_is_loopback",
    "client_ip",
    "generate_presigned_upload_url",
    "r2_presign_get_url",
    "_platform_account_avatar_to_url",
    "_mirror_oauth_profile_image_to_r2",
    "generate_trill_content",
    "notify_signup",
    "notify_mrr",
    "notify_topup",
    "notify_weekly_costs",
    "send_signup_confirmation_email",
    "get_current_user",
    "get_current_user_readonly",
    "require_admin",
    "require_master_admin",
    "_resolve_current_user_bearer",
    "disconnect_account",
    "OAUTH_CONFIG",
    "_oauth_state_set",
    "_oauth_state_pop",
    "TIKTOK_WEBHOOK_SECRET",
    "_verify_tiktok_signature",
    "TIKTOK_WEBHOOK_REPLAY_WINDOW_SEC",
    "META_APP_SECRET",
    "META_APP_ID",
    "FACEBOOK_APP_SECRET",
    "render_yaml",
    "Path",
]

# Do not prefix logger in admin file — use module logger
_ADMIN_M_SUBS = [x for x in _ADMIN_M_SUBS if x != "logger"]

# Blacklist: do not prefix these even if in list (substrings / mistakes)
_skip_m = {"Path"}

for name in _ADMIN_M_SUBS:
    if name in _skip_m:
        continue
    admin_raw = re.sub(rf"\b{name}\b", f"m.{name}", admin_raw)

# Fix double m.
while "m.m." in admin_raw:
    admin_raw = admin_raw.replace("m.m.", "m.")

ADMIN_HEADER = '''"""Admin API routes (/api/admin/*). Uses lazy `import app as m` per handler."""
from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import math
import os
import re
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from pydantic import BaseModel, Field

from api.schemas.admin import (
    AdminEmailJobRunRequest,
    AdminResetPasswordIn,
    AdminUpdateEmailIn,
    AdminUserUpdate,
    AdminWalletAdjust,
)
from api.schemas.thumbnail_studio import (
    AnnouncementRequest,
    MarketingAIGenerateIn,
    PikzelsV2EditBody,
    PikzelsV2FaceswapBody,
    PikzelsV2PikzonalityBody,
    PikzelsV2PromptBody,
    PikzelsV2RecreateBody,
    PikzelsV2ScoreBody,
    PikzelsV2TitlesBody,
)
from core.sql_allowlist import (
    USERS_UPDATE_COLUMNS_ADMIN,
    assert_set_fragments_columns,
    assert_user_update_set_clauses,
    assert_wallet_balance_column,
)
from services.api_errors import api_problem
from services.billing import _tier_is_upgrade, get_plan
from services import metric_definitions as metric_defs
from services.upload_metrics import (
    ANALYTICS_OVERVIEW_PLATFORMS,
    SUCCESSFUL_STATUS_SQL_IN,
    SUCCESSFUL_UPLOAD_STATUSES,
)
from services.wallet import credit_wallet, get_wallet, ledger_entry, reserve_tokens, spend_tokens
from stages.emails import (
    send_admin_account_status_email,
    send_admin_email_change_notice_to_old_email,
    send_admin_reset_password_email,
    send_admin_tier_switch_email,
    send_admin_wallet_topup_email,
    send_agency_welcome_email,
    send_announcement_email,
    send_friends_family_welcome_email,
    send_master_admin_welcome_email,
    send_user_email_change_notice_to_old_email,
)
from stages.emails.base import MAIL_FROM_SUPPORT, SUPPORT_EMAIL, send_email
from stages.entitlements import (
    ENTITLEMENT_KEYS,
    PRIORITY_QUEUE_CLASSES,
    TIER_CONFIG,
    TIER_SLUGS,
    check_queue_depth,
    entitlements_to_dict,
    get_entitlements_for_tier,
    get_entitlements_from_user,
    get_next_public_upgrade_tier,
    get_tier_display_name,
    get_tiers_for_api,
    normalize_tier,
)

logger = logging.getLogger("uploadm8-api")

router = APIRouter(tags=["admin"])


async def _admin_user(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    import app as m

    user = await m.get_current_user(request, authorization)
    return await m.require_admin(user)


async def _master_admin_user(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    import app as m

    user = await m.get_current_user(request, authorization)
    return await m.require_master_admin(user)


'''

# Prefix remaining app-only names that weren't in list — compile and iterate is separate

(ROOT / "routers" / "admin.py").write_text(ADMIN_HEADER + admin_raw, encoding="utf-8")
print("wrote routers/admin.py (raw), lines:", len(admin_raw.splitlines()))
