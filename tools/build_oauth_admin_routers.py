"""
Build routers/oauth.py and routers/admin.py from app.py (run from repo root).
Deletes nothing from app.py — use strip script after verifying import app works.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP_PATH = ROOT / "app.py"
SRC = APP_PATH.read_text(encoding="utf-8")
LINES = SRC.splitlines(keepends=True)


def grab(a: int, b: int) -> str:
    return "".join(LINES[a - 1 : b])


BLACKLIST_CALL = frozenset(
    {
        "print",
        "dict",
        "list",
        "set",
        "str",
        "int",
        "float",
        "bool",
        "len",
        "range",
        "enumerate",
        "zip",
        "map",
        "filter",
        "any",
        "all",
        "min",
        "max",
        "sum",
        "open",
        "isinstance",
        "hasattr",
        "getattr",
        "setattr",
        "type",
        "super",
        "property",
        "staticmethod",
        "classmethod",
        "bytes",
        "repr",
        "vars",
        "dir",
        "abs",
        "round",
        "sorted",
        "reversed",
        "compile",
        "format",
        "iter",
        "next",
        "slice",
        "object",
        "id",
    }
)


def collect_symbols(py: str) -> tuple[set[str], set[str], set[str]]:
    tree = ast.parse(py)
    funcs: set[str] = set()
    consts: set[str] = set()
    explicit: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
            funcs.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            explicit.add(node.target.id)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if not isinstance(t, ast.Name):
                    continue
                nid = t.id
                if nid == "app":
                    continue
                if nid.isupper():
                    consts.add(nid)
                else:
                    explicit.add(nid)
    funcs.discard("app")
    return funcs, consts, explicit


funcs, consts, explicit = collect_symbols(SRC)
# Vars we must prefix even if not picked up (imports use different names)
explicit |= {
    "db_pool",
    "redis_client",
    "admin_settings_cache",
    "ENC_KEYS",
    "CURRENT_KEY_ID",
    "logger",
    "app_shutting_down",
}
explicit.discard("app")

# --- OAuth (line numbers must match current app.py) ---
OAUTH_LO, OAUTH_HI = 9367, 9932
oauth_body = grab(OAUTH_LO, OAUTH_HI)
oauth_body = oauth_body.replace("@app.", "@router.")
oauth_body = oauth_body.replace("Depends(get_current_user)", "Depends(_oauth_user)")
for name in [
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
]:
    oauth_body = re.sub(rf"\b{name}\b", f"m.{name}", oauth_body)
while "m.m." in oauth_body:
    oauth_body = oauth_body.replace("m.m.", "m.")

OAUTH_HEADER = '''"""OAuth routes (/api/oauth/*)."""
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

oauth_body = oauth_body.replace(
    "):\n    \"\"\"Start OAuth flow for a platform\"\"\"\n    if platform not in m.OAUTH_CONFIG",
    "):\n    \"\"\"Start OAuth flow for a platform\"\"\"\n    import app as m\n    if platform not in m.OAUTH_CONFIG",
)
oauth_body = oauth_body.replace(
    "):\n    \"\"\"Handle OAuth callback - returns HTML that communicates with parent window\"\"\"\n    post_target = m.FRONTEND_URL.rstrip(\"/\")",
    "):\n    \"\"\"Handle OAuth callback - returns HTML that communicates with parent window\"\"\"\n    import app as m\n    post_target = m.FRONTEND_URL.rstrip(\"/\")",
)

(ROOT / "routers" / "oauth.py").write_text(OAUTH_HEADER + oauth_body, encoding="utf-8")
print("wrote routers/oauth.py")

# --- Admin: regex slice /api/admin routes ---
pat = re.compile(r'^@app\.(get|post|put|delete|patch)\(\"([^\"]+)\"', re.MULTILINE)
matches = list(pat.finditer(SRC))
admin_chunks: list[str] = []
for i, m in enumerate(matches):
    path = m.group(2)
    start = m.start()
    end = matches[i + 1].start() if i + 1 < len(matches) else len(SRC)
    block = SRC[start:end]
    if "/api/admin/" in path or path.startswith("/api/admin"):
        admin_chunks.append(block)

admin_raw = "\n".join(admin_chunks)
admin_raw = admin_raw.replace("@app.", "@router.")
admin_raw = admin_raw.replace("Depends(require_admin)", "Depends(_admin_user)")
admin_raw = admin_raw.replace("Depends(require_master_admin)", "Depends(_master_admin_user)")
admin_raw = re.sub(r"^(\s*)require_admin\(user\)", r"\1await m.require_admin(user)", admin_raw, flags=re.MULTILINE)

out_lines: list[str] = []
for line in admin_raw.split("\n"):
    stripped = line.lstrip()
    if stripped.startswith("def ") or stripped.startswith("async def "):
        out_lines.append(line)
        continue
    line2 = line
    for name in sorted(funcs, key=len, reverse=True):
        if name in BLACKLIST_CALL or name.startswith("__"):
            continue
        line2 = re.sub(rf"(?<!\.)\b{re.escape(name)}\s*\(", f"m.{name}(", line2)
    sym_words = sorted(consts | explicit, key=len, reverse=True)
    for name in sym_words:
        if not name or name == "app":
            continue
        if name in ("True", "False", "None"):
            continue
        line2 = re.sub(rf"(?<!\.)\b{re.escape(name)}\b", f"m.{name}", line2)
    while "m.m." in line2:
        line2 = line2.replace("m.m.", "m.")
    out_lines.append(line2)

admin_body = "\n".join(out_lines)
# `global admin_settings_cache` must not become `global m.admin_settings_cache`
admin_body = re.sub(r"\bglobal m\.([a-zA-Z_][a-zA-Z0-9_]*)\b", r"global \1", admin_body)
admin_body = re.sub(r"\bnonlocal m\.([a-zA-Z_][a-zA-Z0-9_]*)\b", r"nonlocal \1", admin_body)

# AST insert import app as m into every top-level async def
tree = ast.parse(admin_body)
for node in tree.body:
    if isinstance(node, ast.AsyncFunctionDef):
        node.body.insert(0, ast.Import(names=[ast.alias(name="app", asname="m")]))
admin_body = ast.unparse(tree)

ADMIN_HEADER = '''"""Admin API (/api/admin/*)."""
from __future__ import annotations

import asyncio
import base64
import calendar
import csv
import hashlib
import io
import json
import logging
import math
import os
import random
import re
import secrets
import string
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from urllib.parse import quote, unquote, urlencode, urlparse

import asyncpg
import httpx
import stripe
from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel, EmailStr, Field

from api.schemas.admin_requests import (
    AdminEmailJobRunRequest,
    AdminResetPasswordIn,
    AdminUpdateEmailIn,
    AdminUserUpdate,
    AdminWalletAdjust,
    AnnouncementRequest,
    MarketingAIGenerateIn,
    MarketingCampaignIn,
    MarketingCampaignStatusIn,
    PromoTogglesBody,
)
from api.schemas.pikzels_v2 import (
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
from services.meta_graph_metrics import (
    facebook_page_feed_reel_engagement_rollups,
    instagram_account_degraded_live,
)
from services.notifications import discord_notify as _discord_notify_service
from services.platform_channels import (
    list_analytics_platform_query_values,
    resolve_analytics_platform_filter,
)
from services.upload_metrics import (
    ANALYTICS_OVERVIEW_PLATFORMS,
    SUCCESSFUL_STATUS_SQL_IN,
    SUCCESSFUL_UPLOAD_STATUSES,
)
from services.wallet import credit_wallet, get_wallet, ledger_entry, reserve_tokens, spend_tokens
from stages.context import expand_hashtag_items
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

(ROOT / "routers" / "admin.py").write_text(ADMIN_HEADER + admin_body + "\n", encoding="utf-8")
print("wrote routers/admin.py")
