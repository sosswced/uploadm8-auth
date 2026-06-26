"""E2E environment configuration (reads project .env via run_tests.py / conftest)."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

# Default overnight upload fixtures (override via E2E_TEST_VIDEO / E2E_TEST_TELEMETRY_MAP).
DEFAULT_E2E_VIDEO = Path(r"C:\Users\Earl\Videos\20250224_0073_CAM_EVNT.MP4")
DEFAULT_E2E_TELEMETRY_MAP = Path(r"C:\Users\Earl\Videos\20250224_0073_CAM_EVNT.map")


def e2e_base_url() -> str:
    return (
        os.environ.get("E2E_BASE_URL")
        or os.environ.get("LOCUST_HOST")
        or "http://127.0.0.1:8000"
    ).rstrip("/")


def e2e_master_email() -> str:
    return (
        os.environ.get("E2E_MASTER_ADMIN_EMAIL")
        or os.environ.get("BOOTSTRAP_ADMIN_EMAIL")
        or ""
    ).strip().lower()


def e2e_master_password() -> str:
    return os.environ.get("E2E_MASTER_ADMIN_PASSWORD", "").strip()


def _resolve_e2e_path(raw: str) -> Path | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    p = Path(raw)
    if not p.is_file():
        p = ROOT / raw
    return p if p.is_file() else None


def e2e_test_video() -> Path | None:
    resolved = _resolve_e2e_path(os.environ.get("E2E_TEST_VIDEO", ""))
    if resolved is not None:
        return resolved
    return DEFAULT_E2E_VIDEO if DEFAULT_E2E_VIDEO.is_file() else None


def e2e_test_telemetry_map() -> Path | None:
    resolved = _resolve_e2e_path(os.environ.get("E2E_TEST_TELEMETRY_MAP", ""))
    if resolved is not None:
        return resolved
    return DEFAULT_E2E_TELEMETRY_MAP if DEFAULT_E2E_TELEMETRY_MAP.is_file() else None


def e2e_headed() -> bool:
    return os.environ.get("E2E_HEADED", "").lower() in ("1", "true", "yes")


def e2e_skip_mutations() -> bool:
    return os.environ.get("E2E_SKIP_MUTATIONS", "1").lower() not in ("0", "false", "no")


def e2e_api_timeout_s() -> float:
    try:
        return float(os.environ.get("E2E_API_TIMEOUT_S", "30"))
    except ValueError:
        return 30.0


def e2e_page_timeout_ms() -> int:
    try:
        return int(os.environ.get("E2E_PAGE_TIMEOUT_MS", "45000"))
    except ValueError:
        return 45000


def auth_state_path() -> Path:
    custom = os.environ.get("E2E_AUTH_STATE", "").strip()
    if custom:
        return Path(custom)
    return ROOT / "tests" / "e2e" / ".auth" / "master_admin.json"


DEFAULT_TARGET_USER_ID = "ae995094-abb6-4a41-8d51-460ca8f0fd8c"
DEFAULT_TARGET_USER_NAME = "Johnny Omeadows"


def e2e_target_user_id() -> str:
    return (os.environ.get("E2E_TARGET_USER_ID") or DEFAULT_TARGET_USER_ID).strip()


def e2e_target_user_name() -> str:
    return (os.environ.get("E2E_TARGET_USER_NAME") or DEFAULT_TARGET_USER_NAME).strip()


def e2e_upload_platforms() -> tuple[str, ...]:
    """Platforms to publish to on upload.html (live demo defaults to TikTok only)."""
    raw = (os.environ.get("E2E_UPLOAD_PLATFORMS") or "tiktok").strip()
    parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
    return tuple(parts or ("tiktok",))


def e2e_tiktok_profile() -> str:
    """Optional TikTok profile name/@username substring for account picker."""
    return (os.environ.get("E2E_TIKTOK_PROFILE") or "").strip()


def e2e_target_user_search_terms() -> tuple[str, ...]:
    """Search strings for account-mgmt + wallet UIs (name fragments + id prefix)."""
    name = e2e_target_user_name()
    uid = e2e_target_user_id()
    terms: list[str] = []
    for t in (name, uid, uid.split("-")[0]):
        t = (t or "").strip()
        if t and t not in terms:
            terms.append(t)
    for part in name.split():
        if len(part) >= 3 and part not in terms:
            terms.append(part)
    return tuple(terms)
