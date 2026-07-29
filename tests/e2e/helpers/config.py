"""E2E environment configuration (reads project .env via run_tests.py / conftest)."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


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


def _library_pair() -> tuple[Path, Path] | None:
    """Random matching .mp4+.map from E2E_MEDIA_LIBRARY (or default PNW folder).

    Skips the library when any explicit path env is set. The missing half of a
    one-sided explicit pair is resolved via same-stem sibling in
    ``e2e_test_video`` / ``e2e_test_telemetry_map`` — never a random other clip.
    """
    if (os.environ.get("E2E_TEST_VIDEO") or "").strip() or (
        os.environ.get("E2E_TEST_TELEMETRY_MAP") or ""
    ).strip():
        return None
    try:
        from tests.e2e.helpers.media_library import pick_random_media_pair

        return pick_random_media_pair()
    except Exception:
        return None


def _tup_mode() -> bool:
    return os.environ.get("E2E_TUP", "").lower() in ("1", "true", "yes", "on")


def e2e_test_video() -> Path | None:
    """Explicit ``E2E_TEST_VIDEO``, else sibling of explicit map, else library pair."""
    resolved = _resolve_e2e_path(os.environ.get("E2E_TEST_VIDEO", ""))
    if resolved is not None:
        return resolved
    # Only map set → prefer same-stem video (never a random library clip).
    map_only = _resolve_e2e_path(os.environ.get("E2E_TEST_TELEMETRY_MAP", ""))
    if map_only is not None:
        try:
            from tests.e2e.helpers.media_library import find_sibling_media_path

            sib = find_sibling_media_path(map_only)
            if sib is not None:
                return sib
        except Exception:
            pass
        return None
    pair = _library_pair()
    if pair is not None:
        return pair[0]
    return None


def e2e_test_telemetry_map() -> Path | None:
    """Explicit ``E2E_TEST_TELEMETRY_MAP``, else sibling of explicit video, else library."""
    resolved = _resolve_e2e_path(os.environ.get("E2E_TEST_TELEMETRY_MAP", ""))
    if resolved is not None:
        return resolved
    # Only video set → prefer same-stem .map (never a random library map).
    video_only = _resolve_e2e_path(os.environ.get("E2E_TEST_VIDEO", ""))
    if video_only is not None:
        try:
            from tests.e2e.helpers.media_library import find_sibling_media_path

            sib = find_sibling_media_path(video_only)
            if sib is not None:
                return sib
        except Exception:
            pass
        return None
    pair = _library_pair()
    if pair is not None:
        return pair[1]
    return None


def e2e_youtube_copyright_trim() -> bool:
    """Force-enable YouTube Shorts copyright trim for long-clip TUP tests."""
    raw = os.environ.get("E2E_YOUTUBE_COPYRIGHT_TRIM")
    if raw is None or raw.strip() == "":
        # Default on for /TUP so >60s library clips exercise the cut path.
        return os.environ.get("E2E_TUP", "").lower() in ("1", "true", "yes")
    return raw.lower() not in ("0", "false", "no", "off")


def e2e_headed() -> bool:
    return os.environ.get("E2E_HEADED", "").lower() in ("1", "true", "yes")


def e2e_skip_mutations() -> bool:
    return os.environ.get("E2E_SKIP_MUTATIONS", "1").lower() not in ("0", "false", "no")


def e2e_api_timeout_s() -> float:
    try:
        return float(os.environ.get("E2E_API_TIMEOUT_S", "60"))
    except ValueError:
        return 60.0


def e2e_page_timeout_ms() -> int:
    try:
        return int(os.environ.get("E2E_PAGE_TIMEOUT_MS", "90000"))
    except ValueError:
        return 90000


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


# Canonical publish targets for /TUP (all connected platforms; UI applies each default).
ALL_UPLOAD_PLATFORMS: tuple[str, ...] = ("tiktok", "youtube", "instagram", "facebook")


def e2e_upload_platforms() -> tuple[str, ...]:
    """
    Platforms to publish to on upload.html.

    Default for /TUP is all platforms (`E2E_UPLOAD_PLATFORMS=all` or unset under TUP).
    Legacy live-demo default remains TikTok-only when E2E_UPLOAD_PLATFORMS is unset
    and E2E_TUP is not set.
    """
    raw = (os.environ.get("E2E_UPLOAD_PLATFORMS") or "").strip()
    if not raw:
        if os.environ.get("E2E_TUP", "").lower() in ("1", "true", "yes"):
            return ALL_UPLOAD_PLATFORMS
        return ("tiktok",)
    lowered = raw.lower()
    if lowered in ("all", "*", "every", "connected"):
        return ALL_UPLOAD_PLATFORMS
    parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
    return tuple(parts or ALL_UPLOAD_PLATFORMS)


def e2e_tiktok_profile() -> str:
    """Optional TikTok profile name/@username substring for account picker."""
    return (os.environ.get("E2E_TIKTOK_PROFILE") or "").strip()


def e2e_use_persona() -> bool:
    """Apply linked persona on upload (default on for /TUP)."""
    raw = os.environ.get("E2E_USE_PERSONA")
    if raw is None or raw.strip() == "":
        return os.environ.get("E2E_TUP", "").lower() in ("1", "true", "yes")
    return raw.lower() not in ("0", "false", "no")


def e2e_persona_id() -> str:
    """Optional persona UUID override; empty → use settings default / first linked."""
    return (os.environ.get("E2E_PERSONA_ID") or "").strip()


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
