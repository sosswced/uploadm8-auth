"""
Pikzels public API v2 — canonical endpoint map for UploadM8 integration.

Official docs index: https://docs.pikzels.com/llms.txt
OpenAPI: https://docs.pikzels.com/openapi.json
Base URL: https://api.pikzels.com
Auth: header X-Api-Key: pkz_...

Important:
- Public OpenAPI (checked programmatically) specifies persona/style pikzonalities with
  exactly 3 reference images (minItems=maxItems=3). The Pikzels *web app* may allow more
  (e.g. up to 20) via a different stack; when forwarding to POST /v2/pikzonality/persona,
  you may need to select or downsample to 3 images until the public API documents a higher max.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Final, List, Optional, Tuple

PUBLIC_BASE: Final[str] = "https://api.pikzels.com"


def resolve_public_api_key() -> str:
    """
    API key for https://api.pikzels.com (header X-Api-Key).

    Canonical: PIKZELS_API_KEY (public OpenAPI / llms.txt).
    Fallback: THUMB_RENDER_API_KEY — same secret in many deployments; avoids breaking
    envs that only set the legacy name.
    """
    return (os.environ.get("PIKZELS_API_KEY") or os.environ.get("THUMB_RENDER_API_KEY") or "").strip()


def public_api_key_source() -> Optional[str]:
    """Which env var supplied the v2 key (for ops/debug; never log the secret)."""
    if (os.environ.get("PIKZELS_API_KEY") or "").strip():
        return "PIKZELS_API_KEY"
    if (os.environ.get("THUMB_RENDER_API_KEY") or "").strip():
        return "THUMB_RENDER_API_KEY"
    return None

# Endpoints (paths only; join with PUBLIC_BASE)
V2_THUMBNAIL_TEXT: Final[str] = "/v2/thumbnail/text"
V2_THUMBNAIL_IMAGE: Final[str] = "/v2/thumbnail/image"
V2_THUMBNAIL_EDIT: Final[str] = "/v2/thumbnail/edit"
V2_THUMBNAIL_FACESWAP: Final[str] = "/v2/thumbnail/faceswap"
V2_THUMBNAIL_SCORE: Final[str] = "/v2/thumbnail/score"
V2_TITLE_TEXT: Final[str] = "/v2/title/text"
V2_PIKZONALITY_PERSONA: Final[str] = "/v2/pikzonality/persona"
V2_PIKZONALITY_STYLE: Final[str] = "/v2/pikzonality/style"
V2_PIKZONALITY_BY_ID: Final[str] = "/v2/pikzonality/{id}"

# Feature names (product marketing) → (Pikzels capability, HTTP path key, notes)
# Path key matches our constants above for discoverability.
PIKZELS_FEATURE_MAP: Tuple[Tuple[str, str, str, str], ...] = (
    (
        "Pikzels Score™",
        "score",
        V2_THUMBNAIL_SCORE,
        "POST body: image_url XOR image_base64; optional title. Returns main_score + subscores.",
    ),
    (
        "Prompt",
        "prompt",
        V2_THUMBNAIL_TEXT,
        "POST text-to-thumbnail: prompt, model (e.g. pkz_4), format (e.g. 16:9).",
    ),
    (
        "Recreate",
        "recreate",
        V2_THUMBNAIL_IMAGE,
        "POST create-from-image / reference image workflow (see create-from-image.md).",
    ),
    (
        "One-Click Fix™",
        "one_click_fix",
        V2_THUMBNAIL_EDIT,
        "POST /v2/thumbnail/edit — targeted fixes (pair with /v2/thumbnail/score for suggestions).",
    ),
    (
        "Edit",
        "edit",
        V2_THUMBNAIL_EDIT,
        "Same v2 edit endpoint; distinguish UX in our app via prompt/instruction text.",
    ),
    (
        "Titles",
        "titles",
        V2_TITLE_TEXT,
        "POST /v2/title/text — title generation from text (see docs).",
    ),
    (
        "Personas",
        "personas",
        V2_PIKZONALITY_PERSONA,
        "POST /v2/pikzonality/persona (pikzonality) + optionally /v2/thumbnail/faceswap on outputs; public OpenAPI uses 3 ref images.",
    ),
    (
        "Styles",
        "styles",
        V2_PIKZONALITY_STYLE,
        "POST create style pikzonality; persisted IDs for consistent visual identity.",
    ),
)


def feature_map_for_docs() -> List[Dict[str, Any]]:
    """Structured list for admin/debug JSON or internal tooling."""
    return [
        {
            "feature": name,
            "slug": slug,
            "path": path,
            "notes": notes,
        }
        for name, slug, path, notes in PIKZELS_FEATURE_MAP
    ]


def faceswap_path() -> str:
    """Apply FaceSwap / face alignment on an existing thumbnail (public API path)."""
    return V2_THUMBNAIL_FACESWAP
