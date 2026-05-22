"""Public feature metadata (no auth) — ML / Hugging Face hub links for the web app."""

from typing import Any, Dict

from fastapi import APIRouter

from services.ml_hub_config import build_ml_hub_public_response

router = APIRouter(prefix="/api/features", tags=["features"])


@router.get("/ml-hub")
async def ml_hub_public() -> Dict[str, Any]:
    """
    Safe, cacheable pointers for in-app ML & AI companion.
    No secrets; URLs may be overridden via environment for white-label or forks.
    """
    return build_ml_hub_public_response()
