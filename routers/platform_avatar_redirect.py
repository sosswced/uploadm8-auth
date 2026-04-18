"""
Same-origin /platform-avatars/... URLs (stored in DB as R2 keys) must redirect to a presigned GET
or the browser requests the API host and gets 404. Only the owning user may follow the redirect.
"""

import re

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import RedirectResponse

from core.deps import get_current_user
from core.r2 import generate_presigned_download_url

router = APIRouter(tags=["platform-avatars"])

_SAFE_PLATFORM = re.compile(r"^[a-z0-9_]{1,32}$", re.I)
_SAFE_FILENAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,200}$")


@router.get("/platform-avatars/{user_id}/{platform}/{filename}")
async def redirect_platform_avatar_to_r2(
    user_id: str,
    platform: str,
    filename: str,
    user: dict = Depends(get_current_user),
):
    if str(user["id"]) != str(user_id).strip():
        raise HTTPException(status_code=403, detail="Forbidden")
    plat = (platform or "").strip().lower()
    fn = (filename or "").strip()
    if not _SAFE_PLATFORM.match(plat) or not _SAFE_FILENAME.match(fn):
        raise HTTPException(status_code=400, detail="Invalid path")
    key = f"platform-avatars/{user_id.strip()}/{plat}/{fn}"
    try:
        url = generate_presigned_download_url(key, ttl=3600)
    except Exception:
        raise HTTPException(status_code=404, detail="Avatar not available")
    if not url:
        raise HTTPException(status_code=404, detail="Avatar not available")
    return RedirectResponse(url, status_code=302)
