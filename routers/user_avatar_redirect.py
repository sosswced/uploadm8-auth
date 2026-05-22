"""
Profile avatars in R2 (``avatars/{user_id}/{file}``) are served via same-origin redirect to a presigned GET.
Only the owning user may follow the redirect (session cookie or bearer).
"""

import re

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import RedirectResponse

from core.deps import get_current_user
from core.r2 import generate_presigned_download_url

router = APIRouter(tags=["user-avatars"])

_SAFE_FILENAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,200}$")


@router.get("/user-avatars/{user_id}/{filename}")
async def redirect_user_avatar_to_r2(
    user_id: str,
    filename: str,
    user: dict = Depends(get_current_user),
):
    if str(user["id"]) != str(user_id).strip():
        raise HTTPException(status_code=403, detail="Forbidden")
    fn = (filename or "").strip()
    if not _SAFE_FILENAME.match(fn):
        raise HTTPException(status_code=400, detail="Invalid path")
    key = f"avatars/{user_id.strip()}/{fn}"
    try:
        url = generate_presigned_download_url(key, ttl=3600)
    except Exception:
        raise HTTPException(status_code=404, detail="Avatar not available")
    if not url:
        raise HTTPException(status_code=404, detail="Avatar not available")
    return RedirectResponse(url, status_code=302)
