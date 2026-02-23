"""
UploadM8 Publish Stage
======================
Publishes transcoded videos to each connected platform.

Critical fix: Hashtags MUST be joined as a single string before being
passed to platform APIs. Iterating a list character-by-character was
the root cause of #g #l #o #r #y #b appearing on posts.
"""

import os
import json
import logging
import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any

import httpx

from .context import JobContext, PlatformResult
from .errors import StageError, SkipStage, ErrorCode

logger = logging.getLogger("uploadm8-worker")

DATABASE_URL = os.environ.get("DATABASE_URL", "")
TIKTOK_CLIENT_KEY = os.environ.get("TIKTOK_CLIENT_KEY", "")
TIKTOK_CLIENT_SECRET = os.environ.get("TIKTOK_CLIENT_SECRET", "")


def normalize_hashtags(hashtags: Any) -> List[str]:
    """
    Normalize raw hashtag data to a clean List[str] of complete hashtag words.
    Single character entries are DROPPED — these indicate a prior iteration bug.
    """
    if not hashtags:
        return []

    if isinstance(hashtags, str):
        stripped = hashtags.strip()
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    hashtags = parsed
                else:
                    hashtags = [stripped]
            except json.JSONDecodeError:
                hashtags = [stripped]
        else:
            hashtags = [t.strip() for t in stripped.replace(",", " ").split() if t.strip()]

    if not isinstance(hashtags, list):
        try:
            hashtags = list(hashtags)
        except Exception:
            return []

    cleaned = []
    for tag in hashtags:
        tag = str(tag).strip().lstrip('#')
        if len(tag) < 2:
            continue
        cleaned.append(tag)

    return cleaned


def build_full_description(
    caption: Optional[str],
    hashtags: Any,
    max_len: int = 2200,
) -> str:
    """
    Combine caption and hashtags into a single platform-ready description string.

    Output: "Caption text here...\n\n#dashcam #roadtrip #LasVegas"
    """
    tag_list = normalize_hashtags(hashtags)

    hashtag_str = " ".join(
        f"#{tag}" if not tag.startswith("#") else tag
        for tag in tag_list
    )

    caption_text = (caption or "").strip()

    if caption_text and hashtag_str:
        full = f"{caption_text}\n\n{hashtag_str}"
    elif caption_text:
        full = caption_text
    elif hashtag_str:
        full = hashtag_str
    else:
        full = ""

    return full[:max_len]


async def publish_to_tiktok(
    video_path: Path,
    title: str,
    caption: Optional[str],
    hashtags: Any,
    privacy: str,
    token_data: Dict[str, Any],
    account_id: Optional[str] = None,
) -> PlatformResult:
    """Publish video to TikTok Content Posting API v2."""
    access_token = token_data.get("access_token")
    if not access_token:
        return PlatformResult(platform="tiktok", success=False, error_message="No access token")

    description = build_full_description(caption, hashtags, max_len=2200)

    privacy_map = {
        "public": "PUBLIC_TO_EVERYONE",
        "private": "SELF_ONLY",
        "friends": "FRIEND_ONLY",
        "unlisted": "MUTUAL_FOLLOW_FRIENDS",
    }
    tiktok_privacy = privacy_map.get(privacy, "PUBLIC_TO_EVERYONE")

    try:
        async with httpx.AsyncClient(timeout=300) as client:
            init_resp = await client.post(
                "https://open.tiktokapis.com/v2/post/publish/video/init/",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json; charset=UTF-8",
                },
                json={
                    "post_info": {
                        "title": title[:150] if title else "",
                        "description": description,
                        "privacy_level": tiktok_privacy,
                        "disable_duet": False,
                        "disable_comment": False,
                        "disable_stitch": False,
                        "video_cover_timestamp_ms": 1000,
                    },
                    "source_info": {
                        "source": "FILE_UPLOAD",
                        "video_size": video_path.stat().st_size,
                        "chunk_size": video_path.stat().st_size,
                        "total_chunk_count": 1,
                    },
                },
            )

            if init_resp.status_code not in (200, 201):
                err_text = init_resp.text[:400]
                logger.error(f"TikTok init failed: {init_resp.status_code} — {err_text}")
                return PlatformResult(
                    platform="tiktok", success=False,
                    error_code=str(init_resp.status_code), error_message=err_text,
                )

            init_data = init_resp.json()
            publish_id = init_data.get("data", {}).get("publish_id")
            upload_url = init_data.get("data", {}).get("upload_url")

            if not upload_url:
                return PlatformResult(
                    platform="tiktok", success=False,
                    error_message="No upload URL from TikTok init",
                )

            file_size = video_path.stat().st_size
            with open(video_path, "rb") as f:
                video_bytes = f.read()

            upload_resp = await client.put(
                upload_url,
                content=video_bytes,
                headers={
                    "Content-Type": "video/mp4",
                    "Content-Range": f"bytes 0-{file_size - 1}/{file_size}",
                    "Content-Length": str(file_size),
                },
            )

            if upload_resp.status_code not in (200, 201, 204):
                return PlatformResult(
                    platform="tiktok", success=False,
                    error_code=str(upload_resp.status_code),
                    error_message=upload_resp.text[:300],
                )

            video_url = None
            for attempt in range(12):
                await asyncio.sleep(10)
                status_resp = await client.post(
                    "https://open.tiktokapis.com/v2/post/publish/status/fetch/",
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "Content-Type": "application/json; charset=UTF-8",
                    },
                    json={"publish_id": publish_id},
                )
                if status_resp.status_code == 200:
                    status_data = status_resp.json().get("data", {})
                    status = status_data.get("status", "")
                    if status in ("PUBLISH_COMPLETE", "SUCCESS"):
                        vid_id = status_data.get("publicaly_available_post_id")
                        if vid_id:
                            video_url = f"https://www.tiktok.com/@me/video/{vid_id}"
                        break
                    elif status in ("FAILED", "ERROR"):
                        return PlatformResult(
                            platform="tiktok", success=False,
                            error_message=f"TikTok processing failed: {status_data}",
                        )

            return PlatformResult(
                platform="tiktok", success=True,
                platform_video_id=publish_id, platform_url=video_url,
            )

    except Exception as e:
        logger.exception(f"TikTok publish exception: {e}")
        return PlatformResult(platform="tiktok", success=False, error_message=str(e))


async def publish_to_youtube(
    video_path: Path,
    title: str,
    caption: Optional[str],
    hashtags: Any,
    privacy: str,
    token_data: Dict[str, Any],
    account_id: Optional[str] = None,
) -> PlatformResult:
    """Publish video to YouTube Data API v3."""
    access_token = token_data.get("access_token")
    if not access_token:
        return PlatformResult(platform="youtube", success=False, error_message="No access token")

    description = build_full_description(caption, hashtags, max_len=5000)

    privacy_map = {"public": "public", "private": "private", "unlisted": "unlisted"}
    yt_privacy = privacy_map.get(privacy, "public")

    try:
        async with httpx.AsyncClient(timeout=300) as client:
            init_resp = await client.post(
                "https://www.googleapis.com/upload/youtube/v3/videos"
                "?uploadType=resumable&part=snippet,status",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json",
                    "X-Upload-Content-Type": "video/mp4",
                    "X-Upload-Content-Length": str(video_path.stat().st_size),
                },
                json={
                    "snippet": {
                        "title": (title or "")[:100],
                        "description": description,
                        "categoryId": "22",
                    },
                    "status": {
                        "privacyStatus": yt_privacy,
                        "selfDeclaredMadeForKids": False,
                    },
                },
            )

            if init_resp.status_code != 200:
                return PlatformResult(
                    platform="youtube", success=False,
                    error_code=str(init_resp.status_code),
                    error_message=init_resp.text[:300],
                )

            upload_url = init_resp.headers.get("Location")
            if not upload_url:
                return PlatformResult(
                    platform="youtube", success=False,
                    error_message="No resumable upload URL from YouTube",
                )

            with open(video_path, "rb") as f:
                video_bytes = f.read()

            upload_resp = await client.put(
                upload_url, content=video_bytes,
                headers={"Content-Type": "video/mp4"},
            )

            if upload_resp.status_code not in (200, 201):
                return PlatformResult(
                    platform="youtube", success=False,
                    error_code=str(upload_resp.status_code),
                    error_message=upload_resp.text[:300],
                )

            video_data = upload_resp.json()
            video_id = video_data.get("id")
            video_url = f"https://www.youtube.com/shorts/{video_id}" if video_id else None

            return PlatformResult(
                platform="youtube", success=True,
                platform_video_id=video_id, platform_url=video_url,
            )

    except Exception as e:
        logger.exception(f"YouTube publish exception: {e}")
        return PlatformResult(platform="youtube", success=False, error_message=str(e))


async def publish_to_instagram(
    video_path: Path,
    title: str,
    caption: Optional[str],
    hashtags: Any,
    privacy: str,
    token_data: Dict[str, Any],
    account_id: Optional[str] = None,
) -> PlatformResult:
    """Publish video to Instagram Reels via Meta Graph API."""
    access_token = token_data.get("access_token")
    ig_user_id = token_data.get("ig_user_id") or token_data.get("platform_user_id")

    if not access_token or not ig_user_id:
        return PlatformResult(
            platform="instagram", success=False,
            error_message="No access token or IG user ID",
        )

    full_caption = build_full_description(caption, hashtags, max_len=2200)

    try:
        async with httpx.AsyncClient(timeout=300) as client:
            container_resp = await client.post(
                f"https://graph.facebook.com/v21.0/{ig_user_id}/media",
                params={
                    "access_token": access_token,
                    "media_type": "REELS",
                    "caption": full_caption,
                    "share_to_feed": "true",
                },
            )

            if container_resp.status_code not in (200, 201):
                return PlatformResult(
                    platform="instagram", success=False,
                    error_code=str(container_resp.status_code),
                    error_message=container_resp.text[:300],
                )

            container_id = container_resp.json().get("id")
            if not container_id:
                return PlatformResult(
                    platform="instagram", success=False,
                    error_message="No container ID from Instagram",
                )

            for _ in range(12):
                await asyncio.sleep(10)
                status_resp = await client.get(
                    f"https://graph.facebook.com/v21.0/{container_id}",
                    params={"fields": "status_code,status", "access_token": access_token},
                )
                if status_resp.status_code == 200:
                    s = status_resp.json().get("status_code", "")
                    if s == "FINISHED":
                        break
                    elif s in ("ERROR", "EXPIRED"):
                        return PlatformResult(
                            platform="instagram", success=False,
                            error_message=f"Container status: {s}",
                        )

            publish_resp = await client.post(
                f"https://graph.facebook.com/v21.0/{ig_user_id}/media_publish",
                params={"access_token": access_token, "creation_id": container_id},
            )

            if publish_resp.status_code not in (200, 201):
                return PlatformResult(
                    platform="instagram", success=False,
                    error_code=str(publish_resp.status_code),
                    error_message=publish_resp.text[:300],
                )

            media_id = publish_resp.json().get("id")
            post_url = f"https://www.instagram.com/p/{media_id}/" if media_id else None

            return PlatformResult(
                platform="instagram", success=True,
                platform_video_id=media_id, platform_url=post_url,
            )

    except Exception as e:
        logger.exception(f"Instagram publish exception: {e}")
        return PlatformResult(platform="instagram", success=False, error_message=str(e))


async def publish_to_facebook(
    video_path: Path,
    title: str,
    caption: Optional[str],
    hashtags: Any,
    privacy: str,
    token_data: Dict[str, Any],
    account_id: Optional[str] = None,
) -> PlatformResult:
    """Publish video to Facebook Reels via Meta Graph API."""
    access_token = token_data.get("access_token")
    page_id = token_data.get("page_id") or token_data.get("platform_user_id")

    if not access_token or not page_id:
        return PlatformResult(
            platform="facebook", success=False,
            error_message="No access token or page ID",
        )

    description = build_full_description(caption, hashtags, max_len=5000)

    try:
        async with httpx.AsyncClient(timeout=300) as client:
            with open(video_path, "rb") as f:
                video_bytes = f.read()

            upload_resp = await client.post(
                f"https://graph.facebook.com/v21.0/{page_id}/videos",
                headers={"Authorization": f"Bearer {access_token}"},
                data={
                    "description": description,
                    "title": (title or "")[:255],
                },
                files={"source": ("video.mp4", video_bytes, "video/mp4")},
            )

            if upload_resp.status_code not in (200, 201):
                return PlatformResult(
                    platform="facebook", success=False,
                    error_code=str(upload_resp.status_code),
                    error_message=upload_resp.text[:300],
                )

            video_data = upload_resp.json()
            video_id = video_data.get("id")
            post_url = f"https://www.facebook.com/video/{video_id}" if video_id else None

            return PlatformResult(
                platform="facebook", success=True,
                platform_video_id=video_id, platform_url=post_url,
            )

    except Exception as e:
        logger.exception(f"Facebook publish exception: {e}")
        return PlatformResult(platform="facebook", success=False, error_message=str(e))


async def run_publish_stage(ctx: JobContext, db_pool, token_fetcher) -> JobContext:
    """
    Publish processed video to all requested platforms.

    Args:
        ctx: Job context
        db_pool: asyncpg connection pool
        token_fetcher: Async callable(platform, user_id, target_accounts) -> dict

    Returns:
        Updated context with platform_results populated
    """
    if not ctx.platforms:
        raise SkipStage("No platforms selected", stage="publish")

    def get_video_for_platform(platform: str) -> Optional[Path]:
        if platform in ctx.platform_videos and ctx.platform_videos[platform].exists():
            return ctx.platform_videos[platform]
        if ctx.processed_video_path and ctx.processed_video_path.exists():
            return ctx.processed_video_path
        if ctx.local_video_path and ctx.local_video_path.exists():
            return ctx.local_video_path
        return None

    final_title = ctx.ai_title or ctx.title or ctx.filename
    final_caption = ctx.ai_caption or ctx.caption or ""
    final_hashtags = ctx.ai_hashtags if ctx.ai_hashtags else ctx.hashtags

    logger.info(
        f"[Publish] upload={ctx.upload_id} | "
        f"title='{final_title[:60]}' | "
        f"hashtags={final_hashtags} | "
        f"location={ctx.location_name} | "
        f"platforms={ctx.platforms}"
    )

    results = []
    for platform in ctx.platforms:
        video_path = get_video_for_platform(platform)
        if not video_path:
            logger.error(f"No video file for {platform}")
            results.append(PlatformResult(
                platform=platform, success=False,
                error_message="No video file available",
            ))
            continue

        try:
            token_data = await token_fetcher(platform, ctx.user_id, ctx.target_accounts)
        except Exception as e:
            logger.error(f"Token fetch failed for {platform}: {e}")
            results.append(PlatformResult(
                platform=platform, success=False,
                error_message=f"Token fetch failed: {e}",
            ))
            continue

        if not token_data:
            results.append(PlatformResult(
                platform=platform, success=False,
                error_message="Platform not connected — no token",
            ))
            continue

        logger.info(f"Publishing to {platform}...")

        publish_fn = {
            "tiktok": publish_to_tiktok,
            "youtube": publish_to_youtube,
            "instagram": publish_to_instagram,
            "facebook": publish_to_facebook,
        }.get(platform)

        if not publish_fn:
            results.append(PlatformResult(
                platform=platform, success=False,
                error_message=f"Unsupported platform: {platform}",
            ))
            continue

        result = await publish_fn(
            video_path=video_path,
            title=final_title,
            caption=final_caption,
            hashtags=final_hashtags,
            privacy=ctx.privacy,
            token_data=token_data,
        )

        logger.info(
            f"{platform}: success={result.success} | "
            f"url={result.platform_url} | "
            f"err={result.error_message}"
        )
        results.append(result)

    ctx.platform_results = results
    return ctx
