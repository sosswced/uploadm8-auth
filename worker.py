"""
UploadM8 Worker Service
=======================
Async job consumer for video processing, telemetry HUD overlay, and platform publishing.

This worker:
1. Consumes jobs from Redis queue
2. Downloads videos from R2
3. Processes telemetry data and generates HUD overlays (if .map file present)
4. Calculates Trill scores
5. Uploads processed videos back to R2
6. Publishes to connected platforms (TikTok, YouTube, Instagram, Facebook)
7. Sends notifications via Discord webhooks

Run with: python worker.py
Or as a separate service: gunicorn worker:app -w 1 -k uvicorn.workers.UvicornWorker
"""

import os
import sys
import json
import asyncio
import logging
import tempfile
import subprocess
import time
import hashlib
import base64
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from pathlib import Path

import asyncpg
import httpx
import boto3
from botocore.config import Config
import redis.asyncio as redis
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# ============================================================
# Configuration
# ============================================================

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s [worker] %(message)s")
logger = logging.getLogger("uploadm8-worker")

DATABASE_URL = os.environ.get("DATABASE_URL")
REDIS_URL = os.environ.get("REDIS_URL", "")
UPLOAD_JOB_QUEUE = os.environ.get("UPLOAD_JOB_QUEUE", "uploadm8:jobs")
TELEMETRY_JOB_QUEUE = os.environ.get("TELEMETRY_JOB_QUEUE", "uploadm8:telemetry")

# R2 Configuration
R2_ACCOUNT_ID = os.environ.get("R2_ACCOUNT_ID", "")
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY", "")
R2_BUCKET_NAME = os.environ.get("R2_BUCKET_NAME", "uploadm8-media")
R2_ENDPOINT_URL = os.environ.get("R2_ENDPOINT_URL", "")

# Encryption keys
TOKEN_ENC_KEYS = os.environ.get("TOKEN_ENC_KEYS", "")

# Admin Discord webhook
ADMIN_DISCORD_WEBHOOK_URL = os.environ.get("ADMIN_DISCORD_WEBHOOK_URL", "")

# Trill scoring thresholds
SPEEDING_MPH_DEFAULT = int(os.environ.get("SPEEDING_MPH_DEFAULT", "80"))
EUPHORIA_MPH_DEFAULT = int(os.environ.get("EUPHORIA_MPH_DEFAULT", "100"))

# Worker settings
WORKER_CONCURRENCY = int(os.environ.get("WORKER_CONCURRENCY", "1"))
JOB_TIMEOUT_SECONDS = int(os.environ.get("JOB_TIMEOUT_SECONDS", "600"))
POLL_INTERVAL_SECONDS = float(os.environ.get("POLL_INTERVAL_SECONDS", "1.0"))

# ============================================================
# Global State
# ============================================================

db_pool: Optional[asyncpg.Pool] = None
redis_client: Optional[redis.Redis] = None
ENC_KEYS: Dict[str, bytes] = {}

# ============================================================
# Helpers
# ============================================================

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def parse_enc_keys() -> Dict[str, bytes]:
    if not TOKEN_ENC_KEYS:
        return {}
    keys: Dict[str, bytes] = {}
    clean = TOKEN_ENC_KEYS.strip().strip('"').replace("\\n", "")
    parts = [p.strip() for p in clean.split(",") if p.strip()]
    for part in parts:
        if ":" not in part:
            continue
        kid, b64key = part.split(":", 1)
        try:
            raw = base64.b64decode(b64key.strip())
            if len(raw) == 32:
                keys[kid.strip()] = raw
        except Exception:
            pass
    return keys

def decrypt_blob(blob: Any) -> dict:
    if isinstance(blob, str):
        blob = json.loads(blob)
    kid = blob.get("kid", "v1")
    key = ENC_KEYS.get(kid)
    if not key:
        raise ValueError(f"Unknown key id: {kid}")
    nonce = base64.b64decode(blob["nonce"])
    ciphertext = base64.b64decode(blob["ciphertext"])
    aesgcm = AESGCM(key)
    plaintext = aesgcm.decrypt(nonce, ciphertext, None)
    return json.loads(plaintext.decode("utf-8"))

# ============================================================
# R2 Client
# ============================================================

def get_s3_client():
    endpoint = R2_ENDPOINT_URL or f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )

async def download_from_r2(key: str, local_path: str) -> bool:
    """Download file from R2 to local path."""
    try:
        s3 = get_s3_client()
        s3.download_file(R2_BUCKET_NAME, key, local_path)
        logger.info(f"Downloaded {key} to {local_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download {key}: {e}")
        return False

async def upload_to_r2(local_path: str, key: str, content_type: str = "video/mp4") -> bool:
    """Upload file from local path to R2."""
    try:
        s3 = get_s3_client()
        s3.upload_file(local_path, R2_BUCKET_NAME, key, ExtraArgs={"ContentType": content_type})
        logger.info(f"Uploaded {local_path} to {key}")
        return True
    except Exception as e:
        logger.error(f"Failed to upload to {key}: {e}")
        return False

# ============================================================
# Telemetry Processing (Trill Score + HUD Overlay)
# ============================================================

def parse_map_file(map_path: str) -> list:
    """Parse .map telemetry file and return GPS data points."""
    data_points = []
    try:
        with open(map_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                # Expected format: timestamp,lat,lon,speed_mph,altitude (or similar CSV)
                parts = line.split(',')
                if len(parts) >= 4:
                    try:
                        data_points.append({
                            'timestamp': float(parts[0]),
                            'lat': float(parts[1]),
                            'lon': float(parts[2]),
                            'speed_mph': float(parts[3]),
                            'altitude': float(parts[4]) if len(parts) > 4 else 0
                        })
                    except ValueError:
                        continue
    except Exception as e:
        logger.warning(f"Error parsing map file: {e}")
    return data_points

def calculate_trill_score(data_points: list, speeding_mph: int = 80, euphoria_mph: int = 100) -> int:
    """Calculate Trill score (0-100) based on telemetry data."""
    if not data_points:
        return 0
    
    speeds = [p['speed_mph'] for p in data_points]
    if not speeds:
        return 0
    
    max_speed = max(speeds)
    avg_speed = sum(speeds) / len(speeds)
    
    # Base score from max speed (0-40 points)
    speed_score = min(40, (max_speed / euphoria_mph) * 40)
    
    # Speeding time bonus (0-30 points)
    speeding_time = sum(1 for s in speeds if s >= speeding_mph)
    speeding_ratio = speeding_time / len(speeds)
    speeding_score = speeding_ratio * 30
    
    # Euphoria bonus (0-20 points)
    euphoria_time = sum(1 for s in speeds if s >= euphoria_mph)
    euphoria_ratio = euphoria_time / len(speeds)
    euphoria_score = euphoria_ratio * 20
    
    # Consistency bonus (0-10 points)
    if len(speeds) > 1:
        variance = sum((s - avg_speed) ** 2 for s in speeds) / len(speeds)
        std_dev = variance ** 0.5
        consistency_score = max(0, 10 - (std_dev / 10))
    else:
        consistency_score = 5
    
    total = int(min(100, speed_score + speeding_score + euphoria_score + consistency_score))
    return total

def get_trill_title_modifier(score: int, max_speed: float) -> str:
    """Get title modifier based on Trill score."""
    if score >= 90:
        return " 🔥 #GloryBoyTour"
    elif score >= 80:
        return " ⚡ #Euphoric"
    elif score >= 70:
        return " 🚀 #SendIt"
    elif score >= 60:
        return " 💨 #Spirited"
    elif max_speed >= 100:
        return " ⚡"
    return ""

async def generate_hud_overlay(
    input_video: str,
    output_video: str,
    data_points: list,
    settings: dict
) -> bool:
    """Generate HUD overlay on video using FFmpeg."""
    if not data_points:
        return False
    
    # Get settings
    speed_unit = settings.get('hud_speed_unit', 'mph')
    hud_color = settings.get('hud_color', '#FFFFFF')
    font_family = settings.get('hud_font_family', 'Arial')
    font_size = settings.get('hud_font_size', 24)
    position = settings.get('hud_position', 'bottom-left')
    
    # Convert color from hex to FFmpeg format
    if hud_color.startswith('#'):
        hud_color = hud_color[1:]
    
    # Calculate position
    positions = {
        'top-left': '10:10',
        'top-right': 'W-tw-10:10',
        'bottom-left': '10:H-th-10',
        'bottom-right': 'W-tw-10:H-th-10',
        'center': '(W-tw)/2:(H-th)/2'
    }
    pos = positions.get(position, '10:H-th-10')
    
    # Create subtitle file with speed data
    with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False) as srt:
        srt_path = srt.name
        for i, point in enumerate(data_points):
            start_time = point['timestamp']
            end_time = data_points[i + 1]['timestamp'] if i + 1 < len(data_points) else start_time + 0.5
            
            speed = point['speed_mph']
            if speed_unit == 'kmh':
                speed = speed * 1.60934
                unit_label = 'KM/H'
            else:
                unit_label = 'MPH'
            
            # Format time as SRT timestamp
            def format_srt_time(seconds):
                h = int(seconds // 3600)
                m = int((seconds % 3600) // 60)
                s = int(seconds % 60)
                ms = int((seconds % 1) * 1000)
                return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
            
            srt.write(f"{i + 1}\n")
            srt.write(f"{format_srt_time(start_time)} --> {format_srt_time(end_time)}\n")
            srt.write(f"{int(speed)} {unit_label}\n\n")
    
    try:
        # Use FFmpeg to burn subtitles
        cmd = [
            'ffmpeg', '-y',
            '-i', input_video,
            '-vf', f"subtitles={srt_path}:force_style='FontName={font_family},FontSize={font_size},PrimaryColour=&H{hud_color}&,Alignment=1,MarginV=20'",
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-c:a', 'copy',
            output_video
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=JOB_TIMEOUT_SECONDS)
        
        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr}")
            return False
        
        return True
    except subprocess.TimeoutExpired:
        logger.error("FFmpeg timed out")
        return False
    except FileNotFoundError:
        logger.warning("FFmpeg not found - HUD overlay skipped")
        return False
    except Exception as e:
        logger.error(f"HUD overlay failed: {e}")
        return False
    finally:
        # Clean up SRT file
        try:
            os.unlink(srt_path)
        except:
            pass

# ============================================================
# Platform Publishing
# ============================================================

async def publish_to_tiktok(video_path: str, title: str, token_data: dict) -> dict:
    """Publish video to TikTok."""
    access_token = token_data.get('access_token')
    if not access_token:
        return {"success": False, "error": "No access token"}
    
    try:
        # TikTok Content Posting API
        async with httpx.AsyncClient(timeout=120) as client:
            # Step 1: Initialize upload
            init_resp = await client.post(
                "https://open.tiktokapis.com/v2/post/publish/video/init/",
                headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
                json={
                    "post_info": {
                        "title": title[:150],
                        "privacy_level": "PUBLIC_TO_EVERYONE",
                    },
                    "source_info": {
                        "source": "FILE_UPLOAD",
                        "video_size": os.path.getsize(video_path),
                    }
                }
            )
            
            if init_resp.status_code != 200:
                return {"success": False, "error": f"TikTok init failed: {init_resp.text}"}
            
            init_data = init_resp.json().get("data", {})
            upload_url = init_data.get("upload_url")
            publish_id = init_data.get("publish_id")
            
            if not upload_url:
                return {"success": False, "error": "No upload URL returned"}
            
            # Step 2: Upload video
            with open(video_path, 'rb') as f:
                video_data = f.read()
            
            upload_resp = await client.put(
                upload_url,
                content=video_data,
                headers={"Content-Type": "video/mp4"}
            )
            
            if upload_resp.status_code not in (200, 201):
                return {"success": False, "error": f"TikTok upload failed: {upload_resp.status_code}"}
            
            return {"success": True, "publish_id": publish_id}
    
    except Exception as e:
        logger.error(f"TikTok publish error: {e}")
        return {"success": False, "error": str(e)}

async def publish_to_youtube(video_path: str, title: str, description: str, token_data: dict) -> dict:
    """Publish video to YouTube Shorts."""
    access_token = token_data.get('access_token')
    if not access_token:
        return {"success": False, "error": "No access token"}
    
    try:
        # YouTube uses resumable uploads
        async with httpx.AsyncClient(timeout=300) as client:
            # Step 1: Initialize resumable upload
            metadata = {
                "snippet": {
                    "title": title[:100],
                    "description": description[:5000] if description else "",
                    "categoryId": "22"  # People & Blogs
                },
                "status": {
                    "privacyStatus": "public",
                    "selfDeclaredMadeForKids": False
                }
            }
            
            init_resp = await client.post(
                "https://www.googleapis.com/upload/youtube/v3/videos?uploadType=resumable&part=snippet,status",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json",
                    "X-Upload-Content-Type": "video/mp4",
                    "X-Upload-Content-Length": str(os.path.getsize(video_path))
                },
                json=metadata
            )
            
            if init_resp.status_code != 200:
                return {"success": False, "error": f"YouTube init failed: {init_resp.text}"}
            
            upload_url = init_resp.headers.get("Location")
            if not upload_url:
                return {"success": False, "error": "No upload URL"}
            
            # Step 2: Upload video
            with open(video_path, 'rb') as f:
                video_data = f.read()
            
            upload_resp = await client.put(
                upload_url,
                content=video_data,
                headers={"Content-Type": "video/mp4"}
            )
            
            if upload_resp.status_code not in (200, 201):
                return {"success": False, "error": f"YouTube upload failed: {upload_resp.status_code}"}
            
            video_id = upload_resp.json().get("id")
            return {"success": True, "video_id": video_id}
    
    except Exception as e:
        logger.error(f"YouTube publish error: {e}")
        return {"success": False, "error": str(e)}

async def publish_to_instagram(video_path: str, caption: str, token_data: dict, page_id: str) -> dict:
    """Publish video to Instagram Reels."""
    access_token = token_data.get('access_token')
    if not access_token or not page_id:
        return {"success": False, "error": "Missing access token or page ID"}
    
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            # Get Instagram Business Account ID
            ig_resp = await client.get(
                f"https://graph.facebook.com/v19.0/{page_id}",
                params={"fields": "instagram_business_account", "access_token": access_token}
            )
            
            ig_data = ig_resp.json()
            ig_account_id = ig_data.get("instagram_business_account", {}).get("id")
            
            if not ig_account_id:
                return {"success": False, "error": "No Instagram Business Account linked"}
            
            # Upload video to temporary URL (would need external hosting in production)
            # For now, return placeholder
            return {"success": False, "error": "Instagram Reels upload requires public video URL"}
    
    except Exception as e:
        logger.error(f"Instagram publish error: {e}")
        return {"success": False, "error": str(e)}

async def publish_to_facebook(video_path: str, description: str, token_data: dict, page_id: str) -> dict:
    """Publish video to Facebook Reels."""
    access_token = token_data.get('access_token')
    if not access_token or not page_id:
        return {"success": False, "error": "Missing access token or page ID"}
    
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            # Facebook video upload
            with open(video_path, 'rb') as f:
                files = {"source": ("video.mp4", f, "video/mp4")}
                resp = await client.post(
                    f"https://graph.facebook.com/v19.0/{page_id}/videos",
                    params={"access_token": access_token, "description": description[:5000] if description else ""},
                    files=files
                )
            
            if resp.status_code != 200:
                return {"success": False, "error": f"Facebook upload failed: {resp.text}"}
            
            video_id = resp.json().get("id")
            return {"success": True, "video_id": video_id}
    
    except Exception as e:
        logger.error(f"Facebook publish error: {e}")
        return {"success": False, "error": str(e)}

# ============================================================
# Discord Notifications
# ============================================================

async def notify_discord(webhook_url: str, message: str):
    """Send notification to Discord webhook."""
    if not webhook_url:
        return
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(webhook_url, json={"content": message})
    except Exception as e:
        logger.warning(f"Discord notification failed: {e}")

async def notify_admin(message: str):
    """Send notification to admin Discord webhook."""
    await notify_discord(ADMIN_DISCORD_WEBHOOK_URL, message)

# ============================================================
# Job Handlers
# ============================================================

async def handle_process_upload(job: dict):
    """Process an upload job - download, process telemetry, publish."""
    upload_id = job.get("upload_id")
    user_id = job.get("user_id")
    has_telemetry = job.get("has_telemetry", False)
    
    logger.info(f"Processing upload {upload_id} for user {user_id}")
    
    if not db_pool:
        logger.error("Database not available")
        return False
    
    async with db_pool.acquire() as conn:
        # Get upload details
        upload = await conn.fetchrow("SELECT * FROM uploads WHERE id = $1", upload_id)
        if not upload:
            logger.error(f"Upload {upload_id} not found")
            return False
        
        # Get user settings
        settings = await conn.fetchrow("SELECT * FROM user_settings WHERE user_id = $1", user_id)
        settings = dict(settings) if settings else {}
        
        # Get user webhook
        webhook_url = settings.get("discord_webhook")
        
        # Update status to processing
        await conn.execute(
            "UPDATE uploads SET status = 'processing', processing_started_at = NOW() WHERE id = $1",
            upload_id
        )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Download video
        video_path = os.path.join(tmpdir, upload["filename"])
        if not await download_from_r2(upload["r2_key"], video_path):
            await _mark_upload_failed(upload_id, "r2_download_failed", "Failed to download video from R2")
            return False
        
        processed_path = video_path
        trill_score = None
        title_modifier = ""
        
        # Process telemetry if available
        if has_telemetry and upload.get("telemetry_r2_key"):
            map_path = os.path.join(tmpdir, "telemetry.map")
            if await download_from_r2(upload["telemetry_r2_key"], map_path):
                data_points = parse_map_file(map_path)
                
                if data_points:
                    # Calculate Trill score
                    speeding_mph = settings.get("speeding_mph", SPEEDING_MPH_DEFAULT)
                    euphoria_mph = settings.get("euphoria_mph", EUPHORIA_MPH_DEFAULT)
                    trill_score = calculate_trill_score(data_points, speeding_mph, euphoria_mph)
                    
                    max_speed = max(p['speed_mph'] for p in data_points)
                    title_modifier = get_trill_title_modifier(trill_score, max_speed)
                    
                    # Generate HUD overlay
                    if settings.get("hud_enabled", True):
                        hud_path = os.path.join(tmpdir, "processed_" + upload["filename"])
                        if await generate_hud_overlay(video_path, hud_path, data_points, settings):
                            processed_path = hud_path
                            logger.info(f"HUD overlay generated for {upload_id}")
        
        # Upload processed video back to R2
        processed_key = None
        if processed_path != video_path:
            processed_key = upload["r2_key"].replace("uploads/", "processed/")
            if not await upload_to_r2(processed_path, processed_key):
                logger.warning("Failed to upload processed video, using original")
                processed_key = None
        
        # Update upload record
        async with db_pool.acquire() as conn:
            await conn.execute(
                """UPDATE uploads SET 
                       processed_r2_key = COALESCE($2, processed_r2_key),
                       trill_score = $3,
                       title = COALESCE(title, '') || $4,
                       status = 'publishing',
                       processing_finished_at = NOW()
                   WHERE id = $1""",
                upload_id, processed_key, trill_score, title_modifier
            )
        
        # Get updated upload with modified title
        async with db_pool.acquire() as conn:
            upload = await conn.fetchrow("SELECT * FROM uploads WHERE id = $1", upload_id)
        
        # Publish to platforms
        platforms = upload.get("platforms") or []
        publish_results = {}
        
        async with db_pool.acquire() as conn:
            for platform in platforms:
                # Get platform token
                platform_key = "google" if platform == "youtube" else "meta" if platform in ("instagram", "facebook") else platform
                token_row = await conn.fetchrow(
                    "SELECT token_blob FROM platform_tokens WHERE user_id = $1 AND platform = $2",
                    user_id, platform_key
                )
                
                if not token_row:
                    publish_results[platform] = {"success": False, "error": "Platform not connected"}
                    continue
                
                try:
                    token_data = decrypt_blob(token_row["token_blob"])
                except Exception as e:
                    publish_results[platform] = {"success": False, "error": f"Token decrypt failed: {e}"}
                    continue
                
                title = upload.get("title") or upload.get("filename") or "Video"
                caption = upload.get("caption") or ""
                
                if platform == "tiktok":
                    result = await publish_to_tiktok(processed_path, title, token_data)
                elif platform == "youtube":
                    result = await publish_to_youtube(processed_path, title, caption, token_data)
                elif platform == "instagram":
                    page_id = settings.get("selected_page_id")
                    result = await publish_to_instagram(processed_path, caption, token_data, page_id)
                elif platform == "facebook":
                    page_id = settings.get("selected_page_id")
                    result = await publish_to_facebook(processed_path, caption, token_data, page_id)
                else:
                    result = {"success": False, "error": f"Unknown platform: {platform}"}
                
                publish_results[platform] = result
        
        # Determine final status
        any_success = any(r.get("success") for r in publish_results.values())
        all_success = all(r.get("success") for r in publish_results.values())
        
        if all_success:
            final_status = "completed"
            error_code = None
            error_detail = None
        elif any_success:
            final_status = "partial"
            error_code = "partial_publish"
            error_detail = json.dumps({k: v for k, v in publish_results.items() if not v.get("success")})
        else:
            final_status = "failed"
            error_code = "publish_failed"
            error_detail = json.dumps(publish_results)
        
        # Update final status
        async with db_pool.acquire() as conn:
            await conn.execute(
                """UPDATE uploads SET 
                       status = $2,
                       completed_at = NOW(),
                       error_code = $3,
                       error_detail = $4
                   WHERE id = $1""",
                upload_id, final_status, error_code, error_detail
            )
        
        # Send notifications
        if webhook_url:
            if final_status == "completed":
                msg = f"✅ **Upload Complete**: {upload.get('title') or upload.get('filename')}"
                if trill_score:
                    msg += f"\n⚡ Trill Score: {trill_score}/100"
            elif final_status == "partial":
                msg = f"⚠️ **Partial Upload**: {upload.get('title') or upload.get('filename')} - Some platforms failed"
            else:
                msg = f"❌ **Upload Failed**: {upload.get('title') or upload.get('filename')}"
            
            await notify_discord(webhook_url, msg)
        
        logger.info(f"Upload {upload_id} completed with status: {final_status}")
        return final_status != "failed"

async def handle_process_telemetry(job: dict):
    """Process a telemetry-only job (for videos already uploaded)."""
    upload_id = job.get("upload_id")
    user_id = job.get("user_id")
    telemetry_key = job.get("telemetry_key")
    
    logger.info(f"Processing telemetry for upload {upload_id}")
    
    if not db_pool:
        return False
    
    async with db_pool.acquire() as conn:
        upload = await conn.fetchrow("SELECT * FROM uploads WHERE id = $1 AND user_id = $2", upload_id, user_id)
        if not upload:
            logger.error(f"Upload {upload_id} not found")
            return False
        
        settings = await conn.fetchrow("SELECT * FROM user_settings WHERE user_id = $1", user_id)
        settings = dict(settings) if settings else {}
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Download telemetry file
        map_path = os.path.join(tmpdir, "telemetry.map")
        if not await download_from_r2(telemetry_key, map_path):
            logger.error("Failed to download telemetry file")
            return False
        
        data_points = parse_map_file(map_path)
        if not data_points:
            logger.warning("No valid telemetry data found")
            return False
        
        # Calculate Trill score
        speeding_mph = settings.get("speeding_mph", SPEEDING_MPH_DEFAULT)
        euphoria_mph = settings.get("euphoria_mph", EUPHORIA_MPH_DEFAULT)
        trill_score = calculate_trill_score(data_points, speeding_mph, euphoria_mph)
        
        max_speed = max(p['speed_mph'] for p in data_points)
        title_modifier = get_trill_title_modifier(trill_score, max_speed)
        
        # Update upload with Trill score
        async with db_pool.acquire() as conn:
            await conn.execute(
                """UPDATE uploads SET 
                       trill_score = $2,
                       title = COALESCE(title, '') || $3,
                       telemetry_r2_key = $4
                   WHERE id = $1""",
                upload_id, trill_score, title_modifier, telemetry_key
            )
        
        logger.info(f"Telemetry processed for {upload_id}: Trill score = {trill_score}")
        return True

async def _mark_upload_failed(upload_id: str, error_code: str, error_detail: str):
    """Mark an upload as failed."""
    if not db_pool:
        return
    async with db_pool.acquire() as conn:
        await conn.execute(
            """UPDATE uploads SET 
                   status = 'failed',
                   error_code = $2,
                   error_detail = $3,
                   processing_finished_at = NOW()
               WHERE id = $1""",
            upload_id, error_code, error_detail
        )

# ============================================================
# Main Worker Loop
# ============================================================

async def process_job(job_json: str) -> bool:
    """Process a single job from the queue."""
    try:
        job = json.loads(job_json)
        job_type = job.get("type")
        job_id = job.get("job_id", "unknown")
        
        logger.info(f"Processing job {job_id} of type {job_type}")
        
        if job_type == "process_upload":
            return await handle_process_upload(job)
        elif job_type == "process_telemetry":
            return await handle_process_telemetry(job)
        else:
            logger.warning(f"Unknown job type: {job_type}")
            return False
    
    except json.JSONDecodeError as e:
        logger.error(f"Invalid job JSON: {e}")
        return False
    except Exception as e:
        logger.exception(f"Job processing failed: {e}")
        return False

async def worker_loop():
    """Main worker loop - consume jobs from Redis queues."""
    global db_pool, redis_client, ENC_KEYS
    
    # Initialize encryption keys
    ENC_KEYS = parse_enc_keys()
    
    # Connect to database
    if DATABASE_URL:
        db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
        logger.info("Database connected")
    else:
        logger.error("DATABASE_URL not set")
        return
    
    # Connect to Redis
    if REDIS_URL:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        await redis_client.ping()
        logger.info("Redis connected")
    else:
        logger.error("REDIS_URL not set")
        return
    
    queues = [UPLOAD_JOB_QUEUE, TELEMETRY_JOB_QUEUE]
    logger.info(f"Worker started, listening on queues: {queues}")
    
    await notify_admin("🟢 **UploadM8 Worker Started**")
    
    try:
        while True:
            try:
                # BRPOP blocks until a job is available
                result = await redis_client.brpop(queues, timeout=int(POLL_INTERVAL_SECONDS * 10))
                
                if result:
                    queue_name, job_json = result
                    success = await process_job(job_json)
                    if not success:
                        logger.warning(f"Job failed from queue {queue_name}")
                else:
                    # Timeout - no jobs available
                    await asyncio.sleep(POLL_INTERVAL_SECONDS)
            
            except redis.ConnectionError as e:
                logger.error(f"Redis connection error: {e}")
                await asyncio.sleep(5)
                redis_client = redis.from_url(REDIS_URL, decode_responses=True)
            
            except Exception as e:
                logger.exception(f"Worker loop error: {e}")
                await asyncio.sleep(5)
    
    finally:
        if db_pool:
            await db_pool.close()
        if redis_client:
            await redis_client.close()
        await notify_admin("🔴 **UploadM8 Worker Stopped**")

def main():
    """Entry point."""
    logger.info("Starting UploadM8 Worker...")
    asyncio.run(worker_loop())

if __name__ == "__main__":
    main()
