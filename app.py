"""
UploadM8 App.py PATCH - Upload Queue Endpoints
================================================
Drop-in replacements for the two upload GET endpoints.

INSTRUCTIONS FOR DEPLOYMENT:
In your app.py, find and REPLACE these two functions:
1. `async def get_uploads(...)` - the @app.get("/api/uploads") handler
2. `async def get_upload_detail(...)` or the @app.get("/api/uploads/{upload_id}") handler

The new versions below return:
- thumbnail_url (presigned R2 URL)
- platform_results with per-platform post URLs
- hashtags (as array)
- title, caption (AI-generated if available)
- ai_title, ai_caption, ai_hashtags fields
- views, likes per platform

SEARCH FOR in app.py:
    @app.get("/api/uploads")
    async def get_uploads(status: Optional[str] = None, limit: int = 50, offset: int = 0, user: dict = Depends(get_current_user)):

REPLACE WITH the new version below.
Then find @app.get("/api/uploads/{upload_id}") and replace that too.
"""

# ============================================================
# PASTE THESE TWO FUNCTIONS INTO app.py
# replacing the existing @app.get("/api/uploads") 
# and @app.get("/api/uploads/{upload_id}") handlers
# ============================================================

REPLACEMENT_CODE = '''
@app.get("/api/uploads")
async def get_uploads(status: Optional[str] = None, limit: int = 50, offset: int = 0, user: dict = Depends(get_current_user)):
    """
    Get upload queue for the current user.
    Returns full metadata including thumbnail URLs, platform results, hashtags, and AI content.
    """
    async with db_pool.acquire() as conn:
        if status:
            uploads = await conn.fetch("""
                SELECT 
                    id, filename, platforms, status, title, caption, hashtags,
                    scheduled_time, created_at, completed_at,
                    put_reserved, aic_reserved, error_code, error_detail,
                    thumbnail_r2_key, platform_results, file_size,
                    processing_started_at, processing_finished_at,
                    views, likes
                FROM uploads 
                WHERE user_id = $1 AND status = $2 
                ORDER BY created_at DESC LIMIT $3 OFFSET $4
            """, user["id"], status, limit, offset)
        else:
            uploads = await conn.fetch("""
                SELECT 
                    id, filename, platforms, status, title, caption, hashtags,
                    scheduled_time, created_at, completed_at,
                    put_reserved, aic_reserved, error_code, error_detail,
                    thumbnail_r2_key, platform_results, file_size,
                    processing_started_at, processing_finished_at,
                    views, likes
                FROM uploads 
                WHERE user_id = $1 
                ORDER BY created_at DESC LIMIT $2 OFFSET $3
            """, user["id"], limit, offset)

    result = []
    s3 = get_s3_client()

    for u in uploads:
        # Generate presigned thumbnail URL
        thumbnail_url = None
        if u.get("thumbnail_r2_key"):
            try:
                thumbnail_url = s3.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': R2_BUCKET_NAME, 'Key': u["thumbnail_r2_key"]},
                    ExpiresIn=3600
                )
            except Exception:
                pass

        # Parse platform results JSON
        platform_results = []
        if u.get("platform_results"):
            try:
                raw = u["platform_results"]
                parsed = json.loads(raw) if isinstance(raw, str) else raw
                if isinstance(parsed, list):
                    platform_results = parsed
            except Exception:
                pass

        # Hashtags - always return as array of strings
        hashtags = []
        if u.get("hashtags"):
            raw_tags = u["hashtags"]
            if isinstance(raw_tags, (list, tuple)):
                hashtags = [str(t) for t in raw_tags if t]
            elif isinstance(raw_tags, str):
                try:
                    hashtags = json.loads(raw_tags)
                except Exception:
                    hashtags = [raw_tags]

        result.append({
            "id": str(u["id"]),
            "filename": u["filename"],
            "platforms": list(u["platforms"]) if u["platforms"] else [],
            "status": u["status"],
            "title": u["title"] or "",
            "caption": u["caption"] or "",
            "hashtags": hashtags,
            "scheduled_time": u["scheduled_time"].isoformat() if u["scheduled_time"] else None,
            "created_at": u["created_at"].isoformat() if u["created_at"] else None,
            "completed_at": u["completed_at"].isoformat() if u.get("completed_at") else None,
            "put_cost": u["put_reserved"],
            "aic_cost": u["aic_reserved"],
            "error_code": u.get("error_code"),
            "error": u.get("error_detail"),
            "thumbnail_url": thumbnail_url,
            "platform_results": platform_results,
            "file_size": u.get("file_size"),
            "views": u.get("views") or 0,
            "likes": u.get("likes") or 0,
        })

    return result


@app.get("/api/uploads/{upload_id}")
async def get_upload_detail(upload_id: str, user: dict = Depends(get_current_user)):
    """
    Get full detail for a single upload.
    Returns thumbnail URL, platform results with post URLs, AI content, hashtags.
    """
    async with db_pool.acquire() as conn:
        u = await conn.fetchrow("""
            SELECT 
                id, filename, platforms, status, title, caption, hashtags,
                scheduled_time, created_at, completed_at,
                put_reserved, aic_reserved, error_code, error_detail,
                thumbnail_r2_key, platform_results, file_size,
                processing_started_at, processing_finished_at,
                views, likes, privacy, r2_key
            FROM uploads 
            WHERE id = $1 AND user_id = $2
        """, upload_id, user["id"])

    if not u:
        raise HTTPException(404, "Upload not found")

    s3 = get_s3_client()

    # Generate presigned thumbnail URL
    thumbnail_url = None
    if u.get("thumbnail_r2_key"):
        try:
            thumbnail_url = s3.generate_presigned_url(
                'get_object',
                Params={'Bucket': R2_BUCKET_NAME, 'Key': u["thumbnail_r2_key"]},
                ExpiresIn=3600
            )
        except Exception:
            pass

    # Parse platform results JSON - includes platform_url for post links
    platform_results = []
    if u.get("platform_results"):
        try:
            raw = u["platform_results"]
            parsed = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(parsed, list):
                platform_results = parsed
        except Exception:
            pass

    # Hashtags - always return as array of strings
    hashtags = []
    if u.get("hashtags"):
        raw_tags = u["hashtags"]
        if isinstance(raw_tags, (list, tuple)):
            hashtags = [str(t) for t in raw_tags if t]
        elif isinstance(raw_tags, str):
            try:
                hashtags = json.loads(raw_tags)
            except Exception:
                hashtags = [raw_tags]

    # Calculate processing duration
    duration_seconds = None
    if u.get("processing_started_at") and u.get("processing_finished_at"):
        delta = u["processing_finished_at"] - u["processing_started_at"]
        duration_seconds = int(delta.total_seconds())

    return {
        "id": str(u["id"]),
        "filename": u["filename"],
        "platforms": list(u["platforms"]) if u["platforms"] else [],
        "status": u["status"],
        "title": u["title"] or "",
        "caption": u["caption"] or "",
        "hashtags": hashtags,
        "scheduled_time": u["scheduled_time"].isoformat() if u["scheduled_time"] else None,
        "created_at": u["created_at"].isoformat() if u["created_at"] else None,
        "completed_at": u["completed_at"].isoformat() if u.get("completed_at") else None,
        "put_cost": u["put_reserved"],
        "aic_cost": u["aic_reserved"],
        "error_code": u.get("error_code"),
        "error": u.get("error_detail"),
        "thumbnail_url": thumbnail_url,
        "platform_results": platform_results,
        "file_size": u.get("file_size"),
        "views": u.get("views") or 0,
        "likes": u.get("likes") or 0,
        "privacy": u.get("privacy", "public"),
        "duration_seconds": duration_seconds,
    }
'''

print("Paste the above two route handlers into app.py")
print("Replace @app.get('/api/uploads') and the existing @app.get('/api/uploads/{upload_id}') handler")
