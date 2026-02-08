# ============================================================
# TRILL TELEMETRY INTEGRATION
# Add this to your main.py after the existing imports
# ============================================================

"""
INTEGRATION GUIDE:
1. Add these dependencies to requirements.txt:
   - openai>=1.0.0
   - geopandas (optional, for protected lands)
   - shapely (optional, for protected lands)
   
2. Add environment variables:
   - OPENAI_API_KEY=your_key
   - GAZETTEER_PLACES_PATH=/path/to/2024_Gaz_places_national.txt
   - PADUS_PATH=/path/to/PADUS.gpkg (optional)
   
3. Copy telemetry_trill.py to your project root

4. Add database migrations (add to run_migrations() function)

5. Add API endpoints (copy to app section)

6. Update upload workflow (modify /api/uploads/{upload_id}/complete)
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime, timezone

import openai
from fastapi import HTTPException

# Import your trill module
try:
    import telemetry_trill as tt
except ImportError:
    tt = None
    logging.warning("telemetry_trill.py not found - trill features disabled")

# ============================================================
# Configuration
# ============================================================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GAZETTEER_PLACES_PATH = os.environ.get("GAZETTEER_PLACES_PATH", "")
PADUS_PATH = os.environ.get("PADUS_PATH", "")
PADUS_LAYER = os.environ.get("PADUS_LAYER", "")

# Trill content generation settings
TRILL_SYSTEM_PROMPT = """You are an expert social media content creator specializing in driving/automotive content.
Create viral, high-engagement content that balances excitement with platform safety guidelines.

STYLE GUIDE:
- High-energy, aspirational tone
- Use curiosity gaps and FOMO triggers
- Platform-native language (not corporate)
- Mystery hooks that make people watch
- Avoid: Illegal references, exact speeds, clickbait lies

EMOJI USAGE:
- Titles: 1-2 emojis max (fire 🔥, lightning ⚡, eyes 👀)
- Captions: 2-3 emojis strategically placed
- Never excessive or spammy

HASHTAG STRATEGY:
- Mix viral mega-tags (#fyp, #viral, #trending)
- Niche community tags (#spiriteddrive, #roadtrip)
- Location-based tags (#Utah, #Moab)
- Motion tags when relevant (#curvyroads, #switchbacks)
- Protected lands tags ONLY when verified
"""

# ============================================================
# Database Schema (add to migrations)
# ============================================================
TRILL_MIGRATIONS = [
    (100, """
        -- Trill analysis results
        ALTER TABLE uploads ADD COLUMN IF NOT EXISTS trill_score DECIMAL(5,2);
        ALTER TABLE uploads ADD COLUMN IF NOT EXISTS speed_bucket VARCHAR(50);
        ALTER TABLE uploads ADD COLUMN IF NOT EXISTS trill_metadata JSONB;
        ALTER TABLE uploads ADD COLUMN IF NOT EXISTS ai_generated_title TEXT;
        ALTER TABLE uploads ADD COLUMN IF NOT EXISTS ai_generated_caption TEXT;
        ALTER TABLE uploads ADD COLUMN IF NOT EXISTS ai_generated_hashtags TEXT[];
    """),
    (101, """
        -- Trill places (popular locations for geo-targeting)
        CREATE TABLE IF NOT EXISTS trill_places (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name VARCHAR(255) NOT NULL,
            state VARCHAR(2) NOT NULL,
            lat DECIMAL(10,7) NOT NULL,
            lon DECIMAL(10,7) NOT NULL,
            popularity_score INT DEFAULT 0,
            hashtags TEXT[] DEFAULT '{}',
            is_protected BOOLEAN DEFAULT FALSE,
            protected_name VARCHAR(255),
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(name, state)
        );
        CREATE INDEX IF NOT EXISTS idx_trill_places_state ON trill_places(state);
        CREATE INDEX IF NOT EXISTS idx_trill_places_popularity ON trill_places(popularity_score DESC);
    """),
    (102, """
        -- User trill preferences
        ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS trill_enabled BOOLEAN DEFAULT FALSE;
        ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS trill_min_score INT DEFAULT 60;
        ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS trill_hud_enabled BOOLEAN DEFAULT FALSE;
        ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS trill_ai_enhance BOOLEAN DEFAULT TRUE;
        ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS trill_openai_model VARCHAR(50) DEFAULT 'gpt-4o-mini';
    """),
]

# ============================================================
# OpenAI Content Generation
# ============================================================
def generate_trill_content(trill_metadata: dict, user_prefs: dict = None) -> dict:
    """
    Use OpenAI to generate viral titles, captions, and hashtags based on trill metrics.
    
    Args:
        trill_metadata: Output from telemetry_trill.analyze_video()
        user_prefs: User preferences for generation style
        
    Returns:
        {
            "title": "Generated title",
            "caption": "Generated caption", 
            "hashtags": ["tag1", "tag2", ...],
            "tokens_used": 150,
            "model": "gpt-4o-mini"
        }
    """
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not configured")
    
    openai.api_key = OPENAI_API_KEY
    
    # Extract key metrics
    score = trill_metadata.get("trill_score", 0)
    bucket = trill_metadata.get("speed_bucket", "CRUISE MODE")
    place = trill_metadata.get("place_name", "")
    state = trill_metadata.get("state", "")
    protected_name = trill_metadata.get("protected_name")
    near_protected = trill_metadata.get("near_protected", False)
    elev_gain = trill_metadata.get("elev_gain_m", 0)
    curv_score = trill_metadata.get("curv_score", 0)
    dyn_score = trill_metadata.get("dyn_score", 0)
    turny = trill_metadata.get("turny", False)
    spirited = trill_metadata.get("spirited", False)
    
    # Build context
    location = f"near {place}, {state}" if place and state else state if state else "the open road"
    scene = f"{protected_name} (verified protected lands)" if near_protected and protected_name else "public lands" if near_protected else "backroads"
    
    # User preferences
    prefs = user_prefs or {}
    model = prefs.get("trill_openai_model", "gpt-4o-mini")
    platform_focus = prefs.get("platform_focus", "all")  # tiktok, youtube, instagram, facebook, all
    
    # Build prompt
    user_prompt = f"""Generate viral social media content for a driving video with these metrics:

TRILL SCORE: {score}/100 (higher = more thrilling)
SPEED BUCKET: {bucket}
LOCATION: {location}
SCENE: {scene}
ELEVATION GAIN: {elev_gain}m
CURVATURE: {curv_score}/10 (higher = more twisty/switchbacks)
DYNAMICS: {dyn_score}/10 (higher = more spirited cornering)
MOTION FLAGS: {"Turny roads" if turny else ""} {"Spirited driving" if spirited else ""}

TARGET PLATFORM: {platform_focus}

GENERATE:
1. TITLE (max 80 chars)
   - Create mystery/curiosity gap
   - Use 1-2 emojis strategically
   - Make it stop-the-scroll worthy
   - Examples: "This road changed my perspective 🔥" or "POV: You find the perfect line ⚡"
   
2. CAPTION (max 200 chars)
   - First-person, conversational
   - Create FOMO/aspiration
   - Ask a question or prompt engagement
   - 2-3 emojis max
   
3. HASHTAGS (exactly 15 tags)
   - 3-4 mega viral: #fyp #foryou #viral #trending
   - 4-5 niche community: #roadtrip #driving #explore
   - 3-4 location: #{state} #{place} (if available)
   - 2-3 motion: #curvyroads #spiriteddrive (if applicable)
   - 1-2 protected lands: #publiclands #nationalpark (ONLY if verified: {near_protected})

RETURN ONLY THIS JSON (no markdown, no backticks):
{{
  "title": "your title here",
  "caption": "your caption here",
  "hashtags": ["tag1", "tag2", ...]
}}
"""
    
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": TRILL_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.85,  # High creativity but not random
            max_tokens=500,
            response_format={"type": "json_object"}  # Ensures valid JSON
        )
        
        content = response.choices[0].message.content
        tokens_used = response.usage.total_tokens
        
        # Parse response
        result = json.loads(content)
        
        # Validate and clean hashtags
        hashtags = result.get("hashtags", [])
        hashtags = [h.lower().replace("#", "").replace(" ", "") for h in hashtags]
        hashtags = [h for h in hashtags if h and len(h) <= 30][:15]  # Max 15 tags
        
        return {
            "title": result.get("title", "")[:100],
            "caption": result.get("caption", "")[:250],
            "hashtags": hashtags,
            "tokens_used": tokens_used,
            "model": model,
            "trill_score": score,
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logging.error(f"OpenAI generation failed: {e}")
        # Fallback to trill's built-in generation
        return {
            "title": trill_metadata.get("title", ""),
            "caption": trill_metadata.get("caption", ""),
            "hashtags": trill_metadata.get("hashtags", []),
            "tokens_used": 0,
            "model": "fallback",
            "error": str(e)
        }

# ============================================================
# Trill Places Database
# ============================================================
async def seed_trill_places(conn):
    """Seed database with popular driving locations"""
    places = [
        # Utah
        {"name": "Moab", "state": "UT", "lat": 38.5733, "lon": -109.5498, "popularity": 95, "protected_name": "Arches National Park"},
        {"name": "Zion", "state": "UT", "lat": 37.2982, "lon": -113.0263, "popularity": 90, "protected_name": "Zion National Park"},
        
        # California  
        {"name": "Big Sur", "state": "CA", "lat": 36.2704, "lon": -121.8081, "popularity": 92, "protected_name": "Los Padres National Forest"},
        {"name": "Malibu", "state": "CA", "lat": 34.0259, "lon": -118.7798, "popularity": 85},
        {"name": "Yosemite", "state": "CA", "lat": 37.8651, "lon": -119.5383, "popularity": 88, "protected_name": "Yosemite National Park"},
        
        # Colorado
        {"name": "Rocky Mountain NP", "state": "CO", "lat": 40.3428, "lon": -105.6836, "popularity": 87, "protected_name": "Rocky Mountain National Park"},
        {"name": "Aspen", "state": "CO", "lat": 39.1911, "lon": -106.8175, "popularity": 80},
        
        # Arizona
        {"name": "Sedona", "state": "AZ", "lat": 34.8697, "lon": -111.7610, "popularity": 88, "protected_name": "Coconino National Forest"},
        {"name": "Grand Canyon", "state": "AZ", "lat": 36.1069, "lon": -112.1129, "popularity": 95, "protected_name": "Grand Canyon National Park"},
        
        # Montana
        {"name": "Glacier National Park", "state": "MT", "lat": 48.7596, "lon": -113.7870, "popularity": 85, "protected_name": "Glacier National Park"},
        
        # Wyoming
        {"name": "Yellowstone", "state": "WY", "lat": 44.4280, "lon": -110.5885, "popularity": 92, "protected_name": "Yellowstone National Park"},
        {"name": "Grand Teton", "state": "WY", "lat": 43.7904, "lon": -110.6818, "popularity": 85, "protected_name": "Grand Teton National Park"},
        
        # North Carolina
        {"name": "Blue Ridge Parkway", "state": "NC", "lat": 35.5951, "lon": -82.5515, "popularity": 82, "protected_name": "Blue Ridge Parkway"},
        {"name": "Tail of the Dragon", "state": "NC", "lat": 35.5159, "lon": -83.9293, "popularity": 90},
        
        # Texas
        {"name": "Austin Hill Country", "state": "TX", "lat": 30.2672, "lon": -98.7661, "popularity": 75},
        {"name": "Big Bend", "state": "TX", "lat": 29.1275, "lon": -103.2428, "popularity": 78, "protected_name": "Big Bend National Park"},
        
        # Oregon
        {"name": "Crater Lake", "state": "OR", "lat": 42.8684, "lon": -122.1685, "popularity": 80, "protected_name": "Crater Lake National Park"},
        
        # Washington
        {"name": "Mount Rainier", "state": "WA", "lat": 46.8523, "lon": -121.7603, "popularity": 82, "protected_name": "Mount Rainier National Park"},
    ]
    
    for p in places:
        hashtags = [p["name"].lower().replace(" ", ""), f"{p['state'].lower()}roadtrip"]
        if p.get("protected_name"):
            hashtags.extend(["publiclands", "nationalpark"])
        
        await conn.execute("""
            INSERT INTO trill_places (name, state, lat, lon, popularity_score, is_protected, protected_name, hashtags)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (name, state) DO UPDATE SET
                popularity_score = EXCLUDED.popularity_score,
                is_protected = EXCLUDED.is_protected,
                protected_name = EXCLUDED.protected_name,
                hashtags = EXCLUDED.hashtags,
                updated_at = NOW()
        """, p["name"], p["state"], p["lat"], p["lon"], p.get("popularity", 50), 
             bool(p.get("protected_name")), p.get("protected_name"), hashtags)

async def get_nearby_trill_place(conn, lat: float, lon: float, max_distance_km: float = 50) -> Optional[dict]:
    """Find nearest popular trill place for geo-targeting"""
    places = await conn.fetch("SELECT * FROM trill_places")
    
    from math import radians, sin, cos, sqrt, atan2
    
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371  # Earth radius in km
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
        return 2 * R * atan2(sqrt(a), sqrt(1-a))
    
    best = None
    min_dist = float('inf')
    
    for p in places:
        dist = haversine(lat, lon, float(p["lat"]), float(p["lon"]))
        if dist < min_dist and dist <= max_distance_km:
            min_dist = dist
            best = {**dict(p), "distance_km": dist}
    
    return best

# ============================================================
# Trill Processing Workflow
# ============================================================
async def process_telemetry(conn, upload_id: str, user_id: str, video_path: str, map_path: str, user_prefs: dict) -> dict:
    """
    Process telemetry data and generate content.
    
    Returns:
        {
            "trill_metadata": {...},
            "ai_content": {...},
            "hud_path": "/path/to/hud.mp4" or None
        }
    """
    if not tt:
        raise HTTPException(503, "Telemetry processing not available")
    
    try:
        # Run trill analysis
        result = tt.safe_analyze_video(
            video_path,
            map_path,
            gaz_places_path=GAZETTEER_PLACES_PATH if os.path.exists(GAZETTEER_PLACES_PATH) else None,
            padus_path=PADUS_PATH if os.path.exists(PADUS_PATH) else None,
            padus_layer=PADUS_LAYER,
            hud_enabled=user_prefs.get("trill_hud_enabled", False)
        )
        
        if not result.get("ok"):
            raise Exception(result.get("error", "Analysis failed"))
        
        trill_data = result["data"]
        
        # Check if score meets minimum threshold
        trill_score = trill_data.get("trill_score", 0)
        min_score = user_prefs.get("trill_min_score", 60)
        
        # Enrich with nearby trill place if available
        mid_lat = trill_data.get("place_lat")
        mid_lon = trill_data.get("place_lon")
        if mid_lat and mid_lon:
            trill_place = await get_nearby_trill_place(conn, mid_lat, mid_lon)
            if trill_place:
                trill_data["trill_place"] = trill_place["name"]
                trill_data["trill_place_hashtags"] = trill_place.get("hashtags", [])
        
        # Generate AI content if enabled and score is high enough
        ai_content = None
        if user_prefs.get("trill_ai_enhance", True) and trill_score >= min_score:
            try:
                ai_content = generate_trill_content(trill_data, user_prefs)
            except Exception as e:
                logging.error(f"AI generation failed: {e}")
                # Use trill's built-in generation as fallback
                ai_content = {
                    "title": trill_data.get("title"),
                    "caption": trill_data.get("caption"),
                    "hashtags": trill_data.get("hashtags", []),
                    "model": "trill_fallback"
                }
        
        # Generate HUD if enabled
        hud_path = None
        if user_prefs.get("trill_hud_enabled", False):
            try:
                hud_path = tt.ensure_hud_mp4(video_path, map_path)
            except Exception as e:
                logging.error(f"HUD generation failed: {e}")
        
        # Store in database
        await conn.execute("""
            UPDATE uploads SET 
                trill_score = $1,
                speed_bucket = $2,
                trill_metadata = $3,
                ai_generated_title = $4,
                ai_generated_caption = $5,
                ai_generated_hashtags = $6,
                updated_at = NOW()
            WHERE id = $7
        """, 
            trill_score,
            trill_data.get("speed_bucket"),
            json.dumps(trill_data),
            ai_content.get("title") if ai_content else None,
            ai_content.get("caption") if ai_content else None,
            ai_content.get("hashtags") if ai_content else None,
            upload_id
        )
        
        # Track OpenAI costs if used
        if ai_content and ai_content.get("tokens_used"):
            cost = ai_content["tokens_used"] * 0.00001  # Adjust based on model
            await conn.execute("""
                INSERT INTO cost_tracking (user_id, category, operation, tokens, cost_usd)
                VALUES ($1, 'openai', 'trill_generation', $2, $3)
            """, user_id, ai_content["tokens_used"], cost)
        
        return {
            "trill_metadata": trill_data,
            "ai_content": ai_content,
            "hud_path": hud_path
        }
        
    except Exception as e:
        logging.error(f"Telemetry processing failed: {e}")
        raise HTTPException(500, f"Telemetry processing failed: {str(e)}")

# ============================================================
# API ENDPOINTS (add to your FastAPI app)
# ============================================================

"""
# Add these imports to main.py
from trill_integration import (
    process_telemetry,
    generate_trill_content,
    get_nearby_trill_place,
    seed_trill_places,
    TRILL_MIGRATIONS
)

# Add to run_migrations() function
for version, sql in TRILL_MIGRATIONS:
    if version not in applied:
        try:
            await conn.execute(sql)
            await conn.execute("INSERT INTO schema_migrations (version) VALUES ($1)", version)
            logger.info(f"Trill migration v{version} applied")
        except Exception as e:
            logger.error(f"Trill migration v{version} failed: {e}")

# Add to lifespan startup
await seed_trill_places(db_pool)

# Add these endpoints:

@app.post("/api/trill/analyze/{upload_id}")
async def analyze_telemetry(upload_id: str, user: dict = Depends(get_current_user)):
    \"""Manually trigger trill analysis on an upload with telemetry data\"""
    async with db_pool.acquire() as conn:
        upload = await conn.fetchrow(
            "SELECT * FROM uploads WHERE id = $1 AND user_id = $2",
            upload_id, user["id"]
        )
        
        if not upload:
            raise HTTPException(404, "Upload not found")
        
        if not upload.get("telemetry_r2_key"):
            raise HTTPException(400, "No telemetry data for this upload")
        
        # Download files from R2
        s3 = get_s3_client()
        
        video_obj = s3.get_object(Bucket=R2_BUCKET_NAME, Key=upload["r2_key"])
        video_data = video_obj["Body"].read()
        
        telem_obj = s3.get_object(Bucket=R2_BUCKET_NAME, Key=upload["telemetry_r2_key"])
        telem_data = telem_obj["Body"].read()
        
        # Write to temp files
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as vf:
            vf.write(video_data)
            video_path = vf.name
        
        with tempfile.NamedTemporaryFile(suffix=".map", delete=False) as tf:
            tf.write(telem_data)
            map_path = tf.name
        
        try:
            # Get user preferences
            prefs = await conn.fetchrow(
                "SELECT * FROM user_preferences WHERE user_id = $1",
                user["id"]
            )
            user_prefs = dict(prefs) if prefs else {}
            
            # Process
            result = await process_telemetry(
                conn, upload_id, user["id"],
                video_path, map_path, user_prefs
            )
            
            return {
                "success": True,
                "trill_score": result["trill_metadata"].get("trill_score"),
                "speed_bucket": result["trill_metadata"].get("speed_bucket"),
                "ai_enhanced": bool(result.get("ai_content")),
                "hud_generated": bool(result.get("hud_path"))
            }
        finally:
            os.unlink(video_path)
            os.unlink(map_path)


@app.get("/api/trill/places")
async def get_trill_places(
    state: Optional[str] = Query(None),
    limit: int = Query(20, le=100),
    user: dict = Depends(get_current_user)
):
    \"""Get popular trill places for targeting\"""
    async with db_pool.acquire() as conn:
        if state:
            places = await conn.fetch(
                "SELECT * FROM trill_places WHERE state = $1 ORDER BY popularity_score DESC LIMIT $2",
                state.upper(), limit
            )
        else:
            places = await conn.fetch(
                "SELECT * FROM trill_places ORDER BY popularity_score DESC LIMIT $1",
                limit
            )
    
    return [dict(p) for p in places]


@app.post("/api/trill/generate-preview")
async def generate_trill_preview(
    data: dict = Body(...),
    user: dict = Depends(get_current_user)
):
    \"""Preview AI-generated content without saving\"""
    trill_metadata = data.get("trill_metadata", {})
    
    # Get user preferences
    async with db_pool.acquire() as conn:
        prefs = await conn.fetchrow(
            "SELECT * FROM user_preferences WHERE user_id = $1",
            user["id"]
        )
        user_prefs = dict(prefs) if prefs else {}
    
    result = generate_trill_content(trill_metadata, user_prefs)
    return result
"""

# ============================================================
# USAGE EXAMPLE
# ============================================================
"""
# In your /api/uploads/{upload_id}/complete endpoint:

async def complete_upload(upload_id: str, user: dict = Depends(get_current_user)):
    async with db_pool.acquire() as conn:
        upload = await conn.fetchrow(...)
        
        # Get user preferences
        prefs = await conn.fetchrow(
            "SELECT * FROM user_preferences WHERE user_id = $1",
            user["id"]
        )
        user_prefs = dict(prefs) if prefs else {}
        
        # Process telemetry if enabled and .map file present
        if (user_prefs.get("trill_enabled") and 
            upload.get("telemetry_r2_key")):
            
            try:
                # Download files from R2
                s3 = get_s3_client()
                video_data = s3.get_object(...)
                map_data = s3.get_object(...)
                
                # Process
                result = await process_telemetry(
                    conn, upload_id, user["id"],
                    video_path, map_path, user_prefs
                )
                
                # Use AI-generated content if available
                if result.get("ai_content"):
                    upload_metadata["title"] = result["ai_content"]["title"]
                    upload_metadata["caption"] = result["ai_content"]["caption"]
                    upload_metadata["hashtags"] = result["ai_content"]["hashtags"]
                
                # Use HUD video if generated
                if result.get("hud_path"):
                    video_path = result["hud_path"]
                    
            except Exception as e:
                logger.error(f"Trill processing failed: {e}")
                # Continue with original content
        
        # Continue with platform uploads...
"""
