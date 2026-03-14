# Content Generation Architecture — Titles, Captions, Thumbnails

This document maps how titles, captions, and thumbnails are generated, where settings/preferences flow from, and how to reduce misleading output.

---

## 1. Settings & Preferences Flow

### Storage Sources (merged by `stages/db.py` → `load_user_settings`)

| Source | Priority | Keys Used for Content Generation |
|-------|----------|----------------------------------|
| `user_settings` table | 1 | `styled_thumbnails`, `auto_thumbnails`, `auto_captions`, `thumbnail_interval`, etc. |
| `user_preferences` table | 2 | Same keys, merged if `user_settings` exists |
| `users.preferences` JSONB | 3 (wins) | **Caption**: `captionStyle`, `captionTone`, `captionVoice`, `captionFrameCount`<br>**AI**: `autoCaptions`, `autoThumbnails`, `aiHashtagsEnabled`, `aiHashtagCount`, `aiHashtagStyle`<br>**Hashtags**: `alwaysHashtags`, `blockedHashtags`, `platformHashtags`, `maxHashtags` |

### API Endpoints

- **Save**: `PUT /api/me/preferences` — validates and stores to `users.preferences`
- **Load**: Worker calls `db.load_user_settings(pool, user_id)` → merged dict → `ctx.user_settings`

### Caption & AI Settings (from `PreferencesUpdate` / `app.py`)

| Key | Values | Default | Used By |
|-----|--------|---------|---------|
| `captionStyle` | `story` \| `punchy` \| `factual` | `story` | caption_stage — prompt length/style |
| `captionTone` | `hype` \| `calm` \| `cinematic` \| `authentic` | `authentic` | caption_stage — tone directive |
| `captionVoice` | `default` \| `mentor` \| `hypebeast` \| `best_friend` \| `teacher` \| `cinematic_narrator` | `default` | caption_stage — voice profile |
| `captionFrameCount` | 2–12 | 6 (or tier max) | caption_stage — frames sent to GPT |
| `autoCaptions` | bool | — | caption_stage — whether to generate |
| `autoThumbnails` | bool | — | thumbnail_stage — whether to run |
| `aiHashtagsEnabled` | bool | — | caption_stage — whether to generate hashtags |
| `aiHashtagCount` | int | 15 | caption_stage — max hashtags |
| `aiHashtagStyle` | `trending` \| `niche` \| `mixed` | `mixed` | caption_stage — hashtag style |

---

## 2. Title & Caption Generation

**File**: `stages/caption_stage.py`

### Inputs

- `ctx.filename`, `ctx.title`, `ctx.caption` (user hints)
- `ctx.user_settings` (style, tone, voice, frame count)
- `ctx.platforms`, `ctx.location_name`
- `ctx.telemetry_data` / `ctx.trill_score` (Trill integration)
- `ctx.caption_memory_examples` (few-shot from past uploads)

### Content Category Detection (3-layer)

1. **Layer 1**: Keyword scan of `ctx.caption` and `ctx.title`
2. **Layer 2**: Keyword scan of `ctx.filename`
3. **Layer 3**: Falls back to `general`; GPT vision confirms from frames

Categories: `automotive`, `beauty`, `food`, `home_renovation`, `gardening`, `fitness`, `fashion`, `gaming`, `travel`, `pets`, `education`, `comedy`, `tech`, `music`, `real_estate`, `sports`, `asmr`, `lifestyle`, `general`

### Per-Category Data (can encourage misleading output)

Each category has:

- **tone**: Default tone guidance
- **hook_templates**: Example hooks — AI "adapts, not copies"
- **hashtag_seeds**: Starting vocabulary

**Examples of potentially misleading hooks:**

- `general`: "You need to see this", "Nobody expected this outcome"
- `education`: "Nobody teaches this in school — here's what you need to know", "The 1% of people who know this will get ahead"
- `beauty`: "The secret to glowy skin nobody tells you"
- `real_estate`: "The neighbourhood nobody is talking about yet"
- `home_renovation`: "Nobody believed this would work. I proved them wrong."

### Tone Override (user pref)

| `captionTone` | Instruction |
|--------------|-------------|
| `hype` | "Energy is HIGH. Power words, exclamation, urgency. Stop the scroll." |
| `calm` | "Measured and confident. Let the footage speak. Understated cool." |
| `cinematic` | "Poetic, atmospheric. Paint a picture with words. Film trailer voiceover." |
| `authentic` | "Real talk, first-person, no fluff. Like texting a friend." |

### Existing Anti-Misleading Rules (in prompt)

- "NEVER invent events, locations, or narratives not shown in the video. Base caption ONLY on what is visible in the frames."
- "Content must feel AUTHENTIC — not AI-generated"
- "Be SPECIFIC to what is actually visible — generic content gets buried"

### Gaps

- Hook templates and tone instructions can still push toward sensational language.
- No explicit "do not use clickbait" or "accuracy over engagement" constraint.
- `hype` tone + certain hooks can combine into misleading framing.

---

## 3. Thumbnail Generation

**Files**: `stages/thumbnail_stage.py`, `stages/context.py`

### Frame Selection

1. Extract N frames (tier `max_thumbnails`) across video duration
2. Score each with FFmpeg blurdetect (sharpness)
3. Detect category (same 3-layer as caption)
4. AI selection: GPT-4o-mini picks "best" frame using category-specific criteria
5. Prompt includes: "click-through rate potential", "category-specific quality"

### Thumbnail Brief (styled thumbnails)

When `can_custom_thumbnails` and `styledThumbnails` enabled:

- `THUMBNAIL_BRIEF_PROMPT` in `stages/context.py`
- Variables: `effective_title`, `effective_caption`, `category`, `location_name`, `trill_bucket`, `max_speed_mph`, `platforms_csv`
- Outputs: `selected_headline`, `headline_options`, `badge_text`, `badge_style`, `directional_element`, `props`, `emotion_cue`, `color_mood`, `platform_plan`

### Thumbnail Brief Rules

- "No profanity, no hate, no nudity, no weapons emphasis, no illegal claims"
- "No copyrighted logos/brand marks"
- "Text: 2–6 words total, ALL CAPS"
- "Always include 1 badge (e.g., NEW, FAST, HOW TO, TOP 5) if appropriate"
- **No explicit "no misleading" or "accuracy" rule**

### Gaps

- Badge text (NEW, FAST, TOP 5) can be misleading if not accurate.
- `emotion_cue` (shocked, excited) can encourage over-the-top framing.
- Headlines are optimized for CTR, not accuracy.

---

## 4. Thumbnail Upload Flow

1. **thumbnail_stage**: Sets `ctx.thumbnail_path` (and `platform_thumbnail_map` for styled)
2. **worker.py** (after thumbnail stage): Uploads to R2 at `thumbnails/{user_id}/{upload_id}/thumbnail.jpg`
3. **worker.py**: Updates `uploads.thumbnail_r2_key`
4. **app.py** `GET /api/uploads/{id}`: Returns presigned URL for `thumbnail_r2_key`

---

## 5. Recommended Changes to Reduce Misleading Output

### Caption Stage

1. **Add explicit anti-misleading rule** to `_build_narrative_prompt`:
   - "Do NOT use clickbait patterns ('Nobody expected', 'You need to see this', 'The secret nobody tells you'). Describe what is actually shown. Accuracy over engagement."

2. **Soften hook templates** in `CONTENT_CATEGORIES`:
   - Replace "Nobody expected this outcome" with more neutral options
   - Add "ACCURACY: Hooks must reflect what is actually visible. Never overpromise."

3. **Add user preference** `contentAccuracy` or `avoidClickbait`:
   - When `true`, inject stronger accuracy-first language and avoid hype-heavy hooks

### Thumbnail Brief

1. **Add to HARD RULES** in `THUMBNAIL_BRIEF_PROMPT`:
   - "Headlines and badges must accurately reflect the video content. No misleading claims (e.g. 'TOP 5' when it's not a list, 'NEW' when it's not new)."

2. **Constrain badge_text**:
   - Only suggest badges that match the content (e.g. "FAST" only if speed/telemetry present)

### DB / API

1. **Optional**: Add `contentAccuracy` or `avoidClickbait` to `PreferencesUpdate` and `FIELD_MAP` so users can opt into stricter accuracy mode.

---

## 6. File Reference

| File | Purpose |
|------|---------|
| `stages/caption_stage.py` | Title, caption, hashtag generation; CONTENT_CATEGORIES, VOICE_PROFILES |
| `stages/thumbnail_stage.py` | Frame extraction, AI selection, thumbnail brief, template/AI edit render |
| `stages/context.py` | JobContext, THUMBNAIL_BRIEF_PROMPT, get_thumbnail_brief_vars |
| `stages/db.py` | load_user_settings, FIELD_MAP (camelCase ↔ snake_case) |
| `app.py` | PreferencesUpdate, PUT /api/me/preferences, validation |
| `worker.py` | Pipeline orchestration, thumbnail R2 upload |
