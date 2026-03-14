# Caption & AI Settings — Logic and Mapping

## Why Settings Don't Save

**Root cause**: The settings page uses `GET /api/settings/preferences` and `POST /api/settings/preferences`, which read/write the `user_preferences` table. **Caption Style, Caption Tone, Caption Voice, and Caption Frame Count are NOT in that flow** — they are only handled by `PUT /api/me/preferences`, which writes to `users.preferences` JSONB.

The **worker** reads caption fields from `users.preferences` (via `db.load_user_settings`). So if the frontend saves via POST /api/settings/preferences, caption fields never reach `users.preferences` and the worker falls back to defaults.

---

## Caption Style

**Purpose**: Controls how the AI structures the caption (length and format).

| Value   | Caption Length / Instruction |
|---------|------------------------------|
| `story` | 150–280 characters — tell a narrative arc from start to finish. Uses multi-frame data for best narrative. |
| `punchy`| Under 120 characters — hook in the first 3 words. Short, scroll-stopping. |
| `factual`| 100–200 characters — lead with the most impressive stat. Data-first. |

**Backend**: `stages/caption_stage.py` → `_build_narrative_prompt()` → `caption_length` dict (line ~762).

**Storage key**: `captionStyle` (camelCase) / `caption_style` (snake_case)

---

## Caption Tone

**Purpose**: Controls the AI's emotional register — how the caption *feels*.

| Value       | Instruction |
|-------------|-------------|
| `hype`      | Energy is HIGH. Power words, exclamation, urgency. Stop the scroll. |
| `calm`      | Measured and confident. Let the footage speak. Understated cool. |
| `cinematic` | Poetic, atmospheric. Paint a picture with words. Film trailer voiceover. |
| `authentic` | Real talk, first-person, no fluff. Like texting a friend. *(default)* |

**Backend**: `stages/caption_stage.py` → `tone_instruction` dict (line ~777). User tone **overrides** the category default tone.

**Storage key**: `captionTone` / `caption_tone`

---

## Caption Voice / Persona

**Purpose**: Higher-level personality layered on top of tone. Defines *who* is speaking.

| Value               | Description |
|---------------------|-------------|
| `default`           | Balanced, high-signal storytelling. Confident but not shouty. |
| `mentor`            | Supportive, experienced mentor. Speaks as a guide, clear takeaways, calm confidence. |
| `hypebeast`         | Max energy, hype, slang-forward. Short punchy sentences, built to spike excitement. |
| `best_friend`       | Warm, honest, slightly chaotic. First-person, casual, occasionally self-deprecating, relatable. |
| `teacher`           | Clear, structured educator. Breaks ideas into steps, avoids fluff, one key insight per caption. |
| `cinematic_narrator`| Film-trailer narrator. Descriptive, visual, dramatic, like a voiceover. |

**Backend**: `stages/caption_stage.py` → `VOICE_PROFILES` dict (line ~486). Injected as `VOICE PROFILE: {voice_key} — {voice_instruction}` in the prompt.

**Storage key**: `captionVoice` / `caption_voice`

---

## AI Caption Scan Depth (Caption Frame Count)

**Purpose**: How many video frames to send to GPT for caption generation. More frames = better narrative understanding.

- **Range**: 2–12 (clamped by tier `max_caption_frames`)
- **Default**: 6 (or tier max)
- **Logic**: `caption_stage` extracts N frames across the video, sends them to GPT vision. Story mode benefits most from higher N.

**Backend**: `stages/caption_stage.py` → `num_frames = max(2, min(user_frame_count, max_caption_frames), 12)`

**Storage key**: `captionFrameCount` / `caption_frame_count`

---

## Data Flow

```
Frontend (Settings Page)
    │
    ├─► GET /api/settings/preferences  → user_preferences table (no caption fields)
    │   OR
    │   GET /api/me/preferences        → users.preferences JSONB (has caption fields)
    │
    └─► POST /api/settings/preferences → user_preferences table (no caption fields)  ❌
        OR
        PUT /api/me/preferences        → users.preferences JSONB (has caption fields) ✅

Worker (caption_stage)
    │
    └─► db.load_user_settings() → merges user_settings + user_preferences + users.preferences
        users.preferences WINS for caption fields (priority 3)
```

**Fix applied**: GET /api/settings/preferences now overlays `users.preferences` for caption fields. POST /api/settings/preferences now syncs caption fields to `users.preferences` when present in the payload. The worker continues to read from `users.preferences` via `load_user_settings`.
