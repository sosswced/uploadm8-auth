# UploadM8 — All 12 Patches Summary

**Date:** March 15, 2026  
**Purpose:** Complete implementation of all 12 feature/break fixes for local testing.

---

## Patches Applied

### 1. Settings Flow (Thumbnail, Tone, Voice, Caption)
- **app.py:** `PUT /api/me/preferences` now **merges** with existing (no longer replaces).
- **settings.html:** Sends full prefs to PUT (platformHashtags, alwaysHashtags, blockedHashtags, captionStyle, captionTone, captionVoice, captionFrameCount).

### 2. Platform Hashtags
- Same pipeline as always_hashtags in `context.get_effective_hashtags(platform)`.
- Fixed by PUT merge — platform_hashtags now persist in users.preferences.

### 3. Scheduled Calendar Click
- **scheduled.html:** Event delegation on `#calendarDays` for `.calendar-day.has-uploads` clicks.
- Uses `data-date-iso`; recomputes uploads via `getUploadsForDate`; calls `showDayUploads`.

### 4. Queue Pending vs Processing
- Documented in FEATURE-STATUS-AND-FIXES.md.
- Pending = queued; Processing = worker running; Completed/Failed = done.

### 5. Scheduled Time/Date Per Video
- **Already implemented:** `renderScheduleSlots()` shows datetime-local per file when mode=scheduled.
- Validation in `startUpload()` blocks submit until all files have `scheduledTime`.

### 6. Admin Calculator Live Data
- **Already implemented:** "Load Live Data" button calls `fetchPricingAndRun()` → `GET /api/admin/calculator/pricing`.
- Backend returns live tier/topup data from entitlements.

### 7. Multi-Account Uploads
- **Already implemented:** `resolveTargetAccountIds()` collects all selected accounts (or group members).
- By default all accounts selected; user can uncheck. `target_accounts` sent in presign/complete.
- Backend publish stage iterates over each token_id.

### 8. Admin/Analytics 7/30/90 Day Dropdowns
- **admin-kpi.html:** Time range select with 7d, 30d, 90d, 6m, 1y.
- **analytics.html:** `dateRangeSelect` with 7, 30, 90, 365, all.
- Backend APIs accept `range=` param (7d, 30d, 90d, etc.).

### 9. Discord Upload Complete
- **notify_stage.py:** Per-platform results with account label and `[View Post](url)`.
- Added fallback URL build from `platform_video_id` when `platform_url` missing (TikTok, YouTube, Facebook).

### 10. Queue/Dashboard Hyperlinks + Multi-Account
- **app.py:** `_normalize_platform_results` aliases `platform_url` → `url`.
- **queue.html:** `resolvePlatformUrl()` checks url, platform_url; shows account name in badge.
- Multi-account badges: `TikTok (@handle)`, `YouTube (channel)` etc.

### 11. Voice/Tone Captions — Anti-Misleading
- **caption_stage.py:** Replaced misleading hook templates:
  - Sports: "Nobody thought we'd make it" → "Here's the play that got us to the finals"
  - ASMR: "The most satisfying... you'll watch" → "60 seconds of satisfying sounds"
  - Gaming: "Nobody expects this strat" → "This strat worked in ranked"
  - Lifestyle: "changed my life" → "what I noticed" / "what works for me"

### 12. Default PUT/AIC Credits
- **app.py:** Signup credits from `get_entitlements_for_tier("free")`:
  - `signup_put = max(ent.put_monthly, 80)`  → 80
  - `signup_aic = max(ent.aic_monthly, 50)` → 50

---

## Files Changed

| Repo | File | Changes |
|------|------|---------|
| uploadm8-auth | app.py | PUT merge; signup credits from entitlements |
| uploadm8-auth | stages/caption_stage.py | Anti-misleading hook templates |
| uploadm8-auth | stages/notify_stage.py | Discord URL fallback from platform_video_id |
| uploadm8-frontend | settings.html | Full prefs to PUT |
| uploadm8-frontend | scheduled.html | Event delegation for calendar click |

---

## Local Testing

1. **Auth:** `cd uploadm8-auth && uvicorn app:app --reload --host 127.0.0.1 --port 8000`
2. **Frontend:** Serve `uploadm8-frontend` (e.g. `npx serve` or open via file).
3. **Worker:** Run worker for upload processing (if testing full flow).

### Test Checklist
- [ ] Settings: Save caption style/tone/voice; verify they persist.
- [ ] Settings: Save platform hashtags; run upload; verify per-platform hashtags.
- [ ] Scheduled: Click calendar day with uploads; modal opens.
- [ ] Upload: Scheduled mode; add 3 files; set date/time for each; submit.
- [ ] Admin calculator: Load Live Data; values update.
- [ ] Upload: Select 2+ accounts per platform; verify target_accounts sent.
- [ ] Admin KPI: Change range to 7d, 30d, 90d; data updates.
- [ ] Discord: Complete upload; webhook shows platform + profile + link.
- [ ] Queue: Completed upload shows clickable badges with correct URLs.
- [ ] New signup: User gets 80 PUT, 50 AIC.
