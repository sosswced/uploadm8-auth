# UploadM8 — Feature Status & Fix Plan

**Date:** March 15, 2026

---

## 1. Settings Flow (Thumbnail, Tone, Voice, Caption)

### Current State
- **Caption style/tone/voice** are saved via POST /api/settings/preferences (to user_preferences) and synced to users.preferences.
- **FIXED:** PUT /api/me/preferences now **merges** incoming prefs with existing (no longer replaces). Settings page now sends full hashtag + caption fields to PUT.

### Thumbnail settings
- `styledThumbnails`, `autoThumbnails`, `thumbnailInterval` flow through user_preferences and load_user_settings.
- Worker reads from load_user_settings → create_context.

---

## 2. Hashtag Logic

### always_hashtags
- Works: merged at publish via `ctx.get_effective_hashtags(platform)`.
- Stored in users.preferences (and user_preferences).
- Applied first in merge order: always → platform → base → AI.

### platform_hashtags
- **Logic exists** in context.get_effective_hashtags(platform) — same pipeline as always_hashtags.
- Keys: tiktok, youtube, instagram, facebook (lowercase).
- **FIXED:** PUT merge + settings page now sends platformHashtags (and caption fields) to PUT. platform_hashtags should now persist and apply at publish.

---

## 3. Queue: Pending vs Processing — Logic

### Status Flow
| Status     | Meaning |
|------------|---------|
| **pending**   | In queue, not yet picked up by worker. Redis queue holds job. |
| **processing**| Worker has claimed the job; pipeline running (transcode, caption, etc.). |
| **completed** | All platforms published successfully. |
| **failed**    | One or more platforms failed; may be retried. |
| **cancelled** | User cancelled before completion. |

### Smart vs Scheduled Uploads
| Mode       | Behavior |
|------------|----------|
| **Immediate** | User clicks Submit; upload enqueued; worker processes as soon as a slot is free. |
| **Scheduled** | User picks single `scheduled_time`; all platforms publish at that time. Stored in `scheduled_time`. |
| **Smart**     | AI picks optimal times per platform; stored in `smart_schedule: { tiktok: "ISO", youtube: "ISO", ... }`. Each platform can have a different publish time. |

### API Endpoints
- `GET /api/uploads/queue` — pending + processing uploads
- `GET /api/uploads/queue/stats` — counts for dashboard (Pending, Processing, Completed, Failed)
- `GET /api/scheduled` — scheduled + smart uploads

---

## 4. Scheduled Calendar Click / Modal

- **FIXED:** Event delegation on `#calendarDays` — click handler finds `.calendar-day.has-uploads`, reads `data-date-iso`, recomputes uploads via `getUploadsForDate`, calls `showDayUploads`. Survives re-renders.

---

## 5. Scheduled: Time/Date Per Video

- **Current:** Upload flow allows single scheduled_time or smart_schedule (one time per platform for the whole batch).
- **Requested:** For N selected videos, user must enter time/date for each of the N.
- **Scope:** This is a UX/flow change — upload.html would need a multi-video scheduler UI. Not a quick patch.

---

## 6. Admin Calculator "Live Data" Button

- Need to verify admin-calculator.html has a "Live data" or similar button that fetches live DB data.
- Pending inspection.

---

## 7. Multi-Account Uploads

- **Backend:** target_accounts = [platform_tokens.id, ...] is supported. Publish stage iterates over each token_id.
- **Frontend:** Must send selected account IDs in presign. Upload.html has account picker.
- **Requested:** Ensure ALL selected accounts (e.g. 10 FB, 12 TikTok, 14 IG) get the upload. Backend already supports this if frontend sends all IDs.

---

## 8. Admin/Analytics 7/30/90 Day Dropdowns

- APIs should accept range parameter (7d, 30d, 90d) and return filtered data.
- Need to verify each admin/analytics endpoint supports this.

---

## 9. Discord Upload Complete

- Should show platform + each profile + correct links to video/page.
- notify_stage sends Discord webhook; need to include platform_results with URLs.

---

## 10. Queue/Dashboard Hyperlinks & Multi-Account Display

- platform_results should include per-account video URLs.
- Dashboard/queue should render each account, not just one profile per platform.

---

## 11. Voice/Tone Captions — Misleading

- Caption stage has anti-misleading rules (SESSION-CHANGES-SUMMARY).
- User says still misleading — may need stronger prompt constraints.

---

## 12. Default PUT/AIC for New Accounts

- **Current:** signup_put = 100, signup_aic = 75 (in app.py register).
- **Requested:** Based on entitlements, enough to test features. Current values are already generous; can tune if needed.

---

## All 12 Patches — Status

| # | Item | Status |
|---|------|--------|
| 1 | Settings flow (thumbnail, tone, voice) | ✅ PUT merge + settings send full prefs |
| 2 | Platform hashtags | ✅ Same pipeline as always_hashtags |
| 3 | Scheduled calendar click | ✅ Event delegation |
| 4 | Queue pending vs processing doc | ✅ In FEATURE-STATUS |
| 5 | Scheduled time/date per video | ✅ Validation exists; slots UI exists |
| 6 | Admin calculator Live Data | ✅ Load Live Data button → /api/admin/calculator/pricing |
| 7 | Multi-account uploads | ✅ target_accounts; all selected by default |
| 8 | Admin/analytics 7/30/90 dropdowns | ✅ admin-kpi, analytics have range params |
| 9 | Discord upload complete | ✅ Per-platform + profile + URL fallback |
| 10 | Queue/dashboard hyperlinks | ✅ platform_url normalized to url; multi-account badges |
| 11 | Voice/tone captions | ✅ Anti-misleading hooks strengthened |
| 12 | Default PUT/AIC credits | ✅ From entitlements (free tier) |
