# UploadM8 Feature Analysis & Roadmap

This document answers your questions about current logic and outlines required changes for each feature area.

---

## 1. Queue: Pending vs Processing Logic

| Status | Meaning | Where Defined |
|--------|---------|---------------|
| **pending** | Job created, file uploaded, waiting to enter the Redis queue | `queue.html` line 827: `['pending','queued'].includes(u.status)` |
| **queued** | Job is in Redis queue, waiting for a worker slot | Same filter |
| **processing** | Worker is actively transcoding, running AI, compositing HUD | `queue.html` line 828: `u.status === 'processing'` |
| **ready_to_publish** | All processing done, waiting for scheduled publish time | Scheduled/smart jobs |
| **completed / succeeded** | Published successfully | Terminal |

**Flow:** `staged` → `pending`/`queued` → `processing` → `ready_to_publish` (if scheduled) → `publishing` → `succeeded`/`partial`/`failed`

---

## 2. Smart Uploads vs Scheduled Uploads

| Mode | Behavior | Current UI |
|------|-----------|------------|
| **Publish Now (immediate)** | Upload immediately to all platforms | No date picker |
| **Schedule** | Single date/time for **all** selected videos | One `datetime-local` input; same time for every video |
| **Smart Schedule** | AI picks optimal times per platform, spread across a window | Days input (7–730); different times per platform |

**Current limitation:** Scheduled mode uses **one** date/time for the entire batch. If you select 10 videos, all 10 publish at the same moment.

**Your requirement:** User should enter time/date slots for **each** video (1, 10, or 100 videos = 1, 10, or 100 slots).

---

## 3. Scheduled Not Clickable / No Modals

**Current behavior:**
- The "Schedule" option in `upload.html` (line 333) is a `<label>` with `data-mode="scheduled"`.
- Click handler in `setupScheduleOptions()` (line 860) toggles `scheduleDateTime` visibility and shows a single `datetime-local` input.
- There is **no modal** — the date picker appears inline below the schedule options.

**Possible issues:**
1. **CSS:** `.schedule-option input[type="radio"] { display: none; }` — the label should still be clickable.
2. **Overlap:** Another element may be covering the schedule options.
3. **Expectation mismatch:** You may expect a modal/popup for per-video scheduling; that does not exist yet.

**Recommendation:** Add instrumentation to confirm whether the Schedule option receives clicks and whether `scheduleDateTime` is shown. If the issue is "no per-video modal," that requires new UI (see Section 2).

---

## 4. Platform Hashtags vs Always Hashtags

**Always hashtags (works):**
- Stored in `user_settings` as `always_hashtags` / `alwaysHashtags`.
- Applied to **every** upload on **every** platform.
- Frontend: `upload.html` lines 648–663, `renderPreferencesNotice()`.

**Platform hashtags (broken):**
- Stored as `platform_hashtags` / `platformHashtags`: `{ tiktok: [...], youtube: [...], instagram: [...], facebook: [...] }`.
- Frontend preview: `updatePlatformHashtagPreview()` (line 2073) shows them in the AI pref box.
- **Backend:** The frontend sends prefs to `/api/uploads/{id}/complete` and settings APIs. The backend caption/hashtag pipeline must merge:
  - `always_hashtags` (always)
  - `platform_hashtags[platform]` (per platform)

**Fix:** Update backend hashtag logic to mirror `always_hashtags`:
- For each platform, final hashtags = `always_hashtags` + `platform_hashtags[platform]` (+ AI-generated, respecting max_hashtags and blocked_hashtags).

---

## 5. Admin Calculator "Live Data" Button

**Current behavior:**
- On page load, `fetchPricingAndRun()` calls `/api/admin/calculator/pricing` and prefills inputs.
- Presets (Bootstrap, Growth, Scale, Enterprise) overwrite with static values.
- There is **no** "Live Data" button to revert to or refresh live DB data.

**Your requirement:** Add a "Live Data" button that:
1. Calls the live pricing API again.
2. Refreshes inputs with current DB values.
3. Re-runs the calculator.

**Implementation:** Add a button next to Export/Import that calls `fetchPricingAndRun()`.

---

## 6. Multi-Account Uploads for All Platforms

**Current logic (`upload.html` line 1819, `resolveTargetAccountIds`):**
- Returns `[]` when **no group** is selected.
- Backend then uses "one per platform" (most recent token).
- Only when a **group** is selected does it return `[token_id, ...]` for all accounts in the group.

**Your requirement:** If the user has 10 Facebook, 12 TikTok, 14 Instagram accounts:
- When platforms are selected, load **all** account tokens from DB for those platforms.
- Upload to every account, not just one per platform.

**Options:**
1. **Default to all accounts:** When no group is selected, `resolveTargetAccountIds` returns all connected account token IDs for the selected platforms.
2. **Account picker:** Add UI to select which accounts per platform (checkboxes or multi-select).
3. **Group-only:** Keep current behavior but make groups the primary way to target multiple accounts.

**Backend:** Must accept `target_accounts` and publish to each token; APIs must load tokens by ID.

---

## 7. Admin Pages & Analytics: 7/30/90 Day Dropdowns

**Current state:** Admin pages and analytics may have dropdowns for "Last 7 days," "Last 30 days," "Last 90 days" but:
- APIs may not accept date-range parameters.
- Backend may not filter by date range.

**Requirement:** Ensure:
1. Frontend dropdowns pass `?days=7`, `?days=30`, `?days=90` (or equivalent) to APIs.
2. Backend endpoints filter data by date range and return correct aggregates.

---

## 8. Discord Upload Complete: Platform + Profile + Links

**Current state:** Discord webhooks are configured in Settings and Admin. The upload-complete webhook payload needs to include:
- Platform name
- Each profile/account uploaded to
- Correct links to each video/post

**Requirement:** Backend webhook payload should list each `platform_results` entry with:
- `platform`, `account_name`/`username`, `url` (or constructed URL).

---

## 9. Dashboard/Queue: Multiple Accounts & Correct Hyperlinks

**Current behavior:**
- `platform_results` is a list: `[{ platform, success, video_id, ... }]`.
- `renderPlatformResults()` in `queue.html` iterates over this list and shows one badge per entry.
- If the backend returns **one** entry per platform (not per account), only one profile shows even when uploaded to two accounts.

**Requirement:**
1. **Backend:** When publishing to multiple accounts (e.g., 2 TikTok accounts), return **one** `platform_results` entry per account with `account_id`, `username`, `url`, etc.
2. **Frontend:** Display each account separately with correct links.
3. **URL logic:** `resolvePlatformUrl()` in `queue.html` already handles `video_id`, `shortcode`, etc. Ensure backend populates these fields per account.

---

## 10. Voice and Tone for Dynamic, Accurate Captions

**Current settings (`settings.html`):**
- `captionStyle` (e.g., story, list)
- `captionTone` (e.g., authentic, hype)
- `captionVoice` (e.g., default, mentor, hypebeast, best_friend, teacher, cinematic_narrator)

**Stored as:** `captionStyle`, `captionTone`, `captionVoice` in user preferences.

**Requirement:** Backend AI caption pipeline must:
1. Receive these values from user settings.
2. Use them in prompts so captions match the chosen voice/tone.
3. Avoid "fake stories" or misleading content — ensure prompts emphasize accuracy and user’s chosen persona.

**Frontend:** Settings already save these; verify they are sent with upload/complete requests and that the backend reads them.

---

## 11. Default PUT and AIC Credits for New Accounts

**Requirement:** New accounts should start with default PUT and AIC credits so users can:
- Upload and try features.
- Test auto-uploads.
- Based on entitlements, allocate appropriate starting credits.

**Implementation:** Backend/entitlements logic:
- On account creation or first login, grant default PUT and AIC based on plan/tier.
- Document in entitlements schema.

---

## Summary: Priority Order

| # | Item | Layer | Effort |
|---|------|-------|--------|
| 1 | Platform hashtags (match always logic) | Backend | Medium |
| 2 | Admin calculator "Live Data" button | Frontend | Low |
| 3 | Scheduled not clickable (debug) | Frontend | Low |
| 4 | Per-video date/time slots for scheduled | Frontend + Backend | High |
| 5 | Multi-account uploads (all accounts) | Frontend + Backend | High |
| 6 | Dashboard/queue multi-account display | Backend + Frontend | Medium |
| 7 | Discord webhook: platform + profile + links | Backend | Medium |
| 8 | 7/30/90 day APIs for admin/analytics | Backend | Medium |
| 9 | Voice/tone in caption pipeline | Backend | Medium |
| 10 | Default credits for new accounts | Backend | Low |

---

*Generated from codebase analysis. Backend changes require access to the uploadm8 backend repository.*
