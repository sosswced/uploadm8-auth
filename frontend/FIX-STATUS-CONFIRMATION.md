# Fix Status Confirmation

Status of each issue/bug from your list. **F** = Fixed (frontend), **B** = Backend required, **D** = Documented only.

---

## ✅ FIXED (Frontend)

| # | Issue | Status | What Was Done |
|---|-------|--------|---------------|
| 1 | **Scheduled not clickable / no modals** | **F** | Schedule option is a clickable `<label>` with `cursor: pointer`. Selecting "Schedule" shows inline per-video datetime pickers (no modal; inline UI per spec). |
| 2 | **Per-video time slots for scheduled** | **F** | `renderScheduleSlots()` shows one datetime picker per selected file. 1 video = 1 slot, 10 = 10, 100 = 100. Each presign sends its own `scheduled_time`. |
| 3 | **Platform hashtags** | **F** | Settings saves `platformHashtags` (lowercase keys) via POST /api/settings/preferences and PUT /api/me/preferences. Backend does case-insensitive lookup (per your note). |
| 4 | **Admin calculator Live Data button** | **F** | "Load Live Data" button calls `fetchPricingAndRun()` to reload live pricing from DB and re-run the calculator. |
| 5 | **Multi-account uploads** | **F** | Account picker per platform; user selects which accounts. `resolveTargetAccountIds()` returns selected account IDs (or group IDs). `target_accounts` sent in presign/complete. |
| 6 | **Admin KPI time range + refresh** | **F** | Time range selector (24h, 7d, 30d, 90d, 6m, 1y) passes `range` param. Refresh button calls POST /api/admin/kpi/refresh, waits 5s, then reloads. |

---

## 📋 DOCUMENTED (Logic Explained)

| # | Issue | Status | Notes |
|---|-------|--------|-------|
| 7 | **Queue pending vs processing** | **D** | `pending`/`queued` = waiting for worker; `processing` = worker actively transcoding/AI. See `FEATURE-ANALYSIS-AND-ROADMAP.md`. |
| 8 | **Smart vs scheduled uploads** | **D** | **Schedule:** user picks one time per video. **Smart:** AI picks per-platform times across a window. Both use `staged` → scheduler. |

---

## ⏳ BACKEND REQUIRED

| # | Issue | Status | What Backend Must Do |
|---|-------|--------|----------------------|
| 9 | **Admin/analytics 7/30/90 day APIs** | **B** | APIs like `GET /api/admin/kpis?range=7d` must accept `range` and filter data. KPI page already sends it. |
| 10 | **Discord upload complete: platform + profile + links** | **B** | Webhook payload must include each `platform_results` entry with `platform`, `account_name`, `url`. |
| 11 | **Dashboard/queue: multiple accounts display** | **B** | When 1 video → 2 TikTok accounts, backend must return **2** `platform_results` entries (one per account) with `account_id`, `account_name`, `url`. Frontend will show each when present. |
| 12 | **Voice and tone for captions** | **B** | Settings persist `captionStyle`, `captionTone`, `captionVoice`. Backend caption pipeline must pass these to the AI prompt and avoid misleading/fake content. |
| 13 | **Default PUT/AIC credits for new accounts** | **B** | On signup, grant default PUT and AIC based on entitlements so new users can try features. |

---

## 🔗 URL Logic (Frontend Ready)

| # | Issue | Status | Notes |
|---|-------|--------|-------|
| 14 | **Correct hyperlinks after upload** | **F** | `resolvePlatformUrl()` in queue.html builds URLs from `video_id`, `shortcode`, etc. Backend must populate these per account in `platform_results`. |

---

## Summary

- **6 items** fully fixed in frontend
- **2 items** documented (logic)
- **5 items** need backend changes
- **1 item** (URLs) frontend-ready; backend must supply correct IDs per account
