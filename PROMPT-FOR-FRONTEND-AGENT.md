# Prompt for Cursor Frontend Agent

Copy and paste the following into your **frontend project** (the UploadM8 app UI repo) when talking to Cursor, so the frontend runs correctly against the local backend.

---

## Prompt (copy from here)

**Goal:** Run this frontend project locally so it talks to the UploadM8 **local backend API** instead of production.

**Backend (already running on my machine):**
- Base URL: `http://127.0.0.1:8000`
- Docs: http://127.0.0.1:8000/docs
- The backend is the `uploadm8-auth` repo, running with `uvicorn app:app --reload --host 127.0.0.1 --port 8000` and loading env from `.env`.

**What I need you to do:**
1. **Identify** how this frontend is built and run (e.g. static HTML/JS, Vite, Create React App, Next.js, etc.) and how it currently gets the API base URL (env var, config file, or hardcoded).
2. **Configure** the frontend so that in local development it uses `http://127.0.0.1:8000` (or `http://localhost:8000`) as the API / auth base URL. Update the right place (e.g. `.env.local`, `vite.config.js`, or the config that sets the API origin).
3. **Give me exact commands** to start the frontend dev server from this repo (e.g. `npm install` then `npm run dev`, or `npx serve .`, or whatever applies) so I can run the app in the browser and have it call the local backend.
4. If the app uses CORS, the backend already allows `http://localhost:3000` in `ALLOWED_ORIGINS`; if the frontend runs on a different port, tell me so I can add it to the backend’s allowed origins.

**Summary:** I want to run this frontend in service with my local backend at `http://127.0.0.1:8000`. Configure and document the steps to run the frontend and point it at that API.

---

## Multi-account uploads (backend ready)

Users select **platforms** (YouTube, TikTok, etc.) and **which accounts** within each platform. Uploads post only to **connected and selected** accounts.

**API:**
- `GET /api/platform-accounts` — returns `{ accounts: [{ id, platform, account_name, account_username, ... }] }`. Use `id` (platform_tokens.id UUID) for `target_accounts`.
- `POST /api/uploads/presign` — accepts `target_accounts: string[]`. Send the selected account IDs from the account picker.

**Logic:**
- `target_accounts` provided: publish only to those accounts (e.g. 2 YouTube channels + 1 TikTok = 3 publishes).
- `target_accounts` empty: legacy fallback — one publish per platform using the most recently connected account.

**Frontend UX:**
1. User selects platforms (e.g. YouTube, TikTok).
2. For each selected platform, show connected accounts from `GET /api/platform-accounts`.
3. User selects which accounts to publish to (checkboxes per account).
4. Send selected account `id`s as `target_accounts` in the presign request.

---

## Platform-specific hashtags

There is already UI for this. Platform hashtags now use the **same pipeline** as always_hashtags (merge order: always → platform → base → AI). Save via `PUT /api/me/preferences`:

```json
{
  "platformHashtags": {
    "youtube": ["shorts", "viral", "youtubeshorts"],
    "tiktok": ["fyp", "viral", "foryou"],
    "instagram": ["reels", "explore"],
    "facebook": ["reels", "viral"]
  }
}
```

- Keys must be lowercase: `youtube`, `tiktok`, `instagram`, `facebook`.
- Values are arrays of strings (without `#`; the backend adds them).

---

## Edit uploads in queue and scheduled pages

**Goal:** Allow users to edit title, caption, hashtags, and schedule times for uploads that have not yet been published to platforms.

**Editable statuses:** `pending`, `staged`, `queued`, `scheduled`, `ready_to_publish` (not `processing`, `completed`, `succeeded`, etc.)

**API:**

1. **Get upload details (for edit form):**
   - `GET /api/uploads/{upload_id}` — full upload (queue page)
   - `GET /api/scheduled/{upload_id}` — full upload (scheduled page)
   - Both return: `title`, `caption`, `hashtags`, `scheduled_time`, `schedule_mode`, `schedule_metadata`, `timezone`, `thumbnail` (presigned URL), `platforms`

2. **Update upload (queue page):**
   - `PATCH /api/uploads/{upload_id}`
   - Body (all optional): `{ title?, caption?, hashtags?, scheduled_time?, smart_schedule? }`
   - `smart_schedule`: `{ "tiktok": "2025-03-15T19:00:00Z", "youtube": "2025-03-16T17:00:00Z" }` (platform → ISO datetime string)
   - `scheduled_time`: for single-time scheduled uploads (ISO datetime string)

3. **Update scheduled upload (scheduled page) — smart_schedule only:**
   - `PATCH /api/scheduled/{upload_id}`
   - Body (required): `{ "smart_schedule": { "tiktok": "2025-03-15T19:00:00Z", "youtube": "2025-03-16T17:00:00Z" } }`

4. **Regenerate thumbnail from video:**
   - `POST /api/uploads/{upload_id}/generate-thumbnail` — extracts frame from video

**Frontend UX (queue.html, scheduled.html):**
- Show an "Edit" button on each upload card when status is editable
- Edit modal/form: title, caption, hashtags, and:
  - **Smart schedule:** datetime picker per platform (from `schedule_metadata` in GET response)
  - **Single scheduled:** one datetime picker (from `scheduled_time`)
- Queue page: Save calls `PATCH /api/uploads/{upload_id}` with changed fields
- Scheduled page: Save calls `PATCH /api/scheduled/{upload_id}` with `smart_schedule` only
- Disable edit for uploads that are processing or already published

---

## After the frontend agent responds

- Start the **backend** (in `uploadm8-auth`):  
  `python -m dotenv run -- uvicorn app:app --reload --host 127.0.0.1 --port 8000`
- Start the **frontend** using the commands the agent gave you.
- If the frontend runs on a port other than 3000, add it to the backend’s `ALLOWED_ORIGINS` in your backend `.env`, e.g.:  
  `ALLOWED_ORIGINS=https://app.uploadm8.com,https://uploadm8.com,http://localhost:3000,http://localhost:5173`

---

## Admin Business Calculator — live pricing on load

**Goal:** The admin calculator (`admin-calculator.html`) should load **live** pricing and entitlements from the backend on page load, instead of hardcoded values.

**API:** `GET /api/admin/calculator/pricing` (requires master_admin)

**Response shape:**
```json
{
  "revenue_tiers": {
    "free": { "name": "Free", "price": 0, "put_monthly": 60, "aic_monthly": 30, ... },
    "creator_lite": { "name": "Creator Lite", "price": 9.99, "put_monthly": 360, "aic_monthly": 120, ... },
    "creator_pro": { "name": "Creator Pro", "price": 19.99, "put_monthly": 900, "aic_monthly": 260, ... },
    "studio": { "name": "Studio", "price": 49.99, "put_monthly": 2500, "aic_monthly": 800, ... },
    "agency": { "name": "Agency", "price": 99.99, "put_monthly": 6000, "aic_monthly": 2000, ... }
  },
  "internal_tiers": {
    "friends_family": { "name": "Friends & Family", "price": 0, "put_monthly": 999999, "aic_monthly": 999999 },
    "lifetime": { "name": "Lifetime", ... },
    "master_admin": { "name": "Administrator", ... }
  },
  "topup_packs": [
    { "lookup_key": "uploadm8_put_100", "wallet": "put", "amount": 100, "label": "PUT 100 Pack" },
    ...
  ]
}
```

**Frontend integration:**
1. On `admin-calculator.html` load (after admin access is verified), call `GET /api/admin/calculator/pricing`.
2. Prefill the Inputs form with:
   - **Subscription Prices:** `revenue_tiers.creator_pro.price`, `revenue_tiers.studio.price`, etc.
   - **PUT Allowances:** `revenue_tiers.*.put_monthly` for each tier.
   - **AIC Allowances:** `revenue_tiers.*.aic_monthly` for each tier.
   - **Internal tiers (Friends & Family, Admin/Internal, Lifetime):** `internal_tiers.friends_family.put_monthly`, etc.
3. Top-up pack amounts come from `topup_packs`; prices are in Stripe (or use defaults if not fetched).
4. Run **Calculate** automatically after prefilling, or keep the existing Calculate button flow.

**Source of truth:** `stages/entitlements.py` — matches [index.html pricing](https://app.uploadm8.com/index.html) (Free, Creator Lite, Creator Pro, Studio, Agency) plus internal tiers (Friends & Family, Lifetime, Master Admin).
