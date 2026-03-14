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

## After the frontend agent responds

- Start the **backend** (in `uploadm8-auth`):  
  `python -m dotenv run -- uvicorn app:app --reload --host 127.0.0.1 --port 8000`
- Start the **frontend** using the commands the agent gave you.
- If the frontend runs on a port other than 3000, add it to the backend’s `ALLOWED_ORIGINS` in your backend `.env`, e.g.:  
  `ALLOWED_ORIGINS=https://app.uploadm8.com,https://uploadm8.com,http://localhost:3000,http://localhost:5173`
