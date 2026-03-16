# Running the frontend with your local backend

This frontend is **static** (HTML, CSS, vanilla JS). There is no build step, no `package.json`, and no framework. The API base URL is chosen at runtime: when the app is opened from **localhost** or **127.0.0.1**, it automatically uses your local backend.

---

## 1. How the API base URL is set

- **Production:** Uses `https://auth.uploadm8.com` (hardcoded fallback in `app.js`, `auth-core.js`, `login.html`, `signup.html`, `forgot-password.html`).
- **Local dev:** When the page origin is `http://localhost:*` or `http://127.0.0.1:*`, the app sets `window.API_BASE = 'http://127.0.0.1:8000'` so all fetch calls go to your local API.
- **Override:** You can override the API for the current page with a query param, e.g.  
  `http://localhost:8080/login.html?api=http://127.0.0.1:8000`  
  or by setting `window.API_BASE` in the browser console before any request.

No env vars or config files are used; the check is done in script on each page.

---

## 2. Commands to run the frontend

From the **frontend repo root** (`uploadm8-frontend`):

```bash
# Serve the static files (default port 8080)
python -m http.server 8080
```

Then open in the browser:

- **http://localhost:8080** or **http://127.0.0.1:8080**

Entry point: **http://localhost:8080/index.html** (or **login.html**, **dashboard.html**, etc.)

No `npm install` or other build step is required.

---

## 3. Full local workflow

1. **Start the backend** (in the `uploadm8-auth` repo):

   ```bash
   python -m dotenv run -- uvicorn app:app --reload --host 127.0.0.1 --port 8000
   ```

   API docs: **http://127.0.0.1:8000/docs**

2. **Start the frontend** (in this repo):

   ```bash
   python -m http.server 8080
   ```

3. Open **http://localhost:8080** (or **http://127.0.0.1:8080**) and use the app. It will call **http://127.0.0.1:8000** for auth and API.

---

## 4. CORS (backend `ALLOWED_ORIGINS`)

The frontend is served on **port 8080**, not 3000. Add both origins to your backend’s allowed list so the browser allows requests.

In your **backend** `.env` (or wherever `ALLOWED_ORIGINS` is configured), include:

```env
ALLOWED_ORIGINS=https://app.uploadm8.com,https://uploadm8.com,http://localhost:3000,http://localhost:5173,http://localhost:8080,http://127.0.0.1:8080
```

If the backend only allows comma-separated origins, add at least:

- `http://localhost:8080`
- `http://127.0.0.1:8080`

Then restart the backend so the new CORS settings are loaded.

---

## Summary

| Item              | Value                          |
|-------------------|--------------------------------|
| Frontend type     | Static HTML/JS (no build)      |
| API base (local)  | `http://127.0.0.1:8000` (auto) |
| How it’s set      | Script checks `location.origin`|
| Run frontend      | `python -m http.server 8080`   |
| Frontend URL      | http://localhost:8080          |
| Backend CORS      | Add `http://localhost:8080`, `http://127.0.0.1:8080` to `ALLOWED_ORIGINS` |
