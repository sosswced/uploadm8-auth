# Local Testing Guide

## 1. Start Backend & Frontend

**Terminal 1 — Backend (port 8000):**
```powershell
cd c:\Users\Earl\Dev\uploadm8-auth
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 — Frontend (port 8080):**
```powershell
cd c:\Users\Earl\Dev\uploadm8-frontend
npx serve -p 8080
```

## 2. Use the Correct URL

- **Dashboard:** `http://localhost:8080/dashboard.html` (not `/dashboard`)
- **Login:** `http://localhost:8080/login.html`
- **Upload:** `http://localhost:8080/upload.html`

The `serve` package serves static files by name. `/dashboard` without `.html` may 404.

## 3. Load Modified Files (Bypass Cache)

1. **Hard refresh:** `Ctrl + Shift + R` or `Ctrl + F5`
2. **Or:** F12 → Network tab → check "Disable cache" → refresh
3. **Or:** Use an Incognito/Private window

## 4. Verify Modified Code Is Loading

1. Open `http://localhost:8080/dashboard.html`
2. Press F12 → Console
3. Run: `console.log(window.API_BASE)`
   - Should show: `http://127.0.0.1:8000`
4. Run: `console.log(window.currentUser?.email)`
   - Should show your email when logged in

## 5. Check for Errors

- F12 → Console: look for red errors
- F12 → Network: filter by "Fetch/XHR" — API calls should go to `127.0.0.1:8000` and return 200

## 6. If Dashboard Is Blank or Broken

1. Check Console for JavaScript errors
2. Ensure you're logged in (visit `login.html` first)
3. Ensure backend is running (visit `http://127.0.0.1:8000/docs` to confirm)
