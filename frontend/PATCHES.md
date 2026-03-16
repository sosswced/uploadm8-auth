# UploadM8 Frontend — Break Fix Patches

**Date:** March 15, 2026  
**Purpose:** Fix breaks introduced by commits `7b8d0cd` and `692a09b`.

---

## Break Points Fixed

| # | File | Issue |
|---|------|-------|
| 1 | app.js | Debug `fetch` to `127.0.0.1:7767` — CSP blocks in production |
| 2 | app.js | Tier fallback missing `plan.tier` |
| 3 | app.js | No try/catch around `WalletTokens.renderTokenPills` |
| 4 | dashboard.html | No try/catch around `renderUsageBanner` |

---

## Files Changed

- `app.js` — 3 patches
- `dashboard.html` — 1 patch

---

## Deploy

```bash
git add app.js dashboard.html PATCHES.md
git commit -m "fix: remove debug fetch, add plan.tier fallback, defensive try/catch for WalletTokens"
git push
```
