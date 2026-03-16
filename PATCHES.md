# UploadM8 — Break Fix Patches

**Date:** March 15, 2026  
**Purpose:** Pinpoint and fix breaks introduced by the pricing overhaul commits (auth `a013dc4`, frontend `7b8d0cd` + `692a09b`).

---

## Break Points Identified

| # | Location | Issue | Impact |
|---|----------|-------|--------|
| 1 | `uploadm8-frontend/app.js` | Debug `fetch` to `127.0.0.1:7767` in `updateUserUI` | CSP blocks in production; can cause console errors or unexpected behavior |
| 2 | `uploadm8-frontend/app.js` | Tier fallback missing `plan.tier` | API returns `plan.tier` (from entitlements); frontend only checked `plan.name` / `plan.subscription_tier` |
| 3 | `uploadm8-frontend/app.js` | No try/catch around `WalletTokens.renderTokenPills` | If WalletTokens throws, entire `updateUserUI` fails → broken header/sidebar |
| 4 | `uploadm8-frontend/dashboard.html` | No try/catch around `renderUsageBanner` | Same failure mode for dashboard |

---

## Patches Applied

### 1. Remove Debug Fetch (app.js)

**File:** `uploadm8-frontend/app.js`  
**Change:** Removed the agent debug instrumentation that POSTed to `http://127.0.0.1:7767/ingest/...`.

```diff
-    // #region agent log
-    fetch('http://127.0.0.1:7767/ingest/41b1265e-ab1d-40f0-8242-b596cc87964d',{method:'POST',...}).catch(()=>{});
-    // #endregion
     Object.entries(els).forEach(([id, value]) => {
```

**Why:** CSP `connect-src` does not include `127.0.0.1:7767`. In production this fetch is blocked and can cause errors.

---

### 2. Add plan.tier Fallback (app.js)

**File:** `uploadm8-frontend/app.js`  
**Change:** Include `currentUser.plan?.tier` in the tier resolution chain.

```diff
-    const tier = currentUser.subscription_tier || currentUser.plan?.name || currentUser.plan?.subscription_tier || 'free';
+    const tier = currentUser.subscription_tier || currentUser.plan?.tier || currentUser.plan?.name || currentUser.plan?.subscription_tier || 'free';
```

**Why:** `/api/me` returns `plan` from `entitlements_to_dict()`, which uses `tier` and `tier_display`, not `name` or `subscription_tier`. Fallback ensures correct tier display.

---

### 3. Defensive try/catch for Token Pills (app.js)

**File:** `uploadm8-frontend/app.js`  
**Change:** Wrap `WalletTokens.renderTokenPills` in try/catch.

```diff
     // Token pills in header (PUT/AIC wallet indicators)
+    try {
         if (typeof window.WalletTokens !== 'undefined' && window.WalletTokens.renderTokenPills) {
             const topBarActions = document.querySelector('.top-bar-actions');
             if (topBarActions) window.WalletTokens.renderTokenPills(topBarActions);
         }
+    } catch (e) {
+        console.warn('[WalletTokens] renderTokenPills failed:', e);
+    }
```

**Why:** If `renderTokenPills` throws (e.g. malformed user data), the rest of `updateUserUI` still runs.

---

### 4. Defensive try/catch for Usage Banner (dashboard.html)

**File:** `uploadm8-frontend/dashboard.html`  
**Change:** Wrap `WalletTokens.renderUsageBanner` in try/catch.

```diff
+    try {
         if (typeof window.WalletTokens !== 'undefined' && window.WalletTokens.renderUsageBanner) {
             const container = document.getElementById('usageBannerContainer');
             if (container) window.WalletTokens.renderUsageBanner(container, user);
         }
+    } catch (e) {
+        console.warn('[Dashboard] renderUsageBanner failed:', e);
+    }
```

**Why:** Same failure isolation for dashboard.

---

## Backend (uploadm8-auth) — No Patches Needed

The following were reviewed and are **correct**:

- **Hashtag logic:** `context.get_effective_hashtags(platform)` is used in `publish_stage.py`; `user_settings` is passed into `create_context` from the worker. No change required.
- **`/api/me` response:** Returns `wallet` and `plan` (from `get_plan` → `entitlements_to_dict`) with `put_monthly`, `aic_monthly`, `tier`, `tier_display`.
- **Entitlements:** `creator_lite`, `creator_pro`, etc. are defined; `launch` aliases to creator_lite.

---

## Deployment

1. **Frontend (uploadm8-frontend):**
   ```bash
   cd c:\Users\Earl\Dev\uploadm8-frontend
   git add app.js dashboard.html
   git commit -m "fix: remove debug fetch, add plan.tier fallback, defensive try/catch for WalletTokens"
   git push
   ```

2. **Auth (uploadm8-auth):** No code changes. Patches are frontend-only.

---

## Verification

After deploying:

1. **Dashboard** — Load dashboard; confirm no console errors, token pills and usage banner render.
2. **Queue / Scheduled** — Confirm no `[object Object]` in platform results.
3. **Settings** — Confirm tier and billing section display correctly.
4. **Upload** — Confirm cost estimate and token warnings work.

---

*Patches generated from break analysis. See CHANGES-SUMMARY.md and SESSION-CHANGES-SUMMARY.md for full change history.*
