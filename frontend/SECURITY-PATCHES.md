# Security & Critical Fix Patches

This document summarizes the patches applied to address critical security and correctness issues.

**See also:** [GOOGLE-SECURITY-COMPLIANCE.md](GOOGLE-SECURITY-COMPLIANCE.md) for Google Cloud security best practices (zero-code storage, API restrictions, least privilege, key rotation).

## 1. Security

### Content-Security-Policy (CSP)
- **Added** a restrictive CSP meta tag to:
  - `admin.html`, `account-management.html`, `login.html`, `signup.html`, `forgot-password.html`, `index.html`
- Policy: `default-src 'self'`; scripts/styles from self + CDNs; `connect-src` includes `https://auth.uploadm8.com` and `http://127.0.0.1:8000` / `http://localhost:8000` for local dev.
- **Note:** `script-src` and `style-src` include `'unsafe-inline'` because the app uses inline scripts and styles. Locking down further would require moving to nonce- or hash-based CSP and a build step.

### Subresource Integrity (SRI)
- **Font Awesome** (cdnjs 6.4.0): `integrity="sha384-iw3OoTErCYJJB9mCa8LNS2hbsQ7M3C0EpIsO/H5+EGAkPGc6rk+V8i04oW/K5xq0"` + `crossorigin="anonymous"` added where the CSS is used (admin, account-management, login, signup, forgot-password, index).
- **Chart.js** (jsDelivr): `integrity="sha384-jb8JQMbMoBUzgWatfe6COACi2ljcDdZQ2OxczGA3bGNeWe+6DChMTBJemed7ZnvJ"` + `crossorigin="anonymous"` on `admin.html`.
- Google Fonts CSS does not support SRI in the same way (URL can change); consider self-hosting or accepting the risk.

### XSS mitigations (innerHTML / user or API data)
- **app.js**
  - **Token / logout:** `clearTokens()` now clears every token key used anywhere: `uploadm8_access_token`, `accessToken`, `access_token`, `authToken`, `auth_token`, `token`, `uploadm8_refresh_token`, `refreshToken`, `refresh_token` in both localStorage and sessionStorage.
  - **escapeHTML:** New helper; exported as `window.escapeHTML` for use in other scripts.
  - **Avatar in updateUserUI:** Replaced `userAvatar.innerHTML = ...` with `createElement('img')`, set `src` and styles, `appendChild`. No user-controlled HTML.
  - **Tier badge:** Replaced `innerHTML` with DOM creation and `textContent` for labels.
  - **showToast:** Builds toast with `createElement` and `textContent` for message; close button uses `addEventListener`, not inline `onclick`.
  - **showLoading / showEmptyState / showError:** Message parameter is set via `textContent` (no interpolation into HTML). `showEmptyState` still accepts `actionHtml` for optional static markup; avoid passing user/API data there.
- **auth-init.js**
  - **clearAndRedirect:** Clears the same set of token keys as `app.js` (localStorage + sessionStorage).
- **auth-core.js**
  - **clearTokens:** Clears the same token keys for consistency.
  - **applyAvatarToUI:** Replaced `innerHTML` with `createElement('img')`, validated URL with `^https?:` before use.
- **admin.html**
  - **searchUsers:** No longer builds `onclick="selectUser('${id}','${name}')"` from user data. Builds result rows with DOM, `textContent` for name/email, and `addEventListener('click', () => selectUser(u.id))`.
  - **renderSelectedUsers:** Builds tags with DOM and `textContent`; remove button uses `addEventListener` instead of inline `onclick` with user id in attribute.
- **account-management.html**
  - **Toast fallback:** Uses `textContent` for message.
  - **Sidebar avatar:** Uses `createElement('img')` and `img.src` with cache-bust; no innerHTML with user.avatar_url or displayName.
  - **loadUsers error row:** Error message set via `textContent`.
  - **renderUsers:** Entire table body built with DOM; user name, email, role, tier, status, dates set via `textContent`. Edit/Ban/Unban buttons use `addEventListener` with `u.id` from closure, not inline `onclick` with id in HTML.

### Incomplete logout (tokens left in storage)
- **Fixed** in `app.js`, `auth-init.js`, and `auth-core.js`: all code paths that clear tokens now remove every key that `getAccessToken` / `getRefreshToken` or compatibility code ever read or wrote (`uploadm8_*`, `accessToken`, `access_token`, `authToken`, `auth_token`, `token`, `refreshToken`, `refresh_token` in both localStorage and sessionStorage).

---

## 2. Bugs (behavior)

### auth-init.js – initAuth not async
- **Issue:** `initAuth()` used `await` but was declared as `function initAuth()`, so at runtime the first `await` threw.
- **Fix:** Declared as `async function initAuth()` and left the rest of the logic unchanged. Removed obsolete helper fragments that were no longer referenced.

---

## 3. What was not changed (recommendations)

- **Single auth flow:** The codebase still has multiple auth entry points (`app.js` checkAuth + initApp, `auth-core.js` authenticate, `auth-init.js` initAuth). Unifying to one flow (e.g. always use `checkAuth` + `initApp`) is recommended for maintainability but was not part of this patch set.
- **Tokens in localStorage only:** Access/refresh tokens remain in localStorage/sessionStorage. Moving refresh tokens to httpOnly cookies and using short-lived access tokens would require backend and deployment changes; not done here.
- **CSP on every page:** Only a subset of HTML files received the CSP meta tag. Apply the same (or a stricter) CSP to the remaining pages (dashboard, settings, queue, etc.) for full coverage.
- **SRI on all CDN scripts:** Only Font Awesome and Chart.js were given SRI where they are loaded. Other pages that load Font Awesome (or other CDN assets) can use the same `integrity` values shown above.
- **Remaining innerHTML:** Other files (e.g. `admin-wallet.html`, `queue.html`, `dashboard.html`, `settings.html`, `admin-kpi.html`, etc.) may still use `innerHTML` or inline `onclick` with user/API data. A full audit and conversion to `textContent` / `createElement` / `addEventListener` is recommended.
- **Build pipeline, tests, i18n, error tracking:** Not addressed in these patches; see your roadmap for enterprise readiness.

---

## 4. Verifying

- **Logout:** Log in, then log out. In DevTools → Application → Local Storage / Session Storage, confirm all of the token keys listed above are removed.
- **Auth init:** On a protected page that loads `auth-init.js`, ensure the page still loads and sidebar/user state updates (no console error from `initAuth`).
- **Admin user search:** On Admin Panel, use the user search and select users; confirm no script execution from malicious name/email and that list/selection still work.
- **CSP:** Open the patched pages and check the console for CSP violations; adjust the meta tag if you add new script/style sources.
- **SRI:** If you upgrade Font Awesome or Chart.js, recompute the integrity hashes and update the `integrity` attributes.
