/**
 * UploadM8 — canonical API origin for all static pages.
 *
 * SHELL STACK: This file must be the first external script in the standard auth/hydration
 * bundle. Order for pages that show logged-in chrome:
 *   api-base → auth-stack → session-user-hydrate → public-shell-hydrate → (shared-sidebar on app shell) → …
 * Place that block synchronously at the end of <body> (no defer/async) so it runs before the
 * browser typically paints the markup above it.
 *
 * Priority:
 *   1. ?api=URL  — one-time override; also saved to sessionStorage for later navigations
 *   2. ?api=clear — remove saved override
 *   3. sessionStorage.uploadm8_api_base — persisted from a previous ?api=
 *   4. localhost/127.0.0.1 ports 8000–8002 → same origin (matches “Python: Uvicorn (API + frontend)” on :8001)
 *   5. Static-only server on 8001/8002 with API on :8000 → open once with
 *      ?api=http%3A%2F%2F127.0.0.1%3A8000 (saved in sessionStorage)
 *   6. other localhost (e.g. Vite :3000) → http://127.0.0.1:8000 (local uvicorn; avoids CORS to prod)
 *   7. else → https://auth.uploadm8.com
 *
 * Production API from a dev port: ?api=https%3A%2F%2Fauth.uploadm8.com (requires prod CORS for your origin).
 * Local API: open once with ?api=http%3A%2F%2F127.0.0.1%3A8000 then reload any page;
 * or run frontend on port 8000 with uvicorn serving static. To clear: ?api=clear
 */
(function initUploadM8ApiBase() {
  var SS_KEY = 'uploadm8_api_base';
  try {
    var q = (typeof URLSearchParams !== 'undefined' && typeof location !== 'undefined' && location.search)
      ? new URLSearchParams(location.search).get('api')
      : null;
    if (q && String(q).toLowerCase() === 'clear') {
      try { sessionStorage.removeItem(SS_KEY); } catch (e) {}
      q = null;
    }
    if (q) {
      window.API_BASE = decodeURIComponent(q).replace(/\/$/, '');
      try { sessionStorage.setItem(SS_KEY, window.API_BASE); } catch (e) {}
      return;
    }
    if (typeof window.API_BASE === 'string' && window.API_BASE) return;

    var persisted = null;
    try {
      persisted = sessionStorage.getItem(SS_KEY);
    } catch (e) {}
    if (persisted && /^https?:\/\//i.test(String(persisted).trim())) {
      window.API_BASE = String(persisted).trim().replace(/\/$/, '');
      return;
    }

    var host = location.hostname || '';
    var port = location.port || '';
    var isLocal = host === 'localhost' || host === '127.0.0.1';

    if (isLocal) {
      if (port === '8000' || port === '8001' || port === '8002') {
        window.API_BASE = location.origin.replace(/\/$/, '');
        return;
      }
      window.API_BASE = 'http://127.0.0.1:8000';
      return;
    }
    window.API_BASE = 'https://auth.uploadm8.com';
  } catch (e) {
    if (!window.API_BASE) window.API_BASE = 'https://auth.uploadm8.com';
  }
})();

window.getUploadM8ApiBase = function getUploadM8ApiBase() {
  var SS_KEY = 'uploadm8_api_base';
  try {
    var q = (typeof URLSearchParams !== 'undefined' && typeof location !== 'undefined' && location.search)
      ? new URLSearchParams(location.search).get('api')
      : null;
    if (q && String(q).toLowerCase() !== 'clear') {
      return decodeURIComponent(q).replace(/\/$/, '');
    }
    if (typeof window.API_BASE === 'string' && window.API_BASE)
      return String(window.API_BASE).replace(/\/$/, '');
    var persisted = null;
    try { persisted = sessionStorage.getItem(SS_KEY); } catch (e) {}
    if (persisted && /^https?:\/\//i.test(String(persisted).trim())) {
      return String(persisted).trim().replace(/\/$/, '');
    }
    var host = location.hostname || '';
    var port = location.port || '';
    var isLocal = host === 'localhost' || host === '127.0.0.1';
    if (isLocal) {
      if (port === '8000' || port === '8001' || port === '8002') {
        return location.origin.replace(/\/$/, '');
      }
      return 'http://127.0.0.1:8000';
    }
    return 'https://auth.uploadm8.com';
  } catch (e) {
    return 'https://auth.uploadm8.com';
  }
};

/**
 * Human-readable message from FastAPI JSON: detail string, Pydantic list, or api_problem { code, message }.
 */
window.uploadM8ApiErrorMessage = function uploadM8ApiErrorMessage(body) {
  if (!body || typeof body !== 'object') return '';
  if (typeof body.message === 'string' && body.message) return body.message;
  var d = body.detail;
  if (typeof d === 'string') return d;
  if (Array.isArray(d)) {
    return d
      .map(function (x) {
        if (!x) return '';
        if (typeof x === 'string') return x;
        if (x.msg) return String(x.msg);
        if (x.message) return String(x.message);
        try {
          return JSON.stringify(x);
        } catch (e) {
          return '';
        }
      })
      .filter(Boolean)
      .join('; ');
  }
  if (d && typeof d === 'object') {
    if (typeof d.message === 'string' && d.message) return d.message;
    if (typeof d.msg === 'string' && d.msg) return d.msg;
  }
  return '';
};

/**
 * Origin string for fetch() on full page loads — prefer getUploadM8ApiBase() so ?api= applies per navigation.
 */
window.resolveUploadM8ApiOrigin = function resolveUploadM8ApiOrigin() {
  try {
    if (typeof window.getUploadM8ApiBase === 'function') return window.getUploadM8ApiBase();
  } catch (e) {}
  var b = typeof window.API_BASE === 'string' ? window.API_BASE : '';
  return String(b).replace(/\/$/, '');
};
