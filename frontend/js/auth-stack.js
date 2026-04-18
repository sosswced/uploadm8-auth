/**
 * UploadM8 — single browser auth module (load synchronously right after js/api-base.js).
 *
 * Includes: token storage + migration, refresh/login/logout/register, apiCall(), and checkAuth()
 * (session guard + /api/me pipeline). initApp() in app.js calls checkAuth after other shell scripts run;
 * checkAuth is defined here as soon as this file parses — no second auth-*.js script tag.
 *
 * Primary auth: HttpOnly cookies on /api/auth/* when app and API share a site; apiCall uses credentials:'include'.
 * Same-site: applyUploadM8AuthSession() avoids storing JWT/refresh in JS (cookies only). Cross-host: um8_send_bearer + sessionStorage.
 *
 * Order: api-base → THIS FILE → session-user-hydrate → public-shell-hydrate → shared-sidebar →
 * helpers → upload-utils → app.js → wallet-tokens.js
 *
 * All JSON calls to /api/* should use apiCall(). For CSV/blob/text responses use apiFetch()
 * (same cookies, bearer, X-Request-ID, timeouts, and 401 refresh — set authRedirectOn401:false
 * on public pages like login.html when probing /api/me).
 *
 * Optional: set window.__UPLOADM8_LONG_FETCH_TIMEOUT_MS (ms) to override the long-read cap for
 * dashboard/catalog/wallet/sync-analytics and similar paths (default 120s).
 * GET /api/uploads/{uuid} (upload page poll) uses UPLOAD_STATUS_POLL_TIMEOUT_MS (60s default) —
 * longer than the generic 15s cap but not classified as "long-running" (avoids offline gate).
 *
 * Concurrent identical GETs (same URL, bearer, timeout) share one network request; pass
 * skipGetDedupe: true on apiFetch/apiCall options to force a fresh round-trip (e.g. upload status poll).
 */
(function () {
    const TOKEN_KEY = 'uploadm8_access_token';
    const REFRESH_KEY = 'uploadm8_refresh_token';
    const ACCESS_ALIASES = [TOKEN_KEY, 'accessToken', 'access_token', 'authToken', 'auth_token', 'token', 'uploadm8_token'];
    const REFRESH_ALIASES = [REFRESH_KEY, 'refreshToken', 'refresh_token'];
    const DEFAULT_FETCH_TIMEOUT_MS = 15000;

    function generateRequestId() {
        return 'req_' + Date.now().toString(36) + Math.random().toString(36).substr(2, 9);
    }

    function _isNetworkError(e) {
        return (
            e.name === 'TimeoutError' ||
            e.name === 'AbortError' ||
            e instanceof TypeError ||
            e.message === 'Failed to fetch'
        );
    }

    function _isDefinitiveAuthFailure(status) {
        return status === 401 || status === 403;
    }

    async function fetchWithTimeout(url, options = {}, timeoutMs = DEFAULT_FETCH_TIMEOUT_MS) {
        const controller = new AbortController();
        const tid = setTimeout(() => controller.abort(), timeoutMs);
        try {
            return await fetch(url, { ...options, signal: controller.signal });
        } catch (e) {
            if (e && e.name === 'AbortError') {
                const rid = options?.headers?.['X-Request-ID'] || generateRequestId();
                const err = new Error(`Request timed out after ${timeoutMs}ms`);
                err.name = 'TimeoutError';
                err.status = 408;
                err.requestId = rid;
                throw err;
            }
            throw e;
        } finally {
            clearTimeout(tid);
        }
    }

    function escapeHTML(value) {
        if (value === null || value === undefined) return '';
        return String(value)
            .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
    }

    function _stripAuthFromLocalStorage() {
        ACCESS_ALIASES.concat(REFRESH_ALIASES).forEach(function (k) {
            try { localStorage.removeItem(k); } catch (_) {}
        });
    }

    function _firstValue(storage, keys) {
        for (let i = 0; i < keys.length; i++) {
            const v = storage.getItem(keys[i]);
            if (v) return v;
        }
        return '';
    }

    function _bearerHeaderMode() {
        try {
            return sessionStorage.getItem('um8_send_bearer') === '1';
        } catch (_) {
            return false;
        }
    }

    function getAccessToken() {
        if (!_bearerHeaderMode()) return '';
        let t = _firstValue(sessionStorage, ACCESS_ALIASES);
        if (t) return t;
        t = _firstValue(localStorage, ACCESS_ALIASES);
        if (t) {
            try {
                sessionStorage.setItem(TOKEN_KEY, t);
                _stripAuthFromLocalStorage();
            } catch (_) {}
            return t;
        }
        return '';
    }

    function getRefreshToken() {
        if (!_bearerHeaderMode()) return '';
        let t = _firstValue(sessionStorage, REFRESH_ALIASES);
        if (t) return t;
        t = _firstValue(localStorage, REFRESH_ALIASES);
        if (t) {
            try {
                sessionStorage.setItem(REFRESH_KEY, t);
                _stripAuthFromLocalStorage();
            } catch (_) {}
            return t;
        }
        return '';
    }

    const _SESSION_USER_KEY = 'uploadm8_cached_user';
    const _SESSION_USER_AT_KEY = 'uploadm8_cached_user_at';

    function _b64UrlDecodeUm8(segment) {
        let b = String(segment || '').replace(/-/g, '+').replace(/_/g, '/');
        while (b.length % 4) b += '=';
        try {
            return decodeURIComponent(
                Array.prototype.map.call(atob(b), function (c) {
                    return '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2);
                }).join('')
            );
        } catch (_) {
            try {
                return atob(b);
            } catch (_) {
                return null;
            }
        }
    }

    /** JWT `sub` — must match app.py create_access_jwt(str(user_id)). */
    function _jwtSubFromAccessToken(accessToken) {
        try {
            const parts = String(accessToken || '').split('.');
            if (parts.length < 2) return '';
            const json = _b64UrlDecodeUm8(parts[1]);
            if (!json) return '';
            const pl = JSON.parse(json);
            if (!pl || pl.sub == null) return '';
            return String(pl.sub).trim();
        } catch (_) {
            return '';
        }
    }

    /** After refresh, keep the same-user session snapshot + avatar; only advance tokenMarker. */
    function _patchSessionCacheTokenMarker(accessToken) {
        const marker = accessToken ? String(accessToken).slice(-24) : '';
        if (!marker) return;
        const newSub = _jwtSubFromAccessToken(accessToken);
        if (!newSub) return;
        let raw = '';
        try {
            raw = sessionStorage.getItem(_SESSION_USER_KEY) || '';
        } catch (_) {
            return;
        }
        if (!raw) return;
        let parsed;
        try {
            parsed = JSON.parse(raw);
        } catch (_) {
            return;
        }
        const uid = String(
            parsed.uid ||
                (parsed.user &&
                    (parsed.user.id || parsed.user.user_id || parsed.user.uid)) ||
                ''
        ).trim();
        if (!uid || uid !== newSub) return;
        parsed.tokenMarker = marker;
        try {
            sessionStorage.setItem(_SESSION_USER_KEY, JSON.stringify(parsed));
            sessionStorage.setItem(_SESSION_USER_AT_KEY, String(Date.now()));
        } catch (_) {}
    }

    function purgeStaleUserSnapshot() {
        try {
            window.__um8DomHydratedFromSession = false;
        } catch (_) {}
        try {
            sessionStorage.removeItem('uploadm8_cached_user');
            sessionStorage.removeItem('uploadm8_cached_user_at');
        } catch (_) {}
        try {
            localStorage.removeItem('uploadm8_cached_user');
            localStorage.removeItem('uploadm8_cached_user_at');
        } catch (_) {}
        try {
            sessionStorage.removeItem('uploadm8_avatar_ss_url');
            sessionStorage.removeItem('uploadm8_avatar_ss_sub');
        } catch (_) {}
        try {
            const st = window.__um8AuthState;
            if (st) {
                st.cachedUser = null;
                st.cachedUserAt = 0;
            }
        } catch (_) {}
    }

    /**
     * Session-only token storage. `remember` is ignored — kept for call-site compatibility.
     * Same-user access-token rotation (e.g. /api/auth/refresh) preserves uploadm8_cached_user
     * and avatar hints; only a different JWT sub or logout clears them (via purgeStaleUserSnapshot).
     */
    function _siteKey(hostname) {
        var h = String(hostname || '').toLowerCase();
        if (h === 'localhost' || h === '127.0.0.1') return h;
        var p = h.split('.');
        if (p.length < 2) return h;
        return p.slice(-2).join('.');
    }

    /** True when browser origin and API_BASE are different sites (e.g. localhost vs 127.0.0.1). */
    function _crossHostApi() {
        try {
            if (typeof window.API_BASE !== 'string' || !window.API_BASE) return false;
            var ah = new URL(window.API_BASE).hostname.toLowerCase();
            var lh = String(window.location.hostname || '').toLowerCase();
            if (ah === lh) return false;
            if (_siteKey(ah) === _siteKey(lh)) return false;
            return true;
        } catch (_) {
            return false;
        }
    }

    function setTokens(accessToken, refreshToken, remember = true) {
        let preserveProfileSnapshot = false;
        if (accessToken) {
            const newSub = _jwtSubFromAccessToken(accessToken);
            let raw = '';
            try {
                raw = sessionStorage.getItem(_SESSION_USER_KEY) || '';
            } catch (_) {}
            if (newSub && raw) {
                try {
                    const parsed = JSON.parse(raw);
                    const uid = String(
                        parsed.uid ||
                            (parsed.user &&
                                (parsed.user.id || parsed.user.user_id || parsed.user.uid)) ||
                            ''
                    ).trim();
                    if (uid && uid === newSub) preserveProfileSnapshot = true;
                } catch (_) {}
            }
            if (!preserveProfileSnapshot) {
                purgeStaleUserSnapshot();
            }
        }
        _stripAuthFromLocalStorage();
        ACCESS_ALIASES.forEach(function (k) {
            try {
                sessionStorage.removeItem(k);
            } catch (_) {}
        });
        REFRESH_ALIASES.forEach(function (k) {
            try {
                sessionStorage.removeItem(k);
            } catch (_) {}
        });
        try {
            sessionStorage.removeItem('um8_send_bearer');
        } catch (_) {}
        if (_crossHostApi()) {
            try {
                sessionStorage.setItem('um8_send_bearer', '1');
            } catch (_) {}
            if (accessToken) {
                try {
                    sessionStorage.setItem(TOKEN_KEY, accessToken);
                } catch (_) {}
            }
            if (refreshToken) {
                try {
                    sessionStorage.setItem(REFRESH_KEY, refreshToken);
                } catch (_) {}
            }
        }
        if (accessToken && preserveProfileSnapshot) {
            _patchSessionCacheTokenMarker(accessToken);
        }
    }

    function isLoggedIn() {
        if (getAccessToken()) return true;
        try {
            const raw = sessionStorage.getItem(_SESSION_USER_KEY);
            if (!raw) return false;
            const p = JSON.parse(raw);
            const marker = String((p && p.tokenMarker) || '');
            if (marker.indexOf('cookie_uid:') === 0) return true;
        } catch (_) {}
        return false;
    }

    window.generateRequestId = generateRequestId;
    window._isNetworkError = _isNetworkError;
    window._isDefinitiveAuthFailure = _isDefinitiveAuthFailure;
    window.fetchWithTimeout = fetchWithTimeout;
    window.escapeHTML = escapeHTML;
    window.escapeHtml = escapeHTML;
    window.um8EscapeHtml = escapeHTML;
    window.getAccessToken = getAccessToken;
    window.getRefreshToken = getRefreshToken;
    window.setTokens = setTokens;
    window.isLoggedIn = isLoggedIn;
    window._uploadm8StripLegacyLocalAuth = _stripAuthFromLocalStorage;
    /** True when app origin and API_BASE are same-site (HttpOnly cookies; no bearer in sessionStorage). */
    window._um8CookiePrimaryAuth = function () {
        return !_crossHostApi();
    };
    window.purgeStaleUserSnapshot = purgeStaleUserSnapshot;
})();

/* ── Login / refresh / apiCall (legacy standalone file removed; edit only here) ── */
/**
 * Auth flow helpers layered on top of token primitives above.
 */
(function () {
    const REFRESH_FETCH_TIMEOUT_MS = 10000;
    const DEFAULT_FETCH_TIMEOUT_MS = 15000;
    /** Upload.html poll: DB + optional R2 presign for thumbnail_url; must not use isLongRunning (offline gate). */
    const UPLOAD_STATUS_POLL_TIMEOUT_MS = 60000;
    /** Dashboard/catalog/wallet/sync jobs: remote DB + platform APIs often exceed 45s under load or debug. */
    const SYNC_FETCH_TIMEOUT_MS = 120000;
    /** Shared across dashboard + wallet HUD: pause background polls after connection failures. */
    const API_OFFLINE_COOLDOWN_MS = 60_000;
    /** Dedupe console noise when many background polls fail together (e.g. uvicorn --reload). */
    const API_NOISE_LOG_WINDOW_MS = 12_000;
    /** Coalesce concurrent identical GETs (catalog widget + page + wallet) so one worker is not hit twice for the same URL. */
    const _inFlightGetDedupe = new Map();
    let _apiNoiseLogKey = '';
    let _apiNoiseLogAt = 0;

    function _apiLogNoiseOnce(key, logFn) {
        const now = Date.now();
        if (_apiNoiseLogKey === key && now - _apiNoiseLogAt < API_NOISE_LOG_WINDOW_MS) return;
        _apiNoiseLogKey = key;
        _apiNoiseLogAt = now;
        logFn();
    }

    function _isFetchNetworkFailure(e) {
        if (!e) return false;
        if (e.name === 'AbortError' || e.name === 'TimeoutError') return false;
        if (e instanceof TypeError) return true;
        const m = String(e.message || '');
        return /failed to fetch|load failed|network error|ECONNREFUSED|connection refused/i.test(m);
    }

    function _bumpUploadM8ApiOfflineCooldown(ms) {
        const add = ms == null ? API_OFFLINE_COOLDOWN_MS : ms;
        try {
            const now = Date.now();
            const cur =
                typeof window.__uploadm8ApiOfflineUntil === 'number'
                    ? window.__uploadm8ApiOfflineUntil
                    : 0;
            window.__uploadm8ApiOfflineUntil = Math.max(cur, now + add);
        } catch (_) {}
    }

    function _clearUploadM8ApiOfflineCooldown() {
        try {
            window.__uploadm8ApiOfflineUntil = 0;
        } catch (_) {}
    }

    /**
     * After login/refresh: same-site uses HttpOnly cookies only — do not keep JWT/refresh in JS storage.
     * Cross-host dev keeps bearer tokens in sessionStorage (um8_send_bearer).
     */
    function applyUploadM8AuthSession(access, refresh, remember) {
        remember = remember !== false;
        if (typeof window._um8CookiePrimaryAuth === 'function' && window._um8CookiePrimaryAuth()) {
            try {
                if (typeof window.purgeStaleUserSnapshot === 'function') window.purgeStaleUserSnapshot();
            } catch (_) {}
            window.setTokens('', '', remember);
            return;
        }
        window.setTokens(access || '', refresh || '', remember);
    }

    window.applyUploadM8AuthSession = applyUploadM8AuthSession;

    /**
     * Serialize refresh: the server rotates one refresh token per success. Parallel POST /refresh
     * (e.g. several 401s from Promise.all on wallet + stats + analytics, or HUD poll overlap)
     * makes the second request reuse a revoked hash → 401 "Reuse detected" → clearTokens() and
     * spurious 401s on /api/wallet in DevTools even though the first refresh succeeded.
     */
    let _refreshInFlight = null;

    async function _tryRefreshTokenOnce() {
        const refreshToken = window.getRefreshToken();
        const payloads = refreshToken
            ? [{ refresh_token: refreshToken }, { refreshToken: refreshToken }]
            : [{}];

        try {
            for (const payload of payloads) {
                const requestId = window.generateRequestId();
                let resp;
                try {
                    resp = await window.fetchWithTimeout(`${window.API_BASE}/api/auth/refresh`, {
                        method: 'POST',
                        credentials: 'include',
                        headers: { 'Content-Type': 'application/json', 'X-Request-ID': requestId },
                        body: JSON.stringify(payload),
                    }, REFRESH_FETCH_TIMEOUT_MS);
                } catch (fetchErr) {
                    console.warn('[Auth] Token refresh network error:', fetchErr.message);
                    return false;
                }

                if (resp.ok) {
                    const data = await resp.json().catch(() => ({}));
                    if (!data.access_token && !data.accessToken) {
                        console.warn('[Auth] Refresh response missing access_token');
                        return false;
                    }
                    applyUploadM8AuthSession(
                        data.access_token || data.accessToken,
                        data.refresh_token || data.refreshToken || refreshToken || '',
                        true
                    );
                    return true;
                }

                if (resp.status === 422) continue;
                if (window._isDefinitiveAuthFailure(resp.status)) {
                    if (typeof window.clearTokens === 'function') window.clearTokens();
                }
                return false;
            }

            if (typeof window.clearTokens === 'function') window.clearTokens();
            return false;
        } catch (e) {
            if (!window._isNetworkError(e) && typeof window.clearTokens === 'function') {
                window.clearTokens();
            }
            console.error('[Auth] tryRefreshToken unexpected error:', e);
            return false;
        }
    }

    async function tryRefreshToken() {
        if (_refreshInFlight) {
            return _refreshInFlight;
        }
        _refreshInFlight = (async () => {
            try {
                return await _tryRefreshTokenOnce();
            } finally {
                _refreshInFlight = null;
            }
        })();
        return _refreshInFlight;
    }

    async function login(email, password, remember = true) {
        const requestId = window.generateRequestId();
        try {
            const resp = await window.fetchWithTimeout(`${window.API_BASE}/api/auth/login`, {
                method: 'POST',
                credentials: 'include',
                headers: { 'Content-Type': 'application/json', 'X-Request-ID': requestId },
                body: JSON.stringify({ email, password }),
            }, DEFAULT_FETCH_TIMEOUT_MS);

            const data = await resp.json().catch(() => ({}));
            if (!resp.ok) {
                const msg =
                    (typeof window.uploadM8ApiErrorMessage === 'function' && window.uploadM8ApiErrorMessage(data)) ||
                    'Invalid email or password';
                return { success: false, error: msg, requestId };
            }
            applyUploadM8AuthSession(data.access_token, data.refresh_token, remember);
            if (data.must_reset_password) {
                try {
                    sessionStorage.setItem('uploadm8_must_reset_password', '1');
                } catch (e) {}
                window.location.href = 'settings.html?password_reset_required=1#security';
                return { success: true, requestId, must_reset_password: true };
            }
            return { success: true, requestId };
        } catch (e) {
            console.error(`Login error [${requestId}]:`, e);
            return {
                success: false,
                error: e.name === 'TimeoutError'
                    ? 'Connection timed out. Please try again.'
                    : 'Connection failed. Please check your internet.',
                requestId,
            };
        }
    }

    async function logout() {
        window.fetchWithTimeout(`${window.API_BASE}/api/auth/logout`, {
            method: 'POST',
            credentials: 'include',
            headers: {
                'Content-Type': 'application/json',
                'X-Request-ID': window.generateRequestId(),
            },
            body: JSON.stringify({}),
        }, 8000).catch(() => {});
        if (typeof window.clearTokens === 'function') window.clearTokens();
        window.location.href = 'login.html';
    }

    async function logoutAll() {
        try {
            await window.fetchWithTimeout(`${window.API_BASE}/api/auth/logout-all`, {
                method: 'POST',
                credentials: 'include',
                headers: { 'Content-Type': 'application/json', 'X-Request-ID': window.generateRequestId() },
                body: JSON.stringify({}),
            }, 8000);
        } catch (_) {}
        if (typeof window.clearTokens === 'function') window.clearTokens();
        window.location.href = 'login.html';
    }

    async function register(name, email, password) {
        const requestId = window.generateRequestId();
        try {
            const resp = await window.fetchWithTimeout(`${window.API_BASE}/api/auth/register`, {
                method: 'POST',
                credentials: 'include',
                headers: { 'Content-Type': 'application/json', 'X-Request-ID': requestId },
                body: JSON.stringify({ name, email, password }),
            }, DEFAULT_FETCH_TIMEOUT_MS);

            const data = await resp.json().catch(() => ({}));
            if (!resp.ok) {
                const msg =
                    (typeof window.uploadM8ApiErrorMessage === 'function' && window.uploadM8ApiErrorMessage(data)) ||
                    'Registration failed';
                return { success: false, error: msg, requestId };
            }
            if (data.status === 'pending_verification') {
                if (typeof window.clearTokens === 'function') window.clearTokens();
                return {
                    success: true,
                    email: data.email || '',
                    pendingVerification: true,
                    requestId,
                };
            }
            return { success: true, email: data.email || '', requestId };
        } catch (e) {
            console.error(`Registration error [${requestId}]:`, e);
            return {
                success: false,
                error: e.name === 'TimeoutError'
                    ? 'Connection timed out. Please try again.'
                    : 'Connection failed. Please try again.',
                requestId,
            };
        }
    }

    /**
     * Low-level authorized fetch to API_BASE + endpoint. Pass through authRedirectOn401:false to skip
     * refresh + login redirect (e.g. login page cookie probe).
     */
    async function apiFetch(endpoint, userOptions = {}, _retryDepth = 0) {
        const authRedirectOn401 = userOptions.authRedirectOn401 !== false;
        const skipGetDedupe = userOptions.skipGetDedupe === true;
        const optTimeoutMs = userOptions.timeoutMs;
        const options = { ...userOptions };
        delete options.authRedirectOn401;
        delete options.timeoutMs;
        delete options.skipGetDedupe;

        const token = window.getAccessToken();
        const requestId = window.generateRequestId();

        const hdrs = {
            'Content-Type': 'application/json',
            'X-Request-ID': requestId,
            ...options.headers,
        };
        if (token) {
            hdrs['Authorization'] = `Bearer ${token}`;
        }

        const config = {
            ...options,
            credentials: 'include',
            cache: 'no-store',
            headers: hdrs,
        };
        if (options.body instanceof FormData) delete config.headers['Content-Type'];

        const ep = String(endpoint || '');
        const epPath = ep.split('?')[0];
        const methodUpper = String(config.method || 'GET').toUpperCase();
        /** Thumbnail Studio recreate runs GPT + optional Pikzels per variant — often >15s locally. */
        const isThumbnailStudioRecreatePost =
            methodUpper === 'POST' && epPath === '/api/thumbnail-studio/recreate';
        // GET /api/uploads/{uuid} only (upload status poll). Not list, not /complete, etc.
        const isUploadRecordStatusGet =
            methodUpper === 'GET' && /^\/api\/uploads\/[^/]+$/.test(epPath);
        // Heavy reads: analytics sync jobs, dashboard aggregates, bulk upload lists (limit=500+).
        // Note: GET /api/uploads/{uuid} is NOT isLongRunning — that tied it to the global offline gate
        // after ERR_CONNECTION_RESET. It still needs a timeout > DEFAULT when the API is busy.
        const isLongRunning =
            ep.indexOf('/sync-analytics') >= 0 ||
            ep.indexOf('/analytics/refresh-all') >= 0 ||
            ep.indexOf('/analytics/platform-metrics') >= 0 ||
            ep.indexOf('/api/dashboard/stats') >= 0 ||
            epPath.indexOf('/api/shell') === 0 ||
            epPath.indexOf('/api/analytics') === 0 ||
            epPath.indexOf('/api/catalog') === 0 ||
            epPath.indexOf('/api/wallet') === 0 ||
            epPath === '/api/platform-accounts' ||
            epPath.indexOf('/api/entitlements') === 0 ||
            epPath === '/api/me/coach' ||
            // GET /api/uploads?… list only; skip /api/uploads/{id}/… presign and actions
            (ep.indexOf('/api/uploads') >= 0 && ep.indexOf('/api/uploads/') < 0);
        const customLong =
            typeof window.__UPLOADM8_LONG_FETCH_TIMEOUT_MS === 'number' &&
            window.__UPLOADM8_LONG_FETCH_TIMEOUT_MS > 0
                ? window.__UPLOADM8_LONG_FETCH_TIMEOUT_MS
                : null;
        const longMs = customLong || SYNC_FETCH_TIMEOUT_MS;
        const thumbRecreateMs = Math.max(longMs, 240000);
        const pollMs =
            typeof window.__UPLOADM8_UPLOAD_POLL_TIMEOUT_MS === 'number' &&
            window.__UPLOADM8_UPLOAD_POLL_TIMEOUT_MS > 0
                ? window.__UPLOADM8_UPLOAD_POLL_TIMEOUT_MS
                : UPLOAD_STATUS_POLL_TIMEOUT_MS;
        const timeoutMs = Number(
            optTimeoutMs ||
                (isThumbnailStudioRecreatePost
                    ? thumbRecreateMs
                    : isLongRunning
                      ? longMs
                      : isUploadRecordStatusGet
                        ? pollMs
                        : DEFAULT_FETCH_TIMEOUT_MS)
        );

        const offlineUntil =
            typeof window.__uploadm8ApiOfflineUntil === 'number'
                ? window.__uploadm8ApiOfflineUntil
                : 0;
        if (isLongRunning && Date.now() < offlineUntil) {
            const err = new Error('API temporarily unreachable');
            err.name = 'UploadM8ApiOffline';
            err.requestId = requestId;
            throw err;
        }

        const fullUrl = `${window.API_BASE}${endpoint}`;
        const authKey = token || '';

        async function doNetworkFetch() {
            let resp;
            try {
                resp = await window.fetchWithTimeout(fullUrl, config, timeoutMs);
            } catch (e) {
                if (e.name === 'TimeoutError') {
                    const err = new Error(`Request timed out: ${endpoint}`);
                    err.name = 'TimeoutError';
                    err.status = 408;
                    err.requestId = requestId;
                    if (isLongRunning) {
                        _bumpUploadM8ApiOfflineCooldown(Math.min(API_OFFLINE_COOLDOWN_MS, 30_000));
                        _apiLogNoiseOnce(`to:${epPath}`, () =>
                            console.warn(`API timeout (background) [${requestId}] ${endpoint}`)
                        );
                    } else {
                        console.error(`API Timeout [${requestId}] ${endpoint}`);
                    }
                    throw err;
                }
                if (!e.requestId) e.requestId = requestId;
                const unreachable = _isFetchNetworkFailure(e);
                if (unreachable) {
                    _bumpUploadM8ApiOfflineCooldown();
                    if (isLongRunning) {
                        _apiLogNoiseOnce(`un:${epPath}`, () =>
                            console.warn(`API unreachable (background) [${requestId}] ${endpoint}`)
                        );
                    } else {
                        console.warn(`API unreachable [${requestId}] ${endpoint}`);
                    }
                } else console.error(`API Error [${requestId}] ${endpoint}:`, e);
                throw e;
            }

            _clearUploadM8ApiOfflineCooldown();

            if (resp.status === 401) {
                if (!authRedirectOn401) {
                    return resp;
                }
                if (_retryDepth > 0) {
                    if (typeof window.clearTokens === 'function') window.clearTokens();
                    sessionStorage.setItem('uploadm8_auth_message', 'Session expired. Please log in again.');
                    window.location.href = 'login.html';
                    throw new Error('Session expired');
                }
                const refreshed = await tryRefreshToken();
                if (refreshed) {
                    return apiFetch(endpoint, { ...userOptions, authRedirectOn401 }, _retryDepth + 1);
                }
                if (typeof window.clearTokens === 'function') window.clearTokens();
                sessionStorage.setItem('uploadm8_auth_message', 'Session expired. Please log in again.');
                window.location.href = 'login.html';
                throw new Error('Session expired');
            }

            if (resp.status === 403 && authRedirectOn401) {
                let body = {};
                try {
                    body = await resp.clone().json();
                } catch (_) {}
                const d = body.detail;
                const code = (typeof d === 'object' && d && d.code) || body.code;
                if (code === 'email_not_verified') {
                    if (typeof window.clearTokens === 'function') window.clearTokens();
                    try {
                        sessionStorage.setItem('uploadm8_auth_message', 'Please verify your email to continue.');
                    } catch (_) {}
                    window.location.href = 'check-email.html';
                    throw new Error('Email not verified');
                }
            }

            return resp;
        }

        const canDedupe =
            !skipGetDedupe &&
            _retryDepth === 0 &&
            methodUpper === 'GET' &&
            config.body == null &&
            !(options.body instanceof FormData);
        if (canDedupe) {
            const dedupeKey = methodUpper + '\0' + fullUrl + '\0' + authKey + '\0' + String(timeoutMs);
            let shared = _inFlightGetDedupe.get(dedupeKey);
            if (!shared) {
                shared = doNetworkFetch().finally(() => {
                    if (_inFlightGetDedupe.get(dedupeKey) === shared) _inFlightGetDedupe.delete(dedupeKey);
                });
                _inFlightGetDedupe.set(dedupeKey, shared);
            }
            const r = await shared;
            return r.clone();
        }

        return doNetworkFetch();
    }

    async function apiCall(endpoint, options = {}) {
        const requestId = window.generateRequestId();
        try {
            const resp = await apiFetch(endpoint, { ...options, authRedirectOn401: true });

            if (!resp.ok) {
                const err = await resp.json().catch(() => ({ detail: 'Request failed' }));
                const parsed =
                    typeof window.uploadM8ApiErrorMessage === 'function'
                        ? window.uploadM8ApiErrorMessage(err)
                        : '';
                const error = new Error(
                    parsed || (typeof err.detail === 'string' ? err.detail : '') || `API Error: ${resp.status}`
                );
                error.status = resp.status;
                error.requestId = requestId;
                error.response = err;
                throw error;
            }

            const text = await resp.text();
            return text ? JSON.parse(text) : {};
        } catch (e) {
            if (e.message === 'Session expired' || e.message === 'Email not verified') throw e;
            if (e.name === 'TimeoutError') throw e;
            if (e.name === 'UploadM8ApiOffline') throw e;
            if (!e.requestId) e.requestId = requestId;
            // Network unreachable: apiFetch already logged + bumped __uploadm8ApiOfflineUntil.
            if (!_isFetchNetworkFailure(e)) {
                console.error(`API Error [${requestId}] ${endpoint}:`, e);
            }
            throw e;
        }
    }

    window.tryRefreshToken = tryRefreshToken;
    window.login = login;
    window.logout = logout;
    window.logoutAll = logoutAll;
    window.register = register;
    window.apiFetch = apiFetch;
    window.apiCall = apiCall;
})();

/* ── checkAuth + session refresh (formerly js/auth-check.js) ── */
(function () {
    const AUTH_FETCH_TIMEOUT_MS = 15000;
    /** In-memory only — short window to dedupe parallel checkAuth; token change purges via setTokens. */
    const USER_CACHE_TTL = 12000;
    /**
     * Same-tab sessionStorage snapshot: trust for an instant shell on full page loads.
     * 20m was too tight — users saw a blocking /api/me on almost every navigation once the snapshot aged out.
     * We still call /api/me in the background whenever the snapshot is older than USER_CACHE_TTL.
     */
    const SESSION_SNAPSHOT_COMFORT_MS = 12 * 60 * 60 * 1000;
    const PUBLIC_PAGES = [
        'index.html', 'login.html', 'signup.html', 'forgot-password.html',
        'terms.html', 'privacy.html', 'about.html', 'contact.html',
        'support.html', 'blog.html', 'how-it-works.html', 'data-deletion.html',
        'walkthrough.html', 'check-email.html', 'confirm-email.html', 'reset-password.html',
        'verify-email.html', 'unsubscribe.html', ''
    ];

    function state() {
        return (window.__um8AuthState = window.__um8AuthState || {
            cachedUser: null,
            cachedUserAt: 0,
            currentUser: null,
            isAuthChecking: false,
            authCheckPromise: null,
        });
    }

    function setCurrentUser(user) {
        if (typeof window._setCurrentUserState === 'function') {
            window._setCurrentUserState(user);
        } else {
            window.currentUser = user;
            state().currentUser = user;
        }
    }

    /** Paint sidebar name/tier/avatar + html.um8-user-ready (app.js). Safe after all scripts load. */
    function notifyUserChromeReady(user, chromeOpts) {
        if (!user || !user.email) return;
        try {
            if (typeof window.updateUserUI === 'function') {
                window.updateUserUI(Object.assign({ walletBroadcast: true }, chromeOpts || {}));
            }
        } catch (_) {}
    }

    function isOnAuthPage() {
        const path = window.location.pathname;
        const page = path.split('/').pop() || 'index.html';
        return PUBLIC_PAGES.includes(page) || path.endsWith('/');
    }

    function _cookiePrimarySession() {
        return typeof window._um8CookiePrimaryAuth === 'function' && window._um8CookiePrimaryAuth();
    }

    async function _refreshUserInBackground() {
        const token = window.getAccessToken();
        if (!token && !_cookiePrimarySession()) return;
        try {
            const headers = { 'Content-Type': 'application/json' };
            if (token) headers['Authorization'] = 'Bearer ' + token;
            const resp = await window.fetchWithTimeout(
                `${window.API_BASE}/api/me`,
                { headers: headers, credentials: 'include' },
                AUTH_FETCH_TIMEOUT_MS
            );
            if (!resp.ok) return;
            let user = await resp.json();
            user = window._normalizeUserPayload(user);
            if (!user || !user.email) return;
            setCurrentUser(user);
            state().cachedUser = user;
            state().cachedUserAt = Date.now();
            window._writeSessionCache(user);
            if (typeof window.updateUserUI === 'function') {
                window.updateUserUI();
            }
        } catch (_) {}
    }

    async function _fetchUserFromAPI(token, options = {}, retryDepth = 0) {
        const { redirectOnFail = true, silent = false } = options;
        const requestId = window.generateRequestId();
        try {
            const headers = {
                'X-Request-ID': requestId,
                'Content-Type': 'application/json',
            };
            if (token) headers['Authorization'] = 'Bearer ' + token;
            const resp = await window.fetchWithTimeout(
                `${window.API_BASE}/api/me`,
                { headers: headers, credentials: 'include' },
                AUTH_FETCH_TIMEOUT_MS
            );

            if (resp.status === 401) {
                if (retryDepth > 0) {
                    if (typeof window.clearTokens === 'function') window.clearTokens();
                    if (redirectOnFail && !isOnAuthPage()) {
                        sessionStorage.setItem('uploadm8_auth_message', 'Session expired. Please log in again.');
                        window.location.href = 'login.html';
                    }
                    return null;
                }
                const refreshed = await window.tryRefreshToken();
                if (refreshed) {
                    state().isAuthChecking = false;
                    state().authCheckPromise = null;
                    return checkAuth(options, 1);
                }
                if (typeof window.clearTokens === 'function') window.clearTokens();
                if (redirectOnFail && !isOnAuthPage()) {
                    sessionStorage.setItem('uploadm8_auth_message', 'Session expired. Please log in again.');
                    window.location.href = 'login.html';
                }
                return null;
            }

            if (resp.status === 403) {
                let body = {};
                try {
                    body = await resp.clone().json();
                } catch (_) {}
                const d = body.detail;
                const code = (typeof d === 'object' && d && d.code) || body.code;
                if (code === 'email_not_verified') {
                    if (typeof window.clearTokens === 'function') window.clearTokens();
                    if (redirectOnFail && !isOnAuthPage()) {
                        try {
                            sessionStorage.setItem('uploadm8_auth_message', 'Please verify your email to continue.');
                        } catch (_) {}
                        window.location.href = 'check-email.html';
                    }
                    return null;
                }
            }

            if (!resp.ok) {
                console.error(`Auth check failed: ${resp.status} [${requestId}]`);
                if (window._isDefinitiveAuthFailure(resp.status)) {
                    if (typeof window.clearTokens === 'function') window.clearTokens();
                    if (redirectOnFail && !isOnAuthPage()) window.location.href = 'login.html';
                }
                return null;
            }

            let user = await resp.json();
            user = window._normalizeUserPayload(user);
            if (!user || !user.email) throw new Error('Invalid user data received');

            setCurrentUser(user);
            state().cachedUser = user;
            state().cachedUserAt = Date.now();
            window._writeSessionCache(user);
            // Sidebar paint only — initApp() (or the page) broadcasts uploadm8:user when wallet HUD should attach.
            notifyUserChromeReady(user, { walletBroadcast: false });
            return user;
        } catch (e) {
            console.error('Auth check error:', e);
            if (window._isNetworkError(e)) {
                const fallback = window._readSessionCache && window._readSessionCache();
                if (fallback && fallback.user && typeof window._sessionUserSnapshotTrusted === 'function' &&
                    window._sessionUserSnapshotTrusted()) {
                    console.warn('[Auth] Network error — using same-session user snapshot');
                    setCurrentUser(fallback.user);
                    state().cachedUser = fallback.user;
                    state().cachedUserAt = Date.now() - USER_CACHE_TTL + 5000;
                    return fallback.user;
                }
                if (!silent && redirectOnFail && !isOnAuthPage()) {
                    sessionStorage.setItem('uploadm8_auth_message',
                        e.name === 'TimeoutError'
                            ? 'Connection timed out. Please try again.'
                            : 'Connection error. Please try again.');
                    window.location.href = 'login.html';
                }
                return null;
            }
            if (!silent && redirectOnFail && !isOnAuthPage()) {
                sessionStorage.setItem('uploadm8_auth_message', 'Authentication error. Please log in again.');
                window.location.href = 'login.html';
            }
            return null;
        } finally {
            state().isAuthChecking = false;
            state().authCheckPromise = null;
        }
    }

    async function checkAuth(options = {}, _retryDepth = 0) {
        const { redirectOnFail = true, forceRefresh = false, blockUntilValidated = false } = options;
        if (state().isAuthChecking && state().authCheckPromise) return state().authCheckPromise;

        if (!forceRefresh && !blockUntilValidated && state().cachedUser && state().cachedUserAt) {
            const age = Date.now() - state().cachedUserAt;
            if (age >= 0 && age < SESSION_SNAPSHOT_COMFORT_MS) {
                setCurrentUser(state().cachedUser);
                notifyUserChromeReady(state().cachedUser, { walletBroadcast: false });
                if (age >= USER_CACHE_TTL) void _refreshUserInBackground();
                return state().cachedUser;
            }
        }

        const token = window.getAccessToken();
        const useCookiesOnly = _cookiePrimarySession() && !token;
        if (!token && !useCookiesOnly) {
            if (redirectOnFail && !isOnAuthPage()) {
                sessionStorage.setItem('uploadm8_auth_message', 'Please log in to continue.');
                window.location.href = 'login.html';
            }
            return null;
        }

        // Cookie-primary (or late app.js): session snapshot may exist even when hydrate IIFE skipped.
        if (
            !forceRefresh &&
            !blockUntilValidated &&
            _retryDepth === 0 &&
            typeof window._readSessionCache === 'function'
        ) {
            const snap = window._readSessionCache();
            if (
                snap &&
                snap.user &&
                snap.user.email &&
                snap.age >= 0 &&
                snap.age < SESSION_SNAPSHOT_COMFORT_MS
            ) {
                setCurrentUser(snap.user);
                state().cachedUser = snap.user;
                state().cachedUserAt = Date.now() - snap.age;
                notifyUserChromeReady(snap.user, { walletBroadcast: false });
                void _refreshUserInBackground();
                return snap.user;
            }
        }

        state().isAuthChecking = true;
        state().authCheckPromise = _fetchUserFromAPI(useCookiesOnly ? null : token, options, _retryDepth);
        return state().authCheckPromise;
    }

    window.isOnAuthPage = isOnAuthPage;
    window._refreshUserInBackground = _refreshUserInBackground;
    window.checkAuth = checkAuth;
    window.__UM8_SESSION_COMFORT_MS = SESSION_SNAPSHOT_COMFORT_MS;

    window.addEventListener('pageshow', function (ev) {
        if (!ev.persisted) return;
        const st = state();
        st.cachedUser = null;
        st.cachedUserAt = 0;
        if (
            (window.getAccessToken && window.getAccessToken()) ||
            (typeof window._um8CookiePrimaryAuth === 'function' && window._um8CookiePrimaryAuth())
        ) {
            void _refreshUserInBackground();
        }
    });

    document.addEventListener('visibilitychange', function () {
        if (document.visibilityState !== 'visible') return;
        if (isOnAuthPage()) return;
        var hasBearer = window.getAccessToken && window.getAccessToken();
        var cookiePri = typeof window._um8CookiePrimaryAuth === 'function' && window._um8CookiePrimaryAuth();
        if (!hasBearer && !cookiePri) return;
        const st = state();
        const age = st.cachedUser ? Date.now() - st.cachedUserAt : USER_CACHE_TTL;
        if (age >= USER_CACHE_TTL) {
            void _refreshUserInBackground();
        }
    });
})();
