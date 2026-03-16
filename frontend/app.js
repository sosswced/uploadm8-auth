/**
 * UploadM8 v3.2.0 — Production Application JavaScript
 *
 * Key improvements over v3.1.1:
 *  - Instant UI hydration from sessionStorage cache — zero flash on page load
 *  - /api/me called at most ONCE per session; background-refreshed silently
 *  - Token clearing only on definitive auth failures, never on network blips
 *  - Bounded retry depth (max 1 auto-retry) in both checkAuth and apiCall
 *  - Sidebar toggle safe before DOMContentLoaded via pending-call queue
 *  - Single unified hide-figures system (CSS .blur-value class)
 *  - Dead handleUnauthorized branch removed
 *  - getUploadStats added to window exports
 */

// ============================================================
// Configuration
// ============================================================
if (!window.API_BASE && /^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/i.test(location.origin)) {
    window.API_BASE = 'http://127.0.0.1:8000';
}
const API_BASE = window.API_BASE || 'https://auth.uploadm8.com';
const APP_VERSION  = '3.2.1';

const TOKEN_KEY      = 'uploadm8_access_token';
const REFRESH_KEY    = 'uploadm8_refresh_token';
// Avatar URL persisted to localStorage — survives page refreshes and browser restarts.
// Lets us paint the correct avatar before any network call fires.
const _AVATAR_URL_KEY  = 'uploadm8_avatar_url';

// Timeout constants
const DEFAULT_FETCH_TIMEOUT_MS = 15000;
const REFRESH_FETCH_TIMEOUT_MS = 10000;
const AUTH_FETCH_TIMEOUT_MS    = 15000;

// Session cache — user object persists across page navigations until logout
const _SESSION_CACHE_KEY    = 'uploadm8_cached_user';
const _SESSION_CACHE_AT_KEY = 'uploadm8_cached_user_at';
const _SESSION_CACHE_MAX_AGE = 10 * 60 * 1000; // 10 min — background refresh after this

// In-memory cache — prevents duplicate /api/me calls within a single page load
let _cachedUser   = null;
let _cachedUserAt = 0;
const _USER_CACHE_TTL = 60000; // 1 min in-memory hard cap

// Auth state
let currentUser        = null;
let isAuthChecking     = false;
let authCheckPromise   = null;

// ============================================================
// INSTANT UI HYDRATION — runs synchronously before DOMContentLoaded
// Applies cached user data to the DOM the moment the script executes,
// eliminating the "User / Free" flash seen on every page load.
// ============================================================
(function hydrateFromCache() {
    const token = localStorage.getItem(TOKEN_KEY) || sessionStorage.getItem(TOKEN_KEY);
    if (!token) return;

    // ── Phase 1: Paint avatar from localStorage BEFORE sessionStorage user loads.
    //    This runs synchronously — the browser renders the background-image from its
    //    HTTP cache with zero latency, so the avatar is visible on the very first frame.
    try {
        const cachedAvatarUrl = localStorage.getItem(_AVATAR_URL_KEY);
        if (cachedAvatarUrl) {
            // Inject a <style> tag into <head> (or <html> if head not ready yet).
            // background-image renders instantly from browser cache — no img element flash.
            const style = document.createElement('style');
            style.id = 'um8-avatar-pre';
            // Silence initial letter (raw text AND .um8-initial span) while
                // background-image shows — zero flash before img.onload fires.
                style.textContent =
                    '#userAvatar{background-image:url("' +
                    cachedAvatarUrl.replace(/"/g,'\"') +
                    '");background-size:cover;background-position:center;' +
                    'font-size:0!important;color:transparent!important;}' +
                    '#userAvatar .um8-initial,#userAvatar>span{visibility:hidden!important;}';
            (document.head || document.documentElement).appendChild(style);
        }
    } catch (_) {}

    // ── Phase 2: Hydrate full user object from sessionStorage.
    try {
        const raw      = sessionStorage.getItem(_SESSION_CACHE_KEY);
        const cachedAt = sessionStorage.getItem(_SESSION_CACHE_AT_KEY);
        if (!raw || !cachedAt) return;

        const age = Date.now() - parseInt(cachedAt, 10);
        if (age > _SESSION_CACHE_MAX_AGE * 2) return; // stale beyond 20 min

        const user = JSON.parse(raw);
        if (!user || !user.email || !user.role) return;

        currentUser        = user;
        window.currentUser = user;
        _cachedUser        = user;
        _cachedUserAt      = Date.now();

        _applyUserToDOM(user);
        try { window.dispatchEvent(new CustomEvent('uploadm8:user', { detail: user })); } catch (_) {}
    } catch (_) {}
})();

// ============================================================
// Theme — applied synchronously to prevent FOUC
// ============================================================
function getTheme() {
    return localStorage.getItem('uploadm8_theme') || 'dark';
}

function setTheme(theme) {
    localStorage.setItem('uploadm8_theme', theme);
    document.documentElement.setAttribute('data-theme', theme);
    document.body.classList.toggle('light-mode', theme === 'light');
    document.body.classList.toggle('dark-mode',  theme !== 'light');
    document.querySelectorAll(
        '#themeToggleIcon, #themeToggleIconDesktop, #themeToggleIconMobile'
    ).forEach(el => { if (el) el.className = theme === 'dark' ? 'fas fa-moon' : 'fas fa-sun'; });
}

function toggleTheme() { setTheme(getTheme() === 'dark' ? 'light' : 'dark'); }

// Apply theme immediately — before any HTML renders
(function () { document.documentElement.setAttribute('data-theme', getTheme()); })();

// ============================================================
// Fetch helpers
// ============================================================
function generateRequestId() {
    return 'req_' + Date.now().toString(36) + Math.random().toString(36).substr(2, 9);
}

function _isNetworkError(e) {
    // True for transient blips: timeouts, aborts, DNS failures, offline
    return (
        e.name === 'TimeoutError'  ||
        e.name === 'AbortError'    ||
        e instanceof TypeError     || // fetch() throws TypeError on network failure
        e.message === 'Failed to fetch'
    );
}

function _isDefinitiveAuthFailure(status) {
    // Only 401/403 warrant clearing stored tokens
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
            err.name      = 'TimeoutError';
            err.status    = 408;
            err.requestId = rid;
            throw err;
        }
        throw e;
    } finally {
        clearTimeout(tid);
    }
}

// ============================================================
// Token management
// ============================================================
function escapeHTML(value) {
    if (value === null || value === undefined) return '';
    return String(value)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function getAccessToken() {
    return localStorage.getItem(TOKEN_KEY) || sessionStorage.getItem(TOKEN_KEY);
}

function getRefreshToken() {
    const primary = localStorage.getItem(REFRESH_KEY) || sessionStorage.getItem(REFRESH_KEY);
    if (primary) return primary;
    return (
        localStorage.getItem('refreshToken')    ||
        localStorage.getItem('refresh_token')   ||
        sessionStorage.getItem('refreshToken')  ||
        sessionStorage.getItem('refresh_token') ||
        ''
    );
}

function setTokens(accessToken, refreshToken, remember = true) {
    const store = remember ? localStorage : sessionStorage;
    const other = remember ? sessionStorage : localStorage;

    // Write to intended store
    if (accessToken) {
        const accessKeys = [TOKEN_KEY, 'accessToken', 'access_token', 'authToken', 'auth_token', 'token'];
        accessKeys.forEach(k => { try { store.setItem(k, accessToken); } catch (_) {} });
    }
    if (refreshToken) {
        [REFRESH_KEY, 'refreshToken', 'refresh_token'].forEach(k => {
            try { store.setItem(k, refreshToken); } catch (_) {}
        });
    }

    // Remove stale copies from the other store to prevent split-brain
    try {
        [TOKEN_KEY, REFRESH_KEY, 'accessToken', 'access_token', 'authToken',
         'auth_token', 'token', 'refreshToken', 'refresh_token'
        ].forEach(k => { try { other.removeItem(k); } catch (_) {} });
    } catch (_) {}
}

function clearTokens() {
    _cachedUser   = null;
    _cachedUserAt = 0;
    currentUser        = null;
    window.currentUser = null;

    try { sessionStorage.removeItem(_SESSION_CACHE_KEY);    } catch (_) {}
    try { sessionStorage.removeItem(_SESSION_CACHE_AT_KEY); } catch (_) {}

    const keys = [
        TOKEN_KEY, REFRESH_KEY,
        _AVATAR_URL_KEY,
        'accessToken', 'access_token', 'authToken', 'auth_token', 'token',
        'refreshToken', 'refresh_token'
    ];
    keys.forEach(k => {
        try { localStorage.removeItem(k);   } catch (_) {}
        try { sessionStorage.removeItem(k); } catch (_) {}
    });
}

function isLoggedIn() { return !!getAccessToken(); }

// ============================================================
// Session cache helpers
// ============================================================
function _readSessionCache() {
    try {
        const raw = sessionStorage.getItem(_SESSION_CACHE_KEY);
        const at  = sessionStorage.getItem(_SESSION_CACHE_AT_KEY);
        if (!raw || !at) return null;
        const user = JSON.parse(raw);
        if (!user || !user.email || !user.role) return null;
        return { user, age: Date.now() - parseInt(at, 10) };
    } catch (_) { return null; }
}

function _writeSessionCache(user) {
    try {
        sessionStorage.setItem(_SESSION_CACHE_KEY,    JSON.stringify(user));
        sessionStorage.setItem(_SESSION_CACHE_AT_KEY, String(Date.now()));
    } catch (_) {}
}

// ============================================================
// Core: apply user data to DOM
// Called both from the synchronous hydration IIFE (pre-DOM) and
// from updateUserUI (post-auth). Safe to call multiple times.
// ============================================================
// _setAvatarInitial — renders the fallback initial letter into an avatar element.
// Only called when there is genuinely no avatar URL available.
// Never called when we have a URL — the background-image CSS covers the interim.
function _setAvatarInitial(el, user) {
    if (!el) return;
    // Reuse existing span if present to avoid DOM churn
    let span = el.querySelector('.um8-initial');
    if (!span) {
        span = document.createElement('span');
        span.className = 'um8-initial';
        el.textContent = '';
        el.appendChild(span);
    }
    span.textContent = (user.name || user.email || 'U')[0].toUpperCase();
}

function _applyUserToDOM(user) {
    if (!user) return;

    const role   = user.role || 'user';
    const tier   = user.tier || user.subscription_tier || 'free';
    const isAdminRole       = ['admin', 'master_admin'].includes(role);
    const isMasterAdminRole = role === 'master_admin';

    const displayName = (
        user.name ||
        [user.first_name, user.last_name].filter(Boolean).join(' ').trim() ||
        user.email?.split('@')[0] ||
        'User'
    );

    let displayTier;
    if (isMasterAdminRole)  displayTier = 'Master Admin';
    else if (role === 'admin') displayTier = 'Admin';
    else displayTier = getTierDisplayName(tier);

    // Text fields — safe to call before full DOM is ready via textContent
    const textMap = {
        userName:    displayName,
        userEmail:   user.email || '',
        welcomeName: displayName.split(' ')[0],
        userTier:    displayTier,
        userRole:    role,
    };
    Object.entries(textMap).forEach(([id, val]) => {
        const el = document.getElementById(id);
        if (el) el.textContent = val;
    });

    // ── Avatar — zero-flash approach ──────────────────────────────────────────
    // Strategy:
    //   1. Never cache-bust with Date.now() — that causes a browser cache miss every load.
    //      Instead, persist the clean URL to localStorage and let HTTP cache headers manage
    //      freshness. The URL only changes when the server returns a genuinely new one.
    //   2. Keep the initial letter VISIBLE underneath the image while the image loads.
    //      Use CSS opacity + transition to crossfade from initial → image.
    //      This means there is NEVER a blank/empty state — worst case the user sees their
    //      initial for ~50ms and it crossfades to the image.
    //   3. The pre-DOM <style> injected in hydrateFromCache() already painted the avatar
    //      via CSS background-image from browser cache (zero latency). The <img> element
    //      we create here reinforces that and handles the onerror/update path.
    const userAvatar = document.getElementById('userAvatar');
    if (userAvatar) {
        const rawSrc =
            user.avatarSignedUrl   ||
            user.avatar_signed_url ||
            user.avatarUrl         ||
            user.avatar_url        || null;

        const cleanUrl = rawSrc ? String(rawSrc).split('#')[0] : null;

        // Persist to localStorage — IIFE reads this on the very next page load
        // to inject a background-image <style> tag before the body even renders.
        if (cleanUrl) {
            try { localStorage.setItem(_AVATAR_URL_KEY, cleanUrl); } catch (_) {}
        } else {
            try { localStorage.removeItem(_AVATAR_URL_KEY); } catch (_) {}
        }

        // Fall back to localStorage copy if API response had no avatar field
        const urlToUse = cleanUrl ||
            (() => { try { return localStorage.getItem(_AVATAR_URL_KEY); } catch (_) { return null; } })();

        const existingImg = userAvatar.querySelector('img.um8-avatar-img');

        if (urlToUse) {
            if (existingImg) {
                // Element already exists — only swap src if URL genuinely changed.
                // This path never causes a visual change unless the avatar was updated.
                const prevUrl = existingImg.getAttribute('data-clean-src') || '';
                if (prevUrl !== urlToUse) {
                    existingImg.setAttribute('data-clean-src', urlToUse);
                    existingImg.src = urlToUse;
                }
            } else {
                // First time rendering this element.
                // IMPORTANT: set up the element BEFORE touching userAvatar's children
                // so there is zero frame where the element is empty.
                const img = document.createElement('img');
                img.className  = 'um8-avatar-img';
                img.alt        = '';
                img.setAttribute('data-clean-src', urlToUse);
                // Full-cover styling — same dimensions as the parent avatar circle
                img.style.cssText =
                    'display:block;width:100%;height:100%;' +
                    'border-radius:50%;object-fit:cover;';
                // No opacity animation — image comes from browser HTTP cache on
                // subsequent page loads (same URL was used by the background-image
                // IIFE style tag), so it paints instantly with no fade needed.

                img.onload = function () {
                    // Background-image CSS no longer needed — real <img> is showing
                    const pre = document.getElementById('um8-avatar-pre');
                    if (pre) pre.remove();
                    // Hide any fallback initial letter that may have been in the DOM
                    userAvatar.querySelectorAll('.um8-initial').forEach(el => el.remove());
                };

                img.onerror = function () {
                    // Broken URL — fall back to initial letter, clear cached URL
                    this.remove();
                    try { localStorage.removeItem(_AVATAR_URL_KEY); } catch (_) {}
                    const pre = document.getElementById('um8-avatar-pre');
                    if (pre) pre.remove();
                    _setAvatarInitial(userAvatar, user);
                };

                // Replace element content atomically — textContent wipe then
                // immediate append means the browser batches this into one repaint.
                // We do NOT set textContent to a letter first; the background-image
                // CSS injected by the IIFE covers the element until img.onload fires.
                userAvatar.textContent = '';
                userAvatar.appendChild(img);
                // src assigned last — img element is in the DOM first so the
                // browser never shows a broken-image placeholder.
                img.src = urlToUse;
            }
        } else {
            // No avatar URL anywhere — show initial letter
            if (existingImg) existingImg.remove();
            const pre = document.getElementById('um8-avatar-pre');
            if (pre) pre.remove();
            _setAvatarInitial(userAvatar, user);
        }
    }

    // Admin section visibility
    const adminSection = document.getElementById('adminSection');
    if (adminSection) adminSection.style.display = isAdminRole ? 'block' : 'none';

    document.querySelectorAll('.admin-only').forEach(el => {
        el.classList.toggle('hidden', !isAdminRole);
        el.style.display = isAdminRole ? '' : 'none';
    });
    document.querySelectorAll('.master-admin-only').forEach(el => {
        el.classList.toggle('hidden', !isMasterAdminRole);
        el.style.display = isMasterAdminRole ? '' : 'none';
    });

    // Quota bar (only if elements exist)
    const quotaUsed  = document.getElementById('quotaUsed');
    const quotaTotal = document.getElementById('quotaTotal');
    const quotaBar   = document.getElementById('quotaBar');
    if (quotaUsed || quotaTotal || quotaBar) {
        const isUnlimited = user.unlimited_uploads ||
            ['lifetime', 'friends_family'].includes(tier) || isAdminRole;
        const used  = user.uploads_this_month || 0;
        const limit = user.upload_quota ?? getUploadLimit(user);
        if (quotaUsed)  quotaUsed.textContent  = used;
        if (quotaTotal) quotaTotal.textContent = isUnlimited ? '∞' : limit;
        if (quotaBar) {
            if (isUnlimited) {
                quotaBar.style.width = '10%';
                quotaBar.className   = 'quota-bar bg-green';
            } else {
                const pct = Math.min(100, (used / limit) * 100);
                quotaBar.style.width = `${pct}%`;
                quotaBar.className   = `quota-bar ${pct > 80 ? 'bg-red' : pct > 50 ? 'bg-orange' : 'bg-green'}`;
            }
        }
    }

    // Tier badge
    const tierBadge = document.getElementById('tierBadge');
    if (tierBadge && !tierBadge.hasChildNodes()) {
        const badge = document.createElement('span');
        badge.classList.add('tier-badge');
        if (isMasterAdminRole) {
            badge.classList.add('bg-red');
            badge.textContent = 'Master Admin';
        } else if (role === 'admin') {
            badge.classList.add('bg-orange');
            badge.textContent = 'Admin';
        } else {
            const colorMap = {
                free: 'bg-gray', starter: 'bg-blue', solo: 'bg-blue', launch: 'bg-blue',
                creator: 'bg-orange', creator_lite: 'bg-blue', creator_pro: 'bg-orange',
                growth: 'bg-orange', studio: 'bg-purple', agency: 'bg-purple',
                lifetime: 'bg-gradient', friends_family: 'bg-gradient',
            };
            badge.classList.add(colorMap[tier] || 'bg-gray');
            badge.textContent = getTierDisplayName(tier);
        }
        tierBadge.appendChild(badge);
    }

    window.isUserAdmin       = isAdminRole;
    window.isUserMasterAdmin = isMasterAdminRole;
    window.userRole          = role;
    window.userTier          = tier;

    console.log('[Auth] User:', user.email, '| Role:', role, '| Tier:', tier);
}

// ============================================================
// Authentication
// ============================================================
async function checkAuth(options = {}, _retryDepth = 0) {
    const { redirectOnFail = true, silent = false } = options;

    // De-duplicate concurrent calls
    if (isAuthChecking && authCheckPromise) return authCheckPromise;

    // In-memory cache — fresh enough, skip network entirely
    if (_cachedUser && (Date.now() - _cachedUserAt) < _USER_CACHE_TTL) {
        return _cachedUser;
    }

    const token = getAccessToken();
    if (!token) {
        if (redirectOnFail && !isOnAuthPage()) {
            sessionStorage.setItem('uploadm8_auth_message', 'Please log in to continue.');
            window.location.href = 'login.html';
        }
        return null;
    }

    // Session-storage cache — still valid, use it and schedule a silent background refresh
    const cached = _readSessionCache();
    if (cached && cached.age < _SESSION_CACHE_MAX_AGE) {
        currentUser        = cached.user;
        window.currentUser = cached.user;
        _cachedUser        = cached.user;
        _cachedUserAt      = Date.now();
        // Schedule a background refresh so the cache stays warm without blocking
        setTimeout(() => _refreshUserInBackground(), 0);
        return cached.user;
    }

    // Cache is stale or absent — fetch from API
    isAuthChecking   = true;
    authCheckPromise = _fetchUserFromAPI(token, options, _retryDepth);
    return authCheckPromise;
}

async function _fetchUserFromAPI(token, options = {}, retryDepth = 0) {
    const { redirectOnFail = true, silent = false } = options;
    const requestId = generateRequestId();
    try {
        const resp = await fetchWithTimeout(`${API_BASE}/api/me`, {
            headers: {
                'Authorization': `Bearer ${token}`,
                'X-Request-ID':  requestId,
                'Content-Type':  'application/json',
            },
        }, AUTH_FETCH_TIMEOUT_MS);

        if (resp.status === 401) {
            if (retryDepth > 0) {
                // Already retried — token is definitively invalid
                clearTokens();
                if (redirectOnFail && !isOnAuthPage()) {
                    sessionStorage.setItem('uploadm8_auth_message', 'Session expired. Please log in again.');
                    window.location.href = 'login.html';
                }
                return null;
            }
            const refreshed = await tryRefreshToken();
            if (refreshed) {
                isAuthChecking   = false;
                authCheckPromise = null;
                return checkAuth(options, 1); // depth 1 — no further retries
            }
            clearTokens();
            if (redirectOnFail && !isOnAuthPage()) {
                sessionStorage.setItem('uploadm8_auth_message', 'Session expired. Please log in again.');
                window.location.href = 'login.html';
            }
            return null;
        }

        if (!resp.ok) {
            // Server error (5xx) — do NOT clear tokens; could be transient
            console.error(`Auth check failed: ${resp.status} [${requestId}]`);
            if (_isDefinitiveAuthFailure(resp.status)) {
                clearTokens();
                if (redirectOnFail && !isOnAuthPage()) window.location.href = 'login.html';
            }
            return null;
        }

        const user = await resp.json();
        if (!user || !user.email || !user.role) throw new Error('Invalid user data received');

        currentUser        = user;
        window.currentUser = user;
        _cachedUser        = user;
        _cachedUserAt      = Date.now();
        _writeSessionCache(user);

        try { window.dispatchEvent(new CustomEvent('uploadm8:user', { detail: user })); } catch (_) {}
        return user;

    } catch (e) {
        console.error('Auth check error:', e);
        // Network blips — do NOT redirect or clear tokens
        if (_isNetworkError(e)) {
            // Return cached user if we have one, so the page still works offline/blip
            const fallback = _readSessionCache();
            if (fallback) {
                console.warn('[Auth] Network error — using cached user as fallback');
                currentUser        = fallback.user;
                window.currentUser = fallback.user;
                _cachedUser        = fallback.user;
                _cachedUserAt      = Date.now() - _USER_CACHE_TTL + 5000; // force re-check soon
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
        // Unexpected non-network errors
        if (!silent && redirectOnFail && !isOnAuthPage()) {
            sessionStorage.setItem('uploadm8_auth_message', 'Authentication error. Please log in again.');
            window.location.href = 'login.html';
        }
        return null;
    } finally {
        isAuthChecking   = false;
        authCheckPromise = null;
    }
}

// Background refresh — updates sessionStorage cache without blocking the UI
async function _refreshUserInBackground() {
    const token = getAccessToken();
    if (!token) return;
    try {
        const resp = await fetchWithTimeout(`${API_BASE}/api/me`, {
            headers: { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' },
        }, AUTH_FETCH_TIMEOUT_MS);
        if (!resp.ok) return; // silent — don't touch tokens on background error
        const user = await resp.json();
        if (!user || !user.email || !user.role) return;
        currentUser        = user;
        window.currentUser = user;
        _cachedUser        = user;
        _cachedUserAt      = Date.now();
        _writeSessionCache(user);
        _applyUserToDOM(user); // refresh avatar/name/tier if they changed
        try { window.dispatchEvent(new CustomEvent('uploadm8:user', { detail: user })); } catch (_) {}
    } catch (_) {
        // Background refresh failure is silent — current cached data is still valid
    }
}

function isOnAuthPage() {
    const path = window.location.pathname;
    return (
        path.includes('login.html')          ||
        path.includes('signup.html')         ||
        path.includes('forgot-password.html')||
        path.includes('reset-password.html') ||
        path.endsWith('/')                   ||
        path.includes('index.html')          ||
        path.includes('terms.html')          ||
        path.includes('privacy.html')        ||
        path.includes('support.html')
    );
}

// ============================================================
// Token refresh — never clears tokens on network failures
// ============================================================
async function tryRefreshToken() {
    const refreshToken = getRefreshToken();
    if (!refreshToken) return false;

    // Try snake_case first, camelCase fallback (422 = FastAPI validation failure)
    const payloads = [
        { refresh_token: refreshToken },
        { refreshToken:  refreshToken },
    ];

    try {
        for (const payload of payloads) {
            const requestId = generateRequestId();
            let resp;
            try {
                resp = await fetchWithTimeout(`${API_BASE}/api/auth/refresh`, {
                    method:      'POST',
                    credentials: 'include',
                    headers:     { 'Content-Type': 'application/json', 'X-Request-ID': requestId },
                    body:        JSON.stringify(payload),
                }, REFRESH_FETCH_TIMEOUT_MS);
            } catch (fetchErr) {
                // Network error during refresh — do NOT clear tokens
                console.warn('[Auth] Token refresh network error:', fetchErr.message);
                return false;
            }

            if (resp.ok) {
                const data = await resp.json().catch(() => ({}));
                if (!data.access_token && !data.accessToken) {
                    // Server said OK but gave no token — treat as failure, but keep existing tokens
                    console.warn('[Auth] Refresh response missing access_token');
                    return false;
                }
                setTokens(
                    data.access_token  || data.accessToken,
                    data.refresh_token || data.refreshToken || refreshToken,
                    true
                );
                return true;
            }

            if (resp.status === 422) continue; // wrong payload shape, try next

            // 401/403/other — definitively rejected
            if (_isDefinitiveAuthFailure(resp.status)) {
                clearTokens();
            }
            return false;
        }

        // Both payloads failed with 422
        clearTokens();
        return false;

    } catch (e) {
        // Should not reach here (fetch errors caught inside loop), but belt-and-suspenders
        if (!_isNetworkError(e)) clearTokens();
        console.error('[Auth] tryRefreshToken unexpected error:', e);
        return false;
    }
}

// ============================================================
// Login / Logout / Register
// ============================================================
async function login(email, password, remember = true) {
    const requestId = generateRequestId();
    try {
        const resp = await fetchWithTimeout(`${API_BASE}/api/auth/login`, {
            method:  'POST',
            headers: { 'Content-Type': 'application/json', 'X-Request-ID': requestId },
            body:    JSON.stringify({ email, password }),
        }, DEFAULT_FETCH_TIMEOUT_MS);

        const data = await resp.json().catch(() => ({}));
        if (!resp.ok) {
            return { success: false, error: data.detail || 'Invalid email or password', requestId };
        }
        setTokens(data.access_token, data.refresh_token, remember);
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
    const token        = getAccessToken();
    const refreshToken = getRefreshToken();
    if (token && refreshToken) {
        // Fire-and-forget — don't await, don't let it block redirect
        fetchWithTimeout(`${API_BASE}/api/auth/logout`, {
            method:  'POST',
            headers: {
                'Content-Type':  'application/json',
                'Authorization': `Bearer ${token}`,
                'X-Request-ID':  generateRequestId(),
            },
            body: JSON.stringify({ refresh_token: refreshToken }),
        }, 8000).catch(() => {});
    }
    clearTokens();
    window.location.href = 'login.html';
}

async function logoutAll() {
    const token = getAccessToken();
    if (token) {
        try {
            await fetchWithTimeout(`${API_BASE}/api/auth/logout-all`, {
                method:  'POST',
                headers: { 'Authorization': `Bearer ${token}`, 'X-Request-ID': generateRequestId() },
            }, 8000);
        } catch (_) {}
    }
    clearTokens();
    window.location.href = 'login.html';
}

async function register(name, email, password) {
    const requestId = generateRequestId();
    try {
        const resp = await fetchWithTimeout(`${API_BASE}/api/auth/register`, {
            method:  'POST',
            headers: { 'Content-Type': 'application/json', 'X-Request-ID': requestId },
            body:    JSON.stringify({ name, email, password }),
        }, DEFAULT_FETCH_TIMEOUT_MS);

        const data = await resp.json().catch(() => ({}));
        if (!resp.ok) return { success: false, error: data.detail || 'Registration failed', requestId };
        if (data.access_token) setTokens(data.access_token, data.refresh_token, true);
        return { success: true, requestId };

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

// ============================================================
// API Calls — bounded retry depth
// ============================================================
async function apiCall(endpoint, options = {}, _retryDepth = 0) {
    const token     = getAccessToken();
    const requestId = generateRequestId();

    if (!token) {
        const error     = new Error('Not authenticated');
        error.status    = 401;
        error.requestId = requestId;
        throw error;
    }

    const config = {
        ...options,
        headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type':  'application/json',
            'X-Request-ID':  requestId,
            ...options.headers,
        },
    };
    if (options.body instanceof FormData) delete config.headers['Content-Type'];

    try {
        const resp = await fetchWithTimeout(`${API_BASE}${endpoint}`, config, DEFAULT_FETCH_TIMEOUT_MS);

        if (resp.status === 401) {
            if (_retryDepth > 0) {
                // Already retried once — give up, redirect
                clearTokens();
                sessionStorage.setItem('uploadm8_auth_message', 'Session expired. Please log in again.');
                window.location.href = 'login.html';
                throw new Error('Session expired');
            }
            const refreshed = await tryRefreshToken();
            if (refreshed) return apiCall(endpoint, options, 1); // depth 1 — no further retries
            clearTokens();
            sessionStorage.setItem('uploadm8_auth_message', 'Session expired. Please log in again.');
            window.location.href = 'login.html';
            throw new Error('Session expired');
        }

        if (!resp.ok) {
            const err   = await resp.json().catch(() => ({ detail: 'Request failed' }));
            const error = new Error(err.detail || `API Error: ${resp.status}`);
            error.status    = resp.status;
            error.requestId = requestId;
            error.response  = err;
            throw error;
        }

        const text = await resp.text();
        return text ? JSON.parse(text) : {};

    } catch (e) {
        if (e.message === 'Session expired') throw e;
        if (e.name === 'TimeoutError') {
            const error     = new Error(`Request timed out: ${endpoint}`);
            error.name      = 'TimeoutError';
            error.status    = 408;
            error.requestId = requestId;
            console.error(`API Timeout [${requestId}] ${endpoint}`);
            throw error;
        }
        if (!e.requestId) e.requestId = requestId;
        console.error(`API Error [${requestId}] ${endpoint}:`, e);
        throw e;
    }
}

// ============================================================
// Upload Stats
// ============================================================
function toNum(v) { const n = Number(v); return isNaN(n) ? 0 : n; }

function getUploadStats(upload) {
    let views = toNum(upload?.views), likes    = toNum(upload?.likes);
    let comments = toNum(upload?.comments),    shares  = toNum(upload?.shares);
    if (views === 0 && likes === 0 && comments === 0 && shares === 0) {
        const pr = Array.isArray(upload?.platform_results) ? upload.platform_results : [];
        pr.forEach(r => {
            if (!r || !r.success) return;
            views    += toNum(r.view_count    ?? r.views    ?? r.viewCount ?? r.plays);
            likes    += toNum(r.like_count    ?? r.likes    ?? r.likeCount);
            comments += toNum(r.comment_count ?? r.comments ?? r.commentCount);
            shares   += toNum(r.share_count   ?? r.shares   ?? r.shareCount);
        });
    }
    return { views, likes, comments, shares };
}

// ============================================================
// Upload with Progress
// ============================================================
let activeUploads = new Map();

async function uploadFile(file, metadata, onProgress, onStatusChange) {
    const requestId = generateRequestId();
    try {
        if (onStatusChange) onStatusChange('presigning');
        const presign = await apiCall('/api/uploads/presign', {
            method: 'POST',
            body:   JSON.stringify({ filename: file.name, file_size: file.size, content_type: file.type, ...metadata }),
        });
        const uploadId = presign.upload_id;
        if (onStatusChange) onStatusChange('uploading');
        await uploadToR2WithProgress(presign.presigned_url, file, uploadId, onProgress);
        if (activeUploads.get(uploadId)?.cancelled) return { success: false, error: 'Upload cancelled', uploadId };
        if (onStatusChange) onStatusChange('completing');
        const result = await apiCall(`/api/uploads/${uploadId}/complete`, { method: 'POST' });
        activeUploads.delete(uploadId);
        return { success: true, upload: result, uploadId };
    } catch (e) {
        console.error(`Upload error [${requestId}]:`, e);
        return { success: false, error: e.message, requestId };
    }
}

function uploadToR2WithProgress(url, file, uploadId, onProgress) {
    return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        activeUploads.set(uploadId, { xhr, cancelled: false });
        xhr.upload.addEventListener('progress', e => {
            if (e.lengthComputable && onProgress) onProgress(Math.round((e.loaded / e.total) * 100));
        });
        xhr.addEventListener('load',  () => (xhr.status >= 200 && xhr.status < 300) ? resolve(xhr) : reject(new Error(`Upload failed: ${xhr.status}`)));
        xhr.addEventListener('error', () => reject(new Error('Upload failed: network error')));
        xhr.addEventListener('abort', () => reject(new Error('Upload cancelled')));
        xhr.open('PUT', url);
        xhr.setRequestHeader('Content-Type', file.type);
        xhr.send(file);
    });
}

async function cancelUpload(uploadId) {
    try {
        const state = activeUploads.get(uploadId);
        if (state?.xhr) { state.cancelled = true; state.xhr.abort(); }
        await apiCall(`/api/uploads/${uploadId}/cancel`, { method: 'POST' });
        activeUploads.delete(uploadId);
        return { success: true };
    } catch (e) { return { success: false, error: e.message }; }
}

async function retryUpload(uploadId) {
    try {
        await apiCall(`/api/uploads/${uploadId}/retry`, { method: 'POST' });
        return { success: true };
    } catch (e) { return { success: false, error: e.message }; }
}

// ============================================================
// Entitlements / Tier helpers
// ============================================================
const TIER_SLUGS = ['free','creator_lite','creator_pro','studio','agency','friends_family','lifetime','master_admin','launch'];
const TIER_DISPLAY_NAMES = {
    free: 'Free', creator_lite: 'Creator Lite', creator_pro: 'Creator Pro',
    studio: 'Studio', agency: 'Agency', friends_family: 'Friends & Family',
    lifetime: 'Lifetime', master_admin: 'Admin', launch: 'Creator Lite',
};
const PAID_TIERS = ['creator_lite','creator_pro','studio','agency','friends_family','lifetime'];

function getTier(user) {
    const u = user || currentUser;
    return u?.tier || u?.subscription_tier || 'free';
}

function getTierDisplayName(tierOrUser) {
    if (typeof tierOrUser === 'object' && tierOrUser) {
        return tierOrUser.tier_display || TIER_DISPLAY_NAMES[getTier(tierOrUser)] || 'Free';
    }
    const t = tierOrUser || 'free';
    return TIER_DISPLAY_NAMES[t] || (typeof t === 'string' ? t.replace(/_/g,' ').replace(/\b\w/g,c=>c.toUpperCase()) : 'Free');
}

// updateUserUI — calls _applyUserToDOM plus wallet pill rendering
function updateUserUI() {
    if (!currentUser) return;
    try { window.dispatchEvent(new CustomEvent('uploadm8:user', { detail: currentUser })); } catch (_) {}
    _applyUserToDOM(currentUser);
    try {
        if (window.WalletTokens?.renderTokenPills) {
            const tba = document.querySelector('.top-bar-actions');
            if (tba) window.WalletTokens.renderTokenPills(tba);
        }
    } catch (e) { console.warn('[WalletTokens] renderTokenPills failed:', e); }
}

function isAdmin()       { return !!currentUser && ['admin','master_admin'].includes(currentUser.role); }
function isMasterAdmin() { return !!currentUser && currentUser.role === 'master_admin'; }
function isPaidUser()    { if (!currentUser) return false; if (isAdmin()) return true; return PAID_TIERS.includes(getTier(currentUser)); }
function isFreeUser()    { return !isPaidUser(); }
function isFriendsFamily() { return currentUser?.subscription_tier === 'friends_family'; }
function isLifetime()      { return currentUser?.subscription_tier === 'lifetime'; }

function getUserAccessLevel() {
    if (!currentUser) return 'guest';
    const role = currentUser.role || 'user';
    const tier = getTier(currentUser);
    if (role === 'master_admin') return 'master_admin';
    if (role === 'admin')        return 'admin';
    if (['lifetime','friends_family'].includes(tier)) return 'premium';
    if (['agency','studio'].includes(tier))           return 'business';
    if (tier === 'creator_pro')                       return 'pro';
    if (['creator_lite','launch'].includes(tier))     return 'basic';
    return 'free';
}

function hasEntitlement(feature) {
    if (!currentUser) return false;
    const ent  = currentUser.entitlements || {};
    const role = currentUser.role || 'user';
    if (['admin','master_admin'].includes(role)) return true;
    if (['lifetime','friends_family'].includes(getTier(currentUser))) return true;
    if (feature === 'show_ads')  return ent.show_ads === true;
    if (feature === 'no_ads')    return ent.show_ads === false;
    if (['unlimited_accounts','unlimited_connected_accounts','max_accounts'].includes(feature))
        return (ent.max_accounts || 0) >= 999;
    return !!ent[feature];
}

function showsAds() {
    if (!currentUser) return false;
    if (['admin','master_admin'].includes(currentUser.role)) return false;
    return (currentUser.entitlements || {}).show_ads === true;
}

function hasWatermark() {
    if (!currentUser) return true;
    if (['admin','master_admin'].includes(currentUser.role)) return false;
    return (currentUser.entitlements || {}).can_watermark === true;
}

function getMaxAccounts() {
    if (!currentUser) return 1;
    if (['admin','master_admin'].includes(currentUser.role)) return 999;
    return currentUser.entitlements?.max_accounts ?? 4;
}

function getMaxHashtags() {
    if (!currentUser) return 2;
    if (['admin','master_admin'].includes(currentUser.role)) return 9999;
    return currentUser.entitlements?.max_hashtags ?? 2;
}

function getUploadLimit(tierOrUser) {
    const u = typeof tierOrUser === 'object' ? tierOrUser : currentUser;
    const t = typeof tierOrUser === 'string' ? tierOrUser : getTier(u);
    if (u?.entitlements?.put_monthly != null) return u.entitlements.put_monthly;
    const limits = { free:80, creator_lite:400, launch:400, creator_pro:1200, studio:3500, agency:8000, lifetime:12000, friends_family:12000, master_admin:999999 };
    return limits[t] ?? 80;
}

function getTierBadgeHTML(tierOrUser) {
    const t = typeof tierOrUser === 'object' ? getTier(tierOrUser) : (tierOrUser || 'free');
    const colors = { free:'bg-gray', creator_lite:'bg-blue', launch:'bg-blue', creator_pro:'bg-orange', studio:'bg-purple', agency:'bg-purple', lifetime:'bg-gradient', friends_family:'bg-gradient', master_admin:'bg-red' };
    return `<span class="tier-badge ${colors[t]||'bg-gray'}">${getTierDisplayName(t)}</span>`;
}

function getUserStatusDot(status) {
    const colors = { active:'green', trialing:'yellow', canceled:'red', past_due:'red' };
    return `<span class="status-dot bg-${colors[status]||'gray'}"></span>`;
}

function getUserStatusBadge(user) {
    const tier   = user.subscription_tier || 'free';
    const status = user.subscription_status;
    if (['lifetime','friends_family'].includes(tier)) return '<span class="badge badge-purple">Lifetime</span>';
    if (status === 'trialing' || (user.trial_ends_at && new Date(user.trial_ends_at) > new Date())) return '<span class="badge badge-yellow">Trial</span>';
    if (status === 'active') return '<span class="badge badge-green">Active</span>';
    if (status === 'canceled' || status === 'past_due') return '<span class="badge badge-red">Canceled</span>';
    return '<span class="badge badge-gray">Free</span>';
}

// ============================================================
// UI Helpers
// ============================================================
function showToast(message, type = 'info', duration = 4000) {
    const container = document.getElementById('toastContainer') || createToastContainer();
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    const icons = { success:'check-circle', error:'exclamation-circle', warning:'exclamation-triangle', info:'info-circle' };
    const icon = document.createElement('i');    icon.className = `fas fa-${icons[type]||'info-circle'}`;
    const text = document.createElement('span'); text.textContent = String(message ?? '');
    const close = document.createElement('button');
    close.type = 'button'; close.className = 'toast-close'; close.innerHTML = '&times;';
    close.addEventListener('click', () => toast.remove());
    toast.append(icon, text, close);
    container.appendChild(toast);
    setTimeout(() => toast.classList.add('show'), 10);
    if (duration > 0) setTimeout(() => { toast.classList.remove('show'); setTimeout(() => toast.remove(), 300); }, duration);
    return toast;
}

function createToastContainer() {
    const c = document.createElement('div'); c.id = 'toastContainer';
    document.body.appendChild(c); return c;
}

function showModal(modalId) {
    const m = document.getElementById(modalId);
    if (m) { m.classList.remove('hidden'); document.body.style.overflow = 'hidden'; }
}
function hideModal(modalId) {
    const m = document.getElementById(modalId);
    if (m) { m.classList.add('hidden'); document.body.style.overflow = ''; }
}

function showConfirmModal(title, message, onConfirm) {
    const modal = document.getElementById('confirmModal');
    if (!modal) return;
    document.getElementById('confirmTitle').textContent   = title;
    document.getElementById('confirmMessage').textContent = message;
    document.getElementById('confirmAction').onclick      = () => { hideModal('confirmModal'); if (onConfirm) onConfirm(); };
    showModal('confirmModal');
}

function showLoading(container, message = 'Loading...') {
    if (typeof container === 'string') container = document.getElementById(container);
    if (!container) return;
    container.textContent = '';
    const wrap = document.createElement('div');
    wrap.className = 'flex items-center justify-center py-8 text-secondary';
    const icon = document.createElement('i'); icon.className = 'fas fa-spinner fa-spin mr-2';
    const text = document.createElement('span'); text.textContent = String(message ?? 'Loading...');
    wrap.append(icon, text); container.appendChild(wrap);
}

function showEmptyState(container, icon, message, actionHtml = '') {
    if (typeof container === 'string') container = document.getElementById(container);
    if (!container) return;
    container.textContent = '';
    const wrap = document.createElement('div'); wrap.className = 'empty-state';
    const ico = document.createElement('i'); ico.className = `fas fa-${icon || 'inbox'}`;
    const p = document.createElement('p'); p.textContent = String(message ?? '');
    wrap.append(ico, p);
    if (actionHtml) { const aw = document.createElement('div'); aw.innerHTML = actionHtml; wrap.appendChild(aw); }
    container.appendChild(wrap);
}

function showError(container, message) {
    if (typeof container === 'string') container = document.getElementById(container);
    if (!container) return;
    container.textContent = '';
    const wrap = document.createElement('div'); wrap.className = 'error-state';
    const icon = document.createElement('i'); icon.className = 'fas fa-exclamation-circle';
    const p = document.createElement('p'); p.textContent = String(message ?? 'Error');
    wrap.append(icon, p); container.appendChild(wrap);
}

// ============================================================
// Sidebar — safe to call before DOMContentLoaded via queue
// ============================================================
const _sidebarCallQueue = [];
let   _sidebarReady     = false;

function toggleSidebar() {
    if (_sidebarReady && typeof window._sidebarToggleFn === 'function') {
        window._sidebarToggleFn();
    } else {
        _sidebarCallQueue.push('toggle');
    }
}

// ============================================================
// Formatting helpers
// ============================================================
function formatDate(d) {
    if (!d) return '-';
    return new Date(d).toLocaleDateString('en-US', { month:'short', day:'numeric', year:'numeric' });
}
function formatDateTime(d) {
    if (!d) return '-';
    return new Date(d).toLocaleString('en-US', { month:'short', day:'numeric', year:'numeric', hour:'numeric', minute:'2-digit' });
}
function formatRelativeTime(d) {
    if (!d) return '-';
    const diff = Date.now() - new Date(d).getTime();
    const m = Math.floor(diff / 60000), h = Math.floor(diff / 3600000), days = Math.floor(diff / 86400000);
    if (m < 1)  return 'Just now';
    if (m < 60) return `${m}m ago`;
    if (h < 24) return `${h}h ago`;
    if (days < 7) return `${days}d ago`;
    return formatDate(d);
}
function formatNumber(num) {
    if (num == null) return '-';
    const n = Number(num);
    if (isNaN(n))      return '-';
    if (n >= 1000000)  return (n / 1000000).toFixed(1) + 'M';
    if (n >= 1000)     return (n / 1000).toFixed(1) + 'K';
    return n.toLocaleString();
}
function formatFileSize(bytes) {
    if (!bytes) return '0 B';
    const k = 1024, sizes = ['B','KB','MB','GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}
function formatCurrency(amount, currency = 'USD') {
    return new Intl.NumberFormat('en-US', { style:'currency', currency }).format(amount);
}

// ============================================================
// Platform helpers
// ============================================================
function getPlatformInfo(platform) {
    const map = {
        tiktok:    { name:'TikTok',    icon:'fab fa-tiktok',    color:'#00f2ea' },
        youtube:   { name:'YouTube',   icon:'fab fa-youtube',   color:'#FF0000' },
        instagram: { name:'Instagram', icon:'fab fa-instagram', color:'#E1306C' },
        facebook:  { name:'Facebook',  icon:'fab fa-facebook',  color:'#1877F2' },
        meta:      { name:'Meta',      icon:'fab fa-meta',      color:'#0668E1' },
        google:    { name:'YouTube',   icon:'fab fa-youtube',   color:'#FF0000' },
    };
    const p   = typeof platform === 'string' ? platform : (platform?.platform || platform?.name || String(platform || ''));
    const key = String(p).toLowerCase().trim();
    return map[key] || { name: typeof p === 'string' ? p : 'Unknown', icon:'fas fa-globe', color:'#666' };
}
function getPlatformIcon(platform)  { const i = getPlatformInfo(platform); return `<i class="${i.icon}" style="color:${i.color};" title="${i.name}"></i>`; }
function getPlatformBadge(platform) { const i = getPlatformInfo(platform); return `<span class="platform-badge" style="background:${i.color}20;color:${i.color};"><i class="${i.icon}"></i> ${i.name}</span>`; }

function getStatusBadge(uploadOrStatus) {
    const map = {
        pending:          { label:'Pending',          color:'yellow', icon:'clock' },
        queued:           { label:'Queued',           color:'blue',   icon:'list' },
        staged:           { label:'Staged',           color:'yellow', icon:'clock' },
        scheduled:        { label:'Scheduled',        color:'purple', icon:'calendar-alt' },
        ready_to_publish: { label:'Ready to Publish', color:'purple', icon:'calendar-check' },
        processing:       { label:'Processing',       color:'blue',   icon:'spinner fa-spin' },
        uploading:        { label:'Uploading',        color:'blue',   icon:'cloud-upload-alt' },
        completed:        { label:'Completed',        color:'green',  icon:'check-circle' },
        succeeded:        { label:'Succeeded',        color:'green',  icon:'check-circle' },
        failed:           { label:'Failed',           color:'red',    icon:'exclamation-circle' },
        partial:          { label:'Partial',          color:'orange', icon:'exclamation-triangle' },
        cancelled:        { label:'Cancelled',        color:'gray',   icon:'ban' },
    };
    const isObj   = uploadOrStatus && typeof uploadOrStatus === 'object';
    const status  = isObj ? (uploadOrStatus.status || '') : (uploadOrStatus || '');
    const label   = (isObj && uploadOrStatus.status_label ? String(uploadOrStatus.status_label).trim() : null)
                    || map[status]?.label
                    || (status ? String(status).replace(/_/g,' ') : 'Unknown');
    const info    = map[(status || '').toLowerCase()] || { color:'gray', icon:'circle' };
    return `<span class="status-badge status-${info.color}"><i class="fas fa-${info.icon}"></i> ${escapeHTML(label)}</span>`;
}

// ============================================================
// KPI helpers
// ============================================================
const KPI_RANGES = [
    { value:'30m',    label:'30 Minutes',  minutes:30     },
    { value:'1h',     label:'1 Hour',      minutes:60     },
    { value:'6h',     label:'6 Hours',     minutes:360    },
    { value:'12h',    label:'12 Hours',    minutes:720    },
    { value:'1d',     label:'1 Day',       minutes:1440   },
    { value:'7d',     label:'7 Days',      minutes:10080  },
    { value:'30d',    label:'30 Days',     minutes:43200  },
    { value:'6m',     label:'6 Months',    minutes:262800 },
    { value:'1y',     label:'1 Year',      minutes:525600 },
    { value:'custom', label:'Custom Range',minutes:0      },
];
function getKpiRangeMinutes(range) { return KPI_RANGES.find(r => r.value === range)?.minutes || 43200; }
function buildKpiRangeDropdown(selectId, onChange) {
    const sel = document.getElementById(selectId);
    if (!sel) return;
    sel.innerHTML = KPI_RANGES.map(r => `<option value="${r.value}">${r.label}</option>`).join('');
    if (onChange) sel.addEventListener('change', e => onChange(e.target.value));
}

// ============================================================
// Dashboard card drag & drop
// ============================================================
function initDashboardCustomization() {
    const grid = document.getElementById('dashboardGrid');
    if (!grid) return;
    try {
        const order = JSON.parse(localStorage.getItem('uploadm8_dashboard_order') || 'null');
        if (order) {
            const cards = Array.from(grid.children);
            order.forEach(id => { const c = cards.find(c => c.dataset.cardId === id); if (c) grid.appendChild(c); });
        }
    } catch (_) {}
    grid.querySelectorAll('[data-card-id]').forEach(card => {
        card.draggable = true;
        card.classList.add('draggable-card');
        card.addEventListener('dragstart', e => { e.dataTransfer.setData('text/plain', card.dataset.cardId); card.classList.add('dragging'); setTimeout(() => { card.style.opacity = '0.5'; }, 0); });
        card.addEventListener('dragend',   () => { card.classList.remove('dragging'); card.style.opacity = ''; saveDashboardOrder(); });
        card.addEventListener('dragover',  e => {
            e.preventDefault();
            const dragging = grid.querySelector('.dragging');
            if (dragging && dragging !== card) {
                const mid = card.getBoundingClientRect().top + card.getBoundingClientRect().height / 2;
                card.parentNode.insertBefore(dragging, e.clientY < mid ? card : card.nextSibling);
            }
        });
    });
}
function saveDashboardOrder() {
    const grid = document.getElementById('dashboardGrid');
    if (!grid) return;
    localStorage.setItem('uploadm8_dashboard_order', JSON.stringify(Array.from(grid.querySelectorAll('[data-card-id]')).map(c => c.dataset.cardId)));
    showToast('Dashboard layout saved', 'success');
}
function resetDashboardOrder() { localStorage.removeItem('uploadm8_dashboard_order'); location.reload(); }

// ============================================================
// Hide Figures — UNIFIED CSS-class approach
// All pages use body.classList 'hide-figures' + CSS .blur-value rules.
// The old [data-hide-value] text-swap approach is retired.
// ============================================================
function _initHideFigures() {
    if (localStorage.getItem('uploadm8_hide_figures') === 'true') {
        document.body.classList.add('hide-figures');
        _syncHideFiguresIcon(true);
    }
}

function toggleHideFigures() {
    const hidden = document.body.classList.toggle('hide-figures');
    localStorage.setItem('uploadm8_hide_figures', hidden ? 'true' : 'false');
    _syncHideFiguresIcon(hidden);
    showToast(hidden ? 'Numbers hidden' : 'Numbers visible', 'info');
}

function _syncHideFiguresIcon(hidden) {
    // Handles both the KPI-style icon (hideFiguresIcon) and queue-style (toggleFiguresIcon)
    ['hideFiguresIcon', 'toggleFiguresIcon'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.className = hidden ? 'fas fa-eye-slash' : 'fas fa-eye';
    });
}

// Legacy alias — kept so any old inline calls don't break
function updateHiddenFigures() { _initHideFigures(); }

// ============================================================
// Navigation helpers
// ============================================================
function highlightCurrentNav() {
    const path = window.location.pathname.split('/').pop() || 'dashboard.html';
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.toggle('active', link.getAttribute('href') === path);
    });
}

function initDragDrop(dropZoneId, fileInputId, onFileSelect) {
    const dz = document.getElementById(dropZoneId);
    const fi = document.getElementById(fileInputId);
    if (!dz || !fi) return;
    ['dragenter','dragover'].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.add('drag-over'); }));
    ['dragleave','drop'].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.remove('drag-over'); }));
    dz.addEventListener('drop',   e => { if (e.dataTransfer.files.length) onFileSelect(e.dataTransfer.files[0]); });
    fi.addEventListener('change', e => { if (e.target.files.length) onFileSelect(e.target.files[0]); });
}

// ============================================================
// App initialisation
// ============================================================
async function initApp(pageName) {
    setTheme(getTheme());
    _initHideFigures();

    const authMessage = sessionStorage.getItem('uploadm8_auth_message');
    if (authMessage) {
        sessionStorage.removeItem('uploadm8_auth_message');
        setTimeout(() => showToast(authMessage, 'warning'), 100);
    }

    const user = await checkAuth({ redirectOnFail: true });
    if (!user) return null;

    // updateUserUI re-applies everything — covers cases where the async
    // fetch returned fresher data than the hydration IIFE used
    updateUserUI();
    highlightCurrentNav();

    window.dispatchEvent(new CustomEvent('userLoaded', { detail: user }));

    document.getElementById('themeToggle')?.addEventListener('click', toggleTheme);
    document.querySelectorAll('[data-action="logout"], .logout-btn, #logoutBtn').forEach(btn => {
        btn.addEventListener('click', e => { e.preventDefault(); logout(); });
    });

    if (pageName === 'dashboard') initDashboardCustomization();
    if (!showsAds()) document.querySelectorAll('.ad-container, .ad-banner').forEach(el => el.remove());

    return user;
}

// ============================================================
// Avatar helper (used by settings page)
// ============================================================
function applyAvatarToUI(user) {
    if (!user) return;
    const rawUrl = user.avatar_url || user.avatarUrl || null;
    if (!rawUrl || typeof rawUrl !== 'string') return;
    const cleanUrl = rawUrl.split('#')[0]; // never cache-bust with Date.now()
    if (!/^https?:/i.test(cleanUrl)) return;

    // Persist clean URL for next-load hydration IIFE
    try { localStorage.setItem(_AVATAR_URL_KEY, cleanUrl); } catch (_) {}

    const sidebarAvatar = document.getElementById('sidebarAvatar');
    if (sidebarAvatar && !sidebarAvatar.querySelector('img')) {
        sidebarAvatar.textContent = '';
        const img = document.createElement('img');
        img.src = cleanUrl; img.alt = 'User avatar';
        img.style.cssText = 'width:100%;height:100%;object-fit:cover;border-radius:50%;';
        sidebarAvatar.appendChild(img);
    }
    const avatarImage   = document.getElementById('avatarImage');
    const avatarInitial = document.getElementById('avatarInitial');
    if (avatarImage) {
        if (avatarImage.src.split('#')[0] !== cleanUrl) avatarImage.src = cleanUrl;
        avatarImage.style.display = 'block';
    }
    if (avatarInitial) avatarInitial.style.display = 'none';
}

// ============================================================
// Global Exports
// ============================================================
if (typeof window !== 'undefined') {
    window.API_BASE              = API_BASE;
    window._AVATAR_URL_KEY       = _AVATAR_URL_KEY;
    window.APP_VERSION           = APP_VERSION;
    window.escapeHTML            = escapeHTML;
    window.checkAuth             = checkAuth;
    window.login                 = login;
    window.logout                = logout;
    window.logoutAll             = logoutAll;
    window.register              = register;
    window.isLoggedIn            = isLoggedIn;
    window.getAccessToken        = getAccessToken;
    window.clearTokens           = clearTokens;
    window.apiCall               = apiCall;
    window.uploadFile            = uploadFile;
    window.cancelUpload          = cancelUpload;
    window.retryUpload           = retryUpload;
    window.getUploadStats        = getUploadStats;   // ← was missing
    window.currentUser           = currentUser;
    window.updateUserUI          = updateUserUI;
    window.hasEntitlement        = hasEntitlement;
    window.getTier               = getTier;
    window.getTierDisplayName    = getTierDisplayName;
    window.getMaxAccounts        = getMaxAccounts;
    window.getMaxHashtags        = getMaxHashtags;
    window.getUploadLimit        = getUploadLimit;
    window.showsAds              = showsAds;
    window.hasWatermark          = hasWatermark;
    window.isAdmin               = isAdmin;
    window.isMasterAdmin         = isMasterAdmin;
    window.isPaidUser            = isPaidUser;
    window.isFreeUser            = isFreeUser;
    window.isFriendsFamily       = isFriendsFamily;
    window.isLifetime            = isLifetime;
    window.getUserAccessLevel    = getUserAccessLevel;
    window.getTierBadgeHTML      = getTierBadgeHTML;
    window.getUserStatusDot      = getUserStatusDot;
    window.getUserStatusBadge    = getUserStatusBadge;
    window.showToast             = showToast;
    window.showModal             = showModal;
    window.hideModal             = hideModal;
    window.showConfirmModal      = showConfirmModal;
    window.showLoading           = showLoading;
    window.showEmptyState        = showEmptyState;
    window.showError             = showError;
    window.toggleSidebar         = toggleSidebar;
    window.toggleTheme           = toggleTheme;
    window.getTheme              = getTheme;
    window.setTheme              = setTheme;
    window.toggleHideFigures     = toggleHideFigures;
    window.updateHiddenFigures   = updateHiddenFigures;
    window.initApp               = initApp;
    window.applyAvatarToUI       = applyAvatarToUI;
    window.initDragDrop          = initDragDrop;
    window.initDashboardCustomization = initDashboardCustomization;
    window.resetDashboardOrder   = resetDashboardOrder;
    window.highlightCurrentNav   = highlightCurrentNav;
    window.formatDate            = formatDate;
    window.formatDateTime        = formatDateTime;
    window.formatRelativeTime    = formatRelativeTime;
    window.formatNumber          = formatNumber;
    window.formatFileSize        = formatFileSize;
    window.formatCurrency        = formatCurrency;
    window.getPlatformInfo       = getPlatformInfo;
    window.getPlatformIcon       = getPlatformIcon;
    window.getPlatformBadge      = getPlatformBadge;
    window.getStatusBadge        = getStatusBadge;
    window.KPI_RANGES            = KPI_RANGES;
    window.getKpiRangeMinutes    = getKpiRangeMinutes;
    window.buildKpiRangeDropdown = buildKpiRangeDropdown;
}

// ============================================================
// MOBILE SIDEBAR — DOMContentLoaded
// Single source of truth. Flushes any pending toggleSidebar()
// calls that arrived before the DOM was ready.
// ============================================================
document.addEventListener('DOMContentLoaded', function () {
    // ── Avatar early-inject ─────────────────────────────────────────────────────
    // The IIFE already painted a background-image CSS rule from localStorage.
    // Now that the DOM exists, inject the real <img> element immediately so the
    // browser can confirm the cache hit and start decoding — before checkAuth
    // returns (which could be 200-800 ms away on a slow connection).
    // This means the avatar is in the DOM from the first DOMContentLoaded frame.
    (function injectAvatarImg() {
        try {
            const cachedUrl = localStorage.getItem(_AVATAR_URL_KEY);
            if (!cachedUrl) return;
            const el = document.getElementById('userAvatar');
            if (!el || el.querySelector('img.um8-avatar-img')) return; // already there

            const img = document.createElement('img');
            img.className = 'um8-avatar-img';
            img.alt       = '';
            img.setAttribute('data-clean-src', cachedUrl);
            img.style.cssText =
                'display:block;width:100%;height:100%;' +
                'border-radius:50%;object-fit:cover;';

            img.onload = function () {
                const pre = document.getElementById('um8-avatar-pre');
                if (pre) pre.remove();
                el.querySelectorAll('.um8-initial').forEach(e => e.remove());
            };
            img.onerror = function () {
                this.remove();
                try { localStorage.removeItem(_AVATAR_URL_KEY); } catch (_) {}
                const pre = document.getElementById('um8-avatar-pre');
                if (pre) pre.remove();
                // Initial letter will be set by updateUserUI once user object is available
            };

            el.textContent = '';
            el.appendChild(img);
            img.src = cachedUrl; // src set AFTER append — no broken-image icon
        } catch (_) {}
    })();
    // ── end avatar early-inject ─────────────────────────────────────────────────

    const menuToggle = document.getElementById('menuToggle');
    const sidebar    = document.getElementById('sidebar');
    const overlay    = document.getElementById('sidebarOverlay');
    let   _isOpen    = false;

    function openSidebar() {
        if (!sidebar) return;
        _isOpen = true;
        sidebar.classList.add('open');
        document.body.classList.add('sidebar-open');
        if (overlay) { overlay.classList.remove('hidden'); overlay.classList.add('active'); overlay.style.display = 'block'; }
    }

    function closeSidebar() {
        if (!sidebar) return;
        _isOpen = false;
        sidebar.classList.remove('open');
        document.body.classList.remove('sidebar-open');
        if (overlay) { overlay.classList.add('hidden'); overlay.classList.remove('active'); overlay.style.display = 'none'; }
    }

    function _toggleSidebar() {
        if (!sidebar) return;
        (_isOpen || sidebar.classList.contains('open')) ? closeSidebar() : openSidebar();
    }

    // Expose real implementation — flush any pre-DOM queued calls
    window._sidebarToggleFn = _toggleSidebar;
    _sidebarReady = true;
    window.toggleSidebar = _toggleSidebar;
    _sidebarCallQueue.forEach(cmd => { if (cmd === 'toggle') _toggleSidebar(); });
    _sidebarCallQueue.length = 0;

    // Wire burger button — removeAttribute prevents double-fire from HTML onclick
    if (menuToggle) {
        menuToggle.removeAttribute('onclick');
        menuToggle.addEventListener('click', e => { e.preventDefault(); e.stopPropagation(); _toggleSidebar(); }, { passive: false });
    }

    if (overlay) {
        overlay.removeAttribute('onclick');
        overlay.addEventListener('click', e => { e.preventDefault(); e.stopPropagation(); closeSidebar(); }, { passive: false });
    }

    if (sidebar) {
        sidebar.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', () => {
                if (window.innerWidth <= 1024 && sidebar.classList.contains('open')) closeSidebar();
            }, { passive: true });
        });
    }

    // Initial overlay state
    if (overlay) overlay.style.display = sidebar?.classList.contains('open') ? 'block' : 'none';

    // Back button injection on mobile sub-pages
    const currentPage  = window.location.pathname.split('/').pop() || 'index.html';
    const NO_BACK      = ['dashboard.html','index.html','login.html','signup.html',''];
    const BACK_MAP     = {
        'upload.html':            { label:'Dashboard', href:'dashboard.html' },
        'queue.html':             { label:'Dashboard', href:'dashboard.html' },
        'scheduled.html':         { label:'Dashboard', href:'dashboard.html' },
        'platforms.html':         { label:'Dashboard', href:'dashboard.html' },
        'groups.html':            { label:'Platforms', href:'platforms.html' },
        'analytics.html':         { label:'Dashboard', href:'dashboard.html' },
        'settings.html':          { label:'Dashboard', href:'dashboard.html' },
        'color-preferences.html': { label:'Settings',  href:'settings.html'  },
        'guide.html':             { label:'Dashboard', href:'dashboard.html' },
        'admin.html':             { label:'Dashboard', href:'dashboard.html' },
        'account-management.html':{ label:'Admin',     href:'admin.html'     },
        'admin-kpi.html':         { label:'Admin',     href:'admin.html'     },
        'admin-calculator.html':  { label:'Admin',     href:'admin.html'     },
        'admin-wallet.html':      { label:'Admin',     href:'admin.html'     },
        'billing.html':           { label:'Settings',  href:'settings.html'  },
        'success.html':           { label:'Dashboard', href:'dashboard.html' },
        'walkthrough.html':       { label:'Home',      href:'index.html'     },
        'kpi.html':               { label:'Admin',     href:'admin.html'     },
    };

    if (!NO_BACK.includes(currentPage) && BACK_MAP[currentPage] && window.innerWidth <= 1024) {
        const info   = BACK_MAP[currentPage];
        const tba    = document.querySelector('.top-bar-actions');
        if (tba) {
            const btn = document.createElement('a');
            btn.href      = info.href;
            btn.className = 'back-btn-mobile back-btn';
            btn.innerHTML = `<i class="fas fa-arrow-left"></i><span>${info.label}</span>`;
            btn.style.cssText = 'display:inline-flex;align-items:center;gap:0.4rem;padding:0.4rem 0.75rem;background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.12);border-radius:8px;color:var(--text-secondary);font-size:0.78rem;font-weight:500;text-decoration:none;min-height:36px;white-space:nowrap;';
            tba.prepend(btn);
        }
    }

    // Swipe gestures
    let touchStartX = 0, touchStartY = 0;
    document.addEventListener('touchstart', e => { touchStartX = e.touches[0].clientX; touchStartY = e.touches[0].clientY; }, { passive: true });
    document.addEventListener('touchend', e => {
        if (!sidebar) return;
        const dx = e.changedTouches[0].clientX - touchStartX;
        const dy = Math.abs(e.changedTouches[0].clientY - touchStartY);
        if (touchStartX < 25  && dx > 60  && dy < 80 && !sidebar.classList.contains('open')) _toggleSidebar();
        if (sidebar.classList.contains('open') && dx < -60 && dy < 80) _toggleSidebar();
    }, { passive: true });

    // Apply hide-figures after DOM is ready (in case IIFE ran before body existed)
    _initHideFigures();
});