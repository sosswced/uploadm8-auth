/**
 * UploadM8 v3.2.0 — Production Application JavaScript
 *
 * Key improvements over v3.1.1:
 *  - Session user snapshot (JWT sub + access-token tail) for instant paint; /api/me remains source of truth
 *  - Same-user token refresh patches tokenMarker in sessionStorage (auth-stack setTokens) — no wipe on refresh
 *  - Load order: api-base → auth-stack → session-user-hydrate → public-shell-hydrate → shared-sidebar
 *    (calls __um8ApplyTrustedSessionChrome) → helpers → upload-utils → app.js
 *  - initApp → checkAuth({ blockUntilValidated }) when wallet refresh is on: awaits /api/me before uploadm8:user (avoids 401s if cookie expired but snapshot is fresh); admin lite pages use skipWalletRefresh:true
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
// API origin: canonical logic lives in js/api-base.js (load before this file). Sync window.API_BASE from it only.
(function syncApiBaseFromCanonicalModule() {
    try {
        if (typeof window.resolveUploadM8ApiOrigin === 'function') {
            window.API_BASE = window.resolveUploadM8ApiOrigin();
            return;
        }
        if (typeof window.getUploadM8ApiBase === 'function') {
            window.API_BASE = String(window.getUploadM8ApiBase() || '').replace(/\/$/, '');
            return;
        }
    } catch (_) {}
    window.API_BASE =
        typeof window.API_BASE === 'string' && window.API_BASE
            ? String(window.API_BASE).replace(/\/$/, '')
            : 'https://auth.uploadm8.com';
})();
if (typeof window.getUploadM8ApiBase !== 'function') {
    window.getUploadM8ApiBase = function () {
        return String(window.API_BASE || 'https://auth.uploadm8.com').replace(/\/$/, '');
    };
}
const APP_VERSION  = '3.2.2';

const TOKEN_KEY      = 'uploadm8_access_token';
const REFRESH_KEY    = 'uploadm8_refresh_token';
// Avatar URL hint: sessionStorage only, bound to JWT sub (shared-browser safe).
const _AVATAR_URL_KEY  = 'uploadm8_avatar_url';
const _AVATAR_URL_KEY_PREFIX = 'uploadm8_avatar_url:';
/** Session-only avatar (no localStorage) — paired with JWT sub so shared computers never show wrong face. */
const _AVATAR_SS_URL = 'uploadm8_avatar_ss_url';
const _AVATAR_SS_SUB = 'uploadm8_avatar_ss_sub';

function _b64UrlDecode(segment) {
    let b = String(segment || '').replace(/-/g, '+').replace(/_/g, '/');
    while (b.length % 4) b += '=';
    try {
        return decodeURIComponent(Array.prototype.map.call(atob(b), function (c) {
            return '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2);
        }).join(''));
    } catch (_) {
        try { return atob(b); } catch (_) { return null; }
    }
}

function _jwtPayload(accessToken) {
    try {
        const parts = String(accessToken || '').split('.');
        if (parts.length < 2) return null;
        const json = _b64UrlDecode(parts[1]);
        if (!json) return null;
        return JSON.parse(json);
    } catch (_) {
        return null;
    }
}

/** Access token `sub` (user id) — must match cached /api/me snapshot for shared-browser safety. */
function _jwtSub(accessToken) {
    const pl = _jwtPayload(accessToken);
    if (!pl || pl.sub == null) return '';
    return String(pl.sub).trim();
}

function _cachePayloadMatchesCurrentToken(parsed) {
    if (!parsed || typeof parsed !== 'object') return false;
    const uid = String(
        parsed.uid ||
            (parsed.user && (parsed.user.id || parsed.user.user_id || parsed.user.uid)) ||
            ''
    ).trim();
    const cm = String(parsed.tokenMarker || '');
    if (cm.indexOf('cookie_uid:') === 0 && uid && cm === 'cookie_uid:' + uid) return true;
    const token =
        (typeof window.getAccessToken === 'function' && window.getAccessToken()) ||
        sessionStorage.getItem(TOKEN_KEY) ||
        '';
    if (!token) return false;
    const marker = token ? String(token).slice(-24) : '';
    if (!cm || !marker || cm !== marker) return false;
    const sub = _jwtSub(token);
    if (!sub || !uid || sub !== uid) return false;
    return true;
}

function _getTokenMarkerForCache() {
    try {
        const t = (typeof window.getAccessToken === 'function' && window.getAccessToken()) ||
            sessionStorage.getItem(TOKEN_KEY) || '';
        if (t) return String(t).slice(-24);
        const u = window.currentUser || {};
        const uid = String(u.id || u.user_id || u.uid || '').trim();
        return uid ? 'cookie_uid:' + uid : '';
    } catch (_) {
        return '';
    }
}

function _extractCachedUid(raw) {
    try {
        if (!raw) return '';
        const parsed = JSON.parse(raw);
        const user = (parsed && parsed.user && typeof parsed.user === 'object') ? parsed.user : parsed;
        return String(user && (user.id || user.user_id || user.uid) || '').trim();
    } catch (_) {
        return '';
    }
}

function _getAvatarStorageKey() {
    const current = (window.currentUser && (window.currentUser.id || window.currentUser.user_id || window.currentUser.uid)) || '';
    const tok =
        (typeof window.getAccessToken === 'function' && window.getAccessToken()) ||
        sessionStorage.getItem(TOKEN_KEY) ||
        '';
    const fromJwt = _jwtSub(tok);
    const cached = _extractCachedUid(sessionStorage.getItem('uploadm8_cached_user'));
    const uid = String(current || fromJwt || cached || '').trim();
    return uid ? (_AVATAR_URL_KEY_PREFIX + uid) : _AVATAR_URL_KEY;
}

/**
 * Facebook / Instagram CDN URLs often return 403 when loaded outside Meta apps.
 * Avatars must be R2/signed or user-uploaded — drop these so we do not spam the network.
 */
function _sanitizeAvatarUrlForDisplay(url) {
    if (!url || typeof url !== 'string') return null;
    const u = String(url).trim().split('#')[0];
    if (!u.startsWith('http')) return u;
    try {
        const host = new URL(u).hostname.toLowerCase();
        if (host.includes('fbcdn.net') || host.includes('cdninstagram.com')) return null;
    } catch (_) {
        return null;
    }
    return u;
}
try {
    window._sanitizeUploadM8AvatarUrl = _sanitizeAvatarUrlForDisplay;
} catch (_) {}

function _getSanitizedAvatarUrlFromStorage() {
    try {
        const tok =
            (typeof window.getAccessToken === 'function' && window.getAccessToken()) ||
            sessionStorage.getItem(TOKEN_KEY) ||
            '';
        const sub = _jwtSub(tok);
        if (sub) {
            const owner = sessionStorage.getItem(_AVATAR_SS_SUB);
            const url = sessionStorage.getItem(_AVATAR_SS_URL);
            if (owner === sub) {
                const s = _sanitizeAvatarUrlForDisplay(url);
                if (url && !s) {
                    sessionStorage.removeItem(_AVATAR_SS_URL);
                    sessionStorage.removeItem(_AVATAR_SS_SUB);
                }
                if (s) return s;
            } else if (owner) {
                sessionStorage.removeItem(_AVATAR_SS_URL);
                sessionStorage.removeItem(_AVATAR_SS_SUB);
            }
        }
        return null;
    } catch (_) {
        return null;
    }
}

function _persistAvatarUrlForSession(safeUrl, userIdFallback) {
    try {
        const tok =
            (typeof window.getAccessToken === 'function' && window.getAccessToken()) ||
            sessionStorage.getItem(TOKEN_KEY) ||
            '';
        const sub = _jwtSub(tok) || String(userIdFallback || '').trim();
        if (!sub || !safeUrl) return;
        sessionStorage.setItem(_AVATAR_SS_SUB, sub);
        sessionStorage.setItem(_AVATAR_SS_URL, safeUrl);
    } catch (_) {}
}

function _clearSessionAvatarCache() {
    try {
        sessionStorage.removeItem(_AVATAR_SS_URL);
        sessionStorage.removeItem(_AVATAR_SS_SUB);
    } catch (_) {}
}

// Timeout constants
const DEFAULT_FETCH_TIMEOUT_MS = 15000;
const REFRESH_FETCH_TIMEOUT_MS = 10000;
const AUTH_FETCH_TIMEOUT_MS    = 15000;

// User snapshot — sessionStorage only (never localStorage) so shared computers do not bleed profiles.
const _SESSION_CACHE_KEY    = 'uploadm8_cached_user';
const _SESSION_CACHE_AT_KEY = 'uploadm8_cached_user_at';
/** Trust = JWT sub + token tail match snapshot (see auth-stack setTokens refresh patch). No wall-clock TTL — entitlements refresh via checkAuth /api/me. */

function _sessionUserSnapshotTrusted() {
    try {
        const raw = sessionStorage.getItem(_SESSION_CACHE_KEY);
        const at = sessionStorage.getItem(_SESSION_CACHE_AT_KEY);
        if (!raw || !at) return false;
        const parsed = JSON.parse(raw);
        if (!_cachePayloadMatchesCurrentToken(parsed)) return false;
        const tok = (typeof window.getAccessToken === 'function' && window.getAccessToken()) ||
            sessionStorage.getItem(TOKEN_KEY);
        return !!tok;
    } catch (_) {
        return false;
    }
}

// In-memory cache — prevents duplicate /api/me calls within a single page load
const __um8AuthState = (window.__um8AuthState = window.__um8AuthState || {
    cachedUser: null,
    cachedUserAt: 0,
    currentUser: null,
    isAuthChecking: false,
    authCheckPromise: null,
});

// Auth state
let currentUser = __um8AuthState.currentUser || null;

function _hideRoleGatedUiByDefault() {
    try {
        document.querySelectorAll('.admin-only, .master-admin-only, .admin-only-marketing').forEach(el => {
            if (!el.dataset.roleDisplay) {
                el.dataset.roleDisplay = (el.style && typeof el.style.display === 'string') ? el.style.display : '';
            }
            el.classList.add('hidden');
            el.style.display = 'none';
        });
    } catch (_) {}
}

if (!window.__um8DomHydratedFromSession) {
    _hideRoleGatedUiByDefault();
}

function _setCurrentUserState(user) {
    currentUser = user;
    __um8AuthState.currentUser = user;
    window.currentUser = user;
}

// Pages that do not require authentication (no redirect to login)
const PUBLIC_PAGES = [
    'index.html', 'login.html', 'signup.html', 'forgot-password.html',
    'terms.html', 'privacy.html', 'about.html', 'contact.html',
    'support.html', 'blog.html', 'how-it-works.html', 'data-deletion.html',
    'walkthrough.html', 'check-email.html', 'confirm-email.html', 'reset-password.html',
    'verify-email.html', 'unsubscribe.html', ''
];

/** Ensure role + entitlements exist so cache + /api/me payloads never fail strict checks (free tier included). */
function _normalizeUserPayload(user) {
    if (!user || typeof user !== 'object') return user;

    const planObj = user.plan && typeof user.plan === 'object' ? user.plan : null;
    const planTier = planObj && typeof planObj.tier === 'string'
        ? String(planObj.tier).trim().toLowerCase()
        : '';

    if (!user.subscription_tier || typeof user.subscription_tier !== 'string') {
        const top = typeof user.tier === 'string' && user.tier ? String(user.tier).trim() : '';
        user.subscription_tier = top || (planTier || 'free');
    }
    if (!user.tier || typeof user.tier !== 'string') {
        user.tier = user.subscription_tier || planTier || 'free';
    }
    // Stale/minimal session caches sometimes only have plan.tier — surface it for admin gating.
    const sub = String(user.subscription_tier || '').toLowerCase();
    const topTier = String(user.tier || '').toLowerCase();
    if (planTier && planTier !== 'free') {
        if (!sub || sub === 'free') user.subscription_tier = user.plan.tier;
        if (!topTier || topTier === 'free') user.tier = user.plan.tier;
    }

    const subL = String(user.subscription_tier || '').toLowerCase();
    const tierL = String(user.tier || '').toLowerCase();
    const implicitMaster = subL === 'master_admin' || tierL === 'master_admin' || planTier === 'master_admin';

    const rawRole = user.role != null && String(user.role).trim() !== ''
        ? String(user.role).trim().toLowerCase()
        : '';
    if (implicitMaster && rawRole !== 'admin' && rawRole !== 'master_admin') {
        user.role = 'master_admin';
    } else if (!rawRole) {
        user.role = 'user';
    } else {
        user.role = rawRole;
    }

    if (!user.entitlements || typeof user.entitlements !== 'object') {
        user.entitlements = planObj || {};
    }
    return user;
}

// ============================================================
// INSTANT UI HYDRATION — runs when app.js parses (after shared-sidebar + session-user-hydrate).
// session-user-hydrate.js + shared-sidebar already paint the shell from sessionStorage in one turn.
// Here we bind __um8AuthState / currentUser and skip duplicate DOM if that early pass ran.
// ============================================================
(function hydrateFromCache() {
    const token =
        (typeof window.getAccessToken === 'function' && window.getAccessToken()) ||
        sessionStorage.getItem(TOKEN_KEY) ||
        '';
    if (!token) return;

    // ── Phase 1: Paint avatar from session hint before user object loads.
    // Skip if js/session-user-hydrate.js already applied the full chrome (same synchronous rules).
    // Otherwise a leftover URL could flash before /api/me replaces it.
    try {
        if (!window.__um8DomHydratedFromSession && _sessionUserSnapshotTrusted()) {
            const cachedAvatarUrl = _getSanitizedAvatarUrlFromStorage();
            const safePre = _sanitizeAvatarUrlForDisplay(cachedAvatarUrl);
            if (cachedAvatarUrl && !safePre) {
                try { sessionStorage.removeItem(_AVATAR_SS_URL); sessionStorage.removeItem(_AVATAR_SS_SUB); } catch (_) {}
            } else if (safePre) {
                const style = document.createElement('style');
                style.id = 'um8-avatar-pre';
                style.textContent =
                    '#userAvatar{background-image:url("' +
                    safePre.replace(/"/g,'\"') +
                    '");background-size:cover;background-position:center;' +
                    'font-size:0!important;color:transparent!important;}' +
                    '#userAvatar .um8-initial,#userAvatar>span{visibility:hidden!important;}';
                (document.head || document.documentElement).appendChild(style);
            }
        }
    } catch (_) {}

    // ── Phase 2: Optional paint-from-cache (same tab + same JWT sub + same token tail only).
    function _trySessionUserSnapshot() {
        try {
            const raw = sessionStorage.getItem(_SESSION_CACHE_KEY);
            const at  = sessionStorage.getItem(_SESSION_CACHE_AT_KEY);
            if (!raw || !at) return null;
            const cachedAtMs = parseInt(at, 10);
            if (!cachedAtMs || cachedAtMs <= 0) return null;
            const parsed = JSON.parse(raw);
            if (!_cachePayloadMatchesCurrentToken(parsed)) return null;
            const user = _normalizeUserPayload(parsed && parsed.user ? parsed.user : parsed);
            if (!user || !user.email) return null;
            return { user, cachedAtMs };
        } catch (_) { return null; }
    }
    const hit = _trySessionUserSnapshot();
    if (!hit) return;
    const user = hit.user;

    _setCurrentUserState(user);
    __um8AuthState.cachedUser = user;
    __um8AuthState.cachedUserAt = hit.cachedAtMs;

    if (!window.__um8DomHydratedFromSession) {
        _applyUserToDOM(user);
    }
    _markUserReady();
    // Do not dispatch uploadm8:user here — session cookie may be expired while this JWT-matched
    // snapshot is fresh; wallet-tokens validates via /api/me on first fetch (see wallet-tokens _auto).
})();

function _markUserReady() {
    try { document.documentElement.classList.add('um8-user-ready'); } catch (_) {}
}

// ============================================================
// Theme — applied synchronously to prevent FOUC
// ============================================================
function getTheme() {
    let t = sessionStorage.getItem('uploadm8_theme');
    if (!t) {
        try {
            t = localStorage.getItem('uploadm8_theme');
            if (t) {
                sessionStorage.setItem('uploadm8_theme', t);
                localStorage.removeItem('uploadm8_theme');
            }
        } catch (_) {}
    }
    return t || 'dark';
}

function setTheme(theme) {
    try {
        sessionStorage.setItem('uploadm8_theme', theme);
        localStorage.removeItem('uploadm8_theme');
    } catch (_) {}
    document.documentElement.setAttribute('data-theme', theme);
    document.body.classList.toggle('light-mode', theme === 'light');
    document.body.classList.toggle('dark-mode',  theme !== 'light');
    const iconCls = theme === 'dark' ? 'fas fa-moon' : 'fas fa-sun';
    document.querySelectorAll(
        '#themeToggleIcon, #themeToggleIconDesktop, #themeToggleIconMobile'
    ).forEach(el => { if (el) el.className = iconCls; });
    try {
        document.querySelectorAll('.theme-toggle > i').forEach(el => { el.className = iconCls; });
    } catch (_) {}
}

function toggleTheme() { setTheme(getTheme() === 'dark' ? 'light' : 'dark'); }

// Apply theme immediately — match setTheme() (data-theme + body classes + toggle icons)
(function () { try { setTheme(getTheme()); } catch (_) { document.documentElement.setAttribute('data-theme', getTheme()); } })();

// ============================================================
// Fetch + token primitives: js/auth-stack.js
// ============================================================

function clearTokens() {
    __um8AuthState.cachedUser = null;
    __um8AuthState.cachedUserAt = 0;
    __um8AuthState.isAuthChecking = false;
    __um8AuthState.authCheckPromise = null;
    _setCurrentUserState(null);

    try { window.__um8DomHydratedFromSession = false; } catch (_) {}

    try { sessionStorage.removeItem(_SESSION_CACHE_KEY);    } catch (_) {}
    try { sessionStorage.removeItem(_SESSION_CACHE_AT_KEY); } catch (_) {}
    try { localStorage.removeItem(_SESSION_CACHE_KEY);      } catch (_) {}
    try { localStorage.removeItem(_SESSION_CACHE_AT_KEY);   } catch (_) {}
    try { sessionStorage.removeItem(_AVATAR_SS_URL);        } catch (_) {}
    try { sessionStorage.removeItem(_AVATAR_SS_SUB);        } catch (_) {}

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
    try {
        Object.keys(localStorage).forEach(k => {
            if (String(k).startsWith(_AVATAR_URL_KEY_PREFIX)) localStorage.removeItem(k);
        });
    } catch (_) {}
    function _sweepUploadm8Keys(storage) {
        try {
            const rm = [];
            for (let i = 0; i < storage.length; i++) {
                const k = storage.key(i);
                if (!k) continue;
                if (String(k).startsWith('uploadm8') || String(k).startsWith('wt_')) rm.push(k);
            }
            rm.forEach(k => { try { storage.removeItem(k); } catch (_) {} });
        } catch (_) {}
    }
    _sweepUploadm8Keys(localStorage);
    _sweepUploadm8Keys(sessionStorage);
    if (typeof window._uploadm8StripLegacyLocalAuth === 'function') window._uploadm8StripLegacyLocalAuth();
}

// ============================================================
// Session cache helpers
// ============================================================
function _readSessionCache() {
    try {
        const raw = sessionStorage.getItem(_SESSION_CACHE_KEY);
        const at  = sessionStorage.getItem(_SESSION_CACHE_AT_KEY);
        if (!raw || !at) return null;
        const parsed = JSON.parse(raw);
        if (!_cachePayloadMatchesCurrentToken(parsed)) return null;
        const user = _normalizeUserPayload(parsed && parsed.user ? parsed.user : parsed);
        if (!user || !user.email) return null;
        return { user, age: Date.now() - parseInt(at, 10) };
    } catch (_) { return null; }
}

function _writeSessionCache(user) {
    try {
        user = _normalizeUserPayload({ ...user });
        const ts = String(Date.now());
        const uid = String(user.id || user.user_id || user.uid || '');
        let marker = _getTokenMarkerForCache();
        if (!uid) return;
        if (!marker) marker = 'cookie_uid:' + uid;
        const payload = {
            user: user,
            uid: uid,
            tokenMarker: marker,
        };
        sessionStorage.setItem(_SESSION_CACHE_KEY, JSON.stringify(payload));
        sessionStorage.setItem(_SESSION_CACHE_AT_KEY, ts);
    } catch (_) {}
}

// ============================================================
// User-scoped cache helpers (safe for shared computers)
// ============================================================
const _USER_SCOPED_CACHE_PREFIX = 'uploadm8_user_cache_v1:';

function _getUserCacheScope() {
    const u = currentUser || __um8AuthState.currentUser || window.currentUser || {};
    const uid = String(u.id || u.user_id || u.uid || '').trim();
    let token = '';
    try {
        token = (typeof window.getAccessToken === 'function' && window.getAccessToken()) ||
                sessionStorage.getItem(TOKEN_KEY) || '';
    } catch (_) {}
    const tm = token ? String(token).slice(-24) : uid ? 'cookie_uid:' + uid : '';
    return { uid, tokenMarker: tm };
}

function _userScopedCacheKey(namespace, uid) {
    const ns = String(namespace || '').trim();
    const id = String(uid || '').trim();
    if (!ns || !id) return null;
    return _USER_SCOPED_CACHE_PREFIX + ns + ':' + id;
}

function _readUserScopedCache(namespace, options = {}) {
    const { uid } = _getUserCacheScope();
    const key = _userScopedCacheKey(namespace, uid);
    if (!key) return null;
    const storage = options.storage === 'local' ? localStorage : sessionStorage;
    const maxAgeMs = Number(options.maxAgeMs || 0);
    try {
        const raw = storage.getItem(key);
        if (!raw) return null;
        const parsed = JSON.parse(raw);
        if (!parsed || parsed.uid !== uid) return null;
        const curMarker = _getTokenMarkerForCache();
        const cachedMarker = String(parsed.tokenMarker || '');
        if (cachedMarker && curMarker && cachedMarker !== curMarker) return null;
        const at = Number(parsed.at || 0);
        if (maxAgeMs > 0 && (!at || (Date.now() - at) > maxAgeMs)) return null;
        return parsed.data ?? null;
    } catch (_) {
        return null;
    }
}

function _writeUserScopedCache(namespace, data, options = {}) {
    const scope = _getUserCacheScope();
    const key = _userScopedCacheKey(namespace, scope.uid);
    if (!key) return false;
    const storage = options.storage === 'local' ? localStorage : sessionStorage;
    try {
        storage.setItem(key, JSON.stringify({
            uid: scope.uid,
            tokenMarker: scope.tokenMarker || '',
            at: Date.now(),
            data: data
        }));
        return true;
    } catch (_) {
        return false;
    }
}

function _refreshAnalyticsAllOnPageLoad(user, pageName) {
    if (!user || !user.id) return;
    if (window.__um8RefreshAllTriggered) return;
    window.__um8RefreshAllTriggered = true;
    if (typeof window.apiCall !== 'function') return;
    const payload = JSON.stringify({ page: pageName || null, at: Date.now() });
    const prefetch = window.apiCall('/api/analytics/platform-metrics/cached').catch(() => null);
    const req = window.apiCall('/api/analytics/refresh-all?async_mode=true', {
        method: 'POST',
        body: payload
    });
    Promise.allSettled([prefetch, req])
        .then(async (settled) => {
            const before = (settled && settled[0] && settled[0].status === 'fulfilled') ? settled[0].value : null;
            const res = (settled && settled[1] && settled[1].status === 'fulfilled') ? settled[1].value : null;
            const baseTs = String((before && before.fetched_at) || '');
            let latest = null;
            const started = Date.now();
            while ((Date.now() - started) < 9000) {
                await new Promise((r) => setTimeout(r, 1200));
                try {
                    const pm = await window.apiCall('/api/analytics/platform-metrics/cached');
                    const ts = String((pm && pm.fetched_at) || '');
                    if (ts && ts !== baseTs) {
                        latest = pm;
                        break;
                    }
                } catch (_) {}
            }
            try {
                if (window.UploadM8Cache && typeof window.UploadM8Cache.setUserScoped === 'function') {
                    window.UploadM8Cache.setUserScoped('analytics_refresh_all', {
                        result: res || {},
                        metrics: latest || null,
                        refreshed_at: Date.now()
                    }, { storage: 'session' });
                }
            } catch (_) {}
            try {
                const scope = _getUserCacheScope();
                window.dispatchEvent(new CustomEvent('uploadm8:analytics-refresh-all', {
                    detail: {
                        user_id: scope.uid || String(user.id),
                        result: res || null,
                        metrics: latest || null
                    }
                }));
            } catch (_) {}
        })
        .catch(() => null);
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

/**
 * Admin UI + require_admin parity: admin/master_admin role, or master_admin on
 * subscription_tier / tier / plan.tier (some caches omit role).
 */
function _resolveAdminFlags(user) {
    if (!user || typeof user !== 'object') {
        return { isMasterAdminRole: false, isAdminRole: false, role: 'user', tier: 'free' };
    }
    const planT = user.plan && typeof user.plan === 'object' && typeof user.plan.tier === 'string'
        ? String(user.plan.tier).trim().toLowerCase()
        : '';
    const role = String(user.role || 'user').trim().toLowerCase();
    let tier = String(user.tier || user.subscription_tier || 'free').trim().toLowerCase();
    if ((tier === 'free' || !tier) && planT) tier = planT;
    const tierMaster = tier === 'master_admin' || planT === 'master_admin';
    const isMasterAdminRole = role === 'master_admin' || tierMaster;
    const isAdminRole = isMasterAdminRole || role === 'admin';
    return { isMasterAdminRole, isAdminRole, role, tier };
}

function _applyUserToDOM(user) {
    if (!user) return;

    const { isMasterAdminRole, isAdminRole, role, tier } = _resolveAdminFlags(user);

    const displayName = (
        user.name ||
        [user.first_name, user.last_name].filter(Boolean).join(' ').trim() ||
        user.email?.split('@')[0] ||
        'User'
    );

    let displayTier;
    if (isMasterAdminRole) displayTier = 'Master Admin';
    else displayTier = user.tier_display || getTierDisplayName(tier);

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
    //      Persist safe URL in sessionStorage bound to JWT sub (see _persistAvatarUrlForSession).
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
        const sanitizedApi = _sanitizeAvatarUrlForDisplay(cleanUrl);

        if (sanitizedApi) {
            _persistAvatarUrlForSession(sanitizedApi, user.id || user.user_id);
        } else {
            _clearSessionAvatarCache();
        }

        const urlToUse = sanitizedApi || _getSanitizedAvatarUrlFromStorage();

        const existingImg = userAvatar.querySelector('img.um8-avatar-img');

        if (urlToUse) {
            if (existingImg) {
                // Element already exists — only swap src if URL genuinely changed.
                // This path never causes a visual change unless the avatar was updated.
                const prevUrl = existingImg.getAttribute('data-clean-src') || '';
                if (prevUrl !== urlToUse) {
                    existingImg.setAttribute('data-clean-src', urlToUse);
                    existingImg.referrerPolicy = 'no-referrer';
                    existingImg.src = urlToUse;
                }
            } else {
                // First time rendering this element.
                // IMPORTANT: set up the element BEFORE touching userAvatar's children
                // so there is zero frame where the element is empty.
                const img = document.createElement('img');
                img.className  = 'um8-avatar-img';
                img.alt        = '';
                img.referrerPolicy = 'no-referrer';
                img.loading = 'lazy';
                img.decoding = 'async';
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
                    _clearSessionAvatarCache();
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

    // Admin nav block: must not use the generic .admin-only restore path — the sidebar
    // ships with inline display:none, which was being saved as roleDisplay and re-applied
    // after we set block, leaving admin links permanently hidden.
    const adminSection = document.getElementById('adminSection');
    if (adminSection) {
        adminSection.classList.toggle('hidden', !isAdminRole);
        adminSection.style.display = isAdminRole ? 'block' : 'none';
    }

    document.querySelectorAll('.admin-only, .admin-only-marketing').forEach(el => {
        if (el.id === 'adminSection') return;
        if (!el.dataset.roleDisplay) {
            el.dataset.roleDisplay = (el.style && typeof el.style.display === 'string') ? el.style.display : '';
        }
        el.classList.toggle('hidden', !isAdminRole);
        el.style.display = isAdminRole ? (el.dataset.roleDisplay || '') : 'none';
    });
    document.querySelectorAll('.master-admin-only').forEach(el => {
        if (!el.dataset.roleDisplay) {
            el.dataset.roleDisplay = (el.style && typeof el.style.display === 'string') ? el.style.display : '';
        }
        el.classList.toggle('hidden', !isMasterAdminRole);
        el.style.display = isMasterAdminRole ? (el.dataset.roleDisplay || '') : 'none';
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
                free: 'bg-gray', starter: 'bg-blue', solo: 'bg-blue',
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

    if (sessionStorage.getItem('uploadm8_debug_auth') === '1') {
        console.log('[Auth] User:', user.email, '| Role:', role, '| Tier:', tier);
    }
}

// ============================================================
// checkAuth is defined in js/auth-stack.js (loaded before app.js)
// ============================================================

// ============================================================
// Token refresh + login/logout/register + apiCall + checkAuth: js/auth-stack.js
// ============================================================

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
const TIER_SLUGS = ['free','creator_lite','creator_pro','studio','agency','friends_family','lifetime','master_admin'];
const TIER_DISPLAY_NAMES = {
    free: 'Free', creator_lite: 'Creator Lite', creator_pro: 'Creator Pro',
    studio: 'Studio', agency: 'Agency', friends_family: 'Friends & Family',
    lifetime: 'Lifetime', master_admin: 'Admin',
};
const PAID_TIERS = ['creator_lite','creator_pro','studio','agency','friends_family','lifetime'];

function getEntitlements(user) {
    const u = user || currentUser;
    if (!u || typeof u !== 'object') return {};
    if (u.entitlements && typeof u.entitlements === 'object') return u.entitlements;
    if (u.plan && typeof u.plan === 'object') return u.plan;
    return {};
}

function getTier(user) {
    const u = user || currentUser;
    if (!u) return 'free';
    const p = u.plan && typeof u.plan === 'object' && typeof u.plan.tier === 'string' ? u.plan.tier : null;
    return u.tier || u.subscription_tier || p || 'free';
}

function getTierDisplayName(tierOrUser) {
    if (typeof tierOrUser === 'object' && tierOrUser) {
        return tierOrUser.tier_display || TIER_DISPLAY_NAMES[getTier(tierOrUser)] || 'Free';
    }
    const t = tierOrUser || 'free';
    return TIER_DISPLAY_NAMES[t] || (typeof t === 'string' ? t.replace(/_/g,' ').replace(/\b\w/g,c=>c.toUpperCase()) : 'Free');
}

// updateUserUI — wallet-tokens listens on uploadm8:user; omit event until session is validated (cookie may be stale while snapshot is fresh).
function updateUserUI(options) {
    const opts = typeof options === 'object' && options !== null ? options : {};
    const walletBroadcast = opts.walletBroadcast !== false;
    if (!currentUser) return;
    _applyUserToDOM(currentUser); // paint first so we never reveal placeholders
    _markUserReady();
    if (!walletBroadcast) return;
    try { window.dispatchEvent(new CustomEvent('uploadm8:user', { detail: currentUser })); } catch (_) {}
}

function isAdmin()       { return !!currentUser && _resolveAdminFlags(currentUser).isAdminRole; }
function isMasterAdmin() { return !!currentUser && _resolveAdminFlags(currentUser).isMasterAdminRole; }
function isPaidUser()    { if (!currentUser) return false; if (isAdmin()) return true; return PAID_TIERS.includes(getTier(currentUser)); }
function isFreeUser()    { return !isPaidUser(); }
function isFriendsFamily() { return currentUser?.subscription_tier === 'friends_family'; }
function isLifetime()      { return currentUser?.subscription_tier === 'lifetime'; }

function getUserAccessLevel() {
    if (!currentUser) return 'guest';
    const { isMasterAdminRole, isAdminRole, role } = _resolveAdminFlags(currentUser);
    const tier = getTier(currentUser);
    if (isMasterAdminRole) return 'master_admin';
    if (isAdminRole) return 'admin';
    if (['lifetime','friends_family'].includes(tier)) return 'premium';
    if (['agency','studio'].includes(tier))           return 'business';
    if (tier === 'creator_pro')                       return 'pro';
    if (tier === 'creator_lite')     return 'basic';
    return 'free';
}

function hasEntitlement(feature) {
    if (!currentUser) return false;
    const ent  = currentUser.entitlements || {};
    if (_resolveAdminFlags(currentUser).isAdminRole) return true;
    if (['lifetime','friends_family'].includes(getTier(currentUser))) return true;
    if (feature === 'show_ads')  return ent.show_ads === true;
    if (feature === 'no_ads')    return ent.show_ads === false;
    if (['unlimited_accounts','unlimited_connected_accounts','max_accounts'].includes(feature))
        return (ent.max_accounts || 0) >= 999;
    return !!ent[feature];
}

function showsAds() {
    if (!currentUser) return false;
    if (_resolveAdminFlags(currentUser).isAdminRole) return false;
    return (currentUser.entitlements || {}).show_ads === true;
}

function hasWatermark() {
    if (!currentUser) return true;
    if (_resolveAdminFlags(currentUser).isAdminRole) return false;
    return (currentUser.entitlements || {}).can_watermark === true;
}

function getMaxAccounts() {
    if (!currentUser) return 1;
    if (_resolveAdminFlags(currentUser).isAdminRole) return 999;
    return currentUser.entitlements?.max_accounts ?? 4;
}

function getMaxHashtags() {
    if (!currentUser) return 2;
    if (_resolveAdminFlags(currentUser).isAdminRole) return 9999;
    return currentUser.entitlements?.max_hashtags ?? 2;
}

function getUploadLimit(tierOrUser) {
    const u = typeof tierOrUser === 'object' ? tierOrUser : currentUser;
    const t = typeof tierOrUser === 'string' ? tierOrUser : getTier(u);
    if (u?.entitlements?.put_monthly != null) return u.entitlements.put_monthly;
    const limits = { free:120, creator_lite:500, creator_pro:1800, studio:7000, agency:22000, lifetime:12000, friends_family:12000, master_admin:999999 };
    return limits[t] ?? 80;
}

function getTierBadgeHTML(tierOrUser) {
    const t = typeof tierOrUser === 'object' ? getTier(tierOrUser) : (tierOrUser || 'free');
    const colors = { free:'bg-gray', creator_lite:'bg-blue', creator_pro:'bg-orange', studio:'bg-purple', agency:'bg-purple', lifetime:'bg-gradient', friends_family:'bg-gradient', master_admin:'bg-red' };
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

function showModal(first, second) {
    const modalId =
        second !== undefined && second !== null && second !== ''
            ? String(second)
            : String(first);
    const m = document.getElementById(modalId);
    if (m) { m.classList.remove('hidden'); document.body.style.overflow = 'hidden'; }
}
function hideModal(first, second) {
    const modalId =
        second !== undefined && second !== null && second !== ''
            ? String(second)
            : String(first);
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
        let order = null;
        try {
            const raw = sessionStorage.getItem('uploadm8_dashboard_order') || localStorage.getItem('uploadm8_dashboard_order');
            if (raw) {
                order = JSON.parse(raw);
                if (localStorage.getItem('uploadm8_dashboard_order')) {
                    sessionStorage.setItem('uploadm8_dashboard_order', raw);
                    localStorage.removeItem('uploadm8_dashboard_order');
                }
            }
        } catch (_) {}
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
    sessionStorage.setItem('uploadm8_dashboard_order', JSON.stringify(Array.from(grid.querySelectorAll('[data-card-id]')).map(c => c.dataset.cardId)));
    showToast('Dashboard layout saved', 'success');
}
function resetDashboardOrder() {
    try { sessionStorage.removeItem('uploadm8_dashboard_order'); } catch (_) {}
    try { localStorage.removeItem('uploadm8_dashboard_order'); } catch (_) {}
    location.reload();
}

// ============================================================
// Hide Figures — UNIFIED CSS-class approach
// All pages use body.classList 'hide-figures' + CSS .blur-value rules.
// The old [data-hide-value] text-swap approach is retired.
// ============================================================
function _initHideFigures() {
    let v = sessionStorage.getItem('uploadm8_hide_figures');
    if (v == null) {
        try {
            const leg = localStorage.getItem('uploadm8_hide_figures');
            if (leg != null) {
                sessionStorage.setItem('uploadm8_hide_figures', leg);
                localStorage.removeItem('uploadm8_hide_figures');
                v = leg;
            }
        } catch (_) {}
    }
    if (v === 'true') {
        document.body.classList.add('hide-figures');
        _syncHideFiguresIcon(true);
    }
}

function toggleHideFigures() {
    const hidden = document.body.classList.toggle('hide-figures');
    try {
        sessionStorage.setItem('uploadm8_hide_figures', hidden ? 'true' : 'false');
        localStorage.removeItem('uploadm8_hide_figures');
    } catch (_) {}
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
function um8NavHrefIsActive(href) {
    if (!href || href.indexOf('http') === 0) return false;
    const path = window.location.pathname.split('/').pop() || 'dashboard.html';
    const pageHash = (window.location.hash || '').split('?')[0];
    const i = href.indexOf('#');
    const base = i >= 0 ? href.slice(0, i) : href;
    const linkHash = i >= 0 ? href.slice(i) : '';
    if (base !== path) return false;
    if (path === 'guide.html') {
        if (linkHash === '#feat-settings-playbook') return pageHash === linkHash;
        return pageHash !== '#feat-settings-playbook';
    }
    if (!linkHash) return true;
    return pageHash === linkHash;
}

function highlightCurrentNav() {
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.toggle('active', um8NavHrefIsActive(link.getAttribute('href')));
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
async function initApp(pageName, options = {}) {
    const opts = typeof options === 'object' && options !== null ? options : {};
    /** Only set forceAuthRefresh:true when you must block on a fresh /api/me (rare). Default: trust session snapshot + background refresh. */
    const forceAuthRefresh = opts.forceAuthRefresh === true;
    const skipWalletRefresh = opts.skipWalletRefresh === true;

    setTheme(getTheme());
    _initHideFigures();

    const authMessage = sessionStorage.getItem('uploadm8_auth_message');
    if (authMessage) {
        sessionStorage.removeItem('uploadm8_auth_message');
        setTimeout(() => showToast(authMessage, 'warning'), 100);
    }

    // blockUntilValidated: await a real /api/me (or refresh) before wallet/HUD listeners run — avoids 401s when
    // HttpOnly session expired but in-memory / sessionStorage snapshot is still within comfort window.
    const blockUntilValidated = !skipWalletRefresh;
    let user = null;
    const st = window.__um8AuthState;
    const comfortMs =
        typeof window.__UM8_SESSION_COMFORT_MS === 'number' ? window.__UM8_SESSION_COMFORT_MS : 12 * 60 * 60 * 1000;
    const snapAge = st && st.cachedUserAt ? Date.now() - st.cachedUserAt : Infinity;
    const trustHydrated =
        !forceAuthRefresh &&
        st &&
        st.cachedUser &&
        st.cachedUser.email &&
        snapAge >= 0 &&
        snapAge < comfortMs;

    if (trustHydrated) {
        user = st.cachedUser;
        _setCurrentUserState(user);
        updateUserUI({ walletBroadcast: false });
        highlightCurrentNav();
    }
    user = await checkAuth({
        redirectOnFail: true,
        forceRefresh: forceAuthRefresh,
        blockUntilValidated,
    });
    if (!user) return null;

    // Do not force heavy aggregate refresh on every page load.
    // Refreshes are triggered by explicit actions (manual refresh / connect flows).

    // Broadcast wallet only when this page uses wallet refresh (admin lite pages skip).
    updateUserUI({ walletBroadcast: !skipWalletRefresh });
    highlightCurrentNav();

    // Wallet HUD + aggregates: updateUserUI dispatches uploadm8:user → wallet-tokens _fetchAll.
    // Do not also call WalletTokens.refresh here (was a second forced _fetchAll(true) on every load).

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
    const cleanUrl = rawUrl.split('#')[0];
    const safe = _sanitizeAvatarUrlForDisplay(cleanUrl);
    if (!safe || !/^https?:/i.test(safe)) {
        _clearSessionAvatarCache();
        return;
    }

    _persistAvatarUrlForSession(safe, user.id || user.user_id);

    const avatarImage   = document.getElementById('avatarImage');
    const avatarInitial = document.getElementById('avatarInitial');
    if (avatarImage) {
        if (avatarImage.src.split('#')[0] !== safe) avatarImage.src = safe;
        avatarImage.style.display = 'block';
    }
    if (avatarInitial) avatarInitial.style.display = 'none';
}

// ============================================================
// Global Exports
// ============================================================
if (typeof window !== 'undefined') {
    // window.API_BASE already set at top of file
    window._AVATAR_URL_KEY       = _AVATAR_URL_KEY;
    window.APP_VERSION           = APP_VERSION;
    window._setCurrentUserState  = _setCurrentUserState;
    if (typeof _normalizeUserPayload === 'function') window._normalizeUserPayload = _normalizeUserPayload;
    if (typeof _resolveAdminFlags === 'function') window._resolveAdminFlags = _resolveAdminFlags;
    if (typeof _cachePayloadMatchesCurrentToken === 'function') window._cachePayloadMatchesCurrentToken = _cachePayloadMatchesCurrentToken;
    if (typeof _sessionUserSnapshotTrusted === 'function') window._sessionUserSnapshotTrusted = _sessionUserSnapshotTrusted;
    if (typeof _readSessionCache === 'function') window._readSessionCache = _readSessionCache;
    if (typeof _writeSessionCache === 'function') window._writeSessionCache = _writeSessionCache;
    if (typeof window.escapeHTML === 'function') window.escapeHTML = window.escapeHTML;
    if (typeof window.checkAuth === 'function') window.checkAuth = window.checkAuth;
    if (typeof window.login === 'function') window.login = window.login;
    if (typeof window.logout === 'function') window.logout = window.logout;
    if (typeof window.logoutAll === 'function') window.logoutAll = window.logoutAll;
    if (typeof window.register === 'function') window.register = window.register;
    if (typeof window.isLoggedIn === 'function') window.isLoggedIn = window.isLoggedIn;
    if (typeof window.getAccessToken === 'function') {
        window.getAccessToken = window.getAccessToken;
        window.getToken = window.getAccessToken;
    }
    window.clearTokens           = clearTokens;
    if (typeof window.apiCall === 'function') window.apiCall = window.apiCall;
    window.uploadFile            = uploadFile;
    window.cancelUpload          = cancelUpload;
    window.retryUpload           = retryUpload;
    if (typeof getUploadStats !== 'undefined') window.getUploadStats = getUploadStats;
    window.currentUser           = currentUser;
    window.updateUserUI          = updateUserUI;
    window.hasEntitlement        = hasEntitlement;
    window.getEntitlements       = getEntitlements;
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
    window.UploadM8Cache         = {
        getUserScope: _getUserCacheScope,
        getUserScoped: _readUserScopedCache,
        setUserScoped: _writeUserScopedCache,
    };
    if (typeof window.formatDate === 'function') window.formatDate = window.formatDate;
    if (typeof window.formatDateTime === 'function') window.formatDateTime = window.formatDateTime;
    if (typeof window.formatRelativeTime === 'function') window.formatRelativeTime = window.formatRelativeTime;
    if (typeof window.formatNumber === 'function') window.formatNumber = window.formatNumber;
    if (typeof window.formatFileSize === 'function') window.formatFileSize = window.formatFileSize;
    if (typeof window.formatCurrency === 'function') window.formatCurrency = window.formatCurrency;
    if (typeof window.getPlatformInfo === 'function') window.getPlatformInfo = window.getPlatformInfo;
    if (typeof window.getPlatformIcon === 'function') window.getPlatformIcon = window.getPlatformIcon;
    if (typeof window.getPlatformBadge === 'function') window.getPlatformBadge = window.getPlatformBadge;
    if (typeof window.getStatusBadge === 'function') window.getStatusBadge = window.getStatusBadge;
    window.KPI_RANGES            = KPI_RANGES;
    window.getKpiRangeMinutes    = getKpiRangeMinutes;
    window.buildKpiRangeDropdown = buildKpiRangeDropdown;
}

// ============================================================
// MOBILE SIDEBAR — run when DOM is ready (not only DOMContentLoaded)
// If app.js loads after DOMContentLoaded (defer/async/late injection),
// addEventListener('DOMContentLoaded', …) never fires — mobile menu breaks.
// Single source of truth. Flushes any pending toggleSidebar()
// calls that arrived before the DOM was ready.
// ============================================================
function _onAppDomReady() {
    if (window._um8AppDomReadyRan) return;
    window._um8AppDomReadyRan = true;

    // ── Mobile sidebar FIRST (before avatar / role gates / back-button inject) ──
    // If anything below throws, the hamburger must still work.
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

    window._sidebarToggleFn = _toggleSidebar;
    _sidebarReady = true;
    window.toggleSidebar = _toggleSidebar;
    _sidebarCallQueue.forEach(cmd => { if (cmd === 'toggle') _toggleSidebar(); });
    _sidebarCallQueue.length = 0;

    if (menuToggle) {
        try { menuToggle.setAttribute('type', 'button'); } catch (_) {}
        menuToggle.setAttribute('aria-expanded', 'false');
        if (!menuToggle.getAttribute('aria-label')) {
            try { menuToggle.setAttribute('aria-label', 'Open menu'); } catch (_) {}
        }
        const syncMenuAria = () => {
            try {
                menuToggle.setAttribute('aria-expanded', sidebar && sidebar.classList.contains('open') ? 'true' : 'false');
            } catch (_) {}
        };
        const onMenuActivate = e => {
            e.preventDefault();
            e.stopPropagation();
            _toggleSidebar();
            syncMenuAria();
        };
        // pointerup covers mouse + touch + pen in one path; relying on click alone is flaky on
        // some mobile WebKit builds for position:fixed chrome (e.g. analytics and other heavy pages).
        if (typeof window.PointerEvent === 'function') {
            menuToggle.addEventListener('pointerup', e => {
                if (e.pointerType === 'mouse' && e.button !== 0) return;
                onMenuActivate(e);
            }, { passive: false });
        } else {
            menuToggle.addEventListener('click', onMenuActivate, { passive: false });
        }
    }

    if (overlay) {
        const onOverlayClose = e => {
            e.preventDefault();
            e.stopPropagation();
            closeSidebar();
        };
        if (typeof window.PointerEvent === 'function') {
            overlay.addEventListener('pointerup', onOverlayClose, { passive: false });
        } else {
            overlay.addEventListener('click', onOverlayClose, { passive: false });
        }
    }

    if (sidebar) {
        sidebar.addEventListener('click', e => {
            const a = e.target && e.target.closest ? e.target.closest('a.nav-link') : null;
            if (!a || !a.getAttribute('href')) return;
            if (window.innerWidth <= 1024 && sidebar.classList.contains('open')) closeSidebar();
        }, { passive: true });
    }

    window.addEventListener('resize', () => {
        if (window.innerWidth > 1024 && sidebar && sidebar.classList.contains('open')) closeSidebar();
    }, { passive: true });

    if (overlay) overlay.style.display = sidebar?.classList.contains('open') ? 'block' : 'none';

    let touchStartX = 0, touchStartY = 0;
    document.addEventListener('touchstart', e => { touchStartX = e.touches[0].clientX; touchStartY = e.touches[0].clientY; }, { passive: true });
    document.addEventListener('touchend', e => {
        if (!sidebar) return;
        const dx = e.changedTouches[0].clientX - touchStartX;
        const dy = Math.abs(e.changedTouches[0].clientY - touchStartY);
        if (touchStartX < 25  && dx > 60  && dy < 80 && !sidebar.classList.contains('open')) _toggleSidebar();
        if (sidebar.classList.contains('open') && dx < -60 && dy < 80) _toggleSidebar();
    }, { passive: true });

    if (!window.__um8DomHydratedFromSession) {
        _hideRoleGatedUiByDefault();
    }
    // ── Avatar early-inject ─────────────────────────────────────────────────────
    // The IIFE may have painted a background-image CSS rule from session avatar hint.
    // Now that the DOM exists, inject the real <img> element immediately so the
    // browser can confirm the cache hit and start decoding — before checkAuth
    // returns (which could be 200-800 ms away on a slow connection).
    // This means the avatar is in the DOM from the first DOMContentLoaded frame.
    (function injectAvatarImg() {
        try {
            if (!_sessionUserSnapshotTrusted()) return;
            const cachedUrl = _getSanitizedAvatarUrlFromStorage();
            if (!cachedUrl) return;
            const el = document.getElementById('userAvatar');
            if (!el || el.querySelector('img.um8-avatar-img')) return; // already there

            const img = document.createElement('img');
            img.className = 'um8-avatar-img';
            img.alt       = '';
            img.referrerPolicy = 'no-referrer';
            img.loading = 'lazy';
            img.decoding = 'async';
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
                _clearSessionAvatarCache();
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

    // Re-apply cached / in-memory user now that every id= element exists (fixes
    // rare ordering where hydration ran before full DOM, and keeps sidebar + pills in sync).
    try {
        if (typeof currentUser !== 'undefined' && currentUser && currentUser.email) {
            _applyUserToDOM(currentUser);
            window.dispatchEvent(new CustomEvent('uploadm8:user', { detail: currentUser }));
        }
    } catch (_) {}

    // Back button injection on mobile sub-pages
    const currentPage  = window.location.pathname.split('/').pop() || 'index.html';
    const NO_BACK      = ['dashboard.html', 'index.html', 'login.html', 'signup.html', 'check-email.html', 'confirm-email.html', 'forgot-password.html', 'reset-password.html', 'verify-email.html', 'unsubscribe.html', ''];
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
        'kpi.html':               { label:'Dashboard', href:'dashboard.html' },
        'admin-marketing.html':   { label:'Admin',     href:'admin.html'     },
        'admin-data-integrity.html': { label:'Admin',  href:'admin.html'     },
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

    // Apply hide-figures after DOM is ready (in case IIFE ran before body existed)
    _initHideFigures();
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', _onAppDomReady);
} else {
    _onAppDomReady();
}

/**
 * Full stats refresh: enqueue per-upload platform sync, canonical/analytics refresh,
 * and in-process live platform-metrics poll. Dashboard and queue listen for
 * `uploadm8:analytics-refresh-all` with `detail.master` to reload lists and KPIs.
 */
window.uploadM8MasterRefreshStats = async function uploadM8MasterRefreshStats(options) {
    const opts = options || {};
    if (typeof window.apiCall !== 'function') {
        return { ok: false, error: 'no_apiCall' };
    }
    const u = window.currentUser;
    const uid = u && (u.id || u.user_id || u.uid);
    if (!uid) {
        return { ok: false, error: 'no_user' };
    }
    const showToast = typeof window.showToast === 'function' ? window.showToast : function () {};
    const silent = !!opts.silent;
    if (!silent) {
        showToast('Refreshing stats from platforms…', 'info');
    }
    try {
        await window.apiCall('/api/uploads/sync-analytics/all?max_uploads=800&async_mode=true', {
            method: 'POST',
            body: '{}',
        });
        await window.apiCall('/api/analytics/refresh-all?async_mode=true', {
            method: 'POST',
            body: JSON.stringify({ trigger: 'master_refresh' }),
        }).catch(function () {
            return null;
        });
        await window.apiCall('/api/analytics/platform-metrics?force=true').catch(function () {
            return null;
        });
    } catch (e) {
        if (!silent) {
            showToast(e && e.message ? e.message : 'Refresh failed', 'error');
        }
        return { ok: false, error: 'api' };
    }
    try {
        sessionStorage.setItem('uploadm8_master_refresh_at', String(Date.now()));
    } catch (_) {}
    try {
        window.dispatchEvent(
            new CustomEvent('uploadm8:analytics-refresh-all', {
                detail: { user_id: String(uid), master: true, master_refresh: true },
            })
        );
    } catch (_) {}
    if (!silent) {
        showToast('Stats refresh complete', 'success');
    }
    return { ok: true };
};

// NOTE: Do not attach a document-level capture click handler for #menuToggle / #sidebarOverlay.
// It called stopPropagation() before the target phase, so the real button listener never ran on
// some mobile WebKit builds if the delegated path failed — leaving taps swallowed and the menu dead.
// Toggle wiring lives only in _onAppDomReady (menuToggle + overlay listeners).