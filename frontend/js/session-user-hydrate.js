/**
 * Session snapshot helpers + synchronous app-shell paint (via shared-sidebar only).
 *
 * REQUIRED ORDER (same on every HTML page that uses this bundle):
 *   1. js/api-base.js
 *   2. js/auth-stack.js (tokens + apiCall + checkAuth)
 *   3. js/session-user-hydrate.js   ← this file (registers APIs only; see below)
 *   4. js/public-shell-hydrate.js   (marketing nav + guide banner; runs immediately)
 *   5. shared-sidebar.js (app shell)  → calls __um8ApplyTrustedSessionChrome() after inject
 *   6. … helpers-formatting, upload-utils, app.js …
 *
 * This file does NOT auto-touch the DOM on parse. shared-sidebar.js invokes
 * __um8ApplyTrustedSessionChrome() once #userAvatar exists. Trust rules mirror app.js
 * (JWT sub + access-token tail marker + snapshot uid).
 *
 * PAINT: Keep these four scripts as consecutive, synchronous <script src> tags at the end
 * of <body> so they execute before the first paint of preceding markup in typical browsers.
 * Do not add defer/async to this block.
 *
 * If you change _applyUserToDOM / snapshot rules in app.js, update this file in parallel.
 */
(function () {
    'use strict';

    try {
        if (typeof window !== 'undefined' && typeof window.getAccessToken !== 'function') {
            console.warn('[UploadM8] session-user-hydrate.js: load js/auth-stack.js before this file.');
        }
    } catch (_) {}

    var TOKEN_KEY = 'uploadm8_access_token';
    var _SESSION_CACHE_KEY = 'uploadm8_cached_user';
    var _SESSION_CACHE_AT_KEY = 'uploadm8_cached_user_at';
    var _AVATAR_SS_URL = 'uploadm8_avatar_ss_url';
    var _AVATAR_SS_SUB = 'uploadm8_avatar_ss_sub';

    var TIER_DISPLAY_NAMES = {
        free: 'Free',
        creator_lite: 'Creator Lite',
        creator_pro: 'Creator Pro',
        studio: 'Studio',
        agency: 'Agency',
        friends_family: 'Friends & Family',
        lifetime: 'Lifetime',
        master_admin: 'Admin',
    };

    function _b64UrlDecode(segment) {
        var b = String(segment || '').replace(/-/g, '+').replace(/_/g, '/');
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

    function _jwtPayload(accessToken) {
        try {
            var parts = String(accessToken || '').split('.');
            if (parts.length < 2) return null;
            var json = _b64UrlDecode(parts[1]);
            if (!json) return null;
            return JSON.parse(json);
        } catch (_) {
            return null;
        }
    }

    function _jwtSub(accessToken) {
        var pl = _jwtPayload(accessToken);
        if (!pl || pl.sub == null) return '';
        return String(pl.sub).trim();
    }

    function _getToken() {
        return (
            (typeof window.getAccessToken === 'function' && window.getAccessToken()) ||
            sessionStorage.getItem(TOKEN_KEY) ||
            ''
        );
    }

    function _cachePayloadMatchesCurrentToken(parsed) {
        if (!parsed || typeof parsed !== 'object') return false;
        var token = _getToken();
        if (!token) return false;
        var marker = token ? String(token).slice(-24) : '';
        var cm = String(parsed.tokenMarker || '');
        if (!cm || !marker || cm !== marker) return false;
        var sub = _jwtSub(token);
        var uid = String(
            parsed.uid ||
                (parsed.user && (parsed.user.id || parsed.user.user_id || parsed.user.uid)) ||
                ''
        ).trim();
        if (!sub || !uid || sub !== uid) return false;
        return true;
    }

    function _normalizeUserPayload(user) {
        if (!user || typeof user !== 'object') return user;
        var planObj = user.plan && typeof user.plan === 'object' ? user.plan : null;
        var planTier =
            planObj && typeof planObj.tier === 'string' ? String(planObj.tier).trim().toLowerCase() : '';

        if (!user.subscription_tier || typeof user.subscription_tier !== 'string') {
            var top = typeof user.tier === 'string' && user.tier ? String(user.tier).trim() : '';
            user.subscription_tier = top || planTier || 'free';
        }
        if (!user.tier || typeof user.tier !== 'string') {
            user.tier = user.subscription_tier || planTier || 'free';
        }
        var sub = String(user.subscription_tier || '').toLowerCase();
        var topTier = String(user.tier || '').toLowerCase();
        if (planTier && planTier !== 'free') {
            if (!sub || sub === 'free') user.subscription_tier = user.plan.tier;
            if (!topTier || topTier === 'free') user.tier = user.plan.tier;
        }

        var subL = String(user.subscription_tier || '').toLowerCase();
        var tierL = String(user.tier || '').toLowerCase();
        var implicitMaster =
            subL === 'master_admin' || tierL === 'master_admin' || planTier === 'master_admin';

        var rawRole =
            user.role != null && String(user.role).trim() !== ''
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

    function _resolveAdminFlags(user) {
        if (!user || typeof user !== 'object') {
            return { isMasterAdminRole: false, isAdminRole: false, role: 'user', tier: 'free' };
        }
        var planT =
            user.plan && typeof user.plan === 'object' && typeof user.plan.tier === 'string'
                ? String(user.plan.tier).trim().toLowerCase()
                : '';
        var role = String(user.role || 'user').trim().toLowerCase();
        var tier = String(user.tier || user.subscription_tier || 'free').trim().toLowerCase();
        if ((tier === 'free' || !tier) && planT) tier = planT;
        var tierMaster = tier === 'master_admin' || planT === 'master_admin';
        var isMasterAdminRole = role === 'master_admin' || tierMaster;
        var isAdminRole = isMasterAdminRole || role === 'admin';
        return { isMasterAdminRole: isMasterAdminRole, isAdminRole: isAdminRole, role: role, tier: tier };
    }

    function _getTier(user) {
        if (!user) return 'free';
        var p =
            user.plan && typeof user.plan === 'object' && typeof user.plan.tier === 'string'
                ? user.plan.tier
                : null;
        return user.tier || user.subscription_tier || p || 'free';
    }

    function _getTierDisplayName(user) {
        if (user && typeof user === 'object') {
            return user.tier_display || TIER_DISPLAY_NAMES[_getTier(user)] || 'Free';
        }
        return 'Free';
    }

    function _sanitizeAvatarUrlForDisplay(url) {
        if (!url || typeof url !== 'string') return null;
        var u = String(url).trim().split('#')[0];
        if (!u.startsWith('http')) return u;
        try {
            var host = new URL(u).hostname.toLowerCase();
            if (host.indexOf('fbcdn.net') !== -1 || host.indexOf('cdninstagram.com') !== -1)
                return null;
        } catch (_) {
            return null;
        }
        return u;
    }

    function _getSanitizedAvatarUrlFromStorage() {
        try {
            var tok = _getToken();
            var sub = _jwtSub(tok);
            if (sub) {
                var owner = sessionStorage.getItem(_AVATAR_SS_SUB);
                var url = sessionStorage.getItem(_AVATAR_SS_URL);
                if (owner === sub) {
                    var s = _sanitizeAvatarUrlForDisplay(url);
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
            var tok = _getToken();
            var sub = _jwtSub(tok) || String(userIdFallback || '').trim();
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

    function _setAvatarInitial(el, user) {
        if (!el) return;
        var span = el.querySelector('.um8-initial');
        if (!span) {
            span = document.createElement('span');
            span.className = 'um8-initial';
            el.textContent = '';
            el.appendChild(span);
        }
        span.textContent = (user.name || user.email || 'U')[0].toUpperCase();
    }

    function _readTrustedCachedUser() {
        try {
            var raw = sessionStorage.getItem(_SESSION_CACHE_KEY);
            var at = sessionStorage.getItem(_SESSION_CACHE_AT_KEY);
            if (!raw || !at) return null;
            var parsed = JSON.parse(raw);
            if (!_cachePayloadMatchesCurrentToken(parsed)) return null;
            var user = _normalizeUserPayload(parsed.user ? parsed.user : parsed);
            if (!user || !user.email) return null;
            return user;
        } catch (_) {
            return null;
        }
    }

    function _injectAvatarPreStyle(safePre) {
        try {
            if (document.getElementById('um8-avatar-pre')) return;
            var style = document.createElement('style');
            style.id = 'um8-avatar-pre';
            style.textContent =
                '#userAvatar{background-image:url("' +
                safePre.replace(/"/g, '\\"') +
                '");background-size:cover;background-position:center;' +
                'font-size:0!important;color:transparent!important;}' +
                '#userAvatar .um8-initial,#userAvatar>span{visibility:hidden!important;}';
            (document.head || document.documentElement).appendChild(style);
        } catch (_) {}
    }

    /**
     * Apply cached user to sidebar + common page IDs in one synchronous pass.
     * @returns {boolean} true if a trusted snapshot was applied to the DOM
     */
    function applyTrustedSessionChrome() {
        var token = _getToken();
        if (!token) {
            window.__um8DomHydratedFromSession = false;
            return false;
        }

        var user = _readTrustedCachedUser();
        if (!user) {
            window.__um8DomHydratedFromSession = false;
            return false;
        }

        var hadAppShellAvatarSlot = !!document.getElementById('userAvatar');

        var flags = _resolveAdminFlags(user);
        var isMasterAdminRole = flags.isMasterAdminRole;
        var isAdminRole = flags.isAdminRole;
        var role = flags.role;
        var tier = flags.tier;

        var displayName =
            user.name ||
            [user.first_name, user.last_name]
                .filter(Boolean)
                .join(' ')
                .trim() ||
            (user.email && user.email.split('@')[0]) ||
            'User';

        var displayTier;
        if (isMasterAdminRole) displayTier = 'Master Admin';
        else displayTier = user.tier_display || _getTierDisplayName(user);

        var textMap = {
            userName: displayName,
            userEmail: user.email || '',
            welcomeName: displayName.split(' ')[0],
            userTier: displayTier,
            userRole: role,
        };
        Object.keys(textMap).forEach(function (id) {
            var el = document.getElementById(id);
            if (el) el.textContent = textMap[id];
        });

        var rawSrc =
            user.avatarSignedUrl ||
            user.avatar_signed_url ||
            user.avatarUrl ||
            user.avatar_url ||
            null;
        var cleanUrl = rawSrc ? String(rawSrc).split('#')[0] : null;
        var sanitizedApi = _sanitizeAvatarUrlForDisplay(cleanUrl);

        if (sanitizedApi) {
            _persistAvatarUrlForSession(sanitizedApi, user.id || user.user_id);
        } else {
            _clearSessionAvatarCache();
        }

        var urlToUse = sanitizedApi || _getSanitizedAvatarUrlFromStorage();
        var userAvatar = document.getElementById('userAvatar');
        if (userAvatar) {
            var safePre = urlToUse ? _sanitizeAvatarUrlForDisplay(urlToUse) : null;
            if (safePre) {
                _injectAvatarPreStyle(safePre);
            }
            var existingImg = userAvatar.querySelector('img.um8-avatar-img');
            if (urlToUse) {
                if (existingImg) {
                    var prevUrl = existingImg.getAttribute('data-clean-src') || '';
                    if (prevUrl !== urlToUse) {
                        existingImg.setAttribute('data-clean-src', urlToUse);
                        existingImg.referrerPolicy = 'no-referrer';
                        existingImg.src = urlToUse;
                    }
                } else {
                    var img = document.createElement('img');
                    img.className = 'um8-avatar-img';
                    img.alt = '';
                    img.referrerPolicy = 'no-referrer';
                    img.loading = 'lazy';
                    img.decoding = 'async';
                    img.setAttribute('data-clean-src', urlToUse);
                    img.style.cssText =
                        'display:block;width:100%;height:100%;' +
                        'border-radius:50%;object-fit:cover;';
                    img.onload = function () {
                        var pre = document.getElementById('um8-avatar-pre');
                        if (pre) pre.remove();
                        userAvatar.querySelectorAll('.um8-initial').forEach(function (e) {
                            e.remove();
                        });
                    };
                    img.onerror = function () {
                        this.remove();
                        _clearSessionAvatarCache();
                        var pre = document.getElementById('um8-avatar-pre');
                        if (pre) pre.remove();
                        _setAvatarInitial(userAvatar, user);
                    };
                    userAvatar.textContent = '';
                    userAvatar.appendChild(img);
                    img.src = urlToUse;
                }
            } else {
                if (existingImg) existingImg.remove();
                var pre = document.getElementById('um8-avatar-pre');
                if (pre) pre.remove();
                _setAvatarInitial(userAvatar, user);
            }
        }

        var adminSection = document.getElementById('adminSection');
        if (adminSection) {
            adminSection.classList.toggle('hidden', !isAdminRole);
            adminSection.style.display = isAdminRole ? 'block' : 'none';
        }

        document.querySelectorAll('.admin-only, .admin-only-marketing').forEach(function (el) {
            if (el.id === 'adminSection') return;
            if (!el.dataset.roleDisplay) {
                el.dataset.roleDisplay =
                    el.style && typeof el.style.display === 'string' ? el.style.display : '';
            }
            el.classList.toggle('hidden', !isAdminRole);
            el.style.display = isAdminRole ? el.dataset.roleDisplay || '' : 'none';
        });
        document.querySelectorAll('.master-admin-only').forEach(function (el) {
            if (!el.dataset.roleDisplay) {
                el.dataset.roleDisplay =
                    el.style && typeof el.style.display === 'string' ? el.style.display : '';
            }
            el.classList.toggle('hidden', !isMasterAdminRole);
            el.style.display = isMasterAdminRole ? el.dataset.roleDisplay || '' : 'none';
        });

        if (hadAppShellAvatarSlot) {
            try {
                document.documentElement.classList.add('um8-user-ready');
            } catch (_) {}
            window.__um8DomHydratedFromSession = true;
            return true;
        }

        window.__um8DomHydratedFromSession = false;
        return false;
    }

    window.__um8ApplyTrustedSessionChrome = applyTrustedSessionChrome;
    window.__um8ReadTrustedCachedUser = _readTrustedCachedUser;
})();
