/**
 * Token- and snapshot-based chrome for marketing + guide pages (no network).
 *
 * REQUIRED ORDER:
 *   js/api-base.js → js/auth-stack.js → js/session-user-hydrate.js → THIS FILE → …
 *
 * Depends on: getAccessToken() (auth-stack), __um8ReadTrustedCachedUser() (session-user-hydrate).
 *
 * EXECUTION: Runs once when parsed, and again on DOMContentLoaded if the document was still
 * loading (covers scripts moved earlier in <body>). Idempotent — safe to call multiple times.
 * With the canonical placement (sync scripts at end of <body>), the first run usually runs
 * after all markup exists, which avoids a logged-in flash on index.html / guide.html before paint.
 *
 * Do not use defer/async on this file in the shell block.
 */
(function () {
    'use strict';

    try {
        if (typeof window !== 'undefined' && typeof window.getAccessToken !== 'function') {
            console.warn('[UploadM8] public-shell-hydrate.js: load js/auth-stack.js before this file.');
        }
    } catch (_) {}

    function tokenPresent() {
        return !!(
            (typeof window.getAccessToken === 'function' && window.getAccessToken()) ||
            sessionStorage.getItem('uploadm8_access_token')
        );
    }

    /** index.html — hero + nav swap when a session token exists */
    function applyIndexMarketingNav() {
        var navLinks = document.getElementById('navLinks');
        var navCta = document.getElementById('navCta');
        var heroCta = document.getElementById('heroCta');
        if (!navLinks || !navCta || !heroCta) return;
        if (!tokenPresent()) return;

        navLinks.innerHTML =
            '<a href="dashboard.html">Dashboard</a>' +
            '<a href="upload.html">Upload</a>' +
            '<a href="queue.html">Queue</a>' +
            '<a href="analytics.html">Analytics</a>' +
            '<a href="settings.html">Settings</a>';
        navCta.innerHTML =
            '<a href="dashboard.html" class="btn btn-primary">Go to Dashboard</a>';
        heroCta.innerHTML =
            '<a href="dashboard.html" class="btn btn-primary btn-large"><i class="fas fa-th-large"></i> Go to Dashboard</a>' +
            '<a href="upload.html" class="btn btn-secondary btn-large"><i class="fas fa-upload"></i> Upload Video</a>';
        var ctaBtn = document.getElementById('ctaBtn');
        if (ctaBtn) {
            ctaBtn.setAttribute('href', 'dashboard.html');
            ctaBtn.innerHTML = '<i class="fas fa-th-large"></i> Go to Dashboard';
        }
        try {
            navLinks.classList.remove('open');
            navCta.classList.remove('open');
        } catch (_) {}
    }

    function tierFromUser(u) {
        if (!u || typeof u !== 'object') return 'free';
        var p = u.plan && typeof u.plan === 'object' && u.plan.tier ? u.plan.tier : null;
        return String(u.tier || u.subscription_tier || p || 'free')
            .trim()
            .toLowerCase();
    }

    /** guide.html — plan banner + filter from trusted snapshot before initApp */
    function applyGuidePlanBannerFromCache() {
        var banner = document.getElementById('yourPlanBanner');
        var planName = document.getElementById('bannerPlanName');
        var planDetail = document.getElementById('bannerPlanDetail');
        if (!banner || !planName) return;

        var u =
            typeof window.__um8ReadTrustedCachedUser === 'function' &&
            window.__um8ReadTrustedCachedUser();
        if (!u) return;

        var tier = tierFromUser(u);
        var NAMES = {
            free: 'Free',
            creator_lite: 'Creator Lite',
            launch: 'Creator Lite',
            creator_pro: 'Creator Pro',
            studio: 'Studio',
            agency: 'Agency',
            master_admin: 'Admin',
            lifetime: 'Lifetime',
            friends_family: 'Friends & Family',
        };
        var COLORS = {
            free: '#9ca3af',
            creator_lite: '#f97316',
            launch: '#f97316',
            creator_pro: '#8b5cf6',
            studio: '#3b82f6',
            agency: '#22c55e',
            master_admin: '#ef4444',
            lifetime: '#fbbf24',
            friends_family: '#fbbf24',
        };
        var puts = {
            free: 80,
            creator_lite: 400,
            creator_pro: 1200,
            studio: 3500,
            agency: 8000,
        };
        var aics = {
            free: 50,
            creator_lite: 120,
            creator_pro: 350,
            studio: 1000,
            agency: 2500,
        };
        var name = NAMES[tier] || tier;
        var col = COLORS[tier] || '#f97316';
        planName.innerHTML =
            '<span style="color:' + col + '">' + name + '</span> — Your current plan';
        if (planDetail) {
            var p = puts[tier];
            var a = aics[tier];
            planDetail.textContent = p
                ? p.toLocaleString() + ' PUT / month · ' + a.toLocaleString() + ' AIC / month'
                : 'Unlimited access';
        }
        banner.style.display = 'flex';
        banner.classList.add('visible');

        var fBtn =
            document.querySelector('.plan-btn[data-tier="' + tier + '"]') ||
            document.querySelector('.plan-btn[data-tier="all"]');
        if (fBtn) {
            document.querySelectorAll('.plan-btn').forEach(function (b) {
                b.classList.remove('active');
            });
            fBtn.classList.add('active');
        }
    }

    function run() {
        applyIndexMarketingNav();
        applyGuidePlanBannerFromCache();
    }

    run();
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', run);
    }
    window.__um8PublicShellHydrate = run;
})();
