/**
 * UploadM8 — upgrade-modal.js  v1.0
 * ══════════════════════════════════════════════════════════════════
 * Global upgrade / paywall modal. Include on every page after app.js.
 *
 * Usage:
 *   window.showUpgradeModal(reason, opts)
 *
 * reason strings (shown as headline):
 *   "insufficient_put"     — ran out of upload tokens
 *   "insufficient_aic"     — ran out of AI credits
 *   "account_limit"        — too many connected accounts
 *   "queue_limit"          — queue depth exceeded
 *   "feature_schedule"     — scheduling requires paid plan
 *   "feature_ai"           — AI requires paid plan
 *   "feature_hud"          — HUD requires Pro+
 *   "feature_analytics"    — full analytics requires paid plan
 *   "feature_excel"        — Excel export requires Studio+
 *   "feature_webhooks"     — webhooks require paid plan
 *   "feature_white_label"  — white label requires Agency
 *   Any other string       — shown verbatim as the headline
 *
 * opts:
 *   { currentTier, requiredTier, available, needed, topup: true }
 *
 * Also handles structured 429 errors from apiCall:
 *   window.handleEntitlementError(err)  — pass the caught error object
 * ══════════════════════════════════════════════════════════════════
 */
(function () {
    'use strict';

    /* ── Tier data ─────────────────────────────────────────────── */
    var TIERS = [
        {
            slug: 'free', name: 'Free', price: '$0', color: '#9ca3af',
            put: 60, aic: 10, accounts: 4,
            feats: ['4 platform accounts','Basic analytics'],
            missing: ['AI captions','Scheduling','HUD overlay','Priority queue','Webhooks'],
        },
        {
            slug: 'creator_lite', name: 'Creator Lite', price: '$9.99/mo',
            color: '#f97316', put: 300, aic: 80, accounts: 8,
            lookupKey: 'uploadm8_creatorlite_monthly',
            trial: '7-day free trial',
            feats: ['8 accounts (2/platform)','AI captions','Scheduling','3 thumbnails','Styled thumbnails','Webhooks','Standard analytics'],
            missing: ['HUD overlay','AI thumbnail styling','Priority queue','Excel export'],
        },
        {
            slug: 'creator_pro', name: 'Creator Pro', price: '$19.99/mo',
            color: '#8b5cf6', put: 700, aic: 200, accounts: 20,
            lookupKey: 'uploadm8_creatorpro_monthly',
            trial: '7-day free trial',
            feats: ['20 accounts (5/platform)','Advanced AI','HUD overlay','Priority queue (p2)','5 thumbnails','Styled + AI thumbnails','3 team seats'],
            missing: ['Excel export','White label','Flex transfers'],
        },
        {
            slug: 'studio', name: 'Studio', price: '$49.99/mo',
            color: '#3b82f6', put: 2000, aic: 600, accounts: 60,
            lookupKey: 'uploadm8_studio_monthly',
            trial: '7-day free trial',
            feats: ['60 accounts','Max AI (15 frames)','Excel export','Turbo queue (p1)','10 team seats','Full analytics'],
            missing: ['White label','Flex transfers'],
        },
        {
            slug: 'agency', name: 'Agency', price: '$99.99/mo',
            color: '#22c55e', put: 4500, aic: 1500, accounts: 9999,
            lookupKey: 'uploadm8_agency_monthly',
            trial: '7-day free trial',
            feats: ['Unlimited accounts','Max AI','White label','Top priority (p0)','Flex transfers','25 team seats'],
            missing: [],
        },
    ];

    var TIER_BY_SLUG = {};
    TIERS.forEach(function (t) { TIER_BY_SLUG[t.slug] = t; });
    TIER_BY_SLUG['launch'] = TIER_BY_SLUG['creator_lite'];

    /* ── Reason → copy map ─────────────────────────────────────── */
    var REASON_COPY = {
        insufficient_put:    { icon: '⚡', headline: 'Out of upload tokens (PUT)', sub: 'You\'ve used all your PUT tokens for this billing period. Upgrade for more, or buy a top-up pack.' },
        insufficient_aic:    { icon: '🤖', headline: 'Out of AI credits (AIC)',    sub: 'You\'ve used all your AI credits. Upgrade your plan or buy an AIC top-up pack.' },
        account_limit:       { icon: '🔌', headline: 'Account limit reached',       sub: 'You\'ve connected the maximum number of social accounts for your plan.' },
        queue_limit:         { icon: '📋', headline: 'Queue limit reached',          sub: 'You have too many pending uploads. Upgrade for a larger queue depth.' },
        feature_schedule:    { icon: '📅', headline: 'Scheduling is a paid feature', sub: 'Schedule uploads in advance with Creator Lite or higher.' },
        feature_ai:          { icon: '🤖', headline: 'AI captions require a paid plan', sub: 'Get AI-powered captions, hashtags, and descriptions with Creator Lite+.' },
        feature_hud:         { icon: '🎬', headline: 'HUD overlay requires Creator Pro', sub: 'Burn custom overlays and branding into your videos with Creator Pro+.' },
        feature_analytics:   { icon: '📊', headline: 'Full analytics require a paid plan', sub: 'Unlock views, likes, and engagement tracking across all platforms.' },
        feature_excel:       { icon: '📑', headline: 'Excel export requires Studio', sub: 'Download full analytics reports as formatted Excel files.' },
        feature_webhooks:    { icon: '🔔', headline: 'Webhooks require Creator Lite', sub: 'Get notified via webhook when uploads complete.' },
        feature_white_label: { icon: '🏷️', headline: 'White label requires Agency', sub: 'Remove all UploadM8 branding for your clients.' },
        feature_styled_thumbnails: { icon: '🖼️', headline: 'Styled thumbnails require Creator Lite', sub: 'Add text, badges, and overlays to thumbnails (MrBeast-style) with Creator Lite+.' },
        feature_ai_thumbnail:      { icon: '✨', headline: 'AI thumbnail styling requires Creator Pro', sub: 'Use AI to enhance thumbnails with props and styling. Creator Pro+ only.' },
    };

    /* ── Required tier per feature ─────────────────────────────── */
    var FEATURE_TIER = {
        insufficient_put: null,  // show topup packs
        insufficient_aic: null,  // show topup packs
        account_limit:    'creator_lite',
        queue_limit:      'creator_lite',
        feature_schedule: 'creator_lite',
        feature_ai:       'creator_lite',
        feature_hud:      'creator_pro',
        feature_analytics:'creator_lite',
        feature_excel:    'studio',
        feature_webhooks: 'creator_lite',
        feature_white_label: 'agency',
        feature_styled_thumbnails: 'creator_lite',
        feature_ai_thumbnail:      'creator_pro',
    };

    /* ── Topup packs ───────────────────────────────────────────── */
    var TOPUP_PACKS = {
        put: [
            { key: 'uploadm8_put_50',   label: '50 PUT',     price: '$2.99' },
            { key: 'uploadm8_put_100',  label: '100 PUT',    price: '$4.99' },
            { key: 'uploadm8_put_250',  label: '250 PUT',    price: '$9.99',  best: true },
            { key: 'uploadm8_put_1000', label: '1,000 PUT',  price: '$29.99' },
        ],
        aic: [
            { key: 'uploadm8_aic_100',  label: '100 AIC',    price: '$3.99' },
            { key: 'uploadm8_aic_250',  label: '250 AIC',    price: '$7.99',  best: true },
            { key: 'uploadm8_aic_500',  label: '500 AIC',    price: '$14.99' },
            { key: 'uploadm8_aic_2500', label: '2,500 AIC',  price: '$49.99' },
        ],
    };

    /* ── Helpers ───────────────────────────────────────────────── */
    function currentTierSlug() {
        return (window.currentUser && window.currentUser.subscription_tier) || 'free';
    }
    function apiBase() {
        return window.API_BASE || (typeof location !== 'undefined' && /^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/i.test(location.origin) ? 'http://127.0.0.1:8000' : 'https://auth.uploadm8.com');
    }
    function getToken() {
        return localStorage.getItem('uploadm8_access_token') || sessionStorage.getItem('uploadm8_access_token') || '';
    }
    async function callApi(path, opts) {
        var token = getToken();
        var res = await fetch(apiBase() + path, Object.assign({
            headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + token }
        }, opts || {}));
        if (!res.ok) { var e = await res.json().catch(function(){return{detail:'Error'}}); throw new Error(e.detail||res.status); }
        return res.json();
    }

    /* ── CSS (injected once) ───────────────────────────────────── */
    function injectCSS() {
        if (document.getElementById('um8-upgrade-css')) return;
        var s = document.createElement('style');
        s.id = 'um8-upgrade-css';
        s.textContent = [
            '#um8UgOverlay{position:fixed;inset:0;z-index:99999;background:rgba(0,0,0,.8);backdrop-filter:blur(6px);-webkit-backdrop-filter:blur(6px);display:flex;align-items:center;justify-content:center;padding:1rem;opacity:0;transition:opacity .25s}',
            '#um8UgOverlay.open{opacity:1}',
            '#um8UgModal{background:var(--bg-card,#1a1a24);border:1px solid rgba(255,255,255,.1);border-radius:18px;width:100%;max-width:580px;max-height:90vh;overflow-y:auto;box-shadow:0 40px 80px rgba(0,0,0,.6);transform:translateY(16px) scale(.97);transition:transform .3s cubic-bezier(.16,1,.3,1)}',
            '#um8UgOverlay.open #um8UgModal{transform:none}',
            '#um8UgModal::-webkit-scrollbar{width:4px}#um8UgModal::-webkit-scrollbar-thumb{background:rgba(255,255,255,.1);border-radius:2px}',
            '.um8-rainbow{height:3px;background:linear-gradient(90deg,#f97316,#a855f7,#3b82f6,#22c55e);border-radius:18px 18px 0 0}',
            '.um8-body{padding:1.75rem 2rem 2rem}',
            '.um8-icon{font-size:2.5rem;text-align:center;margin-bottom:.75rem}',
            '.um8-hl{font-size:1.2rem;font-weight:700;text-align:center;color:var(--text-primary,#fff);margin-bottom:.35rem}',
            '.um8-sub{font-size:.83rem;color:var(--text-secondary,#9ca3af);text-align:center;line-height:1.55;margin-bottom:1.25rem}',
            '.um8-sec-label{font-size:.62rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--text-muted,#6b7280);margin-bottom:.6rem}',
            '.um8-plan-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:.6rem;margin-bottom:1.5rem}',
            '.um8-plan-card{border:1px solid rgba(255,255,255,.08);border-radius:10px;padding:.875rem .75rem;cursor:pointer;transition:border-color .15s,transform .15s;text-align:center;position:relative;background:rgba(255,255,255,.02)}',
            '.um8-plan-card:hover{transform:translateY(-2px)}',
            '.um8-plan-card.recommended{border-width:2px}',
            '.um8-plan-name{font-size:.8rem;font-weight:700;margin-bottom:.25rem}',
            '.um8-plan-price{font-size:.72rem;color:var(--text-secondary,#9ca3af);margin-bottom:.5rem}',
            '.um8-plan-feats{font-size:.68rem;color:var(--text-muted,#6b7280);line-height:1.5;text-align:left;padding:0 .25rem}',
            '.um8-plan-trial{font-size:.62rem;color:#22c55e;margin-top:.4rem;font-weight:600}',
            '.um8-rec-badge{position:absolute;top:-10px;left:50%;transform:translateX(-50%);background:var(--accent-orange,#f97316);color:#fff;font-size:.58rem;font-weight:700;padding:.15rem .55rem;border-radius:999px;white-space:nowrap;letter-spacing:.06em}',
            '.um8-btn-primary{display:flex;align-items:center;justify-content:center;gap:.5rem;width:100%;background:linear-gradient(135deg,#f97316,#ea580c);color:#fff;border:none;border-radius:10px;padding:.875rem;font-size:.9rem;font-weight:600;cursor:pointer;transition:transform .15s,box-shadow .15s;text-decoration:none;margin-bottom:.6rem}',
            '.um8-btn-primary:hover{transform:translateY(-1px);box-shadow:0 6px 20px rgba(249,115,22,.3)}',
            '.um8-btn-secondary{display:flex;align-items:center;justify-content:center;gap:.5rem;width:100%;background:transparent;color:var(--text-secondary,#9ca3af);border:1px solid rgba(255,255,255,.1);border-radius:10px;padding:.75rem;font-size:.85rem;font-weight:500;cursor:pointer;transition:border-color .15s;text-decoration:none;margin-bottom:.5rem}',
            '.um8-btn-secondary:hover{border-color:rgba(249,115,22,.35);color:var(--text-primary,#fff)}',
            '.um8-topup-grid{display:grid;grid-template-columns:1fr 1fr;gap:.5rem;margin-bottom:1.25rem}',
            '.um8-topup-item{border:1px solid rgba(255,255,255,.07);border-radius:8px;padding:.6rem .75rem;display:flex;align-items:center;justify-content:space-between;font-size:.78rem;background:rgba(255,255,255,.02);transition:border-color .15s}',
            '.um8-topup-item.best{border-color:rgba(249,115,22,.3)}',
            '.um8-topup-item button{background:rgba(249,115,22,.12);color:#f97316;border:1px solid rgba(249,115,22,.25);border-radius:6px;padding:.25rem .6rem;font-size:.7rem;font-weight:600;cursor:pointer;transition:background .15s}',
            '.um8-topup-item button:hover{background:rgba(249,115,22,.25)}',
            '.um8-divider{height:1px;background:rgba(255,255,255,.06);margin:1.25rem 0}',
            '.um8-close{position:absolute;top:1rem;right:1rem;width:30px;height:30px;background:rgba(255,255,255,.07);border:none;border-radius:50%;color:var(--text-secondary,#9ca3af);cursor:pointer;display:flex;align-items:center;justify-content:center;font-size:.85rem;transition:background .15s}',
            '.um8-close:hover{background:rgba(255,255,255,.12)}',
            '@media(max-width:480px){.um8-body{padding:1.5rem 1.25rem}.um8-plan-grid{grid-template-columns:1fr 1fr}.um8-topup-grid{grid-template-columns:1fr}}',
        ].join('');
        document.head.appendChild(s);
    }

    /* ── Build modal DOM ───────────────────────────────────────── */
    function buildModal() {
        injectCSS();
        var existing = document.getElementById('um8UgOverlay');
        if (existing) existing.remove();

        var overlay = document.createElement('div');
        overlay.id = 'um8UgOverlay';
        overlay.innerHTML = [
            '<div id="um8UgModal" role="dialog" aria-modal="true" aria-labelledby="um8UgHeadline" style="position:relative;">',
            '  <div class="um8-rainbow"></div>',
            '  <div class="um8-body" id="um8UgBody"></div>',
            '</div>',
        ].join('');

        // Close on backdrop click
        overlay.addEventListener('click', function (e) {
            if (e.target === overlay) closeModal();
        });
        // Close on Esc
        document.addEventListener('keydown', function esc(e) {
            if (e.key === 'Escape') { closeModal(); document.removeEventListener('keydown', esc); }
        });

        document.body.appendChild(overlay);
        requestAnimationFrame(function () { overlay.classList.add('open'); });
    }

    function closeModal() {
        var ov = document.getElementById('um8UgOverlay');
        if (!ov) return;
        ov.classList.remove('open');
        setTimeout(function () { if (ov.parentNode) ov.remove(); }, 280);
    }

    /* ── Checkout call ─────────────────────────────────────────── */
    async function goToCheckout(lookupKey, btnEl) {
        var orig = btnEl.innerHTML;
        btnEl.disabled = true;
        btnEl.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading…';
        try {
            var res = await callApi('/api/billing/checkout', {
                method: 'POST',
                body: JSON.stringify({ lookup_key: lookupKey, kind: 'subscription' }),
            });
            if (res && res.checkout_url) window.location.href = res.checkout_url;
            else throw new Error('No checkout URL');
        } catch (e) {
            btnEl.disabled = false;
            btnEl.innerHTML = orig;
            alert('Checkout error: ' + e.message);
        }
    }

    async function goToTopup(lookupKey, btnEl) {
        var orig = btnEl.innerHTML;
        btnEl.disabled = true;
        btnEl.innerHTML = '…';
        try {
            var res = await callApi('/api/billing/checkout', {
                method: 'POST',
                body: JSON.stringify({ lookup_key: lookupKey, kind: 'topup' }),
            });
            if (res && res.checkout_url) window.location.href = res.checkout_url;
            else throw new Error('No checkout URL');
        } catch (e) {
            btnEl.disabled = false;
            btnEl.innerHTML = orig;
            alert('Top-up error: ' + e.message);
        }
    }

    /* ── Render modal content ──────────────────────────────────── */
    function renderContent(reason, opts) {
        opts = opts || {};
        var copy         = REASON_COPY[reason] || { icon: '🔒', headline: reason, sub: 'This feature requires a higher plan.' };
        var curSlug      = opts.currentTier || currentTierSlug();
        var reqSlug      = opts.requiredTier || FEATURE_TIER[reason] || 'creator_lite';
        var curIdx       = TIERS.findIndex(function (t) { return t.slug === curSlug; });
        var showTopupPut = reason === 'insufficient_put';
        var showTopupAic = reason === 'insufficient_aic';
        var showTopup    = showTopupPut || showTopupAic;

        // Which plans to recommend (current+1 up to agency, min 2)
        var eligible = TIERS.filter(function (t) { return t.lookupKey; });
        var reqIdx   = eligible.findIndex(function (t) { return t.slug === reqSlug; });
        var startIdx = Math.max(0, reqIdx);
        var plans    = eligible.slice(startIdx, startIdx + 3);
        if (plans.length === 0) plans = [eligible[0]];

        // Recommended = smallest plan that unlocks the feature
        var recSlug = plans[0] && plans[0].slug;

        var html = [
            '<button class="um8-close" onclick="(function(){var o=document.getElementById(\'um8UgOverlay\');if(o){o.classList.remove(\'open\');setTimeout(function(){o.remove()},280)}})()">',
            '  <i class="fas fa-times"></i>',
            '</button>',
            '<div class="um8-icon">' + copy.icon + '</div>',
            '<div class="um8-hl" id="um8UgHeadline">' + copy.headline + '</div>',
            '<div class="um8-sub">' + copy.sub + '</div>',
        ];

        // Low balance detail
        if ((showTopupPut || showTopupAic) && opts.available != null && opts.needed != null) {
            var tokenType = showTopupPut ? 'PUT' : 'AIC';
            html.push('<div style="background:rgba(239,68,68,.07);border:1px solid rgba(239,68,68,.2);border-radius:8px;padding:.625rem .875rem;font-size:.8rem;color:#ef4444;text-align:center;margin-bottom:1.25rem;">');
            html.push('<i class="fas fa-exclamation-triangle" style="margin-right:.4rem;"></i>');
            html.push(opts.available + ' ' + tokenType + ' available — ' + opts.needed + ' needed</div>');
        }

        // ── Topup packs (for insufficient wallet) ─────────────────
        if (showTopup) {
            var packType = showTopupPut ? 'put' : 'aic';
            var packs    = TOPUP_PACKS[packType] || [];
            html.push('<div class="um8-sec-label">' + (showTopupPut ? '⚡ Buy PUT tokens' : '🤖 Buy AI credits') + '</div>');
            html.push('<div class="um8-topup-grid">');
            packs.forEach(function (p) {
                html.push('<div class="um8-topup-item' + (p.best ? ' best' : '') + '">');
                html.push('<div><div style="font-weight:600;">' + p.label + '</div><div style="font-size:.7rem;color:#9ca3af;">' + p.price + (p.best ? ' · Best value' : '') + '</div></div>');
                html.push('<button data-topup-key="' + p.key + '">Buy</button>');
                html.push('</div>');
            });
            html.push('</div>');
            html.push('<div class="um8-divider"></div>');
            html.push('<div class="um8-sec-label" style="margin-top:0">Or upgrade for more monthly credits</div>');
        }

        // ── Plan cards ─────────────────────────────────────────────
        html.push('<div class="um8-sec-label"' + (showTopup ? ' style="display:none"' : '') + '>Choose a plan</div>');
        html.push('<div class="um8-plan-grid">');
        plans.forEach(function (plan) {
            var isRec = plan.slug === recSlug;
            var col   = plan.color || '#f97316';
            html.push('<div class="um8-plan-card' + (isRec ? ' recommended' : '') + '" style="' + (isRec ? 'border-color:' + col + ';' : '') + '">');
            if (isRec) html.push('<div class="um8-rec-badge" style="background:' + col + '">Recommended</div>');
            html.push('<div class="um8-plan-name" style="color:' + col + '">' + plan.name + '</div>');
            html.push('<div class="um8-plan-price">' + plan.price + '</div>');
            html.push('<div class="um8-plan-feats">');
            plan.feats.slice(0, 4).forEach(function (f) {
                html.push('<div style="margin-bottom:.2rem;"><span style="color:#22c55e;margin-right:.3rem;">✓</span>' + f + '</div>');
            });
            html.push('</div>');
            if (plan.trial) html.push('<div class="um8-plan-trial">🎁 ' + plan.trial + '</div>');
            html.push('<button data-checkout-key="' + plan.lookupKey + '" style="margin-top:.75rem;width:100%;background:' + (isRec ? col : 'rgba(255,255,255,.07)') + ';color:#fff;border:none;border-radius:7px;padding:.5rem;font-size:.75rem;font-weight:600;cursor:pointer;transition:opacity .15s;" onmouseover="this.style.opacity=\'.8\'" onmouseout="this.style.opacity=\'1\'">' + (isRec ? 'Start Free Trial' : 'Select') + '</button>');
            html.push('</div>');
        });
        html.push('</div>');

        // View full pricing
        html.push('<a href="settings.html#billing" class="um8-btn-secondary"><i class="fas fa-credit-card"></i> View all plans & billing</a>');

        // Already on paid plan — open billing portal
        if (curSlug !== 'free' && (showTopupPut || showTopupAic)) {
            html.push('<a href="' + apiBase() + '" class="um8-btn-secondary" id="um8PortalBtn" onclick="event.preventDefault();um8OpenPortal(this)"><i class="fas fa-external-link-alt"></i> Manage subscription</a>');
        }

        return html.join('');
    }

    /* ── Wire up buttons in modal ──────────────────────────────── */
    function wireButtons() {
        var body = document.getElementById('um8UgBody');
        if (!body) return;

        body.querySelectorAll('[data-checkout-key]').forEach(function (btn) {
            btn.addEventListener('click', function () { goToCheckout(this.dataset.checkoutKey, this); });
        });
        body.querySelectorAll('[data-topup-key]').forEach(function (btn) {
            btn.addEventListener('click', function () { goToTopup(this.dataset.topupKey, this); });
        });
    }

    /* ── Public API ────────────────────────────────────────────── */
    function showUpgradeModal(reason, opts) {
        buildModal();
        var body = document.getElementById('um8UgBody');
        body.innerHTML = renderContent(reason || 'account_limit', opts || {});
        wireButtons();
    }

    async function openPortal(btn) {
        var orig = btn && btn.innerHTML;
        if (btn) { btn.disabled = true; btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Opening…'; }
        try {
            var res = await callApi('/api/billing/portal', { method: 'POST' });
            if (res && res.portal_url) window.location.href = res.portal_url;
        } catch (e) {
            if (btn) { btn.disabled = false; btn.innerHTML = orig; }
        }
    }

    /**
     * Pass a caught apiCall error — if it's a billing 429 this auto-opens the modal.
     * Returns true if it handled the error, false if caller should handle it themselves.
     *
     * Example in upload.html:
     *   } catch (err) {
     *     if (!window.handleEntitlementError(err)) throw err;
     *   }
     */
    function handleEntitlementError(err) {
        if (!err || err.status !== 429) return false;
        var resp = err.response || {};
        var code = resp.code || '';
        var opts = {
            available: resp.available,
            needed:    resp.needed,
            topup_url: resp.topup_url,
        };
        if (code === 'insufficient_put') { showUpgradeModal('insufficient_put', opts); return true; }
        if (code === 'insufficient_aic') { showUpgradeModal('insufficient_aic', opts); return true; }
        // Generic queue/rate 429
        if (err.status === 429) { showUpgradeModal('queue_limit', opts); return true; }
        return false;
    }

    // Exports
    window.showUpgradeModal     = showUpgradeModal;
    window.handleEntitlementError = handleEntitlementError;
    window.um8OpenPortal        = openPortal;

})();
