/**
 * UploadM8 — Billing Success + Post-verify onboarding modal
 * ==========================================================
 * Drop this script tag AFTER app.js and js/tier-catalog.js on the page.
 *
 * 1) Billing: URL ?billing_success=1 (billing/success.html) — Stripe session + /api/me.
 * 2) Onboarding: sessionStorage uploadm8_post_verify_onboarding=1 (set by confirm-email.html
 *    after first successful email verification) — same modal shell, tier/entitlements/steps.
 */

(function () {
    'use strict';

    const QUICK_STEPS = [
        { icon: 'fa-plug', title: 'Connect your platforms', desc: 'Go to <strong>Connected Accounts</strong> and link TikTok, YouTube, Instagram, and Facebook.', link: 'platforms.html', cta: 'Connect now →' },
        { icon: 'fa-cloud-upload-alt', title: 'Upload your first video', desc: 'Drop your video, choose platforms, and hit publish. We handle the rest.', link: 'upload.html', cta: 'Upload now →' },
        { icon: 'fa-robot', title: 'Let AI write your captions', desc: 'Enable AI in the upload form to auto-generate platform-specific captions and hashtag sets.', link: 'guide.html#ai-features', cta: 'Learn more →' },
        { icon: 'fa-calendar-alt', title: 'Schedule your content', desc: 'Set a future publish time and UploadM8 queues, processes, and posts on time — automatically.', link: 'guide.html#scheduling', cta: 'How it works →' },
    ];

    // ── CSS injected once ────────────────────────────────────────────────────
    const CSS = `
#um8-success-overlay {
    position: fixed; inset: 0;
    background: rgba(0,0,0,.75);
    backdrop-filter: blur(6px);
    z-index: 99999;
    display: flex; align-items: center; justify-content: center;
    padding: 1rem;
    animation: um8-fade-in .25s ease both;
}
@keyframes um8-fade-in { from { opacity:0 } to { opacity:1 } }

#um8-success-modal {
    background: #1a1d27;
    border: 1px solid rgba(249,115,22,.3);
    border-radius: 20px;
    width: 100%; max-width: 640px;
    max-height: 90vh;
    display: flex; flex-direction: column;
    overflow: hidden;
    box-shadow: 0 0 80px rgba(249,115,22,.12), 0 30px 80px rgba(0,0,0,.5);
    animation: um8-modal-in .4s cubic-bezier(.16,1,.3,1) both;
    position: relative;
}
@keyframes um8-modal-in {
    from { transform: translateY(30px) scale(.97); opacity:0 }
    to   { transform: translateY(0)    scale(1);   opacity:1 }
}

/* Header glow */
#um8-modal-header {
    background: linear-gradient(135deg, rgba(249,115,22,.15) 0%, rgba(168,85,247,.08) 100%);
    border-bottom: 1px solid rgba(255,255,255,.06);
    padding: 1.5rem 1.75rem 0;
    flex-shrink: 0;
}
.um8-header-top {
    display: flex; align-items: flex-start; justify-content: space-between; gap: 1rem;
    margin-bottom: 1.25rem;
}
.um8-plan-info { display: flex; align-items: center; gap: .75rem; }
.um8-plan-icon {
    width: 48px; height: 48px;
    background: linear-gradient(135deg,#f97316,#ea580c);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem; color: #fff;
    box-shadow: 0 4px 16px rgba(249,115,22,.4);
    animation: um8-icon-pop .4s cubic-bezier(.16,1,.3,1) .15s both;
}
@keyframes um8-icon-pop { from { transform:scale(0) rotate(-20deg); opacity:0 } to { transform:scale(1) rotate(0); opacity:1 } }
.um8-plan-badge {
    display: inline-flex; align-items: center; gap: .35rem;
    background: rgba(249,115,22,.15);
    border: 1px solid rgba(249,115,22,.35);
    border-radius: 999px;
    padding: .25rem .75rem;
    font-size: .8rem; font-weight: 700;
    color: #f97316;
    letter-spacing: .02em;
}
.um8-close-btn {
    background: rgba(255,255,255,.06);
    border: 1px solid rgba(255,255,255,.1);
    border-radius: 8px;
    width: 32px; height: 32px;
    display: flex; align-items: center; justify-content: center;
    cursor: pointer; color: #9ca3af; font-size: .85rem;
    transition: background .15s, color .15s;
    flex-shrink: 0;
}
.um8-close-btn:hover { background: rgba(239,68,68,.15); color: #ef4444; border-color: rgba(239,68,68,.3); }

/* Tab bar */
.um8-tabs {
    display: flex; gap: 0;
    border-bottom: 1px solid rgba(255,255,255,.06);
    margin: 0 -1.75rem;
    padding: 0 1.75rem;
}
.um8-tab {
    padding: .6rem .9rem;
    font-size: .82rem; font-weight: 600;
    color: #6b7280;
    cursor: pointer;
    border-bottom: 2px solid transparent;
    transition: color .15s, border-color .15s;
    white-space: nowrap;
}
.um8-tab:hover { color: #d1d5db; }
.um8-tab.active { color: #f97316; border-bottom-color: #f97316; }

/* Body */
#um8-modal-body {
    flex: 1; overflow-y: auto; padding: 1.5rem 1.75rem;
    scrollbar-width: thin; scrollbar-color: rgba(255,255,255,.1) transparent;
}
.um8-tab-panel { display: none; animation: um8-panel-in .2s ease both; }
.um8-tab-panel.active { display: block; }
@keyframes um8-panel-in { from { opacity:0; transform:translateY(8px) } to { opacity:1; transform:translateY(0) } }

/* ── Tab 1: Welcome ─────────────────────────────────────── */
.um8-welcome-hero { text-align: center; padding: .5rem 0 1.25rem; }
.um8-check-ring {
    width: 72px; height: 72px;
    background: linear-gradient(135deg,#f97316,#ea580c);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 1rem;
    font-size: 1.75rem; color: #fff;
    box-shadow: 0 0 0 0 rgba(249,115,22,.5);
    animation: um8-ring .4s cubic-bezier(.16,1,.3,1) .2s both, um8-ping 2s ease-out .6s;
}
@keyframes um8-ring { from { transform:scale(0); opacity:0 } to { transform:scale(1); opacity:1 } }
@keyframes um8-ping {
    0%   { box-shadow: 0 0 0 0 rgba(249,115,22,.5) }
    70%  { box-shadow: 0 0 0 20px rgba(249,115,22,0) }
    100% { box-shadow: 0 0 0 0 rgba(249,115,22,0) }
}
.um8-welcome-headline { font-size: 1.5rem; font-weight: 700; margin-bottom: .4rem; }
.um8-welcome-sub { font-size: .9rem; color: #9ca3af; line-height: 1.6; max-width: 400px; margin: 0 auto; }
.um8-trial-pill {
    display: inline-flex; align-items: center; gap: .4rem;
    background: rgba(245,158,11,.1); border: 1px solid rgba(245,158,11,.3);
    color: #f59e0b; border-radius: 999px;
    padding: .35rem .9rem; font-size: .78rem; font-weight: 600;
    margin-top: .75rem;
}
.um8-wallet-row {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: .75rem; margin-top: 1.25rem;
}
.um8-wallet-card {
    background: rgba(255,255,255,.04);
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 12px; padding: 1rem; text-align: center;
}
.um8-wallet-card:hover { border-color: rgba(249,115,22,.3); }
.um8-wallet-emoji { font-size: 1.3rem; margin-bottom: .3rem; }
.um8-wallet-num {
    font-size: 1.7rem; font-weight: 800;
    background: linear-gradient(135deg,#f97316,#fbbf24);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.um8-wallet-lbl { font-size: .7rem; color: #6b7280; text-transform: uppercase; letter-spacing: .08em; margin-top: .1rem; }

/* ── Tab 2: What's Included ─────────────────────────────── */
.um8-feature-grid {
    display: flex; flex-direction: column; gap: .4rem;
}
.um8-feature-row {
    display: flex; align-items: flex-start; gap: .75rem;
    padding: .65rem .75rem; border-radius: 10px;
    background: rgba(255,255,255,.03);
    border: 1px solid transparent;
    transition: background .15s, border-color .15s;
}
.um8-feature-row:hover { background: rgba(255,255,255,.05); border-color: rgba(255,255,255,.08); }
.um8-feature-row.included { border-color: rgba(16,185,129,.1); }
.um8-feature-row.locked {
    opacity: .45;
    background: rgba(0,0,0,.15);
}
.um8-feat-icon {
    width: 28px; height: 28px; flex-shrink: 0;
    background: rgba(16,185,129,.12); border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    font-size: .75rem; color: #10b981;
}
.um8-feat-icon.locked-icon { background: rgba(107,114,128,.1); color: #6b7280; }
.um8-feat-label { font-size: .88rem; font-weight: 600; color: #f3f4f6; margin-bottom: .15rem; }
.um8-feat-desc  { font-size: .78rem; color: #6b7280; line-height: 1.4; }
.um8-feat-upgrade {
    margin-left: auto; flex-shrink: 0;
    font-size: .68rem; color: #f97316;
    background: rgba(249,115,22,.08);
    border: 1px solid rgba(249,115,22,.2);
    border-radius: 999px; padding: .1rem .5rem;
    white-space: nowrap;
}

/* ── Tab 3: Quick Start ─────────────────────────────────── */
.um8-steps { display: flex; flex-direction: column; gap: .75rem; }
.um8-step {
    display: flex; align-items: flex-start; gap: 1rem;
    background: rgba(255,255,255,.03);
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 12px; padding: 1rem 1.1rem;
    transition: border-color .15s, background .15s;
    text-decoration: none; color: inherit;
}
.um8-step:hover { border-color: rgba(249,115,22,.35); background: rgba(249,115,22,.04); }
.um8-step-num {
    width: 36px; height: 36px; flex-shrink: 0;
    background: linear-gradient(135deg,#f97316,#ea580c);
    border-radius: 50%; display: flex; align-items: center; justify-content: center;
    font-size: .8rem; font-weight: 800; color: #fff;
}
.um8-step-title { font-size: .9rem; font-weight: 600; margin-bottom: .2rem; }
.um8-step-desc  { font-size: .8rem; color: #9ca3af; line-height: 1.5; }
.um8-step-cta   { font-size: .78rem; color: #f97316; margin-top: .3rem; font-weight: 500; }

/* Footer */
#um8-modal-footer {
    padding: 1rem 1.75rem;
    border-top: 1px solid rgba(255,255,255,.06);
    display: flex; align-items: center; justify-content: space-between; gap: 1rem;
    flex-wrap: wrap; flex-shrink: 0;
    background: rgba(0,0,0,.2);
}
.um8-footer-hint { font-size: .78rem; color: #6b7280; }
.um8-footer-hint a { color: #f97316; }
.um8-btn-primary {
    display: inline-flex; align-items: center; gap: .5rem;
    background: linear-gradient(135deg,#f97316,#ea580c);
    color: #fff; border: none; border-radius: 8px;
    padding: .6rem 1.25rem; font-size: .875rem; font-weight: 600;
    cursor: pointer; text-decoration: none;
    transition: transform .15s, box-shadow .15s;
}
.um8-btn-primary:hover { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(249,115,22,.35); }
.um8-btn-ghost {
    display: inline-flex; align-items: center; gap: .4rem;
    background: transparent; color: #9ca3af;
    border: 1px solid rgba(255,255,255,.1);
    border-radius: 8px; padding: .6rem 1rem; font-size: .875rem;
    cursor: pointer; text-decoration: none;
    transition: border-color .15s, color .15s;
}
.um8-btn-ghost:hover { border-color: rgba(255,255,255,.25); color: #f3f4f6; }

/* Confetti */
.um8-confetti {
    position: fixed; pointer-events: none; z-index: 100000;
    width: 8px; height: 8px; border-radius: 2px;
    animation: um8-fall linear both;
}
@keyframes um8-fall {
    0%   { transform: translateY(-20px) rotate(0deg);   opacity: 1 }
    100% { transform: translateY(105vh)  rotate(720deg); opacity: 0 }
}
`;

    // ── Helpers ──────────────────────────────────────────────────────────────
    function getToken() {
        return (typeof window.getAccessToken === 'function' && window.getAccessToken()) ||
            sessionStorage.getItem('uploadm8_access_token') ||
            sessionStorage.getItem('access_token') ||
            null;
    }

    async function apiGet(path) {
        if (typeof window.apiCall !== 'function') throw new Error('success-modal: load js/auth-stack.js');
        return window.apiCall(path, { method: 'GET' });
    }

    function fmt(n) {
        if (n >= 999999) return '∞';
        return n.toLocaleString();
    }

    function fireConfetti() {
        const colors = ['#f97316','#fb923c','#fbbf24','#34d399','#60a5fa','#a78bfa','#f472b6'];
        for (let i = 0; i < 80; i++) {
            const el = document.createElement('div');
            el.className = 'um8-confetti';
            el.style.cssText = [
                `left:${Math.random() * 100}vw`,
                `top:-15px`,
                `background:${colors[Math.floor(Math.random() * colors.length)]}`,
                `width:${5 + Math.random() * 9}px`,
                `height:${5 + Math.random() * 9}px`,
                `animation-duration:${1.4 + Math.random() * 2.2}s`,
                `animation-delay:${Math.random() * .9}s`,
                `border-radius:${Math.random() > .5 ? '50%' : '2px'}`,
            ].join(';');
            document.body.appendChild(el);
            el.addEventListener('animationend', () => el.remove());
        }
    }

    // ── Build modal HTML ─────────────────────────────────────────────────────
    function buildModal(user, sessionData, opts) {
        opts = opts || {};
        const onboarding = !!opts.onboarding;
        const TC = window.UploadM8TierCatalog;
        if (!TC || typeof TC.getTierMeta !== 'function') return null;
        const tier = (typeof window.getTier === 'function' && user)
            ? window.getTier(user)
            : (user?.tier || user?.subscription_tier || user?.plan?.tier || sessionData?.tier || 'free');
        const meta = TC.getTierMeta(tier, 'free');
        if (!meta) return null;
        const ent = (typeof window.getEntitlements === 'function' && user)
            ? window.getEntitlements(user)
            : (user?.entitlements || user?.plan || null);
        const isTrialing = !onboarding && (user?.subscription_status === 'trialing' || sessionData?.subscription_status === 'trialing');
        const wallet = user?.wallet || {};
        const putBal = wallet.put_balance != null ? wallet.put_balance : meta.put;
        const aicBal = wallet.aic_balance != null ? wallet.aic_balance : meta.aic;
        const features = TC.successModalFeatureList(tier, ent);
        const trialDays = meta.trial_days || 7;

        const welcomeSub = onboarding
            ? `Your email is verified. You are on <strong>${meta.name}</strong> — here is what you can use right now. Open the <strong>Setup Handbook</strong> anytime from the Feature Guide.`
            : `Your <strong>${meta.name}</strong> subscription is active and your wallet has been funded for this billing period.`;

        // ── Tab 1: Welcome ────────────────────────────────────────────────
        const welcomeHTML = `
<div class="um8-welcome-hero">
    <div class="um8-check-ring"><i class="fas fa-check"></i></div>
    <div class="um8-welcome-headline">You're in, ${(user?.name || user?.email || '').split(' ')[0] || 'friend'}!</div>
    <div class="um8-welcome-sub">${welcomeSub}</div>
    ${isTrialing ? `<div class="um8-trial-pill"><i class="fas fa-clock"></i> ${trialDays}-day free trial — your card charges when the trial ends</div>` : ''}
</div>
<div class="um8-wallet-row">
    <div class="um8-wallet-card">
        <div class="um8-wallet-emoji"></div>
        <div class="um8-wallet-num">${fmt(putBal)}</div>
        <div class="um8-wallet-lbl">PUT Tokens Available</div>
    </div>
    <div class="um8-wallet-card">
        <div class="um8-wallet-emoji"></div>
        <div class="um8-wallet-num">${fmt(aicBal)}</div>
        <div class="um8-wallet-lbl">AI Credits Available</div>
    </div>
</div>`;

        // ── Tab 2: What's Included ────────────────────────────────────────
        const featureRows = features.map(([label, desc, included, req]) => `
<div class="um8-feature-row ${included ? 'included' : 'locked'}">
    <div class="um8-feat-icon ${!included ? 'locked-icon' : ''}">
        <i class="fas ${included ? 'fa-check' : 'fa-lock'}"></i>
    </div>
    <div>
        <div class="um8-feat-label">${label}</div>
        <div class="um8-feat-desc">${desc}</div>
    </div>
    ${!included && req ? `<span class="um8-feat-upgrade">${req.replace('_', ' ')}+</span>` : ''}
</div>`).join('');

        const includedHTML = `<div class="um8-feature-grid">${featureRows}</div>`;

        // ── Tab 3: Quick Start ────────────────────────────────────────────
        const stepsHTML = `
<div class="um8-steps">
${QUICK_STEPS.map((s, i) => `
<a href="${s.link}" class="um8-step" data-um8-fn="um8Close">
    <div class="um8-step-num">${i + 1}</div>
    <div>
        <div class="um8-step-title"><i class="fas ${s.icon}" style="color:#f97316;margin-right:.4rem;"></i>${s.title}</div>
        <div class="um8-step-desc">${s.desc}</div>
        <div class="um8-step-cta">${s.cta}</div>
    </div>
</a>`).join('')}
</div>`;

        return { welcomeHTML, includedHTML, stepsHTML, meta, tier, trialDays, onboarding };
    }

    // ── Render into DOM ──────────────────────────────────────────────────────
    function renderModal(user, sessionData, opts) {
        const built = buildModal(user, sessionData, opts);
        if (!built) {
            console.warn('[success-modal] tier catalog not ready');
            return;
        }
        const { welcomeHTML, includedHTML, stepsHTML, meta, onboarding } = built;

        const ariaLabel = onboarding ? 'Welcome to UploadM8' : 'Subscription confirmed';
        const badgeLine = onboarding
            ? `<i class="fas fa-envelope-circle-check"></i> ${meta.name} — email verified`
            : `<i class="fas fa-check-circle"></i> ${meta.name} — Active`;

        const overlay = document.createElement('div');
        overlay.id = 'um8-success-overlay';
        overlay.innerHTML = `
<div id="um8-success-modal" role="dialog" aria-modal="true" aria-label="${ariaLabel}">
    <div id="um8-modal-header">
        <div class="um8-header-top">
            <div class="um8-plan-info">
                <div class="um8-plan-icon" style="background:linear-gradient(135deg,${meta.color},${meta.color}cc)">
                    <i class="fas ${meta.icon}"></i>
                </div>
                <div>
                    <div class="um8-plan-badge">
                        ${badgeLine}
                    </div>
                    <div style="font-size:.75rem;color:#6b7280;margin-top:.25rem;">
                        ${meta.put >= 999999 ? 'Unlimited' : meta.put.toLocaleString() + ' PUT'} · 
                        ${meta.aic >= 999999 ? 'Unlimited' : meta.aic.toLocaleString() + ' AIC'} per month
                    </div>
                </div>
            </div>
            <button type="button" class="um8-close-btn" data-um8-fn="um8Close" aria-label="Close">
                <i class="fas fa-times"></i>
            </button>
        </div>
        <div class="um8-tabs">
            <div class="um8-tab active" data-panel="welcome" role="button" tabindex="0" data-um8-fn="um8Tab" data-um8-arg="welcome"> Welcome</div>
            <div class="um8-tab" data-panel="included" role="button" tabindex="0" data-um8-fn="um8Tab" data-um8-arg="included"> What's Included</div>
            <div class="um8-tab" data-panel="quickstart" role="button" tabindex="0" data-um8-fn="um8Tab" data-um8-arg="quickstart"> Quick Start</div>
        </div>
    </div>
    <div id="um8-modal-body">
        <div class="um8-tab-panel active" id="um8-panel-welcome">${welcomeHTML}</div>
        <div class="um8-tab-panel" id="um8-panel-included">${includedHTML}</div>
        <div class="um8-tab-panel" id="um8-panel-quickstart">${stepsHTML}</div>
    </div>
    <div id="um8-modal-footer">
        <div class="um8-footer-hint">
            <i class="fas fa-book-open" style="color:#f97316;margin-right:.3rem;"></i>
            Full feature guide → <a href="guide.html" data-um8-fn="um8Close">guide.html</a>
        </div>
        <div style="display:flex;gap:.6rem;">
            <a href="guide.html" class="um8-btn-ghost" data-um8-fn="um8Close">
                <i class="fas fa-book"></i> Full Guide
            </a>
            <a href="upload.html" class="um8-btn-primary" data-um8-fn="um8Close">
                <i class="fas fa-cloud-upload-alt"></i> Start Uploading
            </a>
        </div>
    </div>
</div>`;

        document.body.appendChild(overlay);

        // Close on backdrop click
        overlay.addEventListener('click', e => { if (e.target === overlay) um8Close(); });

        // Keyboard close
        document.addEventListener('keydown', function handler(e) {
            if (e.key === 'Escape') { um8Close(); document.removeEventListener('keydown', handler); }
        });

        fireConfetti();
    }

    // ── Global tab / close functions (need global scope for onclick attrs) ──
    window.um8Tab = function (first, second) {
        const el = first && first.currentTarget ? first.currentTarget : first;
        const panelId =
            second !== undefined && second !== null && second !== '' ? String(second) : '';
        if (!el || !panelId) return;
        document.querySelectorAll('.um8-tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.um8-tab-panel').forEach(p => p.classList.remove('active'));
        el.classList.add('active');
        const panel = document.getElementById('um8-panel-' + panelId);
        if (panel) panel.classList.add('active');
    };

    window.um8Close = function () {
        const overlay = document.getElementById('um8-success-overlay');
        if (!overlay) return;
        overlay.style.opacity = '0';
        overlay.style.transition = 'opacity .2s ease';
        setTimeout(() => overlay.remove(), 210);
        // Clean URL
        const url = new URL(window.location.href);
        url.searchParams.delete('billing_success');
        url.searchParams.delete('session_id');
        url.searchParams.delete('onboarding');
        window.history.replaceState({}, '', url.toString());
    };

    // ── Init ─────────────────────────────────────────────────────────────────
    async function init() {
        const params = new URLSearchParams(window.location.search);
        let onboarding = false;
        try {
            if (sessionStorage.getItem('uploadm8_post_verify_onboarding') === '1') {
                sessionStorage.removeItem('uploadm8_post_verify_onboarding');
                onboarding = true;
            }
        } catch (_) {}

        if (!params.has('billing_success') && !onboarding) return;

        // dashboard.html already shows billingSuccessOverlay for ?billing_success=1 — skip duplicate modal
        if (params.has('billing_success') && !onboarding) {
            try {
                if (document.getElementById('billingSuccessOverlay')) return;
            } catch (_) {}
        }

        // Inject CSS once
        if (!document.getElementById('um8-success-css')) {
            const style = document.createElement('style');
            style.id = 'um8-success-css';
            style.textContent = CSS;
            document.head.appendChild(style);
        }

        const sessionId = params.get('session_id');
        let user = window.currentUser || null;
        let sessionData = null;

        if (window.UploadM8TierCatalog && typeof window.UploadM8TierCatalog.load === 'function') {
            try { await window.UploadM8TierCatalog.load(); } catch (e) { console.warn('[success-modal] tier catalog', e); }
        }

        if (onboarding) {
            for (let i = 0; i < 8; i++) {
                try {
                    const me = await apiGet('/api/me');
                    user = (typeof window._normalizeUserPayload === 'function' && me)
                        ? window._normalizeUserPayload(Object.assign({}, me))
                        : me;
                    if (user && user.email) break;
                } catch (e) {
                    if (i < 7) await new Promise(r => setTimeout(r, 400));
                }
            }
            renderModal(user, null, { onboarding: true });
            return;
        }

        // Fetch session data from our billing endpoint
        if (sessionId) {
            try {
                sessionData = await apiGet('/api/billing/session?session_id=' + encodeURIComponent(sessionId));
            } catch (e) {
                console.warn('[UploadM8] Could not fetch billing session:', e);
            }
        }

        // Poll /api/me until subscription shows active (webhook may be in flight)
        if (!user || user.subscription_tier === 'free') {
            for (let i = 0; i < 8; i++) {
                try {
                    const me = await apiGet('/api/me');
                    user = (typeof window._normalizeUserPayload === 'function' && me)
                        ? window._normalizeUserPayload(Object.assign({}, me))
                        : me;
                    if (user?.subscription_status === 'active' || user?.subscription_status === 'trialing') {
                        break;
                    }
                    if (i < 7) await new Promise(r => setTimeout(r, 1500));
                } catch (e) { break; }
            }
        }

        renderModal(user, sessionData, { onboarding: false });
    }

    // Run after DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        // Small delay so app.js can set window.currentUser first
        setTimeout(init, 600);
    }
})();
