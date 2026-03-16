/**
 * UploadM8 Billing Module v2
 * ===========================
 * Handles:
 *   - Wallet display (PUT balance, AIC balance, usage bars)
 *   - Plan selection → Stripe Checkout (with trial badge)
 *   - Billing Portal redirect
 *   - Top-up pack purchases
 *   - Real-time cost estimator
 *   - Trial countdown banner
 *
 * Requires: auth-core.js (apiCall, toast helpers)
 * Include AFTER auth-core.js on any page that has billing UI.
 */

(function () {
    'use strict';

    // ── Constants ────────────────────────────────────────────────────────────
    const TIER_META = {
        free:         { name: 'Free',         price: 0,      put: 60,    aic: 10,    color: '#6b7280' },
        creator_lite: { name: 'Creator Lite', price: 9.99,   put: 300,   aic: 80,    color: '#f97316' },
        creator_pro:  { name: 'Creator Pro',  price: 19.99,  put: 700,   aic: 200,   color: '#f97316' },
        studio:       { name: 'Studio',       price: 49.99,  put: 2000,  aic: 600,   color: '#a855f7' },
        agency:       { name: 'Agency',       price: 99.99,  put: 4500,  aic: 1500,  color: '#ec4899' },
        launch:       { name: 'Launch',       price: 9.99,   put: 300,   aic: 80,    color: '#f97316' },
        master_admin: { name: 'Admin',        price: 0,      put: 999999, aic: 999999, color: '#ef4444' },
        friends_family:{ name: 'F&F',         price: 0,      put: 999999, aic: 999999, color: '#10b981' },
        lifetime:     { name: 'Lifetime',     price: 0,      put: 999999, aic: 999999, color: '#10b981' },
    };

    const TOPUP_PACKS = {
        put: [
            { key: 'uploadm8_put_50',   amount: 50,   price: '$2.99',  label: '50 PUT' },
            { key: 'uploadm8_put_100',  amount: 100,  price: '$4.99',  label: '100 PUT' },
            { key: 'uploadm8_put_250',  amount: 250,  price: '$9.99',  label: '250 PUT',  badge: 'Best Value' },
            { key: 'uploadm8_put_500',  amount: 500,  price: '$17.99', label: '500 PUT' },
            { key: 'uploadm8_put_1000', amount: 1000, price: '$29.99', label: '1000 PUT', badge: 'Best Deal' },
        ],
        aic: [
            { key: 'uploadm8_aic_100',  amount: 100,  price: '$3.99',  label: '100 AIC' },
            { key: 'uploadm8_aic_250',  amount: 250,  price: '$7.99',  label: '250 AIC',  badge: 'Best Value' },
            { key: 'uploadm8_aic_500',  amount: 500,  price: '$14.99', label: '500 AIC' },
            { key: 'uploadm8_aic_1000', amount: 1000, price: '$24.99', label: '1000 AIC', badge: 'Best Deal' },
            { key: 'uploadm8_aic_2500', amount: 2500, price: '$49.99', label: '2500 AIC', badge: 'Agency Pick' },
        ],
    };

    // ── Billing state ────────────────────────────────────────────────────────
    let _user = null;
    let _wallet = null;
    let _entitlements = null;

    // ── Init ─────────────────────────────────────────────────────────────────
    async function initBilling(userData) {
        _user = userData;

        // Load wallet data
        try {
            const w = await apiCall('/api/wallet');
            _wallet = w?.wallet || {};
            _entitlements = w?.entitlements || null;
        } catch (e) {
            console.warn('[Billing] Wallet load failed:', e);
            _wallet = {};
        }

        renderCurrentPlan();
        renderWalletBars();
        renderPlanCards();
        renderTopupPacks();
        renderTrialBanner();
        bindEvents();
    }

    // ── Current Plan Card ────────────────────────────────────────────────────
    function renderCurrentPlan() {
        const tier = _user?.subscription_tier || 'free';
        const meta = TIER_META[tier] || TIER_META.free;
        const status = _user?.subscription_status || 'inactive';

        const planNameEl = document.getElementById('billingPlanName');
        if (planNameEl) {
            planNameEl.textContent = meta.name;
            planNameEl.style.color = meta.color;
        }

        // Show billing status badge
        const statusEl = document.getElementById('billingStatusBadge');
        if (statusEl) {
            const statusMap = {
                active:   { text: 'Active',     cls: 'badge-success' },
                trialing: { text: '7-Day Trial', cls: 'badge-warning' },
                past_due: { text: 'Past Due',    cls: 'badge-danger'  },
                canceled: { text: 'Cancelled',   cls: 'badge-gray'    },
                inactive: { text: tier === 'free' ? 'Free' : 'Inactive', cls: 'badge-gray' },
            };
            const s = statusMap[status] || statusMap.inactive;
            statusEl.innerHTML = `<span class="badge ${s.cls}">${s.text}</span>`;
        }

        // Period end
        const periodEl = document.getElementById('billingPeriodEnd');
        if (periodEl && _user?.current_period_end) {
            const d = new Date(_user.current_period_end);
            periodEl.textContent = 'Renews ' + d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
        }

        // Manage subscription card
        const manageCard = document.getElementById('manageSubCard');
        if (manageCard) {
            manageCard.style.display = _user?.stripe_customer_id ? 'block' : 'none';
        }
    }

    // ── Wallet Bars ──────────────────────────────────────────────────────────
    function renderWalletBars() {
        const tier = _user?.subscription_tier || 'free';
        const meta = TIER_META[tier] || TIER_META.free;

        const putBalance = parseFloat(_wallet?.put_balance || 0);
        const aicBalance = parseFloat(_wallet?.aic_balance || 0);
        const putReserved = parseFloat(_wallet?.put_reserved || 0);
        const aicReserved = parseFloat(_wallet?.aic_reserved || 0);

        const putAvail = putBalance - putReserved;
        const aicAvail = aicBalance - aicReserved;
        const putMonthly = meta.put;
        const aicMonthly = meta.aic;

        // PUT bar
        _setBar('putBar', 'putText', putAvail, putBalance, putMonthly, 'PUT');
        // AIC bar
        _setBar('aicBar', 'aicText', aicAvail, aicBalance, aicMonthly, 'AIC');

        // Reserved indicator
        const resEl = document.getElementById('putReservedText');
        if (resEl && putReserved > 0) {
            resEl.textContent = `${putReserved} reserved`;
            resEl.style.display = 'inline';
        }

        // Warn if low
        if (putAvail < 20 && putAvail >= 0) {
            _showLowBalanceWarning('put', putAvail);
        }
        if (aicAvail < 10 && aicAvail >= 0 && aicMonthly > 0) {
            _showLowBalanceWarning('aic', aicAvail);
        }
    }

    function _setBar(barId, textId, available, balance, monthly, label) {
        const bar = document.getElementById(barId);
        const txt = document.getElementById(textId);
        if (!bar && !txt) return;

        const pct = monthly > 0 ? Math.min(100, (available / monthly) * 100) : 0;
        const isUnlimited = monthly >= 999999;

        if (bar) {
            bar.style.width = isUnlimited ? '100%' : pct + '%';
            // Color: green > 50%, orange 20-50%, red < 20%
            bar.className = 'usage-fill ' + (
                isUnlimited ? 'fill-unlimited' :
                pct > 50    ? 'fill-good' :
                pct > 20    ? 'fill-warn' : 'fill-low'
            );
        }
        if (txt) {
            txt.textContent = isUnlimited
                ? `${balance.toLocaleString()} ${label} (unlimited)`
                : `${available.toLocaleString()} / ${monthly.toLocaleString()} ${label} available`;
        }
    }

    function _showLowBalanceWarning(type, amount) {
        const el = document.getElementById(`${type}LowWarning`);
        if (el) {
            el.textContent = `Low ${type.toUpperCase()} balance (${amount} remaining). Top up to continue uploading.`;
            el.style.display = 'block';
        }
    }

    // ── Plan Cards ───────────────────────────────────────────────────────────
    function renderPlanCards() {
        const currentTier = _user?.subscription_tier || 'free';
        const tierOrder = ['free', 'creator_lite', 'creator_pro', 'studio', 'agency'];
        const currentIdx = tierOrder.indexOf(currentTier);

        document.querySelectorAll('.plan-option[data-tier]').forEach(function (card) {
            const tier = card.dataset.tier;
            const btn = card.querySelector('button[data-plan]');
            const idx = tierOrder.indexOf(tier);
            const meta = TIER_META[tier];

            // Mark current plan
            if (tier === currentTier) {
                card.classList.add('current');
                if (btn) { btn.textContent = '✓ Current Plan'; btn.disabled = true; btn.classList.add('btn-current'); }
            } else {
                card.classList.remove('current');
                if (btn) {
                    btn.disabled = false;
                    btn.classList.remove('btn-current');
                    btn.textContent = idx > currentIdx
                        ? (meta?.trial_days > 0 ? 'Start Free Trial' : 'Upgrade →')
                        : 'Switch Plan';
                }
            }

            // Update PUT/AIC features in card
            const putEl = card.querySelector('[data-feature="put"]');
            if (putEl && meta) putEl.textContent = meta.put >= 999999 ? 'Unlimited PUT' : `${meta.put.toLocaleString()} PUT/month`;
            const aicEl = card.querySelector('[data-feature="aic"]');
            if (aicEl && meta) aicEl.textContent = meta.aic >= 999999 ? 'Unlimited AIC' : `${meta.aic.toLocaleString()} AIC/month`;
        });
    }

    // ── Top-up Packs ─────────────────────────────────────────────────────────
    function renderTopupPacks() {
        ['put', 'aic'].forEach(function (type) {
            const container = document.getElementById(`${type}TopupGrid`);
            if (!container) return;
            container.innerHTML = '';
            TOPUP_PACKS[type].forEach(function (pack) {
                const div = document.createElement('div');
                div.className = 'topup-pack' + (pack.badge ? ' topup-pack-featured' : '');
                div.innerHTML = `
                    ${pack.badge ? `<span class="topup-badge">${pack.badge}</span>` : ''}
                    <div class="topup-amount">${pack.label}</div>
                    <div class="topup-price">${pack.price}</div>
                    <button class="btn btn-sm btn-outline-primary topup-btn"
                            data-key="${pack.key}" data-type="${type}">
                        Buy
                    </button>`;
                container.appendChild(div);
            });
        });
    }

    // ── Trial Banner ─────────────────────────────────────────────────────────
    function renderTrialBanner() {
        const banner = document.getElementById('trialBanner');
        if (!banner) return;

        const status = _user?.subscription_status;
        const trialEnd = _user?.trial_end;

        if (status === 'trialing' && trialEnd) {
            const daysLeft = Math.ceil((new Date(trialEnd) - new Date()) / 86400000);
            if (daysLeft > 0) {
                banner.innerHTML = `
                    <div class="trial-banner-inner">
                        <i class="fas fa-clock"></i>
                        <strong>Free Trial:</strong> ${daysLeft} day${daysLeft !== 1 ? 's' : ''} remaining.
                        Your card will be charged when the trial ends.
                        <a href="#" id="cancelTrialLink">Cancel anytime</a>
                    </div>`;
                banner.style.display = 'block';
                const cancelLink = document.getElementById('cancelTrialLink');
                if (cancelLink) cancelLink.addEventListener('click', openBillingPortal);
            }
        } else {
            banner.style.display = 'none';
        }
    }

    // ── Cost Estimator ───────────────────────────────────────────────────────
    // numPublishTargets: backend uses this for cost (len(target_accounts) or len(platforms) when empty).
    function updateCostEstimate(numPublishTargets, useAi, useHud) {
        const tier = _user?.subscription_tier || 'free';
        const isInternal = ['master_admin','friends_family','lifetime'].includes(tier);

        // PUT formula (matches backend: 10 + (n-1)*2 + HUD + priority + thumbnails)
        const PRIORITY_TIERS = ['creator_pro', 'studio', 'agency'];
        const HUD_TIERS = ['creator_pro', 'studio', 'agency'];
        const AI_DEPTH = { none: 0, basic: 2, enhanced: 3, advanced: 4, max: 6 };
        const AI_DEPTH_BY_TIER = {
            free: 'basic', creator_lite: 'enhanced', creator_pro: 'advanced',
            studio: 'max', agency: 'max', launch: 'enhanced',
        };

        const isPriority = PRIORITY_TIERS.includes(tier);
        const canHud = HUD_TIERS.includes(tier);
        let put = 10;
        if (useHud && canHud) put += 5;
        if (isPriority) put += 5;
        put += Math.max(0, numPublishTargets - 1) * 2;

        const depth = AI_DEPTH_BY_TIER[tier] || 'none';
        const aic = useAi ? AI_DEPTH[depth] || 0 : 0;

        const putEl = document.getElementById('estimatedPutCost');
        const aicEl = document.getElementById('estimatedAicCost');
        const totalEl = document.getElementById('estimatedCostSummary');

        if (putEl) putEl.textContent = isInternal ? '∞' : put + ' PUT';
        if (aicEl) aicEl.textContent = isInternal ? '∞' : aic + ' AIC';
        if (totalEl) {
            const putAvail = (_wallet?.put_balance || 0) - (_wallet?.put_reserved || 0);
            const aicAvail = (_wallet?.aic_balance || 0) - (_wallet?.aic_reserved || 0);
            const canAfford = putAvail >= put && (aic === 0 || aicAvail >= aic);
            totalEl.className = 'cost-summary ' + (canAfford ? 'cost-ok' : 'cost-low');
            totalEl.innerHTML = canAfford
                ? `<i class="fas fa-check-circle"></i> You have enough balance`
                : `<i class="fas fa-exclamation-circle"></i> Insufficient balance — <a href="/settings.html#billing">top up</a>`;
        }
        return { put, aic };
    }

    // ── API Calls ────────────────────────────────────────────────────────────
    async function startCheckout(lookupKey, kind) {
        const btn = document.querySelector(`[data-plan="${lookupKey}"]`);
        if (btn) { btn.disabled = true; btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading...'; }

        try {
            const res = await apiCall('/api/billing/checkout', {
                method: 'POST',
                body: JSON.stringify({ lookup_key: lookupKey, kind: kind || 'subscription' }),
            });
            if (res?.checkout_url) {
                window.location.href = res.checkout_url;
            } else {
                _billingError('Could not start checkout. Please try again.');
            }
        } catch (e) {
            _billingError(e?.message || 'Checkout failed.');
        } finally {
            if (btn) { btn.disabled = false; btn.innerHTML = 'Start Free Trial'; }
        }
    }

    async function openBillingPortal(e) {
        if (e) e.preventDefault();
        const btn = document.getElementById('billingPortalBtn');
        if (btn) { btn.disabled = true; btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Opening...'; }

        try {
            const res = await apiCall('/api/billing/portal', { method: 'POST' });
            if (res?.portal_url) {
                window.open(res.portal_url, '_blank');
            } else {
                _billingError('Could not open billing portal.');
            }
        } catch (e) {
            _billingError('Billing portal unavailable.');
        } finally {
            if (btn) { btn.disabled = false; btn.innerHTML = '<i class="fas fa-external-link-alt"></i> Open Billing Portal'; }
        }
    }

    async function buyTopup(lookupKey, type) {
        const btn = document.querySelector(`.topup-btn[data-key="${lookupKey}"]`);
        if (btn) { btn.disabled = true; btn.textContent = '...'; }

        try {
            const res = await apiCall('/api/billing/checkout', {
                method: 'POST',
                body: JSON.stringify({ lookup_key: lookupKey, kind: 'topup' }),
            });
            if (res?.checkout_url) {
                window.location.href = res.checkout_url;
            } else {
                _billingError('Could not start top-up checkout.');
            }
        } catch (e) {
            _billingError(e?.message || 'Top-up failed.');
        } finally {
            if (btn) { btn.disabled = false; btn.textContent = 'Buy'; }
        }
    }

    // ── Event Bindings ───────────────────────────────────────────────────────
    function bindEvents() {
        // Plan selection buttons
        document.querySelectorAll('[data-plan]').forEach(function (btn) {
            if (btn.dataset.bound) return;
            btn.dataset.bound = 'true';
            btn.addEventListener('click', function () {
                startCheckout(this.dataset.plan, 'subscription');
            });
        });

        // Billing portal
        const portalBtn = document.getElementById('billingPortalBtn');
        if (portalBtn && !portalBtn.dataset.bound) {
            portalBtn.dataset.bound = 'true';
            portalBtn.addEventListener('click', openBillingPortal);
        }

        // Top-up buttons (delegated)
        document.addEventListener('click', function (e) {
            const btn = e.target.closest('.topup-btn');
            if (btn) {
                buyTopup(btn.dataset.key, btn.dataset.type);
            }
        });

        // Tab visibility: refresh wallet when billing tab shown
        document.querySelectorAll('[data-tab="billing"]').forEach(function (tab) {
            tab.addEventListener('click', function () {
                if (_user) renderWalletBars();
            });
        });
    }

    function _billingError(msg) {
        if (typeof toast === 'function') {
            toast(msg, 'error');
        } else {
            alert(msg);
        }
    }

    // ── Public API ───────────────────────────────────────────────────────────
    window.UploadM8Billing = {
        init: initBilling,
        updateCostEstimate: updateCostEstimate,
        openPortal: openBillingPortal,
        startCheckout: startCheckout,
        buyTopup: buyTopup,
        getWallet: function () { return _wallet; },
        getUser: function () { return _user; },
    };

})();
