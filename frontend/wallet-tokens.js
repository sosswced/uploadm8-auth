/**
 * UploadM8 — Wallet Token Engine v2.1
 * Aligned to app.py Production Build v4 (entitlements.py single source of truth)
 *
 * Key API contracts consumed:
 *   GET /api/me              → wallet, tier (canonical), tier_display, entitlements
 *   GET /api/entitlements/tiers → canonical tier list + PUT/AIC monthly values
 *   GET /api/dashboard/stats → uploads, engagement, quota
 *   GET /api/analytics?range=30d → put_used, aic_used, views, likes
 *   GET /api/wallet          → ledger (last 50 entries)
 *
 * /api/me response fields used:
 *   user.tier           → canonical slug (normalize_tier applied server-side)
 *   user.tier_display   → human-readable display name
 *   user.subscription_tier → raw DB value (fallback)
 *   user.entitlements.put_monthly
 *   user.entitlements.aic_monthly
 *   user.entitlements.can_ai
 *   user.entitlements.can_schedule
 *   user.wallet.put_balance / put_reserved / aic_balance / aic_reserved
 *
 * HTML usage:
 *   Pills:  <div data-wt-pills></div>
 *   Banner: <div data-wt-banner></div>
 *   HUD FAB: auto-injected (pass noHud:true to opts to disable)
 *
 * Events:
 *   window.dispatchEvent(new CustomEvent('uploadm8:user', { detail: userObj }))
 *   → re-renders everything with fresh user data immediately
 *
 * Backward compat (BILLING-UPSELL-BANNERS.md contract fully preserved):
 *   WalletTokens.getWalletFromUser(user)
 *   WalletTokens.getUsageLevel(available, monthly)
 *   WalletTokens.renderTokenPills(container)
 *   WalletTokens.renderUsageBanner(container, user)
 *   WalletTokens.UPLOAD_ESTIMATE
 */

(function () {
  'use strict';

  /* ─────────────────────────────────────────────
   * CONSTANTS — mirrors entitlements.py defaults
   * ───────────────────────────────────────────── */
  const PUT_PER_UPLOAD  = 16;   // compute_upload_cost base (1 platform, no AI, no HUD)
  const AIC_PER_UPLOAD  = 3;    // AIC per job
  const BILLING_URL     = 'settings.html#billing';
  const PRICING_URL     = 'index.html#pricing';
  const API_BASE        = (typeof window !== 'undefined' && window.API_BASE)
    ? window.API_BASE
    : (typeof location !== 'undefined' && /^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/i.test(location.origin)
      ? 'http://127.0.0.1:8000'
      : '');
  const CACHE_TTL_MS    = 60_000; // 60s min between full fetches
  const HUD_REFRESH_MS  = 90_000; // re-render interval

  /* Fallback tier metadata — overwritten by /api/entitlements/tiers on init */
  let _tierMeta = {
    free:          { label: 'Free',         badge: 'FREE',   color: '#6b7280', next: 'creator_lite',  put: 80,   aic: 50   },
    creator_lite:  { label: 'Creator Lite', badge: 'LITE',   color: '#3b82f6', next: 'creator_pro',   put: 400,  aic: 120  },
    creator_pro:   { label: 'Creator Pro',  badge: 'PRO',    color: '#8b5cf6', next: 'studio',        put: 1200, aic: 350  },
    studio:        { label: 'Studio',       badge: 'STUDIO', color: '#f59e0b', next: 'agency',        put: 3500, aic: 1000 },
    agency:        { label: 'Agency',       badge: 'AGENCY', color: '#10b981', next: null,            put: 8000, aic: 2500 },
    friends_family:{ label: 'Friends & Family', badge: 'F&F', color: '#ec4899', next: null, put: 12000,aic: 5000 },
    lifetime:      { label: 'Lifetime',     badge: 'LIFE',   color: '#f97316', next: null,            put: 12000,aic: 5000 },
    master_admin:  { label: 'Admin',        badge: 'ADMIN',  color: '#ef4444', next: null,            put: 999999,aic: 999999 },
    launch:        { label: 'Creator Lite', badge: 'LITE',   color: '#3b82f6', next: 'creator_pro',   put: 400,  aic: 120  },
  };

  const TIER_FEATURES = {
    creator_lite: ['3× PUT tokens', 'AI Captions & Hashtags', 'Priority Queue', '4 platforms'],
    creator_pro:  ['6× PUT tokens', 'Advanced AI', 'Smart Scheduler', 'Multi-account'],
    studio:       ['18× PUT tokens', 'White Label', 'Studio Analytics', 'Team Support'],
    agency:       ['60× PUT tokens', 'API Access', 'Custom Domains', 'Dedicated Manager'],
  };

  /* ─────────────────────────────────────────────
   * STATE
   * ───────────────────────────────────────────── */
  let _s = {
    user:       null,
    wallet:     null,
    ent:        null,         // user.entitlements (from /api/me)
    dash:       null,         // /api/dashboard/stats
    analytics:  null,         // /api/analytics
    ledger:     [],           // /api/wallet ledger
    tierConfig: null,         // /api/entitlements/tiers (server canonical)
    lastFetch:  0,
    inFlight:   false,
  };

  /* ─────────────────────────────────────────────
   * CSS INJECTION
   * ───────────────────────────────────────────── */
  function _css() {
    if (document.getElementById('wt-css')) return;
    const el = document.createElement('style');
    el.id = 'wt-css';
    el.textContent = `
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&display=swap');

/* ── Pills ──────────────────────────────────── */
.wt-pills-wrap{display:inline-flex;align-items:center;gap:6px}
.wt-tier-badge{display:inline-flex;align-items:center;padding:2px 8px;border-radius:4px;
  font-size:10px;font-weight:800;letter-spacing:.08em;font-family:'DM Mono',monospace}
.wt-pills{display:inline-flex;align-items:center;gap:5px;font-family:'DM Mono',monospace}
.wt-pill{display:inline-flex;align-items:center;gap:4px;padding:3px 10px;border-radius:20px;
  font-size:11px;font-weight:700;letter-spacing:.04em;text-transform:uppercase;
  text-decoration:none;cursor:pointer;position:relative;overflow:hidden;white-space:nowrap;
  border:1.5px solid transparent;transition:transform .15s,box-shadow .15s}
.wt-pill::before{content:'';position:absolute;inset:0;
  background:linear-gradient(135deg,rgba(255,255,255,.1),transparent 60%);pointer-events:none}
.wt-pill:hover{transform:translateY(-1px) scale(1.04);box-shadow:0 4px 16px rgba(0,0,0,.35)}
.wt-pill-icon{font-size:10px}
.wt-pill-bar{position:absolute;bottom:0;left:0;height:2px;border-radius:0 0 20px 20px;
  transition:width .6s cubic-bezier(.4,0,.2,1);opacity:.7}
.wt-pill-healthy {background:linear-gradient(135deg,#052e16,#14532d);color:#4ade80;border-color:#16a34a33}
.wt-pill-notice  {background:linear-gradient(135deg,#1c1917,#292524);color:#a8a29e;border-color:#57534e33}
.wt-pill-low     {background:linear-gradient(135deg,#1c1400,#2a1d00);color:#fbbf24;border-color:#d9770633}
.wt-pill-critical{background:linear-gradient(135deg,#1f0000,#3b0000);color:#f87171;border-color:#dc262633;
  animation:wt-pulse 1.8s ease-in-out infinite}
.wt-pill-empty   {background:linear-gradient(135deg,#0f0f0f,#1a1a1a);color:#4b5563;border-color:#37415133;
  cursor:not-allowed;filter:grayscale(.4)}
.wt-pill-healthy .wt-pill-bar{background:#4ade80}
.wt-pill-low     .wt-pill-bar{background:#fbbf24}
.wt-pill-critical .wt-pill-bar{background:#f87171}
@keyframes wt-pulse{0%,100%{box-shadow:0 0 0 0 rgba(239,68,68,0)}50%{box-shadow:0 0 0 5px rgba(239,68,68,.18)}}

/* ── Banners ────────────────────────────────── */
.wt-banner{width:100%;border-radius:12px;padding:14px 16px;margin:12px 0;
  display:flex;align-items:flex-start;gap:12px;position:relative;overflow:hidden;
  font-family:-apple-system,'Segoe UI',sans-serif;
  animation:wt-banner-in .35s cubic-bezier(.34,1.56,.64,1) forwards}
@keyframes wt-banner-in{from{opacity:0;transform:translateY(-8px)}to{opacity:1;transform:translateY(0)}}
.wt-banner::after{content:'';position:absolute;inset:0;
  background:linear-gradient(135deg,rgba(255,255,255,.035),transparent 55%);pointer-events:none}
.wt-banner-low{background:linear-gradient(135deg,#1c1400,#221800);
  border:1px solid rgba(251,191,36,.22);box-shadow:0 4px 24px rgba(251,191,36,.07)}
.wt-banner-critical{background:linear-gradient(135deg,#1f0000,#2d0000);
  border:1px solid rgba(248,113,113,.28);
  animation:wt-banner-in .35s cubic-bezier(.34,1.56,.64,1) forwards,wt-crit-glow 3s ease-in-out infinite .5s}
@keyframes wt-crit-glow{0%,100%{box-shadow:0 4px 24px rgba(239,68,68,.1)}
  50%{box-shadow:0 4px 40px rgba(239,68,68,.25)}}
.wt-banner-empty{background:linear-gradient(135deg,#0f0f0f,#151515);
  border:1px solid rgba(107,114,128,.22)}
.wt-banner-upsell{background:linear-gradient(135deg,#0d0d1a,#130d26);
  border:1px solid rgba(139,92,246,.28);box-shadow:0 4px 24px rgba(139,92,246,.1)}
.wt-banner-icon{font-size:22px;line-height:1;flex-shrink:0;margin-top:1px}
.wt-banner-body{flex:1;min-width:0}
.wt-banner-title{font-size:13px;font-weight:700;line-height:1.2;margin-bottom:3px}
.wt-banner-low    .wt-banner-title{color:#fde68a}
.wt-banner-critical .wt-banner-title{color:#fca5a5}
.wt-banner-empty  .wt-banner-title{color:#9ca3af}
.wt-banner-upsell .wt-banner-title{color:#c4b5fd}
.wt-banner-msg{font-size:12px;line-height:1.5}
.wt-banner-low    .wt-banner-msg{color:#fbbf24}
.wt-banner-critical .wt-banner-msg{color:#f87171}
.wt-banner-empty  .wt-banner-msg{color:#6b7280}
.wt-banner-upsell .wt-banner-msg{color:#a78bfa}
.wt-banner-sub{font-size:11px;margin-top:4px;opacity:.75;line-height:1.4;color:#d1d5db}
.wt-banner-upsell .wt-banner-sub{color:#8b7cb6}
.wt-feature-list{display:flex;flex-wrap:wrap;gap:5px;margin-top:8px}
.wt-feat-chip{display:inline-flex;align-items:center;gap:3px;padding:2px 7px;border-radius:99px;
  font-size:10px;font-weight:600;background:rgba(139,92,246,.14);
  border:1px solid rgba(139,92,246,.24);color:#c4b5fd}
.wt-roi-badge{display:inline-flex;align-items:center;gap:5px;padding:4px 10px;border-radius:6px;
  background:rgba(16,185,129,.1);border:1px solid rgba(16,185,129,.2);
  font-family:'DM Mono',monospace;font-size:11px;font-weight:700;color:#34d399;margin-top:8px}
.wt-banner-actions{display:flex;gap:6px;margin-top:10px;flex-wrap:wrap}
.wt-bbtn{display:inline-flex;align-items:center;gap:4px;padding:6px 12px;border-radius:8px;
  font-size:11px;font-weight:700;letter-spacing:.04em;text-decoration:none;
  cursor:pointer;border:none;transition:all .15s;font-family:inherit}
.wt-bbtn-primary{background:linear-gradient(135deg,#f97316,#dc2626);color:#fff;
  box-shadow:0 2px 10px rgba(249,115,22,.3)}
.wt-bbtn-primary:hover{filter:brightness(1.1);box-shadow:0 4px 16px rgba(249,115,22,.4)}
.wt-bbtn-purple{background:linear-gradient(135deg,#7c3aed,#6d28d9);color:#fff;
  box-shadow:0 2px 10px rgba(124,58,237,.3)}
.wt-bbtn-purple:hover{filter:brightness(1.1)}
.wt-bbtn-secondary{background:rgba(255,255,255,.06);color:#d1d5db;
  border:1px solid rgba(255,255,255,.1)}
.wt-bbtn-secondary:hover{background:rgba(255,255,255,.1);color:#f9fafb}
.wt-dismiss{position:absolute;top:8px;right:10px;background:none;border:none;cursor:pointer;
  color:rgba(255,255,255,.22);font-size:16px;line-height:1;padding:2px;transition:color .15s}
.wt-dismiss:hover{color:rgba(255,255,255,.6)}

/* ── HUD ────────────────────────────────────── */
.wt-fab{position:fixed;bottom:24px;right:24px;z-index:9989;width:48px;height:48px;
  border-radius:50%;background:linear-gradient(135deg,#1f1f1f,#111);
  border:1.5px solid rgba(255,255,255,.1);box-shadow:0 8px 24px rgba(0,0,0,.45);
  display:flex;align-items:center;justify-content:center;cursor:pointer;
  transition:transform .2s,box-shadow .2s;font-size:20px}
.wt-fab:hover{transform:scale(1.08);box-shadow:0 12px 32px rgba(0,0,0,.55)}
.wt-fab-dot{position:absolute;top:3px;right:3px;width:10px;height:10px;border-radius:50%;
  background:#f97316;border:2px solid #111;animation:wt-dot 2s ease-in-out infinite}
@keyframes wt-dot{0%,100%{transform:scale(1);opacity:1}50%{transform:scale(1.4);opacity:.8}}
.wt-hud{position:fixed;bottom:24px;right:24px;z-index:9990;width:320px;border-radius:16px;
  overflow:hidden;font-family:'DM Mono','Fira Code',monospace;background:#0a0a0a;
  box-shadow:0 20px 60px rgba(0,0,0,.55),0 0 0 1px rgba(255,255,255,.06);
  transform:translateY(120%);transition:transform .35s cubic-bezier(.34,1.56,.64,1)}
.wt-hud.open{transform:translateY(0)}
.wt-hud-head{display:flex;align-items:center;justify-content:space-between;
  padding:14px 16px 10px;background:linear-gradient(135deg,#111,#1a1a1a);
  border-bottom:1px solid rgba(255,255,255,.07)}
.wt-hud-title{font-size:11px;font-weight:800;letter-spacing:.12em;text-transform:uppercase;color:#e5e7eb}
.wt-hud-x{background:none;border:none;color:#6b7280;cursor:pointer;font-size:16px;line-height:1;
  padding:0;width:20px;height:20px;display:flex;align-items:center;justify-content:center;
  border-radius:4px;transition:color .15s,background .15s}
.wt-hud-x:hover{color:#e5e7eb;background:rgba(255,255,255,.08)}
.wt-hud-body{padding:14px 16px;max-height:80vh;overflow-y:auto}
.wt-hud-body::-webkit-scrollbar{width:3px}
.wt-hud-body::-webkit-scrollbar-thumb{background:rgba(255,255,255,.12);border-radius:99px}
.wt-meter{margin-bottom:14px}
.wt-meter-row{display:flex;align-items:center;justify-content:space-between;margin-bottom:5px}
.wt-meter-lbl{font-size:10px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#9ca3af}
.wt-meter-val{font-size:13px;font-weight:700;color:#f9fafb}
.wt-meter-sub{font-size:10px;color:#6b7280;margin-left:2px}
.wt-track{height:5px;border-radius:99px;background:rgba(255,255,255,.06);overflow:hidden}
.wt-fill{height:100%;border-radius:99px;transition:width .8s cubic-bezier(.4,0,.2,1)}
.wt-fill-healthy {background:linear-gradient(90deg,#16a34a,#4ade80)}
.wt-fill-low     {background:linear-gradient(90deg,#d97706,#fbbf24)}
.wt-fill-critical{background:linear-gradient(90deg,#b91c1c,#f87171);animation:wt-shim 1.5s linear infinite}
.wt-fill-empty   {background:#374151}
@keyframes wt-shim{0%,100%{opacity:1}50%{opacity:.5}}
.wt-kpi-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:12px 0}
.wt-kpi{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.06);
  border-radius:8px;padding:8px 10px}
.wt-kpi-n{font-size:16px;font-weight:800;color:#f9fafb;line-height:1}
.wt-kpi-l{font-size:9px;color:#6b7280;text-transform:uppercase;letter-spacing:.08em;margin-top:2px}
.wt-kpi-d{font-size:9px;font-weight:700;margin-left:3px}
.wt-kpi-d.pos{color:#4ade80}.wt-kpi-d.neg{color:#f87171}
.wt-burn{background:rgba(249,115,22,.08);border:1px solid rgba(249,115,22,.2);
  border-radius:8px;padding:8px 10px;margin:10px 0;display:flex;align-items:center;gap:8px}
.wt-burn-ico{font-size:14px}
.wt-burn-txt{font-size:10px;color:#d1d5db;line-height:1.4}
.wt-burn-txt strong{color:#fb923c}
.wt-vel-wrap{margin:10px 0}
.wt-vel-lbl{display:flex;justify-content:space-between;font-size:9px;color:#6b7280;
  text-transform:uppercase;letter-spacing:.07em;margin-bottom:4px}
.wt-vel-track{height:3px;background:rgba(255,255,255,.06);border-radius:99px;overflow:hidden}
.wt-vel-fill{height:100%;border-radius:99px;background:linear-gradient(90deg,#8b5cf6,#f97316);
  transition:width .6s ease}
.wt-ledger{margin-top:12px;border-top:1px solid rgba(255,255,255,.06);padding-top:10px}
.wt-ledger-ttl{font-size:9px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;
  color:#6b7280;margin-bottom:6px}
.wt-ledger-row{display:flex;align-items:center;justify-content:space-between;
  padding:3px 0;border-bottom:1px solid rgba(255,255,255,.04);font-size:9px}
.wt-ledger-row:last-child{border-bottom:none}
.wt-ledger-type{color:#6b7280;font-size:8px;width:28px}
.wt-ledger-rsn{color:#9ca3af;text-transform:capitalize;flex:1;margin:0 6px}
.wt-ledger-d{font-weight:700}
.wt-ledger-d.pos{color:#4ade80}.wt-ledger-d.neg{color:#f87171}
.wt-ledger-dt{color:#4b5563;font-size:8px;white-space:nowrap}
.wt-cta-row{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:14px}
.wt-hbtn{display:flex;align-items:center;justify-content:center;gap:4px;padding:8px 10px;
  border-radius:8px;font-size:10px;font-weight:800;letter-spacing:.06em;text-transform:uppercase;
  text-decoration:none;cursor:pointer;border:none;transition:all .15s;font-family:inherit}
.wt-hbtn-primary{background:linear-gradient(135deg,#f97316,#ea580c);color:#fff;
  box-shadow:0 4px 12px rgba(249,115,22,.3)}
.wt-hbtn-primary:hover{filter:brightness(1.1);transform:translateY(-1px)}
.wt-hbtn-secondary{background:rgba(255,255,255,.06);color:#d1d5db;
  border:1px solid rgba(255,255,255,.1)}
.wt-hbtn-secondary:hover{background:rgba(255,255,255,.1);color:#f9fafb}
.wt-ai-badge{display:inline-flex;align-items:center;gap:5px;padding:3px 8px;border-radius:6px;
  background:rgba(139,92,246,.12);border:1px solid rgba(139,92,246,.22);
  font-size:9px;font-weight:700;color:#c4b5fd;margin-top:6px;font-family:'DM Mono',monospace}
@media(max-width:480px){
  .wt-hud{width:calc(100vw - 32px);right:16px;bottom:16px}
  .wt-fab{right:16px;bottom:16px}
}
`;
    document.head.appendChild(el);
  }

  /* ─────────────────────────────────────────────
   * LEVEL LOGIC
   * ───────────────────────────────────────────── */
  function _lvl(avail, monthly) {
    if (avail <= 0) return 'empty';
    if (!monthly || monthly <= 0) return 'healthy';
    const pct = (avail / monthly) * 100;
    if (pct <= 5  || avail <= PUT_PER_UPLOAD)     return 'critical';
    if (pct <= 15 || avail <= PUT_PER_UPLOAD * 3) return 'low';
    if (pct <= 30) return 'notice';
    return 'healthy';
  }

  function _lvlIcon(l) {
    return { healthy: '●', notice: '◐', low: '◑', critical: '⚠', empty: '⊘' }[l] || '●';
  }

  function _fillCls(l) {
    return { healthy: 'wt-fill-healthy', notice: 'wt-fill-healthy',
             low: 'wt-fill-low', critical: 'wt-fill-critical',
             empty: 'wt-fill-empty' }[l] || 'wt-fill-healthy';
  }

  /* ─────────────────────────────────────────────
   * TIER RESOLUTION
   * New app.py returns user.tier (canonical, normalized) and user.tier_display.
   * Falls back gracefully to subscription_tier for older tokens.
   * ───────────────────────────────────────────── */
  function _resolveTier(u) {
    const slug = u?.tier || u?.subscription_tier || 'free';
    const meta = _tierMeta[slug] || _tierMeta.free;
    const label = u?.tier_display || meta.label;
    return { slug, label, meta: { ...meta, label } };
  }

  /* ─────────────────────────────────────────────
   * METRICS DERIVATION
   * ───────────────────────────────────────────── */
  const _INTERNAL_TIERS = ['master_admin', 'friends_family', 'lifetime'];

  function _isInternal(u) {
    const t = u?.tier || u?.subscription_tier || '';
    const r = u?.role || '';
    return _INTERNAL_TIERS.includes(t) || r === 'master_admin' || r === 'admin';
  }

  function _metrics() {
    const u   = _s.user    || {};
    const w   = u.wallet   || _s.wallet || {};
    const ent = u.entitlements || _s.ent || {};
    const plan= u.plan     || {};
    const d   = _s.dash    || {};
    const a   = _s.analytics || {};

    const internal = _isInternal(u);

    const putBal  = Math.max(0, Math.floor(parseFloat(w.put_balance  || 0)));
    const aicBal  = Math.max(0, Math.floor(parseFloat(w.aic_balance  || 0)));
    const putRes  = Math.max(0, Math.floor(parseFloat(w.put_reserved || 0)));
    const aicRes  = Math.max(0, Math.floor(parseFloat(w.aic_reserved || 0)));
    let putAvail = Math.max(0, putBal - putRes);
    let aicAvail = Math.max(0, aicBal - aicRes);

    const putMo  = parseInt(ent.put_monthly || plan.put_monthly || 80,  10) || 80;
    const aicMo  = parseInt(ent.aic_monthly || plan.aic_monthly || 50,  10) || 50;

    // Internal tiers bypass wallet balance — they always have unlimited tokens
    if (internal) {
      putAvail = Math.max(putAvail, putMo);
      aicAvail = Math.max(aicAvail, aicMo);
    }

    const putPct = putMo  > 0 ? Math.min(100, (putAvail / putMo)  * 100) : 100;
    const aicPct = aicMo  > 0 ? Math.min(100, (aicAvail / aicMo) * 100) : 100;

    const putLvl  = _lvl(putAvail, putMo);
    const aicLvl  = _lvl(aicAvail, aicMo);
    const overall = (putLvl === 'empty'    || aicLvl === 'empty')    ? 'empty'
                  : (putLvl === 'critical' || aicLvl === 'critical') ? 'critical'
                  : (putLvl === 'low'      || aicLvl === 'low')      ? 'low'
                  : (putLvl === 'notice'   || aicLvl === 'notice')   ? 'notice'
                  : 'healthy';

    const uploadsLeft = Math.floor(Math.min(putAvail / PUT_PER_UPLOAD, aicAvail / AIC_PER_UPLOAD));

    const totalUps  = parseInt(d.uploads?.total     || 0, 10);
    const doneUps   = parseInt(d.uploads?.completed || 0, 10);
    const inQueue   = parseInt(d.uploads?.in_queue  || 0, 10);
    const putUsed   = parseInt(d.quota?.put_used    || a.put_used || 0, 10);
    const aicUsed   = parseInt(a.aic_used  || 0, 10);
    const views30   = parseInt(d.engagement?.views  || a.views   || 0, 10);
    const likes30   = parseInt(d.engagement?.likes  || a.likes   || 0, 10);

    const burnDay   = putUsed > 0 ? putUsed / 30 : 0;
    const daysLeft  = burnDay > 0 ? Math.floor(putAvail / burnDay) : 999;
    const roi       = putUsed > 0 ? Math.round(views30 / putUsed) : 0;
    const successPct= totalUps > 0 ? Math.round((doneUps / totalUps) * 100) : 0;
    const velUsed   = putMo > 0 ? Math.min(100, ((putMo - putAvail) / putMo) * 100) : 0;

    const canAi       = !!(ent.can_ai       || plan.ai);
    const canSchedule = !!(ent.can_schedule || plan.scheduler);

    const { slug: tier, label: tierLabel, meta: tierM } = _resolveTier(u);
    const nextTier  = tierM.next;

    return {
      putAvail, aicAvail, putRes, aicRes, putBal, aicBal,
      putMo, aicMo, putPct, aicPct,
      putLvl, aicLvl, overall,
      uploadsLeft, totalUps, doneUps, inQueue,
      putUsed, aicUsed, views30, likes30,
      burnDay, daysLeft, roi, successPct, velUsed,
      canAi, canSchedule,
      tier, tierLabel, tierM, nextTier,
    };
  }

  /* ─────────────────────────────────────────────
   * PUT GAIN LOOKUP
   * ───────────────────────────────────────────── */
  function _putGain(currentTier, nextTier) {
    const current = (_s.tierConfig?.[currentTier]?.put_monthly) || _tierMeta[currentTier]?.put || 80;
    const next    = (_s.tierConfig?.[nextTier]?.put_monthly)    || _tierMeta[nextTier]?.put    || current;
    const delta   = next - current;
    return delta > 0 ? `+${delta}` : 'more';
  }

  /* ─────────────────────────────────────────────
   * UPSELL COPY ENGINE
   * ───────────────────────────────────────────── */
  function _upsellCfg(m) {
    const { tier, nextTier, uploadsLeft, daysLeft, roi,
            views30, putUsed, totalUps, doneUps, successPct, canAi } = m;

    if (!nextTier) {
      if (m.putAvail < PUT_PER_UPLOAD * 10) {
        return {
          icon: '💳', cls: 'upsell',
          title: 'Top Up Your Arsenal',
          msg: `${m.putAvail} PUT remaining. Top up to stay in the game.`,
          sub: null, features: [],
          cta: [{ label: '⚡ Add Tokens', href: BILLING_URL, style: 'primary' }],
        };
      }
      return null;
    }

    const nextM    = _tierMeta[nextTier] || {};
    const gainStr  = _putGain(tier, nextTier) + ' PUT/mo';
    const feats    = TIER_FEATURES[nextTier] || [];

    if (roi > 50 && views30 > 300) {
      return {
        icon: '🚀', cls: 'upsell',
        title: `${roi}× Return — Time to Scale`,
        msg: `Every PUT token generates ~${roi} views. ${nextM.label || nextTier} unlocks ${gainStr} automatically.`,
        sub: 'Your content converts. The ceiling is your token limit.',
        features: feats, roi, views: views30,
        cta: [
          { label: `Upgrade to ${nextM.label || nextTier}`, href: BILLING_URL, style: 'purple' },
          { label: 'See Plans', href: PRICING_URL, style: 'secondary' },
        ],
      };
    }

    if (daysLeft < 10 && daysLeft < 999 && putUsed > 0) {
      return {
        icon: '🔥', cls: 'upsell',
        title: `Tokens Gone in ~${daysLeft} Day${daysLeft !== 1 ? 's' : ''}`,
        msg: `At your pace you'll hit zero mid-month. ${nextM.label || nextTier} auto-refills ${gainStr} every cycle.`,
        sub: null, features: feats,
        cta: [
          { label: `Upgrade to ${nextM.label || nextTier}`, href: BILLING_URL, style: 'primary' },
          { label: 'Top Up Instead', href: BILLING_URL + '?tab=topup', style: 'secondary' },
        ],
      };
    }

    if (uploadsLeft < 5 && totalUps > 2) {
      return {
        icon: '📈', cls: 'upsell',
        title: `Only ${uploadsLeft} Upload${uploadsLeft !== 1 ? 's' : ''} Left`,
        msg: `${doneUps} published · ${successPct}% success. ${nextM.label || nextTier} adds ${gainStr} to keep momentum.`,
        sub: "Don't let an empty wallet break your streak.",
        features: feats,
        cta: [
          { label: `Upgrade to ${nextM.label || nextTier}`, href: BILLING_URL, style: 'purple' },
          { label: 'Top Up Now', href: BILLING_URL + '?tab=topup', style: 'secondary' },
        ],
      };
    }

    if (tier === 'free' && totalUps >= 1) {
      return {
        icon: '⚡', cls: 'upsell',
        title: "You've Got the Flow — Go Pro",
        msg: `${totalUps} upload${totalUps !== 1 ? 's' : ''} done. Creator Lite gives 3× tokens${!canAi ? ', AI captions' : ''} & priority queue.`,
        sub: 'First 7 days free. Cancel any time.',
        features: feats,
        cta: [
          { label: 'Start Free Trial', href: BILLING_URL, style: 'purple' },
          { label: 'Compare Plans',    href: PRICING_URL, style: 'secondary' },
        ],
      };
    }

    return null;
  }

  /* ─────────────────────────────────────────────
   * DATA FETCH
   * ───────────────────────────────────────────── */
  function _tok() {
    try {
      return localStorage.getItem('uploadm8_access_token')
          || localStorage.getItem('uploadm8_token')
          || localStorage.getItem('access_token')
          || localStorage.getItem('accessToken')
          || localStorage.getItem('token') || '';
    } catch { return ''; }
  }

  async function _get(path) {
    const t = _tok();
    try {
      const r = await fetch(API_BASE + path, {
        headers: t ? { Authorization: `Bearer ${t}` } : {},
        credentials: 'include',
      });
      return r.ok ? r.json() : null;
    } catch { return null; }
  }

  function _applyTierConfig(data) {
    if (!data || !data.tiers) return;
    _s.tierConfig = {};
    data.tiers.forEach(t => {
      _s.tierConfig[t.slug] = t;
      if (!_tierMeta[t.slug]) _tierMeta[t.slug] = { label: t.name || t.slug, badge: (t.slug || '').toUpperCase(), color: '#6b7280', next: null, put: 80, aic: 50 };
      if (t.put_monthly) _tierMeta[t.slug].put = t.put_monthly;
      if (t.aic_monthly) _tierMeta[t.slug].aic = t.aic_monthly;
      if (t.name)        _tierMeta[t.slug].label = t.name;
    });
  }

  async function _fetchAll(force = false) {
    if (!force && Date.now() - _s.lastFetch < CACHE_TTL_MS) return;
    if (_s.inFlight) return;
    _s.inFlight = true;

    try {
      // Use window.currentUser if available (from app.js session cache or checkAuth) — avoids duplicate /api/me
      _resolve();
      if (!_s.user) {
        const me = await _get('/api/me');
        if (me) {
          _s.user   = me;
          _s.wallet = me.wallet;
          _s.ent    = me.entitlements;
        }
      } else {
        _s.wallet = _s.wallet || _s.user.wallet;
        _s.ent    = _s.ent || _s.user.entitlements;
      }

      let tierData = null;
      if (!_s.tierConfig) {
        tierData = await _get('/api/entitlements/tiers');
        if (!tierData) tierData = await _get('/api/entitlements');
      }
      const [d, a, w] = await Promise.allSettled([
        _get('/api/dashboard/stats'),
        _get('/api/analytics?range=30d'),
        _get('/api/wallet'),
      ]);

      if (d.status === 'fulfilled' && d.value) _s.dash      = d.value;
      if (a.status === 'fulfilled' && a.value) _s.analytics = a.value;
      if (w.status === 'fulfilled' && w.value) {
        _s.ledger = (w.value.ledger || []).slice(0, 8);
        if (w.value.wallet) _s.wallet = w.value.wallet;
      }
      if (tierData) _applyTierConfig(tierData);

      _s.lastFetch = Date.now();
    } finally {
      _s.inFlight = false;
    }
  }

  function _resolve() {
    if (!_s.user && window.currentUser) {
      _s.user   = window.currentUser;
      _s.wallet = window.currentUser.wallet;
      _s.ent    = window.currentUser.entitlements;
    }
  }

  /* ─────────────────────────────────────────────
   * PILL RENDERER
   * ───────────────────────────────────────────── */
  function _renderPills(el) {
    if (!el) return;
    _resolve();
    const m = _metrics();
    el.querySelector('.wt-pills-wrap')?.remove();

    const wrap = document.createElement('div');
    wrap.className = 'wt-pills-wrap';

    const badge = document.createElement('span');
    badge.className = 'wt-tier-badge';
    badge.style.cssText = `background:${m.tierM.color}1a;border:1px solid ${m.tierM.color}44;color:${m.tierM.color}`;
    badge.title = m.tierLabel + ' tier';
    badge.textContent = m.tierM.badge;

    const pills = document.createElement('div');
    pills.className = 'wt-pills';

    const internal = _isInternal(_s.user);
    const _fmt = (v) => internal && v >= 99999 ? '∞' : v.toLocaleString();

    const mk = (label, avail, pct, lvl, ttl) => {
      const a = document.createElement('a');
      a.href      = BILLING_URL;
      a.className = `wt-pill wt-pill-${lvl}`;
      a.title     = ttl;
      a.innerHTML = `<span class="wt-pill-icon">${_lvlIcon(lvl)}</span>
        <span>${label} ${_fmt(avail)}</span>
        <span class="wt-pill-bar" style="width:${Math.max(2, pct)}%"></span>`;
      return a;
    };

    pills.appendChild(mk('PUT', m.putAvail, m.putPct, m.putLvl,
      internal ? 'PUT: ∞ (unlimited)' : `PUT: ${m.putAvail} / ${m.putMo} available · ~${Math.floor(m.putAvail / PUT_PER_UPLOAD)} uploads left`));
    pills.appendChild(mk('AIC', m.aicAvail, m.aicPct, m.aicLvl,
      internal ? 'AIC: ∞ (unlimited)' : `AIC: ${m.aicAvail} / ${m.aicMo} · Powers AI captions, hashtags & thumbnails`));

    wrap.appendChild(badge);
    wrap.appendChild(pills);
    el.insertBefore(wrap, el.firstChild);
  }

  /* ─────────────────────────────────────────────
   * BANNER RENDERER
   * ───────────────────────────────────────────── */
  function _renderBanner(el, userOverride) {
    if (!el) return;
    if (userOverride) { _s.user = userOverride; _s.wallet = userOverride.wallet; _s.ent = userOverride.entitlements; }
    _resolve();
    const m   = _metrics();
    const key = `wt_dismissed_${m.overall}_${new Date().toDateString()}`;
    el.querySelectorAll('.wt-banner').forEach(b => b.remove());
    if (localStorage.getItem(key)) return;

    const cfg = _bannerCfg(m);
    if (!cfg) return;

    const featsHtml = (cfg.features && cfg.features.length)
      ? `<div class="wt-feature-list">${cfg.features.map(f => `<span class="wt-feat-chip">✓ ${f}</span>`).join('')}</div>` : '';

    const roiHtml = (cfg.roi && cfg.views)
      ? `<div class="wt-roi-badge">📊 ${cfg.roi}× ROI · ${cfg.views.toLocaleString()} views / 30d</div>` : '';

    const aiBadge = m.canAi
      ? `<div class="wt-ai-badge">✦ AI Captions & Hashtags Enabled</div>` : '';

    const banner = document.createElement('div');
    banner.className = `wt-banner wt-banner-${cfg.cls}`;
    banner.innerHTML = `
      <div class="wt-banner-icon">${cfg.icon}</div>
      <div class="wt-banner-body">
        <div class="wt-banner-title">${cfg.title}</div>
        <div class="wt-banner-msg">${cfg.msg}</div>
        ${cfg.sub ? `<div class="wt-banner-sub">${cfg.sub}</div>` : ''}
        ${featsHtml}${roiHtml}${aiBadge}
        <div class="wt-banner-actions">${cfg.cta.map(c => `<a href="${c.href}" class="wt-bbtn wt-bbtn-${c.style}">${c.label}</a>`).join('')}</div>
      </div>
      <button class="wt-dismiss" aria-label="Dismiss">×</button>`;

    banner.querySelector('.wt-dismiss').addEventListener('click', () => {
      try { localStorage.setItem(key, '1'); } catch {}
      banner.style.cssText += ';opacity:0;transform:translateY(-6px);transition:all .25s';
      setTimeout(() => banner.remove(), 260);
    });

    el.insertBefore(banner, el.firstChild);
  }

  function _bannerCfg(m) {
    const { overall, putAvail, aicAvail, uploadsLeft, burnDay, daysLeft } = m;

    if (overall === 'empty') return {
      cls: 'empty', icon: '⛔',
      title: 'Zero Tokens — Uploads Paused',
      msg: "You've hit your limit. Add tokens to resume publishing.",
      sub: null, features: [],
      cta: [
        { label: '💳 Top Up Tokens', href: BILLING_URL, style: 'primary' },
        { label: 'Upgrade Plan',     href: PRICING_URL, style: 'secondary' },
      ],
    };

    if (overall === 'critical') return {
      cls: 'critical', icon: '🚨',
      title: `Only ${uploadsLeft} Upload${uploadsLeft !== 1 ? 's' : ''} Left!`,
      msg: `${putAvail} PUT · ${aicAvail} AIC remaining. Next upload may fail.`,
      sub: burnDay > 0 ? `At your pace: ~${daysLeft} day${daysLeft !== 1 ? 's' : ''} until empty.` : 'Top up to avoid interruption.',
      features: [],
      cta: [
        { label: '🔥 Top Up Now',              href: BILLING_URL, style: 'primary' },
        { label: 'Upgrade for Auto-Refill',    href: BILLING_URL, style: 'secondary' },
      ],
    };

    if (overall === 'low') return {
      cls: 'low', icon: '⚠️',
      title: 'Running Low on Tokens',
      msg: `${putAvail} PUT · ${aicAvail} AIC — about ${uploadsLeft} upload${uploadsLeft !== 1 ? 's' : ''} remaining.`,
      sub: 'Top up now to keep publishing without interruption.',
      features: [],
      cta: [
        { label: '➕ Add Tokens',  href: BILLING_URL, style: 'primary' },
        { label: 'Upgrade Plan',   href: PRICING_URL, style: 'secondary' },
      ],
    };

    return _upsellCfg(m);
  }

  /* ─────────────────────────────────────────────
   * HUD PANEL
   * ───────────────────────────────────────────── */
  function _buildHud() {
    if (document.getElementById('wt-hud')) return;
    _resolve();
    const m = _metrics();

    const fab = document.createElement('button');
    fab.id = 'wt-fab'; fab.className = 'wt-fab';
    fab.title = 'Wallet & Token Stats';
    fab.setAttribute('aria-label', 'Open wallet panel');
    fab.innerHTML = '💰';
    if (m.overall === 'critical' || m.overall === 'empty') {
      const dot = document.createElement('div'); dot.className = 'wt-fab-dot'; fab.appendChild(dot);
    }

    const hud = document.createElement('div');
    hud.id = 'wt-hud'; hud.className = 'wt-hud';
    hud.setAttribute('role', 'dialog');
    hud.setAttribute('aria-modal', 'true');
    hud.setAttribute('aria-label', 'Wallet and Token Stats');
    hud.innerHTML = `
      <div class="wt-hud-head">
        <span class="wt-hud-title">⬡ Token Wallet</span>
        <button class="wt-hud-x" id="wt-close" aria-label="Close">✕</button>
      </div>
      <div class="wt-hud-body" id="wt-hud-body">${_hudInner(m)}</div>`;

    document.body.appendChild(fab);
    document.body.appendChild(hud);

    fab.addEventListener('click', () => hud.classList.toggle('open'));
    document.getElementById('wt-close').addEventListener('click', () => hud.classList.remove('open'));
    document.addEventListener('keydown', e => { if (e.key === 'Escape') hud.classList.remove('open'); });
  }

  function _hudInner(m) {
    return `${_hudMeters(m)}${_hudEntBadges(m)}${_hudKpis(m)}${_hudBurn(m)}${_hudVelocity(m)}${_hudLedger()}${_hudCtas(m)}`;
  }

  function _hudMeters(m) {
    const intl = _isInternal(_s.user);
    const fmtA = (v) => intl && v >= 99999 ? '∞' : v.toLocaleString();
    const fmtM = (v) => intl && v >= 99999 ? '∞' : v.toLocaleString();
    return `
      <div class="wt-meter">
        <div class="wt-meter-row">
          <span class="wt-meter-lbl">PUT Tokens</span>
          <span class="wt-meter-val">${fmtA(m.putAvail)}<span class="wt-meter-sub">/ ${fmtM(m.putMo)}</span></span>
        </div>
        <div class="wt-track"><div class="wt-fill ${_fillCls(m.putLvl)}" style="width:${Math.max(1,m.putPct)}%"></div></div>
      </div>
      <div class="wt-meter">
        <div class="wt-meter-row">
          <span class="wt-meter-lbl">AIC Credits</span>
          <span class="wt-meter-val">${fmtA(m.aicAvail)}<span class="wt-meter-sub">/ ${fmtM(m.aicMo)}</span></span>
        </div>
        <div class="wt-track"><div class="wt-fill ${_fillCls(m.aicLvl)}" style="width:${Math.max(1,m.aicPct)}%"></div></div>
      </div>`;
  }

  function _hudEntBadges(m) {
    const badges = [];
    if (m.canAi)       badges.push('✦ AI');
    if (m.canSchedule) badges.push('⏱ Scheduler');
    if (!badges.length) return '';
    return `<div style="display:flex;gap:5px;flex-wrap:wrap;margin-bottom:10px">${
      badges.map(b => `<span class="wt-ai-badge">${b}</span>`).join('')
    }</div>`;
  }

  function _hudKpis(m) {
    const rCls = m.roi > 50 ? 'pos' : m.roi > 10 ? '' : 'neg';
    return `
      <div class="wt-kpi-grid">
        <div class="wt-kpi"><div class="wt-kpi-n">${m.uploadsLeft}</div><div class="wt-kpi-l">Uploads left</div></div>
        <div class="wt-kpi"><div class="wt-kpi-n">${m.successPct}%</div><div class="wt-kpi-l">Success rate</div></div>
        <div class="wt-kpi"><div class="wt-kpi-n">${m.views30.toLocaleString()}</div><div class="wt-kpi-l">Views / 30d</div></div>
        <div class="wt-kpi">
          <div class="wt-kpi-n">${m.roi > 0 ? m.roi + 'x' : '—'}${m.roi > 0 ? `<span class="wt-kpi-d ${rCls}">${m.roi > 50 ? '▲' : '→'}</span>` : ''}</div>
          <div class="wt-kpi-l">Views / PUT</div>
        </div>
      </div>`;
  }

  function _hudBurn(m) {
    if (m.burnDay <= 0 || m.daysLeft >= 999) return '';
    const urgCss = m.daysLeft < 5 ? 'color:#f87171' : m.daysLeft < 14 ? 'color:#fbbf24' : 'color:#34d399';
    return `
      <div class="wt-burn">
        <span class="wt-burn-ico">🔥</span>
        <div class="wt-burn-txt">Burning <strong>~${m.burnDay.toFixed(1)} PUT/day</strong> —
          <span style="${urgCss};font-weight:700">~${m.daysLeft}d left</span> at this pace.
        </div>
      </div>`;
  }

  function _hudVelocity(m) {
    if (m.putMo <= 0) return '';
    return `
      <div class="wt-vel-wrap">
        <div class="wt-vel-lbl"><span>Token Velocity (30d)</span><span>${Math.round(m.velUsed)}% consumed</span></div>
        <div class="wt-vel-track"><div class="wt-vel-fill" style="width:${m.velUsed}%"></div></div>
      </div>`;
  }

  function _hudLedger() {
    if (!_s.ledger.length) return '';
    const rows = _s.ledger.slice(0, 6).map(l => {
      const d   = parseInt(l.delta || 0, 10);
      const cls = d > 0 ? 'pos' : 'neg';
      const dt  = l.created_at
        ? new Date(l.created_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
        : '';
      return `<div class="wt-ledger-row">
        <span class="wt-ledger-type">${(l.token_type || '').toUpperCase()}</span>
        <span class="wt-ledger-rsn">${(l.reason || '').replace(/_/g, ' ')}</span>
        <span class="wt-ledger-d ${cls}">${d > 0 ? '+' : ''}${d}</span>
        <span class="wt-ledger-dt">${dt}</span>
      </div>`;
    }).join('');
    return `<div class="wt-ledger"><div class="wt-ledger-ttl">Recent Ledger</div>${rows}</div>`;
  }

  function _hudCtas(m) {
    const nextM = m.nextTier ? (_tierMeta[m.nextTier] || {}) : null;
    return `
      <div class="wt-cta-row">
        <a href="${BILLING_URL}?tab=topup" class="wt-hbtn wt-hbtn-primary">⚡ Top Up</a>
        ${nextM
          ? `<a href="${BILLING_URL}" class="wt-hbtn wt-hbtn-secondary">↑ ${nextM.label || m.nextTier}</a>`
          : `<a href="${BILLING_URL}" class="wt-hbtn wt-hbtn-secondary">⚙ Billing</a>`}
      </div>`;
  }

  function _refreshHud() {
    const body = document.getElementById('wt-hud-body');
    if (!body) return;
    _resolve();
    body.innerHTML = _hudInner(_metrics());

    const fab = document.getElementById('wt-fab');
    if (fab) {
      fab.querySelector('.wt-fab-dot')?.remove();
      const m = _metrics();
      if (m.overall === 'critical' || m.overall === 'empty') {
        const dot = document.createElement('div'); dot.className = 'wt-fab-dot'; fab.appendChild(dot);
      }
    }
  }

  /* ─────────────────────────────────────────────
   * PUBLIC API
   * ───────────────────────────────────────────── */
  async function init(opts = {}) {
    _css();
    if (opts.user) { _s.user = opts.user; _s.wallet = opts.user.wallet; _s.ent = opts.user.entitlements; }
    _resolve();
    await _fetchAll();

    document.querySelectorAll('[data-wt-pills]').forEach(_renderPills);
    document.querySelectorAll('[data-wt-banner]').forEach(el => _renderBanner(el));
    if (!opts.noHud) _buildHud();

    setInterval(async () => {
      await _fetchAll(true);
      document.querySelectorAll('[data-wt-pills]').forEach(_renderPills);
      document.querySelectorAll('[data-wt-banner]').forEach(el => _renderBanner(el));
      _refreshHud();
    }, HUD_REFRESH_MS);
  }

  function _auto() { init({ user: window.currentUser || null }); }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', _auto);
  } else {
    setTimeout(_auto, 0);
  }

  window.addEventListener('uploadm8:user', e => {
    if (!e.detail) return;
    _s.user = e.detail;
    _s.wallet = e.detail.wallet;
    _s.ent    = e.detail.entitlements;
    document.querySelectorAll('[data-wt-pills]').forEach(_renderPills);
    document.querySelectorAll('[data-wt-banner]').forEach(el => _renderBanner(el));
    _refreshHud();
  });

  /* ─────────────────────────────────────────────
   * EXPOSE (backward-compat + new surface)
   * ───────────────────────────────────────────── */
  window.WalletTokens = {
    init,
    refresh: async () => {
      await _fetchAll(true);
      document.querySelectorAll('[data-wt-pills]').forEach(_renderPills);
      document.querySelectorAll('[data-wt-banner]').forEach(el => _renderBanner(el));
      _refreshHud();
    },
    getMetrics: () => { _resolve(); return _metrics(); },
    getState:   () => _s,

    getWalletFromUser(user) {
      const w   = (user && user.wallet) || {};
      const ent = (user && user.entitlements) || {};
      const plan= (user && user.plan)  || {};
      const src = (ent.put_monthly != null || ent.aic_monthly != null) ? ent : plan;
      const internal = _isInternal(user);

      const putBal  = Math.floor(parseFloat(w.put_balance  || 0));
      const aicBal  = Math.floor(parseFloat(w.aic_balance  || 0));
      const putRes  = Math.floor(parseFloat(w.put_reserved || 0));
      const aicRes  = Math.floor(parseFloat(w.aic_reserved || 0));
      let putAvail = Math.max(0, putBal - putRes);
      let aicAvail = Math.max(0, aicBal - aicRes);
      const putMo  = parseInt(src.put_monthly || plan.put_monthly || 0, 10) || 80;
      const aicMo  = parseInt(src.aic_monthly || plan.aic_monthly || 0, 10) || 50;

      if (internal) {
        putAvail = Math.max(putAvail, putMo);
        aicAvail = Math.max(aicAvail, aicMo);
      }

      const putLvl = _lvl(putAvail, putMo);
      const aicLvl = _lvl(aicAvail, aicMo);
      const overallLevel = (putLvl === 'empty'    || aicLvl === 'empty')    ? 'empty'
                         : (putLvl === 'critical' || aicLvl === 'critical') ? 'critical'
                         : (putLvl === 'low'      || aicLvl === 'low')      ? 'low' : 'healthy';

      const tier = (user && (user.tier || user.subscription_tier || user.plan?.tier || user.entitlements?.tier)) || 'free';

      return {
        putAvail, aicAvail, putRes, aicRes, putMo, aicMo,
        putLevel: putLvl, aicLevel: aicLvl, overallLevel,
        uploadsLeft: Math.floor(Math.min(putAvail / PUT_PER_UPLOAD, aicAvail / AIC_PER_UPLOAD)),
        tier,
      };
    },

    getUsageLevel: _lvl,

    renderTokenPills(container) {
      _css(); _resolve(); _renderPills(container);
    },

    renderUsageBanner(container, user) {
      _css(); _renderBanner(container, user);
    },

    UPLOAD_ESTIMATE: { put: PUT_PER_UPLOAD, aic: AIC_PER_UPLOAD },
  };

})();
