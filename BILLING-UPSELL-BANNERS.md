# Billing Upsell System — PUT & AIC Usage Indicators & Banners

Design spec for cool, interactive banners that incentivize and alert customers about token usage.

---

## 1. Usage Thresholds & Alert Levels

| Level | PUT % of monthly | AIC % of monthly | UX treatment |
|-------|------------------|------------------|---------------|
| **Healthy** | > 25% | > 25% | Subtle indicator only, no banner |
| **Notice** | 10–25% | 10–25% | Soft banner, "You're doing great" |
| **Low** | 5–10% | 5–10% | Amber banner, "Running low" |
| **Critical** | 1–5% | 1–5% | Red banner, "Almost out" |
| **Empty** | 0 | 0 | Blocking banner, "Top up to continue" |

**Alternative (absolute thresholds):** Use fixed numbers for consistency across tiers:
- **Low**: ≤ 3 uploads worth (~50 PUT, ~15 AIC)
- **Critical**: ≤ 1 upload worth (~16 PUT, ~5 AIC)
- **Empty**: 0 available (balance − reserved ≤ 0)

---

## 2. Banner Types & Placement

### A. Global Header Token Bar (always visible when logged in)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  UploadM8    Dashboard  Upload  Queue  ...    [PUT 142] [AIC 38]  [⚙️]  │
└─────────────────────────────────────────────────────────────────────────┘
```

- **Healthy**: Gray pill badges, no animation
- **Low**: Amber pill, subtle pulse
- **Critical**: Red pill, gentle glow
- **Empty**: Red pill + "Top up" link inline

**Interaction**: Click pill → expand mini dropdown with:
- Available vs reserved
- Monthly allowance
- "Top up" / "Upgrade plan" CTA

---

### B. Dashboard Hero Banner (contextual, above main content)

**Healthy / Notice** (optional positive reinforcement):
```
┌──────────────────────────────────────────────────────────────────────────┐
│  🎯  You've used 45% of your monthly tokens — on track for ~12 more uploads │
│      [View usage] [Upgrade for more]                                        │
└──────────────────────────────────────────────────────────────────────────┘
```

**Low**:
```
┌──────────────────────────────────────────────────────────────────────────┐
│  ⚠️  Running low — 42 PUT & 12 AIC left (~2–3 uploads)                     │
│      Top up now to keep publishing without interruption.                  │
│      [Top Up Tokens]  [Upgrade Plan]                              [×]     │
└──────────────────────────────────────────────────────────────────────────┘
```
- Amber gradient background
- Dismissible (localStorage, 24h)

**Critical**:
```
┌──────────────────────────────────────────────────────────────────────────┐
│  🚨  Almost out! Only 18 PUT & 4 AIC — about 1 upload left                │
│      Your next upload may fail. Top up now.                               │
│      [Top Up Now — 50 PUT $2.99]  [See all packs]                 [×]     │
└──────────────────────────────────────────────────────────────────────────┘
```
- Red/orange gradient, subtle pulse animation
- Primary CTA = best-value top-up pack

**Empty** (blocking):
```
┌──────────────────────────────────────────────────────────────────────────┐
│  ⛔  No tokens left — uploads are paused                                   │
│      Add tokens to resume publishing.                                     │
│      [Add PUT] [Add AIC] [Upgrade Plan]                                   │
└──────────────────────────────────────────────────────────────────────────┘
```
- Full-width, cannot dismiss
- Shown on dashboard, upload page, queue

---

### C. Upload Page Pre-Submit Warning

Before user clicks "Upload", if balance is low:
```
┌──────────────────────────────────────────────────────────────────────────┐
│  ⚠️  This upload will use ~16 PUT + ~3 AIC. You have 24 PUT & 8 AIC.     │
│      After this, you'll have ~1 upload left. [Top up] to add more.       │
└──────────────────────────────────────────────────────────────────────────┘
```

If empty:
```
┌──────────────────────────────────────────────────────────────────────────┐
│  ⛔  Insufficient tokens. Add PUT and AIC to upload. [Top Up Now]          │
└──────────────────────────────────────────────────────────────────────────┘
```
- Disable "Start Upload" button until topped up

---

### D. Post-Upload Toast (after successful upload)

```
┌──────────────────────────────────────────────────────────────────────────┐
│  ✅  Upload queued!  Used 16 PUT + 3 AIC.  ~142 PUT & 35 AIC remaining.   │
└──────────────────────────────────────────────────────────────────────────┘
```
- If low: append " — [Top up] to avoid running out"
- Auto-dismiss 5s, or click to dismiss

---

### E. Settings / Billing Page — Usage Visualization

**Progress rings or bars** (interactive):
```
PUT  [████████████░░░░░░░░] 142 / 400 this month  (35%)
AIC  [██████████░░░░░░░░░░]  38 / 120 this month  (32%)
```
- Hover: tooltip "~8 uploads left at current rate"
- Click bar: scroll to top-up grid

**"Estimated uploads left"** badge:
```
~8 uploads left  (at ~16 PUT + 3 AIC each)
```
- Updates when user changes platform count in upload form (if that's pre-filled)

---

## 3. Copy & Messaging

### Tone
- **Healthy**: Celebratory, minimal — "You're all set"
- **Low**: Helpful, not alarming — "Running low — top up when convenient"
- **Critical**: Urgent but not panic — "Almost out — add tokens to avoid interruption"
- **Empty**: Clear, actionable — "No tokens. Add some to continue."

### Value Props to Weave In
- "First top-up: +25% bonus"
- "~X uploads left"
- "Upgrade for 5× more tokens"
- "Best value: 250 PUT for $9.99"

---

## 4. Interactive Elements

### Token Pill (header)
- **Hover**: Tooltip with available, reserved, monthly limit
- **Click**: Dropdown with quick top-up buttons (50 PUT, 100 AIC)
- **Animation**: Subtle pulse when critical, glow when low

### Banner
- **Dismiss**: × button, store in localStorage `banner_dismissed_{level}_{date}`
- **CTA**: Primary = top-up or upgrade; secondary = "Learn more"
- **Progress**: Animated fill on progress bar (CSS transition)

### Usage Widget (settings)
- **Animated ring**: SVG or CSS conic-gradient, transitions on load
- **Click segment**: Scroll to relevant top-up section (PUT vs AIC)

---

## 5. API Data (Already Available)

| Endpoint | Data |
|----------|------|
| `GET /api/me` | `wallet.put_balance`, `wallet.aic_balance`, `wallet.put_reserved`, `wallet.aic_reserved`, `plan.put_monthly`, `plan.aic_monthly` |
| `GET /api/dashboard/stats` | `wallet.put_available`, `wallet.aic_available`, `quota.put_used`, `quota.put_limit` |
| `GET /api/pricing` | `topups` (lookup_key, amount, price_usd, wallet) |

**Computed client-side:**
- `put_available = put_balance - put_reserved`
- `aic_available = aic_balance - aic_reserved`
- `put_pct_used = (put_monthly - put_available) / put_monthly` (approx; or use quota if exposed)
- `uploads_left ≈ min(put_available/16, aic_available/3)` (rough)

---

## 6. Implementation Checklist

### Phase 1 — Core Indicators
- [ ] Header token pills (PUT, AIC) on all authenticated pages
- [ ] Color state: gray → amber → red based on thresholds
- [ ] Click → mini dropdown with balance + "Top up" link

### Phase 2 — Banners
- [ ] Dashboard hero banner (Low, Critical, Empty)
- [ ] Dismiss logic with localStorage
- [ ] CTAs to `/settings.html#billing` or Stripe Checkout

### Phase 3 — Upload Flow
- [ ] Pre-submit warning on upload page when low
- [ ] Disable upload button when empty
- [ ] Post-upload toast with remaining balance

### Phase 4 — Settings Enhancement
- [ ] Progress bars for PUT/AIC usage
- [ ] "Estimated uploads left" calculator
- [ ] Quick top-up buttons in wallet section

### Phase 5 — Polish
- [ ] Animations (pulse, glow, progress fill)
- [ ] A/B test copy for conversion
- [ ] Telemetry: banner impressions, CTA clicks, top-up conversion

---

## 7. CSS Snippets (Starter)

```css
/* Token pill states */
.token-pill { border-radius: 9999px; padding: 0.25rem 0.6rem; font-size: 0.8rem; font-weight: 600; }
.token-pill.healthy { background: #27272a; color: #a1a1aa; }
.token-pill.low { background: rgba(245, 158, 11, 0.2); color: #fbbf24; animation: pulse-amber 2s infinite; }
.token-pill.critical { background: rgba(239, 68, 68, 0.2); color: #f87171; animation: pulse-red 2s infinite; }
.token-pill.empty { background: #7f1d1d; color: #fca5a5; }

@keyframes pulse-amber { 0%, 100% { opacity: 1; } 50% { opacity: 0.85; } }
@keyframes pulse-red { 0%, 100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.4); } 50% { box-shadow: 0 0 0 6px rgba(239,68,68,0); } }

/* Banner gradient */
.banner-low { background: linear-gradient(135deg, #78350f 0%, #92400e 100%); }
.banner-critical { background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%); }
```

---

## 8. Example Banner HTML (Low State)

```html
<div class="usage-banner banner-low" data-level="low">
  <div class="banner-content">
    <span class="banner-icon">⚠️</span>
    <div>
      <strong>Running low</strong> — <span id="put-avail">42</span> PUT & <span id="aic-avail">12</span> AIC left (~2–3 uploads)
      <p class="banner-sub">Top up now to keep publishing without interruption.</p>
    </div>
    <div class="banner-actions">
      <a href="/settings.html#billing" class="btn btn-primary">Top Up Tokens</a>
      <a href="/index.html#pricing" class="btn btn-outline">Upgrade Plan</a>
    </div>
    <button class="banner-dismiss" aria-label="Dismiss">×</button>
  </div>
</div>
```

---

## 9. Threshold Constants (Frontend)

```javascript
const UPLOAD_ESTIMATE = { put: 16, aic: 3 };  // per typical upload

function getUsageLevel(available, monthly) {
  const pct = monthly > 0 ? (1 - available / monthly) * 100 : 0;
  if (available <= 0) return 'empty';
  if (pct >= 95 || available <= UPLOAD_ESTIMATE.put) return 'critical';
  if (pct >= 80 || available <= UPLOAD_ESTIMATE.put * 3) return 'low';
  if (pct >= 75) return 'notice';
  return 'healthy';
}
```

---

*Use this doc as the spec for implementing the billing upsell system. Adjust thresholds and copy based on real usage data.*

---

## 10. Implementation Status (uploadm8-frontend)

**Implemented:**
- **wallet-tokens.js** — Shared module: `getUsageLevel`, `getWalletFromUser`, `renderTokenPills`, `renderUsageBanner`
- **Token pills** — In header (top-bar-actions) on all authenticated pages; color states: healthy (gray), low (amber pulse), critical (red pulse), empty (red)
- **Dashboard usage banner** — Low/Critical/Empty banners above page content; dismissible (localStorage, per-day)
- **Upload page** — Pre-submit token warning when low; blocking empty state; upload button disabled when empty
- **CSS** — Token pill and usage banner styles in styles.css

**Pages updated:** dashboard, queue, upload, settings, scheduled, platforms, groups, analytics, guide, color-preferences, admin, admin-kpi, admin-wallet, admin-calculator, account-management
