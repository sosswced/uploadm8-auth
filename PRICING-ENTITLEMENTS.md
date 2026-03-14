# UploadM8 Pricing & Entitlements — Stripe Setup Guide

**Version:** v2 (Value-first, taste → exponential scaling)  
**Last updated:** March 2025

---

## 1. Overview

- **Free:** Taste of paid — enough to experience value, limits create upgrade need
- **Paid tiers:** Exponential value scaling — each tier feels meaningfully better
- **Top-ups:** Attractive pricing that builds experience, value, and profit

---

## 2. Subscription Tiers (Stripe Products)

### Stripe lookup keys (must match exactly)

| Tier | Stripe lookup_key | Price (USD) | Billing |
|------|-------------------|-------------|---------|
| Creator Lite | `uploadm8_creatorlite_monthly` | $9.99 | monthly |
| Creator Pro | `uploadm8_creatorpro_monthly` | $19.99 | monthly |
| Studio | `uploadm8_studio_monthly` | $49.99 | monthly |
| Agency | `uploadm8_agency_monthly` | $99.99 | monthly |

**Legacy (keep until migration):**
- `uploadm8_launch_monthly` → maps to creator_lite
- `uploadm8_creator_pro_monthly` → maps to creator_pro

---

## 3. Tier Entitlements (Full Structure)

### Public tiers

| Field | Free | Creator Lite | Creator Pro | Studio | Agency |
|-------|------|--------------|-------------|--------|--------|
| **Price** | $0 | $9.99 | $19.99 | $49.99 | $99.99 |
| **PUT/month** | 80 | 400 | 1,200 | 3,500 | 8,000 |
| **AIC/month** | 50 | 120 | 350 | 1,000 | 2,500 |
| **Max accounts** | 4 | 10 | 25 | 75 | 300 |
| **Per-platform** | 2 | 3 | 6 | 20 | 100 |
| **Watermark** | Yes | No | No | No | No |
| **Ads** | Yes | No | No | No | No |
| **AI** | Yes | Yes | Yes | Yes | Yes |
| **Scheduling** | Yes | Yes | Yes | Yes | Yes |
| **Webhooks** | No | Yes | Yes | Yes | Yes |
| **White label** | No | No | No | No | Yes |
| **HUD burn** | No | No | Yes | Yes | Yes |
| **Excel export** | No | No | No | Yes | Yes |
| **Flex (transfer)** | No | No | No | No | Yes |
| **Queue depth** | 10 | 100 | 500 | 2,500 | 99,999 |
| **Lookahead (h)** | 4 | 12 | 24 | 72 | 168 |
| **Max thumbnails** | 3 | 5 | 8 | 12 | 20 |
| **AI depth** | basic | enhanced | advanced | max | max |
| **Caption frames** | 3 | 5 | 8 | 15 | 15 |
| **Parallel uploads** | 1 | 2 | 3 | 4 | 6 |
| **Custom thumbnails** | Yes | Yes | Yes | Yes | Yes |
| **AI thumbnail styling** | No | No | Yes | Yes | Yes |
| **Team seats** | 1 | 1 | 3 | 10 | 25 |
| **Analytics** | basic | standard | full | full_export | full_export |
| **Trial days** | 0 | 7 | 7 | 7 | 7 |

---

## 4. Perks & Descriptions (for marketing / Stripe product copy)

### Free
- **Tagline:** Taste the full workflow
- **Perks:** AI captions, scheduling (4h lookahead), 3 thumbnails, 80 PUT + 50 AIC/month
- **Limits:** Watermark, ads, 4 accounts, 10-upload queue, basic AI

### Creator Lite — $9.99/mo
- **Tagline:** Ship consistently, stop manual posting
- **Perks:** No watermark, scheduling (12h), webhooks, 400 PUT + 120 AIC, 10 accounts
- **Best for:** Solo creators posting 2–3×/week

### Creator Pro — $19.99/mo [MOST POPULAR]
- **Tagline:** Weekend batching + higher AI precision
- **Perks:** HUD burn, AI thumbnail styling, 1,200 PUT + 350 AIC, 25 accounts, 3 team seats
- **Best for:** Creators who batch weekly, want better AI

### Studio — $49.99/mo
- **Tagline:** Turbo throughput + export-grade reporting
- **Perks:** Excel export, 3,500 PUT + 1,000 AIC, 75 accounts, 10 team seats, 72h lookahead
- **Best for:** Small teams, agencies starting out

### Agency — $99.99/mo
- **Tagline:** Built for agencies managing multiple clients
- **Perks:** White label, Flex (transfer PUT between platforms), 8,000 PUT + 2,500 AIC, 300 accounts, 25 team seats, 1-week lookahead
- **Best for:** Agencies, multi-client management

---

## 5. Top-Up Products (Stripe one-time prices)

### Suggested prices (experience + value + profit)

| lookup_key | Wallet | Amount | Price (USD) | $/token | Notes |
|------------|--------|--------|-------------|---------|-------|
| `uploadm8_put_50` | put | 50 | $2.99 | $0.06 | Starter — low barrier |
| `uploadm8_put_100` | put | 100 | $4.99 | $0.05 | Sweet spot |
| `uploadm8_put_250` | put | 250 | $9.99 | $0.04 | Best value |
| `uploadm8_put_500` | put | 500 | $17.99 | $0.036 | Power users |
| `uploadm8_put_1000` | put | 1000 | $29.99 | $0.03 | Bulk |
| `uploadm8_aic_50` | aic | 50 | $2.99 | $0.06 | AI taste |
| `uploadm8_aic_100` | aic | 100 | $4.99 | $0.05 | Sweet spot |
| `uploadm8_aic_250` | aic | 250 | $9.99 | $0.04 | Best value |
| `uploadm8_aic_500` | aic | 500 | $17.99 | $0.036 | Power users |
| `uploadm8_aic_1000` | aic | 1000 | $29.99 | $0.03 | Bulk |

**Note:** Replace `uploadm8_aic_2500` with `uploadm8_aic_50` in Stripe if migrating. Create new price for `uploadm8_aic_50` if not exists.

**First top-up bonus:** +25% tokens (one-time, applied automatically)

---

## 6. PUT/AIC Cost Formulas

### PUT cost per upload
```
base 10 + HUD(5) + priority(5) + (platforms-1)×2 + (thumbnails-1)×1
```
- Typical 4-platform upload: 16 PUT
- With HUD + priority: 26 PUT

### AIC cost per upload
```
base by ai_depth: basic=2, enhanced=3, advanced=4, max=6
+ frame surcharge: 0 (≤6), +1 (7–12), +2 (13–24), +3 (25+)
```
- Typical basic AI: 2 AIC
- Max AI, 15 frames: 6–9 AIC

---

## 7. MRR & Margin Breakdown

### Revenue per tier (MRR)

| Tier | Price | Est. uploads* | $/upload (100% use) |
|------|-------|---------------|----------------------|
| Lite | $9.99 | ~25 | $0.40 |
| Pro | $19.99 | ~75 | $0.27 |
| Studio | $49.99 | ~215 | $0.23 |
| Agency | $99.99 | ~500 | $0.20 |

\*Using ~16 PUT + 3 AIC per upload

### Cost assumptions (per upload)
- Compute/transcode: ~$0.02
- OpenAI (AI captions): ~$0.02–0.05
- Storage/bandwidth: ~$0.01
- **Total:** ~$0.05–0.08/upload

### Margin
- Subscription: 2–8× cost at typical usage
- Top-ups: 3–5× subscription $/token → high margin on overflow

---

## 8. Signup Bonus

- **PUT:** 100 (one-time)
- **AIC:** 75 (one-time)

Free users get this on signup; use until depleted or upgrade/top-up.

---

## 9. Stripe Setup Checklist

1. **Products**
   - Create/update products for Creator Lite, Pro, Studio, Agency
   - Set `lookup_key` on each Price object

2. **Top-up products**
   - Create one-time payment products for each PUT and AIC pack
   - Set `lookup_key` exactly as in TOPUP_PRODUCTS

3. **Webhook**
   - `checkout.session.completed` — subscription + top-up handling
   - `invoice.paid` — monthly refill
   - `customer.subscription.updated` — tier changes
   - `invoice.payment_failed` — dunning

4. **Success/Cancel URLs**
   - Success: `{FRONTEND_URL}/billing/success.html?session_id={CHECKOUT_SESSION_ID}`
   - Cancel: `{FRONTEND_URL}/index.html#pricing`
