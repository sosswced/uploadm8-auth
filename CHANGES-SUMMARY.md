# UploadM8 — File & Change Summary

**Generated:** March 14, 2026  
**Repos:** `uploadm8-auth` (backend) + `uploadm8-frontend` (frontend)

---

## Overview

This document summarizes all files and recent changes across the UploadM8 project, including what was modified and why.

---

## 1. Backend (uploadm8-auth)

### Recent Commits (chronological)

| Commit | Date | Summary |
|--------|------|---------|
| `a013dc4` | Mar 14, 2026 | Pricing overhaul: entitlements, frontend, first top-up bonus, caption settings |
| `bcf8cda` | — | Merge: Bug fixes doc for frontend agent |
| `341d4d0` | — | docs: Bug Fixes section (scheduled, queue, dashboard, feature guide) |
| `f6175ac` | Mar 14, 2026 | Queue/dashboard status logic, platform hashtags, multi-account, Discord sync, default credits |
| `5265e82` | — | KPI collector: Stripe/Mailgun/OpenAI cost sync every 30 min |
| `b59cdab` | — | fix: admin Discord webhook from saved admin_settings |
| `3968c74` | — | max_parallel_uploads in entitlements for batch upload mode |
| `22c77f5` | — | Security: SECURITY.md, .env.example, JWT fix; TOS; emails |

---

### Core Application

| File | Purpose | Recent Changes |
|------|---------|----------------|
| **app.py** | FastAPI app, routes, auth, billing | Added `GET /api/pricing` (public), first top-up +25% bonus, `TIER_CONFIG` fix for `admin_assign_tier`, billing portal redirect |
| **worker.py** | Background job processor | Added caption settings support, entitlements integration, queue/scheduled handling |
| **requirements.txt** | Python dependencies | Added packages for KPI collector, Stripe/Mailgun/OpenAI sync |

---

### Stages (Pipeline)

| File | Purpose | Recent Changes |
|------|---------|----------------|
| **stages/entitlements.py** | Tier limits, Stripe mapping | Value-first tiers (Free 80/50, Lite 400/120, Pro 1200/350, Studio 3500/1000, Agency 8000/2500); `TOPUP_PRODUCTS` with price/price_usd; `STRIPE_LOOKUP_TO_TIER`; launch tier mapping |
| **stages/caption_stage.py** | AI caption generation | Caption style/tone/voice/frame count from `users.preferences`; narrative prompt updates |
| **stages/context.py** | Pipeline context, user settings | Extended for caption prefs, platform hashtags, multi-account |
| **stages/db.py** | DB helpers, user settings | `load_user_settings` merges `users.preferences`; new queries for pricing/entitlements |
| **stages/thumbnail_stage.py** | AI thumbnail generation | AI styling, interval logic, tier-based limits |
| **stages/publish_stage.py** | Multi-platform publishing | Multi-account `target_accounts`, platform hashtags, Discord sync |
| **stages/notify_stage.py** | Notifications | Discord webhook, status updates |
| **stages/kpi_collector.py** | Cost/metrics sync | Stripe, Mailgun, OpenAI cost sync every 30 min |

---

### Emails

| File | Purpose | Recent Changes |
|------|---------|----------------|
| **stages/emails/billing_changes.py** | Billing change emails | Top-up receipt shows first-time +25% bonus |
| **stages/emails/base.py** | Email base, templates | — |
| **stages/emails/auth.py** | Auth-related emails | — |
| **stages/emails/lifecycle.py** | Welcome, onboarding | — |
| **stages/emails/uploads.py** | Upload notifications | — |
| **stages/emails/admin_actions.py** | Admin action emails | — |
| **stages/emails/announcements.py** | Announcement emails | — |
| **stages/emails/billing_subscriptions.py** | Subscription emails | — |
| **stages/emails/welcome_special.py** | Welcome/special offers | — |
| **stages/emails_backup/** | Backup of email modules | Full copy of `stages/emails` before refactors |

---

### Frontend (in auth repo)

| File | Purpose | Recent Changes |
|------|---------|----------------|
| **frontend/index.html** | Landing / pricing page | Pricing grid, tier comparison, top-ups, billing portal CTA |
| **frontend/settings.html** | User settings | Billing section, cost/margin display, top-up links, billing portal API |
| **frontend/README.md** | Frontend notes | Added docs for auth repo frontend assets |

---

### Documentation

| File | Purpose | Recent Changes |
|------|---------|----------------|
| **PRICING-ENTITLEMENTS.md** | Stripe setup, tier entitlements | New: value-first tiers, PUT/AIC limits, top-up products, Stripe lookup keys |
| **CAPTION-SETTINGS-LOGIC.md** | Caption/AI settings flow | New: why settings don't save, style/tone/voice mapping, storage keys |
| **CONTENT-GENERATION-ARCHITECTURE.md** | Titles, captions, thumbnails | New: settings flow, API endpoints, content category detection |
| **BILLING-UPSELL-BANNERS.md** | Token usage UX spec | New: usage thresholds, banner types, placement, pre-submit warnings |
| **PROMPT-FOR-FRONTEND-AGENT.md** | Frontend agent instructions | Bug fixes section for scheduled, queue, dashboard, guide |
| **SECURITY.md** | Security practices | JWT fix, .env.example, TOS |
| **README.md** | Project overview | General updates |

---

### Config & Misc

| File | Purpose | Recent Changes |
|------|---------|----------------|
| **.env.example** | Env var template | Security-related vars |
| **render.yaml** | Render deployment | — |
| **Dockerfile** | Container build | — |
| **.gitignore** | Git ignore rules | Secure rules for secrets |
| **auth_upstream.txt** | Upstream reference | — |
| **commit-a013dc4-diff.txt** | Diff archive | Full diff of pricing overhaul commit |

---

## 2. Frontend (uploadm8-frontend)

### Recent Commits

| Commit | Date | Summary |
|--------|------|---------|
| `692a09b` | Mar 14, 2026 | Update styles.css |
| `7b8d0cd` | Mar 14, 2026 | Update HTML and JS: dashboard, queue, scheduled, settings, upload, admin, app.js, wallet-tokens |
| `d0a4696` | — | Merge: scheduled-queue-dashboard-guide fixes |
| `3b84b99` | — | fix: scheduled date modal, queue [object Object], dashboard user/plan, feature guide nav |
| `8f796a9` | — | Dashboard stats, queue/dashboard status logic, status_label display |

---

### HTML Pages

| File | Purpose | Recent Changes |
|------|---------|----------------|
| **index.html** | Landing / marketing | Pricing, signup CTA |
| **dashboard.html** | Main dashboard | Stats, status logic, user/plan display, token indicators |
| **upload.html** | Upload flow | Cost estimate, target_accounts, top-up prompts |
| **queue.html** | Upload queue | Status labels, edit metadata, [object Object] fix |
| **scheduled.html** | Scheduled uploads | Date modal fix, edit capability, status display |
| **settings.html** | User settings | Billing, cost/margin, top-ups, caption preferences |
| **admin.html** | Admin panel | — |
| **admin-calculator.html** | Cost calculator | Live pricing from API |
| **admin-kpi.html** | KPI dashboard | — |
| **admin-wallet.html** | Admin wallet | API spec, encodeURIComponent for adjust URL |
| **account-management.html** | Account management | — |
| **analytics.html** | Analytics view | — |
| **color-preferences.html** | Color prefs | — |
| **groups.html** | Groups management | — |
| **guide.html** | Feature guide | Nav, delete confirmations |
| **platforms.html** | Platform connections | — |
| **billing.html** | Billing page | — |
| **terms.html**, **privacy.html**, **security.html** | Legal pages | — |
| **login.html**, **signup.html**, **forgot-password.html** | Auth flows | — |
| **verify-email.html**, **reset-password.html** | Auth verification | — |

---

### JavaScript

| File | Purpose | Recent Changes |
|------|---------|----------------|
| **app.js** | Main app logic | API calls, dashboard stats, queue/scheduled status, pricing fetch |
| **wallet-tokens.js** | Token wallet UI | New: top-up flows, token display, billing portal |
| **billing.js** | Billing logic | — |
| **auth-init.js**, **auth-core.js** | Auth helpers | — |
| **upgrade-modal.js** | Upgrade modal | — |
| **success-modal.js** | Success modal | — |
| **shared-sidebar.js** | Shared sidebar | — |

---

### Styles

| File | Purpose | Recent Changes |
|------|---------|----------------|
| **styles.css** | Global styles | +29 lines: token bar, banners, top-up UI, status colors |

---

## 3. Why These Changes Were Made

### Pricing Overhaul (a013dc4)

- **Value-first tiers**: Free tier gives enough to experience value; paid tiers scale exponentially so each feels meaningfully better.
- **Top-ups**: Attractive pricing for one-off token purchases; first top-up gets +25% bonus to encourage trial.
- **Public pricing API**: `GET /api/pricing` lets frontend show live pricing without auth.
- **Caption settings**: Style, tone, voice, frame count stored in `users.preferences` and used by worker; fixes "settings don't save" issue.

### Frontend Alignment (7b8d0cd, 692a09b)

- **Dashboard/queue/scheduled**: Status logic matches backend; fixes `[object Object]` display; edit metadata support.
- **Token display**: PUT/AIC shown in header; top-up CTAs when low.
- **Settings**: Billing section, cost/margin, billing portal link.
- **Upload**: Live cost estimate, target_accounts support.
- **wallet-tokens.js**: Dedicated module for token wallet and top-up flows.

### Queue/Dashboard Status (f6175ac)

- **Status labels**: Clear status for queue/scheduled items (pending, processing, published, failed).
- **Platform hashtags**: Per-platform hashtag support in publish stage.
- **Multi-account**: `target_accounts` for publishing to multiple accounts.
- **Discord sync**: Admin webhook for upload status.
- **Default credits**: New users get tier-appropriate starting balance.

### Documentation

- **PRICING-ENTITLEMENTS.md**: Single source of truth for Stripe products, lookup keys, tier limits.
- **CAPTION-SETTINGS-LOGIC.md**: Explains why caption settings must use `PUT /api/me/preferences`.
- **CONTENT-GENERATION-ARCHITECTURE.md**: Maps settings flow from API → worker → caption/thumbnail stages.
- **BILLING-UPSELL-BANNERS.md**: UX spec for token usage banners and upsell flows.

---

## 4. File Count Summary

| Repo | Category | Count |
|------|----------|-------|
| uploadm8-auth | Python (app, worker, stages) | ~40 |
| uploadm8-auth | Documentation (.md) | 8 |
| uploadm8-auth | Frontend (HTML in auth repo) | 2 |
| uploadm8-frontend | HTML pages | 35+ |
| uploadm8-frontend | JavaScript | 10+ |
| uploadm8-frontend | CSS | 1 |

---

*This summary was generated from git history and file inspection. For exact diffs, see `git log` and `git show <commit>`.*
