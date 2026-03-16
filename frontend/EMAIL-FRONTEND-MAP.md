# Email stages → Frontend map (uploadm8-auth/stages/emails)

This doc aligns [uploadm8-auth/stages/emails](https://github.com/sosswced/uploadm8-auth/tree/b77bd4940d798240d2b8542fd5b66adcacdf9d69/stages/emails) with this frontend. Every CTA and link in those emails must point to an existing, actionable page.

## Auth base URLs (from stages/emails/base.py)

| Constant        | Default URL              | Frontend page / behavior |
|-----------------|--------------------------|---------------------------|
| `URL_DASHBOARD` | `FRONTEND_URL/dashboard.html` | **dashboard.html** ✓ |
| `URL_SETTINGS`  | `FRONTEND_URL/settings.html`  | **settings.html** ✓ |
| `URL_BILLING`   | `FRONTEND_URL/billing.html`   | **billing.html** → redirects to **settings.html#billing** ✓ |
| `URL_PRICING`   | `FRONTEND_URL/index.html#pricing` | **index.html#pricing** ✓ |
| `URL_UNSUBSCRIBE` | `FRONTEND_URL/unsubscribe.html` | **unsubscribe.html** ✓ |

## Email-specific CTAs and logic

### Auth (auth.py)
- **Password reset** – link: `reset-password?token=…` → **reset-password.html** (reads `token`, POSTs to `/api/auth/reset-password`) ✓
- **Password changed** – CTA “Sign In” → **login.html** ✓
- **Account deleted** – copy mentions “app.uploadm8.com” for re-signup → **index.html** / signup ✓
- **Email change verification** – link: `verify-email?token=…` → **verify-email.html** (reads `token`, GET `/api/auth/verify-email?token=…`) ✓
- **Admin reset password** – CTA “Sign In Now” → **login.html** ✓

### Uploads (uploads.py)
- **Upload completed / failed** – “View Upload Details”, “Retry Upload” → **dashboard.html** (queue / history) ✓

### Billing (billing_changes.py, billing_subscriptions.py)
- Plan upgraded/downgraded, top-up receipt, renewal – CTAs “Dashboard”, “Manage Billing”, “Settings”, “View Plans”, “Start Uploading”, “Token Wallet” → **dashboard.html**, **billing.html** (→ settings#billing), **settings.html**, **index.html#pricing** ✓

### Admin actions (admin_actions.py)
- Wallet top-up / tier switch – “Use My Tokens Now”, “Token Balance”, “See What’s Unlocked”, “Billing” → **dashboard.html**, **billing.html** ✓

### Welcome / lifecycle (welcome_special.py, lifecycle.py)
- **Master admin welcome** – CTA “Open Admin Dashboard” → **admin.html** ✓
- Payment failed, trial ending, low tokens – “Update Payment Method”, “Continue Subscription”, “Top Up” → **billing.html** ✓

### Unsubscribe
- Footer “Manage email preferences” in base.py → **unsubscribe.html** ✓

## Frontend requirements (all satisfied)

1. **reset-password.html** – Accepts `?token=…`, shows form; submits to `/api/auth/reset-password` with `token` and `new_password`.
2. **verify-email.html** – Accepts `?token=…`, calls `/api/auth/verify-email?token=…`.
3. **unsubscribe.html** – Lets users manage email preferences (unsubscribe/categories).
4. **billing.html** – Exists and redirects to **settings.html#billing** so all email “Billing” / “Token Wallet” links land in the right place.
5. **settings.html** – On load and `hashchange`, if URL is `#billing` (or `#preferences`, etc.), the corresponding tab is opened so deep links work.

## Redirects & key flows (frontend)

- **Logout (this device)** – clears tokens, redirects to **login.html**.
- **Log out all other sessions** – Settings → Security → “Log Out All Other Sessions”; calls `/api/auth/logout-all`, then clears tokens and redirects to **login.html**.
- **Delete account** – Settings → Security → Danger Zone → Delete account. **Free users:** Immediate deletion; clears tokens, redirects to **index.html**. **Paid users:** Returns `deletion_scheduled` with `access_until`; user retains access until period end, then backend runs full deletion via webhook.
- **Top-up / Billing** – “Top up”, “Token Wallet”, “Manage Billing” from dashboard, upgrade modal, or emails → **billing.html** or **settings.html#billing** (billing.html redirects to settings.html#billing).
- **After Stripe checkout (subscription or top-up)** – success/cancel URLs are configured in backend; frontend **billing/success.html** (or equivalent) typically offers a link back to **settings.html#billing**.

## Keeping in sync

- When adding a new email in uploadm8-auth that links to the app, add its CTA URL here and ensure the target page exists.
- If the auth repo changes `FRONTEND_URL` or any `URL_*` constant, ensure this frontend still has the matching route or redirect (e.g. `billing.html`).
