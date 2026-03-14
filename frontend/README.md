# UploadM8 Frontend — Pricing & Billing

These files provide the **pricing section** and **billing section** for the UploadM8 app.

## Files

- **index.html** — Landing page with pricing grid and top-up section. Fetches from `GET /api/pricing`.
- **settings.html** — Settings page with wallet display, top-up grid, and cost/margin overview.

## Integration

If your frontend lives in a **separate repo** (e.g. `uploadm8-frontend`):

1. Copy the pricing grid markup and `loadPricing()` / `renderTiers()` / `renderTopups()` logic from `index.html` into your landing page.
2. Copy the billing section and `loadBilling()` logic from `settings.html` into your settings page.
3. Ensure `API_BASE` or `API_BASE_URL` points to your backend (e.g. `https://auth.uploadm8.com` in prod, `http://127.0.0.1:8000` in dev).

## API

- **GET /api/pricing** (public, no auth) — Returns `{ tiers, topups }` with PUT/AIC, prices, lookup keys.
- **GET /api/me** (auth) — Returns wallet, plan, entitlements.
- **POST /api/wallet/topup** (auth) — Body: `{ lookup_key }`. Returns `{ checkout_url }`.

## Checkout URLs

- Subscription: Your checkout flow (e.g. `/billing/checkout.html?lookup=uploadm8_creatorpro_monthly`).
- Top-up: `POST /api/wallet/topup` returns Stripe Checkout URL.
- Manage subscription: Stripe Customer Portal (e.g. `/billing/portal.html` or `GET /api/billing/portal`).
