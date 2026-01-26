# UploadM8 - Multi-Platform Video Upload SaaS

## Overview

UploadM8 is a SaaS platform that allows content creators to upload videos once and distribute them simultaneously to TikTok, YouTube Shorts, Instagram Reels, and Facebook Reels.

## Features by Tier

| Feature | Starter | Creator | Growth | Studio | Agency |
|---------|---------|---------|--------|--------|--------|
| **Monthly Uploads** | 10 | 200 | 500 | 1,500 | 5,000 |
| **Connected Accounts** | 1 | 4 | 8 | 15 | 40 |
| **Auto Captions** | ❌ | ✅ | ✅ | ✅ | ✅ |
| **HUD Overlay** | ❌ | ✅ | ✅ | ✅ | ✅ |
| **AI Captions** | ❌ | ❌ | ✅ | ✅ | ✅ |
| **Priority Processing** | ❌ | ✅ | ✅ | ✅ | ✅ |
| **Team Seats** | 1 | 1 | 1 | 3 | 10 |
| **History Retention** | 7 days | 30 days | 30 days | 90 days | 365 days |
| **Support** | Basic | Standard | Standard | Priority | SLA |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    app.uploadm8.com (Frontend)                  │
│                    Render Static Site                           │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                   auth.uploadm8.com (API)                       │
│                   Render Web Service                            │
│                        app.py                                   │
├─────────────────────────────────────────────────────────────────┤
│  • Authentication (JWT + OAuth)                                 │
│  • Upload management                                            │
│  • Stripe billing                                               │
│  • Admin APIs                                                   │
└─────────────────────────────────────────────────────────────────┘
         │              │               │               │
         ▼              ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  PostgreSQL  │ │    Redis     │ │ Cloudflare   │ │   Stripe     │
│   (Render)   │ │  (Upstash)   │ │     R2       │ │  Payments    │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Background Worker                             │
│                   Render Worker Service                         │
│                       worker.py                                 │
├─────────────────────────────────────────────────────────────────┤
│  Stages:                                                        │
│  1. Telemetry parsing (.map files)                              │
│  2. Caption generation (Trill-based)                            │
│  3. HUD overlay (FFmpeg)                                        │
│  4. Platform publishing (TikTok, YouTube, Instagram, Facebook)  │
│  5. Discord notifications                                       │
└─────────────────────────────────────────────────────────────────┘
```

## Deployment on Render

### 1. Create PostgreSQL Database

1. Dashboard → New → PostgreSQL
2. Name: `uploadm8-db`
3. Copy the **Internal Database URL**

### 2. Create Redis (Upstash)

1. Go to [upstash.com](https://upstash.com)
2. Create Redis database
3. Copy the Redis URL

### 3. Create Web Service (API)

1. Dashboard → New → Web Service
2. Connect your GitHub repo
3. **Settings:**
   - Name: `uploadm8-auth`
   - Runtime: Python 3
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - Custom Domain: `auth.uploadm8.com`

### 4. Create Background Worker

1. Dashboard → New → Background Worker
2. Connect same repo
3. **Settings:**
   - Name: `uploadm8-worker`
   - Runtime: Python 3
   - Build Command: `pip install -r requirements.txt && apt-get update && apt-get install -y ffmpeg`
   - Start Command: `python worker.py`

### 5. Create Static Site (Frontend)

1. Dashboard → New → Static Site
2. Connect repo (or upload `static/` folder)
3. **Settings:**
   - Name: `uploadm8-app`
   - Publish Directory: `static`
   - Custom Domain: `app.uploadm8.com`

## Environment Variables

### Required for API (app.py)

```env
# Database
DATABASE_URL=postgresql://...

# JWT
JWT_SECRET=your-64-char-random-string
JWT_ISSUER=https://auth.uploadm8.com
JWT_AUDIENCE=uploadm8-app

# Token Encryption (generate with: python -c "import secrets,base64; print('v1:' + base64.b64encode(secrets.token_bytes(32)).decode())")
TOKEN_ENC_KEYS=v1:base64-encoded-32-byte-key

# URLs
BASE_URL=https://auth.uploadm8.com
FRONTEND_URL=https://app.uploadm8.com

# Redis
REDIS_URL=redis://...

# R2 Storage
R2_ACCOUNT_ID=your-account-id
R2_ACCESS_KEY_ID=your-access-key
R2_SECRET_ACCESS_KEY=your-secret-key
R2_BUCKET_NAME=uploadm8-media
R2_ENDPOINT_URL=https://your-account-id.r2.cloudflarestorage.com

# Stripe
STRIPE_SECRET_KEY=sk_live_...
STRIPE_WEBHOOK_SECRET=whsec_...

# OAuth (TikTok)
TIKTOK_CLIENT_KEY=your-key
TIKTOK_CLIENT_SECRET=your-secret

# OAuth (Google/YouTube)
GOOGLE_CLIENT_ID=your-id
GOOGLE_CLIENT_SECRET=your-secret

# OAuth (Meta/Instagram/Facebook)
META_APP_ID=your-app-id
META_APP_SECRET=your-secret

# Admin
BOOTSTRAP_ADMIN_EMAIL=your@email.com
ADMIN_API_KEY=your-admin-key

# Billing (set to "test" initially)
BILLING_MODE=test
BILLING_LIVE_ALLOWED=0
PRODUCTION_HOSTS=auth.uploadm8.com,app.uploadm8.com

# CORS
ALLOWED_ORIGINS=https://app.uploadm8.com,https://uploadm8.com
```

### Required for Worker

Same as API, plus:
```env
UPLOAD_JOB_QUEUE=uploadm8:jobs
ADMIN_DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
```

## Stripe Setup

### Create Products

1. Go to Stripe Dashboard → Products
2. Create products with these **lookup keys**:
   - `uploadm8_creator_monthly` - $19.99/month
   - `uploadm8_growth_monthly` - $29.99/month
   - `uploadm8_studio_monthly` - $49.99/month
   - `uploadm8_agency_monthly` - $99.99/month

### Configure Webhook

1. Stripe Dashboard → Developers → Webhooks
2. Add endpoint: `https://auth.uploadm8.com/api/billing/webhook`
3. Events to send:
   - `checkout.session.completed`
   - `customer.subscription.updated`
   - `customer.subscription.deleted`
   - `invoice.paid`
   - `invoice.payment_failed`

## Admin Access

### Restore Admin (if locked out)

```bash
curl -X POST https://auth.uploadm8.com/api/admin/restore \
  -H "Content-Type: application/json" \
  -d '{"secret_key": "YOUR_ADMIN_API_KEY", "email": "your@email.com"}'
```

### Grant Entitlements (via Admin Panel)

```bash
curl -X POST https://auth.uploadm8.com/api/admin/entitlements/grant \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "tier": "growth", "upload_quota": 500}'
```

## File Structure

```
uploadm8-final/
├── app.py              # FastAPI backend
├── worker.py           # Background job processor
├── requirements.txt    # Python dependencies
├── runtime.txt         # Python version for Render
├── stages/             # Worker processing stages
│   ├── __init__.py
│   ├── entitlements.py # Tier configuration
│   ├── context.py      # Job context
│   ├── errors.py       # Error codes
│   ├── db.py           # Database operations
│   ├── r2.py           # R2 storage operations
│   ├── telemetry_stage.py
│   ├── caption_stage.py
│   ├── hud_stage.py
│   ├── publish_stage.py
│   └── notify_stage.py
└── static/             # Frontend files
    ├── index.html      # Landing page
    ├── dashboard.html  # Main app
    ├── admin.html      # Admin panel
    ├── login.html
    ├── signup.html
    └── images/
        └── logo.svg
```

## Going Live Checklist

- [ ] Deploy API to Render
- [ ] Deploy Worker to Render
- [ ] Deploy Frontend to Render
- [ ] Configure custom domains
- [ ] Set up SSL certificates (automatic on Render)
- [ ] Configure Stripe products
- [ ] Configure Stripe webhook
- [ ] Set up OAuth apps (TikTok, Google, Meta)
- [ ] Test full flow in test mode
- [ ] Set `BILLING_MODE=live` and `BILLING_LIVE_ALLOWED=1`
- [ ] Monitor first real transactions

## Support

For issues, contact support@uploadm8.com
