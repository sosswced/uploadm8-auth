# UploadM8 Backend

Production-ready FastAPI backend for the UploadM8 multi-platform video upload SaaS.

## Features

- **Authentication**: Email/password + JWT access/refresh tokens with rotation
- **Stripe Billing**: Checkout sessions, customer portal, webhooks, trials, auto-tax
- **Multi-Platform OAuth**: TikTok, YouTube, Instagram, Facebook
- **R2 Storage**: Cloudflare R2 presigned URLs for direct uploads
- **Redis Job Queue**: Async video processing with telemetry HUD overlay
- **Distributed Rate Limiting**: Redis-backed sliding window
- **Commercial KPIs**: Throughput, reliability, latency, platform mix, error analysis
- **Admin Dashboard**: Role-based admin panel with user management

## Architecture

```
┌─────────────────────┐     ┌─────────────────────┐
│   Frontend (HTML)   │────▶│    FastAPI API      │
│  app.uploadm8.com   │     │  auth.uploadm8.com  │
└─────────────────────┘     └──────────┬──────────┘
                                       │
         ┌─────────────────────────────┼─────────────────────────────┐
         │                             │                             │
         ▼                             ▼                             ▼
┌─────────────────┐         ┌─────────────────────┐       ┌─────────────────┐
│   PostgreSQL    │         │       Redis         │       │  Cloudflare R2  │
│   (Neon/Supabase)         │   (Queue + Cache)   │       │   (Video Store) │
└─────────────────┘         └──────────┬──────────┘       └─────────────────┘
                                       │
                                       ▼
                            ┌─────────────────────┐
                            │    Worker Service   │
                            │   (worker.py)       │
                            │  - Telemetry/HUD    │
                            │  - Platform Publish │
                            └─────────────────────┘
```

## Environment Variables

### Required

```bash
# Database (PostgreSQL)
DATABASE_URL=postgres://user:pass@host:5432/dbname

# JWT Authentication
JWT_SECRET=your-256-bit-secret-key

# Token Encryption (for OAuth tokens)
# Format: version:base64_32_byte_key (comma-separated for key rotation)
TOKEN_ENC_KEYS=v1:BASE64_ENCODED_32_BYTE_KEY
```

### R2 Storage

```bash
R2_ACCOUNT_ID=your-cloudflare-account-id
R2_ACCESS_KEY_ID=your-r2-access-key
R2_SECRET_ACCESS_KEY=your-r2-secret-key
R2_BUCKET_NAME=uploadm8-media
R2_ENDPOINT_URL=https://ACCOUNT_ID.r2.cloudflarestorage.com
```

### Redis (Job Queue)

```bash
REDIS_URL=redis://:password@host:6379/0
UPLOAD_JOB_QUEUE=uploadm8:jobs
TELEMETRY_JOB_QUEUE=uploadm8:telemetry
```

### Stripe Billing

```bash
STRIPE_SECRET_KEY=sk_live_or_test_xxx
STRIPE_WEBHOOK_SECRET=whsec_xxx
STRIPE_LOOKUP_KEYS=uploadm8_starter_monthly,uploadm8_solo_monthly,uploadm8_creator_monthly,uploadm8_growth_monthly,uploadm8_studio_monthly,uploadm8_agency_monthly
STRIPE_DEFAULT_LOOKUP_KEY=uploadm8_creator_monthly
STRIPE_SUCCESS_URL=https://app.uploadm8.com/billing-success.html?session_id={CHECKOUT_SESSION_ID}
STRIPE_CANCEL_URL=https://app.uploadm8.com/index.html#pricing
STRIPE_PORTAL_RETURN_URL=https://app.uploadm8.com/dashboard.html
STRIPE_TRIAL_DAYS_DEFAULT=0
STRIPE_AUTOMATIC_TAX=0
```

### Platform OAuth

```bash
# TikTok
TIKTOK_CLIENT_KEY=your-tiktok-client-key
TIKTOK_CLIENT_SECRET=your-tiktok-client-secret

# Google/YouTube
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret

# Meta (Facebook/Instagram)
META_APP_ID=your-meta-app-id
META_APP_SECRET=your-meta-app-secret
META_API_VERSION=v23.0
```

### Mailgun (Email)

```bash
MAILGUN_API_KEY=your-mailgun-api-key
MAILGUN_DOMAIN=your-mailgun-domain
MAIL_FROM=no-reply@uploadm8.com
```

### Admin & Notifications

```bash
ADMIN_API_KEY=your-admin-api-key
ADMIN_DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/xxx
BOOTSTRAP_ADMIN_EMAIL=your-email@example.com
```

### Application URLs

```bash
BASE_URL=https://auth.uploadm8.com
FRONTEND_URL=https://app.uploadm8.com
ALLOWED_ORIGINS=https://app.uploadm8.com,https://uploadm8.com
```

### Optional Tuning

```bash
LOG_LEVEL=INFO
ACCESS_TOKEN_MINUTES=15
REFRESH_TOKEN_DAYS=30
RATE_LIMIT_WINDOW_SEC=60
RATE_LIMIT_MAX=60
WORKER_CONCURRENCY=1
JOB_TIMEOUT_SECONDS=600
```

## Deployment

### API Service (Render)

1. Create new **Web Service**
2. Connect your GitHub repo
3. **Build Command**: `pip install -r requirements.txt`
4. **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. Add all environment variables
6. Set custom domain: `auth.uploadm8.com`

### Worker Service (Render)

1. Create new **Background Worker**
2. Connect same GitHub repo
3. **Build Command**: `pip install -r requirements.txt`
4. **Start Command**: `python worker.py`
5. Add same environment variables
6. Set instance type based on video processing needs

### Redis (Upstash/Redis Cloud)

1. Create Redis instance
2. Copy connection URL to `REDIS_URL`

## API Endpoints

### Authentication
- `POST /api/auth/register` - Create account
- `POST /api/auth/login` - Login
- `POST /api/auth/refresh` - Refresh token
- `POST /api/auth/logout` - Logout
- `GET /api/auth/me` - Get current user
- `POST /api/auth/password-reset` - Request reset
- `POST /api/auth/password-reset/confirm` - Confirm reset

### Uploads
- `POST /api/uploads/presign` - Get presigned upload URL
- `POST /api/uploads/{id}/complete` - Mark upload complete
- `POST /api/uploads/{id}/cancel` - Cancel upload
- `GET /api/uploads` - List uploads
- `GET /api/uploads/{id}` - Get upload details
- `POST /api/uploads/{id}/telemetry` - Get telemetry presigned URL
- `POST /api/uploads/{id}/telemetry/complete` - Complete telemetry upload

### Billing
- `GET /api/billing/prices` - Get Stripe prices
- `POST /api/billing/checkout` - Create checkout session
- `POST /api/billing/portal` - Create customer portal session
- `POST /api/billing/webhook` - Stripe webhook handler

### Analytics & KPI
- `GET /api/analytics/overview` - User analytics
- `GET /api/analytics/timeseries` - Time series data
- `GET /api/kpi/summary` - Commercial KPI summary
- `GET /api/kpi/raw` - Raw upload data for export

### Platforms
- `GET /api/platforms` - Get connected platforms
- `GET /oauth/{platform}/start` - Start OAuth flow
- `GET /oauth/{platform}/callback` - OAuth callback

### Settings
- `GET /api/settings` - Get user settings
- `PUT /api/settings` - Update settings

### Admin
- `GET /api/admin/overview` - Admin dashboard data
- `GET /api/admin/kpi/global` - Global KPIs
- `GET /api/admin/users/search` - Search users
- `POST /api/admin/users/role` - Set user role
- `POST /api/admin/entitlements/grant` - Grant entitlement

## Tier Structure

| Tier | Price | Uploads/mo | Accounts | Features |
|------|-------|------------|----------|----------|
| Starter | $0 | 10 | 1 | Manual queue, community support |
| Solo | $9.99 | 60 | 2 | Basic scheduling, Discord alerts |
| Creator | $19.99 | 200 | 4 | Smart scheduling, auto-transcode, resumable |
| Growth | $29.99 | 500 | 8 | Caption templates, advanced alerts, exports |
| Studio | $49.99 | 1,500 | 15 | 3 team seats, approval workflows, webhooks |
| Agency | $99.99 | 5,000 | 40 | 10 team seats, audit logs, SLA support |

## Setting Up Stripe Products

Create products in Stripe Dashboard with these lookup keys:
- `uploadm8_starter_monthly`
- `uploadm8_solo_monthly`
- `uploadm8_creator_monthly`
- `uploadm8_growth_monthly`
- `uploadm8_studio_monthly`
- `uploadm8_agency_monthly`

## License

Proprietary - UploadM8 © 2026
