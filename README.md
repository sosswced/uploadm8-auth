# UploadM8 v3.1 - Production Ready

**Multi-Platform Video Upload SaaS**

Upload once, publish everywhere. TikTok, YouTube Shorts, Instagram Reels, Facebook Reels.

## What's New in v3.1

### Critical Fixes
- **Auth Flow Fixed**: Hard auth gate on all protected pages, no more login loops
- **Session Expiry Handling**: Graceful redirect with message when session expires
- **Logout Fixed**: Properly clears tokens and redirects
- **Server-Authoritative Cancel**: Cancel in-progress uploads reliably

### New Features
- **Dark/Light Mode Toggle**: Theme persists across sessions
- **Admin KPI Dashboard**: Time-range selector (30m to 1y + custom), leaderboard, analytics
- **Request ID Tracking**: Every API call tagged for debugging
- **Professional Landing Page**: Trust indicators, security badges, pricing tiers

### Pages
- `index.html` - Landing page with pricing
- `login.html` - Sign in with password toggle, remember me
- `signup.html` - Registration with password strength indicator
- `forgot-password.html` - Password reset request
- `dashboard.html` - User dashboard with stats
- `upload.html` - Drag-drop upload with progress
- `queue.html` - Upload queue with cancel/retry
- `scheduled.html` - Scheduled uploads management
- `platforms.html` - Connected account management
- `groups.html` - Account group management
- `analytics.html` - Upload performance analytics
- `settings.html` - User settings and billing
- `admin.html` - Admin panel
- `admin-users.html` - User management
- `admin-kpi.html` - KPI dashboard
- `terms.html` - Terms of Service
- `privacy.html` - Privacy Policy
- `support.html` - Support/FAQ

## Pricing Tiers

| Tier | Price | Uploads/mo | Accounts | Features |
|------|-------|------------|----------|----------|
| Free | $0 | 5 | 1 | Basic scheduling |
| Starter | $19/mo | 50 | 5 | Smart scheduling, basic captions |
| Pro | $49/mo | 500 | 15 | AI captions, thumbnails, analytics, no watermark |
| Agency | $149/mo | Unlimited | 50 | Team collab, white-label, API access |
| Lifetime | $499 | Unlimited | 100 | Everything in Pro, forever |

## Tech Stack

**Frontend:**
- Vanilla JS (no framework dependencies)
- CSS with CSS variables for theming
- Font Awesome icons
- Inter font family

**Backend (FastAPI):**
- PostgreSQL (Neon)
- Redis (Upstash)
- Cloudflare R2 storage
- Stripe billing
- JWT authentication

## Deployment

### Frontend (Cloudflare Pages)
1. Connect to git repo or drag-drop upload
2. Set build command: (none, static files)
3. Set output directory: `/`
4. Deploy to `app.uploadm8.com`

### Backend (Render)
1. Connect to git repo
2. Set Python 3.11 runtime
3. Set start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
4. Add environment variables
5. Deploy to `auth.uploadm8.com`

## Environment Variables

```
DATABASE_URL=postgresql://...
JWT_SECRET=your-secret-key
REDIS_URL=redis://...
R2_ACCOUNT_ID=...
R2_ACCESS_KEY_ID=...
R2_SECRET_ACCESS_KEY=...
R2_BUCKET_NAME=uploadm8-media
STRIPE_SECRET_KEY=sk_...
STRIPE_WEBHOOK_SECRET=whsec_...
FRONTEND_URL=https://app.uploadm8.com
BASE_URL=https://auth.uploadm8.com
```

## File Structure

```
uploadm8-complete-v3/
├── index.html          # Landing page
├── login.html          # Auth
├── signup.html         # Registration
├── forgot-password.html
├── dashboard.html      # Main app
├── upload.html
├── queue.html
├── scheduled.html
├── platforms.html
├── groups.html
├── analytics.html
├── settings.html
├── admin.html          # Admin
├── admin-users.html
├── admin-kpi.html
├── terms.html          # Legal
├── privacy.html
├── support.html
├── app.js              # Core JS
├── styles.css          # Styles
├── logo.png
├── logo.svg
└── backend/
    ├── app.py          # FastAPI (3200+ lines)
    ├── worker.py       # Background jobs
    ├── requirements.txt
    └── stages/         # Pipeline stages
```

## API Endpoints (65+)

### Auth
- `POST /api/auth/register`
- `POST /api/auth/login`
- `POST /api/auth/logout`
- `POST /api/auth/refresh`
- `POST /api/auth/forgot-password`
- `GET /api/me`

### Uploads
- `POST /api/uploads/presign`
- `POST /api/uploads/{id}/complete`
- `POST /api/uploads/{id}/cancel` - Server-authoritative cancel
- `POST /api/uploads/{id}/retry`
- `GET /api/uploads`
- `GET /api/uploads/{id}`
- `DELETE /api/uploads/{id}`

### Platforms
- `GET /api/accounts`
- `POST /api/accounts`
- `DELETE /api/accounts/{id}`
- `GET /api/oauth/{platform}/start`
- `GET /api/oauth/{platform}/callback`

### Admin
- `GET /api/admin/users`
- `GET /api/admin/kpis`
- `GET /api/admin/leaderboard`
- `GET /api/admin/countries`

## Contact

Earl @ uploadm8.com

---

© 2025 UploadM8. All rights reserved.
