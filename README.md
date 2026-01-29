# UploadM8 v3.2 - Production Ready

**Multi-Platform Video Upload SaaS**

Upload once, publish everywhere. TikTok, YouTube Shorts, Instagram Reels, Facebook Reels.

## What's New in v3.2

### Smart Upload Scheduling
- **AI-Powered Scheduling**: Automatically picks optimal posting times for each platform
- **Platform-Specific Hot Times**: Uses engagement data to schedule during peak hours
- **Different Days**: Spreads uploads across multiple days (no same-day posts)
- **Configurable Range**: Choose 7, 14, 21, or 30 day scheduling windows

### Always Hashtags Feature
- **Static Hashtags**: Set hashtags that are added to every upload
- **AI-Generated Hashtags**: Optionally generate relevant hashtags using AI
- **Platform-Specific Hashtags**: Different hashtags for TikTok, YouTube, Instagram, Facebook
- **Blocked Hashtags**: Prevent certain hashtags from ever being used
- **Hashtag Position**: Choose start, end, or first comment placement

### FFmpeg Video Transcoding
- **Automatic Format Conversion**: Converts videos to platform-required formats
- **YouTube Shorts Optimization**: H.264 codec, AAC audio, max 60 seconds
- **TikTok Optimization**: H.264 codec, proper aspect ratio, up to 10 minutes
- **Instagram Reels**: H.264 codec, max 90 seconds, proper dimensions
- **Smart Analysis**: Only transcodes when necessary (saves processing time)

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
- `upload.html` - Drag-drop upload with progress, smart scheduling
- `queue.html` - Upload queue with cancel/retry
- `scheduled.html` - Scheduled uploads management
- `platforms.html` - Connected account management
- `groups.html` - Account group management
- `analytics.html` - Upload performance analytics
- `settings.html` - User settings, hashtag settings, and billing
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

**Worker:**
- FFmpeg for video transcoding
- Platform-specific video optimization
- Background job processing

## System Requirements

### FFmpeg Installation (Required for Worker)

The worker requires FFmpeg to be installed for video transcoding.

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Docker (Render/Railway):**
Add to your Dockerfile:
```dockerfile
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*
```

**Verify installation:**
```bash
ffmpeg -version
ffprobe -version
```

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
5. **Ensure FFmpeg is installed** (use Docker or apt-get in build)
6. Deploy to `auth.uploadm8.com`

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
TOKEN_ENC_KEYS=v1:base64key...
```

## Video Processing Pipeline

1. **Download** - Fetch video from R2 storage
2. **Transcode** - Convert to platform-specific formats (H.264/AAC)
3. **Telemetry** - Parse driving data if provided
4. **Thumbnail** - Extract thumbnail frames
5. **Caption** - Generate AI captions (if enabled)
6. **HUD** - Add speed/telemetry overlay (if enabled)
7. **Watermark** - Add branding (free tier)
8. **Publish** - Upload to each platform with optimized video

## Platform Video Requirements

| Platform | Max Duration | Codec | Max Resolution | Max FPS |
|----------|--------------|-------|----------------|---------|
| YouTube Shorts | 60s | H.264 | 1080x1920 | 60 |
| TikTok | 10min | H.264 | 1080x1920 | 60 |
| Instagram Reels | 90s | H.264 | 1080x1920 | 30 |
| Facebook Reels | 90s | H.264 | 1080x1920 | 30 |

## File Structure

```
uploadm8-complete-v3/
├── index.html          # Landing page
├── login.html          # Auth
├── signup.html         # Registration
├── forgot-password.html
├── dashboard.html      # Main app
├── upload.html         # Smart scheduling
├── queue.html
├── scheduled.html
├── platforms.html
├── groups.html
├── analytics.html
├── settings.html       # Hashtag settings
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
    ├── app.py          # FastAPI
    ├── worker.py       # Background jobs
    ├── requirements.txt
    └── stages/         # Pipeline stages
        ├── transcode_stage.py  # FFmpeg video conversion
        ├── publish_stage.py    # Platform publishing
        └── ...
```

## API Endpoints (70+)

### Auth
- `POST /api/auth/register`
- `POST /api/auth/login`
- `POST /api/auth/logout`
- `POST /api/auth/refresh`
- `POST /api/auth/forgot-password`
- `GET /api/me`
- `PUT /api/me/preferences` - Hashtag and upload settings

### Uploads
- `POST /api/uploads/presign`
- `POST /api/uploads/smart-schedule/preview` - Preview smart schedule
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
