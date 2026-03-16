# UploadM8 v3.1 - Implementation Summary

## Critical Issues to Fix

### 1. Auth Token Keys (login.html)
```javascript
// WRONG (line 286, 394):
localStorage.getItem('access_token')
// CORRECT:
localStorage.getItem('uploadm8_access_token')
```

### 2. Footer Links (login.html, signup.html)
Add working href to Terms, Privacy, Support links

### 3. Home Button (login.html, signup.html)
Already implemented - verify visibility

### 4. Remove/Mark OAuth Buttons (login.html)
Google & Discord - mark as "Coming Soon" with disabled state

---

## New Database Tables

```sql
CREATE TABLE platform_accounts (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    platform VARCHAR(50),
    account_id VARCHAR(255),
    account_name VARCHAR(255),
    account_avatar TEXT,
    token_blob JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ
);

CREATE TABLE white_label_settings (
    user_id UUID PRIMARY KEY,
    enabled BOOLEAN DEFAULT FALSE,
    logo_url TEXT,
    company_name VARCHAR(255),
    primary_color VARCHAR(20)
);

CREATE TABLE revenue_tracking (
    id UUID PRIMARY KEY,
    amount DECIMAL(10,2),
    source VARCHAR(100),
    user_id UUID,
    stripe_payment_id VARCHAR(255),
    created_at TIMESTAMPTZ
);

CREATE TABLE cost_tracking (
    id UUID PRIMARY KEY,
    category VARCHAR(100),
    amount DECIMAL(10,2),
    description TEXT,
    created_at TIMESTAMPTZ
);
```

---

## User Settings Additions

```sql
ALTER TABLE user_settings ADD COLUMN ffmpeg_screenshot_interval INT DEFAULT 5;
ALTER TABLE user_settings ADD COLUMN auto_generate_thumbnails BOOLEAN DEFAULT TRUE;
ALTER TABLE user_settings ADD COLUMN auto_generate_captions BOOLEAN DEFAULT TRUE;
ALTER TABLE user_settings ADD COLUMN auto_generate_hashtags BOOLEAN DEFAULT TRUE;
ALTER TABLE user_settings ADD COLUMN default_hashtag_count INT DEFAULT 5;
ALTER TABLE user_settings ADD COLUMN watermark_enabled BOOLEAN DEFAULT TRUE;
```

---

## Tier Entitlements

| Feature | Free | Creator | Studio | Agency | Lifetime |
|---------|------|---------|--------|--------|----------|
| Uploads/mo | 5 | 200 | 1500 | 5000 | ∞ |
| Accounts | 1 | 4 | 15 | 50 | 100 |
| Hashtags | 2 | 10 | 30 | ∞ | ∞ |
| AI Captions | ❌ | ✅ | ✅ | ✅ | ✅ |
| Thumbnails | ❌ | ✅ | ✅ | ✅ | ✅ |
| Watermark | ✅ | ❌ | ❌ | ❌ | ❌ |
| Ads | ✅ | ❌ | ❌ | ❌ | ❌ |
| White Label | ❌ | ❌ | ✅ | ✅ | ✅ |
| Export | ❌ | ❌ | ✅ | ✅ | ✅ |

---

## New API Endpoints

```
POST /api/schedule/smart
POST /api/uploads/{id}/thumbnail/generate
POST /api/ai/generate-content
GET  /api/exports/excel
GET  /api/white-label
PUT  /api/white-label
GET  /api/platforms/{platform}/accounts
POST /api/platforms/{platform}/accounts
DELETE /api/platforms/{platform}/accounts/{id}
GET  /api/groups
POST /api/groups
PUT  /api/groups/{id}
DELETE /api/groups/{id}
GET  /api/admin/kpi/comprehensive
GET  /api/admin/revenue
GET  /api/admin/expenses
POST /api/admin/stripe/refund
```

---

## Discord Webhooks

Set these environment variables:
```
SIGNUP_DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
TRIAL_DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
MRR_DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
```

Triggered on:
- User signup
- Trial started
- Payment received (MRR)

---

## Key Files to Update

1. **backend/app.py** - Add new endpoints, migrations, webhooks
2. **backend/stages/thumbnail_stage.py** - New file for thumbnail generation
3. **backend/worker.py** - Add thumbnail stage to pipeline
4. **app.js** - Add new helper functions for features
5. **index.html** - Updated with new headline and features
6. **login.html** - Fix auth, add home button
7. **signup.html** - Fix alerts, add home button
8. **settings.html** - Add new settings fields
9. **platforms.html** - Rewrite for multi-account
10. **admin-kpi.html** - Add revenue/expense/hide figures
11. **admin-users.html** - Add Stripe integration
12. **analytics.html** - Add charts and KPIs
