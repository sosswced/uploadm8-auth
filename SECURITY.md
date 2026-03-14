# Security & Credential Management

This document outlines security best practices for UploadM8, aligned with Google Cloud's security framework and industry standards.

---

## 1. Zero-Code Storage (Never Commit Keys)

**Requirement:** Never commit API keys, secrets, or credentials to source code or version control.

**Current compliance:**
- All credentials are loaded via `os.environ.get()` — no hardcoded secrets in source
- `.env` and `.env.*` are in `.gitignore` — never commit env files
- Use `.env.example` as a template only — it contains placeholder values, never real secrets

**Action items:**
- Use [Google Secret Manager](https://cloud.google.com/secret-manager) (or your host's secret store) to inject credentials at runtime in production
- On Render/Railway: use Environment Variables in the dashboard — values are encrypted at rest
- Verify `.env` is never committed: `git status` should never show `.env`

---

## 2. Google OAuth / YouTube Credentials

**Env vars used:** `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` (or `YOUTUBE_CLIENT_ID`, `YOUTUBE_CLIENT_SECRET`)

**Google Cloud Console actions (you must do these):**

### API restrictions
- Go to [Google Cloud Console → APIs & Services → Credentials](https://console.cloud.google.com/apis/credentials)
- Edit your OAuth 2.0 Client ID
- Under **API restrictions**: select "Restrict key"
- Add only: **YouTube Data API v3** (and any other APIs you actually use)
- Never leave the key unrestricted

### Application restrictions
- Under **Application restrictions**: restrict by type
  - For web: add your authorized JavaScript origins (e.g. `https://app.uploadm8.com`, `http://localhost:3000`)
  - For server-side: add authorized redirect URIs only

---

## 3. Disable Dormant Keys

**Requirement:** Decommission keys with no activity in the last 30 days.

**Action items:**
- In [Google Cloud Console → APIs & Services → Credentials](https://console.cloud.google.com/apis/credentials), review each key
- Check **Key usage** / **Last used** — delete or disable keys with no recent activity
- For OAuth client IDs: if you have multiple, remove unused ones

---

## 4. Least Privilege

**Requirement:** Grant only the minimum permissions required.

**OAuth scopes (YouTube):**
- Request only the scopes you need for YouTube Shorts upload
- Typical: `https://www.googleapis.com/auth/youtube.upload`, `https://www.googleapis.com/auth/youtube.readonly`

**Service accounts (if any):**
- Use [IAM Recommender](https://cloud.google.com/iam/docs/recommender-overview) to prune unused permissions
- Avoid `roles/owner` or broad roles — use narrow custom roles

---

## 5. Mandatory Rotation

**Requirement:** Rotate credentials periodically and enforce key expiry where supported.

**Action items:**
- Rotate `GOOGLE_CLIENT_SECRET` (and other OAuth client secrets) at least annually
- Rotate `JWT_SECRET`, `TOKEN_ENC_KEYS`, database passwords on a schedule
- For GCP service account keys: set `iam.serviceAccountKeyExpiryHours` if using key-based auth
- If you don't need user-managed service account keys: `iam.managed.disableServiceAccountKeyCreation`

---

## 6. Sensitive Data in Logs

**Current compliance:**
- `_SENSITIVE_KEYS` in `app.py` redacts `access_token`, `client_secret`, `code`, `refresh_token`, `fb_exchange_token` from logged URLs
- Never log raw credentials

---

## 7. If Credentials Are Exposed

If `.env` or any secret was ever committed to git or shared:

1. **Rotate immediately:** Regenerate every key in the exposed file (Google, Stripe, JWT, DB, Mailgun, etc.)
2. **Remove from git history:** Use `git filter-repo` or BFG Repo-Cleaner to purge the file from history
3. **Revoke and recreate:** OAuth client secrets, API keys, database passwords
4. **Notify:** If user data was at risk, follow your incident response plan

---

## Env Vars Reference

See `.env.example` for the list of required variables. All must be set via environment (never hardcoded).

---

## Optional: Secret Detection in CI

To catch accidental commits of secrets, add a check to your CI:

```bash
# Fail if .env is tracked
git ls-files | grep -E '^\.env$|^\.env\.' && exit 1 || exit 0
```

Or use [gitleaks](https://github.com/gitleaks/gitleaks) or [trufflehog](https://github.com/trufflesecurity/trufflehog) for broader secret scanning.
