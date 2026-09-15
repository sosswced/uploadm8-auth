# UploadM8 — stage security ops checklist
#
# Product controls live in code. This file is the operator habit list for
# pre-scale growth (not SOC 2 / ISO evidence).

## Monitored inboxes

| Address | Purpose | Habit |
|---------|---------|--------|
| security@uploadm8.com | Vulnerability disclosure + suspicious account reports | Acknowledge within 48 hours; triage within 10 business days (see security.html) |
| privacy@uploadm8.com | Data deletion / rights requests | Acknowledge within 5 business days; complete within 30 days when verified |

Owner (fill in): __________________
Last inbox check (date): __________

## Secrets live only in host env

Never commit `.env`, `uploadm8-auth.env`, `*.local.js` brand/pixel files, or PEM/keys.

Prod secrets belong in Render (or equivalent) environment variables / secret store:

- `JWT_SECRET`
- `TOKEN_ENC_KEYS` (AES-GCM kids for OAuth blobs)
- `DATABASE_URL` / Redis URL
- Stripe secret + webhook secret
- Platform OAuth client secrets (Google / Meta / TikTok)
- R2 access keys
- Mailgun API key
- Optional: `SENTRY_DSN`

Rotation checklist (do in host console, then restart API/worker):

1. [ ] Generate new value offline
2. [ ] Set in Render (or host) secrets
3. [ ] For `TOKEN_ENC_KEYS`: keep old kid until all blobs re-encrypted / dual-key window ends
4. [ ] Restart API + worker
5. [ ] Confirm login + one platform publish path
6. [ ] Retire old secret

If a secret was ever in a local file, chat, or git history: rotate it.

Scan (from repo root):

```powershell
python scripts/check_secret_hygiene.py
python scripts/verify_prod_env.py
```

## Auth / session notes

- Same-site browser sessions: HttpOnly cookies (`uploadm8_access` / `uploadm8_refresh`).
- Cross-host / API clients: Bearer still supported.
- `POST /api/auth/logout` and `POST /api/auth/logout-all` both revoke all refresh rows for the user.
- `POST /api/auth/logout-other-sessions` keeps the current refresh token.
- Password reset email links: `/reset-password.html?token=...`

## Postgres backup restore drill

Hosting: Render PostgreSQL (or current provider). App does not implement its own dump.

Steps (operator):

1. Confirm automatic backups are enabled in the host dashboard.
2. Restore a backup into a **non-production** instance (or follow provider point-in-time restore into a scratch DB).
3. Verify `SELECT COUNT(*) FROM users` (or equivalent) succeeds on the restored copy.
4. Record below — do **not** overwrite production.

| Field | Value |
|-------|--------|
| Last successful restore drill | YYYY-MM-DD |
| Operator | |
| Provider / method | e.g. Render PITR / download restore |
| Notes | |

## Host defaults

- Public HTTPS via Render / Cloudflare edge (app does not terminate TLS itself).
- Postgres and Redis: private network / not publicly exposed.
- Card data: Stripe Checkout + Customer Portal only — UploadM8 does not store PANs.
- Optional observability: Sentry when `SENTRY_DSN` is set.

## Legal honesty

Public pages must match code. Do not claim admin MFA, 24/7 on-call paging, or quarterly restore drills unless those programs are real. Prefer this checklist + accurate security.html wording.
