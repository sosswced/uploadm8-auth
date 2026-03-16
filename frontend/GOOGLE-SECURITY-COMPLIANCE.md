# Google Cloud Security Best Practices – Compliance

This document records how this repository aligns with [Google's security best practices](https://cloud.google.com/iam/docs/best-practices-service-accounts) for service account keys and API keys.

## Scope

- **This repo (uploadm8-frontend):** Static frontend. No Google API keys or service account keys are used or stored here. All API calls go to `auth.uploadm8.com` (our backend).
- **Backend (uploadm8-auth, etc.):** If you use Google Cloud (e.g., YouTube Data API, GCP services), apply the practices below there.

## Checklist

| Practice | Frontend | Backend (if using GCP) |
|----------|----------|------------------------|
| **Zero-Code Storage** | ✓ No keys in code | Use Secret Manager; inject at runtime |
| **Disable Dormant Keys** | N/A | Audit keys; decommission unused >30 days |
| **Enforce API Restrictions** | N/A | Restrict keys to specific APIs and IP/referrer |
| **Least Privilege** | N/A | Use IAM recommender; minimal permissions |
| **Mandatory Rotation** | N/A | Use `iam.serviceAccountKeyExpiryHours` or disable key creation |

## What This Repo Does

1. **`.gitignore`** – Never commit `.env`, `*.pem`, `*.key`, `credentials.json`, or service account JSON files.
2. **No hardcoded secrets** – No API keys, JWT secrets, or credentials in source code.
3. **API base URL** – `auth.uploadm8.com` is a public endpoint, not a secret.
4. **README** – Documents env vars as placeholders only; backend must use Secret Manager.

## Backend Requirements (If Using GCP)

If your backend uses Google Cloud (YouTube API, etc.):

- Store credentials in **Secret Manager**; inject at runtime.
- Restrict API keys by API and environment (IP, referrer, bundle ID).
- Use IAM recommender to prune unused permissions.
- Apply `iam.serviceAccountKeyExpiryHours` or `iam.managed.disableServiceAccountKeyCreation`.
- Decommission keys with no activity in 30+ days.

## Audit

- Run: `git grep -E 'AIza|ya29\.|-----BEGIN|sk_live|whsec_|AKIA|GOCSPX'` to detect leaked keys.
- Ensure no `.env` or credential files are tracked: `git status` and `git diff --cached`.
