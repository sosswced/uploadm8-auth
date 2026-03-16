# Dashboard Frontend Updates

Backend changes are in place. The frontend (dashboard.html, queue.html, etc.) should be updated as follows.

## 1. Credits Display (replace "Monthly Upload Quota")

**Current:** Dashboard shows "Monthly Upload Quota 0 / 999999"

**Change to:** Display PUT and AIC credits from wallet, not monthly quota.

**API:** `GET /api/dashboard/stats` now returns:

```json
{
  "credits": {
    "put": {
      "available": 80,
      "reserved": 0,
      "total": 80,
      "monthly_allowance": 80
    },
    "aic": {
      "available": 50,
      "reserved": 0,
      "total": 50,
      "monthly_allowance": 50
    }
  },
  "wallet": { "put_available": 80, "put_total": 80, "aic_available": 50, "aic_total": 50 },
  "quota": { "put_used": 0, "put_limit": 80 }
}
```

**Suggested UI:**
- **PUT credits:** `credits.put.available` / `credits.put.total` (or show "X available")
- **AIC credits:** `credits.aic.available` / `credits.aic.total`
- Label: "PUT Credits" and "AIC Credits" (not "Monthly Upload Quota")

## 2. Upload History — "All" Tab

**API:** `GET /api/uploads?view=all&meta=true` now returns all uploads (no status filter).

**Frontend:** Ensure the "All" tab sends `view=all` (not `view=completed` or no view). Same structure as Succeeded; just no status filter.

## 3. Platform Results — Multi-Account Links

**API:** Each item in `platform_results` now includes:
- `account_id` — platform_tokens.id (use for correct token lookup)
- `account_name` — display name
- `account_username` — e.g. @handle
- `url` / `platform_url` — publish link for this specific account

**Frontend:** When rendering platform badges/links:
- Iterate over `platform_results` (one entry per account published to)
- Use `account_username` or `account_name` for the label
- Use `url` for the link
- Do not group by platform only — each result is a distinct account

## 4. Stats Pending

**Backend fixes:**
- Worker analytics sync now includes `succeeded` and `partial` (not just `completed`)
- Fixed SQL bugs in worker analytics loop
- Multi-account: uses `account_id` from platform_results to pick correct token for stats API

**Frontend:** The "Sync" button calls `POST /api/uploads/{upload_id}/sync-analytics`. Stats should now populate correctly. If "Stats pending" persists, the worker may need to run (it syncs every 6h by default).
