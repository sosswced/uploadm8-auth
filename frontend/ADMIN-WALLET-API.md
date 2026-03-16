# Admin Wallet Manager — Required Backend API

The **admin-wallet.html** page requires these endpoints. A 500 Internal Server Error usually means one or more are missing or failing.

## Endpoints

### 1. User search (admin)

```
GET /api/admin/users?search={query}&limit=8
```

**Auth:** Bearer token, admin role required.

**Response:**
```json
{
  "users": [
    {
      "id": "uuid-or-int",
      "email": "user@example.com",
      "name": "User Name",
      "subscription_tier": "free"
    }
  ]
}
```

---

### 2. Get user wallet

```
GET /api/admin/users/{user_id}/wallet
```

**Auth:** Bearer token, admin role required.

**Response:**
```json
{
  "wallet": {
    "put_balance": 100,
    "aic_balance": 50,
    "put_reserved": 0,
    "aic_reserved": 0
  },
  "plan_limits": {
    "put_monthly": 200,
    "aic_monthly": 100
  },
  "ledger": [
    {
      "id": "...",
      "wallet": "put",
      "delta": 10,
      "reason": "Admin top-up",
      "created_at": "2025-03-11T12:00:00Z"
    }
  ]
}
```

---

### 3. Adjust wallet

```
POST /api/admin/users/{user_id}/wallet/adjust
Content-Type: application/json

{
  "wallet": "put" | "aic",
  "mode": "add" | "subtract" | "set",
  "amount": 50,
  "reason": "Support refund"
}
```

**Auth:** Bearer token, admin role required.

**Response:**
```json
{
  "before": 100,
  "after": 150,
  "delta": 50
}
```

---

## Implementation notes

- `user_id` can be UUID or integer; ensure the route accepts it.
- All three endpoints must enforce admin (or master_admin) role.
- The wallet/ledger tables must exist; missing schema often causes 500s.
- Check backend logs at auth.uploadm8.com for the actual exception.
