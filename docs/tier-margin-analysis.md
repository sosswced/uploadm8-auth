# UploadM8 tier margin analysis (v1)

## Assumptions

- Typical upload: 60s video, 1 platform, caption + thumbnail pipeline.
- AIC burn: ~12–45 credits depending on tier `ai_depth` and enabled smart services.
- PUT cost: 1 base + duration multiplier (see `stages/entitlements.compute_put_cost`).
- Target gross margin on paid tiers: ~70% before infra/support.

## Estimated COGS per typical upload (60s)

| Tier | PUT | AIC (typical) | Est. COGS @ $0.002/AIC, $0.05/PUT |
|------|-----|---------------|-------------------------------------|
| free | 1 | 8–12 | ~$0.07 |
| creator_lite | 1 | 15–20 | ~$0.09 |
| creator_pro | 1 | 25–35 | ~$0.12 |
| studio | 1 | 40–55 | ~$0.16 |
| agency | 1 | 55–80 | ~$0.21 |

## Margin at list price (monthly, moderate usage)

| Tier | Price | PUT/mo | AIC/mo | ~Max uploads/mo @ 1 PUT + 20 AIC | Est. COGS cap | Gross margin |
|------|-------|--------|--------|-----------------------------------|---------------|--------------|
| free | $0 | 100 | 80 | ~4 full-smart | $0 | N/A |
| creator_lite | $12 | 600 | 200 | ~10 | ~$1.20 | ~90% |
| creator_pro | $29 | 2000 | 600 | ~28 | ~$3.50 | ~88% |
| studio | $79 | 6000 | 2000 | ~85 | ~$12 | ~85% |
| agency | $199 | high | high | 200+ | ~$35 | ~82% |

Paid tiers have headroom; free tier is acquisition cost.

## Recommendations applied in `TIER_CONFIG`

1. **creator_lite**: Slightly raise `aic_monthly` 200 → 220 for watermark-free positioning without margin risk.
2. **creator_pro**: Keep PUT/AIC; add clarity via marketing copy (not config).
3. **studio**: Raise `put_monthly` 6000 → 6500 for turbo positioning.
4. **free**: Keep limits; ensure smart depth stays `basic` to cap COGS.

## Follow-up

- Reconcile with live ledger burn from `wallet_ledger` monthly aggregates.
- Sync Stripe catalog after config changes via `scripts/sync_stripe_catalog.py`.
