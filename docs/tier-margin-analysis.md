# UploadM8 tier margin analysis (v2 — Jul 2026)

## Anchor (includes profit)

| Unit | Meaning |
|------|---------|
| **1 AIC** | ≈ **$0.01** intended **retail** wallet debit (not raw vendor $) |
| **1 PUT** | Publish/process unit; **base job = 10 PUT** (do not drop to 1) |
| Target AI retail | **$0.15–$0.25** wallet debit for a **60s full-smart** upload |
| Pikzels markup | Wallet debit ≥ **2.5×** vendor avg → ~**60% gross** on Studio calls after vendor $ |

**Profit room:** Pikzels is explicit 2.5× COGS. Upload AIC weights are sized to the *retail* target ($0.01/AIC), not break-even vendor $ — so when OpenAI/GCP/TL invoices rise, you still have headroom vs top-up list price (~$0.008–0.02/AIC). Subscriptions cover fixed upkeep (Render seat, Redis); tokens are overage + margin.

## Live vendor spend (statements + consoles)

| Source | Period | Amount | Notes |
|--------|--------|--------|-------|
| **Render** (`unbilled-charges.csv`) | Jul 2026 MTD (~224h) | **~$41.35** | API $2.11 + worker $7.54 + DB $2.70 + **team seat $29** |
| **Upstash Redis** | Jun 2026 | **$16.16** | 7.9M commands ($15.72) + storage $0.44 |
| **Google Cloud** (inv 5624832401) | Jun 2026 | **$0.42** | Vision/VI barely used at current volume |
| **Pikzels** | Jan 31–Jul 9 2026 | **$30.91** | Spike Apr–May; see unit table below |

**Fixed/upkeep dominates at current scale** (Render team seat + Redis). Variable AI risk is concentrated in **Pikzels** when Studio is used, not GCP.

### Pikzels unit costs (usage console)

| Endpoint | Avg vendor $ | 2.5× target retail | AIC @ $0.01 |
|----------|-------------:|-------------------:|------------:|
| Image to Thumbnail (PKZ-4) | $0.36 | $0.90 | **90** |
| Text to Thumbnail (PKZ-4) | $0.37 | $0.93 | **90–93** |
| Create Persona | $0.38 | $0.95 | **95** |
| Faceswap Thumbnail | $0.37 | $0.93 | **93** |
| Edit Thumbnail (PKZ-4.5) | $0.12 | $0.30 | **30** |
| Score Thumbnail | $0.03 | $0.08 | **8** |

## PUT formula (unchanged)

```
base 10
+ 2 × max(0, publish_targets − 1)
+ 5 if priority lane
+ 1 × max(0, thumbnails − 1)
```

Example: 1 platform, no priority, 1 thumb → **10 PUT** (~$0.10–0.20 of top-up value). PUT is an **abuse/rate brake**, not the primary profit meter.

## AIC — upload pipeline (recalibrated weights)

Default light job (captions + thumbs + Vision + light audio; **no** Whisper / Twelve Labs / VI / dashcam):

| Service | Weight |
|---------|-------:|
| caption_llm | 4 |
| thumbnail_ai | 3 |
| vision_google | 2 |
| audio_gpt_classify | 1 |
| audio_yamnet | 1 |
| trend_intel | 1 |
| **Typical total** | **~12–15 AIC** (≈ $0.12–0.15 retail) |

Full-smart 60s (all services, duration_mult ≈ 1.0):

| Extra (opt-in) | Weight |
|----------------|-------:|
| twelvelabs | 4 (× duration) |
| video_intelligence | 3 (× duration) |
| audio_whisper | 3 (× duration) |
| dashcam_osd | 1 |
| audio_acr | 1 |
| telemetry_trill | 1 |
| **Typical total** | **~22–25 AIC** (≈ $0.22–0.25 retail) |

Defaults ship **lighter than max** — biggest free margin lever. Users opt into minute-metered APIs in Settings.

## AIC — Thumbnail Studio / Pikzels

Debits in `estimate_pikzels_v2_call_cost` / `estimate_studio_cost` match the 2.5× table above (e.g. recreate **90 AIC**, persona **95 AIC**, edit **30 AIC**).

## Subscription allotments vs burn

| Tier | Price | PUT/mo | AIC/mo | ~Default-light uploads (10 PUT + 14 AIC) | ~Full-smart (10 PUT + 24 AIC) |
|------|------:|-------:|-------:|------------------------------------------:|------------------------------:|
| free | $0 | 100 | 80 | ~7 | ~4 |
| creator_lite | $12 | 600 | 220 | ~15 | ~9 |
| creator_pro | $29 | 2000 | 600 | ~42 | ~25 |
| studio | $79 | 6500 | 2000 | ~140 | ~83 |
| agency | $199 | 20000 | 7000 | ~500 | ~290 |

Paid tiers still have headroom if defaults stay light. **Studio/Pikzels** can burn a month of AIC in a handful of persona+recreate runs — intentional; top-ups recover COGS.

## Margin model (layers)

1. **Variable AI (AIC + Pikzels)** — recover via AIC burn + top-ups; target ≥70% contribution after vendor $.
2. **Semi-variable infra (PUT)** — R2, ffmpeg, worker minutes, Redis commands; soft recovery.
3. **Fixed upkeep** — Render seats, DB, Upstash baseline — paid by **subscription MRR**.

## Follow-up

- **Go-live for upload AIC weights (DB — live at presign):**

  | Mode | When | How |
  |------|------|-----|
  | Soft (default) | Deploy / API restart | Boot runs `migrate_legacy_weights_to_code_defaults` — only rows still on old seeds (28/24/22…) |
  | Hard (UI) | After deploy if drift remains | Master Admin → Debit weights → **Reset AIC to code defaults** |
  | Hard (API) | Scripted | `POST /api/admin/billing/service-weights/reset-to-code` (master admin auth) |
  | Hard (env, one boot) | Force every row to code | Set `BILLING_WEIGHTS_FORCE_SYNC_FROM_CODE=1`, restart API once, then unset |

- **Automate before frontend /333 push:**
  ```powershell
  python scripts/pre_ship_pricing.py                  # surfaces always
  python scripts/pre_ship_pricing.py --force-db-weights  # + DB upsert when DATABASE_URL set
  ```
  Wired into `/333` (step 1b) and `uploadm8-frontend-push` (step 0). Unit tests still auto-check surface drift.
- **Stripe catalog / tier allotments:** separate. Run only when plan prices, monthly PUT/AIC allowances, or product cards change:
  ```powershell
  python -m scripts.sync_stripe_catalog --all
  ```
  Admin catalog edits already `auto_sync` to Stripe by default (`docs/catalog-sync.md`). Do **not** auto-run full Stripe sync on every deploy — it creates/migrates Stripe Prices.
- Reconcile monthly with `token_ledger` burn × OpenAI/TwelveLabs invoices when those statements are available.
