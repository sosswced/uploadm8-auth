# UploadM8 — Features, Services & User Benefits

**Last reviewed:** May 20, 2026  
**Source of truth for limits:** `stages/entitlements.py` (tiers, caps) · worker pipeline in `worker.py`

Use this doc when updating product/feature claims and tier limits. For **page copy, hooks, and SEO/meta tags**, use **`docs/copy-seo-spec.md`** as the marketing source of truth (implemented in `frontend/js/marketing-copy.js` and `frontend/js/app-shell-copy.js`). Pair with `docs/marketing-rollout-hooks.md` for short social hooks.

---

## A. Final feature & service list

### Core product (all tiers, with plan limits)

| Feature | What it is | Stand-out hook |
|--------|------------|----------------|
| **One upload → four platforms** | Single master video → TikTok, YouTube, Instagram, Facebook with per-platform transcode & publish | *4 Platforms, 1 Upload, 0 Chaos* |
| **AI packaging (M8)** | Multi-frame vision + LLM titles, captions, per-platform hashtags | *Hooks that match what’s actually in the clip* |
| **Thumbnail pipeline** | Frame extract, sharpness scoring, auto-pick; optional manual override | *Stop guessing which frame wins the click* |
| **Thumbnail Studio** | Prompt, recreate, score thumbnails in-app (AI credits) | *Thumbnails without leaving your publish workflow* |
| **Smart + manual scheduling** | Peak-window suggestions or exact time; deferred processing before go-live | *Batch Sunday, ship all week—without early publishes* |
| **Queue & timeline** | Live status, cancel, per-platform results, resume-friendly uploads | *A queue your client can actually read* |
| **Publishing credits (PUT)** | Pay for transcode, multi-destination publish, priority, extra thumbs | *Credits move only when work runs* |
| **AI credits (AIC)** | Captions, hashtags, thumbnail AI, deep frame scans | *AI spend you can explain on an invoice* |
| **Connected accounts** | OAuth to each platform; caps by total + per-platform | *Many brands, one login—no password vault* |
| **Client / brand groups** | Route one upload to a preset set of accounts | *Friday drops to every client page in one click* |
| **Parallel batch upload** | Upload.html parallel mode capped by tier | *Weekend backlog without serial torture* |
| **Speech-to-text** | Whisper transcription feeds captions (all plans when enabled) | *Voice-led videos get captions that heard you* |
| **Webhooks / Discord** | Completion alerts to your stack | *Alerts where your team already lives* |
| **Analytics** | Cross-platform performance signals; export on Studio+ | *One dashboard, not four native apps* |

### Tier-gated power features

| Feature | Available from | User benefit |
|--------|----------------|--------------|
| **Watermark removed** | Creator Lite+ | Brand-clean exports clients won’t flinch at |
| **Priority / turbo lanes** | Pro (p2), Studio (p1), Agency (p0) | Busy days still ship—your jobs aren’t stuck behind free tier |
| **AI thumbnail styling** | Creator Pro+ | On-brand composite thumbs, not generic frames |
| **Excel export** | Studio+ | Client reporting without copy-paste marathons |
| **White-label** | Agency | Client-facing surfaces without UploadM8 branding |
| **Flex transfers** | Agency | Move credits between brands when one client spikes |
| **Extended scheduling window** | Up to 168h (Agency) | Pre-process up to a week early for agency calendars |
| **Dense AI scan** | Up to 15 caption frames / 20 thumbnails (Studio/Agency) | Packaging stays sharp at high volume |

### Drive / Trill (optional niche)

| Feature | What it is | Benefit |
|--------|------------|---------|
| **Telemetry + Trill** | `.map` lap data, scoring, leaderboards | Map-aware captions without re-editing |
| **Dashcam OSD Reader** | OCR of speed/GPS text already on the clip | Backfill route context when no `.map`—no new overlay burned in |

### Platforms & billing services

- **Stripe subscriptions:** Starter (free), Creator Lite $12, Creator Pro $29, Studio $79, Agency $199 (+ annual)
- **Add-on credit packs:** PUT, AIC, bundles — never expire
- **7-day trial:** Paid tiers at checkout
- **Enterprise:** Custom via contact

---

## B. Benefits by audience

| Audience | Primary pain | UploadM8 outcome |
|----------|--------------|------------------|
| **Solo creator** | Re-upload fatigue & weak metadata | One session → four lives; AI writes the boring parts |
| **Serious creator / small team** | Batch weekends + brand polish | Pro lane, seats, deeper AI |
| **Studio / production shop** | Throughput + client proof | Turbo queue, exports, deep queue depth |
| **Agency** | Many clients, many accounts | Groups, white-label, flex, dedicated lane |
| **Automotive creator** | Telemetry story + multi-platform | Trill + dashcam OSD + same ops stack as everyone else |

---

## C. Killer hooks (SEO + social + hero)

1. **You focus on creating. We handle uploads, timing, captions, and thumbnails.** — primary hero  
2. **Create the videos. We’ll do the rest.** — promise / CTA  
3. **Mass content in. Accurate AI out. Stay consistent.** — volume + consistency  
4. **Struggle with consistency?** — empathy hook; we remove manual upload + timing  
5. **4 Platforms, 1 Upload, 0 Chaos** — secondary brand / OG  
6. **Calendars schedule posts. UploadM8 packages the video.** — soft compare  
7. **Publishing credits for goes-live. AI credits for smart extras. Both honest.** — trust  
8. **Batch the week on Sunday. Ship on time every day.** — scheduling  
9. **Thumbnail Studio: prompt, recreate, score—then publish.** — product-led  
10. **Agency Friday: one upload, every client page, one queue.** — B2B  
11. **Your lap on screen. Your clip on four feeds.** — automotive  
12. **Start free. No card. See the full loop.** — conversion  

---

## D. Pages to keep aligned

| Page | Role |
|------|------|
| `frontend/index.html` | Hero, feature grid, pricing narratives |
| `frontend/guide.html` | Deep feature reference |
| `frontend/how-it-works.html` | Short funnel / SEO |
| `frontend/settings.html#billing` | In-app plan bullets |
| `frontend/js/marketing-copy.js` | Shared tier narratives & perks |
| SEO landers (`ai-social-media-scheduler.html`, etc.) | Keyword entry points |
| `docs/marketing-rollout-hooks.md` | Social / OG snippets |
