# UploadM8 — Copy, Hooks & SEO Spec

**Purpose:** Drop-in marketing copy, hero hooks, and SEO/meta tags for every page. Hand this to your implementation AI. Each page block tells you exactly what goes where.

**How to read this doc:**
- **PUBLIC** pages get the full SEO suite (title tag, meta description, OG/Twitter tags, keywords). These are crawlable and are your acquisition surface.
- **APP** pages are behind login. They get a browser `<title>`, an in-app header/subhead, and microcopy. They must be **`<meta name="robots" content="noindex, nofollow">`** — never let Google index dashboards, admin tools, or wallet pages.
- Anything in `code font` is meant to be used literally. Everything else is a menu — pick what fits.

**Repo mapping (no new HTML files):**
- Feature Guide → `frontend/guide.html`
- Setup Handbook → `frontend/guide.html#feat-settings-playbook`
- Connected Accounts → `frontend/platforms.html`
- Discord → external invite link (no `discord.html`)

**Implementation:** `frontend/js/marketing-copy.js` (public) · `frontend/js/app-shell-copy.js` (app/admin)

---

## 0. Brand Foundation

**Product:** UploadM8 — upload a video once, publish everywhere (TikTok, YouTube Shorts, Instagram Reels, Facebook Reels) with AI captions, AI thumbnails, and automatic scheduling.

**Voice:** Confident, fast, creator-to-creator. Short punchy lines. Benefit first, mechanics second.

**The promise:** You create. We'll do the rest.

**Locked canonical lines** (verbatim):
- `Post everywhere. Stay consistent. Just keep creating.`
- `From one video to four platforms — captioned, thumbnailed, and scheduled. Automatically.`
- `You create. We'll do the rest.`
- `Upload once. UploadM8 publishes to TikTok, YouTube, Instagram, and Facebook with AI captions, thumbnails, and the consistent posting schedule you've been struggling to keep.`

---

# PART 1 — PUBLIC PAGES

## index.html

**Hero (launch):** Eyebrow `One video. Four platforms. Zero busywork.` · H1 `Post everywhere. Stay consistent. Just keep creating.` · Subhead (hero block) · CTAs `Start free` + `See how it works`

**Hero subhead:** Upload once and UploadM8 publishes to TikTok, YouTube, Instagram, and Facebook — with accurate AI captions, auto-generated thumbnails, and the consistent posting schedule you've been struggling to keep.

**Section headers:** Everything happens after you hit upload · Captions that actually heard your video · Scroll-stopping thumbnails · It posts at the right time · Trill leaderboard · Built for creators who'd rather create · Final CTA: Make the video. We'll handle the rest.

**SEO:** Title `UploadM8 — Upload Once, Post to TikTok, YouTube, Instagram & Facebook` · OG `Upload once. Post everywhere.` · JSON-LD `MultimediaApplication`

## guide.html (Feature Guide)

- H1: `Everything UploadM8 does for you`
- Subhead: A walkthrough of captions, thumbnails, scheduling, multi-account publishing, and the Trill leaderboard — what each feature does and when to use it.
- Title: `Feature Guide — UploadM8`
- Public SEO, indexed (no noindex)

## guide.html#feat-settings-playbook (Setup Handbook)

- H2: `Get set up in 5 minutes`
- Subhead: Connect your accounts, upload your first video, and let UploadM8 take it from there.
- Steps: `1. Connect your platforms` · `2. Upload your first video` · `3. Set your schedule` · `4. Let it run`

---

# PART 2 — APP PAGES (noindex)

See `frontend/js/app-shell-copy.js` for per-page titles, headers, subheads, empty states, and microcopy.

| Page | Title | Header | Subhead |
|------|-------|--------|---------|
| dashboard | Dashboard — UploadM8 | (keep welcome personalization) | Your uploads, schedule, and platform health at a glance. |
| upload | Upload — UploadM8 | Upload once. Post everywhere. | Drop your video, pick your platforms… |
| thumbnail-studio | Thumbnail Studio — UploadM8 | Thumbnails that stop the scroll | Generate, tweak, and lock in… |
| queue | Queue — UploadM8 | What's processing right now | Live status for every upload… |
| scheduled | Scheduled — UploadM8 | Your posting calendar, handled | Everything lined up to publish… |
| platforms | Platforms — UploadM8 | Your platforms, all in one place | TikTok, YouTube, Instagram, Facebook… |
| groups | Account Groups — UploadM8 | Group accounts, post in bulk | Bundle accounts into one-tap targets… |
| analytics | Analytics — UploadM8 | Your numbers, all four platforms | Views, engagement, and growth… |
| kpi | Upload KPIs — UploadM8 | Upload performance & throughput | Success rates, processing times… |
| settings | Settings — UploadM8 | Settings | Tune captions, scheduling, hashtags… |

---

# PART 3 — ADMIN (noindex)

Operator titles and one-line headers in `app-shell-copy.js` — Admin Panel, User Management, Platform KPIs, Marketing Ops, ML Observability, Incidents, AI Trace, etc.

---

# PART 4 — Reusable Snippets

**Taglines:** You create. We'll do the rest. · Just keep creating. · One upload. Everywhere. · We handle the busywork.

**Loading:** Loading your studio… · Lining up your platforms… · Almost there — your feeds await.

**Default OG:** UploadM8 — Upload Once, Post Everywhere · Auto-publish one video to TikTok, YouTube, Instagram, and Facebook with AI captions, thumbnails, and scheduling.

**Global:** `theme-color` `#f97316` · `og:site_name` UploadM8 · `twitter:card` summary_large_image · canonical on public pages · noindex on app/admin.

---

## Implementation notes

1. **Crawlable:** index, guide, acquisition pages (signup, how-it-works, landers, compare, contact, legal). Everything else `noindex, nofollow`.
2. **Use locked lines verbatim** where specified.
3. **One H1 per page.** Trill is a feature card on index, not the hero lead.
4. **No keyword-stuffing** on app pages.
