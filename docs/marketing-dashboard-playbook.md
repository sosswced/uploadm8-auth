# UploadM8 Marketing Dashboard & Ads Playbook

**Audience:** Founder / master admin  
**Goal:** Run the in-app AI/ML marketing ecosystem automatically; only paste Meta/TikTok ads yourself for acquisition, verification, and budget control.  
**Companion PDF:** `docs/marketing-dashboard-playbook.pdf` (regenerate with `python docs/generate_marketing_playbook_pdf.py`)

---

## 1. Mental model

| Layer | What it is | Who runs it |
|--------|------------|-------------|
| **Paid ads** | Meta / TikTok Ads Manager | **You** — paste creative + UTM URL, set budget |
| **Pixels** | GA4 / Meta / TikTok browser pixels | Config + live site |
| **Acquisition CRM** | UTM signup → paid attribution | Automatic once UTMs land on signup |
| **Marketing Funnel** | In-app nudge shown / click / dismiss / attributed revenue | Automatic from wallet events |
| **Business Need Signals** | Free uploaders, low PUT, low AIC, multi-platform | Automatic from DB |
| **Promo Timing** | Best hours + comms hints | Automatic when enough click events |
| **AI Strategist → Deploy** | Plan + create campaigns from Truth metrics | **You click** Generate / Deploy; then in-app runs |
| **Wallet / in-app nudges** | Offers to logged-in users | **Automatic** |
| **Path A** | Rule-based email/Discord to qualifying users | Env + **Enable outbound** gate |
| **Path B** | ML propensity campaigns (email/Discord/mixed) | Deploy/apply → approve → **execution tick** |

**Funnel CTR ≠ ad CTR.** Funnel is **wallet nudge** performance inside the app. Ads feed **Acquisition CRM** via UTMs. Pixels optimize **ad platforms**; they do not write UploadM8 `marketing_events`.

---

## 2. What you do vs what runs itself

| You (manual) | Automated / almost automated |
|--------------|------------------------------|
| Paste ads + set budget in Ads Manager | Wallet / in-app nudges for logged-in users |
| Copy UTM landing per flight | AI Strategist Generate → Deploy In-App |
| Optionally Enable Path A outbound later | Path A loop when env + gate on |
| Approve + execution tick for email/Discord Path B | Truth Dashboard / audit after Generate/Deploy |
| Tinker homepage / hero creative | Business Need Signals + Funnel refresh from live data |

---

## 3. Prerequisites (do once)

### 3.1 Access

- Log into production as **admin / master admin**
- Open `admin-marketing.html` on production
- Hard refresh (Ctrl+Shift+R)

### 3.2 Conversion pixels

Configured in gitignored `frontend/js/conversion-pixels.local.js` (must also exist on the **production** frontend host):

| Platform | ID |
|----------|-----|
| GA4 | `G-ND7HVJC1BL` |
| Meta | `1613855987129945` |
| TikTok | `DAFFIFBC77UEK70HRUG0` |

On admin-marketing → Conversion pixels:

- Confirm fields filled / status pill green
- Confirm same file on production deploy host
- Test on a non-localhost page (Network: GA / fbevents / TikTok)

Pixels stay **off on localhost** unless `allow_localhost: true`.

### 3.3 Server env (verify in prod secrets — names only)

- `MARKETING_AUTOMATION_ENABLED=true` — Path A loop
- `MARKETING_ML_TARGETING_ENABLED=1` — Path B propensity
- Mailgun configured — email Path A/B
- `OPENAI_API_KEY` — optional; improves Strategist copy
- Discord webhook — if using Discord channel

### 3.4 Creative for ads (~15s homepage hero)

| File | Use |
|------|-----|
| `frontend/videos/hero-teaser.mp4` | Homepage hero (~14–15s). **Use this for first ads.** |
| `frontend/videos/hero-demo.mp4` | Longer “how it works” (~72s) when you want a fuller story |
| `frontend/images/hero-final.png` / `hero-poster.png` | Still / poster fallbacks |

You can keep adjusting the homepage. When creative changes, re-export and re-upload in Ads Manager. Attack pack UI may reference `hero-demo.mp4`; for a 15s ad, prefer **`hero-teaser.mp4`**.

Optional: also export a **9:16** crop for TikTok / Reels.

---

## 4. Acquisition Command Center

Open **admin-marketing** → Acquisition Command Center.

### 4.1 Official social kit

1. Save brand email / phone / handle if not locked.
2. For each network: open create flow → paste bio + link-in-bio.
3. Store passwords in 1Password / Bitwarden (not a spreadsheet).
4. Click **Mark live** when the account exists.

### 4.2 Landing URL factory

1. Click preset: **Meta paid** or **TikTok paid**.
2. Keep `utm_campaign=traction_v1` for the first flight.
3. Keep `utm_content=hook_create` for the first ad.
4. **Copy landing URL** → paste as Ads Manager (or bio) destination.
5. When you start a **new flight**, rename `utm_campaign` (e.g. `traction_v2`) so Acquisition CRM can separate cohorts.

**Meta (first flight):**

```text
https://app.uploadm8.com/signup.html?utm_source=meta&utm_medium=paid_social&utm_campaign=traction_v1&utm_content=hook_create
```

**TikTok (first flight):**

```text
https://app.uploadm8.com/signup.html?utm_source=tiktok&utm_medium=paid_social&utm_campaign=traction_v1&utm_content=hook_create
```

### 4.3 Attack pack (manual ads — nothing auto-publishes)

1. Set **Channel** to `meta` (then later `tiktok`).
2. **Copy pack** or **Download CSV**.
3. Paste primary / headline / CTA into Ads Manager.
4. Attach `hero-teaser.mp4` (or `hero-demo.mp4`).
5. Destination URL = matching landing for that hook’s `utm_content`.

---

## 5. How to implement ads

### 5.1 Shared rules

- One channel first (recommend **Meta**), one hook (`hook_create`), **small daily budget**
- Always use a UTM landing from Landing URL factory
- Creative: homepage teaser (~15s) is enough to verify
- Confirm pixel PageView on landing; signup/purchase when users convert

### 5.2 Meta Ads Manager

1. Ads Manager → Create campaign  
2. Objective: **Traffic** (simplest verify) or **Sales/Conversions** once the pixel has events  
3. Budget: start tiny (e.g. $5–15/day) — you control spend **only** here  
4. Audience: broad or “content creators / social media managers” — don’t over-narrow at first  
5. Placements: Advantage+ or Feed + Reels  
6. Ad:
   - Primary text ← Attack pack `primary`
   - Headline ← `headline`
   - CTA ← Sign up / Learn more
   - Media ← `hero-teaser.mp4`
   - Website URL ← Meta UTM landing above  
7. Publish → check Events Manager + UploadM8 Acquisition CRM  

### 5.3 TikTok Ads

1. TikTok Ads → Campaign → Traffic or Conversions  
2. Small daily budget  
3. Ad group: interest / broad  
4. Ad: upload teaser (prefer 9:16), paste Attack pack copy  
5. Destination = TikTok UTM URL  
6. Confirm TikTok Pixel receives events  

### 5.4 Verify ads worked

- Click your ad (or preview) → land on `signup.html` with UTMs in the address bar  
- Test or real signup → user has `utm_source` / `utm_campaign`  
- Acquisition CRM shows UTM signups after intel refresh  
- Pixel platforms show activity (may lag)  

---

## 6. Marketing dashboard panels

These are mostly **readouts**. They fill when users and events exist.

| Panel | How it gets data | What you do |
|--------|------------------|---------------|
| **Marketing Funnel** | Wallet nudge shown/clicked/dismissed + attributed revenue | Get users in app; Deploy In-App |
| **Business Need Signals** | Free uploaders 7d, low PUT, low AIC, 3+ platforms | Organic use + paid signups using product |
| **Promo Timing** | Click-hour histogram | Wait until “not enough signals” clears |
| **AI Strategist** | Truth metrics + OpenAI (optional) | Generate → Deploy |
| **Truth Dashboard** | Last AI decision / audit | Appears after Generate/Deploy |
| **Acquisition CRM** | `users.utm_*` + revenue | Ads with UTM landings |
| **Path A gate** | Master toggle + env | Enable only after landings work |
| **Automation outbound log** | Path A sends | After Enable outbound |

### 6.1 AI Strategist — recommended first settings

- Objective: **Revenue Growth**
- Tone: **Executive Clear**
- Offer Style: **Value First**
- Channel Mix: **In-App (wallet nudges)** ← fastest automatic path
- Force Deploy: **No** (unless confidence stuck under ~55)
- Allow PII: **No (sanitized)**

**Steps:**

1. **Generate AI Plan**
2. Read the plan output
3. If confidence ≥ ~55 → **Deploy AI Campaign**
4. In-app / discount campaigns can go **active** without email approval
5. Check **Truth Dashboard** for the decision record
6. Target users should see wallet nudges → Funnel fills

### 6.2 Path A (email / Discord) — later

1. Leave **Disabled** until ads/bios produce real landings  
2. Confirm prod `MARKETING_AUTOMATION_ENABLED=1`  
3. Master admin → **Enable outbound**  
4. Watch Automation outbound log (`sent` / `dedupe_skip` / `failed`)  
5. **Disable** anytime  

### 6.3 Path B (ML email / Discord)

1. Generate / Apply to Campaign Builder / save templates  
2. Master **approve** (approval ticket)  
3. **Run execution tick** (or tick-after-approve) — not fully hands-off  
4. In-app Path B needs Set Active only  

---

## 7. Minimal ops to drive its own ecosystem

1. Keep posting **small-budget ads** with UTM landings (`traction_v1` → bump name per flight) so Acquisition CRM attributes signups. Pixels already set.  
2. Use the product / get users into the app so Free Uploaders / Low PUT / Low AIC update and wallet events fill Marketing Funnel.  
3. AI Strategist: **Generate** → **Deploy In-App** when confidence is OK.  
4. Leave **Path A Disabled** until landings work; then Enable outbound only if you want email/Discord.  
5. Do **not** expect Funnel/Strategist to place Meta ads — Attack pack is copy-to-paste only.  
6. Tinker the homepage freely; re-upload teaser in Ads Manager when creative changes.  

---

## 8. Definition of done

### Acquisition (ads)

- [ ] At least one Meta **or** TikTok ad live with UTM landing  
- [ ] Budget capped in Ads Manager  
- [ ] Pixel events visible in platform  
- [ ] ≥1 UTM signup visible in Acquisition CRM  

### Product marketing loop

- [ ] Business Need Signals updating over time  
- [ ] Marketing Funnel showing shown/clicks after Deploy In-App  
- [ ] At least one AI Generate + Deploy in Truth Dashboard  
- [ ] Wallet nudges visible when logged in as a target user  

### Optional outbound

- [ ] Path A still Disabled **or** Enabled with a clean outbound log  
- [ ] Path B only if you intentionally approve email campaigns  

### Creative

- [ ] Ads using `hero-teaser.mp4` (~15s) or successor  
- [ ] Homepage edits don’t block ads (re-upload when ready)  

---

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Pixels not firing | local.js missing on prod; or localhost | Deploy local.js; don’t test on localhost |
| Acquisition CRM empty | Ads without UTMs / wrong domain | Must use `signup.html?utm_...` |
| Funnel all zeros | No wallet nudge events | Deploy In-App; use app as user |
| Promo Timing “not enough signals” | Sparse clicks | Wait; drive more in-app engagement |
| Deploy blocked | Confidence &lt; 55 | Force Deploy Yes once, or wait for more data |
| Path A silent | Gate Disabled or env off | Enable gate + confirm prod env |
| Email Path B not sending | Not approved / no tick | Approve + Run execution tick |

---

## 10. First-week schedule

| Day | Focus |
|-----|--------|
| 1 | Confirm pixels on prod; copy Meta landing; launch 1 Meta ad with teaser + `hook_create` |
| 2 | Mark 1–2 socials live; Generate + Deploy In-App Strategist |
| 3 | Check Acquisition CRM + Funnel; adjust budget only |
| 4 | Optional second hook or TikTok with same teaser |
| 5 | Review Truth Dashboard; tweak homepage; re-export teaser if needed |
| 6–7 | Only then consider Path A Enable outbound |

---

## 11. One-page cheat sheet

```text
ADS:      Attack pack + hero-teaser.mp4 + UTM URL → Ads Manager (you)
ATTR:     UTMs → Acquisition CRM | Pixels → Meta/GA4/TikTok
PRODUCT:  Users → signals + wallet events → Funnel
AI:       Generate → Deploy In-App → nudges auto
OUTBOUND: Path A gate later | Path B approve + tick
NEVER:    Expect Strategist to publish Meta/TikTok ads
```

---

## 12. Related repo paths

| Path | Role |
|------|------|
| `frontend/admin-marketing.html` | Marketing dashboard UI |
| `frontend/js/acquisition-command-center.js` | UTM factory, attack pack, readiness |
| `frontend/js/conversion-pixels.js` + `.local.js` | Pixel loader + IDs |
| `frontend/videos/hero-teaser.mp4` | Homepage / first ad creative |
| `frontend/videos/README.md` | Hero asset contract |
| `services/marketing_touchpoint_runner.py` | Path A |
| `services/marketing_execution.py` | Path B |
| `services/growth_intelligence.py` | Funnel / signals / UTM intel |
| `docs/marketing-rollout-hooks.md` | Organic social copy hooks |
