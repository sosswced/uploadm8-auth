#!/usr/bin/env python3
"""Generate UploadM8 Marketing Dashboard & Ads Playbook PDF from the study guide."""

from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "marketing-dashboard-playbook.pdf"


class Doc(FPDF):
    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 100, 100)
        self.cell(
            0,
            8,
            f"UploadM8 Marketing Dashboard Playbook  |  Page {self.page_no()}/{{nb}}",
            align="C",
        )

    def _reset_x(self):
        self.set_x(self.l_margin)

    def h1(self, text):
        self._reset_x()
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(20, 20, 20)
        self.multi_cell(0, 8, text)
        self.ln(2)
        self._reset_x()

    def h2(self, text):
        self.ln(3)
        self._reset_x()
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 7, text)
        self.ln(1)
        self._reset_x()

    def h3(self, text):
        self.ln(1)
        self._reset_x()
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 6, text)
        self.ln(0.5)
        self._reset_x()

    def body(self, text):
        self._reset_x()
        self.set_font("Helvetica", "", 9)
        self.set_text_color(25, 25, 25)
        self.multi_cell(0, 5, text)
        self.ln(1)
        self._reset_x()

    def bullet(self, text):
        self._reset_x()
        self.set_font("Helvetica", "", 9)
        self.set_text_color(25, 25, 25)
        self.multi_cell(0, 5, f"- {text}")
        self._reset_x()

    def check(self, text):
        self._reset_x()
        self.set_font("Helvetica", "", 9)
        self.set_text_color(25, 25, 25)
        self.multi_cell(0, 5, f"[ ] {text}")
        self._reset_x()

    def callout(self, title, text):
        self._reset_x()
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(120, 70, 0)
        self.multi_cell(0, 5, title)
        self._reset_x()
        self.set_font("Helvetica", "", 9)
        self.set_text_color(25, 25, 25)
        self.multi_cell(0, 5, text)
        self.ln(2)
        self._reset_x()

    def paste(self, text):
        self._reset_x()
        self.set_fill_color(245, 245, 245)
        self.set_font("Courier", "", 8)
        self.set_text_color(20, 20, 20)
        self.multi_cell(0, 4.2, text, fill=True)
        self.ln(2)
        self._reset_x()

    def kv(self, key, value):
        self._reset_x()
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(25, 25, 25)
        self.write(5, f"{key}: ")
        self.set_font("Helvetica", "", 9)
        self.multi_cell(0, 5, value)
        self._reset_x()


def build():
    pdf = Doc(format="Letter")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.set_margins(16, 16, 16)
    pdf.add_page()

    pdf.h1("UploadM8 Marketing Dashboard & Ads Playbook")
    pdf.body(
        "Audience: Founder / master admin. Goal: run the in-app AI/ML marketing "
        "ecosystem automatically; only paste Meta/TikTok ads yourself for acquisition, "
        "verification, and budget control."
    )
    pdf.body("Source markdown: docs/marketing-dashboard-playbook.md")

    pdf.h2("1. Mental model")
    pdf.bullet("Paid ads (Meta/TikTok)  -  YOU paste creative + UTM URL and set budget")
    pdf.bullet("Pixels (GA4/Meta/TikTok)  -  config on live site; optimize ad platforms")
    pdf.bullet("Acquisition CRM  -  UTM signup to paid attribution (automatic)")
    pdf.bullet("Marketing Funnel  -  in-app nudge shown/click/dismiss (NOT ad CTR)")
    pdf.bullet("Business Need Signals  -  free uploaders, low PUT/AIC, multi-platform")
    pdf.bullet("AI Strategist Generate/Deploy  -  you click; then in-app campaigns run")
    pdf.bullet("Wallet nudges  -  automatic for logged-in users")
    pdf.bullet("Path A  -  rule-based email/Discord when env + Enable outbound")
    pdf.bullet("Path B  -  ML campaigns need approve + execution tick")
    pdf.callout(
        "Remember",
        "Funnel CTR is wallet nudge CTR inside the product. Ads feed Acquisition CRM "
        "via UTMs. Pixels do not write UploadM8 marketing_events.",
    )

    pdf.h2("2. You vs automated")
    pdf.h3("You (manual)")
    pdf.bullet("Paste ads + set budget in Ads Manager")
    pdf.bullet("Copy UTM landing per flight")
    pdf.bullet("Optionally Enable Path A outbound later")
    pdf.bullet("Approve + execution tick for email/Discord Path B")
    pdf.bullet("Tinker homepage / hero creative")
    pdf.h3("Automated / almost automated")
    pdf.bullet("Wallet / in-app nudges")
    pdf.bullet("AI Strategist Generate -> Deploy In-App")
    pdf.bullet("Path A loop when env + gate on")
    pdf.bullet("Truth Dashboard audit after Generate/Deploy")
    pdf.bullet("Business Need Signals + Funnel from live data")

    pdf.h2("3. Prerequisites (do once)")
    pdf.h3("3.1 Access")
    pdf.check("Log in as admin / master admin")
    pdf.check("Open production admin-marketing.html")
    pdf.check("Hard refresh (Ctrl+Shift+R)")

    pdf.h3("3.2 Conversion pixels (already configured locally)")
    pdf.kv("GA4", "G-ND7HVJC1BL")
    pdf.kv("Meta", "1613855987129945")
    pdf.kv("TikTok", "DAFFIFBC77UEK70HRUG0")
    pdf.bullet("File: frontend/js/conversion-pixels.local.js (gitignored  -  must exist on prod host)")
    pdf.bullet("Confirm Conversion pixels pill is green on admin-marketing")
    pdf.bullet("Pixels stay OFF on localhost unless allow_localhost is true")

    pdf.h3("3.3 Server env (verify in prod  -  names only)")
    pdf.bullet("MARKETING_AUTOMATION_ENABLED=true  -  Path A loop")
    pdf.bullet("MARKETING_ML_TARGETING_ENABLED=1  -  Path B propensity")
    pdf.bullet("Mailgun configured  -  email Path A/B")
    pdf.bullet("OPENAI_API_KEY optional  -  improves Strategist copy")
    pdf.bullet("Discord webhook if using Discord channel")

    pdf.h3("3.4 Creative (~15s homepage hero)")
    pdf.bullet("frontend/videos/hero-teaser.mp4  -  homepage (~14-15s). USE THIS FOR FIRST ADS.")
    pdf.bullet("frontend/videos/hero-demo.mp4  -  longer how-it-works (~72s) later")
    pdf.bullet("images/hero-final.png / hero-poster.png  -  still/poster fallbacks")
    pdf.bullet("Optional: 9:16 crop for TikTok/Reels")
    pdf.body(
        "Tinker the homepage anytime. When creative changes, re-export and re-upload "
        "in Ads Manager. Prefer hero-teaser.mp4 for a short ad."
    )

    pdf.add_page()
    pdf.h2("4. Acquisition Command Center")
    pdf.h3("4.1 Official social kit")
    pdf.bullet("Save brand email / phone / handle if not locked")
    pdf.bullet("Open create flow per network; paste bio + link-in-bio")
    pdf.bullet("Passwords in 1Password/Bitwarden  -  not a spreadsheet")
    pdf.bullet("Mark live when the account exists")

    pdf.h3("4.2 Landing URL factory")
    pdf.bullet("Click Meta paid or TikTok paid preset")
    pdf.bullet("First flight: utm_campaign=traction_v1, utm_content=hook_create")
    pdf.bullet("Copy landing URL -> paste as Ads Manager destination")
    pdf.bullet("New flight: rename utm_campaign (e.g. traction_v2) for CRM attribution")
    pdf.body("Meta first flight:")
    pdf.paste(
        "https://app.uploadm8.com/signup.html?utm_source=meta"
        "&utm_medium=paid_social&utm_campaign=traction_v1&utm_content=hook_create"
    )
    pdf.body("TikTok first flight:")
    pdf.paste(
        "https://app.uploadm8.com/signup.html?utm_source=tiktok"
        "&utm_medium=paid_social&utm_campaign=traction_v1&utm_content=hook_create"
    )

    pdf.h3("4.3 Attack pack (nothing auto-publishes)")
    pdf.bullet("Channel = meta (then tiktok)")
    pdf.bullet("Copy pack or Download CSV")
    pdf.bullet("Paste primary/headline/CTA into Ads Manager")
    pdf.bullet("Attach hero-teaser.mp4; destination = matching hook landing URL")

    pdf.h2("5. How to implement ads")
    pdf.h3("5.1 Shared rules")
    pdf.bullet("One channel first (Meta), one hook (hook_create), small daily budget")
    pdf.bullet("Always use UTM landing from Landing URL factory")
    pdf.bullet("Confirm pixel PageView; signup/purchase when users convert")

    pdf.h3("5.2 Meta Ads Manager")
    pdf.bullet("Create campaign  -  Traffic (verify) or Sales/Conversions later")
    pdf.bullet("Budget $5-15/day to start  -  spend control lives ONLY here")
    pdf.bullet("Audience: broad or creators / social managers")
    pdf.bullet("Placements: Advantage+ or Feed + Reels")
    pdf.bullet("Ad: Attack pack copy + hero-teaser.mp4 + Meta UTM URL")
    pdf.bullet("Publish -> Events Manager + Acquisition CRM")

    pdf.h3("5.3 TikTok Ads")
    pdf.bullet("Traffic or Conversions; small budget; broad/interest")
    pdf.bullet("Prefer 9:16 teaser; paste Attack pack copy; TikTok UTM URL")
    pdf.bullet("Confirm TikTok Pixel receives events")

    pdf.h3("5.4 Verify")
    pdf.check("Ad click lands on signup.html with UTMs in the address bar")
    pdf.check("Signup stores utm_source / utm_campaign")
    pdf.check("Acquisition CRM shows UTM signups after intel refresh")
    pdf.check("Pixel platforms show activity (may lag)")

    pdf.add_page()
    pdf.h2("6. Marketing dashboard panels")
    pdf.bullet("Marketing Funnel  -  wallet nudge events; Deploy In-App to feed it")
    pdf.bullet("Business Need Signals  -  DB cohorts; grow with product usage")
    pdf.bullet("Promo Timing  -  needs enough click signals")
    pdf.bullet("AI Strategist  -  Generate then Deploy")
    pdf.bullet("Truth Dashboard  -  appears after Generate/Deploy")
    pdf.bullet("Acquisition CRM  -  needs UTM ads")
    pdf.bullet("Path A gate  -  enable only after landings work")
    pdf.bullet("Automation outbound log  -  after Enable outbound")

    pdf.h3("6.1 AI Strategist  -  first settings")
    pdf.bullet("Objective: Revenue Growth")
    pdf.bullet("Tone: Executive Clear")
    pdf.bullet("Offer Style: Value First")
    pdf.bullet("Channel Mix: In-App (wallet nudges)  -  fastest automatic path")
    pdf.bullet("Force Deploy: No (unless confidence stuck under ~55)")
    pdf.bullet("Allow PII: No (sanitized)")
    pdf.body(
        "Steps: Generate AI Plan -> if confidence >= ~55 Deploy AI Campaign -> "
        "in-app/discount can go active without email approval -> check Truth Dashboard "
        "-> wallet nudges appear -> Funnel fills."
    )

    pdf.h3("6.2 Path A (later)")
    pdf.bullet("Leave Disabled until ads/bios produce real landings")
    pdf.bullet("Confirm MARKETING_AUTOMATION_ENABLED on prod")
    pdf.bullet("Master admin Enable outbound; watch sent/dedupe_skip/failed")
    pdf.bullet("Disable anytime")

    pdf.h3("6.3 Path B (ML email/Discord)")
    pdf.bullet("Apply to Campaign Builder / save templates")
    pdf.bullet("Master approve (approval ticket)")
    pdf.bullet("Run execution tick (or tick-after-approve)")
    pdf.bullet("In-app Path B: Set Active only")

    pdf.h2("7. Minimal ops (own ecosystem)")
    pdf.bullet("Small-budget ads with UTM landings; bump utm_campaign per flight")
    pdf.bullet("Get users into the app so signals + wallet events grow")
    pdf.bullet("Generate -> Deploy In-App when confidence OK")
    pdf.bullet("Path A Disabled until landings work")
    pdf.bullet("Never expect Strategist to publish Meta/TikTok ads")
    pdf.bullet("Homepage edits: re-upload teaser in Ads Manager when ready")

    pdf.h2("8. Definition of done")
    pdf.h3("Acquisition")
    pdf.check("One Meta or TikTok ad live with UTM landing")
    pdf.check("Budget capped in Ads Manager")
    pdf.check("Pixel events visible")
    pdf.check(">=1 UTM signup in Acquisition CRM")
    pdf.h3("Product loop")
    pdf.check("Business Need Signals updating over time")
    pdf.check("Funnel shows shown/clicks after Deploy In-App")
    pdf.check("AI Generate + Deploy in Truth Dashboard")
    pdf.check("Wallet nudges visible as a target user")
    pdf.h3("Creative")
    pdf.check("Ads use hero-teaser.mp4 (~15s) or successor")

    pdf.add_page()
    pdf.h2("9. Troubleshooting")
    pdf.kv("Pixels not firing", "local.js missing on prod or testing on localhost")
    pdf.kv("Acquisition CRM empty", "Ads missing UTMs  -  must use signup.html?utm_...")
    pdf.kv("Funnel all zeros", "No wallet events  -  Deploy In-App; use the app")
    pdf.kv("Promo Timing sparse", "Need more in-app clicks")
    pdf.kv("Deploy blocked", "Confidence < 55  -  Force Deploy once or wait")
    pdf.kv("Path A silent", "Gate Disabled or env off")
    pdf.kv("Path B email silent", "Not approved / no execution tick")

    pdf.h2("10. First-week schedule")
    pdf.bullet("Day 1: Confirm pixels; Meta landing; 1 Meta ad with teaser + hook_create")
    pdf.bullet("Day 2: Mark 1-2 socials live; Generate + Deploy In-App")
    pdf.bullet("Day 3: Check Acquisition CRM + Funnel; adjust budget only")
    pdf.bullet("Day 4: Optional second hook or TikTok")
    pdf.bullet("Day 5: Truth Dashboard; homepage tweaks; re-export teaser if needed")
    pdf.bullet("Day 6-7: Only then consider Path A Enable outbound")

    pdf.h2("11. Cheat sheet")
    pdf.paste(
        "ADS:      Attack pack + hero-teaser.mp4 + UTM URL -> Ads Manager (you)\n"
        "ATTR:     UTMs -> Acquisition CRM | Pixels -> Meta/GA4/TikTok\n"
        "PRODUCT:  Users -> signals + wallet events -> Funnel\n"
        "AI:       Generate -> Deploy In-App -> nudges auto\n"
        "OUTBOUND: Path A gate later | Path B approve + tick\n"
        "NEVER:    Expect Strategist to publish Meta/TikTok ads"
    )

    pdf.h2("12. Related repo paths")
    pdf.bullet("frontend/admin-marketing.html  -  dashboard UI")
    pdf.bullet("frontend/js/acquisition-command-center.js  -  UTM + attack pack")
    pdf.bullet("frontend/js/conversion-pixels.js + .local.js  -  pixel loader + IDs")
    pdf.bullet("frontend/videos/hero-teaser.mp4  -  homepage / first ad creative")
    pdf.bullet("services/marketing_touchpoint_runner.py  -  Path A")
    pdf.bullet("services/marketing_execution.py  -  Path B")
    pdf.bullet("services/growth_intelligence.py  -  Funnel / signals / UTM intel")
    pdf.bullet("docs/marketing-rollout-hooks.md  -  organic social hooks")

    pdf.output(str(OUT))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    build()

