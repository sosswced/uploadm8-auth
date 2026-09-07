#!/usr/bin/env python3
"""
Assisted official-brand social signup (headed Playwright).

Stop-and-go: on errors/CAPTCHA/OTP the script PAUSES (browser stays open).
  Continue → write anything to scripts/agent/.brand_continue  (or reply "go" in chat)
  Stop     → write anything to scripts/agent/.brand_stop
  OTP      → one line in scripts/agent/.brand_otp

Secrets: frontend/js/official-brand-secrets.local.js (reloaded after password pauses).
Optional JSON: scripts/agent/.brand_secrets.local.json  { "email", "passwords": {...} }

Usage:
  python scripts/agent/assisted_brand_signup.py --platform tiktok --mode rebrand
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SECRETS_JS = ROOT / "frontend" / "js" / "official-brand-secrets.local.js"
SECRETS_JSON = Path(__file__).resolve().parent / ".brand_secrets.local.json"
OTP_FILE = Path(__file__).resolve().parent / ".brand_otp"
CONTINUE_FILE = Path(__file__).resolve().parent / ".brand_continue"
STOP_FILE = Path(__file__).resolve().parent / ".brand_stop"
STATUS_FILE = Path(__file__).resolve().parent / ".brand_signup_status.json"
SCREEN_DIR = Path(__file__).resolve().parent / ".brand_signup_shots"

BIO = (
    "Official UploadM8 — upload once, post to TikTok, YouTube, Instagram & Facebook. "
    "AI captions, thumbnails, scheduling. You create — we'll do the rest."
)
BIO_TIKTOK = "Official UploadM8 — upload once, post everywhere. You create; we do the rest."
BIO_LINK = (
    "https://app.uploadm8.com/signup.html"
    "?utm_source=bio&utm_medium=organic_social&utm_campaign=official_bio&utm_content=uploadm8"
)
LOGO_PATH = ROOT / "frontend" / "images" / "logo.png"
HANDLE_FALLBACK = "officialuploadm8"
HANDLE_X_FALLBACK = "getuploadm8"  # X username max 15 chars (officialuploadm8 is 16)
PERSONAL_FIRST = "Cedrick"
PERSONAL_LAST = "Gillespie"
PERSONAL_FULL = "Cedrick Gillespie"
BRAND_NAME = "Official UploadM8"
BIRTHDAY = {"month": "February", "month_num": "2", "day": "26", "year": "1999"}
FAST_MS = 250
FILL_TO = 900
CLICK_TO = 900

PLATFORMS = {
    "tiktok": {
        "url": "https://www.tiktok.com/signup/phone-or-email/email",
        "alt_url": "https://www.tiktok.com/signup",
        "login_url": "https://www.tiktok.com/login/phone-or-email/email",
        "generic_profile": "https://www.tiktok.com/@user3984681198447",
    },
    "instagram": {"url": "https://www.instagram.com/accounts/emailsignup/"},
    "youtube": {"url": "https://accounts.google.com/signup"},
    "facebook": {"url": "https://www.facebook.com/r.php"},
    "x": {"url": "https://x.com/i/flow/signup"},
    "linkedin": {"url": "https://www.linkedin.com/signup"},
    "discord": {"url": "https://discord.com/register"},
}


class UserStop(Exception):
    """Operator requested stop via .brand_stop."""


class SoftBlock(Exception):
    """Recoverable UI block — pause then continue."""


def _clear_flag(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


def say(msg: str) -> None:
    text = str(msg)
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        print(text.encode("ascii", "replace").decode("ascii"), flush=True)



def load_secrets() -> dict:
    data: dict = {
        "email": "",
        "phone": "",
        "handle": "uploadm8",
        "display_name": "Official UploadM8",
        "passwords": {},
    }

    if SECRETS_JS.exists():
        text = SECRETS_JS.read_text(encoding="utf-8")
        m = re.search(r"__UM8_BRAND_SECRETS__\s*=\s*(\{.*?\});", text, re.S)
        if m:
            raw = m.group(1)
            email = re.search(r"email:\s*'([^']+)'", raw)
            phone = re.search(r"phone(?:_e164)?:\s*'([^']+)'", raw)
            handle = re.search(r"handle:\s*'([^']+)'", raw)
            display = re.search(r"display_name:\s*'([^']+)'", raw)
            if email:
                data["email"] = email.group(1)
            if phone:
                data["phone"] = phone.group(1)
            if handle:
                data["handle"] = handle.group(1).lstrip("@")
            if display:
                data["display_name"] = display.group(1)
            for k in ("tiktok", "instagram", "youtube", "facebook", "x", "linkedin", "discord"):
                pm = re.search(rf"{k}:\s*'([^']*)'", raw)
                if pm:
                    data["passwords"][k] = pm.group(1)
        else:
            say("[secrets] could not parse official-brand-secrets.local.js object")

    # JSON overlay wins (for mid-run password fixes)
    if SECRETS_JSON.exists():
        try:
            raw_j = json.loads(SECRETS_JSON.read_text(encoding="utf-8"))
            if raw_j.get("email"):
                data["email"] = raw_j["email"]
            if raw_j.get("phone") or raw_j.get("phone_e164"):
                data["phone"] = raw_j.get("phone") or raw_j.get("phone_e164")
            if raw_j.get("handle"):
                data["handle"] = str(raw_j["handle"]).lstrip("@")
            if raw_j.get("display_name"):
                data["display_name"] = raw_j["display_name"]
            for k, v in (raw_j.get("passwords") or {}).items():
                if v and v != "REPLACE":
                    data["passwords"][k] = v
            say(f"[secrets] JSON overlay applied ({SECRETS_JSON.name})")
        except Exception as e:
            say(f"[secrets] JSON overlay unreadable — JS only ({e})")

    if not SECRETS_JS.exists() and not SECRETS_JSON.exists():
        raise SoftBlock(f"No secrets file. Create {SECRETS_JS.name} or {SECRETS_JSON.name}")

    return data


def set_status(**kwargs) -> None:
    prev = {}
    if STATUS_FILE.exists():
        try:
            prev = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
        except Exception:
            prev = {}
    prev.update(kwargs)
    prev["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    STATUS_FILE.write_text(json.dumps(prev, indent=2), encoding="utf-8")
    # Keep status quiet unless phase change
    if "phase" in kwargs or "error" in kwargs:
        say(f"[status] {kwargs}")


def pause_go(title: str, how_to_fix: str, timeout_s: int = 1800) -> None:
    """Friendly stop-and-go. Browser should already be open for the human."""
    _clear_flag(CONTINUE_FILE)
    _clear_flag(STOP_FILE)
    set_status(phase="paused", error=title, hint=how_to_fix)
    say("")
    say("=" * 60)
    say(f"[PAUSED] {title}")
    say(f"    {how_to_fix}")
    say(f"    Continue: create empty file  {CONTINUE_FILE.name}")
    say(f"               (or tell the agent: go)")
    say(f"    Stop:     create empty file  {STOP_FILE.name}")
    say("=" * 60)
    say("")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if STOP_FILE.exists():
            _clear_flag(STOP_FILE)
            raise UserStop(title)
        if CONTINUE_FILE.exists():
            _clear_flag(CONTINUE_FILE)
            say("[GO] Continuing...")
            set_status(phase="resumed")
            return
        time.sleep(0.4)
    raise UserStop(f"Timed out while paused: {title}")


def wait_for_otp(timeout_s: int = 600, preexisting: str | None = None) -> str:
    if preexisting and preexisting.strip():
        return preexisting.strip()
    _clear_flag(OTP_FILE)
    set_status(phase="awaiting_otp", hint=f"Write OTP to {OTP_FILE.name}")
    say("")
    say("=" * 60)
    say("[WAITING FOR OTP]")
    say(f"    Paste the code in chat, or write one line to {OTP_FILE.name}")
    say(f"    Stop anytime with {STOP_FILE.name}")
    say("=" * 60)
    say("")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if STOP_FILE.exists():
            _clear_flag(STOP_FILE)
            raise UserStop("Stopped while waiting for OTP")
        if OTP_FILE.exists():
            code = OTP_FILE.read_text(encoding="utf-8").strip()
            if code:
                _clear_flag(OTP_FILE)
                return code
        time.sleep(0.4)
    pause_go("OTP timed out", "Paste OTP into .brand_otp then continue, or stop.")
    return wait_for_otp(timeout_s=timeout_s)


def page_alive(page) -> bool:
    try:
        _ = page.url
        return True
    except Exception:
        return False


def shot(page, name: str) -> None:
    if not page_alive(page):
        say(f"[shot] skipped ({name}) — browser closed")
        return
    SCREEN_DIR.mkdir(parents=True, exist_ok=True)
    path = SCREEN_DIR / f"{int(time.time())}_{name}.png"
    try:
        page.screenshot(path=str(path), full_page=False)
        say(f"[shot] {path.name}")
    except Exception as e:
        say(f"[shot] skipped ({name}): {e}")


def page_text(page) -> str:
    try:
        return page.inner_text("body") or ""
    except Exception:
        return ""


def detect_tiktok_login_error(page) -> str | None:
    t = page_text(page)
    patterns = [
        (r"doesn't match our records", "Username or password doesn't match. Update Bitwarden / secrets, then continue."),
        (r"maximum number of attempts", "Too many login attempts. Wait a bit, then continue."),
        (r"Something went wrong", "TikTok hiccup. Refresh/login manually in the window, then continue."),
        (r"Verify|CAPTCHA|captcha|Drag the puzzle", "CAPTCHA / puzzle — solve it in the Chromium window, then continue."),
    ]
    for rx, msg in patterns:
        if re.search(rx, t, re.I):
            return msg
    return None



def react_set(page, value: str, matchers: list[str]) -> bool:
    """Set a React-controlled input by placeholder / name / aria-label."""
    if not page_alive(page):
        return False
    try:
        ok = page.evaluate(
            """({ value, matchers }) => {
              const inputs = [...document.querySelectorAll('input, textarea')];
              const el = inputs.find((i) => {
                const bits = [
                  i.placeholder || '',
                  i.name || '',
                  i.getAttribute('aria-label') || '',
                  i.getAttribute('autocomplete') || '',
                  i.id || '',
                ].join(' ').toLowerCase();
                return matchers.some((m) => bits.includes(m.toLowerCase()));
              });
              if (!el) return false;
              el.focus();
              const proto = el.tagName === 'TEXTAREA'
                ? window.HTMLTextAreaElement.prototype
                : window.HTMLInputElement.prototype;
              const desc = Object.getOwnPropertyDescriptor(proto, 'value');
              if (desc && desc.set) desc.set.call(el, value);
              else el.value = value;
              el.dispatchEvent(new Event('input', { bubbles: true }));
              el.dispatchEvent(new Event('change', { bubbles: true }));
              return el.value === value;
            }""",
            {"value": value, "matchers": matchers},
        )
        return bool(ok)
    except Exception:
        return False


def fill_first(page, selectors: list[str], value: str, label: str) -> bool:
    if not page_alive(page) or value is None:
        return False
    for sel in selectors:
        loc = page.locator(sel).first
        try:
            loc.wait_for(state="visible", timeout=FILL_TO)
            loc.click(timeout=CLICK_TO, force=True)
            loc.fill(value, timeout=FILL_TO)
            say(f"[fill] {label}")
            return True
        except Exception:
            continue
    return False


def click_first(page, selectors: list[str], label: str) -> bool:
    if not page_alive(page):
        return False
    for sel in selectors:
        if sel.startswith(("button", "input", "a", "[", "#", ".")) or ":has-text" in sel or "role=" in sel:
            loc = page.locator(sel).first
            try:
                loc.wait_for(state="visible", timeout=FILL_TO)
                loc.click(timeout=CLICK_TO, force=True)
                say(f"[click] {label}")
                return True
            except Exception:
                continue
    for text in selectors:
        if text.startswith(("button", "input", "a", "[", "#", ".")) or ":has-text" in text:
            continue
        try:
            page.get_by_role("button", name=re.compile(text, re.I)).first.click(timeout=CLICK_TO, force=True)
            say(f"[click] {label}")
            return True
        except Exception:
            pass
        try:
            page.get_by_text(text, exact=False).first.click(timeout=CLICK_TO, force=True)
            say(f"[click] {label}")
            return True
        except Exception:
            continue
    return False


def set_birthday_fast(page) -> None:
    """Native <select> birthday if present. Never pause if missing."""
    if not page_alive(page):
        return
    month, day, year = BIRTHDAY["month"], BIRTHDAY["day"], BIRTHDAY["year"]
    try:
        selects = page.locator("select")
        n = selects.count()
        if n >= 3:
            try:
                selects.nth(0).select_option(label=month)
            except Exception:
                selects.nth(0).select_option(value=BIRTHDAY["month_num"])
            try:
                selects.nth(1).select_option(value=day)
            except Exception:
                selects.nth(1).select_option(label=day)
            selects.nth(2).select_option(label=year)
            say(f"[fill] birthday {month} {day} {year}")
            return
    except Exception:
        pass
    for name, val in (
        ("birthday_month", BIRTHDAY["month_num"]),
        ("birthday_day", day),
        ("birthday_year", year),
        ("month", BIRTHDAY["month_num"]),
        ("day", day),
        ("year", year),
    ):
        try:
            loc = page.locator(f"select[name='{name}'], #{name}")
            if loc.count():
                loc.first.select_option(value=str(val))
        except Exception:
            pass


def _tiktok_set_birthday(page) -> None:
    """Month / Day / Year dropdowns on TikTok email signup — fixed 02/26/1999."""
    month, day, year = BIRTHDAY["month"], BIRTHDAY["day"], BIRTHDAY["year"]
    selects = page.locator("select")
    try:
        n = selects.count()
    except Exception:
        n = 0
    if n >= 3:
        try:
            selects.nth(0).select_option(label=month)
            try:
                selects.nth(1).select_option(value=day)
            except Exception:
                selects.nth(1).select_option(label=day)
            selects.nth(2).select_option(label=year)
            print(f"[fill] birthday via select {month} {day} {year}", flush=True)
            return
        except Exception as e:
            print(f"[fill] birthday select failed: {e}", flush=True)

    for placeholder, choice in (("Month", month), ("Day", day), ("Year", year)):
        try:
            box = page.get_by_text(placeholder, exact=True).first
            box.click(timeout=2000)
            page.wait_for_timeout(300)
            page.get_by_text(choice, exact=True).first.click(timeout=2000)
            print(f"[fill] birthday {placeholder}={choice}", flush=True)
        except Exception as e:
            print(f"[fill] birthday {placeholder} skip: {e}", flush=True)


def tiktok_signup(page, secrets: dict, otp: str | None) -> None:
    handle = secrets["handle"]
    email = secrets["email"]
    password = secrets["passwords"].get("tiktok") or ""
    if not password or password == "REPLACE":
        raise SystemExit("Missing tiktok password in secrets file")

    set_status(platform="tiktok", phase="open_signup", handle=handle, email=email)
    page.goto(PLATFORMS["tiktok"]["url"], wait_until="domcontentloaded", timeout=60000)
    page.wait_for_timeout(2500)
    shot(page, "tiktok_open")

    _tiktok_set_birthday(page)

    filled_email = fill_first(
        page,
        [
            'input[placeholder*="Email" i]',
            'input[name="email"]',
            'input[type="email"]',
            'input[autocomplete="email"]',
        ],
        email,
        "email",
    )
    filled_pw = fill_first(
        page,
        [
            'input[placeholder*="Password" i]',
            'input[type="password"]',
            'input[name="password"]',
            'input[autocomplete="new-password"]',
        ],
        password,
        "password",
    )
    if not filled_email or not filled_pw:
        raise SystemExit("Could not fill TikTok email/password fields — check the Chromium window")

    shot(page, "tiktok_filled")

    # Request email OTP (button is on the same form as the code field)
    set_status(phase="send_code")
    if not click_first(
        page,
        [
            'button:has-text("Send code")',
            'button:has-text("Send Code")',
            "Send code",
        ],
        "send code",
    ):
        print(">>> Click **Send code** in the browser if it did not auto-click.", flush=True)

    page.wait_for_timeout(1500)
    shot(page, "tiktok_after_send_code")
    set_status(phase="maybe_captcha_or_otp", note="Solve CAPTCHA in the browser window if shown")
    print(">>> If CAPTCHA/puzzle appears after Send code, solve it in Chromium.", flush=True)

    otp_selectors = [
        'input[placeholder*="6-digit" i]',
        'input[placeholder*="code" i]',
        'input[placeholder*="Code" i]',
        'input[name="code"]',
        'input[autocomplete="one-time-code"]',
        'input[inputmode="numeric"]',
    ]

    code = wait_for_otp(preexisting=otp)
    set_status(phase="submitting_otp")
    if not fill_first(page, otp_selectors, code, "otp"):
        try:
            page.keyboard.type(code, delay=80)
        except Exception:
            pass

    click_first(
        page,
        [
            'button:has-text("Next")',
            'button[type="submit"]',
            'button:has-text("Verify")',
            "Next",
        ],
        "otp submit / next",
    )
    page.wait_for_timeout(3000)
    shot(page, "tiktok_after_otp")

    set_status(
        phase="post_otp",
        note=f"Claim @{handle} in profile if prompted; fallback @officialuploadm8",
        bio=BIO,
        bio_link=BIO_LINK,
    )
    print(
        f"\n>>> OTP submitted. In the browser:\n"
        f"  - Set username @{handle} (or @officialuploadm8)\n"
        f"  - Display: {secrets['display_name']}\n"
        f"  - Bio + link from kit\n"
        f"  - Then Mark live in local admin Official social kit\n"
        f"Keeping browser open ~3 minutes for you to finish profile.\n",
        flush=True,
    )
    page.wait_for_timeout(180000)
    set_status(phase="done_awaiting_mark_live")
    shot(page, "tiktok_final")


def tiktok_login(page, secrets: dict, otp: str | None) -> dict:
    """Attempt login with stop-and-go on bad password / CAPTCHA. Returns (possibly reloaded) secrets."""
    for attempt in range(1, 6):
        if not page_alive(page):
            pause_go("Browser closed", "Re-run the script when ready, or open Chromium again.")
            return secrets

        secrets = load_secrets()  # pick up password updates between pauses

        # Already logged in (e.g. human finished login during a pause)
        try:
            url = (page.url or "").lower()
        except Exception:
            url = ""
        if url and "login" not in url and "signup" not in url:
            if page.locator('a[data-e2e="nav-profile"]').count() or page.locator('button:has-text("Edit profile")').count():
                say("[ok] Already logged in — skipping login form.")
                return secrets
            # Heuristic: on tiktok.com home/foryou/profile
            if "tiktok.com" in url and "/login" not in url:
                say("[ok] On TikTok without login URL — treating as logged in.")
                return secrets

        email = secrets["email"]
        password = secrets["passwords"].get("tiktok") or ""
        if not password or password == "REPLACE":
            pause_go(
                "TikTok password missing",
                f"Put passwords.tiktok in {SECRETS_JS.name} or {SECRETS_JSON.name}, then continue.",
            )
            continue
        if len(password) > 20:
            # Signup UI says max 20; existing accounts may already use longer — warn, don't block login.
            say(f"[warn] TikTok password is {len(password)} chars (signup UI max is 20); trying login anyway.")
        if len(password) < 8:
            pause_go(
                f"TikTok password is only {len(password)} chars (min 8)",
                "Use at least 8 chars with letters, numbers, and a special character, then continue.",
            )
            continue

        set_status(platform="tiktok", phase="login", attempt=attempt)
        try:
            page.goto(PLATFORMS["tiktok"]["login_url"], wait_until="domcontentloaded", timeout=60000)
        except Exception as e:
            pause_go("Could not open TikTok login", f"{e}\nFix network / reopen, then continue.")
            continue

        page.wait_for_timeout(1500)
        shot(page, "tiktok_login_open")

        fill_first(
            page,
            [
                'input[placeholder*="Email" i]',
                'input[name="username"]',
                'input[type="text"]',
                'input[autocomplete="username"]',
            ],
            email,
            "login email",
        )
        fill_first(
            page,
            ['input[type="password"]', 'input[placeholder*="Password" i]'],
            password,
            "login password",
        )
        click_first(
            page,
            ['button[type="submit"]', 'button:has-text("Log in")', "Log in"],
            "log in",
        )
        page.wait_for_timeout(2800)
        shot(page, "tiktok_login_after")

        err = detect_tiktok_login_error(page)
        if err and "CAPTCHA" in err.upper():
            pause_go("CAPTCHA / puzzle", f"{err}\nSolve it in the window, then continue.")
            if "login" not in (page.url or "").lower():
                say("[ok] Looks logged in after CAPTCHA.")
                return secrets
            continue
        if err and re.search(r"too many|maximum number of attempts|try again later|rate.?limit", err, re.I):
            pause_go(
                "TikTok rate-limited login",
                "Log in manually in this Chromium window (Bitwarden autofill is fine). "
                "When you see your profile / For You feed, reply go.",
            )
            if "login" not in ((page.url or "").lower()):
                say("[ok] Manual login detected.")
                return secrets
            continue
        if err:
            pause_go(
                f"Login blocked (try {attempt}/5)",
                f"{err}\nUpdate password in secrets/Bitwarden if needed, then continue to retry.",
            )
            continue

        # OTP field?
        for sel in (
            'input[placeholder*="code" i]',
            'input[placeholder*="6-digit" i]',
            'input[autocomplete="one-time-code"]',
        ):
            try:
                if page.locator(sel).first.is_visible(timeout=1200):
                    code = wait_for_otp(preexisting=otp)
                    fill_first(page, [sel], code, "login otp")
                    click_first(
                        page,
                        ['button:has-text("Next")', 'button[type="submit"]', "Next"],
                        "otp next",
                    )
                    page.wait_for_timeout(2000)
                    break
            except Exception:
                continue

        # Success heuristic: left login URL or Profile visible
        url = (page.url or "").lower()
        if "login" not in url or page.locator('a[data-e2e="nav-profile"]').count():
            say("[ok] Login looks good.")
            return secrets

        pause_go(
            "Not sure if login worked",
            "Finish login manually in Chromium (or fix password), then continue.",
        )
        if "login" not in (page.url or "").lower():
            return secrets

    pause_go("Login retries exhausted", "Log in manually in the window, then continue once.")
    return secrets


def _set_input_by_labelish(page, labels: list[str], value: str, field: str) -> bool:
    for lab in labels:
        try:
            loc = page.get_by_label(lab, exact=False).first
            loc.wait_for(state="visible", timeout=2000)
            loc.click()
            loc.fill("")
            loc.fill(value)
            say(f"[fill] {field} via label:{lab}")
            return True
        except Exception:
            pass
        try:
            row = page.locator(f"div:has-text('{lab}')").filter(has=page.locator("input, textarea")).first
            box = row.locator("input, textarea").first
            box.wait_for(state="visible", timeout=2000)
            box.click()
            box.fill("")
            box.fill(value)
            say(f"[fill] {field} via row:{lab}")
            return True
        except Exception:
            continue
    return False


def tiktok_rebrand_profile(page, secrets: dict, otp: str | None) -> None:
    """Login (stop-and-go) then set username / name / bio / avatar."""
    handle = secrets["handle"].lstrip("@") or "uploadm8"
    display = secrets.get("display_name") or "Official UploadM8"
    bio = BIO_TIKTOK
    if len(bio) > 80:
        pause_go("Bio too long for TikTok", f"{len(bio)} chars — trim BIO_TIKTOK in script, then continue.")

    if not LOGO_PATH.exists():
        pause_go("Logo missing", f"Add {LOGO_PATH}, then continue.")

    set_status(platform="tiktok", phase="rebrand_start", handle=handle, bio_len=len(bio))
    secrets = tiktok_login(page, secrets, otp)
    if not page_alive(page):
        pause_go("Browser closed after login", "Re-run when ready.")
        return

    shot(page, "tiktok_rebrand_logged_in")

    # Open known generic profile, else Profile nav
    try:
        page.goto(PLATFORMS["tiktok"]["generic_profile"], wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(2000)
    except Exception as e:
        pause_go("Could not open profile URL", f"{e}\nOpen your profile manually, then continue.")

    if not page_alive(page):
        return

    body_l = (page.content() or "").lower()
    if "couldn't find" in body_l:
        if not click_first(page, ['a[data-e2e="nav-profile"]', "Profile"], "profile nav"):
            pause_go("Open your TikTok profile", "Click Profile in the left nav, then continue.")

    if not click_first(
        page,
        [
            'button:has-text("Edit profile")',
            '[data-e2e="edit-profile-entrance"]',
            "Edit profile",
        ],
        "edit profile",
    ):
        pause_go(
            "Edit profile not found",
            "Click Edit profile in the window (modal should open), then continue.",
        )

    page.wait_for_timeout(1000)
    shot(page, "tiktok_edit_open")

    claimed = handle
    if not _set_input_by_labelish(page, ["Username", "username"], claimed, "username"):
        if not fill_first(page, ['input[placeholder*="Username" i]'], claimed, "username"):
            pause_go("Could not fill username", f"Type @{claimed} (or @{HANDLE_FALLBACK}) yourself, then continue.")

    page.wait_for_timeout(600)
    body = page_text(page)
    if re.search(r"already|taken|unavailable|not available", body, re.I):
        claimed = HANDLE_FALLBACK
        say(f"[warn] @{handle} unavailable → trying @{claimed}")
        _set_input_by_labelish(page, ["Username", "username"], claimed, "username_fallback")

    if not _set_input_by_labelish(page, ["Name", "Nickname", "name"], display, "name"):
        pause_go("Could not fill Name", f"Set Name to “{display}” (7-day lock), then continue.")

    if not _set_input_by_labelish(page, ["Bio", "bio"], bio, "bio"):
        if not fill_first(page, ["textarea", 'textarea[placeholder*="Bio" i]'], bio, "bio"):
            pause_go("Could not fill Bio", f"Paste this (77 chars):\n{bio}\nThen continue.")

    uploaded = False
    try:
        file_inputs = page.locator('input[type="file"]')
        if file_inputs.count():
            file_inputs.first.set_input_files(str(LOGO_PATH))
            uploaded = True
            say("[fill] avatar via file input")
    except Exception as e:
        say(f"[warn] avatar auto-upload skipped: {e}")

    if not uploaded:
        pause_go(
            "Avatar needs a manual click",
            f"Click the pencil on the photo → pick {LOGO_PATH.name}, then continue.",
        )

    shot(page, "tiktok_edit_filled")
    if not click_first(page, ['button:has-text("Save")', "Save"], "save profile"):
        pause_go("Save not clicked", "Click Save in the Edit profile modal, then continue.")

    page.wait_for_timeout(2000)
    shot(page, "tiktok_edit_saved")
    set_status(
        phase="rebrand_saved",
        handle=claimed,
        display_name=display,
        bio=bio,
        avatar_uploaded=uploaded,
        next="Mark live in local admin Official social kit → TikTok",
    )
    say("")
    say(f"[OK] TikTok rebrand flow finished for @{claimed}")
    say(f"  Name: {display}")
    say(f"  Bio:  {bio}")
    say("  Next: Mark live on http://127.0.0.1:8000/admin-marketing.html")
    pause_go(
        "Confirm profile looks right",
        "If username/name/bio/logo look good, continue to close. Or stop to keep the browser open.",
    )


REMAINING_ORDER = ("instagram", "youtube", "facebook", "x", "linkedin", "discord")

# Researched required fields — fill all of these, pause ONLY for OTP / CAPTCHA.
PLATFORM_KIT = {
    "instagram": {
        "needs": "email, password, birthday 02/26/1999, full name Cedrick Gillespie, username uploadm8",
        "otp": "email or SMS 6-digit",
    },
    "youtube": {
        "needs": "Google: first Cedrick, last Gillespie, email Earl@..., password, birthday. Then Brand channel Official UploadM8 / @uploadm8",
        "otp": "Google email/SMS/2FA",
    },
    "facebook": {
        "needs": "personal: first Cedrick, last Gillespie, email, password, birthday, then Page Official UploadM8",
        "otp": "email or SMS",
    },
    "x": {
        "needs": "name Official UploadM8, email, birthday, username uploadm8 (max 15; fallback getuploadm8)",
        "otp": "email or SMS; web may force mobile app",
    },
    "linkedin": {
        "needs": "first Cedrick, last Gillespie, email, password. Company Page later (profile >=1 day + 2 connections)",
        "otp": "email verify",
    },
    "discord": {
        "needs": "email, display Official UploadM8, username uploadm8, password, birthday. Join discord.gg/TVDAc8fnwu",
        "otp": "email link or CAPTCHA; phone only if flagged",
    },
}


def _password_for(secrets: dict, platform: str) -> str:
    return (secrets.get("passwords") or {}).get(platform) or ""


def _handle(secrets: dict) -> str:
    return (secrets.get("handle") or "uploadm8").lstrip("@")


def maybe_otp(page, otp: str | None, platform: str) -> bool:
    """If an OTP box is visible, wait for chat/file and fill. Return True if handled."""
    if not page_alive(page):
        return False
    sels = [
        'input[autocomplete="one-time-code"]',
        'input[name="email_confirmation_code"]',
        'input[name="code"]',
        'input[placeholder*="code" i]',
        'input[placeholder*="6-digit" i]',
        'input[inputmode="numeric"]',
    ]
    visible = False
    for sel in sels:
        try:
            if page.locator(sel).first.is_visible(timeout=400):
                visible = True
                break
        except Exception:
            continue
    if not visible:
        return False
    code = wait_for_otp(preexisting=otp)
    fill_first(page, sels, code, f"{platform} otp")
    click_first(
        page,
        ['button[type="submit"]', 'button:has-text("Confirm")', 'button:has-text("Next")', "Next", "Verify"],
        f"{platform} otp submit",
    )
    return True


def captcha_visible(page) -> bool:
    """Only real challenge widgets — do not pause on generic 'verify' copy."""
    if not page_alive(page):
        return False
    try:
        if page.locator("iframe[src*='recaptcha'], iframe[src*='hcaptcha'], iframe[title*='reCAPTCHA' i]").count():
            return True
    except Exception:
        pass
    t = page_text(page)
    return bool(re.search(r"i.?m not a robot|drag the puzzle|select all images", t, re.I))


def fill_instagram(page, secrets: dict) -> None:
    email = secrets["email"]
    password = _password_for(secrets, "instagram")
    handle = _handle(secrets)
    react_set(page, email, ["email", "mobile number", "emailorphone"]) or fill_first(
        page,
        ['input[placeholder*="Mobile number or email" i]', 'input[name="emailOrPhone"]', 'input[type="text"]'],
        email,
        "instagram email",
    )
    react_set(page, password, ["password"]) or fill_first(
        page, ['input[type="password"]', 'input[name="password"]'], password, "instagram password"
    )
    set_birthday_fast(page)
    try:
        _instagram_dropdowns(page)
    except Exception as e:
        say(f"[warn] IG birthday widgets: {e}")
    try:
        box = page.get_by_placeholder("Full name")
        box.click(force=True, timeout=FILL_TO)
        box.fill(PERSONAL_FULL)
        say("[fill] instagram name Cedrick Gillespie")
    except Exception:
        react_set(page, PERSONAL_FULL, ["full name", "fullname"])
    user = HANDLE_FALLBACK if handle == "uploadm8" else handle
    try:
        box = page.get_by_placeholder("Username")
        box.click(force=True, timeout=FILL_TO)
        box.fill(user)
        say(f"[fill] instagram username @{user}")
    except Exception:
        react_set(page, user, ["username"]) or fill_first(
            page, ['input[placeholder="Username"]', 'input[name="username"]'], user, "instagram username"
        )
    page.wait_for_timeout(200)
    try:
        if "not available" in page_text(page).lower() and user != HANDLE_FALLBACK:
            fill_first(page, ['input[placeholder="Username"]'], HANDLE_FALLBACK, "instagram username fallback")
    except Exception:
        pass
    click_first(page, ['button:has-text("Submit")', 'button[type="submit"]', "Submit", "Sign up"], "instagram submit")


def instagram_complete_profile(page, secrets: dict, otp: str | None) -> None:
    """Logged-in profile template: name, bio (150), website, avatar."""
    email = secrets["email"]
    password = _password_for(secrets, "instagram")
    ig_handle = "officialuploadm8"
    bio = BIO[:150]
    set_status(platform="instagram", phase="profile")
    say("Instagram profile template — name / bio / link / photo")
    page.goto("https://www.instagram.com/accounts/login/", wait_until="domcontentloaded", timeout=45000)
    page.wait_for_timeout(FAST_MS)
    fill_first(
        page,
        ['input[name="username"]', 'input[aria-label*="username" i]', 'input[type="text"]'],
        email,
        "ig login user",
    )
    fill_first(page, ['input[name="password"]', 'input[type="password"]'], password, "ig login pw")
    click_first(page, ['button[type="submit"]', 'button:has-text("Log in")', "Log in"], "ig login")
    page.wait_for_timeout(1200)
    if maybe_otp(page, otp, "instagram"):
        page.wait_for_timeout(800)
    click_first(page, ['button:has-text("Not now")', "Not now"], "not now")
    page.goto(f"https://www.instagram.com/{ig_handle}/", wait_until="domcontentloaded", timeout=45000)
    page.wait_for_timeout(800)
    if not click_first(page, ['a:has-text("Edit profile")', 'button:has-text("Edit profile")', "Edit profile"], "edit profile"):
        page.goto("https://www.instagram.com/accounts/edit/", wait_until="domcontentloaded")
        page.wait_for_timeout(800)
    shot(page, "instagram_edit_open")
    react_set(page, BRAND_NAME, ["name"]) or fill_first(
        page, ['input[id="pepName"]', 'input[name="name"]', 'input[aria-label="Name"]'], BRAND_NAME, "ig display name"
    )
    if not fill_first(page, ['textarea[id="pepBio"]', 'textarea[name="biography"]', "textarea"], bio, "ig bio"):
        react_set(page, bio, ["bio", "biography"])
    fill_first(
        page,
        ['input[id="pepWebsite"]', 'input[name="website"]', 'input[placeholder*="Website" i]'],
        BIO_LINK,
        "ig website",
    )
    try:
        fi = page.locator('input[type="file"]')
        if fi.count() and LOGO_PATH.exists():
            fi.first.set_input_files(str(LOGO_PATH))
            say("[fill] ig avatar")
    except Exception as e:
        say(f"[warn] avatar: {e}")
    click_first(page, ['button:has-text("Submit")', 'div[role="button"]:has-text("Submit")', "Submit"], "ig save")
    shot(page, "instagram_profile_saved")
    pause_go(
        "Instagram profile — confirm in Chromium",
        f"Name: {BRAND_NAME}\nBio: {bio}\nLink: {BIO_LINK}\nPhoto: logo.png\n"
        "If any field is blank, paste from this chat, then go.",
    )


def _instagram_dropdowns(page) -> None:
    month, day, year = BIRTHDAY["month"], BIRTHDAY["day"], BIRTHDAY["year"]
    if page.locator("select").count() >= 3:
        set_birthday_fast(page)
        return

    def pick(label: str, value: str) -> None:
        page.locator(f'span:text-is("{label}")').first.click(force=True, timeout=FILL_TO)
        page.wait_for_timeout(150)
        try:
            page.get_by_role("option", name=value, exact=True).first.click(timeout=FILL_TO)
        except Exception:
            page.locator(f'[role="option"]:has-text("{value}")').first.click(force=True, timeout=FILL_TO)
        say(f"[fill] IG {label}={value}")

    pick("Month", month)
    pick("Day", day)
    pick("Year", year)


def fill_youtube(page, secrets: dict) -> None:
    email = secrets["email"]
    password = _password_for(secrets, "youtube")
    react_set(page, PERSONAL_FIRST, ["first", "given"]) or fill_first(
        page, ['input[name="firstName"]', 'input[id="firstName"]'], PERSONAL_FIRST, "google first"
    )
    react_set(page, PERSONAL_LAST, ["last", "family"]) or fill_first(
        page, ['input[name="lastName"]', 'input[id="lastName"]'], PERSONAL_LAST, "google last"
    )
    local = email.split("@")[0]
    react_set(page, local, ["username"]) or fill_first(
        page, ['input[name="Username"]', 'input[id="username"]', 'input[type="email"]'], email, "google username"
    )
    click_first(page, ['button:has-text("Next")', "#collectNameNext", "Next"], "google next")
    page.wait_for_timeout(FAST_MS)
    fill_first(page, ['input[type="password"]', 'input[name="Passwd"]'], password, "google password")
    fill_first(page, ['input[name="ConfirmPasswd"]', 'input[name="PasswdAgain"]'], password, "google confirm pw")
    click_first(page, ['button:has-text("Next")', "Next"], "google pw next")
    page.wait_for_timeout(600)
    # Basic information: Month dropdown + Day/Year + Gender Male
    try:
        page.get_by_label("Month", exact=False).first.select_option(value=BIRTHDAY["month_num"])
        say("[fill] google month")
    except Exception:
        try:
            page.locator("select").first.select_option(label=BIRTHDAY["month"])
        except Exception:
            pass
    fill_first(page, ['input[name="day"]', 'input[aria-label="Day"]', 'input[placeholder="Day"]'], BIRTHDAY["day"], "google day")
    fill_first(page, ['input[name="year"]', 'input[aria-label="Year"]', 'input[placeholder="Year"]'], BIRTHDAY["year"], "google year")
    try:
        page.get_by_label("Gender", exact=False).first.select_option(label="Male")
        say("[fill] google gender Male")
    except Exception:
        click_first(page, ['option:has-text("Male")', "Male"], "google gender")
        try:
            page.locator("select").nth(1).select_option(label="Male")
        except Exception:
            pass
    click_first(page, ['button:has-text("Next")', "Next"], "google birthday next")


def fill_facebook(page, secrets: dict) -> None:
    email = secrets["email"]
    password = _password_for(secrets, "facebook")
    fill_first(page, ['input[name="firstname"]', 'input[name="firstName"]'], PERSONAL_FIRST, "fb first")
    fill_first(page, ['input[name="lastname"]', 'input[name="lastName"]'], PERSONAL_LAST, "fb last")
    fill_first(page, ['input[name="reg_email__"]', 'input[name="email"]', 'input[type="text"]'], email, "fb email")
    fill_first(page, ['input[name="reg_email_confirmation__"]'], email, "fb email confirm")
    fill_first(page, ['input[name="reg_passwd__"]', 'input[type="password"]'], password, "fb password")
    set_birthday_fast(page)
    # Facebook: sex value 2 = male
    try:
        page.locator('input[name="sex"][value="2"]').first.check(timeout=FILL_TO)
        say("[fill] facebook gender male")
    except Exception:
        click_first(page, ['label:has-text("Male")', 'input[value="2"]', "Male"], "fb gender male")
    click_first(page, ['button[name="websubmit"]', 'button:has-text("Sign Up")', "Sign Up"], "fb submit")


def fill_x(page, secrets: dict) -> None:
    handle = _handle(secrets)
    if len(handle) > 15:
        handle = HANDLE_X_FALLBACK
    react_set(page, BRAND_NAME, ["name"]) or fill_first(
        page, ['input[name="name"]', 'input[autocomplete="name"]'], BRAND_NAME, "x name"
    )
    react_set(page, secrets["email"], ["email", "phone"]) or fill_first(
        page, ['input[name="email"]', 'input[type="email"]', 'input[autocomplete="email"]'], secrets["email"], "x email"
    )
    set_birthday_fast(page)
    click_first(page, ['button:has-text("Use email instead")', "Use email instead"], "x email instead")
    click_first(page, ['button:has-text("Next")', 'button[data-testid="ocfSignupNextLink"]', "Next"], "x next")
    react_set(page, handle, ["username"]) or fill_first(
        page, ['input[name="username"]'], handle, "x username"
    )


def fill_linkedin(page, secrets: dict) -> None:
    email = secrets["email"]
    password = _password_for(secrets, "linkedin")
    fill_first(page, ['input[name="email-address"]', 'input[id="email-address"]', 'input[type="email"]'], email, "li email")
    fill_first(page, ['input[name="password"]', 'input[id="password"]', 'input[type="password"]'], password, "li password")
    click_first(page, ['button[type="submit"]', 'button:has-text("Agree")', "Agree & Join"], "li join")
    page.wait_for_timeout(FAST_MS)
    fill_first(page, ['input[name="first-name"]', 'input[id="first-name"]'], PERSONAL_FIRST, "li first")
    fill_first(page, ['input[name="last-name"]', 'input[id="last-name"]'], PERSONAL_LAST, "li last")
    click_first(page, ['button[type="submit"]', 'button:has-text("Continue")', "Continue"], "li continue")


def fill_discord(page, secrets: dict) -> None:
    email = secrets["email"]
    password = _password_for(secrets, "discord")
    handle = _handle(secrets)
    fill_first(page, ['input[name="email"]', 'input[type="email"]'], email, "discord email")
    fill_first(page, ['input[name="global_name"]', 'input[placeholder*="Display" i]'], BRAND_NAME, "discord display")
    fill_first(page, ['input[name="username"]'], handle, "discord username")
    fill_first(page, ['input[name="password"]', 'input[type="password"]'], password, "discord password")
    set_birthday_fast(page)
    click_first(page, ['button[type="submit"]', 'button:has-text("Create")', "Continue"], "discord submit")


FILLERS = {
    "instagram": fill_instagram,
    "youtube": fill_youtube,
    "facebook": fill_facebook,
    "x": fill_x,
    "linkedin": fill_linkedin,
    "discord": fill_discord,
}



def assisted_platform_signup(page, secrets: dict, platform: str, otp: str | None) -> None:
    kit = PLATFORM_KIT.get(platform, {})
    url = PLATFORMS[platform]["url"]
    set_status(platform=platform, phase="open")
    say("")
    say("#" * 60)
    say(f"# {platform.upper()}  needs: {kit.get('needs', '')}")
    say(f"# OTP: {kit.get('otp', 'if shown')}")
    say("#" * 60)
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=45000)
    except Exception as e:
        say(f"[warn] {platform} load: {e}")
        return
    page.wait_for_timeout(FAST_MS)
    shot(page, f"{platform}_open")
    filler = FILLERS.get(platform)
    if filler:
        try:
            filler(page, secrets)
        except UserStop:
            raise
        except Exception as e:
            say(f"[warn] {platform} fill continued: {e}")
    page.wait_for_timeout(FAST_MS)
    shot(page, f"{platform}_after_fill")
    # Stay on THIS platform until OTP or you confirm done. Do not hop.
    for _ in range(8):
        if maybe_otp(page, otp, platform):
            break
        if captcha_visible(page):
            pause_go(f"{platform}: CAPTCHA", "Solve it in this Chromium window, then continue.")
            continue
        page.wait_for_timeout(400)
    pause_go(
        f"{platform}: stay here until this account is done",
        "I filled this platform only. Paste OTP in chat if asked. "
        "When this account exists, reply go — I will start the NEXT platform in a new step.",
    )
    set_status(platform=platform, phase="one_done")
    say(f"[OK] {platform} one-at-a-time step finished.")


def run_remaining(page, secrets: dict, otp: str | None) -> None:
    say("TikTok is MANUAL. Fast-filling remaining platforms. Pause only for OTP/CAPTCHA.")
    say(f"Personal name: {PERSONAL_FULL}  Brand: {BRAND_NAME}  Birthday: 02/26/1999")
    for i, platform in enumerate(REMAINING_ORDER, start=1):
        if not page_alive(page):
            pause_go("Browser closed", "Re-run --platform remaining when ready.")
            return
        say(f">>> [{i}/{len(REMAINING_ORDER)}] {platform}")
        secrets = load_secrets()
        try:
            assisted_platform_signup(page, secrets, platform, otp)
        except UserStop:
            raise
        except Exception as e:
            say(f"[warn] {platform} skipped ahead: {e}")
    say("All remaining platforms walked. Mark live on local admin-marketing when each account exists.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Assisted brand social — stop-and-go on errors")
    ap.add_argument("--platform", default="tiktok", choices=sorted(PLATFORMS) + ["remaining"])
    ap.add_argument("--mode", default="rebrand", choices=("rebrand", "signup", "profile"))
    ap.add_argument("--otp", default=None)
    ap.add_argument("--keep-open-s", type=int, default=60)
    args = ap.parse_args()

    _clear_flag(STOP_FILE)
    _clear_flag(CONTINUE_FILE)

    try:
        secrets = load_secrets()
    except SoftBlock as e:
        say(f"[!] {e}")
        return 2

    say(
        f"Loaded brand: email={secrets['email']} handle=@{secrets['handle']} "
        f"platform={args.platform} mode={args.mode}"
    )
    say(f"Stop-and-go files: {CONTINUE_FILE.name} | {STOP_FILE.name} | {OTP_FILE.name}")
    set_status(platform=args.platform, phase="starting", handle=secrets["handle"], mode=args.mode)

    from playwright.sync_api import sync_playwright

    browser = context = page = None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=False,
                slow_mo=0,
                args=["--disable-blink-features=AutomationControlled"],
            )
            context = browser.new_context(
                viewport={"width": 1280, "height": 900},
                locale="en-US",
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
                ),
            )
            page = context.new_page()
            try:
                if args.platform == "instagram" and args.mode == "profile":
                    instagram_complete_profile(page, secrets, args.otp)
                elif args.platform == "remaining":
                    run_remaining(page, secrets, args.otp)
                elif args.platform == "tiktok" and args.mode == "rebrand":
                    tiktok_rebrand_profile(page, secrets, args.otp)
                elif args.platform == "tiktok":
                    tiktok_signup(page, secrets, args.otp)
                else:
                    assisted_platform_signup(page, secrets, args.platform, args.otp)
            except UserStop as e:
                say(f"[STOP] Stopped: {e}")
                set_status(phase="stopped", error=str(e))
                return 0
            except SoftBlock as e:
                say(f"[!] {e}")
                pause_go(str(e), "Fix the issue in the browser or secrets, then continue or stop.")
                return 0
            except Exception as e:
                say(f"[!] Unexpected issue (browser left open for you): {e}")
                set_status(phase="error_soft", error=str(e))
                try:
                    pause_go(
                        "Unexpected error",
                        f"{e}\nFix manually in Chromium if possible, then continue or stop.",
                    )
                except UserStop:
                    say("[STOP] Stopped after error.")
                return 1
            finally:
                try:
                    if page and page_alive(page):
                        page.wait_for_timeout(min(args.keep_open_s, 20) * 1000)
                except Exception:
                    pass
                try:
                    if context:
                        context.close()
                    if browser:
                        browser.close()
                except Exception:
                    pass
    except UserStop as e:
        say(f"[STOP] Stopped: {e}")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
