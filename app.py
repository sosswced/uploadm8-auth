from __future__ import annotations

import os
import re
import sys
import json
import time
import random
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auth_manager_updated import UnifiedAuthManager  # noqa: E402
from ig_tunnel import CloudflaredTunnel  # noqa: E402

# Upload adapters (public API)
from TikTok_Upload import upload_tiktok_video  # noqa: E402
from Facebook_Upload import upload_facebook_video  # noqa: E402
from Instagram_Upload import upload_instagram_reel  # noqa: E402
from YouTube_Upload import upload_youtube_short  # noqa: E402

# Optional: telemetry scoring + HUD generation (no impact unless TELEMETRY_ENABLED=1)
try:
    import telemetry_trill_v2 as tt  # type: ignore
except Exception:
    tt = None


def ensure_env_exists() -> Path:
    env_path = ROOT / ".env"
    if env_path.exists():
        return env_path

    template = """# ===========================================================
# UPLOADM8 BUSINESS CONFIGURATION (NO BACKEND)
# Each user supplies their own app keys.
# ===========================================================

# --- DIRECTORIES ---
VIDEOS_DIR=videos

# --- DISCORD (optional but recommended) ---
DISCORD_WEBHOOK_URL=

# --- OAUTH LOOPBACK ---
OAUTH_LOCAL_HOST=127.0.0.1
OAUTH_LOCAL_PORT=8421

# --- TIKTOK ---
TIKTOK_CLIENT_KEY=
TIKTOK_CLIENT_SECRET=
TIKTOK_REDIRECT_URI=http://127.0.0.1:8421/tiktok/callback
TIKTOK_SCOPES=user.info.basic,video.upload
DIRECT_POST=0

TIKTOK_ACCESS_TOKEN=
TIKTOK_REFRESH_TOKEN=
TIKTOK_TOKEN_EXPIRES_AT=0

# --- META (FACEBOOK + INSTAGRAM) ---
FB_APP_ID=
FB_APP_SECRET=
FB_PAGE_ID=
FB_OAUTH_REDIRECT=http://127.0.0.1:8421/facebook/callback

FB_USER_ACCESS_TOKEN=
FB_TOKEN_EXPIRES_AT=0
FB_PAGE_ACCESS_TOKEN=

# --- INSTAGRAM CLOUD (only used for IG media_url fetch) ---
# Will be auto-set each run when IG is selected (trycloudflare rotates).
IG_PUBLIC_BASE_URL=

# --- TELEMETRY (optional) ---
TELEMETRY_ENABLED=0
HUD_ENABLED=0
MAPS_DIR=videos
GAZETTEER_PLACES_PATH=
PADUS_PATH=
PADUS_LAYER=
"""
    env_path.write_text(template, encoding="utf-8")
    print("[SETUP] Created .env template. Fill keys, then re-run.")
    return env_path


def read_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    lines = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = raw.strip()
        if s:
            lines.append(s)
    return lines


def cycle_pick(lines: List[str], idx: int) -> str:
    if not lines:
        return ""
    return lines[idx % len(lines)]


def normalize_privacy(raw: str) -> str:
    # legacy; keep for backward compatibility
    s = (raw or "").strip().lower()
    if s in ("public", "pub", "p", "1"):
        return "public"
    if s in ("private", "priv", "prv", "2"):
        return "private"
    return "public"


def normalize_privacy_plus(raw: str, default: str = "public") -> str:
    """
    Your required UX:
      - 1 / public => PUBLIC
      - 2 / private => PRIVATE
    """
    s = (raw or "").strip().lower()
    if not s:
        return default

    if s in ("1", "public", "pub", "u"):
        return "public"
    if s in ("2", "private", "priv", "p"):
        return "private"

    # fallback to legacy mapping
    return normalize_privacy(s)


def parse_duration_to_seconds(raw: str) -> int:
    """
    Accepts:
      - integer seconds: "600"
      - compound: "10s", "15m", "3h", "1h30m", "2h 10m", "90m"
    """
    s = (raw or "").strip().lower().replace(" ", "")
    if not s:
        return 0
    if s.isdigit():
        return int(s)

    total = 0
    for value, unit in re.findall(r"(\d+)([hms])", s):
        v = int(value)
        if unit == "h":
            total += v * 3600
        elif unit == "m":
            total += v * 60
        elif unit == "s":
            total += v
    return total


def countdown(seconds: int) -> None:
    if seconds <= 0:
        return
    end = time.time() + seconds
    while True:
        remaining = int(end - time.time())
        if remaining <= 0:
            print("\r[NEXT] Upload window opened.                      ")
            return
        hh = remaining // 3600
        mm = (remaining % 3600) // 60
        ss = remaining % 60
        print(f"\r[TIMER] Next upload in {hh:02d}:{mm:02d}:{ss:02d}", end="", flush=True)
        time.sleep(1)


def discord_notify(webhook: str, payload: dict) -> None:
    if not webhook:
        return
    try:
        import requests

        requests.post(webhook, json=payload, timeout=15)
    except Exception:
        pass


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip() == "1"


# ===================== TELEMETRY HOOKS (UNCHANGED BEHAVIOR UNLESS ENABLED) =====================

def find_map_for_video(video_path: Path, maps_dir: Path) -> Optional[Path]:
    stem = video_path.stem
    for ext in (".map", ".MAP"):
        cand = maps_dir / f"{stem}{ext}"
        if cand.exists():
            return cand
    for ext in (".map", ".MAP"):
        cand = video_path.with_suffix(ext)
        if cand.exists():
            return cand
    return None


def telemetry_analyze_all(mp4s: List[Path], videos_dir: Path) -> Tuple[List[Path], Dict[str, dict]]:
    """Returns (ranked_mp4s, meta_by_name). Safe/no-crash."""
    if not _env_flag("TELEMETRY_ENABLED", "0"):
        return (mp4s, {})
    if tt is None:
        print("[TRILL] telemetry_trill not available; skipping telemetry features.")
        return (mp4s, {})

    maps_dir = Path(os.getenv("MAPS_DIR", str(videos_dir))).resolve()
    gaz = os.getenv("GAZETTEER_PLACES_PATH", "").strip() or None
    padus = os.getenv("PADUS_PATH", "").strip() or None
    padus_layer = os.getenv("PADUS_LAYER", "").strip() or None

    meta: Dict[str, dict] = {}
    scored: List[Tuple[float, Path]] = []

    for fp in mp4s:
        mp = find_map_for_video(fp, maps_dir)
        if not mp:
            scored.append((-1.0, fp))
            continue

        res = tt.safe_analyze_video(
            mp4_path=str(fp),
            map_path=str(mp),
            gaz_places_path=gaz,
            padus_path=padus,
            padus_layer=padus_layer,
            hud_enabled=_env_flag("HUD_ENABLED", "0"),
        )
        if not res.get("ok"):
            scored.append((-1.0, fp))
            continue

        data = res["data"]
        score = float(data.get("trill_score", -1.0))
        meta[fp.name] = data
        scored.append((score, fp))

    scored_sorted = sorted(scored, key=lambda x: x[0], reverse=True)
    ranked = [p for _, p in scored_sorted]
    return (ranked, meta)


def telemetry_preflight_hud(next_fp: Path, videos_dir: Path) -> Optional[str]:
    """Generates _HUD.mp4 for the next candidate clip (optional)."""
    if not _env_flag("TELEMETRY_ENABLED", "0"):
        return None
    if not _env_flag("HUD_ENABLED", "0"):
        return None
    if tt is None:
        return None

    maps_dir = Path(os.getenv("MAPS_DIR", str(videos_dir))).resolve()
    mp = find_map_for_video(next_fp, maps_dir)
    if not mp:
        return None
    try:
        return tt.ensure_hud_mp4(str(next_fp), str(mp), out_dir=str((ROOT / "generated").resolve()))
    except Exception:
        return None


# ===================== SMART vs MANUAL (ADD-ON ONLY; DOES NOT TOUCH OAUTH/TUNNEL) =====================

SMART_CFG = {
    "CAPS": {
        "tiktok": 4,
        "instagram": 3,
        "facebook": 3,
        "youtube": 2,
    },
    "SMART_SLOTS": {
        "tiktok":    ["07:00", "11:45", "18:15", "22:00"],
        "instagram": ["06:45", "16:00", "19:30"],
        "facebook":  ["07:00", "16:15", "20:00"],
        "youtube":   ["18:00", "21:00"],
    },
    "JITTER_MIN_SEC": -420,
    "JITTER_MAX_SEC":  420,
    "MANUAL_INTERVAL_FALLBACK_SEC": 3 * 3600,
}

RUN_SHEET_FILE = ROOT / "run_sheet.json"


def _pick_mode(raw: str, default: str = "smart") -> str:
    """
    smart:  1 / smart / s
    manual: 2 / manual / m
    """
    r = (raw or "").strip().lower()
    if not r:
        return default
    if r in ("1", "s", "smart"):
        return "smart"
    if r in ("2", "m", "manual"):
        return "manual"
    if r.startswith("s"):
        return "smart"
    if r.startswith("m"):
        return "manual"
    return default


def _seed_daily() -> None:
    today = date.today()
    random.seed(today.toordinal())


def _jitter(dt: datetime) -> datetime:
    j = random.randint(SMART_CFG["JITTER_MIN_SEC"], SMART_CFG["JITTER_MAX_SEC"])
    return dt + timedelta(seconds=j)


def _hhmm_to_h_m(hhmm: str) -> Tuple[int, int]:
    h, m = hhmm.split(":")
    return int(h), int(m)


def _next_occurrence(hhmm: str, base_dt: datetime) -> datetime:
    h, m = _hhmm_to_h_m(hhmm)
    candidate = base_dt.replace(hour=h, minute=m, second=0, microsecond=0)
    if candidate < base_dt:
        candidate = candidate + timedelta(days=1)
    return candidate


def _smart_count_min_cap(platforms: List[str], max_available: int) -> int:
    caps = [SMART_CFG["CAPS"].get(p, 2) for p in platforms]
    cap = max(1, min(caps)) if caps else 1
    return max(1, min(cap, max_available))


def _smart_schedule(platforms: List[str], count: int, now: datetime) -> List[datetime]:
    slot_set = set()
    for p in platforms:
        for s in SMART_CFG["SMART_SLOTS"].get(p, []):
            slot_set.add(s)
    base_slots = sorted(slot_set)

    out: List[datetime] = []
    cursor = now
    while len(out) < count:
        for hhmm in base_slots:
            dt = _jitter(_next_occurrence(hhmm, cursor))
            if dt < now + timedelta(seconds=3):
                continue
            out.append(dt)
            if len(out) >= count:
                break
        cursor = (cursor + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)

    out.sort()
    return out[:count]


def _manual_schedule(user_in: str, count: int, now: datetime) -> List[datetime]:
    """
    Manual timing accepts:
      A) interval: "20s" / "15m" / "3h" / "120"
      B) explicit slots: "07:00,12:30,19:00"
      C) window: "08:00-23:00"  (randomized inside window)
    """
    s = (user_in or "").strip()

    # B) explicit slots
    if "," in s and ":" in s and "-" not in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        dts = [_jitter(_next_occurrence(p, now)) for p in parts]
        dts.sort()
        while len(dts) < count:
            dts.append(_jitter(dts[-1] + timedelta(days=1)))
        return dts[:count]

    # C) window
    if "-" in s and ":" in s:
        a, b = [x.strip() for x in s.split("-", 1)]
        start_dt = _next_occurrence(a, now)
        end_dt = _next_occurrence(b, now)
        if end_dt <= start_dt:
            end_dt = end_dt + timedelta(days=1)

        span = int((end_dt - start_dt).total_seconds())
        if span < 300:
            interval_sec = SMART_CFG["MANUAL_INTERVAL_FALLBACK_SEC"]
            out = [_jitter(now + timedelta(seconds=i * interval_sec)) for i in range(count)]
            out.sort()
            return out

        out = []
        for _ in range(count):
            out.append(_jitter(start_dt + timedelta(seconds=random.randint(0, span))))
        out.sort()
        return out

    # A) interval
    try:
        interval_sec = parse_duration_to_seconds(s) if s else SMART_CFG["MANUAL_INTERVAL_FALLBACK_SEC"]
    except Exception:
        interval_sec = SMART_CFG["MANUAL_INTERVAL_FALLBACK_SEC"]

    out = [_jitter(now + timedelta(seconds=i * interval_sec)) for i in range(count)]
    out.sort()
    return out


def _persist_run_sheet(slots: List[datetime], mode: str, platforms: List[str]) -> None:
    try:
        payload = {
            "date": date.today().isoformat(),
            "mode": mode,
            "platforms": platforms,
            "slots": [dt.isoformat() for dt in slots],
        }
        RUN_SHEET_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass


def _load_run_sheet_if_today(mode: str, platforms: List[str]) -> Optional[List[datetime]]:
    try:
        if not RUN_SHEET_FILE.exists():
            return None
        payload = json.loads(RUN_SHEET_FILE.read_text(encoding="utf-8"))
        if payload.get("date") != date.today().isoformat():
            return None
        if (payload.get("mode") or "") != mode:
            return None
        if (payload.get("platforms") or []) != platforms:
            return None
        return [datetime.fromisoformat(x) for x in payload.get("slots", [])]
    except Exception:
        return None


# ===================== CORE MASTER FLOW (OAUTH/TUNNEL/PORT LOGIC UNCHANGED) =====================

def platform_defaults_interval_seconds(platforms: List[str]) -> int:
    # kept for backward compatibility; no longer drives timing when run_sheet is used
    defaults = {
        "tiktok": 3 * 3600,
        "instagram": 3 * 3600,
        "facebook": 3 * 3600,
        "youtube": 3 * 3600,
    }
    return max(defaults.get(p, 3 * 3600) for p in platforms) if platforms else 0


def parse_platform_selection(raw: str) -> List[str]:
    mapping = {
        "1": "tiktok",
        "2": "instagram",
        "3": "facebook",
        "4": "youtube",
        "tiktok": "tiktok",
        "tt": "tiktok",
        "t": "tiktok",
        "instagram": "instagram",
        "ig": "instagram",
        "i": "instagram",
        "facebook": "facebook",
        "fb": "facebook",
        "f": "facebook",
        "youtube": "youtube",
        "yt": "youtube",
        "y": "youtube",
    }
    tokens = re.split(r"[,\s]+", (raw or "").strip().lower())
    out = []
    for tok in tokens:
        if not tok:
            continue
        if tok in mapping:
            out.append(mapping[tok])
            continue
        for key in ("tiktok", "instagram", "facebook", "youtube"):
            if key.startswith(tok):
                out.append(key)
                break
    seen = set()
    final = []
    for p in out:
        if p not in seen:
            seen.add(p)
            final.append(p)
    return final


def menu() -> None:
    print("Select platforms by name, number, or letter (comma/space separated):")
    print("  1) TikTok     2) Instagram  3) Facebook   4) YouTube")


def main() -> None:
    env_path = ensure_env_exists()
    load_dotenv(env_path)

    videos_dir = Path(os.getenv("VIDEOS_DIR", "videos"))
    if not videos_dir.is_absolute():
        videos_dir = (ROOT / videos_dir).resolve()

    titles_path = ROOT / "titles.txt"
    captions_path = ROOT / "captions.txt"
    hashtags_path = ROOT / "hashtag.txt"

    titles = read_lines(titles_path)
    captions = read_lines(captions_path)
    hashtags = read_lines(hashtags_path)

    oauth_host = os.getenv("OAUTH_LOCAL_HOST", "127.0.0.1")
    oauth_port = int(os.getenv("OAUTH_LOCAL_PORT", "8421"))

    auth = UnifiedAuthManager(env_path=str(env_path), host=oauth_host, port=oauth_port)
    print(f"[AUTH] Loopback OAuth server running on http://{oauth_host}:{oauth_port}/health")

    webhook = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

    try:
        while True:
            menu()
            raw = input("Your selection: ").strip()
            platforms = parse_platform_selection(raw)
            if not platforms:
                print("[WARN] No valid selection. Use 1-4 or platform names.")
                continue

            # Privacy (uniform input; mapped per module)
            privacy = normalize_privacy_plus(
                input("Upload privacy [1=public, 2=private]: ").strip() or "1",
                default="public",
            )

            mp4s = sorted(videos_dir.glob("*.mp4"))
            if not mp4s:
                print(f"[BLOCKER] No .mp4 files found in: {videos_dir}")
                continue

            # Count mode (smart = MIN cap across selected platforms)
            count_mode = _pick_mode(input("How many videos? [1=smart, 2=manual]: ").strip(), default="smart")
            if count_mode == "smart":
                count = _smart_count_min_cap(platforms, len(mp4s))
            else:
                try:
                    count = int(input("Enter number of videos to upload: ").strip())
                except Exception:
                    count = 1
            count = max(1, min(count, len(mp4s)))

            # Timing mode (smart slots w/ jitter OR manual interval/slots/window)
            timing_mode = _pick_mode(input("Upload timing [1=smart, 2=manual]: ").strip(), default="smart")

            _seed_daily()
            now = datetime.now()

            run_sheet = _load_run_sheet_if_today(timing_mode, platforms)
            if (not run_sheet) or (len(run_sheet) < count):
                if timing_mode == "smart":
                    run_sheet = _smart_schedule(platforms, count, now)
                else:
                    user_sched = input(
                        "Manual timing: interval (10s/15m/3h) OR window (08:00-23:00) OR slots (07:00,12:30,...): "
                    ).strip()
                    run_sheet = _manual_schedule(user_sched, count, now)

                _persist_run_sheet(run_sheet, timing_mode, platforms)

            # IG tunnel starts only if IG selected (unchanged logic)
            tunnel: Optional[CloudflaredTunnel] = None
            ig_public_base = os.getenv("IG_PUBLIC_BASE_URL", "").strip()

            if "instagram" in platforms:
                tunnel = CloudflaredTunnel(local_host=oauth_host, local_port=oauth_port, timeout_seconds=45)
                ig_public_base = tunnel.ensure_started()
                if not ig_public_base:
                    print("[IG] Tunnel unavailable. Instagram will be skipped (no crash).")
                    platforms = [p for p in platforms if p != "instagram"]
                else:
                    auth.write_env("IG_PUBLIC_BASE_URL", ig_public_base)

            # Telemetry ranking (optional; only changes mp4 order when TELEMETRY_ENABLED=1)
            telemetry_meta: Dict[str, dict] = {}
            try:
                mp4s, telemetry_meta = telemetry_analyze_all(mp4s, videos_dir)
            except Exception:
                telemetry_meta = {}

            print("\n[PLAN]")
            print(f"  Platforms: {', '.join(platforms)}")
            print(f"  Videos: {count} (from {videos_dir})")
            print(f"  Privacy: {privacy.upper()}")
            print(f"  Timing: {timing_mode.upper()} (jittered run sheet)")
            print("  Slots:")
            for i in range(min(count, len(run_sheet))):
                print(f"    - {run_sheet[i].strftime('%Y-%m-%d %H:%M:%S')}")
            print("")

            # Ensure auth only when needed (unchanged)
            if "tiktok" in platforms:
                auth.ensure_tiktok()
            if ("facebook" in platforms) or ("instagram" in platforms):
                auth.ensure_meta()

            for idx in range(count):
                # Wait until scheduled slot for this job (replaces fixed interval)
                target_dt = run_sheet[idx]
                wait_sec = max(0, int((target_dt - datetime.now()).total_seconds()))
                if wait_sec > 0:
                    countdown(wait_sec)

                video_path = mp4s[idx]

                # Default text inputs (existing behavior)
                title = cycle_pick(titles, idx)
                cap = cycle_pick(captions, idx)
                tag = cycle_pick(hashtags, idx)
                final_caption = (cap + ("\n\n" + tag if tag else "")).strip()

                # Optional telemetry-driven text (only if meta exists)
                tmeta = telemetry_meta.get(video_path.name)
                if tmeta:
                    title = tmeta.get("title") or title
                    tele_caption = (tmeta.get("caption") or "").strip()
                    tele_tags = tmeta.get("hashtags") or []
                    if tele_caption:
                        final_caption = tele_caption
                        if tele_tags:
                            final_caption = (final_caption + "\n\n" + " ".join(tele_tags)).strip()

                # Optional HUD file selection (generated between uploads)
                upload_fp = video_path
                if _env_flag("HUD_ENABLED", "0"):
                    hud_candidate = (ROOT / "generated" / f"{video_path.stem}_HUD.mp4")
                    if hud_candidate.exists():
                        upload_fp = hud_candidate

                # Telemetry printout (visibility only)
                if tmeta:
                    print(
                        f"[TRILL] score={tmeta.get('trill_score')} "
                        f"bucket={tmeta.get('speed_bucket')} "
                        f"place={tmeta.get('place_name')},{tmeta.get('state')} "
                        f"elev_gain_m={tmeta.get('elev_gain_m')} "
                        f"protected={tmeta.get('near_protected')}"
                    )

                print("\n" + "=" * 70)
                print(f"[JOB] #{idx+1}/{count}")
                print(f"[FILE] {video_path.name}")
                print(f"[TITLE] {title}")
                print(f"[CAPTION] {final_caption[:180]}{'...' if len(final_caption) > 180 else ''}")
                print(f"[PRIVACY] {privacy.upper()}")

                results = []

                # TikTok
                if "tiktok" in platforms:
                    try:
                        r = upload_tiktok_video(video_path=str(upload_fp), title=title, caption=final_caption, privacy=privacy)
                        results.append(r)
                        print(f"[TT] OK  url={r.get('url') or 'n/a'}  id={r.get('id') or 'n/a'}")
                    except Exception as e:
                        print(f"[TT] FAIL {e}")
                        results.append({"platform": "tiktok", "ok": False, "error": str(e)})

                # Facebook
                if "facebook" in platforms:
                    try:
                        r = upload_facebook_video(video_path=str(upload_fp), title=title, caption=final_caption, privacy=privacy)
                        results.append(r)
                        print(f"[FB] OK  url={r.get('url') or 'n/a'}  id={r.get('id') or 'n/a'}")
                    except Exception as e:
                        print(f"[FB] FAIL {e}")
                        results.append({"platform": "facebook", "ok": False, "error": str(e)})

                # Instagram
                if "instagram" in platforms:
                    try:
                        r = upload_instagram_reel(
                            video_path=str(upload_fp),
                            title=title,
                            caption=final_caption,
                            privacy=privacy,
                            public_base_url=ig_public_base,
                            oauth_server=auth.server,
                        )
                        results.append(r)
                        print(f"[IG] OK  url={r.get('url') or 'n/a'}  id={r.get('id') or 'n/a'}")
                    except Exception as e:
                        print(f"[IG] FAIL {e}")
                        results.append({"platform": "instagram", "ok": False, "error": str(e)})

                # YouTube
                if "youtube" in platforms:
                    try:
                        r = upload_youtube_short(video_path=str(upload_fp), title=title, caption=final_caption, privacy=privacy, root=str(ROOT))
                        results.append(r)
                        print(f"[YT] OK  url={r.get('url') or 'n/a'}  id={r.get('id') or 'n/a'}")
                    except Exception as e:
                        print(f"[YT] FAIL {e}")
                        results.append({"platform": "youtube", "ok": False, "error": str(e)})

                # Discord payload (structured, per upload batch)
                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                embed_fields = [
                    {"name": "Privacy (requested)", "value": privacy.upper(), "inline": True},
                    {"name": "Uploaded at", "value": ts, "inline": True},
                    {"name": "Video", "value": video_path.name, "inline": False},
                ]
                for r in results:
                    plat = r.get("platform", "unknown").upper()
                    ok = r.get("ok", False)
                    url = r.get("url") or "n/a"
                    mode = r.get("mode") or "n/a"
                    applied_priv = r.get("privacy_applied") or "n/a"
                    embed_fields.append(
                        {
                            "name": f"{plat} {'OK' if ok else 'FAIL'}",
                            "value": f"mode={mode} | privacy={applied_priv} | url={url}",
                            "inline": False,
                        }
                    )

                discord_notify(
                    webhook,
                    {
                        "content": f"UploadM8 batch result: {video_path.name}",
                        "embeds": [
                            {
                                "title": title[:240] if title else "UploadM8 Upload",
                                "description": (final_caption[:400] + ("..." if len(final_caption) > 400 else "")) if final_caption else "",
                                "fields": embed_fields,
                            }
                        ],
                    },
                )

                # Preflight HUD for next candidate while between uploads (unchanged behavior; still optional)
                if idx < (count - 1):
                    try:
                        telemetry_preflight_hud(mp4s[idx + 1], videos_dir)
                    except Exception:
                        pass

            if tunnel:
                tunnel.stop()

            again = input("\nRun another module set? [y]/n: ").strip().lower()
            if again in ("n", "no"):
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted (Ctrl+C). Exiting.")
    finally:
        try:
            auth.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
