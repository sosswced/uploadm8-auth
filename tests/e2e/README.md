# UploadM8 overnight E2E (Playwright + API)

Master-admin sweep of **every static page**, **sidebar link**, **safe in-page controls**, and **all read-only GET routes** from OpenAPI.

## Prerequisites

1. API running (serves frontend on the same origin):

   ```powershell
   python -m uvicorn app:app --host 127.0.0.1 --port 8000
   ```

2. Worker running if you want upload pipeline to finish (optional for page-load tests).

3. Add to `.env`:

   ```env
   E2E_MASTER_ADMIN_EMAIL=you@example.com
   E2E_MASTER_ADMIN_PASSWORD=your-master-admin-password
   E2E_BASE_URL=http://127.0.0.1:8000
   E2E_TEST_VIDEO=C:\Users\Earl\Videos\your-clip.MP4
   E2E_TEST_TELEMETRY_MAP=C:\Users\Earl\Videos\your-clip.map
   ```

   `E2E_MASTER_ADMIN_EMAIL` falls back to `BOOTSTRAP_ADMIN_EMAIL` if unset.

4. Install browsers:

   ```powershell
   pip install -r requirements-e2e.txt
   python -m playwright install chromium
   ```

## Run overnight

```powershell
.\tools\run_overnight_e2e.ps1 -Video "D:\clips\test.mp4"
```

Or manually:

```powershell
python run_tests.py overnight -v --html=tests/e2e/artifacts/report.html --self-contained-html
```

## What runs

| Suite | Marker | Description |
|-------|--------|-------------|
| API GET smoke | `api_smoke` | Every safe `GET /api/*` from OpenAPI (404/422 OK; 5xx fails) |
| Page load | `ui_smoke` | All authenticated + public HTML pages |
| Sidebar | `ui_smoke` | Click every sidebar nav link as master admin |
| Safe clicks | `ui_clicks` | Tabs/toggles on key pages (skips delete/logout/charge) |
| Upload UI | `upload_ui` | Drop test MP4 on upload.html (needs `E2E_TEST_VIDEO`) |

API-only (faster):

```powershell
.\tools\run_overnight_e2e.ps1 -ApiOnly
```

## Live demo journey (upload + browse + verify)

**One module:** `tests/e2e/helpers/live_demo.py`  
**One browser helper:** `tests/e2e/helpers/browser_session.py` → `SingleBrowserSession`  
**CLI:** `scripts/run_live_demo_journey.py` (thin wrapper — does not launch its own browser)

Human headed run: login form, video + `.map` upload, then on **every page while the worker processes** run router lint + OpenAPI GET smoke + target-user admin API, browse tabs/sidebar/safe clicks, then confirm on dashboard and queue:

```powershell
.\tools\run_live_demo.ps1
# or
python scripts/run_live_demo_journey.py --pipeline-timeout-min 120 --video "C:\Users\Earl\Videos\20250301_0058_CAM_EVNT.MP4" --telemetry "C:\Users\Earl\Videos\20250301_0058_CAM_EVNT.map"
```

Optional: `--include-slow-api`, `--api-per-page 12`, `--skip-api-smoke`

Upload platform defaults to **all connected platforms** under `/TUP` (`E2E_UPLOAD_PLATFORMS=all`).
Legacy live-demo without `E2E_TUP` still defaults to TikTok. Set `E2E_TIKTOK_PROFILE=@yourhandle` when multiple TikTok accounts are connected.
Persona is applied when linked (`E2E_USE_PERSONA=1`). Pikzels engine runs **once per setup cycle** (see `tup_pikzels_setup.json`).

**Worker-safe mode** (`E2E_WORKER_SAFE=1`, default): 90s quiet period after publish, ~10s between pages, 2 API checks/page, skips ML/KPI-heavy pages until upload finishes. Prevents starving `worker.py` FFmpeg slots on a dev machine.

Requires uvicorn + `worker.py`. Report: `tests/e2e/artifacts/live_demo_*.json`

## Useful env flags

| Variable | Default | Purpose |
|----------|---------|---------|
| `E2E_HEADED=1` | headless | Show browser |
| `E2E_FORCE_RELOGIN=1` | reuse cache | Fresh login form + storage state |
| `E2E_HUMAN_LOGIN=1` | on when headed | Use login.html form (cookie-primary; no bearer injection) |
| `E2E_SKIP_API_SMOKE=1` | run API | Skip OpenAPI sweep |
| `E2E_SKIP_MUTATIONS=1` | skip writes | Reserved for future POST mutation tests |
| `E2E_INCLUDE_SLOW_API=1` | off | Include coach/KPI/ML-heavy GETs (much longer run) |
| `E2E_SMOKE_READ_TIMEOUT_S` | 12 | Per-endpoint read timeout for API sweep |

## Regenerate API catalog snapshot

```powershell
python scripts/generate_e2e_api_catalog.py
```

Writes `tests/e2e/catalog/read_endpoints.json` for review before expanding coverage.

## Artifacts

- HTML report: `tests/e2e/artifacts/overnight_*.html`
- JUnit XML: `tests/e2e/artifacts/overnight_*.xml`
- Auth cache: `tests/e2e/.auth/master_admin.json` (gitignored)
