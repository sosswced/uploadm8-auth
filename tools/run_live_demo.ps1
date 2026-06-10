<#
.SYNOPSIS
  Unified session: master-admin login → upload → admin API + Johnny UI + full app tour
  WHILE upload is pending → verify queue/dashboard.

.EXAMPLE
  .\tools\run_live_demo.ps1
  .\tools\run_live_demo.ps1 -UploadTimeoutMin 90
#>
[CmdletBinding()]
param(
    [string] $BaseUrl = "http://127.0.0.1:8000",
    [string] $Video = "C:\Users\Earl\Videos\20250301_0058_CAM_EVNT.MP4",
    [string] $Telemetry = "C:\Users\Earl\Videos\20250301_0058_CAM_EVNT.map",
    [int] $UploadTimeoutMin = 120
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

python -m pip install -q -r requirements-e2e.txt
python -m playwright install chromium 2>$null

$env:E2E_BASE_URL = $BaseUrl
$env:E2E_TEST_VIDEO = $Video
$env:E2E_TEST_TELEMETRY_MAP = $Telemetry
$env:E2E_HEADED = "1"
$env:E2E_FORCE_RELOGIN = "0"
$env:E2E_HUMAN_LOGIN = "1"
$env:E2E_SLOW_MO_MS = "150"
$env:E2E_CLICK_DELAY_MS = "500"
$env:E2E_REQUEST_DELAY_MS = "1200"
$env:E2E_WORKER_SAFE = "1"
$env:E2E_UPLOAD_QUIET_SEC = "90"
$env:E2E_UPLOAD_PAGE_DELAY_MS = "10000"
$env:E2E_WORKER_SAFE_API_PER_PAGE = "2"
$env:E2E_WORKER_SAFE_MAX_CLICKS = "4"
$env:E2E_UPLOAD_POLL_SEC = "60"
$env:RATE_LIMIT_LOOPBACK_BYPASS = "1"
$env:E2E_TARGET_USER_ID = "ae995094-abb6-4a41-8d51-460ca8f0fd8c"
$env:E2E_TARGET_USER_NAME = "Johnny Omeadows"

Write-Host "Single Chrome window only - stop overnight/checklist Playwright runs first." -ForegroundColor Yellow

python scripts/run_live_demo_journey.py `
    --base-url $BaseUrl `
    --video $Video `
    --telemetry $Telemetry `
    --pipeline-timeout-min $UploadTimeoutMin

exit $LASTEXITCODE
