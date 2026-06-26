<#
.SYNOPSIS
  Human App Tour — ONE Chrome window, scroll + click every page, full API sweep, Excel.

  Real-user pace. Loopback rate-limit bypass. No extra browsers or pytest windows.

.EXAMPLE
  .\tools\run_human_app_tour.ps1
  .\tools\run_human_app_tour.ps1 -WithUpload
#>
[CmdletBinding()]
param(
    [string] $BaseUrl = "http://127.0.0.1:8000",
    [switch] $WithUpload,
    [switch] $IncludeSlowApi
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

python -m pip install -q -r requirements-e2e.txt
python -m playwright install chromium 2>$null

$env:E2E_BASE_URL = $BaseUrl
$env:E2E_HEADED = "1"
$env:RATE_LIMIT_LOOPBACK_BYPASS = "1"
$env:E2E_WORKER_SAFE = "0"
$env:E2E_REQUEST_DELAY_MS = "450"
$env:E2E_REQUEST_JITTER_MS = "200"
$env:E2E_CLICK_DELAY_MS = "350"
$env:E2E_SLOW_MO_MS = "120"
$env:E2E_TOUR_PAGE_DELAY_MS = "800"
$env:E2E_TOUR_MAX_CLICKS = "28"
$env:E2E_TOUR_ADMIN_MAX_CLICKS = "35"
$env:E2E_API_PER_PAGE = "12"
$env:E2E_SMOKE_READ_TIMEOUT_S = "12"

Write-Host ""
Write-Host "Human App Tour — single Chrome window" -ForegroundColor Cyan
Write-Host "  Rate limit bypass: ON (loopback)" -ForegroundColor Green
Write-Host "  Excel: tests\e2e\artifacts\UploadM8_Human_App_Tour_*.xlsx" -ForegroundColor Green
Write-Host "  Close other Playwright / checklist runs first." -ForegroundColor Yellow
Write-Host ""

$pyArgs = @("scripts/run_human_app_tour.py", "--base-url", $BaseUrl, "--keep-open-sec", "90")
if ($WithUpload) { $pyArgs += "--with-upload" }
if ($IncludeSlowApi) { $pyArgs += "--include-slow-api" }

python @pyArgs
exit $LASTEXITCODE
