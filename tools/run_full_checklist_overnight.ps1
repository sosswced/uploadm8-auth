<#
.SYNOPSIS
  Full-app ~500-point checklist: headed Chrome, paced API, Excel report.

.EXAMPLE
  .\tools\run_full_checklist_overnight.ps1
#>
[CmdletBinding()]
param(
    [string] $BaseUrl = "http://127.0.0.1:8000",
    [switch] $IncludeSlowApi
)

$ErrorActionPreference = "Continue"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

python -m pip install -q -r requirements-e2e.txt
python -m playwright install chromium 2>$null

$env:E2E_BASE_URL = $BaseUrl
$env:E2E_HEADED = "1"
$env:E2E_FORCE_RELOGIN = "1"
$env:E2E_REQUEST_DELAY_MS = "450"
$env:E2E_REQUEST_JITTER_MS = "200"
$env:E2E_CLICK_DELAY_MS = "350"
$env:E2E_SLOW_MO_MS = "150"
$env:E2E_SMOKE_READ_TIMEOUT_S = "12"
# Loopback bypasses Redis rate limits; pacing avoids hammering DB/worker.
$env:RATE_LIMIT_LOOPBACK_BYPASS = "1"

$args = @("scripts/run_full_app_checklist.py", "--headed")
if ($IncludeSlowApi) { $args += "--include-slow-api" }

$log = "tests\e2e\artifacts\overnight_run_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
Write-Host "Logging to $log" -ForegroundColor Cyan
Write-Host "Excel will appear in tests\e2e\artifacts\UploadM8_Full_App_Checklist_*.xlsx" -ForegroundColor Green

python @args 2>&1 | Tee-Object -FilePath $log
exit $LASTEXITCODE
