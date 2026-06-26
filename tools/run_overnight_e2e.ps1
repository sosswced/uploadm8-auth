<#
.SYNOPSIS
  Run the full UploadM8 overnight Playwright + API E2E suite.

.EXAMPLE
  # API + worker already running on :8000; credentials in .env
  .\tools\run_overnight_e2e.ps1

.EXAMPLE
  .\tools\run_overnight_e2e.ps1 -Video "D:\clips\test.mp4" -Headed

.EXAMPLE
  .\tools\run_overnight_e2e.ps1 -ApiOnly
#>
[CmdletBinding()]
param(
    [string] $BaseUrl = "http://127.0.0.1:8000",
    [string] $Video = "",
    [switch] $Headed,
    [switch] $ApiOnly,
    [switch] $ForceRelogin,
    [string] $ReportDir = "tests\e2e\artifacts"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

if (-not (Test-Path "requirements-e2e.txt")) {
    Write-Error "Run from uploadm8-auth repo root (requirements-e2e.txt missing)."
}

$py = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

Write-Host "Acquiring pipeline lock (do not run checklist + overnight E2E in parallel)..." -ForegroundColor Cyan
& $py scripts/agent/pipeline_lock.py acquire --name overnight_e2e
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Installing E2E deps (if needed)..." -ForegroundColor Cyan
python -m pip install -q -r requirements-e2e.txt
python -m playwright install chromium 2>$null

$env:E2E_BASE_URL = $BaseUrl
if ($Headed) { $env:E2E_HEADED = "1" }
if ($ForceRelogin) { $env:E2E_FORCE_RELOGIN = "1" }
if ($Video) { $env:E2E_TEST_VIDEO = (Resolve-Path -LiteralPath $Video).Path }

New-Item -ItemType Directory -Force -Path $ReportDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$report = Join-Path $ReportDir "overnight_$stamp.html"
$junit = Join-Path $ReportDir "overnight_$stamp.xml"

Write-Host "Probing API at $BaseUrl ..." -ForegroundColor Cyan
try {
    Invoke-WebRequest -Uri "$BaseUrl/api/auth/session-probe" -UseBasicParsing -TimeoutSec 15 | Out-Null
} catch {
    Write-Error "API not reachable at $BaseUrl. Start: python -m uvicorn app:app --host 127.0.0.1 --port 8000"
}

$marker = if ($ApiOnly) { "api_smoke" } else { "overnight" }
$pytestArgs = @(
    "run_tests.py", "e2e",
    "-m", $marker,
    "--html=$report", "--self-contained-html",
    "--junitxml=$junit",
    "-v", "--tb=short",
    "--timeout=600"
)

Write-Host "Starting overnight suite (marker=$marker)..." -ForegroundColor Green
Write-Host "  HTML report: $report"
try {
    python @pytestArgs
    $code = $LASTEXITCODE
} finally {
    & $py scripts/agent/pipeline_lock.py release | Out-Null
}

Write-Host ""
if ($code -eq 0) {
    Write-Host "Overnight E2E finished OK." -ForegroundColor Green
} else {
    Write-Host "Overnight E2E finished with failures (exit $code). See $report" -ForegroundColor Yellow
}
Write-Host "JUnit: $junit"
exit $code
