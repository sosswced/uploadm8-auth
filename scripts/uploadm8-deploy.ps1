<#
.SYNOPSIS
  UploadM8 — regenerate product card PNGs and optionally run Stripe PowerShell sync.

.DESCRIPTION
  GenerateImages  Runs python scripts/generate_product_cards.py → frontend/images
  SyncStripe       Patches and runs frontend/images/stripe-update-all-products.ps1 if present
  AddAnnual        (Optional) same as legacy script block for yearly prices

.PARAMETER RepoRoot
  Path to uploadm8-auth repo root. Defaults to parent of this script's directory.

.PARAMETER Steps
  Comma-separated: GenerateImages, SyncStripe, AddAnnual, or All

.PARAMETER StripeKey
  Stripe secret key for SyncStripe. If empty, uses env STRIPE_SECRET_KEY / STRIPE_API_KEY.

.PARAMETER PythonExe
  Python executable. Default: python
#>

[CmdletBinding()]
param(
    [string]$Steps = 'GenerateImages',
    [string]$RepoRoot = '',
    [string]$StripeKey = '',
    [string]$PythonExe = 'python'
)

$ErrorActionPreference = 'Stop'

if (-not $RepoRoot) {
    $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
}

$ImagesDir = Join-Path $RepoRoot 'frontend\images'
$GenScript = Join-Path $RepoRoot 'scripts\generate_product_cards.py'

function Test-StepRequested([string]$Name) {
    if ($Steps -eq 'All') { return $true }
    $list = $Steps -split ',' | ForEach-Object { $_.Trim() }
    return $list -contains $Name
}

function Write-Section([string]$Title) {
    Write-Host ""
    Write-Host "=== $Title ===" -ForegroundColor Cyan
}

if (Test-StepRequested 'GenerateImages') {
    Write-Section 'GenerateImages'
    if (-not (Test-Path $GenScript)) { throw "Missing $GenScript" }
    & $PythonExe $GenScript --out $ImagesDir
    if ($LASTEXITCODE -ne 0) { throw "generate_product_cards failed: $LASTEXITCODE" }
    Write-Host "Cards written to $ImagesDir" -ForegroundColor Green
}

if (Test-StepRequested 'SyncStripe') {
    Write-Section 'SyncStripe'
    $key = $StripeKey
    if (-not $key) { $key = $env:STRIPE_SECRET_KEY }
    if (-not $key) { $key = $env:STRIPE_API_KEY }
    if (-not $key) { throw "SyncStripe requires -StripeKey or STRIPE_SECRET_KEY in environment" }
    $sync = Join-Path $ImagesDir 'stripe-update-all-products.ps1'
    if (-not (Test-Path $sync)) {
        Write-Host "stripe-update-all-products.ps1 not found at $sync — skipping." -ForegroundColor Yellow
    }
    else {
        $body = Get-Content $sync -Raw
        $body = $body -replace '\$ApiKey\s*=\s*"sk_test[^"]*"',
            ('$ApiKey = "' + $key + '"')
        $escaped = ($ImagesDir -replace '\\', '\\')
        $body = $body -replace '\$ImageFolder\s*=\s*"[^"]*"',
            ('$ImageFolder = "' + $escaped + '"')
        $tempScript = Join-Path $env:TEMP ("stripe-sync-" + [guid]::NewGuid().ToString('N').Substring(0, 8) + ".ps1")
        Set-Content -Path $tempScript -Value $body -Encoding UTF8
        & powershell -NoProfile -ExecutionPolicy Bypass -File $tempScript
        Remove-Item $tempScript -Force -ErrorAction SilentlyContinue
        Write-Host "Stripe sync finished." -ForegroundColor Green
    }
}

if (Test-StepRequested 'AddAnnual') {
    Write-Host "AddAnnual: use legacy uploadm8-deploy.ps1 block or Stripe Dashboard — not wired in this slim script." -ForegroundColor Yellow
}

Write-Host "Done." -ForegroundColor Green
