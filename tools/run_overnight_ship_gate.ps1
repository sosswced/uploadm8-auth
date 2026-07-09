<#
.SYNOPSIS
  Final live-ready gate before /333 — evals, router lint, checklist Excel, TRILL pre-ship.

.DESCRIPTION
  Scriptable checks + agent flags. Sentry clearance comes from
  tests/e2e/artifacts/sentry_triage.json with mcp_verified=true
  (Phase 6: MCP search_issues → sentry_triage.py --record --mcp-verified).
  - /parallel-audit bugbot: no critical findings (-BugbotClear)
  - Self-heal green (-SelfHealed)

.EXAMPLE
  # After overnight Phase 6 wrote mcp_verified artifact + bugbot/self-heal confirmed:
  .\tools\run_overnight_ship_gate.ps1 -BugbotClear -SelfHealed

.EXAMPLE
  .\tools\run_overnight_ship_gate.ps1 -Checklist "tests\e2e\artifacts\UploadM8_Full_App_Checklist_20260609_143426.xlsx" -OvernightJUnit "tests\e2e\artifacts\overnight_report.xml"
#>
[CmdletBinding()]
param(
    [string] $Checklist = "",
    [string] $OvernightJUnit = "tests\e2e\artifacts\overnight_report.xml",
    [switch] $SkipChecklist,
    [switch] $SentryCleared,
    [switch] $AllowSentryFlagOnly,
    [switch] $BugbotClear,
    [switch] $SelfHealed
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$args = @("scripts/agent/ship_gate.py", "--json")
if ($Checklist) { $args += @("--checklist", $Checklist) }
if ($SkipChecklist) { $args += "--skip-checklist" }
if ($OvernightJUnit -and (Test-Path -LiteralPath $OvernightJUnit)) {
    $args += @("--require-overnight-junit", $OvernightJUnit)
}
if ($SentryCleared) { $args += "--sentry-cleared" }
if ($AllowSentryFlagOnly) { $args += "--allow-sentry-flag-only" }
if ($BugbotClear) { $args += "--bugbot-clear" }
if ($SelfHealed) { $args += "--self-healed" }

Write-Host "Ship gate (live-ready check before /333)..." -ForegroundColor Cyan
python @args
$code = $LASTEXITCODE

if ($code -eq 0) {
    Write-Host "GREEN — safe to run /333 when user asks to deploy." -ForegroundColor Green
} else {
    Write-Host "BLOCKED — fix blockers above; do not /333." -ForegroundColor Yellow
}
exit $code
