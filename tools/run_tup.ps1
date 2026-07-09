# Run /TUP — Test Upload Pipeline (live journey + heal)
param(
    [string]$Video = $env:E2E_TEST_VIDEO,
    [string]$Telemetry = $env:E2E_TEST_TELEMETRY_MAP,
    [int]$PipelineTimeoutMin = 120,
    [switch]$SkipOvernightPytest,
    [switch]$HealOnly,
    [switch]$Headless,
    [switch]$ForcePikzels,
    [switch]$SkipPikzels,
    [switch]$ResetPikzels,
    [switch]$Status
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$argsList = @("scripts/agent/tup.py", "--json")
if ($Status) { $argsList = @("scripts/agent/tup.py", "--status"); & python @argsList; exit $LASTEXITCODE }
if ($ResetPikzels) { $argsList = @("scripts/agent/tup.py", "--reset-pikzels", "--json"); & python @argsList; exit $LASTEXITCODE }
if ($HealOnly) { $argsList += "--heal-only" }
if ($SkipOvernightPytest) { $argsList += "--skip-overnight-pytest" }
if ($Headless) { $argsList += "--headless" }
if ($ForcePikzels) { $argsList += "--force-pikzels" }
if ($SkipPikzels) { $argsList += "--skip-pikzels" }
if ($Video) { $argsList += @("--video", $Video) }
if ($Telemetry) { $argsList += @("--telemetry", $Telemetry) }
$argsList += @("--pipeline-timeout-min", "$PipelineTimeoutMin")

Write-Host "TUP → python $($argsList -join ' ')"
& python @argsList
exit $LASTEXITCODE
