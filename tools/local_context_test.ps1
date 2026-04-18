<#
.SYNOPSIS
  Run tools/simulate_full_pipeline.py with sane local defaults: real .env, minimal external vision APIs,
  audio + thumbnail + M8 title/caption/hashtags. Optional telemetry .map for Trill/GPS context.

.EXAMPLE
  .\tools\local_context_test.ps1 -Video "D:\clips\run1.mp4" -TelemetryMap "D:\clips\run1.map"

.EXAMPLE
  .\tools\local_context_test.ps1 -Video ".\samples\a.mp4" -IncludeVision -IncludeTwelveLabs

.EXAMPLE
  # Run Vision + Twelve Labs + Video Intelligence (respects .env; stages skip with a log if not configured)
  .\tools\local_context_test.ps1 -Video ".\clip.mp4" -TryAllWorkerServices
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, HelpMessage = "Path to the MP4 to process")]
    [string] $Video,

    [Parameter(HelpMessage = "Path to a .map file (paired with --trill-enabled)")]
    [string] $TelemetryMap = "",

    [switch] $TryAllWorkerServices,

    [switch] $IncludeVision,
    [switch] $IncludeTwelveLabs,
    [switch] $IncludeVideoIntelligence,

    [string] $Platforms = "youtube,tiktok",
    [string] $Tier = "creator_pro",
    [string] $DebugDumpDir = "",

    [switch] $ShowBilling,
    [switch] $NoStyledThumb
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

if (-not (Test-Path -LiteralPath $Video)) {
    Write-Error "Video not found: $Video"
}
$videoResolved = (Resolve-Path -LiteralPath $Video).Path

$pyArgs = @(
    "tools/simulate_full_pipeline.py",
    $videoResolved,
    "--platforms", $Platforms,
    "--tier", $Tier
)

if ($TelemetryMap) {
    if (-not (Test-Path -LiteralPath $TelemetryMap)) {
        Write-Error "Telemetry map not found: $TelemetryMap"
    }
    $pyArgs += @("--telemetry-map", (Resolve-Path -LiteralPath $TelemetryMap).Path, "--trill-enabled")
}

if ($TryAllWorkerServices) {
    $IncludeVision = $true
    $IncludeTwelveLabs = $true
    $IncludeVideoIntelligence = $true
}

if (-not $IncludeVision) { $pyArgs += "--skip-vision" }
if (-not $IncludeTwelveLabs) { $pyArgs += "--skip-12labs" }
if (-not $IncludeVideoIntelligence) { $pyArgs += "--skip-video-intelligence" }

if (-not $ShowBilling) { $pyArgs += "--no-billing" }
if ($NoStyledThumb) { $pyArgs += "--no-styled-thumb" }

if ($DebugDumpDir) {
    $d = $DebugDumpDir.Trim()
    if ($d) { $pyArgs += @("--debug-dump-dir", $d) }
}

Write-Host "Repo: $repoRoot" -ForegroundColor DarkGray
Write-Host "python $($pyArgs -join ' ')" -ForegroundColor Cyan
& python @pyArgs
exit $LASTEXITCODE
