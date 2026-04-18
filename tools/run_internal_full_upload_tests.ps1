param(
    [Parameter(Mandatory = $true)]
    [string]$VideoPath,

    [Parameter(Mandatory = $false)]
    [string]$TelemetryMapPath = "",

    [Parameter(Mandatory = $false)]
    [string]$OutputRoot = "tools\internal-test-results",

    [Parameter(Mandatory = $false)]
    [int]$RandomRuns = 2,

    [Parameter(Mandatory = $false)]
    [int]$SeedBase = 1000,

    [Parameter(Mandatory = $false)]
    [switch]$IncludeInternalTiers
,
    [Parameter(Mandatory = $false)]
    [ValidateSet("DEBUG", "INFO", "WARNING", "ERROR")]
    [string]$LogLevel = "DEBUG",

    [Parameter(Mandatory = $false)]
    [switch]$FullTranscode
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Ensure-PathExists([string]$PathValue, [string]$Label) {
    if (-not (Test-Path -Path $PathValue)) {
        throw "$Label does not exist: $PathValue"
    }
}

function Run-OneCase(
    [string]$Tier,
    [string]$CaseName,
    [string[]]$ExtraArgs,
    [string]$Video,
    [string]$MapPath,
    [string]$RunRoot
) {
    $safeCase = $CaseName -replace '[^a-zA-Z0-9_\-]', '_'
    $caseDir = Join-Path $RunRoot ("{0}_{1}" -f $Tier, $safeCase)
    $dumpDir = Join-Path $caseDir "ctx_dumps"
    New-Item -ItemType Directory -Force -Path $dumpDir | Out-Null

    $args = @(
        "tools/simulate_full_pipeline.py"
        $Video
        "--tier", $Tier
        "--platforms", "youtube,tiktok,instagram,facebook"
        "--privacy", "private"
        "--caption-frames", "18"
        "--caption-style", "punchy"
        "--caption-tone", "cinematic"
        "--caption-voice", "mentor"
        "--hashtag-style", "mixed"
        "--ai-hashtag-count", "5"
        "--max-hashtags", "20"
        "--always-hashtags", "tester,qwe"
        "--blocked-hashtags", "no"
        "--platform-tiktok", "1,2"
        "--platform-youtube", "3,4"
        "--platform-instagram", "5,6"
        "--platform-facebook", "7,8"
        "--thumbnail-interval", "10"
        "--openai-model", "gpt-4o-mini"
        "--debug-dump-dir", $dumpDir
        "--log-level", $LogLevel
        "--trace-decisions"
    )

    if ($MapPath -ne "") {
        $args += @("--telemetry-map", $MapPath)
    }
    if ($FullTranscode) {
        $args += @("--full-transcode")
    }

    if ($ExtraArgs -and $ExtraArgs.Count -gt 0) {
        $args += $ExtraArgs
    }

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "Tier: $Tier | Case: $CaseName"
    Write-Host "Dump dir: $dumpDir"
    Write-Host "============================================================"

    $logPath = Join-Path $caseDir "run.log"
    New-Item -ItemType Directory -Force -Path $caseDir | Out-Null

    # Run without PowerShell treating native stderr lines as terminating errors.
    $stdoutPath = Join-Path $caseDir "stdout.log"
    $stderrPath = Join-Path $caseDir "stderr.log"
    $argLine = ($args | ForEach-Object {
        if ($_ -match '\s') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ }
    }) -join " "

    $proc = Start-Process -FilePath "python" -ArgumentList $argLine -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath

    if (Test-Path $stdoutPath) {
        Get-Content -Path $stdoutPath | Tee-Object -FilePath $logPath | Out-Host
    }
    if (Test-Path $stderrPath) {
        Get-Content -Path $stderrPath | Tee-Object -FilePath $logPath -Append | Out-Host
    }

    if ($proc.ExitCode -ne 0) {
        throw "Case failed (tier=$Tier case=$CaseName). See: $logPath"
    }
}

Ensure-PathExists $VideoPath "VideoPath"
if ($TelemetryMapPath -ne "") {
    Ensure-PathExists $TelemetryMapPath "TelemetryMapPath"
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runRoot = Join-Path $OutputRoot ("run_{0}" -f $timestamp)
New-Item -ItemType Directory -Force -Path $runRoot | Out-Null

$tiers = @("free", "creator_lite", "creator_pro", "studio", "agency")
if ($IncludeInternalTiers) {
    $tiers += @("friends_family", "lifetime", "master_admin")
}

Write-Host "Running internal full upload-flow tests..."
Write-Host "Run root: $runRoot"
Write-Host "Tiers: $($tiers -join ', ')"
Write-Host "LogLevel: $LogLevel | FullTranscode: $FullTranscode"

foreach ($tier in $tiers) {
    # Baseline full-flow case (no randomization)
    $baseArgs = @()

    if ($tier -eq "free") {
        $baseArgs += @(
            "--no-trill-enabled",
            "--no-hud-enabled",
            "--no-billing-use-ai"
        )
    } else {
        $baseArgs += @(
            "--trill-enabled",
            "--trill-ai-enhance",
            "--trill-min-score", "60"
        )

        # HUD only where tier supports it (creator_pro+; simulator will skip if not allowed)
        if ($tier -in @("creator_pro", "studio", "agency", "friends_family", "lifetime", "master_admin")) {
            $baseArgs += @("--hud-enabled", "--billing-hud")
        } else {
            $baseArgs += @("--no-hud-enabled")
        }
    }

    if ($TelemetryMapPath -eq "") {
        # If no map provided, force trill off to avoid noisy "no telemetry file" paths.
        $baseArgs += @("--no-trill-enabled", "--no-hud-enabled")
    }

    Run-OneCase -Tier $tier -CaseName "baseline_full_flow" -ExtraArgs $baseArgs -Video $VideoPath -MapPath $TelemetryMapPath -RunRoot $runRoot

    # Randomized stress case for caption/hashtag preferences
    $randArgs = @(
        "--randomize-caption-ai",
        "--random-runs", "$RandomRuns",
        "--random-seed", "$SeedBase"
    )
    if ($tier -eq "free") {
        $randArgs += @("--no-billing-use-ai", "--no-trill-enabled", "--no-hud-enabled")
    } else {
        $randArgs += @("--trill-enabled")
        if ($tier -in @("creator_pro", "studio", "agency", "friends_family", "lifetime", "master_admin")) {
            $randArgs += @("--hud-enabled", "--billing-hud")
        } else {
            $randArgs += @("--no-hud-enabled")
        }
    }
    if ($TelemetryMapPath -eq "") {
        $randArgs += @("--no-trill-enabled", "--no-hud-enabled")
    }

    Run-OneCase -Tier $tier -CaseName "randomized_caption_hashtag" -ExtraArgs $randArgs -Video $VideoPath -MapPath $TelemetryMapPath -RunRoot $runRoot
}

Write-Host ""
Write-Host "All cases passed."
Write-Host "Artifacts:"
Write-Host "  $runRoot"
Write-Host ""
Write-Host "Each case contains:"
Write-Host "  - run.log"
Write-Host "  - ctx_dumps/*.json (stage-by-stage context snapshots)"
