param(
    [Parameter(Mandatory = $true)]
    [string]$BranchPrefix = "feat/parallel"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (git -C $PSScriptRoot rev-parse --show-toplevel 2>$null)
if (-not $RepoRoot) {
    Write-Error "Not inside a git repository."
}

$lanes = @(
    @{ Lane = "implement"; Suffix = "implement" },
    @{ Lane = "fix";       Suffix = "fix" },
    @{ Lane = "frontend";  Suffix = "frontend" },
    @{ Lane = "tests";     Suffix = "tests" },
    @{ Lane = "docs";      Suffix = "docs" }
)

Write-Host "Spawning 5-window worktree pattern..." -ForegroundColor Cyan
Write-Host ""

foreach ($item in $lanes) {
    $branch = "$BranchPrefix-$($item.Suffix)"
    & "$PSScriptRoot\worktree_spawn.ps1" -Branch $branch -Lane $item.Lane
    Write-Host ""
}

Write-Host "Done. Open each worktree in a separate Cursor window:" -ForegroundColor Green
git -C $RepoRoot worktree list
