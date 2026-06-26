param(
    [Parameter(Mandatory = $true)]
    [string]$Branch,
    [ValidateSet("implement", "fix", "frontend", "tests", "docs")]
    [string]$Lane = "implement"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (git -C $PSScriptRoot rev-parse --show-toplevel 2>$null)
if (-not $RepoRoot) {
    Write-Error "Not inside a git repository."
}
$Parent = Split-Path $RepoRoot -Parent
$WorktreePath = Join-Path $Parent "uploadm8-auth-wt-$Lane"

if (Test-Path $WorktreePath) {
    Write-Host "Worktree already exists: $WorktreePath"
} else {
    git -C $RepoRoot worktree add $WorktreePath -b $Branch
    if ($LASTEXITCODE -ne 0) {
        git -C $RepoRoot worktree add $WorktreePath $Branch
    }
}

Write-Host ""
Write-Host "Lane:     $Lane"
Write-Host "Branch:   $Branch"
Write-Host "Worktree: $WorktreePath"
Write-Host ""
Write-Host "Open in a new Cursor window for parallel agent work."
Write-Host "List worktrees: git -C `"$RepoRoot`" worktree list"
