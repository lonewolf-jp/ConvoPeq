# check-authority-boundary.ps1
# P0-5: Publication Authority Audit

param(
    [string]$RepoRoot = (Resolve-Path "$PSScriptRoot/../..")
)

$exitCode = 0

$engineHeader = Join-Path $RepoRoot "src/audioengine/AudioEngine.h"
$coordinatorHeader = Join-Path $RepoRoot "src/core/RuntimePublicationCoordinator.h"
$orchestratorHeader = Join-Path $RepoRoot "src/audioengine/RuntimePublicationOrchestrator.h"

# --- 1. Coordinator direct usage warning ---
Write-Host "`n[1/3] makeRuntimePublicationCoordinator() direct usage:" -ForegroundColor Yellow
$coordCalls = Select-String -Path "$RepoRoot\src\**\*.h", "$RepoRoot\src\**\*.cpp" `
    -Pattern 'makeRuntimePublicationCoordinator(' `
    -SimpleMatch `
    -ErrorAction SilentlyContinue | `
    Where-Object { $_.Path -notlike "*\JUCE\*" -and $_.Path -notlike "*\r8brain-free-src\*" }

if ($coordCalls) {
    Write-Host "  WARNING: direct usage detected:" -ForegroundColor Yellow
    $coordCalls | ForEach-Object {
        Write-Host "    $($_.Path):$($_.LineNumber)"
    }
}
else {
    Write-Host "  OK: no direct usage detected" -ForegroundColor Green
}

# --- 2. Coordinator direct API warning ---
Write-Host "`n[2/3] RuntimePublicationCoordinator direct API usage:" -ForegroundColor Yellow
$directApiCalls = Select-String -Path "$RepoRoot\src\**\*.h", "$RepoRoot\src\**\*.cpp" `
    -Pattern '(RuntimePublicationCoordinator::acquireWriteAccess|::acquireWriteAccess\()' `
    -ErrorAction SilentlyContinue | `
    Where-Object { $_.Path -notlike "*\JUCE\*" -and $_.Path -notlike "*\r8brain-free-src\*" }

if ($directApiCalls) {
    Write-Host "  WARNING: direct API usage detected:" -ForegroundColor Yellow
    $directApiCalls | ForEach-Object {
        Write-Host "    $($_.Path):$($_.LineNumber): $($_.Line.Trim())"
    }
}
else {
    Write-Host "  OK: no direct API usage detected" -ForegroundColor Green
}

# --- 3. friend proliferation audit ---
Write-Host "`n[3/3] friend declaration audit:" -ForegroundColor Yellow
$allowedFriends = @(
    'RuntimePublicationOrchestrator',
    'PublicationExecutor',
    'RuntimePublicationStateOwner',
    'RuntimeBuilder',
    'AudioEngine',
    'DSPTransition',
    'NoiseShaperLearner',
    'EQEditProcessor',
    'RuntimePublicationCoordinator'
)

$engineFriendLines = Select-String -Path $engineHeader -Pattern 'friend class' -ErrorAction SilentlyContinue
$coordFriendLines = Select-String -Path $coordinatorHeader -Pattern 'friend class' -ErrorAction SilentlyContinue
$orchFriendLines = Select-String -Path $orchestratorHeader -Pattern 'friend class' -ErrorAction SilentlyContinue

$allFriendLines = @()
if ($engineFriendLines) { $allFriendLines += $engineFriendLines }
if ($coordFriendLines) { $allFriendLines += $coordFriendLines }
if ($orchFriendLines) { $allFriendLines += $orchFriendLines }

$allFriendLines = $allFriendLines | Where-Object {
    $line = $_.Line.Trim()
    $line -notmatch '^\s*(//|/\*|\*)'
}

$violations = @()
foreach ($line in $allFriendLines) {
    $found = $false
    foreach ($allowed in $allowedFriends) {
        if ($line.Line -match $allowed) {
            $found = $true
            break
        }
    }
    if (-not $found) {
        $violations += $line
    }
}

if ($violations.Count -gt 0) {
    Write-Host "  FAIL: unauthorized friend declaration detected!" -ForegroundColor Red
    $violations | ForEach-Object {
        Write-Host "    $($_.Path):$($_.LineNumber): $($_.Line.Trim())"
    }
    $exitCode = 1
}
else {
    Write-Host "  OK: friend declarations are within allowlist" -ForegroundColor Green
}

Write-Host "`n=== Audit Complete ===" -ForegroundColor Cyan
exit $exitCode
