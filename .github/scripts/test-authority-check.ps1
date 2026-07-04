# Test script for authority compliance checks

Write-Host "Testing P1 Phase1-B: Check legacy PublicationIntent remnants..."
$errors = @()

if (Select-String -Path src\**\* -Pattern "struct PublicationIntent|struct PublicationLog" -Quiet) {
    $errors += "PublicationIntent/PublicationLog struct still exists"
}

if (Select-String -Path src\**\* -Pattern "\bappendPublicationIntentForCommit\s*\(|\bdrainPublicationLogForShutdown\s*\(|\bhasPublicationLogPending\s*\(" -Quiet) {
    $errors += "Legacy PublicationLog functions still exist"
}

if ($errors.Count -gt 0) {
    foreach ($e in $errors) {
        Write-Host "error::$e"
    }
    Write-Host "Phase1-B check failed"
    exit 1
}
Write-Host "Phase1-B: All PublicationIntent/PublicationLog remnants removed."

Write-Host ""
Write-Host "Testing P14: Check partial publication interfaces..."
$found = $false
if (Select-String -Path src\**\* -Pattern "publish\(generation|publish\(dsp" -Quiet) {
    Write-Host "error::partial publication interface detected (publish(generation) or publish(dsp))"
    $found = $true
    exit 1
}
if (-not $found) {
    Write-Host "No partial publication interfaces detected."
}

Write-Host ""
Write-Host "Testing P3: Check direct EpochRetire in new code..."
$diff = git diff origin/main...HEAD -- src/ 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "warning::Cannot compare with base branch - check skipped"
    exit 0
}
$violations = $diff | Select-String -Pattern "^\+\s*.*EpochDomain.*enqueueRetire" | Where-Object { $_ -notmatch "Coordinator" }
if ($violations) {
    Write-Host "error::New code uses direct enqueueRetire (must use Coordinator::enqueueRetire instead)"
    $violations | ForEach-Object { Write-Host "  $_" }
    exit 1
}
Write-Host "No direct EpochDomain::enqueueRetire in new code."
