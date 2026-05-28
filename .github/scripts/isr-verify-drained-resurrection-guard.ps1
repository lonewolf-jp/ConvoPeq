Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$releasePath = Join-Path $repoRoot 'src\audioengine\AudioEngine.Processing.ReleaseResources.cpp'
$dispatchPath = Join-Path $repoRoot 'src\audioengine\AudioEngine.RebuildDispatch.cpp'

foreach ($path in @($releasePath, $dispatchPath)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Missing required source file: $path"
    }
}

$releaseText = Get-Content -LiteralPath $releasePath -Raw -Encoding UTF8
$dispatchText = Get-Content -LiteralPath $dispatchPath -Raw -Encoding UTF8

function Assert-Contains {
    param(
        [string]$Text,
        [string]$Pattern,
        [string]$Message
    )

    if (-not [regex]::IsMatch($Text, $Pattern, [System.Text.RegularExpressions.RegexOptions]::Singleline)) {
        throw $Message
    }
}

function Get-IndexOrThrow {
    param(
        [string]$Text,
        [string]$Token,
        [string]$Message
    )

    $idx = $Text.IndexOf($Token, [System.StringComparison]::Ordinal)
    if ($idx -lt 0) {
        throw $Message
    }

    return $idx
}

# Rule-11/23 guard: release entry must stop accepting and clear deferred restart paths.
Assert-Contains -Text $releaseText -Pattern 'setShutdownPhase\(ShutdownPhase::StopAcceptingWork,\s*"releaseResources"\);' -Message 'Missing StopAcceptingWork transition at release entry'
Assert-Contains -Text $releaseText -Pattern 'clearRebuildReason\(RebuildReason::StructuralFromNonMT\);' -Message 'Missing clearRebuildReason(StructuralFromNonMT)'
Assert-Contains -Text $releaseText -Pattern 'clearRebuildReason\(RebuildReason::DeferredStructural\);' -Message 'Missing clearRebuildReason(DeferredStructural)'
Assert-Contains -Text $releaseText -Pattern 'clearRebuildReason\(RebuildReason::DeferredFinalizeAware\);' -Message 'Missing clearRebuildReason(DeferredFinalizeAware)'
Assert-Contains -Text $releaseText -Pattern 'cancelPendingUpdate\(\);' -Message 'Missing cancelPendingUpdate() at release entry'

# Strict drained checks: publication/rebuild/fallback all required before final completion.
# Current implementation uses bounded wait + full-drain fallback.
Assert-Contains -Text $releaseText -Pattern 'const bool drainedWithinBudget = waitForDrain\(2000,\s*2\);' -Message 'Missing bounded waitForDrain gate'
Assert-Contains -Text $releaseText -Pattern 'if \(!drainedWithinBudget \|\| !isFullyDrained\(\)\)' -Message 'Missing strict drained conjunction gate (drainedWithinBudget/isFullyDrained)'
Assert-Contains -Text $releaseText -Pattern 'drainPublicationLogForShutdown\(\);' -Message 'Missing publication drain in release path'
Assert-Contains -Text $releaseText -Pattern 'drainDeferredRetireQueues\(true\);' -Message 'Missing deferred retire drain in release path'
Assert-Contains -Text $releaseText -Pattern 'm_epochDomain\.drainAll\(\);' -Message 'Missing epochDomain.drainAll in strict drain fallback path'
Assert-Contains -Text $releaseText -Pattern 'shutdownRuntime_\.transitionTo\(convo::isr::ShutdownPhase::ShutdownComplete\);' -Message 'Missing shutdown complete transition'

# Resurrection guard ordering: clear + cancel must happen before worker stop.
$idxClear = Get-IndexOrThrow -Text $releaseText -Token 'clearRebuildReason(RebuildReason::StructuralFromNonMT);' -Message 'Cannot find clearRebuildReason token index'
$idxCancel = Get-IndexOrThrow -Text $releaseText -Token 'cancelPendingUpdate();' -Message 'Cannot find cancelPendingUpdate token index'
$idxStopWorkers = Get-IndexOrThrow -Text $releaseText -Token 'setShutdownPhase(ShutdownPhase::StopWorkers, "releaseResources");' -Message 'Cannot find StopWorkers token index'
if ($idxClear -gt $idxStopWorkers -or $idxCancel -gt $idxStopWorkers) {
    throw 'Resurrection guard ordering violation: clear/cancel must occur before StopWorkers phase'
}

# Async bridge must short-circuit when shutdown is in progress.
Assert-Contains -Text $dispatchText -Pattern 'void AudioEngine::handleAsyncUpdate\(\)\s*\{\s*if \(isShutdownInProgress\(\)\)\s*return;' -Message 'handleAsyncUpdate must early-return when shutdown is in progress'

Write-Host '[PASS] drained/resurrection guard policy verified (Rule-11/23)'
