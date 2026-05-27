$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$runbookPath = Join-Path $repoRoot "doc\work\ISR_Shared_EpochDomain_SplitMigration_Runbook_2026-05-27.md"
$comparisonPath = Join-Path $repoRoot "doc\work\ISR_Shared_EpochDomain_Shared_vs_Split_Comparison_2026-05-27.md"
$goNoGoPath = Join-Path $repoRoot "doc\work\ISR_Shared_EpochDomain_GoNoGo_2026-05-27.md"
$compareScriptPath = Join-Path $repoRoot ".github\scripts\isr-compare-shared-split-epoch.ps1"
$recordScriptPath = Join-Path $repoRoot ".github\scripts\isr-record-shared-split-go-no-go.ps1"

foreach ($path in @($runbookPath, $comparisonPath, $goNoGoPath, $compareScriptPath, $recordScriptPath)) {
    if (-not (Test-Path $path)) {
        throw "Missing file: $path"
    }
}

$runbookText = Get-Content -LiteralPath $runbookPath -Raw -Encoding UTF8
$comparisonText = Get-Content -LiteralPath $comparisonPath -Raw -Encoding UTF8
$goNoGoText = Get-Content -LiteralPath $goNoGoPath -Raw -Encoding UTF8

$runbookRequiredPhrases = @(
    'rollback',
    'split',
    'shared'
)
foreach ($phrase in $runbookRequiredPhrases) {
    if ($runbookText -notmatch [regex]::Escape($phrase)) {
        throw "Split migration runbook missing required keyword: $phrase"
    }
}

$comparisonRequiredPhrases = @(
    'latency',
    'jitter',
    'reclaim burst',
    'shutdown drain'
)
foreach ($phrase in $comparisonRequiredPhrases) {
    if ($comparisonText -notmatch [regex]::Escape($phrase)) {
        throw "Shared vs split comparison document missing required metric axis: $phrase"
    }
}

if ($goNoGoText -notmatch '(?m)^-\s*Decision:\s*(Pending|Go\(shared継続\)|Go\(split移行\)|No-Go)\s*$') {
    throw 'Go/No-Go record must include a valid Decision field.'
}

if ($goNoGoText -notmatch '(?m)^-\s*Reason:\s*.+$') {
    throw 'Go/No-Go record must include a non-empty Reason field.'
}

Write-Host '[PASS] R10 shared/split migration readiness artifacts verified'
