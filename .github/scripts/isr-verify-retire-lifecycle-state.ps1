$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$timelinePath = Join-Path $repoRoot 'evidence\retire_timeline.json'
$sourceHeaderPath = Join-Path $repoRoot 'src\audioengine\ISRRetireRuntimeEx.h'

if (-not (Test-Path -LiteralPath $sourceHeaderPath)) {
    throw "Missing retire runtime header: $sourceHeaderPath"
}

$headerText = Get-Content -LiteralPath $sourceHeaderPath -Raw -Encoding UTF8
$requiredStates = @('Visible', 'CompareEligible', 'TelemetryRetained', 'ReplayRetainedOptional', 'ReclaimEligible', 'Reclaimed')
foreach ($state in $requiredStates) {
    if (-not $headerText.Contains($state)) {
        throw "Retire lifecycle enum token missing in header: $state"
    }
}

if (-not (Test-Path -LiteralPath $timelinePath)) {
    throw "Missing retire timeline evidence: $timelinePath"
}

$data = Get-Content -LiteralPath $timelinePath -Raw -Encoding UTF8 | ConvertFrom-Json
if ("$($data.schema)" -ne 'retire_timeline_v2') {
    throw "Schema mismatch for retire timeline evidence. expected=retire_timeline_v2 actual=$($data.schema)"
}

if ($null -eq $data.lifecycleCounters) {
    throw 'retire timeline missing lifecycleCounters section'
}

foreach ($field in @('visible', 'compareEligible', 'telemetryRetained', 'replayRetainedOptional', 'reclaimEligible', 'reclaimed')) {
    if ($null -eq $data.lifecycleCounters.PSObject.Properties[$field]) {
        throw "retire timeline lifecycleCounters missing field: $field"
    }
}

if ($null -eq $data.lifecycleSample) {
    throw 'retire timeline missing lifecycleSample section'
}

Write-Host "[PASS] retire lifecycle state verification passed (reclaimed=$($data.lifecycleCounters.reclaimed))"
