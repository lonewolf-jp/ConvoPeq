$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$policyPath = Join-Path $repoRoot "doc\work\ISR_DSPHandle_Allocator_Policy.md"
$dspHandleHeader = Join-Path $repoRoot "src\audioengine\ISRDSPHandle.h"
$dspHandleCpp = Join-Path $repoRoot "src\audioengine\ISRDSPHandle.cpp"
$releaseResourcesCpp = Join-Path $repoRoot "src\audioengine\AudioEngine.Processing.ReleaseResources.cpp"
$audioEngineHeader = Join-Path $repoRoot "src\audioengine\AudioEngine.h"

foreach ($path in @($policyPath, $dspHandleHeader, $dspHandleCpp, $releaseResourcesCpp, $audioEngineHeader)) {
    if (-not (Test-Path $path)) {
        throw "Missing file: $path"
    }
}

$policyText = Get-Content -LiteralPath $policyPath -Raw -Encoding UTF8
$dspHandleHeaderText = Get-Content -LiteralPath $dspHandleHeader -Raw -Encoding UTF8
$dspHandleCppText = Get-Content -LiteralPath $dspHandleCpp -Raw -Encoding UTF8
$releaseResourcesText = Get-Content -LiteralPath $releaseResourcesCpp -Raw -Encoding UTF8
$audioEngineText = Get-Content -LiteralPath $audioEngineHeader -Raw -Encoding UTF8
$audioEngineCppRoot = Join-Path $repoRoot "src\audioengine"
$audioEngineCppText = (Get-ChildItem -Path $audioEngineCppRoot -Recurse -File -Include *.cpp,*.cxx,*.cc |
    ForEach-Object { Get-Content -LiteralPath $_.FullName -Raw -Encoding UTF8 }) -join "`n"

$requiredPolicyPhrases = @(
    'A1. Slot Table',
    'A2. Reuse Policy',
    'A3. Generation 更新',
    'A4. Overflow Policy',
    'A5. Fragmentation Policy',
    '2回の epoch advance 完了まで再利用禁止',
    'generation は uint64',
    '非RTメンテナンスフェーズで compaction 計画を実行'
)

foreach ($phrase in $requiredPolicyPhrases) {
    if ($policyText -notmatch [regex]::Escape($phrase)) {
        throw "DSPHandle allocator policy text missing required phrase: $phrase"
    }
}

$requiredHeaderTokens = @(
    'DSPHandle create\(',
    'ResolvedDSP resolve\(',
    'void retire\(',
    'void reclaim\(',
    'void quarantine\(',
    'CrossfadeAuthorityRuntime'
)

foreach ($token in $requiredHeaderTokens) {
    if ($dspHandleHeaderText -notmatch $token) {
        throw "DSPHandle runtime header missing required API token: $token"
    }
}

if ($dspHandleCppText -notmatch 'currentGen != handle.generation') {
    throw 'DSPHandleRuntime must reject stale generation handles.'
}

if ($dspHandleCppText -notmatch 'state == DSPState::Reclaimed \|\| state == DSPState::Quarantined') {
    throw 'DSPHandleRuntime must reject reclaimed/quarantined handles.'
}

if ($dspHandleCppText -notmatch 'void DSPHandleRuntime::quarantine\(') {
    throw 'DSPHandleRuntime quarantine path missing.'
}

if ($releaseResourcesText -notmatch 'dspHandleRuntime_\.reclaim\(') {
    throw 'Shutdown reclaim path must route through DSPHandleRuntime::reclaim.'
}

$hasHandleRuntimeObservePath = ($audioEngineCppText -match 'dspHandleRuntime_\.resolve\(')
$hasRuntimeWorldObservePath =
    ($audioEngineCppText -match 'readAudioRuntimeView\(\)') -and
    ($audioEngineCppText -match 'getRuntimeGraph\(runtimeReadView\)') -and
    ($audioEngineCppText -match 'runtimeGraph->activeNode')

if (-not $hasHandleRuntimeObservePath -and -not $hasRuntimeWorldObservePath) {
    throw 'AudioEngine processing path must observe runtime DSP via DSPHandleRuntime or RuntimeWorld read path.'
}

if ($audioEngineCppText -notmatch 'dspHandleRuntime_\.reclaim\(' -and $audioEngineText -notmatch 'dspHandleRuntime_\.reclaim\(') {
    throw 'AudioEngine must keep reclaim routed through DSPHandleRuntime.'
}

Write-Host '[PASS] R3 DSP handle allocator policy verified'
