$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'latency_alignment_report.json'

if (-not (Test-Path $evidenceDir)) {
    New-Item -Path $evidenceDir -ItemType Directory | Out-Null
}

$headerPath = Join-Path $repoRoot 'src\audioengine\AudioEngine.h'
$initPath = Join-Path $repoRoot 'src\audioengine\AudioEngine.Init.cpp'
$preparePath = Join-Path $repoRoot 'src\audioengine\AudioEngine.Processing.PrepareToPlay.cpp'
$commitPath = Join-Path $repoRoot 'src\audioengine\AudioEngine.Commit.cpp'

foreach ($path in @($headerPath, $initPath, $preparePath, $commitPath)) {
    if (-not (Test-Path $path)) {
        throw "Missing required source file: $path"
    }
}

$headerText = Get-Content -LiteralPath $headerPath -Raw -Encoding UTF8
$initText = Get-Content -LiteralPath $initPath -Raw -Encoding UTF8
$prepareText = Get-Content -LiteralPath $preparePath -Raw -Encoding UTF8
$commitText = Get-Content -LiteralPath $commitPath -Raw -Encoding UTF8

$violations = New-Object System.Collections.Generic.List[string]

foreach ($required in @('publishLatencyDelayAtomics', 'resetLatencyDelayRtState')) {
    if ($headerText -notmatch [regex]::Escape($required)) {
        $violations.Add("AudioEngine.h missing latency alignment helper: $required")
    }
}

if ($initText -notmatch 'resetLatencyDelayRtState\(\)') {
    $violations.Add('AudioEngine.Init.cpp must call resetLatencyDelayRtState()')
}
if ($initText -match 'latencyDelayOld_RT\s*=|latencyDelayNew_RT\s*=') {
    $violations.Add('AudioEngine.Init.cpp must not directly assign latencyDelay*_RT')
}

if ($prepareText -notmatch 'resetLatencyDelayRtState\(\)') {
    $violations.Add('AudioEngine.Processing.PrepareToPlay.cpp must call resetLatencyDelayRtState()')
}
if ($prepareText -match 'latencyDelayOld_RT\s*=|latencyDelayNew_RT\s*=') {
    $violations.Add('AudioEngine.Processing.PrepareToPlay.cpp must not directly assign latencyDelay*_RT')
}

if ($commitText -notmatch 'publishLatencyDelayAtomics\(') {
    $violations.Add('AudioEngine.Commit.cpp must call publishLatencyDelayAtomics()')
}
if ($commitText -match 'publishAtomic\(latencyDelayOld,|publishAtomic\(latencyDelayNew,') {
    $violations.Add('AudioEngine.Commit.cpp must not directly publish latencyDelay atomics')
}

# v5.5 P5: RT latency sync can be done either by legacy runtimeGraph helper or by prepared snapshot handoff.
$hasLegacyRuntimeGraphSync = $headerText -match 'syncLatencyDelayRtState\(runtimeGraph\)'
$hasPreparedSnapshotSyncLegacy =
    ($headerText -match 'runtime\.latencyDelayOld\s*=\s*prepared\.latencyDelayOld;') -and
    ($headerText -match 'runtime\.latencyDelayNew\s*=\s*prepared\.latencyDelayNew;')

$hasPreparedSnapshotSyncCurrent =
    ($headerText -match 'makeCrossfadePreparedSnapshotFromWorld\(const RuntimePublishWorld& world\)') -and
    ($headerText -match 'snapshot\.latencyDelayOld\s*=\s*world\.latency\.latencyDelayOld;') -and
    ($headerText -match 'snapshot\.latencyDelayNew\s*=\s*world\.latency\.latencyDelayNew;')

$hasPreparedSnapshotSync = $hasPreparedSnapshotSyncLegacy -or $hasPreparedSnapshotSyncCurrent

if (-not $hasLegacyRuntimeGraphSync -and -not $hasPreparedSnapshotSync) {
    $violations.Add('AudioEngine.h must sync RT latency state via legacy runtimeGraph helper or prepared snapshot handoff')
}

$report = [ordered]@{
    schema = 'latency_alignment_report_v2'
    generatedAt = (Get-Date -Format 'o')
    headerPath = $headerPath
    initPath = $initPath
    preparePath = $preparePath
    commitPath = $commitPath
    hasLegacyRuntimeGraphSync = $hasLegacyRuntimeGraphSync
    hasPreparedSnapshotSync = $hasPreparedSnapshotSync
    violations = $violations
}

Set-Content -LiteralPath $reportPath -Value ($report | ConvertTo-Json -Depth 6) -Encoding UTF8
Write-Host "[INFO] latency alignment report written: $reportPath"

if ($violations.Count -gt 0) {
    foreach ($violation in $violations) {
        Write-Host "[ERROR] $violation"
    }
    throw "Latency alignment violations detected. count=$($violations.Count)"
}

Write-Host '[PASS] latency alignment gate verified'
