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

foreach ($required in @('publishLatencyDelayAtomics', 'resetLatencyDelayRtState', 'syncLatencyDelayRtState')) {
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

if ($headerText -notmatch 'syncLatencyDelayRtState\(runtimeGraph\)') {
    $violations.Add('AudioEngine.h must sync RT latency state through syncLatencyDelayRtState(runtimeGraph)')
}

$report = [ordered]@{
    schema = 'latency_alignment_report_v1'
    generatedAt = (Get-Date -Format 'o')
    headerPath = $headerPath
    initPath = $initPath
    preparePath = $preparePath
    commitPath = $commitPath
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
