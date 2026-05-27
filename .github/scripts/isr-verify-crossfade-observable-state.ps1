$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'crossfade_observable_state_report.json'

if (-not (Test-Path $evidenceDir)) {
    New-Item -Path $evidenceDir -ItemType Directory | Out-Null
}

$headerPath = Join-Path $repoRoot 'src\audioengine\AudioEngine.h'
$initPath = Join-Path $repoRoot 'src\audioengine\AudioEngine.Init.cpp'

foreach ($path in @($headerPath, $initPath)) {
    if (-not (Test-Path $path)) {
        throw "Missing required source file: $path"
    }
}

$headerText = Get-Content -LiteralPath $headerPath -Raw -Encoding UTF8
$initText = Get-Content -LiteralPath $initPath -Raw -Encoding UTF8

$violations = New-Object System.Collections.Generic.List[string]

foreach ($required in @(
    'bool dspCrossfadeArmed_RT = false;',
    'int dspCrossfadeStartDelayBlocks_RT = 0;',
    'dspCrossfadeArmed_RT = true;',
    'dspCrossfadeArmed_RT = false;'
)) {
    if ($headerText -notmatch [regex]::Escape($required) -and $initText -notmatch [regex]::Escape($required)) {
        $violations.Add("Missing crossfade observable state token: $required")
    }
}

if ($headerText -notmatch 'if \(!hasPendingCrossfade\)\s*\{\s*dspCrossfadeArmed_RT = false;\s*dspCrossfadeStartDelayBlocks_RT = 0;') {
    $violations.Add('Crossfade pending-miss path must clear armed/start-delay observable state')
}

if ($headerText -notmatch '&& !dspCrossfadeArmed_RT\)') {
    $violations.Add('Crossfade arm path must guard on !dspCrossfadeArmed_RT')
}

if ($initText -notmatch 'dspCrossfadeArmed_RT = false;') {
    $violations.Add('AudioEngine.initialize() must reset dspCrossfadeArmed_RT to false')
}

$report = [ordered]@{
    schema = 'crossfade_observable_state_report_v1'
    generatedAt = (Get-Date -Format 'o')
    headerPath = $headerPath
    initPath = $initPath
    violations = $violations
}

Set-Content -LiteralPath $reportPath -Value ($report | ConvertTo-Json -Depth 6) -Encoding UTF8
Write-Host "[INFO] crossfade observable state report written: $reportPath"

if ($violations.Count -gt 0) {
    foreach ($violation in $violations) {
        Write-Host "[ERROR] $violation"
    }
    throw "Crossfade observable state violations detected. count=$($violations.Count)"
}

Write-Host '[PASS] crossfade observable state gate verified'
