$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$audioRoot = Join-Path $repoRoot "src\audioengine"
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'phase4_generation_drift_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -Path $evidenceDir -ItemType Directory | Out-Null
}

$requiredFiles = [ordered]@{
    dspHandleCpp = (Join-Path $audioRoot 'ISRDSPHandle.cpp')
    commitCpp    = (Join-Path $audioRoot 'AudioEngine.Commit.cpp')
    timerCpp     = (Join-Path $audioRoot 'AudioEngine.Timer.cpp')
    audioHeader  = (Join-Path $audioRoot 'AudioEngine.h')
}

$violations = New-Object System.Collections.Generic.List[string]
$fileContents = @{}

foreach ($key in $requiredFiles.Keys) {
    $path = $requiredFiles[$key]
    if (-not (Test-Path -LiteralPath $path)) {
        $violations.Add("Phase4 drift gate: missing file $path")
        continue
    }

    $fileContents[$key] = Get-Content -LiteralPath $path -Raw -Encoding UTF8
}

$dspHandleCpp = $fileContents['dspHandleCpp']
$commitCpp = $fileContents['commitCpp']
$timerCpp = $fileContents['timerCpp']
$audioHeader = $fileContents['audioHeader']

# 1) stale generation handle は reject 必須
if ($null -ne $dspHandleCpp) {
    if ($dspHandleCpp -notmatch 'currentGen\s*!=\s*handle\.generation') {
        $violations.Add('Phase4 drift gate: stale generation mismatch check missing in DSPHandleRuntime::resolve.')
    }
    if ($dspHandleCpp -notmatch '\{\s*nullptr\s*,\s*false\s*,\s*true\s*\}') {
        $violations.Add('Phase4 drift gate: stale generation path must mark isStale=true.')
    }
}

# 2) crossfade 開始時に activeCrossfadeId を publish すること
if ($null -ne $commitCpp) {
    if ($commitCpp -notmatch 'dspHandleRuntime_\.beginCrossfade\s*\(') {
        $violations.Add('Phase4 drift gate: beginCrossfade call missing in commit path.')
    }
    if ($commitCpp -notmatch 'publishAtomic\(activeCrossfadeId_,\s*crossfadeId') {
        $violations.Add('Phase4 drift gate: activeCrossfadeId publish on crossfade start missing.')
    }
}

# 3) crossfade 完了時に handle runtime / authority runtime / activeCrossfadeId clear の3点を実行
if ($null -ne $timerCpp) {
    if ($timerCpp -notmatch 'consumeAtomic\(activeCrossfadeId_,\s*std::memory_order_acquire\)') {
        $violations.Add('Phase4 drift gate: timer must consume activeCrossfadeId before completion.')
    }
    if ($timerCpp -notmatch 'dspHandleRuntime_\.endCrossfade\s*\(') {
        $violations.Add('Phase4 drift gate: timer must call endCrossfade.')
    }
    if ($timerCpp -notmatch 'crossfadeAuthorityRuntime_\.unregisterCrossfade\s*\(') {
        $violations.Add('Phase4 drift gate: timer must unregister crossfade authority entry.')
    }
    if ($timerCpp -notmatch 'publishAtomic\(activeCrossfadeId_,\s*static_cast<convo::isr::CrossfadeId>\(0u\)') {
        $violations.Add('Phase4 drift gate: timer must clear activeCrossfadeId after completion.')
    }
}

# 4) header に activeCrossfadeId atomic が存在すること
if ($null -ne $audioHeader) {
    if ($audioHeader -notmatch 'std::atomic<convo::isr::CrossfadeId>\s+activeCrossfadeId_\s*\{\s*0u\s*\}') {
        $violations.Add('Phase4 drift gate: activeCrossfadeId atomic declaration missing in AudioEngine.h.')
    }
}

$report = [ordered]@{
    schema        = 'phase4_generation_drift_report_v1'
    generatedAt   = (Get-Date -Format 'o')
    audioRoot     = $audioRoot
    requiredFiles = $requiredFiles
    violations    = $violations
}

Set-Content -LiteralPath $reportPath -Value ($report | ConvertTo-Json -Depth 6) -Encoding UTF8
Write-Host "[INFO] phase4 generation drift report written: $reportPath"

if ($violations.Count -gt 0) {
    foreach ($violation in $violations) {
        Write-Host "[ERROR] $violation"
    }
    throw "Phase4 generation drift violations detected. count=$($violations.Count)"
}

Write-Host '[PASS] Phase4 generation drift gate verified'
