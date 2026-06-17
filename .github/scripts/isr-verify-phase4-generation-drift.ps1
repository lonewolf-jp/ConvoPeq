$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$audioRoot = Join-Path $repoRoot "src\audioengine"
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'phase4_generation_drift_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -Path $evidenceDir -ItemType Directory | Out-Null
}

$requiredFiles = [ordered]@{
    dspHandleCpp  = (Join-Path $audioRoot 'ISRDSPHandle.cpp')
    dspTransition = (Join-Path $audioRoot 'DSPTransition.h')
    timerCpp      = (Join-Path $audioRoot 'AudioEngine.Timer.cpp')
    audioHeader   = (Join-Path $audioRoot 'AudioEngine.h')
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
$dspTransition = $fileContents['dspTransition']
$timerCpp = $fileContents['timerCpp']
$audioHeader = $fileContents['audioHeader']

# 1) stale generation handle は reject 必須
if ($null -ne $dspHandleCpp) {
    if ($dspHandleCpp -notmatch 'currentGen\s*!=\s*handle\.generation') {
        $violations.Add('Phase4 drift gate: stale generation mismatch check missing in DSPHandleRuntime::resolve.')
    }
}

# 2) Crossfade 開始: DSPTransition で registerCrossfade + beginCrossfade 連携
if ($null -ne $dspTransition) {
    if ($dspTransition -notmatch 'crossfadeAuthorityRuntime_\.registerCrossfade\(') {
        $violations.Add('Crossfade gate: registerCrossfade call missing in DSPTransition.')
    }
    if ($dspTransition -notmatch 'dspHandleRuntime_\.beginCrossfade\(') {
        $violations.Add('Crossfade gate: beginCrossfade call missing in DSPTransition.')
    }
    # activeCrossfadeId_ への参照が存在しないこと
    if ($dspTransition -match 'activeCrossfadeId_') {
        $violations.Add('Crossfade gate: activeCrossfadeId_ reference must be removed from DSPTransition.')
    }
}

# 3) Crossfade 完了: Timer で getActiveCrossfades + SPSC round-trip + endCrossfade/unregister
if ($null -ne $timerCpp) {
    # Authority から ID を取得
    if ($timerCpp -notmatch 'getActiveCrossfades\(\)') {
        $violations.Add('Crossfade gate: timer must use getActiveCrossfades() for completion.')
    }
    # SPSC round-trip
    if ($timerCpp -notmatch 'notifyFadeComplete\(') {
        $violations.Add('Crossfade gate: timer must notifyFadeComplete to SPSC.')
    }
    if ($timerCpp -notmatch 'consumeCompletedFade\(') {
        $violations.Add('Crossfade gate: timer must consumeCompletedFade from SPSC.')
    }
    # 状態遷移
    if ($timerCpp -notmatch 'dspHandleRuntime_\.endCrossfade\s*\(') {
        $violations.Add('Crossfade gate: timer must call endCrossfade.')
    }
    if ($timerCpp -notmatch 'crossfadeAuthorityRuntime_\.unregisterCrossfade\s*\(') {
        $violations.Add('Crossfade gate: timer must unregister crossfade authority entry.')
    }
    # activeCrossfadeId_ が Timer から完全に消えていること
    if ($timerCpp -match 'activeCrossfadeId_') {
        $violations.Add('Crossfade gate: activeCrossfadeId_ must be removed from Timer.cpp.')
    }
    # 単一前提表明（jassert）
    if ($timerCpp -notmatch 'jassert\(records\.size\(\)') {
        $violations.Add('Crossfade gate: timer must assert single-crossfade assumption.')
    }
}

# 4) AudioEngine.h に activeCrossfadeId_ が存在しないこと
if ($null -ne $audioHeader) {
    if ($audioHeader -match 'activeCrossfadeId_') {
        $violations.Add('Crossfade gate: activeCrossfadeId_ must be removed from AudioEngine.h.')
    }
}

# 5) crossfadeRecords_ は維持されていること（削除禁止）
if ($null -ne $dspHandleCpp) {
    if ($dspHandleCpp -notmatch 'crossfadeRecords_') {
        $violations.Add('Crossfade gate: crossfadeRecords_ must be preserved in DSPHandleRuntime.')
    }
}

$report = [ordered]@{
    schema        = 'phase4_generation_drift_report_v2'
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
