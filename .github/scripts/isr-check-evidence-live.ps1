#============================================================================
# isr-check-evidence-live.ps1 - verify Live Runtime evidence (read only)
#
# Purpose:
#   Verify that evidence/live_runtime_evidence.json comes from an actual
#   Runtime execution. This script never creates or fills in values.
#
# Checks (fail-closed):
#   schema / generationMode=runtime-live / evidenceSource=process /
#   runId matches CONVO_ISR_RUNTIME_RUN_ID / startedAtNs>0 /
#   emittedAtNs>=startedAtNs / engineInitialized=true / audioCallbackCount>0 /
#   sampleRateHz>0 / blockSamples>0 / derived duty ratio within [0.25, 4.0] /
#   file mtime inside the run window / no evidence written after process exit /
#   manifest (if present) has generationMode=runtime and the same runtimeRunId
#
# Naming note: this file is NOT named isr-verify-*.ps1. The execution-layer
#   policy pins the verifier count at 37 and registering a new verifier is out
#   of scope for this change.
#
# NOTE: this file is intentionally ASCII-only (see the runner's note).
#============================================================================

param(
    [datetime]$StartedUtc = [datetime]::MinValue,
    [datetime]$ExitedUtc = [datetime]::MinValue
)

$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$evidenceRoot = Join-Path $repoRoot "evidence"
$evidencePath = Join-Path $evidenceRoot "live_runtime_evidence.json"

function Fail([string]$message) {
    Write-Host "[FAIL] $message"
    exit 1
}

if (-not (Test-Path -LiteralPath $evidencePath)) {
    Fail "missing live evidence: $evidencePath"
}

$raw = Get-Content -LiteralPath $evidencePath -Raw -Encoding UTF8
if ([string]::IsNullOrWhiteSpace($raw)) {
    Fail "live evidence is empty: $evidencePath"
}

try {
    $evidence = $raw | ConvertFrom-Json
} catch {
    Fail "live evidence is not valid JSON: $($_.Exception.Message)"
}

if ($null -eq $evidence) {
    Fail "live evidence parsed to null"
}

# -- identity: schema, mode, source -----------------------------------------
if ("$($evidence.schema)" -ne 'live_runtime_evidence_v1') {
    Fail "schema mismatch: actual=$($evidence.schema) expected=live_runtime_evidence_v1"
}
if ("$($evidence.generationMode)" -ne 'runtime-live') {
    Fail "generationMode mismatch: actual=$($evidence.generationMode) expected=runtime-live"
}
if ("$($evidence.evidenceSource)" -ne 'process') {
    Fail "evidenceSource mismatch: actual=$($evidence.evidenceSource) expected=process"
}

$runId = [string]$env:CONVO_ISR_RUNTIME_RUN_ID
if ([string]::IsNullOrWhiteSpace($runId)) {
    Fail "CONVO_ISR_RUNTIME_RUN_ID is required to verify live evidence"
}
if ("$($evidence.runId)" -ne $runId) {
    Fail "runId mismatch: evidence=$($evidence.runId) expected=$runId"
}

# -- timestamps (monotonic steady clock inside the process) ------------------
$startedAtNs = [long]$evidence.startedAtNs
$emittedAtNs = [long]$evidence.emittedAtNs
if ($startedAtNs -le 0) {
    Fail "startedAtNs must be > 0 (actual=$startedAtNs)"
}
if ($emittedAtNs -lt $startedAtNs) {
    Fail "emittedAtNs < startedAtNs (started=$startedAtNs emitted=$emittedAtNs)"
}

# -- runtime observations ----------------------------------------------------
$runtime = $evidence.runtime
if ($null -eq $runtime) {
    Fail "runtime block is missing"
}

$requiredRuntimeFields = @(
    'engineInitialized',
    'audioCallbackCount',
    'sampleRateHz',
    'blockSamples'
)
foreach ($field in $requiredRuntimeFields) {
    if (-not ($runtime.PSObject.Properties.Name -contains $field)) {
        Fail "runtime.$field is missing"
    }
}

# engineInitialized must be a JSON boolean: PowerShell 5.1 treats the non-empty
# string "false" as $true, so a cast alone would accept a forged string.
$engineInitialized = $runtime.engineInitialized
if ($engineInitialized -isnot [bool]) {
    Fail "engineInitialized must be a JSON boolean (actual type: $($engineInitialized.GetType().Name))"
}
if (-not $engineInitialized) {
    Fail "engineInitialized is false (runtime did not initialize)"
}

$callbackCount = [long]$runtime.audioCallbackCount
$sampleRateHz = [double]$runtime.sampleRateHz
$blockSamples = [int]$runtime.blockSamples

if ($callbackCount -le 0) {
    Fail "audioCallbackCount must be > 0 (runtime processed no audio)"
}
if ($sampleRateHz -le 0) {
    Fail "sampleRateHz must be > 0 (actual=$sampleRateHz)"
}
if ($blockSamples -le 0) {
    Fail "blockSamples must be > 0 (actual=$blockSamples)"
}

# -- derived values: recompute from the observations and compare --------------
#    The gate must not trust the derived block; it is recomputed from the
#    counters and the steady-clock timestamps, then compared with the stored
#    values (the exporter rounds to about six digits).
$derived = $evidence.derived
if ($null -eq $derived) {
    Fail "derived block is missing"
}
foreach ($field in @('audioSecondsProcessed', 'wallSeconds', 'dutyRatio')) {
    if (-not ($derived.PSObject.Properties.Name -contains $field)) {
        Fail "derived.$field is missing"
    }
}

$audioSeconds = [double]$derived.audioSecondsProcessed
$wallSeconds = [double]$derived.wallSeconds
$dutyRatio = [double]$derived.dutyRatio

$recomputedAudioSeconds = ($callbackCount * $blockSamples) / $sampleRateHz
$recomputedWallSeconds = ($emittedAtNs - $startedAtNs) / 1.0e9
if ($recomputedWallSeconds -le 0) {
    Fail "recomputed wall time is not positive (startedAtNs=$startedAtNs emittedAtNs=$emittedAtNs)"
}
$recomputedDutyRatio = $recomputedAudioSeconds / $recomputedWallSeconds

function Test-CloseEnough([double]$stored, [double]$recomputed, [double]$relative, [double]$absolute) {
    $delta = [Math]::Abs($stored - $recomputed)
    if ($delta -le $absolute) { return $true }
    if ([Math]::Abs($recomputed) -gt 0.0 -and ($delta / [Math]::Abs($recomputed)) -le $relative) { return $true }
    return $false
}

if (-not (Test-CloseEnough $audioSeconds $recomputedAudioSeconds 0.01 0.01)) {
    Fail ("derived.audioSecondsProcessed does not match callbacks*block/rate: stored=$audioSeconds " +
        "recomputed=$recomputedAudioSeconds (callbacks=$callbackCount block=$blockSamples rate=$sampleRateHz)")
}
if (-not (Test-CloseEnough $wallSeconds $recomputedWallSeconds 0.01 0.01)) {
    Fail ("derived.wallSeconds does not match emittedAtNs-startedAtNs: stored=$wallSeconds " +
        "recomputed=$recomputedWallSeconds")
}
if (-not (Test-CloseEnough $dutyRatio $recomputedDutyRatio 0.01 0.01)) {
    Fail ("derived.dutyRatio does not match audioSeconds/wallSeconds: stored=$dutyRatio " +
        "recomputed=$recomputedDutyRatio")
}
if ($recomputedDutyRatio -lt 0.25 -or $recomputedDutyRatio -gt 4.0) {
    $rounded = [Math]::Round($recomputedDutyRatio, 3)
    Fail "dutyRatio out of range: $rounded (audioSeconds=$recomputedAudioSeconds wallSeconds=$recomputedWallSeconds)"
}

# -- file mtime inside the run window (wall clock) ---------------------------
#    Normalise the caller's window to UTC: a bound DateTime parsed from an ISO
#    string is Local kind, and comparing it with a UTC mtime compares raw ticks.
$windowStartUtc = [datetime]::MinValue
$windowEndUtc = [datetime]::MinValue
if ($StartedUtc -ne [datetime]::MinValue) { $windowStartUtc = $StartedUtc.ToUniversalTime() }
if ($ExitedUtc -ne [datetime]::MinValue) { $windowEndUtc = $ExitedUtc.ToUniversalTime() }

$mtimeUtc = (Get-Item -LiteralPath $evidencePath).LastWriteTimeUtc
if ($windowStartUtc -ne [datetime]::MinValue) {
    if ($mtimeUtc -lt $windowStartUtc.AddSeconds(-2)) {
        Fail "live evidence is stale: mtime=$mtimeUtc started=$windowStartUtc"
    }
}
if ($windowEndUtc -ne [datetime]::MinValue) {
    if ($mtimeUtc -gt $windowEndUtc.AddSeconds(2)) {
        Fail "live evidence was written after process exit: mtime=$mtimeUtc exited=$windowEndUtc"
    }
}

# -- nothing may be written to evidence/ after the process exited ------------
if ($windowEndUtc -ne [datetime]::MinValue) {
    $lateFiles = @(Get-ChildItem -LiteralPath $evidenceRoot -File -Filter '*.json' -ErrorAction SilentlyContinue |
        Where-Object { $_.LastWriteTimeUtc -gt $windowEndUtc.AddSeconds(2) })
    if ($lateFiles.Count -gt 0) {
        $names = ($lateFiles | ForEach-Object { $_.Name }) -join ', '
        Fail "evidence modified after process exit: $names"
    }
}

# -- consistency with the existing manifest (only when present) --------------
$manifestPath = Join-Path $evidenceRoot "evidence_manifest.json"
if (Test-Path -LiteralPath $manifestPath) {
    $manifestRaw = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8
    if ([string]::IsNullOrWhiteSpace($manifestRaw)) {
        # An empty manifest would skip both checks below; reject it instead.
        Fail "evidence_manifest.json exists but is empty"
    }
    try {
        $manifest = $manifestRaw | ConvertFrom-Json
    } catch {
        Fail "evidence_manifest.json is not valid JSON: $($_.Exception.Message)"
    }

    if ("$($manifest.generationMode)" -ne 'runtime') {
        Fail "manifest generationMode is not runtime: actual=$($manifest.generationMode)"
    }
    if ("$($manifest.runtimeRunId)" -ne $runId) {
        Fail "manifest runtimeRunId mismatch: actual=$($manifest.runtimeRunId) expected=$runId"
    }
}

Write-Host "[PASS] live evidence gate: callbacks=$callbackCount rate=$sampleRateHz block=$blockSamples duty=$([Math]::Round($recomputedDutyRatio, 3)) runId=$runId"
exit 0
