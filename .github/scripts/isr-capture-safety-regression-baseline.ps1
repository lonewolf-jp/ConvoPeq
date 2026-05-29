param(
    [switch]$Apply,
    [string]$Owner = 'audioengine-runtime-bridge',
    [string]$Issue = 'BRIDGE-SAFETY-BASELINE-001',
    [string]$Expiry = '2026-09-30'
)

$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$evidenceDir = Join-Path $repoRoot 'evidence'
$baselinePath = Join-Path $repoRoot '.github\isr-safety-regression-baseline.json'
$candidatePath = Join-Path $evidenceDir 'isr-safety-regression-baseline.candidate.json'
$reportPath = Join-Path $evidenceDir 'safety_regression_baseline_capture_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null
}

function Read-JsonFile {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return $null
    }
    return Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json
}

$rebuildMetrics = Read-JsonFile -Path (Join-Path $repoRoot 'evidence\rebuild_admission_8_1_metrics_report.json')
$observeShim = Read-JsonFile -Path (Join-Path $repoRoot 'evidence\observe_shim_usage_report.json')
$cleanupDeferred = Read-JsonFile -Path (Join-Path $repoRoot 'evidence\cleanup_deferred_report.json')
$retireLatency = Read-JsonFile -Path (Join-Path $repoRoot 'evidence\retire_latency_report.json')
$triggerAudit = Read-JsonFile -Path (Join-Path $repoRoot 'evidence\trigger_audit_report.json')

$capturedMetrics = [ordered]@{
    xrunDelta = if ($null -ne $rebuildMetrics -and [string]$rebuildMetrics.status -eq 'evaluated') { [double]$rebuildMetrics.evaluation.xrunDelta } else { 0 }
    staleObserveCount = if ($null -ne $observeShim) { [double]$observeShim.blockedMatches } else { 0 }
    retireBacklogSlope = if ($null -ne $cleanupDeferred) { [double]$cleanupDeferred.remainingCount } else { 0 }
    worldLeakCount = if ($null -ne $triggerAudit) { [double]$triggerAudit.metrics.runtimeExecutionViewUsageCount } else { 0 }
    publicationLatencyP99Ratio = if ($null -ne $retireLatency -and $null -ne $retireLatency.withinThreshold) { if ([bool]$retireLatency.withinThreshold) { 1 } else { 2 } } else { 1 }
}

$candidate = [ordered]@{
    schema = 'isr_safety_regression_baseline_v1'
    owner = $Owner
    issue = $Issue
    rationale = 'captured from latest verified evidence snapshot for SafetyPass baseline comparison'
    expiry = $Expiry
    capturedAt = (Get-Date -Format 'o')
    metrics = $capturedMetrics
}

$candidate | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $candidatePath -Encoding UTF8

$applied = $false
if ($Apply) {
    $candidate | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $baselinePath -Encoding UTF8
    $applied = $true
}

$report = [ordered]@{
    schema = 'safety_regression_baseline_capture_report_v1'
    generatedAt = (Get-Date -Format 'o')
    baselinePath = $baselinePath
    candidatePath = $candidatePath
    applied = $applied
    metrics = $capturedMetrics
    dataSources = [ordered]@{
        rebuildMetrics = 'evidence/rebuild_admission_8_1_metrics_report.json'
        observeShim = 'evidence/observe_shim_usage_report.json'
        cleanupDeferred = 'evidence/cleanup_deferred_report.json'
        retireLatency = 'evidence/retire_latency_report.json'
        triggerAudit = 'evidence/trigger_audit_report.json'
    }
}

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8

Write-Host "[INFO] safety baseline candidate written: $candidatePath"
Write-Host "[INFO] safety baseline capture report written: $reportPath"
if ($applied) {
    Write-Host "[PASS] safety baseline applied: $baselinePath"
} else {
    Write-Host '[PASS] safety baseline captured (preview mode, not applied)'
}
