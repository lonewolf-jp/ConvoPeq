param(
    [string]$BaselinePath = (Join-Path $PSScriptRoot '..\isr-safety-regression-baseline.json')
)

$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$baselineFullPath = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\isr-safety-regression-baseline.json'))
$reportPath = Join-Path $repoRoot 'evidence\safety_regression_report.json'

if (-not (Test-Path -LiteralPath $baselineFullPath)) {
    throw "Missing safety regression baseline: $baselineFullPath"
}

$baseline = Get-Content -LiteralPath $baselineFullPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($baseline.schema -ne 'isr_safety_regression_baseline_v1') {
    throw "Unexpected safety baseline schema: $($baseline.schema)"
}

$remeasureReportPath = Join-Path $repoRoot 'evidence\safety_regression_remeasure_report.json'
$failureHistoryPath = Join-Path $repoRoot 'evidence\safety_failure_window_history.json'

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

$currentMetrics = [ordered]@{
    xrunDelta = if ($null -ne $rebuildMetrics -and [string]$rebuildMetrics.status -eq 'evaluated') { [double]$rebuildMetrics.evaluation.xrunDelta } else { 0 }
    staleObserveCount = if ($null -ne $observeShim) { [double]$observeShim.blockedMatches } else { 0 }
    retireBacklogSlope = if ($null -ne $cleanupDeferred) { [double]$cleanupDeferred.remainingCount } else { 0 }
    worldLeakCount = if ($null -ne $triggerAudit) { [double]$triggerAudit.metrics.runtimeExecutionViewUsageCount } else { 0 }
    publicationLatencyP99Ratio = if ($null -ne $retireLatency -and $null -ne $retireLatency.withinThreshold) { if ([bool]$retireLatency.withinThreshold) { 1 } else { 2 } } else { 1 }
}

$baselineMetrics = $baseline.metrics
$violations = New-Object System.Collections.Generic.List[string]

$xrunPass = ($currentMetrics.xrunDelta -le [double]$baselineMetrics.xrunDelta)
$staleObservePass = ($currentMetrics.staleObserveCount -le [double]$baselineMetrics.staleObserveCount)
$retireBacklogPass = ($currentMetrics.retireBacklogSlope -le [double]$baselineMetrics.retireBacklogSlope)
$worldLeakPass = ($currentMetrics.worldLeakCount -eq 0 -and $currentMetrics.worldLeakCount -le [double]$baselineMetrics.worldLeakCount)
$publicationLatencyThreshold = [Math]::Round(([double]$baselineMetrics.publicationLatencyP99Ratio * 1.05), 6)
$publicationLatencyPass = ($currentMetrics.publicationLatencyP99Ratio -le $publicationLatencyThreshold)

$history = Read-JsonFile -Path $failureHistoryPath
if ($null -eq $history -or "$($history.schema)" -ne 'safety_failure_window_history_v1') {
    $history = [ordered]@{
        schema = 'safety_failure_window_history_v1'
        generatedAt = $null
        streaks = [ordered]@{
            classDBacklogDivergence = 0
            classERetentionLeak = 0
        }
    }
}

$previousClassDStreak = [int]$history.streaks.classDBacklogDivergence
$previousClassEStreak = [int]$history.streaks.classERetentionLeak
$currentClassDStreak = if (-not $retireBacklogPass) { $previousClassDStreak + 1 } else { 0 }
$currentClassEStreak = if (-not $worldLeakPass) { $previousClassEStreak + 1 } else { 0 }

$classDFailClosed = ($currentClassDStreak -ge 2)
$classEFailClosed = ($currentClassEStreak -ge 2)

$retireBacklogPassEffective = ($retireBacklogPass -or -not $classDFailClosed)
$worldLeakPassEffective = ($worldLeakPass -or -not $classEFailClosed)

$warnings = New-Object System.Collections.Generic.List[string]
if (-not $retireBacklogPass -and -not $classDFailClosed) {
    $warnings.Add("Class-D backlog divergence detected but below consecutive-fail threshold: streak=$currentClassDStreak required=2")
}
if (-not $worldLeakPass -and -not $classEFailClosed) {
    $warnings.Add("Class-E retention leak detected but below consecutive-fail threshold: streak=$currentClassEStreak required=2")
}

if (-not $xrunPass) {
    $violations.Add("XRUN count regressed: current=$($currentMetrics.xrunDelta) baseline=$($baselineMetrics.xrunDelta)")
}
if (-not $staleObservePass) {
    $violations.Add("stale observe count regressed: current=$($currentMetrics.staleObserveCount) baseline=$($baselineMetrics.staleObserveCount)")
}
if (-not $retireBacklogPassEffective) {
    $violations.Add("retire backlog slope regressed (Class-D consecutive window fail): current=$($currentMetrics.retireBacklogSlope) baseline=$($baselineMetrics.retireBacklogSlope) streak=$currentClassDStreak")
}
if (-not $worldLeakPassEffective) {
    $violations.Add("world leak count policy violated (Class-E consecutive window fail): current=$($currentMetrics.worldLeakCount) baseline=$($baselineMetrics.worldLeakCount) requiredCurrent=0 streak=$currentClassEStreak")
}
if (-not $publicationLatencyPass) {
    $violations.Add("publication latency p99 ratio regressed: current=$($currentMetrics.publicationLatencyP99Ratio) threshold=$publicationLatencyThreshold baseline=$($baselineMetrics.publicationLatencyP99Ratio)")
}

$safetyPass = ($xrunPass -and $staleObservePass -and $retireBacklogPassEffective -and $worldLeakPassEffective -and $publicationLatencyPass)

$noisePolicyApplied = $false
$noisePolicyAccepted = $false
$noisePolicyDetails = $null

if (-not $safetyPass) {
    $failedChecks = New-Object 'System.Collections.Generic.List[string]'
    if (-not $xrunPass) { $failedChecks.Add('xrunDelta') | Out-Null }
    if (-not $staleObservePass) { $failedChecks.Add('staleObserveCount') | Out-Null }
    if (-not $retireBacklogPassEffective) { $failedChecks.Add('retireBacklogSlope') | Out-Null }
    if (-not $worldLeakPassEffective) { $failedChecks.Add('worldLeakCount') | Out-Null }
    if (-not $publicationLatencyPass) { $failedChecks.Add('publicationLatencyP99Ratio') | Out-Null }

    if ($failedChecks.Count -eq 1) {
        $failedMetric = $failedChecks[0]
        $baselineValue = [double]$baselineMetrics.$failedMetric
        $currentValue = [double]$currentMetrics.$failedMetric
        $noiseThreshold = [Math]::Round(($baselineValue * 1.03), 6)

        if ($failedMetric -ne 'worldLeakCount' -and $currentValue -le $noiseThreshold) {
            $noisePolicyApplied = $true
            $noisePolicyDetails = [ordered]@{
                failedMetric = $failedMetric
                baselineValue = $baselineValue
                currentValue = $currentValue
                allowedThreshold = $noiseThreshold
                remeasureReportPath = $remeasureReportPath
            }

            if (Test-Path -LiteralPath $remeasureReportPath) {
                try {
                    $remeasureReport = Get-Content -LiteralPath $remeasureReportPath -Raw -Encoding UTF8 | ConvertFrom-Json
                    if ("$($remeasureReport.schema)" -eq 'safety_regression_remeasure_report_v1' -and $null -ne $remeasureReport.samples) {
                        $samples = @($remeasureReport.samples)
                        if ($samples.Count -eq 3) {
                            $values = @()
                            foreach ($sample in $samples) {
                                if ($null -eq $sample.metrics -or $sample.metrics.PSObject.Properties.Name -notcontains $failedMetric) {
                                    throw "remeasure sample missing metric: $failedMetric"
                                }
                                $values += [double]$sample.metrics.$failedMetric
                            }
                            $sorted = @($values | Sort-Object)
                            $median = [double]$sorted[1]
                            if ($median -le $noiseThreshold) {
                                $noisePolicyAccepted = $true
                                $safetyPass = $true
                                $violations.Clear()
                            }
                            $noisePolicyDetails['remeasureValues'] = @($values)
                            $noisePolicyDetails['remeasureMedian'] = $median
                        }
                    }
                }
                catch {
                    $noisePolicyDetails['remeasureError'] = "$($_.Exception.Message)"
                }
            }
        }
    }
}

$report = [ordered]@{
    schema = 'safety_regression_report_v1'
    generatedAt = (Get-Date -Format 'o')
    baselinePath = $baselineFullPath
    baseline = $baseline
    currentMetrics = $currentMetrics
    checks = [ordered]@{
        xrun = $xrunPass
        staleObserve = $staleObservePass
        retireBacklogSlope = $retireBacklogPass
        worldLeak = $worldLeakPass
        publicationLatencyP99 = $publicationLatencyPass
    }
    publicationLatencyPolicy = [ordered]@{
        baseline = [double]$baselineMetrics.publicationLatencyP99Ratio
        multiplier = 1.05
        threshold = $publicationLatencyThreshold
    }
    noiseAllowancePolicy = [ordered]@{
        maxFailedMetrics = 1
        maxRelativeDrift = 1.03
        requiredRemeasureRuns = 3
        medianAcceptance = 'median<=baseline*1.03'
        applied = $noisePolicyApplied
        accepted = $noisePolicyAccepted
        details = $noisePolicyDetails
    }
    taxonomyWindowPolicy = [ordered]@{
        classD = [ordered]@{
            metric = 'retireBacklogSlope'
            failOnConsecutiveWindows = 2
            previousStreak = $previousClassDStreak
            currentStreak = $currentClassDStreak
            rawMetricPass = $retireBacklogPass
            effectiveMetricPass = $retireBacklogPassEffective
            failClosed = $classDFailClosed
        }
        classE = [ordered]@{
            metric = 'worldLeakCount'
            failOnConsecutiveWindows = 2
            previousStreak = $previousClassEStreak
            currentStreak = $currentClassEStreak
            rawMetricPass = $worldLeakPass
            effectiveMetricPass = $worldLeakPassEffective
            failClosed = $classEFailClosed
        }
    }
    safetyPass = $safetyPass
    warnings = @($warnings)
    violations = @($violations)
}

$updatedHistory = [ordered]@{
    schema = 'safety_failure_window_history_v1'
    generatedAt = (Get-Date -Format 'o')
    streaks = [ordered]@{
        classDBacklogDivergence = $currentClassDStreak
        classERetentionLeak = $currentClassEStreak
    }
    latest = [ordered]@{
        retireBacklogSlope = [double]$currentMetrics.retireBacklogSlope
        worldLeakCount = [double]$currentMetrics.worldLeakCount
    }
}

$updatedHistory | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $failureHistoryPath -Encoding UTF8
Write-Host "[INFO] Safety failure window history written: $failureHistoryPath"

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] Safety regression report written: $reportPath"

if (-not $safetyPass) {
    foreach ($violation in $violations) {
        Write-Host "[ERROR] $violation"
    }
    throw "Safety regression verification failed. violations=$($violations.Count)"
}

Write-Host '[PASS] Safety regression verification passed'
