param(
    [string]$RegistryPath = '.github/isr-metric-governance.json',
    [string]$MetricGovernanceReportPath = 'evidence/metric_governance_report.json',
    [string]$RebuildAdmissionMetricsPath = 'evidence/rebuild_admission_8_1_metrics_report.json',
    [string]$RetireLatencyReportPath = 'evidence/retire_latency_report.json',
    [string]$RuntimeBudgetReportPath = 'evidence/runtime_budget_report.json',
    [string]$CrossfadeObservableReportPath = 'evidence/crossfade_observable_state_report.json'
)

$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'canary_baseline_normalization_report.json'

function Resolve-RepoPath {
    param([string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return $Path
    }
    return (Join-Path $repoRoot $Path)
}

$resolvedRegistryPath = Resolve-RepoPath -Path $RegistryPath
$resolvedMetricGovernanceReportPath = Resolve-RepoPath -Path $MetricGovernanceReportPath
$resolvedRebuildAdmissionMetricsPath = Resolve-RepoPath -Path $RebuildAdmissionMetricsPath
$resolvedRetireLatencyReportPath = Resolve-RepoPath -Path $RetireLatencyReportPath
$resolvedRuntimeBudgetReportPath = Resolve-RepoPath -Path $RuntimeBudgetReportPath
$resolvedCrossfadeObservableReportPath = Resolve-RepoPath -Path $CrossfadeObservableReportPath

if (-not (Test-Path -LiteralPath $resolvedRegistryPath)) {
    throw "Missing metric governance registry: $resolvedRegistryPath"
}
if (-not (Test-Path -LiteralPath $resolvedMetricGovernanceReportPath)) {
    throw "Missing metric governance report: $resolvedMetricGovernanceReportPath"
}
if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -Path $evidenceDir -ItemType Directory | Out-Null
}

$strictEvidence = ($env:ISR_REQUIRE_RUNTIME_EVIDENCE -eq '1')
$registry = Get-Content -LiteralPath $resolvedRegistryPath -Raw -Encoding UTF8 | ConvertFrom-Json
$metricGovernanceReport = Get-Content -LiteralPath $resolvedMetricGovernanceReportPath -Raw -Encoding UTF8 | ConvertFrom-Json

if ($registry.schema -ne 'metric_governance_v1') {
    throw "Unexpected metric governance schema: $($registry.schema)"
}
if ($metricGovernanceReport.schema -ne 'metric_governance_report_v2') {
    throw "Unexpected metric governance report schema: $($metricGovernanceReport.schema)"
}
if ($null -eq $registry.normalizationPolicy) {
    throw 'Metric governance registry missing normalizationPolicy'
}
if ($null -eq $metricGovernanceReport.normalizationPolicy) {
    throw 'Metric governance report missing normalizationPolicy'
}

$strictModeRequireAllMetrics = [bool]$registry.normalizationPolicy.strictModeRequireAllMetrics

$requiredMetricIds = @('xrunDelta', 'callbackJitter', 'retireLatency', 'crossfadePeak')
$warnings = New-Object System.Collections.Generic.List[string]
$violations = New-Object System.Collections.Generic.List[string]
$metricObservations = New-Object System.Collections.Generic.List[object]

$rebuildMetricsReport = $null
if (Test-Path -LiteralPath $resolvedRebuildAdmissionMetricsPath) {
    $rebuildMetricsReport = Get-Content -LiteralPath $resolvedRebuildAdmissionMetricsPath -Raw -Encoding UTF8 | ConvertFrom-Json
}

$retireLatencyReport = $null
if (Test-Path -LiteralPath $resolvedRetireLatencyReportPath) {
    $retireLatencyReport = Get-Content -LiteralPath $resolvedRetireLatencyReportPath -Raw -Encoding UTF8 | ConvertFrom-Json
}

$runtimeBudgetReport = $null
if (Test-Path -LiteralPath $resolvedRuntimeBudgetReportPath) {
    $runtimeBudgetReport = Get-Content -LiteralPath $resolvedRuntimeBudgetReportPath -Raw -Encoding UTF8 | ConvertFrom-Json
}

$crossfadeObservableReport = $null
if (Test-Path -LiteralPath $resolvedCrossfadeObservableReportPath) {
    $crossfadeObservableReport = Get-Content -LiteralPath $resolvedCrossfadeObservableReportPath -Raw -Encoding UTF8 | ConvertFrom-Json
}

foreach ($metricId in $requiredMetricIds) {
    $entry = $registry.metrics | Where-Object { $_.id -eq $metricId } | Select-Object -First 1
    if ($null -eq $entry) {
        $violations.Add("Missing metric governance entry: id=$metricId")
        continue
    }

    $evidenceFound = $false
    $evidencePath = $null
    $evidenceSchema = $null
    $evidenceSummary = $null

    switch ($metricId) {
        'xrunDelta' {
            if ($null -ne $rebuildMetricsReport -and "$($rebuildMetricsReport.schema)" -eq 'rebuild_admission_8_1_metrics_report_v1') {
                $evidenceFound = $true
                $evidencePath = $resolvedRebuildAdmissionMetricsPath
                $evidenceSchema = "$($rebuildMetricsReport.schema)"
                $evidenceSummary = "status=$($rebuildMetricsReport.status) reason=$($rebuildMetricsReport.reason) readyToClose8_1=$($rebuildMetricsReport.readyToClose8_1)"

                if ($strictEvidence -and "$($rebuildMetricsReport.status)" -ne 'evaluated') {
                    $violations.Add("Strict mode requires evaluated xrunDelta evidence: status=$($rebuildMetricsReport.status) reason=$($rebuildMetricsReport.reason)")
                }
            }
        }
        'callbackJitter' {
            if ($null -ne $runtimeBudgetReport -and "$($runtimeBudgetReport.schema)" -eq 'runtime_budget_report_v1') {
                $evidenceFound = $true
                $evidencePath = $resolvedRuntimeBudgetReportPath
                $evidenceSchema = "$($runtimeBudgetReport.schema)"
                $evidenceSummary = "provenance=$($runtimeBudgetReport.provenance)"
            }
        }
        'retireLatency' {
            if ($null -ne $retireLatencyReport -and "$($retireLatencyReport.schema)" -eq 'retire_latency_report_v1') {
                $evidenceFound = $true
                $evidencePath = $resolvedRetireLatencyReportPath
                $evidenceSchema = "$($retireLatencyReport.schema)"
                $evidenceSummary = "withinThreshold=$($retireLatencyReport.withinThreshold)"
            }
        }
        'crossfadePeak' {
            if ($null -ne $crossfadeObservableReport -and "$($crossfadeObservableReport.schema)" -eq 'crossfade_observable_state_report_v1') {
                $evidenceFound = $true
                $evidencePath = $resolvedCrossfadeObservableReportPath
                $evidenceSchema = "$($crossfadeObservableReport.schema)"
                $evidenceSummary = "violations=$(@($crossfadeObservableReport.violations).Count)"
            }
        }
    }

    $normalizationApplied = ("$($entry.normalization)" -eq 'baselineWindowNormalized')
    if (-not $normalizationApplied) {
        $violations.Add("Metric normalization contract mismatch: id=$metricId normalization=$($entry.normalization)")
    }

    $isBlocking = ("$($entry.blocking)" -eq 'yes')
    if (-not $evidenceFound) {
        $warnings.Add("Canary evidence missing: id=$metricId expectedNormalization=baselineWindowNormalized")
        if ($strictEvidence -and ($strictModeRequireAllMetrics -or $isBlocking)) {
            $violations.Add("Strict mode requires evidence for canary metric: id=$metricId strictModeRequireAllMetrics=$strictModeRequireAllMetrics")
        }
    }

    $metricObservations.Add([ordered]@{
            id = "$metricId"
            blocking = "$($entry.blocking)"
            normalization = "$($entry.normalization)"
            normalizationApplied = $normalizationApplied
            evidenceFound = $evidenceFound
            evidencePath = $evidencePath
            evidenceSchema = $evidenceSchema
            evidenceSummary = $evidenceSummary
        }) | Out-Null
}

$report = [ordered]@{
    schema = 'canary_baseline_normalization_report_v1'
    generatedAt = (Get-Date -Format 'o')
    strictEvidence = $strictEvidence
    registryPath = $resolvedRegistryPath
    metricGovernanceReportPath = $resolvedMetricGovernanceReportPath
    normalizationPolicy = [ordered]@{
        enabled = [bool]$registry.normalizationPolicy.enabled
        baselineWindowMinutes = [int]("$($registry.normalizationPolicy.baselineWindowMinutes)")
        cpuThermalOsNormalization = [bool]$registry.normalizationPolicy.cpuThermalOsNormalization
        bucketBy = "$($registry.normalizationPolicy.bucketBy)"
        strictModeRequireAllMetrics = $strictModeRequireAllMetrics
        issue = "$($registry.normalizationPolicy.issue)"
    }
    metrics = $metricObservations
    warnings = $warnings
    violations = $violations
}

Set-Content -LiteralPath $reportPath -Value ($report | ConvertTo-Json -Depth 8) -Encoding UTF8
Write-Host "[INFO] canary baseline normalization report written: $reportPath"

if ($warnings.Count -gt 0) {
    foreach ($warning in $warnings) {
        Write-Host "[WARN] $warning"
    }
}

if ($violations.Count -gt 0) {
    foreach ($violation in $violations) {
        Write-Host "[ERROR] $violation"
    }
    throw "Canary baseline normalization violations detected. count=$($violations.Count)"
}

if ($warnings.Count -gt 0) {
    Write-Host '[PASS] canary baseline normalization gate completed (monitor mode with warnings)'
}
else {
    Write-Host '[PASS] canary baseline normalization gate verified'
}
