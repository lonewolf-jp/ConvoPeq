param(
    [switch]$RequireAstEvidence,
    [string]$DeferredRegistryPath = '.github/isr-cleanup-deferred.json'
)

$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$reportPath = Join-Path $repoRoot 'evidence\trigger_audit_report.json'
$sourceRoot = Join-Path $repoRoot 'src'
$resolvedDeferredRegistryPath = if ([System.IO.Path]::IsPathRooted($DeferredRegistryPath)) { $DeferredRegistryPath } else { Join-Path $repoRoot $DeferredRegistryPath }
$expectedTriggerAuditReportPath = Join-Path $repoRoot 'evidence\trigger_audit_report.json'
$expectedDeferredRegistryPath = Join-Path $repoRoot '.github\isr-cleanup-deferred.json'
$expectedSourceRoot = Join-Path $repoRoot 'src'

if (-not (Test-Path $reportPath)) {
    throw "Missing trigger audit report: $reportPath"
}

if (-not (Test-Path $sourceRoot)) {
    throw "Missing source directory: $sourceRoot"
}
if (-not (Test-Path -LiteralPath $resolvedDeferredRegistryPath)) {
    throw "Missing deferred cleanup registry: $resolvedDeferredRegistryPath"
}

$report = Get-Content -LiteralPath $reportPath -Raw -Encoding UTF8 | ConvertFrom-Json
$deferredRegistry = Get-Content -LiteralPath $resolvedDeferredRegistryPath -Raw -Encoding UTF8 | ConvertFrom-Json

$violations = New-Object System.Collections.Generic.List[string]

if ($report.schema -ne 'trigger_audit_report_v1') {
    $violations.Add("Unexpected trigger audit schema: $($report.schema)")
}

if ($RequireAstEvidence) {
    if ($null -eq $report.astEvidenceRequired -or -not [bool]$report.astEvidenceRequired) {
        $violations.Add('Trigger cleanup completion requires trigger_audit_report.astEvidenceRequired=true')
    }
}

if ("$($deferredRegistry.schema)" -ne 'cleanup_deferred_registry_v1') {
    $violations.Add("Unexpected deferred cleanup registry schema: $($deferredRegistry.schema)")
}

$deferredEntryCount = @($deferredRegistry.entries).Count
if ($deferredEntryCount -ne 0) {
    $violations.Add("Trigger cleanup completion requires deferred cleanup registry to be empty, but entries=$deferredEntryCount")
}

$metricMap = [ordered]@{
    activeDspRefCount                 = 'activeDspRefCount'
    fadingOutDspWriteCount            = 'fadingOutDspWriteCount'
    retireFacadeDirectDependencyCount = 'retireFacadeDirectDependencyCount'
    retireFacadeRuntimeExecutionCount = 'retireFacadeRuntimeExecutionCount'
    runtimeExecutionViewUsageCount    = 'runtimeExecutionViewUsageCount'
    legacyDirectObserveUsageCount     = 'legacyDirectObserveUsageCount'
}

if ($null -eq $report.metrics) {
    $violations.Add('Trigger audit report missing metrics block')
}
else {
    foreach ($field in $metricMap.Keys) {
        if ($null -eq $report.metrics.$field) {
            $violations.Add("Trigger audit metrics missing field: $field")
            continue
        }

        if ([int64]$report.metrics.$field -ne 0) {
            $violations.Add("Trigger cleanup not complete: metrics.$field=$($report.metrics.$field)")
        }
    }
}

if ($null -eq $report.policyEvaluations) {
    $violations.Add('Trigger audit report missing policyEvaluations block')
}
else {
    foreach ($entry in $report.policyEvaluations) {
        if ($entry.expired -eq $true) {
            $violations.Add("Trigger policy expired: $($entry.id)")
        }

        if ($null -eq $entry.allowedMax -or $null -eq $entry.actual) {
            $violations.Add("Trigger policy evaluation missing actual/allowedMax: $($entry.id)")
            continue
        }

        if ([double]$entry.actual -gt [double]$entry.allowedMax) {
            $violations.Add("Trigger policy evaluation exceeded allowedMax: $($entry.id) actual=$($entry.actual) allowedMax=$($entry.allowedMax)")
        }
    }
}

$legacyHelperNames = @(
    'getActiveDSP',
    'resolveActiveDSPFromRuntimeWorldOnly',
    'resolveFadingDSPFromRuntimeWorldOnly',
    'exchangeFadingOutDSP'
)

$sourceText = Get-ChildItem -LiteralPath $sourceRoot -Recurse -File | ForEach-Object {
    Get-Content -LiteralPath $_.FullName -Raw -Encoding UTF8
}

foreach ($legacyName in $legacyHelperNames) {
    if (($sourceText -join "`n") -match [regex]::Escape($legacyName)) {
        $violations.Add("Legacy helper name still present in source: $legacyName")
    }
}

$cleanupCompleted = $violations.Count -eq 0

$triggerAuditReportPathSatisfied = [System.IO.Path]::GetFullPath($reportPath) -eq $expectedTriggerAuditReportPath
$deferredRegistryPathSatisfied = [System.IO.Path]::GetFullPath($resolvedDeferredRegistryPath) -eq $expectedDeferredRegistryPath
$sourceRootSatisfied = [System.IO.Path]::GetFullPath($sourceRoot) -eq $expectedSourceRoot

if (-not $triggerAuditReportPathSatisfied) {
    $violations.Add("Trigger cleanup completion reference mismatch: triggerAuditReport expected=$expectedTriggerAuditReportPath actual=$([System.IO.Path]::GetFullPath($reportPath))")
}

if (-not $deferredRegistryPathSatisfied) {
    $violations.Add("Trigger cleanup completion reference mismatch: deferredRegistryPath expected=$expectedDeferredRegistryPath actual=$([System.IO.Path]::GetFullPath($resolvedDeferredRegistryPath))")
}

if (-not $sourceRootSatisfied) {
    $violations.Add("Trigger cleanup completion reference mismatch: sourceRoot expected=$expectedSourceRoot actual=$([System.IO.Path]::GetFullPath($sourceRoot))")
}

$output = [ordered]@{
    schema                     = 'trigger_cleanup_completion_report_v1'
    generatedAt                = (Get-Date -Format 'o')
    triggerAuditReport         = $reportPath
    deferredRegistryPath       = $resolvedDeferredRegistryPath
    deferredRegistryEntryCount = $deferredEntryCount
    sourceRoot                 = $sourceRoot
    referenceConsistency       = [ordered]@{
        triggerAuditReport   = [ordered]@{
            expected  = $expectedTriggerAuditReportPath
            actual    = [System.IO.Path]::GetFullPath($reportPath)
            satisfied = $triggerAuditReportPathSatisfied
        }
        deferredRegistryPath = [ordered]@{
            expected  = $expectedDeferredRegistryPath
            actual    = [System.IO.Path]::GetFullPath($resolvedDeferredRegistryPath)
            satisfied = $deferredRegistryPathSatisfied
        }
        sourceRoot           = [ordered]@{
            expected  = $expectedSourceRoot
            actual    = [System.IO.Path]::GetFullPath($sourceRoot)
            satisfied = $sourceRootSatisfied
        }
    }
    cleanupCompleted           = $cleanupCompleted
    violations                 = $violations
}

$outputPath = Join-Path $repoRoot 'evidence\trigger_cleanup_completion_report.json'
Set-Content -LiteralPath $outputPath -Value ($output | ConvertTo-Json -Depth 6) -Encoding UTF8
Write-Host "[INFO] trigger cleanup completion report written: $outputPath"

if ($violations.Count -gt 0) {
    foreach ($violation in $violations) {
        Write-Host "[ERROR] $violation"
    }
    throw "Trigger cleanup completion violations detected. count=$($violations.Count)"
}

Write-Host '[PASS] trigger cleanup completion gate verified'
