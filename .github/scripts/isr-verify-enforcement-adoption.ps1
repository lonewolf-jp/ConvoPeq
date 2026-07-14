param(
    [string]$TriggerAuditReportPath = 'evidence/trigger_audit_report.json',
    [string]$PolicyPath = '.github/isr-enforcement-adoption-policy.json',
    [ValidateSet('smoke', 'standard', 'exhaustive')]
    [string]$Tier = 'standard',
    [double]$MinAdvancedCoverageRatio = [double]::NaN
)

$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'enforcement_adoption_report.json'
$resolvedTriggerAuditPath = if ([System.IO.Path]::IsPathRooted($TriggerAuditReportPath)) { $TriggerAuditReportPath } else { Join-Path $repoRoot $TriggerAuditReportPath }
$resolvedPolicyPath = if ([System.IO.Path]::IsPathRooted($PolicyPath)) { $PolicyPath } else { Join-Path $repoRoot $PolicyPath }

if (-not (Test-Path $resolvedTriggerAuditPath)) {
    Write-Host "[SKIP] Trigger audit report not found: $resolvedTriggerAuditPath (generated in CI only)" -ForegroundColor Yellow
    Write-Host "[SKIP] Enforcement adoption check requires CI-generated evidence files. Skipping." -ForegroundColor Yellow
    exit 0
}
if (-not (Test-Path $resolvedPolicyPath)) {
    Write-Host "[SKIP] Enforcement policy not found: $resolvedPolicyPath (generated in CI only)" -ForegroundColor Yellow
    Write-Host "[SKIP] Enforcement adoption check requires CI-generated evidence files. Skipping." -ForegroundColor Yellow
    exit 0
}
if (-not (Test-Path $evidenceDir)) {
    New-Item -Path $evidenceDir -ItemType Directory | Out-Null
}

$policy = Get-Content -LiteralPath $resolvedPolicyPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($policy.schema -ne 'isr_enforcement_adoption_policy_v1') {
    throw "Unexpected enforcement adoption policy schema: $($policy.schema)"
}

foreach ($field in @('owner', 'issue', 'rationale', 'expiry')) {
    if ([string]::IsNullOrWhiteSpace("$($policy.$field)")) {
        throw "enforcement adoption policy missing required field: $field"
    }
}

$expiry = [datetime]::ParseExact("$($policy.expiry)", 'yyyy-MM-dd', [System.Globalization.CultureInfo]::InvariantCulture)
if ((Get-Date).Date -gt $expiry.Date) {
    throw "enforcement adoption policy expired: $($policy.expiry)"
}

if ([double]::IsNaN($MinAdvancedCoverageRatio)) {
    if (-not ($policy.thresholds.PSObject.Properties.Name -contains $Tier)) {
        throw "enforcement adoption policy missing tier threshold: $Tier"
    }
    $MinAdvancedCoverageRatio = [double]$policy.thresholds.$Tier
}

$triggerAudit = Get-Content -LiteralPath $resolvedTriggerAuditPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($triggerAudit.schema -ne 'trigger_audit_report_v1') {
    throw "Unexpected trigger audit schema: $($triggerAudit.schema)"
}

$sourceFields = @(
    'activeDspMetricSource',
    'fadingOutDspMetricSource',
    'retireFacadeMetricSource',
    'runtimeExecutionViewMetricSource',
    'legacyDirectObserveMetricSource'
)

$advancedPattern = '^(symbol|triggerAst|observeShim)'
$total = 0
$advanced = 0
$rawRegex = 0
$unknown = 0
$details = New-Object System.Collections.Generic.List[object]

foreach ($field in $sourceFields) {
    if (-not ($triggerAudit.PSObject.Properties.Name -contains $field)) {
        throw "Trigger audit report missing source field: $field"
    }

    $source = "$($triggerAudit.$field)"
    $classification = 'unknown'
    if ($source -match $advancedPattern) {
        $classification = 'advanced'
        $advanced++
    }
    elseif ($source -match '^rawRegex') {
        $classification = 'rawRegex'
        $rawRegex++
    }
    else {
        $unknown++
    }

    $total++
    $details.Add([ordered]@{
        field = $field
        source = $source
        classification = $classification
    }) | Out-Null
}

$ratio = if ($total -gt 0) { [double]$advanced / [double]$total } else { 0.0 }
$withinTarget = ($ratio -ge $MinAdvancedCoverageRatio)

$report = [ordered]@{
    schema = 'enforcement_adoption_report_v1'
    generatedAt = (Get-Date -Format 'o')
    triggerAuditReportPath = $resolvedTriggerAuditPath
    policyPath = $resolvedPolicyPath
    tier = $Tier
    minAdvancedCoverageRatio = $MinAdvancedCoverageRatio
    totalTrackedSources = $total
    advancedSourceCount = $advanced
    rawRegexSourceCount = $rawRegex
    unknownSourceCount = $unknown
    advancedCoverageRatio = [math]::Round($ratio, 4)
    withinTarget = $withinTarget
    details = $details
}

$reportJson = $report | ConvertTo-Json -Depth 8
Set-Content -LiteralPath $reportPath -Value $reportJson -Encoding UTF8
Write-Host "[INFO] enforcement adoption report written: $reportPath"
Write-Host "[INFO] advancedCoverageRatio=$($report.advancedCoverageRatio) target>=$MinAdvancedCoverageRatio"

if (-not $withinTarget) {
    throw "Advanced enforcement adoption ratio below threshold. ratio=$($report.advancedCoverageRatio) threshold=$MinAdvancedCoverageRatio"
}

Write-Host '[PASS] enforcement adoption gate verified'
