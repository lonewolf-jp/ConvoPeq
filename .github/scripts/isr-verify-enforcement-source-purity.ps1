param(
    [string]$TriggerAuditReportPath = 'evidence/trigger_audit_report.json',
    [switch]$RequireAstEvidence,
    [string]$PolicyPath = '.github/isr-enforcement-source-policy.json'
)

$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'enforcement_source_purity_report.json'
$resolvedTriggerAuditPath = if ([System.IO.Path]::IsPathRooted($TriggerAuditReportPath)) { $TriggerAuditReportPath } else { Join-Path $repoRoot $TriggerAuditReportPath }
$resolvedPolicyPath = if ([System.IO.Path]::IsPathRooted($PolicyPath)) { $PolicyPath } else { Join-Path $repoRoot $PolicyPath }

foreach ($path in @($resolvedTriggerAuditPath, $resolvedPolicyPath)) {
    if (-not (Test-Path $path)) {
        throw "Missing required file: $path"
    }
}
if (-not (Test-Path $evidenceDir)) {
    New-Item -Path $evidenceDir -ItemType Directory | Out-Null
}

$policy = Get-Content -LiteralPath $resolvedPolicyPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($policy.schema -ne 'isr_enforcement_source_policy_v1') {
    throw "Unexpected enforcement source policy schema: $($policy.schema)"
}
foreach ($field in @('owner', 'issue', 'rationale', 'expiry')) {
    if ($null -eq $policy.$field -or [string]::IsNullOrWhiteSpace("$($policy.$field)")) {
        throw "enforcement source policy missing required field: $field"
    }
}
$policyExpiry = [datetime]::ParseExact("$($policy.expiry)", 'yyyy-MM-dd', [System.Globalization.CultureInfo]::InvariantCulture)
if ((Get-Date).Date -gt $policyExpiry.Date) {
    throw "enforcement source policy expired: $($policy.expiry)"
}

$allowedRawRegexFields = @()
if ($policy.allowedRawRegexFields) {
    $allowedRawRegexFields = @($policy.allowedRawRegexFields | ForEach-Object { "$_" })
}
$allowedUnknownFields = @()
if ($policy.allowedUnknownFields) {
    $allowedUnknownFields = @($policy.allowedUnknownFields | ForEach-Object { "$_" })
}

$triggerAudit = Get-Content -LiteralPath $resolvedTriggerAuditPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($triggerAudit.schema -ne 'trigger_audit_report_v1') {
    throw "Unexpected trigger audit schema: $($triggerAudit.schema)"
}

if ($RequireAstEvidence) {
    if ($null -eq $triggerAudit.astEvidenceRequired -or -not [bool]$triggerAudit.astEvidenceRequired) {
        throw 'trigger audit astEvidenceRequired must be true when RequireAstEvidence is specified'
    }
}

$sourceFields = @(
    'activeDspMetricSource',
    'fadingOutDspMetricSource',
    'retireFacadeMetricSource',
    'retireFacadeRuntimeExecutionMetricSource',
    'runtimeExecutionViewMetricSource',
    'legacyDirectObserveMetricSource'
)

$rawRegexViolations = New-Object System.Collections.Generic.List[string]
$unknownViolations = New-Object System.Collections.Generic.List[string]
$details = New-Object System.Collections.Generic.List[object]

foreach ($field in $sourceFields) {
    if (-not ($triggerAudit.PSObject.Properties.Name -contains $field)) {
        throw "trigger audit missing source field: $field"
    }

    $source = "$($triggerAudit.$field)"
    $classification = 'advanced'

    if ($source -match '^rawRegex') {
        $classification = 'rawRegex'
        if ($allowedRawRegexFields -notcontains $field) {
            $rawRegexViolations.Add("rawRegex source is not allowed: field=$field source=$source")
        }
    }
    elseif ($source -match '^(symbol|triggerAst|observeShim)') {
        $classification = 'advanced'
    }
    else {
        $classification = 'unknown'
        if ($allowedUnknownFields -notcontains $field) {
            $unknownViolations.Add("unknown source is not allowed: field=$field source=$source")
        }
    }

    $details.Add([ordered]@{
        field = $field
        source = $source
        classification = $classification
    }) | Out-Null
}

$report = [ordered]@{
    schema = 'enforcement_source_purity_report_v1'
    generatedAt = (Get-Date -Format 'o')
    policyPath = $resolvedPolicyPath
    triggerAuditReportPath = $resolvedTriggerAuditPath
    allowedRawRegexFields = $allowedRawRegexFields
    allowedUnknownFields = $allowedUnknownFields
    rawRegexViolationCount = $rawRegexViolations.Count
    unknownViolationCount = $unknownViolations.Count
    rawRegexViolations = $rawRegexViolations
    unknownViolations = $unknownViolations
    details = $details
}

$reportJson = $report | ConvertTo-Json -Depth 8
Set-Content -LiteralPath $reportPath -Value $reportJson -Encoding UTF8
Write-Host "[INFO] enforcement source purity report written: $reportPath"

if ($rawRegexViolations.Count -gt 0 -or $unknownViolations.Count -gt 0) {
    foreach ($v in $rawRegexViolations) { Write-Host "[ERROR] $v" }
    foreach ($v in $unknownViolations) { Write-Host "[ERROR] $v" }
    throw "enforcement source purity gate failed. rawRegex=$($rawRegexViolations.Count) unknown=$($unknownViolations.Count)"
}

Write-Host '[PASS] enforcement source purity gate verified'
