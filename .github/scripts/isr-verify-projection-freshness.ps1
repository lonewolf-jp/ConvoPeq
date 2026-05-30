param(
    [int]$MaxStalenessSec = 600
)

$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'projection_freshness_report.json'

$triggerAuditReportPath = Join-Path $evidenceDir 'trigger_audit_report.json'
$observeShimReportPath = Join-Path $evidenceDir 'observe_shim_usage_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null
}

$violations = New-Object 'System.Collections.Generic.List[string]'

function Add-Violation {
    param([string]$Message)
    $violations.Add($Message) | Out-Null
}

function Read-JsonOrNull {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        Add-Violation "Missing required evidence: $Path"
        return $null
    }

    try {
        return (Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json)
    }
    catch {
        Add-Violation "Invalid JSON format: $Path"
        return $null
    }
}

function Get-ReportAgeSec {
    param([Parameter(Mandatory = $true)]$Report)

    if ($null -eq $Report.generatedAt) {
        return $null
    }

    try {
        $generatedAt = [datetime]::Parse("$($Report.generatedAt)")
        return [int][Math]::Floor(((Get-Date) - $generatedAt).TotalSeconds)
    }
    catch {
        return $null
    }
}

$triggerAudit = Read-JsonOrNull -Path $triggerAuditReportPath
$observeShim = Read-JsonOrNull -Path $observeShimReportPath

$triggerAuditAgeSec = $null
$observeShimAgeSec = $null

if ($null -ne $triggerAudit) {
    if ("$($triggerAudit.schema)" -ne 'trigger_audit_report_v1') {
        Add-Violation "Trigger audit schema mismatch: expected=trigger_audit_report_v1 actual=$($triggerAudit.schema)"
    }

    $triggerAuditAgeSec = Get-ReportAgeSec -Report $triggerAudit
    if ($null -eq $triggerAuditAgeSec) {
        Add-Violation 'Trigger audit freshness check failed: generatedAt is missing or invalid.'
    }
    elseif ($triggerAuditAgeSec -gt $MaxStalenessSec) {
        Add-Violation "Trigger audit report is stale: ageSec=$triggerAuditAgeSec limitSec=$MaxStalenessSec"
    }

    if ($null -eq $triggerAudit.metrics -or $null -eq $triggerAudit.metrics.runtimeExecutionViewUsageCount) {
        Add-Violation 'Trigger audit report missing metrics.runtimeExecutionViewUsageCount'
    }
    else {
        $usageCount = [int]$triggerAudit.metrics.runtimeExecutionViewUsageCount
        if ($usageCount -gt 0) {
            Add-Violation "Projection freshness violation: runtimeExecutionViewUsageCount must be 0 but was $usageCount"
        }
    }
}

if ($null -ne $observeShim) {
    if ("$($observeShim.schema)" -ne 'observe_shim_usage_report_v1') {
        Add-Violation "Observe shim schema mismatch: expected=observe_shim_usage_report_v1 actual=$($observeShim.schema)"
    }

    $observeShimAgeSec = Get-ReportAgeSec -Report $observeShim
    if ($null -eq $observeShimAgeSec) {
        Add-Violation 'Observe shim freshness check failed: generatedAt is missing or invalid.'
    }
    elseif ($observeShimAgeSec -gt $MaxStalenessSec) {
        Add-Violation "Observe shim report is stale: ageSec=$observeShimAgeSec limitSec=$MaxStalenessSec"
    }

    if ($null -eq $observeShim.blockedMatches) {
        Add-Violation 'Observe shim report missing blockedMatches'
    }
    else {
        $blocked = [int]$observeShim.blockedMatches
        if ($blocked -gt 0) {
            Add-Violation "Projection freshness violation: observe shim blockedMatches must be 0 but was $blocked"
        }
    }
}

$report = @{
    schema              = 'projection_freshness_report_v1'
    generatedAt         = (Get-Date -Format 'o')
    maxStalenessSec     = $MaxStalenessSec
    triggerAuditPath    = $triggerAuditReportPath
    triggerAuditAgeSec  = $triggerAuditAgeSec
    observeShimPath     = $observeShimReportPath
    observeShimAgeSec   = $observeShimAgeSec
    violations          = $violations
    ready               = ($violations.Count -eq 0)
}

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] projection freshness report written: $reportPath"

if ($violations.Count -gt 0) {
    foreach ($v in $violations) {
        Write-Host "[ERROR] $v"
    }
    throw "Projection freshness verification failed. violations=$($violations.Count)"
}

Write-Host '[PASS] projection freshness verification passed'
