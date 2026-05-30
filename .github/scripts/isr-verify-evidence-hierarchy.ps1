$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'evidence_hierarchy_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null
}

$manifestPath = Join-Path $evidenceDir 'evidence_manifest.json'
$safetyReportPath = Join-Path $evidenceDir 'safety_regression_report.json'
$prSlaReportPath = Join-Path $evidenceDir 'pr_sla_report.json'

$violations = New-Object 'System.Collections.Generic.List[string]'

function Add-Violation {
    param([string]$Message)
    $violations.Add($Message) | Out-Null
}

function Read-JsonOrNull {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
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

$manifest = Read-JsonOrNull -Path $manifestPath
$safety = Read-JsonOrNull -Path $safetyReportPath
$prSla = Read-JsonOrNull -Path $prSlaReportPath

if ($null -eq $manifest) {
    Add-Violation "Missing required evidence: $manifestPath"
}
else {
    if ("$($manifest.schema)" -ne 'evidence_manifest_v1') {
        Add-Violation "Manifest schema mismatch: expected=evidence_manifest_v1 actual=$($manifest.schema)"
    }

    if ($null -eq $manifest.artifacts -or @($manifest.artifacts).Count -eq 0) {
        Add-Violation 'Manifest must include non-empty artifacts list.'
    }
}

if ($null -eq $safety) {
    Add-Violation "Missing required evidence: $safetyReportPath"
}
else {
    if ("$($safety.schema)" -ne 'safety_regression_report_v1') {
        Add-Violation "Safety report schema mismatch: expected=safety_regression_report_v1 actual=$($safety.schema)"
    }

    if ($null -eq $safety.safetyPass -or -not [bool]$safety.safetyPass) {
        Add-Violation 'Safety regression evidence must pass (safetyPass=true).'
    }
}

if ($null -eq $prSla) {
    Add-Violation "Missing required evidence: $prSlaReportPath"
}
else {
    if ("$($prSla.schema)" -ne 'pr_sla_report_v1') {
        Add-Violation "PR SLA report schema mismatch: expected=pr_sla_report_v1 actual=$($prSla.schema)"
    }

    if ($null -eq $prSla.ready -or -not [bool]$prSla.ready) {
        Add-Violation 'PR SLA evidence must be ready=true.'
    }
}

$hierarchy = [ordered]@{
    staticEvidenceReady      = ($null -ne $manifest)
    soakEvidenceReady        = ($null -ne $safety -and [bool]$safety.safetyPass)
    operationalEvidenceReady = ($null -ne $prSla -and [bool]$prSla.ready)
}

if (-not $hierarchy.staticEvidenceReady) {
    Add-Violation 'Evidence hierarchy violation: static evidence layer is not ready.'
}

if (-not $hierarchy.soakEvidenceReady) {
    Add-Violation 'Evidence hierarchy violation: soak evidence layer is not ready.'
}

if (-not $hierarchy.operationalEvidenceReady) {
    Add-Violation 'Evidence hierarchy violation: operational evidence layer is not ready.'
}

$report = [ordered]@{
    schema            = 'evidence_hierarchy_report_v1'
    generatedAt       = (Get-Date -Format 'o')
    manifestPath      = $manifestPath
    safetyReportPath  = $safetyReportPath
    prSlaReportPath   = $prSlaReportPath
    hierarchy         = $hierarchy
    violations        = @($violations)
    ready             = ($violations.Count -eq 0)
}

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] evidence hierarchy report written: $reportPath"

if ($violations.Count -gt 0) {
    foreach ($v in $violations) {
        Write-Host "[ERROR] $v"
    }
    throw "Evidence hierarchy verification failed. violations=$($violations.Count)"
}

Write-Host '[PASS] evidence hierarchy verification passed'
