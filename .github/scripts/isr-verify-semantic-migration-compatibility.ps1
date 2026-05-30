$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'semantic_migration_compatibility_report.json'

$ownershipReportPath = Join-Path $evidenceDir 'ownership_migration_report.json'
$rollbackReportPath = Join-Path $evidenceDir 'rollback_compatibility_report.json'

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

$ownership = Read-JsonOrNull -Path $ownershipReportPath
$rollback = Read-JsonOrNull -Path $rollbackReportPath

if ($null -ne $ownership) {
    if ("$($ownership.schema)" -ne 'ownership_migration_report_v2') {
        Add-Violation "Ownership migration schema mismatch: expected=ownership_migration_report_v2 actual=$($ownership.schema)"
    }

    if ($null -eq $ownership.allStepsSatisfied -or -not [bool]$ownership.allStepsSatisfied) {
        Add-Violation 'Ownership migration evidence must satisfy allStepsSatisfied=true.'
    }

    if ($null -eq $ownership.violations) {
        Add-Violation 'Ownership migration evidence missing violations field.'
    }
    elseif (@($ownership.violations).Count -gt 0) {
        Add-Violation "Ownership migration evidence contains violations: count=$(@($ownership.violations).Count)"
    }
}

if ($null -ne $rollback) {
    if ("$($rollback.schema)" -ne 'rollback_compatibility_report_v1') {
        Add-Violation "Rollback compatibility schema mismatch: expected=rollback_compatibility_report_v1 actual=$($rollback.schema)"
    }

    if ($null -eq $rollback.violations) {
        Add-Violation 'Rollback compatibility evidence missing violations field.'
    }
    elseif (@($rollback.violations).Count -gt 0) {
        Add-Violation "Rollback compatibility evidence contains violations: count=$(@($rollback.violations).Count)"
    }

    if ($null -eq $rollback.scenarioCount -or [int]$rollback.scenarioCount -le 0) {
        Add-Violation 'Rollback compatibility evidence must report scenarioCount > 0.'
    }
}

$report = @{
    schema              = 'semantic_migration_compatibility_report_v1'
    generatedAt         = (Get-Date -Format 'o')
    ownershipReportPath = $ownershipReportPath
    rollbackReportPath  = $rollbackReportPath
    ownershipReady      = ($null -ne $ownership -and [bool]$ownership.allStepsSatisfied -and @($ownership.violations).Count -eq 0)
    rollbackReady       = ($null -ne $rollback -and @($rollback.violations).Count -eq 0 -and [int]$rollback.scenarioCount -gt 0)
    violations          = $violations
    ready               = ($violations.Count -eq 0)
}

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] semantic migration compatibility report written: $reportPath"

if ($violations.Count -gt 0) {
    foreach ($v in $violations) {
        Write-Host "[ERROR] $v"
    }
    throw "Semantic migration compatibility verification failed. violations=$($violations.Count)"
}

Write-Host '[PASS] semantic migration compatibility verification passed'
