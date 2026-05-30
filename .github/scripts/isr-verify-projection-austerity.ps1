$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$audioRoot = Join-Path $repoRoot 'src\audioengine'
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'projection_austerity_report.json'
$triggerAuditReportPath = Join-Path $evidenceDir 'trigger_audit_report.json'

if (-not (Test-Path -LiteralPath $audioRoot)) {
    throw "Missing audioengine source root: $audioRoot"
}

if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null
}

$forbiddenPatterns = @(
    'cachedResolvedRouting',
    'cachedEffectiveTopology',
    'cachedTransitionInfo'
)

$violations = New-Object 'System.Collections.Generic.List[string]'
$matchEntries = New-Object 'System.Collections.Generic.List[object]'

function Add-Violation {
    param([string]$Message)
    $violations.Add($Message) | Out-Null
}

$files = Get-ChildItem -Path $audioRoot -Recurse -File -Include *.h,*.hpp,*.cpp,*.cxx,*.cc
foreach ($file in $files) {
    $relativePath = $file.FullName.Substring($repoRoot.Length + 1).Replace('\\', '/')
    $lineNumber = 0

    foreach ($line in Get-Content -LiteralPath $file.FullName -Encoding UTF8) {
        $lineNumber++
        foreach ($pattern in $forbiddenPatterns) {
            if ($line -match [regex]::Escape($pattern)) {
                $entry = [ordered]@{
                    pattern = $pattern
                    path    = $relativePath
                    line    = $lineNumber
                }
                $matchEntries.Add($entry) | Out-Null
                Add-Violation "Projection austerity violation: pattern='$pattern' path=$relativePath line=$lineNumber"
            }
        }
    }
}

$triggerAuditReady = $false
$runtimeExecutionViewUsageCount = $null

if (-not (Test-Path -LiteralPath $triggerAuditReportPath)) {
    Add-Violation "Missing trigger audit evidence: $triggerAuditReportPath"
}
else {
    $triggerAudit = Get-Content -LiteralPath $triggerAuditReportPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ("$($triggerAudit.schema)" -ne 'trigger_audit_report_v1') {
        Add-Violation "Trigger audit schema mismatch: expected=trigger_audit_report_v1 actual=$($triggerAudit.schema)"
    }
    else {
        $triggerAuditReady = $true
        if ($null -eq $triggerAudit.metrics -or $null -eq $triggerAudit.metrics.runtimeExecutionViewUsageCount) {
            Add-Violation 'Trigger audit report missing metrics.runtimeExecutionViewUsageCount'
        }
        else {
            $runtimeExecutionViewUsageCount = [int]$triggerAudit.metrics.runtimeExecutionViewUsageCount
            if ($runtimeExecutionViewUsageCount -gt 0) {
                Add-Violation "Projection austerity violation: runtimeExecutionViewUsageCount must be 0 but was $runtimeExecutionViewUsageCount"
            }
        }
    }
}

$report = @{
    schema                         = 'projection_austerity_report_v1'
    generatedAt                    = (Get-Date -Format 'o')
    audioRoot                      = $audioRoot
    forbiddenPatterns              = $forbiddenPatterns
    triggerAuditReportPath         = $triggerAuditReportPath
    triggerAuditReady              = $triggerAuditReady
    runtimeExecutionViewUsageCount = $runtimeExecutionViewUsageCount
    totalMatches                   = $matchEntries.Count
    matches                        = $matchEntries
    violations                     = $violations
    ready                          = ($violations.Count -eq 0)
}

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] projection austerity report written: $reportPath"

if ($violations.Count -gt 0) {
    foreach ($v in $violations) {
        Write-Host "[ERROR] $v"
    }
    throw "Projection austerity verification failed. violations=$($violations.Count)"
}

Write-Host '[PASS] projection austerity verification passed'
