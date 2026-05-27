param(
    [string]$BacklogPath = 'doc/work/ISR_Completeness_Risk_Backlog.md',
    [switch]$EnforceNoSpecFixed
)

$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$resolvedBacklogPath = if ([System.IO.Path]::IsPathRooted($BacklogPath)) { $BacklogPath } else { Join-Path $repoRoot $BacklogPath }
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'backlog_specfixed_residual_report.json'

if (-not (Test-Path -LiteralPath $resolvedBacklogPath)) {
    throw "Missing backlog file: $resolvedBacklogPath"
}
if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -Path $evidenceDir -ItemType Directory | Out-Null
}

$lines = Get-Content -LiteralPath $resolvedBacklogPath -Encoding UTF8
$rStatus = New-Object System.Collections.Generic.List[object]

$currentRId = $null
foreach ($line in $lines) {
    if ($line -match '^##\s+R(?<id>\d+)\.') {
        $currentRId = [int]$Matches.id
        continue
    }

    if ($null -eq $currentRId) {
        continue
    }

    if ($line -match '^-\s*状態:\s*(?<status>.+)$') {
        $statusText = "$($Matches.status)".Trim()
        $rStatus.Add([pscustomobject]@{
            id = $currentRId
            status = $statusText
            hasSpecFixed = ($statusText -match 'Spec-Fixed')
            hasClosed = ($statusText -match 'Closed')
        }) | Out-Null
        $currentRId = $null
    }
}

$residual = @($rStatus | Where-Object { $_.hasSpecFixed -and -not $_.hasClosed })
$totalRStatusCount = [int]$rStatus.Count
$specFixedResidualCount = [int]$residual.Count

$closureReadinessByR = @{
    1 = @(
        @{ path = '.github/scripts/isr-verify-v1-immutability.ps1'; type = 'script' },
        @{ path = '.github/scripts/isr-verify-v2-seal.ps1'; type = 'script' },
        @{ path = 'evidence/mutation_fault_trace.json'; type = 'artifact'; schema = 'mutation_fault_trace_v1' }
    )
    2 = @(
        @{ path = '.github/scripts/isr-verify-v3-runtime-graph-immutability.ps1'; type = 'script' }
    )
    3 = @(
        @{ path = '.github/scripts/isr-verify-v4-dsp-handle-policy.ps1'; type = 'script' }
    )
    4 = @(
        @{ path = '.github/scripts/isr-verify-v5-retire-authority-lane.ps1'; type = 'script' }
    )
    5 = @(
        @{ path = '.github/scripts/isr-verify-v6-domain-f-ordering.ps1'; type = 'script' }
    )
    6 = @(
        @{ path = 'evidence/hb_graph_trace.json'; type = 'artifact'; schema = 'hb_trace_v1' },
        @{ path = 'evidence/hb_violation_report.json'; type = 'artifact'; schema = 'hb_violation_report_v1' }
    )
    7 = @(
        @{ path = 'evidence/closure_graph.json'; type = 'artifact'; schema = 'closure_graph_v1' }
    )
    8 = @(
        @{ path = 'evidence/shutdown_trace.json'; type = 'artifact'; schema = 'shutdown_trace_v1' }
    )
    9 = @(
        @{ path = '.github/scripts/isr-verify-v7-rt-nonrt-retire-bridge.ps1'; type = 'script' }
    )
    10 = @(
        @{ path = '.github/scripts/isr-verify-v8-shared-split-readiness.ps1'; type = 'script' }
    )
}

$residualWithReadiness = New-Object System.Collections.Generic.List[object]
$promotableResidualCount = 0

foreach ($entry in $residual) {
    $rId = [int]$entry.id
    $requirements = if ($closureReadinessByR.ContainsKey($rId)) { $closureReadinessByR[$rId] } else { @() }
    $missing = New-Object System.Collections.Generic.List[string]

    foreach ($req in $requirements) {
        $absolutePath = Join-Path $repoRoot $req.path
        if (-not (Test-Path -LiteralPath $absolutePath)) {
            $missing.Add("missing:$($req.path)") | Out-Null
            continue
        }

        if ($req.type -eq 'artifact' -and -not [string]::IsNullOrWhiteSpace("$($req.schema)")) {
            try {
                $artifact = Get-Content -LiteralPath $absolutePath -Raw -Encoding UTF8 | ConvertFrom-Json
                if ("$($artifact.schema)" -ne "$($req.schema)") {
                    $missing.Add("schema:$($req.path):expected=$($req.schema):actual=$($artifact.schema)") | Out-Null
                }
            }
            catch {
                $missing.Add("parse:$($req.path)") | Out-Null
            }
        }
    }

    $readinessSatisfied = ($missing.Count -eq 0)
    if ($readinessSatisfied) {
        $promotableResidualCount++
    }

    $residualWithReadiness.Add([pscustomobject]@{
        id = $rId
        status = "$($entry.status)"
        hasSpecFixed = [bool]$entry.hasSpecFixed
        hasClosed = [bool]$entry.hasClosed
        readinessSatisfied = $readinessSatisfied
        readinessMissing = @($missing)
    }) | Out-Null
}

$report = @{}
$report['schema'] = 'backlog_specfixed_residual_report_v1'
$report['generatedAt'] = (Get-Date -Format 'o')
$report['backlogPath'] = $resolvedBacklogPath
$report['enforceNoSpecFixed'] = [bool]$EnforceNoSpecFixed
$report['totalRStatusCount'] = $totalRStatusCount
$report['specFixedResidualCount'] = $specFixedResidualCount
$report['promotableResidualCount'] = $promotableResidualCount
$report['residual'] = [object[]]$residualWithReadiness.ToArray()

Set-Content -LiteralPath $reportPath -Value ($report | ConvertTo-Json -Depth 8) -Encoding UTF8
Write-Host "[INFO] backlog spec-fixed residual report written: $reportPath"
Write-Host "[INFO] totalRStatusCount=$totalRStatusCount specFixedResidualCount=$specFixedResidualCount"

if ($EnforceNoSpecFixed -and $specFixedResidualCount -gt 0) {
    throw "Backlog Spec-Fixed residuals detected under enforce mode. count=$specFixedResidualCount"
}

if ($specFixedResidualCount -gt 0) {
    Write-Host '[WARN] backlog Spec-Fixed residuals remain (monitor mode)'
}

Write-Host '[PASS] backlog Spec-Fixed residual gate completed'
