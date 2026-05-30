$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$evidencePath = Join-Path $repoRoot 'evidence\shadow_compare_cadence.json'

if (-not (Test-Path -LiteralPath $evidencePath)) {
    throw "Missing shadow compare cadence evidence: $evidencePath"
}

$data = Get-Content -LiteralPath $evidencePath -Raw -Encoding UTF8 | ConvertFrom-Json
if ("$($data.schema)" -ne 'shadow_compare_cadence_v1') {
    throw "Schema mismatch for shadow compare cadence evidence. expected=shadow_compare_cadence_v1 actual=$($data.schema)"
}

foreach ($field in @('minCadenceMs', 'burstWindowMs', 'burstEscalationThreshold', 'totalObservations', 'mismatchCount', 'monotonicViolationCount', 'cadenceViolationCount', 'escalationCount', 'lastSequenceId')) {
    if ($null -eq $data.PSObject.Properties[$field]) {
        throw "Missing required field in shadow compare cadence evidence: $field"
    }
}

if ([int]$data.minCadenceMs -gt 1000) {
    throw "Shadow compare cadence threshold too loose. minCadenceMs=$($data.minCadenceMs)"
}

Write-Host "[PASS] shadow compare cadence verification passed (observations=$($data.totalObservations), cadenceViolations=$($data.cadenceViolationCount), escalations=$($data.escalationCount))"
