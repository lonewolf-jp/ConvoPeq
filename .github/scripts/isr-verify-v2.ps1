$ErrorActionPreference = 'Stop'
$data = . "$PSScriptRoot\isr-verify-common.ps1" -ArtifactName "mutation_fault_trace.json" -Schema "mutation_fault_trace_v1" -RequiredKeys @("status")

Assert-ValueInSet -Value $data.status -Allowed @("generated", "ok") -FieldName "status"

if ($data.PSObject.Properties.Name -contains "violations") {
    Assert-NonNegativeInteger -Value $data.violations -FieldName "violations"
    if ([long]$data.violations -gt 0) {
        throw "Seal integrity violations detected. violations=$($data.violations)"
    }
}
