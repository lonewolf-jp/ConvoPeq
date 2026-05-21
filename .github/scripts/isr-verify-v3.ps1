$ErrorActionPreference = 'Stop'
$data = . "$PSScriptRoot\isr-verify-common.ps1" -ArtifactName "closure_graph.json" -Schema "closure_graph_v1" -RequiredKeys @("status", "descriptorCoverageComplete", "externalMutableDependencies")

Assert-ValueInSet -Value $data.status -Allowed @("generated", "valid", "invalid") -FieldName "status"
Assert-NonNegativeInteger -Value $data.externalMutableDependencies -FieldName "externalMutableDependencies"

if ($data.descriptorCoverageComplete -isnot [bool]) {
    throw "Field 'descriptorCoverageComplete' must be boolean"
}

if (-not $data.descriptorCoverageComplete) {
    throw "Closure descriptor coverage is incomplete"
}

if ([long]$data.externalMutableDependencies -gt 0) {
    throw "External mutable dependencies detected. count=$($data.externalMutableDependencies)"
}

if ($data.status -eq "invalid") {
    throw "Recursive closure validation failed"
}
