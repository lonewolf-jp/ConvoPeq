$ErrorActionPreference = 'Stop'

$data = . "$PSScriptRoot\isr-verify-common.ps1" -ArtifactName "closure_graph.json" -Schema "closure_graph_v1" -RequiredKeys @("status")

Assert-ValueInSet -Value $data.status -Allowed @("generated", "valid", "invalid") -FieldName "status"

if ($data.PSObject.Properties.Name -contains "nodeCount") {
    Assert-NonNegativeInteger -Value $data.nodeCount -FieldName "nodeCount"
}

if ($data.PSObject.Properties.Name -contains "edgeCount") {
    Assert-NonNegativeInteger -Value $data.edgeCount -FieldName "edgeCount"
}

if ($data.PSObject.Properties.Name -contains "validationErrors") {
    Assert-IsArray -Value $data.validationErrors -FieldName "validationErrors"
}

if ($data.status -eq "invalid") {
    $messages = @()
    if ($data.PSObject.Properties.Name -contains "validationErrors") {
        $messages = @($data.validationErrors | ForEach-Object { [string]$_ })
    }

    $cycleDetected = $false
    foreach ($m in $messages) {
        if ($m -match "cycle") {
            $cycleDetected = $true
            break
        }
    }

    if ($cycleDetected) {
        throw "Ownership cycle detected in closure graph"
    }

    throw "Closure graph validation failed (treated as ownership-cycle gate violation for V10)"
}

Write-Host "[PASS] ownership cycle gate"
