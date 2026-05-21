$ErrorActionPreference = 'Stop'
$data = . "$PSScriptRoot\isr-verify-common.ps1" -ArtifactName "payload_tier_report.json" -Schema "payload_tier_report_v1" -RequiredKeys @("status", "families")

Assert-ValueInSet -Value $data.status -Allowed @("generated", "ok") -FieldName "status"

if ($data.families -isnot [System.Collections.IEnumerable]) {
    throw "Field 'families' must be an array"
}

$requiredFamilies = @("activeNode", "fadingNode", "transitionNext", "retireSlot")
$observedFamilies = @{}

foreach ($family in $data.families) {
    if (-not ($family.PSObject.Properties.Name -contains "name") -or -not ($family.PSObject.Properties.Name -contains "tier")) {
        throw "Each family entry must include 'name' and 'tier'"
    }

    if ([string]::IsNullOrWhiteSpace([string]$family.name) -or [string]::IsNullOrWhiteSpace([string]$family.tier)) {
        throw "Family name/tier must not be empty"
    }

    $observedFamilies[[string]$family.name] = $true
}

foreach ($required in $requiredFamilies) {
    if (-not $observedFamilies.ContainsKey($required)) {
        throw "Missing payload tier family assignment: $required"
    }
}

if ($data.PSObject.Properties.Name -contains "violations") {
    Assert-NonNegativeInteger -Value $data.violations -FieldName "violations"
    if ([long]$data.violations -gt 0) {
        throw "Payload tier violations detected. violations=$($data.violations)"
    }
}
