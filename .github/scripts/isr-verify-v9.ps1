$ErrorActionPreference = 'Stop'
$data = . "$PSScriptRoot\isr-verify-common.ps1" -ArtifactName "payload_tier_report.json" -Schema "payload_tier_report_v1" -RequiredKeys @("status")

Assert-ValueInSet -Value $data.status -Allowed @("generated") -FieldName "status"

if ($data.PSObject.Properties.Name -contains "violations") {
	Assert-NonNegativeInteger -Value $data.violations -FieldName "violations"
	if ([long]$data.violations -gt 0) {
		throw "Forbidden capability scan failed. violations=$($data.violations)"
	}
}

if ($data.PSObject.Properties.Name -contains "rtLocalLeaks") {
	Assert-NonNegativeInteger -Value $data.rtLocalLeaks -FieldName "rtLocalLeaks"
	if ([long]$data.rtLocalLeaks -gt 0) {
		throw "RTLocal capability leak detected. rtLocalLeaks=$($data.rtLocalLeaks)"
	}
}
