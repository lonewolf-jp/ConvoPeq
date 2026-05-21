$ErrorActionPreference = 'Stop'
$latency = . "$PSScriptRoot\isr-verify-common.ps1" -ArtifactName "retire_latency_report.json" -Schema "retire_latency_report_v1" -RequiredKeys @("withinThreshold")

if ($latency.withinThreshold -isnot [bool]) {
	throw "Field 'withinThreshold' must be boolean"
}

if (-not $latency.withinThreshold) {
	throw "Retire latency threshold exceeded"
}

$timeline = . "$PSScriptRoot\isr-verify-common.ps1" -ArtifactName "retire_timeline.json" -Schema "retire_timeline_v1" -RequiredKeys @("totalTransitions", "epochMode", "rollbackMode", "rollbackReady")
Assert-NonNegativeInteger -Value $timeline.totalTransitions -FieldName "totalTransitions"

Assert-ValueInSet -Value $timeline.epochMode -Allowed @("shared", "split", "hybrid") -FieldName "epochMode"
Assert-ValueInSet -Value $timeline.rollbackMode -Allowed @("shared", "split", "hybrid") -FieldName "rollbackMode"

if ($timeline.rollbackReady -isnot [bool]) {
	throw "Field 'rollbackReady' must be boolean"
}

if ($timeline.PSObject.Properties.Name -contains "laneCounters") {
	Assert-HasProperty -Object $timeline.laneCounters -Name "rtIntent"
	Assert-HasProperty -Object $timeline.laneCounters -Name "coordination"
	Assert-HasProperty -Object $timeline.laneCounters -Name "epoch"
	Assert-HasProperty -Object $timeline.laneCounters -Name "reclaim"
	Assert-HasProperty -Object $timeline.laneCounters -Name "quarantine"

	Assert-NonNegativeInteger -Value $timeline.laneCounters.rtIntent -FieldName "laneCounters.rtIntent"
	Assert-NonNegativeInteger -Value $timeline.laneCounters.coordination -FieldName "laneCounters.coordination"
	Assert-NonNegativeInteger -Value $timeline.laneCounters.epoch -FieldName "laneCounters.epoch"
	Assert-NonNegativeInteger -Value $timeline.laneCounters.reclaim -FieldName "laneCounters.reclaim"
	Assert-NonNegativeInteger -Value $timeline.laneCounters.quarantine -FieldName "laneCounters.quarantine"
}
