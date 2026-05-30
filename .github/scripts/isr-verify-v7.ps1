$ErrorActionPreference = 'Stop'
$latency = . "$PSScriptRoot\isr-verify-common.ps1" -ArtifactName "retire_latency_report.json" -Schema "retire_latency_report_v1" -RequiredKeys @("withinThreshold")

if ($latency.withinThreshold -isnot [bool]) {
	throw "Field 'withinThreshold' must be boolean"
}

if (-not $latency.withinThreshold) {
	throw "Retire latency threshold exceeded"
}

$timelinePath = Join-Path (Join-Path $PSScriptRoot '..\..') 'evidence\retire_timeline.json'
if (-not (Test-Path -LiteralPath $timelinePath)) {
	throw "Missing required artifact: retire_timeline.json"
}

$timeline = Get-Content -LiteralPath $timelinePath -Raw -Encoding UTF8 | ConvertFrom-Json
$timelineSchema = "$($timeline.schema)"
if ($timelineSchema -ne 'retire_timeline_v1' -and $timelineSchema -ne 'retire_timeline_v2') {
	throw "Schema mismatch for retire_timeline.json. expected=retire_timeline_v1|retire_timeline_v2 actual=$timelineSchema"
}

foreach ($requiredField in @('totalTransitions', 'epochMode', 'rollbackMode', 'rollbackReady', 'rollbackFlags')) {
	Assert-HasProperty -Object $timeline -Name $requiredField
}
Assert-NonNegativeInteger -Value $timeline.totalTransitions -FieldName "totalTransitions"

Assert-ValueInSet -Value $timeline.epochMode -Allowed @("shared", "split", "hybrid") -FieldName "epochMode"
Assert-ValueInSet -Value $timeline.rollbackMode -Allowed @("shared", "split", "hybrid") -FieldName "rollbackMode"

if ($timeline.rollbackReady -isnot [bool]) {
	throw "Field 'rollbackReady' must be boolean"
}

Assert-HasProperty -Object $timeline.rollbackFlags -Name "global"
Assert-HasProperty -Object $timeline.rollbackFlags -Name "publicationOnly"
Assert-HasProperty -Object $timeline.rollbackFlags -Name "crossfadeOnly"
Assert-HasProperty -Object $timeline.rollbackFlags -Name "retirePathOnly"

if ($timeline.rollbackFlags.global -isnot [bool]) {
	throw "Field 'rollbackFlags.global' must be boolean"
}
if ($timeline.rollbackFlags.publicationOnly -isnot [bool]) {
	throw "Field 'rollbackFlags.publicationOnly' must be boolean"
}
if ($timeline.rollbackFlags.crossfadeOnly -isnot [bool]) {
	throw "Field 'rollbackFlags.crossfadeOnly' must be boolean"
}
if ($timeline.rollbackFlags.retirePathOnly -isnot [bool]) {
	throw "Field 'rollbackFlags.retirePathOnly' must be boolean"
}

$expectedRollbackReady = ($timeline.rollbackFlags.global -and $timeline.rollbackFlags.retirePathOnly)
if ($timeline.rollbackReady -ne $expectedRollbackReady) {
	throw "Field 'rollbackReady' must equal rollbackFlags.global && rollbackFlags.retirePathOnly"
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
