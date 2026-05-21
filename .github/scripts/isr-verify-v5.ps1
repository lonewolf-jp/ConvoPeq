$ErrorActionPreference = 'Stop'
$trace = . "$PSScriptRoot\isr-verify-common.ps1" -ArtifactName "hb_graph_trace.json" -Schema "hb_trace_v1" -RequiredKeys @("eventCount", "events")
Assert-NonNegativeInteger -Value $trace.eventCount -FieldName "eventCount"
if ([long]$trace.eventCount -le 0) {
    throw "HB trace must contain at least one event. eventCount=$($trace.eventCount)"
}

Assert-IsArray -Value $trace.events -FieldName "events"
$events = @($trace.events)
if ($events.Count -ne [int64]$trace.eventCount) {
    throw "HB trace event count mismatch. eventCount=$($trace.eventCount) actualEvents=$($events.Count)"
}

$prevTs = -1L
foreach ($ev in $events) {
    Assert-HasProperty -Object $ev -Name "ts"
    Assert-HasProperty -Object $ev -Name "from"
    Assert-HasProperty -Object $ev -Name "to"
    Assert-HasProperty -Object $ev -Name "fromEpoch"
    Assert-HasProperty -Object $ev -Name "toEpoch"
    Assert-HasProperty -Object $ev -Name "mo"
    Assert-HasProperty -Object $ev -Name "release"
    Assert-HasProperty -Object $ev -Name "acquire"

    Assert-NonNegativeInteger -Value $ev.ts -FieldName "events.ts"
    Assert-NonNegativeInteger -Value $ev.from -FieldName "events.from"
    Assert-NonNegativeInteger -Value $ev.to -FieldName "events.to"
    Assert-NonNegativeInteger -Value $ev.fromEpoch -FieldName "events.fromEpoch"
    Assert-NonNegativeInteger -Value $ev.toEpoch -FieldName "events.toEpoch"
    Assert-NonNegativeInteger -Value $ev.mo -FieldName "events.mo"

    if ($ev.release -isnot [bool]) {
        throw "Field 'events.release' must be boolean"
    }
    if ($ev.acquire -isnot [bool]) {
        throw "Field 'events.acquire' must be boolean"
    }

    if ($prevTs -gt [long]$ev.ts) {
        throw "HB trace timestamp must be monotonic non-decreasing"
    }
    $prevTs = [long]$ev.ts
}

$data = . "$PSScriptRoot\isr-verify-common.ps1" -ArtifactName "hb_violation_report.json" -Schema "hb_violation_report_v1" -RequiredKeys @("status", "violations", "scenarioResults")

Assert-ValueInSet -Value $data.status -Allowed @("ok", "violation") -FieldName "status"
Assert-IsArray -Value $data.violations -FieldName "violations"

if ($data.status -eq "violation") {
    throw "HB violation report indicates violation"
}

Assert-IsArray -Value $data.scenarioResults -FieldName "scenarioResults"
$scenarioCount = @($data.scenarioResults).Count
if ($scenarioCount -ne 4) {
    throw "scenarioResults count must be 4. actual=$scenarioCount"
}

$expectedNames = @("forced_reorder", "epoch_lag", "retire_delay", "observe_race")
$actualNames = @($data.scenarioResults | ForEach-Object { $_.name })

foreach ($name in $expectedNames) {
    if ($actualNames -notcontains $name) {
        throw "HB scenario missing: $name"
    }
}

foreach ($scenario in $data.scenarioResults) {
    Assert-HasProperty -Object $scenario -Name "name"
    Assert-HasProperty -Object $scenario -Name "result"
    Assert-ValueInSet -Value $scenario.result -Allowed @("pass", "fail") -FieldName "scenarioResults.result"
    if ($scenario.result -eq "fail") {
        throw "HB reorder simulation failed for scenario '$($scenario.name)'"
    }
}
