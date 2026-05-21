$ErrorActionPreference = 'Stop'
$data = . "$PSScriptRoot\isr-verify-common.ps1" -ArtifactName "shutdown_trace.json" -Schema "shutdown_trace_v1" -RequiredKeys @("phase", "verified", "sh1_callbackCount", "sh2_activeCrossfade", "sh3_pendingRetire", "sh4_observerCount", "sh5_lateCallbackCount", "sh6_postStopEnqueueCount")

Assert-NonNegativeInteger -Value $data.phase -FieldName "phase"
Assert-NonNegativeInteger -Value $data.sh1_callbackCount -FieldName "sh1_callbackCount"
Assert-NonNegativeInteger -Value $data.sh2_activeCrossfade -FieldName "sh2_activeCrossfade"
Assert-NonNegativeInteger -Value $data.sh3_pendingRetire -FieldName "sh3_pendingRetire"
Assert-NonNegativeInteger -Value $data.sh4_observerCount -FieldName "sh4_observerCount"
Assert-NonNegativeInteger -Value $data.sh5_lateCallbackCount -FieldName "sh5_lateCallbackCount"
Assert-NonNegativeInteger -Value $data.sh6_postStopEnqueueCount -FieldName "sh6_postStopEnqueueCount"

if ([long]$data.sh5_lateCallbackCount -gt 0) {
	throw "Late callbacks detected after audio stop. count=$($data.sh5_lateCallbackCount)"
}

if ([long]$data.sh6_postStopEnqueueCount -gt 0) {
	throw "Post-stop retire enqueue detected. count=$($data.sh6_postStopEnqueueCount)"
}

if ($data.verified -isnot [bool]) {
	throw "Field 'verified' must be boolean"
}

if (-not $data.verified) {
	throw "Shutdown FSM verification failed"
}
