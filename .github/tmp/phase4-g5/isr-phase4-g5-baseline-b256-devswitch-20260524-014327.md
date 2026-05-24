# ISR Phase4 G5 Auto Measurement (baseline-b256-devswitch)

- RunId: "20260524-014327"
- GeneratedAt: 2026-05-24 01:43:27
- AppExitCode: -1073740940

## CPU (Process Sampling)
- CPU Usage: avg=3.165633%, p95=8.741437%, min=0%, max=9.062552%, n=32

## Processing Time Raw (CLI_PERF_RAW)
- Process Time: avg=1492.75us, p95=2119.1us, min=866.4us, max=2119.1us, n=2

## Rebuild Latency (from log latencyMs=...)
- Rebuild Latency: avg=269.748949ms, p95=300ms, min=1.446ms, max=300ms, n=59

## Telemetry Counters (delta in this run)
- deferred_finalize_ready: 0
- task_queued: 6
- rebuild_forced_dispatch: 0
- deferred_finalize_rebuild_req: 0
- policy_must_execute: 0
- same_as_pending_would_merge: 107
- pending_duplicate: 0
- suppressed_mixed_phase_intermediate: 0
- requestRebuild_sr_bs: 6

## Continuity Proxy (log-based)
- fatal_like_entries: 0
- ir_reload_iterations: 0
- runtime_publish_events: 0
- bypass_burst_scheduled: 0

## Fixed Audio Config Validation
- Requested: deviceType=Windows Audio, bufferSamples=256, sampleRateHz=192000
- Effective readback: bufferSamples=1024, sampleRateHz=192000, readbackCount=2
- appExitOk: False
- bufferMatch: False
- sampleRateMatch: True
- fixedConfigSatisfied: False

## G5 Template Paste Helper
- Avg processing time (raw us): 1492.75
- P95 processing time (raw us): 2119.1
- mix=0 segment avg (Baseline / Candidate / Delta / Verdict): not auto-computed (no mix-segment tagging in logs).
- mix=0 segment P95 (Baseline / Candidate / Delta / Verdict): not auto-computed (no mix-segment tagging in logs).
- Click peak (dBFS): not auto-computed (no output waveform capture).
- 20ms window RMS delta (dB): not auto-computed (no output waveform capture).

## Caveats
- CPU comparison is based on process CPU sampling (Get-Process CPU).
- Click peak (dBFS) / 20ms RMS delta (dB) cannot be computed from current logs and are out of scope for auto evaluation.
- mix=0 segment split is not auto-computed because log events do not currently provide explicit segment tags.
- Fixed-config validation failed. requested(buffer=256, sampleRateHz=192000) effective(buffer=1024, sampleRateHz=192000) appExitCode=-1073740940 appExitOk=False bufferMatch=False sampleRateMatch=True
