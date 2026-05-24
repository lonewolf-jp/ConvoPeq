# ISR Phase4 G5 Auto Measurement (test-b256-ds-noprobe)

- RunId: "20260524-015442"
- GeneratedAt: 2026-05-24 01:54:42
- AppExitCode: -1073740940

## CPU (Process Sampling)
- CPU Usage: avg=2.116272%, p95=7.62016%, min=0%, max=9.488882%, n=33

## Processing Time Raw (CLI_PERF_RAW)
- Process Time: avg=295.6us, p95=590.3us, min=0.9us, max=590.3us, n=2

## Rebuild Latency (from log latencyMs=...)
- Rebuild Latency: avg=1.970667ms, p95=2.468ms, min=1.422ms, max=2.468ms, n=6

## Telemetry Counters (delta in this run)
- deferred_finalize_ready: 0
- task_queued: 6
- rebuild_forced_dispatch: 0
- deferred_finalize_rebuild_req: 0
- policy_must_execute: 0
- same_as_pending_would_merge: 0
- pending_duplicate: 0
- suppressed_mixed_phase_intermediate: 0
- requestRebuild_sr_bs: 6

## Continuity Proxy (log-based)
- fatal_like_entries: 0
- ir_reload_iterations: 0
- runtime_publish_events: 0
- bypass_burst_scheduled: 0

## Fixed Audio Config Validation
- Requested: deviceType=DirectSound, bufferSamples=256, sampleRateHz=44100
- Effective readback: bufferSamples=256, sampleRateHz=44100, readbackCount=2
- appExitOk: False
- bufferMatch: True
- sampleRateMatch: True
- fixedConfigSatisfied: False

## G5 Template Paste Helper
- Avg processing time (raw us): 295.6
- P95 processing time (raw us): 590.3
- mix=0 segment avg (Baseline / Candidate / Delta / Verdict): not auto-computed (no mix-segment tagging in logs).
- mix=0 segment P95 (Baseline / Candidate / Delta / Verdict): not auto-computed (no mix-segment tagging in logs).
- Click peak (dBFS): not auto-computed (no output waveform capture).
- 20ms window RMS delta (dB): not auto-computed (no output waveform capture).

## Caveats
- CPU comparison is based on process CPU sampling (Get-Process CPU).
- Click peak (dBFS) / 20ms RMS delta (dB) cannot be computed from current logs and are out of scope for auto evaluation.
- mix=0 segment split is not auto-computed because log events do not currently provide explicit segment tags.
- Fixed-config validation failed. requested(buffer=256, sampleRateHz=44100) effective(buffer=256, sampleRateHz=44100) appExitCode=-1073740940 appExitOk=False bufferMatch=True sampleRateMatch=True
