# ISR Phase4 G5 Auto Measurement (baseline-b256-directsound)

- RunId: "20260524-015229"
- GeneratedAt: 2026-05-24 01:52:29
- AppExitCode: -1073740940

## CPU (Process Sampling)
- CPU Usage: avg=2.2344%, p95=8.076536%, min=0%, max=8.560506%, n=35

## Processing Time Raw (CLI_PERF_RAW)
- Process Time: avg=275.4us, p95=549.6us, min=1.2us, max=549.6us, n=2

## Rebuild Latency (from log latencyMs=...)
- Rebuild Latency: avg=252.917632ms, p95=300ms, min=1.287ms, max=300ms, n=38

## Telemetry Counters (delta in this run)
- deferred_finalize_ready: 0
- task_queued: 6
- rebuild_forced_dispatch: 0
- deferred_finalize_rebuild_req: 0
- policy_must_execute: 0
- same_as_pending_would_merge: 64
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
- Avg processing time (raw us): 275.4
- P95 processing time (raw us): 549.6
- mix=0 segment avg (Baseline / Candidate / Delta / Verdict): not auto-computed (no mix-segment tagging in logs).
- mix=0 segment P95 (Baseline / Candidate / Delta / Verdict): not auto-computed (no mix-segment tagging in logs).
- Click peak (dBFS): not auto-computed (no output waveform capture).
- 20ms window RMS delta (dB): not auto-computed (no output waveform capture).

## Caveats
- CPU comparison is based on process CPU sampling (Get-Process CPU).
- Click peak (dBFS) / 20ms RMS delta (dB) cannot be computed from current logs and are out of scope for auto evaluation.
- mix=0 segment split is not auto-computed because log events do not currently provide explicit segment tags.
- Fixed-config validation failed. requested(buffer=256, sampleRateHz=44100) effective(buffer=256, sampleRateHz=44100) appExitCode=-1073740940 appExitOk=False bufferMatch=True sampleRateMatch=True
