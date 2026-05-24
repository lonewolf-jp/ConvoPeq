# ISR Phase4 G5 Auto Measurement (baseline-b256-directsound-cleanlog)

- RunId: "20260524-082345"
- GeneratedAt: 2026-05-24 08:23:45
- AppExitCode: 0

## CPU (Process Sampling)
- CPU Usage: avg=2.020308%, p95=6.164688%, min=0%, max=8.844595%, n=93

## Processing Time Raw (CLI_PERF_RAW)
- Process Time: avg=506.114286us, p95=738.3us, min=442.3us, max=917.8us, n=35

## Rebuild Latency (from log latencyMs=...)
- Rebuild Latency: avg=265.277433ms, p95=300ms, min=1.187ms, max=300ms, n=60

## Telemetry Counters (delta in this run)
- deferred_finalize_ready: 1
- task_queued: 7
- rebuild_forced_dispatch: 0
- deferred_finalize_rebuild_req: 1
- policy_must_execute: 4
- same_as_pending_would_merge: 106
- pending_duplicate: 0
- suppressed_mixed_phase_intermediate: 0
- requestRebuild_sr_bs: 7

## Continuity Proxy (log-based)
- fatal_like_entries: 0
- ir_reload_iterations: 0
- runtime_publish_events: 0
- bypass_burst_scheduled: 0

## Fixed Audio Config Validation
- Requested: deviceType=DirectSound, bufferSamples=256, sampleRateHz=44100
- Effective readback: bufferSamples=256, sampleRateHz=44100, readbackCount=35
- appExitOk: True
- bufferMatch: True
- sampleRateMatch: True
- fixedConfigSatisfied: True

## G5 Template Paste Helper
- Avg processing time (raw us): 506.114286
- P95 processing time (raw us): 738.3
- mix=0 segment avg (Baseline / Candidate / Delta / Verdict): not auto-computed (no mix-segment tagging in logs).
- mix=0 segment P95 (Baseline / Candidate / Delta / Verdict): not auto-computed (no mix-segment tagging in logs).
- Click peak (dBFS): not auto-computed (no output waveform capture).
- 20ms window RMS delta (dB): not auto-computed (no output waveform capture).

## Caveats
- CPU comparison is based on process CPU sampling (Get-Process CPU).
- Click peak (dBFS) / 20ms RMS delta (dB) cannot be computed from current logs and are out of scope for auto evaluation.
- mix=0 segment split is not auto-computed because log events do not currently provide explicit segment tags.
