# ISR Phase4 G5 Auto Measurement (baseline-b256-capprobe)

- RunId: "20260524-001259"
- GeneratedAt: 2026-05-24 00:12:59
- AppExitCode: -1073740940

## CPU (Process Sampling)
- CPU Usage: avg=3.323493%, p95=9.503504%, min=0%, max=9.996467%, n=30

## Processing Time Raw (CLI_PERF_RAW)
- Process Time: avg=2324.6us, p95=2348.1us, min=2301.1us, max=2348.1us, n=2

## Rebuild Latency (from log latencyMs=...)
- Rebuild Latency: avg=269.708898ms, p95=300ms, min=1.386ms, max=300ms, n=59

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
- Requested: bufferSamples=256, sampleRateHz=192000
- Effective readback: bufferSamples=1024, sampleRateHz=192000, readbackCount=2
- appExitOk: False
- bufferMatch: False
- sampleRateMatch: True
- fixedConfigSatisfied: False

## G5 Template Paste Helper
- Avg processing time (raw us): 2324.6
- P95 processing time (raw us): 2348.1
- mix=0 segment avg (Baseline / Candidate / Delta / Verdict): not auto-computed (no mix-segment tagging in logs).
- mix=0 segment P95 (Baseline / Candidate / Delta / Verdict): not auto-computed (no mix-segment tagging in logs).
- Click peak (dBFS): not auto-computed (no output waveform capture).
- 20ms window RMS delta (dB): not auto-computed (no output waveform capture).

## Caveats
- CPU comparison is based on process CPU sampling (Get-Process CPU).
- Click peak (dBFS) / 20ms RMS delta (dB) cannot be computed from current logs and are out of scope for auto evaluation.
- mix=0 segment split is not auto-computed because log events do not currently provide explicit segment tags.
- Requested fixed audio setup was not satisfied. requested(buffer=256, sampleRateHz=192000) effective(buffer=1024, sampleRateHz=192000) appExitCode=-1073740940
