# ISR Phase4 G5 Auto Measurement (baseline-b256-safe-v2)

- RunId: "20260523-232419"
- GeneratedAt: 2026-05-23 23:24:19
- AppExitCode: -1073740940

## CPU (Process Sampling)
- CPU Usage: avg=2.995211%, p95=10.929274%, min=0%, max=12.344884%, n=30

## Processing Time Raw (CLI_PERF_RAW)
- Process Time: avg=133113.2us, p95=133113.2us, min=133113.2us, max=133113.2us, n=1

## Rebuild Latency (from log latencyMs=...)
- Rebuild Latency: avg=1.5795ms, p95=2.178ms, min=1.266ms, max=2.178ms, n=4

## Telemetry Counters (delta in this run)
- deferred_finalize_ready: 0
- task_queued: 4
- rebuild_forced_dispatch: 0
- deferred_finalize_rebuild_req: 0
- policy_must_execute: 0
- same_as_pending_would_merge: 0
- pending_duplicate: 0
- suppressed_mixed_phase_intermediate: 0
- requestRebuild_sr_bs: 4

## Continuity Proxy (log-based)
- fatal_like_entries: 0
- ir_reload_iterations: 0
- runtime_publish_events: 0
- bypass_burst_scheduled: 0

## Fixed Audio Config Validation
- Requested: bufferSamples=256, sampleRateHz=0
- Effective readback: bufferSamples=1024, sampleRateHz=192000, readbackCount=1
- appExitOk: False
- bufferMatch: False
- sampleRateMatch: True
- fixedConfigSatisfied: False

## G5 Template Paste Helper
- Avg processing time (raw us): 133113.2
- P95 processing time (raw us): 133113.2
- mix=0 segment avg (Baseline / Candidate / Delta / Verdict): not auto-computed (no mix-segment tagging in logs).
- mix=0 segment P95 (Baseline / Candidate / Delta / Verdict): not auto-computed (no mix-segment tagging in logs).
- Click peak (dBFS): not auto-computed (no output waveform capture).
- 20ms window RMS delta (dB): not auto-computed (no output waveform capture).

## Caveats
- CPU comparison is based on process CPU sampling (Get-Process CPU).
- Click peak (dBFS) / 20ms RMS delta (dB) cannot be computed from current logs and are out of scope for auto evaluation.
- mix=0 segment split is not auto-computed because log events do not currently provide explicit segment tags.
- Requested fixed audio setup was not satisfied. requested(buffer=256, sampleRateHz=0) effective(buffer=1024, sampleRateHz=192000) appExitCode=-1073740940
