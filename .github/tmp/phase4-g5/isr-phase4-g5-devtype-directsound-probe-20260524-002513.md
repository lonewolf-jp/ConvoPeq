# ISR Phase4 G5 Auto Measurement (devtype-directsound-probe)

- RunId: "20260524-002513"
- GeneratedAt: 2026-05-24 00:25:13
- AppExitCode: -1073740940

## CPU (Process Sampling)
- CPU Usage: avg=2.195572%, p95=8.115794%, min=0%, max=10.922652%, n=34

## Processing Time Raw (CLI_PERF_RAW)
- Process Time: avg=473.9us, p95=473.9us, min=473.9us, max=473.9us, n=1

## Rebuild Latency (from log latencyMs=...)
- Rebuild Latency: avg=0ms, p95=0ms, min=0ms, max=0ms, n=0

## Telemetry Counters (delta in this run)
- deferred_finalize_ready: 0
- task_queued: 0
- rebuild_forced_dispatch: 0
- deferred_finalize_rebuild_req: 0
- policy_must_execute: 0
- same_as_pending_would_merge: 0
- pending_duplicate: 0
- suppressed_mixed_phase_intermediate: 0
- requestRebuild_sr_bs: 0

## Continuity Proxy (log-based)
- fatal_like_entries: 0
- ir_reload_iterations: 0
- runtime_publish_events: 0
- bypass_burst_scheduled: 0

## Fixed Audio Config Validation
- Requested: deviceType=DirectSound, bufferSamples=256, sampleRateHz=192000
- Effective readback: bufferSamples=256, sampleRateHz=44100, readbackCount=1
- appExitOk: False
- bufferMatch: True
- sampleRateMatch: False
- fixedConfigSatisfied: False

## G5 Template Paste Helper
- Avg processing time (raw us): 473.9
- P95 processing time (raw us): 473.9
- mix=0 segment avg (Baseline / Candidate / Delta / Verdict): not auto-computed (no mix-segment tagging in logs).
- mix=0 segment P95 (Baseline / Candidate / Delta / Verdict): not auto-computed (no mix-segment tagging in logs).
- Click peak (dBFS): not auto-computed (no output waveform capture).
- 20ms window RMS delta (dB): not auto-computed (no output waveform capture).

## Caveats
- CPU comparison is based on process CPU sampling (Get-Process CPU).
- Click peak (dBFS) / 20ms RMS delta (dB) cannot be computed from current logs and are out of scope for auto evaluation.
- mix=0 segment split is not auto-computed because log events do not currently provide explicit segment tags.
- Requested fixed audio setup was not satisfied. requested(buffer=256, sampleRateHz=192000) effective(buffer=256, sampleRateHz=44100) appExitCode=-1073740940
