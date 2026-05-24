# ISR Phase4 G5 Auto Measurement (candidate-b256-directsound-soak9)

- RunId: "20260524-085125"
- GeneratedAt: 2026-05-24 08:51:25
- AppExitCode: 0

## CPU (Process Sampling)
- CPU Usage: avg=2.215354%, p95=5.924737%, min=0%, max=11.845167%, n=92

## Processing Time Raw (CLI_PERF_RAW)
- Process Time: avg=519.066667us, p95=619.2us, min=445.1us, max=1066.6us, n=36

## Rebuild Latency (from log latencyMs=...)
- Rebuild Latency: avg=260.27195ms, p95=300ms, min=1.109ms, max=300ms, n=60

## Telemetry Counters (delta in this run)
- deferred_finalize_ready: 2
- task_queued: 8
- rebuild_forced_dispatch: 0
- deferred_finalize_rebuild_req: 2
- policy_must_execute: 8
- same_as_pending_would_merge: 104
- pending_duplicate: 0
- suppressed_mixed_phase_intermediate: 0
- requestRebuild_sr_bs: 8

## Continuity Proxy (log-based)
- fatal_like_entries: 0
- ir_reload_iterations: 0
- runtime_publish_events: 0
- bypass_burst_scheduled: 0

## Fixed Audio Config Validation
- Requested: deviceType=DirectSound, bufferSamples=256, sampleRateHz=44100
- Effective readback: bufferSamples=256, sampleRateHz=44100, readbackCount=36
- appExitOk: True
- bufferMatch: True
- sampleRateMatch: True
- fixedConfigSatisfied: True

## G5 Template Paste Helper
- Avg processing time (raw us): 519.066667
- P95 processing time (raw us): 619.2
- mix=0 segment avg (Baseline / Candidate / Delta / Verdict): not auto-computed (no mix-segment tagging in logs).
- mix=0 segment P95 (Baseline / Candidate / Delta / Verdict): not auto-computed (no mix-segment tagging in logs).
- Click peak (dBFS): not auto-computed (no output waveform capture).
- 20ms window RMS delta (dB): not auto-computed (no output waveform capture).

## Caveats
- CPU comparison is based on process CPU sampling (Get-Process CPU).
- Click peak (dBFS) / 20ms RMS delta (dB) cannot be computed from current logs and are out of scope for auto evaluation.
- mix=0 segment split is not auto-computed because log events do not currently provide explicit segment tags.
