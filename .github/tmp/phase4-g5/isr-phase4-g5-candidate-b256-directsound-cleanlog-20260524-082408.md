# ISR Phase4 G5 Auto Measurement (candidate-b256-directsound-cleanlog)

- RunId: "20260524-082408"
- GeneratedAt: 2026-05-24 08:24:08
- AppExitCode: 0

## CPU (Process Sampling)
- CPU Usage: avg=2.095302%, p95=5.46055%, min=0.473449%, max=9.428713%, n=92

## Processing Time Raw (CLI_PERF_RAW)
- Process Time: avg=536.805714us, p95=936.6us, min=440.8us, max=1205.5us, n=35

## Rebuild Latency (from log latencyMs=...)
- Rebuild Latency: avg=269.677983ms, p95=300ms, min=1.087ms, max=300ms, n=59

## Telemetry Counters (delta in this run)
- deferred_finalize_ready: 1
- task_queued: 6
- rebuild_forced_dispatch: 0
- deferred_finalize_rebuild_req: 1
- policy_must_execute: 4
- same_as_pending_would_merge: 106
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
- Effective readback: bufferSamples=256, sampleRateHz=44100, readbackCount=35
- appExitOk: True
- bufferMatch: True
- sampleRateMatch: True
- fixedConfigSatisfied: True

## G5 Template Paste Helper
- Avg processing time (raw us): 536.805714
- P95 processing time (raw us): 936.6
- mix=0 segment avg (Baseline / Candidate / Delta / Verdict): not auto-computed (no mix-segment tagging in logs).
- mix=0 segment P95 (Baseline / Candidate / Delta / Verdict): not auto-computed (no mix-segment tagging in logs).
- Click peak (dBFS): not auto-computed (no output waveform capture).
- 20ms window RMS delta (dB): not auto-computed (no output waveform capture).

## Caveats
- CPU comparison is based on process CPU sampling (Get-Process CPU).
- Click peak (dBFS) / 20ms RMS delta (dB) cannot be computed from current logs and are out of scope for auto evaluation.
- mix=0 segment split is not auto-computed because log events do not currently provide explicit segment tags.
