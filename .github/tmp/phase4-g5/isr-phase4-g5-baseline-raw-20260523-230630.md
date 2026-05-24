# ISR Phase4 G5 Auto Measurement (baseline-raw)

- RunId: "20260523-230630"
- GeneratedAt: 2026-05-23 23:06:30
- AppExitCode: 0

## CPU (Process Sampling)
- CPU Usage: avg=2.697151%, p95=8.555181%, min=0.471544%, max=10.92861%, n=96

## Processing Time Raw (CLI_PERF_RAW)
- Process Time: avg=922.831429us, p95=2068us, min=693.5us, max=2203.4us, n=35

## Rebuild Latency (from log latencyMs=...)
- Rebuild Latency: avg=275.368607ms, p95=300ms, min=2.006ms, max=2068.579ms, n=61

## Telemetry Counters (delta in this run)
- deferred_finalize_ready: 0
- task_queued: 11
- rebuild_forced_dispatch: 1
- deferred_finalize_rebuild_req: 2
- policy_must_execute: 5
- same_as_pending_would_merge: 107
- pending_duplicate: 0
- suppressed_mixed_phase_intermediate: 0
- requestRebuild_sr_bs: 11

## Continuity Proxy (log-based)
- fatal_like_entries: 0
- ir_reload_iterations: 0
- runtime_publish_events: 0
- bypass_burst_scheduled: 0

## G5 Template Paste Helper
- Avg processing time (raw us): 922.831429
- P95 processing time (raw us): 2068
- mix=0 segment avg (Baseline / Candidate / Delta / Verdict): not auto-computed (no mix-segment tagging in logs).
- mix=0 segment P95 (Baseline / Candidate / Delta / Verdict): not auto-computed (no mix-segment tagging in logs).
- Click peak (dBFS): not auto-computed (no output waveform capture).
- 20ms window RMS delta (dB): not auto-computed (no output waveform capture).

## Caveats
- CPU comparison is based on process CPU sampling (Get-Process CPU).
- Click peak (dBFS) / 20ms RMS delta (dB) cannot be computed from current logs and are out of scope for auto evaluation.
- mix=0 segment split is not auto-computed because log events do not currently provide explicit segment tags.
