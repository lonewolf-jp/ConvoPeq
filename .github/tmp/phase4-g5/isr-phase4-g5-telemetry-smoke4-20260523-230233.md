# ISR Phase4 G5 Auto Measurement (telemetry-smoke4)

- RunId: "20260523-230233"
- GeneratedAt: 2026-05-23 23:02:33
- AppExitCode: 0

## CPU (Process Sampling)
- CPU Usage: avg=3.007456%, p95=7.133268%, min=0%, max=9.478452%, n=33

## Processing Time Raw (CLI_PERF_RAW)
- Process Time: avg=1030.8us, p95=1406.4us, min=709.1us, max=1406.4us, n=9

## Rebuild Latency (from log latencyMs=...)
- Rebuild Latency: avg=1.9995ms, p95=3.2ms, min=1.592ms, max=3.2ms, n=4

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

## G5 Template Paste Helper
- Avg processing time (raw us): 1030.8
- P95 processing time (raw us): 1406.4
- mix=0 segment avg (Baseline / Candidate / Delta / Verdict): not auto-computed (no mix-segment tagging in logs).
- mix=0 segment P95 (Baseline / Candidate / Delta / Verdict): not auto-computed (no mix-segment tagging in logs).
- Click peak (dBFS): not auto-computed (no output waveform capture).
- 20ms window RMS delta (dB): not auto-computed (no output waveform capture).

## Caveats
- CPU comparison is based on process CPU sampling (Get-Process CPU).
- Click peak (dBFS) / 20ms RMS delta (dB) cannot be computed from current logs and are out of scope for auto evaluation.
- mix=0 segment split is not auto-computed because log events do not currently provide explicit segment tags.
