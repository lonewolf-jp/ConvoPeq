# ISR Phase4 G5 Auto Measurement (candidate-raw)

- RunId: "20260523-230656"
- GeneratedAt: 2026-05-23 23:06:56
- AppExitCode: 0

## CPU (Process Sampling)
- CPU Usage: avg=2.845391%, p95=8.533211%, min=0%, max=10.007375%, n=97

## Processing Time Raw (CLI_PERF_RAW)
- Process Time: avg=876.016667us, p95=1833.9us, min=675.5us, max=2703.6us, n=36

## Rebuild Latency (from log latencyMs=...)
- Rebuild Latency: avg=265.883101ms, p95=300ms, min=1.673ms, max=2068.065ms, n=79

## Telemetry Counters (delta in this run)
- deferred_finalize_ready: 0
- task_queued: 15
- rebuild_forced_dispatch: 1
- deferred_finalize_rebuild_req: 2
- policy_must_execute: 5
- same_as_pending_would_merge: 135
- pending_duplicate: 0
- suppressed_mixed_phase_intermediate: 0
- requestRebuild_sr_bs: 15

## Continuity Proxy (log-based)
- fatal_like_entries: 0
- ir_reload_iterations: 0
- runtime_publish_events: 0
- bypass_burst_scheduled: 0

## G5 Template Paste Helper
- Avg processing time (raw us): 876.016667
- P95 processing time (raw us): 1833.9
- mix=0 segment avg (Baseline / Candidate / Delta / Verdict): not auto-computed (no mix-segment tagging in logs).
- mix=0 segment P95 (Baseline / Candidate / Delta / Verdict): not auto-computed (no mix-segment tagging in logs).
- Click peak (dBFS): not auto-computed (no output waveform capture).
- 20ms window RMS delta (dB): not auto-computed (no output waveform capture).

## Caveats
- CPU comparison is based on process CPU sampling (Get-Process CPU).
- Click peak (dBFS) / 20ms RMS delta (dB) cannot be computed from current logs and are out of scope for auto evaluation.
- mix=0 segment split is not auto-computed because log events do not currently provide explicit segment tags.
