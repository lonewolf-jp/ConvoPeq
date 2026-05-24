# ISR Phase4 G5 Auto Measurement (baseline)

- RunId: "20260523-213445"
- GeneratedAt: 2026-05-23 21:34:45
- AppExitCode: 0

## CPU (Process Sampling)
- CPU Usage: avg=2.884742%, p95=7.612145%, min=0%, max=10.415939%, n=97

## Rebuild Latency (from log latencyMs=...)
- Rebuild Latency: avg=230.025627ms, p95=300ms, min=2.048ms, max=300ms, n=51

## Telemetry Counters (delta in this run)
- deferred_finalize_ready: 1
- task_queued: 12
- rebuild_forced_dispatch: 0
- deferred_finalize_rebuild_req: 1
- policy_must_execute: 4
- same_as_pending_would_merge: 88
- pending_duplicate: 0
- suppressed_mixed_phase_intermediate: 0
- requestRebuild_sr_bs: 12

## Continuity Proxy (log-based)
- fatal_like_entries: 0
- ir_reload_iterations: 0
- runtime_publish_events: 0
- bypass_burst_scheduled: 0

## G5 Template Paste Helper
- Avg processing time (Baseline / Candidate / Delta / Verdict): not auto-computed (no raw processing-time metric in logs).
- P95 processing time (Baseline / Candidate / Delta / Verdict): not auto-computed (no raw processing-time metric in logs).
- mix=0 segment avg (Baseline / Candidate / Delta / Verdict): not auto-computed (no mix-segment tagging in logs).
- mix=0 segment P95 (Baseline / Candidate / Delta / Verdict): not auto-computed (no mix-segment tagging in logs).
- Click peak (dBFS): not auto-computed (no output waveform capture).
- 20ms window RMS delta (dB): not auto-computed (no output waveform capture).

## Caveats
- CPU comparison is based on process CPU sampling (Get-Process CPU).
- Click peak (dBFS) / 20ms RMS delta (dB) cannot be computed from current logs and are out of scope for auto evaluation.
