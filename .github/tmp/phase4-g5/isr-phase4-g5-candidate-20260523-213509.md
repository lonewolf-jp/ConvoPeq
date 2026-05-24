# ISR Phase4 G5 Auto Measurement (candidate)

- RunId: "20260523-213509"
- GeneratedAt: 2026-05-23 21:35:09
- AppExitCode: 0

## CPU (Process Sampling)
- CPU Usage: avg=2.897887%, p95=8.06689%, min=0%, max=10.926685%, n=97

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
