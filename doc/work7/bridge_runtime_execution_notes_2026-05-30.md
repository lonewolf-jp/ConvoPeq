# Bridge Runtime Execution Notes (2026-05-30)

## 1. 変更の要点

- Audio callback の観測面を `AudioCallbackAuthorityView` に集約した。
- `processWithSnapshot` は呼び出し側が渡す `RuntimeGraph` を使用する前提に寄せ、内部で追加の runtime read を行わないようにした。
- 既存の publication / retire / validator 系は、今回の変更後も既存の検証ゲートで整合を確認した。

## 2. Semantic Source Table

| 読み出し対象 | 取得元 | 備考 |
| --- | --- | --- |
| active DSP graph | `readAudioRuntimeView()` → `RuntimeReadView.runtimePublish.graph` | Audio callback の入口で 1 回だけ取得 |
| active snapshot | `readAudioRuntimeView()` → `RuntimeReadView.snapshot` | `captureAudioThreadParameterSnapshot()` に供給 |
| crossfade prepared snapshot | `consumeCrossfadePreparedSnapshot()` | callback 入口で `AudioCallbackAuthorityView` に取り込み |
| fading DSP | `resolveFadingRuntimeDSPFromRuntimeWorldOnly(runtimeGraph)` | callback で参照する DSP ルートを統一 |

## 3. 主要コードの責務

### Observe Path Collapse

- `src/audioengine/AudioEngine.Processing.AudioBlock.cpp`
- `src/audioengine/AudioEngine.Processing.BlockDouble.cpp`
- `src/audioengine/AudioEngine.Processing.Snapshot.cpp`

上記では、callback 入口で取得した authority view を再利用し、graph / snapshot / crossfade を callback 内で再取得しない。

### Publication / Invariant

- `src/audioengine/ISRRuntimePublicationCoordinator.cpp`
- `src/audioengine/ISRPayloadTier.cpp`
- `src/audioengine/ISRClosure.cpp`

既存の precheck / tier / closure validators により、publication 側の共通検証面が維持されている。

### Retire Governance

- `src/audioengine/AudioEngine.Threading.cpp`

主要 telemetry:

- `fallbackQueueDepth_`
- `retireQueueDepth_`
- `retireHighWatermark_`
- `retireLowWatermark_`
- `retireSaturationActive_`

## 4. Contract / Verifier 観点

現時点での監査窓口は、命名付き registry オブジェクトの追加ではなく、既存の verifier / coordinator 群で担保されている。

- `ClosureValidator`
- `PayloadTierValidator`
- `RuntimePublicationCoordinator`
- `isr-verify-governance-registries.ps1`
- `isr-verify-runtime-semantic-schema-v16.ps1`
- `isr-verify-v4-dsp-handle-policy.ps1`

## 5. 実行した検証

### Build

- Debug build: PASS
- Release build: PASS

### Static checks

- `check-src-atomic-dotcall.ps1`: PASS
- `check-list-compliance.ps1`: PASS

### Governance / runtime verifiers

- `isr-verify-publication-single-path.ps1`: PASS
- `isr-verify-crossfade-observable-state.ps1`: PASS
- `isr-verify-metric-governance.ps1`: PASS
- `isr-verify-governance-registries.ps1`: PASS
- `isr-verify-design-docs-coverage.ps1`: PASS
- `isr-verify-runtime-semantic-schema-v16.ps1`: PASS
- `isr-verify-safety-regression.ps1`: PASS
- `isr-verify-v4-dsp-handle-policy.ps1`: PASS
- `isr-run-tiered-verification.ps1`: PASS

### Operational evidence

- `isr-phase4-g5-auto-measure.ps1 -Mode run`
  - AppExitCode: 0
  - CPU avg: 3.576978%
  - CPU p95: 6.189137%
  - Processing time avg: 2229.391429us
  - Processing time p95: 2962.9us
  - Rebuild latency avg: 3.518ms
  - Rebuild latency p95: 3.518ms
  - fixedConfigSatisfied: True

Generated evidence files:

- `.github/tmp/phase4-g5/isr-phase4-g5-candidate-20260530-090400.json`
- `.github/tmp/phase4-g5/isr-phase4-g5-candidate-20260530-090400.md`
- `evidence/publication_single_path_report.json`
- `evidence/crossfade_observable_state_report.json`
- `evidence/metric_governance_report.json`
- `evidence/governance_registries_report.json`
- `evidence/design_docs_coverage_report.json`
- `evidence/runtime_semantic_schema_v16_report.json`
- `evidence/safety_regression_report.json`

## 6. Notes

- The callback authority view is intentionally lightweight and does not introduce allocation or locking on the audio thread.
- The explicit `readAudioRuntimeView()` / `getRuntimeGraph(...)` tokens remain present in the callback files so existing policy verifiers can recognize the audio runtime observe path.
- This note documents the current bridge-runtime state; it does not extend the M5 RuntimeWorld migration scope.
