# ISR Bridge Runtime 設計凍結仕様（v5.5 FINAL FREEZE）

- Project: ConvoPeq
- Date: 2026-05-28
- Status: **FINAL FREEZE**（この版を基本設計として固定）
- Scope: DAW 実運用で破綻しにくい ISR Bridge Runtime hardening

---

## 0. 目的（固定）

本仕様は、理論的完全性よりも実運用耐性を優先し、以下の failure mode を段階的に封じ込めることを目的とする。

- rebuild storm
- retire saturation
- shutdown race
- stale runtime reuse
- crossfade authority drift
- publication race

優先軸（固定）:

1. deterministic shutdown
2. rebuild causality isolation
3. retire backpressure
4. RT visibility separation
5. execution-local mutability

---

## 1. 非目的（固定）

本凍結版では以下を目的化しない。

- ISR purity の理論追求
- 完全 immutable runtime graph への全面移行
- 全 lock-free 化
- runtime graph / publication system の全面再設計
- DSP topology の全面再構築

---

## 2. 基本原則（固定）

### 2.1 ownership mutation と RT visibility の分離

Runtime 所有権変化と RT 可視状態を分離し、可観測境界を明確化する。

### 2.2 execution-local mutable state の許可

以下は ISR 違反とみなさない。

- local fade progression
- thread-local scratch
- smoothing accumulators
- readonly cache materialization

### 2.3 shared mutable runtime authority の禁止

以下を禁止する。

- runtime pointer 由来の rebuild decision
- cross-runtime mutable progression
- publish 後 topology mutation

### 2.4 admission state machine 単一路線

`lifecycleState / shutdownPhase / shutdownRuntime_` に統合し、独立 state machine 追加を禁止する。

---

## 3. Unified Admission Gate（固定）

### 3.1 API

```cpp
bool acceptsRuntimePublication() const noexcept;
```

### 3.2 判定表（固定）

| lifecycle / shutdown state | acceptsRuntimePublication() |
| --- | --- |
| Running | true |
| StopAcceptingWork | false |
| AudioStopped | false |
| ReleasingResources | false |
| Destroyed | false |

### 3.3 mandatory gate points（固定）

以下すべてで gate を適用する。

- `requestRebuild`
- `enqueuePublication`（実装対応: publication intent append / prepareCommit 経路）
- `prepareCommit`
- `executeCommit`
- `appendPublicationIntent`

### 3.4 mandatory behavior

```cpp
if (!acceptsRuntimePublication())
{
    return;
}
```

---

## 4. RuntimeBuildSnapshot 契約（固定）

### 4.1 構造

```cpp
struct RuntimeBuildFingerprint
{
    uint32_t fingerprintVersion = 1;
    uint64_t irIdentityHash = 0;
    uint64_t convolutionConfigHash = 0;
    uint64_t dspParameterHash = 0;
    double sampleRate = 0.0;
    int blockSize = 0;
};

struct RuntimeBuildSnapshot
{
    RuntimeGeneration generation;
    double sampleRate;
    int blockSize;
    int oversamplingFactor;
    ConvolverBuildSnapshot convolver;
    ParameterBlock parameters;
    RuntimeFlags flags;
    RuntimeBuildFingerprint rebuildFingerprint;
};
```

### 4.2 構築フェーズ

`capture -> finalize -> seal -> worker handoff` を固定する。

### 4.3 finalize の定義（deterministic）

同一 semantic input から同一 finalized snapshot を生成する純粋正規化段階。

semantic input:

- normalized DSP parameters
- IR identity
- oversampling configuration
- topology class
- runtime policy version
- rebuildFingerprintVersion

### 4.4 finalize 許可 / 禁止

許可:

- canonical ordering
- parameter normalization
- immutable fingerprint generation
- bounded validation

禁止:

- runtime state injection
- wall clock / timing / thread-order 依存
- allocation order 依存
- pointer identity 依存
- environment-derived mutation
- mutable patch injection
- async augmentation

### 4.5 worker 側契約

許可:

- readonly access
- derived local cache creation
- local scratch allocation

禁止:

- snapshot mutation
- runtime pointer derived override

### 4.6 versioning 規約

- fingerprint 構成要素変更時は `fingerprintVersion` increment mandatory
- version mismatch 時は reuse prohibited

---

## 5. Retire Backpressure（固定）

### 5.1 基本値

```cpp
highWatermark = 3072;
lowWatermark  = 1024;
```

### 5.2 scaling clamp（固定）

全 scale factor に clamp 適用:

```cpp
scale = clamp(scale, 0.75, 1.50);
```

対象:

- sampleRateScale
- irComplexityScale
- oversamplingScale
- memoryPressureScale

### 5.3 scaling source（固定）

`memoryPressureScale` は runtime-local deterministic metrics only:

- retire queue depth
- fallback queue depth
- rebuild backlog count
- quarantine resident count
- reclaim latency
- runtime allocation retry count
- publication backlog

禁止:

- OS global memory usage
- DAW process memory usage
- allocator opaque heuristics
- external plugin state
- system pressure callbacks

### 5.4 saturation semantics（固定）

saturation state 中は system stabilization direction only。

許可:

- HWM increase
- LWM increase
- reject aggressiveness increase
- rebuild coalescing increase
- obsolete rebuild discard increase

禁止:

- HWM/LWM decrease
- reject relaxation
- rebuild expansion
- queue growth encouragement

### 5.5 hysteresis / recovery（固定）

- `highWatermark > lowWatermark` mandatory
- 推奨: `(HWM - LWM) >= 512`
- saturation 解除条件: `queueDepth < lowWatermark`
- recovery: stepwise conservative relaxation mandatory
- 推奨 step:

```cpp
HWM -= 128;
LWM -= 128;
```

---

## 6. Rebuild Collapse Determinism（固定）

### 6.1 latest-generation-wins

mandatory。

### 6.2 safe-to-collapse rebuild 定義

以下をすべて満たす場合のみ collapse 可。

- UI-driven transient rebuild
- newer equivalent rebuild exists
- runtime publication not started
- no externally-visible state committed
- rebuildFingerprint equivalent
- rebuildClass identical

### 6.3 must-execute rebuild（collapse 禁止）

- prepareToPlay derived rebuild
- shutdown transition rebuild
- topology migration rebuild
- runtime recovery rebuild
- safety recovery rebuild

### 6.4 禁止

- cross-class collapse
- ambiguous reuse
- partially-matching fingerprint reuse

---

## 7. Queue / Drain 用語固定

- retire queue: epoch retire pending queue
- fallback queue: overflow / deferred retire queue
- quarantine: temporarily non-reclaimable retired runtime residency
- publication backlog: publication-intent pending work count
- rebuild backlog: non-committed rebuild work count

---

## 8. Drained 完了条件（最終固定）

drained は以下をすべて満たす状態。

- retireQueue empty
- fallbackQueue empty
- quarantine empty
- publicationCoordinator drained
- rebuildWorker stopped

`publicationCoordinator drained` の意味:

- no pending publication intent
- no active prepareCommit
- no pending executeCommit
- no publication retry scheduled

drained 後禁止:

- enqueue resurrection
- deferred retry restart
- publication revival
- rebuild relaunch

---

## 9. 実装統制 Rule（v5.5 最終固定）

- Rule-O: scaling clamp mandatory
- Rule-P: monotonic saturation stabilization mandatory
- Rule-Q: runtime-local deterministic metrics only
- Rule-R: latest-generation-wins mandatory
- Rule-S: drained resurrection prohibited
- Rule-T: finalize determinism mandatory
- Rule-U: recovery must be stepwise conservative
- Rule-V: cross-class rebuild collapse prohibited
- Rule-W: finalize must not observe runtime timing/environment
- Rule-X: saturation state may only move toward stabilization
- Rule-Y: publication drained means no publishable work remains
- Rule-Z: rebuild collapse allowed only for safe-to-collapse rebuilds

---

## 10. 段階導入順序（固定）

1. Unified Admission Gate
2. RuntimeBuildSnapshot 完全移行
3. Retire Backpressure Hardening
4. DSP Execution State 分離
5. Crossfade Authority Isolation

---

## 11. ロック宣言

本書 `ISR_Bridge_Runtime_v5.5_FINAL_FREEZE_2026-05-28.md` を、
ConvoPeq ISR Bridge Runtime hardening の基本設計（Final Freeze）として固定する。

- 以後の変更は差分追補（v5.5.x）で管理
- 本体方針（DAW-safe hardening / deterministic shutdown / collapse determinism / saturation recoverability）は不変とする

---

## 12. v5.5.1 文書追補（監査是正反映）

### 12.1 `shutdownRuntime_` の役割明確化

`shutdownRuntime_` は、shutdown/drain 期間における runtime 可視性と排出順序を管理する単一路線の参照点であり、
独立 admission state machine を追加せず `lifecycleState / shutdownPhase / shutdownRuntime_` の統合で運用する。

### 12.2 publication 経路の命名統一

`appendPublicationIntent` を正規名とし、`enqueuePublication` は legacy alias として扱う。
v5.5.1 以降の設計・実装文書では、原則 `appendPublicationIntent` を使用する。

### 12.3 Telemetry 識別子（単一情報源）

Rule-27 相当の観測項目識別子を以下に固定する。

- `retireQueueDepth`
- `fallbackQueueDepth`
- `quarantineResident`
- `publicationBacklog`
- `rebuildBacklog`
- `saturationEnterCount`
- `saturationExitCount`
- `publicationRejectCount`
- `rebuildCollapseCount`
- `reclaimLatency`
