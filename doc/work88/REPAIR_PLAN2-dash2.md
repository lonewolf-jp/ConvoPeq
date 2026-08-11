# REPAIR_PLAN2-dash2 将来対応・一部実装の詳細設計

**★ 作成日:** 2026-08-11
**★ 対象:** `doc/work88/REPAIR_PLAN2-dash.md`（本実装は完了済み）で「将来対応」「一部実装」とされた項目の**整理・詳細設計**
**★ 前提:** REPAIR_PLAN2-dash.md の本実装（P2-1〜P2-4 / X1〜X6 / X4-B / X3-R4 Phase 7）は完了済み・ctest 28/28 PASS・CI 検証 PASS
**★ normative source:** REPAIR_PLAN2-dash.md（現行版・A-2.42 以降 + §5/§6 の「現在形」記述）＋実コード照合

---

## 実装状況サマリ

| 分類 | 項目 | 現状 | ISR review 対応 |
|---|---|---|---|
| 将来対応 | 1.1 R1: recoveryIntentQueue_ の MPSC 化 | Phase 5 将来拡張（未実装） | 🟢 条件付きGO — INV-R1-1/INV-R1-2 + AC-ISR-1 追加 |
| 将来対応 | 1.2 Recovery coalesce（マージ）実装 | 四次レビュー NO-GO → 別タスク（P3） | 🟡 NO-GO継続 — LogicalRecoveryIdentity / RecoveryProvenance 分離 |
| 将来対応 | 1.3 §4.3: ConvolverProcessor の LinearRamp 分離 | 対象外・文書化（未実装） | ⚠️ **修正GO** — 1 known RT contract violation (`mixSmoother.reset()` at Runtime.cpp:360, asserts `ASSERT_NON_RT_THREAD` from RT). Phase 0.5 fix required before Tier 1 |
| 将来対応 | 1.4 isFullyDrained の他カウンタ実測上書きの全廃 | 別タスク（P2 範囲外） | 🟡 設計先行必須 — 16-condition drain semantic (reclaimInFlight + publicationIntentResidency 追加) |
| 将来対応 | 1.5 PublishReceiptWaiter の sparse completion | 将来 MPSC completion 許容時のみ | 🟢 将来保留 — FIFO invariant を維持 |
| 将来対応 | 1.6 X2 の wraparound / out-of-order テスト | 将来 sparse 化時のみ追加 | 🟢 GO — sequence arithmetic の定義も必要 |
| 将来対応 | 1.7 X4-B 案2（currentWorld_ 廃止） | 将来タスク（read-source singularization） | 🟡 高リスク — AC-PUB-1 (identity consistency) が前提 |
| 将来対応 | 1.8 BuildError 保険分類 | enum + toString のみ・catch 拡張未実装（work32 Step 3 未対応) | 🔴 現案NO-GO → 1.8.5.2 FailureClassification/RetryDisposition 分離. ⚠️ No exponential backoff in retry (RebuildDispatch.cpp:990-1048, `kMaxRecoveryConsecutiveFailures=4` only) |
| 将来対応 | 1.9 初回 publish 前 quarantine の無駄な起床 | Phase 5 最適化候補 | 🟡 条件付きGO — 3-condition wake + lost-wake proof |
| 一部実装 | 2.1 R4: retire 順序逆転の完全解消 | runtime 経路対応済み・完全解消は保留 | 🟡 条件付きGO — epoch safety と FIFO を分離 (UAF/リークは既に排除済み) |
| 一部実装 | 2.2 shutdown lifetime contract の明文化 | R4 詳細設計で ShutdownQuiescenceProof 確定済み | 🟢 強く GO — ReclaimPermit pattern (ShutdownRuntime only) |

---

## 総合判定 (ISR Architecture Review 2026-08-11)

| 項目 | 判定 | ISR観点 |
|---|---|---|
| 1.1 Recovery Queue MPSC化 | 🟢 条件付きGO | RT producer禁止を明文化した点は正しい |
| 1.2 Recovery coalesce | 🟡 NO-GO継続 | Logical identity定義がまだ不十分（1.2.1 で修正済み） |
| 1.3 LinearRamp owner audit | 🟢 GO | 変更せず監査だけなら安全 |
| 1.4 isFullyDrained再設計 | 🟡 設計先行必須 | shutdown/lifetime safetyに直結 |
| 1.5 sparse completion | 🟢 将来保留 | 現行FIFO invariantを崩す必要なし |
| 1.6 sequenceテスト | 🟢 GO | sparse化時に必要 |
| 1.7 currentWorld_廃止 | 🟡 高リスク | ISR read semantic変更を伴う |
| 1.8 BuildError wiring | 🔴 現案のままはNO-GO | retry分類が粗すぎる（1.8.5.2 で分離追加済み） |
| 1.9 quarantine wake最適化 | 🟡 条件付きGO | lost wake防止の証明が必要（1.9 AC で追加済み） |
| 2.1 retire順序FIFO化 | 🟡 条件付きGO | epoch safetyとFIFOを混同しない（2.1 で修正済み） |
| 2.2 shutdown lifetime proof | 🟢 強くGO | ISR lifetime safetyを強化する |

> **「ISR安全」と「lifetime安全」は別だが、後者を壊せば結果的にISR安全ではなくなる**

### Tier 優先順位

| Tier | 内容 | 項目 |
|---|---|---|
| **Tier 1** (lifetime safety) | 最優先 | 2.2 → 1.4 → 2.1 → 1.7 |
| **Tier 2** (recovery correctness) | 5-7 | 1.1 → 1.9 → 1.2 |
| **Tier 3** (infrastructure hardening) | 8-10 | 1.8 → 1.5/1.6 |

---

# 1. 将来対応（Phase 5 以降・将来タスク）

## 1.1 R1: recoveryIntentQueue_ の MPSC 化

**現状（実コード照合）**
- `recoveryIntentQueue_` は `LockFreeRingBuffer<RecoveryIntent, 256>`（SPSC、ISRRuntimePublicationCoordinator.h:433-434）
- Producer は **CoordinatorLoop 単一スレッド**（`submitRecoveryRequest` → intentQueue_ → processIntent → RecoveryIntentHandler → `submitRecoveryIntent`）
- `popRecoveryRequest`（Builder Work Queue 消費）も単一スレッド

**課題 / 動機**
- 将来 Timer 等から直接 `submitRecoveryRequest` を呼ぶ場合、**MPSC 化が必要**
- 現状は SPSC 成立のため**緊急性なし**（Phase 5 将来拡張）

**設計方針**
```
LockFreeRingBuffer<RecoveryIntent, 256>  →  MpscBoundedRing<RecoveryIntent, 256>
Producer 側 submitRecoveryRequest:  §1.1（REPAIR_PLAN2-dash）の reservation→push→rollback に変更
Consumer 側 popRecoveryRequest:     pop 成功時 fetchSub（pendingIntentCount_）
```

> ⚠️ **MPSC queue 自体が lock-free だから RT safe、ではない。** MPSC CAS retry が bounded/wait-free である保証と、**RT producer を許可することは別問題。** Producer は **NonRT-only** に限定する architectural invariant を追加すること (INV-R1-MPSC-4)。

**⚠️ レビュー指摘: reservation の不変条件を明示する:**

**INV-R1-1**
```text
pendingIntentCount_
=
成功したreservationの総数
−
成功したconsumer pop/discardの総数
```

**INV-R1-2**
```text
pendingIntentCount_ == 0
⇒
queue visible residency == 0
AND
producer reservation == 0
```

INV-R1-2 は shutdown proof 側から使える形にする必要がある（isFullyDrained との整合）。

#### 1.1.1 ⚠️ レビュー指摘: MPSC queue だけでは不足 — `pendingRecoveryAdmission_` もMPSC-safeにする必要

**現行コードの実態 (実コード照合):**

```text
intentQueue_         → MpscBoundedRing (既にMPSC！ ISRRuntimePublicationCoordinator.h:602)
quarantineFallbackQueue_ → MpscBoundedRing (既にMPSC！ ISRRuntimePublicationCoordinator.h:610)
recoveryIntentQueue_ → LockFreeRingBuffer (SPSC未変更！ h:551)
pendingRecoveryAdmission_ → plain struct (SPSC専用 — atomicなし！ h:590)
```

`intentQueue_` と `quarantineFallbackQueue_` は既に `MpscBoundedRing` に移行済み。しかし、`recoveryIntentQueue_` は**依然SPSC** (`LockFreeRingBuffer`)。

**問題:** `submitRecoveryRequest` (cpp:721-781) は2つのパスを持つ:
1. **Transport path**: `recoveryIntentQueue_.push()` → queue residency
2. **Durable path**: queue full時 → `pendingRecoveryAdmission_` へ fallback

`pendingRecoveryAdmission_` は `// SPSC（plain 構造体 — atomic 不要）` と明記されている (h:590)。

**→ `recoveryIntentQueue_` をMPSC化するなら、`pendingRecoveryAdmission_` もmutexまたは原子的操作にする必要がある。**

2つのProducerが同時に queue full → durable fallback すると、以下のraceが発生する:

```text
Producer A: push fail → rollback(pendingIntentCount_--) → write pendingRecoveryAdmission_ = Recovery A
Producer B: push fail → rollback(pendingIntentCount_--) → write pendingRecoveryAdmission_ = Recovery B
                        ↑ lost update / data race / identity corruption
```

**推奨アプローチ (review §3 P0-A):**

Recovery Admissionを**single NonRT admission authority**に集納する:

```text
RecoveryIntent
      ↓
RecoveryAdmissionAuthority (NonRT only, single producer)
      ├─ recoveryIntentQueue_     (transport)
      └─ pendingRecoveryAdmission_ (durable)
```

つまり、**producerをCoordinatorLoop singleに限定**するか、**両方をmutex保護**するかのいずれか。

`intentQueue_` が既にMPSCなので、`pendingRecoveryAdmission_` は**single-writer invariant**を保つことでSPSC-safeに維持されているが、`recoveryIntentQueue_`をMPSC化するとこの保証が壊れる。

---

**実装手順**
1. `recoveryIntentQueue_` の型を `MpscBoundedRing<RecoveryIntent, 256>` へ置換
2. `submitRecoveryRequest` に reservation→push→rollback（pendingIntentCount_ 管理）を追加
3. `popRecoveryRequest` で pop 成功時 fetchSub
4. テスト追加: 2 Producer からの並行 enqueue / queue full → rollback / 重複なし / underflow なし

**Acceptance Criteria**
- MPSC 2 Producer で全件 enqueue・**重複なし・underflow なし**（reservation-before-push の happens-before チェーン）
- `isFullyDrained`（pendingIntentCount_ + queue emptiness）が整合
- **INV-R1-1**: pendingIntentCount_ = Σreservations − Σpops (数式的定義)
- **INV-R1-2**: pendingIntentCount_ == 0 ⇒ queue residency == 0 AND producer reservation == 0
- **AC-ISR-1**: Audio Thread は MPSC enqueue producer にならない
- 既存 Recovery テスト（ISRSemanticValidationTests / AudioEngineHarness 統合）全 PASS

**影響範囲 / リスク**
- 影響: `ISRRuntimePublicationCoordinator.h`（recoveryIntentQueue_ 型）、`submitRecoveryRequest`、`popRecoveryRequest`、テスト
- リスク: **中**（MPSC 化に伴う memory_order / Vyukov bounded queue の producer hole の正しさ）。MpscBoundedRing は P2-3 でテスト固定済み

---

## 1.2 Recovery coalesce（マージ）実装

**現状**
- REPAIR_PLAN2-dash.md §2.2 で **四次レビュー NO-GO**（`lastRecoveryHandle_` tracking は**正当な Recovery を silent loss** するため）
- 現行は「1 logical Recovery admission = exactly one reservation（INV-X1-5）」「durable は transport と二重計上しない（INV-X1-6）」で成立

**課題 / 動機**
- 同一 handle の重複 Recovery が発生した場合の**安全なマージ**（reservation 増加なし・正当な Recovery を喪失しない）
- coalesce 実装は**別タスク（P3）**として記録

**設計方針（NO-GO 理由を踏まえた安全設計）**
```
現方式（単一 last handle tracking）は採用しない（NO-GO の再発防止）
```

#### 1.2.1 ⚠️ レビュー指摘: "same handle = duplicate" は不十分

**⚠️ handle identity ≠ logical Recovery identity**

The current proposed approach (detecting duplicate by "same handle") is insufficient because:

```text
Recovery A: handle=H, generation=10, buildSource=S1
Recovery B: handle=H, generation=10, buildSource=S2

A と B は本当に同一 logical recovery か？
```

Handle equality alone cannot answer this — they could be:
- Same logical recovery, different build sources (legitimate update → coalesce)
- Independent recoveries that happen to share a handle (e.g., re-admission after partial failure)

**⚠️ レビュー追加指摘: `admission source` を identity に含めることは不適切**

> source = Transport と source = Durable が同じ logical recovery を表すことがある。
> identity に admission source を入れると、むしろ二重計上のリスクが高まる。

**Required definition — `LogicalRecoveryIdentity` (admission sourceを除外):**

```text
LogicalRecoveryIdentity =
    handle
  + generation (epoch-consistent)
  + semantic build identity (IR hash, sample rate, channel count, ...)
  + activation/publication epoch
```

**Separately — `RecoveryProvenance` (identity ≠ provenance):**

```text
RecoveryProvenance =
    Transport   (transport-layer failure: IR load timeout in flight)
    Durable     (durable layer failure: build snapshot failed)
    Retry       (transient retry after failure)
    Quarantine  (quarantined due to age or error)
```

```text
identity = 「何を復旧するのか」
provenance = 「なぜこの要求が存在するのか」
```

This separation aligns with X1's invariant: **1 logical Recovery admission = exactly one reservation**, regardless of how many provenance paths converge.

Without this, the "obsolete policy" (A(G10), B(G10), C(G11) → keep only G11) cannot be proven correct — because **G10 might not be fully subsumed by G11**.

#### 1.2.2 ⚠️ レビュー指摘: identity equality だけでは不十分 — `RecoverySupersession` が必要

> Handle equality alone cannot answer this...
> `generation 11 > generation 10` だけでは、G11 が G10 の Recovery obligation を完全に代替できるとは限らない。

例えば:

```text
G10 = EQ change   (recovery: EQ coefficients)
G11 = IR change   (recovery: convolver IR)
```

G11がG10を完全に包含できるとは限らない。

**必要な追加概念 — `RecoverySupersession`:**

```cpp
// G11がG10のRecovery obligationを完全に代替できるかの純粋関数
bool canSupersede(
    const LogicalRecoveryIdentity& newer,
    const LogicalRecoveryIdentity& older)
{
    // handle が同じ
    if (newer.handle != older.handle) return false;
    // generation が strictly greater
    if (newer.generation <= older.generation) return false;
    // semantic build identity: newer が older のsupersetか
    //   → EQ change (G10)  vs IR change (G11) → NOT supersede
    //   → IR+EQ change (G10) vs IR+EQ+OS change (G11) → supersede
    return isSemanticSuperset(newer.buildIdentity, older.buildIdentity);
}
```

**coalesce の正しい条件:**

```text
A(G10), B(G10), C(G11)
    ↓
canSupersede(C, A) == true かつ  canSupersede(C, B) == true
    ↓
A, B を drop — C のみを保持
```

`generation >` だけではなく、**semantic build identity の superset 関係**を証明しなければならない。

**Future implementation when coalescing:**
1. Define `LogicalRecoveryIdentity` struct with full semantic key (handle + generation + build identity hash + epoch) — **NOT admission source**
2. Define `RecoveryProvenance` enum (Transport / Durable / Retry / Quarantine) — separate from identity
3. State machine: `NoAdmission → DurablePending → Building → [TransientFail → DurablePending] → [Success → NoAdmission]`
4. Enforce: `exactly 1 logical admission = exactly 1 durable obligation` at every state transition
5. Test: verify `logicalAdmissionCount == 1` invariant holds after every coalesce operation

**実装手順**
1. `LogicalRecoveryIdentity` を定義 (handle + generation + build identity hash + epoch — **admission source を含めない**)
2. `RecoveryProvenance` enum を定義 (Transport / Durable / Retry / Quarantine) — identity と分離
3. State machine を定義: `NoAdmission → DurablePending → Building → [TransientFail → DurablePending] → [Success → NoAdmission]`
4. `exactly 1 logical admission = exactly 1 durable obligation` の invariant を各 transition で保証
5. `PendingRecoveryAdmission` に `LogicalRecoveryIdentity` ベースの重複検出を追加
6. 重複時: buildSource を最新化・reservation は増やさない (INV-X1-5 / INV-X1-6)
7. `hasDuplicates` API を新設
8. テスト: Recovery A,A coalesce / queue full → durable / 二重計上なし / 正当な Recovery が喪失されない / logicalAdmissionCount == 1

**Acceptance Criteria**
- **INV-X1-5**: 1 logical Recovery admission = exactly one reservation（coalesce で reservation 増加なし）
- **INV-X1-6**: durable Recovery は transport residency と二重計上しない
- A(G10) / B(G10) / C(G11) → G11 のみが必要（obsolete policy 再利用）

**影響範囲 / リスク**
- 影響: `PendingRecoveryAdmission`、`submitRecoveryRequest`、RebuildDispatch recovery
- リスク: **高**（coalesce が正当な Recovery を喪失しないことの保証が困難 → NO-GO 理由）。実装時は徹底的なテストが必要

---

## 1.3 §4.3: ConvolverProcessor の LinearRamp 分離

**現状（実コード照合）**
- `ConvolverProcessor.h:910,935,945` の `latencySmoother` / `crossfadeGain` / `mixSmoother` は、CrossfadeRuntime の `gain_` / `dryScaleGain_` とは**別個**の `LinearRamp` (`DspNumericPolicy.h:319-406`)
- **⚠️ 2026-08-11 追加 (2026-08-11 review Amendment 1)**: **`mixSmoother.reset()` at `ConvolverProcessor.Runtime.cpp:360` is a KNOWN RT CONTRACT VIOLATION** — called from audio processing thread (RT), but `LinearRamp::reset()` asserts `ASSERT_NON_RT_THREAD()` (DspNumericPolicy.h:333). This is **not a future audit item** — it is an existing bug requiring Phase 0.5 fix before any Tier 1 work.
- 設計判断として**対象外・文書化**（四次レビュー承認） — however, the known RT violation must be resolved first

**課題 / 動機**
- 将来の**独立 RT-safety 検証**（各 smoother の RT スレッド所有権・atomic 性）

**設計方針**
- 対象外（文書化）を維持しつつ、将来各 smoother の read/write 箇所を棚卸しして独立検証

**実装手順**
1. `ConvolverProcessor` の 3 つの smoother（latencySmoother / crossfadeGain / mixSmoother）の read/write 箇所を棚卸し
2. 各 smoother が RT スレッドのみで write されることを検証
3. 非 RT からの write がある場合のみ修正（RT 安全性の担保）

**Acceptance Criteria**
- 各 smoother が **RT スレッドのみで write** される（非 RT write なし）
- **RT write → RT read**（NonRT write = 0）
- **NonRT read/write による data race = 0**
- **lifetime mutation = 0**
- 既存 ConvolverProcessor テスト・オーディオパス検証（HeadlessAudioPathVerification）全 PASS

   **影響範団 / リスク**
   - 影響: `ConvolverProcessor.h`, `ConvolverProcessor.Lifecycle.cpp`, `ConvolverProcessor.Runtime.cpp`, `LinearRamp` (`DspNumericPolicy.h`)
   - リスク: **Medium** (1 known RT contract violation — `mixSmoother.reset()` at Runtime.cpp:360 asserts `ASSERT_NON_RT_THREAD` from RT path). Must fix in Phase 0.5 before Tier 1 work.
   - **AC-ISR-1**: Audio Thread は LinearRamp の `reset()` / `setCurrentAndTargetValue()` (NonRT-only) を呼び出さない。`applyImmediateValueRT()` / `setTargetValue()` / `getNextValue()` (RT-only) のみ呼び出す。

  **⚠️ 実コード照合 (2026-08-11 深層調査) — LinearRamp RT write 違反: 1 件検出**

  LinearRamp (`DspNumericPolicy.h:319-406`) のメソッド別スレッド属性:

  | Method | ASSERT | NonRT write | RT write |
  |---|---|---|---|
  | `reset()` | `ASSERT_NON_RT_THREAD` (h:333) | ✅ NonRT (prepareToPlay) | ⚠️ **RT violation** (see below) |
  | `setCurrentAndTargetValue()` | `ASSERT_NON_RT_THREAD` (h:341) | ✅ NonRT only | ❌ |
  | `applyImmediateValueRT()` | `ASSERT_AUDIO_THREAD` (h:352) | ❌ | ✅ RT only |
  | `setTargetValue()` | `ASSERT_AUDIO_THREAD` (h:363) | ❌ | ✅ RT only |
  | `getNextValue()` | `ASSERT_AUDIO_THREAD` (h:374) | ❌ | ✅ RT only |
  | `getCurrentValue()` | none | ✅ | ✅ (const, no assert) |
  | `getTargetValue()` | none | ✅ | ✅ (const, no assert) |

  **発見事項 — `mixSmoother.reset()` が RT スレッドから呼び出されている (Severity: Medium)**

  `ConvolverProcessor.Runtime.cpp:360` において、**RT オーディオスレッド**（process() / processReplacing() のコールスタック）が `activeMixSmoother.reset(sampleRate, newTime)` を呼び出す。この `reset()` は `ASSERT_NON_RT_THREAD()`（`DspNumericPolicy.h:333`）を含むため、**Debug builds では jassert 違反**を引き起こす。

  起動経路: `setSmoothingTime()` (NonRT, ConvolverProcessor.Runtime.cpp:912) が `smoothingTimeChangePendingGen` を fetchAdd (acq_rel) → RT スレッドの `process()` が generation を acquire で検知 → `reset()` + `applyImmediateValueRT()` + `setTargetValue()` を呼び出す (Runtime.cpp:350-362)。

  影響: `reset()` は `totalSteps`（ランプ長 in samples）のみを変更する。このフィールドは `LinearRamp` のコメントで「Audio Thread 中は不変」とされているが、RT スレッドから変更されている。`totalSteps` は read-only なので data race ではないが、**assertion contract violation** であり、debug ビルドでの false positive jassert の根源となっている可能性。

  **🔴 現段階: 実装不要 (対象外・文書化)**. 将来 LinearRamp 抽象化実装時 (1.3 の将来タスク) に以下を対処:
  - `reset()` を RT-safe なバージョンに分割（`resetRT()` は `totalSteps` のみ atomic 更新）
  - または `totalSteps` を `std::atomic<int>` に変更し `reset()` の ASSERT を撤去
  - 非 RT の `setCurrentAndTargetValue()` は Lifecycle.cpp:377 でのみ呼び出され ✅

  **read/write 箇所棚卸し (2026-08-11 完成):**

  | Smoother | File | Line | Method | Thread |
  |---|---|---|---|---|
  | latencySmoother | Lifecycle.cpp | 375 | `reset()` | NonRT ✅ |
  | latencySmoother | Lifecycle.cpp | 382 | `setCurrentAndTargetValue()` | NonRT ✅ |
  | latencySmoother | Lifecycle.cpp | 388 | `getTargetValue()` | NonRT ✅ |
  | latencySmoother | Runtime.cpp | 279 | `getTargetValue()` | RT ✅ |
  | latencySmoother | Runtime.cpp | 283 | `getCurrentValue()` | RT ✅ |
  | latencySmoother | Runtime.cpp | 284-286 | `applyImmediateValueRT()` / `setTargetValue()` | RT ✅ |
  | latencySmoother | Runtime.cpp | 317 | `applyImmediateValueRT()` | RT ✅ |
  | latencySmoother | Runtime.cpp | 395 | `getTargetValue()` | RT ✅ |
  | crossfadeGain | Lifecycle.cpp | 376 | `reset()` | NonRT ✅ |
  | crossfadeGain | Lifecycle.cpp | 377 | `setCurrentAndTargetValue()` | NonRT ✅ |
  | crossfadeGain | Runtime.cpp | 284-285 | `applyImmediateValueRT()` / `setTargetValue()` | RT ✅ |
  | crossfadeGain | Runtime.cpp | 333-334 | `applyImmediateValueRT()` / `setTargetValue()` | RT ✅ |
  | mixSmoother | Lifecycle.cpp | 370 | `reset()` | NonRT ✅ |
  | mixSmoother | Lifecycle.cpp | 370-371 | `setCurrentAndTargetValue()` | NonRT ✅ |
  | mixSmoother | Lifecycle.cpp | 489 | `reset()` (pending gen bump) | NonRT ✅ |
  | mixSmoother | Runtime.cpp | 345 | `applyImmediateValueRT()` | RT ✅ |
  | mixSmoother | Runtime.cpp | 358-362 | `getCurrentValue()` / `getTargetValue()` / **`reset()`** / `applyImmediateValueRT()` / `setTargetValue()` | **RT ⚠️ (reset violates ASSERT)** |
  | mixSmoother | Runtime.cpp | 368 | `getTargetValue()` / `setTargetValue()` | RT ✅ |
  | mixSmoother | Runtime.cpp | 373 | `setTargetValue()` | RT ✅ |
  | mixSmoother | Runtime.cpp | 601 | `getNextValue()` | RT ✅ |

  ---
## 1.4 isFullyDrained の他カウンタ実測上書きの全廃

- `AudioEngine::isFullyDrained`（Threading.cpp:114-156）が、`fallbackBacklog` / `retireBacklog` / `deferredRetire` / `quarantineResident` を**外部 setter で実測上書き**している
  - ⚠️ **実際のline refは Threading.cpp:114-156**（document says :117,131 — close enough but :126-128 is where the 3 setters are called）
- `AudioEngine::isFullyDrained`（Threading.cpp:114-156）は **2層構造**:
  - **Layer 1 (AudioEngine)**: `!hasDeferredCommit`（:116 via hasDeferredRequest）+ `pendingReclaimHandles_.empty()`（:145-149, mutex-guarded）+ `overflowRing.residentCount()`（:134-135）+ `dspQuarantineManager.residentCount()`（:136）+ `retireRouter.quarantineResidentCount()`（:137-138）+ `runtimePublicationBridge_.isFullyDrained()`（:156）
  - **Layer 2 (Coordinator)**: `ShutdownScheduler::isFullyDrained()` at ISRRuntimePublicationCoordinator.cpp:484-526 (1 pre-check `swapPending_` + 15 return-body conditions)

**課題 / 動機**
- 上書きが「drain 判定の正しさ」を担保している面があり、**廃止には全カウンタの正確な増減管理が必要**
- Coordinator 内部の純粋 accounting に一本化したい

**設計方針**
```
外部 setter（setFallbackBacklogCount / setRetireBacklogCount / setDeferredRetireResidencyCount /
quarantineResidentCount 上書き）を廃止
→ isFullyDrained で実測値（queue size / DSPQuarantineManager::residentCount）を直接判定
```

#### 1.4.1 ⚠️ レビュー指摘: `queue size == 0` だけでは不十分

**現行コードの16条件 (ISRRuntimePublicationCoordinator.cpp:498-526):**

```cpp
// Layer 1: queue emptiness
intentQueue_.sizeApprox() == 0
observeDeferredRing_.size() == 0
quarantineFallbackQueue_.sizeApprox() == 0
recoveryIntentQueue_.size() == 0
// Layer 2: backlog counters
retireBacklogCount_ == 0
publicationBacklogCount_ == 0
publicationIntentResidencyCount_ == 0    // ✅ review §6 addition
pendingIntentCount_ == 0                  // reservation hole
fallbackBacklogCount_ == 0
reclaimInFlightCount_ == 0                // ✅ review §6 addition
deferredRetireResidencyCount_ == 0
quarantineIntentResidencyCount_ == 0      // X6
quarantineRingResidencyCount_ == 0        // X6
quarantineResidentCount_ == 0             // physical DSP objects (Phase 2)
!recoveryAdmissionPending_                // durable admission
```

**Drain semantics を定義する必要がある:**

```text
FULL_DRAINED =
    publication transport == 0        (intentQueue_ residency)
AND recovery transport == 0          (recoveryIntentQueue_ residency)
AND quarantine transport == 0        (quarantineIntentResidencyCount_)
AND producer reservations == 0       (pendingIntentCount_ — reservation hole)
AND durable recovery == none         (recoveryAdmissionPending_ — DurablePending/Building)
AND retire pending == 0              (retireBacklogCount_)
AND deferred retire == 0             (deferredRetireResidencyCount_)
AND reclaim in-flight == 0           (reclaimInFlightCount_)
AND quarantine physical == 0         (quarantineResidentCount_ — actual DSP objects)
AND publication intent residency == 0 (publicationIntentResidencyCount_)
```

> ⚠️ `isFullyDrained` を単なる `queue.size() == 0` に変更する前に、**drain semantic の定義が必要**。
> 現行コード（`ISRRuntimePublicationCoordinator.cpp:500-525`）ではすでに16の複合条件で判定しているが、
> `reclaimInFlightCount_` と `publicationIntentResidencyCount_` は dash2 設計方針で見落としていた。
>
> MPSC producer hole では `queue visible items == 0` かつ `reservation count == 1` という状態があり得る。
> この場合 `isFullyDrained()` が `true` を返すと shutdown proof が欠陥する。

**INV-ISFULLDRAINED-1〜5** (現行コードから派生):
- **INV-ISFULLDRAINED-1**: transport residency == 0 (publication + recovery + quarantine intent transports)
- **INV-ISFULLDRAINED-2**: producer reservations == 0 (pendingIntentCount_ — reservation hole)
- **INV-ISFULLDRAINED-3**: durable admission == none (recoveryAdmissionPending_ == false)
- **INV-ISFULLDRAINED-4**: retire/backlog/deferred == 0 (retireBacklogCount_ + deferredRetireResidencyCount_ + reclaimInFlightCount_)
- **INV-ISFULLDRAINED-5**: quarantine physical residency == 0 (quarantineResidentCount_ from DSPQuarantineManager)

**実装順序:**
1. drain semantic を定義/文書化 (INV-ISFULLDRAINED-1〜5) — **完了済み（現行16条件）**
2. 各カウンタの増減箇所を棚卸し (push/pop の実測)
3. `isFullyDrained` で複合条件を判定 (現行16条件を維持・ドキュメント化)
4. 外部 setter を廃止

**実装手順**
1. drain semantic を定義/文書化 (INV-ISFULLDRAINED-1〜5: transport + reservation + durable admission + retire + quarantine + reclaim + publication intent)
2. 各カウンタの増減箇所を棚単し（push/pop の実測）
3. `isFullyDrained` で複合条件を判定（現行16条件を維持・ドキュメント化）
4. 外部 setter を廃止
5. テスト: isFullyDrained が実測と整合 / ctest 全 PASS

**Acceptance Criteria**
- drain semantic が定義済み (INV-ISFULLDRAINED-1〜5)
- 外部 setter が**廃止**される（コード参照 0）
- `isFullyDrained` が複合条件を判定 (transport + reservation + durable + retire + quarantine + reclaim + publication intent)
- `isFullyDrained` が実測と整合 (X5/X6 のカウンタと照合)
- `isr-verify-backlog-specfixed-residual.ps1` が PASS

**影響範囲 / リスク**
- 影響: `AudioEngine.Threading.cpp`、`AudioEngine.Commit.cpp`、CI スクリプト
- リスク: **中**（上書き廃止による drain 判定の過検出/未検出。X5/X6 でカウンタが正確化されているため、廃止可能な基盤は整っている）

---

## 1.5 PublishReceiptWaiter の sparse completion

**現状（実コード照合）**
- `PublishReceiptWaiter`（AudioEngine.h:3631）は **high-water mark（`lastCompleted_`）+ contiguous completion**（INV-X2-6: completion order == publication sequence order）
- `complete()` は `if (seqId > lastCompleted_)` で monotonic watermark（INV-X2-4: stale completion は上書きしない）

**課題 / 動機**
- 将来 **MPSC completion / parallel publish / async completion** を許す場合、contiguous 前提が壊れる
- 現在は PublishExecutor sole gateway + FIFO で**不要**

**設計方針**
```
completedThrough_（contiguous frontier）+ completedOutOfOrder_（sparse set）の二重構造
waitFor(seq) は frontier と sparse を併用
INV-X2-5（sole completion writer）は維持
```

**実装手順**
1. `completedThrough_` / `completedOutOfOrder_` を導入
2. `complete()` を CAS max + sparse set 更新に変更
3. `waitFor()` を frontier + sparse 併用に変更
4. テスト追加: out-of-order / duplicate / wraparound

**Acceptance Criteria**
- out-of-order completion でも `waitFor(seq)` が正しく完了
- 既存 contiguous テスト（PublishPipelineIntegrationTests / PublishCompletionMonotonicity）維持
- INV-X2-6 の architectural test（PublishExecutor 以外から completion が発生しない）維持

**影響範囲 / リスク**
- 影響: `AudioEngine.h`（PublishReceiptWaiter）、`commitRuntimePublication`（waitForPublishReceipt）
- リスク: **高**（completion 意味論の変更。contiguous 前提が architecture に埋め込まれている。**現状は実装不要**）

---

## 1.6 X2 の wraparound / out-of-order テスト

**現状**
- INV-X2-6（completion order == publication sequence order）を architectural test で固定
- 正常系 10→11→12（FIFO completion invariant）を検証
- out-of-order は **PublishExecutor sole gateway である限り不要**（:1844）

**課題 / 動機**
- 将来 sparse completion（1.5）を導入する場合に備え、テストを準備しておく

**設計方針**
- 現状は architectural invariant 固定を維持
- **sequence arithmetic を先に定義**: `isBefore(a,b)`, `isAfter(a,b)`, `isCompleted(seq, watermark)` を modulo UINT64 で仕様化
- **sparse 化（1.5）と同時に**、`11→10（out-of-order）/ 10→10（duplicate）/ UINT64_MAX→0（wraparound）` のテストを追加

#### 1.6.1 ⚠️ レビュー指摘: sequence comparison は `a < b` では不十分

```text
UINT64_MAX - 1 → UINT64_MAX → 0 → 1
```

を

```cpp
if (a < b)  // WRONG: wraparound 後 a > b になる
```

では壊れる。必要なのは:

```cpp
bool isBefore(uint64_t a, uint64_t b) {
    return (b - a) < (UINT64_MAX / 2);  // modular arithmetic
}
```

この `isBefore` を定義してから1.6のテストを追加する。

**実装手順**
1. 現状維持（INV-X2-6 の architectural test）
2. 1.5（sparse completion）実装時にテスト追加
3. 追加テスト: 10→11→12（正常）/ 11→10（out-of-order）/ 10→10（duplicate）/ UINT64_MAX 近傍（wraparound）

**Acceptance Criteria**
- 正常系テスト維持
- sparse 化時に out-of-order / duplicate / wraparound テストが全 PASS

**影響範囲 / リスク**
- 影響: テストのみ
- リスク: **低**（現状維持・テスト準備）

---

## 1.7 X4-B 案2（currentWorld_ 廃止 = read-source singularization）

**現状**
- `currentWorld_`（ISR メタデータ observation alias）と `runtimeStore.current`（物理 publication source）の **dual-pointer**
- X4-B は **write authority singularization** に限定（案1）
- dual-pointer は「暫定正常状態」として明示（INV-X4-6: publish 完了後は同一 PublicationIdentity を保証）

**課題 / 動機**
- read-source singularization（currentWorld_ 廃止・読み取りを全て runtimeStore.current に一本化）
- ISR commit の意味論変更を伴うため**独立タスク**（A-2.20 に記録）

**設計方針**
```
ISR 側 read（currentEpoch / sequence / version）を RuntimeState::publication から直接導出
（FUTURE-4 で persistentState_ 削除済み → currentWorld_ も削除可能な方向）
commit() の currentWorld_ 更新を廃止し、メタデータ source を runtimeStore.current に一本化
```

**実装手順**
1. ISR read（currentEpoch / sequence / version）を `runtimeStore.current` 経由に変更
2. `commit()` の `currentWorld_` 更新を廃止
3. INV-X4-6 の検証を identity 一致に一本化
4. テスト: Test 9（dual-pointer identity）/ read 経路が runtimeStore のみ / INV-X4-A/B/C 充足

**Acceptance Criteria**
- `currentWorld_` が**廃止**される
- 全 read が `runtimeStore.current`（INV-X4-B）のみ
- INV-X4-A / INV-X4-C（RT API は currentWorld_ から ownership/lifetime を導出しない）充足
- **AC-PUB-1**: `RuntimeStore::current` と publication metadata が同一 `PublicationIdentity` を必ず共有する
- Test 9（PublicationIdentity 一致）・Test 10（INV-X4-7）維持

**⚠️ レビュー指摘: read-source singularization = 単純なポインタ置換ではない**

`RuntimeWorldAuthority::publish()` は `commit metadata → publishAndSwap` の順序を契約にしている（現行コードで commit-before-swap が明示済み）。`currentWorld_` を消すには、`RuntimeStore::current` にある `RuntimeState` の publication metadata が**物理store swap の時点で必ず完全に観測可能**であることを証明しなければならない。

**影響範囲 / リスク**
- 影響: `RuntimeWorldAuthority`、`ISRRuntimePublicationCoordinator`、`RuntimeIntentCoordinator`、read path 全箇所（~13 箇所）
- リスク: **高**（ISR commit 意味論変更・大規模リファクタ）。**dual-pointer は「暫定正常状態」として許容済みのため、実装は任意・将来タスク**

---

## 1.8 BuildError 保険分類

### 1.8.1 補足: C-2 起源（work32 implementation_plan.md §7 / implementation_audit_report.md §3.6）

`MKLFailure` / `ConvolverFailure` / `PrepareFailure` は **work32 タスク C-2「BuildError 分類拡充」** によって enum と `toString` に追加された。C-2 は 4 ステップの計画だったが、checklist.md:80-81 によれば**Step 1（enum 拡張）と Step 2（toString 拡張）のみ実装・確認済み**。

| Step | 内容 | 実装状況 | 備考 |
|---|---|---|---|
| 1 | enum 拡張（MKLFailure / ConvolverFailure / PrepareFailure） | ✅ 完了 | RuntimeBuilder.h:107-113 |
| 2 | toString() switch-case 拡張 | ✅ 完了 | RuntimeBuilder.cpp:53-72 |
| 3 | build() catch 拡張（`catch(mkl::exception)→MKLFailure` 等） | ❌ **未実装** | work32 plan:803-835 に仕様記載ありが、実装はされていない |
| 4 | rebuildThreadLoop telemetry（buildErrorCount_） | ❌ **未実装** | work32 plan:842-861 は optional。`buildErrorCount_` フィールドも存在しない |

**監査の範囲不足**: work32/implementation_audit_report.md:102-107 は C-2 について**enum 追加 + toString 拡張の 2 項目のみ**を検証した（コード証拠: RuntimeBuilder.h:13-15, RuntimeBuilder.cpp:47-52）。**Step 3 の catch 拡張は監査対象外**。現行コードでも `mkl::exception` 型は存在しない（AiDex 検索: 0 ヒット）。

**監査の行番号不一致**: work32 監査は RuntimeBuilder.h:13-15 / RuntimeBuilder.cpp:47-52 を参照するが、現行コードは v9.5 リオーガナイゼーション後に RuntimeBuilder.h:107-116 / RuntimeBuilder.cpp:53-76 に移動している。監査は旧版（v9.5 以前）の行番号で実施された。

### 1.8.2 実コード照合 — BuildError の完全な分類

#### 1.8.2.1 Enum 定義（RuntimeBuilder.h:107-116）

```cpp
enum class BuildError {
    None,
    InvalidInput,
    ResourceUnavailable,
    MKLFailure,          // ★ C-2: MKL 初期化・FFT 計画失敗
    ConvolverFailure,    // ★ C-2: Convolver Build 失敗
    PrepareFailure,      // ★ C-2: DSPCore::prepare() 失敗
    WarmupFailed,
    InternalError
};
```

8 値すべて。`toString`（RuntimeBuilder.cpp:53-76）は**全 8 値に対応済み**。`switch` に `default:` はなく、フォールバックは `switch` の外に `return "Unknown"`（:75）がある（work32 監査行144 の「`default` がある」という主張は**不正確** — `default:` ラベルは存在しない）。

#### 1.8.2.2 build() 関数が実際に返す値（RuntimeBuilder.cpp:428-469）

`build()` は **try/catch** で囲まれた 3 段階の検証を行う:

| ステップ | コード | スロー可能性 | 現状のマッピング |
|---|---|---|---|
| Input validation | :433-435 (`sampleRate <= 0 \|\| blockSize <= 0`) | なし | `InvalidInput` ✅ |
| try ブロック | :441-457 | | |
| - `aligned_make_unique<DSPCore>()` | :443 | `std::bad_alloc` | `ResourceUnavailable` (:461) ✅ |
| - `convolverRt().setVisualizationEnabled(false)` | :444 | **not noexcept**（ConvolverProcessor.h:533, trivial inline） | `catch(...)` → `InternalError` |
| - `convolverRt().applyBuildSnapshot(snapshot)` | :445 | **not noexcept**（ConvolverProcessor.h:477）| `catch(...)` → `InternalError` |
| - `convolverRt().transferIRStateFrom(...)` | :447 | **not noexcept**（ConvolverProcessor.h:1132） | `catch(...)` → `InternalError` |
| - `runtime->prepare(...)` | :448-454 | **not noexcept**（AudioEngine.h:866） | `catch(...)` → `InternalError` |
| catch `std::bad_alloc` | :459-462 | | `ResourceUnavailable` ✅ |
| catch `...` | :464-467 | | `InternalError` ✅ |

**重要な設計的ギャップ — Catch-based なアプローチは根本的に不適**:
build() の try ブロック内で呼び出される convolver/prepare 操作は**すべてステータスコードベースのエラー処理**を使用する:

- `ConvolverProcessor::prepareToPlay()`（ConvolverProcessor.Lifecycle.cpp:211）: `jassert` + ステータスコードベースの初期化。C++ 例外をスローしない（`GlobalGuard` で ReaderLock を取得し、`fftHandle.reset()`、`juce::ScopedLock` を使用）。
- `ConvolverProcessor::applyBuildSnapshot()`（ConvolverProcessor.StateAndUI.cpp:271）: `juce::ScopedLock` + atomic publish。`publishRuntimeProcessSnapshot()` 内でのエラーは `bool` 戻り値や `jassert` で処理される可能性があるが、例外はスローしない。
- `DSPCore::prepare()`（AudioEngine.Processing.DSPCoreLifecycle.cpp:72）: `oversampling.prepare()`、`convolverState->prepare()`（内部で `ConvolverProcessor::prepareToPlay`）、`eqState->prepare()`、`dither.prepare()` を呼ぶ。メモリ確保失敗以外はステータスコード/`jassert` で処理される。
- MKL: `MklFftEvaluator.h` はコメントに「[v2.1] MKL DFTI を Intel IPP に換装」（:13）とあり、IPP の `IppStatus` コード（`ippsFFTInit_R_64f` 等）を使用。`DftiHandle.h` は `DftiFreeDescriptor`（C API）を RAII でラップ。**MKL は C++ 例外をスローしない** — ステータスコードベース。

したがって、work32 Step 3 の `catch(const mkl::exception&) → MKLFailure` は**アーキテクチャ的に不可能**。`mkl::exception` は存在しない上、MKL/IPP は例外をスローしない。同様に `ConvolverFailure`/`PrepareFailure` も例外ベースで捕捉できない。**正しい wiring アプローチは return-code / status-code チェックである**（1.8.4 参照）。

#### 1.8.2.3 validateWarmup() が実際に返す値（RuntimeBuilder.cpp:471-479）

```cpp
if (runtime.convolverRt().isIRLoaded() && !runtime.convolverRt().isIRFinalized())
    return BuildError::WarmupFailed;  // :476
return BuildError::None;  // :478
```

`WarmupFailed` / `None` の 2 値のみ。`convolverRt().isIRLoaded()` / `isIRFinalized()` / `isLoadingIR()` はすべて `noexcept`（AudioEngine.h:910, ConvolverProcessor.h）。

#### 1.8.2.4 実際に返る値の総計

`build()` + `validateWarmup()` の組み合わせで返り得る値:

| BuildError | build() | validateWarmup() | 実コード照合 |
|---|---|---|---|
| `None` | ✓ (implicit, :455-457) | ✓ (:478) | ✅ |
| `InvalidInput` | ✓ (:435) | ✗ | ✅ |
| `ResourceUnavailable` | ✓ (:461) | ✗ | ✅ |
| `WarmupFailed` | ✗ | ✓ (:476) | ✅ |
| `InternalError` | ✓ (:466) | ✗ | ✅ |
| `MKLFailure` | ✗ | ✗ | ✅ (未使用) |
| `ConvolverFailure` | ✗ | ✗ | ✅ (未使用) |
| `PrepareFailure` | ✗ | ✗ | ✅ (未使用) |

**結論**: 実際に返るのは `None` / `InvalidInput` / `ResourceUnavailable` / `WarmupFailed` / `InternalError` の 5 値。`MKLFailure` / `ConvolverFailure` / `PrepareFailure` は enum 定義 + `toString` のみで** build パスでは未使用**（保険分類）。

### 1.8.3 Build パスアーキテクチャ — 2 系統の分離

BuildError は**build() / validateWarmup() の 2 関数でのみ**使用される。メインの publish パス `buildRuntimePublishWorld()`（RuntimeBuilder.h:134-196, RuntimeBuilder.cpp:178-426）は **BuildError を一切使用しない**。

| 関数 | ファイル | noexcept | BuildError 使用 | 呼び出し元 |
|---|---|---|---|---|
| `build()` | h:206-207, cpp:428-469 | ✅ | ✅（BuildResult.error） | RebuildDispatch.cpp:941, :1015, :1087 の 3 サイトのみ |
| `validateWarmup()` | h:209, cpp:471-479 | ✅ | ✅（直接返り値） | RebuildDispatch.cpp:955, :1032, :1143 の 3 サイトのみ |
| `buildRuntimePublishWorld()` | h:134-137, cpp:178-426 | ✅ | ❌（別経路） | PrepareToPlay.cpp:149,271; ReleaseResources.cpp:169; Transition.cpp:22; Timer.cpp:912; Init.cpp:53; etc. |

`buildRuntimePublishWorld` は `engine.makeEngineRuntimeState()`（cpp:208）で DSPCore を間接的に作成するが、**build() を呼ばない**（cpp:428 は独立した関数定義）。失敗時は世界が nullptr になるか `jassert` で停止する（noexcept であるため例外はスローされない）。したがって **BuildError の影響範囲は rebuildThreadLoop（RebuildDispatch.cpp:897-1162）に限定**される。

### 1.8.4 3 箇所の build() 呼び出しサイトと caller パターン

すべて `AudioEngine.RebuildDispatch.cpp` の `rebuildThreadLoop` 内。**すべて `runtime == nullptr`（ポインタの null 判定）で失敗を検知し、`error` フィールドは toString でログ出力のみ**（`error != None` ではない）。

| サイト | 行 | パス | failure check | error 使用 | リトライ方針 |
|---|---|---|---|---|---|
| 1 | :941-948 | transient recovery | `recoveryResult.runtime == nullptr` | `toString()` ログのみ | `continue`（破棄） |
| 2 | :1015-1027 | durable recovery | `recoveryResult.runtime == nullptr` | `toString()` ログのみ | `settlePendingRecoveryAdmission(true)` → retry（上限: `kMaxRecoveryConsecutiveFailures=4`） |
| 3 | :1087-1098 | main rebuild task | `buildResult.runtime == nullptr` | `toString()` ログのみ | `continue`（破棄） |

validateWarmup() の 3 サイトも同様のパターン:

| サイト | 行 | パス | failure check | error 使用 | リトライ方針 |
|---|---|---|---|---|---|
| 1 | :955-971 | transient recovery warmup | `!= BuildError::None` | `toString()` ログ + destroyDSP | `continue`（破棄） |
| 2 | :1032-1049 | durable recovery warmup | `!= BuildError::None` | `toString()` ログ + destroyDSP + retry | `settlePendingRecoveryAdmission(true)` → retry（上限: `kMaxRecoveryConsecutiveFailures=4`） |
| 3 | :1143-1162 | main rebuild warmup | `!= BuildError::None` | `toString()` ログ + `shouldRetryWarmupFailure()` | `shouldRetryWarmupFailure(dsp)` → `isLoadingIR()` で判定。retryable なら `submitRebuildIntent(Structural)` |

`shouldRetryWarmupFailure()`（RebuildDispatch.cpp:78-81）: `return dsp.convolverRt().isLoadingIR();` — IR ロード中であれば retryable。

⚠️ **WarmupFailed の retryability は call-site で異なる**:
- **Site 3 (main rebuild, :1143-1162)**: `shouldRetryWarmupFailure()` (isLoadingIR check) → retryable
- **Site 2 (durable recovery, :1032-1049)**: 常に retry (`settlePendingRecoveryAdmission(true)`) — `isLoadingIR` チェックなし
- **Site 1 (transient recovery, :955-971)**: 常に discard (`continue`) — `isLoadingIR` チェックなし

これは **WarmupFailed の retryability が context-dependent** であることを裏付ける — `BuildError` だけでは判定できない。

### 1.8.5 再試行方針（REPAIR_PLAN2-dash.md:1602-1613 との整合）

REPAIR_PLAN2-dash.md の BuildError × retry マトリクス（十八次別視点11 / 別視点13）と実コードの整合:

#### 1.8.5.1 ⚠️ レビュー指摘: 現行分類は retryability 判定に不十分

> `MKLFailure` / `ConvolverFailure` / `PrepareFailure` を一律「transient → retry」とするのは安全ではない。
> 各 failure の **根本原因** によって retryability が異なる。

**問題の本質:**

| BuildError | 例として起こり得る root cause | retry可能性 |
|---|---|---|
| `MKLFailure` | temporary resource exhaustion | ✅ transient |
| `MKLFailure` | library initialization failure | ❌ permanent |
| `MKLFailure` | invalid parameter / unsupported config | ❌ permanent |
| `ConvolverFailure` | IR load timeout (in flight) | ✅ transient |
| `ConvolverFailure` | corrupted IR data | ❌ permanent |
| `PrepareFailure` | memory allocation (oversampling buffer) | ✅ transient |
| `PrepareFailure` | invalid sample rate / channel config | ❌ permanent |

**→ BuildError だけでは retryability は判定できない。**

#### 1.8.5.2 Recommended: BuildError と RetryDisposition を分離

現行: `BuildError → retry/discard` (1段階)

推奨: `BuildError → BuildFailureClass → RetryDisposition` (3段階)

```cpp
// ★ work88 (1.8.5.2): retryability 判定用の分離
// ⚠️ WarmupFailed は call-site によって retryability が変わる（isLoadingIR check）
//   → RetryDisposition に ContextDependent を追加、caller が追加 context を渡す
enum class BuildFailureClass {
    InvalidConfiguration,  // InvalidInput, some ConvolverFailure, some PrepareFailure
    ResourceTransient,     // ResourceUnavailable, some ConvolverFailure, some PrepareFailure
    DSPTransient,          // WarmupFailed + isLoadingIR() (main rebuild only), some ConvolverFailure
    DSPPermanent,          // ConvolverFailure (corrupted), PrepareFailure (invalid config)
    Internal               // InternalError
};

enum class RetryDisposition {
    Retry,           // Building → DurablePending
    Discard,         // permanent — state clear
    Inspect,         // detail-dependent — additional context required
    ContextDependent // call-site specific (e.g. WarmupFailed: isLoadingIR)
};
```

**変換マトリクス (BuildError → Class → Disposition):**

| BuildError | build() で返り得るか | BuildFailureClass | RetryDisposition |
|---|---|---|---|
| `None` | ✅ (success) | — | — |
| `InvalidInput` | ✅ (build:435) | InvalidConfiguration | Discard |
| `ResourceUnavailable` | ✅ (build:461) | ResourceTransient | Retry |
| `WarmupFailed` | ✅ (validateWarmup:476) | DSPTransient | **context-dependent** (site 3: isLoadingIR check / site 2: always retry / site 1: always discard) |
| `InternalError` | ✅ (build:466) | Internal | Discard |
| `MKLFailure` | ❌ (enum only) | Inspect | Inspect |
| `ConvolverFailure` | ❌ (enum only) | Inspect | Inspect |
| `PrepareFailure` | ❌ (enum only) | Inspect | Inspect |

#### 1.8.5.3 Original matrix (REPAIR_PLAN2-dash.md design)

| BuildError | classification (dash design) | retry 方針 (dash design) | 実装状況 |
|---|---|---|---|
| `ResourceUnavailable` | **一時的** | Building → DurablePending（retry） | ✅ build(:461) で返り、durable recovery は retry（:1015-1048） |
| `WarmupFailed` | **一時的** | Building → DurablePending（retry） | ✅ validateWarmup(:476) で返り、main task は `shouldRetryWarmupFailure` で retryable 判定 |
| `InternalError` | **永続** | Discarded（state clear） | ✅ build(:466) で返り、caller は `continue` |
| `MKLFailure` | **一時的** (設計済み) | Building → DurablePending（retry） | ❌ enum のみ。build() では InternalError に丸められる |
| `ConvolverFailure` | **一時的** (2026-08-10 確定) | Building → DurablePending（retry） | ❌ enum のみ。build() では InternalError に丸められる |
| `PrepareFailure` | **一時的** | Building → DurablePending（retry） | ❌ enum のみ。build() では InternalError に丸められる |

**設計整合の問題**: REPAIR_PLAN2-dash.md:1602-1613 は MKLFailure/ConvolverFailure/PrepareFailure を**「一時的（retry）」** として設計しているが、実コードではこれらは**永続の InternalError に丸められており retry 対象外**になっている。保険分類として「一時的」を想定しているのに、実際の failure パスでは「永続（Discarded）」になっている**意味論の不一致**が存在する。将来 wiring する際はこの不一致を解消する必要がある。

### 1.8.6 コンパイル時安全機構の欠如

| メカニズム | 状態 | 備考 |
|---|---|---|
| `-Wswitch` / `-Wswitch-enum` | ❌ 未指定 | CMakeLists.txt:1264 `/W4` は MSVC C4061（enum switch 未網羅）を含むが、gcc/clang は `-Wall -Wextra`（:1327）のみで `-Wswitch-enum` は未指定 |
| `static_assert` enum↔toString 網羅性 | ❌ なし | RuntimeBuilder.cpp に static_assert は 0 件 |
| clang-tidy `enum-ast-visibility` / `switch-default` | ❌ OFF | CMakeLists.txt:40-57: `CONVOPEQ_ENABLE_CLANG_TIDY OFF`。CI 環境（`CONVO_CI_BUILD`）では ON だが、ローカル開発では無効 |
| `return "Unknown"` フォールバック | ⚠️ あるが `default:` なし | switch の外にフォールバック return はあるが、`default:` ラベルはない。新規 enum 値追加時、コンパイラは C4061 で警告する可能性があるが静的解析ツール（clang-tidy）未設定の環境では静静に "Unknown" が返る |
| **テストによる enum↔toString 一致検証** | ❌ **なし** | **すべてのテストファイルに BuildError/toString 参照ゼロ**（AiDex 検索: 0 ヒット）。ISRSemanticValidationTests.cpp（941行）は ISR closure/payload セマンティクスのみをテストし、BuildError は未テスト |

**work32 監査の不正確さ**（implementation_audit_report.md:144）:
> `BuildError` enum 拡張 | 既存の switch-case に `default` があるため互換性を維持。| 低

この主張は**不正確**。toString の switch には `default:` ラベルがない（`switch` の後に `return "Unknown"` があるだけ）。しかし実質的には「未処理値が暗黙に `Unknown` になる」ため、**ランタイム安全性は確保されているが、**静的検出はない。

### 1.8.7 現状 / 課題 / 動機

**現状**
- `MKLFailure` / `ConvolverFailure` / `PrepareFailure` は enum 定義 + `toString` のみで build パス未使用（保険分類）
- `build()` が実際に返すのは `InvalidInput`（:435）/ `ResourceUnavailable`（:461）/ `InternalError`（:466）/ `None`（implicit success）、`validateWarmup()` が返すのは `WarmupFailed`（:476）/ `None`（:478）
- work32 C-2 Step 3（catch 拡張）は**未実装**。`mkl::exception` は存在しない。MKL/IPP はステータスコードベースで例外をスローしない
- caller は `runtime == nullptr` で失敗検知。`error` フィードはログ出力のみ
- buildRuntimePublishWorld（メイン publish パス）は BuildError を全く使用しない

**課題 / 動機**
- 将来 convolver build / prepare 失敗を**適切に分類**（transient/DurablePending）して retry できるようにする
- 現状では `InternalError`（永続/Discarded）に丸められており、一時的な convolver/prepare 問題で正当な Recovery が喪失される（REPAIR_PLAN2-dash.md:1614 の「安全側は一時的」設計方針と矛盾）
- `toString` の enum 網羅性に対する**静的検証ギャップ**（テストなし・static_assert なし）

### 1.8.8 設計方針 — 将来 wiring の正しいアプローチ

#### 1.8.8.1 Catch-based アプローチは不適（work32 Step 3 は破棄すべき）

work32 plan:803-835 の Step 3 は `catch(const mkl::exception&) → MKLFailure` を提案するが、これは**実装不可能**：
1. `mkl::exception` は oneMKL C++ wrappers にのみ存在し、本プロジェクトは **C API**（`mkl.h`, `mkl_dfti.h`）を使用
2. MKL / IPP は **ステータスコード** (`IppStatus`, `DFTI_STATUS`) でエラーを返す。C++ 例外をスローしない
3. `ConvolverProcessor::prepareToPlay()` は `jassert` + ステータスコードで処理

#### 1.8.8.2 正しい wiring アプローチ — Return-code チェック

`build()` の try ブロック内で呼び出される各操作の**戻り値/ステータスをチェック**し、対応する BuildError を設定する:

```cpp
// 将来の build() 内 (案)
BuildResult result {};

if (in.sampleRate <= 0.0 || in.blockSize <= 0) {
    result.error = BuildError::InvalidInput;
    return result;
}

try {
    runtime = convo::aligned_make_unique<AudioEngine::DSPCore>();
    runtime->convolverRt().setVisualizationEnabled(false);
    runtime->convolverRt().applyBuildSnapshot(convolverBuildSnapshot);
    runtime->convolverRt().transferIRStateFrom(engine.getConvolverProcessor());

    // ★ 将来: prepare() の戻り値/ステータスをチェック
    // prepare() は現在 void だが、将来的に Status を返すように変更、
    // または prepare() 内で ConvolverFailure/PrepareFailure/MKLFailure を
    // internal ステータスとして設定し、build() に伝播する
    auto prepareStatus = runtime->prepareWithStatus(...);
    if (prepareStatus == PrepareStatus::ConvolverInitFailed) {
        result.error = BuildError::ConvolverFailure;
        return result;
    }
    if (prepareStatus == PrepareStatus::MKLFailed) {
        result.error = BuildError::MKLFailure;
        return result;
    }
    if (prepareStatus == PrepareStatus::ResourceExhausted) {
        result.error = BuildError::PrepareFailure;
        return result;
    }

    result.runtime = runtime.release();
    result.prepared = true;
    return result;
}
catch (const std::bad_alloc&) {
    result.error = BuildError::ResourceUnavailable;
    return result;
}
catch (...) {
    result.error = BuildError::InternalError;
    return result;
}
```

**前提条件**: `DSPCore::prepare()` をステータスコードを返すバージョンに変更する必要がある（現状は `void`）。また、`applyBuildSnapshot`/`transferIRStateFrom` にステータスチェックを追加する。

#### 1.8.8.3 Retry ポリシーの整合

将来 wiring する際は REPAIR_PLAN2-dash.md:1602-1613 の分類に合わせる:
- **Transient**（Building → DurablePending, retry）: `ResourceUnavailable`, `MKLFailure`, `ConvolverFailure`, `PrepareFailure`, `WarmupFailed`
- **Permanent**（Discarded）: `InvalidInput`, `InternalError`

現在の caller パターン（`runtime == nullptr` チェック）は**変更不要** — insurance 分類を wiring する場合でも、`prepare()`/`convolverRt()` 操作が失敗した場合は `runtime` が nullptr にならない可能性がある（部分的に初期化された DSPCore）。この場合は `result.error != None` もチェックする必要がある:

```cpp
// 将来の caller (案) — error チェックを追加
if (buildResult.runtime == nullptr || buildResult.error != BuildError::None) {
    diagLog("...build failed error=" + toString(buildResult.error));
    if (buildResult.error == BuildError::InvalidInput ||
        buildResult.error == BuildError::InternalError) {
        // permanent → discard
        continue;  // or settle(false)
    }
    // transient → retry (DurablePending)
    runtimePublicationBridge_.settlePendingRecoveryAdmission(true);
    continue;
}
```

### 1.8.9 実装手順

**フェーズ 1（今すぐ可能 — safety hardening）**
1. `toString()` に `default:` ケースを追加し、未知の enum 値を「検出可能」にする（`return "Unknown"` を `default: return "Unknown-!!";` などの目印付きに変更）
2. `static_assert` による enum↔toString 網羅性のコンパイル時検証を追加（詳細は 1.8.10）
3. **テスト追加**: `BuildErrorToStringTests.cpp` — 全 8 enum 値の `toString()` 戻り値を検証

**フェーズ 1（今すぐ可能 — safety hardening）**
1. `kBuildErrorDescriptors` constexpr table を RuntimeBuilder.h に追加 (name + failureClass + disposition + telemetryCode) — single source of truth
2. `toString` を constexpr table にリダイレクト（switch-case は廃止、table lookup に統一）
3. `static_assert(kBuildErrorDescriptors.size() == static_cast<int>(BuildError::InternalError) + 1)` を追加
4. `default: return "Unknown-!!";` を toString に追加（runtime safety net）
5. `BuildErrorClassificationTests.cpp` を追加 — constexpr table と enum の一致を検証
6. `-Wswitch-enum` を CMakeLists.txt に追加（GCC/Clang: `:1327`, MSVC: `/W4 C4061`）

**フェーズ 2（将来 wiring — insurance 分類の活用）**
7. `BuildFailureClass` / `RetryDisposition` enum を `RuntimeBuilder.h` に追加 (1.8.5.2) — `ContextDependent` variant for WarmupFailed
8. `DSPCore::prepare()` の戻り値を `void` → ステータス型（`PrepareResult` with status + subsystem + retryability）
9. `ConvolverProcessor::applyBuildSnapshot()` と `transferIRStateFrom()` にステータス/エラー伝播を追加
10. build() の try ブロック内でステータスチェック → `MKLFailure`/`ConvolverFailure`/`PrepareFailure` をセット
11. `BuildError → BuildFailureClass → RetryDisposition` 変換関数を実装
12. caller（RebuildDispatch.cpp:941/1015/1087）の failure check を `runtime == nullptr || error != None` に強化
13. WarmupFailed call-site ごとに context-dependent retry (RebuildDispatch.cpp:1146 `shouldRetryWarmupFailure` / :1022 `settlePendingRecoveryAdmission(true)` / :943 `continue`)
14. `kMaxRecoveryConsecutiveFailures=4` (RebuildDispatch.cpp:1003) による infinite retry loop prevention を維持
15. `buildErrorCount_` telemetry カウンタを AudioEngine に追加（work32 Step 4）

### 1.8.10 static_assert による enum↔toString 網羅性検証

#### 1.8.10.1 現在のアプローチ（限定的 — enum ordering のみ検証）

```cpp
// ★ C-2: toString の enum 網羅性をコンパイル時検証
// 新規 enum 値を追加した場合、この static_assert がコンパイルエラーになる。
// （switch は default を持たないため、新規値は "Unknown" に落ちる静的デグロードを防止）
#define BUILD_ERROR_ENUM_COUNT 8
static_assert(static_cast<int>(BuildError::None) + 1 == 1, "");  // 0
static_assert(static_cast<int>(BuildError::InvalidInput) + 1 == 2, "");
static_assert(static_cast<int>(BuildError::ResourceUnavailable) + 1 == 3, "");
static_assert(static_cast<int>(BuildError::MKLFailure) + 1 == 4, "");
static_assert(static_cast<int>(BuildError::ConvolverFailure) + 1 == 5, "");
static_assert(static_cast<int>(BuildError::PrepareFailure) + 1 == 6, "");
static_assert(static_cast<int>(BuildError::WarmupFailed) + 1 == 7, "");
static_assert(static_cast<int>(BuildError::InternalError) + 1 == 8, "");
static_assert(static_cast<int>(BuildError::InternalError) == BUILD_ERROR_ENUM_COUNT - 1,
              "BuildError enum count mismatch — update toString switch");
```

#### 1.8.10.2 レビュー指摘: static_assert は `toString` の網羅性を保証しない

**⚠️ 判定: static_assert only verifies enum sequential numbering (0..7), NOT `toString` exhaustiveness.**

The current `static_assert` chain only checks that `BuildError` values are `0, 1, 2, ..., 7` in order. If a `case` arm is removed from `toString`'s switch (e.g. `case BuildError::MKLFailure: return "MKLFailure";` is deleted), but the enum value still exists, **all `static_assert` checks still pass** — `toString` silently falls through to `return "Unknown"`.

**Code verification confirms:** `RuntimeBuilder.cpp:53-79` — `toString` has **no `default:` case** in its switch; it falls through to `return "Unknown"` (line 78). This is exactly the vulnerability.

#### 1.8.10.3 Recommended stronger approach — constexpr descriptor table (preferred)

⚠️ **レビュー指摘: X-macroは可能だが、constexpr descriptor tableの方がConvoPeqのsemantic architectureにより適合する**

> `enum + toString + classification + telemetry` を4つに分けて保守すると、追加maintenance surfaceができる。
> `BuildError` を `semantic architecture` (ISRRuntimeSemanticSchema.h の AuthorityClass/VisibilityClass など) と統合することを推奨。

**Recommended: Single constexpr descriptor table:**

```cpp
struct BuildErrorDescriptor {
    const char* name;
    BuildFailureClass failureClass;
    RetryDisposition disposition;
    const char* telemetryCode;  // work32 Step 4
};

// Single source of truth — enum, toString, classification, telemetry が全てここから生成
inline constexpr std::array<BuildErrorDescriptor, 8> kBuildErrorDescriptors = {{
    {"None",              BuildFailureClass::None,          RetryDisposition::None,      "B000"},
    {"InvalidInput",      BuildFailureClass::InvalidConfiguration, RetryDisposition::Discard, "B001"},
    {"ResourceUnavailable",BuildFailureClass::ResourceTransient, RetryDisposition::Retry,   "B002"},
    {"MKLFailure",        BuildFailureClass::Inspect,       RetryDisposition::Inspect,   "B003"},
    {"ConvolverFailure",  BuildFailureClass::Inspect,       RetryDisposition::Inspect,   "B004"},
    {"PrepareFailure",    BuildFailureClass::Inspect,       RetryDisposition::Inspect,   "B005"},
    {"WarmupFailed",      BuildFailureClass::DSPTransient,  RetryDisposition::ContextDependent, "B006"},
    {"InternalError",     BuildFailureClass::Internal,      RetryDisposition::Discard,   "B007"},
}};

// Compile-time exhaustiveness: enum count と array size を照合
static_assert(kBuildErrorDescriptors.size() == 8, "Update descriptors when enum changes");
static_assert(static_cast<int>(BuildError::InternalError) + 1 == 8, "Enum ordering check");
// → enum追加 → count mismatch → compile error
// → toString case追加忘れ → kBuildErrorDescriptors未更新 → runtime test failure
```

**利点:**
- 追加: `failureClass` + `disposition` + `telemetryCode` が enum 変更時自動追従
- `-Wswitch-enum` は補完的（enum switch 未網羅警告）
- runtime test: `kBuildErrorDescriptors` に全 enum 値が存在することを検証

### 1.8.11 Test 設計

**現状**: テストファイルに `BuildError` または `toString` の参照は**ゼロ**（AiDex 検索: 0 ヒット）。ISRSemanticValidationTests.cpp（941行）は ISR closure/payload セマンティクスのみをテスト。

**提案するテストファイル**: `src/tests/BuildErrorClassificationTests.cpp`

```cpp
#include <catch2/catch.hpp>  // or JUCE UnitTest
#include "RuntimeBuilder.h"

TEST_CASE("BuildError to string and classification") {
    using namespace convo;
    for (int i = 0; i < static_cast<int>(BuildError::InternalError) + 1; ++i) {
        const auto error = static_cast<BuildError>(i);
        const auto& desc = kBuildErrorDescriptors[i];
        REQUIRE(std::string(toString(error)) == desc.name);
        REQUIRE(desc.failureClass != BuildFailureClass::None || error == BuildError::None);
    }
}
```

**Acceptance Criteria**
- `BuildError::toString` が全 8 enum 値に対応済み ✅（実コード照合済み）
- `toString` switch に `default: return "Unknown-!!";` を追加し、未検知 enum 値をランタイムで目視検出可能にする（⚠️ `static_assert` alone does NOT guard this — see 1.8.10.2）
- `kBuildErrorDescriptors` constexpr table を導入し、enum ↔ name ↔ classification ↔ retryability ↔ telemetry の single source of truth とする (1.8.10.3) ✅
- `static_assert(kBuildErrorDescriptors.size() == static_cast<int>(BuildError::InternalError) + 1)` でコンパイル時網羅性検証 ✅
- `BuildErrorClassificationTests.cpp` で constexpr table と enum の一致をテストする ✅
- **1.8.5.2**: `BuildFailureClass` / `RetryDisposition` separation を実装 — MKLFailure/ConvolverFailure/PrepareFailure は `Inspect` として分類 ✅
- **1.8.5.3**: `WarmupFailed` は `RetryDisposition::ContextDependent` — call-site が追加 context (isLoadingIR)を渡す ✅
- フェーズ2実装時: `PrepareResult` (status + subsystem + retryability) ヒエラルキーを導入 ✅
- 既存 ctest 28/28 PASS（影響なし）

**影響範囲 / リスク**
- **フェーズ 1**: RuntimeBuilder.h/cpp (kBuildErrorDescriptors + static_assert), 新規テストファイル — 影響: なし。リスク: **低**
- **フェーズ 2**: RuntimeBuilder.h/cpp (BuildFailureClass + RetryDisposition + PrepareResult), AudioEngine.h (prepare ステータス), AudioEngine.Processing.DSPCoreLifecycle.cpp (prepare ステータス返却), ConvolverProcessor.h/cpp (ステータス伝播), RebuildDispatch.cpp (caller failure check 強化 + WarmupFailed context-dependent retry), AudioEngine.h (buildErrorCount_ 追加) — 影響: **中**（error handling パスの変更）。リスク: **中**（retry ポリシーの不整合による無限リトライ / 回復不能エラーの永久リトライ）。⚠️ **MKLFailure/ConvolverFailure/PrepareFailure を一律 retry すると、永続的な failure (corrupted IR / invalid config) が無限リトライループを引き起こす** — 1.8.5.2 の classification 分離が必須。**kMaxRecoveryConsecutiveFailures=4 (RebuildDispatch.cpp:1003)** による無限リトライ防止が既存 — 将来 wiring 時は維持すること。

### 1.8.12 未解決課題 / 調査完了事項

| # | 課題 | 調査方法 | ステータス | 調査結果 |
|---|---|---|---|---|
| 1 | `DSPCore::prepare()` 内部の各サブシステムの失敗モードを列挙 | `AudioEngine.Processing.DSPCoreLifecycle.cpp:72` を解析 | **✅ 調査完了** | `prepare()` は**void** を返す (h:78)。内部で呼び出す全サブシステムも void: `ramp.prepare()`、`oversampling.prepare()`、`convolverState->prepare()`、`eqState->prepare()`、`dcBlockers().init()`、`noiseShaper prepare`、`outputFilter.prepare()`、`truePeakDetector.prepare()`、`loudnessMeter.prepare()`、`peakLimiter.prepare()`。**いずれも失敗を返さない** — allocation 失敗は `std::bad_alloc` 例外 (catch で `ResourceUnavailable` に変換済み at `RuntimeBuilder.cpp:459-462`)、他のサブシステムの内部失敗はログ出力のみ (juce::Logger) または黙殻。**ステータス型導入にはすべてのサブシステムの戻り値変更が必要**。 |
| 2 | `applyBuildSnapshot()` / `transferIRStateFrom()` の失敗検知 | `ConvolverProcessor.h:477,1132` + `ConvolverProcessor.StateAndUI.cpp:271` を解析 | **✅ 調査完了** | `applyBuildSnapshot()` — void、atomic publish のみ (ScopedLock + publishAtomic)。例外なし。`transferIRStateFrom()` — void、IR AudioBuffer コピー。成功/失敗を `juce::Logger::writeToLog` でログ出力 (`[CONV_IR] transferIRStateFrom: IR transferred/failed/no IR data`) が、戻り値なし。**いずれもステータスチェック不能 — 将来的な status-code wiring には戻り値追加が必要**。 |
| 3 | MKL/IPP の使用箇所とステータスコード | `MKLNonUniformConvolver.h` + `ConvolverProcessor.Lifecycle.cpp:276-279` を解析 | **✅ 調査完了** | **⚠️ MKL は実際には使用されていない** — h:5 コメント「v2.0 FFT backend を MKL DFTI → Intel IPP に換装」。`MKLNonUniformConvolver.cpp/h` は `#include <mkl.h>` (h:47) を持つが、実際の FFT 計算は **Intel IPP** (`IppsFFTSpec_R_64f`, `ippsFFTFwd_RToCCS_64f`, `ippsFFTInv_CCSToR_64f`)。IPP は `IppStatus` コードを返す (ステータスコードベース、C++ 例外なし)。`newConv->init()` (L276-279) は bool を返す — `false` 時は log + existing engine を保持する fallback (L288-295)。**したがって `MKLFailure` は現在どのコードパスからも生成されない** (build() は `InvalidInput`/`ResourceUnavailable`/`InternalError`/`None` のみ返す)。 |
| 4 | `publishRuntimeProcessSnapshot()` の戻り値 | `ConvolverProcessor.h:1027` を解析 | **✅ 調査完了** | `void` を返す。`runtimeProcessSnapshots[next]` への書き込み (`pendingOverride` からの ScopedLock + atomics) は全て成功する (allocation なし)。**エラー伝播パスなし** — この関数は failure を返しません。 |
| 5 | `buildRuntimePublishWorld` の failure handling | `RuntimeBuilder.cpp:178-426` を解析 | **✅ 調査完了** | `buildRuntimePublishWorld` は `BuildError` を**使用しない** — 内部で try/catch し `worldOwner` を返す (失敗時 nullptr)。**設計の意図: `build()` (DSPCore construction) と `buildRuntimePublishWorld()` (World assembly) は責務分離**。`build()` は DSPCore + IR + prepare (BuildError で分類可能)。`buildRuntimePublishWorld` は World の構成・トポロジ — ここでの failure は構造的問題 (internal) なので InternalError で統一されている。**将来的な分離は不要 — 現設計で正しい**。 |
 | 6 | `build()` の `diagLog` / diagnostics guard | `RuntimeBuilder.cpp:428-469` + `RebuildDispatch.cpp` を解析 | **✅ 誕張完了** | `build()` は **`#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` ガードなし**。try/catch (`std::bad_alloc` → ResourceUnavailable, `...` → InternalError) は **常に有効** (production code)。`diagLog` は `RebuildDispatch.cpp:1073/1093` で使用されるが、**build() 自体には診断 guard なし** — build() の `catch` は本番環境でもアクティブ。 |
| 7 | **⚠️ retry storm / backoff analysis (2026-08-11 review Amendment 1.8)** | `RebuildDispatch.cpp:990-1052` (durable recovery retry loop) | **✅ 調査完了** | `kMaxRecoveryConsecutiveFailures = 4`（:1003）でbounded。しかし**exponential backoff なし** — `continue` で即座にretry。IR loading中の warm-up failure は `settlePendingRecoveryAdmission(true)` で DurablePending に戻り、loop内で即座に再`take`される。**System-level RT safety impact**: Recovery retry storm が Builder thread を飽和させ、新規 world publish が遅延し publication latency が増加する可能性あり（NonRT → RT indirect). No retry-count reset on success within loop (counterはloop scope)。 |

---

## 1.8.9 RetryDisposition 実装時の Acceptance Criteria（2026-08-11 review Amendment 1.8）

**⚠️ `RetryDisposition::Retry` を「retry forever」にしないこと。** ISR観点では:

| Concern | Acceptance Criterion |
|---|---|
| retry count | `RetryDisposition::Retry` は max retry count（configurable, default 3）を伴う |
| backoff | retry間に exponential backoff（min 1ms, max 100ms）を挿入する |
| generation validity | retry対象のgenerationが無効（superseded）なら即座にdiscard |
| shutdown state | `isShutdownInProgress()` をretry loopのpredicateに含める |
| system-level RT safety | retry storm を telemetry で observability（retryCount / retryLatency / retryStormDetected） |
| ISR safety | retry decision はすべて NonRT スレッド（CoordinatorLoop/RebuildThread）で行われる — RT からは呼ばれない |

---

## 1.9 初回 publish 前 quarantine の無駄な起床の最適化

**現状（実コード照合）**
- `submitRecoveryIntent` (AudioEngine.h:4285-4297) は **無条件** に `recoveryPending = true` + `rebuildCV.notify_all()` を実行
- `submitRecoveryRequest` (ISRRuntimePublicationCoordinator.cpp:721-781) は reservation + push または durable admission への fallback を行う
- RebuildThread wake predicate (RebuildDispatch.cpp:828-833): `hasPendingTask || publishRetryReady || recoveryPending || rebuildThreadShouldExit`
- `recoveryPending` は `AudioEngine.h` メンバ変数（mutex + CV で保護） — `submitRecoveryIntent` と `rebuildThreadLoop` の間の通信チャネル
- 1.9 の最適化: `submitRecoveryIntent` が**実際に recovery が存在する場合のみ** `recoveryPending` を set + notify する

**⚠️ 現実の wake パス:**

```text
QuarantineIntentHandler
  → submitRecoveryIntent(handle, buildSource)
    → submitRecoveryRequest(handle, buildSource)
      → reservation fetchAdd + push to recoveryIntentQueue_
      └─ push success: recovery is in transport
      └─ push fail (queue full): rollback + DurablePending state set
    → recoveryPending = true + notify_all()  ← 無条件
```

**wake の3条件** — 実際は `submitRecoveryRequest` が2つのパスを区別している:
- **Transport path**: `recoveryIntentQueue_.push()` success → queue has recoverable item
- **Durable path**: push fail → `recoveryAdmissionPending_ = true` + `pendingRecoveryAdmission_.pending = true`

1.9 の最適化は、`submitRecoveryRequest` の結果に基づいて `recoveryPending` を set するかどうかを決定すること:
- push success → `recoveryPending = true` (transport recovery exists)
- push fail → `recoveryPending = true` (durable recovery exists)  
- shutdown gate hit → `recoveryPending = false` (recovery discarded, no wake needed)

**課題 / 動機**
- 機能上の問題はないが、**無意味な RebuildThread 起床**による CPU 無駄（Phase 5 最適化候補）

**設計方針**
- `submitRecoveryIntent` が**実際に recovery が存在する場合のみ** `recoveryPending` を set + notify するように変更
- **wake 判定条件**（将来的な coalesce 実装時）:
  - `hasTransportRecovery` (queue/reservation に pending あり)
  - **OR** `hasDurableRecovery` (quarantine/fallback に永続的な退避あり)
  - **OR** `hasPendingRecoveryAdmission` (DurablePending state)
  > ⚠️ `hasDurableRecovery` だけをチェックすると、transport-level retry が pending なのに wake されない可能性がある。現行コードでは `recoveryPending` を set するたびに notify が走るため、現時点では過剰起床が問題にならない。最適化時は3つの条件の論理和を判定すること。

#### 1.9.1 ⚠️ レビュー指摘: 3条件では「初回 publish 前の無意味な wake」を解決できない

**問題**: 3条件 (`hasTransportRecovery || hasDurableRecovery || hasPendingRecoveryAdmission`) は、quarantineが検出された状況では**常にtrue**になる。

初回 publish 前の quarantine は、`QuarantineIntentHandler` が `submitRecoveryIntent` を呼ぶ。この時点で:
- `submitRecoveryRequest` は queue push に成功する（queue は空なので） → transport recovery あり
- 3条件は true → `recoveryPending = true` + notify

→ **3条件の論理和だけでは、初回 publish 前の quarantine による無意味な wake を除去できない**。

**必要なのは `RecoveryAdmission` ポリシー**:

```text
Recovery requested
       ↓
hasAuthoritativePublishedRuntime()
       ├─ false → discard/absorb (initial quarantine — no authoritative runtime yet)
       └─ true → transport/durable
```

このポリシーは、

```text
hasAuthoritativePublishedRuntime
    = RuntimeStore::current != nullptr
      AND current->publication.epoch > 0
```

あるいは、`AudioEngine.h:3556` の `observePublishedWorld() != nullptr`（= `worldAuthority_.observePublishedWorld()`）をチェックすることで判定可能。⚠️ **訂正（§補足2.5.3）**: `runtimePublishWorld_` は存在しない。代わりに `RuntimeWorldAuthority::observePublishedWorld()` を使用。

**実装方針**:
- `submitRecoveryRequest` に `RecoveryAdmissionPolicy` パラメータを追加
- `NoAuthoritativeRuntime` モード: quarantine detection を silent absorb (log only, no wake)
- `HasAuthoritativeRuntime` モード: current 3-path (transport/durable/shutdown)

この最適化は**quarantine semantic の問題**であり、単なる wake 最適化ではない。

**実装手順**
1. `submitRecoveryIntent` の recoveryPending set 条件を精査（transport/durable/pending admission の3条件の論理和）
2. 全3条件が false の場合の early return を追加
3. テスト: 無意味な起床の削減 / 機能回帰なし（既存 Recovery テスト）

**Acceptance Criteria**
- 初回 publish 前の無意味な起床が発生しない（recoveryPending の誤 set なし）
- ⚠️ **lost wake proof**: `predicate = transport || durable || pending` とし、state observation + state transition + notify の間で lost wake が起きないことを証明する
- shutdown 時に `AdmissionClosed` と競合しないことを証明する
- 既存 Recovery / soak テスト全 PASS

**影響範囲 / リスク**
- 影響: `submitRecoveryIntent`、RebuildDispatch
- リスク: **低**（最適化のみ・機能変更なし）

---

# 2. 一部実装（保留・部分対応）

## 2.1 R4: retire 順序逆転の完全解消

**現状（実コード照合）**
- **runtime 経路は対応済み**: `retireDSPHandleForRuntime` / `retireByHandle` は `requestReclaimHandle` 経由に一本化
- `shutdownReclaim` は **X3-R4 Phase 7 で完全削除済み**（AC-R4-1 call sites==0 / AC-R4-2 symbol absent）
- `reclaim(ReclaimMode, ...)` は RuntimeEBR / ShutdownQuiescent の2モードで一本化（ReclaimAuthority）

**残る課題**
- **retire 順序逆転**（後から発行された retire が先に処理される可能性）の**完全解消は保留**
- ただし quarantine fallback で **UAF / リークは排除済み**

**設計方針**
- R4 詳細設計（REPAIR_PLAN2-dash.md §6.3 末尾・Phase 0-7）に基づき、retire の epoch 順序保証（FIFO）を強化

⚠️ **レビュー指摘: epoch safety と FIFO を混同しない**

```text
Epoch safety ≠ FIFO
```

- `retire A @ epoch 10, retire B @ epoch 11` で `B → reclaim, A → reclaim` が起きても、両方 epoch-safe なら UAF ではない
- `retire A @ epoch 10, retire B @ epoch 11` を FIFO にしただけでは、`reader @ epoch 10` 残存時の安全性は保証されない
- **FIFO order = optimization / determinism / telemetry の property** — memory safety そのものではない

現行コードは `retireDSPHandleForRuntime` / `retireByHandle` を `requestReclaimHandle` 経由に一本化し、epoch unsafe 時は `pendingReclaimHandles_` に保持して `drainDeferredRetireQueues()` で再試行する構造になっている。`requestReclaim()` 内部でも epoch を再確認する TOCTOU対策がある。

**TOCTOU 対策 (AudioEngine.Retire.cpp:41-117):**
```text
drainDeferredRetireQueues
    ↓
pendingReclaimHandles_ から mutex で batch 抽出
    ↓
各 handle: isRetired() チェック → epoch 再確認
    ├─ retireEpoch < minReaderEpoch → requestReclaim → success/false
    │   └─ false: pendingReclaimHandles_ へ再登録 (TOCTOU 対策)
    └─ retireEpoch >= minReaderEpoch → pendingReclaimHandles_ へ再登録
```

さらに、`Quarantined` 状態から `Retired` への誤上書きを防ぐため、`isRetired()` チェックが行われる (AudioEngine.Retire.cpp:75: `if (dspHandleRuntime_.isRetired(handle))`)。

したがって、**「retire順序逆転 = UAF」** とは評価すべきではない。dash2の「UAF/リークは既に排除済み」という位置付けの方が適切。

**実装手順**
1. retire の epoch 順序保証（FIFO）を強化（enqueue 順と epoch 順の整合）
2. `drainDeferredRetireQueues` の順序検証（pendingReclaimHandles_ の再試行機構と整合）
3. テスト追加: retire 順序逆転の回帰テスト（AC-R4-T1〜T7 を拡張）

**Acceptance Criteria**
- retire が epoch 順に処理される（順序逆転なし）
- quarantine fallback でリークなし（既存）
- **AC-R4-1〜10 全充足**（shutdownReclaim 0 / ReclaimAuthority 一本化 / epoch unsafe は pendingReclaimHandles_ に残る / Faulted で pending を clear しない）
- **AC-ISR-1**: Audio Thread は retire ordering API を呼び出さない
- epoch safety と FIFO の分離をテスト: `retire order reversal` だけで UAF が起きないことを検証

**影響範囲 / リスク**
- 影響: `ISRRetireRouter`、`AudioEngine.Retire.cpp`（drainDeferredRetireQueues）、`ReleaseResources.cpp`
- リスク: **中**（retire 順序の変更は epoch 安全性に影響。runtime 経路は既に対応済みのため、残余は順序保証の強化）

---

## 2.2 shutdown lifetime contract の明文化

**現状**
- REPAIR_PLAN2-dash.md §4（:837-846）で「shutdown lifetime contract の明文化」を**将来タスクとして記録**
- R4 詳細設計で **ShutdownQuiescenceProof**（admissionClosed / producersJoined / coordinatorStopped / builderStopped / audioStopped / readerRegistrationClosed / readersZero / epochSettled）が**答えとして確定済み**
- `readerRegistrationClosed`（EpochDomain.h）は実装済み（X3）・`reclaim(ShutdownQuiescent)` の precondition は実装済み
- **⚠️ 2026-08-11 追加**: `ShutdownQuiescenceProof` / `ReclaimPermit` オブジェクトは**現在コードに存在しない**。現行は `reclaim(..., bool readerRegistrationClosed=false)` のboolパラメータで証明を受け取っている（ISRRuntimePublicationCoordinator.h:373-377）。3つのproduction call site（ReleaseResources.cpp:423,433 / AudioEngine.h:2032）が全て `m_epochDomain.readerRegistrationClosed()` をboolで渡している — callerが `true` を偽って渡す可能性あり（§5.19/§5.12 詳細調査確認済み）。

**課題 / 動機**
- shutdown 側の「本当に reader が存在しないこと」の**保証をコード上の契約として明文化**
- AC-X3-11〜18 を満たす（ReclaimPermit の生成を ShutdownRuntime のみに限定等）

**設計方針**
```
ShutdownQuiescenceProof を独立オブジェクト化（valid() は完全条件）
ReclaimPermit は ShutdownRuntime のみが生成（caller cannot manufacture — AC-X3-11）
shutdownPhase >= Destroy は ShutdownQuiescent reclaim の証明にならない（AC-X3-12）

⚠️ 現行コード（ISRRuntimePublicationCoordinator.cpp:631）では
 `reclaim(ReclaimMode, ..., bool readerRegistrationClosed)` が
 bool を直接受け取っている。このパターンは形骸化の危陳 —
 caller が `true` を偽って渡すことが可能。
 → 将来的に ShutdownQuiescenceProof / ReclaimPermit オブジェクトを渡すべき。
```

**⚠️ レビュー指摘: ShutdownQuiescenceProof は bool の wrapper にしない**

> 各 bool を外から渡して `valid()` すると、proof が形骨化する。
> 理想は ShutdownRuntime が proof を生成する構造:
>
> ShutdownRuntime
>     ├── closes admission
>     ├── joins producers
>     ├── stops coordinator
>     ├── stops builder
>     ├── stops audio
>     ├── closes reader registration
>     ├── waits readers == 0
>     ├── settles epoch
>     └── postStopEnqueue == 0  ← ★ review §16 addition
>              ↓
>        ShutdownQuiescenceProof  (immutable, only ShutdownRuntime can create)
>              ↓
>        ReclaimPermit  (passed to reclaim(ShutdownQuiescent, permit))

#### 2.2.1 ⚠️ レビュー指摘: 現行コードの ShutdownQuiescent reclaim は `shutdownPhase >= Destroy` で判定（形骨化リスク）

**CacheMap::~CacheMap() (AudioEngine.h:2015-2047) — 実コード照合:**

```cpp
~CacheMap()
{
    if (consumeAtomic(owner->shutdownPhase, ...) >= ShutdownPhase::Destroy)
    {
        // Shutdown path: resolve → delete → reclaim(ShutdownQuiescent, ..., readerRegistrationClosed)
        const bool reclaimed = owner->runtimePublicationBridge_.reclaim(
            RuntimeIntentCoordinator::ReclaimMode::ShutdownQuiescent,
            entry.second, rt, *owner->m_retireRouter,
            owner->m_epochDomain.readerRegistrationClosed());
    }
    else {
        // Normal path: retire only
        owner->dspHandleRuntime_.retire(entry.second);
    }
}
```

**問題**: `shutdownPhase >= Destroy` は proof の代わりになっているが、これは**事後的なstate snapshot**であり、**証明ではない**。

`readerRegistrationClosed()` を個別に `reclaim()` に渡しているが、この bool は caller が偽って渡せる（現コードでも `true` を強制渡し可能）。

**→ 将来的に ShutdownQuiescenceProof / ReclaimPermit オブジェクトを渡すべき (AC-LIFE-2 / AC-2.2-PERMIT)**

#### 2.2.2 postStopEnqueue tracking (already implemented)

`ShutdownRuntime` はすでに `postStopEnqueue` を追跡している:

- `ISRShutdown.h:144`: `ShutdownResult.postStopEnqueueCount`
- `ISRShutdown.h:202`: `markPostStopEnqueue()`
- `ISRShutdown.h:223`: `sh6PostStopEnqueueCount_` atomic counter
- `AudioEngine.Commit.cpp:459`: `if (shutdownRuntime_.isShutdownInProgress()) shutdownRuntime_.markPostStopEnqueue();`

**→ `postStopEnqueue == 0` を ShutdownQuiescenceProof の条件に追加 (review §16)**

**実装手順**
1. `ShutdownQuiescenceProof` の `valid()` を完全条件（上記8項目）で実装
2. `ReclaimPermit` の生成を ShutdownRuntime のみに限定（friend class・AC-X3-11）
3. `registerReaderThread()` が閉鎖後 -1 を返す（INV-X3-4・AC-X3-15）— 実装済み
4. CacheMap / ReleaseResources は ShutdownRuntime の Permit 経由で reclaim（AC-X3-12）
5. テスト: AC-R4-T1〜T7 / AC-X3-11〜18

**Acceptance Criteria**
- **AC-X3-11〜18 全充足**（ReclaimPermit 生成限定 / phase 依存禁止 / readerRegistrationClosed 証明 / pendingReclaimHandles empty は producer join 後 / registerReaderThread 失敗 / 二重 retire 防止 / Audio Thread から不可 / physical delete と独立）
- **AC-LIFE-2** (review §22): `ShutdownQuiescenceProof` を持たない caller は `ShutdownQuiescent` reclaim を実行できない
- `isFullyDrained` が proof と整合

**影響範囲 / リスク**
- 影響: `ShutdownRuntime`、`RuntimeIntentCoordinator`、`CacheMap`（AudioEngine.h）、`ReleaseResources.cpp`
- リスク: **中**（shutdown 意味論の厳密化。**readerRegistrationClosed は実装済み**のため、残余は Proof オブジェクトの完全化と Permit 導入）

---

# 3. 実装順序（推奨）

```
## 推奨実装順序 (ISR Architecture Review §21 修正版 + 2026-08-11 review amendments)

**⚠️ Amendment (2026-08-11 review)**: Add Phase 0.5 — `mixSmoother.reset()` RT violation must be fixed BEFORE any other Phase, because it is an existing contract violation, not a future audit item (see §1.3 audit finding).

Phase 0（現行Invariant固定）: 現行コードの invariants 棚卸し・テスト固定
    ↓
**Phase 0.5（既知bug修正 — NEW）**: `mixSmoother.reset()` RT thread assertion violation の修正
  - `LinearRamp::reset()` を RT-safe に分割（`resetRT()` は `totalSteps` のみ atomic 更新）
  - または `totalSteps` を `std::atomic<int>` に変更し `reset()` の ASSERT を撤去
  - NonRT path（Lifecycle.cpp:375, 376, 489）は既に NonRT なので影響なし
    ↓
Phase A（Tier 1: lifetime safety）:  2.2（ShutdownQuiescenceProof + ReclaimPermit）
    ↓
Phase B（Tier 1: lifetime safety）:  1.4（isFullyDrained drain semantic 確定 + setter廃止）
    ↓
Phase C（Tier 1: lifetime safety）:  2.1（R4 retire 順序の完全解消 — epoch safety vs FIFO 分離）
    ↓
Phase D（Tier 3: infrastructure hardening）:  1.8（BuildError -> FailureClassification -> RetryDisposition）
    ↓
Phase E（Tier 2: recovery correctness）:  1.9（quarantine wake optimization — lost-wake proof）
    ↓
Phase F（Tier 2: recovery correctness）:  1.1（R1 MPSC 化 — INV-R1-1/INV-R1-2）
    ↓
Phase G（Tier 2: recovery correctness）:  1.7（currentWorld_ read-source singularization — AC-PUB-1）
    ↓
Phase H（Tier 3: infrastructure hardening）:  1.5/1.6（sparse completion + sequence tests）
    ↓
Phase I（Tier 2: recovery correctness）:  1.2（Recovery coalesce — LogicalRecoveryIdentity + state machine）

理由: shutdown proof -> drain semantic -> retire safety を先に固定しないまま、
MPSCやcoalesceを追加すると drain semantic -> shutdown proof の関係を再構築する必要がある。
**Phase 0.5 を追加した理由**: mixSmoother.reset() の RT assertion violation は既知のbugであり、
将来タスクではなく現行コードのcontract違反。これを放置するとdebugビルドでのjassert false positiveが
発生し、実際の問題の検知がマスキングされる。最重要のlifetime safety fix（Phase A: 2.2）の前提として、
まずこの既知bugを解消する必要がある。

---

## ⚠️ 2026-08-11 Review Amendments

### Amendment 1: §1.3 LinearRamp — reclassify as known bug, not future audit

**⚠️ mixSmoother.reset() at ConvolverProcessor.Runtime.cpp:360 is a KNOWN RT CONTRACT VIOLATION, not a future audit item.**

The LinearRamp audit (§1.3) discovered that `activeMixSmoother.reset()` is called from the audio processing thread (RT), but `LinearRamp::reset()` asserts `ASSERT_NON_RT_THREAD()` (DspNumericPolicy.h:333). This is an existing bug that must be fixed in Phase 0.5 before any other work.

**Severity**: Medium (debug-build jassert violation; `reset()` only modifies `totalSteps` which is read-only, so no data race in release builds, but contract violation is real)

### Amendment 2: §2.2 ShutdownQuiescenceProof — not yet implemented (design only)

**`ShutdownQuiescenceProof` and `ReclaimPermit` objects do NOT exist in the current codebase.** The current shutdown proof mechanism is:

```cpp
// Current approach (ISRRuntimePublicationCoordinator.h:373-377):
bool reclaim(ReclaimMode mode, const DSPHandle& handle, 
             class DSPHandleRuntime& handleRuntime,
             class ISRRetireRouter& router,
             bool readerRegistrationClosed = false) noexcept;
```

**3 production call sites** pass `readerRegistrationClosed` as a plain bool:
- `AudioEngine.Processing.ReleaseResources.cpp:423` — `m_epochDomain.readerRegistrationClosed()`
- `AudioEngine.Processing.ReleaseResources.cpp:433` — `m_epochDomain.readerRegistrationClosed()`
- `AudioEngine.h:2032` (CacheMap::~CacheMap) — `owner->m_epochDomain.readerRegistrationClosed()`

**⚠️ This is a bool parameter — caller can pass `true` without actual proof.** Phase A (2.2) must replace this with a `ShutdownQuiescenceProof` object whose generation is limited to `ShutdownRuntime` only (friend class / AC-X3-11).

### Amendment 3: §1.4 isFullyDrained — external setters must be eliminated before Tier 1

**⚠️ Threading.cpp:126-128 still calls 3 external setters.** This is a Phase 0.5 concern — these setters overwrite the authoritative counters that Phase A (ShutdownQuiescenceProof) and Phase B (drain semantic) depend on. If Phase A proceeds before setters are eliminated, the proof object validates against corrupted data.

### Amendment 4: §1.9 quarantine wake — semantic change requires recovery admission policy

The 1.9 optimization (suppress wake when no authoritative runtime exists) introduces a semantic change: recovery requests may be **silently absorbed** during initial quarantine pre-publish. This must be:
1. Observable via telemetry (quarantineAbsorptionCount_)
2. Proven not to lose valid recovery obligations (INV-X1-2: queue full ≠ Recovery lost)
3. The `hasAuthoritativePublishedRuntime()` check must use `observePublishedWorld() != nullptr` (AudioEngine.h:3556), NOT `runtimePublishWorld_` (which does not exist — §5.3 correction confirmed)

### Amendment 5: §2.2 — AudioEngine::ShutdownPhase vs isr::ShutdownPhase non-1:1 mapping

Confirmed: AudioEngine has 7-phase enum (AudioEngine.h:2533), ISR has 11-phase enum (ISRShutdown.h:25). CacheMap::~CacheMap (AudioEngine.h:2019) checks `AudioEngine::ShutdownPhase::Destroy` — NOT `isr::ShutdownPhase::ReclaimComplete`. The two enums represent different lifecycle layers (Engine vs Coordinator). Any code comparing across enums must bridge via explicit mapping.

### Amendment 6: Phase ordering — mixSmoother fix must precede Phase A

The reviewer (2026-08-11 ISR Architecture Review) explicitly recommends inserting Phase 0.5 before Phase A:

```
Phase 0: invariant freeze
Phase 0.5: mixSmoother.reset() RT violation fix  ← NEW
Phase A: ShutdownQuiescenceProof (2.2)
Phase B: isFullyDrained drain semantic (1.4)
...
```

**Rationale**: `mixSmoother.reset()` is an existing RT contract violation. If Phase A/B modify the shutdown/drain paths, the mixSmoother issue could mask or compound real lifetime safety problems. Fix known bugs first, then build new safety on a clean foundation.

- **各 Phase でビルド・ctest を通過**させる（rollback point 確保）
- Phase D 以降（1.8, 1.9, 1.1, 1.7, 1.5/1.6, 1.2）は**高い影響範囲**のため、実施前に現行 REPAIR_PLAN2-dash.md の該当設計（§5 R1/R4・§6.2 sparse・§6.4-X4 案2）を再確認する

---

## 4. クロスカッティング Acceptance Criteria

> ⚠️ **ISR安全 ≠ lifetime安全**。後者を壊せば結果的にISR安全ではなくなる。
> 以下の AC は全ての将来実装項目で守るべき最小契約。

### AC-ISR-1

```text
Audio Thread は新設APIを呼び出さない
```

対象: MPSC enqueue / coalesce / BuildError / ShutdownProof / retire ordering / wake optimization

### AC-LIFE-1

```text
isFullyDrained() == true
⇒
physical DSP objectがAudio Threadから到達不能
AND
all retire/quarantine/reclaim obligations == 0
```

### AC-LIFE-2

```text
ShutdownQuiescenceProofを持たないcallerは
ShutdownQuiescent reclaimを実行できない
```

### AC-PUB-1

```text
RuntimeStore::current と publication metadata が
同一PublicationIdentityを必ず共有する
```

(1.7 currentWorld_ 廃止の前提条件)

### AC-1.4-DRAIN

```text
isFullyDrained は 現行コードの16条件すべてを満たす
(queue sizes + reservations + durable + retire + deferred + reclaim + 
 quarantine + publication intent + physical residency)
```

### AC-1.8-RETRY

```text
MKLFailure / ConvolverFailure / PrepareFailure は
BuildFailureClass 経由で retryability を判定する
(一律 retry 禁止)
```

### AC-2.2-PERMIT

```text
ShutdownQuiescenceProof / ReclaimPermit は
ShutdownRuntime のみが生成可能
(caller cannot manufacture)
```

---

## 補足
- 本ドキュメント（dash2）の項目は、いずれも**設計・アーキテクチャ上の将来拡張・最適化**です。
- ⚠️ **「現在の ISR パイプラインの正しさには影響しない」は項目ごとには正しいが、全体としては無条件ではない**。特に `isFullyDrained`（§1.4）、shutdown proof（§2.2）、retire順序（§2.1）、BuildError retry（§1.8）を変更すると、ISR 本体ではなくても **ISR から参照される Runtime の lifetime safety を壊し得る**。
- 各項目の実装着手時は、本ドキュメントの Acceptance Criteria をテストとして固定してから実施してください（REPAIR_PLAN2-dash.md の Phase 0 方針と同一）。
- **2026-08-11 追加**: 1.8.12 の 6 課題すべて調査完了。重要な知見:
  - `DSPCore::prepare()` は void — 全サブシステム (ramp/convolver/eq/dither/oversampling/...) が void。ステータス型導入にはすべてのサブシステムの戻り値変更必要
  - `MKLFailure` は**どのコードパスからも生成されない** — MKL は使用されていない (IPP ステータスコードベース)
  - `build()` の try/catch は**production code で常に有効** — `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` ガードなし
  - caller (RebuildDispatch.cpp:1091) は `buildResult.runtime == nullptr` のみチェック — `buildResult.error` は log のみ
  - `publishRuntimeProcessSnapshot()` は void — ログ出力パスなし
  - `buildRuntimePublishWorld()` は BuildError を使用しない — build() (DSPCore construction) と World assembly は責務分離されている
- **2026-08-11 追加 (lifetime)**: 現行コードの `isFullyDrained` は16条件で判定済み（ISRRuntimePublicationCoordinator.cpp:500-525） — `reclaimInFlightCount_` と `publicationIntentResidencyCount_` を含む
- **2026-08-11 追加 (lifetime)**: `reclaim(ReclaimMode::ShutdownQuiescent)` は `readerRegistrationClosed && activeReaderCount()==0` を precondition として実装済み (ISRRuntimePublicationCoordinator.cpp:638-646)
- **2026-08-11 追加 (lifetime)**: `ShutdownPhase` enum (ISRShutdown.h:25) は**11段階** (Running, AudioStopped, ObserverDrained, RetireClosed, EpochSettled, ReclaimComplete, EmergencyDrain, VerifyDrained, TimedOut, Failed, ShutdownComplete) — ⚠️ AudioEngine::ShutdownPhase (AudioEngine.h:2533) は別に7段階 (Running, StopAcceptingWork, StopAudio, StopWorkers, ForceEpochAdvance, DrainRetire, Destroy) 存在。2つのenumは非1:1対応。CacheMap::~CacheMap は AudioEngine::ShutdownPhase::Destroy をチェック
- **2026-08-11 追加 (lifetime)**: `AudioEngine::isFullyDrained` (Threading.cpp:114-156) は Layer 1 (AudioEngine: deferred commit + pendingReclaim + 3 quarantine sources) + Layer 2 (Coordinator: 16 conditions) の 2層構造
- **2026-08-11 追加 (recovery)**: `submitRecoveryIntent` (AudioEngine.h:4285-4297) は**無条件**に `recoveryPending=true + notify_all()` — 1.9 の最適化対象。wake predicate: `hasPendingTask || publishRetryReady || recoveryPending || rebuildThreadShouldExit` (RebuildDispatch.cpp:828-833)
- **2026-08-11 追加 (recovery)**: `submitRecoveryRequest` (ISRRuntimePublicationCoordinator.cpp:721-781) は reservation-before-push + push-success OR durable fallback（queue full） の2-path。`settlePendingRecoveryAdmission(bool retry)` は retry=true で DurablePending→Building レースを管理
- **2026-08-11 追加 (publish)**: `RuntimeWorldAuthority::publish()` (RuntimeWorldAuthority.h:223-264) は `coordinator_.commit()`（metadata bake + currentWorld_ set）→ `publishAndSwap()`（RuntimeStore::current swap）の commit-before-swap 順序を保証。PublicationSemantic は RuntimeState.publication フィールドとして物理的に co-located
- **2026-08-11 追加 (retire)**: `drainDeferredRetireQueues` (AudioEngine.Retire.cpp:41-117) は `pendingReclaimHandles_` から抽出して `requestReclaim` 再試行。TOCTOU対策: `isRetired()` チェック + epoch 再確認 — `Quarantined` 状態から `Retired` への誤上書きを防止

---

## 補足2: Source Code Verification Appendix (2026-08-11 深層調査)

> **調査方法**: AiDex + serena + cocoindex + graphify + semble + WSL rg/grep の組み合わせで、全ソースファイルを検査。調査対象ファイルの行番号は実コードに照合済み。

### 5.1 MpscBoundedRing — CAS-based true MPSC lock-free (src/MpscBoundedRing.h:50)

```cpp
// src/MpscBoundedRing.h:49-54
template<typename T, size_t Capacity>
class MpscBoundedRing {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of 2");
    static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");
```

- **Producer**: CAS (`compareExchangeAtomic` at line 82) — Multiple producers can race; slot reservation via CAS guarantees uniqueness. `reservation order == seqId order` (line 82 comment). Producer hole: CAS 予約 → payload 書込み → seq release. Consumer validates seq 番号で torn read を防止。
- **Consumer**: Single consumer (`CoordinatorLoop`) — `pop()` at line 119 uses `consumeAtomic(dequeuePos_, acquire)` + `publishAtomic` release. Only consumer, so no CAS needed.
- **`sizeApprox()`** (line 134): best-effort — `enqueuePos_ - dequeuePos_` (acquire). Only exact with single producer. Multi-producer は transient over-count あり。
- **`clear()`** (line 142): Must be called only after all producers/consumers stopped (same contract as LockFreeRingBuffer).
- **RT safe**: Producer is CAS (wait-free bounded retry), but **architectural invariant AC-ISR-1** requires Producer = NonRT only. Comment at line 25-27: `"RT パス（Audio Thread）からは push/pop されない（Producer は全て NonRT）"`.
- **Lock-free assertion**: No `static_assert(std::atomic<...>::is_always_lock_free)` exists in the file. `enqueuePos_` and `dequeuePos_` are `std::atomic<uint32_t>` — on x64 these are lock-free (native `uint32_t` atomics). `is_always_lock_free` check exists at AudioEngine.h:2380-2384 for `MmcppPolicy` enum but **NOT** for MpscBoundedRing counters.

### 5.2 Two ShutdownPhase enums — Non-1:1 correspondence (correction to §2.2)

**AudioEngine::ShutdownPhase** (AudioEngine.h:2533-2542) — **7 values** (int):

| Value | Name |
|---|---|
| 0 | Running |
| 1 | StopAcceptingWork |
| 2 | StopAudio |
| 3 | StopWorkers |
| 4 | ForceEpochAdvance |
| 5 | DrainRetire |
| 6 | Destroy |

**isr::ShutdownPhase** (ISRShutdown.h:25-41) — **11 values** (uint8_t):

| Value | Name |
|---|---|
| 0 | Running |
| 1 | AudioStopped |
| 2 | ObserverDrained |
| 3 | RetireClosed |
| 4 | EpochSettled |
| 5 | ReclaimComplete |
| 6 | EmergencyDrain (C-2) |
| 7 | VerifyDrained (P3) |
| 8 | TimedOut |
| 9 | Failed |
| 10 | ShutdownComplete |

**Non-1:1 correspondence**: AudioEngine (7) and ISR (11) have **different granularities and non-overlapping names**. `AudioEngine::ShutdownPhase::Destroy` ≠ `isr::ShutdownPhase::ReclaimComplete`. CacheMap::~CacheMap (AudioEngine.h:2019-2020) checks `AudioEngine::ShutdownPhase::Destroy` — NOT `isr::ShutdownPhase`. This means the 2 enums represent different lifecycle layers (Engine layer vs ISR Coordinator layer). ⚠️ **This is a correctness risk for any code comparing across the two enums — they must be bridged via `StopWorkers` → `AudioStopped/ObserverDrained/RetireClosed/EpochSettled/ReclaimComplete` mapping.**

### 5.3 currentWorld_ location correction (correction to §1.7)

**⚠️ The document's reference to `AudioEngine.h:2537`'s `runtimePublishWorld_` is incorrect — this field does not exist.**

- `currentWorld_` is a member of `RuntimeIntentCoordinator` (ISRRuntimePublicationCoordinator.h:465), NOT `AudioEngine`:
  ```cpp
  // ISRRuntimePublicationCoordinator.h:465
  std::atomic<const void*> currentWorld_;
  ```
- `AudioEngine` uses `worldAuthority_` (`convo::isr::RuntimeWorldAuthority`, AudioEngine.h:4685) which wraps a `RuntimeStore` (RuntimeWorldAuthority.h:114-119):
  ```cpp
  // RuntimeWorldAuthority.h:114-119
  using Store = convo::RuntimeStore<RuntimeState, RuntimeWorldAuthority>;
  explicit RuntimeWorldAuthority(RuntimeIntentCoordinator& coordinator) noexcept
      : coordinator_(coordinator)
      , runtimeStore_()
      , writeAccess_(runtimeStore_.acquireWriteAccess())  // ★ X4-B-3
  ```
- The read API is `RuntimeWorldAuthority::observePublishedWorld()` (AudioEngine.h:3556-3558):
  ```cpp
  [[nodiscard]] const RuntimePublishWorld* observePublishedWorld() const noexcept {
      return worldAuthority_.observePublishedWorld();
  }
  ```
- `submitRecoveryRequest` (ISRRuntimePublicationCoordinator.cpp:739-740) reads `currentWorld_` from the Coordinator:
  ```cpp
  const auto world = static_cast<const RuntimeState*>(
      convo::consumeAtomic(currentWorld_, std::memory_order_acquire));
  ```
- INV-X4-B comment at RuntimeWorldAuthority.h:102-103: `"read-source singularization（currentWorld_ 廃止）は Future（二十四次レビュー §27-C）。現状は PublishExecutor が sole gateway + 一時生成 Coordinator が唯一の store-swap ため INV-X4-3 は de facto 成立。"` — **read-source singularization is still a future task, not done.**

### 5.4 Commit-before-swap ordering (verification of §1.7 AC-PUB-1)

Confirmed at `RuntimeIntentCoordinator::commit()` (ISRRuntimePublicationCoordinator.cpp:75-115). The actual publication steps (lines 104-114) within the commit function body:

```cpp
/// commit() at ISRRuntimePublicationCoordinator.cpp:75-115
/// (overloaded — line 75 is the full-arg version, line 62 delegates to it)
void RuntimeIntentCoordinator::commit(
    PublishAuthority,
    RuntimeBoundary boundary,
    const void* newWorld,
    std::uint64_t /*version*/,
    PublicationSequenceId sequenceId,
    PublicationEpoch epoch,
    std::uint64_t mappedGeneration) {

    // (lines 82-102): boundary + monotonicity validation → Faulted if invalid

    // ★ publication steps (104-114):
    convo::publishAtomic(state_, CoordinatorState::Publishing, std::memory_order_release);  // (1) set publishing state
    convo::publishAtomic(swapPending_, true, std::memory_order_release);                     // (2) mark swap pending
    // (3) bake PublicationSemantic onto newWorld (metadata commit BEFORE physical swap)
    auto* pubWorld = const_cast<RuntimeState*>(static_cast<const RuntimeState*>(newWorld));
    pubWorld->publication = PublicationSemantic{sequenceId, epoch,
        static_cast<PublicationGeneration>(mappedGeneration), prevSeqId};
    convo::publishAtomic(currentWorld_, newWorld, std::memory_order_release);  // (4) publish currentWorld_ (physical swap)
    convo::publishAtomic(swapPending_, false, std::memory_order_release);      // (5) clear swap pending
    convo::publishAtomic(state_, CoordinatorState::Ready, std::memory_order_release);  // (6) set ready
}
```cpp
convo::publishAtomic(state_, CoordinatorState::Publishing, std::memory_order_release);  // (1) set publishing state
convo::publishAtomic(swapPending_, true, std::memory_order_release);                     // (2) mark swap pending
// (3) bake PublicationSemantic onto newWorld (metadata commit BEFORE physical swap)
auto* pubWorld = const_cast<RuntimeState*>(static_cast<const RuntimeState*>(newWorld));
pubWorld->publication = PublicationSemantic{sequenceId, epoch, ...};
// (4) publish currentWorld_ (physical swap — RT observers can now read)
convo::publishAtomic(currentWorld_, newWorld, std::memory_order_release);
convo::publishAtomic(swapPending_, false, std::memory_order_release);                    // (5) clear swap pending
convo::publishAtomic(state_, CoordinatorState::Ready, std::memory_order_release);       // (6) set ready
```
**AC-PUB-1 holds**: After commit (step 4), `currentWorld_` and `pubWorld->publication` are co-located — the publication metadata is baked onto the RuntimeState BEFORE the pointer is published. Any RT observer reading `currentWorld_` acquire-observes the full `PublicationSemantic` (release at step 3 + step 4).

### 5.5 QuarantineIntentHandler → Recovery path (verification of §1.9)

**Primary Recovery path** (NOT through intentQueue_ Dispatcher):

```
QuarantineIntentHandler::handle() (ISRRuntimePublicationCoordinator_ProcessIntent.cpp:110-141)
  → QuarantineService::executeQuarantine() (ISRRuntimePublicationCoordinator.cpp:688-713)
    → handleRuntime.quarantine(request.handle)  [State change]
    → quarantineManager.quarantineHandle()       [Audit]
  → if (qResult.stateChanged) → submitRecoveryIntent() (AudioEngine.h:4285, renamed to n())
    → submitRecoveryRequest() (ISRRuntimePublicationCoordinator.cpp:721-780)
      → reservation-before-push: fetchAdd(pendingIntentCount_) → recoveryIntentQueue_.push()
      → push success: transport recovery in queue
      → push fail (queue full): rollback fetchSub + durable admission (pendingRecoveryAdmission_)
```

**RecoveryIntentHandler** (ISRIntentDispatcher.h:43-47, ISRRuntimePublicationCoordinator_ProcessIntent.cpp:163-169): **Dead code** — no producer enqueues `IntentType::Recovery` intents into `intentQueue_`. Comment at ProcessIntent.cpp:158-162 confirms: "現状は誰も intentQueue_ に Recovery Intent を push しないため dead code". This is a **future extension path only**. Recovery is issued via `QuarantineIntentHandler → submitRecoveryIntent → submitRecoveryRequest → recoveryIntentQueue_` (bypasses intentQueue_ Dispatcher entirely).

**Intent dispatch table** (ISRIntentDispatcher.h:60-68):
```cpp
constexpr const IntentHandler* kDispatchTable[kIntentTypeCount] = {
    &g_observeIntentHandler,   // IntentType::Observe   (0)
    &g_publishIntentHandler,   // IntentType::Publish   (1)
    &g_recoveryIntentHandler,  // IntentType::Recovery  (2)  ← dead code
    &g_quarantineIntentHandler // IntentType::Quarantine(3)  ← primary recovery trigger
};
```

### 5.6 DSPCore::prepare() subsystem void-return chain (correction to §1.8.12)

`DSPCore::prepare()` (AudioEngine.Processing.DSPCoreLifecycle.cpp:72) is **void** and calls 10 sub-systems, ALL returning void:

| Line | Sub-system | Return type | Failure handling |
|---|---|---|---|
| 179 | `ramp.prepare(newSampleRate)` | void | — |
| 183 | `oversampling.prepare(...)` | void | — |
| 188 | `softClipOS.prepareSingleStage(...)` | void | — |
| 196 | `convolverState->prepare(owner, ...)` | void | → calls ConvolverProcessor::prepareToPlay |
| 200 | `eqState->prepare(...)` | void | — |
| 204 | `dcBlockers().init(...)` | void | — |
| 208-211 | `dither.prepare()` / `fixedNoiseShaper.prepare()` / `adaptiveNoiseShaper.prepare()` | void | — |
| 216 | `outputFilter.prepare(...)` | void | — |
| 220 | `truePeakDetector.prepare(...)` | void | — |
| 224 | `loudnessMeter.prepare(...)` | void | — |
| 228 | `peakLimiter.prepare(...)` | void | — |

`convolverState->prepare()` (AudioEngine.h:682-688) calls `ConvolverProcessor::prepareToPlay()` (ConvolverProcessor.Lifecycle.cpp:211) which is **void**, but internally calls `newConv->init()` returning **bool** (ConvolverProcessor.h:741):
```cpp
if (newConv->init(irL.release(), irR.release(),
                  conv->irDataLength, sampleRate, conv->irLatency, ...))
{
    newConv = newConvHolder.release();
    // exchange + retire old
}
else
{
    juce::Logger::writeToLog("ConvolverProcessor::prepareToPlay: NUC re-init failed. Keeping existing engine.");
}
```
⚠️ **The bool return of `newConv->init()` is NOT propagated upward** — `prepareToPlay` returns void, `convolverState->prepare` returns void, `DSPCore::prepare` returns void. The failure is logged and the existing engine is kept (fallback). `build()` at RuntimeBuilder.cpp:448 calls `runtime->prepare(...)` — which is void — and **never checks the internal `newConv->init()` failure**. This is the root cause of `MKLFailure`/`ConvolverFailure`/`PrepareFailure` never being returned from `build()`.

### 5.7 pendingRecoveryAdmission_ State machine (confirmation of §1.8.5.2)

`PendingRecoveryAdmission` struct (ISRRuntimePublicationCoordinator.h:575-593):
```cpp
struct PendingRecoveryAdmission {
    enum class State : uint8_t {
        NoAdmission = 0,
        DurablePending,
        Building
    };
    State state = State::NoAdmission;
    bool pending = false;
    uint64_t recoveryGeneration = 0;
    convo::RuntimeBuildSnapshot buildSource{};
    bool reservationOwned = false;  // 1 admission = 1 reservation (INV-X1-5)
    DSPHandle handle{};
    PublicationEpoch epoch{0};
    uint64_t intentId{0};
};
PendingRecoveryAdmission pendingRecoveryAdmission_;  // SPSC (plain struct — atomic none)
std::atomic<bool> recoveryAdmissionPending_{false};   // durable flag (isFullyDrained reads this)
```

**`settlePendingRecoveryAdmission(bool retry)`** (ISRRuntimePublicationCoordinator.cpp:831-842):
- `retry=true`: `DurablePending → Building` transition (Building→DurablePending also possible). `recoveryAdmissionPending_` stays true.
- `retry=false`: Full clear (`= PendingRecoveryAdmission{}`) + `recoveryAdmissionPending_ = false`.

### 5.8 isFullyDrained Layer 1 detail (AudioEngine.Threading.cpp:114-157)

Layer 1 checks (in order):
1. `!hasDeferredCommit` — `runtimeOrchestrator_->hasDeferredRequest()` (deferred publication commit)
2. `pendingReclaimHandles_.empty()` — mutex-guarded (pendingReclaimHandlesMutex_)
3. `ringResident == 0` — overflow ring resident count (3 sources: overflowRing + dspQuarantine + retireQuarantineStore)
4. `dspQuarantineResident == 0` — DSPQuarantineManager::residentCount()
5. `retireQuarantineResident == 0` — m_retireRouter->quarantineResidentCount()
6. `runtimePublicationBridge_.isFullyDrained()` — Layer 2 (16 conditions)

**⚠️ External setter still active**: Threading.cpp:126-128 still calls `setFallbackBacklogCount(fallbackDepth)`, `setRetireBacklogCount(retireDepth)`, `setDeferredRetireResidencyCount(fallbackDepth)` — these are the "実測上書き" setters that §1.4 says should be eliminated. The document's claim that setters are "現行コードの16条件で判定済み" is partially inaccurate — Layer 1 still uses 3 external setters to overwrite Coordinator counters.

### 5.9 build() try/catch is always active (clarification of §1.8.12 task 6)

The document claims `build()` has no `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` guard — this is **confirmed correct**. The `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` blocks at RuntimeBuilder.cpp:7 (file header) and 195, 418 only guard `diagLog` calls (diagnostic logging), NOT the try/catch logic. The try/catch at RuntimeBuilder.cpp:441-468 is **always compiled** (not behind any macro). `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` is `OFF` by default (CMakeLists.txt:129), meaning `diagLog` calls are compiled out, but the `catch(std::bad_alloc) → ResourceUnavailable` and `catch(...) → InternalError` are still active in production.

### 5.10 Two-layer isFullyDrained — Layer 2 exact conditions (ISRRuntimePublicationCoordinator.cpp:500-525)

All 16 conditions confirmed — **⚠️ correction: the `swapPending_` pre-check is condition 0 (early-return at line 485),  NOT listed in the document's 15-item return body**:

- **Condition 0 (pre-check, not in return body):** `coordinator_.swapPending_ == false` (line 485-486 — early return)
- **Conditions 1-15 (in return body, lines 500-525):**
1. `intentQueue_.sizeApprox() == 0`
2. `observeDeferredRing_.size() == 0`
3. `quarantineFallbackQueue_.sizeApprox() == 0`
4. `recoveryIntentQueue_.size() == 0`
5. `retireBacklogCount_ == 0` (consumeAtomic acquire)
6. `publicationBacklogCount_ == 0`
7. `publicationIntentResidencyCount_ == 0` (INV-X5-1 addition)
8. `pendingIntentCount_ == 0` (reservation hole)
9. `fallbackBacklogCount_ == 0`
10. `reclaimInFlightCount_ == 0`
11. `deferredRetireResidencyCount_ == 0`
12. `quarantineIntentResidencyCount_ == 0` (X6)
13. `quarantineRingResidencyCount_ == 0` (X6)
14. `quarantineResidentCount_ == 0` (physical DSP objects)
15. `!recoveryAdmissionPending_` (durable admission)

- **Condition 16 (caller-side, NOT inside isFullyDrained):** `state_ == CoordinatorState::ShuttingDown` checked by `markShutdownComplete()` at line 533 — ensures isFullyDrained() is not evaluated during a Faulted state. This is an explicit caller-side guard, not an "implicit" check. The caller at ISRRuntimePublicationCoordinator.cpp:533-535 checks `state != ShuttingDown → return`, then calls `isFullyDrained()` at line 538.

**⚠️ Critical: `swapPending_` is NOT one of the 15 return-body conditions** — it is a separate early-return check that prevents evaluating the 15 conditions during an in-progress swap. The document's §1.4 table incorrectly omits this pre-check. If the external setters are eliminated (§1.4 goal), `swapPending_` must still be checked (it guards against torn reads during publication swap, independent of the backlog counters).

### 5.11 postStopEnqueue tracking (ISRShutdown.h:144,202,223 / ISRShutdown.cpp:162 / AudioEngine.Commit.cpp:459)

- `ISRShutdown.h:144`: `uint32_t postStopEnqueueCount{0};` (in ShutdownResult struct)
- `ISRShutdown.h:202`: `void markPostStopEnqueue() noexcept;` (API declaration)
- `ISRShutdown.h:223`: `std::atomic<uint32_t> sh6PostStopEnqueueCount_{0};` (storage)
- `AudioEngine.Commit.cpp:459`: `if (shutdownRuntime_.isShutdownInProgress()) shutdownRuntime_.markPostStopEnqueue();`
- `ISRShutdown.cpp:162`: `result.postStopEnqueueCount = convo::consumeAtomic(sh6PostStopEnqueueCount_, ...);` (read at shutdown completion)
- `ISRShutdown.cpp:257`: `file << "  \"sh6_postStopEnqueueCount\": " << sh6 << "\n";` (trace output)

This is the `postStopEnqueue == 0` condition mentioned in §2.2.

### 5.12 ShutdownQuiescent reclaim precondition (ISRRuntimePublicationCoordinator.cpp:638-647)

```cpp
if (mode == ReclaimMode::ShutdownQuiescent)
{
    // INV-X3-4 / INV-ISR-04: ShutdownQuiescent reclaim は
    //   (a) reader registration が永久に閉じている（CloseReaderRegistration フェーズ）
    //   (b) active reader が 0（readersZero — graceful drain 完了）
    // の両方が必須（R4 Phase 3: ShutdownQuiescenceProof の構成要素）。
    if (!readerRegistrationClosed || router.activeReaderCount() != 0)
        return false;
}
```
The `readerRegistrationClosed` bool is passed from the caller — in CacheMap::~CacheMap (AudioEngine.h:2035): `owner->m_epochDomain.readerRegistrationClosed()`. ⚠️ **This is a bool parameter — caller can pass true without actual proof (§2.2 review 指摘).**

### 5.13 External setters still present (correction to §1.4)

Despite the document's claim that external setters are being eliminated, the following setters are **still declared** at ISRRuntimePublicationCoordinator.h:121-127:

| Setter | h:line | Called from AudioEngine.Threading.cpp |
|---|---|---|
| `setFallbackBacklogCount` | h:124 | ✅ Threading.cpp:126 |
| `setRetireBacklogCount` | h:122 | ✅ Threading.cpp:127 |
| `setPendingIntentCount` | h:123 | ❌ (Coordinator internal) |
| `setDeferredRetireResidencyCount` | h:126 | ✅ Threading.cpp:128 |
| `setReclaimInFlightCount` | h:125 | ❌ (Coordinator internal, called at cpp:660,671) |
| `setQuarantineResidentCount` | h:127 | ❌ (Phase2, called from DSPQuarantineManager) |

⚠️ **Threading.cpp:126-128 still calls 3 setters to overwrite real-measured values** — these are the "実測上書きの全廃" targets in §1.4 that have NOT yet been eliminated. The document says "廃止済み" but source code shows they are still active.

### 5.14 convolver init() bool return NOT propagated — root cause of unused MKLFaliure/ConvolverFailure/PrepareFailure

The chain: `build()` → `runtime->prepare()` (void) → `convolverState->prepare()` (void) → `ConvolverProcessor::prepareToPlay()` (void) → `newConv->init()` (bool, checked locally but NOT returned upward).

At ConvolverProcessor.Lifecycle.cpp:276-291: `newConv->init()` returns false → log message + keep existing engine. No exception thrown, no failure status propagated to DSPCore::prepare() → no failure status propagated to `build()`.

**Therefore MKLFailure/ConvolverFailure/PrepareFailure are structurally unreachable from the build path** — the enum values exist in `toString` (RuntimeBuilder.cpp:53-76) but no code path sets them on a `BuildResult`. The only values `build()` returns are `InvalidInput` (:435), `ResourceUnavailable` (:461), `InternalError` (:466), and `None` (implicit success at :455-457).

### 5.15 INV-ISR-01 through INV-ISR-07 (top-level ISR invariants)

Defined as comments at ISRRuntimePublicationCoordinator.h:68-83:

| Invariant | Content |
|---|---|
| INV-ISR-01 | `isFullyDrained == true` ⇒ all producers stopped AND all producer joins completed AND all transport queues empty AND all deferred state empty AND all reclaim-in-flight == 0 AND all reader inactive AND reader registration closed |
| INV-ISR-02 | `pendingIntentCount_` は queue size ではなく transport residency + producer reservation である（residency + reservation — 二重計上禁止） |
| INV-ISR-03 | 異なる semantic state を一つの counter で表現しない。特に Intent / DSP resident / Retire resident を混ぜない（§6.6 X6 の4層分離と整合） |
| INV-ISR-04 | ShutdownQuiescent reclaim は readerRegistrationClosed なしでは絶対に許可しない（§6.3 X3 と整合） |
| INV-ISR-05 | completion watermark を publication committed と同一視しない（§6.2 X2 と整合） |
| INV-ISR-06 | currentWorld_ を ownership source として扱わない（§6.4-X4 INV-X4-8 と整合） |
| INV-ISR-07 | currentWorld_ と RuntimeStore::current が存在する間は、両者の identity consistency を検証可能にする（§6.4-X4 Test 9 / INV-X4-6 と整合） |

### 5.16 BuildError: catch(...) maps std::bad_alloc-only to ResourceUnavailable, everything else to InternalError (confirmed)

build() try/catch (RuntimeBuilder.cpp:441-468) actual failure mapping:

| Thrown by | Caught by | Result |
|---|---|---|
| aligned_make_unique<DSPCore>() -> std::bad_alloc | catch(std::bad_alloc) | ResourceUnavailable |
| aligned_make_unique<DSPCore>() -> other std::exception | catch(...) | InternalError |
| convolverRt().setVisualizationEnabled() (non-noexcept inline) | catch(...) | InternalError |
| convolverRt().applyBuildSnapshot() (void, non-noexcept) | catch(...) | InternalError |
| convolverRt().transferIRStateFrom() (void, non-noexcept) | catch(...) | InternalError |
| runtime->prepare() (void, non-noexcept) -> newConv->init() failure | NOT caught — void return, init() bool logged locally, NOT propagated | InternalError only if exception thrown (which it isn't) |

Critical: newConv->init() failure at ConvolverProcessor.Lifecycle.cpp:276 returns false, but prepareToPlay() does NOT convert this to any BuildError. Logs + void return. build() at RuntimeBuilder.cpp:448 calls runtime->prepare() (void) so build() only sees success or InternalError.

Root cause of MKLFailure/ConvolverFailure/PrepareFailure never being returned: failure chain is newConv->init() -> bool check -> log -> void return -> void return -> void return. Bool lost at every hop.

### 5.17 buildRuntimePublishWorld — no BuildError usage (confirmed)

buildRuntimePublishWorld() (RuntimeBuilder.h:134-137, RuntimeBuilder.cpp:179-426) is noexcept, does NOT return BuildResult or use BuildError. Returns aligned_unique_ptr<RuntimePublishWorld> (nullptr on failure). build() (RuntimeBuilder.h:206-209, RuntimeBuilder.cpp:428-469) is a SEPARATE function handling DSPCore construction with BuildError.

Callers:
- buildRuntimePublishWorld(): PrepareToPlay.cpp:149,271; ReleaseResources.cpp:169; Transition.cpp:22; Timer.cpp:912; Init.cpp:53; AudioEngine.Commit.cpp:390
- build(): only RebuildDispatch.cpp:941,1015,1087

Design intent confirmed: build() (DSPCore construction, BuildError-classified) vs buildRuntimePublishWorld() (World assembly, noexcept) are intentionally separated.

### 5.18 Recovery Admission State Machine — full lifecycle (ISRRuntimePublicationCoordinator.h:575-593, cpp:721-842)

PendingRecoveryAdmission 3-state SPSC machine (Producer=CoordinatorLoop, Consumer=Builder Loop):

| State | Trigger | recoveryAdmissionPending_ | pendingIntentCount_ |
|---|---|---|---|
| NoAdmission | Initial / settle(false) | false | 0 |
| DurablePending | Queue full -> fallback (submitRecoveryRequest:772) | true | 1 (reservation owned) |
| Building | takePendingRecoveryAdmission() (cpp:790-804) | true | 1 (unchanged) |
| -> DurablePending | settle(true) | true | 1 |
| -> NoAdmission | settle(false) | false | 0 |

recoveryShutdownDiscardCount_ (cpp:735, 820): telemetry only, NOT a drain condition. discardPendingRecoveryAdmission() sets recoveryAdmissionPending_=false which IS drain condition #15. Invariant: pendingIntentCount_ tracks transport reservations only. Durable admission tracked separately (INV-X1-6: no double-counting).

### 5.19 Recovery shutdown gate + discard call site

Shutdown gate (submitRecoveryRequest:733): Checks state_ == ShuttingDown BEFORE incrementing pendingIntentCount_. If shutdown, recoveryShutdownDiscardCount_, returns (no reservation, no push).
Graceful discard (discardPendingRecoveryAdmission:816-824): Called from stopRebuildThread() (RebuildDispatch.cpp:798), AFTER Builder thread joined. Clears pendingRecoveryAdmission_ + sets recoveryAdmissionPending_=false.

Call chain: AudioEngine::shutdown() -> stopRebuildThread() (line 771: phase->StopWorkers, notify CV, join) -> discardPendingRecoveryAdmission() (line 798) -> waitForDrain() (Threading.cpp:159) -> isFullyDrained() (Threading.cpp:114) -> Coordinator::isFullyDrained() (ISRRuntimePublicationCoordinator.cpp:484)

**Test coverage**: `ISRSemanticValidationTests.cpp:631` (`testRecoveryDurableAdmission`) covers the full state machine: fill 256 transport slots -> 257th triggers durable admission -> pop 256 transport items -> takePendingRecoveryAdmission (lease DurablePending→Building) → settle(false) (clear) → take returns nullopt (empty). Additional test at line 886 (`testRecoveryRequestEnqueueAndPop`) covers basic enqueue/pop. Lines 914-917 call `testRecoveryDurableAdmission()` from the main test entry.

### 5.20 enqueuePublicationIntent rollback path (ISRRuntimePublicationCoordinator.h:324-339)

`enqueuePublicationIntent` (h:324-339) uses the same reservation-before-push pattern as `submitRecoveryRequest`:

```cpp
fetchAddAtomic(publicationIntentResidencyCount_, 1);  // reservation
if (intentQueue_.push(prepared))
    return true;
fetchSubAtomic(publicationIntentResidencyCount_, 1);  // rollback on full
return false;
```

When `enqueuePublicationIntent` returns false (queue full), the caller at AudioEngine.h:4424-4429 performs rollback:
```cpp
if (!runtimePublicationBridge_.enqueuePublicationIntent(intent))
{
    (void)worldAuthority_.ownerChannel().take(
        convo::isr::OwnerChannelKey{ seqId, epoch, mappedGen });
    worldAuthority_.registry().unregister(seqId);
}
```

This is INV-X5-1 (publicationIntentResidencyCount = Publish intent queue residency + producer reservation). The 3 enqueue paths (normal rebuild / Recovery publish / deferred re-enqueue) all funnel through `enqueuePublicationIntent` — single reservation site.

**⚠️ Note**: `enqueuePublicationIntentForRuntimeCommit` (AudioEngine.h:2473, implemented at AudioEngine.Commit.cpp:707) is a HIGHER-LEVEL wrapper that constructs the `Intent` struct and then calls `enqueuePublicationIntent`. The `enqueuePublicationIntentForRuntimeCommit` does NOT touch `publicationIntentResidencyCount_` directly — only `enqueuePublicationIntent` does (at h:334/337). This is correct: single reservation site.

---

## ⚠️ 2026-08-11 Review: Immediate Fix Candidates (P1)

Per the ISR Architecture Review (2026-08-11), the following are existing contract violations in the current codebase that must be fixed BEFORE Phase A/B implementation:

### A-1. `mixSmoother.reset()` RT thread violation
- **File**: `ConvolverProcessor.Runtime.cpp:360`
- **Issue**: `LinearRamp::reset()` asserts `ASSERT_NON_RT_THREAD()` but is called from audio processing thread
- **Severity**: Medium (debug jassert violation; no data race in release since `totalSteps` is read-only)
- **Fix**: Phase 0.5 — either split `reset()` into RT-safe and NonRT variants, or make `totalSteps` atomic and remove ASSERT
- **NonRT callers** (`ConvolverProcessor.Lifecycle.cpp:370,371,376,489`) are unaffected (already NonRT)

### A-2. `isFullyDrained` external setters
- **File**: `Threading.cpp:126-128`
- **Issue**: `setFallbackBacklogCount()` / `setRetireBacklogCount()` / `setDeferredRetireResidencyCount()` overwrite authoritative counters computed from actual queue state
- **Risk**: If Phase A (ShutdownQuiescenceProof) proceeds before setters are eliminated, the proof object validates against corrupted data
- **Fix**: Phase B — eliminate all external setters; counters must be derived exclusively from push/pop operations

### A-3. `reclaim()` bool parameter (ShutdownQuiescenceProof not yet implemented)
- **File**: `ISRRuntimePublicationCoordinator.h:373-377`
- **Issue**: `reclaim(..., bool readerRegistrationClosed=false)` accepts a plain bool — caller can pass `true` without proof
- **3 call sites**: `ReleaseResources.cpp:423`, `ReleaseResources.cpp:433`, `AudioEngine.h:2032`
- **Fix**: Phase A — replace with `ReclaimPermit` object whose generation is limited to `ShutdownRuntime` only

### A-4. Retry storm / no backoff in recovery loop
- **File**: `AudioEngine.RebuildDispatch.cpp:990-1048`
- **Issue**: `kMaxRecoveryConsecutiveFailures=4` bounds consecutive failures, but no exponential backoff between retries. `continue` immediately retries
- **Risk**: System-level RT safety — retry storm saturates Builder thread → publication latency increases
- **Fix**: Phase D — implement exponential backoff (min 1ms, max 100ms) + telemetry (retryCount/retryLatency/retryStormDetected)

### A-5. RetryDisposition without retry count/backoff specification
- **File**: §1.8.9 (new)
- **Issue**: `RetryDisposition::Retry` does not specify max retry count or backoff
- **Fix**: Implement AC-ISR-4: Retry is bounded (max 3 configurable), exponential backoff, generation validity check, shutdown state check

---

## ⚠️ 2026-08-11 Review: Final GO/NO-GO Assessment

### 🟢 Immediate Design Fix (Phase 0.5)
- **mixSmoother.reset() RT violation** — must fix before any Tier 1 work

### 🟢 Now Safe to Design (Phase A-C)
| Item | Status | Condition |
|------|--------|-----------|
| 2.2 ShutdownQuiescenceProof | Design ready | Must replace bool param with proof object |
| 1.4 isFullyDrained semantic | Design ready | Must eliminate external setters (Phase B) |
| 2.1 Retire epoch safety / FIFO | Design ready | Confirmed SPSC producer = CoordinatorLoop only |
| 1.6 sequence arithmetic tests | Ready | Add `wraparound comparison rule` to AC |
| 1.1 MPSC design | Design ready | Producer = CoordinatorLoop (NonRT) only — no RT producers |

### 🟡 Design ready, implement after Phase A-C
| Item | Status | Condition |
|------|--------|-----------|
| 1.3 LinearRamp | Fix known bug first | Phase 0.5 before remaining work |
| 1.8 BuildError classification | Design ready | Must add retry count + backoff AC |
| 1.9 RecoveryAdmission / wake | Design ready | Must add telemetry for absorption |
| 1.7 currentWorld_ singularization | High risk | Must fix PublicationIdentity invariant in tests first |
| 1.5 sparse completion | Correct to defer | Only implement when MPSC completion needed |

### 🔴 Do not implement
- **1.2 Recovery coalesce** — NO-GO (correctly identified)

---

## Verification Summary: NonRT Producer Confirmation

**ISR safety check for §1.1 (Recovery MPSC):**

All `submitObserve` callers (the only RT-facing producer) were verified:
- `AudioEngine.Timer.cpp:896, 1029, 1568` — Timer callback (MessageThread, NonRT)
- `DSPTransition.h:156` — explicitly documented as NonRT (`AudioEngine.h:1201-1202`: "DSPTransition は Non-RT スレッドで動作する")

`processIntent` (which dispatches QuarantineIntentHandler → `submitRecoveryIntent`) runs only on:
- `CoordinatorLoop` (juce::Thread, ISRCoordinatorLoop.cpp:7-8) — NonRT worker thread
- Called via `AudioEngine::runCoordinatorPhase()` (Threading.cpp:225-230)

**`submitObserve` is the RT-facing producer → `observeDeferred_` ring buffer.**  
**`submitRecoveryRequest` is the NonRT-only producer → `recoveryIntentQueue_` + `pendingRecoveryAdmission_`.**

→ Current SPSC design is correct. MPSC conversion is for future NonRT producer scalability, NOT RT safety.

---

## ⚠️ Deep Investigation Appendix (2026-08-11)

### Appendix A: §1.1 — RT Producer Exhaustive Verification

**All `submitObserve` callers verified NonRT:**

| Caller | File:Line | Thread | Evidence |
|--------|-----------|--------|----------|
| Timer callback | AudioEngine.Timer.cpp:896 | MessageThread (NonRT) | AudioEngine inherits `juce::Timer` (h:585); timerCallback runs on MessageThread |
| Timer callback | AudioEngine.Timer.cpp:1029 | MessageThread (NonRT) | Same — Timer Thread = MessageThread |
| Timer callback | AudioEngine.Timer.cpp:1568 | MessageThread (NonRT) | Same |
| DSPTransition | DSPTransition.h:156 | NonRT | AudioEngine.h:1201-1202: "DSPTransition は Non-RT スレッドで動作する" |

**Audio thread registration mechanism:**
- `ScopedThreadRole(ThreadRole::AudioRealtime)` at `AudioEngine.Processing.AudioBlock.cpp:102` and `BlockDouble.cpp:104` — registers the audio thread hash in `audioThreadSlots[]` (DspNumericPolicy.h:37-40)
- `isAudioThread()` (DspNumericPolicy.h:118-127) checks current thread hash against slots
- `ASSERT_NON_RT_THREAD()` = `jassert(!isAudioThread())` (DspNumericPolicy.h:183)

**All `submitRecoveryRequest` / `submitRecoveryIntent` callers verified NonRT:**

| Caller | File:Line | Thread | Evidence |
|--------|-----------|--------|----------|
| QuarantineIntentHandler | ISRRuntimePublicationCoordinator_ProcessIntent.cpp:139 | CoordinatorLoop (NonRT) | Called from `processIntent` which runs on CoordinatorLoop (ISRCoordinatorLoop.cpp:31-42, `juce::Thread`) |
| (future) RecoveryIntentHandler | ProcessIntent.cpp:168 | CoordinatorLoop (NonRT) | Dead code (nobody pushes Recovery Intent to intentQueue_) |

**`enqueuePublicationIntent` callers verified NonRT:**
- `PublishIntentHandler` (ProcessIntent.cpp) — called from CoordinatorLoop
- `AudioEngine.h:4424` (rollback path) — called from NonRT

**⚠️ Correction**: `AudioEngine.h:4285 submitRecoveryIntent` unconditionally calls `rebuildCV.notify_all()` (line 4296). The §1.9 optimization of suppressing this is only safe if the rebuild thread's `recoveryPending` predicate (RebuildDispatch.cpp:831) is preserved. The rebuild thread is started at `AudioEngine.Init.cpp:33` — before Bootstrap World publish (Init.cpp:45-49). Bootstrap World is published synchronously via `worldAuthority_.publish()` before the CoordinatorLoop starts (Init.cpp:123). Therefore, **`observePublishedWorld()` always returns non-null during normal operation** — the "no authoritative runtime" case only exists in the sub-millisecond window between rebuild thread start and bootstrap publish.

### Appendix B: §1.2 — Recovery Coalesce / Identity Tracking

**No existing dedup mechanism:**
- `RecoveryIntent` (ISRRuntimePublicationCoordinator.h:195-208): Contains `DSPHandle handle`, `PublicationEpoch epoch`, `uint64_t intentId`, `RuntimeBuildSnapshot buildSource` — no previous-handle tracking field
- `recoveryIntentQueue_` (LockFreeRingBuffer, 256 slots): Simple FIFO — **no dedup**, multiple recovery requests for the same handle are queued separately
- `PendingRecoveryAdmission` (h:575-590): Has `handle` and `recoveryGeneration` fields. When queue is full → 257th request overwrites `buildSource` and `handle` (latest-wins coalesce). But **no `SupersessionProof`** — the overwrite is silent, no reason recorded

**Confirmed: §1.2 NO-GO is correct — no existing coalesce infrastructure exists.** The `recoveryGeneration` field at h:583 is labeled "coalesce 判定用" but is only set to `intent.intentId` (cpp:774) — never used for dedup decisions. No `lastRecoveryHandle_` tracking exists anywhere in the codebase.

### Appendix C: §1.8 — Retry Backoff / Retry Storm Analysis

**Current retry mechanism:**
```
kMaxRecoveryConsecutiveFailures = 4  (RebuildDispatch.cpp:1003)
```

**Durable recovery retry loop** (RebuildDispatch.cpp:1005-1052):
1. `takePendingRecoveryAdmission()` → leases DurablePending → Building (cpp:1005)
2. `runtimeBuilder.build(recovery->buildSource.buildInput, ...)` → if `runtime == nullptr` (cpp:1017):
   - Log error
   - `settlePendingRecoveryAdmission(true)` → returns to DurablePending (cpp:1022) — retry
   - `++recoveryConsecutiveFailures` — if `>= 4`, `break` (cpp:1024-1025)
   - `continue` — immediately retries (NO backoff)
3. `runtimeBuilder.validateWarmup(*recoveryDSP)` → if `!= None` (cpp:1033):
   - Log error + destroy DSP
   - `settlePendingRecoveryAdmission(true)` → retry (cpp:1044)
   - `++recoveryConsecutiveFailures` — if `>= 4`, `break` (cpp:1046-1047)
   - `continue` — immediately retries (NO backoff)

**Normal rebuild warmup retry** (RebuildDispatch.cpp:1143-1159):
- `shouldRetryWarmupFailure(dsp)` (cpp:78-81) checks only `dsp.convolverRt().isLoadingIR()`
- If retryable: `submitRebuildIntent(Structural, RebuildThreadWarmupRetry)` — re-enqueues via Coordinator deferred path (different from direct loop retry)
- This path has natural backoff via the periodic Coordinator cadence

**⚠️ Retry storm risk:**
- Durable recovery retry loop has **no exponential backoff** between iterations
- The `continue` at cpp:1026/1048 immediately re-takes the same admission
- Only bounded by `kMaxRecoveryConsecutiveFailures=4` before yielding to next loop iteration
- `recoveryPending` flag (set by `submitRecoveryIntent`) is re-set if a new recovery arrives during retry
- **System-level RT safety impact**: Builder thread saturation → delayed world publication → increased publication latency (indirect RT safety impact)
- **No telemetry**: No retry count, retry latency, or retry storm detection counters for recovery retries

### Appendix D: §1.4 — isFullyDrained 16 Conditions (exact verification)

**Pre-check (line 485):**
1. `swapPending_ == false` — guards against in-flight publication swap

**Return body (lines 500-525) — 15 conditions:**
| # | Condition | Line | Counter Source |
|---|-----------|------|----------------|
| 2 | `intentQueue_.sizeApprox() == 0` | 500 | Observe/Publish/Quarantine transport (MPSC MpscBoundedRing) |
| 3 | `observeDeferredRing_.size() == 0` | 501 | Observe overflow (SPSC LockFreeRingBuffer) |
| 4 | `quarantineFallbackQueue_.sizeApprox() == 0` | 502 | Quarantine fallback (MPSC MpscBoundedRing) |
| 5 | `recoveryIntentQueue_.size() == 0` | 503 | Recovery (SPSC LockFreeRingBuffer) |
| 6 | `retireBacklogCount_ == 0` | 504 | Retire backlog (atomic, incremented at Retire.cpp:48 |
| 7 | `publicationBacklogCount_ == 0` | 505 | Publication backlog (atomic) |
| 8 | `publicationIntentResidencyCount_ == 0` | 509 | Publish intent residency + producer reservation (fetchAdd cpp:889, fetchSub cpp:649) |
| 9 | `pendingIntentCount_ == 0` | 510 | Producer reservation hole (fetchAdd at submitObserve:602, submitRecoveryRequest:750, submitPublishRequest:889) |
| 10 | `fallbackBacklogCount_ == 0` | 511 | Fallback backlog (atomic) |
| 11 | `reclaimInFlightCount_ == 0` | 512 | Reclaim in-flight (set at Retire.cpp:48, cleared at :52) |
| 12 | `deferredRetireResidencyCount_ == 0` | 513 | Deferred retire residency (atomic) |
| 13 | `quarantineIntentResidencyCount_ == 0` | 518 | Quarantine intent residency (atomic, incremented at cpp:678, decremented at cpp:681) |
| 14 | `quarantineRingResidencyCount_ == 0` | 519 | Quarantine ring residency (atomic) |
| 15 | `quarantineResidentCount_ == 0` | 520 | Physical quarantine residency (from DSPQuarantineManager) |
| 16 | `!recoveryAdmissionPending_` | 525 | Durable Pending/Building flag (atomic) |

**External setter locations** (Threading.cpp:126-128) that corrupt Layer 1:
- `setFallbackBacklogCount()` — called from Threading.cpp:126
- `setRetireBacklogCount()` — called from Threading.cpp:127
- `setDeferredRetireResidencyCount()` — called from Threading.cpp:128

These setters overwrite the Coordinator's authoritative counters with AudioEngine-side measurements. Must be eliminated in Phase B.

### Appendix E: §1.6 — Sequence Number Comparison (no wraparound-safe logic)

**Sequence comparison** at `ISRRuntimePublicationCoordinator.cpp:96-97`:
```cpp
if (!(static_cast<std::uint64_t>(sequenceId) > static_cast<std::uint64_t>(prevSeqId)
    && static_cast<std::uint64_t>(epoch) > static_cast<std::uint64_t>(prevEpoch)
    && mappedGeneration > prevGen)) {
```

**⚠️ Simple `>` comparison — not modular/wraparound-safe.** Since `PublicationSequenceId` and `PublicationEpoch` are both `std::uint64_t`, wraparound would require 2^64 values (~584 billion years at 1ns intervals). Practically safe, but if §1.5 sparse completion or §1.6 tests are implemented, a proper modular comparison (`(int64_t)(a - b) > 0`) should be used.

**Intent sequenceId** (`nextObserveIntentId_`, `nextIntentId_`, `nextRecoveryIntentId_`) all use `fetch_add(1, relaxed)` — 64-bit counters, no wraparound handling needed.

### Appendix F: §1.7 — Dual-Pointer Sync Verification

**Two parallel world pointers confirmed synced via commit-before-swap:**

1. **`RuntimeStore::current`** (RuntimeStore.h:93) — `std::atomic<RuntimeState*>`, initialized to `nullptr`
   - Written only by `WriteAccess::publishAndSwap()` via `exchangeAtomic(current, next, acq_rel)` (RuntimeStore.h:49)
   - Only `RuntimeWorldAuthority` can call `acquireWriteAccess()` (RuntimeStore.h:86-91)
   - Read by `observe()` via `consumeAtomic(current, acquire)` (RuntimeStore.h:77-82)

2. **`currentWorld_`** (ISRRuntimePublicationCoordinator.h:465) — `std::atomic<const void*>`, initialized to `nullptr`
   - Written in `commit()` at cpp:112: `publishAtomic(currentWorld_, newWorld, release)` — same `newWorld` pointer as `publishAndSwap`

**Sync ordering** (RuntimeWorldAuthority.h:248-261):
```
1. coordinator_.commit(...)  → currentWorld_ = newWorld (release)  [metadata baked onto pubWorld first]
2. if state_ == Faulted → return nullptr (no swap)  [both stay in sync]
3. writeAccess_.publishAndSwap(next) → RuntimeStore::current = newWorld (acq_rel)  [physical swap]
```

**INV-X4-6 satisfied**: After step 3, both pointers point to `newWorld`. If `commit()` fails (step 2), both stay pointing to the previous world. No desync path exists.

**RT reader safety**: `observe()` (acquire load) sees `newWorld` only after both `commit()`'s release (step 1) and `publishAndSwap()`'s acq_rel (step 3). The `PublicationSemantic` metadata is baked onto `newWorld` at cpp:110 before `currentWorld_` update — readers always see consistent (world + metadata).

### Appendix G: §2.2 — Reclaim Call Sites Complete Inventory

**3 production call sites for `RuntimeIntentCoordinator::reclaim(ReclaimMode::ShutdownQuiescent, ...)`:**

| File:Line | Context | readerRegistrationClosed source |
|-----------|---------|--------------------------------|
| AudioEngine.Processing.ReleaseResources.cpp:423 | `dspHandleRuntime_.retire(activeHandle)` → `reclaim(ShutdownQuiescent, ...)` | `m_epochDomain.readerRegistrationClosed()` |
| AudioEngine.Processing.ReleaseResources.cpp:433 | `dspHandleRuntime_.retire(fadingHandle)` → `reclaim(ShutdownQuiescent, ...)` | `m_epochDomain.readerRegistrationClosed()` |
| AudioEngine.h:2032 (CacheMap::~CacheMap) | `resolve` → `delete` → `reclaim(ShutdownQuiescent, ...)` | `owner->m_epochDomain.readerRegistrationClosed()` |

**⚠️ All 3 pass bool directly.** The `reclaim()` function at ISRRuntimePublicationCoordinator.cpp:631 checks `readerRegistrationClosed` internally. Phase A must convert to `ReclaimPermit` object.

**CacheMap::~CacheMap** additionally checks `shutdownPhase >= AudioEngine::ShutdownPhase::Destroy` (AudioEngine.h:2019) before entering the reclaim loop — this is the AudioEngine 7-phase enum, not the ISR 11-phase enum. The two enums are non-1:1 mapped (see Amendment 5).
