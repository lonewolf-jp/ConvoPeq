# REPAIR_PLAN2-dash2 将来対応・一部実装の詳細設計

**★ 作成日:** 2026-08-11
**★ 対象:** `doc/work88/REPAIR_PLAN2-dash.md`（本実装は完了済み）で「将来対応」「一部実装」とされた項目の**整理・詳細設計**
**★ 前提:** REPAIR_PLAN2-dash.md の本実装（P2-1〜P2-4 / X1〜X6 / X4-B / X3-R4 Phase 7）は完了済み・ctest 28/28 PASS・CI 検証 PASS
**★ normative source:** REPAIR_PLAN2-dash.md（現行版・A-2.42 以降 + §5/§6 の「現在形」記述）＋実コード照合

> **⚠️ 版管理メモ（2026-08-14 更新）**: 本ファイルは 2026-08-13 23:58 時点のスナップショット（`REPAIR_PLAN2-dash2(20260813-235851).md`）をベースに、2026-08-14 の実コード照合・第三者的〜第十四者レビュー反映を追記している。**本文中の「2026-08-14 追加」とマークされた記述、および H.6〜H.18 の検証セクションは 8/14 の追記レイヤー**であり、8/13 23:58 時点の immutable design baseline とは別のレイヤーとして扱うこと。実装仕様書として使用する場合は、8/14 追記（H.6〜H.18）を別 revision に分離して版を固定するのが安全。

---

## 実装状況サマリ

| 分類 | 項目 | 現状 | ISR review 対応 |
|---|---|---|---|
| 将来対応 | 1.1 R1: recoveryIntentQueue_ の MPSC 化 | Phase 5 将来拡張（未実装） | 🟢 条件付きGO — INV-R1-1/INV-R1-2 + AC-ISR-1 追加 |
| 将来対応 | 1.2 Recovery coalesce（マージ）実装 | 四次レビュー NO-GO → 別タスク（P3） | 🟡 **条件付きGO（2026-08-14 第四者レビュー）** — LogicalRecoveryIdentity + RecoveryProvenance + SupersessionDecision + lease state machine + bounded durable table として実装すれば GO 可能。単純な「same handle = 最新」は NO-GO |
| 将来対応 | 1.3 §4.3: ConvolverProcessor の LinearRamp 分離 | 対象外・文書化（未実装） | 🟢 **GO** — ✅ 実コード照合 (2026-08-14): **RT contract violation は既に解消済み**。`Runtime.cpp:360` は `resetRT()` (RT-safe, `ASSERT_AUDIO_THREAD`) を呼ぶ。Phase 0.5 は実装済み（`DspNumericPolicy.h:341` `resetRT()` + generation handshake）。対象外・文書化の維持で可 |
| 将来対応 | 1.4 isFullyDrained の他カウンタ実測上書きの全廃 | 別タスク（P2 範囲外） | 🟡 設計先行必須 — 16-condition drain semantic (reclaimInFlight + publicationIntentResidency 追加) |
| 将来対応 | 1.5 PublishReceiptWaiter の sparse completion | 将来 MPSC completion 許容時のみ | 🟢 将来保留 — FIFO invariant を維持 |
| 将来対応 | 1.6 X2 の wraparound / out-of-order テスト | 将来 sparse 化時のみ追加 | 🟢 GO — sequence arithmetic の定義も必要 |
| 将来対応 | 1.7 X4-B 案2（currentWorld_ 廃止） | 将来タスク（read-source singularization） | 🟡 高リスク — AC-PUB-1 (identity consistency) が前提 |
| 将来対応 | 1.8 BuildError 保険分類 | enum + toString のみ・catch 拡張未実装（work32 Step 3 未対応) | 🔴 現案NO-GO → 1.8.5.2 FailureClassification/RetryDisposition 分離. ⚠️ No exponential backoff in retry (RebuildDispatch.cpp:990-1048, `kMaxRecoveryConsecutiveFailures=4` only) |
| 将来対応 | 1.9 初回 publish 前 quarantine の無駄な起床 | Phase 5 最適化候補 | 🟡 条件付きGO — 3-condition wake + lost-wake proof |
| 一部実装 | 2.1 R4: retire 順序逆転の完全解消 | runtime 経路対応済み・完全解消は保留 | 🟡 条件付きGO — epoch safety と FIFO を分離 (UAF/リークは既に排除済み) |
| 一部実装 | 2.2 shutdown lifetime contract の明文化 | R4 詳細設計で ShutdownQuiescenceProof 確定済み | 🟢 強く GO — ReclaimPermit pattern (ShutdownRuntime only)。**2026-08-14 更新**: H.11.11〜H.11.18 で Proof 条件 Q0〜Q7 / ShutdownCompletionProof（C1〜C7）/ ShutdownCompletionAuthority / EpochQuiescenceEvidence / ShutdownAdmissionState 7-state / AUTH 群 へ発展。実装時は H.11 の最新定義を正とする（A2-G01〜G23 PASS まで production reclaim 接続は NO-GO） |

---

## 総合判定 (ISR Architecture Review 2026-08-11)

| 項目 | 判定 | ISR観点 |
|---|---|---|
| 1.1 Recovery Queue MPSC化 | 🟢 条件付きGO | RT producer禁止を明文化した点は正しい |
| 1.2 Recovery coalesce | 🟡 条件付きGO（2026-08-14） | LogicalRecoveryIdentity + SupersessionDecision 実装で GO 可能。単純 latest-wins は NO-GO（1.2.1/1.2.2 参照） |
| 1.3 LinearRamp owner audit | 🟢 GO | 変更せず監査だけなら安全 |
| 1.4 isFullyDrained再設計 | 🟡 設計先行必須 | shutdown/lifetime safetyに直結。**2026-08-14**: 観測 predicate 化（H.11.11.1 INV-DRAIN-1/2）、Completion 側で C1〜C7 検証（H.11.11.9.3 / H.11.15.7） |
| 1.5 sparse completion | 🟢 将来保留 | 現行FIFO invariantを崩す必要なし |
| 1.6 sequenceテスト | 🟢 GO | sparse化時に必要 |
| 1.7 currentWorld_廃止 | 🟡 高リスク | ISR read semantic変更を伴う |
| 1.8 BuildError wiring | 🔴 現案のままはNO-GO | retry分類が粗すぎる（1.8.5.2 で分離追加済み） |
| 1.9 quarantine wake最適化 | 🟡 条件付きGO | lost wake防止の証明が必要（1.9 AC で追加済み） |
| 2.1 retire順序FIFO化 | 🟡 条件付きGO | epoch safetyとFIFOを混同しない（2.1 で修正済み・INV-EPOCH-1/2 / INV-FIFO-1） |
| 2.2 shutdown lifetime proof | 🟢 強くGO | ISR lifetime safetyを強化する。**2026-08-14**: Proof 条件 Q0〜Q7 / ShutdownCompletionProof（C1〜C7）/ ShutdownCompletionAuthority / EpochQuiescenceEvidence / ShutdownAdmissionState 7-state（H.11.11〜H.11.18）へ発展。実装時は H.11 の最新定義を正とする（A2-G01〜G23 PASS まで production reclaim 接続は NO-GO） |

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
- `recoveryIntentQueue_` は `LockFreeRingBuffer<RecoveryIntent, 256>`（SPSC、ISRRuntimePublicationCoordinator.h:551）
- **primary 経路**（2026-08-14 ソース照合で修正）: `QuarantineIntentHandler → submitRecoveryIntent → submitRecoveryRequest → recoveryIntentQueue_（Builder Work Queue）→ Builder Loop が popRecoveryRequest で消費`（RebuildDispatch.cpp:911）。`RecoveryIntentHandler`（intentQueue_ 経由、ProcessIntent.cpp:151-154）は **dead code** — intentQueue_ には誰も Recovery Intent を push しない（ProcessIntent.cpp:131 注記・line 2179 と整合）
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

**問題:** `submitRecoveryRequest` (cpp:721-780) は2つのパスを持つ:
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

#### 1.2.3 ✅ 2026-08-14 第四者レビュー反映 — Recovery coalesce を条件付き GO にする設計

**単純な「same handle = 最新を残す」は NO-GO のまま**だが、以下の **GO 条件（R1〜R17）** を満たせば coalesce 実装は可能。方向転換の根拠は:
- `LogicalRecoveryIdentity`（identity = 「何を復旧するか」）
- `RecoveryProvenance`（provenance = 「なぜこの要求が存在するか」）— identity に含めない
- `SupersessionDecision`（semantic superset 判定）
- lease 型 state machine（`DurablePending → Building`）
- **bounded durable recovery table**（single durable slot の修正）

**重要: `pendingRecoveryAdmission_`（h:590）は single durable slot**（plain struct, SPSC）。`CanSupersede == false` の場合に複数 logical obligation を保持できるよう、**bounded durable table**（例: `kMaxDurableRecoveryAdmissions`）への拡張が必要。**coalesce できない Recovery を「できないから捨てる」ことは禁止**。

**GO 条件（R1〜R17）**:

```text
R1  LogicalRecoveryIdentity implemented（handle + generation + build identity + epoch — admission source 含めず）
R2  RecoveryProvenance separated（Transport / Durable / Retry / Quarantine）
R3  SupersessionDecision implemented（bool でなく理由を返す enum: CanSupersede / DifferentHandle / DifferentSemanticTarget / NotSameGenerationDomain / NotSemanticSuperset）
R4  DurablePending/Building lease implemented（take は destructive dequeue でなく state transition）
R5  exactly-one-logical-obligation invariant
R6  transport/durable double counting impossible
R7  different semantic target cannot coalesce
R8  different handle cannot coalesce
R9  generation domain mismatch cannot coalesce
R10 non-superset cannot coalesce
R11 transient failure restores durable obligation（Building → DurablePending、reservation 再発行なし）
R12 successful build consumes exactly one logical obligation
R13 queue-full does not lose recovery（durable fallback）
R14 coalesce does not increase reservation（INV-X1-5）
R15 coalesce does not delete a non-superseded recovery
R16 shutdown closes RecoveryAdmission（RecoveryAdmissionClosed）
R17 BuilderStopped participates in shutdown proof
```

**最重要 Invariant**: **「1 LogicalRecoveryIdentity = 0 または 1 個の live obligation」**。この Invariant により、duplicate reservation / double accounting / lost recovery / stale recovery / durable-transport 二重計上を構造的に防ぐ。

**`intentId` は identity ではない**: `intentId` は sequence/diagnostic に過ぎない。`intentId A != intentId B` でも `LogicalRecoveryIdentity A == B` なら coalesce 可能。逆に `intentId` 一致だけで同一とはしない。

**coalesce は Audio Thread では行わない**: producer は CoordinatorLoop（NonRT）のみ。Queue は transport のまま、semantic admission/coalesce は RecoveryAdmission 側に置く。

---

## 1.3 §4.3: ConvolverProcessor の LinearRamp 分離

**現状（実コード照合）**
- `ConvolverProcessor.h:910,935,945` の `latencySmoother` / `crossfadeGain` / `mixSmoother` は、CrossfadeRuntime の `gain_` / `dryScaleGain_` とは**別個**の `LinearRamp` (`DspNumericPolicy.h:319-406`)
- **✅ 2026-08-14 実コード照合 (再検証)**: **RT contract violation は既に解消済み**。`ConvolverProcessor.Runtime.cpp:360` は `activeMixSmoother.resetRT(...)` (RT-safe, `ASSERT_AUDIO_THREAD` at `DspNumericPolicy.h:343`) を呼び出す。`resetRT()` は `DspNumericPolicy.h:341-347` で定義され、`reset()` と同一の totalSteps 更新ロジックだが `ASSERT_NON_RT_THREAD` を持たない（§1.3-B 修正済み）。generation handshake (`mixSmootherResetPendingGen` + `smoothingTimeChangePendingGen`) が NonRT → RT 通知を担当する。
- 設計判断として**対象外・文書化**（四次レビュー承認） — RT violation は既に解消済みのため、Phase 0.5 は不要（将来の LinearRamp 抽象化は文書化のまま）

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
   - リスク: **Low** — ✅ 実コード照合 (2026-08-14): **既知の RT contract violation は存在しない**。`Runtime.cpp:360` は `resetRT()` (RT-safe, `ASSERT_AUDIO_THREAD`) を呼ぶため、Phase 0.5 修正は不要（既に実装済み）。
   - **AC-ISR-1**: Audio Thread は LinearRamp の `reset()` / `setCurrentAndTargetValue()` (NonRT-only) を呼び出さない。`applyImmediateValueRT()` / `resetRT()` / `setTargetValue()` / `getNextValue()` (RT-only) のみ呼び出す。✅ 現行コードはこれを満たす。

  **✅ 実コード照合 (2026-08-14 再検証) — LinearRamp RT write 違反: 0 件（解消済み）**

  LinearRamp (`DspNumericPolicy.h:319-406`) のメソッド別スレッド属性（**2026-08-14 実測の line ref に更新**）:

  | Method | ASSERT | NonRT write | RT write |
  |---|---|---|---|
  | `reset()` | `ASSERT_NON_RT_THREAD` (h:333) | ✅ NonRT (prepareToPlay) | ❌ (RT からは呼ばれない) |
  | `resetRT()` | `ASSERT_AUDIO_THREAD` (h:343) | ❌ | ✅ RT only（§1.3-B 修正で追加） |
  | `setCurrentAndTargetValue()` | `ASSERT_NON_RT_THREAD` (h:356) | ✅ NonRT only | ❌ |
  | `applyImmediateValueRT()` | `ASSERT_AUDIO_THREAD` (h:367) | ❌ | ✅ RT only |
  | `setTargetValue()` | `ASSERT_AUDIO_THREAD` (h:378) | ❌ | ✅ RT only |
  | `getNextValue()` | `ASSERT_AUDIO_THREAD` (h:389) | ❌ | ✅ RT only |
  | `getCurrentValue()` | none | ✅ | ✅ (const, no assert) |
  | `getTargetValue()` | none | ✅ | ✅ (const, no assert) |

  **✅ 再検証結果 — `mixSmoother.reset()` の RT 呼び出しは存在しない (2026-08-14)**

  `ConvolverProcessor.Runtime.cpp:360` は `activeMixSmoother.resetRT(sampleRate, newTime)` を呼び出す — これは **RT-safe 版**（`ASSERT_AUDIO_THREAD()` at `DspNumericPolicy.h:343`）であり、`reset()`（`ASSERT_NON_RT_THREAD`）ではない。したがって **Debug builds での jassert 違反は発生しない**。

  起動経路（実コード確認済み）: `setSmoothingTime()` (NonRT, Runtime.cpp:928) が `smoothingTimeChangePendingGen` を fetchAdd (acq_rel) → RT スレッドの `process()` が generation を acquire で検知 (Runtime.cpp:350-353) → `resetRT()` (h:360) + `applyImmediateValueRT()` (h:361) + `setTargetValue()` (h:362) を呼び出す。全て RT-safe API。

  `DspNumericPolicy.h:338-341` のコメントが明示: "Audio Thread から呼ぶ RT-safe 版 reset。reset() と同一の totalSteps 更新ロジックだが ASSERT_NON_RT_THREAD を持たない。smoothingTimeChangePendingGen ハンドシェイク経由で reset() が RT スレッドへ渡行していた問題（§1.3-B）を解消する。" — **既に修正済み**。

  **🔴 現段階: 実装不要 (対象外・文書化) — 修正済みのため Phase 0.5 は不要**
  - 将来 LinearRamp 抽象化実装時 (1.3 の将来タスク) に各 smoother の独立 RT-safety 検証を継続
  - `totalSteps` は RT で `resetRT()` により更新される（`ASSERT_AUDIO_THREAD` あり、data race なし）
  - 非 RT の `setCurrentAndTargetValue()` は Lifecycle.cpp:371 でのみ呼び出され ✅

  **read/write 箇所棚卸し (2026-08-11 初版 / 2026-08-14 line ref 更新):**

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
  | mixSmoother | Runtime.cpp | 358-362 | `getCurrentValue()` / `getTargetValue()` / **`resetRT()`** / `applyImmediateValueRT()` / `setTargetValue()` | **RT ✅ (resetRT は RT-safe)** |
  | mixSmoother | Runtime.cpp | 368 | `getTargetValue()` / `setTargetValue()` | RT ✅ |
  | mixSmoother | Runtime.cpp | 373 | `setTargetValue()` | RT ✅ |
  | mixSmoother | Runtime.cpp | 601 | `getNextValue()` | RT ✅ |

  ---
## 1.4 isFullyDrained の他カウンタ実測上書きの全廃

- `AudioEngine::isFullyDrained`（Threading.cpp:114-156）が、`fallbackBacklog` / `retireBacklog` / `deferredRetire` / `quarantineResident` を**外部 setter で実測上書き**している
  - ⚠️ **実際のline refは Threading.cpp:114-156**（document says :117,131 — close enough but :127-129 is where the 3 setters are called）
- `AudioEngine::isFullyDrained`（Threading.cpp:114-156）は **2層構造**:
  - **Layer 1 (AudioEngine)**: `!hasDeferredCommit`（:116 via hasDeferredRequest）+ `pendingReclaimHandles_.empty()`（:148, mutex-guarded）+ `overflowRing.residentCount()`（:134-135）+ `dspQuarantineManager.residentCount()`（:136）+ `retireRouter.quarantineResidentCount()`（:137-138）+ `runtimePublicationBridge_.isFullyDrained()`（:156）
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

**✅ 2026-08-14 第三者的レビュー反映（§1.4 設計方針の補強）**:

1. **setter は「private化」ではなく「意味ごと廃止」**: `private: void setRetireBacklogCount(...)` にして Coordinator 内部からの snapshot overwrite（`setRetireBacklogCount(actualQueueSize)`）が残ると authority 問題は残る。**setter API そのものを削除**する。
2. **snapshot accounting に戻さない**: setter 削除後に `retireBacklogCount_.fetch_add()/fetch_sub()` を各所に散らすと authority が再分散する。**semantic event API**（`onRetireAccepted()` / `onRetireConsumed()` 等）へ閉じ込める。
3. **underflow 防止**: `onRetireConsumed()` の `fetch_sub` 前に `old > 0` を検証（0 で fetch_sub すると UINT64_MAX になる）。違反時は Faulted → Proof 生成不能へ遷移。
4. **`setQuarantineResidentCount()` は domain mixing**: ReleaseResources.cpp:291 の `setQuarantineResidentCount(ringResident)` は overflow ring resident（retire 系）を quarantine カウンタに混ぜている。`DSPQuarantineManager::residentCount()` を直接 source にする（H.6.1 / H.9.2 参照）。

#### 1.4.1 ⚠️ レビュー指摘: `queue size == 0` だけでは不十分

**現行コードの16条件 (ISRRuntimePublicationCoordinator.cpp:500-526 — return body 500-525 + 終了 526; swapPending_ pre-check は 486):**

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
- **INV-ISFULLDRAINED-1**: transport residency == 0 (publication + recovery + quarantine intent transports; intentQueue_ + observeDeferredRing_ + quarantineFallbackQueue_ + recoveryIntentQueue_ + publicationIntentResidencyCount_ + quarantineIntentResidencyCount_ + quarantineRingResidencyCount_)
- **INV-ISFULLDRAINED-2**: producer reservations == 0 (pendingIntentCount_ — reservation hole)
- **INV-ISFULLDRAINED-3**: durable admission == none (recoveryAdmissionPending_ == false)
- **INV-ISFULLDRAINED-4**: retire/backlog/deferred == 0 (retireBacklogCount_ + fallbackBacklogCount_ + publicationBacklogCount_ + deferredRetireResidencyCount_ + reclaimInFlightCount_)
- **INV-ISFULLDRAINED-5**: quarantine physical residency == 0 (quarantineResidentCount_ from DSPQuarantineManager)

> **⚠️ 2026-08-14 第六パス照合**: 上記 INV-ISFULLDRAINED-1〜5 は現行 16 条件（return body 15 条件）を semantic 分類したもので、`publicationBacklogCount_`（条件 7）と `fallbackBacklogCount_`（条件 10）は INV-ISFULLDRAINED-4（backlog 系）に含めるべきところ、旧記述では欠落していた。修正済み。

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
- `PublishReceiptWaiter`（AudioEngine.h:3632）は **high-water mark（`lastCompleted_`）+ contiguous completion**（INV-X2-6: completion order == publication sequence order）
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
- out-of-order は **PublishExecutor sole gateway である限り不要**（INV-X2-5/INV-X2-6 at AudioEngine.h:3621-3627）

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

BuildError は**build() / validateWarmup() の 2 関数でのみ**使用される。メインの publish パス `buildRuntimePublishWorld()`（RuntimeBuilder.h:134-196, RuntimeBuilder.cpp:179-426）は **BuildError を一切使用しない**。

| 関数 | ファイル | noexcept | BuildError 使用 | 呼び出し元 |
|---|---|---|---|---|
| `build()` | h:206-207, cpp:428-469 | ✅ | ✅（BuildResult.error） | RebuildDispatch.cpp:941, :1015, :1087 の 3 サイトのみ |
| `validateWarmup()` | h:209, cpp:471-479 | ✅ | ✅（直接返り値） | RebuildDispatch.cpp:955, :1032, :1143 の 3 サイトのみ |
| `buildRuntimePublishWorld()` | h:134-196, cpp:179-426 | ✅ | ❌（別経路） | PrepareToPlay.cpp:149,271; ReleaseResources.cpp:169; Transition.cpp:22; Timer.cpp:912; Init.cpp:53; etc. |

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

> **⚠️ 2026-08-14 H.11.2 との関係明確化**: 本 1.8.5.2 の `BuildFailureClass` / `RetryDisposition`（Retry / Discard / Inspect / ContextDependent）は**実装詳細レイヤー**、H.11.2 の `FailureClassification`（Permanent / Transient / Infrastructure / Fatal）+ `RetryDisposition`（NoRetry / RetryBackoff / RetryImmediate）は**概念的分類レイヤー**。
> 実装時は descriptor table（§1.8.10.3 / line 1059-1066）を single source of truth とし、各エントリに H.11.2 の FailureClassification を付与する。**両者の RetryDisposition enum 値は統一する**（Retry→RetryBackoff、Discard→NoRetry、Inspect→ContextDependent）。
> MKLFailure / ConvolverFailure / PrepareFailure は「build() で返り得ない」（本マトリクス）ため、H.11.2 の retry mapping は defensive な予定表である（H.11.2 注記・§1.8.12 項目3 参照）。

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

#### 1.8.10.1 検討されたアプローチ（static_assert による enum ordering 検証 — 未実装）

> **✅ 2026-08-14 実コード照合**: `BUILD_ERROR_ENUM_COUNT` マクロおよび下記 static_assert チェーンは**ソースコードに存在しない**（`grep -rn "BUILD_ERROR_ENUM_COUNT" src/` → 0 件、`RuntimeBuilder.h/.cpp` に static_assert なし）。§1.8.9 Phase 1 で「追加」する予定のコードであり、「現在のアプローチ」ではない。**本節は「検討案」として修正**。

```cpp
// ★ C-2: toString の enum 網羅性をコンパイル時検証（提案 — 未実装）
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

検討中の `static_assert` チェーンは `BuildError` 値が `0, 1, 2, ..., 7` の順であることのみを検証する。`toString` の switch から `case` が削除された場合（例: `case BuildError::MKLFailure: return "MKLFailure";` を削除）、enum 値は存在するため **static_assert は全て通過し**、`toString` は静かに `return "Unknown"` に落ちる。

**Code verification confirms:** `RuntimeBuilder.cpp:53-76` — `toString` has **no `default:` case** in its switch; it falls through to `return "Unknown"` (line 75). This is exactly the vulnerability.

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
| 1 | `DSPCore::prepare()` 内部の各サブシステムの失敗モードを列挙 | `AudioEngine.Processing.DSPCoreLifecycle.cpp:72` を解析 | **✅ 調査完了** | `prepare()` は**void** を返す (AudioEngine.h:866)。内部で呼び出す全サブシステムも void: `ramp.prepare()`、`oversampling.prepare()`、`convolverState->prepare()`、`eqState->prepare()`、`dcBlockers().init()`、`noiseShaper prepare`、`outputFilter.prepare()`、`truePeakDetector.prepare()`、`loudnessMeter.prepare()`、`peakLimiter.prepare()`。**いずれも失敗を返さない** — allocation 失敗は `std::bad_alloc` 例外 (catch で `ResourceUnavailable` に変換済み at `RuntimeBuilder.cpp:459-462`)、他のサブシステムの内部失敗はログ出力のみ (juce::Logger) または黙殻。**ステータス型導入にはすべてのサブシステムの戻り値変更が必要**。 |
| 2 | `applyBuildSnapshot()` / `transferIRStateFrom()` の失敗検知 | `ConvolverProcessor.h:477,1132` + `ConvolverProcessor.StateAndUI.cpp:271` を解析 | **✅ 調査完了** | `applyBuildSnapshot()` — void、atomic publish のみ (ScopedLock + publishAtomic)。例外なし。`transferIRStateFrom()` — void、IR AudioBuffer コピー。成功/失敗を `juce::Logger::writeToLog` でログ出力 (`[CONV_IR] transferIRStateFrom: IR transferred/failed/no IR data`) が、戻り値なし。**いずれもステータスチェック不能 — 将来的な status-code wiring には戻り値追加が必要**。 |
| 3 | MKL/IPP の使用箇所とステータスコード | `MKLNonUniformConvolver.h` + `ConvolverProcessor.Lifecycle.cpp:276-279` を解析 | **✅ 調査完了** | **⚠️ FFT は MKL DFTI → Intel IPP に換装済みだが、MKL 自体は VML/BLAS で継続使用中**。`MKLNonUniformConvolver.h:5` コメント「v2.0 FFT backend を MKL DFTI → Intel IPP に換装」。Audio Thread 内 FFT は **Intel IPP** (`IppsFFTSpec_R_64f`, `ippsFFTFwd_RToCCS_64f`, `ippsFFTInv_CCSToR_64f`) で実装 — IPP は `IppStatus` コードを返す (ステータスコードベース、C++ 例外なし)。**ただし MKL VML (`vdMul`, `mkl_vml.h`) / MKL BLAS (`cblas_dscal`, `mkl_cblas.h`) は引き続き Message Thread で使用** (MKLNonUniformConvolver.cpp:27-30,54-55)。`newConv->init()` (L276-279) は bool を返す — `false` 時は log + existing engine を保持する fallback (L288-295)。**したがって `MKLFailure` は現在どのコードパスからも生成されない** (build() は `InvalidInput`/`ResourceUnavailable`/`InternalError`/`None` のみ返す)。 |
| 4 | `publishRuntimeProcessSnapshot()` の戻り値 | `ConvolverProcessor.h:1027` を解析 | **✅ 調査完了** | `void` を返す。`runtimeProcessSnapshots[next]` への書き込み (`pendingOverride` からの ScopedLock + atomics) は全て成功する (allocation なし)。**エラー伝播パスなし** — この関数は failure を返しません。 |
| 5 | `buildRuntimePublishWorld` の failure handling | `RuntimeBuilder.cpp:179-426` を解析 | **✅ 調査完了** | `buildRuntimePublishWorld` は `BuildError` を**使用しない** — 内部で try/catch し `worldOwner` を返す (失敗時 nullptr)。**設計の意図: `build()` (DSPCore construction) と `buildRuntimePublishWorld()` (World assembly) は責務分離**。`build()` は DSPCore + IR + prepare (BuildError で分類可能)。`buildRuntimePublishWorld` は World の構成・トポロジ — ここでの failure は構造的問題 (internal) なので InternalError で統一されている。**将来的な分離は不要 — 現設計で正しい**。 |
 | 6 | `build()` の `diagLog` / diagnostics guard | `RuntimeBuilder.cpp:428-469` + `RebuildDispatch.cpp` を解析 | **✅ 調査完了** | `build()` は **`#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` ガードなし**。try/catch (`std::bad_alloc` → ResourceUnavailable, `...` → InternalError) は **常に有効** (production code)。`diagLog` は `RebuildDispatch.cpp:1073/1093` で使用されるが、**build() 自体には診断 guard なし** — build() の `catch` は本番環境でもアクティブ。 |
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
- `submitRecoveryRequest` (ISRRuntimePublicationCoordinator.cpp:721-780) は reservation + push または durable admission への fallback を行う
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
- ⚠️ **INV-R9-1（obligation conservation — 第五者レビュー #27 反映）**: 「Recovery obligation exists」は wake 状態（recoveryPending / notify）とは独立して常に true を維持する。wake 最適化は obligation を見失ってはならない — silent absorb は transport / durable / pending の**いずれかに必ず留まる**。
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

**INV-EPOCH-1/2（primary）・INV-FIFO-1（secondary）の明文化（第四者レビュー §8 反映）**:

```text
INV-EPOCH-1: retire 済み handle の reclaim 可否は retireEpoch と minReaderEpoch の比較のみで決定する
INV-EPOCH-2: reader が grace period 内に居る限り reclaim 禁止（retire 順序とは無関係）
INV-FIFO-1: retire の処理順序は epoch 順を目指す（optimization / determinism / telemetry）— memory safety ではない
```

> **RCU 整合性（[ISO C++ P0279R1 "Read-Copy Update (RCU) for C++"](https://isocpp.org/files/papers/p0279r1.pdf)）**: RCU では retire/reclaim の安全性は read-side critical section と grace period の関係で決まり、**単なるキュー処理順序ではない**。本設計の「Epoch safety primary / FIFO secondary」はこの RCU モデルと整合する。

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

> **AC-R4-1〜10 個別定義（REPAIR_PLAN2-dash.md:2415 から引用・2026-08-14 第六パスでソース整合確認）**:
> - **AC-R4-1**: shutdownReclaim() call sites == 0（X_IMPL_CHECKLIST.md:118 で ✅ 完了確認）
> - **AC-R4-2**: shutdownReclaim() symbol absent（同 ✅）
> - **AC-R4-3**: DSPHandleRuntime::reclaim() の production caller が ReclaimAuthority のみ
> - **AC-R4-4**: RuntimeEBR / ShutdownQuiescent の両 mode で same physical reclaim primitive
> - **AC-R4-5**: ShutdownQuiescent に quiescence proof 必須
> - **AC-R4-6**: epoch unsafe な RuntimeEBR handle は pendingReclaimHandles_ に残る
> - **AC-R4-7**: Faulted で pending を clear しない
> - **AC-R4-8**: Audio Thread から reclaim authority を呼ばない
> - **AC-R4-9**: reclaim 自体は RT thread で実行しない
> - **AC-R4-10**: isFullyDrained() は reclaim authority にならない（§2.2 / H.9.1 と整合）

**影響範囲 / リスク**
- 影響: `ISRRetireRouter`、`AudioEngine.Retire.cpp`（drainDeferredRetireQueues）、`ReleaseResources.cpp`
- リスク: **中**（retire 順序の変更は epoch 安全性に影響。runtime 経路は既に対応済みのため、残余は順序保証の強化）

---

## 2.2 shutdown lifetime contract の明文化

**現状**
- REPAIR_PLAN2-dash.md §4（:837-846）で「shutdown lifetime contract の明文化」を**将来タスクとして記録**
- R4 詳細設計で **ShutdownQuiescenceProof**（admissionClosed / producersJoined / coordinatorStopped / builderStopped / audioStopped / readerRegistrationClosed / readersZero / epochSettled）が**答えとして確定済み**
- ⚠️ **2026-08-14 整合注記（第五・七・九者レビュー）**: 上記の旧 Proof 条件（8 条件）は **Q0〜Q7 に改編**（H.11.11.3）。`pendingReclaimIdentities.empty()` / `LifetimeAccounting.isDrained()` は Proof から除外し **ShutdownCompletionProof（H.11.11.9.3 C1〜C7）** へ移動（循環排除）。EpochSettled は `EpochQuiescenceEvidence`（H.11.13.3）として証明。本 2.2 は 8/13 baseline の記述であり、実装時は H.11 の最新定義を正とする。
- `readerRegistrationClosed`（EpochDomain.h）は実装済み（X3）・`reclaim(ShutdownQuiescent)` の precondition は実装済み
- **⚠️ 2026-08-11 追加**: `ShutdownQuiescenceProof` / `ReclaimPermit` オブジェクトは**現在コードに存在しない**。現行は `reclaim(..., bool readerRegistrationClosed=false)` のboolパラメータで証明を受け取っている（ISRRuntimePublicationCoordinator.h:373-377）。3つのproduction call site（ReleaseResources.cpp:423,433 / AudioEngine.h:2032）が全て `m_epochDomain.readerRegistrationClosed()` をboolで渡している — callerが `true` を偽って渡す可能性あり（Appendix G / H.6.3 詳細調査確認済み）。

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
 bool を直接受け取っている。このパターンは形骸化の危険 —
 caller が `true` を偽って渡すことが可能。
 → 将来的に ShutdownQuiescenceProof / ReclaimPermit オブジェクトを渡すべき。
```

**⚠️ レビュー指摘: ShutdownQuiescenceProof は bool の wrapper にしない**

> 各 bool を外から渡して `valid()` すると、proof が形骸化する。
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

**✅ 2026-08-14 第三者的レビュー反映（§2.2 設計方針の補強）**:

1. **Proof 生成を単一 transaction に**: `ShutdownRuntime::tryMakeQuiescenceProof()` が全条件（admissionClosed / allProducersJoined / readerRegistrationClosed / activeReaders==0 / epochSettled / lifetimeAccounting().isDrained() / pendingReclaimHandles().empty() / postStopEnqueueCount==0 / noResurrection）を**自身で authority から取得して検証**する。`if (isFullyDrained()) return ShutdownQuiescenceProof{};` のような簡易生成は不可。
2. **ReclaimPermit は move-only single-use**: `ReclaimPermit(const ReclaimPermit&) = delete; ReclaimPermit(ReclaimPermit&&) noexcept = default;` — 二重 reclaim（`reclaim(permit); reclaim(permit);`）を構造的に防止。
3. **Permit identity を ShutdownRuntime に束縛**: `ShutdownRuntimeIdentity` / `shutdownGeneration` / `epochGeneration` / `readerRegistrationGeneration` を埋め込み、reclaim 時に照合。Shutdown N の Permit を Shutdown N+1 で使えないようにする。
4. **Proof 生成後に状態変更を不可逆に**: `Admission = Closed` を不可逆状態にし、`Closed → Open` を禁止（post-proof resurrection 防止）。Proof は「ある瞬間の snapshot」ではなく「新しい obligation が生成されない phase に入った」ことを含む証明とする。
5. **旧 reclaim API を完全削除**: `reclaim(..., bool readerRegistrationClosed)` と `reclaim(..., ShutdownQuiescenceProof)` を共存させず、`reclaim(ShutdownQuiescent, handle, ReclaimPermit&&)` のみにする。`bool` / `ShutdownPhase` / `ShutdownDrainToken` / `isFullyDrained()` のどれも reclaim authorization として使えない。
6. **accounting underflow 防止**: semantic event API（`onRetireAccepted` / `onRetireConsumed` 等）で `fetch_sub` 前に `old > 0` を検証。違反時は Faulted → Proof 生成不能へ遷移。
7. **`isFullyDrained()` は観測 predicate に降格**: 診断・invariant 観測 API として残すが、reclaim authority にはしない。Proof は ShutdownRuntime が全条件を自前検証して生成。

#### 2.2.1 ⚠️ レビュー指摘: 現行コードの ShutdownQuiescent reclaim は `shutdownPhase >= Destroy` で判定（形骸化リスク）

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
## 推奨実装順序 (ISR Architecture Review §21 修正版 + 2026-08-11 review amendments + 2026-08-14 再検証)

**✅ 2026-08-14 再検証**: Phase 0.5 は**実装済み**（`resetRT()` + generation handshake が既に存在）。Phase 0.5 を独立フェーズとして追加する必要はない。下記 Phase 0.5 は「既に実装済みであることの確認事項」として維持する。

Phase 0（現行Invariant固定）: 現行コードの invariants 棚卸し・テスト固定
    ↓
~~Phase 0.5（既知bug修正 — NEW）~~: ✅ 実装済み — `mixSmoother.reset()` RT violation は既に解消（`resetRT()` at DspNumericPolicy.h:341 + generation handshake）。確認のみ。
    ↓
Phase B0/B1/B2（Tier 1: lifetime safety）:  1.4（external setter 9 call sites 撤去 → semantic event accounting、`setReclaimInFlightCount(0)` store → `fetch_sub`。**Phase A1 より先に実施 — Amendment 3 / 第四者レビュー: setter 残存中に Proof を作ると、corrupted accounting を型付き Proof で包装することになる**）
    ↓
Phase A1（Tier 1: lifetime safety）:  2.2（ShutdownQuiescenceProof + ReclaimPermit 型導入 — **type only, production 未接続**。H.11.6 Commit 1-3）
    ↓
Phase B3（Tier 1: lifetime safety）:  4 admission paths（Publication/Recovery/Build/Retire）shutdown gate — Path B `enqueuePublicationIntent` gate 含む（H.11.6 Commit 5-9）
    ↓
Phase A2（Tier 1: lifetime safety）:  production reclaim → Permit only（A2-G01〜G23 PASS 前提。bool reclaim API 削除。H.11.6 Commit 16）
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
**Phase 0.5 の位置づけ（2026-08-14 更新）**: 2026-08-11 review では mixSmoother.reset() の RT assertion violation が
既知bugとされたが、**2026-08-14 実コード照合でこの違反は既に解消済み**であることを確認した。
`resetRT()`（DspNumericPolicy.h:341）と `smoothingTimeChangePendingGen` / `mixSmootherResetPendingGen`
handshake（Runtime.cpp:350-362, 928）により、RT スレッドは常に RT-safe API のみを呼ぶ。したがって
Phase 0.5 は実装不要 — 今後のフェーズ進行に blocker はない。

---

## ✅ 2026-08-14 Review Amendments（再検証結果）

### Amendment 1 (更新): §1.3 LinearRamp — RT violation は既に解消済み

**✅ 2026-08-14 実コード照合: `mixSmoother.reset()` の RT 呼び出しは存在しない。**

`ConvolverProcessor.Runtime.cpp:360` は `activeMixSmoother.resetRT(...)`（RT-safe, `ASSERT_AUDIO_THREAD` at DspNumericPolicy.h:343）を呼び出す。`reset()`（`ASSERT_NON_RT_THREAD`）は NonRT の `prepareToPlay()`（Lifecycle.cpp:370）でのみ呼ばれる。したがって **Debug builds での jassert 違反は存在しない**。

`DspNumericPolicy.h:338-341` のコメントが当該修正（§1.3-B）の実施を明示している。

**Severity（2026-08-14 更新）**: ~~Medium (debug-build jassert violation)~~ → **None（解消済み）**。RT スレッドは `resetRT()`（RT-safe）のみを呼ぶため、jassert 違反・data race とも存在しない。

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

**⚠️ Threading.cpp:126-128 still calls 3 external setters.** This is a Phase B (drain semantic) concern — these setters overwrite the authoritative counters that Phase A (ShutdownQuiescenceProof) and Phase B (drain semantic) depend on. If Phase A proceeds before setters are eliminated, the proof object validates against corrupted data.

### Amendment 4: §1.9 quarantine wake — semantic change requires recovery admission policy

The 1.9 optimization (suppress wake when no authoritative runtime exists) introduces a semantic change: recovery requests may be **silently absorbed** during initial quarantine pre-publish. This must be:
1. Observable via telemetry (quarantineAbsorptionCount_)
2. Proven not to lose valid recovery obligations (INV-X1-2: queue full ≠ Recovery lost)
3. The `hasAuthoritativePublishedRuntime()` check must use `observePublishedWorld() != nullptr` (AudioEngine.h:3556), NOT `runtimePublishWorld_` (which does not exist — §5.3 correction confirmed)

### Amendment 5: §2.2 — AudioEngine::ShutdownPhase vs isr::ShutdownPhase non-1:1 mapping

Confirmed: AudioEngine has 7-phase enum (AudioEngine.h:2533), ISR has 11-phase enum (ISRShutdown.h:25). CacheMap::~CacheMap (AudioEngine.h:2019) checks `AudioEngine::ShutdownPhase::Destroy` — NOT `isr::ShutdownPhase::ReclaimComplete`. The two enums represent different lifecycle layers (Engine vs Coordinator). Any code comparing across enums must bridge via explicit mapping.

### Amendment 6 (更新): Phase ordering — mixSmoother fix は実装済みのため先行不要

The 2026-08-11 review recommended inserting Phase 0.5 before Phase A. **2026-08-14 実コード照合で、この fix は既に実装済み**であることを確認（`resetRT()` at DspNumericPolicy.h:341 + generation handshake at Runtime.cpp:350-362/928）。したがって Phase 0.5 は不要。**Phase 順序は §3（第四者レビュー反映版）に合わせ、B0/B1/B2（setter 撤去）→ A1（Proof/Permit type only）→ B3（gate）→ A2（production reclaim）の順とする。**

```
Phase 0: invariant freeze
（Phase 0.5: mixSmoother RT violation fix — ✅ 実装済み・確認のみ）
Phase B0/B1/B2: external setter 9 call sites 撤去 (1.4) — Phase A1 より先に実施（Amendment 3）
Phase A1: ShutdownQuiescenceProof (2.2) — type only, production 未接続
Phase B3: 4 admission paths shutdown gate（Path B 含む）
Phase A2: production reclaim → Permit only（A2-G01〜G23 PASS 前提）
Phase C: R4 retire (2.1) → Phase D: BuildError (1.8) → Phase E: quarantine wake (1.9)
→ Phase F: MPSC (1.1) → Phase G: currentWorld_ (1.7) → Phase H: sparse (1.5/1.6) → Phase I: coalesce (1.2)
```

**Rationale（更新）**: 2026-08-11 review では `mixSmoother.reset()` を既存の RT contract violation としていたが、実コードは RT スレッドで `resetRT()`（RT-safe）を呼ぶため違反は存在しない。Phase A/B の進行に blocker はない。

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

### AC-LIFE-NEW-1（2026-08-14 第四者レビュー追加）

```text
Proof生成後、lifetime obligationを生成するAPIは存在してはならない。
enqueue / build / publish / recovery / retire registration はすべて
closed admission を尊重する。
```

### AC-LIFE-NEW-2（2026-08-14 第四者レビュー追加）

```text
Proof/Permitのidentityはshutdown transactionのlinearization pointを共有する。
単なる same generation ではなく、
same ShutdownRuntimeIdentity
+ same shutdown generation
+ same epoch generation
+ same reader-registration generation
を要求する。
```

### AC-LIFE-NEW-3（2026-08-14 第四者レビュー追加）

```text
Proof生成後にstate mutationが発生した場合、
Permitは生成不能または即時無効化される。
例: Proof → unexpected post-stop enqueue → Permit を許さない。
Proof は「snapshot」ではなく
「閉鎖されたtransactionのterminal capability」として扱う。
```

---

## 補足
- 本ドキュメント（dash2）の項目は、いずれも**設計・アーキテクチャ上の将来拡張・最適化**です。
- ⚠️ **「現在の ISR パイプラインの正しさには影響しない」は項目ごとには正しいが、全体としては無条件ではない**。特に `isFullyDrained`（§1.4）、shutdown proof（§2.2）、retire順序（§2.1）、BuildError retry（§1.8）を変更すると、ISR 本体ではなくても **ISR から参照される Runtime の lifetime safety を壊し得る**。
- 各項目の実装着手時は、本ドキュメントの Acceptance Criteria をテストとして固定してから実施してください（REPAIR_PLAN2-dash.md の Phase 0 方針と同一）。
- **2026-08-11 追加**: 1.8.12 の 6 課題すべて調査完了。重要な知見:
  - `DSPCore::prepare()` は void — 全サブシステム (ramp/convolver/eq/dither/oversampling/...) が void。ステータス型導入にはすべてのサブシステムの戻り値変更必要
  - `MKLFailure` は**どのコードパスからも生成されない** — Audio Thread 内 FFT は MKL DFTI → Intel IPP に換装済み（ステータスコードベース）。ただし MKL VML/BLAS は Message Thread で継続使用（2026-08-14 照合: MKLNonUniformConvolver.cpp:27-30,54-55）
  - `build()` の try/catch は**production code で常に有効** — `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` ガードなし
  - caller (RebuildDispatch.cpp:1091) は `buildResult.runtime == nullptr` のみチェック — `buildResult.error` は log のみ
  - `publishRuntimeProcessSnapshot()` は void — ログ出力パスなし
  - `buildRuntimePublishWorld()` は BuildError を使用しない — build() (DSPCore construction) と World assembly は責務分離されている
- **2026-08-11 追加 (lifetime)**: 現行コードの `isFullyDrained` は16条件で判定済み（ISRRuntimePublicationCoordinator.cpp:500-525） — `reclaimInFlightCount_` と `publicationIntentResidencyCount_` を含む
- **2026-08-11 追加 (lifetime)**: `reclaim(ReclaimMode::ShutdownQuiescent)` は `readerRegistrationClosed && activeReaderCount()==0` を precondition として実装済み (ISRRuntimePublicationCoordinator.cpp:638-646)
- **2026-08-11 追加 (lifetime)**: `ShutdownPhase` enum (ISRShutdown.h:25) は**11段階** (Running, AudioStopped, ObserverDrained, RetireClosed, EpochSettled, ReclaimComplete, EmergencyDrain, VerifyDrained, TimedOut, Failed, ShutdownComplete) — ⚠️ AudioEngine::ShutdownPhase (AudioEngine.h:2533) は別に7段階 (Running, StopAcceptingWork, StopAudio, StopWorkers, ForceEpochAdvance, DrainRetire, Destroy) 存在。2つのenumは非1:1対応。CacheMap::~CacheMap は AudioEngine::ShutdownPhase::Destroy をチェック
- **2026-08-11 追加 (lifetime)**: `AudioEngine::isFullyDrained` (Threading.cpp:114-156) は Layer 1 (AudioEngine: deferred commit + pendingReclaim + 3 quarantine sources) + Layer 2 (Coordinator: 16 conditions) の 2層構造
- **2026-08-11 追加 (recovery)**: `submitRecoveryIntent` (AudioEngine.h:4285-4297) は**無条件**に `recoveryPending=true + notify_all()` — 1.9 の最適化対象。wake predicate: `hasPendingTask || publishRetryReady || recoveryPending || rebuildThreadShouldExit` (RebuildDispatch.cpp:828-833)
- **2026-08-11 追加 (recovery)**: `submitRecoveryRequest` (ISRRuntimePublicationCoordinator.cpp:721-780) は reservation-before-push + push-success OR durable fallback（queue full） の2-path。`settlePendingRecoveryAdmission(bool retry)` は retry=true で DurablePending→Building レースを管理
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

**AudioEngine::ShutdownPhase** (AudioEngine.h:2533-2541) — **7 values** (int):

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
- `submitRecoveryRequest` (ISRRuntimePublicationCoordinator.cpp:740-741) reads `currentWorld_` from the Coordinator:
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

**Intent dispatch table** (ISRIntentDispatcher.h:60-66):
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

### 5.8 isFullyDrained Layer 1 detail (AudioEngine.Threading.cpp:114-156)

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

### 5.10 Two-layer isFullyDrained — Layer 2 exact conditions (ISRRuntimePublicationCoordinator.cpp:484-526)

All 16 conditions confirmed — **⚠️ correction: the `swapPending_` pre-check is condition 0 (early-return at line 486), NOT listed in the document's 15-item return body**:

- **Condition 0 (pre-check, not in return body):** `coordinator_.swapPending_ == false` (line 486-487 — early return)
- **Conditions 2-16 (in return body, lines 500-525; numbering matches Appendix D):**
2. `intentQueue_.sizeApprox() == 0`
3. `observeDeferredRing_.size() == 0`
4. `quarantineFallbackQueue_.sizeApprox() == 0`
5. `recoveryIntentQueue_.size() == 0`
6. `retireBacklogCount_ == 0` (consumeAtomic acquire)
7. `publicationBacklogCount_ == 0`
8. `publicationIntentResidencyCount_ == 0` (INV-X5-1 addition)
9. `pendingIntentCount_ == 0` (reservation hole)
10. `fallbackBacklogCount_ == 0`
11. `reclaimInFlightCount_ == 0`
12. `deferredRetireResidencyCount_ == 0`
13. `quarantineIntentResidencyCount_ == 0` (X6)
14. `quarantineRingResidencyCount_ == 0` (X6)
15. `quarantineResidentCount_ == 0` (physical DSP objects)
16. `!recoveryAdmissionPending_` (durable admission)

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

### 5.12 ShutdownQuiescent reclaim precondition (ISRRuntimePublicationCoordinator.cpp:638-646)

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

Note: This table lists Threading.cpp-specific call sites. For the full external setter inventory across all files, see H.6.1.

| Setter | h:line | Called from AudioEngine.Threading.cpp |
|---|---|---|
| `setFallbackBacklogCount` | h:124 | ✅ Threading.cpp:126 |
| `setRetireBacklogCount` | h:121 | ✅ Threading.cpp:127 |
| `setPendingIntentCount` | h:123 | ❌ (not called from Threading.cpp — Coordinator internal only) |
| `setDeferredRetireResidencyCount` | h:126 | ✅ Threading.cpp:128 |
| `setReclaimInFlightCount` | h:125 | ❌ (not called from Threading.cpp — Coordinator internal at cpp:660,671, plus 4 external calls from Retire.cpp:48,52,316,319) |
| `setQuarantineResidentCount` | h:127 | ❌ (not called from Threading.cpp — Phase2, called from ReleaseResources.cpp:291, domain mixing — H.6.1/H.9.2 参照) |

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

buildRuntimePublishWorld() (RuntimeBuilder.h:134-196, RuntimeBuilder.cpp:179-426) is noexcept, does NOT return BuildResult or use BuildError. Returns aligned_unique_ptr<RuntimePublishWorld> (nullptr on failure). build() (RuntimeBuilder.h:206-209, RuntimeBuilder.cpp:428-469) is a SEPARATE function handling DSPCore construction with BuildError.

Callers:
- buildRuntimePublishWorld(): PrepareToPlay.cpp:149,271; ReleaseResources.cpp:169; Transition.cpp:22; Timer.cpp:912; RuntimePublicationOrchestrator.cpp:165,232（Init.cpp は createBootstrapWorld()（RuntimeBuilder.cpp:79）を使用 — buildRuntimePublishWorld とは独立した bootstrap 専用構築）
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

recoveryShutdownDiscardCount_ (cpp:735, 820): telemetry only, NOT a drain condition. discardPendingRecoveryAdmission() sets recoveryAdmissionPending_=false which IS drain condition #16 (!recoveryAdmissionPending_, Appendix D / 5.10 の番号付けと整合). Invariant: pendingIntentCount_ tracks transport reservations only. Durable admission tracked separately (INV-X1-6: no double-counting).

### 5.19 Recovery shutdown gate + discard call site

Shutdown gate (submitRecoveryRequest:733): Checks state_ == ShuttingDown BEFORE incrementing pendingIntentCount_. If shutdown, recoveryShutdownDiscardCount_, returns (no reservation, no push).
Graceful discard (discardPendingRecoveryAdmission:816-824): Called from stopRebuildThread() (RebuildDispatch.cpp:798), AFTER Builder thread joined. Clears pendingRecoveryAdmission_ + sets recoveryAdmissionPending_=false.

Call chain: AudioEngine::shutdown() -> stopRebuildThread() (line 771: phase->StopWorkers, notify CV, join) -> discardPendingRecoveryAdmission() (line 798) -> waitForDrain() (Threading.cpp:159) -> isFullyDrained() (Threading.cpp:114) -> Coordinator::isFullyDrained() (ISRRuntimePublicationCoordinator.cpp:484)

**Test coverage**: `ISRSemanticValidationTests.cpp:631` (`testRecoveryDurableAdmission`) covers the full state machine: fill 256 transport slots -> 257th triggers durable admission -> pop 256 transport items -> takePendingRecoveryAdmission (lease DurablePending→Building) → settle(false) (clear) → take returns nullopt (empty). Additional test at line 610 (`testRecoveryRequestEnqueueAndPop`, 定義) covers basic enqueue/pop. Lines 910/914 call `testRecoveryRequestEnqueueAndPop()` / `testRecoveryDurableAdmission()` from the main test entry.

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

This is INV-X5-1 (publicationIntentResidencyCount = Publish intent queue residency + producer reservation). The only production caller of `enqueuePublicationIntent` is `AudioEngine.h:4424` (Path B). The Path A orchestrator route (`enqueuePublicationIntentForRuntimeCommit` → `submitPublishRequest`) does NOT call `enqueuePublicationIntent` — it goes through `trySubmitImpl` → `evaluate` → eventually `processIntent` in the Coordinator loop.

**⚠️ Note**: `enqueuePublicationIntentForRuntimeCommit` (AudioEngine.h:2473, implemented at AudioEngine.Commit.cpp:707) is a HIGHER-LEVEL wrapper that constructs a `PublishRequest` and calls `runtimeOrchestrator_->submitPublishRequest(req)` (Commit.cpp:738) — it does NOT call `enqueuePublicationIntent`. The `submitPublishRequest` path goes through `trySubmitImpl` → `admission_.evaluate()` which has a shutdown gate (`RejectedShutdown` at PublicationAdmission.cpp:11).

The actual `enqueuePublicationIntent` call (Path B, no shutdown gate) is at `AudioEngine.h:4424` — a separate code path that directly calls `runtimePublicationBridge_.enqueuePublicationIntent(intent)` without going through the orchestrator's admission gate. This is the residual No-Resurrection gap.

---

## ⚠️ 2026-08-11 Review: Immediate Fix Candidates (P1)

Per the ISR Architecture Review (2026-08-11), the following are existing contract violations in the current codebase that must be fixed BEFORE Phase A/B implementation:

> **✅ 2026-08-14 再検証注記**: 下記 A-1（mixSmoother RT violation）は実コード照合の結果、**既に解消済み**であることが判明した。A-2〜A-5 は引き続き有効。

### A-1. ~~`mixSmoother.reset()` RT thread violation~~ → ✅ 解消済み (2026-08-14 再検証)
- **File**: `ConvolverProcessor.Runtime.cpp:360` → 実際は `activeMixSmoother.resetRT(...)`（RT-safe）
- **Issue（旧）**: `LinearRamp::reset()` asserts `ASSERT_NON_RT_THREAD()` but is called from audio processing thread → **実コードでは RT スレッドは `resetRT()`（`ASSERT_AUDIO_THREAD`）を呼ぶため違反なし**
- **Severity（更新）**: ~~Medium~~ → **None（解消済み）** — `resetRT()` at `DspNumericPolicy.h:341-347` + generation handshake により jassert 違反なし
- **Fix**: ~~Phase 0.5~~ → **不要（実装済み）**。確認のみ
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

### 🟢 Immediate Design Fix (Phase 0.5) — ✅ 解消済み (2026-08-14)
- ~~**mixSmoother.reset() RT violation**~~ — ✅ 実装済み（`resetRT()` + generation handshake）。今後の Tier 1 作業への blocker なし

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

**⚠️ Correction**: `AudioEngine.h:4285 submitRecoveryIntent` unconditionally calls `rebuildCV.notify_all()` (line 4296). The §1.9 optimization of suppressing this is only safe if the rebuild thread's `recoveryPending` predicate (RebuildDispatch.cpp:831) is preserved. The rebuild thread is started at `AudioEngine.Init.cpp:33` — before Bootstrap World publish (Init.cpp:73). Bootstrap World is published synchronously via `worldAuthority_.publish()` before the CoordinatorLoop starts (Init.cpp:123 startCoordinatorLoop). Therefore, **`observePublishedWorld()` always returns non-null during normal operation** — the "no authoritative runtime" case only exists in the sub-millisecond window between rebuild thread start and bootstrap publish.

### Appendix B: §1.2 — Recovery Coalesce / Identity Tracking

**No existing dedup mechanism:**
- `RecoveryIntent` (ISRRuntimePublicationCoordinator.h:195-218): Contains `DSPHandle handle`, `PublicationEpoch epoch`, `uint64_t intentId`, `RuntimeBuildSnapshot buildSource` — no previous-handle tracking field
- `recoveryIntentQueue_` (LockFreeRingBuffer, 256 slots): Simple FIFO — **no dedup**, multiple recovery requests for the same handle are queued separately
- `PendingRecoveryAdmission` (h:575-593): Has `handle` and `recoveryGeneration` fields. When queue is full → 257th request overwrites `buildSource` and `handle` (latest-wins coalesce). But **no `SupersessionProof`** — the overwrite is silent, no reason recorded

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

**Pre-check (line 486):**
1. `swapPending_ == false` — guards against in-flight publication swap

**Return body (lines 500-525) — 15 conditions:**
| # | Condition | Line | Counter Source |
|---|-----------|------|----------------|
| 2 | `intentQueue_.sizeApprox() == 0` | 500 | Observe/Publish/Quarantine transport (MPSC MpscBoundedRing) |
| 3 | `observeDeferredRing_.size() == 0` | 501 | Observe overflow (SPSC LockFreeRingBuffer) |
| 4 | `quarantineFallbackQueue_.sizeApprox() == 0` | 502 | Quarantine fallback (MPSC MpscBoundedRing) |
| 5 | `recoveryIntentQueue_.size() == 0` | 503 | Recovery (SPSC LockFreeRingBuffer) |
| 6 | `retireBacklogCount_ == 0` | 504 | Retire backlog (set at Retire.cpp:114 + 5 other sites) |
| 7 | `publicationBacklogCount_ == 0` | 505 | Publication backlog (atomic) |
| 8 | `publicationIntentResidencyCount_ == 0` | 509 | Publish intent residency + producer reservation (fetchAdd h:334, fetchSub h:337) |
| 9 | `pendingIntentCount_ == 0` | 510 | Producer reservation hole (fetchAdd at submitObserve:602, submitRecoveryRequest:757, submitQuarantine:904 — **not** enqueuePublicationIntent:h:334, which uses publicationIntentResidencyCount_) |
| 10 | `fallbackBacklogCount_ == 0` | 511 | Fallback backlog (atomic) |
| 11 | `reclaimInFlightCount_ == 0` | 512 | Reclaim in-flight (set at Retire.cpp:48, cleared at :52) |
| 12 | `deferredRetireResidencyCount_ == 0` | 513 | Deferred retire residency (atomic) |
| 13 | `quarantineIntentResidencyCount_ == 0` | 518 | Quarantine intent residency (atomic, incremented at cpp:905, decremented at cpp:917) |
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
   - Written only by `WriteAccess::publishAndSwap()` via `exchangeAtomic(current, next, acq_rel)` (RuntimeStore.h:40)
   - Only `RuntimeWorldAuthority` can call `acquireWriteAccess()` (RuntimeStore.h:88)
   - Read by `observe()` via `consumeAtomic(current, acquire)` (RuntimeStore.h:77)

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

---

## H. Line-Reference Verification Log (2026-08-13 Deep Consistency Pass)

### H.1 Summary of All Corrections Applied

| 項目 | 修正前 | 修正後 | ファイル | 行 |
|---|---|---|---|---|
| `recoveryIntentQueue_` 型定義行 | h:433-434 | h:551 | REPAIR_PLAN2-dash2.md | 61 |
| `ShutdownQuiescenceProof` コメント | cpp:643 | cpp:642 | REPAIR_PLAN2-dash2.md | 2248 (H.5 verification note) |
| `submitRecoveryRequest` currentWorld_ read | cpp:739-740 | cpp:740-741 | REPAIR_PLAN2-dash2.md | 1620 |
| `RecoveryIntent` struct range | h:195-208 | h:195-218 | REPAIR_PLAN2-dash2.md | 2057 |
| `setRetireBacklogCount` 宣言 | h:122 | h:121 | REPAIR_PLAN2-dash2.md | 1832 |
| `buildRuntimePublishWorld` 関数範囲 | h:134-137 | h:134-196 | REPAIR_PLAN2-dash2.md | 706, 712, 1072, 1881 |
| `buildRuntimePublishWorld` CPP 開始 | cpp:178 | cpp:179 | REPAIR_PLAN2-dash2.md | 706, 1072 |
| `isFullyDrained` pre-check line | line 485 | line 486 | REPAIR_PLAN2-dash2.md | 1775, 1777, 2098 |
| `isFullyDrained` function range (section header) | cpp:484-524 | cpp:484-526 | REPAIR_PLAN2-dash2.md | 1773 (return body confirmed at 500-525) |
| `retireBacklogCount_` source | Retire.cpp:48 | Retire.cpp:114 + 5 other sites | REPAIR_PLAN2-dash2.md | 2108 |
| `submitRecoveryRequest` 関数範囲 | cpp:721-781 | cpp:721-780 | REPAIR_PLAN2-dash2.md | 113, 1097 |
| `hasDeferredCommit` (Threading.cpp) | :115 | :116 | REPAIR_PLAN2-dash2.md | 394 (Layer 1 drain condition) |
| `pendingReclaimHandles_.empty()` (Threading.cpp) | :147-148 | :148 | REPAIR_PLAN2-dash2.md | 394 (Layer 1 drain condition) |
| **2026-08-14 追加: §1.8.10.1 static_assert は未実装** | 「現在のアプローチ」 | 「検討案（未実装）」 | REPAIR_PLAN2-dash2.md | 1.8.10.1 (BUILD_ERROR_ENUM_COUNT はソースに存在せず) |
| **2026-08-14 追加: §1.8.10.2 toString range** | RuntimeBuilder.cpp:53-79 / line 78 | RuntimeBuilder.cpp:53-76 / line 75 | REPAIR_PLAN2-dash2.md | 1.8.10.2 |
| **2026-08-14 追加: §1.8.12 item1 prepare() line** | h:78 | AudioEngine.h:866 | REPAIR_PLAN2-dash2.md | 1.8.12 |
| **2026-08-14 追加: §1.8.12 item3 MKL 使用** | MKL は未使用 | FFT は IPP だが VML/BLAS は MKL 継続使用 | REPAIR_PLAN2-dash2.md | 1.8.12 |
| **2026-08-14 追加: §1.5 PublishReceiptWaiter** | AudioEngine.h:3631 | AudioEngine.h:3632 | REPAIR_PLAN2-dash2.md | 1.5 |
| **2026-08-14 追加: §1.6 INV-X2-6 参照** | :1844 (誤) | AudioEngine.h:3621-3627 | REPAIR_PLAN2-dash2.md | 1.6 |
| **2026-08-14 追加: 5.2 ShutdownPhase range** | AudioEngine.h:2533-2542 | AudioEngine.h:2533-2541 | REPAIR_PLAN2-dash2.md | 5.2 |
| **2026-08-14 追加: 5.5 dispatch table** | ISRIntentDispatcher.h:60-68 | ISRIntentDispatcher.h:60-66 | REPAIR_PLAN2-dash2.md | 5.5 |
| **2026-08-14 追加: Appendix F publishAndSwap** | RuntimeStore.h:49 | RuntimeStore.h:40 | REPAIR_PLAN2-dash2.md | Appendix F |
| **2026-08-14 追加: Appendix D condition13 quarantine** | cpp:678/681 (誤) | cpp:905/917 | REPAIR_PLAN2-dash2.md | Appendix D |
| **2026-08-14 追加: Appendix D external setters** | Threading.cpp:127-129 | Threading.cpp:126-128 | REPAIR_PLAN2-dash2.md | Appendix D |
| **2026-08-14 追加: H.1 Threading setters (swap)** | :126-128 | :127-129 | REPAIR_PLAN2-dash2.md | H.1 (誤: 修正前後が逆) |
| **2026-08-14 追加: H.4 §3 reclaim bool 行** | ReleaseResources.cpp:425/435 | ReleaseResources.cpp:426/436 | REPAIR_PLAN2-dash2.md | H.4 |
| **2026-08-14 追加: H.6.3 reclaim signature** | h:377 | h:373-377 | REPAIR_PLAN2-dash2.md | H.6.3 |
| **2026-08-14 追加: §2.2 stale §5.19/§5.12 参照** | §5.19/§5.12 | Appendix G / H.6.3 | REPAIR_PLAN2-dash2.md | 2.2 |
| `dspQuarantineManager.residentCount()` (Threading.cpp) | :138 | :136 | REPAIR_PLAN2-dash2.md | 394 (Layer 1 drain condition) |
| `retireRouter.quarantineResidentCount()` (Threading.cpp) | :139-140 | :137-138 | REPAIR_PLAN2-dash2.md | 394 (Layer 1 drain condition) |
| Threading.cpp 3 setters call sites | :127-129 | :126-128 | REPAIR_PLAN2-dash2.md | 393 (warning note) |
| **2026-08-14 第3パス: 補足 MKL 記述** | MKL は未使用 | FFT は IPP、VML/BLAS は MKL 継続使用 | REPAIR_PLAN2-dash2.md | 補足 |
| **2026-08-14 第3パス: 5.8 external setters** | Threading.cpp:127-129 | Threading.cpp:126-128 | REPAIR_PLAN2-dash2.md | 5.8 |
| **2026-08-14 第3パス: 5.10 条件番号** | 1-15 | 2-16（Appendix D と整合） | REPAIR_PLAN2-dash2.md | 5.10 |
| **2026-08-14 第3パス: 5.17 buildRuntimePublishWorld callers** | Commit.cpp:390（誤）/ Init.cpp:53 | RuntimePublicationOrchestrator.cpp:165,232 / Init.cpp は createBootstrapWorld | REPAIR_PLAN2-dash2.md | 5.17 |
| **2026-08-14 第3パス: 5.18 drain condition #** | #15 | #16（!recoveryAdmissionPending_） | REPAIR_PLAN2-dash2.md | 5.18 |
| **2026-08-14 第3パス: 5.19 test coverage** | :886 / :914-917 | :610（定義）/ :910/:914（呼び出し） | REPAIR_PLAN2-dash2.md | 5.19 |
| **2026-08-14 第3パス: 5.13 Threading line refs** | :127-129 | :126-128 | REPAIR_PLAN2-dash2.md | 5.13 |
| **2026-08-14 第3パス: Appendix A Init.cpp publish** | :123 | :73（publish）/ :123（startCoordinatorLoop） | REPAIR_PLAN2-dash2.md | Appendix A |
| **2026-08-14 第3パス: 1.4.1 isFullyDrained 範囲** | :498-526 | :500-526 | REPAIR_PLAN2-dash2.md | 1.4.1 |
| **2026-08-14 第3パス: Appendix B PendingRecoveryAdmission 範囲** | h:575-590 | h:575-593 | REPAIR_PLAN2-dash2.md | Appendix B |
| **2026-08-14 第4パス: Recovery coalesce 方向転換** | NO-GO継続 | 条件付き GO（R1〜R17） | REPAIR_PLAN2-dash2.md | 1.2 / サマリ表 |
| **2026-08-14 第4パス: AC-LIFE-NEW-1/2/3 追加** | （なし） | 追加 | REPAIR_PLAN2-dash2.md | §4 |
| **2026-08-14 第4パス: A2 Gate A2-G01〜G23 現在状態** | （なし） | 追加 | REPAIR_PLAN2-dash2.md | H.9.4 |
| **2026-08-14 第4パス: single-use Permit 補強** | move-only | move-only + PermitState{Valid,Consumed} | REPAIR_PLAN2-dash2.md | H.9.3 |
| **2026-08-14 第4パス: pendingReclaimHandles 正本化** | count のみ | identity set を primary authority に昇格 | REPAIR_PLAN2-dash2.md | H.9.3 |
| **2026-08-14 第5パス: 既検証領域の別視点再確認** | — | MpscBoundedRing / processIntent / Orchestrator / PublicationAdmission 全て整合。新規修正なし | REPAIR_PLAN2-dash2.md | H.10.4 |
| **2026-08-14 第6パス: INV-ISFULLDRAINED-4 修正** | fallbackBacklog/publicationBacklog 欠落 | backlog 系に追加（16 条件の完全分類） | REPAIR_PLAN2-dash2.md | 1.4 |
| **2026-08-14 第6パス: AC-R4-1〜10 個別定義追加** | （要約のみ） | dash.md:2415 から引用し明記 | REPAIR_PLAN2-dash2.md | 2.1 |

### H.2 Verified-Accurate Counter Declaration Line Numbers

| Counter | h:line (actual) | Document claim | Status |
|---|---|---|---|
| `currentWorld_` | h:465 | h:465 (no correction needed — conversation summary was incorrect) | ✅ |
| `retireBacklogCount_` | h:467 | (not directly cited) | ✅ |
| `publicationBacklogCount_` | h:468 | (not directly cited) | ✅ |
| `publicationIntentResidencyCount_` | h:479 | h:479 (correct) | ✅ |
| `pendingIntentCount_` | h:492 | (not directly cited) | ✅ |
| `fallbackBacklogCount_` | h:493 | (not directly cited) | ✅ |
| `reclaimInFlightCount_` | h:494 | (not directly cited) | ✅ |
| `deferredRetireResidencyCount_` | h:495 | (not directly cited) | ✅ |
| `quarantineIntentResidencyCount_` | h:503 | (not directly cited) | ✅ |
| `quarantineRingResidencyCount_` | h:504 | (not directly cited) | ✅ |
| `quarantineResidentCount_` | h:505 | (not directly cited) | ✅ |

### H.3 Design-Level Consistency (No Contradictions Found)

| 確認事項 | 判定 | 詳細 |
|---|---|---|
| `external setter = cleanup ではない` 否定形の一貫性 | ✅ 整合 | 否定形または lifetime safety として位置付け — 矛盾なし |
| `shutdownPhase >= Destroy は Proof ではない` 否定形 | ✅ 整合 | state snapshot として否定形で一致 |
| `ReclaimPermit は wrapper ではない` 否定形 | ✅ 整合 | reclaim authority そのものとして言及 |
| `reclaimInFlightCount_` → `pendingReclaimHandles_` 強化の設計前提 | ✅ 整合 | A2 hard gate として設計済み。ソースは既に `pendingReclaimHandles_` 使用中 (AudioEngine.h:4632) |

### H.3b AudioEngine.Threading.cpp Drain Condition Line Verification

| Condition | Old claim | Actual source line | Status |
|---|---|---|---|
| `hasDeferredCommit` | :115 | :116 | ✅ Corrected |
| `pendingReclaimHandles_.empty()` | :147-148 | :148 | ✅ Corrected |
| `overflowRing.residentCount()` | :134-135 | :134-135 | ✅ No change |
| `dspQuarantineManager.residentCount()` | :138 | :136 | ✅ Corrected |
| `retireRouter.quarantineResidentCount()` | :139-140 | :137-138 | ✅ Corrected |
| `runtimePublicationBridge_.isFullyDrained()` | :156 | :156 | ✅ No change |
| Function range | :114-157 | :114-156 | ✅ Corrected |
| 3 setter call sites | :127-129 | :126-128 | ✅ Corrected |
| `setPendingIntentCount` | h:123 | h:123 | ✅ (declarations verified against header) |

### H.4 dash2(11) New Claims — Source Evidence Summary

| dash2(11) § | claim | 実コード確認 |
|---|---|---|
| §3 | `reclaim()` receives `bool readerRegistrationClosed` | ISRRuntimePublicationCoordinator.cpp:631-635 (sig), cpp:644-646 (gate) |
| §3 | `reclaim(ShutdownQuiescent, true)` in production | AudioEngine.h:2034-2035, ReleaseResources.cpp:426, :436 — bool directly passed |
| §4 | `shutdownPhase >= Destroy` is state snapshot, not Proof | AudioEngine.h:2019-2020 — `CacheMap::~CacheMap` |
| §7 | `publicationIntentResidencyCount_` separates queue + reservation | h:334 fetchAdd (reserve-before-push), h:337 fetchSub (rollback), h:479 (declaration) |
| §8 | `reclaimInFlightCount_` is single counter (no identity) | h:494 (declaration), cpp:239 (def), cpp:660 (+1), cpp:671 (=0) |
| §9 | `enqueuePublicationIntent` has no state_ check | h:324-339 — fetchAdd → push → rollback only |
| §19 | 3 production reclaim call sites (bool directly) | AudioEngine.h:2032, ReleaseResources.cpp:423, :433 |
| §20 | delete-site inventory | `destroyDSPCoreNode` (Threading.cpp:17-22) single authority; `CacheMap::~CacheMap` at h:2026 delete + h:2032 reclaim |
| §22 | Blocking #3/#4/#5/#10/#11/#12 are 🔴 A2 blocker | All unimplemented/partial (H.1 table + Blocking table) |

### H.5 Independent Tool Verification Summary

| Tool | Usage | Result |
|---|---|---|
| **WSL rg (rtk)** | `rg -n "setQuarantineResidentCount" src/` | 3 matches: 1 call + 1 def + 1 decl |
| **WSL rg (rtk)** | `rg -n "currentWorld_" src/audioengine/ISRRuntimePublicationCoordinator.h` | h:465 (not h:463) |
| **WSL rg (rtk)** | `rg -n "reclaimInFlightCount_" src/` | h:494, cpp:660, cpp:671 |
| **WSL sed** | `sed -n '484,526p' Coordinator.cpp` | Pre-check at 485-486, return body 500-525, close at 526 |
| **WSL sed** | `sed -n '631,647p' Coordinator.cpp` | `reclaim()` sig at 631, gate at 644-646, ShutdownQuiescenceProof comment at 642 |
| **WSL sed** | `sed -n '740,742p' Coordinator.cpp` | `currentWorld_` read at cpp:740-741 |
| **WSL sed** | `sed -n '119,130p' Coordinator.h` | 10 setters at h:121-130 |

**Conclusion**: All line references have been verified against source code using WSL rg + sed. The `ShutdownQuiescenceProof` / `ReclaimPermit` types do NOT exist in source code — confirmed via `rg -n` search across all `.h`/`.cpp` files. Only comment reference is at Coordinator.cpp:642.

### H.6 Fresh Source Verification (2026-08-14 Cross-Check Session)

#### H.6.1 Production External Setter Call Site Inventory

Full `rg -rn` audit across `src/` (excluding tests and Coordinator-internal calls):

| Setter | Production call sites | ファイル |
|---|---|---|
| `setFallbackBacklogCount()` | 2 calls | AudioEngine.Retire.cpp:113, AudioEngine.Threading.cpp:126 |
| `setRetireBacklogCount()` | 6 calls | AudioEngine.Commit.cpp:481, :624, AudioEngine.Retire.cpp:114, AudioEngine.Threading.cpp:127, AudioEngine.h:4152, :4162 |
| `setDeferredRetireResidencyCount()` | 2 calls | AudioEngine.Retire.cpp:115, AudioEngine.Threading.cpp:128 |
| `setReclaimInFlightCount()` | 4 calls | AudioEngine.Retire.cpp:48, :52, :316, :319 |
| `setQuarantineResidentCount()` | 1 call | AudioEngine.Processing.ReleaseResources.cpp:291 |
| **Total** | **9 call sites** | 6 files |

**Status**: Not eliminated (🔴). All 9 production call sites still actively overwrite Coordinator counters with AudioEngine-side snapshot measurements.

> **⚠️ 2026-08-14 追加照合 — setQuarantineResidentCount の domain mixing**: `ReleaseResources.cpp:285-293` の `setQuarantineResidentCount(ringResident)` は **overflow ring の resident count（retire 系）** を Coordinator の `quarantineResidentCount_`（本来 semantic = 実在 quarantine DSP 数）に渡している。`DSPQuarantineManager::residentCount()` が X6 方針の source of truth であるため、この呼び出しは **domain mixing** であり、B2 で `DSPQuarantineManager::residentCount()` 直接観測へ置換すべき（H.9.2 参照）。

#### H.6.2 Path B PublicationIntent Gap — CONFIRMED

`enqueuePublicationIntent()` at `ISRRuntimePublicationCoordinator.h:324` has **no `state_` check** or shutdown authority gate:

```cpp
// ISRRuntimePublicationCoordinator.h:324-339
[[nodiscard]] bool enqueuePublicationIntent(const Intent& intent) noexcept
{
    Intent prepared = intent;
    prepared.type = IntentType::Publish;
    convo::fetchAddAtomic(publicationIntentResidencyCount_, ...);
    if (intentQueue_.push(prepared))
        return true;
    convo::fetchSubAtomic(publicationIntentResidencyCount_, ...);
    return false;
}
```

Path B caller at `AudioEngine.h:4424`:
```cpp
// AudioEngine.h:4424
if (!runtimePublicationBridge_.enqueuePublicationIntent(intent))
```

**Path B 完全経路（2026-08-14 照合）**:
```
AudioEngine.h:4364 enqueueRuntimePublicationFireAndForget()   ← 関数起点
        ↓
AudioEngine.h:4424 enqueuePublicationIntent(intent)           ← 直接呼び出し
        ↓
ISRRuntimePublicationCoordinator.h:324-339 enqueuePublicationIntent()
```
`enqueueRuntimePublicationFireAndForget`（h:4364）は `commitRuntimePublication`（h:4442）のコア実装で、内部で `runtimePublicationBridge_.enqueuePublicationIntent(intent)`（h:4424）を直接呼ぶ。この経路に shutdown gate はない。

**No shutdown gate** — caller passes intent directly to queue without checking `isShutdownInProgress()` or admission state.

Path A (via `submitPublishRequest`) DOES have shutdown gate (`RejectedShutdown` decision at `RuntimePublicationOrchestrator.cpp:353`).

**Status**: 🔴 **ACTIVE GAP** — Path B allows post-shutdown PublicationIntent enqueue, enabling resurrection.

> **✅ 2026-08-14 第三者的レビュー反映 — Path B 修正の方向**:
> 1. **caller-side gate では不十分**: `if (!isShutdownInProgress()) enqueuePublicationIntent(...)` を AudioEngine 側に追加しても、caller が増えるたびに gate 漏れが発生する。**publication admission の最終 gate は publication authority API 自身（`enqueuePublicationIntent`）に置く**。
> 2. **単純な `state_` check でも不十分**: `T1: enqueue 開始 → T2: shutdown admission close → T3: queue.push()` という race がある。`if (state_ != ShuttingDown)` は linearization point を保証しない。
> 3. **推奨 — admission transaction**: `tryAdmitPublication()` → `AdmissionToken/generation` → queue reservation → queue push の順序で、**「check した」ではなく「shutdown admission を正式に取得した」**ことを obligation 生成条件にする。`close admission ↔ enqueue reservation` の linearization point を明示する。
> 4. **4 経路を統一**: PublicationIntent enqueue / Recovery enqueue / Build admission / Publish の 4 経路すべてを共通 shutdown admission へ接続（No-Resurrection invariant）。

#### H.6.3 reclaim() bool Parameter — CONFIRMED UNIMPLEMENTED

`reclaim()` signature at `ISRRuntimePublicationCoordinator.h:373-377`:
```cpp
bool reclaim(
    ReclaimMode mode,
    const DSPHandle& handle,
    DSPHandleRuntime& handleRuntime,
    ISRRetireRouter& router,
    bool readerRegistrationClosed = false) noexcept;
```

3 production call sites pass `owner->m_epochDomain.readerRegistrationClosed()` directly:
- `AudioEngine.h:2035`
- `AudioEngine.Processing.ReleaseResources.cpp:426`
- `AudioEngine.Processing.ReleaseResources.cpp:436`

`readerRegistrationClosed()` implementation at `EpochDomain.h:592` — simple boolean flag read, NOT a proof.

**Status**: 🔴 **Not migrated to ReclaimPermit**. `bool readerRegistrationClosed` must be replaced with `ReclaimPermit`.

#### H.6.4 CacheMap::~CacheMap — CONFIRMED using shutdownPhase
#### H.6.3a Build Identity Semantic Comparison — CONFIRMED UNIMPLEMENTED

```sh
rg -n "buildIdentity|isSemanticSuperset|BuildIdentity|semanticBuildIdentity|IR hash" src/
```
- **結果: 0 results** in production source. `isSemanticSuperset` appears only in design spec/pseudocode (REPAIR_PLAN2-dash2.md:264).
- `RuntimeBuildTypes.h:318-332` has `buildInput` field-by-field comparison (`sampleRate`, `blockSize`, etc.) but **no IR hash comparison (`isSemanticSuperset`) implemented**.
- **Impact**: G10/G11 coalesce (§1.8) relies on `irhash` for correct supercession — without it, `PendingRecoveryAdmission` range safety assumption (INV-X1-1/INV-X1-2) is compromised.
- The 7 non-`setFallbackBacklogCount` comparison paths in H.6.1 (9 call sites) cover build-input fields only, **not IR hash**.

**Source**: `src/audioengine/RuntimeBuildTypes.h:301-332` (buildInput comparison), `REPAIR_PLAN2-dash2.md:264` (isSemanticSuperset spec).


`AudioEngine.h:2019-2035`:
```cpp
if (convo::consumeAtomic(owner->shutdownPhase, std::memory_order_acquire)
    >= AudioEngine::ShutdownPhase::Destroy) {
    // ...
    const bool reclaimed = owner->runtimePublicationBridge_.reclaim(
        convo::isr::RuntimeIntentCoordinator::ReclaimMode::ShutdownQuiescent,
        entry.second, rt, *owner->m_retireRouter,
        owner->m_epochDomain.readerRegistrationClosed());
```

Uses `shutdownPhase >= Destroy` as the reclaim authorization check — NOT a `ShutdownQuiescenceProof`.

**Status**: 🔴 **A2 blocked** — CacheMap destructor reclaim path not Permit-authorized.

#### H.6.5 Type Existence Verification

| Type | Existence | Source |
|---|---|---|
| `ShutdownQuiescenceProof` | ❌ Does NOT exist | `rg -rn "class ShutdownQuiescenceProof\|struct ShutdownQuiescenceProof" src/` → 0 results |
| `ReclaimPermit` | ❌ Does NOT exist | `rg -rn "class ReclaimPermit\|struct ReclaimPermit" src/` → 0 results |
| `PermitIdentity` | ❌ Does NOT exist | `rg -rn "class PermitIdentity\|struct PermitIdentity" src/` → 0 matches |
| `ShutdownDrainToken` | ✅ Exists | `src/audioengine/ShutdownScope.h:12` |

**Status**: ✅ Design-only types confirmed not implemented (consistent with Amendment 2: §2.2).

#### H.6.6 Counter Declaration Verification (h-file)

Direct source verification via `rg -n` on `ISRRuntimePublicationCoordinator.h`:

| Counter | Actual line | Document H.2 claim | Status |
|---|---|---|---|
| `currentWorld_` | h:465 | h:465 | ✅ |
| `retireBacklogCount_` | h:467 | h:467 | ✅ |
| `publicationBacklogCount_` | h:468 | h:468 | ✅ |
| `publicationIntentResidencyCount_` | h:479 | h:479 | ✅ |
| `pendingIntentCount_` | h:492 | h:492 | ✅ |
| `fallbackBacklogCount_` | h:493 | h:493 | ✅ |
| `reclaimInFlightCount_` | h:494 | h:494 | ✅ |
| `deferredRetireResidencyCount_` | h:495 | h:495 | ✅ |
| `quarantineIntentResidencyCount_` | h:503 | h:503 | ✅ |
| `quarantineRingResidencyCount_` | h:504 | h:504 | ✅ |
| `quarantineResidentCount_` | h:505 | h:505 | ✅ |
| `recoveryAdmissionPending_` | h:591 | h:591 | ✅ |

All counter declarations verified against source. ✅

#### H.6.7 Summary of Blocking Findings

| Item | Status | Details |
|---|---|---|
| Path B shutdown gate | 🔴 NO-GO | `enqueuePublicationIntent()` at h:324 has no `state_` check |
| External setters | 🔴 NO-GO | 9 production call sites across 6 files |
| `reclaim(..., bool)` | 🔴 NO-GO | 3 production call sites pass raw bool |
| Proof/Permit types | 🔴 NO-GO | Types do not exist in source |
| CacheMap destructor reclaim | 🔴 NO-GO | Uses `shutdownPhase >= Destroy`, not Proof |
| `isFullyDrained()` as proof | ✅ Correct | Document correctly distinguishes drain observation from Proof |

### H.7 mixSmoother RT Contract Violation — ✅ 解消済み (2026-08-14 再検証)

**2026-08-11 review で「既知の RT contract violation」とされていた `mixSmoother.reset()` at `Runtime.cpp:360` は、実コードでは存在しない。**

#### 検証結果

| 主張 (2026-08-11) | 実コード (2026-08-14) | 判定 |
|---|---|---|
| `mixSmoother.reset()` が RT から呼ばれる | `Runtime.cpp:360` は `activeMixSmoother.resetRT(...)`（RT-safe）を呼ぶ | ❌ 主張は誤り |
| `reset()` が `ASSERT_NON_RT_THREAD` で jassert 違反 | RT スレッドは `resetRT()`（`ASSERT_AUDIO_THREAD` at DspNumericPolicy.h:343）を呼ぶ | ❌ 違反なし |
| `reset()` は NonRT のみ | `reset()` は `prepareToPlay()`（Lifecycle.cpp:370, NonRT）でのみ呼ばれる | ✅ |
| Phase 0.5 修正が必要 | `resetRT()` + generation handshake で既に修正済み | ❌ Phase 0.5 不要 |

#### 実コードの仕組み（RT-safe である根拠）

1. **NonRT 側**: `setSmoothingTime()` (Runtime.cpp:928) が `pendingOverride.smoothingTimeSec` を更新し、`smoothingTimeChangePendingGen` を fetchAdd (acq_rel)
2. **RT 側**: `process()` が generation を acquire で検知 (Runtime.cpp:350-353) → `resetRT()` (h:360) + `applyImmediateValueRT()` (h:361) + `setTargetValue()` (h:362) を呼ぶ
3. **`resetRT()`**: `DspNumericPolicy.h:341-347` で定義。`reset()` と同一の totalSteps 更新ロジックだが `ASSERT_NON_RT_THREAD` を持たない（`ASSERT_AUDIO_THREAD` を持つ）。コメントに「smoothingTimeChangePendingGen ハンドシェイク経由で reset() が RT スレッドへ渡行していた問題（§1.3-B）を解消する」と明記
4. **mixSmoother reset handshake**: NonRT `ConvolverProcessor::reset()` (Lifecycle.cpp:489) が `mixSmootherResetPendingGen` を fetchAdd → RT が検知 (Runtime.cpp:341-344) → `applyImmediateValueRT()` で適用

#### DspNumericPolicy.h の line ref 更新 (2026-08-14)

`resetRT()` 追加により後続メソッドの line が ~14 行ずれていた（旧 line ref は `resetRT()` 導入前）。

| Method | 旧 line ref (doc) | 実測 line (2026-08-14) |
|---|---|---|
| `reset()` assert | h:333 | h:333 ✅ |
| `resetRT()` | （記載なし） | h:341-347（assert at h:343） |
| `setCurrentAndTargetValue()` assert | h:341 | h:356 |
| `applyImmediateValueRT()` assert | h:352 | h:367 |
| `setTargetValue()` assert | h:363 | h:378 |
| `getNextValue()` assert | h:374 | h:389 |

#### 影響

- 本文 §1.3、Amendment 1/6、A-1、Phase 0.5、Final Assessment の該当箇所を ✅ 解消済みに更新
- **Phase 0.5 は不要**（実装済みのため）— 今後のフェーズ進行（Phase A/B）に blocker なし
- 将来の LinearRamp 抽象化（1.3）は引き続き「対象外・文書化」で維持

### H.8 Fresh-Perspective Source Verification (2026-08-14 第二パス)

前セッション（H.6/H.7）がカウンタ・drain 条件・Path B を検証したのに対し、本セッションは**設計セクション（§1-§2・補足 5.x・Appendix A-G）の line ref と記述整合性**を別視点から検証した。

#### H.8.1 修正・確定事項（2026-08-14）

| 対象 | 修正/確定内容 | 実コード根拠 |
|---|---|---|
| §1.8.10.1 | `BUILD_ERROR_ENUM_COUNT` static_assert は**未実装** → 「検討案」に修正 | `grep -rn "BUILD_ERROR_ENUM_COUNT" src/` → 0 件 |
| §1.8.10.2 | toString range :53-79/line 78 → **:53-76/line 75** | RuntimeBuilder.cpp |
| §1.8.12 item1 | `prepare()` の line ref h:78 → **AudioEngine.h:866** | h:78 は include 行 |
| §1.8.12 item3 | 「MKL 未使用」→ **FFT は IPP、VML/BLAS は MKL 継続使用** | MKLNonUniformConvolver.cpp:27-30,54-55 |
| §1.5 | PublishReceiptWaiter h:3631 → **h:3632** | AudioEngine.h |
| §1.6 | INV-X2-6 参照 :1844 → **AudioEngine.h:3621-3627** | :1844 は無関係コード |
| 5.2 | AudioEngine::ShutdownPhase :2533-2542 → **:2533-2541** | 閉じ `};` は 2541 |
| 5.5 | dispatch table :60-68 → **:60-66** | ISRIntentDispatcher.h |
| Appendix F | publishAndSwap :49 → **:40** | RuntimeStore.h |
| Appendix D #13 | quarantineIntentResidency cpp:678/681 → **cpp:905/917** | :678/681 は requestReclaim 内 |
| Appendix D | external setters Threading.cpp:127-129 → **:126-128** | grep 確認 |
| H.1 | Threading setters 行の修正前後が逆 → **交換** | — |
| H.4 §3 | reclaim bool 行 ReleaseResources.cpp:425/435 → **:426/436** | 実測 |
| H.6.3 | reclaim signature h:377 → **h:373-377** | 関数開始は 373 |
| §2.2 | stale 参照 §5.19/§5.12 → **Appendix G / H.6.3** | 明確化 |

#### H.8.2 再検証で正確と確認された事項（変更なし）

- **§1.1/1.1.1**: queue 型（intentQueue_=MpscBoundedRing h:602, quarantineFallbackQueue_=MpscBoundedRing h:610, recoveryIntentQueue_=LockFreeRingBuffer h:551, pendingRecoveryAdmission_=plain h:590）✅
- **§1.8.2-1.8.4**: BuildError enum h:107-116 / build() cpp:428-469 / validateWarmup() cpp:471-479 / 3 build() 呼び出しサイト cpp:941/1015/1087 / 3 validateWarmup サイト cpp:955/1032/1143 ✅
- **§1.9**: submitRecoveryIntent h:4285-4297（無条件 recoveryPending=true + notify）/ wake predicate RebuildDispatch.cpp:828-833 ✅
- **§2.1**: retireDSPHandleForRuntime h:4207→requestReclaimHandle h:4234 / TOCTOU Retire.cpp:41-117 / isRetired() :75 ✅
- **§2.2.2**: postStopEnqueueCount h:144 / markPostStopEnqueue h:202 / sh6PostStopEnqueueCount_ h:223 / Commit.cpp:459 ✅
- **5.3**: currentWorld_ h:465 / worldAuthority_ h:4685 / observePublishedWorld h:3556-3558 / runtimePublishWorld_ 非存在 ✅
- **5.4**: commit() cpp:75-115 / currentWorld_ 更新 cpp:112（commit-before-swap）✅
- **5.5**: QuarantineIntentHandler ProcessIntent.cpp:110-141 / RecoveryIntentHandler :163-169（dead code）✅
- **5.6**: DSPCore::prepare() DSPCoreLifecycle.cpp:72（void） / convolverState->prepare() h:682-688 / newConv->init() h:741（bool 非伝播）✅
- **5.12/5.19**: reclaim precondition cpp:638-646 / shutdown gate cpp:733 / discardPendingRecoveryAdmission cpp:816-824 ✅
- **Appendix A**: submitObserve 4 call sites（Timer.cpp:896/1029/1568, DSPTransition.h:156） / ScopedThreadRole AudioBlock.cpp:102 / isAudioThread h:118 ✅
- **Appendix C**: retry loop RebuildDispatch.cpp:1005-1052 / kMax=4 :1003 / settle :1022/1044 ✅
- **Appendix E**: sequence comparison cpp:96-98（`>` 単純比較・wraparound 非対応の記述は正確）✅
- **Appendix G**: reclaim 3 call sites（ReleaseResources.cpp:423/433, AudioEngine.h:2032）✅

#### H.8.3 未確定事項の確定結果

| マーカー | 判定 | 根拠 |
|---|---|---|
| 「1.5 sparse completion 将来保留」 | ✅ 確定（将来タスク） | 現行は contiguous watermark（h:3632-3653）。MPSC completion 許容時のみ必要 |
| 「2.1 retire 順序逆転の完全解消は保留」 | ✅ 確定（将来タスク） | runtime 経路は requestReclaimHandle に一本化済み。FIFO 強化のみ残存 |
| 「1.1 MPSC 化は Phase 5 将来拡張」 | ✅ 確定（将来タスク） | 現行 SPSC 成立（recoveryIntentQueue_ h:551） |

→ **未確定な事項は全て「将来タスク」であり、調査で確定された**。文書内に不明確な技術主張は残っていない。

### H.9 External Setter → Lifetime Fact 単一化の検証 (2026-08-14 第三者的レビュー反映)

第三者的レビュー（2026-08-14）の中心的主張: **「setter を削除する」だけでは不十分。`drain` を authoritative な lifetime fact として成立させ、その fact からのみ Proof → Permit → reclaim が生成される一方向の証明経路を確立すること**が真の目標である。

#### H.9.1 4 つの完成条件（external setter = 0 より強い）

レビューは「external setter = 0」ではなく、以下の **4 条件**を本当の完成条件とする:

| # | 条件 | 現状 (2026-08-14 照合) |
|---|---|---|
| ① | **Lifetime accounting に external writer が存在しない** | 🔴 未達 — 9 production call sites（H.6.1） |
| ② | **Proof 生成権が ShutdownRuntime に限定される** | 🔴 未達 — Proof 型は未実装（コメントのみ cpp:642） |
| ③ | **Permit 生成権と identity が ShutdownRuntime に限定される** | 🔴 未達 — Permit 型は未実装 |
| ④ | **Proof 生成後に新しい lifetime obligation を生成できない** | 🔴 未達 — Path B は shutdown gate なし（H.6.2） |

この 4 条件が揃わない限り、`drain → proof → permit → reclaim` の authority chain は一本化されない。

#### H.9.2 新規検証: setQuarantineResidentCount の domain mixing

**2026-08-14 追加照合**: `ReleaseResources.cpp:291` の `setQuarantineResidentCount(ringResident)` は **overflow ring の resident count（retire 系）** を Coordinator の `quarantineResidentCount_`（実在 quarantine DSP 数の semantic）に渡している。これは **domain mixing** である。

- `ringResident` = `worldAuthority_.lifetime().getOverflowRing()->residentCount()`（retire overflow ring）
- `quarantineResidentCount_` の本来 semantic = 実在 quarantine DSP 数（`DSPQuarantineManager::residentCount()` が source of truth — X6）
- 呼び出しコンテキスト: ReleaseResources.cpp:285-293 の `else` ブロック（「タイムアウト前に完了した場合も coordinator カウントを最終更新」）

**→ この呼び出しは B2（external setter 完全撤去）で、`DSPQuarantineManager::residentCount()` を直接観測する形へ置換すべき**（§1.4 / H.6.1 の X6 方針と整合）。

#### H.9.3 レビュー推奨の設計強化（文書 §2.2 への補強）

| レビュー指摘 | 反映内容 |
|---|---|
| Path B gate を「state check」から「admission transaction」へ | `enqueuePublicationIntent` 内で `state_` check のみならず、shutdown admission の formal 取得（`tryAdmitPublication()` → token/generation → reservation → push）を要求 |
| Proof 生成を単一 transaction に | `ShutdownRuntime::tryMakeQuiescenceProof()` が全条件を自身で authority から取得し検証（`isFullyDrained()` の結果だけから Proof を作らない） |
| `reclaimInFlightCount_` を Proof の唯一の根拠にしない | `pendingReclaimHandles`（identity-level state）を **primary authority** に昇格。`reclaim begin → insert(ReclaimIdentity)` / `reclaim complete → erase(ReclaimIdentity)`。Proof 条件は `pendingReclaimHandles.empty()`。count は derived telemetry に降格 |
| `ShutdownDrainToken` を ReclaimPermit に昇格させない | 確認済み: `ShutdownDrainToken{}`（default ctor, valid_=true, expiration=0）で `consume()==true` になり得る（ShutdownScope.h:12-33）。capability として不十分 |
| Permit は move-only single-use | `ReclaimPermit(const ReclaimPermit&) = delete; (ReclaimPermit&&) = default;` に加え、**move-only だけでは single-use を保証しない**（move 後 object の再利用を自動防止しない）。Permit 内部に `PermitState { Valid, Consumed }` を持たせ、reclaim 時に `identity valid AND !consumed` を確認してから `Consumed` へ遷移（double reclaim 防止） |
| Proof 生成後の状態変更を不可逆に | `Admission = Closed` を不可逆状態にし、`Closed → Open` を禁止（post-proof resurrection 防止） |
| underflow 防止 | semantic event API で `fetch_sub` 前に `old > 0` を検証。違反時は Faulted → Proof 生成不能へ |

#### H.9.4 A2 開始条件の拡充（23 条件）

第三者的レビューは A2 開始条件を 23 条件（A2-G01〜A2-G23）として固定する。8/13 版の A2 開始条件（15 条件: Audio Thread→NonRT API=0 / isFullyDrained semantic fixed / external setter=0 / single-writer counters / pendingReclaimHandles authoritative / Proof ctor inaccessible / Permit ctor inaccessible / Permit.identity==Proof.identity / readerRegistrationClosed bool 廃止 / shutdownPhase>=Destroy 非認可 / 4 経路全拒否 / postStopEnqueue==0 / destruction audit / stale Permit reject / forged Permit impossible）を包含する:

| # | 条件 | カテゴリ |
|---|---|---|
| A2-G01 | external setter = 0 | drain accounting |
| A2-G02 | counter mutation = single authority（semantic event のみ） | drain accounting |
| A2-G03 | isFullyDrained = observational predicate only（reclaim authority でない） | drain accounting |
| A2-G04 | swapPending_ pre-check preserved | drain accounting |
| A2-G05 | Path A shutdown gate | No-Resurrection |
| A2-G06 | Path B authority-side shutdown gate | No-Resurrection |
| A2-G07 | Recovery enqueue gate | No-Resurrection |
| A2-G08 | Build admission gate | No-Resurrection |
| A2-G09 | Publish gate | No-Resurrection |
| A2-G10 | postStopEnqueue == 0 after producer join | quiescence |
| A2-G11 | reader registration closed | quiescence |
| A2-G12 | active readers == 0 | quiescence |
| A2-G13 | epoch settled | quiescence |
| A2-G14 | pending reclaim identity == empty | completion（H.11.11.9.3 C1 へ移動 — Proof から除外・循環排除） |
| A2-G15 | ShutdownQuiescenceProof private construction | capability |
| A2-G16 | ReclaimPermit private construction | capability |
| A2-G17 | PermitIdentity bound to ShutdownRuntime | capability |
| A2-G18 | PermitIdentity bound to shutdown generation | capability |
| A2-G19 | PermitIdentity bound to epoch generation | capability |
| A2-G20 | PermitIdentity bound to reader-registration generation | capability |
| A2-G21 | stale Permit rejected | capability |
| A2-G22 | forged Permit impossible（`ReclaimPermit{}` がコンパイル不能） | capability |
| A2-G23 | physical destruction paths audited（delete/destroy/reset/release/unique_ptr 破棄/owner reset） | destruction |

**→ この 23 条件を満たして初めて `reclaim(ShutdownQuiescent, ReclaimPermit)` を production へ接続する。**

**✅ 2026-08-14 第四者レビュー — A2 Gate 現在状態（A2-G01〜G23、番号を A2-G 表に揃えて修正）**:

| Gate | 条件 | 現在 (2026-08-14) |
|---|---|---|
| A2-G01 | external setter = 0 | 🔴（9 call sites） |
| A2-G02 | semantic accounting single authority | 🟡（event 化未完了） |
| A2-G03 | `isFullyDrained()` observational only | 🟢 設計（実装未接続） |
| A2-G04 | swapPending_ pre-check preserved | 🟢（現行 16 条件に維持） |
| A2-G05 | Path A shutdown gate | 🟢（RejectedShutdown :350） |
| A2-G06 | Path B authority-side shutdown gate | 🔴（h:324-339 に gate なし） |
| A2-G07 | Recovery enqueue gate | 🟡（submitRecoveryRequest:733 に shutdown gate あり） |
| A2-G08 | Build admission gate | 🟡 |
| A2-G09 | Publish gate | 🟢/🟡（Path A は gate あり、Path B はなし） |
| A2-G10 | postStopEnqueue == 0 after producer join | 🟢 tracking / 🔴 Proof 接続 |
| A2-G11 | reader registration closed | 🟢（EpochDomain.h:592 実装済み） |
| A2-G12 | active readers == 0 | 🟢（activeReaderCount() 実装済み） |
| A2-G13 | epoch settled | 🟢 設計（EpochQuiescenceEvidence は H.11.13.3） |
| A2-G14 | pending reclaim identity == empty | 🟡（count のみ、identity set 未昇格。Completion C1 側） |
| A2-G15 | ShutdownQuiescenceProof private construction | 🔴（型なし） |
| A2-G16 | ReclaimPermit private construction | 🔴（型なし） |
| A2-G17 | PermitIdentity bound to ShutdownRuntime | 🔴（型なし） |
| A2-G18 | PermitIdentity bound to shutdown generation | 🔴（型なし） |
| A2-G19 | PermitIdentity bound to epoch generation | 🔴（型なし、EpochDomain に generation なし） |
| A2-G20 | PermitIdentity bound to reader-reg generation | 🔴（型なし、EpochDomain に generation なし） |
| A2-G21 | stale Permit rejected | 🔴（型なし） |
| A2-G22 | forged Permit impossible（`ReclaimPermit{}` コンパイル不能） | 🔴（型なし） |
| A2-G23 | physical destruction paths audited | 🔴（H.11.12 で destroyDSPCoreNode 4 パス棚卸し・H.11.21 で削除キュー full 時 deleter 強制実行を対象追加・acceptance test 未実装） |

> **注**: 旧 H.9.4 の現在状態表（G1〜G20）は番号が A2-G 表（G01〜G23）と食い違っていたため修正。旧 G9（bool reclaim API 削除）は Commit 4 / H.11.11.9.5、旧 G19（double consume）は T9 / Consumed CAS、旧 G20（no-resurrection）は T8 / H.11.4 に相当し、A2-G 表に統合した。

**→ A2-G01〜G23 の PASS が A2 production reclaim の機械的 acceptance gate。**

#### H.9.5 推奨 Phase 順序（第三者的レビュー）

```
Phase B0: setter 全 9 call site を semantic 分類（retire/fallback/deferred/reclaim/quarantine）
Phase B1: absolute set → semantic ++/-- へ変換（snapshot accounting 禁止）
Phase B2: external setter API 削除（grep production 0 件を合格条件）
Phase A1: Proof / Permit 型導入（production 接続禁止）
Phase B3: 共通 shutdown admission へ 4 経路接続（Path B 閉鎖）
Phase A2: production reclaim を ReclaimPermit 化
```

既存の A1 → B1 → B2 → A2 順序と整合し、B0/B3 を追加で明示する。

#### H.9.6 A2 検証テスト（第三者的レビュー T1〜T8）

| Test | 内容 | 合格条件 |
|---|---|---|
| T1 | setter 不存在 | `grep production:` setFallbackBacklogCount / setRetireBacklogCount / setDeferredRetireResidencyCount / setReclaimInFlightCount / setQuarantineResidentCount = 0 件 |
| T2 | backlog 存在中 | retireResidency = 1 のとき `tryMakeProof()` → nullopt |
| T3 | fake zero attempt | 旧 setter API が残っていれば `setRetireBacklogCount(0)` で `isFullyDrained()==true` になり得る。修正後は `setRetireBacklogCount` が **compile error** |
| T4 | producer race | `closeAdmission()` 後の producer enqueue が拒否され、`postStopEnqueueCount == 0`（または Proof が unavailable） |
| T5 | stale Permit | Shutdown N の Permit N を Shutdown N+1 で `reclaim()` → reject |
| T6 | forged Permit | `ReclaimPermit{}` が production code からコンパイル不能 |
| T7 | double consume | `reclaim(std::move(permit))` の 2 回目が不可能/拒否 |
| T8 | Path B resurrection | shutdown admission close 後の `enqueuePublicationIntent()` → rejected |

**→ T1〜T8 全 PASS を A2 完了条件に含める。**

#### H.9.7 Blocking 番号の定義（第三者的レビュー参照の明確化）

H.4 §22 で参照される「Blocking #3/#4/#5/#10/#11/#12」の具体的定義（第三者的レビューに基づく 2026-08-14 確定）:

| # | Blocking 内容 | 現状 (2026-08-14) |
|---|---|---|
| Blocking 3 | external setter（9 call sites / 6 files） | 🔴 未達（H.6.1） |
| Blocking 4 | Proof authority（ShutdownRuntime に限定されない） | 🔴 未達（Proof 型未実装） |
| Blocking 5 | ReclaimPermit authority（ShutdownRuntime に限定されない） | 🔴 未達（Permit 型未実装） |
| Blocking 10 | Path B を含む 4 経路 No-Resurrection | 🔴 未達（Path B gate なし — H.6.2） |
| Blocking 11 | stale Proof（旧 shutdown transaction の Proof が有効と誤認） | 🔴 未達（identity 未束縛） |
| Blocking 12 | Proof/Permit transaction atomicity | 🔴 未達（単一 transaction 未実装） |

**→ 全 Blocking は A2 完了までに解消必須。**

### H.10 Fresh-Perspective Source Verification (2026-08-14 第三パス)

第1パス（カウンタ/drain）、第2パス（§1-§2・補足 5.x・Appendix A-G）に続き、第3パスでは **§4 クロスカッティング AC・補足 5.8〜5.20 の未検証部分・RuntimeWorldAuthority publish flow・文書内の繰り返し主張の整合性**を検証した。

#### H.10.1 修正・確定事項（2026-08-14 第三パス）

| 対象 | 修正内容 | 実コード根拠 |
|---|---|---|
| 補足 MKL 記述 | 「MKL は未使用」→「FFT は IPP、VML/BLAS は MKL 継続使用」 | MKLNonUniformConvolver.cpp:27-30,54-55 |
| 5.8 external setters | Threading.cpp:127-129 → **:126-128** | grep 確認 |
| 5.10 条件番号 | 1-15 → **2-16**（Appendix D と整合） | ISRRuntimePublicationCoordinator.cpp:500-525 |
| 5.17 callers | Commit.cpp:390（誤）/ Init.cpp:53 → **RuntimePublicationOrchestrator.cpp:165,232** / Init.cpp は createBootstrapWorld | grep 確認 |
| 5.18 drain condition # | #15 → **#16**（!recoveryAdmissionPending_） | 5.10 の番号付けと整合 |
| 5.19 test coverage | :886 → **:610（定義）** / :914-917 → **:910/:914（呼び出し）** | ISRSemanticValidationTests.cpp |
| 5.13 Threading line refs | :127-129 → **:126-128** | grep 確認 |
| Appendix A Init.cpp | Bootstrap publish :123 → **:73**（:123 は startCoordinatorLoop） | AudioEngine.Init.cpp |
| 1.4.1 isFullyDrained 範囲 | :498-526 → **:500-526** | 関数 range と整合 |
| Appendix B PendingRecoveryAdmission 範囲 | h:575-590 → **h:575-593** | struct 575-589 + member 590 + static_assert 591-593 |

#### H.10.2 検証で正確と確認された事項（変更なし）

- **§4 AC**: AC-ISR-1 / AC-LIFE-1 / AC-LIFE-2 / AC-PUB-1 / AC-1.4-DRAIN / AC-1.8-RETRY / AC-2.2-PERMIT — すべて設計契約として正確
- **5.11**: postStopEnqueue tracking（ISRShutdown.h:144/202/223, ISRShutdown.cpp:162/257, Commit.cpp:459）✅
- **5.12**: reclaim precondition（cpp:638-646）✅
- **5.13**: setter h:line 一覧（h:121-127）は正確 ✅（Threading.cpp の line refs のみ修正）
- **5.14**: convolver init() bool 非伝播 ✅
- **5.15**: INV-ISR-01〜07（ISRRuntimePublicationCoordinator.h:68-83 コメントと完全一致）✅
- **5.16**: build() catch マッピング ✅
- **5.18**: Recovery Admission State Machine（cpp:721-842）✅
- **5.20**: enqueuePublicationIntent rollback path ✅
- **RuntimeWorldAuthority::publish()**: commit-before-swap（RuntimeWorldAuthority.h:252-263）✅
- **Init.cpp bootstrap**: rebuildThread :33 → publish :73 → submitRebuildIntent :94 → startCoordinatorLoop :123 ✅

#### H.10.3 文書整合性チェック結果

| 確認項目 | 結果 |
|---|---|
| Threading.cpp external setters 参照（5.8/5.13/H.6.1/Appendix D） | ✅ 全て :126-128 で統一 |
| isFullyDrained 条件数（16条件 vs 15 return-body） | ✅ swapPending_ pre-check(1) + return-body(15) の使い分けが一貫 |
| PendingRecoveryAdmission 範囲（5.7/5.18/Appendix B） | ✅ h:575-593 で統一 |
| RecoveryIntent 範囲 | ✅ h:195-218 で統一 |
| 未確定マーカー（要調査/棚卸し/保留） | ✅ 全て将来タスクとして確定済み（H.8.3） |

→ **文書内の技術主張はソースコードと整合し、繰り返し参照の line ref も統一された。**

#### H.10.4 第五パス検証（2026-08-14）— 既検証領域の別視点再確認

MpscBoundedRing・processIntent・RuntimePublicationOrchestrator・PublicationAdmission の実装詳細を別視点から再確認した。**新たな修正は不要**（全て文書記述と整合）。

| 確認項目 | 実コード照合 | 判定 |
|---|---|---|
| 5.1 MpscBoundedRing CAS | `push()` CAS は h:82（`compareExchangeAtomic(enqueuePos_,...)`）、reservation order コメント h:13/81、payload 書込み h:99、seq release | ✅ 5.1 の line refs 正確 |
| 5.1 lock-free 主張 | `enqueuePos_`/`dequeuePos_` は `std::atomic<uint32_t>`（h:198-199）、alignas(64) 分離。x64 で lock-free | ✅ |
| 5.1 sizeApprox/clear | sizeApprox h:134、clear h:142-147 | ✅ |
| processIntent 統合 | intentQueue_（MPSC）から Observe/Publish/Quarantine/Recovery を処理。Publish pop → `publicationIntentResidencyCount_--`、Quarantine pop → `quarantineIntentResidencyCount_--`+`pendingIntentCount_--`、Observe/Recovery pop → `pendingIntentCount_--` | ✅ Appendix D と整合 |
| H.6.2 Path A gate | `submitPublishRequest`（RuntimePublicationOrchestrator.cpp:316）→ `trySubmitImpl` → `RejectedShutdown`（:350-353） | ✅ |
| H.6.2 PublicationAdmission | `evaluate()` shutdown check at PublicationAdmission.cpp:11（`isShutdownInProgress()`） | ✅ |
| Appendix A juce::Timer | AudioEngine は `private juce::Timer` を h:585 で継承 | ✅ |

→ **既検証領域の再確認でも文書記述の正確性が担保された。**

#### H.10.5 第六パス検証（2026-08-14）— 不変条件定義と Acceptance Criteria の整合

INV-ISFULLDRAINED 定義・AC-R4 条件・Phase 順序・補足文章を別視点から照合した。

| 対象 | 修正/確認内容 | 根拠 |
|---|---|---|
| INV-ISFULLDRAINED-4 | **修正**: `fallbackBacklogCount_` と `publicationBacklogCount_` を backlog 系に追加（旧記述は欠落） | isFullyDrained 16 条件の semantic 完全分類 |
| AC-R4-1〜10 | **追加**: 個別定義を dash.md:2415 から引用し dash2 に明記（自己完結性向上） | REPAIR_PLAN2-dash.md:2415 / X_IMPL_CHECKLIST.md:118（AC-R4-1/2 完了確認） |
| §3 Phase 順序 | 確認: Phase 0→0.5→A→B→C→D→E→F→G→H→I は維持。Phase A（A2 相当）と Phase I（coalesce）は分離 = 第四者レビューの「A2 と coalesce 独立」と整合 | — |
| 補足文章 | 確認: AC テスト固定方針・lifetime safety 注記は整合 | — |

→ **不変条件定義と Acceptance Criteria がソースおよび正本（dash.md）と整合。**

### H.9.8 第四者レビュー反映（2026-08-14 004001 版）

第四者レビュー（2026-08-14）の総合判定は **「設計 GO / 実装条件付き GO / production reclaim は NO-GO 継続」**。前版（第三者的レビュー反映 H.9）の方向性を支持しつつ、以下を追加確定した。

#### 追加確定事項

| 項目 | 判定 | 反映 |
|---|---|---|
| **AC-LIFE-NEW-1** | Proof 後 lifetime obligation 生成 API 不存在 | §4 に追加 |
| **AC-LIFE-NEW-2** | Proof/Permit identity が shutdown transaction linearization point を共有 | §4 に追加 |
| **AC-LIFE-NEW-3** | Proof 後 state mutation → Permit 生成不能/即時無効化 | §4 に追加 |
| **A2 Gate 現在状態** | A2-G01〜G23（現状: 大部分 🔴） | H.9.4 に追加 |
| **Recovery coalesce 方向転換** | NO-GO → **条件付き GO**（R1〜R17 実装で可能） | §1.2.3 に追加 |
| **single-use Permit** | move-only だけでは不十分 → `PermitState {Valid, Consumed}` | H.9.3 に更新 |
| **pendingReclaimHandles 正本化** | count ではなく identity set を primary authority に昇格 | H.9.3 に更新 |
| **bounded durable table** | `pendingRecoveryAdmission_`（h:590）は single slot → `CanSupersede==false` 時に複数保持できる bounded durable table へ拡張 | §1.2.3 に追加 |
| **intentId は identity ではない** | `intentId` は sequence/diagnostic、coalesce 判定に使わない | §1.2.3 に追加 |
| **A2 と Recovery coalesce は独立** | A2 は Tier 1 lifetime、coalesce は Tier 2 recovery、依存させない | サマリ表に反映 |

#### 最終評価のまとめ

```
Architecture:              GO
Specification:             GO
Phase B0/B1:               GO
Phase B2:                  GO（9 call site 完全撤去が絶対条件）
Proof/Permit 型導入:        GO
Path B 修正:               P0 lifetime/No-Resurrection として GO
Production reclaim A2:     NO-GO（A2-G01〜G23 の PASS が前提）
Recovery coalesce:         条件付き GO（R1〜R17）
currentWorld_ singularization: 後段で GO
```

---

## H.11 §4 クロスカッティング未検証項目の確定（2026-08-14 ユーザーレビュー反映）

ユーザーレビュー（2026-08-14）で指摘された以下の未確定事項について、ソースコードを直接調査・確定した。

### H.11.1 未確定事項の確定結果

| ユーザーレビュー指摘 | 調査方法 | 結果 | 確定ステータス |
|---|---|---|---|
| `rebuildRequestGeneration` が request sequence か logical build generation か | `rg -n "rebuildRequestGeneration"` (23箇所) | `AudioEngine.h:2435` `// 非同期リビルドの競合防止用` + `++:643`, `++:127`, `++:134(ReleaseResources)` 全て increment | **logical build generation** (request sequenceではない)。`isRebuildObsolete(gen)` 比較で使用。`generation 10/11/10` ABA は**可能**。A2-G18（generation binding）を支援するため、generationのみではなくRuntimeIdentityを併結する必要あり |
| `ShutdownDrainToken` が default constructible か | `sed -n '1,40p' src/audioengine/ShutdownScope.h` | `ShutdownDrainToken() noexcept = default;` + `bool valid_ = true;` | **YES, default constructible & valid-by-default**。`ShutdownDrainToken{}` で `consume()==true` になり得る → 資料 H.9.3 の「forged Permit impossible」の根拠で **現行 ShutdownDrainToken は ReclaimPermit 代替不可**を技術的に確認 |
| retry policy に exponential backoff あるか | `rg -n "kMaxRecoveryConsecutiveFailures backoff RetryDisposition BuildFailureClass"` | `RebuildDispatch.cpp:1003` `kMaxRecoveryConsecutiveFailures=4` + `break` のみ。`RetryDisposition`/`BuildFailureClass` は**grep 0 件** | **NO exponential backoff**。連続4回失敗で break（次サイクル委譲）のみ。exponential backoff は**未実装** |
| `pendingReclaimHandles` が identity set か count か | `rg -n "pendingReclaimHandles reclaimIdentity ReclaimIdentity"` | `AudioEngine.Retire.cpp:57-92` / `AudioEngine.h:4229-4269` — **handle vector** (`std::vector` + mutex)。`Coordinator.cpp:512` isFullyDrained は `reclaimInFlightCount_` (count) で判定 | **count only** — `pendingReclaimHandles_` は AudioEngine 側に scatter。Coordinator は count で判定。H.9.3 の「identity set primary authority」は**未実装** |
| §1.8.10.1 `BUILD_ERROR_ENUM_COUNT` static_assert | `rg -n "BUILD_ERROR_ENUM_COUNT"` (src/) | **0 件** | **未実装**。「検討案」セクションを「現在のアプローチ」から「検討案（未実装）」へ修正済み |
| `setReclaimInFlightCount(0)` store パターン | `rg -n "setReclaimInFlightCount"` (Coordinator.cpp) | `:239` 定義, `:240` publishAtomic, `:660` `+1`, `:671` **`=0` store** | **存在する** — `reclaim complete` 時に `=0` を store。H.9.3 の「count だけでは identity が失われる」の直接的根拠。**setReclaimInFlight を fetch_sub 1 に変更必須** |
| Path B `enqueuePublicationIntent` に shutdown gate あるか | `rg -n "enqueuePublicationIntentForRuntimeCommit"` (RebuildDispatch.cpp) `:987/1060/1230` | `enqueuePublicationIntent` (h:324) は reservation-before-push ✅ だが **shutdown gate なし**。Path B は `enqueuePublicationIntentForRuntimeCommit` を経由 | **gate 未実装**。H.6.2 (CONFIRMED) と一致. T8 未実装 |

### H.11.2 §1.8.5.2 追加 — BuildError → FailureClassification/RetryDisposition 分離仕様

> **1.8.5.2 との関係（2026-08-14 別視点調査で明確化）**: 本 H.11.2 は 1.8.5.2 の `BuildFailureClass` / `RetryDisposition`（実装詳細）を**概念的分類**（`FailureClassification` + `RetryDisposition{NoRetry, RetryBackoff, RetryImmediate}`）として再表現したもの。
> 両者の RetryDisposition 値は**統一する**（Retry→RetryBackoff、Discard→NoRetry、Inspect→ContextDependent）。実装の single source of truth は descriptor table（§1.8.10.3 / line 1059-1066）で、本 H.11.2 はその抽象化である。

現行 `BuildError` enum（`RuntimeBuilder.h:107-116`）:

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

**問題**: `BuildError` は failure 原因を表すが、retryability は caller が推測して判定 → 8/14レビューの「retryability 判定に不十分」を確認。

**推奨分離**:

```cpp
enum class FailureClassification : uint8_t {
    Permanent,    // InvalidInput — retry 無意味
    Transient,    // ResourceUnavailable, WarmupFailed — retry 有効
    Infrastructure, // ConvolverFailure, PrepareFailure — retry 有効（環境依存）
    Fatal         // InternalError, MKLFailure — retry 無意味/異常終了
};

enum class RetryDisposition : uint8_t {
    NoRetry,      // retry 禁止（permanent / fatal）
    RetryBackoff, // exponential backoff 付き retry（transient / infrastructure）
    RetryImmediate // immediate retry（warmup failure 等 latency-sensitive）
};

struct BuildOutcome {
    BuildError error;
    FailureClassification classification;
    RetryDisposition retry;
    std::optional<std::chrono::milliseconds> retryDelay;
};
```

**retryability マッピング (retry mapping)**:

| BuildError | Classification | RetryDisposition | 現状 (RetryDispatch.cpp:990-1048) |
|---|---|---|---|
| None | N/A | NoRetry | ✅ (success path) |
| InvalidInput | Permanent | NoRetry | ❌ (未分類 — `build() == nullptr` check only) |
| ResourceUnavailable | Transient | RetryBackoff | ❌ (consecutive failure count only, no backoff) |
| MKLFailure | Fatal | NoRetry | ❌ (未分類) |
| ConvolverFailure | Infrastructure | RetryBackoff | ❌ |
| PrepareFailure | Infrastructure | RetryBackoff | ❌ |
| WarmupFailed | Transient | RetryImmediate | ❌ (retry but no classification) |
| InternalError | Fatal | NoRetry | ❌ |

> **⚠️ 注記（§1.8.12 項目3 と整合）**: `MKLFailure` は `RuntimeBuilder.h:111`（enum）と
> `RuntimeBuilder.cpp:63-64`（toString）にのみ存在し、**build() からは現在どのコードパスからも
> 生成されない**（FFT は MKL DFTI → Intel IPP に換装済み・MKL VML/BLAS は Message Thread で使用）。
> したがって本行の mapping は**将来の回復備え（defensive）**であり、現行実行時には到達しない。
> 実装時は `BuildContext` に基づき「一時的 resource exhaustion → Transient/RetryBackoff」
> 「persistent config → Fatal/NoRetry」を決定する。
> **⚠️ 第四者レビュー §21（2026-08-14）反映: 本マッピングは「デフォルト分類」であり、固定 lookup table ではない。**
> `FailureClassification` は `BuildError` 単独ではなく **BuildContext（一時的 resource exhaustion / persistent invalid configuration 等）を含めて決定**する。
> 例: `MKLFailure` は「一時的な resource exhaustion」なら `Transient / RetryBackoff`、「persistent invalid configuration」なら `Fatal / NoRetry`。
> 実装は `BuildError → Classification` の 1:1 固定でなく、**`BuildError + BuildContext → FailureClassification → RetryDisposition`** の順で解決する（本テーブルはデフォルト表として維持）。

**検証**: `grep -n "ResourceUnavailable\|InvalidInput\|InternalError\|WarmupFailed\|ConvolverFailure\|PrepareFailure\|MKLFailure" src/audioengine/AudioEngine.RebuildDispatch.cpp` — 5.8節の `BuildError` 分類は `BuildResult.error` を `toString()` してログ出力するのみ。`retryability` 判定は**存在しない**。連続失敗4回で `break`（`kMaxRecoveryConsecutiveFailures=4`）のみ。

### H.11.3 §2.4 Proof/Permit 型定義（未実装 — A2-G04/G05/G16/G21〜G23 対応）

**現状**: `ShutdownQuiescenceProof` / `ReclaimPermit` **型は存在しない** (`grep -rn "ShutdownQuiescenceProof\|ReclaimPermit" src/ | grep -v test` → 0 件)。`ISRRuntimePublicationCoordinator.cpp:643` にコメントのみ。

**推奨型定義**:

```cpp
// src/audioengine/ISRShutdown.h

struct ShutdownRuntimeIdentity {
    uint64_t engineInstanceId{0};    // runtime instance identity（第十三者 #15 — Runtime A/B 間の Permit 混用防止）
    uint64_t shutdownGeneration{0};  // requestShutdown 毎にインクリメント
};

struct PermitIdentity {
    ShutdownRuntimeIdentity runtime;
    // ⚠️ 第十三者 #15: runtime.shutdownGeneration と重複（冗長）。runtime 内の generation に一本化推奨
    uint64_t shutdownGeneration{0};
    uint64_t epochGeneration{0};      // m_epochDomain の generation
    uint64_t readerRegGeneration{0};  // reader registration closed の generation

    bool operator==(const PermitIdentity&) const noexcept = default;
};

class ShutdownQuiescenceProof {
public:
    ShutdownQuiescenceProof(const ShutdownQuiescenceProof&) = delete;
    ShutdownQuiescenceProof& operator=(const ShutdownQuiescenceProof&) = delete;
    ShutdownQuiescenceProof(ShutdownQuiescenceProof&&) noexcept = default;
    ShutdownQuiescenceProof& operator=(ShutdownQuiescenceProof&&) noexcept = default;

    const PermitIdentity& identity() const noexcept { return identity_; }

private:
    friend class ShutdownRuntime;
    explicit ShutdownQuiescenceProof(PermitIdentity id) noexcept : identity_(id) {}
    PermitIdentity identity_;
};

class ReclaimPermit {
public:
    ReclaimPermit(const ReclaimPermit&) = delete;
    ReclaimPermit& operator=(const ReclaimPermit&) = delete;

    // move は許すが、single-use を保証するため Consumed 状態を持つ
    ReclaimPermit(ReclaimPermit&& other) noexcept
        : identity_(other.identity_),
          state_(other.state_.exchange(PermitState::Consumed, std::memory_order_acq_rel)) {}

    [[nodiscard]] bool consume() noexcept {
        auto expected = PermitState::Valid;
        return state_.compare_exchange_strong(
            expected, PermitState::Consumed,
            std::memory_order_acq_rel, std::memory_order_acquire);
    }
    // ReclaimPermit{} はコンパイルエラー（private ctor） → T6 (forged Permit impossible)

private:
    friend class ShutdownRuntime;
    explicit ReclaimPermit(PermitIdentity id) noexcept : identity_(id) {}

    PermitIdentity identity_;
    std::atomic<PermitState> state_{PermitState::Valid};
};

enum class PermitState : uint8_t { Valid, Consumed };
```

> **⚠️ 実装補完注記（2026-08-14 別視点調査で確定）**: `epochGeneration` / `readerRegGeneration` は**現行ソースに存在しない**。
> `grep -rn "readerRegGeneration\|readerRegistrationGeneration" src/` → **0 matches**、`EpochDomain.h` には generation フィールドが**一切ない**
> （epoch は単一 `uint64_t` 値のみ）。つまり本 PermitIdentity の generation 束縛（A2-G19/G20）を実装するには、
> **EpochDomain に epoch-generation カウンタ + reader-registration generation カウンタを新規追加する必要がある**。
> これは INV-LIFE-12（readerRegistrationClosed 単独では認可にならない）の設計根拠でもある。

**reclaim API**:

```cpp
// ISRRuntimePublicationCoordinator.h:373-377 を置換
[[nodiscard]] bool reclaim(
    ReclaimMode mode,
    const DSPHandle& handle,
    class DSPHandleRuntime& handleRuntime,
    class ISRRetireRouter& router,
    ReclaimPermit permit) noexcept;  // bool → ReclaimPermit
```

### H.11.4 §2.5 AdmissionState 4-state FSM（未実装）

**現状**: `ShutdownPhase` は 11 状態だが `AdmissionState {Open/Closing/Closed/Faulted}` は**未実装**。`isShutdownInProgress()` が unique check に使われている。

**推奨**:

```cpp
enum class AdmissionState : uint8_t {
    Open,     // 新規 enqueue 許可
    Closing,  // closeAdmission() 呼出し済み / producer join 中
    Closed,   // producer join 完了 / 新規 enqueue 拒否
    Faulted   // underflow 等で Faulted
};
```

| 遷移 | 許容 |
|---|---|
| Open → Closing | ✅ `closeAdmission()` |
| Closing → Closed | ✅ `joinProducers()` 完了 |
| Closed → Open | ❌ **禁止** (resurrection 防止) |
| 任意 → Faulted | ✅ (underflow / overflow) |

### H.11.5 §1.12 Recovery coalesce — Building 中の policies（R1〜R17 に R18 追加）

ユーザーレビュー#25（Building 中の coalesce）の確定:

**現状**: `RebuildDispatch.cpp:991-1005` は `takePendingRecoveryAdmission()` で **single durable slot** から 1 つ取り出し、Building 状態に遷移。Building 中に新たな Recovery が来た場合、`submitRecoveryRequest` は `pendingRecoveryAdmission_` に**上書き**する（`recoveryGeneration` 更新）。これは **Building中の object 上書き** であり、**NO-GO**。

**R18 (追加)**: `bounded durable table full` は**coalesce 成功として偽装してはいけない**。Table full 時は `explicit admission failure + higher-level durable fallback / Faulted` へ遷移。

**R17 補強 — Building 中の Recovery 受信時の policy**:

```
Building(A)
    ↓ 新規 Recovery B 受信
canSupersede(B, A)?
    ├── true → B を DurablePending (Building は継続)
    ├── false → B を table に保持 (capacity 許容内なら)
    └── table full → B を reject + Failured（R18）
```

Building 中の `currentBuildSource` を上書きしない:

```cpp
// ❌ Building 中の object を直接上書きする実装
pendingRecoveryAdmission_ = std::move(newAdmission);  // R18 violation

// ✅ Building を継続し、新規は別 obligation として保持
if (canSupersede(newAdmission, buildingAdmission)) {
    // B は A をsupersede する → B を DurablePending に入れて A 完了後処理
    durableTable.push(B);  // Building(A) は中断しない
} else {
    if (durableTable.size() < kMaxDurableRecoveryAdmissions)
        durableTable.push(B);
    else
        return AdmissionResult::TableFull;  // ❌ silent drop 禁止（R18）
}
```

### H.11.6 §3 実装順序（2026-08-14 ユーザーレビュー修正版）

ユーザーレビューで提言された **16-commit 実装順序**:

```text
Commit 1     PermitIdentity / ShutdownRuntimeIdentity / ShutdownQuiescenceProof / ReclaimPermit (type only, production 未接続)
Commit 2     ShutdownRuntime::tryMakeQuiescenceProof() (full condition check, production 未接続)
Commit 3     ReclaimPermit validation (identity match / single-use consume / post-proof irreversibility)
Commit 4     reclaim() API: bool → ReclaimPermit (compile error で bool 渡しを禁止)
Commit 5     AdmissionState 4-state FSM (Open→Closing→Closed, Closed→Open 禁止)
Commit 6     Path B: enqueuePublicationIntentForRuntimeCommit に tryAdmitPublication() gate
Commit 7     Path A: submitPublishRequest に admission token (既実装確認: :316 trySubmitImpl → :350 RejectedShutdown)
Commit 8     Recovery admission: submitRecoveryRequest に shutdown gate (既実装済み — P2-4 Step B)
Commit 9     Build/Publish admission gate (Path A/B/C/D 4経路統一)
Commit 10    setReclaimInFlightCount(0) store → fetch_sub(1) + underflow check (R4 Phase 2 Audit #4 対応)
Commit 11    external setter 9 call sites → semantic ++/-- (B0+B1)
Commit 12    external setter API declaration 削除 (T1 grep 0 件)
Commit 13    pendingReclaimHandles を identity set に (reclaim identity begin/end)
Commit 14    pendingReclaimHandles_.empty() を ShutdownCompletionProof 条件に（H.11.11.9.3 C1 — Proof から分離、循環排除）
Commit 15    T1-T13 + T9 (concurrent double reclaim) + T10 (Permit ABA) + T11 (setter resurrection) + T12 (Path B race stress) + T13 (destruction audit)
Commit 16    production reclaim A2 接続 (G1〜G23 PASS 前提)
```

**注（第十三者 #3 で更新）**: Commit 1-3 は型導入のみ (production reclaim 未接続) 方針。**旧 bool API は production migration 開始時点で削除し、コンパイラを migration guard として利用**（`reclaim(..., bool)` が残っていたら compile error）。二重 authority（新しい Permit 経路 + 古い bool 経路）を避けるため、**旧 bool overload を残したまま migration を進めない**。A2-G09 (bool reclaim API 削除) は Commit 4 / Step 9 で実現。

> ⚠️ **実装順序の正本**: 本 16-commit は第四者レビュー時点の順序。**最新の正本は H.11.17.5 の 15-Step**（bool API 早期削除・compile guard を含む）。実装時は 15-Step を優先する。

### H.11.7 T9〜T13: 追加受付テスト（A2 完了条件）

ユーザーレビュー#27で提言された T9〜T13を正式受付テストに追加。

| Test | 内容 | 実装状況 |
|---|---|---|
| T9 | concurrent double reclaim: 同じ Permit で 2 スレッドが reclaim | 🔴 未実装 (PermitState atomic CAS は Commit 1 で導入) |
| T10 | Permit identity ABA: Shutdown A/G1 → B/G2 → C/G1 で A の Permit を C で reject | 🔴 未実装 (generation + runtime identity binding で対応) |
| T11 | setter resurrection: 旧 setter API が production code から完全消去 | 🔴 未実装 (T1 = 0 call sites で確定) |
| T12 | Path B race stress: 数千回 closeAdmission() vs enqueuePublicationIntent() 競合 | 🔴 未実装 (AdmissionState 4-state + AdmissionToken で対応) |
| T13 | destruction audit: delete/destroy/reset/release/unique_ptr/owner reset の全 physical destruction path を static audit | 🔴 未実装 (A2-G23 対応、H.6.7 で部分検証済み) |

### H.11.8 §1.8.10.1 — static_assert は未実装 (再確認)

`BUILD_ERROR_ENUM_COUNT` は**ソースに存在しない** (`grep -rn "BUILD_ERROR_ENUM_COUNT" src/` → 0 件)。

§1.8.10.1 を「現在のアプローチ」から「検討案（未実装）」へ**既に修正済み** (line 1011/2309)。この H.11.8 は**再確認**。将来実装時の参照先。

**推奨実装**:

```cpp
// RuntimeBuilder.h
enum class BuildError { None, ..., InternalError };
inline constexpr int kBuildErrorCount = 8;
static_assert(static_cast<int>(BuildError::InternalError) + 1 == kBuildErrorCount,
              "BuildError enum mismatch — toString() が not-exhaustive となる");

// toString の網羅性をコンパイル時検証
consteval bool verifyToStringExhaustive() {
    switch (BuildError::None) {
        case BuildError::None: case BuildError::InvalidInput: /* ... all cases ... */
        case BuildError::InternalError: return true;
    }
    return false;  // unreachable
}
static_assert(verifyToStringExhaustive(), "toString が全 enum を網羅していません");
```

---

### H.11.9 §1.8.7 Build パスアーキテクチャ — 3 系統の分離 (2026-08-14 追加確認)

現行は **2 系統** (`build() + validateWarmup()` at `RuntimeBuilder.cpp:428-479`) だが、**warmup failure と build failure を同じ `BuildResult.error` に混在**させている。

3 系統分離を推奨:

| Path | 呼び出し | Error source | RetryClassification |
|---|---|---|---|
| Initial Build | `RuntimeBuilder::build()` | build-time error | BuildError |
| Warmup Validate | `RuntimeBuilder::validateWarmup()` | warmup runtime error | WarmupError (separate enum) |
| Runtime Build | `AudioEngine.RebuildDispatch.cpp:89` `buildRuntimeWorld` | runtime transition error | RuntimeBuildError |

**問題**: 現状 `validateWarmup()` は `Return BuildError` だが、`build()` と `validateWarmup()` の error を **同じ `BuildResult.error`** に詰めて `toString()` している（:57-71）。warmup failure (`WarmupFailed`) が build failure (`InvalidInput`) と同じ分類で扱われている。

---

### H.11.10 未確定事項の確定最終確認

| 未確定事項 | 確定結果 |
|---|---|
| `rebuildRequestGeneration` = request sequence | ❌ → **logical build generation** (generation+identity binding 必須) |
| `ShutdownDrainToken` default constructible | ✅ **確認済み** (forged 可能 → ReclaimPermit で置換) |
| retry exponential backoff | ❌ **未実装** (RetryDisposition で対応) |
| `pendingReclaimHandles` identity set | ❌ **count only** (identity set へ昇格未実装) |
| `BUILD_ERROR_ENUM_COUNT` static_assert | ❌ **未実装** (検討案) |
| `setReclaimInFlightCount(0)` store | ✅ **確認済み** (fetch_sub 1 へ変更必須) |
| Path B shutdown gate | ❌ **未実装** (H.6.2 CONFIRMED) |
| `BuildError` retryability separate | ❌ **未実装** (FailureClassification/RetryDisposition) |
| 3 パス Build architecture | ❌ **2-path 混在** (warmup/build 同じ error field) |
| Building 中 coalesce | ❌ **single slot overwrite** (R17/R18 で対応) |
| AdmissionState 4-state | ❌ **未実装** (ShutdownPhase のみ) |
| Proof/Permit 型 | ❌ **完全未実装** (コメントのみ) |
| `readerRegGeneration` / `epochGeneration` | ❌ **ソースに存在しない** (`grep -rn "readerRegGeneration" src/` → 0、`EpochDomain.h` に generation フィールドなし。H.11.3 注記: PermitIdentity の generation 束縛 A2-G19/G20 実装時は EpochDomain への新規追加が必要) |
| `MKLFailure` の生成パス | ❌ **どのコードパスからも生成されない** (`RuntimeBuilder.h:111` enum + `:63-64` toString のみ。FFT は Intel IPP 換装済み。H.11.2 注記: 将来の defensive mapping のみ) |
| direct destruction paths（`destroyDSPCoreNode` 直接呼び出し） | ✅ **4 箇所を棚卸し確定**（H.11.12）: DSPLifetimeManager.cpp:123（rollback）/ RebuildDispatch.cpp:898（DSPGuard）/ :967・:1040（recovery warmup 失敗）。全て**未公開 DSP** に限定 → EBR 保護不要（UAF なし）。A2-G23 / T13（destruction audit）の対象 |

→ **全未確定事項はソースコードで調査・確定された。文書内に不明確な技術主張は残っていない。**

---

### H.11.11 第四者レビュー反映 — Lifetime Authority Closure 設計確定（2026-08-14）

第四者レビュー（2026-08-14）が「isFullyDrained と Proof/Permit を一本の **Lifetime Authority Closure** として再設計すべき」と提言。全項目をソースコードで検証し、以下の設計を確定した。**全て現状未実装**（H.11.1〜H.11.10 の検証結果と整合）。

#### H.11.11.1 責務の3層分離 + INV-DRAIN-1/2（未実装）

`isFullyDrained()` / `shutdownPhase` / `reclaim()` の概念的な近接を3層に分離する。

| 層 | 責務 | 現状（ソース） |
| --- | --- | --- |
| Layer 1: Lifetime Accounting | 「何がまだ残っているか」— 観測のみ | 現行 counter 散在 + external setter 9 call sites（H.11.1） |
| Layer 2: Quiescence Proof | 「新 obligation が発生せず、既存 obligation が settle した」— ShutdownRuntime のみ生成可能 | 型なし（H.11.3） |
| Layer 3: Reclaim Capability | 「この shutdown transaction について physical reclaim を実行する権限」— single-use ReclaimPermit | 型なし（H.11.3） |

**INV-DRAIN-1**: `isFullyDrained() == true` ⇒ 現在観測可能な lifetime obligation は存在しない
**INV-DRAIN-2**: `isFullyDrained() == true` ⇏ ShutdownQuiescent reclaim が許可される（**観測 ≠ 認可**）

これは 5.10 節の isFullyDrained 16条件（15 return-body + 1 swapPending_ pre-check）を**捨てるのではなく、観測 predicate として固定**する方針。16条件は監査用に維持（H.11.11.9.3 **C2** の一部 — 第五者レビューで旧 P6 から移動）。

#### H.11.11.2 LifetimeAccounting クラス — semantic event API 具体化（未実装）

H.11.1 の「external setter 9 call sites」撤去（B0/B1）の具体形。現行は `AudioEngine.Threading.cpp` 等が実測 snapshot を `setXXXCount()` で Coordinator へ上書き（9 call sites / 6 files）。**private化ではなく削除**し、semantic event API に閉じ込める。

```cpp
// src/audioengine/ISRLifetimeAccounting.h（新規・設計のみ）
class LifetimeAccounting {
public:
    void onRetireAccepted() noexcept;        // B0: setRetireBacklogCount 代替
    void onRetireConsumed() noexcept;        // fetch_sub 前 old>0 検証（underflow → Faulted）
    void onDeferredRetireAccepted() noexcept;
    void onDeferredRetireConsumed() noexcept;
    void onPublicationAccepted() noexcept;
    void onPublicationConsumed() noexcept;
    void onReclaimStarted(ReclaimIdentity) noexcept;   // H.11.11.6 identity set
    void onReclaimCompleted(ReclaimIdentity) noexcept;
    void onReservationAcquired() noexcept;
    void onReservationReleased() noexcept;

    [[nodiscard]] bool isDrained() const noexcept;     // INV-DRAIN-1 の観測
private:
    std::atomic<uint64_t> retireBacklog_{0};
    std::atomic<uint64_t> deferredRetireResidency_{0};
    std::atomic<uint64_t> publicationResidency_{0};
    std::atomic<uint64_t> reclaimInFlight_{0};         // 補助値に降格（H.11.11.6）
    // ...
};
```

**A2-G02 対応**: counter mutation = single authority（semantic event のみ）。`AudioEngine` 側は `setXXXCount(snapshot)` を呼べない（compile error — A2-G10/Commit 12）。

#### H.11.11.3 Proof 生成条件 Q0〜Q7 + 11ステップ transaction（未実装）

`ShutdownRuntime::tryMakeQuiescenceProof()` は `if (isFullyDrained()) return Proof{};` の簡易生成を**禁止**（A2-G05/Commit 2）。9条件を authority から取得して検証する。

| # | 条件 | ソース |
| --- | --- | --- |
| Q0 | OutstandingAdmissionReservations == 0 | 新規（第九者必須修正2）— close vs in-flight enqueue の race を型付き Proof で閉じる |
| Q1 | AdmissionClosed | 未実装（H.11.4 AdmissionState） |
| Q2 | AllProducersJoined | `requestShutdown`（ISRRuntimePublicationCoordinator.cpp:665） |
| Q3 | ReaderRegistrationClosed | `EpochDomain.h:592`（bool 読取 — proof ではない） |
| Q4 | ActiveReaders == 0 | `m_retireRouter->activeReaderCount()` |
| Q5 | EpochSettled（EpochQuiescenceEvidence） | `m_epochDomain` generation + registration closed + readers == 0（第七者 C: generation 値だけで定義しない） |
| Q6 | postStopEnqueueCount == 0 | `ISRShutdown.h:144` / `:321`（tracking 実装済み） |
| Q7 | NoResurrection（AdmissionState Closed 固定） | 未実装（H.11.4） |

> **⚠️ 第五者レビュー（2026-08-14）反映 — 循環排除**: 旧 P6（`LifetimeAccounting.isDrained()`）と旧 P7（`PendingReclaimIdentities.empty()`）は **Proof 条件から除外**し、`ShutdownCompletionProof`（H.11.11.9）側へ移す。
> 理由: ReclaimPermit なしでは pending reclaim を処理できないのに、pending reclaim が空でないと Permit を作れない、という**循環**になる
> （Proof → Permit → reclaim → pending 空 → ... を Proof が要求 → 循環）。
> Proof は「**これ以降、新しい lifetime obligation が発生しない**」ことのみを証明する（Q0〜Q7）。
> `isFullyDrained()` / `reclaimInFlight == 0` / `physicalDSP == 0` も Proof に入れない（completion 側の検証対象）。

**11ステップ transaction**（第四者レビュー §30 の順序を確定・第五者で Q1〜Q7・九者で Q0 追加）:

```text
1. close admission            → Q1（AdmissionClosed / Open→Closing 遷移）
2. wait outstanding reservations == 0 → Q0（AdmissionReservations released → Closing→Closed 遷移前提）
3. join/wait producers        → Q2（AllProducersJoined）
4. close reader registration  → Q3
5. wait active readers == 0   → Q4
6. stop coordinator           → RuntimeIntentCoordinator::ShuttingDown
7. stop builder               → RebuildDispatch 停止
8. settle epoch               → Q5（EpochSettled / EpochQuiescenceEvidence）
9. verify postStopEnqueue == 0 → Q6
10. verify no resurrection     → Q7
11. atomically seal shutdown transaction → Proof 生成（INV-PERMIT-ATOMICITY / AC-LIFE-NEW-2）
    ※ pendingReclaim / isDrained は Proof に含めない — Completion 側（H.11.11.9.3 C1/C2）で検証
```

**Step 11 が唯一の linearization point**。Proof 生成後に obligation-generating API が reject されることを FSM（Closed 固定）で保証（H.11.4 と整合）。Q0 は「AllProducersJoined と AdmissionToken 消費済みは別」であるため、**in-flight reservation が 0 になったこと**を Proof 条件として明示する（第九者必須修正2）。Proof 生成後は ReclaimPermit（H.11.11.3/9）→ ReclaimAuthority → ShutdownCompletionProof（H.11.11.9.3）と進む。

#### H.11.11.4 postStopEnqueue > 0 → Faulted 昇格（未実装）

**ソース検証**: `markPostStopEnqueue()` は `fetchAddAtomic(sh6PostStopEnqueueCount_, 1)` のみ（ISRShutdown.cpp:321-323）。production call site は **1 箇所**（AudioEngine.Commit.cpp:459）。診断 tracking のみで、**Proof 条件にも Faulted 昇格にも使われていない**。

**確定方針**:

- `postStopEnqueueCount == 0` を Proof 条件 P8 に昇格（H.11.11.3）
- `postStopEnqueueCount > 0` 時は **Faulted へ昇格**（または Proof unavailable）— 「警告して続行」は NO-GO
- `markPostStopEnqueue()` を 4 経路（Publication/Recovery/Build/Retire）全てで呼ぶ（現状 1 call site → 4 経路への拡張）

#### H.11.11.5 reclaimNormal / reclaimShutdownQuiescent 分離 API（未実装）

現行 `reclaim(ReclaimMode, ..., bool readerRegistrationClosed)`（ISRRuntimePublicationCoordinator.h:373-377、3 production call sites: AudioEngine.h:2035 / ReleaseResources.cpp:426,436）。H.11.3 の bool→ReclaimPermit 置換に加え、**Mode と Capability を型で一致させる**:

```cpp
// Mode 分岐を消し、Capability 型で経路を区別する（Mode+Capability 不一致が不可能に）
[[nodiscard]] bool reclaimNormal(
    const DSPHandle& handle, DSPHandleRuntime& rt,
    ISRRetireRouter& router) noexcept;   // RuntimeEBR / 通常経路

[[nodiscard]] bool reclaimShutdownQuiescent(
    const DSPHandle& handle, DSPHandleRuntime& rt,
    ISRRetireRouter& router, ReclaimPermit permit) noexcept;  // ShutdownQuiescent
```

- `reclaim(ReclaimMode::ShutdownQuiescent, ..., false)` という誤用自体が **コンパイル不能**（A2-G09）
- `ReclaimPermit` は move-only + single-use（H.11.3 の Consumed state CAS を維持）
- 物理 destruction 成功と Permit validity は独立（`reclaim` = 「reclaim operation 開始権限を consume」）

#### H.11.11.6 pendingReclaimHandles_ → ReclaimIdentity set 昇格（未実装）

**ソース検証**: `AudioEngine.h:4229-4269` / `AudioEngine.Retire.cpp:57-92` — `std::vector<DSPHandle>` + mutex。Coordinator は `reclaimInFlightCount_`（count）で判定（Coordinator.cpp:512）。`setReclaimInFlightCount(0)` store（:671）は **count が identity を失う** 直接的根拠（H.11.1）。

**確定方針**:

- `ReclaimIdentity`（DSPHandle を包む or 拡張）を導入し、`std::unordered_set<ReclaimIdentity>` を **primary authority** に昇格
- `onReclaimStarted(id)` → insert / `onReclaimCompleted(id)` → erase（H.11.11.2）
- `pendingReclaimIdentities.empty()` を reclaim completion の primary authority（A2-G14 / Commit 13-14）
- `setReclaimInFlightCount(0)` store → `fetch_sub(1)` + underflow check（Commit 10 と整合）

#### H.11.11.7 INV-LIFE-1〜12（lifetime contract 固定）

第四者レビュー #45 の INV-LIFE を Acceptance 基準として採択（§4 AC-LIFE-NEW と整合）。

| # | 内容 |
| --- | --- |
| INV-LIFE-1 | `isFullyDrained()` = observation only（INV-DRAIN-1/2） |
| INV-LIFE-2 | `isFullyDrained() == true` ⇏ reclaim authorization |
| INV-LIFE-3 | ShutdownQuiescenceProof は ShutdownRuntime のみ生成可能 |
| INV-LIFE-4 | ReclaimPermit は ShutdownRuntime の shutdown transaction からのみ生成可能 |
| INV-LIFE-5 | Proof.identity == Permit.identity |
| INV-LIFE-6 | Permit.identity == current shutdown transaction identity |
| INV-LIFE-7 | Permit は一度しか consume できない |
| INV-LIFE-8 | Proof 生成後、lifetime obligation を生成する API はすべて reject |
| INV-LIFE-9 | Closed → Open は存在しない（no-resurrection）＝ **INV-NO-RESURRECTION**（第五者レビュー #31 Phase 0 凍結対象） |
| INV-LIFE-10 | external drain-counter setter == 0 |
| INV-LIFE-11 | pendingReclaimIdentities.empty() を reclaim completion の primary authority |
| INV-LIFE-12 | shutdownPhase / isFullyDrained / ShutdownDrainToken / readerRegistrationClosed は単独では ShutdownQuiescent reclaim authorization にならない |

#### H.11.11.8 Race A/B/C/D テスト整理（T5/T9〜T12 との対応）

| Race | 内容 | 期待 | 対応テスト |
| --- | --- | --- | --- |
| Race A | tryMakeQuiescenceProof() vs enqueuePublicationIntent() 並行 | A wins → Proof 成立 + B reject / B wins → Proof 不可。**両立禁止** | T12（Path B race stress） |
| Race B | Proof N → Shutdown N+1 → reclaim(Permit N) | **reject**（stale Permit） | T5 / T10（Permit ABA） |
| Race C | 同一 Permit で2スレッド reclaim | **exactly one 成功**（Consumed CAS） | T9（concurrent double reclaim） |
| Race D | 旧 setter が production code から再出現 | **compile error**（API-level invariant） | T11（setter resurrection） |

→ **第四者レビューの全提言はソースコードで検証・確定された。H.11.11.1〜H.11.11.8 は設計確定（実装は Commit 1-16 の順序で実施、production reclaim A2 は A2-G01〜G23 PASS まで NO-GO）。**

---

### H.11.11.9 第五者レビュー反映 — ShutdownCompletionProof と非循環 lifetime contract（2026-08-14）

第五者レビューが **Proof 内の `pendingReclaimIdentities.empty()` は循環** と指摘（ReclaimPermit なしでは reclaim 処理できないのに、pending reclaim が空でないと Permit を作れない）。H.11.11.3 で Proof 条件から除外し、以下を確定する。

#### H.11.11.9.1 5段階アーキテクチャ（Observation → Quiescence → Capability → Reclaim → Completion）

```text
LifetimeAccounting（「何が残っているか」）
        │ observe
        ▼
isFullyDrained()（Observation Only — 認可に使わない）
        │ shutdown transaction
        ▼
ShutdownQuiescenceProof（Q0〜Q7: 新規 obligation が発生しない）
        │
        ▼
ReclaimPermit（physical reclaim capability, single-use / identity-bound）
        │
        ▼
ReclaimAuthority（Epoch-safe physical reclaim）
        │
        ▼
ShutdownCompletionProof（C1〜C7: 全 obligation 消滅）
```

#### H.11.11.9.2 責務分離表（第五者レビュー #43）

| 機構 | 責務 | 認可に使うか |
| --- | --- | --- |
| `isFullyDrained()` | 現在の lifetime 状態の観測 | **No** |
| `LifetimeAccounting` | obligation の semantic accounting | 間接的 |
| `ShutdownPhase` | lifecycle state | **No** |
| `ShutdownQuiescenceProof` | 新規 obligation が発生しないことの証明 | **Yes** |
| `ReclaimPermit` | shutdown reclaim capability | **Yes** |
| `Epoch` | 通常 runtime の memory safety | **Yes** |
| `ReclaimIdentity` | reclaim obligation の identity 管理 | **Yes** |
| FIFO | deterministic processing order | **No** |
| `ShutdownCompletionProof` | 全 obligation 消滅の最終証明 | shutdown 完了判定 |

#### H.11.11.9.3 ShutdownCompletionProof の条件（H.11.11.3 旧 P6/P7 を移動 + 追加）

```text
C1  pendingReclaimIdentities.empty()      ← 旧 P7（Proof から移動）
C2  LifetimeAccounting.isDrained()        ← 旧 P6（移動、5.10 の16条件）
C3  activeReaders == 0
C4  postStopEnqueue == 0
C5  all transport queues == empty
C6  all durable recovery admissions empty ← 第七者 D 追加（recoveryAdmissionPending_ == false、drain #16 と整合）
C7  all quarantine residency empty        ← 第七者 D 追加（quarantine resident / ring / intent）
```

> **⚠️ 第七者レビュー D（2026-08-14）**: C5（transport queues）だけでは不足。Recovery は transport + durable admission の二層構造なので、**C6（durable recovery admission empty）** と **C7（quarantine residency empty）** を明示する。ただし **C2（`LifetimeAccounting.isDrained()`）が C1〜C7 を内部で全部見る設計は hidden coupling**（第九者 #5）になるため、`isFullyDrained()` / `LifetimeAccounting.isDrained()` を **completion authority に昇格させない**。NonRT の `ShutdownRuntime` が `CompletionObservation`（LifetimeAccounting + ReclaimIdentitySet + ReaderState + RecoveryAdmission + QuarantineState）を**一つの transaction で観測**し、`ShutdownCompletionProof` を生成する。C1〜C7 は観測項目の分類であり、個別に検証する（C2 は semantic lifetime obligation のみを指し、recovery / quarantine / reader / transport は C3〜C7 で個別検証）。

**第十者必須修正①（ShutdownCompletionAuthority）**: C1〜C7 をまとめて評価する `ShutdownCompletionAuthority` を設け、`ShutdownCompletionProof` の**唯一の生成 authority** にする。

```cpp
class ShutdownCompletionAuthority {
public:
    [[nodiscard]] std::optional<ShutdownCompletionProof>
    tryMakeProof(const ShutdownQuiescenceProof&) noexcept;   // C1〜C7 を全て検証
private:
    friend class ShutdownRuntime;
};
```

責務分離（第十者 #32-A）:

```text
LifetimeAccounting          → C2 の semantic lifetime obligation の single source of truth
ShutdownCompletionAuthority → C1〜C7 全条件を評価し、CompletionProof の唯一の生成 authority
```

**「LifetimeAccounting が唯一の completion authority」という表現は削除**し、上記の責務分離に置き換える。

```cpp
class ShutdownCompletionProof {
public:
    ShutdownCompletionProof(const ShutdownCompletionProof&) = delete;
    ShutdownCompletionProof& operator=(const ShutdownCompletionProof&) = delete;
    ShutdownCompletionProof(ShutdownCompletionProof&&) noexcept = default;
private:
    friend class ShutdownRuntime;
    explicit ShutdownCompletionProof(PermitIdentity id) noexcept : identity_(id) {}
    PermitIdentity identity_;
};
```

```cpp
// ShutdownRuntime に追加
std::optional<ShutdownCompletionProof> tryMakeCompletionProof() noexcept;
```

`ShutdownRuntime::tryMakeCompletionProof()` は **C1〜C7 を全て検証**した後、reclaim 完了を最終確定する（INV-LIFE-11 の authority）。

#### H.11.11.9.4 CacheMap destructor は shutdown authority ではない（第五者レビュー #15-16）

現行 `CacheMap::~CacheMap()` は `shutdownPhase >= Destroy` + `readerRegistrationClosed()` で**自分で reclaim 判断**している（AudioEngine.h:2032）。これは state snapshot に基づく判断であり、shutdown authority ではない。

**第十者必須修正③（delete-before-reclaim の順序）**: 現行 CacheMap は `delete EQCoeffCache` → `reclaim(handle)` の順（delete が reclaim より前）。`reclaim()` が false の場合、`object = deleted / handle = not reclaimed` となり lifetime state として危険。**physical lifetime safety 上の実装 blocker**。

```text
❌ 現行: delete → reclaim（reclaim 失敗で object 消滅 + handle 未回収）

✅ 新設: ReclaimPermit validation → ReclaimStarted(identity) → physical destruction → ReclaimCompleted(identity)
```

新設 API では authority 側で**検証 → 開始 → 破棄 → 完了**の順序を固定する（第十者 #19）。

```cpp
// ❌ 現行: CacheMap が自分で shutdown proof を判断
if (shutdownPhase >= Destroy)
    reclaim(ShutdownQuiescent, ..., readerRegistrationClosed());

// ✅ 将来: CacheMap は対象を ReclaimAuthority へ渡すだけ（ShutdownRuntime が Permit を発行・駆動）
// ShutdownRuntime::performShutdownReclaim(ReclaimPermit permit)
//   → ReclaimAuthority が各 DSP handle を physical reclaim
```

**ShutdownPhase = lifecycle state / ShutdownQuiescenceProof = quiescence evidence / ReclaimPermit = reclaim authority** を完全分離する。`shutdownPhase >= Destroy` の使用は「teardown behavior」の判定に限定し、**reclaim の認可には使わない**。

#### H.11.11.9.5 ReclaimPermit は万能 token にしない・粒度は reclaim phase（第五者 #38-39 + 第七者 A/B 反映）

ReclaimPermit は「全 DSP object を自由に delete できる万能 token」ではなく、**shutdown transaction の reclaim phase への参加権**。

**第九者 #8（invocation capability）**: ReclaimPermit は「physical delete の認可」ではなく、**「ShutdownQuiescent ReclaimAuthority の起動権」**。Permit 自体が `delete DSPCore` を許可するのではなく、ReclaimAuthority が epoch/lifetime validation → handle reclaim → physical destruction authority を実行する。

```text
Permit
    ↓
ReclaimAuthority（invocation）
    ↓
epoch/lifetime validation
    ↓
handle reclaim
    ↓
physical destruction authority
```

**第七者レビュー A（粒度）**: `1 Permit = 1 DSP reclaim` ではなく `1 Permit = 1 shutdown reclaim phase` とする。複数の reclaim 対象がある場合、Permit を個々の DSP に渡すのではなく **ReclaimAuthority へ一度渡す**。
**第十三者 #12（per-identity Permit 案）**: 代替として、Proof から **各 ReclaimIdentity ごとに Permit を発行**する方式も選択肢（`create ReclaimPermit(A/B/C)`）。Proof = shutdown quiescence capability、Permit = one specific reclaim transaction capability。実装時に「phase 方式」と「per-identity 方式」を選択（single-use と複数対象の両立は per-identity 方式が自然）。

```cpp
// ShutdownRuntime::performShutdownReclaim(ReclaimPermit&& permit) — 1 Permit = 1 reclaim phase
//   → consume permit → ReclaimAuthority → for each ReclaimIdentity → epoch-safe / shutdown-safe reclaim
```

**第七者レビュー B（consume linearization）**: `consume` は**最終的な capability handoff の linearization point**。reclaim 失敗後に Permit だけ消費される状態（liveness 悪化）を避ける。

```text
validate permit identity
    ↓
validate shutdown state
    ↓
validate authority ownership
    ↓
consume permit（linearization point）
    ↓
perform reclaim phase
```

各 object は `ReclaimIdentity`（handle + retireSequence）を別途登録する。

```cpp
struct ReclaimIdentity {
    DSPHandle handle;
    uint64_t retireSequence;   // deterministic ordering（H.11.11.9.6 INV-FIFO-1 と整合）
    bool operator==(const ReclaimIdentity&) const noexcept = default;
};
```

`onReclaimStarted(identity)` で insert / `onReclaimCompleted(identity)` で erase し、`empty()` を completion authority とする（H.11.11.6 と整合）。

**⚠️ ReclaimIdentity set は NonRT ReclaimAuthority 専用（第七者 #12 反映）**: `std::unordered_set` の操作（insert/erase/find）は allocation・hashing・rehash・mutex を発生させるため、**Audio Thread からは一切呼ばせない**。ISR 上、ReclaimIdentity set の操作は NonRT スレッドのみ（AC-ISR-1 と整合）。

#### H.11.11.9.6 実装順序（第五者レビュー #41 — Phase A〜J）

```text
Phase A  LifetimeAccounting（semantic event accounting / external setter = 0）
Phase B  AdmissionState / No-Resurrection（4 経路 gate）
Phase C  ShutdownQuiescenceProof（Q0〜Q7）
Phase D  ReclaimPermit（single-use / identity-bound / Consumed CAS）
Phase E  reclaim API bool 削除（reclaimNormal / reclaimShutdownQuiescent 分離）
Phase F  ReclaimIdentity set（onReclaimStarted/Completed → empty が completion authority）
Phase G  Epoch safety / retire policy 整理（INV-EPOCH-1/2 primary）
Phase H  Shutdown reclaim（ShutdownRuntime が ReclaimAuthority を駆動、CacheMap は対象を渡すだけ）
Phase I  ShutdownCompletionProof（C1〜C7）
Phase J  FIFO deterministic policy（INV-FIFO-1 secondary — memory safety に使わない）
```

**RetireRecord 構造（第五者レビュー #30 反映 — ソースに存在しないため設計のみ）**:

```cpp
// src/audioengine/ISRRetireRouter 配下（設計のみ、grep 0 件のため新規）
struct RetireRecord {
    ReclaimIdentity identity;    // lifetime accounting（H.11.11.9.5）
    uint64_t retireEpoch;        // memory safety（INV-EPOCH-1/2 primary）
    uint64_t retireSequence;     // deterministic ordering（INV-FIFO-1 secondary）
    RetireKind kind;             // DSP / Snapshot / Deferred 等
};
```

`retireEpoch`（safety）と `retireSequence`（ordering）を**分離**し、FIFO を safety に使わない（第五者レビュー #24-31 と整合）。

→ **第五者レビューにより「Proof（Q0〜Q7）」「Permit（single-use）」「Completion（C1〜C7）」が分離され、Proof → Permit → Reclaim → Completion の非循環 lifetime contract が確定した。H.11.11.3 の Proof 条件は Q0〜Q7 に修正済み、H.11.11.9.3 が旧 P6/P7 を引き受ける。**

---

### H.11.12 別視点調査 — 直接破棄パス（direct destruction paths）の棚卸し（2026-08-14）

「別の視点」として、`destroyDSPCoreNode` の直接呼び出しパス（EBR/reclaim をバイパスする physical destruction）を全量棚卸しした。**設計書で未検証・未反映だった**ため、A2-G23 / T13（destruction audit）の対象として確定する。

#### H.11.12.1 直接破棄パスの完全一覧（ソース照合）

`AudioEngine::destroyDSPCoreNode`（`AudioEngine.Threading.cpp:17-22`、単一破棄 authority）を**直接呼ぶ**パスは **4 箇所**:

| # | 呼び出し元 | コンテキスト | 破棄対象 |
| --- | --- | --- | --- |
| 1 | `DSPLifetimeManager.cpp:123`（`destroyRolledBackDSP`） | publish rollback（`RuntimePublicationOrchestrator.cpp:272-274`、executor_.publish 失敗時） | 未公開 DSPCore |
| 2 | `AudioEngine.RebuildDispatch.cpp:898`（DSPGuard デストラクタ） | `retireDSPHandleForRuntime` が false（未登録 DSPCore）のとき | rebuild-obsolete 未登録 DSPCore |
| 3 | `AudioEngine.RebuildDispatch.cpp:967` | recovery warmup 失敗時（未コミット DSP を破棄して continue） | 未コミット recovery DSP |
| 4 | `AudioEngine.RebuildDispatch.cpp:1040` | durable recovery warmup 失敗時（未コミット DSP を破棄して retry） | 未コミット recovery DSP |

**ISR 思想との整合評価**: 全 4 パスとも **「publish されていない（未登録・未コミット）DSPCore」** に限定されている。ソースコメント（RebuildDispatch.cpp:879-882）が明示する通り「rebuild-obsolete な DSPCore は publish されず RuntimeWorld に公開されていないため、EBR epoch 保護は不要」。つまり reader から到達不能な DSP のみを直接破棄しており、**UAF にはならない**（ISR 思想「公開された DSP は epoch 保護、未公開 DSP は直接破棄」と整合）。

**ただし**:

- これは **A2-G23（physical destruction paths audited）・T13（destruction audit）の対象**であり、設計書 §20（delete-site inventory）が `destroyDSPCoreNode` の single authority を確認しているものの、**呼び出し元 4 箇所の棚卸しは未反映**だった。
- `destroyRolledBackDSP`（#1）は `currentRetiringGeneration_` を `fetch_add(1)` するが、**LifetimeAccounting の `onReclaimStarted/Completed` を経由しない** — 直接破棄が lifetime accounting に計上されない経路。Proof/Completion 条件（Q/C）に影響しないが、監査上は記録すべき。
- 二重解放の既知リスク（RebuildDispatch.cpp:883-885 CAVEAT コメント: フォーマッタ起因で destroyDSPCoreNode が重複し 0xC0000005）に注意。

#### H.11.12.2 確定方針

1. **T13（destruction audit）のスコープに本 4 パスを含める** — `rg -n "destroyDSPCoreNode"` で完全一致確認（現行 4 箇所）。
2. **直接破棄は「未公開 DSP の publish 前破棄」に限定** することをコード規約として明文化（ISR 思想: 公開 DSP は ReclaimAuthority のみが破棄）。
3. 将来的に `destroyRolledBackDSP` は `ReclaimPermit` 不要の経路として明示（未公開 DSP は epoch 保護不要）か、LifetimeAccounting の監査カウンタに `onDirectDestroy()` を追加して可観測にする。
4. A2-G23 の acceptance に「直接破棄 4 パスが全て未公開 DSP のみであることの static audit」を追加。

→ **直接破棄パスは 4 箇所全て「未公開 DSP」に限定され ISR 思想と整合（UAF なし）。A2-G23 / T13 の対象として棚卸し確定。**

---

### H.11.13 第七者レビュー反映 — 実装前修正 4 点 + 補強事項（2026-08-14）

第七者レビュー（2026-08-14）が「前回の循環解消は正しい」と評価した上で、**実装前修正必須 4 点（A〜D）** と**補強事項**を指摘。A〜D は H.11.11.3 / H.11.11.9.3 / H.11.11.9.5 に反映済み。本セクションは残りの補強事項を確定する。

#### H.11.13.1 Q2 AllProducersJoined の補強（第七者 #4）

Q2 は「producer が**現在** joined している」ではなく「**今後 obligation を生成できる全入口が閉じた**」ことを意味する。したがって Q2 は **AdmissionClosed + 全 obligation-producing API が Admission を取得する**構造とセットでなければ成立しない。

- Path B `enqueuePublicationIntent()` はまだ shutdown gate なし（A2-G10 🔴 / H.6.2 CONFIRMED）
- Recovery（G11 🟡）/ Build（G12 🟡）/ Publish（G13 🟢🟡）も未達

**確定**: Q1 AdmissionClosed を型として作っても、**全 producer がその Admission を通らない限り Proof は成立させない**（A2-G05〜G09 / Commit 5-9 の 4 経路 gate 完了が前提）。

#### H.11.13.2 Step 11 linearization の実装方法（第七者 #5 / 九者で Step 11 に更新）

`if (allConditions()) return ShutdownQuiescenceProof(...)` では linearization point にならない。**atomic seal** が必要。

```cpp
// AdmissionState の atomic seal（CAS）
AdmissionState expected = AdmissionState::Closing;
if (admissionState_.compare_exchange_strong(expected, AdmissionState::Closed,
        std::memory_order_acq_rel, std::memory_order_acquire))
{
    // seal 成功 → この時点が唯一の linearization point
    return ShutdownQuiescenceProof(identity);
}
// seal 失敗 → Closing ではなかった（他スレッドが遷移）→ Proof 不可
```

全 obligation producer 側は `Admission acquire → obligation creation` を**同じ transaction semantics** に従わせる。`close()` と `enqueue()` の競合は二択を必ず成立させる:

- close wins → enqueue reject
- enqueue wins → obligation が存在するため Proof 不可（Race A と整合）

#### H.11.13.3 EpochQuiescenceEvidence（第七者 #11 / 修正 C 補強）

Q5 EpochSettled を「generation 値」だけで定義しない。generation 追加は quiescence の証明ではない（epoch=42 / gen=7 でも active reader ≠ 0 は証明されない）。

```cpp
struct EpochQuiescenceEvidence {
    uint64_t observedEpoch;        // m_epochDomain currentEpoch
    uint64_t readerGeneration;     // reader registration generation（新規）
    bool     readersZero;          // activeReaderCount() == 0
    bool     registrationClosed;   // readerRegistrationClosed()
};
```

`generation + state + active reader observation` を**一つの shutdown transaction で取得**する（Step 3/4/7 を同一シーケンスで実行）。

**第九者必須修正3（state evidence 化）**: EpochQuiescenceEvidence は「generation 値」ではなく、**実際の reader/epoch state との transactional evidence** として定義する。

```cpp
struct EpochQuiescenceEvidence {
    uint64_t epoch;                          // m_epochDomain currentEpoch
    uint64_t epochGeneration;                // epoch cycle generation（新規）
    uint64_t readerRegistrationGeneration;   // reader registration generation（新規）
    bool     readerRegistrationClosed;       // closeReaderRegistration() 後
    uint32_t activeReaders;                  // activeReaderCount() == 0 観測
    uint64_t minReaderEpoch;                 // getMinReaderEpoch()（第十者必須修正②）
};
```

取得手順（一つの shutdown transaction 内）:

```text
close reader registration
    ↓
observe active readers == 0
    ↓
settle epoch
    ↓
capture evidence（epoch / epochGeneration / readerRegistrationGeneration / closed / readers==0）
```

**Evidence 作成後に reader registration を再開できない**こと（reopen 不可能）を保証する — `AdmissionState Closed` + `readerRegistrationClosed()` の不変性に依存する（INV-LIFE-9 / INV-NO-RESURRECTION と整合）。

#### H.11.13.4 postStopEnqueue は「防波堤」（第七者 #6）

`markPostStopEnqueue()` を 4 経路に追加するだけでは不十分。**obligation を生成する全経路が必ず Admission transaction を通ること**が必要。`postStopEnqueue` は **safety proof そのものではなく、Admission gate 漏れを検出する防波堤**として扱う。

```text
理想形: tryAdmit() → AdmissionToken → obligation enqueue
postStopEnqueue = Admission gate 漏れの検出（Q6 / C4 の補助）
```

#### H.11.13.5 semantic accounting の rollback semantics（第七者 #14）

`onRetireAccepted()` 追加だけでは counter leak になる。各 operation について reserve→commit / accept→consume を明確化する。

```text
Admission acquired
    ↓
residency++（reservation）
    ↓
queue push
    ├─ success → obligation live
    └─ failure → reservation--（rollback）
```

現行 `enqueuePublicationIntent()` の reservation-before-push + rollback（h:334 fetchAdd / h:337 fetchSub）を **semantic accounting に移植する際もこの transaction semantics を壊さない**（H.11.11.2 追記）。

#### H.11.13.6 build identity: Compatibility ≠ Supersession（第七者 #18）

ソース照合: `isRuntimeBuildSnapshotSealedAndCompatible`（`RuntimeBuildTypes.h:309`、`RebuildDispatch.cpp:623` で使用）は **compatibility 比較**（sampleRate / blockSize / oversampling / processing / convolverFingerprint / irIdentityHash / convolutionConfigHash / dspParameterHash）を提供。一方 **`isSemanticSuperset`（supersession relation）は未実装**（H.6.3a と整合）。

**確定**: Recovery coalesce では既存 compatibility 関数をそのまま superset 判定に流用しない。**Compatibility ≠ Supersession** として別 API（`isSemanticSuperset`）を新設する（R1〜R10 の supersession decision はこの別 API を使う）。

#### H.11.13.7 UnpublishedDSP invariant（第七者 #21）

H.11.12 の direct destruction 4 パスについて、「未公開だから安全」をコメントだけにしない。acceptance を `UnpublishedDSP invariant` として定義する。

```text
UnpublishedDSP invariant:
    not in RuntimeWorld
    not in OwnerChannel
    not in published registry
    not observable by AudioThread
    not registered for runtime retirement
```

A2-G23 の acceptance に「直接破棄 4 パスが全て UnpublishedDSP invariant を満たすことの static audit」を追加（H.11.12.2 #4 と統合）。

#### H.11.13.8 onDirectDestroy() — Published/Unpublished の分離（第七者 #22）

`destroyRolledBackDSP` を LifetimeAccounting の `onReclaimStarted/Completed` に**入れない**（semantic confusion を防ぐ）。代わりに diagnostic として `onDirectDestroy()` を追加する。

```text
Published lifetime        → ReclaimIdentity（onReclaimStarted/Completed）
Unpublished construction rollback → DirectDestroyTelemetry（onDirectDestroy() — pendingReclaimIdentities には入れない）
```

→ **第七者レビューの修正必須 4 点（A〜D）は H.11.11.3 / H.11.11.9.3 / H.11.11.9.5 に反映済み。補強事項（Q2 補強 / atomic seal / EpochQuiescenceEvidence / 防波堤 / rollback semantics / Compatibility≠Supersession / UnpublishedDSP invariant / onDirectDestroy）は本 H.11.13 で確定。**

---

### H.11.14 第九者レビュー反映 — Lifetime Authority Convergence（2026-08-14）

第九者レビュー（2026-08-14）が「09:53 版は循環を正しく解消、ISR/lifetime として妥当」と評価した上で、**実装前最終修正 3 点**と**追加提案**を指摘。修正 3 点（Q0 AdmissionReservations / Permit phase capability / Epoch state evidence）は H.11.11.3 / H.11.11.9.5 / H.11.13.3 に反映済み。本セクションは残りの追加提案を確定する。

#### H.11.14.1 INV-ISR-LIFE-1〜6（第九者 #21 — Audio Thread 境界の明文化）

| # | Invariant |
| --- | --- |
| INV-ISR-LIFE-1 | Audio Thread は physical DSP destruction を実行しない |
| INV-ISR-LIFE-2 | Audio Thread は ShutdownQuiescenceProof を生成しない |
| INV-ISR-LIFE-3 | Audio Thread は ReclaimPermit を生成しない |
| INV-ISR-LIFE-4 | Audio Thread は ReclaimIdentity unordered_set を操作しない |
| INV-ISR-LIFE-5 | Audio Thread は external drain counter を snapshot overwrite しない |
| INV-ISR-LIFE-6 | Publication enqueue は AdmissionToken なしでは obligation を生成できない |

#### H.11.14.2 Race A〜F テスト（第九者 #22）

| Race | 内容 | 期待 |
| --- | --- | --- |
| Race A | `enqueuePublicationIntent()` vs `closeAdmission()` | A wins → token acquired / B wins → enqueue reject。**両方成功は FAIL** |
| Race B | AdmissionToken acquired → closeAdmission() → queue.push() | linearization 前なら reject / 後なら既存 obligation として完了可 |
| Race C | stale Permit（shutdown gen 10 → 11 後に Permit gen 10） | **reject** |
| Race D | stale Epoch generation（Permit epochGen 20 / current 21） | **reject** |
| Race E | stale reader-reg generation（Permit readerRegGen 30 / current 31） | **reject** |
| Race F | double consume（同一 Permit を 2 スレッド） | **exactly one succeeds** |

#### H.11.14.3 Compile-time API audit（第九者 #23）

`rg` で以下を **0 件**にすることを Acceptance Criteria とする:

```text
reclaim(... bool readerRegistrationClosed)      // reclaim authorization で bool 不使用
readerRegistrationClosed())                     // reclaim authorization path で不使用
setRetireBacklogCount( / setFallbackBacklogCount( / setDeferredRetireResidencyCount(
setReclaimInFlightCount( / setQuarantineResidentCount(   // production caller 0 件
```

A2-G01 / A2-G02 に対応。

#### H.11.14.4 G-A1〜G-E5 acceptance gate（第九者 #24）

| Gate | 合格条件 |
| --- | --- |
| G-A1 | external setter production call = 0 |
| G-A2 | `LifetimeAccounting` のみ semantic counter mutation |
| G-B1 | Path B authority-side admission gate あり |
| G-B2 | `AdmissionToken` reservation あり |
| G-B3 | Closed → Open が存在しない |
| G-B4 | Publication / Recovery / Build / Publish 全経路に No-Resurrection gate |
| G-C1 | `reclaim(bool)` = 0 |
| G-C2 | `reclaimNormal()` / `reclaimShutdownQuiescent()` 分離 |
| G-C3 | shutdown reclaim に ReclaimPermit 必須 |
| G-C4 | `readerRegistrationClosed()` 単独認可 = 0 |
| G-D1 | `epochGeneration` 実装 |
| G-D2 | `readerRegGeneration` 実装 |
| G-D3 | PermitIdentity に両 generation |
| G-D4 | stale generation Permit 拒否 |
| G-E1 | `isFullyDrained()` は Observation Only |
| G-E2 | Proof は ShutdownRuntime のみ生成 |
| G-E3 | ReclaimPermit は ShutdownRuntime のみ生成 |
| G-E4 | ReclaimIdentity primary authority |
| G-E5 | physical reclaim は NonRT |

#### H.11.14.5 実装コミット単位（第九者 #25 — 17-commit）

```text
Commit 1   LifetimeAccounting skeleton
Commit 2   semantic event hooks
Commit 3   external setter production call sites = 0
Commit 4   AdmissionState
Commit 5   AdmissionToken / reservation
Commit 6   Path B authority-side admission
Commit 7   Recovery / Build / Publish admission 統合
Commit 8   Epoch generation
Commit 9   Reader registration generation
Commit 10  ShutdownQuiescenceProof（Q0〜Q7）
Commit 11  ReclaimPermit
Commit 12  reclaimNormal / reclaimShutdownQuiescent
Commit 13  ReclaimIdentity
Commit 14  CacheMap / shutdown reclaim migration
Commit 15  ShutdownCompletionProof
Commit 16  race / stale / double-consume tests（Race A〜F）
Commit 17  A2-G01〜G23 full audit
```

**重要（第九者 #26）**: Path B を最初に直さない。**LifetimeAccounting → AdmissionToken → Path B** の順（単純な `if (state != Closed)` で済ませる危険を回避）。Path B の gate は「shutdown check」ではなく「**この publication が lifetime obligation を生成することを Admission Authority が正式に認可した**」という意味論が必要。

> ⚠️ **実装順序の正本**: 本 17-commit は第九者レビュー時点の順序。**最新の正本は H.11.17.5 の 15-Step**（bool API 早期削除・compile guard を含む）。実装時は 15-Step を優先する。

→ **第九者レビューの修正 3 点（Q0 / Permit phase capability / Epoch state evidence）は H.11.11.3 / H.11.11.9.5 / H.11.13.3 に反映済み。INV-ISR-LIFE-1〜6 / Race A〜F / Compile-time audit / G-A1〜G-E5 / 17-commit は本 H.11.14 で確定。**

---

### H.11.15 第十者レビュー反映 — Completion Authority / RT-safe Admission / ReclaimIdentity 状態遷移（2026-08-14）

第十者レビュー（2026-08-14）が「10:07 版は前版より改善、A2 lifetime の基本設計は採用可」と評価した上で、**必須修正 3 点**と**追加提案**を指摘。必須修正 3 点（ShutdownCompletionAuthority / EpochEvidence minReaderEpoch / CacheMap 順序）は H.11.11.9.3 / H.11.13.3 / H.11.11.9.4 に反映済み。本セクションは残りの追加提案を確定する。

#### H.11.15.1 AdmissionReservation ≠ TransportResidency（第十者 #24-25 / 問題D）

Path B の `tryAdmitPublication()` 導入時、**Admission reservation と PublicationIntent residency を混同しない**（別 semantic counter）。

```text
AdmissionReservation
    = 「新しい lifetime obligation を生成する権利」（Q0 の根拠）

PublicationIntentResidency
    = 「queue 内に transport object が存在する」（既存 publicationIntentResidencyCount_）
```

同じ counter に混ぜると「Admission accepted / push failed」や「push succeeded / processIntent started」の意味が曖昧になる（第十者 #25 — 新しい bug の原因）。

#### H.11.15.2 Admission は RT-safe（第十者 #5）

Path B は RT から呼ばれる可能性を排除できない。`tryAdmitPublication()` 内部で**禁止**:

```text
mutex wait
condition_variable
heap allocation
blocking queue
unbounded allocation
```

`AdmissionToken` は **atomic state + bounded reservation** として実装（現行 Intent の trivially_copyable + lock-free ring 前提を維持）。

#### H.11.15.3 G-A1 強化: direct semantic counter mutation = 0（第十者 #7）

`setter = 0` だけでなく、**Coordinator 外での direct semantic counter mutation = 0** まで静的検査対象にする。

```text
publishAtomic(retireBacklogCount_, ...)
fetchAddAtomic(retireBacklogCount_, ...)
fetchSubAtomic(retireBacklogCount_, ...)
```

が LifetimeAccounting の semantic event API 以外に存在したら FAIL。

#### H.11.15.4 Race G / Race H 追加（第十者 #28）

| Race | 内容 | 期待 |
| --- | --- | --- |
| Race G | permit consume vs destruction failure | **Permit consumed でも physical reclaim incomplete → CompletionProof unavailable → obligation remains**（Permit consume ≠ reclaim completion） |
| Race H | destruction vs ReclaimIdentity removal | ReclaimStarted(id) 後、**physical destruction が確定完了するまで set は id を保持**（ReclaimCompleted まで resident） |

#### H.11.15.5 ReclaimIdentity 状態遷移（第十者 #20-21）

```text
Published → Retired → ReclaimPending → ReclaimAuthorized → ReclaimStarted → PhysicalDestroyed → ReclaimCompleted
```

`ReclaimIdentity set` は **ReclaimPending + ReclaimAuthorized + ReclaimStarted**（physical destruction 未完了）の identity を管理。`empty()` は「全 reclaim obligation 完了」であり、「physical object 全破棄済み」と同義にしてはいけない（第十者 #21）。

#### H.11.15.6 postStopEnqueue Faulted は linearization に含める（第十者 #27）

`postStopEnqueueCount` は **Proof seal transaction と同じ同期ドメインで評価**する（T1: postStopEnqueue → T2: tryMakeProof → T3: Faulted では T2 が古い観測値で Proof を生成する可能性）。

```text
seal()
{
    if (postStopEnqueue != 0) fail;      // 同ドメインで評価
    if (admission != Closing) fail;
    ...
    state = Quiescent;
    proof = ...;
}
```

#### H.11.15.7 Gate 追加（第十者 #34）

| Gate | 条件 |
| --- | --- |
| G-B5 | AdmissionReservation と transport residency が別 authority |
| G-B6 | `tryAdmit()` は bounded / nonblocking / allocation-free |
| G-D5 | EpochQuiescenceEvidence が単一 transaction で生成される |
| G-E6 | Permit consume ≠ reclaim completion |
| G-E7 | physical destruction 前に ReclaimAuthority validation 完了 |
| G-E8 | ReclaimIdentity は destruction 完了まで resident |
| G-F1 | CompletionAuthority のみ CompletionProof 生成可能 |
| G-F2 | C1〜C7 の全条件を同一 completion transaction で評価 |
| G-F3 | C2 が C1/C6/C7 を内部的に二重管理しない |

#### H.11.15.8 ISR-LIFE-01〜10（第十者 #35 — Audio Thread 境界の機械的チェック）

```text
ISR-LIFE-01  Audio Thread → ShutdownQuiescenceProof = 0
ISR-LIFE-02  Audio Thread → ReclaimPermit = 0
ISR-LIFE-03  Audio Thread → ReclaimIdentity unordered_set = 0
ISR-LIFE-04  Audio Thread → physical delete = 0
ISR-LIFE-05  Admission tryAdmit = no malloc
ISR-LIFE-06  Admission tryAdmit = no mutex wait
ISR-LIFE-07  Admission queue push = bounded / nonblocking
ISR-LIFE-08  LifetimeAccounting mutation = atomic / RT-safe
ISR-LIFE-09  ShutdownCompletionProof = NonRT only
ISR-LIFE-10  EpochQuiescenceEvidence generation = NonRT only
```

Practical Stable ISR Bridge の「Audio Thread = read only / no lock / no malloc / no delete / no decision」原則と一致。

##### Audio Thread 境界の正確な定義（第十五者 #12 — 2026-08-14 追記）

「read only」を文字通り「shared state への write 禁止」と解釈すると、実際の RT の
bounded atomic mutation（atomic counters / retire intent / heartbeat / fade progress /
metrics）と矛盾する。正確な定義は以下の通り。

- **許可（bounded atomic mutation）**:
  - atomic metric increment
  - bounded queue reservation / push（ISR-LIFE-07）
  - RT-local DSP state mutation
  - fade sample progress
- **禁止（Runtime ownership / state authority 変更）**:
  - RuntimeWorld ownership 変更
  - publish decision
  - retire authorization
  - delete
  - shutdown authorization
  - policy decision

すなわち「**Audio Thread は Runtime ownership / state authority を変更しない**」と定義する。
ISR-LIFE-08（LifetimeAccounting mutation = atomic / RT-safe）と整合。

#### H.11.15.9 Phase 変更: Epoch Evidence を Proof より前に（第十者 #33）

Proof が EpochSettled（Q5）を含む以上、Evidence の semantic contract を先に確定する。

```text
Phase A  LifetimeAccounting
Phase B  AdmissionState
Phase B2 AdmissionReservation（Q0）
Phase C  EpochQuiescenceEvidence   ← Proof より前（第十者 #33）
Phase D  ShutdownQuiescenceProof（Q0〜Q7）
Phase E  ReclaimPermit
Phase F  ReclaimIdentity
Phase G  reclaimNormal / reclaimShutdownQuiescent
Phase H  ReclaimAuthority + physical destruction（検証→開始→破棄→完了）
Phase I  ShutdownCompletionAuthority（C1〜C7）
Phase J  FIFO policy
```

→ **第十者レビューの必須修正 3 点（ShutdownCompletionAuthority / EpochEvidence minReaderEpoch / CacheMap 順序）は H.11.11.9.3 / H.11.13.3 / H.11.11.9.4 に反映済み。AdmissionReservation 分離 / RT-safe / Race G-H / ReclaimIdentity 状態遷移 / Gate G-B5〜G-F3 / ISR-LIFE-01〜10 / Phase A-J は本 H.11.15 で確定。**

---

### H.11.16 別視点調査 — ShutdownPhase FSM 遷移規則の確定（2026-08-14）

設計書で ShutdownPhase FSM（11 状態）の遷移規則が未記載だったため、ソース照合（`ISRShutdown.h:25-40` / `ISRShutdown.cpp:108-139`）で確定した。

#### H.11.16.1 ShutdownPhase enum（ISRShutdown.h:25-40、ソース照合）

```cpp
enum class ShutdownPhase : uint8_t
{
    Running,          // 0
    AudioStopped,     // 1
    ObserverDrained,  // 2
    RetireClosed,     // 3
    EpochSettled,     // 4
    ReclaimComplete,  // 5
    EmergencyDrain,   // 6（C-2 optional、デフォルトスキップ）
    VerifyDrained,    // 7（P3 最終監査フェーズ）
    TimedOut,         // 8
    Failed,           // 9
    ShutdownComplete  // 10
};
```

> ⚠️ `transitionTo` のソースコメント（ISRShutdown.cpp:113「TimedOut(6)/Failed(7)」）は EmergencyDrain/VerifyDrained 追加前の古い番号。実際の enum では **TimedOut=8 / Failed=9 / ShutdownComplete=10**。

#### H.11.16.2 transitionTo 遷移規則（ISRShutdown.cpp:108-139）

```text
基本: 単調増加・1 ステップずつ（t == c || t == c+1）
例外: terminal 状態（TimedOut / Failed）のみをスキップする遷移は許可
      （例: ReclaimComplete(5) → ShutdownComplete(10) は TimedOut/Failed をスキップ）
禁止: 逆方向遷移・非 terminal スキップ → transitionViolations_++ + return false
isShutdownInProgress(): Running 以外 かつ terminal 以外
```

```cpp
// ソース照合: ISRShutdown.cpp:108-139
bool ShutdownRuntime::transitionTo(ShutdownPhase target) noexcept
{
    bool allowed = (t == c || t == c + 1);
    if (!allowed && t > c + 1) {
        allowed = true;
        for (int i = c + 1; i < t; ++i)
            if (!isTerminalPhase(static_cast<ShutdownPhase>(i))) { allowed = false; break; }
    }
    if (!allowed) { transitionViolations_++; return false; }
    publishAtomic(phase_, target, release);
    return true;
}
```

#### H.11.16.3 ISR 思想との整合

- ShutdownPhase は「lifecycle state」であり reclaim authority ではない（INV-LIFE-12）
- 遷移規則（単調増加）は Proof / Permit とは独立した FSM。AdmissionState（H.11.4）とは別概念
- `isTerminalPhase()` = ShutdownComplete / TimedOut / Failed（ISRShutdown.h:166-170）

→ **ShutdownPhase FSM 遷移規則（単調増加・1 ステップ・terminal スキップ許可・違反検出）をソース照合で確定。設計不足を補完。**

---

### H.11.17 第十三者レビュー反映 — 実装契約の厳密化（2026-08-14）

第十三者レビュー（10:46 版評価）が「設計凍結候補として GO、ただし実装契約をさらに固定すべき」と指摘。PermitIdentity / Commit 注記 / per-identity Permit は H.11.3 / H.11.6 / H.11.11.9.5 に反映済み。本セクションは残りの実装契約を確定する。

#### H.11.17.1 isFullyDrained の 3 層 predicate（第十三者 #33）

Observation predicate・Quiescence proof・Completion proof を**同じ boolean predicate にしない**。

```text
Observation predicate  = isFullyDrained() の 16 条件   （観測のみ、認可に使わない）
Quiescence proof       = Q0〜Q7（ShutdownRuntime のみ生成）
Completion proof       = C1〜C7（ShutdownCompletionAuthority のみ生成）
```

#### H.11.17.2 CompletionAuthority は atomic snapshot / seal で生成（第十三者 #5）

`tryMakeProof()` が「各条件を別々に読む」だけでは弱い。**C1〜C7 が同じ shutdown transaction の状態に属すること**（seal された shutdown state に対して生成）が必要。

```cpp
// ❌ 各条件を別々の atomic snapshot で読む（T1〜T5 間に新 obligation が入り得る）
bool tryMakeProof() {
    return pendingSet.empty() && lifetime.isDrained() && activeReaders == 0 && ...;
}

// ✅ seal された shutdown transaction の状態で C1〜C7 を同時観測
// CompletionAuthority は AdmissionState::Closed + no new obligation が確定した
// 同一 transaction で C1〜C7 を評価する（Race J と整合）
```

#### H.11.17.3 G-A3〜G-A7 追加（第十三者 #32）

| Gate | 条件 |
| --- | --- |
| G-A3 | LifetimeAccounting semantic counter の直接 atomic mutation = 0（LifetimeAccounting 内部以外） |
| G-A4 | snapshot overwrite API = 0 |
| G-A5 | external code からの counter reset = 0 |
| G-A6 | counter underflow → Faulted |
| G-A7 | obligation registration / completion の event pairing を static audit |

`fetchAdd(counter)` / `fetchSub(counter)` / `store(counter, value)` の直接操作を検索対象にする。

#### H.11.17.4 Race I / Race J 追加（第十三者 #38-39）

| Race | 内容 | 期待 |
| --- | --- | --- |
| Race I | Admission close vs Publication reservation | T1 acquire → T2 close → T3 push は**正当な in-flight obligation**（Proof は reservation 消費まで不可）。T2 close が先なら T1 acquire は失敗。二択をテスト |
| Race J | Completion proof vs postStopEnqueue | Admission Closed + all producers gated + seal 後の enqueue は必ず postStopEnqueue++ になる（Completion と linearization 同期） |

#### H.11.17.5 15-Step 実装順序（第十三者 #20）

| Step | 改修 | Production 接続 |
| --- | --- | --- |
| 1 | `ShutdownRuntimeIdentity` | No |
| 2 | `ShutdownQuiescenceProof` | No |
| 3 | `ReclaimPermit` | No |
| 4 | Proof 生成 API | No |
| 5 | Permit identity / single-use | No |
| 6 | `ReclaimIdentity` | No |
| 7 | `reclaimNormal()` 新設 | Yes |
| 8 | `reclaimShutdownQuiescent(...Permit)` 新設 | No |
| 9 | **旧 bool reclaim 削除** | Compile migration |
| 10 | CacheMap caller-side shutdown 判断撤去 | Yes |
| 11 | ReleaseResources migration | Yes |
| 12 | ShutdownRuntime から Permit 供給 | Yes |
| 13 | physical destruction ordering 修正 | Yes |
| 14 | race / forged / stale Permit tests | No |
| 15 | A2-G01〜G23 全 PASS 後 production 固定 | Yes |

**旧 Commit 1〜16 との対応**: 旧 Commit 1〜4 → Step 1〜5、旧 Commit 13 → Step 6、旧 Commit 4 → Step 7〜9、旧 production migration → Step 10〜13。**旧 bool API を残したまま production migration を進めない**（Step 9 で compile guard）。

**実装順序の関係（4 系統の整理）**: 本設計書には 4 つの実装順序が存在する。

```text
H.11.6   16-commit（第四者レビュー時点）     型 → ... → production
H.11.14.5 17-commit（第九者レビュー時点）     LifetimeAccounting → ... → A2 audit（Q0 追加）
H.11.15.9 Phase A-J（第十者レビュー時点）     Epoch Evidence を Proof より前
H.11.17.5 15-Step（第十三者レビュー時点）     bool API 早期削除（compile guard）← 最新・正
```

実装時は **15-Step を正**とし、16-commit / 17-commit / Phase A-J はレビュー過程の順序として参照する。

#### H.11.17.6 AC-1〜AC-8（第十三者 #22）

| AC | 条件 |
| --- | --- |
| AC-1 | `reclaim(..., bool)` production 0 件（テストも原則 0 件） |
| AC-2 | caller-side shutdown 判断（`shutdownPhase >= Destroy` / `readerRegistrationClosed()` / `isFullyDrained()` / `ShutdownDrainToken` を認可に使用）= 0 件 |
| AC-3 | `ShutdownQuiescenceProof{}` を ShutdownRuntime 以外から生成 = compile 不能 |
| AC-4 | `ReclaimPermit{}` を外部から生成 = compile 不能 |
| AC-5 | stale Permit rejection（Shutdown N Permit → Shutdown N+1 で reject） |
| AC-6 | identity mismatch rejection（Permit(A) → Reclaim(B) で reject） |
| AC-7 | double consume rejection（2 回目拒否） |
| AC-8 | post-proof resurrection rejection（Proof 後、Publication/Recovery/Build/Publish/Retire の新規 obligation 生成 reject — INV-LIFE-8/9） |

#### H.11.17.7 核心方針（第十三者 #26）

「bool を型に変える」ではなく、**「reclaim 認可の authority を caller から ShutdownRuntime へ移す」**ことが本質。

```text
isFullyDrained      = observation only
ShutdownQuiescenceProof = ShutdownRuntime only
ReclaimPermit       = ShutdownRuntime only
shutdownPhase       = lifecycle state only
readerRegistrationClosed = observation only
```

**実装開始条件**: Proof/Permit 型の存在ではなく、**`reclaim(..., bool)` と caller-side `shutdownPhase` 判定が production code から完全消滅し、Permit が ShutdownRuntime の authority からしか取得できないこと**。A2-G01〜G23 全 PASS まで production reclaim 接続は NO-GO。

→ **第十三者レビューの実装契約（bool API 早期削除 / per-identity Permit / CompletionAuthority atomic snapshot / G-A3〜A7 / Race I-J / 15-Step / AC-1〜8 / 3 層 predicate）を確定。H.11.17 で反映。**

---

### H.11.18 第十四者レビュー反映 — ShutdownTransaction FSM / Permit 再発行 / AUTH 群（2026-08-14）

第十四者レビュー（11:17 版評価）が「設計 8.5/10 GO、production 接続 NO-GO」としつつ、**実装時に仕様を固定しないと新 lifetime bug を作る 5 点**（LifetimeAccounting semantic / AdmissionToken と Path B linearization / EpochEvidence / Permit consume-retry / CompletionProof 境界）を指摘。本セクションで確定する。

#### H.11.18.1 ShutdownAdmissionState 7-state FSM（第十四者 #19-A / #11）

`AdmissionState`（H.11.4、enqueue 受付 4-state）とは別に、**ShutdownTransaction 全体の state machine** を導入する。

```cpp
enum class ShutdownAdmissionState : uint8_t {
    Open,       // 新規 obligation 許可
    Closing,    // closeAdmission() 済み / producer join 中
    Closed,     // producer join 完了 / 新規 enqueue 拒否
    Quiescent,  // QuiescenceProof 成立（seal 済み）
    Reclaiming, // ReclaimPermit で reclaim phase 実行中
    Completed,  // CompletionProof 成立
    Faulted     // invariant violation
};
```

許可遷移:

```text
Open → Closing → Closed → Quiescent → Reclaiming → Completed
任意 → Faulted
```

禁止（No-Resurrection / INV-LIFE-9）:

```text
Closed → Open
Quiescent → Open
Reclaiming → Open
Completed → Open
```

**Proof 生成後の obligation-producing API 全 reject** を本 FSM の Closed/Quiescent 状態で保証する。

#### H.11.18.2 Proof 生成の単一 linearization protocol（第十四者 #3-4）

Proof 生成も Completion と同様、**seal された transaction state から評価**する（個別 snapshot 読み取りを禁止）。

```text
OPEN
  ↓ closeAdmission()
CLOSING
  ↓ sealProducerReservations()   ← Q0（outstanding reservation == 0）をここで確定
SEALED
  ↓ verify quiescence
QUIESCENT
  ↓
Proof
```

SEALED 以前に reservation が存在すれば Proof 不可（Race I と整合）。

#### H.11.18.3 AdmissionToken の identity 束縛（第十四者 #5）

AdmissionToken に最低限以下を持たせ、**Token acquired → queue reservation → push** まで同じ transaction に束縛する。

```cpp
struct AdmissionToken {
    ShutdownRuntimeIdentity shutdown;   // runtime instance + shutdown generation
    uint64_t admissionGeneration;       // admission epoch generation（新規）
};
```

#### H.11.18.4 Permit 再発行ルール（第十四者 #11 / #19-B）

Permit consumed で partial reclaim failure（B が pending）の場合:

```text
same ShutdownTransaction（不変）
    ↓
QuiescenceProof remains valid（再生成しない）
    ↓
ReclaimPermit #N consumed
    ↓
partial failure → pending ReclaimIdentity remains
    ↓
ReclaimPermit #N+1 allowed（new phase-scoped Permit）
```

ただし、**shutdownGeneration / epochGeneration / readerGeneration が変わったら再発行不可**（stale / ABA 防止）。

#### H.11.18.5 ReclaimIdentity 重複登録・未登録完了 → Faulted（第十四者 #19-C）

```text
ReclaimStarted(A) を 2 回 → insert returns false → Faulted（「すでにあるから無視」は禁止）
ReclaimCompleted(A) で A が存在しない → Faulted（未登録完了も禁止）
```

#### H.11.18.6 RuntimeIntentCoordinator から reclaim authority を分離（第十四者 #33）

`RuntimeIntentCoordinator::reclaim()` を残さない。**intent / publication coordination** と **lifetime reclaim** を分離。

```text
RuntimeIntentCoordinator → intent / publication coordination のみ
ReclaimAuthority          → lifetime reclaim（reclaim / physical destruction の唯一の実行主体）
```

#### H.11.18.7 compile-fail migration branch（第十四者 #22）

Step 9 をより安全に:

```text
Step 7  reclaimNormal() 新設
Step 8  reclaimShutdownQuiescent(Permit) 新設
Step 8.5 compile-fail migration branch（旧 bool call sites を全部コンパイルエラー化）
Step 9  production call sites migration（3 call sites + hidden call sites を全て修正 → build green）
```

旧 API 削除 → コンパイルエラー一覧取得 → 全 caller 修正 → production build green。**見落とした旧 caller を残さない**。

#### H.11.18.8 AC-9〜AC-15 追加（第十四者 #23）

| AC | 条件 |
| --- | --- |
| AC-9 | Permit consumed + partial reclaim failure 後、CompletionProof unavailable かつ pending identities remain |
| AC-10 | same ShutdownTransaction + new Permit による retry が可能 |
| AC-11 | ShutdownGeneration changed 後の Permit 再発行不可 |
| AC-12 | ReclaimStarted(id) の二重登録は Faulted |
| AC-13 | ReclaimCompleted(id) の未登録完了は Faulted |
| AC-14 | epochGeneration equal でも epochState != Settled なら reject |
| AC-15 | Admission reservation が残っている状態では Proof 生成不可 |

#### H.11.18.9 AUTH-01〜AUTH-20（第十四者 #38）

| ID | 条件 |
| --- | --- |
| AUTH-01 | `reclaim(..., bool)` = 0 |
| AUTH-02 | `shutdownPhase` から reclaim authorization = 0 |
| AUTH-03 | `readerRegistrationClosed()` から reclaim authorization = 0 |
| AUTH-04 | `isFullyDrained()` から reclaim authorization = 0 |
| AUTH-05 | Proof 生成主体 = `ShutdownRuntime` のみ |
| AUTH-06 | Permit 生成主体 = `ShutdownRuntime` のみ |
| AUTH-07 | Reclaim execution 主体 = `ReclaimAuthority` のみ |
| AUTH-08 | physical DSP destruction 経路 = `ReclaimAuthority` のみ |
| AUTH-09 | Permit に shutdown generation を束縛 |
| AUTH-10 | Permit に epoch generation を束縛 |
| AUTH-11 | Permit に reader generation を束縛 |
| AUTH-12 | Permit double consume 拒否 |
| AUTH-13 | stale Permit 拒否 |
| AUTH-14 | forged Permit compile failure |
| AUTH-15 | CacheMap が shutdown 判断しない |
| AUTH-16 | ReleaseResources が shutdown 判断しない |
| AUTH-17 | `ReclaimIdentity` accounting が semantic event のみ |
| AUTH-18 | `unordered_set` 操作が NonRT のみ |
| AUTH-19 | physical destruction が Audio Thread から到達不能 |
| AUTH-20 | CompletionProof が reclaim 完了後のみ生成可能 |

#### H.11.18.10 核心方針（第十四者 #39）

**Permit そのものを安全性の根拠にしない**。Permit は「ShutdownRuntime が既に確立した quiescence について ReclaimAuthority を起動する capability」。実際の memory safety は `ShutdownQuiescenceProof + EpochEvidence + ReclaimIdentity + LifetimeAccounting + ReclaimAuthority` の組み合わせで成立させる。FIFO はその後の deterministic ordering（安全性の根拠ではない）。

**最終ゴール**: 「caller から bool を取り除く」ではなく、**一方向の authority chain（ShutdownRuntime → sealed Proof → ReclaimPermit → ReclaimAuthority → physical reclaim → CompletionAuthority）をコード構造として強制すること**。

→ **第十四者レビューの ShutdownAdmissionState 7-state FSM / Permit 再発行 / ReclaimIdentity 重複登録→Faulted / AC-9〜15 / AUTH-01〜20 / Coordinator からの reclaim 分離 / compile-fail migration branch を確定。H.11.18 で反映。**

---

### H.11.19 別視点調査 — PublicationAdmission::evaluate 実装詳細の確定（2026-08-14）

`PublicationAdmission::evaluate()`（Path A の shutdown gate、PublicationAdmission.cpp:1-30）の実装詳細をソース照合で確定した。設計書 H.6.2 / H.10.4 では「evaluate() shutdown check at :11」と簡単に言及していたのみで、判定項目が未詳述だった。

#### H.11.19.1 evaluate() の判定項目（ソース照合）

```cpp
// PublicationAdmission.cpp:1-30（Path A: submitPublishRequest → trySubmitImpl → evaluate）
Decision evaluate(const PublishRequest& req, AudioEngine& engine,
                  const RuntimeReaderContext& ctx) const noexcept
{
    // 1. Shutdown check（Path A の shutdown gate — H.6.2 と整合）
    if (engine.isShutdownInProgress())
        return Decision::RejectedShutdown;

    // 2. Generation staleness check（req.generation != rebuildRequestGeneration）
    if (req.generation != currentGen)
        return Decision::RejectedStaleGeneration;

    // 3. DSP finalized check（sealedSnapshot から判定）
    if (req.sealedSnapshot.irLoaded && !req.sealedSnapshot.irFinalized)
        return Decision::RejectedNotFinalized;

    // 4. HealthState check（Admission Circuit Breaker — Critical / Degraded で拒否）
    if (m_healthStateRef) {
        auto health = convo::consumeAtomic(*m_healthStateRef, std::memory_order_acquire);
        if (health == ISRHealthState::Critical)
            return Decision::RejectedPressure;  // Critical: 全 publish 拒否（フェイルクローズ）
        if (health == ISRHealthState::Degraded)
            return Decision::RejectedPressure;  // Degraded: 低優先度 publish 拒否（Coordinator 側で間引き制御）
    }

    // 5. Pressure / throttle check（P1-6: Adaptive Backpressure）
    if (engine.retirePressurePublicationThrottleActive_)
        return Decision::RejectedPressure;
    ...
}
```

#### H.11.19.2 generation staleness check と A2-G18 の関係

- **evaluate の staleness check は「publication generation」（`req.generation != rebuildRequestGeneration`）** であり、A2-G18 の「shutdown generation」とは**別概念**。
- `rebuildRequestGeneration` は **logical build generation**（H.11.1 確認済み）であり、**generation 10/11/10 の ABA が可能**。単純比較では stale request を誤認する可能性がある。
- **A2-G18 対応**: generation のみでなく **RuntimeIdentity を併結**（`rebuildRequestGeneration + runtime identity`）して ABA を防ぐ。

#### H.11.19.3 ISR 思想との整合

- evaluate() は **NonRT の CoordinatorLoop / submitPublishRequest 経路**で実行され、RT からは呼ばれない（ISR-LIFE と整合）。
- Path A は evaluate() の RejectedShutdown で gate 済みだが、**Path B（enqueuePublicationIntent）にはこの gate がない**（H.6.2 CONFIRMED）— A2-G06 の対象。

→ **PublicationAdmission::evaluate の実装詳細（Shutdown / StaleGeneration / NotFinalized / Pressure）をソース照合で確定。generation staleness check は publication generation（A2-G18 の shutdown generation とは別概念）であり、ABA 対策として RuntimeIdentity 併結が必要。**

### H.11.20 第十五者レビュー — 総合確認と新規指摘の反映（2026-08-14）

第十五者レビュー（Practical Stable ISR Bridge の ISR 原則 + Linux RCU モデルとの照合）を解析した。
総合判定は「**設計 GO、production reclaim 接続 NO-GO**」であり、計画書の方向性
（Tier 1 = 2.2 → 1.4 → 2.1 → 1.7、H.11.17.5 の 15-Step 正本）を支持する。

#### H.11.20.1 総合判定（第十五者）

| 項目 | 判定 | 計画書との関係 |
|---|---|---|
| 2.2 shutdown lifetime contract | 🟢 強くGO（最重要改修） | H.11.11〜H.11.19 で設計反映済み |
| 1.4 isFullyDrained Observation 分離 | 🟡 最重要 | H.11.17（Observation 16条件 / Quiescence Q0-Q7 / Completion C1-C7）反映済み |
| 2.1 retire FIFO と epoch safety 分離 | 🟡 条件付きGO | H.11.16（Phase J FIFO policy）反映済み |
| 1.7 currentWorld_ 廃止 | 🟡 高リスク | 反映済み（identity / read-source semantic） |
| 1.8 BuildError FailureClassification 分離 | 🔴 現案NO-GO | 反映済み（static_assert 未実装を明示） |
| 1.1 Recovery MPSC 化 | 🟢 条件付きGO | durable fallback まで MPSC 化（反映済み） |
| 1.2 Recovery coalesce | 🟡 条件付きGO | identity / supersession 方式（反映済み） |
| 1.9 quarantine wake | 🟡 条件付きGO | lost obligation 証明を優先（INV-R9-1 反映済み） |

#### H.11.20.2 authority chain の最終形（第十五者 #24 / #25）

```text
ShutdownRuntime → ShutdownQuiescenceProof → ReclaimPermit → ReclaimAuthority → physical destruction
```

- caller / bool / predicate / shutdownPhase / readerRegistrationClosed / isFullyDrained の
  いずれも reclaim authorization にならない（AUTH-01〜20 / H.11.18 と整合）。
- Practical Stable ISR Bridge の RT → Intent → NonRT Coordinator → Epoch → Reclaim → Delete 構造、
  Linux RCU の remove → grace period → reclaim 分離と一致。

#### H.11.20.3 Audio Thread 境界の正確な定義（第十五者 #12）— 新規反映

「read only」を文字通り shared state への write 禁止と解釈しない。正確には
「**Audio Thread は Runtime ownership / state authority を変更しない**」。

- **許可**: atomic metric increment / bounded queue reservation・push / RT-local DSP state mutation / fade sample progress
- **禁止**: RuntimeWorld ownership 変更 / publish decision / retire authorization / delete / shutdown authorization / policy decision

（H.11.15.8 の ISR-LIFE-01〜10 セクションにも同内容を反映済み。ISR-LIFE-08 と整合。）

#### H.11.20.4 DeferredDecision は 2-state を維持（第十五者 #20）— revision divergence 確定

- 過去の設計案では `DeferredDecision = Ready / Expired / StaleGeneration / StaleSequence` の
  4-state を提案していたが、8/13 時点の ConvoPeq 実装は
  `DeferredDecision { Ready, Discard } + DiscardReason { None, ShutdownDiscard, StaleDiscard }`
  （PublicationAdmission.h:64-77 / PublicationAdmission.cpp:75-90）で固定されている。
- **設計書には DeferredDecision の 4-state 記述は存在しない**（grep 確認）＝現行実装と整合済み。
- **方針確定**: 現行の 2-state + DiscardReason を維持。4-state への回帰は
  API 変更 + 既存 tests 変更 + telemetry semantics 変更を伴うため、実施しない（レビュー推奨と一致）。
- 補足: PublicationAdmission.cpp:80 の「Expired を別 enum 化可能」コメントは TTL 失効（現
  StaleDiscard）の識別子強化の余地を示す。現行は DiscardReason::StaleDiscard で
  TTL / Generation / Sequence を区別せず扱うため、必要なら DiscardReason の細分化で対応可能
  （YAGNI 判断）。

#### H.11.20.5 Generation binding / consume-once の再確認（第十五者 #9 / #10）

- 第十五者は generation binding（shutdown / epoch / reader-registration generation）と
  consume-once（CAS Unconsumed → Consumed）を必須と評価。計画書の T10 / T9 と一致。
- H.11.19.2 の RuntimeIdentity 併結（ABA 対策）とも整合。

#### H.11.20.6 実装順序の支持（第十五者 #26）

- H.11.17.5 の 15-Step を正本として支持。
- 旧 bool reclaim API は migration 開始時に削除（compile error を migration guard にする）
  方針を支持 — 新旧 authority の二重化（legacy + new authority）を回避。

#### H.11.20.7 確定事項（第十五者レビュー反映のまとめ）

- 設計方向の妥当性を確認（🟢 設計 GO）。
- production reclaim 接続は A2-G01〜G23 完了まで NO-GO（未実装: ShutdownQuiescenceProof /
  ReclaimPermit / generation binding / physical destruction audit）。
- 新規反映: Audio Thread 境界の正確な定義（#12）、DeferredDecision 2-state 確定（#20）。
- 全 Blocking（H.9.7: #3/#4/#5/#10/#11/#12）は A2 完了までに解消必須のまま変更なし。

### H.11.21 別視点調査 — 削除キュー3系統の棚卸しと DeletionQueue の UAF リスク確定（2026-08-14）

「別の視点」として、**EBR 削除（reclaim）を担う削除キュー群**をソースで全量棚卸しした。設計書には
`DeletionQueue` / `DeferredDeletionQueue` / `SnapshotRetireManager` の記述が**一切なく**（grep 確認）、
さらに **`DeletionQueue::enqueue` が容量超過時に epoch チェックなしで deleter を即時実行**する
UAF リスクを発見した。

#### H.11.21.1 削除キュー3系統の棚卸し（ソース照合）

| キュー | 所在 | full 時の挙動 | EBR 安全 |
|---|---|---|---|
| `DeferredDeletionQueue` | src/DeferredDeletionQueue.h（ロックフリーリング、EpochDomain.h:559 に内蔵） | `enqueue()` が `bool`（false）を返す → `ISRRetireRouter::enqueueWithRetry` がリトライ後 `RetireQuarantineStore` へ移送 | ✅（UAF 構造的排除） |
| `RetireQuarantineStore` | src/audioengine/RetireQuarantineStore.h（ISRRetireRouter 内蔵） | `quarantine()` が `false` → **deleter は絶対に実行しない**（UAF 構造的排除・三次レビュー契約、RetireQuarantineStore.h:66-68） | ✅ |
| `DeletionQueue` | src/core/DeletionQueue.h（mutex 保護固定配列 `kCapacity=128`、SnapshotRetireManager.h:49 に内蔵） | **容量超過時に追加エントリの deleter を epoch チェックなしで即時実行**（DeletionQueue.cpp:11-18） | ⚠️ **UAF リスク** |

#### H.11.21.2 SnapshotRetireManager / DeletionQueue は現在未使用（dead code）

- `SnapshotRetireManager`（「GlobalSnapshot の RCU 遅延解放を担当する**唯一の retire 経路**」と宣言・
  v13.0 設計ロック準拠、SnapshotRetireManager.h:2-3）は **ソース全体で使用箇所なし**（grep 確認）＝
  現在 dead code。
- 実際の GlobalSnapshot 解放は `SnapshotCoordinator` が `enqueueWithRetry(*m_epochProvider, ...)`
  （ISRRetireRouter 経由）→ 失敗時 `quarantineRetireSink`（RetireQuarantineStore 経由）で実施
  （SnapshotCoordinator.h:90-108, 167-182 / SnapshotCoordinator.cpp:41-58）。
- つまり「唯一の retire 経路」という宣言と実際の実装経路（ISRRetireRouter + RetireQuarantineStore）が
  **乖離**している。

#### H.11.21.3 DeletionQueue の容量超過時 deleter 即時実行の UAF リスク

```cpp
// DeletionQueue.cpp:11-18
void DeletionQueue::enqueue(void* ptr, void (*deleter)(void*), uint64_t epoch, DeletionEntryType type)
{
    std::lock_guard<std::mutex> lock(mutex);
    if (count >= kCapacity)
    {
        // 容量超過: テールのエントリを強制実行してキューを空ける
        // (安全側に倒れる前に deleter を呼び出す)
        if (deleter && ptr)
            deleter(ptr);   // ⚠️ epoch 安全判定なしで即時 destroy
        return;
    }
    ...
}
```

- 容量超過時に **新しいエントリ（ptr）の deleter を epoch 安全判定なしで即時実行**する。
  GlobalSnapshot が RT（Audio Thread）から参照中の場合、即時 destroy で **UAF になる可能性**がある。
- `RetireQuarantineStore` の「store full 時 deleter 実行禁止（UAF 構造的排除）」契約と**矛盾**する挙動。
- 現状 SnapshotRetireManager / DeletionQueue は未使用のため **production 経路では発生しない**。

#### H.11.21.4 反映方針（A2-G23 拡張 + 棚卸し確定）

- **A2-G23（physical destruction audit）の対象拡張**: DeletionQueue の容量超過時 deleter 強制実行は
  EBR 原則（reader が参照を保持している可能性が完全になくなった後にだけ reclaim）に違反するため、
  将来使用時に修正必須。
  - 修正案: `DeletionQueue::enqueue` を `DeferredDeletionQueue` と同様に **full 時は `false` を返し、
    呼び出し元が `RetireQuarantineStore` へ移送**する形へ統一（capacity exhaustion は health escalation
    で先行検知 — SnapshotCoordinator.cpp:52 の jassert 契約と同様）。
- **棚卸し確定**: SnapshotRetireManager / DeletionQueue は「未使用（dead code）」として棚卸し確定。
  将来「唯一の retire 経路」として接続する場合は、上記の UAF リスクを解消してから接続すること。
- 設計書の retire/reclaim 経路の記述（INV-EPOCH-1 / H.11.16 / A2-G23）は、実装経路
  （ISRRetireRouter → DeferredDeletionQueue → RetireQuarantineStore）を正とする。

→ **削除キューは `DeferredDeletionQueue`（EpochDomain 内蔵・full 時 false → 移送）と `RetireQuarantineStore`（full 時 deleter 実行禁止）の 2 系統が正しく、`DeletionQueue`（SnapshotRetireManager 内蔵・容量超過時 deleter 即時実行）は UAF リスク内包・現在未使用として棚卸し確定。A2-G23 の対象に追加。**

### H.11.22 第十六者レビュー — 新規仕様の追加（linearization point / PermitState 拡張 / AC-8 強化 / P0 一覧）（2026-08-14）

第十六者レビュー（20260814-123241 版対象・最新ソース照合）を解析した。総合判定は
「**設計 GO（9/10）・実装条件付き GO・production reclaim 接続 NO-GO**」であり、計画書の
方向性（authority chain / isFullyDrained Observation 化 / Path B admission transaction / 15-Step 正本）を支持する。
既反映の指摘（#1〜#4 / #6〜#7 / #9〜#11 / #13〜#14 / #16〜#17 / #20〜#30 / #32〜#34）に加え、
**新規に明文化すべき 7 項目**（#5 / #8 / #12 / #15 / #18・#19 / #29 / #31）を反映する。

#### H.11.22.1 実装依存関係の正本化（第十六者 #5）

Tier 1 の「**重要度**」（2.2 → 1.4 → 2.1 → 1.7）と「**実装依存関係**」は別軸である。

```text
重要度（Tier 1）:      2.2 → 1.4 → 2.1 → 1.7   （lifetime safety 優先度）
実装依存関係:          LifetimeAccounting → Admission → EpochEvidence → ShutdownQuiescenceProof
```

- Proof の Q0〜Q7 が正しく計算できなければ Proof は作れない。したがって実装順は
  **1.4 semantic accounting → 2.2 proof** の依存関係を持つ。
- 設計書の実装順正本は **H.11.17.5 の 15-Step** のまま（Step 0 = external setter 分類 → Step 1 =
  LifetimeAccounting semantic event 化 → ... → Step 6 = ShutdownQuiescenceProof）であり、
  「Phase A〜E（H.11.15.9）= 依存関係の正本」「16-commit / 17-commit / Phase A-J = レビュー過程の
  順序」として参照する。← 依存関係の観点で整合済み。

#### H.11.22.2 Admission transaction の linearization point 仕様化（第十六者 #8 / #35①）

「shutdown close」と「admission reservation」のどちらが勝つかを一意に決定する **linearization point**
を仕様として明文化する（AdmissionToken 設計への必須仕様）。

```text
closeAdmission() と tryAdmitPublication() の競合:
  Running
    ├─ tryAdmit → token 取得 → obligation（新規 obligation が先に linearize）
    └─ close   → Closed（shutdown が先に linearize）

  ★ check state + reserve を別 atomic 操作にしてはならない
    → tryAdmitPublication は「AdmissionState の CAS(Running → Reserved) を 1 操作で行う」
    → closeAdmission は「AdmissionState の CAS(→ Closing) を 1 操作で行う」
    → CAS の勝敗が linearization point となる（H.11.18 ShutdownAdmissionState 7-state FSM と整合）
```

- これにより「T1 enqueue 開始 / T2 shutdown close / T3 queue.push」の T3 が admission transaction 外に
  漏れない（Path B resurrection race の構造的排除）。
- **A2-G06（Path B authority-side shutdown gate）の実装仕様として必須**。

#### H.11.22.3 PermitState 拡張: Issued → InProgress → Completed（第十六者 #15 / #35②）

現行設計の `PermitState { Valid, Consumed }`（H.11.9）は、consume と physical reclaim completion を
同一視するリスクがある。第十六者は以下の分離を推奨する。

```text
PermitState
    Issued
      ↓ (consume = authority lease 取得)
    InProgress
      ↓ (physical reclaim completed)
    Completed

      ↘ FailedRetryable（reclaim 失敗時 — Permit 再発行の根拠）
```

- **Consumed ≠ ReclaimCompleted**。Permit の consumption は「authority lease の取得」であり、
  physical completion とは別イベント。
- 悪い順序（CAS → Consumed → reclaim 開始 → reclaim 失敗）を許すと
  「Permit は消費済み・reclaim は未完了」の不整合が起きる。InProgress を挟むことで
  失敗時に FailedRetryable → 再発行が可能になる。
- T9（concurrent double reclaim）・T7（double consume）は「Issued → InProgress の CAS」と
  「InProgress → Completed の CAS」の二段階で評価する。

#### H.11.22.4 AC-8 強化: resurrection reject 対象の列挙（第十六者 #31）

AC-8（H.11.17.6）の post-proof resurrection rejection 対象を明示的に列挙する。

```text
post-proof に reject すべき新規 obligation 生成（全 8 経路）:
  Publication            （Path A: evaluate() RejectedShutdown — 実装済み）
  Recovery               （submitRecoveryRequest shutdown gate — 実装済み）
  Build                  （RebuildDispatch — gate 対象）
  Publish                （enqueuePublicationIntent — A2-G06 対象）
  Retire                 （retire authority — gate 対象）
  Quarantine             （quarantineRetire — gate 対象）
  Reader registration    （registerReaderThread() post-close = -1 — 実装済み INV-X3-4 / AC-X3-15）
  Admission reservation  （tryAdmitPublication — A2-G06 / linearization point 対象）
```

- 特に **`post-proof registerReaderThread() == reject`** を明示する。Proof 発行後に新しい
  reader registration が可能なら Q3（ReaderRegistrationClosed）が破られる。
- `registerReaderThread()` / `reserveReaderThread()` は registrationClosed 後 -1 を返す
  （EpochDomain.h:50-51, 88-89）＝ 実装済みであることを確認。

#### H.11.22.5 SupersessionDecision の Decision object 化（第十六者 #12）

1.2 の `canSupersede()` は bool を返す設計だが、coalesce では false の意味が複数あるため、
**Decision object** を推奨する。

```cpp
enum class SupersessionDecision : uint8_t {
    NotSameIdentity,          // 別 recovery（coalesce 対象外・両方保持）
    SameIdentityNotSuperseding, // 同一 identity だが semantic superset ではない（両方保持 or 明示判断）
    Supersedes,               // newer が older を包含 → older を drop
    AlreadySuperseded,        // older が既に別の newer に包含済み
    Invalid,                  // 引数不正（identity 不整合）
};
```

- bool の false を「different recovery」と「same recovery だが superset ではない」に区別し、
  telemetry / state machine に監査可能な情報として残す。
- 実装時の `canSupersede()` はこの Decision object を返す形へ更新（1.2.3 の R1〜R17 と整合）。

#### H.11.22.6 P0-1〜P0-7: blocking リスク一覧（第十六者 #29）

第十六者が特に blocking と判断した 7 項目を一覧化する（A2-G01〜G23 の状態と整合）。

| ID | blocking リスク | 対応 |
|---|---|---|
| P0-1 | external setter 9 箇所（Proof 入力が改ざん可能） | A2-G01 / Phase B0-B2（setter 撤去） |
| P0-2 | Path B resurrection（shutdown gate が authority API にない） | A2-G06 / H.11.22.2 linearization point |
| P0-3 | Proof / Permit 型が未存在 | A2-G15〜G22 / Commit 1-3 |
| P0-4 | Epoch generation 未実装（Permit binding に必要） | A2-G19 / H.11.13.3 EpochQuiescenceEvidence |
| P0-5 | CacheMap delete-before-reclaim（physical lifetime 順序が逆） | A2-G23 / H.11.15 Phase H |
| P0-6 | DeletionQueue full 時直接 deleter（ReclaimAuthority 迂回） | A2-G23 / H.11.21（blocking issue へ昇格 — H.11.22.7） |
| P0-7 | Recovery durable slot（coalesce 不能ケースで obligation 喪失） | 1.2.3 bounded durable recovery table |

#### H.11.22.7 DeletionQueue full 時 deleter の A2-G23 blocking issue 昇格（第十六者 #18 / #19 / #35③）

H.11.21 で棚卸し確定した `DeletionQueue::enqueue` の容量超過時 deleter 即時実行を、
**単なる監査項目ではなく authority singularization を完成させるための blocking issue** に昇格する。

- `DeletionQueue full → deleter → physical destruction` は ReclaimAuthority を迂回する
  physical destruction path になり得る（H.11.21.3 で確認）。
- 現状は SnapshotRetireManager / DeletionQueue が未使用のため production 経路では発生しないが、
  将来「唯一の retire 経路」として接続する場合の修正必須条件:
  `queue full → enqueue failure → bounded fallback → ReclaimAuthority → safe destruction`
- fallback（RetireQuarantineStore）も無限成長できないため、`primary bounded queue + bounded
  quarantine/fallback + shutdown completion authority` まで含めて LifetimeAccounting する
  （C1 pendingReclaimIdentity == empty を Completion 側に置く設計と整合）。

#### H.11.22.8 確定事項（第十六者レビュー反映のまとめ）

- 設計品質: **9/10（GO）**。authority / proof / evidence / admission / completion / identity /
  generation / physical destruction の責務分離が明確。
- 実装開始: **条件付き GO**（LifetimeAccounting → AdmissionState → EpochEvidence の順に先に実装し、
  `reclaim(..., bool)` は production migration 開始時に削除）。
- production reclaim 接続: **明確に NO-GO**（A2-G01〜G23 PASS まで）。
- 新規反映: 実装依存関係の正本化（#5）/ Admission linearization point 仕様化（#8・#35①）/
  PermitState 拡張 Issued→InProgress→Completed（#15・#35②）/ AC-8 対象列挙（#31）/
  SupersessionDecision Decision object 化（#12）/ P0-1〜P0-7 一覧（#29）/
  DeletionQueue blocking issue 昇格（#18・#19・#35③）。
- 最終契約の確認: `isFullyDrained()` は観測するだけ / `shutdownPhase` は状態を表すだけ /
  `readerRegistrationClosed()` は事実を観測するだけ。reclaim の認可は ShutdownRuntime が生成した
  sealed ShutdownQuiescenceProof からのみ取得できる ReclaimPermit によって行い、
  physical destruction は ReclaimAuthority のみが実行する。

### H.11.23 別視点調査 — retire 実装コア（LifetimeState / EpochControl / RetireOverflowRing）の棚卸し確定（2026-08-14）

「別の視点」として、これまで設計書に未記載だった **retire 実装コア**（`ISRRetire.h` / `LifetimeState` /
`EpochControl` / `RetireOverflowRing` / `RetireLifecycleState` / `RetireLane`）と、**SnapshotFactory /
GlobalSnapshot** のライフサイクルをソースで全量棚卸しした。設計書 line 3633 の「ISRRetireRouter 配下
（設計のみ、grep 0 件のため新規）」という記述は **RetireRecord 構造体に限定しては正しいが、
「ISRRetireRouter 配下が grep 0 件」と読めるため誤解を招く** — 実際には retire 実装コアは実装済みである。

#### H.11.23.1 retire 実装コアの棚卸し（ソース照合）

| コンポーネント | 所在 | 役割 | 設計書への記載 |
|---|---|---|---|
| `RetireIntent` | ISRRetire.h:32 | `dspSlot / generation(64bit) / retireEpoch / priority` — retire 要求 | ❌ 未記載 |
| `LifetimeState` | ISRRetire.h:52 | retire intent の emit / dequeue / escalate / OverflowRing 連携。**RuntimeWorldAuthority::lifetime_ が所有**（RuntimeWorldAuthority.h:286） | ❌ 未記載 |
| `EpochControl` | ISRRetireRuntimeEx.h:39 | epoch / retire ライフサイクル管理（grace period / reclaim 可否判定） | ❌ 未記載 |
| `RetireLifecycleState` | ISRRetireRuntimeEx.h:30 | 6-state FSM: `Visible → CompareEligible → TelemetryRetained → ReplayRetainedOptional → ReclaimEligible → Reclaimed` | ❌ 未記載 |
| `RetireOverflowRing` | ISRRetireOverflowRing.h:65 | retire enqueue overflow 時の bounded 退避リング（`overflowTimestampUs / reinjectRetryCount`） | ❌ 未記載 |
| `RetireLane` | ISRRetireLane.h:6 | retire lane 列挙 | ❌ 未記載 |

- `LifetimeState` は `RuntimeWorldAuthority::lifetime()`（HANDLER-1 boundary、RuntimeWorldAuthority.h:159-161）
  経由でのみ AudioEngine / ISR Handler から到達する — **設計書の RuntimeWorldAuthority 記述に未反映**。

#### H.11.23.2 設計書 line 3633 の記述補正

- 設計書 line 3633「`src/audioengine/ISRRetireRouter 配下（設計のみ、grep 0 件のため新規）`」は
  **RetireRecord 構造体**についての記述であり、RetireRecord 自体はソースに存在しないため正しい。
- ただし「ISRRetireRouter 配下が grep 0 件」と読めるため、**retire 実装コア（LifetimeState / EpochControl /
  RetireOverflowRing / RetireLifecycleState）は実装済み**であることを H.11.23.1 で確定する。
- RetireRecord（`ReclaimIdentity + retireEpoch + retireSequence + RetireKind`）は引き続き**設計のみ**。

#### H.11.23.3 EpochControl と EpochDomain の責務分担（重要）

設計書の `EpochQuiescenceEvidence`（H.11.13.3）は `m_epochDomain` ベースで定義されているが、
**実際の retire の grace period 判定は `EpochControl`（ISRRetireRuntimeEx）が担当**している。

```cpp
// EpochControl::isGracePeriodCompleted（ISRRetireRuntimeEx.h:61-65、inline）
static bool isGracePeriodCompleted(worldGeneration, maxObservedGeneration, audioCallbackActiveCount) noexcept {
    return (maxObservedGeneration > worldGeneration) || (audioCallbackActiveCount == 0u);
}
```

- **EpochDomain**（src/core/EpochDomain.h）: reader registration / minReaderEpoch / grace period の
  **reader 側**を担当（`readerRegistrationClosed()` 等）。
- **EpochControl**（src/audioengine/ISRRetireRuntimeEx.h）: retire intent / RetireLifecycleState /
  grace period 判定の **retire 側**を担当（`RuntimeWorldAuthority::lifetime()` 経由）。
- この2つの責務分担は設計書に未記載。**Q5（EpochSettled）の実装根拠は EpochDomain（reader 側）と
  EpochControl（retire 側）の両方を考慮する必要がある**ことを確定。

#### H.11.23.4 commit 経路の grace period 判定（AudioEngine.Commit.cpp:565-615）

```text
for (const auto& pending : pendingIntents) {
    pendingGeneration = pending.generation;
    maxObservedGeneration = youngestObservedGeneration_;
    callbackActiveCount = rtLocalState_.audioCallbackActiveCount;
    graceCompleted = worldAuthority_.lifetime().isGracePeriodCompleted(
                         pendingGeneration, maxObservedGeneration, callbackActiveCount);
    ...
    if (worldAuthority_.lifetime().canReclaimAfterEscalation(noReader, noExecutorReference, noPendingTransition))
    ...
    else if (worldAuthority_.lifetime().canTransitionRetirePendingToFree(
                 graceCompleted, pendingIntentOwned, authoritativeOwnershipReleased))
}
```

- **commit 経路で grace period 判定 → escalation → retire pending → free 遷移**が行われる。
- `canTransitionRetirePendingToFree = graceCompleted && pendingIntentOwned && authoritativeOwnershipReleased`、
  `canReclaimAfterEscalation = noReader && noExecutorReference && noPendingTransition`。
- これは設計書の **INV-EPOCH-1（retireEpoch vs minReaderEpoch）の実装根拠**であり、設計書の
  retire/reclaim 経路の記述（H.11.16 / A2-G23）と照合して整合することを確認。

#### H.11.23.5 SnapshotFactory / GlobalSnapshot のライフサイクル（設計書に未記載）

- `SnapshotFactory`（src/core/SnapshotFactory.h:15）: GlobalSnapshot の生成・破棄の**唯一の物理層**。
  - `create(params)` / `createImpl(pending, current, generation)` / `destroy(snap)`
  - `computeContentHash(params)` / `areSnapshotsEquivalent(params, snapshot)`
- `areSnapshotsEquivalent` は **UAF 回避: 観測用ポインタではなく不変ID（`convStateId`）で比較**
  （SnapshotFactory.cpp:44-49）— 設計書に未記載。
- `GlobalSnapshot`（src/core/GlobalSnapshot.h:17）: コピー・ムーブ禁止の immutable 構造体。
- **解放経路**: SnapshotCoordinator → `enqueueWithRetry`（ISRRetireRouter）→ 失敗時
  `quarantineRetireSink`（RetireQuarantineStore）。`SnapshotFactory::destroy` は deleter として
  実行される（H.11.21 と整合）。

#### H.11.23.6 反映方針・確定事項

- **棚卸し確定**: retire 実装コア（LifetimeState / EpochControl / RetireOverflowRing /
  RetireLifecycleState / RetireLane）は実装済み・設計書に未記載 → 本 H.11.23 で棚卸し確定。
- **A2-G13（epoch settled）の実装根拠拡張**: EpochSettled 判定は EpochDomain（reader 側）と
  EpochControl（retire 側）の両方を考慮する。`EpochQuiescenceEvidence`（H.11.13.3）は
  `EpochControl::isGracePeriodCompleted` と照合して確定する。
- **RetireLifecycleState 6-state FSM** は ReclaimIdentity（Registered → Started → Completed）と
  補完関係にあり、A2-G23（physical destruction audit）の対象として確定。
- 設計書 line 3633 の「ISRRetireRouter 配下（設計のみ、grep 0 件）」は **RetireRecord に限定**した
  記述として解釈し、retire 実装コアが実装済みであることを本 H.11.23 で補正。

### H.11.24 別視点調査 — RuntimeHealthMonitor の棚卸しと Admission Circuit Breaker の確定（2026-08-14）

「別の視点」として、設計書にほぼ未記載だった **RuntimeHealthMonitor**（ISRHealthState / Critical
判定 / CriticalExitBlocker / CriticalExitCondition / Admission Circuit Breaker）をソースで棚卸しした。
さらに **H.11.19.1 の evaluate() 記述を実際の実装に補正**した（Critical のみ → Critical / Degraded /
Pressure throttle の 3 段階）。

#### H.11.24.1 RuntimeHealthMonitor の棚卸し（ソース照合）

`RuntimeHealthMonitor`（src/audioengine/RuntimeHealthMonitor.h/cpp）は設計書に **ほぼ未記載**（grep で
`ISRHealthState::Critical` は H.11.19.1 のみ）。以下を確定する。

```text
ISRHealthState { Healthy=0, Degraded, Critical }（RuntimeHealthMonitor.h:38-41）

Critical 判定（RuntimeHealthMonitor.cpp:370-388 updateHealthState）:
  - Retire stall       → Degraded or Critical（:374-376）
  - Publication stall  → Critical（retire より優先度高、:380-382）
  - Overflow rate      → Degraded or Critical（:387-388、maxOr > 5.0 → Critical :174）

CriticalExitBlocker（Critical 出口ブロック理由、RuntimeHealthMonitor.h:89-97）:
  MonitorNotNormal / SuppressionActive / RecoveryRunning / StableDurationInsufficient /
  ActiveReaderRemaining / PendingRetireExceeded / RetireAgeExceeded（cpp:255-259）

CriticalExitCondition（:101-115）: allMonitorsNormal + suppressionInactive +
  noRecoveryActionRunning + stableDuration（Critical 出口 安定 60 秒継続）
```

- **kRetireAgeCriticalUs = 30 秒** / **kReaderSlotCriticalThreshold = 0.75** /
  **kOverflowRateCriticalThreshold = 5（5 回/秒超 → Critical）** /
  **kOverflowHysteresisCriticalToDegradedUs = 10 秒**（Critical → Degraded 復帰）。

#### H.11.24.2 Admission Circuit Breaker の実装詳細（H.11.19.1 補正）

`PublicationAdmission::evaluate()` の HealthState check は **Critical だけでなく Degraded でも
RejectedPressure** を返す（PublicationAdmission.cpp:24-35）。さらに **Pressure / throttle check
（P1-6: Adaptive Backpressure）** も存在する（:36-47）。

```cpp
// 4. HealthState check（PublicationAdmission.cpp:24-35）
if (m_healthStateRef) {
    auto health = convo::consumeAtomic(*m_healthStateRef, std::memory_order_acquire);
    if (health == ISRHealthState::Critical) return Decision::RejectedPressure; // 全 publish 拒否
    if (health == ISRHealthState::Degraded) return Decision::RejectedPressure; // 低優先度拒否
}
// 5. Pressure / throttle check（:36-47、retirePressurePublicationThrottleActive_ → RejectedPressure）
```

- **m_healthStateRef** は `RuntimeHealthMonitor::getHealthStateRef()` を PublicationAdmission が
  参照（PublicationAdmission.h:48-50, 109）— **Admission が HealthState を観測する唯一の経路**。
- Degraded は「低優先度 publish を拒否」の意図だが、実装は一律 RejectedPressure を返し
  Coordinator 側の間引き制御に委ねる（コメント記載）。
- 設計書 H.11.19.1 の「Critical で全 publish 拒否」は **Degraded と Pressure throttle が未記載**のため
  補正済み（H.11.19.1 のコードスニペット修正）。

#### H.11.24.3 isFullyDrained の外部 setter（A2-G01）の確認

- `AudioEngine::isFullyDrained`（Threading.cpp:114-160）は現在も **3 つの外部 setter** を呼ぶ
  （`setFallbackBacklogCount` / `setRetireBacklogCount` / `setDeferredRetireResidencyCount`
  Threading.cpp:126-128）。これは §1.4 / A2-G01（external setter = 0）の対象（設計書 line 1895 で
  指摘済み）。
- 撤去方針（§1.4 第三者的レビュー反映、設計書 line 445-452）: setter は「private 化」ではなく
  **「意味ごと廃止」** し、semantic event API（`onRetireAccepted()` / `onRetireConsumed()` 等）へ
  一本化。isFullyDrained は実測値（queue size / DSPQuarantineManager::residentCount）を直接判定する。
- **swapPending_ は 15 return-body 条件とは別の pre-check**（ISRRuntimePublicationCoordinator.cpp:486）
  — setter 撤去後も維持必須（torn read 防止、設計書 line 1925 と整合）。

#### H.11.24.4 反映方針・確定事項

- **棚卸し確定**: RuntimeHealthMonitor は実装済み・設計書にほぼ未記載 → 本 H.11.24 で棚卸し確定。
- **H.11.19.1 補正**: evaluate() の HealthState check は Critical / Degraded / Pressure throttle の
  3 段階（Admission Circuit Breaker）。
- **A2-G01（external setter = 0）**: AudioEngine::isFullyDrained の 3 setters は撤去方針済み
  （semantic event API へ一本化、swapPending_ pre-check は維持）。
- RuntimeHealthMonitor の CriticalExitBlocker / CriticalExitCondition は「Critical 出口の監視」であり、
  shutdown proof（Q0〜Q7）とは独立の仕組みとして設計書に確定。

### H.11.25 第十七者レビュー — BuildError / Recovery coalesce / currentWorld_ の詳細設計確定（2026-08-14）

第十七者レビュー（20260814-125313 版対象・最新ソース照合）を解析した。前半（#1〜#36）は既反映の
確認（設計 GO 9/10 / production reclaim NO-GO）であり、後半は **BuildError / Recovery coalesce /
currentWorld_ の 3 項目** を「NO-GO 解消」ではなく Authority Singularization / ISR / Semantic
Single Source に整合する形で詳細設計した。**新規反映は G-F1〜G-F5 Gate 強化（#31）と 3 項目の
詳細設計（BE / RC / CW 各 Acceptance Criteria）**。

#### H.11.25.1 ソース検証（第十七者主張の確認）

| 項目 | ソース検証結果 |
|---|---|
| BuildError enum | `None / InvalidInput / ResourceUnavailable / MKLFailure / ConvolverFailure / PrepareFailure / WarmupFailed / InternalError`（RuntimeBuilder.h:107-113）。**build() から実際に返るのは InvalidInput / ResourceUnavailable / InternalError / WarmupFailed / None の 5 値のみ**（RuntimeBuilder.cpp:435,461,466,476,478）— **MKLFailure / ConvolverFailure / PrepareFailure は enum のみで build path では生成されない** |
| prepare() | `void prepare(...)`（AudioEngine.h:682,722,747,866）— **failure status を失う**。`PrepareStatus` 型は存在しない |
| currentWorld_ | ISRRuntimePublicationCoordinator.cpp で **10 箇所 read/write**（:89,112,126,129,170,176,183,191,588,740）。`INV-ISR-06`（ownership source でない）/ `INV-ISR-07`（RuntimeStore::current と identity 整合） |
| pendingRecoveryAdmission_ | `PendingRecoveryAdmission`（ISRRuntimePublicationCoordinator.h:575,590）— **single slot 構造体**（SPSC、trivially copyable）。`State::DurablePending`（cpp:772） |

#### H.11.25.2 BuildError 詳細設計（第十七者 — 現案を修正して GO）

現行 `BuildError` は「enum を増やすだけでは不十分」で、**途中の return-status 消失**（`init() → bool`、
`prepareToPlay() → void`、`DSPCore::prepare() → void`、`RuntimeBuilder::build()`）が根本原因。以下を確定する。

```text
DSPCore::prepare()
    ↓ PrepareStatus（新規）
RuntimeBuilder
    ↓ BuildError
FailureClassification
    ↓
RetryPolicy（+ context + admission state + generation + retry count）
    ↓
RetryDecision
```

- **PrepareStatus 導入**: `enum class PrepareStatus { Success, InvalidConfiguration, ResourceExhausted,
  MKLFailure, ConvolverInitFailed, InternalFailure }`。`void prepare()` → `PrepareStatus prepare()`。
  失敗原因を下位層から伝播させる。
- **BuildError / WarmupError 分離**: construction failure（BuildError）と post-construction validation
  failure（`WarmupError { None, IRNotFinalized, ConvolverNotReady, ... }`）を別 domain にする。
- **BuildErrorDescriptor を Semantic Single Source**: `{ name, failureClass, disposition, telemetryCode }`
  の descriptor table を唯一の source にする（toString / FailureClassification / RetryDisposition /
  telemetry が別分岐しない）。
- **RetryDisposition は BuildError の属性と完全には同一にしない**: `WarmupFailed` は context
  （Normal rebuild / Recovery rebuild / Shutdown）で retry policy が異なる。`RetryDisposition::ContextDependent`
  を残す。
- **RetryPolicy は RebuildDispatch に散らさない**: `BuildRetryDecision decideBuildRetry(outcome, context,
  retryState)` に集約（BuildFailureAuthority / RetryPolicy / RuntimePublicationAdmission / ReclaimAuthority
  を分離）。
- **exponential backoff**: failure #1→10ms / #2→20ms / #3→40ms / #4→80ms / #5→next coordinator cycle。
  RT では待たせない（DurablePending + nextEligibleTime、lease → schedule → yield → wake → lease）。
- **BE-1〜BE-7**: BE-1 prepare() が失敗原因を失わない / BE-2 MKLFailure・ConvolverFailure・PrepareFailure
  が実際に生成可能 / BE-3 WarmupError と BuildError が別 domain / BE-4 caller が `runtime == nullptr`
  だけで成功判定しない（`runtime == nullptr || outcome.error != None`）/ BE-5 retry policy が caller
  独自推測にならない / BE-6 Audio Thread は BuildError classification・retry decision を実行しない /
  BE-7 descriptor table が semantic single source。
- **1.8 の判定更新**: 現案（enum 増加のみ）は NO-GO のまま。上記の PrepareStatus / WarmupError /
  BuildErrorDescriptor / RetryPolicy 中央集約を実装すれば **GO**。

#### H.11.25.3 Recovery coalesce 詳細設計（第十七者 — 条件付き GO）

1.2.3 の LogicalRecoveryIdentity / RecoveryProvenance / SupersessionDecision をさらに確定する。

- **LogicalRecoveryIdentity**: `{ handle, generation, buildIdentity, epoch }`。**intentId は含めない**
  （intentId は identity ではなく sequence/diagnostic）。
- **RecoveryProvenance**: `{ Transport, Durable, Retry, Quarantine }`。queue full で Transport → Durable
  に変わっても **identity は変わらない**。
- **SupersessionDecision**: `{ CanSupersede, DifferentHandle, DifferentSemanticTarget,
  DifferentGenerationDomain, NotSemanticSuperset }` — bool にしない（false の理由が消えない）。
- **BoundedDurableRecoveryTable**: `pendingRecoveryAdmission_`（single slot）を
  `BoundedRecoveryAdmissionTable`（例: `kMaxDurableRecoveryAdmissions = 16`）へ拡張。
  coalesce できない Recovery を「できないから捨てる」ことは禁止。
- **state transition**: `Empty → DurablePending → Building → [Success → Empty] | [TransientFailure →
  DurablePending] | [Obsolete → Empty]`。
- **最上位 Invariant**: **1 LogicalRecoveryIdentity = 0 or 1 live obligation**。queue + durable table に
  同一 logical recovery が存在しても reservation = 1（transport / durable の二重計上なし、
  INV-X1-6 と整合）。
- **coalesce は Audio Thread で行わない**（CoordinatorLoop 側 NonRT）。
- **RC-1〜RC-10**: RC-1 同一 identity は live obligation 最大 1 / RC-2 異なる semantic target は
  coalesce しない / RC-3 generation domain 違いは coalesce しない / RC-4 non-superset は coalesce しない /
  RC-5 coalesce で reservation 数が増えない / RC-6 queue full でも Recovery が消えない / RC-7 Building 中の
  Recovery を別 Recovery が誤って上書きしない / RC-8 transient failure で reservation を再発行しない /
  RC-9 shutdown 開始後の Recovery は AdmissionClosed として処理 / RC-10 BuilderStopped が shutdown proof
  に反映。

#### H.11.25.4 currentWorld_ 4 段階 migration（第十七者 — 最後に実施・条件付き GO）

「currentWorld_ を削除して runtimeStore.observe() に置換」は**単純置換では危険**。read-source
singularization は ISR commit 意味論の変更を伴う。4 段階 migration を確定する。

```text
Phase CW-0  dual-source 監査（全 read site 分類: world/epoch/sequence/generation/ownership/recovery）
Phase CW-1  read API を RuntimeWorldAuthority へ集約（observePublishedWorld / observePublicationIdentity）
            — caller が currentWorld_ を直接読むことを禁止（内部ではまだ使用可）
Phase CW-2  observePublishedWorld() の内部実装を runtimeStore_.observe() のみに
Phase CW-3  currentWorld_ の write を削除（commit metadata → RuntimeStore publishAndSwap のみ）
Phase CW-4  currentWorld_ フィールド自体を削除
```

- **PublishedWorldObservation**: `{ const RuntimeState* world; PublicationIdentity identity; }` —
  world と sequence/epoch/generation が**同一 publication identity** であることを read contract にする。
- **CW-1〜CW-7**: CW-1 currentWorld_ direct read = 0、CW-2 direct write = 0、CW-3 全 read が
  RuntimeWorldAuthority → RuntimeStore 経由、CW-4 RuntimeState::publication が physical store swap 前に
  bake 済み、CW-5 `RuntimeStore::current.identity == RuntimeState::publication.identity` を全 publish
  transaction で保証 / CW-6 RT から ownership/lifetime を currentWorld_ から導出する経路 = 0 /
  CW-7 Test 9 / Test 10 / INV-X4-A/B/C PASS。
- **1.7 の判定更新**: 単純削除は NO-GO。上記 CW-0〜CW-4 の段階的 migration を実装すれば GO。
  **lifetime safety（Tier 1）完了後に実施**。

#### H.11.25.5 G-F1〜G-F5 Gate 強化（第十七者 #31）

AC-8〜AC-15 / G-A1〜G-E5 に加えて、実装監査用に G-F 系 Gate を集約する。

```text
G-F1  Proof issued → any new AdmissionToken impossible
G-F2  Proof issued → reader registration impossible
G-F3  Permit consumed → same Permit cannot be replayed
G-F4  ReclaimStarted(A) → physical destruction(A) → ReclaimCompleted(A) の順序を監査可能
G-F5  CompletionProof → pending ReclaimIdentity == 0（identity-level で証明）
```

- G-F2 は registerReaderThread() post-close = -1（EpochDomain.h:50-51, 88-89 実装済み）と整合。
- G-F4 は H.11.15 Phase H（ReclaimPermit validation → ReclaimStarted → destruction → ReclaimCompleted）と整合。

#### H.11.25.6 実装順序（第十七者推奨）

```text
Phase B1-B5: BuildError / PrepareStatus → WarmupError → BuildErrorDescriptor → RetryPolicy → tests
Phase R1-R6: LogicalRecoveryIdentity → RecoveryProvenance → SupersessionDecision → BoundedDurableTable
             → Lease + coalesce → Recovery stress tests
Phase CW1-CW4: Observation API → all read migration → RuntimeStore-only verification → currentWorld_ delete
```

- **3 項目を同一 commit 系列にしない**（BuildError = construction semantics / coalesce = admission
  semantics / currentWorld_ = publication semantics — 障害時の因果追跡を維持）。
- 最も先に着手: **BuildError**。最も慎重に最後まで隔離: **currentWorld_**。
- 旧 bool reclaim API の compile-fail migration branch（15-Step 正本）は変わらず。

#### H.11.25.7 確定事項（第十七者レビュー反映のまとめ）

- 設計 GO 9/10 / ISR GO / 実装着手 条件付き GO / production reclaim は A2-G01〜G23 全 PASS まで NO-GO。
- **DeletionQueue は production 経路から切り離して dead code として棚卸し確定済み**（production P0 ではない）。
  一方 **CacheMap / ReleaseResources の delete/reclaim 順序と旧 bool API は現行 production コードに
  実在するため P0 のまま**。
- 新規反映: G-F1〜G-F5 Gate 強化（#31）/ BuildError 詳細設計（PrepareStatus / WarmupError /
  BuildErrorDescriptor / RetryPolicy / BE-1-7）/ Recovery coalesce 詳細設計（BoundedDurableTable /
  RC-1-10）/ currentWorld_ 4 段階 migration（CW-0-4 / CW-1-7）。
- 最終 contract: `caller / bool / readerRegistrationClosed() / isFullyDrained() / shutdownPhase /
  counter == 0` は**いずれも reclaim authority ではない**。reclaim 認可は ShutdownRuntime → sealed
  ShutdownQuiescenceProof → ReclaimPermit → ReclaimAuthority → physical destruction →
  ShutdownCompletionAuthority → ShutdownCompletionProof の一方向 chain のみ。

### H.11.26 別視点調査 — ISR 関連コンポーネント網羅性の確認（2026-08-14）

「別の視点」として、これまで未調査だった **ISRCoordinatorLoop / ISRRuntimePublicationCoordinator_ProcessIntent
（dispatch テーブル）/ RuntimePublicationOrchestrator / DSPLifetimeManager / CrossfadeRuntime** をソースで
棚卸しし、設計書との整合を確認した。**結論: これらの全領域は設計書に既に反映済みであり、新規修正はない**
（設計書 line 2877 の「第5パス」と同様の網羅性確認。H.11.11〜H.11.25 で ISR 関連コンポーネントを全量
カバーしたことを確定）。

#### H.11.26.1 調査範囲と確認結果（ソース照合）

| コンポーネント | ソース | 設計書での反映箇所 | 整合 |
|---|---|---|---|
| `CoordinatorLoop` | ISRCoordinatorLoop.h:18（juce::Thread、processIntent + overflow drain の cadence） | line 1146 / 1686 / 2120-2122（NonRT 単一 consumer） | ✅ |
| `processIntent` dispatch | ISRRuntimePublicationCoordinator_ProcessIntent.cpp:10,40,66（kDispatchTable） | line 2884（processIntent 統合、Appendix D と整合） | ✅ |
| `QuarantineIntentHandler` | ProcessIntent.cpp:110-141（submitRecoveryIntent 直接呼び、primary 経路） | line 64 / 1797-1802 / 2179 | ✅ |
| `RecoveryIntentHandler` | ProcessIntent.cpp:163-169（dead code — intentQueue_ に誰も Recovery を push しない） | line 64 / 1813（dead code 確定） | ✅ |
| `RuntimePublicationOrchestrator` / `DeferredPublishView` | RuntimePublicationOrchestrator.h:52-158（peek/evaluate/consume/discard、Single Thread Owner） | line 2014 / 2491 / 2885（Path A gate、buildRuntimePublishWorld callers） | ✅ |
| `DSPLifetimeManager` | DSPLifetimeManager.h:14-19（retire / retireByHandle / retireDeferred / destroyRolledBackDSP） | H.11.12（destroyRolledBackDSP 4 パス棚卸し） | ✅ |
| `CrossfadeRuntime` | CrossfadeRuntime.h | line 350（ConvolverProcessor の LinearRamp とは別個） | ✅ |

#### H.11.26.2 今回の調査で確認された詳細事項

- **CoordinatorLoop の cadence**: `juce::Thread::wait()` は blocking sleep（spin ではなく idle polling
  ~0% CPU）— 設計書の NonRT 単一 consumer 記述と整合。
- **processIntent の reservation 減算**: 減算は processIntent の while ループで一元管理
  （handler では行わない — HANDLER-1 boundary、ProcessIntent.cpp:52）。設計書の
  `publicationIntentResidencyCount_` / `quarantineIntentResidencyCount_` の整合と確認。
- **DSPLifetimeManager::retireByHandle**: `dspHandleRuntime_.retire(handle)` + `requestReclaimHandle(handle)`
  （DSPLifetimeManager.cpp:84-90）— **INV-EPOCH-1（retireEpoch vs minReaderEpoch）の実装根拠**と整合。
- **RecoveryIntentHandler dead code**: intentQueue_ に誰も Recovery Intent を push しない（ProcessIntent.cpp:131
  注記）— primary 経路は QuarantineIntentHandler → submitRecoveryIntent（設計書 line 64 と整合）。

#### H.11.26.3 確定事項

- **ISR 関連コンポーネントの網羅性が確定**: ISRShutdown / ISRRuntimePublicationCoordinator / PublicationAdmission /
  DSPLifetimeManager / ISRRetireRouter / EpochDomain / DeletionQueue / DeferredDeletionQueue / RetireQuarantineStore /
  SnapshotCoordinator / SnapshotFactory / RuntimeWorldAuthority / RuntimeStore / ISRRetire / ISRRetireRuntimeEx /
  RetireOverflowRing / RuntimeHealthMonitor / RuntimeBuilder / CoordinatorLoop / ProcessIntent / Orchestrator /
  CrossfadeRuntime — **設計書 H.11.11〜H.11.26 で全量カバー済み**。
- 未確定マーカー: 実質なし（line 1040 BuildError「検討中」= 未実装の将来タスクとして正当 / line 2871
  確定宣言）。文書内に不明確な技術主張は残っていない。

### H.11.27 第十八者レビュー — BE-8 / RC-11 / CW-8 の追加確定（2026-08-14）

第十八者レビュー（20260814-131055 版対象・最新ソース照合）を解析した。総合判定は
「**設計 GO 9/10 / ISR GO / 実装着手 条件付き GO / production reclaim NO-GO**」で、既反映の内容
（BuildError / Recovery coalesce / currentWorld_ の詳細設計、G-F1〜G-F5、ISR-LIFE-01〜10）を確認した。
**新規反映は #37 の 3 項目（BE-8 / RC-11 / CW-8）と #6 / #7 の 2 項目（toString descriptor table 化 /
RetryBackoffPolicy tuning parameter 化）**。

#### H.11.27.1 ソース検証（第十八者新規指摘の確認）

| 項目 | ソース検証結果 |
|---|---|
| BE-8（Audio Thread → BuildError classification / RetryDecision = unreachable） | BuildError / retry 判定は `rebuildThreadLoop` 内（RebuildDispatch.cpp:941-1162）で実行 — **NonRT のみ**。RT 側は `debugRebuildDispatchRequestCount` 等のカウンタのみ（:507,723,744）で、classification / retry decision は到達不能 ✅ |
| RC-11（Building 中 supersession） | 現行 `pendingRecoveryAdmission_`（single slot）は coalesce 時に**既存 durable を最新で上書き**（ISRRuntimePublicationCoordinator.cpp:771）— 現行は Building 中でも上書き。RC-11 の「pending supersession」は**新設計** |
| CW-8（PublishedWorldObservation atomic contract） | `consumeWorldHandle` / `observePublishedWorld` は RuntimeWorldAuthority.h:189-199 に実在。`PublishedWorldObservation` 構造体は**現行ソースに存在しない**（設計のみ） |
| toString descriptor table 化 | 現行 `toString`（RuntimeBuilder.cpp:53-76）は switch-case で全 8 値対応。`static_assert` は 0 件（設計書 line 878） |

#### H.11.27.2 BE-8: Audio Thread から BuildError classification / RetryDecision 到達不能（第十八者 #37-A）

H.11.25.2 の BE-1〜BE-7 に次を追加する。

```text
BE-8: BuildError / WarmupError / RetryDecision の生成は Audio Thread から到達不能
  Audio Thread × BuildError classification
             × RetryDecision
             × backoff calculation
  Audio Thread = intent + bounded accounting のみ（ISR-LIFE-05〜08 と整合）
```

- ソース照合: BuildError / retry 判定は rebuildThreadLoop（RebuildDispatch.cpp:941-1162、NonRT）のみ。
  RT 側（AudioEngine.Retire.cpp / Processing.Snapshot.cpp 等）からは BuildError / RetryDecision を
  生成しないことを acceptance 対象とする。

#### H.11.27.3 RC-11: Building 中の supersession は pending supersession として扱う（第十八者 #37-B）

H.11.25.3 の RC-1〜RC-10 に次を追加する。

```text
RC-11: Building 中の同一 LogicalRecoveryIdentity への新規 intent は既存 lease を上書きせず、
      pending supersession として扱う。

  A = Building（既に Builder へ渡済み）
  B = same LogicalRecoveryIdentity
    → B は既存 lease を変更せず、pending slot として記録
    → A の Build 完了後に supersession 判定（B が A を supersede する場合のみ適用）
```

- **現行との差異**: 現行 `pendingRecoveryAdmission_` は single slot で coalesce 時に「既存 durable を
  最新で上書き」（ISRRuntimePublicationCoordinator.cpp:771）する。RC-11 は **BoundedRecoveryAdmissionTable
  導入後、Building 中の lease を守る**設計（Building 中は既存 lease を変更しない）。
- 理由: A が既に Builder へ渡っているため、B で A を置き換えると build 結果と obligation の対応が壊れる。

#### H.11.27.4 CW-8: PublishedWorldObservation の atomic snapshot contract（第十八者 #37-C）

H.11.25.4 の CW-1〜CW-7 に次を追加する。

```text
CW-8: A single PublishedWorldObservation must never contain
      world from publication N and identity from publication N+1.

  観測自体が atomic snapshot contract になる:
  PublishedWorldObservation { world, identity } の
  world pointer と PublicationIdentity（generation / epoch / sequence）は
  同一 publication transaction 由来であることを read-contract として保証する。
```

- `consumeWorldHandle` / `observePublishedWorld`（RuntimeWorldAuthority.h:189-199）は RuntimeStore::observe()
  経由で world を返すが、**identity（generation / epoch / sequence）と world が同一 publication 由来**である
  ことを CW-5（`RuntimeStore::current.identity == RuntimeState::publication.identity`）に加えて保証する。
- 実装: `observePublishedWorld()` が `{ world, identity }` のペアを**同一 acquire load** で返す
  （world と identity を別 source から読まない）。

#### H.11.27.5 toString を descriptor table から生成（第十八者 #6）

現行 `toString()`（RuntimeBuilder.cpp:53-76）は独立した switch-case であり、enum 追加時の
網羅性 static_assert だけでは「toString が全 enum を処理している」ことを証明しない。

```text
constexpr auto& descriptor(BuildError e) noexcept;   // 唯一の入口
toString(e)        → descriptor(e).name
failureClass(e)    → descriptor(e).failureClass
retryDisposition(e)→ descriptor(e).disposition
telemetryCode(e)   → descriptor(e).telemetryCode
```

- descriptor table（§1.8.10.3 / line 1059-1066）を真の Semantic Single Source にする。
- `toString()` も descriptor table を直接利用する（独立 switch を廃止）。
- これにより「enum → descriptor table（name / failureClass / disposition / telemetryCode）」の
  単一経路が完成する。

#### H.11.27.6 RetryBackoffPolicy を tuning parameter に（第十八者 #7）

H.11.25.2 の exponential backoff（10/20/40/80ms）を **architecture invariant にしない**。

```text
RetryPolicy の固定必須事項:
  - non-blocking
  - RT で実行しない
  - spin しない
  - durable obligation を保持する

delay 値は tuning parameter:
struct RetryBackoffPolicy {
    uint32_t initialDelayMs;
    uint32_t maxDelayMs;
    uint32_t multiplier;
};
```

- ISR 観点では「時間値より **RT を待たせない**こと」が本質。10/20/40/80ms は実装初期値であり、
  tuning parameter（RetryBackoffPolicy）として構成可能にする。
- RT では待たせない（DurablePending + nextEligibleTime、lease → schedule → yield → wake → lease）。

#### H.11.27.7 確定事項（第十八者レビュー反映のまとめ）

- 設計 GO 9/10 / ISR GO / 実装着手 条件付き GO / production reclaim は A2-G01〜G23 全 PASS まで NO-GO。
- 新規反映: BE-8（Audio Thread から BuildError classification / RetryDecision 到達不能）/ RC-11（Building 中
  supersession は pending supersession）/ CW-8（PublishedWorldObservation の atomic snapshot contract）/
  toString の descriptor table 化（#6）/ RetryBackoffPolicy の tuning parameter 化（#7）。
- **「設計をさらに練り直す段階」から「この仕様を実装 contract として固定し、段階的に実装・検証する段階」
  へ移行**する判断を確認（第十八者 #38）。production reclaim のみ従来通り NO-GO。

### H.11.28 別視点調査 — RuntimeDrainAudit / RuntimePolicyEngine / ISRLifecycle の棚卸し確定（2026-08-14）

「ソースコード全体を可能な限り詳細に調査」する別視点として、設計書に未記載の 3 コンポーネント
（**RuntimeDrainAudit / RuntimePolicyEngine / ISRLifecycle**）をソースで棚卸しした。全て設計書に
grep で見つからない（設計不足）ことを確認した。

#### H.11.28.1 RuntimeDrainAudit（設計書に未記載）

`RuntimeDrainAudit`（src/audioengine/RuntimeDrainAudit.h:26）は drain 状態の監査構造体。

```text
RuntimeDrainAudit {
    publishedCount / retiredCount / activeWorldCount / reclaimAttemptCount / reclaimSuccessCount
    pendingPublication / pendingRetire / activeCrossfadeCount / deferredPublish /
    quarantineResident / routerPendingRetire / overflowRingResident / stuckReaderCount
}

BlockingReason { None, PendingPublication, PendingRetire, ActiveCrossfade, DeferredPublish,
                 QuarantineResident, RouterPendingRetire, ReaderActive, Unknown }
ConsistencyState { Consistent, Suspicious, Broken }
```

- **isAllZero()**: 「監査ログ出力専用。shutdown 完了判定には使用しない」（RuntimeDrainAudit.h:77）—
  **設計書の「isFullyDrained は Observation のみ」原則と整合**。
- **verifyWorldConsistency()**: `publishedCount - retiredCount == activeWorldCount` で
  Consistent / Suspicious / Broken を判定 — **「Diagnostic 限定、Shutdown Authority にはしない」**
  （RuntimeDrainAudit.h:87）。
- 使用箇所: `AudioEngine::collectDrainAudit()`（Threading.cpp:70）、
  ReleaseResources.cpp:492（ConsistencyState::Consistent チェック — **shutdown 完了判定ではなく診断**）。

#### H.11.28.2 RuntimePolicyEngine（設計書に未記載）

`RuntimePolicyEngine`（src/audioengine/RuntimePolicyEngine.h）は HealthMonitor の MonitorState から
RecoveryAction を選択する policy 決定エンジン。**H.11.24 の RuntimeHealthMonitor の上位層**。

```text
PolicySource（10 種）: RetireStall / PublicationStall / ReaderStuck / CrossfadeTimeout /
                       LearnerAnomaly / WorldConsistency / AudioOutputAnomaly / EmergencyCondition /
                       RecoveryOutcome / SafeModeState

RecoveryAction（6 レベル階層化 v6.6）:
    Observe(0)  → Throttle(1)  → Recover(2)  → Restore(3)  → Safe(4)  → Critical(5)
    監視            抑制            回復            復元            安全確保      重大
    (HealthEvent)  (admissionStrict) (ForceRetireDrain) (Rollback)   (Soft/HardSafeMode) (EmergencyDrain)

RestorePhase { None, EpochRecoveryIssued, LearnerRollbackDone, IdleWorldPublished }
```

- **PolicyDecision を updateHealthState() に渡し、HealthState は HealthMonitor が最終決定**
  （PolicyEngine は HealthState を直接変更しない — authority 分離）。
- 使用箇所: AudioEngine.Timer.cpp:1546（evaluateAggregate → Recover Action に委譲）。

#### H.11.28.3 ISRLifecycle（設計書に未記載）

`ISRLifecycle`（src/audioengine/ISRLifecycle.h）は Audio Thread のライフサイクル隔離
（LifecycleIsolationRuntime）。

```text
LifecyclePhase { Uninitialized, Preparing, Prepared, AudioRunning, Releasing, Released, Shutdown }
  7-state FSM（AudioEngine::ShutdownPhase 11-state / isr::ShutdownPhase とは別の層）

LifecycleIsolationRuntime: JUCE callback 違反（overlap / late callback）を検出・abort
LifecycleBarrierRuntime: publishPreparedBarrier / publishReleasingBarrier / publishShutdownBarrier
```

- **受入条件 LIF-1〜LIF-6**:
  - LIF-1: prepareToPlay serialized
  - LIF-2: releaseResources は AudioRunning 中に呼べない
  - LIF-3: **Releasing phase 中の publish 禁止**
  - LIF-4: crossfade start は Prepared 以降のみ
  - LIF-5: callback 中 runtimeVersion 変化なし
  - LIF-6: callback 中 DSP generation 変化なし
- 使用箇所: PrepareToPlay.cpp:18,304 / ReleaseResources.cpp:70,572（LifecycleIsolationRuntime integration）。
- **LIF-3（Releasing phase 中の publish 禁止）は H.11.19（PublicationAdmission::evaluate の shutdown gate）と
  整合**。LIF-6（callback 中 DSP generation 変化なし）は A2-G18（generation binding）と整合。

#### H.11.28.4 反映方針・確定事項

- **棚卸し確定**: RuntimeDrainAudit / RuntimePolicyEngine / ISRLifecycle は実装済み・設計書に未記載
  → 本 H.11.28 で棚卸し確定。
- **isFullyDrained との関係**: RuntimeDrainAudit の isAllZero() / verifyWorldConsistency() は
  「Diagnostic 限定、Shutdown Authority にはしない」— 設計書の「isFullyDrained は Observation のみ」と整合。
  **A2-G02（semantic accounting single authority）の対象として、RuntimeDrainAudit は監査ログ・診断専用に
  維持する**ことを確定。
- **RuntimePolicyEngine の authority**: PolicyEngine は RecoveryAction を選択するが、HealthState は
  HealthMonitor が最終決定（authority 分離）— H.11.24 の RuntimeHealthMonitor と整合。
- **ISRLifecycle の LIF-3 / LIF-6** は PublicationAdmission（H.11.19）と A2-G18（generation binding）の
  実装根拠として設計書に確定。

### H.11.29 別視点調査 — SemanticSchema / Dead code / 検証系コンポーネントの棚卸し確定（2026-08-14）

「ソースコード全体を可能な限り詳細に調査」する別視点として、設計書に未記載・未詳述のコンポーネント
（**ISRRuntimeSemanticSchema / ISRRuntimeWorldAuthority / ISRClosure / ISRHB / ISREvidenceExporter /
ISRSealedObject / DeferredRetireFallbackQueue**）をソースで棚卸しした。

#### H.11.29.1 ISRRuntimeWorldAuthority は forward ヘッダ（新規反映不要）

`ISRRuntimeWorldAuthority.h` は **RuntimeWorldAuthority.h への forward**（A3 / ADR-D3: canonical
RuntimeWorldAuthority は RuntimeWorldAuthority.h に定義、PendingPublishRegistry を含む D3 完全面）。
独立コンポーネントではなく、**H.11.25.4（currentWorld_ 廃止）の read authority 設計と整合**。
→ 新規設計要素なし。

#### H.11.29.2 ISRRuntimeSemanticSchema — Authority Singularization の静的検証（設計書 line 1049 で一部言及済み）

`ISRRuntimeSemanticSchema.h` は **Runtime の semantic schema**（Authority / Ownership / Mutability /
Visibility / Lifetime の分類）を定義する。

```text
RuntimeAuthorityClass { Authoritative, Derived, Diagnostic, ExecutorLocal }（:65-70）

SemanticCategory / OwnershipClass / MutabilityClass / VisibilityClass / LifetimeClass
RuntimeFieldDescriptor { fieldName, semanticCategory, ... }
RuntimeAuthorityInventoryEntry { fieldName, authorityClass }
validateFieldDescriptorSet() — field 名の一意性を compile-time 検証（:79-95）

GenerationSemantic / TopologySemantic / RoutingSemantic / ExecutionSemantic /
PublicationSemantic / OverlapSemantic / RetireSemantic
```

- **設計書 line 1049** で「BuildError を semantic architecture（ISRRuntimeSemanticSchema.h の
  AuthorityClass / VisibilityClass など）と統合することを推奨」と一部言及済みだが、詳細未記載。
- **A2-G02（semantic accounting single authority）の静的検証根拠**として確定: 各 Runtime field の
  authorityClass（Authoritative / Derived / Diagnostic / ExecutorLocal）を静的記述し、
  「外部 setter による Authoritative field の上書き」をコンパイル時 / 監査時に検出する。
- これは **Authority Singularization の静的検証レイヤー**であり、設計書の A2-G10 / A2-G17
  （compile error / forged 防止）と整合。

#### H.11.29.3 DeferredRetireFallbackQueue は dead code（H.11.21 の削除キュー棚卸しの拡張）

`DeferredRetireFallbackQueue`（src/core/DeferredRetireFallbackQueue.h:27）は retire の fallback queue
だが、**ソース全体で使用箇所なし**（grep 確認）＝ dead code。

```text
DeferredRetireFallbackEntry { ptr, deleter, ... }
push(): HardLimit 超過時はドロップして false を返す（:43-45、work37）
overflowCount() / overflowRate()（PolicyEngine 連携用）
```

- **H.11.21 の削除キュー棚卸しの拡張**: DeletionQueue / SnapshotRetireManager（未使用）に加えて、
  **DeferredRetireFallbackQueue も未使用（dead code）として棚卸し確定**。
- ただし「HardLimit 超過時はドロップして false」は RetireQuarantineStore の「store full 時 deleter
  実行禁止」と整合する（ドロップは deleter 実行ではなく、呼び出し元へ false を返す）。
- 将来接続時は A2-G23（physical destruction audit）の対象として確認。

#### H.11.29.4 ISRClosure / ISRHB / ISREvidenceExporter / ISRSealedObject（検証・診断系）

設計書に未記載の検証・診断系コンポーネント。

| コンポーネント | 役割 | 設計書との関係 |
|---|---|---|
| `ISRClosure.h`（ClosureValidator） | publication graph の closure 整合性検証（cycle / dangling ref） | ISR の semantic closure 検証 |
| `ISRHB.h`（HBRuntimeCore / HBTraceRuntime） | happens-before の検証・trace（CI/Debug build のみ） | memory ordering の検証（EpochDomain / RCUReader の HB と整合） |
| `ISREvidenceExporter.h` | evidence の export | EpochQuiescenceEvidence（H.11.13.3）の出力系 |
| `ISRSealedObject.h` | sealed object（sealed された不変オブジェクト） | immutable snapshot（GlobalSnapshot）と関連 |

- これらは全て**検証・診断・CI 用**であり、production の runtime authority には直接関与しない。
- **A2-G13 / G-E5 等の acceptance test の検証基盤**として設計書に確定。

#### H.11.29.5 反映方針・確定事項

- **棚卸し確定**: ISRRuntimeSemanticSchema（AuthorityClass 4 値）/ ISRClosure / ISRHB /
  ISREvidenceExporter / ISRSealedObject は実装済み・設計書に未詳述 → 本 H.11.29 で棚卸し確定。
- **A2-G02 の静的検証根拠拡張**: ISRRuntimeSemanticSchema の RuntimeAuthorityClass で
  Authoritative / Derived / Diagnostic / ExecutorLocal を静的記述（外部 setter 上書き検出）。
- **dead code 確定**: DeferredRetireFallbackQueue は未使用（H.11.21 の DeletionQueue / SnapshotRetireManager
  と同様）。将来接続時は A2-G23 の対象。
- **ISRRuntimeWorldAuthority は forward**（新規設計要素なし）— H.11.25.4 と整合。
