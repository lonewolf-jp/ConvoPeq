# Phase C-0 事前監査 — R4 retire 順序（FIFO）の完全解消

- **日付**: 2026-08-19
- **対象**: R4「retire 順序逆転の完全解消」（`REPAIR_PLAN2-dash2.md §2.1` / `REPAIR_PLAN2-dash.md §6.3`）
- **目的**: R4 の残余課題（retire 順序逆転の完全解消 = FIFO 強化）の実装 GO/NO-GO を判定するための producer / consumer / 順序保証 / 安全性契約の完全列挙
- **判定**: **NO-GO（現時点では実装しない）** — 条件付き GO（FIFO 順序が determinism / telemetry 要件として必須になる設計確定時）

---

## 1. R4 仕様の定義（対象セクション）

### 1.1 dash2 §2.1（REPAIR_PLAN2-dash2.md:1248-1340）

**現状（実コード照合）**:

- runtime 経路は対応済み: `retireDSPHandleForRuntime` / `retireByHandle` は `requestReclaimHandle` 経由に一本化
- `shutdownReclaim` は X3-R4 Phase 7 で完全削除済み（AC-R4-1 call sites==0 / AC-R4-2 symbol absent）
- `reclaim(ReclaimMode, ...)` は RuntimeEBR / ShutdownQuiescent の2モードで一本化（ReclaimAuthority）

**残る課題**:

- retire 順序逆転（後から発行された retire が先に処理される可能性）の完全解消は保留
- ただし quarantine fallback で UAF / リークは排除済み

**設計方針（INV 明文化）**:

```text
INV-EPOCH-1: retire 済み handle の reclaim 可否は retireEpoch と minReaderEpoch の比較のみで決定する
INV-EPOCH-2: reader が grace period 内に居る限り reclaim 禁止（retire 順序とは無関係）
INV-FIFO-1: retire の処理順序は epoch 順を目指す（optimization / determinism / telemetry）— memory safety ではない
```

**レビュー指摘**: Epoch safety ≠ FIFO。FIFO order = optimization / determinism / telemetry の property — memory safety そのものではない。RCU 整合性（ISO C++ P0279R1）: retire/reclaim の安全性は read-side critical section と grace period の関係で決まり、単なるキュー処理順序ではない。

**実装手順**:

1. retire の epoch 順序保証（FIFO）を強化（enqueue 順と epoch 順の整合）
2. `drainDeferredRetireQueues` の順序検証（pendingReclaimHandles_ の再試行機構と整合）
3. テスト追加: retire 順序逆転の回帰テスト（AC-R4-T1〜T7 を拡張）

**Acceptance Criteria**: AC-R4-1〜10 全充足 / AC-ISR-1（Audio Thread は retire ordering API を呼び出さない）/ epoch safety と FIFO の分離をテスト

**影響範囲 / リスク**: `ISRRetireRouter` / `AudioEngine.Retire.cpp`（drainDeferredRetireQueues）/ `ReleaseResources.cpp`。リスク: 中。

### 1.2 dash.md §6.3（REPAIR_PLAN2-dash.md:2377-2500）— R4 詳細設計 Phase 0-7

- **Phase 0**: 現状の安全性契約を固定（コード変更なし）
- **Phase 1**: `shutdownReclaim()` の中身を Authority へ移す（ReclaimMode / ReclaimRequest 導入）
- **Phase 2**: RuntimeEBR を先に一本化（pendingReclaimHandles_ 再試行は維持）
- **Phase 3**: ShutdownQuiescent を実装（ShutdownQuiescenceProof 独立オブジェクト化）
- **Phase 4**: releaseResources 移行（reclaim(ShutdownQuiescent) は quiescence 確立後に移動）
- **Phase 5**: CacheMap destructor 移行
- **Phase 6**: `shutdownReclaim()` を deprecated 化（call site=0 確認）
- **Phase 7**: `DSPHandleRuntime::shutdownReclaim()` 完全削除

**NG（やってはいけないこと）**: NG1 名前だけ ReclaimAuthority / NG2 shutdown なら無条件 reclaim / NG3 pending を shutdown 時に無条件 clear / NG4 activeReaderCount()==0 だけで shutdown reclaim / NG5 Faulted で pending を破棄

**テスト R4-T1〜T7** / **AC-R4-1〜10** / **実装順序 R4-0〜R4-12**（R4-5 以前には shutdownReclaim() を削除しない）

## 2. retire intent の生成元（producer）完全列挙

| # | 生成元 | 経路 | 実行スレッド | 状態 |
| --- | --- | --- | --- | --- |
| 1 | `retireDSPHandleForRuntime` / `retireByHandle` | → `requestReclaimHandle`（AudioEngine.h:4248）→ `requestReclaim` → epoch 確認 → reclaim / pending | Non-RT（CoordinatorLoop / Builder） | **PRIMARY runtime 経路** |
| 2 | `enqueueDeferredDeleteNonRtWithResult` | → `enqueueWithRetry`（D→Q→E→T chain） | Non-RT | 通常経路 |
| 3 | `enqueueDeferredDeleteNonRtWithResult`（shutdown 時） | → `shutdownReclaim` → TerminalReclaimAuthority | Non-RT（shutdown） | shutdown 所有権移転 |
| 4 | `emitRetireIntent` / `emitRetireIntentRT` | → LifetimeState MPSC → `dequeuePendingRetireIntents` → `enqueueRetire` | RT（emitRetireIntentRT）/ Non-RT（emitRetireIntent）→ Non-RT consumer | retire intent 経路 |
| 5 | `quarantineRetire`（SnapshotCoordinator.cpp:26 / DSPLifetimeManager） | → RetireQuarantineStore（Q） | Non-RT | quarantine 経路 |
| 6 | `retireRT` | → D queue only（lock-free） | RT 想定 | **production 呼び出し元ゼロ**（RefCountedDeferred::releaseRT も未使用） |
| 7 | `enqueueRetireEpochBounded` | → `enqueueRetire` | — | **production 呼び出し元ゼロ** |
| 8 | `retire`（NonRT retry） | → `enqueueWithRetry` | Non-RT | リトライ込み |

- **実質的な producer は全て Non-RT**（#4 の RT 側は MPSC push のみで、D への enqueue は Non-RT consumer が実行）。
- AC-ISR-1（Audio Thread は retire ordering API を呼び出さない）は成立。

## 3. retire キュー / ストアの構造

| 層 | 型 | capacity | 同期 | 順序保証 |
| --- | --- | --- | --- | --- |
| D | `DeferredDeletionQueue`（Vyukov bounded MPMC） | 4096 | lock-free | **FIFO-strict**（reclaim は dequeue 先頭と一致した時のみ削除。先頭が不安全なら break） |
| Q | `RetireQuarantineStore`（std::array + index） | 512 | mutex（Non-RT） | **epoch-gated・非 FIFO**（全走査で epoch-safe を抽出） |
| E | `EmergencyQuarantineStore`（Q と同型） | 512 | mutex（Non-RT） | **epoch-gated・非 FIFO** |
| T | `TerminalReclaimAuthority`（growable std::vector） | 無制限 | mutex（Non-RT） | **epoch-gated・非 FIFO** |
| pending | `std::vector<ReclaimIdentity{handle, retireSequence}>` | 動的 | mutex | **epoch-gated 再試行・非厳密 FIFO**（INV-FIFO-1） |

- D は構造的に FIFO だが、MPMC のため enqueue 順 ≠ epoch 順（DeferredDeletionQueueReclaimTests Test 7 が測定）。
- Q/E/T は epoch 安全なエントリを順序無関係に回収するため、retire 順序逆転が発生し得る。

## 4. 消費経路（consumer / drain）完全列挙

| # | 呼び出し元 | 実行スレッド | 内容 |
| --- | --- | --- | --- |
| 1 | `CoordinatorLoop::runCoordinatorPhase` → `drainDeferredRetireQueues(false)` | CoordinatorLoop（1ms Non-RT） | E-1.9-B event-driven（waitForDrainSignalOrTimeout）+ 1ms fallback |
| 2 | `Timer::timerCallback` → `tryReclaimResources` + `drainDeferredRetireQueues(false)` | MessageThread Timer（100ms Non-RT） | 定期 drain |
| 3 | `tryReclaim` | Non-RT | `provider_->tryReclaim()` + `drainQuarantineStore` + `drainEmergencyAndTerminal` |
| 4 | `enqueueWithRetry` Stage 2 | Non-RT | QueuePressure 時に `provider_->tryReclaim()` + `drainEmergencyAndTerminal` |
| 5 | shutdown: `drainDeferredRetireQueues(true)` + `drainAllQuarantineStore` + `drainAllUnsafe` | shutdown | 全強制解放 |
| 6 | `requestReclaimHandle` → `requestReclaim` | Non-RT | epoch 確認 → reclaim / pending |

- 全 consumer は Non-RT。AC-R4-8/9（Audio Thread から reclaim authority を呼ばない / reclaim を RT で実行しない）成立。

## 5. 現在すでに成立している FIFO / epoch 順序の性質

- **D queue**: FIFO-strict reclaim（先頭のみ削除）。ただし MPMC のため enqueue 順 ≠ epoch 順（逆転は既知・許容）。
- **Q/E/T**: epoch-gated・非 FIFO。retire 順序逆転が発生し得るが、INV-EPOCH-1/2 が memory safety を保証。
- **pendingReclaimHandles_**: epoch-gated 再試行。ベクタ順に処理し、不安全なものは末尾へ再登録（順序逆転あり）。
- **INV-FIFO-1**: FIFO 順序は optimization / determinism / telemetry の property — memory safety ではない。
- **結論**: 「retire順序逆転 = UAF」とは評価しない（dash2 の「UAF/リークは既に排除済み」が正しい位置付け）。

## 6. A2 / E-1.9 / P-8〜P-14 との関係

| 関連 | 内容 | FIFO 強化の影響 |
| --- | --- | --- |
| A2（ISRLifetimeProof.h） | pendingReclaimHandles_ → ReclaimIdentity（handle + retireSequence, G14） | 影響なし（identity は順序と無関係に保持） |
| E-1.9（quarantine wake） | signalDrainWakeup / waitForDrainSignalOrTimeout（Non-RT のみ） | 影響なし（predicate は pendingRetireCount / residentCountAtomic） |
| P-8〜P-14（shutdown authority / terminal ownership） | shutdownReclaim → TerminalReclaimAuthority（epoch-gated destruction） | 影響なし（所有権移転は順序と無関係） |
| INV-EPOCH-1/2（primary） | reclaim 可否は epoch 比較のみ / grace period 内は reclaim 禁止 | FIFO 強化で変更しない（primary は不変） |

- FIFO は secondary（INV-FIFO-1）のため、A2 / E-1.9 / P-8〜P-14 の memory safety 契約に影響しない。

## 7. RT 安全性

- `retireRT`: RT-safe（D queue only・lock-free）— ただし production 呼び出し元ゼロ。
- `enqueueWithRetry`: Non-RT 限定（jassert(!isAudioThread())。Q/E/T は mutex + allocation のため）。
- `reclaim` / deleter 実行: Non-RT のみ（AC-R4-8/9）。
- `emitRetireIntentRT`（AudioEngine.Commit.cpp:485）: RT から MPSC push のみ。D への enqueue は Non-RT consumer。
- **AC-ISR-1 成立**: Audio Thread は retire ordering API を呼び出さない。

## 8. 既存テスト

| テスト | 内容 | 関連 |
| --- | --- | --- |
| `DeferredDeletionQueueReclaimTests.cpp` Test 2 | FIFO 順序保証（先頭から順に回収） | D queue FIFO |
| `DeferredDeletionQueueReclaimTests.cpp` Test 7 | MPMC epoch 逆転シナリオ測定（複数スレッド同時 enqueue で逆転が実際に発生するか） | **順序逆転の実測** |
| `RetireGraceSemanticsTests.cpp` | 同 priority 内: 古い epoch が先（FIFO）/ OverflowRing FIFO | retire 順序 |
| `invariant_INV3_INV5.cpp` | INV-3-1: retire → epoch safe → reclaim の順序 | epoch 順序 |
| `ISRSemanticValidationTests.cpp` Test 7 | commit-before-swap ordering | publish 順序 |
| `MpscBoundedRingTests.cpp` | SPSC / cross-type / producer-hole FIFO | MPSC FIFO |
| `PriorityIntegrationTests.cpp` | High 優先度 retire intent → dequeue 順序 | retire intent 順序 |
| `ShutdownRetireIntentDrainTests.cpp` | shutdown 時 retire intent drain | shutdown |
| `NormalRetireDSPHandleCompareTests.cpp` | retire handle 比較 | retire 経路 |
| R4-T1〜T7（dash.md §6.3 指定） | RuntimeEBR / Shutdown happy path / premature / stale reader / pending / CacheMap / API architectural | R4 テスト仕様 |

- 順序逆転の回帰テスト（AC-R4-T1〜T7 拡張）は**未追加**（R4 実装手順 3 の保留項目）。

## 9. R4 実装との差分分析（AC-R4-1〜10 充足状況）

| AC | 内容 | 現状 |
| --- | --- | --- |
| AC-R4-1 | shutdownReclaim() call sites == 0 | ✅ `DSPHandleRuntime::shutdownReclaim()` 削除済み（ISRDSPHandle.h:175 コメント）。`ISRRetireRouter::shutdownReclaim()` は別関数（shutdown 所有権移転・P-4 設計）で存続 |
| AC-R4-2 | shutdownReclaim() symbol absent | ✅ DSPHandleRuntime 側は symbol 不在 |
| AC-R4-3 | DSPHandleRuntime::reclaim() の production caller が ReclaimAuthority のみ | ✅ |
| AC-R4-4 | 両 mode で same physical reclaim primitive | ✅ reclaim(ReclaimMode, ...) |
| AC-R4-5 | ShutdownQuiescent に quiescence proof 必須 | ✅ readerRegistrationClosed を precondition として実装済み |
| AC-R4-6 | epoch unsafe な RuntimeEBR handle は pendingReclaimHandles_ に残る | ✅ AudioEngine.Retire.cpp:71-117 |
| AC-R4-7 | Faulted で pending を clear しない | ✅ |
| AC-R4-8 | Audio Thread から reclaim authority を呼ばない | ✅ AC-ISR-1 |
| AC-R4-9 | reclaim 自体は RT thread で実行しない | ✅ |
| AC-R4-10 | isFullyDrained() は reclaim authority にならない | ✅ |

- **AC-R4-1〜10 は全充足**。R4 の核心（shutdownReclaim 排除・ReclaimAuthority 一本化）は実装済み。
- **残る差分**: 「retire 順序逆転の完全解消」（FIFO 強化）のみ。これは INV-FIFO-1 で secondary（optimization / determinism / telemetry）と定義され、memory safety ではない。

## 10. GO/NO-GO 判定

**判定: NO-GO（現時点では実装しない）** — 条件付き GO

### 根拠

1. **R4 の核心は実装済み**: AC-R4-1〜10 全充足。shutdownReclaim 排除・ReclaimAuthority 一本化・pendingReclaimHandles_ 再試行・Faulted 非 clear が全て成立。
2. **残る課題は secondary**: 「retire 順序逆転の完全解消」は INV-FIFO-1 で明示的に optimization / determinism / telemetry の property と定義。memory safety ではない。
3. **Epoch safety ≠ FIFO**: FIFO を強化しても memory safety は向上しない（INV-EPOCH-1/2 が既に保証）。
4. **UAF / リークは排除済み**: quarantine fallback + RetireQuarantineStore で構造的に排除。
5. **順序逆転は既知・許容**: DeferredDeletionQueueReclaimTests Test 7 が MPMC epoch 逆転を実測済み。
6. **F-0 / H-0 の前例と整合**: speculative な変更はしない。実装するなら determinism / telemetry の具体的な不具合が先に必要。
7. **リスク: 中**: retire 順序の変更は epoch 安全性に影響する可能性があるため、実装するなら慎重な検証（R4-T1〜T7 拡張）が必須。

### 条件付き GO トリガー

以下のいずれかが確定した場合に再評価:

- FIFO 順序が determinism / telemetry 要件として必須になる設計（例: 順序依存の telemetry 集計・デバッグ再現性要件）
- retire 順序逆転が実運用で観測される具体的な不具合（例: 順序依存のリソース競合・測定値の非決定性）

### 推奨アクション

- production code は変更しない（C-0 監査完了まで変更禁止ゲートを維持）
- 既存の INV 明文化（AudioEngine.Retire.cpp:71-117）は現状のまま維持
- 将来 FIFO 強化を実装する場合は: 1) enqueue 順と epoch 順の整合 2) drainDeferredRetireQueues の順序検証 3) AC-R4-T1〜T7 拡張テスト追加
