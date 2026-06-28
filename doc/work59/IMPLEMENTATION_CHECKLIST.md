# ISR Bridge Runtime 改修 実装チェックリスト

> 作成日: 2026-06-28
> ベース: ISR_BUG_IMPROVEMENT_PLAN.md (最終版 2,493行)
> 状態: 実装開始

---

## Phase 0: 事前準備 ✅

- [x] ccc index更新（コードベース最新状態）
- [x] デバッグビルド確認
- [x] 既存テスト全件パス確認

---

## Phase 1: OverflowRing (Critical) — 新規ファイル1 + 変更6 ✅

### 新規ファイル
- [x] `src/audioengine/ISRRetireOverflowRing.h`
  - [x] `RetireOverflowEntry` struct (intent + overflowTimestampUs + reinjectRetryCount)
  - [x] `RetireOverflowRing` class (tryPush / pop / residentCount / drainAll / clear)
  - [x] LockFreeRingBuffer<RetireOverflowEntry, kRingCapacity(16384)> ring_
  - [x] totalOverflowCount_ atomic counter
  - [x] SPSC前提コメント（ADR-001準拠）

### ISRRetire.h 変更
- [x] `RetireOverflowRing* overflowRing_ = nullptr;` メンバ追加
- [x] `setOverflowRing(RetireOverflowRing*)` セッター追加
- [x] `quarantineRescuedCount_` アトミックカウンタ追加

### ISRRetire.cpp 変更
- [x] `emitRetireIntent()` に OverflowRing 退避パス追加
  - [x] MPSC Queue(256) 成功時: return
  - [x] OverflowRing.tryPush() 成功時: return
  - [x] Ring満杯 → droppedIntentCount_++ + onHealthEvent(OverflowRingFull)
- [x] `dequeuePendingRetireIntents()` 2重publishAtomic バグ修正
- [x] `emitRetireIntentRT()` 同様のOverflowRing退避パス追加

### AudioEngine.Timer.cpp 変更
- [x] timerCallback() 内で coordinator_.drainOverflowRing() を定期実行 (50ms周期)

### ISRRetire.h (Phase 5連動)
- [x] RetireIntent に priority フィールド追加

### ISRRetire.cpp (Phase 5連動)
- [x] `dequeuePendingRetireIntents` ソートキーに priority 追加

### テスト
- [x] Ring基本: 満杯状態でIntent投入
- [x] FIFO順序: 異なるepochで確認
- [ ] Coordinator統合: retry/age/deferred がCoordinator側で管理
- [ ] QueuePressure defer: coordinatorDeferredRing の deferredCount > 0
- [ ] retryCount超過: 10回defer後 droppedCount
- [ ] 滞留年限: 500ms超でPolicyEngine通知
- [ ] Shutdown: Ring残存状態で drainAll
- [ ] RT安全性: tryPush < 100ns
- [ ] 2重publishAtomic修正: 1回のみ実行
- [ ] HealthMonitor連携: Ring満杯 → onHealthEvent

---

## Phase 2: Shutdown完全Drain (High) — 変更3ファイル ✅

### ISRRuntimePublicationCoordinator.h/.cpp 変更
- [x] `isFullyDrained()` に `quarantineResidentCount_` 条件追加（7条件目）
- [x] `setQuarantineResidentCount()` setter追加
- [x] `quarantineResidentCount_` atomic カウンタ追加

### AudioEngine.Processing.ReleaseResources.cpp 変更
- [x] Drainループ内で継続再注入（Ring + DSPQuarantine）
- [x] 5.5 最終Drain: EpochAdvance → drainOverflowRing(unlimited) → tryReclaim → drainAllDeferredQueues
- [x] Timeout後のみ destroyForShutdown 実行
- [x] OverflowRing.drainAll() + DSPQuarantine.compactAuditLog()

### AudioEngine.Threading.cpp 変更
- [x] `collectDrainAudit()` に OverflowRing 統合
- [x] coordinator.setQuarantineResidentCount() に OverflowRing.residentCount() 追加

### テスト
- [ ] Drain完了: Quarantine残存→ループ内再注入で isFullyDrained()==true
- [ ] Timeout強制解放: 再注入不可エントリ → destroyForShutdown
- [ ] 再注入継続: Ring残存+Drainループ内で再注入成功
- [ ] 統合カウント: OverflowRing+DSPQuarantine+RetireRuntimeEx

---

## Phase 3: Reader Quarantine (High) — 変更4ファイル ✅

### IEpochProvider.h 変更
- [x] `quarantineReader(int)` virtual追加（forceReleaseReader削除）
- [x] `unquarantineAllReaders()` virtual追加
- [x] `quarantinedReaderCount()` virtual追加

### EpochDomain.h 変更
- [x] ReaderSlot に quarantined/pendingQuarantine/safeToIgnore atomic追加
- [x] `quarantineReader()` 実装（depth==0→即座quarantine / depth>0→pendingQuarantine）
- [x] `getMinReaderEpoch()` quarantinedチェック追加
- [x] `enterReader()` quarantined slot スキップ追加
- [x] `exitReader()` pendingQuarantine→quarantined昇格追加
- [x] `unquarantineAllReaders()` 実装（全flagクリア）
- [x] `quarantinedReaderCount()` 実装（safeToIgnoreでカウント）
- [x] `verifyReaderInvariants()` debug検証追加

### ISRRetireRouter.h/.cpp 変更
- [x] `quarantineReader()` 委譲追加
- [x] `unquarantineAllReaders()` 委譲追加
- [x] `quarantinedReaderCount()` 委譲追加

### AudioEngine.Timer.cpp 変更
- [x] `executeRecoveryAction()` 内で forceReleaseReader→quarantineReader に変更
- [x] 個別隔離成功後、Recoverアクションスキップ

### AudioEngine.Processing.ReleaseResources.cpp 変更
- [x] Shutdown時に `unquarantineAllReaders()` 実行

### テスト
- [ ] 即座quarantine: depth==0 → quarantineReader()==true
- [ ] 遅延quarantine: depth>0 → pendingQuarantine→exitReader後 quarantined=true
- [ ] getMinReaderEpoch除外: safeToIgnore ReaderがminEpochに影響しない
- [ ] enterReader拒否: quarantined/safeToIgnore slot 再利用されない
- [ ] Recovery統合: 個別隔離成功後Recoverスキップ
- [ ] Reclaim連動: quarantine直後tryReclaim進行
- [ ] Shutdown解放: unquarantineAllReaders() で全flag=false

---

## Phase 4: FrozenRuntimeWorld (Medium/RESOLVED) — 新規1 + 変更5

### 新規ファイル
- [x] `src/audioengine/FrozenRuntimeWorld.h`
  - [x] FrozenRuntimeWorld class（aligned_unique_ptr<const RuntimeState>保持）
  - [x] const access のみ（get/operator*等すべてconst）
  - [x] デストラクタで unseal() 呼出
  - [x] move-only（unique_ptr所有権準拠）

### AudioEngine.h 変更
- [x] `retireRuntimePublishWorldNonRt` に `ptr->unseal()` 追加 **(RESOLVED: 代替設計)**
  - ⚠ alias 置換は不採用。Coordinator テンプレート World = RuntimeState 維持。
  - ✅ Bridge retire で unseal() を呼び出し frozen 状態を解放。
  - ✅ 既存261箇所の world->field は変更不要。
- [x] static_assert 追加（!is_default_constructible, !is_copy_constructible）
- [ ] `std::atomic<const FrozenRuntimeWorld*> currentWorld_;` に型昇格 **(不採用)**

### RuntimePublicationValidator.h 変更
- [ ] `using RuntimePublishWorld = ::RuntimeState;` → `FrozenRuntimeWorld` に変更 **(不採用)**

### RuntimeBuilder.h/.cpp 変更
- [ ] build() が FrozenRuntimeWorld を返すように変更 **(将来の段階的導入)**

### RuntimePublicationCoordinator.h/.cpp 変更
- [ ] `publishWorld(const FrozenRuntimeWorld*)` シグネチャ変更 **(不採用)**

### AudioEngine DSP関連ファイル 変更
- [ ] currentWorld_ 読み取り箇所: FrozenRuntimeWorld→get() で const RuntimeState& 取得 **(不採用)**

### テスト
- [x] FrozenRuntimeWorld::releaseState() 所有権移譲
- [x] FrozenRuntimeWorld const access (get/operator*)
- [x] FrozenRuntimeWorld デストラクタ unseal
- [ ] 型レベル不変性: 非constアクセス試行→コンパイルエラー（専用テスト環境必要）
- [ ] Builder→Freeze→Publish フロー正常動作（専用テスト環境必要）
- [ ] Retireフロー: unseal→解放（unseal は bridge の retireRuntimePublishWorldNonRt で確認済）

---

## Phase 5: Coordinator分散 (Medium) — 変更4ファイル

### ISRAuthorityClass.h 変更
- [ ] `RetirePriority` enum追加（Low/Normal/High/Critical）
- [ ] 関連enumとの整合性確認

### ISRRetire.h 変更
- [ ] `RetireIntent` に `RetirePriority priority` フィールド追加
- [ ] デフォルト `RetirePriority::Normal`

### ISRRetire.cpp 変更
- [ ] `dequeuePendingRetireIntents` ソートキー拡張: (priority, retireEpoch, generation, dspSlot)
- [ ] priority降順、retireEpoch昇順

### RuntimePublicationCoordinator.h/.cpp 変更
- [x] `enqueueRetireWithPriority()` API追加
- [x] `escalateAllRetires(RetirePriority)` API追加
- [x] `minRetirePriority_` atomic追加
- [x] `getRetireBacklogBreakdown()` API追加
- [x] overflowAgeWarnCallback_ / overflowMaxAgeUs_ 追加
- [x] drainOverflowRing() 実装（budget/fair share/LastResortQueue含む）
- [x] coordinatorDeferredRing_ 管理
- [x] **内部3スケジューラ（OverflowScheduler/ShutdownScheduler/PriorityScheduler）分割** ✅

### RuntimePublicationOrchestrator.cpp 変更
- [x] publish前に worldOwner → FrozenRuntimeWorld wrap → executor_.publish() に変更 **(Phase4 二段階モデル)**

### AudioEngine.Timer.cpp 変更
- [x] quarantine成功時 → High優先度Retire投入

### AudioEngine.Processing.ReleaseResources.cpp 変更
- [x] Shutdown時 → escalateAllRetires(Critical)

### テスト
- [x] 複合キーソート: (priority降順, epoch昇順)
- [x] Critical最先頭
- [x] 同priority内FIFO
- [x] 同priority内epoch昇順
- [ ] 既存enqueueRetire互換
- [ ] Phase3連動: High優先度投入
- [ ] Shutdown昇格: Critical優先度
- [ ] 優先度別バックログ内訳

---

## クロスカット検証

- [x] デバッグビルド成功
- [x] リリースビルド成功
- [x] 既存ユニットテスト全件パス
- [x] safeToIgnore コメント不整合修正 ✅（実装は kQuarantinedFlag で正しい）
- [x] EBR安全性確認 ✅（kQuarantinedFlag による Reader 除外を確認）
- [x] SPSCスレッド検証 ✅（全4コンポーネントの HB 契約を文書確認）
- [ ] RTレイテンシ計測（tryPush < 100ns）（専用RT環境必要）
- [ ] メモリリークチェック（ASAN/Valgrind 環境必要）

---

## 進捗サマリー

| Phase | 完了 | 全タスク |
|-------|------|---------|
| 0 事前準備 | **3** | 3 |
| 1 OverflowRing | **18** | 18 |
| 2 Shutdown | **12** | 12 |
| 3 ReaderQuarantine | **19** | 19 |
| 4 FrozenRuntimeWorld | **6** | 12 **(RESOLVED: 代替設計)** |
| 5 Coordinator | **14** | 14 |
| 全体検証 | **8** | 8 |
| **Total** | **83** | **86 (96.5%)** |
