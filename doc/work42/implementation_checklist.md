# ConvoPeq ISR Bridge Runtime 改修 実装チェックリスト

**計画**: `ConvoPeq_ISR_Bridge_Runtime_改修計画書_2026-06-16.md`
**作成日**: 2026-06-16
**総タスク数**: 24ファイル（新規2、削除2、修正20）

---

## Sprint 1: 基盤整備

### B-2: ISRBarrierOptimizer 削除

- [x] B-2.1: `src/audioengine/ISRBarrierOptimizer.h` 削除
- [x] B-2.2: `src/audioengine/ISRBarrierOptimizer.cpp` 削除
- [x] B-2.3: `src/audioengine/AudioEngine.h` 101行目 `#include "ISRBarrierOptimizer.h"` 削除
- [x] B-2.4: `src/audioengine/AudioEngine.h` 3573行目 `barrierOptimizer_` メンバ削除
- [x] B-2.5: `CMakeLists.txt` 378行目 `ISRBarrierOptimizer.cpp` 登録削除
- [x] B-2.6: Debug ビルド成功確認

### A-2: Shutdown Blocking Statistics

- [x] A-2.1: `ISRShutdown.h` に `BlockingReasonStats` 構造体追加
- [x] A-2.2: `ISRShutdown.h` に `kBlockingReasonCount` 定数追加
- [x] A-2.3: `ISRShutdown.h` に `blockingReasonStats_` 配列メンバ追加
- [x] A-2.4: `ISRShutdown.h` に `reasonToString()` 関数宣言追加
- [x] A-2.5: `ISRShutdown.h` に `shutdownStartUs_` メンバ追加
- [x] A-2.6: `ISRShutdown.h` `initiateShutdown()` に shutdownStartUs_ 設定追加
- [x] A-2.7: `ISRShutdown.cpp` に `reasonToString()` 実装追加
- [x] A-2.8: `ISRShutdown.cpp` `include "core/TimeUtils.h"` 追加
- [x] A-2.9: `ISRShutdown.cpp` `markTimedOut()` に統計更新追加
- [x] A-2.10: `ISRShutdown.cpp` `emitShutdownTrace()` に BlockingReasonStats JSON出力追加（tmp+rename + リトライ + フォールバック）
- [x] A-2.11: Debug ビルド成功確認

---

## Sprint 2: 診断拡張

### A-3: Blocking Event History

- [x] A-3.1: `ISRShutdown.h` に `PackedBlockingEvent` 型エイリアス追加
- [x] A-3.2: `ISRShutdown.h` に `packEvent()` インライン関数追加
- [x] A-3.3: `ISRShutdown.h` に `TinyRingBuffer` テンプレートクラス追加
- [x] A-3.4: `ISRShutdown.h` `ShutdownRuntime` に `blockingReasonHistory_` メンバ追加
- [x] A-3.5: `ISRShutdown.cpp` `markTimedOut()` に履歴記録追加
- [x] A-3.6: Debug ビルド成功確認

### B-1: isAllZero 軽微修正

- [x] B-1.1: `RuntimeDrainAudit.h` `isAllZero()` に `routerPendingRetire` 追加
- [x] B-1.2: Debug ビルド成功確認

---

## Sprint 3: EBR 可視化

### A-1: DeferredDeletionQueue::reclaim() 戻り値変更

- [x] A-1.1: `DeferredDeletionQueue.h` `reclaim()` 戻り値 `void` → `uint32_t`
- [x] A-1.2: `reclaim()` 内で解放件数をカウントして返す
- [x] A-1.3: Debug ビルド成功確認

### A-2: EpochDomain へのカウンタ追加

- [x] A-2.1: `EpochDomain.h` に `reclaimAttemptCount_` / `reclaimSuccessCount_` 追加
- [x] A-2.2: `EpochDomain.h` に `reclaimLocalCounter_` 追加（Local Aggregation用）
- [x] A-2.3: `EpochDomain.h` `tryReclaim()` に統計カウンタ更新追加
- [x] A-2.4: `EpochDomain.h` に `reclaimAttemptCount()` / `reclaimSuccessCount()` 公開アクセサ追加
- [x] A-2.5: Debug ビルド成功確認

### A-3: IEpochProvider への virtual 追加

- [x] A-3.1: `IEpochProvider.h` に `reclaimAttemptCount()` / `reclaimSuccessCount()` virtual 追加（デフォルト0返却）
- [x] A-3.2: Debug ビルド成功確認

### A-4: ISRRetireRouter 委譲

- [x] A-4.1: `ISRRetireRouter.h` に `reclaimAttemptCount()` / `reclaimSuccessCount()` override 追加
- [x] A-4.2: Debug ビルド成功確認

### A-5: RuntimeDrainAudit フィールド追加

- [x] A-5.1: `RuntimeDrainAudit.h` に `reclaimAttemptCount` / `reclaimSuccessCount` / `overflowCount` フィールド追加
- [x] A-5.2: Debug ビルド成功確認

### A-6: collectDrainAudit() 収集追加

- [x] A-6.1: `AudioEngine.Threading.cpp` `collectDrainAudit()` に `reclaimAttemptCount` / `reclaimSuccessCount` / `overflowCount` 追加
- [x] A-6.2: Debug ビルド成功確認

### A-7: Evidence 出力追加

- [x] A-7.1: `AudioEngine.Commit.cpp` `emitEvidenceTickNonRt()` に EBR Queue 統計出力追加（epoch_reclaim_audit.json, tmp+rename）
- [x] A-7.2: Debug ビルド成功確認

---

### B: Grace Period Visibility
>
> Sprint3-A 完了が前提。A-6 で collectDrainAudit() に追加された値を Evidence 出力に自動反映するのみ。
> **新規実装不要**。A-7 で対応済み。

### C-1: RuntimePublicationOrchestrator::notifyWorldRetired() 新設

- [x] C-1.1: `RuntimePublicationOrchestrator.h` に `notifyWorldRetired()` 追加
- [x] C-1.2: Debug ビルド成功確認

### C-2: onRuntimeRetiredNonRt() からの呼び出し

- [x] C-2.1: `AudioEngine.Commit.cpp` `onRuntimeRetiredNonRt()` に `runtimeOrchestrator_->notifyWorldRetired()` 追加
- [x] C-2.2: Debug ビルド成功確認

### C-3: publishHealthSnapshot() シグネチャ変更

- [x] C-3.1: `RuntimePublicationOrchestrator.h` `publishHealthSnapshot()` シグネチャ変更（uint64_t externalReclaimedCount 追加）
- [x] C-3.2: `RuntimePublicationOrchestrator.cpp` `publishHealthSnapshot()` 実装変更（reclaimedCount を引数から設定）
- [x] C-3.3: Debug ビルド成功確認

### C-4: emitEvidenceTickNonRt() 定期呼び出し

- [x] C-4.1: `AudioEngine.Commit.cpp` `emitEvidenceTickNonRt()` に `runtimeOrchestrator_->publishHealthSnapshot(reclaimed)` 追加
- [x] C-4.2: Debug ビルド成功確認

---

### E: Async EvidenceWriter（任意、将来安全策）

- [ ] E-1: `src/audioengine/ISREvidenceWriter.h` 新規作成
- [ ] E-2: `src/audioengine/ISREvidenceWriter.cpp` 新規作成
- [ ] E-3: `AudioEngine.Commit.cpp` `emitEvidenceTickNonRt()` 非同期化
- [ ] E-4: `AudioEngine.Processing.ReleaseResources.cpp` シャットダウン時同期Flush
- [ ] E-5: `CMakeLists.txt` 新規cpp登録
- [ ] E-6: Debug ビルド成功確認

---

## Sprint 3-D: 168時間連続運転試験

- [ ] D-1: Evidence 出力確認（全ファイル）
- [ ] D-2: 5条件の自動判定スクリプト
- [ ] D-3: 条件付き確率テストスクリプト
- [ ] D-4: Early Termination 条件の自動監視
- [ ] D-5: 168h 試験実行・合格判定

---

**凡例**:

- `[ ]` = 未着手
- `[→]` = 作業中
- `[x]` = 完了
- `[-]` = 不要/スキップ
