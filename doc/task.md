# Stage B（Epoch統合）タスク分解 `task.md`

本書は `doc/detailed_design_plan4_rule4_jp.md` の `15.9`（移行統制の再設計）に基づき、
**Stage B: Epoch統合** をそのまま実行可能な依存順タスクへ分解したものである。

---

## 0. 目的

- `Epoch authority` を `EpochDomain` に一元化する。
- `dual epoch = 0` を達成する。
- `reclaimAllIgnoringEpoch` を撤去し、shutdown 停止保証を設計準拠へ移行する。

---

## 1. 事前固定（着手条件）

以下が満たされない場合、Stage B を開始しない。

- [x] Stage A 凍結が有効（新機能/UI変更/DSP最適化を停止）
- [x] 変更は ISR 移行専用ブランチで実施
- [x] `doc/runtime_causality.md` を参照し memory order 契約を固定
- [x] `doc/rule4-coding.md` の Phase 1 規約（3.1〜3.4）を順守

---

## 2. 依存順タスク（必須順序）

> 実施順は固定。**飛ばし・逆順は禁止**。

共通運用（B-1〜B-7 全タスクに適用）:

- [x] 着手時に `doc/implementation_preflight_checklist.md` の「1. 着手可否ゲート」「2. 因果・同期の安全ゲート」を確認
- [x] 完了時に同チェックリストの「4. 変更レビュー前チェック」「5. 検証手順」「6. 証跡ログ」を更新
- [x] 1タスク完了ごとに「旧経路削除が同一タスク内で閉じたか」を記録（移設のみ禁止）

### B-1: `EpochDomain` 骨格導入（最適化禁止）

対象（新規）:

- `src/core/EpochDomain.h`
- `src/core/EpochDomain.cpp`（必要なら）

作業:

- [x] reader register/unregister API
- [x] enter/leave（`recursionDepth` 対応）
- [x] retire enqueue API
- [x] reclaim API（safe 判定）
- [x] `drainAll()` API

実装拘束:

- [x] underflow assert
- [x] overflow fail-fast
- [x] `readerEpoch > retiredEpoch` の不等号契約を保持

完了条件:

- [x] `EpochDomain` 単体APIがコンパイル可能
- [x] Phase 1 規約（3.1〜3.4）に対する違反ゼロ

実行時参照（必須）:

- [x] [doc/implementation_preflight_checklist.md](implementation_preflight_checklist.md) を開き、Section 1/2 を着手前に確認
- [x] 同チェックリスト Section 6 に B-1 の証跡ログを追記

証跡ログ記入例（1行テンプレ）:

- `YYYY-MM-DD | Stage B-1 | files: src/core/EpochDomain.h,.cpp | removed-legacy: none | verify: build=OK scan=OK | risk: none | next: B-2`

---

### B-2: 旧APIアダプタ作成（短命）

対象:

- `src/core/EpochManager.h`
- `src/core/EpochCore.h`
- `src/core/RCUReader.h`
- `src/DeferredDeletionQueue.h`
- `src/core/SnapshotCoordinator.h`
- `src/core/SnapshotCoordinator.cpp`

作業:

- [x] 旧呼び出しを `EpochDomain` へ中継する暫定アダプタを作成
- [x] 呼び出し元から見たシグネチャ互換を最小維持
- [x] 変換層での独自reclaim禁止
- [x] `RCUReader` の内部依存（`EpochManager::enter/exit`）を `EpochDomain` API へ接続
- [x] `SnapshotCoordinator(EpochCore&)` 依存を `EpochDomain` 依存へ移行可能な形へ変更

実装拘束:

- [x] アダプタは短命（次タスクで全callsite移行後に削除）

完了条件:

- [x] 旧APIの実体責務が `EpochDomain` に委譲されている

実行時参照（必須）:

- [x] [doc/implementation_preflight_checklist.md](implementation_preflight_checklist.md) を開き、Section 1/2 を着手前に確認
- [x] 同チェックリスト Section 6 に B-2 の証跡ログを追記

証跡ログ記入例（1行テンプレ）:

- `YYYY-MM-DD | Stage B-2 | files: src/core/EpochManager.h,EpochCore.h,RCUReader.h | removed-legacy: adapter-only | verify: build=OK api-bridge=OK | risk: adapter TTL | next: B-3`

---

### B-3: callsite 全移設（EpochManager/EpochCore/g_deletionQueue）

重点対象:

- `src/audioengine/AudioEngine.*`
- `src/eqprocessor/EQProcessor.*`
- `src/convolver/*`
- `src/ConvolverProcessor.h`
- `src/DeferredFreeThread.h`
- `src/core/SnapshotCoordinator.*`
- `src/core/DeletionQueue.*`
- `src/RefCountedDeferred.h`
- `src/SafeStateSwapper.h`

作業:

- [x] `EpochManager::instance()` 呼び出しを全置換
- [x] `EpochCoreReaderGuard` 使用箇所を新 reader guard へ置換
- [x] `g_deletionQueue.enqueue/reclaim` を `EpochDomain` 経由へ置換
- [x] `SafeStateSwapper` の epoch 依存を `EpochDomain&` へ置換
- [x] `AudioEngine::m_epochCore` を `EpochDomain`（または同等の単一 authority）へ置換
- [x] `publishRcuEpoch / enterRcuReader / exitRcuReader` の旧 epoch 呼び出しを刷新
- [x] `SnapshotCoordinator` のローカル `DeletionQueue` 経由 reclaim を `EpochDomain` 経路へ統合

完了条件:

- [x] `EpochManager::instance(` の参照 0 件
- [x] `EpochCoreReaderGuard` の参照 0 件
- [x] `g_deletionQueue.` 参照 0 件
- [x] `#include "core/EpochManager.h"` 参照 0 件
- [x] `#include "core/EpochCore.h"` 参照 0 件

実行時参照（必須）:

- [x] [doc/implementation_preflight_checklist.md](implementation_preflight_checklist.md) を開き、Section 1/2 を着手前に確認
- [x] 同チェックリスト Section 4/5 で「旧経路削除」と grep 検証結果を記録
- [x] 同チェックリスト Section 6 に B-3 の証跡ログを追記

証跡ログ記入例（1行テンプレ）:

- `YYYY-MM-DD | Stage B-3 | files: src/audioengine/*,src/eqprocessor/* | removed-legacy: EpochManager/EpochCore/g_deletionQueue callsites | verify: grep=0 build=OK | risk: callsite漏れ | next: B-4/B-5`

---

### B-4: shutdown 停止保証プロトコルへ切替

対象:

- `src/audioengine/AudioEngine.CtorDtor.cpp`
- `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp`
- `src/audioengine/AudioEngine.Threading.cpp`

作業:

- [x] registration close
- [x] worker/rebuild/補助スレッド stop
- [x] join with timeout
- [x] active reader = 0 確認
- [x] `EpochDomain::drainAll()` 実行

削除必須:

- [x] `reclaimAllIgnoringEpoch()` 呼び出し
- [x] `reclaimAllIgnoringEpoch()` API 定義

完了条件:

- [x] shutdown 経路に epoch 無視回収が存在しない

実行時参照（必須）:

- [x] [doc/implementation_preflight_checklist.md](implementation_preflight_checklist.md) を開き、Section 1/2 を着手前に確認
- [x] 同チェックリスト Section 5 で shutdown 手順検証を実施
- [x] 同チェックリスト Section 6 に B-4 の証跡ログを追記

証跡ログ記入例（1行テンプレ）:

- `YYYY-MM-DD | Stage B-4 | files: AudioEngine.CtorDtor.cpp,ReleaseResources.cpp,Threading.cpp | removed-legacy: reclaimAllIgnoringEpoch | verify: shutdown-sequence=OK hang=none | risk: timeout tuning | next: B-5`

---

### B-5: `RefCountedDeferred` / retire 経路統合

対象:

- `src/RefCountedDeferred.h`
- retire helper 実装箇所（`AudioEngine.h` 周辺）

作業:

- [x] `RefCountedDeferred::release(EpochDomain&)` へ変更
- [x] retire authority を `EpochDomain` 経路に一本化

完了条件:

- [x] reclaim outside EpochDomain が 0 件

実行時参照（必須）:

- [x] [doc/implementation_preflight_checklist.md](implementation_preflight_checklist.md) を開き、Section 1/2 を着手前に確認
- [x] 同チェックリスト Section 4 で authority 一元化の確認を実施
- [x] 同チェックリスト Section 6 に B-5 の証跡ログを追記

証跡ログ記入例（1行テンプレ）:

- `YYYY-MM-DD | Stage B-5 | files: src/RefCountedDeferred.h,AudioEngine.h | removed-legacy: reclaim outside EpochDomain | verify: authority=single build=OK | risk: deferred path漏れ | next: B-6`

---

### B-6: 旧epoch実装の物理削除

削除対象候補:

- `src/core/EpochManager.h`
- `src/core/EpochCore.h`
- `src/DeferredDeletionQueue.h`（旧設計に依存する場合）
- `src/core/DeletionQueue.*`（責務重複分）
- `src/core/RCUReader.h`（旧 EpochManager 依存を保持する場合は実装刷新後に再評価）

作業:

- [x] 参照ゼロ確認後に削除
- [x] include 残骸削除

完了条件:

- [x] `dual epoch = 0`
- [x] 旧経路のビルド参照 0

実行時参照（必須）:

- [x] [doc/implementation_preflight_checklist.md](implementation_preflight_checklist.md) を開き、Section 1/2 を着手前に確認
- [x] 同チェックリスト Section 4/5 で削除残骸ゼロを確認
- [x] 同チェックリスト Section 6 に B-6 の証跡ログを追記

証跡ログ記入例（1行テンプレ）:

- `YYYY-MM-DD | Stage B-6 | files: src/core/EpochManager.h,EpochCore.h,DeletionQueue.* | removed-legacy: physical delete done | verify: grep=0 include=0 build=OK | risk: orphan include | next: B-7`

---

### B-7: 検証（必須ゲート）

静的確認:

- [x] `EpochManager` ヒット 0
- [x] `EpochCore` ヒット 0（新実装上必要な型名を除く）
- [x] `g_deletionQueue` ヒット 0
- [x] `reclaimAllIgnoringEpoch` ヒット 0
- [x] `SnapshotCoordinator(EpochCore&)` 署名ヒット 0
- [x] `DeletionQueue::reclaim(const EpochCore&)` 署名ヒット 0

動作確認:

- [x] build 成功（Debug/Release いずれか必須）
- [x] shutdown でクラッシュ/ハングなし
- [x] retire 後 reclaim の順序が崩れない

監査更新:

- [x] `doc/detailed_design_plan4_rule4_jp.md` の `15.2 C-03/C-08` を更新
- [x] `15.8/15.9` の Stage B 状態を更新

実行時参照（必須）:

- [x] [doc/implementation_preflight_checklist.md](implementation_preflight_checklist.md) を開き、Section 4/5 を実行
- [x] 同チェックリスト Section 7 で Go/No-Go を確定
- [x] 同チェックリスト Section 6 に B-7 の証跡ログを追記

証跡ログ記入例（1行テンプレ）:

- `YYYY-MM-DD | Stage B-7 | files: docs + verification outputs | removed-legacy: confirmed | verify: scan=OK build=OK shutdown=OK Go/No-Go=Go | risk: none | next: Stage C`

---

## 3. 依存関係マップ

- `B-1` 完了前に `B-3` 着手禁止
- `B-3` 完了前に `B-6` 着手禁止
- `B-4` は `B-3` と並行可だが、最終マージは `B-5` 完了後
- `B-7` は `B-1`〜`B-6` 完了後のみ

---

## 4. 禁止事項（Stage B期間）

- [x] 新機能追加なし
- [x] UI変更なし
- [x] DSP最適化なし
- [x] `RuntimeStore` 先行導入（Epoch統合前）なし
- [x] `PublicationLog` 先行導入（Epoch統合前）なし
- [x] 旧経路と新経路の長期併存なし

---

## 5. 受入判定（Stage B Exit Criteria）

- [x] `dual epoch = 0`
- [x] `reclaim outside EpochDomain = 0`
- [x] `reclaimAllIgnoringEpoch = 0`
- [x] shutdown 手順が `registration close -> stop -> join -> drainAll` を満たす
- [x] 監査章（15章）に Stage B 完了証跡を反映済み

## 6. 実施証跡（2026-05-17）

- `2026-05-17 | Stage B-1 | files: src/core/EpochDomain.h,src/DeferredDeletionQueue.h | removed-legacy: none | verify: scan=OK build=OK grep=OK shutdown=N/A | risk: none | next: B-2`
- `2026-05-17 | Stage B-2 | files: src/core/RCUReader.h,src/core/SnapshotCoordinator.*,src/core/DeletionQueue.* | removed-legacy: EpochCore ctor dependency | verify: scan=OK build=OK grep=OK shutdown=N/A | risk: none | next: B-3`
- `2026-05-17 | Stage B-3 | files: src/audioengine/*,src/eqprocessor/*,src/convolver/*,src/RefCountedDeferred.h | removed-legacy: EpochManager/EpochCore/g_deletionQueue callsites | verify: scan=OK build=OK grep=OK shutdown=N/A | risk: none | next: B-4`
- `2026-05-17 | Stage B-4 | files: src/audioengine/AudioEngine.CtorDtor.cpp,AudioEngine.Processing.ReleaseResources.cpp,AudioEngine.Threading.cpp | removed-legacy: reclaimAllIgnoringEpoch call path | verify: scan=OK build=OK grep=OK shutdown=OK | risk: none | next: B-5`
- `2026-05-17 | Stage B-5 | files: src/RefCountedDeferred.h,src/audioengine/AudioEngine.h | removed-legacy: reclaim outside EpochDomain | verify: scan=OK build=OK grep=OK shutdown=OK | risk: none | next: B-6`
- `2026-05-17 | Stage B-6 | files: src/core/EpochManager.h,src/core/EpochCore.h,CMakeLists.txt | removed-legacy: EpochManager物理削除 + EpochCore互換shim化 | verify: scan=OK build=OK grep=OK shutdown=OK | risk: low (shim file residual) | next: B-7`
- `2026-05-17 | Stage B-7 | files: docs + verification outputs | removed-legacy: confirmed | verify: scan=OK build=OK grep=OK shutdown=OK Go/No-Go=Go | risk: none | next: Stage C`

## 7. Stage C（observe寿命強制）初動タスク

- [x] `SnapshotCoordinator` に `ObservedSnapshot`（move-only）を導入
- [x] raw `getCurrent()` を廃止し、`observeCurrent(readerIndex)` へ置換
- [x] `AudioEngine` / `SpectrumAnalyzerComponent` の observe 呼び出しを寿命付き API に置換
- [x] `src/**` から `getCurrent(` 呼び出しをゼロ化
- [x] `Strict Atomic Dot-Call Scan` 成功
- [x] `Debug Build (cmd env retry)` 成功
- [x] `Release` ビルド成功（`Stage D release-revalidate-1` で再確認済み）

### Stage C 実施証跡（2026-05-17）

- `2026-05-17 | Stage C-1 | files: src/core/SnapshotCoordinator.h,src/core/EpochDomain.h,src/audioengine/AudioEngine.*,src/SpectrumAnalyzerComponent.cpp | removed-legacy: raw SnapshotCoordinator::getCurrent observe path | verify: scan=OK build=OK(Debug) grep(getCurrent)=0 | risk: Release task launcher instability | next: Stage C-2 (thread handoff禁止のlint/監査)`
- `2026-05-18 | Stage C-2 | files: src/core/SnapshotCoordinator.h,src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp,src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp | removed-legacy: observeCurrent(0) 直値呼び出し | verify: scan=OK build=OK(Debug) grep(observeCurrent(0))=0 | risk: thread handoff は runtime guard 検知（lintは次段） | next: Stage D 事前調査`
- `2026-05-18 | Stage D-1 | files: src/audioengine/AudioEngine.h,AudioEngine.Commit.cpp,AudioEngine.Timer.cpp | removed-legacy: reentrant commit drain path | verify: scan=OK build=OK(Debug) | risk: queue+mutex 経路自体は残存 | next: Stage D-2 (PublicationIntent log 化)`
- `2026-05-18 | Stage D-2 | files: src/audioengine/AudioEngine.h,src/audioengine/AudioEngine.Commit.cpp,src/audioengine/AudioEngine.Processing.ReleaseResources.cpp,src/audioengine/AudioEngine.CtorDtor.cpp | removed-legacy: direct queue push for deferred commit + pending log leak on shutdown | verify: scan=OK build=OK(Debug) | risk: queue drain path still exists as compatibility cleanup | next: Stage D-3 (legacy queue責務縮小)`
- `2026-05-18 | Stage D-3 | files: src/audioengine/AudioEngine.h,src/audioengine/AudioEngine.Commit.cpp,src/audioengine/AudioEngine.Processing.ReleaseResources.cpp,src/audioengine/AudioEngine.CtorDtor.cpp | removed-legacy: deferredCommitQueue/deferredCommitMutex and CommitStaging queue drain path | verify: scan=OK build=OK(Debug) | risk: PublicationLog is still vector-backed and not yet MS queue exact form | next: Stage D-4 (PublicationLog lock-free化)`
- `2026-05-18 | Stage D-4 | files: src/audioengine/AudioEngine.h,src/audioengine/AudioEngine.Commit.cpp,src/audioengine/AudioEngine.CtorDtor.cpp | removed-legacy: vector-backed PublicationLog staging + publication mutex | verify: scan=OK build=OK(Debug) | risk: final shutdown drain still uses epoch retire fallback | next: Stage D-5 (shutdown/retire polish)`
- `2026-05-18 | Stage D-5 | files: src/audioengine/AudioEngine.h,src/audioengine/AudioEngine.Commit.cpp | removed-legacy: unguarded publication append against epoch retire window | verify: scan=OK build=OK(Debug) build=NG(Release: std headers unresolved, environment issue) | risk: full Production validation requires Release env recovery | next: Stage E (PublicationCoordinator authority分離)`
- `2026-05-18 | Stage D-6 | files: src/audioengine/AudioEngine.Commit.cpp,src/audioengine/AudioEngine.CtorDtor.cpp,src/audioengine/AudioEngine.h,src/audioengine/AudioEngine.Timer.cpp | removed-legacy: direct atomic dot-call in PublicationLog path + stale deferred naming | verify: scan=OK build=OK(Debug) | risk: PublicationCoordinator class boundary is still implicit in AudioEngine | next: Stage E-1 (authority API分離)`
- `2026-05-18 | Stage E-1 | files: src/audioengine/AudioEngine.h,src/audioengine/AudioEngine.Commit.cpp,src/audioengine/AudioEngine.Processing.PrepareToPlay.cpp,src/audioengine/AudioEngine.Processing.ReleaseResources.cpp,src/audioengine/AudioEngine.Timer.cpp | removed-legacy: distributed direct runtime publish callsites outside coordinator entry | verify: scan=OK build=OK(Debug) | risk: coordinator is still an internal API, class extraction remains | next: Stage E-2 (coordinator type抽出)`
- `2026-05-18 | Stage E-1 hotfix | files: src/audioengine/AudioEngine.h,src/audioengine/AudioEngine.Processing.PrepareToPlay.cpp | removed-legacy: duplicate in-class coordinator declarations + direct prepareToPlay snapshot publish | verify: scan=OK build=OK(Debug) errors=No | risk: coordinator remains internal (not yet extracted type) | next: Stage E-2 (coordinator type抽出)`
- `2026-05-18 | Stage E-2 | files: src/audioengine/AudioEngine.h,src/audioengine/AudioEngine.Commit.cpp,src/audioengine/AudioEngine.Processing.PrepareToPlay.cpp,src/audioengine/AudioEngine.Processing.ReleaseResources.cpp,src/audioengine/AudioEngine.Timer.cpp | removed-legacy: function-level coordinator wrappers as authority surface (migrated to RuntimePublicationCoordinator type boundary) | verify: scan=OK build=OK(Debug) errors=No grep(old coordinator calls)=0 | risk: extracted type is still nested in AudioEngine (external class split deferred to next stage) | next: Stage E-3 (PublicationCoordinator外部型化/責務分割)`
- `2026-05-18 | Stage E-3 prep-1 | files: src/core/RCUReader.h,src/audioengine/AudioEngine.h,src/eqprocessor/EQProcessor.h,src/eqprocessor/EQProcessor.Core.cpp,src/eqprocessor/EQProcessor.Coefficients.cpp,src/eqprocessor/EQProcessor.Parameters.cpp | removed-legacy: EQ/AudioEngine reader-retire path の globalEpochDomain 直参照（局所） | verify: scan=OK build=OK(Debug) errors=No grep(globalEpochDomain in eqprocessor)=0 | risk: convolver 系の globalEpochDomain 直参照は未着手 | next: Stage E-3 prep-2 (convolver 側の同系統整理)`
- `2026-05-18 | Stage E-3 prep-2 | files: src/ConvolverProcessor.h,src/convolver/ConvolverProcessor.LoadPipeline.cpp,src/convolver/ConvolverProcessor.StateAndUI.cpp | removed-legacy: convolver path の globalEpochDomain 直参照（advanceEpoch） | verify: scan=OK build=OK(Debug) errors=No grep(globalEpochDomain in convolver)=0 | risk: globalEpochDomain 直参照の残件は全src再監査が必要 | next: Stage E-3 prep-3 (src 全体の残件収束)`
- `2026-05-18 | Stage E-3 prep-3 | files: src/core/RCUReader.h,src/audioengine/AudioEngine.h,src/audioengine/AudioEngine.Cache.cpp,src/RefCountedDeferred.h | removed-legacy: RCUReader global fallback + RefCountedDeferred default global release path | verify: scan=OK build=OK(Debug) errors=No grep(globalEpochDomain in src/**)=definition-only | risk: globalEpochDomain API 自体の削除は互換性確認が未完了 | next: Stage E-3 prep-4 (globalEpochDomain API の利用実態監査と段階削除可否判定)`
- `2026-05-18 | Stage E-3 prep-4 | files: src/core/EpochDomain.h,src/audioengine/AudioEngine.Globals.cpp,doc/task.md | removed-legacy: globalEpochDomain API（unused global entrypoint） | verify: scan=OK build=OK(Debug) errors=No grep(globalEpochDomain in src/**)=0 | risk: src外コード（将来拡張）で旧API参照が再導入される可能性 | next: Stage E-4 (epoch authority 監査ルールを lint/CI に固定)`
- `2026-05-18 | Stage E-4 lint-1 | files: .github/scripts/check-src-atomic-dotcall.ps1,src/DeferredDeletionQueue.h,src/RefCountedDeferred.h,src/audioengine/AudioEngine.Processing.PrepareToPlay.cpp,src/audioengine/AudioEngine.Processing.ReleaseResources.cpp | removed-legacy: helper外 compare_exchange dot-call + globalEpochDomain 再導入の見逃し | verify: scan=OK build=OK(Debug) errors=No | risk: helper外 fetch_add/fetch_sub 直呼びは別ルールで追加監査が必要 | next: Stage E-4 lint-2 (helper外 atomic操作の網羅監査)`
- `2026-05-18 | Stage E-4 lint-2 | files: .github/scripts/check-src-atomic-dotcall.ps1,src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp,src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp,src/audioengine/AudioEngine.Processing.DSPCoreIO.cpp,src/MKLNonUniformConvolver.cpp,src/NoiseShaperLearner.cpp,src/PsychoacousticDither.h,src/SafeStateSwapper.h,src/ConvolverControlPanel.cpp,src/RefCountedDeferred.h,src/core/SnapshotFactory.cpp,src/core/WorkerThread.cpp,src/ConvolverState.h | removed-legacy: helper外 fetch_add/fetch_sub dot-call（src全域） | verify: scan=OK build=OK(Debug) errors=No grep(fetch_* dot-call in src/**)=0 | risk: helper API を経由しない新規atomic導入の回帰 | next: Stage E-4 lint-3 (ルール運用の継続監査)`
- `2026-05-18 | Stage E-4 lint-3 | files: .github/scripts/check-src-atomic-dotcall.ps1,src/audioengine/AtomicAccess.h,src/audioengine/AudioEngine.h,src/audioengine/AudioEngine.Commit.cpp | removed-legacy: atomic_flag/test_and_set 直呼びの lint 抜け穴 | verify: scan=OK build=OK(Debug) errors=No grep(atomic_flag|test_and_set in src/**)=0 | risk: std::atomic wait/notify 系は condition_variable/future と誤検知衝突するため別設計が必要 | next: Stage E-5 (残存 rule4 逸脱の継続監査)`
- `2026-05-18 | Stage E-5 seqcst-1 | files: .github/scripts/check-src-atomic-dotcall.ps1,src/audioengine/AudioEngine.h,src/LockFreeRingBuffer.h,src/NoiseShaperLearner.cpp | removed-legacy: helper外 memory_order_seq_cst（安全縮小可能な箇所） | verify: scan=OK build=OK(Debug) errors=No grep(memory_order_seq_cst in src/**)=AtomicAccess.h only | risk: AtomicAccess helper default の seq_cst は API 既定値見直しを別途要精査 | next: Stage E-5 seqcst-2 (AtomicAccess既定順序のHB監査)`
- `2026-05-18 | Stage E-5 seqcst-2 | files: .github/scripts/check-src-atomic-dotcall.ps1,src/audioengine/AtomicAccess.h | removed-legacy: AtomicAccess helper default の memory_order_seq_cst | verify: scan=OK build=OK(Debug) errors=No grep(memory_order_seq_cst in src/**)=0 | risk: helper default を利用する既存 callsite の順序意味は継続監査が必要 | next: Stage E-5 seqcst-3 (default依存callsiteの明示順序化)`
- `2026-05-18 | Stage E-5 seqcst-3 batch-1 | files: src/ConvolverProcessor.h,src/convolver/ConvolverProcessor.LoadPipeline.cpp,src/convolver/ConvolverProcessor.LoaderThread.cpp,src/convolver/ConvolverProcessor.Lifecycle.cpp | removed-legacy: default-order依存 helper 呼び出し（Convolver周辺） | verify: scan=OK build=OK(Debug) errors=No | risk: 他モジュールの default-order 依存は継続監査が必要 | next: Stage E-5 seqcst-3 batch-2 (AudioEngine周辺の順序明示化)`
- `2026-05-18 | Stage E-5 seqcst-3 batch-2 | files: src/audioengine/AudioEngine.Init.cpp,src/audioengine/AudioEngine.Timer.cpp | removed-legacy: default-order依存 helper 呼び出し（AudioEngine初期化/クロスフェード完了通知） | verify: scan=OK build=OK(Debug) errors=No | risk: default-order 依存の残件は全src横断で継続削減が必要 | next: Stage E-5 seqcst-3 batch-3 (残存default-order依存の横断抽出と明示化)`
- `2026-05-18 | Stage E-5 seqcst-3 batch-3 | files: src/NoiseShaperLearnerTypes.h,src/NoiseShaperLearner.h,src/NoiseShaperLearner.cpp | removed-legacy: default-order依存 helper 呼び出し（NoiseShaperLearner系） | verify: scan=OK build=OK(Debug) errors=No residual(default-order helper calls)=173 | risk: 依存残件が多いためモジュール別に段階解消が必要 | next: Stage E-5 seqcst-3 batch-4 (AudioEngine.h/Parameters系の明示化)`
- `2026-05-18 | Stage E-5 seqcst-3 batch-4 | files: src/audioengine/AudioEngine.Parameters.cpp,src/audioengine/AudioEngine.StateIO.cpp | removed-legacy: default-order依存 helper 呼び出し（AudioEngine Parameters/StateIO） | verify: scan=OK build=OK(Debug) errors=No residual(default-order helper calls)=81 | risk: AudioEngine.h inline/helper周辺に残件集中 | next: Stage E-5 seqcst-3 batch-5 (AudioEngine.hの順序明示化)`
- `2026-05-18 | Stage E-5 seqcst-3 batch-5 | files: src/audioengine/AudioEngine.Commit.cpp,src/audioengine/AudioEngine.Processing.PrepareToPlay.cpp,src/audioengine/AudioEngine.Processing.ReleaseResources.cpp,src/audioengine/AudioEngine.Processing.Latency.cpp,src/audioengine/AudioEngine.RebuildDispatch.cpp | removed-legacy: default-order依存 helper 呼び出し（commit/prepare/release/latency/rebuild dispatch） | verify: scan=OK build=OK(Debug) errors=No | risk: header inline経路に default-order 依存が残存 | next: Stage E-5 seqcst-3 batch-6 (AudioEngine.h + 周辺ヘッダ仕上げ)`
- `2026-05-18 | Stage E-5 seqcst-3 batch-6 | files: src/audioengine/AudioEngine.h,src/PsychoacousticDither.h | removed-legacy: AudioEngine.h inline/default-order依存 + PsychoacousticDither instance counter default order | verify: scan=OK build=OK(Debug) errors=No residual(non-memory_order regex hits)=19 | risk: 残件19は order転送ラッパー中心（regex偽陽性） | next: Stage E-5 seqcst-3 batch-7 (残件の真偽分類とlint指標整合)`
- `2026-05-18 | Stage E-5 seqcst-3 batch-7 | files: src/audioengine/AudioEngine.h,src/ConvolverProcessor.h,src/eqprocessor/EQProcessor.h | removed-legacy: order転送ラッパー呼び出しの監査ノイズ（memory_order明示表記へ正規化） | verify: scan=OK build=OK(Debug) errors=No residual(non-memory_order regex hits)=0 | risk: low（意味変更なしの表記正規化） | next: Stage E-5 exit-check (rule4監査ログ最終化)`
- `2026-05-18 | Stage E-5 exit-check | files: src/core/SnapshotCoordinator.h,src/core/SnapshotCoordinator.cpp,src/audioengine/AudioEngine.Timer.cpp,doc/task.md,doc/implementation_preflight_checklist.md | removed-legacy: completion side-channel atomic (m_fadeCompleted) | verify: scan=OK build=OK(Debug) errors=No grep(m_fadeCompleted)=0 grep(seq_cst)=0 grep(old-epoch-path)=0 | risk: low（tryCompleteFadeはstate+remaining CASで一回性維持） | next: Stage F 準備（RuntimeState統合の未着手項目を棚卸し）`
- `2026-05-18 | Stage E-5 exit-check-2 | files: src/core/SnapshotCoordinator.h,src/core/SnapshotCoordinator.cpp | removed-legacy: abortFade symbol + abort direct-destroy path（retire queue経由へ統一） | verify: scan=OK build=OK(Debug) errors=No grep(abortFade)=0 grep(m_fadeCompleted)=0 | risk: low（destructor/abort系の解放はDeletionQueue経路で一元化） | next: Stage F 準備（C-01/C-02/C-06主軸の移行計画を具体化）`
- `2026-05-18 | Stage F prep inventory | files: src/audioengine/AudioEngine.h,src/audioengine/AudioEngine.Commit.cpp,src/audioengine/AudioEngine.CtorDtor.cpp,src/audioengine/AudioEngine.Processing.ReleaseResources.cpp | removed-legacy: deferredCommitQueue/deferredCommitMutex/CommitStaging（残存なし確認） | verify: grep(RuntimeStore/PublicationCoordinator/RuntimeState class)=none grep(multi-sync-plane: currentDSPBits/fadingOutDSPBits/runtimePublishWorldState)=present | risk: C-01/C-02 は未収束（単一publish unit化とRuntimeStore導入が未着手） | next: Stage F batch-1（RuntimeStore最小導入とpublish authority閉鎖）`
- `2026-05-18 | Stage F batch-1 | files: src/core/RuntimeStore.h,src/audioengine/AudioEngine.h | removed-legacy: runtimePublishWorldState atomic直保持（RuntimeStoreへ置換） | verify: scan=OK build=OK(Debug) errors=No grep(runtimePublishWorldState)=0 grep(RuntimeStore)=present | risk: publish authority閉鎖はAudioEngine内限定（PublicationCoordinator専有化は次段） | next: Stage F batch-2（RuntimeStore publish API の権限制御を導入）`
- `2026-05-18 | Stage F batch-2 | files: src/core/RuntimeStore.h,src/audioengine/AudioEngine.h | removed-legacy: RuntimeStore publish API の無制限公開（owner-scope化） | verify: scan=OK build=OK(Debug) errors=No grep(friend Owner)=present grep(RuntimeStore<RuntimePublishWorld, AudioEngine>)=present | risk: PublicationCoordinator 専有化は未導入（現状owner=AudioEngine） | next: Stage F batch-3（RuntimeStore API を observe専用公開 + publish経路の更なる閉鎖）`
- `2026-05-18 | Stage F batch-3 | files: src/audioengine/AudioEngine.h | removed-legacy: RuntimeStore write API の散在呼び出し（publish/clearゲートへ集約） | verify: scan=OK build=OK(Debug) errors=No grep(runtimeStore.publishAndSwap|clearAndSwapNull)=helper内のみ | risk: owner=AudioEngine のため書込み権限は依然広い | next: Stage F batch-4（PublicationCoordinator専有化に向けたowner分離準備）`
- `2026-05-18 | Stage F batch-4 | files: src/audioengine/AudioEngine.h | removed-legacy: AudioEngine本体からのRuntimeStore write直アクセス（RuntimePublicationCoordinatorへ移管） | verify: scan=OK build=OK(Debug) errors=No grep(runtimeStore.publishAndSwap|clearAndSwapNull)=RuntimePublicationCoordinator内のみ | risk: core/PublicationCoordinator への外部型分離は未着手 | next: Stage F batch-5（PublicationCoordinator外部型導入の前段リファクタ）`
- `2026-05-18 | Stage F batch-5 | files: src/audioengine/AudioEngine.h | removed-legacy: RuntimeStore owner=AudioEngine の型権限（owner を RuntimePublicationCoordinator へ縮小） | verify: scan=OK build=OK(Debug) errors=No grep(RuntimeStore<RuntimePublishWorld, RuntimePublicationCoordinator>)=present grep(runtimeStore.publishAndSwap|clearAndSwapNull)=RuntimePublicationCoordinator内のみ | risk: coordinator は依然 nested class（外部分離未着手） | next: Stage F batch-6（PublicationCoordinator の外部型化検討）`
- `2026-05-18 | Stage F batch-6 | files: src/audioengine/AudioEngine.h, src/audioengine/AudioEngine.CtorDtor.cpp, src/audioengine/AudioEngine.Processing.ReleaseResources.cpp | removed-legacy: AudioEngine本体に残っていた runtime publish/clear 実装（coordinator へ実体移管） | verify: scan=OK build=OK(Debug) errors=No grep(publishRuntimeSnapshots|clearPublishedRuntimeSnapshotsNonRt)=RuntimePublicationCoordinator経路のみ | risk: coordinator 外部型化（nested解除）は未着手 | next: Stage F batch-7（coordinator 外部型化の依存切り出し）`
- `2026-05-18 | Stage F batch-7a | files: src/audioengine/AudioEngine.h | removed-legacy: coordinator 内での owner atomic helper 依存（owner.fetchAddAtomic / owner.publishAtomic） | verify: scan=OK build=OK(Debug) errors=No grep(owner.fetchAddAtomic|owner.publishAtomic(owner.runtimeGraphRevision))=0 | risk: coordinator 自体は依然 nested class | next: Stage F batch-7b（nested解除可否の型依存整理）`
- `2026-05-18 | Stage F batch-7b | files: src/audioengine/AudioEngine.h | removed-legacy: RuntimePublishWorld の AudioEngine ネスト依存（header-scope へ切り出し） | verify: scan=OK build=OK(Debug) errors=No grep(struct RuntimePublishWorld)=header-scope grep(AudioEngine::RuntimePublishWorld)=0 | risk: coordinator の nested 依存（DSPCore/owner私有API）は残存 | next: Stage F batch-7c（coordinator 外部型化の最小ブリッジ設計）`
- `2026-05-18 | Stage F batch-7c | files: src/audioengine/AudioEngine.h | removed-legacy: coordinator から owner内部実装への直接依存の一部（generation/version/retire/revision reset を bridge API 化） | verify: scan=OK build=OK(Debug) errors=No grep(reserveNextRuntimeGraphGeneration|reserveNextRuntimeVersion|retireRuntimePublishWorldNonRt|resetRuntimeGraphRevisionNonRt)=present | risk: RuntimeStore write 呼び出しは friend 制約上 coordinator 内直呼びを維持 | next: Stage F batch-8（coordinator 外部型化の実装可否を最小PoCで検証）`
- `2026-05-18 | Stage F batch-8 stabilize-1 | files: src/audioengine/AudioEngine.h,src/core/RuntimeStore.h | removed-legacy: 外部型化PoC由来の不完全型依存（RuntimeStore owner 前方宣言不足） | verify: scan=OK build=OK(Debug) errors=No grep(runtimeStore.write path)=RuntimePublicationCoordinator内のみ | risk: coordinator 外部型化は未再開（現時点は nested 構成で安定化） | next: Stage F batch-8 stabilize-2（外部型化再挑戦前の依存境界整理）`
- `2026-05-18 | Stage F batch-8 stabilize-2 | files: src/audioengine/AudioEngine.h | removed-legacy: RuntimePublicationCoordinator の低レベル swap API 公開面（public） | verify: scan=OK build=OK(Debug Build cmd env retry) errors=No grep(runtimeStore.publishAndSwap|clearAndSwapNull)=RuntimePublicationCoordinator private内のみ | risk: nested coordinator の外部型化は未着手 | next: Stage F batch-8 stabilize-3（外部型化前の owner 依存面の更なる局所化）`
- `2026-05-18 | Stage F batch-8 stabilize-3 | files: src/audioengine/AudioEngine.h | added: buildRuntimePublishWorld bridge API（reserveNextRuntimeGraphGeneration/makeEngineRuntimeState/makeRuntimeGraphState/reserveNextRuntimeVersion を内部化）| removed-legacy: publishRuntimeSnapshots 内の coordinator→owner 4 直接呼び出し（→bridge 1本に集約）| verify: scan=OK build=OK(Debug Build cmd env retry) errors=No | risk: coordinator の owner 依存は publishRuntimeTransitionState/publishCurrentDSPAndTakeOwnership/retireRuntimePublishWorldNonRt/resetRuntimeGraphRevisionNonRt が残存 | next: Stage F batch-8 stabilize-4 または batch-9（coordinator 外部型化）`
- `2026-05-18 | Stage F batch-8 stabilize-4 | files: src/audioengine/AudioEngine.h | added: retireAndResetRuntimePublishWorldNonRt bridge（retireRuntimePublishWorldNonRt + resetRuntimeGraphRevisionNonRt を 1本に集約）| removed-legacy: clearPublishedRuntimeSnapshotsNonRt 内の owner 直接呼び出し 2本→bridge 1本に簡略化; 冗長な friend class RuntimePublicationCoordinator 宣言を削除（nested class は C++11 で自動アクセス権）| verify: scan=OK build=OK(Debug Build cmd env retry) errors=No | risk: coordinator の owner 依存は publishRuntimeTransitionState/publishCurrentDSPAndTakeOwnership が残存 | next: Stage F batch-8 stabilize-5（adoptAndPublish/publishState の owner 依存を bridge 化してさらに局所化）`
- `2026-05-18 | Stage F batch-8 stabilize-5 | files: src/audioengine/AudioEngine.h | added: adoptDSPAndPublishTransitionState bridge（publishCurrentDSPAndTakeOwnership + publishRuntimeTransitionState を 1本に集約）| removed-legacy: adoptAndPublish 内の owner 直接呼び出し 2本（publishCurrentDSPAndTakeOwnership + publishState→publishRuntimeTransitionState）→ bridge 1本 + publishRuntimeSnapshots 直呼びに簡略化 | 順序: publishCurrentDSP は world swap 前を維持（RT安全）、publishRuntimeTransitionState(デバッグ atomic のみ)は swap 前移動してもRT安全 | verify: scan=OK build=OK(Debug Build cmd env retry) errors=No | risk: coordinator の owner 依存は publishRuntimeSnapshots 内の buildRuntimePublishWorld/retireRuntimePublishWorldNonRt が残存（swap をブラケットする不可分依存） | next: Stage F batch-9（coordinator 外部型化に向けた bridge API グループ化・インターフェイス設計）`
- `2026-05-18 | Stage F batch-9a | files: src/audioengine/AudioEngine.h | changed: publishRuntimeSnapshots を public→private 移動（publishState/adoptAndPublish の実装詳細であり外部から呼ばれない）| coordinator public API: 3メソッド（clearPublishedRuntimeSnapshotsNonRt/publishState/adoptAndPublish）に整理 | verify: errors=No scan=OK build=OK(CMakeTools Debug) | risk: なし（可視性変更のみ、挙動変化なし）| next: Stage F batch-9b（coordinator bridge 依存を IRuntimePublicationBridge interface に抽象化）`
- `2026-05-18 | Stage F batch-9b | files: src/audioengine/AudioEngine.h | added: buildWorldAndPublishTransition bridge（publishRuntimeTransitionState + buildRuntimePublishWorld を 1本に集約）; adoptAndBuildPublishWorld bridge（adoptDSPAndPublishTransitionState + buildRuntimePublishWorld を 1本に集約）| changed: publishState → buildWorldAndPublishTransition + fence + swap + retire の一様パターンに変更; adoptAndPublish → adoptAndBuildPublishWorld + fence + swap + retire の一様パターンに変更 | removed: private publishRuntimeSnapshots（各メソッドに直接展開、共有不要となった）| coordinator private: publishRuntimeWorldAndSwap + clearRuntimeWorldAndSwapNull + owner のみ | verify: errors=No scan=OK build=OK(CMakeTools Debug) | risk: publishRuntimeTransitionState の実行タイミングが swap 後→swap 前に変化したが、デバッグ atomic のみのため RT 安全 | next: Stage F batch-9c（coordinator の AudioEngine& 依存を interface 型に置換し nested class からの外部型化）`
- `2026-05-18 | Stage F batch-9c | files: src/audioengine/AudioEngine.h | added: //=== RuntimePublicationCoordinator Bridge API (NonRT-only) ===// と //=== End RuntimePublicationCoordinator Bridge API ===// セクションコメントで bridge API 領域を明示的に括る（retireRuntimePublishWorldNonRt, resetRuntimeGraphRevisionNonRt, retireAndResetRuntimePublishWorldNonRt, adoptDSPAndPublishTransitionState, buildRuntimePublishWorld, buildWorldAndPublishTransition, adoptAndBuildPublishWorld を対象）| fix: patch 初期適用でコメントが adoptAndBuildPublishWorld 関数内部に挿入され return 文が欠落し C4716 エラー → 修正で return buildRuntimePublishWorld(5引数) 復元・コメントを関数外に移動（C2660 も修正: latencyDeltaSamples 引数過剰を除去）| verify: errors=No scan=OK build=OK(Debug) | risk: コメント追加・return 文復元のみ、挙動変化なし | note: buildRuntimePublishWorld は 5引数(current,next,policy,fadeTimeSec,active) のみ — latencyDeltaSamples は渡さない | next: Stage F batch-9d`
- `2026-05-18 | Stage F batch-9d | files: src/audioengine/AudioEngine.h,doc/task.md | changed: RuntimePublicationCoordinator を bridge adapter 内包型へ整合化（BridgeImpl + IRuntimePublicationBridge& + owner_）し、owner/owner_ 混在と未置換参照を解消 | fixed: clear/publish/adopt の bridge 呼び出しを bridge_.X() に統一、runtimeStore write は owner_.runtimeStore のみに限定、publicationCoordinator() 返却経路を存置して lifetime 安全を確保 | verify: errors=No scan=OK build=OK(Debug Build cmd env retry) | risk: coordinator の core 外部型化は未着手（nested構成は維持） | next: Stage F batch-9e（外部型化可否の再評価）`
- `2026-05-18 | Stage F batch-9e | files: src/audioengine/AudioEngine.h,doc/task.md | changed: RuntimePublicationCoordinator の保持参照を owner_ から runtimeStore_ へ縮退（BridgeImpl + bridge_ + runtimeStore_）し、AudioEngine 生参照依存を更に縮小 | fixed: publishRuntimeWorldAndSwap/clearRuntimeWorldAndSwapNull のアクセスを runtimeStore_ 直参照へ統一、constructor 初期化を runtimeStore_(ownerIn.runtimeStore) に変更 | verify: errors=No scan=OK build=OK(Debug Build cmd env retry) | risk: coordinator の core 外部型化は依然未着手（DSPCore 型境界と friend Owner 制約の整理が次段） | next: Stage F batch-9f（外部型化PoCの再挑戦）`
- `2026-05-18 | Stage F batch-9f | files: src/audioengine/AudioEngine.h,doc/task.md | changed: RuntimeStore テンプレート型を RuntimePublishStore alias 化し、AudioEngine/coordinator 間の型境界を簡素化（runtimeStore メンバ型・coordinator 内参照型を統一） | fixed: template 直書きの重複を除去して外部型化前の依存面を軽量化 | verify: errors=No scan=OK build=OK(Debug Build cmd env retry) | risk: 外部型化本体は未着手（friend Owner 制約と DSPCore 境界の分離設計が次段） | next: Stage F batch-10（coordinator 外部型化PoCの設計固定）`
- `2026-05-18 | Stage F batch-10a | files: src/audioengine/AudioEngine.h,doc/task.md | changed: RuntimePublicationCoordinator を move-only 化（copy ctor/assign を削除、move ctor/assign を許可）し、publish 経路での accidental copy を防止 | fixed: publicationCoordinator() の返却を brace 初期化へ統一して move 前提を明示 | verify: errors=No scan=OK build=OK(Debug Build cmd env retry) | risk: 外部型化本体は未着手（friend Owner 制約と DSPCore 境界は次段） | next: Stage F batch-10b（coordinator ctor の注入境界を AudioEngine から更に分離）`
- `2026-05-18 | Stage F batch-10b | files: src/audioengine/AudioEngine.h,doc/task.md | changed: RuntimePublicationCoordinator ctor を AudioEngine 直受けから BridgeImpl + RuntimePublishStore 注入へ変更し、生成境界を分離 | fixed: publicationCoordinator() を RuntimePublicationCoordinator{BridgeImpl{*this}, runtimeStore} へ更新し、外部型化PoC向けの注入形を確立 | verify: errors=No scan=OK build=OK(Debug Build cmd env retry) | risk: BridgeImpl は依然 AudioEngine private bridge API へ依存 | next: Stage F batch-10c（注入時の無駄コピー削減）`
- `2026-05-18 | Stage F batch-10c | files: src/audioengine/AudioEngine.h,doc/task.md | changed: ctor引数を BridgeImpl&& に変更し bridgeImpl_ を std::move で構築、<utility> を明示 include | fixed: publicationCoordinator 経路の一時 BridgeImpl コピーを削減し所有権意図を明確化 | verify: errors=No scan=OK build=OK(Debug Build cmd env retry) | risk: 外部型化本体は未着手（friend Owner 制約と DSPCore 境界の分離は次段） | next: Stage F batch-11（RuntimeStore owner 制約を維持した外部型化 PoC の最小設計）`
- `2026-05-18 | Stage F batch-11a | files: src/audioengine/AudioEngine.h,doc/task.md | changed: RuntimePublicationCoordinator::Dependencies を導入し、bridge/store 注入を1束へ集約（ctor を Dependencies&& 受けへ変更） | fixed: makePublicationCoordinatorDependencies() を追加して publicationCoordinator() の依存生成を1箇所へ集中、将来の外部型化時に差し替え点を最小化 | verify: errors=No scan=OK build=OK(Debug Build cmd env retry) | risk: friend Owner 制約・DSPCore 型境界は依然 nested 構成依存 | next: Stage F batch-11b（外部型化PoCの境界定義を task/spec に固定）`
- `2026-05-18 | Stage F batch-11b | files: src/audioengine/AudioEngine.h,doc/task.md | changed: RuntimePublicationCoordinator::Dependencies を move-only 契約へ強化（copy禁止、move許可、明示 ctor で BridgeImpl&& + RuntimePublishStore& を受理） | fixed: 依存束の accidental copy を禁止し、外部型化PoCでの注入境界を型レベルで固定 | verify: errors=No scan=OK build=OK(Debug Build cmd env retry) | risk: coordinator 本体の external class 化は未着手（friend Owner と DSPCore 境界は次段） | next: Stage F batch-11c（spec側へ境界契約を追記）`
- `2026-05-18 | Stage F batch-11c | files: src/audioengine/AudioEngine.h,doc/task.md | changed: move-only 契約を static_assert で固定（Dependencies / RuntimePublicationCoordinator の copy constructibility を禁止） | fixed: 将来変更で copy 経路が復活した場合にコンパイル時に即検出できるよう境界を強化 | verify: errors=No scan=OK build=OK(Debug Build cmd env retry) | risk: external class 化本体は未着手（friend Owner 制約と DSPCore 境界は次段） | next: Stage F batch-12（外部型化PoCの最小スケルトン検討）`
- `2026-05-18 | Stage F batch-11d | files: src/audioengine/AudioEngine.h,doc/task.md | fix: RuntimePublicationCoordinator 内部に置いた is_copy_constructible static_assert が未完了型参照となり C2139 を発生 → クラス定義直後（完了型）へ移動して回復 | verify: errors=No scan=OK build=OK(Debug Build cmd env retry) | risk: 低（配置修正のみ、契約内容は不変） | next: Stage F batch-12（外部型化PoCの最小スケルトン検討）`
- `2026-05-18 | Stage F batch-12a | files: src/core/RuntimeStore.h,src/audioengine/AudioEngine.h,doc/task.md | changed: RuntimeStore に owner専用 WriteAccess ハンドル（acquireWriteAccess + publishAndSwapImpl/clearAndSwapNullImpl）を導入し、coordinator 側の RuntimeStore private API 直呼びを排除 | changed: RuntimePublicationCoordinator::Dependencies/内部保持を RuntimePublishStore& から RuntimePublishWriteAccess へ移行 | verify: errors=No scan=OK build=OK(Debug Build cmd env retry) | risk: friend Owner 制約は維持（外部型化本体は次段） | next: Stage F batch-12b（PoCスケルトンに向けた最小境界分離）`
- `2026-05-18 | Stage F batch-12b | files: src/audioengine/AudioEngine.h,doc/task.md | changed: coordinator/bridge 境界に DspCoreHandle alias（=DSPCore*）を導入し、IRuntimePublicationBridge / BridgeImpl / RuntimePublicationCoordinator のシグネチャを型境界経由に統一 | verify: errors=No scan=OK build=OK(Debug Build cmd env retry) | risk: 低（型エイリアス化のみ、挙動不変） | next: Stage F batch-12c（write権限ハンドル契約の強化）`
- `2026-05-18 | Stage F batch-12c | files: src/core/RuntimeStore.h,doc/task.md | changed: RuntimeStore::WriteAccess を copy禁止・move-only 化し、write authority の偶発コピー経路を遮断 | verify: errors=No scan=OK build=OK(Debug Build cmd env retry) | risk: 低（権限制約の強化のみ、実行時挙動不変） | next: Stage F batch-12d（Dependencies 依存面の更なる薄化候補を選定）`
- `2026-05-18 | Stage F batch-12d | files: src/audioengine/AudioEngine.h,doc/task.md | changed: RuntimePublicationCoordinator の IRuntimePublicationBridge& bridge_ 参照メンバを削除し、bridgeImpl_ を直接呼ぶように変更。= default move ctor が参照メンバ経由で moved-from オブジェクトを指す latent dangling-reference bug を排除。IRuntimePublicationBridge インターフェイスは BridgeImpl の契約ドキュメントとして残存 | verify: errors=No scan=OK build=OK(Debug Build cmd env retry) | risk: 低（呼び出し経路の短縮のみ、挙動不変） | next: Stage F batch-13（coordinator external class 化の準備 or Stage G RuntimeState 統合）`
- `2026-05-18 | Stage F batch-13 | files: src/audioengine/AudioEngine.h,doc/task.md | changed: BridgeImpl から IRuntimePublicationBridge 継承を除去（: IRuntimePublicationBridge → 削除、override × 4 削除）。batch-12d で bridge_（抽象 ref）が削除されたことで vtable は dead code となっていた。IRuntimePublicationBridge は外部型化時の DI 仕様文書として残存 | verify: errors=No scan=OK build=OK(Debug Build cmd env retry) | risk: 低（vtable削除のみ、実行時挙動不変・呼び出し元なし確認済み） | next: Stage F batch-14 候補選定`
- `2026-05-18 | Stage F batch-14 | files: src/audioengine/AudioEngine.h,doc/task.md | changed: makePublicationCoordinatorDependencies() を publicationCoordinator() にインライン化して削除（1対1ヘルパー）。IRuntimePublicationBridge ブロックのコメントを post-batch-12d/13 の実態（vtable不使用・DI仕様文書として残存）に合わせ更新 | verify: errors=No scan=OK build=OK(Debug Build cmd env retry) | risk: 低（ヘルパー削除・コメント更新のみ、挙動不変） | next: Stage F batch-15 候補選定`
- `2026-05-18 | Stage F batch-15 | files: src/audioengine/AudioEngine.h,doc/task.md | changed: Bridge API セクション冒頭コメント（"IRuntimePublicationBridge の具体実装。各メソッドは stage F の bridge API を呼ぶ override。"）を batch-13 の vtable 除去後の実態（AudioEngine 内部 bridge API 実装・BridgeImpl が仲介）に合わせ修正 | verify: errors=No scan=OK build=OK(Debug Build cmd env retry) | risk: なし（コメントのみ） | next: Stage F batch-16 候補選定`
- `2026-05-18 | Stage F batch-16 | files: src/audioengine/AudioEngine.h,doc/task.md | changed: (1) resetRuntimeGraphRevisionNonRt() を retireAndResetRuntimePublishWorldNonRt() にインライン化して削除（1対1）。(2) adoptDSPAndPublishTransitionState() を adoptAndBuildPublishWorld() にインライン化して削除（1対1）。(3) BridgeImpl 先頭コメントを実態（IRuntimePublicationBridge 継承なし・シム）に合わせ更新 | verify: errors=No scan=OK build=OK(Debug Build cmd env retry) | risk: 低（インライン化・コメント更新のみ、挙動不変） | next: Stage F batch-17 候補選定`
- `2026-05-18 | Stage F batch-17 | files: src/audioengine/AudioEngine.h,doc/task.md | changed: coordinator::adoptAndPublish() 内コメントで batch-16 で削除した adoptDSPAndPublishTransitionState への参照を除去し現状記述に更新。同メソッドのパラメータリスト不揃いインデントも修正 | verify: errors=No scan=OK build=OK(Debug Build cmd env retry) | risk: なし（コメント・整形のみ） | next: Stage F batch-18 候補選定`
- `2026-05-18 | Stage F batch-18 | files: src/audioengine/AudioEngine.h,doc/task.md | changed: IRuntimePublicationBridge 内 buildWorldAndPublishTransition / adoptAndBuildPublishWorld の 2 メソッドのパラメータリスト不揃いインデントを修正（ブレース開き位置に合わせて揃え） | verify: errors=No scan=OK build=OK(Debug Build cmd env retry) | risk: なし（整形のみ） | next: Stage F batch-19 候補選定`
- `2026-05-18 | Stage F batch-19 | files: src/audioengine/AudioEngine.h,doc/task.md | changed: retireAndResetRuntimePublishWorldNonRt の batch-8 stabilize-3/4 + batch-16 の積層歴史コメントを現状機能説明に集約。adoptAndBuildPublishWorld の末尾 batch-16 歴史注釈を除去し、先頭コメントに統合 | verify: errors=No scan=OK build=OK(Debug Build cmd env retry) | risk: なし（コメント整理のみ） | next: Stage F batch-20 候補選定`
- `2026-05-18 | Stage F batch-20 | files: src/audioengine/AudioEngine.h,doc/task.md | changed: coordinator 内 clearPublishedRuntimeSnapshotsNonRt / publishState / adoptAndPublish の batch 歴史コメントを現状説明に整理。publishState の 2 番目パラメータのインデント不揃いも修正 | verify: errors=No scan=OK build=OK(Debug Build cmd env retry) | risk: なし（コメント・整形のみ） | next: Stage F batch-21 候補選定`
- `2026-05-18 | Stage F batch-21 | files: src/audioengine/AudioEngine.h,doc/task.md | changed: buildWorldAndPublishTransition の先頭コメントから Stage F batch-9b: 歴史参照を除去し現状の 1 行説明に整理 | verify: errors=No scan=OK build=OK(Debug Build cmd env retry) | risk: なし（コメント整理のみ） | next: Stage F batch-22 候補選定`
- `2026-05-18 | Stage F batch-22 | files: src/audioengine/AudioEngine.h,doc/task.md | changed: BridgeImpl 内 buildWorldAndPublishTransition / adoptAndBuildPublishWorld の 2 メソッドのパラメータ不揃いインデントを修正 | verify: errors=No scan=OK build=OK(Debug Build cmd env retry) | risk: なし（整形のみ） | next: Stage F batch-23 候補選定`
- `2026-05-18 | Stage F batch-23 | files: src/audioengine/AudioEngine.h,doc/task.md | changed: RuntimePublicationCoordinator ctor の Stage F batch-11a: 歴史参照を除去し 1 行の現状説明に整理。src/ 内の Stage F batch-* コメントがすべて除去されゼロになった | verify: errors=No scan=OK build=OK(Debug Build cmd env retry) | risk: なし（コメント整理のみ） | next: Stage F batch-24 候補選定（AudioEngine.h 外のクリーンアップまたは Stage G 準備）`
- `2026-05-18 | Stage F batch-24 | files: src/audioengine/AudioEngine.h,doc/task.md | changed: IRuntimePublicationBridge コメントから batch-12d/batch-13 歴史参照を除去し現状説明に整理。BridgeImpl コメントも同様に整理。src/ 内の batch-* 歴史参照コメントがゼロになった | verify: errors=No scan=OK build=OK(Debug Build cmd env retry) | risk: なし（コメント整理のみ） | next: Stage G 準備または AudioEngine.h 外の残存整理`
- `2026-05-18 | Stage F batch-25 | files: src/audioengine/AudioEngine.h,doc/task.md | changed: toDspBits の二重キャスト（outer static_cast<uintptr_t>）と DSP ビットヘルパー/static wrapper 11 箇所の冗長 static_cast<std::memory_order>(order) を除去 | verify: errors=No scan=OK build=OK(Debug Build cmd env retry) | risk: なし（挙動不変の冗長キャスト除去） | next: Stage G 準備（currentDSPBits/fadingOutDSPBits の RuntimePublishWorld 包含調査）`
- `2026-05-18 | Stage F batch-26 | files: src/audioengine/AudioEngine.h,doc/task.md | changed: DSP ビットヘルパー 8 関数（toDspBits/fromDspBits/loadCurrentDSP/exchangeCurrentDSP/publishCurrentDSP/loadFadingOutDSP/exchangeFadingOutDSP/publishFadingOutDSP）のインデント不揃い（8→4 スペース）を修正 | verify: errors=No scan=OK build=OK(Debug Build cmd env retry) | risk: なし（整形のみ） | next: AudioEngine.h 内の他の不揃い調査または Stage G 候補選定`
- `2026-05-18 | Stage F batch-27 | files: src/audioengine/AudioEngine.h,src/ConvolverControlPanel.cpp,doc/task.md | changed: rule2 Phase 1: プレフィックス 2 箇所を除去し、設計意図コメントのみ残す | verify: errors=No scan=OK build=OK(Debug Build cmd env retry) | risk: なし（コメント整形のみ） | next: src/ 内の残存プラン参照コメント調査`
- `2026-05-18 | Stage F batch-28 | files: src/audioengine/AudioEngine.h,doc/task.md | changed: A-6 Phase 1: プレフィックス 1 箇所（AudioEngine.h L1781）を除去し、設計意図コメントのみ残す | verify: errors=No scan=OK build=OK(Debug Build cmd env retry) | risk: なし（コメント整形のみ） | next: src/ 内の残存プラン参照コメント追加調査`
- `2026-05-18 | Stage F batch-29 | files: src/audioengine/AudioEngine.h,src/audioengine/AudioEngine.RebuildDispatch.cpp,doc/task.md | changed: A-12 first step / A-14 の歴史参照プレフィックス 2 箇所を除去し、設計意図コメントのみ残す | verify: errors=No scan=OK build=OK(Debug Build cmd env retry) grep(A-12|A-14)=0 | risk: なし（コメント整形のみ） | next: src/ 内の残存 A/B/F 系履歴参照の精査または Stage G 候補選定`
- `2026-05-18 | Stage F batch-30 | files: src/ConvolverProcessor.h,src/convolver/ConvolverProcessor.Runtime.cpp,src/convolver/ConvolverProcessor.Lifecycle.cpp,doc/task.md | changed: [F-01 fix] 履歴ラベル 9 箇所を除去し、世代カウンター設計意図コメントへ正規化 | verify: errors=No scan=OK build=OK(Debug Build cmd env retry) grep(F-01 fix)=0 | risk: なし（コメント整形のみ） | next: src/ 内の残存 A/B/F 系履歴ラベルを継続精査`
- `2026-05-18 | Stage F batch-31 | files: src/AllpassDesigner.cpp,src/AllpassDesigner.h,doc/task.md | changed: A-4/B-1/A-2/B-12 の履歴ラベル 4 箇所を除去し、説明コメントへ正規化 | verify: errors=No scan=OK build=OK(Debug Build cmd env retry) residual-label-grep=math-only | risk: なし（コメント整形のみ） | next: Stage G 候補（runtime side-channel ゼロ化）へ移行 or 追加ラベル精査`
- `2026-05-18 | Stage F batch-32 | files: src/audioengine/AudioEngine.h,src/audioengine/AudioEngine.Processing.AudioBlock.cpp,src/audioengine/AudioEngine.Processing.BlockDouble.cpp,src/audioengine/AudioEngine.Processing.Snapshot.cpp,doc/task.md | changed: RT専用の runtime world-only DSP 解決 helper を追加し、Audio Thread 処理/スナップショット処理から currentDSPBits/fadingOutDSPBits への atomic fallback を除去 | verify: errors=No scan=OK build=OK(build.bat Debug nopause) grep(worldOnly helpers)=present | risk: 中低（runtime world 未確立時は silent fallback へ寄せるため prepare/bootstrap 順序に依存） | next: Timer/PrepareToPlay 等の非RT fallback 面を整理し Stage G へ接続`
- `2026-05-18 | Stage F batch-33 | files: src/audioengine/AudioEngine.h,src/audioengine/AudioEngine.Processing.AudioBlock.cpp,src/audioengine/AudioEngine.Processing.BlockDouble.cpp,doc/task.md | changed: RT専用の runtime world-only crossfade state helper を追加し、Audio Thread の crossfade 判定から prepared snapshot/atomic fallback を除去 | verify: errors=No scan=OK build=OK(build.bat Debug nopause) grep(crossfadeWorldOnly helpers)=present | risk: 低（runtime world 未確立時は先行する silent fallback で処理打ち切りのため RT で false 側へ安全収束） | next: RT sample-rate / fade-cleanup 周辺の残存 side-channel 依存を継続削減`
- `2026-05-18 | Stage F batch-34 | files: src/audioengine/AudioEngine.h,src/audioengine/AudioEngine.Processing.AudioBlock.cpp,src/audioengine/AudioEngine.Processing.BlockDouble.cpp,doc/task.md | changed: RT専用の runtime world-only sample-rate helper を追加し、Audio Thread の sample-rate 整合チェックから currentSampleRate atomic fallback を除去 | verify: errors=No scan=OK build=OK(build.bat Debug nopause) grep(runtimeSampleRateWorldOnly)=present | risk: 低（runtime world の sampleRate が無効なら silent fallback へ倒すため RT 側で不整合音を出さない） | next: RT crossfade cleanup/diagnostics 周辺の残存 side-channel 依存を継続削減`
- `2026-05-18 | Stage F batch-35 | files: src/audioengine/AudioEngine.h,src/audioengine/AudioEngine.Processing.AudioBlock.cpp,src/audioengine/AudioEngine.Processing.BlockDouble.cpp,doc/task.md | changed: RT の crossfade finalize/cleanup helper を current/fading 引数受けへ変更し、Audio Thread helper 内の loadCurrentDSP/loadFadingOutDSP atomic 読みを除去 | verify: errors=No scan=OK build=OK(build.bat Debug nopause) grep(finalize/cleanup helper args)=present | risk: 低（呼び出し元で既に world-only 解決した dsp/fading をそのまま再利用するだけで挙動不変） | next: RT diagnostic/latency helper 周辺の残存 side-channel 依存を継続削減`
- `2026-05-18 | Stage F batch-36 | files: src/audioengine/AudioEngine.h,src/audioengine/AudioEngine.Timer.cpp,doc/task.md | changed: RT publish が既に消えた debugLatencyAlign* 診断状態と Timer 側 consume/log 経路を削除し、死んだ latency side-channel を除去 | verify: errors=No scan=OK build=OK(build.bat Debug nopause) grep(debugLatencyAlign*)=0 | risk: 低（書込み元が既に存在しない死んだ診断のみ削除） | next: RT level meter / block counter 系の残存 side-channel 依存を継続監査`
- `2026-05-18 | Stage F batch-37 | files: src/audioengine/AudioEngine.h,src/audioengine/AudioEngine.Snapshot.cpp,src/audioengine/AudioEngine.Init.cpp,doc/task.md | changed: 未更新 m_audioBlockCounter に依存して常時 timeout していた waitForAudioBlockBoundary の宣言/実装/呼出しを削除し、死んだ snapshot boundary wait を除去 | verify: errors=No scan=OK build=OK(build.bat Debug nopause) grep(waitForAudioBlockBoundary)=0 | risk: 低（既存挙動は常時 timeout 後の即時適用であり、待機削除は実質 no-op 簡素化） | next: block counter/debugLastCreateAudioBlockCounter 系の残存死経路を継続整理`
- `2026-05-18 | Stage F batch-38 | files: src/audioengine/AudioEngine.h,src/audioengine/AudioEngine.Processing.AudioBlock.cpp,src/audioengine/AudioEngine.Processing.BlockDouble.cpp,src/audioengine/AudioEngine.Snapshot.cpp,src/audioengine/AudioEngine.Timer.cpp,doc/task.md | changed: 書込みのない m_audioBlockCounter / m_audioBlockCounterRtLocal / debugLastCreateAudioBlockCounter と、それに依存して実行不能だった EQ reflection recovery state を削除し、死んだ counter side-channel を除去 | verify: errors=No scan=OK build=OK(build.bat Debug nopause) grep(dead counter/recovery symbols)=0 | risk: 低（recovery 条件は未更新 counter により常時不成立で、削除は死経路整理のみ） | next: RT level meter publish の必要性と snapshot 化可能性を継続監査`
- `2026-05-18 | Stage F batch-39 | files: src/audioengine/AudioEngine.h,src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp,src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp,src/audioengine/AudioEngine.Processing.DSPCoreToBuffer.cpp,src/audioengine/AudioEngine.Processing.AudioBlock.cpp,src/audioengine/AudioEngine.Processing.BlockDouble.cpp,src/audioengine/AudioEngine.Processing.Snapshot.cpp,doc/task.md | changed: DSPCore process API を nullable meter pointer 受けへ変更し、fading/crossfade/snapshot 補助経路では input/output meter atomic publish を行わないよう整理 | verify: errors=No scan=OK build=OK(build.bat Debug nopause) grep(fadingInputMeter|fadingOutputMeter)=0 | risk: 低（UI meter は主経路のみ維持し、補助経路の未使用 meter 更新だけ停止） | next: debugAppliedEqHashVersion 等の RT 診断 atomic の必要性を継続監査`
- `2026-05-18 | Stage F batch-40 | files: src/audioengine/AudioEngine.h,src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp,src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp,src/audioengine/AudioEngine.Timer.cpp,doc/task.md | changed: write 元の無い debugLastAppliedEqHash / debugAppliedEqHashVersion / debugObservedEqHashVersion と、それに依存した EQ reflection applied/version 診断を削除し、RT debug fetch_add を除去 | verify: errors=No scan=OK build=OK(build.bat Debug nopause) grep(debugLastAppliedEqHash|debugAppliedEqHashVersion|debugObservedEqHashVersion|getLastAppliedEqHashForDebug)=0 | risk: 低（createdHash 診断は維持し、成立していない applied/version side-channel だけ撤去） | next: runtimeTransitionState debug mirror と world publish の重複診断を継続監査`
- `2026-05-18 | Stage F batch-41 | files: src/audioengine/AudioEngine.h,src/audioengine/AudioEngine.Timer.cpp,src/audioengine/AudioEngine.Processing.PrepareToPlay.cpp,doc/task.md | changed: transition 診断の読み元を RuntimePublishWorld.engine.transition へ統一し、runtimeTransitionState mirror / debugRuntimeTransition* atomic / getRuntimeTransitionStateForDebug / publishRuntimeTransitionState を削除 | verify: errors=No scan=OK build=OK(build.bat Debug nopause) grep(getRuntimeTransitionStateForDebug|publishRuntimeTransitionState|debugRuntimeTransitionActive|debugRuntimeTransitionPolicy|debugRuntimeTransitionCurrentPtr|debugRuntimeTransitionNextPtr|debugRuntimeTransitionFadeSec|debugRuntimeTransitionLatencyDeltaSamples)=0 | risk: 低（transition 診断は publish world を継続参照し、重複 side-channel だけ撤去） | next: currentDSPBits / fadingOutDSPBits の RT fallback 依存と world 観測の重複を継続監査`
- `2026-05-18 | Stage F batch-42 | files: src/audioengine/AudioEngine.Processing.Latency.cpp,doc/task.md | changed: getCurrentLatencyBreakdown の current DSP 参照を resolveCurrentDSPFromRuntimePublish から runtimePublishedCurrentDSP へ切り替え、latency query を publish 済み runtime world のみ参照するよう整理 | verify: errors=No scan=OK build=OK(build.bat Debug nopause) grep(resolveCurrentDSPFromRuntimePublish\(runtimeGraph\))=1(AudioEngine.Learning.cpp のみ残存) | risk: 低（PDC/UI latency は publish 済み runtime に揃い、未公開 currentDSP side-channel 参照のみ停止） | next: Learning 開始判定の current DSP 参照を publish world に寄せられるか継続監査`
- `2026-05-18 | Stage F batch-43 | files: src/audioengine/AudioEngine.Learning.cpp,doc/task.md | changed: processLearningCommands の学習開始判定で current DSP 取得を resolveCurrentDSPFromRuntimePublish から runtimePublishedCurrentDSP へ切り替え、learning 判定を publish 済み runtime world 参照に統一 | verify: errors=No scan=OK build=OK(build.bat Debug nopause) grep(resolveCurrentDSPFromRuntimePublish\(runtimeGraph\)|runtimePublishedCurrentDSP\(runtimeGraph\))=3(AudioEngine.h helper,Latency.cpp,Learning.cpp) | risk: 低（未公開 currentDSP への fallback 判定を停止し、公開済み runtime の可視状態に整合） | next: resolveCurrentDSPFromRuntimePublish / resolveFadingDSPFromRuntimePublish helper 自体の残存用途を継続監査`
- `2026-05-18 | Stage F batch-44 | files: src/audioengine/AudioEngine.h,src/audioengine/AudioEngine.Commit.cpp,src/audioengine/AudioEngine.Timer.cpp,src/audioengine/AudioEngine.Processing.PrepareToPlay.cpp,doc/task.md | changed: resolveCurrentDSPFromRuntimePublish / resolveFadingDSPFromRuntimePublish fallback helper を削除し、Commit/Timer/PrepareToPlay の残存参照を runtimePublishedCurrentDSP / resolveFadingDSPFromRuntimeWorldOnly へ統一して world-only 観測へ寄せた | verify: errors=No scan=OK build=OK(build.bat Debug nopause) grep(resolveCurrentDSPFromRuntimePublish\(|resolveFadingDSPFromRuntimePublish\()=0 | risk: 低（publish world 不在時の atomic fallback を停止し、公開済み runtime 可視状態に整合） | next: currentDSPBits/fadingOutDSPBits の残存 direct 読取（NonRT helper含む）を継続監査`
- `2026-05-18 | Stage F batch-45 | files: src/audioengine/AudioEngine.h,src/audioengine/AudioEngine.Commit.cpp,src/audioengine/AudioEngine.Timer.cpp,doc/task.md | changed: runtimeCrossfadePending / runtimeCrossfadeUseDryAsOld fallback helper を削除し、Commit/Timer の crossfade 判定を runtimeCrossfadePendingWorldOnly / runtimeCrossfadeUseDryAsOldWorldOnly へ統一して hidden fallback を除去 | verify: errors=No scan=OK build=OK(build.bat Debug nopause) grep(runtimeCrossfadePending\(|runtimeCrossfadeUseDryAsOld\()=0 | risk: 低（crossfade 判定は publish runtime world を継続参照し、prepared snapshot は明示 OR 条件のみ維持） | next: loadCurrentDSP/loadFadingOutDSP の direct 読取が必要な helper と world-only へ寄せられる helper を継続監査`
- `2026-05-18 | Stage F batch-46 | files: src/audioengine/AudioEngine.h,doc/task.md | changed: nonRT helper（logRuntimeTransitionEvent / startImmediateSmoothTransition / publishHardResetForCurrentDSP / armDryAsOldCrossfadeForCurrentDSP）の current 参照を loadCurrentDSP direct read から activeDSP（NonOwningPtr::get）へ切替し、transition event の fading 参照を resolveFadingDSPFromRuntimeWorldOnly へ統一 | verify: errors=No scan=OK build=OK(build.bat Debug nopause) grep(logRuntimeTransitionEvent\(|startImmediateSmoothTransition\(|publishHardResetForCurrentDSP\(|armDryAsOldCrossfadeForCurrentDSP\()=present | risk: 低（機能ロジック不変、診断/補助 helper の読取源のみ整理） | next: loadCurrentDSP/loadFadingOutDSP 直読のうち診断・検証専用箇所を world-only へ寄せられるか継続監査`
- `2026-05-18 | Stage F batch-47 | files: src/audioengine/AudioEngine.h,src/NoiseShaperLearner.cpp,doc/task.md | changed: makeEngineRuntimeState の fading 取得を resolveFadingDSPFromRuntimeWorldOnly へ切替し、NoiseShaperLearner の session signature 取得を runtimePublishedCurrentDSP(getRuntimePublishView().graph) へ切替して current/fading の direct read をさらに縮小 | verify: errors=No scan=OK build=OK(build.bat Debug nopause) grep(loadCurrentDSP\(|loadFadingOutDSP\()=definitions+comment only | risk: 低（session signature と publish world 構築の参照源のみ整理） | next: 残る direct read は helper 定義/コメント/RT 必須箇所のみを継続監査`
- `2026-05-18 | Stage F batch-48 | files: src/audioengine/AudioEngine.Commit.cpp,src/audioengine/AudioEngine.CtorDtor.cpp,src/audioengine/AudioEngine.Processing.ReleaseResources.cpp,src/audioengine/AudioEngine.h,doc/task.md | changed: commitNewDSP / shutdown / releaseResources の診断用 fading 参照を loadFadingOutDSP direct read から resolveFadingDSPFromRuntimeWorldOnly(getRuntimePublishView().graph) へ切替し、残存 direct read を定義・コメント・RT必須箇所だけに縮小 | verify: errors=No scan=OK build=OK(build.bat Debug nopause) grep(loadFadingOutDSP\()=definition+RT comment only | risk: 低（診断 slot 参照の源泉を publish world に寄せ、挙動は不変） | next: さらに削れる nonRT 診断 helper がないか継続監査`
- `2026-05-18 | Stage F batch-49 | files: src/core/RuntimeStore.h,doc/task.md | changed: RuntimeStore::WriteAccess::clearAndSwapNull() と RuntimeStore::clearAndSwapNullImpl() の実装重複を解消し、clear を publishAndSwapImpl(nullptr) へ一本化 | verify: errors=No scan=OK build=OK(build.bat Debug nopause) grep(clearAndSwapNullImpl\()=0 | risk: なし（書込み経路の薄化と重複削除のみ） | next: RuntimeStore/PublicationCoordinator 周辺の重複 helper を継続監査`
- `2026-05-18 | Stage F batch-50 | files: src/core/RuntimeStore.h,src/audioengine/AudioEngine.h,doc/task.md | changed: RuntimeStore::WriteAccess::clearAndSwapNull() API を削除し、AudioEngine 側の clearRuntimeWorldAndSwapNull() を publishAndSwap(nullptr) へ直接統一して public surface を縮小 | verify: errors=No scan=OK build=OK(build.bat Debug nopause) grep(clearAndSwapNull\()=0 | risk: 低（clear 経路の薄化のみ、publish 実装は不変） | next: RuntimeStore/PublicationCoordinator 周辺の冗長 API を継続監査`
- `2026-05-18 | Stage F batch-51 | files: src/audioengine/AudioEngine.h,src/audioengine/AudioEngine.Timer.cpp,src/audioengine/AudioEngine.Processing.PrepareToPlay.cpp,doc/task.md | changed: getRuntimePublishWorld() 中継を撤去し、RuntimeStore.observe() を getRuntimePublishView()/Timer/PrepareToPlay から直接参照するよう整理して観測経路を1枚削減 | verify: errors=No scan=OK build=OK(build.bat Debug nopause) grep(getRuntimePublishWorld\()=0 | risk: 低（観測の中継削減のみ、publish world の実体は不変） | next: runtimeStore.observe 直結の他 helper も継続監査`
- `2026-05-18 | Stage F batch-52 | files: src/core/RuntimeStore.h,doc/task.md | changed: RuntimeStore::publishAndSwapImpl() を削除し、WriteAccess::publishAndSwap() から current へ直接 exchange するよう整理して write 中継を1枚削減 | verify: errors=No scan=OK build=OK(build.bat Debug nopause) grep(publishAndSwapImpl\()=0 | risk: 低（書込みの薄い中継を除去したのみ） | next: RuntimeStore の残る単一用途 wrapper を継続監査`
- `2026-05-18 | Stage F batch-53 | files: src/audioengine/AudioEngine.h,doc/task.md | changed: startImmediateSmoothTransition / publishHardResetForCurrentDSP / armDryAsOldCrossfadeForCurrentDSP で runtimePublishView をキャッシュして validateDistinctRuntimeSlots への world 参照を再利用するよう整理し、同一関数内の getRuntimePublishView() 重複呼び出しを削減 | verify: errors=No scan=OK build=OK(build.bat Debug nopause) grep(getRuntimePublishView\()=helper+cached uses only | risk: 低（観測結果の再利用のみ、挙動不変） | next: runtimePublishView 直結の他 helper も継続監査`
- `2026-05-18 | Stage F batch-54 | files: src/core/RuntimeStore.h,doc/task.md | changed: RuntimeStore::WriteAccess の move ctor/assign を明示実装に変更し、moved-from ハンドルの store_ を nullptr 化して所有移譲を明確化 | verify: errors=No scan=OK build=OK(build.bat Debug nopause) grep(WriteAccess\(WriteAccess&&|operator=\(WriteAccess&&\))=present | risk: 低（move安全性の強化のみ、publish semantics 不変） | next: RuntimeStore/PublicationCoordinator 周辺の move-only 境界を継続監査`
- `2026-05-18 | Stage F batch-55 | files: src/audioengine/AudioEngine.h,doc/task.md | changed: RuntimePublicationCoordinator 内の単一用途 private wrapper（publishRuntimeWorldAndSwap / clearRuntimeWorldAndSwapNull）を削除し、writeAccess_.publishAndSwap(...) へ直結して publish/clear 経路を直線化 | verify: errors=No scan=OK build=OK(build.bat Debug nopause) grep(publishRuntimeWorldAndSwap\(|clearRuntimeWorldAndSwapNull\()=0 | risk: 低（中継削減のみ、publication semantics 不変） | next: RuntimePublicationCoordinator の残存 thin wrapper を継続監査`
- `2026-05-18 | Stage F batch-56 | files: src/core/RuntimeStore.h,doc/task.md | changed: RuntimeStore::WriteAccess::publishAndSwap() に store_ null ガードを追加し、moved-from ハンドル誤用時のクラッシュ経路を抑止（nullptr返却） | verify: errors=No scan=OK build=OK(build.bat Debug nopause) grep(if \(store_ == nullptr\))=1 | risk: 低（誤用時フォールトトレランス向上のみ、通常 publish semantics 不変） | next: RuntimeStore/PublicationCoordinator の move-only 境界を継続監査`
- `2026-05-18 | Stage F batch-57 | files: src/audioengine/AudioEngine.Timer.cpp,src/audioengine/AudioEngine.Processing.PrepareToPlay.cpp,doc/task.md | changed: Timer/PrepareToPlay で runtime world 観測を単一スナップショット化し、同一処理内の getRuntimePublishView()/observe 二重取得を削減（runtimeGraph を再利用） | verify: errors=No scan=OK build=OK(build.bat Debug nopause) | risk: 低（同一タイミングの world 再取得を減らす整理のみ、公開状態遷移は不変） | next: runtime world 参照の重複取得を継続監査`
- `2026-05-18 | Stage F batch-58 | files: src/core/RuntimeStore.h,doc/task.md | changed: RuntimeStore::WriteAccess::publishAndSwap() の null ガードに debug assert を追加し、moved-from ハンドル誤用を Debug で即検知しつつ Release では従来どおり nullptr フォールバック維持 | verify: errors=No scan=OK build=OK(build.bat Debug nopause) grep(RuntimeStore::WriteAccess used after move)=1 | risk: 低（デバッグ検知強化のみ、通常 publish semantics 不変） | next: RuntimeStore/PublicationCoordinator の thin wrapper と move-only 境界を継続監査`
- `2026-05-18 | Stage F batch-59 | files: src/core/RuntimeStore.h,doc/task.md | changed: RuntimeStore::WriteAccess の move-only 契約を static_assert（copy不可 / move可）で固定し、将来の accidental copy 回帰をコンパイル時に検出可能化 | verify: errors=No scan=OK build=OK(build.bat Debug nopause) | risk: 低（型契約の明文化のみ、実行時セマンティクス不変） | next: RuntimeStore/PublicationCoordinator の型境界・thin wrapper を継続監査`
- `2026-05-18 | Stage F batch-60 | files: src/core/RuntimeStore.h,doc/task.md | changed: RuntimeStore 本体の move/copy を明示 delete（RuntimeStore(RuntimeStore&&)=delete / operator=(RuntimeStore&&)=delete）して non-movable 境界を型システムで固定（不完全型 static_assert 問題を回避） | verify: errors=No scan=OK build=OK(build.bat Debug nopause) | risk: 低（型契約の明示のみ、runtime publish/observe 挙動は不変） | next: RuntimePublicationCoordinator 側の薄い中継/冗長境界を継続監査`
- `2026-05-18 | Stage F batch-61 | files: src/audioengine/AudioEngine.h,src/audioengine/AudioEngine.Commit.cpp,src/audioengine/AudioEngine.Processing.PrepareToPlay.cpp,src/audioengine/AudioEngine.Processing.ReleaseResources.cpp,src/audioengine/AudioEngine.Timer.cpp,doc/task.md | changed: RuntimePublicationCoordinator/Bridge の未使用引数 latencyDeltaSamples を publishState/adoptAndPublish/buildWorldAndPublishTransition/adoptAndBuildPublishWorld 連鎖から削除し、あわせて publishSmoothTransitionState/armDryAsOldCrossfadeForCurrentDSP の未使用引数も除去して thin wrapper 境界を簡素化 | verify: errors=No scan=OK build=OK(build.bat Debug nopause) | risk: 低（未使用パラメータ削減のみ、runtime world 構築・publish 挙動は不変） | next: RuntimePublicationCoordinator 周辺の DI/bridge 層で残る1:1転送を継続監査`
- `2026-05-18 | Stage F batch-62 | files: src/audioengine/AudioEngine.h,doc/task.md | changed: RuntimePublicationCoordinator / Dependencies の型境界に static_assert を追加し、non-copy（construct/assign）と move-constructible 契約を明文化（move-assignable は実体が満たさないため対象外） | verify: errors=No scan=OK build=OK(build.bat Debug nopause) | risk: 低（型契約の固定のみ、runtime publish semantics 不変） | next: RuntimePublicationCoordinator の bridge 経路で残る薄い1:1転送を継続監査`
- `2026-05-18 | Stage F batch-63 | files: src/audioengine/AudioEngine.h,doc/task.md | changed: RuntimePublicationCoordinator の clear 経路で retire+revision reset を直接化し、retireAndResetRuntimePublishWorldNonRt（interface/engine/bridge の1:1転送）を削除。nested class からの外側メンバ参照制約に合わせて revision reset は BridgeImpl::resetRuntimePublishRevisionNonRt() へ集約 | verify: errors=No scan=OK build=OK(build.bat Debug nopause) | risk: 低（clear publish 後の retire/revision-reset 順序は維持、薄い転送削減のみ） | next: IRuntimePublicationBridge 仕様層で実装と乖離した契約コメント/未使用面を継続監査`
- `2026-05-18 | Stage F batch-64 | files: src/audioengine/AudioEngine.h,doc/task.md | changed: 実コード参照のない IRuntimePublicationBridge 仕様構造体を削除し、BridgeImpl コメントを実体（NonRT shim）に一致させて bridge 層の文書/実装乖離を解消 | verify: errors=No scan=OK build=OK(build.bat Debug nopause) | risk: 低（未使用仕様層とコメント整理のみ、runtime publish/retire 処理は不変） | next: RuntimePublicationCoordinator の Dependencies/BridgeImpl 境界で残る冗長型ラップを継続監査`
- `2026-05-19 | Stage F batch-65 | files: src/audioengine/AudioEngine.h,doc/task.md | changed: RuntimePublicationCoordinator の Dependencies ラップを削除して ctor を (BridgeImpl, RuntimePublishWriteAccess) 直受け化。publicationCoordinator() は coordinator 内 static factory（friend境界内）経由で writeAccess を取得する形へ整理し、生成経路の1段ラップを削減 | verify: errors=No scan=OK build=OK(build.bat Debug nopause) | risk: 低（生成経路の簡素化のみ、publish/retire シーケンス不変） | next: BridgeImpl の reset/publish helper で残る単純転送を継続監査`
- `2026-05-19 | Stage F batch-66 | files: src/audioengine/AudioEngine.h,doc/task.md | changed: RuntimePublicationCoordinator から BridgeImpl を撤去し、AudioEngine& を直接保持して build/publish/retire/revision-reset を直呼び化。create factory は friend 境界（RuntimeStore::acquireWriteAccess）を維持したまま動作 | verify: errors=No scan=OK build=OK(build.bat Debug nopause) | risk: 低（中継削減のみ、runtime world publish/retire 順序は不変） | next: RuntimePublicationCoordinator 内コメント/境界名（Bridge API表現）を実装実態へさらに整合`
- `2026-05-19 | Stage F batch-67 | files: src/audioengine/AudioEngine.h,doc/task.md | changed: RuntimePublicationCoordinator 周辺コメントの "bridge" 表現を実装実態（engine直結 helper）に合わせて更新し、文書上の転送層誤認を解消 | verify: errors=No scan=OK build=OK(build.bat Debug nopause) | risk: 低（コメント整合のみ、runtime publish/retire ロジック不変） | next: RuntimePublicationCoordinator create/ctor の境界コメントをさらに簡潔化し可読性を継続改善`
- `2026-05-19 | Stage F batch-68 | files: src/core/RuntimeStore.h,doc/task.md | changed: RuntimeStore::WriteAccess の noexcept move 契約（move ctor/assign）を static_assert で明示固定し、write authority ハンドルの例外仕様ドリフトをコンパイル時に検出可能化 | verify: errors=No scan=OK build=OK(build.bat Debug nopause) | risk: 低（型契約の強化のみ、publish/observe セマンティクス不変） | next: RuntimePublicationCoordinator create/ctor 境界コメントの簡潔化を継続`
- `2026-05-19 | Stage F batch-69 | files: src/audioengine/AudioEngine.h,doc/task.md | changed: RuntimePublicationCoordinator の create factory コメントを owner(friend) 境界実態に合わせて明確化し、adoptAndPublish コメントの旧表現（publishTransitionState(debug)）を現行順序（publishCurrentDSP→world swap）へ整合 | verify: errors=No scan=OK build=OK(build.bat Debug nopause) | risk: 低（コメント整合のみ、publish/retire ロジック不変） | next: RuntimePublicationCoordinator コメントの残存冗長表現を継続整理`
- `2026-05-19 | Stage F batch-70 | files: src/core/RuntimeStore.h,src/audioengine/AudioEngine.h,doc/task.md | changed: RuntimeStore で RuntimeStore本体への型trait static_assert を試行したが不完全型文脈で C2139 を再発したため即時撤回し、既存の =delete 契約を維持したまま WriteAccess の noexcept move 契約のみ保持。あわせて RuntimePublicationCoordinator の create/adopt コメント整合を継続 | verify: errors=No scan=OK build=OK(build.bat Debug nopause) | risk: 低（挙動変更なし、契約表現とコメント整合のみ） | next: RuntimePublicationCoordinator 周辺コメントの簡潔化を継続しつつ不完全型 trait の再導入を回避`
- `2026-05-19 | Stage F batch-71 | files: src/audioengine/AudioEngine.h,doc/task.md | changed: RuntimePublicationCoordinator 近傍の履歴タグ混入コメント（stabilize-4）を削除し、C++11 nested class アクセス規則の実装説明だけに正規化 | verify: errors=No scan=OK build=OK(build.bat Debug nopause) | risk: 低（コメント整形のみ、動作不変） | next: RuntimePublicationCoordinator 周辺の残存コメントを継続監査`
- `2026-05-19 | Stage F batch-72 | files: src/core/RuntimeStore.h,doc/task.md | changed: RuntimeStore::WriteAccess の型契約に default-constructible 禁止 static_assert を追加し、write authority ハンドルが acquireWriteAccess 経由でのみ生成される境界を明示固定 | verify: errors=No scan=OK build=OK(build.bat Debug nopause) | risk: 低（型契約の明示のみ、publish/observe 挙動不変） | next: RuntimePublicationCoordinator 周辺コメントの残存冗長表現を継続整理`
- `2026-05-19 | Stage F batch-73 | files: src/audioengine/AudioEngine.h,doc/task.md | changed: RuntimePublicationCoordinator::create の引数インデントを整形し、adoptAndPublish コメントの "loadCurrentDSP()" 参照を実態表現（RT 側 current 参照）へ更新して文書整合を向上 | verify: errors=No scan=OK build=OK(build.bat Debug nopause) | risk: 低（コメント/整形のみ、動作不変） | next: RuntimePublicationCoordinator 周辺の残存コメントを継続監査`
- `2026-05-19 | Stage F batch-74 | files: src/audioengine/AudioEngine.h,doc/task.md | changed: RuntimePublicationCoordinator の noexcept move 契約強化を試行したが move-assignable の noexcept 条件が実体と不一致で C2338 を検出したため、追加 static_assert を即時撤回して既存 move-constructible 契約のみ維持し green 復帰 | verify: errors=No scan=OK build=OK(build.bat Debug nopause) | risk: 低（挙動変更なし、契約は実体一致へ回復） | next: RuntimePublicationCoordinator 周辺の残存コメント整合を継続`
- `2026-05-19 | Stage F batch-75 | files: src/core/RuntimeStore.h,doc/task.md | changed: RuntimeStore::WriteAccess::publishAndSwap と RuntimeStore::observe に HB 意図（publish=acq_rel exchange / observe=acquire load）を明示するコメントを追加し、rule4 のメモリ順序契約をコード近傍に固定 | verify: errors=No scan=OK build=OK(build.bat Debug nopause) | risk: 低（コメントのみ、動作不変） | next: RuntimeStore/RuntimePublicationCoordinator 周辺の残存コメント整合を継続`
- `2026-05-19 | Stage F batch-76 | files: src/core/RuntimeStore.h,doc/task.md | changed: RuntimeStore の friend Owner 宣言直前に write authority（acquireWriteAccess）が owner のみに限定される契約コメントを追加し、rule4 の publish authority 集約意図をコード近傍で明示 | verify: errors=No scan=OK build=OK(build.bat Debug nopause) | risk: 低（コメントのみ、動作不変） | next: RuntimeStore/RuntimePublicationCoordinator 周辺のコメント整合を継続監査`
- `2026-05-19 | Stage F batch-77 | files: src/core/RuntimeStore.h,doc/task.md | changed: RuntimeStore::WriteAccess::publishAndSwap の moved-from ハンドル誤用時挙動（Debug assert + Release no-op fallback）をコメントで明示し、既存 fail-safe 契約をコード近傍に固定 | verify: errors=No scan=OK build=OK(build.bat Debug nopause) | risk: 低（コメントのみ、動作不変） | next: RuntimeStore/RuntimePublicationCoordinator 周辺の残存コメント整合を継続`
- `2026-05-19 | Stage F batch-78 | files: src/core/RuntimeStore.h,doc/task.md | changed: RuntimeStore::observe の返却ポインタが borrow（非所有）参照である契約コメントを追加し、publish/retire 側との寿命責務分離をコード近傍で明示 | verify: errors=No scan=OK build=OK(build.bat Debug nopause) | risk: 低（コメントのみ、動作不変） | next: RuntimeStore/RuntimePublicationCoordinator 周辺コメントの残存整合を継続`
- `2026-05-19 | Stage F batch-79 | files: src/audioengine/AudioEngine.Commit.cpp,src/ConvolverProcessor.h,src/PsychoacousticDither.h,doc/task.md | changed: memory_order_relaxed 使用箇所3件に rule4-coding §2.2 準拠の根拠コメント追加（intent->next 初期化・CAS failure 側 head 更新×2・publishRuntimeProcessSnapshot 内自己読み・instanceCounter インクリメント）| verify: errors=No scan=OK build=OK(build.bat Debug nopause) | risk: 低（コメントのみ、動作不変） | next: memory_order に根拠コメントが欠けている残存箇所を継続監査`
- `2026-05-19 | Stage F batch-80 | files: src/core/EpochDomain.h,doc/task.md | changed: enterReader/exitReader の fetchAdd/fetchSub acq_rel および publishAtomic release 操作に HB 根拠コメントを追加（enterReader: acquire で直前 exitReader 観測・release で epoch publish を depth > 0 後に保証、exitReader: acquire で読み取り完了観測・release で epoch inactive 化 HB 保証）| verify: errors=No scan=OK build=OK(build.bat Debug nopause) | risk: 低（コメントのみ、動作不変） | next: EpochDomain の残存 memory_order および registerReaderThread/getMinReaderEpoch 根拠コメントを継続監査`
- `2026-05-19 | Stage F batch-81 | files: src/core/EpochDomain.h,doc/task.md | changed: registerReaderThread の CAS acq_rel/acquire（スロット取得公開・競合観測）・depth release（スロット取得後の可視性）、getMinReaderEpoch の depth/epoch acquire（enterReader release との HB）に根拠コメント追加 | verify: errors=No scan=OK build=OK(build.bat Debug nopause) | risk: 低（コメントのみ、動作不変） | next: EpochDomain constructor・reserveReaderThread・advanceEpoch の残存 memory_order コメントを監査`
- `2026-05-19 | Stage F batch-82 | files: src/core/EpochDomain.h,doc/task.md | changed: EpochDomain constructor（release で初期化後の他スレッド可視性保証）・reserveReaderThread（CAS acq_rel/acquire・depth release）・currentEpoch（acquire で advanceEpoch の release-side HB）・advanceEpoch（acq_rel で retire 観測と新 epoch 可視化）に根拠コメント追加 | verify: errors=No scan=OK build=OK(build.bat Debug nopause) | risk: 低（コメントのみ、動作不変） | next: src/core 配下の残存ファイルおよび AudioSegmentBuffer.h の memory_order コメント監査`
- `2026-05-19 | Stage F batch-83 | files: src/AudioSegmentBuffer.h,doc/task.md | changed: clear/pushBlock/copyLatest/getNumAvailableSamples の全 publishAtomic/consumeAtomic に HB 根拠コメント追加（clear=release で後続 acquire との HB 保証、pushBlock=acquire で直前 release 観測・release で更新公開、copyLatest/getNumAvailableSamples=acquire で pushBlock release と HB） | verify: errors=No scan=OK build=OK(build.bat Debug nopause) | risk: 低（コメントのみ、動作不変） | next: ConvolverControlPanel.cpp の irPreviewRequestId acq_rel/acquire 等の残存 memory_order コメント監査`
- `2026-05-19 | Stage F batch-84 | files: src/ConvolverControlPanel.cpp,doc/task.md | changed: irPreviewRequestId の fetchAddAtomic acq_rel（新 requestId の release 公開）・finishAsyncIRLoadPreview の consumeAtomic acquire（ステールネス防止ガード）・ダイアログコールバック内 consumeAtomic acquire（コールバック内 HB）に根拠コメント追加。build 失敗（return;/setIRPreviewInProgress が置換で欠落）を即修正して再 build=OK | verify: errors=No scan=OK build=OK(build.bat Debug nopause) | risk: 低（コメントのみ、動作不変） | next: AudioEngine.h のインライン getter/setter 内 acquire/release memory_order コメント監査`
- `2026-05-19 | Stage F batch-85 | files: src/audioengine/AudioEngine.h,doc/task.md | changed: getSampleRate/getInputLevel/getOutputLevel/isEqBypassRequested 等 bypass getter 群・getProcessingOrder・setAnalyzerSource/getAnalyzerSource/setAnalyzerEnabled/isAnalyzerEnabled・setIRFadeSamples（3 操作）/setEQFadeSamples/getIRFadeSamples/getEQFadeSamples・setIRChangeFlag・getLastCreatedEqHashForDebug・getLatestEqParamsFallback の全 memory_order 操作に HB 根拠コメントをグループ/インラインで追加 | verify: errors=No scan=OK build=OK(build.bat Debug nopause) | risk: 低（コメントのみ、動作不変） | next: AudioEngine.h 残存 memory_order 行または他 src ファイルの監査継続`
- `2026-05-19 | Stage F batch-86 | files: src/audioengine/AudioEngine.h,doc/task.md | changed: isShutdownInProgress（lifecycleState acquire）・getNoiseShaperLearningMode（pendingLearningMode acquire）・isRebuildObsolete（rebuildGeneration acquire）・setShutdownPhase（shutdownPhase acq_rel）・hasRebuildReason acquire/setRebuildReason acq_rel/clearRebuildReason acq_rel グループ・consumeCrossfadePreparedSnapshot acquire/publishCrossfadePreparedSnapshot acquire+release ペアに HB 根拠コメント追加 | verify: errors=No scan=OK build=OK(build.bat Debug nopause) | risk: 低（コメントのみ、動作不変） | next: AudioEngine.h template helper wrappers または他 src/ ファイルの残存 memory_order 監査`
- `2026-05-19 | Stage F batch-87 | files: src/core/CommandBuffer.h,src/LockFreeRingBuffer.h,doc/task.md | changed: CommandBuffer.h SPSCRingBuffer::push/pop に SPSC HB 根拠コメント（acquire×2 観測 + release 公開）追加。LockFreeRingBuffer.h push/pushWithWriter/size に同様コメント追加 | verify: errors=No scan=OK build=OK(build.bat Debug nopause) | risk: 低（コメントのみ、動作不変） | next: src/core/RCUReader.h・src/core/SnapshotCoordinator.h の memory_order 監査`
- `2026-05-19 | Stage F batch-88 | files: src/core/RCUReader.h,src/core/SnapshotCoordinator.h,doc/task.md | changed: RCUReader.h enter（fetchAdd acq_rel HB+CAS acq_rel/acquire）・exit（fetchSub acq_rel、publishAtomic release×2、exchangeAtomic acq_rel）・acquireThreadSlot（consumeAtomic acquire×2 + publishAtomic release）に HB 根拠コメント追加。SnapshotCoordinator.h コンストラクタ release×6・デストラクタ exchangeAtomic acq_rel×2・observeCurrent/switchImmediate/updateFade/isFading の各 acquire に HB 根拠コメント追加 | verify: errors=No scan=OK build=OK(build.bat Debug nopause) | risk: 低（コメントのみ、動作不変） | next: src/core/SnapshotCoordinator.cpp, src/LockFreeAudioRingBuffer.h, src/DeferredDeletionQueue.h など残存ファイルの memory_order 監査`
- `2026-05-19 | Stage F batch-89 | files: src/core/SnapshotCoordinator.cpp,src/LockFreeAudioRingBuffer.h,src/RefCountedDeferred.h,src/DeferredFreeThread.h,doc/task.md | changed: SnapshotCoordinator.cpp startFade(exchangeAtomic acq_rel + publishAtomic release×4)・advanceFade(acquire×2 + release×3)・tryCompleteFade(acquire×2 + CAS acq_rel/acquire)・resetFadeStateAndRetireTarget(exchangeAtomic acq_rel + release×3)・completeFade(exchangeAtomic acq_rel×2 + release×2)に HB コメント追加。LockFreeAudioRingBuffer.h prepare/reset/getAvailableSamples/push/popMixToMono/skip の全 acquire/release に HB コメント追加。RefCountedDeferred.h addRef acq_rel + release fetchSub acq_rel + fence acquire + tryAddRef CAS acq_rel/acquire に HB コメント追加。DeferredFreeThread.h stop() release + run() acquire に HB コメント追加 | verify: errors=No scan=OK build=OK(build.bat Debug nopause) | risk: 低（コメントのみ、動作不変） | next: src/DeferredDeletionQueue.h, src/GenerationManager.h, src/SafeStateSwapper.h の memory_order 監査`
- `2026-05-19 | Stage F batch-90 | files: src/DeferredDeletionQueue.h,src/GenerationManager.h,src/SafeStateSwapper.h,src/ConvolverState.h,doc/task.md | changed: DeferredDeletionQueue.h コンストラクタ release・enqueue(acquire×2+CAS acq_rel/acquire+publishAtomic release+retry acquire)・reclaim(acquire×2+CAS release/acquire+publishAtomic release+retry acquire)・drainAllUnsafe(acquire×2+CAS acq_rel/acquire+publishAtomic release)に HB コメント追加。GenerationManager.h getCurrentGeneration/isCurrentGeneration acquire に HB コメント追加。SafeStateSwapper.h swap(fetchAdd acq_rel×2+exchangeAtomic acq_rel+consumeAtomic acquire×2+publishAtomic release×3)・enterReader/exitReader/getState/tryReclaim(acquire/release 全箇所)・getSafeEpoch/getPendingRetiredCount/getMinReaderEpoch acquire に HB コメント追加。ConvolverState.h generateNewStateId acq_rel・コンストラクタ publishAtomic release×3・cleanup exchangeAtomic acq_rel×4・ムーブコンストラクタ/代入演算子 exchange+publishAtomic コンボ×3+cleanedUp release×2 に HB コメント追加 | verify: errors=No scan=OK build=OK(build.bat Debug nopause) | risk: 低（コメントのみ、動作不変） | next: src/audioengine/AtomicAccess.h, src/core/WorkerThread.h/.cpp, src/eqprocessor/EQProcessor.*, src/convolver/ConvolverProcessor.* の memory_order 監査`
- `2026-05-19 | Stage F batch-91 | files: src/audioengine/AtomicAccess.h,src/eqprocessor/EQProcessor.h,doc/task.md | changed: AtomicAccess helper と EQProcessor state/band-node helper の memory_order デフォルト引数に HB 根拠コメントを追記し、rule4-coding §2.2 の順序根拠明示を補完。既存挙動は不変（コメント追加のみ） | verify: errors=No scan=OK build=OK(Debug Build cmd env retry) grep(uncommented memory_order defaults in target files)=0 | risk: 低（コメントのみ、動作不変） | next: src/convolver/ConvolverProcessor.* の残存 memory_order 根拠コメント監査 + Release build 再検証（task起動系の不安定要因切り分け）`
- `2026-05-19 | Stage F batch-92 | files: src/ConvolverProcessor.h,doc/task.md | changed: ConvolverProcessor の one-line getter/setter と active-engine helper（load/exchange/publish）の memory_order 行に HB 根拠コメントを補完し、convolver 系の順序意図をコード近傍で明示。既存挙動は不変（コメント追加のみ） | verify: errors=No scan=OK build=OK(Debug Build cmd env retry) | risk: 低（コメントのみ、動作不変） | next: Stage D の Release ビルド再検証（task実行系の不安定要因を切り分け）`
- `2026-05-19 | Stage D release-revalidate-1 | files: .vscode/tasks.json,doc/task.md | changed: tasks.json の不正構文（壊れた Debug Build(cmd env) args / 末尾カンマ）を修正し、Release Build (cmd env retry) タスクを追加。run_task("Release") の起動不安定を切り分け、new task で安定起動を確認 | verify: build=NG(1st: ninja recompaction permission denied) -> lock cleanup(build dir remove + process cleanup) -> build=OK(Release Build cmd env retry), scan=OK | risk: 低（導線不安定は解消、今後は release-retry task を優先） | next: rule4系継続改修（残存 memory_order 根拠コメント監査）`
- `2026-05-19 | Stage F batch-93 | files: src/core/RuntimePublicationCoordinator.h,src/audioengine/AudioEngine.h,doc/task.md | changed: RuntimePublicationCoordinator を core 外部型へ抽出し、AudioEngine 側は RuntimePublicationBridge + publicationCoordinator() factory のみを保持する構成へ変更。RuntimeStore owner も外部 coordinator 型へ固定され、write authority を AudioEngine 本体から切り離した | verify: errors=No scan=OK build=OK(Debug cmd env) | risk: 低（bridge は依然 AudioEngine helper に依存するため C-03 は未完） | next: C-03（SnapshotCoordinator 解体）の継続`
- `2026-05-19 | Stage C-03 prep-1 | files: src/core/ObservedRuntime.h,src/core/SnapshotCoordinator.h,src/audioengine/AudioEngine.h,src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp,src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp | changed: observe 寿命契約の型 ObservedRuntime を SnapshotCoordinator 本体から core 独立ヘッダへ切り出し、利用側の型依存を SnapshotCoordinator::ObservedRuntime から convo::ObservedRuntime へ置換 | verify: errors=No scan=OK build=OK(Debug cmd env) | risk: 低（fade/publish/reclaim の責務はまだ SnapshotCoordinator に残存するため C-03 は継続） | next: C-03 prep-2（fade/state/reclaim 境界の切り出し候補整理）`
- `2026-05-19 | Stage C-03 prep-2 | files: src/core/SnapshotFadeState.h,src/core/SnapshotCoordinator.h,src/core/SnapshotCoordinator.cpp,doc/task.md | changed: fade 状態の atomic/storage/advance/complete 判定ロジックを SnapshotFadeState へ切り出し、SnapshotCoordinator から fade state 管理責務を分離した | verify: errors=No scan=OK build=OK(Debug cmd env) | risk: 低（current/target 保管と reclaim はまだ SnapshotCoordinator に残存するため C-03 は継続） | next: C-03 prep-3（current-target storage と reclaim の切り分け候補整理）`
- `2026-05-19 | Stage C-03 prep-3 | files: src/core/SnapshotRetireManager.h,src/core/SnapshotCoordinator.h,src/core/SnapshotCoordinator.cpp,doc/task.md | changed: GlobalSnapshot RCU 遅延解放の唯一経路として SnapshotRetireManager を新規抽出し、SnapshotCoordinator の DeletionQueue 直接操作と SnapshotFactory::destroy lambda を SnapshotRetireManager::retire() に集約。retire authority 分離（Phase 5 要件）が達成された | verify: errors=No scan=OK build=OK(Debug cmd env) | risk: 低（current/target pointer swap ロジックはまだ SnapshotCoordinator に残存するため C-03 は継続） | next: C-03 prep-4（SnapshotCoordinator の残存責務評価と C-03 クローズ可否判断）`
- `2026-05-19 | Stage C-03 prep-4 | files: src/core/SnapshotSlotStore.h,src/core/SnapshotCoordinator.h,src/core/SnapshotCoordinator.cpp,doc/task.md | changed: current/target atomic ポインタペアを SnapshotSlotStore へ切り出し、SnapshotCoordinator から std::atomic の直接操作と AtomicAccess.h への直接依存を除去。SnapshotCoordinator のプライベートメンバが EpochDomain&・SnapshotSlotStore・SnapshotFadeState・SnapshotRetireManager の 4 要素のみになり、純粋コーディネータ構造を実現。Phase 5 単一責務要件を達成 | verify: errors=No scan=OK build=OK(Debug cmd env) | risk: 低 | next: C-03 クローズ可否の最終審査（SnapshotCoordinator が単一責務違反を持たないことの確認）`

## 8. Stage D（commit 線形化）進捗

- [x] `PublicationIntent` / `PublicationLog` の最小導入
- [x] `prepareCommit()` を PublicationLog 入口へ接続
- [x] `executeCommit()` で PublicationLog を consumer 側へ取り込み
- [x] `Debug Build (cmd env retry)` 成功
- [x] `Release` ビルド再確認

### Stage D 実施証跡（2026-05-18）

- `2026-05-18 | Stage D-2 | files: src/audioengine/AudioEngine.h,src/audioengine/AudioEngine.Commit.cpp | removed-legacy: prepareCommit direct queue push | verify: scan=OK build=OK(Debug) | risk: publication storage is transitional and still coexists with deferredCommitQueue | next: Stage D-3 (queue責務縮小)`

## 9. C/H/M 再採番同期ステータス（15.11 / 10章 同期）

本セクションは `doc/detailed_design_plan4_rule4_jp.md` の `15.11` と
`doc/runtime_causality.md` の `10章` と同一採番で運用する。

### 9.1 Critical（Open）

- [x] C-01 単一 publish unit 未達（`currentDSPBits` / `fadingOutDSPBits` 残存）
- [x] C-02 `RuntimeState` 最終モデル未導入
- [x] C-03 SnapshotCoordinator 解体完了（prep-1〜prep-4: ObservedRuntime/SnapshotFadeState/SnapshotRetireManager/SnapshotSlotStore 抽出により純粋コーディネータ構造を達成）

### 9.2 High（Open）

- [x] H-01 PublicationCoordinator の core 独立型未確立（`src/core/RuntimePublicationCoordinator.h` へ抽出して解消）
- [x] H-02 observe lifetime 契約の最終統一未完（`observeCurrentRuntime` 主契約化 + 旧公開面撤去で解消）
- [x] H-03 RuntimeStore 書込み権限の最終閉鎖未完（owner を外部 `RuntimePublicationCoordinator` 型へ固定して解消）
- [x] H-04 監査章本文と実装進捗の時差（15章/10章/task の C/H/M 再採番同期で解消）
- [x] H-05 Stage 運用チェックの更新遅延（未チェック項目の再採番正規化で解消）

### 9.3 Medium（Open）

- [x] M-01 EQ cache snapshot ownership の最終判定（`AudioEngine.Cache.cpp` の `CacheMap` RAII + deferred retire 経路で再確認）
- [x] M-02 completion通知の SPSC 一本化の最終判定（`SnapshotCoordinator.cpp` の `tryCompleteFade` CAS 一回性で再確認）
- [x] M-03 thread handoff 禁止の静的検証（lint/CI）
- [x] M-04 PublicationLog 線形化の受入証跡形式の定型化

### 9.5 PublicationLog 受入証跡テンプレート（M-04）

- `YYYY-MM-DD | PublicationLog acceptance | build=<Debug/Release:OK> | strict-scan=<OK/NG> | append-order=<FIFO trace id> | consume-order=<FIFO trace id> | retire->reclaim=<epoch evidence id> | notes=<optional>`

### 9.4 旧ID解消（Closed）

- [x] 旧 C-03 dual epoch（`EpochManager::instance(` / `EpochCoreReaderGuard` / `g_deletionQueue` ヒット0）
- [x] 旧 C-08 shutdown ignoring epoch（`reclaimAllIgnoringEpoch` ヒット0）
- [x] 旧 C-04 completion side-channel（`m_fadeCompleted` ヒット0）
- [x] 旧 C-05 abort direct destroy（`abortFade` ヒット0）
- [x] 旧 H-01 helper外 `seq_cst`（`memory_order_seq_cst` ヒット0）
