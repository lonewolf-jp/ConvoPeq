# Practical Stable ISR Bridge Runtime — 実装チェックシート

作成日: 2026-06-05
最終更新: 2026-06-06
ベース: doc/work18/refactoring_plan.md

---

## 凡例

- [x] 完了
- [-] スキップ/不要

---

## PR-1: Coordinator Pipeline ✅ 完了

**新規作成ファイル** (構文エラー0):

| ファイル | 責務 |
| --------- | ------ |
| `src/audioengine/PublicationAdmission.h/.cpp` | Admission: evaluate/accepted/deferred/rejected |
| src/audioengine/PublicationExecutor.h/.cpp | Executor: publish + PublishResult enum |
| src/audioengine/DSPTransition.h | Transition: onPublishCompleted/onTransitionComplete |
| src/audioengine/DSPLifetimeManager.h | DSP activate/crossfade/retire 一元管理 |
| src/audioengine/CrossfadeAuthority.h/.cpp | Decision + evaluate + register |
| src/audioengine/RuntimePublicationOrchestrator.h/.cpp | submitPublishRequest |

**CMakeLists.txt**: 5 .cpp files added

---

## PR-2: CrossfadeAuthority 昇格 ✅ 完了

- evaluateAndRegister(): decision + registration 統合
- computeDecision(): computeCrossfadeContext ロジック移植
- Orchestrator.submitPublishRequest 内で CrossfadeAuthority 使用

---

## PR-1.5: AudioEngine 接続 ✅ 完了

- [x] AudioEngine に RuntimePublicationOrchestrator メンバ追加
- [x] enqueuePublicationIntentForRuntimeCommit で Orchestrator 試用(新旧並行)

---

## PR-3: 旧 Commit 経路削除 ✅ 完了

- [x] processPendingCommit() 削除
- [x] applyRuntimeCommitFromIntent() 削除 (~550行)
- [x] `PendingCommitData` / `pendingCommitFlag_` / `pendingCommit_` / `pendingCommitMutex_` 削除
- [x] timer → Orchestrator::hasDeferredRequest() に置換
- [x] isFullyDrained() の pendingCommitFlag_ → Orchestrator::hasDeferredRequest()
- [x] テスト (BuildInputSemanticContractTests) 更新

---

## PR-4: Semantic 移行 ✅ 完了

- [x] RuntimeState に DSPSemanticProjection 構造体追加 (dspProjection)
- [x] kFieldDescriptors / kRuntimeAuthorityInventory / kRuntimeReadAuthorityInventory 更新
- [x] RuntimeBuilder::buildRuntimePublishWorld() で dspProjection 設定
- [x] CrossfadeAuthority::evaluateFromWorlds() 追加 (投影値ベース判断)

---

## PR-5: 硬化 ✅ 完了

- [x] publishWorld() 内で sealRecursively() を必須呼び出し
- [x] sealedSnapshot authority 明文化コメント追加 (RuntimeBuilder.cpp)

---

## 作成/変更ファイル一覧

| ファイル | 構文エラー | PR |
| --------- | ----------- | ----- |
| `PublicationAdmission.h/.cpp` | 0 | PR-1 |
| PublicationExecutor.h/.cpp | 0 | PR-1 |
| DSPTransition.h | 0 | PR-1 |
| DSPLifetimeManager.h | 0 | PR-1 |
| CrossfadeAuthority.h/.cpp | 0 | PR-1/2/4 |
| RuntimePublicationOrchestrator.h/.cpp | 0 | PR-1/3 |
| AudioEngine.h | 0 (pre-existing warnings) | PR-3/4 |
| AudioEngine.Commit.cpp | 0 | PR-3 |
| AudioEngine.RebuildDispatch.cpp | 0 (pre-existing warnings) | PR-3 |
| AudioEngine.Timer.cpp | 0 | PR-3 |
| AudioEngine.Threading.cpp | 0 (pre-existing warnings) | PR-3 |
| RuntimeBuilder.cpp | 0 | PR-4/5 |
| RuntimePublicationCoordinator.h | 0 | PR-5 |
| BuildInputSemanticContractTests.cpp | 0 | PR-3 |
| **Total 14 files** | **0 new errors** | **全PR** |
