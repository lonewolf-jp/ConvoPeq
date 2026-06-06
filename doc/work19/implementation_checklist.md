# Practical Stable ISR Bridge Runtime — 実装チェックリスト

作成日: 2026-06-06
ベース: `doc/work19/refactoring_plan_v2.md`

---

## PR-0 (Gate-0): 事前監査フェーズ

### Crossfade Decision Input 棚卸し

- [x] 0-1: `computeDecision()` の全分岐で参照している DSPCore フィールドを列挙
- [x] 0-2: 列挙したフィールドが `RuntimeWorld.dspProjection` に存在するか確認
- [x] 0-3: 不足フィールドがないことを確認 (3フィールド全て対応済み)
- [x] 0-4: 棚卸し結果を `doc/work19/crossfade_input_inventory.md` に文書化
- [x] 0-5: `buildRuntimePublishWorld()` が CrossfadeDecision / TransitionPolicy に依存している箇所を棚卸し
- [x] 0-6: `evaluate()` の参照フィールド一覧と dspProjection のフィールド一覧を自動比較する仕組みを設計 (kEvaluateRelevantFieldNames 追加)
- [x] 0-7: `getActiveRuntimeDSP()` 全使用箇所を Execution用途 / Semantic用途 に事前分類
- [x] 0-8: Snapshot Authority Inventory: RuntimeBuildSnapshot に不足している全フィールドを棚卸し
- [x] 0-9: Observe Source Audit: DSP操作系APIの全使用箇所を Execution用途 / Semantic用途 に分類
- [x] 0-10: RuntimeWorld 全フィールド Decision Input Inventory (runtimeworld_decision_input_inventory.md)
- [x] 0-11: CrossfadeAuthority Output Inventory: Decision出力の伝搬先を一覧化
- [x] 0-12: Handle Resolution Authority の決定 (Orchestrator 担当確定)
- [x] 0-13: Current Decision Input Inventory (crossfade_input_inventory.md と統合)
- [x] 0-14: CrossfadeAuthority Dual-Path Audit (evaluateOnly vs evaluateFromWorlds 比較) → doc/work19/dual_path_audit.md
- [x] 0-15: Builder Projection Coverage Audit (snapshot_authority_inventory.md)
- [x] 0-Audit: Admission Input Audit (PublishRequest 全フィールド Semantic/Execution 分類 → Semantic 用途ゼロ確認済み)

### Gate-0 完了条件

- [x] Crossfade Decision Input Inventory 完了 (crossfade_input_inventory.md)
- [x] Snapshot Authority Inventory 完了 (snapshot_authority_inventory.md)
- [x] RuntimeBuildSnapshot 追加フィールド確定 (6フィールド追加済み)
- [x] 0-6 自動比較機構 実装完了 (kEvaluateRelevantFieldNames in CrossfadeAuthority.h)
- [x] Dual-Path Audit 完了 (doc/work19/dual_path_audit.md)
- [x] publishWorld 呼び出し8箇所分類完了 (publish_calls_classification.md)
- [x] PR-4/PR-7 実施順決定 → PR-3 → PR-7 → PR-4 推奨 (Admission の deferred が予想より単純なため)
- [x] Observe Source Audit 完了 (observe_source_audit.md)
- [x] Handle Resolution Authority 決定完了 (Orchestrator 担当確定)
- [x] Decision Candidate Inventory 完了 (crossfade_input_inventory.md + runtimeworld_decision_input_inventory.md)
- [x] `getActiveRuntimeDSP()` Semantic 用途ゼロの Gate 条件確立 (Observe Source Audit 結果)
- [x] Builder Projection Coverage Audit 完了 (snapshot_authority_inventory.md)
- [x] **Admission Input Audit 完了** (PublishRequest Semantic用途ゼロ確認)

---

## PR-2: RuntimeBuilder Snapshot Authority 化

- [x] 2-1: `dspProjection` 構築を DSPCore 直読から `RuntimeBuildSnapshot` 経由に変更
- [x] 2-2: `RuntimeBuildSnapshot` に不足フィールドを追加 (irLoaded, irFinalized, structuralHash, oversamplingFactor, sampleRate, baseLatencySamples)
- [x] 2-3: DSPCore* 引数は Execution Object として維持（削除しないことを確認）
- [x] 2-4: `buildRuntimePublishWorld()` 内で DSPCore を Execution Object として使用する経路を確認
- [x] ビルド通過確認 (get_errors: 新規エラーなし)
- [x] Authority Regression Gate 通過確認 ✅

---

## PR-1: CrossfadeAuthority RuntimeWorld 化

- [x] 1-1: Orchestrator の処理順序を「build → evaluate → update → publish」に変更 (option B/選択肢B)
- [x] 1-2: `runtimeStore.observe()` で oldWorld を取得する経路を追加
- [x] 1-3: `evaluateFromWorlds()` → `evaluate()` に API 変更 (DSPCore 非依存)
- [x] 1-4: `CrossfadeAuthority::evaluateOnly()` / `evaluateAndRegister()` を削除
- [x] 1-5: `CrossfadeAuthority::computeDecision(DSPCore*, DSPCore*)` を削除
- [x] 1-6: DSPCore 直読ロジックが残っていないことを確認 (evaluateOnly/evaluateAndRegister/computeDecision/doRegister 全て削除済み)
- [x] ビルド通過確認 (get_errors: 新規エラーなし)
- [x] Authority Regression Gate 通過確認 ✅

---

## PR-3: Admission DSPCore 直読排除 ✅ 完了（Authority 修正クローズ）

**Gate-0 判断**: Authority 修正として完了。Admission の DSPCore 直読は sealedSnapshot 経由に修正済み。
PublishRequest の DSPHandle 化は低優先度 PR-3A へ分離。

- [x] 3-4: ✅ Admission の `evaluate()` が sealedSnapshot 経由で判断 (DSPCore 直読ゼロ)
- [x] Decision 系クラスの DSPCore 直読ゼロ確認済み
- [x] Authority Regression Gate 通過確認

---

---

## Phase2: ISR Runtime 整理

---

## PR-3A: Execution Path Handle Normalization ✅ 完了

**位置付け**: Phase2 ISR Runtime 整理の中核タスク。Gateway監査で `PublishRequest.newDSP` の全使用箇所が Execution用途のみであることが確認されたため、Authority修正ではなく型安全性改善として Phase2 に配置。

**設計**: Option B（DSPHandle 化）を採用。既存 `DSPHandleRuntime` インフラを活用。

### 変更概要

| # | 変更 | ファイル | 状態 |
| --- | --- | --- | --- |
| 3A-1 | `PublishRequest::newDSP`: `void*` → `DSPHandle` | `PublicationAdmission.h` | ✅ 完了 |
| 3A-2 | `AudioEngine::resolveDSPHandle(DSPHandle) → DSPCore*` 追加 | `AudioEngine.h` | ✅ 完了 |
| 3A-3 | commit 時 `registerDSPHandleForRuntime()` で handle 事前登録 | `AudioEngine.Commit.cpp` | ✅ 完了 |
| 3A-4 | Orchestrator 内 `resolveDSPHandle()` 経由で Handle → DSPCore* 解決 | `RuntimePublicationOrchestrator.cpp` | ✅ 完了 |

### 完了条件

- [x] `PublishRequest::newDSP` が `DSPHandle` である（`void*` ではない）
- [x] commit 経路で `registerDSPHandleForRuntime()` が呼ばれている
- [x] Orchestrator が `resolveDSPHandle()` 経由で DSPCore* を取得している
- [x] `static_cast<AudioEngine::DSPCore*>(req.newDSP)` がコードベースに存在しない
- [x] `AudioEngine::resolveDSPHandle()` が定義されている
- [x] ビルド通過確認（変更4ファイル: エラーゼロ）
- [x] Authority Regression Gate 通過確認 ✅（Semantic 用途ゼロ維持）

---

---

## PR-5: Crossfade Registration 規約化 ✅ 完了

**Architecture Invariant の明文化**: Gate-0 確認により registerCrossfade() の呼び出し元が DSPTransition のみであることが確認済み。責務境界を DSPTransition.h に Authority コメントとして明文化。

- [x] 5-1: DSPTransition.h に Registration Authority コメント追加 (Decision/Execution/Registration 責務境界)
- [x] 5-2: registerCrossfade() 呼び出し元 DSPTransition のみ確認済み (grep)
- [x] 5-3: PR-4 完了後再確認 → DSPTransition のみのまま ✅

---

## PR-4: publishWorld() 直接呼び出しの統一

- [x] 4-1: Publication カテゴリの呼び出しを `submitPublishRequest()` 経由に変更 (Commit パスは既に対応済み)
- [x] 4-2: Bootstrap/Shutdown カテゴリは専用モードまたは現状維持
- [x] 4-3: `PublicationExecutor::publish()` の `coordinator.publishWorld()` は維持
- [x] 4-4: デッドコード `publishRuntimeStateNonRt()` を削除 (宣言+定義を削除、RuntimeBuilder.h include も削除)
- [x] ビルド通過確認 (get_errors: 新規エラーなし)
- [x] Authority Regression Gate 通過確認 ✅

---

## PR-7: Deferred Queue 移設 ✅ 完了

- [x] 7-1: `RuntimePublicationOrchestrator` に deferred queue を新設 (deferredRequest_/hasDeferred_)
- [x] 7-2: `PublicationAdmission::deferredRequest_` / `hasDeferred_` を削除
- [x] 7-3: Coordinator が Orchestrator のキュー経由で deferred publish を管理 (notifyTransitionComplete で再試行)
- [x] ビルド通過確認 (get_errors: 新規エラーなし)
- [x] Authority Regression Gate 通過確認 ✅

---

## サービス分離・副作用整理 (各PR内で実施)

- [x] C1: デッドコード `publishRuntimeStateNonRt()` を削除 (PR-4 で実施済み) ✅
- [x] C2: `sendChangeMessage()` / `triggerAsyncUpdate()` の publish pipeline 混在確認 → 問題なし ✅
- [x] C3-Latency: `LatencyService` 分離設計 → **Phase2後半以降** (Authority効果なし、AudioThread再監査リスク大)
- [x] C3-Warmup: `WarmupService` 分離設計 → **実施不要 (No-op)** (Builder責務のまま維持)

---

## Authority Regression Gate (全PR共通)

- [x] DSPCore* を判断入力として読む箇所数が非増加 → **0箇所** (CrossfadeAuthority 完全非依存化)
- [x] `getActiveRuntimeDSP()` の Semantic 利用箇所数が非増加 → **0箇所** (Orchestrator の1箇所はExecution用途)
- [x] `publishWorld()` の直接呼び出し箇所数が非増加 → **6箇所** (Lifecycle/Transition/Internalのみ)
- [x] `registerCrossfade()` の DSPTransition 以外からの呼び出し数が非増加 → **0箇所**
- [x] `evaluateOnly()` / `computeDecision(DSPCore*)` / `evaluateFromWorlds()` の残存呼び出し数が非増加 → **0箇所** (全削除済み)
- [x] `evaluate()` 参照フィールド数と dspProjection 供給フィールド数の一致度が維持 → **被覆率100%** (3/3)
- [x] Decision 系クラスの DSPCore 直読箇所数が非増加 → **0箇所**
