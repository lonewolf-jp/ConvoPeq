# ConvoPeq ISR 実装前・最終安全チェックリスト

本チェックリストは、`doc/runtime_causality.md` / `doc/detailed_design_plan4_rule4_jp.md` / `doc/task.md` を横断して、
**実装に入る直前の最終安全確認**を 1 枚で実施するための運用文書。

- 対象: ISR 移行作業（特に Stage B 以降）
- 目的: 中間状態の長期化を防ぎ、Phase 境界を強制する
- 原則: `1 Phase 完了 -> 旧経路削除 -> 次 Phase 着手`

---

## 1. 着手可否ゲート（Go / No-Go）

### 1.1 ドキュメント固定

- [ ] `doc/runtime_causality.md` の HB / memory_order を変更していない
- [ ] `doc/detailed_design_plan4_rule4_jp.md` 15.9 の強制順序を理解済み
- [ ] `doc/task.md` の当該 Stage タスクを読み、依存順を固定済み

### 1.2 スコープ固定

- [ ] 今回の変更が **1 Stage** に閉じている
- [ ] 新機能/UI/DSP最適化が混入していない
- [ ] `JUCE/` と `r8brain-free-src/` を編集対象に含めていない

### 1.3 失敗時方針

- [ ] 失敗時は「ロールバック or Stage再実行」を選択する（継ぎ足し修正しない）
- [ ] 一時アダプタは短命であることを事前合意した

> 判定: いずれか1項目でも未達なら **No-Go**。

---

## 2. 因果・同期の安全ゲート（Phase 0 準拠）

- [ ] publish/observe が release/acquire を維持している
- [ ] reclaim 判定が `readerEpoch > retiredEpoch`（`>=` なし）
- [ ] helper 外 atomic dot-call を新規追加していない
- [ ] publication path に `relaxed` を混入していない
- [ ] `seq_cst` を便宜的に追加していない

---

## 3. Stage B（Epoch統合）着手前チェック

### 3.1 対象乖離IDの明示（A/C ID）

- [ ] A-02 / C-03（dual epoch）
- [ ] A-03（reclaim authority 分散）
- [ ] A-04 / C-08（`reclaimAllIgnoringEpoch`）

### 3.2 実施順固定（逆順禁止）

- [ ] B-1 `EpochDomain` 骨格
- [ ] B-2 旧APIアダプタ（短命）
- [ ] B-3 全 callsite 移設
- [ ] B-4 shutdown 停止保証
- [ ] B-5 retire 経路統合
- [ ] B-6 旧 epoch 物理削除
- [ ] B-7 検証ゲート

---

## 4. 変更レビュー前チェック（PR前）

### 4.1 旧経路削除確認

- [ ] 旧経路を「残したまま新経路追加」していない
- [ ] 互換コードを恒久化していない
- [ ] 削除対象が削除され、参照ゼロになっている

### 4.2 RT安全性

- [ ] Audio Thread で lock/alloc/free/IO/logger/例外を追加していない
- [ ] RT で publish/retire/reclaim を追加していない
- [ ] callback 中 multiple observe を増やしていない

### 4.3 所有権

- [ ] direct delete/free/destroy 経路を追加していない
- [ ] retire/reclaim 経路が単一 authority に収束している

---

## 5. 検証手順（最小セット）

1. 静的検査
   - [ ] `Strict Atomic Dot-Call Scan` が成功
2. ビルド
   - [ ] Debug または Release ビルド成功（少なくとも1構成）
3. 重点 grep（Stage B）
   - [ ] `EpochManager::instance(` ヒット 0
   - [ ] `EpochCoreReaderGuard` ヒット 0
   - [ ] `g_deletionQueue.` ヒット 0
   - [ ] `reclaimAllIgnoringEpoch` ヒット 0
4. shutdown 動作
   - [ ] `registration close -> stop -> join -> drainAll` 順を満たす

---

## 6. 証跡ログ（毎回記入）

- 作業日: 2026-05-18
- Stage: Stage B 完了 + Stage C-1/C-2 実装
- 対象ID（A/C/H/M）: A-02/A-03/A-04, C-03/C-08, C-07
- 変更ファイル: `src/core/EpochDomain.h`, `src/core/SnapshotCoordinator.h`, `src/audioengine/AudioEngine.*`, `src/SpectrumAnalyzerComponent.cpp`（ほか Stage B 変更群）
- 削除した旧経路: `EpochManager::instance()`, `EpochCoreReaderGuard`, `g_deletionQueue`, `reclaimAllIgnoringEpoch`, `SnapshotCoordinator::getCurrent()` 直接観測経路, `observeCurrent(0)` 直値呼び出し
- 検証結果（scan/build/grep/shutdown）: `scan=OK`, `build=OK(Debug Build cmd env retry)`, `grep(getCurrent)=0`, `grep(observeCurrent(0))=0`, `shutdown=OK(既存 Stage B 経路)`
- 未解決リスク: `ObservedSnapshot` handoff は runtime guard で検知済み、静的 lint 連携は次段で追加
- 次アクション: Stage D（Publication線形化）の事前調査

- 作業日: 2026-05-18
- Stage: Stage E-2（publish authority 境界の型抽出）
- 対象ID（A/C/H/M）: C-01, C-06, H-04（境界明示の前進）
- 変更ファイル: `src/audioengine/AudioEngine.h`, `src/audioengine/AudioEngine.Commit.cpp`, `src/audioengine/AudioEngine.Processing.PrepareToPlay.cpp`, `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp`, `src/audioengine/AudioEngine.Timer.cpp`, `doc/task.md`
- 削除した旧経路: `coordinatorPublishRuntimeState(...)`, `coordinatorAdoptAndPublishRuntime(...)` の呼び出し経路
- 検証結果（scan/build/grep/shutdown）: `scan=OK`, `build=OK(Debug Build cmd env retry)`, `grep(old coordinator calls)=0`
- 未解決リスク: `RuntimePublicationCoordinator` は `AudioEngine` ネスト型であり、外部クラス分離は次段
- 次アクション: Stage E-3（PublicationCoordinator 外部型化 / RetireManager との境界強化）

- 作業日: 2026-05-18
- Stage: Stage E-3 prep-1（epoch domain de-globalization）
- 対象ID（A/C/H/M）: A-02, C-03（single authority 経路の整理継続）
- 変更ファイル: `src/core/RCUReader.h`, `src/audioengine/AudioEngine.h`, `src/eqprocessor/EQProcessor.h`, `src/eqprocessor/EQProcessor.Core.cpp`, `src/eqprocessor/EQProcessor.Coefficients.cpp`, `src/eqprocessor/EQProcessor.Parameters.cpp`, `doc/task.md`
- 削除した旧経路: `EQProcessor` 系 `globalEpochDomain()` 直参照、`AudioEngine` の `RCUReader` グローバル依存初期化
- 検証結果（scan/build/grep/shutdown）: `scan=OK`, `build=OK(Debug Build cmd env retry)`, `grep(globalEpochDomain in src/eqprocessor/**)=0`, `errors=No`
- 未解決リスク: `src/convolver/**` の `globalEpochDomain()` は未移行
- 次アクション: Stage E-3 prep-2（convolver 側 epoch authority 収束）

- 作業日: 2026-05-18
- Stage: Stage E-3 prep-2（convolver epoch authority 収束）
- 対象ID（A/C/H/M）: A-02, C-03（single authority 経路の整理継続）
- 変更ファイル: `src/ConvolverProcessor.h`, `src/convolver/ConvolverProcessor.LoadPipeline.cpp`, `src/convolver/ConvolverProcessor.StateAndUI.cpp`, `doc/task.md`
- 削除した旧経路: `ConvolverProcessor` 系 `globalEpochDomain().advanceEpoch()` 直参照
- 検証結果（scan/build/grep/shutdown）: `scan=OK`, `build=OK(Debug Build cmd env retry)`, `grep(globalEpochDomain in src/convolver/**)=0`, `errors=No`
- 未解決リスク: `src/**` 全体での `globalEpochDomain()` 残件の再監査は未完了
- 次アクション: Stage E-3 prep-3（src 全体残件の収束）

- 作業日: 2026-05-18
- Stage: Stage E-3 prep-3（global fallback 経路の収束）
- 対象ID（A/C/H/M）: A-02, C-03（single authority 経路の整理継続）
- 変更ファイル: `src/core/RCUReader.h`, `src/audioengine/AudioEngine.h`, `src/audioengine/AudioEngine.Cache.cpp`, `src/RefCountedDeferred.h`, `doc/task.md`
- 削除した旧経路: `RCUReader` の `globalEpochDomain()` fallback、`RefCountedDeferred::release()` の default global 経路
- 検証結果（scan/build/grep/shutdown）: `scan=OK`, `build=OK(Debug Build cmd env retry)`, `grep(globalEpochDomain in src/**)=definition/comment only`, `errors=No`
- 未解決リスク: `globalEpochDomain` API 自体の物理削除は互換性監査（src外含む）が未実施
- 次アクション: Stage E-3 prep-4（globalEpochDomain API 削除可否の監査）

- 作業日: 2026-05-18
- Stage: Stage E-3 prep-4（global epoch API 物理削除）
- 対象ID（A/C/H/M）: A-02, C-03（single authority 完全収束）
- 変更ファイル: `src/core/EpochDomain.h`, `src/audioengine/AudioEngine.Globals.cpp`, `doc/task.md`
- 削除した旧経路: `globalEpochDomain()` API（未使用グローバル入口）
- 検証結果（scan/build/grep/shutdown）: `scan=OK`, `build=OK(Debug Build cmd env retry)`, `grep(globalEpochDomain in src/**)=0`, `errors=No`
- 未解決リスク: src外（将来追加コード）で旧API再導入の可能性
- 次アクション: Stage E-4（epoch authority 違反を検出する lint/CI ルール化）

- 作業日: 2026-05-18
- Stage: Stage E-4 lint-1（lint強化と既存違反是正）
- 対象ID（A/C/H/M）: C-03, 11.1（helper外 atomic禁止の検査強化）
- 変更ファイル: `.github/scripts/check-src-atomic-dotcall.ps1`, `src/DeferredDeletionQueue.h`, `src/RefCountedDeferred.h`, `src/audioengine/AudioEngine.Processing.PrepareToPlay.cpp`, `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp`, `doc/task.md`
- 削除した旧経路: helper外 `compare_exchange_weak` dot-call、`globalEpochDomain()` 再導入の検知漏れ
- 検証結果（scan/build/grep/shutdown）: `scan=OK`, `build=OK(Debug Build cmd env retry)`, `errors=No`
- 未解決リスク: helper外 `fetch_add/fetch_sub` 直呼びは lint 未検知（別ルール追加要）
- 次アクション: Stage E-4 lint-2（helper外 atomic操作の網羅監査）

- 作業日: 2026-05-18
- Stage: Stage E-4 lint-2（helper外 fetch系atomic直呼びの収束）
- 対象ID（A/C/H/M）: C-03, 11.1（helper外 atomic禁止の網羅化）
- 変更ファイル: `.github/scripts/check-src-atomic-dotcall.ps1`, `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp`, `src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp`, `src/audioengine/AudioEngine.Processing.DSPCoreIO.cpp`, `src/MKLNonUniformConvolver.cpp`, `src/NoiseShaperLearner.cpp`, `src/PsychoacousticDither.h`, `src/SafeStateSwapper.h`, `src/ConvolverControlPanel.cpp`, `src/RefCountedDeferred.h`, `src/core/SnapshotFactory.cpp`, `src/core/WorkerThread.cpp`, `src/ConvolverState.h`, `doc/task.md`
- 削除した旧経路: helper外 `fetch_add/fetch_sub` dot-call（src全域）
- 検証結果（scan/build/grep/shutdown）: `scan=OK`, `build=OK(Debug Build cmd env retry)`, `grep(fetch_* dot-call in src/**)=0`, `errors=No`
- 未解決リスク: ルールに反する新規atomic直呼びの再導入可能性（継続監査で抑止）
- 次アクション: Stage E-4 lint-3（運用監査の定着）

- 作業日: 2026-05-18
- Stage: Stage E-4 lint-3（atomic_flag系の lint 穴閉塞）
- 対象ID（A/C/H/M）: C-03, 11.1（helper外 atomic禁止の継続監査）
- 変更ファイル: `.github/scripts/check-src-atomic-dotcall.ps1`, `src/audioengine/AtomicAccess.h`, `src/audioengine/AudioEngine.h`, `src/audioengine/AudioEngine.Commit.cpp`, `doc/task.md`
- 削除した旧経路: `std::atomic_flag` + `.test_and_set()` / `.clear()` による helper外 atomic 操作
- 検証結果（scan/build/grep/shutdown）: `scan=OK`, `build=OK(Debug Build cmd env retry)`, `grep(atomic_flag|test_and_set in src/**)=0`, `errors=No`
- 未解決リスク: `std::atomic::wait/notify_*` は `condition_variable` / `future` と静的パターン衝突するため専用検査設計が別途必要
- 次アクション: Stage E-5（残存 rule4 逸脱の継続監査）

- 作業日: 2026-05-18
- Stage: Stage E-5 seqcst-1（helper外 seq_cst の削減とlint固定）
- 対象ID（A/C/H/M）: A-08, H-01, 2.3（helper外 `seq_cst` 禁止の実装反映）
- 変更ファイル: `.github/scripts/check-src-atomic-dotcall.ps1`, `src/audioengine/AudioEngine.h`, `src/LockFreeRingBuffer.h`, `src/NoiseShaperLearner.cpp`, `doc/task.md`
- 削除した旧経路: helper外 `memory_order_seq_cst`（`AudioEngine.h` / `LockFreeRingBuffer.h` / `NoiseShaperLearner.cpp`）
- 検証結果（scan/build/grep/shutdown）: `scan=OK`, `build=OK(Debug Build cmd env retry)`, `grep(memory_order_seq_cst in src/**)=AtomicAccess.h only`, `errors=No`
- 未解決リスク: `AtomicAccess.h` の helper 既定順序は API 既定値の見直しになるため、HB 契約監査とセットで別タスク化が必要
- 次アクション: Stage E-5 seqcst-2（AtomicAccess helper既定順序のHB監査）

- 作業日: 2026-05-18
- Stage: Stage E-5 seqcst-2（AtomicAccess helper既定順序のseq_cst撤去）
- 対象ID（A/C/H/M）: A-08, H-01, 2.3（`seq_cst` 完全撤去）
- 変更ファイル: `.github/scripts/check-src-atomic-dotcall.ps1`, `src/audioengine/AtomicAccess.h`, `doc/task.md`
- 削除した旧経路: `AtomicAccess` helper 既定値の `memory_order_seq_cst`（publish/consume/exchange/CAS/fetch系/ptr helper）
- 検証結果（scan/build/grep/shutdown）: `scan=OK`, `build=OK(Debug via Build_CMakeTools)`, `grep(memory_order_seq_cst in src/**)=0`, `errors=No`
- 未解決リスク: helper default に依存していた callsite は暗黙順序が変わるため、段階的に明示順序化してHB意図を固定する必要あり
- 次アクション: Stage E-5 seqcst-3（default依存callsiteの順序明示化）

- 作業日: 2026-05-18
- Stage: Stage E-5 seqcst-3 batch-1（default依存callsite順序明示化・Convolver周辺）
- 対象ID（A/C/H/M）: H-01, 9.5（helper既定値依存の明示化）
- 変更ファイル: `src/ConvolverProcessor.h`, `src/convolver/ConvolverProcessor.LoadPipeline.cpp`, `src/convolver/ConvolverProcessor.LoaderThread.cpp`, `src/convolver/ConvolverProcessor.Lifecycle.cpp`, `doc/task.md`
- 削除した旧経路: `consumeAtomic/publishAtomic` の default-order 依存（Convolver周辺の代表箇所）
- 検証結果（scan/build/grep/shutdown）: `scan=OK`, `build=OK(Debug via Build_CMakeTools)`, `errors=No`
- 未解決リスク: AudioEngine/NoiseShaper 側に default-order 依存が残存
- 次アクション: Stage E-5 seqcst-3 batch-2（AudioEngine周辺の順序明示化）

- 作業日: 2026-05-18
- Stage: Stage E-5 seqcst-3 batch-2（default依存callsite順序明示化・AudioEngine初期化/Timer）
- 対象ID（A/C/H/M）: H-01, 9.5（helper既定値依存の明示化）
- 変更ファイル: `src/audioengine/AudioEngine.Init.cpp`, `src/audioengine/AudioEngine.Timer.cpp`, `doc/task.md`
- 削除した旧経路: `publishAtomic` の default-order 依存（`maxSamplesPerBlock/currentSampleRate` 初期化、crossfade完了通知）
- 検証結果（scan/build/grep/shutdown）: `scan=OK`, `build=OK(Debug via Build_CMakeTools)`, `errors=No`
- 未解決リスク: default-order 依存の残件が複数モジュールに分散
- 次アクション: Stage E-5 seqcst-3 batch-3（残存default-order依存の横断抽出と明示化）

- 作業日: 2026-05-18
- Stage: Stage E-5 seqcst-3 batch-3（default依存callsite順序明示化・NoiseShaperLearner系）
- 対象ID（A/C/H/M）: H-01, 9.5（helper既定値依存の明示化）
- 変更ファイル: `src/NoiseShaperLearnerTypes.h`, `src/NoiseShaperLearner.h`, `src/NoiseShaperLearner.cpp`, `doc/task.md`
- 削除した旧経路: `consumeAtomic/publishAtomic` の default-order 依存（settings/state save-load周辺）
- 検証結果（scan/build/grep/shutdown）: `scan=OK`, `build=OK(Debug via Build_CMakeTools)`, `errors=No`, `residual(default-order helper calls)=173`
- 未解決リスク: default-order 依存残件が多く、一括置換はHB誤変更リスクが高い
- 次アクション: Stage E-5 seqcst-3 batch-4（AudioEngine.h/Parameters系の明示化）

- 作業日: 2026-05-18
- Stage: Stage E-5 seqcst-3 batch-4（default依存callsite順序明示化・AudioEngine Parameters/StateIO）
- 対象ID（A/C/H/M）: H-01, 9.5（helper既定値依存の明示化）
- 変更ファイル: `src/audioengine/AudioEngine.Parameters.cpp`, `src/audioengine/AudioEngine.StateIO.cpp`, `doc/task.md`
- 削除した旧経路: `consumeAtomic/publishAtomic` の default-order 依存（Parameters/StateIO の状態反映経路）
- 検証結果（scan/build/grep/shutdown）: `scan=OK`, `build=OK(Debug via Build_CMakeTools)`, `errors=No`, `residual(default-order helper calls)=81`
- 未解決リスク: `AudioEngine.h` inline helper 経路に残件集中
- 次アクション: Stage E-5 seqcst-3 batch-5（commit/prepare/release/latency/rebuild dispatch の明示化）

- 作業日: 2026-05-18
- Stage: Stage E-5 seqcst-3 batch-5（default依存callsite順序明示化・AudioEngine commit周辺）
- 対象ID（A/C/H/M）: H-01, 9.5（helper既定値依存の明示化）
- 変更ファイル: `src/audioengine/AudioEngine.Commit.cpp`, `src/audioengine/AudioEngine.Processing.PrepareToPlay.cpp`, `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp`, `src/audioengine/AudioEngine.Processing.Latency.cpp`, `src/audioengine/AudioEngine.RebuildDispatch.cpp`, `doc/task.md`
- 削除した旧経路: commit/prepare/release/latency/rebuild dispatch に散在する default-order helper 呼び出し
- 検証結果（scan/build/grep/shutdown）: `scan=OK`, `build=OK(Debug via Build_CMakeTools)`, `errors=No`
- 未解決リスク: ヘッダ inline 経路（`AudioEngine.h`）に default-order 依存が残存
- 次アクション: Stage E-5 seqcst-3 batch-6（AudioEngine.h + 周辺ヘッダの仕上げ）

- 作業日: 2026-05-18
- Stage: Stage E-5 seqcst-3 batch-6（default依存callsite順序明示化・AudioEngine.h/PsychoacousticDither）
- 対象ID（A/C/H/M）: H-01, 9.5（helper既定値依存の明示化）
- 変更ファイル: `src/audioengine/AudioEngine.h`, `src/PsychoacousticDither.h`, `doc/task.md`
- 削除した旧経路: `AudioEngine.h` inline/default-order 依存、`PsychoacousticDither` instance counter の default order
- 検証結果（scan/build/grep/shutdown）: `scan=OK`, `build=OK(Debug via Build_CMakeTools)`, `errors=No`, `residual(non-memory_order regex hits)=19`
- 未解決リスク: 残件19は `order` 転送ラッパー中心の regex 偽陽性を含むため、指標定義との整合確認が必要
- 次アクション: Stage E-5 seqcst-3 batch-7（残件の真偽分類と lint 指標整合）

- 作業日: 2026-05-18
- Stage: Stage E-5 seqcst-3 batch-7（order転送ラッパーの監査指標整合）
- 対象ID（A/C/H/M）: H-01, 9.5（helper順序明示と監査整合）
- 変更ファイル: `src/audioengine/AudioEngine.h`, `src/ConvolverProcessor.h`, `src/eqprocessor/EQProcessor.h`, `doc/task.md`
- 削除した旧経路: `order` 転送ラッパーが `memory_order` 検出条件外になることで発生していた監査ノイズ
- 検証結果（scan/build/grep/shutdown）: `scan=OK`, `build=OK(Debug via Build_CMakeTools)`, `errors=No`, `residual(non-memory_order regex hits)=0`
- 未解決リスク: 低（意味変更なし、表記正規化のみ）
- 次アクション: Stage E-5 exit-check（rule4/rule4-coding 監査ログ最終化）

- 作業日: 2026-05-18
- Stage: Stage E-5 exit-check（rule4/rule4-coding 監査ログ最終化）
- 対象ID（A/C/H/M）: C-04, H-01, 11.1（completion side-channel除去 + atomic運用監査）
- 変更ファイル: `src/core/SnapshotCoordinator.h`, `src/core/SnapshotCoordinator.cpp`, `src/audioengine/AudioEngine.Timer.cpp`, `doc/task.md`
- 削除した旧経路: `m_fadeCompleted` atomic side-channel（`requestFadeCompletion` 経路を廃止、`tryCompleteFade` を `FadeState + remaining` のCAS判定へ置換）
- 検証結果（scan/build/grep/shutdown）: `scan=OK`, `build=OK(Debug via Build_CMakeTools)`, `errors=No`, `grep(m_fadeCompleted)=0`, `grep(memory_order_seq_cst)=0`, `grep(EpochManager::instance|EpochCoreReaderGuard|g_deletionQueue|reclaimAllIgnoringEpoch)=0`
- 未解決リスク: 低（状態遷移ベース完了判定へ移行済み。Phase F の構造移行タスクは別途継続）
- 次アクション: Stage F 準備（RuntimeState統合の未着手項目棚卸し）

- 作業日: 2026-05-18
- Stage: Stage E-5 exit-check-2（abort経路の禁止語・解放経路是正）
- 対象ID（A/C/H/M）: C-05, 7.2（abort direct destroy 経路の縮退）
- 変更ファイル: `src/core/SnapshotCoordinator.h`, `src/core/SnapshotCoordinator.cpp`, `doc/task.md`
- 削除した旧経路: `abortFade` シンボル、abort時 direct destroy 経路（`resetFadeStateAndRetireTarget` + `DeletionQueue` retire 経路へ統一）
- 検証結果（scan/build/grep/shutdown）: `scan=OK`, `build=OK(Debug via Build_CMakeTools)`, `errors=No`, `grep(abortFade)=0`, `grep(m_fadeCompleted)=0`
- 未解決リスク: 低（SnapshotCoordinator はなお分割前だが、C-05の主要違反は縮退）
- 次アクション: Stage F 準備（C-01/C-02/C-06 の構造移行をタスク化）

- 作業日: 2026-05-18
- Stage: Stage F prep inventory（構造移行の着手前棚卸し）
- 対象ID（A/C/H/M）: C-01, C-02, C-06（残存確認と着手順固定）
- 変更ファイル: `doc/task.md`（証跡更新）
- 削除した旧経路: `deferredCommitQueue/deferredCommitMutex/CommitStaging` は残存なしを再確認
- 検証結果（scan/build/grep/shutdown）: `scan=OK`, `build=OK(Debug via Build_CMakeTools)`, `grep(RuntimeStore/PublicationCoordinator/RuntimeState class)=none`, `grep(currentDSPBits|fadingOutDSPBits|runtimePublishWorldState)=present`
- 未解決リスク: C-01/C-02 は構造改修未着手（single publish unit 未成立）
- 次アクション: Stage F batch-1（RuntimeStore最小導入 + publish authority閉鎖の実装開始）

- 作業日: 2026-05-18
- Stage: Stage F batch-1（RuntimeStore最小導入）
- 対象ID（A/C/H/M）: C-01, C-02（single publish unit への移行土台）
- 変更ファイル: `src/core/RuntimeStore.h`, `src/audioengine/AudioEngine.h`, `doc/task.md`
- 削除した旧経路: `runtimePublishWorldState` の atomic直保持
- 検証結果（scan/build/grep/shutdown）: `scan=OK`, `build=OK(Debug via Build_CMakeTools)`, `errors=No`, `grep(runtimePublishWorldState in AudioEngine.h)=0`, `grep(RuntimeStore<RuntimePublishWorld>)=1`
- 未解決リスク: publish authority はまだ `PublicationCoordinator` 専有ではない（AudioEngine内部の移行段階）
- 次アクション: Stage F batch-2（RuntimeStore publish API の権限制御導入）

- 作業日: 2026-05-18
- Stage: Stage F batch-2（RuntimeStore publish API の権限制御）
- 対象ID（A/C/H/M）: C-02, C-06（publish authority の閉鎖方向）
- 変更ファイル: `src/core/RuntimeStore.h`, `src/audioengine/AudioEngine.h`, `doc/task.md`
- 削除した旧経路: RuntimeStore publish/clear API の無制限 public アクセス
- 検証結果（scan/build/grep/shutdown）: `scan=OK`, `build=OK(Debug via Build_CMakeTools)`, `errors=No`, `grep(friend Owner)=present`, `grep(RuntimeStore<RuntimePublishWorld, AudioEngine>)=present`
- 未解決リスク: publish authority は `PublicationCoordinator` 専有まで未達（owner=AudioEngine段階）
- 次アクション: Stage F batch-3（RuntimeStore API の observe専用公開 + publish経路の更なる閉鎖）

- 作業日: 2026-05-18
- Stage: Stage F batch-3（RuntimeStore write経路のゲート集約）
- 対象ID（A/C/H/M）: C-01, C-06（publish経路の閉鎖強化）
- 変更ファイル: `src/audioengine/AudioEngine.h`, `doc/task.md`
- 削除した旧経路: `runtimeStore.publishAndSwap` / `runtimeStore.clearAndSwapNull` の直接呼び出し散在
- 検証結果（scan/build/grep/shutdown）: `scan=OK`, `build=OK(Debug via Build_CMakeTools)`, `errors=No`, `grep(runtimeStore.publishAndSwap|clearAndSwapNull in AudioEngine.h)=helper内のみ`
- 未解決リスク: owner が AudioEngine のため publish権限はクラス全域に残る
- 次アクション: Stage F batch-4（PublicationCoordinator専有化に向けた owner 分離準備）

- 作業日: 2026-05-18
- Stage: Stage F batch-4（RuntimeStore write の coordinator 専有化）
- 対象ID（A/C/H/M）: C-01, C-02（publish authority の境界縮小）
- 変更ファイル: `src/audioengine/AudioEngine.h`, `doc/task.md`
- 削除した旧経路: AudioEngine 本体からの RuntimeStore write API 呼び出し（`publish/clear` を `RuntimePublicationCoordinator` に移管）
- 検証結果（scan/build/grep/shutdown）: `scan=OK`, `build=OK(Debug via Build_CMakeTools)`, `errors=No`, `grep(runtimeStore.publishAndSwap|clearAndSwapNull in AudioEngine.h)=RuntimePublicationCoordinator内のみ`
- 未解決リスク: `RuntimePublicationCoordinator` はまだ `AudioEngine` ネスト型（core 外部型分離は未着手）
- 次アクション: Stage F batch-5（PublicationCoordinator 外部型導入の前段リファクタ）

- 作業日: 2026-05-18
- Stage: Stage F batch-5（RuntimeStore owner 型の権限縮小）
- 対象ID（A/C/H/M）: C-02, C-06（型レベルで publish authority を縮小）
- 変更ファイル: `src/audioengine/AudioEngine.h`, `doc/task.md`
- 削除した旧経路: `RuntimeStore<RuntimePublishWorld, AudioEngine>` による owner 権限の過大化
- 検証結果（scan/build/grep/shutdown）: `scan=OK`, `build=OK(Debug via Build_CMakeTools)`, `errors=No`, `grep(RuntimeStore<RuntimePublishWorld, RuntimePublicationCoordinator>)=present`, `grep(runtimeStore.publishAndSwap|clearAndSwapNull in AudioEngine.h)=RuntimePublicationCoordinator内のみ`
- 未解決リスク: coordinator が nested class のため、C-01/C-02 完了には core 外部型化と責務切り出しが必要
- 次アクション: Stage F batch-6（PublicationCoordinator の外部型化可否を検証）

- 作業日: 2026-05-18
- Stage: Stage F batch-6（runtime publish/clear 実体の coordinator 集約）
- 対象ID（A/C/H/M）: C-01, C-06（publish authority の実装境界を coordinator へ集中）
- 変更ファイル: `src/audioengine/AudioEngine.h`, `src/audioengine/AudioEngine.CtorDtor.cpp`, `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp`, `doc/task.md`
- 削除した旧経路: AudioEngine 本体 inline に残っていた `publishRuntimeSnapshots` / `clearPublishedRuntimeSnapshotsNonRt` 実装
- 検証結果（scan/build/grep/shutdown）: `scan=OK`, `build=OK(Debug via Build_CMakeTools)`, `errors=No`, `grep(runtimeStore.publishAndSwap|clearAndSwapNull)=RuntimePublicationCoordinator内のみ`, `grep(clearPublishedRuntimeSnapshotsNonRt callsite)=publicationCoordinator経由`
- 未解決リスク: `RuntimePublicationCoordinator` は nested class のまま（外部型化は未着手）
- 次アクション: Stage F batch-7（coordinator 外部型化に向けた依存切り出し）

- 作業日: 2026-05-18
- Stage: Stage F batch-7a（coordinator の owner 依存縮小）
- 対象ID（A/C/H/M）: C-02, H-01（helper依存の縮小と memory_order 明示維持）
- 変更ファイル: `src/audioengine/AudioEngine.h`, `doc/task.md`
- 削除した旧経路: `RuntimePublicationCoordinator` からの `owner.fetchAddAtomic` / `owner.publishAtomic` 依存
- 検証結果（scan/build/grep/shutdown）: `scan=OK`, `build=OK(Debug via Build_CMakeTools)`, `errors=No`, `grep(owner.fetchAddAtomic|owner.publishAtomic(owner.runtimeGraphRevision))=0`
- 未解決リスク: coordinator 自体は nested class のため、外部型化には `RuntimePublishWorld/DSPCore` 型境界の追加整理が必要
- 次アクション: Stage F batch-7b（nested解除可否の型依存整理）

- 作業日: 2026-05-18
- Stage: Stage F batch-7b（RuntimePublishWorld のスコープ切り出し）
- 対象ID（A/C/H/M）: C-02（coordinator 外部型化へ向けた型境界整理）
- 変更ファイル: `src/audioengine/AudioEngine.h`, `doc/task.md`
- 削除した旧経路: `AudioEngine` ネスト型 `RuntimePublishWorld`
- 検証結果（scan/build/grep/shutdown）: `scan=OK`, `build=OK(Debug via Build_CMakeTools)`, `errors=No`, `grep(struct RuntimePublishWorld)=header-scope`, `grep(AudioEngine::RuntimePublishWorld)=0`
- 未解決リスク: coordinator は nested class のため、`DSPCore` と owner private API 依存の橋渡し設計が必要
- 次アクション: Stage F batch-7c（coordinator 外部型化の最小ブリッジ設計）

- 作業日: 2026-05-18
- Stage: Stage F batch-7c（coordinator 外部型化に向けた最小ブリッジ導入）
- 対象ID（A/C/H/M）: C-02, C-06（publish authority 維持のまま依存境界を明確化）
- 変更ファイル: `src/audioengine/AudioEngine.h`, `doc/task.md`
- 削除した旧経路: `RuntimePublicationCoordinator` から generation/version/retire/reset 処理への直接実装依存
- 検証結果（scan/build/grep/shutdown）: `scan=OK`, `build=OK(Debug via Build_CMakeTools)`, `errors=No`, `grep(reserveNextRuntimeGraphGeneration|reserveNextRuntimeVersion|retireRuntimePublishWorldNonRt|resetRuntimeGraphRevisionNonRt)=present`
- 未解決リスク: `RuntimeStore` write API は owner friend 制約により coordinator 内直接呼び出しを維持
- 次アクション: Stage F batch-8（coordinator 外部型化の実装可否を最小PoCで検証）

- 作業日: 2026-05-18
- Stage: Stage F batch-8 stabilize-1（PoCロールバック後の型境界安定化）
- 対象ID（A/C/H/M）: C-02, C-06（publish authority 閉鎖を維持したままビルド回復を固定）
- 変更ファイル: `src/audioengine/AudioEngine.h`, `src/core/RuntimeStore.h`, `doc/task.md`
- 削除した旧経路: 外部型化PoCで混入した不完全型依存（`RuntimeStore<..., RuntimePublicationCoordinator>` 宣言位置と owner 型可視順の破綻）
- 検証結果（scan/build/grep/shutdown）: `scan=OK`, `build=OK(Debug via Build_CMakeTools)`, `errors=No`, `grep(runtimeStore.publishAndSwap|clearAndSwapNull)=RuntimePublicationCoordinator内のみ`
- 未解決リスク: coordinator の core 外部型化は未再開（nested 構成を維持）
- 次アクション: Stage F batch-8 stabilize-2（外部型化再挑戦前に owner 依存面を追加分離）

- 作業日: 2026-05-18
- Stage: Stage F batch-8 stabilize-2（publish authority API露出面の縮小）
- 対象ID（A/C/H/M）: C-02, C-06（RuntimeStore write authority の境界をさらに閉鎖）
- 変更ファイル: `src/audioengine/AudioEngine.h`, `doc/task.md`
- 削除した旧経路: `RuntimePublicationCoordinator::publishRuntimeWorldAndSwap` / `clearRuntimeWorldAndSwapNull` の public 露出
- 検証結果（scan/build/grep/shutdown）: `scan=OK`, `build=OK(Debug Build cmd env retry)`, `errors=No`, `grep(runtimeStore.publishAndSwap|clearAndSwapNull)=RuntimePublicationCoordinator private内のみ`
- 未解決リスク: coordinator の core 外部型化は未着手（nested 維持）
- 次アクション: Stage F batch-8 stabilize-3（外部型化前の owner 依存面をさらに局所化）

---

- 作業日: 2026-05-18
- Stage: Stage F batch-8 stabilize-3（world 構築を bridge API へ集約し coordinator の owner 直接呼び出しを削減）
- 対象ID（A/C/H/M）: C-02, C-06（RuntimePublicationCoordinator の owner 依存面局所化）
- 変更ファイル: `src/audioengine/AudioEngine.h`, `doc/task.md`
- 追加: `buildRuntimePublishWorld` bridge API（reserveNextRuntimeGraphGeneration / makeEngineRuntimeState / makeRuntimeGraphState / reserveNextRuntimeVersion を AudioEngine 内部に隠蔽）
- 削除した旧経路: `publishRuntimeSnapshots` 内の `owner.reserveNextRuntimeGraphGeneration()` / `owner.makeEngineRuntimeState()` / `owner.makeRuntimeGraphState()` / `owner.reserveNextRuntimeVersion()` 直接呼び出し（4本 → bridge 1本に集約）
- 検証結果（scan/build/grep/shutdown）: `scan=OK`, `build=OK(Debug Build cmd env retry)`, `errors=No`
- 未解決リスク: coordinator の owner 依存は `publishRuntimeTransitionState` / `publishCurrentDSPAndTakeOwnership` / `retireRuntimePublishWorldNonRt` / `resetRuntimeGraphRevisionNonRt` が残存
- 次アクション: Stage F batch-8 stabilize-4 または batch-9（coordinator 外部型化）

- 作業日: 2026-05-19
- Stage: Stage F batch-91（memory_order 根拠コメントの残件補完）
- 対象ID（A/C/H/M）: H-01, 2.2, 9.5（順序根拠の明示）
- 変更ファイル: `src/audioengine/AtomicAccess.h`, `src/eqprocessor/EQProcessor.h`, `doc/task.md`
- 削除した旧経路: 「デフォルト memory_order 既定値の根拠がコード近傍にない」状態
- 検証結果（scan/build/grep/shutdown）: `scan=OK`, `build=OK(Debug Build cmd env retry)`, `errors=No`, `grep(uncommented memory_order defaults in target files)=0`
- 未解決リスク: Release ビルド再検証は task 起動系の不安定要因があり未完了
- 次アクション: `src/convolver/ConvolverProcessor.*` の残存 memory_order 根拠コメント監査と Release 検証導線の切り分け

- 作業日: 2026-05-19
- Stage: Stage F batch-92（convolver 系 memory_order 根拠コメント補完）
- 対象ID（A/C/H/M）: H-01, 2.2, 9.5（順序根拠の明示）
- 変更ファイル: `src/ConvolverProcessor.h`, `doc/task.md`
- 削除した旧経路: 「ConvolverProcessor one-line memory_order 行の根拠コメント不足」状態
- 検証結果（scan/build/grep/shutdown）: `scan=OK`, `build=OK(Debug Build cmd env retry)`, `errors=No`
- 未解決リスク: Stage D の `Release` ビルド再確認が未完了（task 実行系の不安定を要切り分け）
- 次アクション: `Release` ビルド再検証（task 起動失敗原因の切り分け）

- 作業日: 2026-05-19
- Stage: Stage D release-revalidate-1（Release導線の切り分けと復旧）
- 対象ID（A/C/H/M）: 運用導線（task起動不安定）
- 変更ファイル: `.vscode/tasks.json`, `doc/task.md`
- 削除した旧経路: `run_task("Release")` で「Task started but no terminal was found」が発生する不安定導線（構文不整合 tasks.json 起因）
- 検証結果（scan/build/grep/shutdown）: `scan=OK`, `build=NG(first: ninja recompaction permission denied) -> build dir cleanup 実施 -> build=OK(Release Build (cmd env retry))`
- 未解決リスク: 低（ninja permission denied は build dir ロック由来。再発時は kill+build削除で回避可能）
- 次アクション: rule4/rule4-coding 継続改修へ復帰

汎用版1行フォーマット（`doc/task.md` の B-1〜B-7 と同一順序）:

- `YYYY-MM-DD | Stage B-x | files: <changed files> | removed-legacy: <removed paths> | verify: scan=<OK/NG> build=<OK/NG> grep=<OK/NG> shutdown=<OK/NG> | risk: <open risks> | next: <next action>`

記入例:

- `2026-05-17 | Stage B-3 | files: src/audioengine/*,src/eqprocessor/* | removed-legacy: EpochManager/EpochCore/g_deletionQueue callsites | verify: scan=OK build=OK grep=OK shutdown=N/A | risk: callsite漏れ監視 | next: B-4`

---

## 7. 最終判定

- [x] **Go**: すべて達成
- [ ] **No-Go**: 未達あり（未達項目を上記へ記録）

No-Go の場合、実装続行禁止。まず未達を解消して再判定する。
