# 修正版バグリスト

以下、これまでのレビュー検証結果を反映した修正版バグリストを示す。重要度は「実害の可能性」「規約違反の程度」「修正優先度」を総合的に判断して再分類した。

---

## 🔴 Critical（至急修正・クラッシュまたはRT違反）

| No | ファイル | 行（目安） | バグ内容 | 修正方向性 | 状態 |
| --- | --- | --- | --- | --- | --- |
| **C1** | `src/ProgressiveUpgradeThread.cpp` | `upgradeStep()` 内の `callAsync` | `juce::MessageManager::callAsync([this, ...])` で `this`（`ProgressiveUpgradeThread`）をキャプチャ。スレッド停止後にラムダ実行されるとUAF | `juce::WeakReference<ConvolverProcessor>` を使用。または `AsyncUpdater` に置換 | ✅ 修正済み（2026-05-17） |
| **C2** | `src/eqprocessor/EQProcessor.Processing.cpp` | `process()` 先頭 | `RCUReaderGuard` が `loadCurrentState()` **後**に宣言。状態ポインタ取得からガードまでの微小ウィンドウでUAFの可能性（理論上） | `RCUReaderGuard` を `loadCurrentState` の**前に**移動 | ✅ 対応済み（確認 2026-05-17） |
| **C3** | `src/CmaEsOptimizerDynamic.cpp` | `update()` 関数内 | コスト関数が `NaN` を返すと共分散行列・平均ベクトルが汚染され、以後の最適化が完全停止 | `update()` 冒頭で `fitness` の有限値チェック。異常時は `resetIdentityCovariance()` + シグマ初期化 | ✅ 対応済み（確認 2026-05-17） |
| **C4** | `src/LockFreeRingBuffer.h` | `push()` / `pushWithWriter()` | Audio Thread から `compare_exchange_weak`（CAS retry loop）を実行。RTスレッドでの非決定性・無限リトライの可能性 → 規約3.2違反 | SPSC設計に変更。RTスレッドでは `store(release)` のみ使用。またはRTからのエンキューを禁止 | ✅ 対応済み（確認 2026-05-17） |

---

## 🟡 High（修正推奨・アーキテクチャ違反または潜在的不具合）

| No | ファイル | 行（目安） | バグ内容 | 修正方向性 | 状態 |
| --- | --- | --- | --- | --- | --- |
| **H1** | `src/audioengine/AudioEngine.Commit.cpp` | `commitNewDSP()` 内 | `runtimeCrossfadePending` と `preparedCrossfade.pending` の二重状態管理。タイミングによりフラグ不一致でクロスフェード開始漏れ | 条件式を統合。`preparedCrossfade.pending` を `hasPendingCrossfade` に確実反映 | ✅ 対応済み（確認 2026-05-17） |
| **H2** | `src/NoiseShaperLearner.cpp` | `workerThreadMain()` 内の `jthread` 生成 | `std::stop_token` を無視し独自フラグでポーリング。RAIIによる自動停止が機能せず | `stop_token` をループ条件に正しく伝播。または `std::thread` に戻す | ✅ 修正済み（2026-05-17） |
| **H3** | `src/convolver/ConvolverProcessor.h` 他 | `mixedTransitionStartHz`, `activeCacheKey` 等多数 | Snapshot 以外の `std::atomic` 状態が多数残存。Source of Truth 分散、パラメータ競合 | 全 shadow atomic を廃止。`GlobalSnapshot` のみを唯一の真理値源に | ✅ 対応済み（確認 2026-05-17） |
| **H4** | `src/convolver/ConvolverProcessor.cpp` 他 | `setMixedTransitionStartHz()` 等の setter | setter 内で `requestDebouncedRebuild()` を直接呼び出し。mutable live runtime model 残骸 | rebuild 要求を全て `enqueueSnapshotCommand()` 経由に統一。`requestDebouncedRebuild` は廃止 | ✅ 対応済み（確認 2026-05-17） |
| **H5** | `src/convolver/ConvolverProcessor.cpp` 他 | `setMix()` 等の setter | `listeners.call(&Listener::convolverParamsChanged, this)` を直接呼び出し。snapshot boundary violation | リスナーはUI更新のみ担当。パラメータ変更通知はキューイングまたは一括適用 | ✅ 修正済み（2026-05-17） |

---

## 🟢 Medium（品質向上・稀なリスク）

| No | ファイル | 行（目安） | バグ内容 | 修正方向性 | 状態 |
| --- | --- | --- | --- | --- | --- |
| **M1** | `src/DeferredDeletionQueue.h` | `enqueue()` 内の `compare_exchange_weak` | 非RTスレッド専用だが、shutdown時の `reclaimAllIgnoringEpoch()` 呼び出しタイミング次第でrace可能性 | shutdown sequence の厳格化。デストラクタで確実に `reclaimAllIgnoringEpoch()` を呼ぶ | ✅ 対応済み（確認 2026-05-17） |
| **M2** | `src/PsychoacousticDither.h` | `popUniformFromRing()` | リングバッファ枯渇時フォールバック乱数に切替。統計的偏りや可聴ノイズ（クリック）のリスク（稀） | 非RTスレッドで `refillRandomRingNonRt()` を定期呼び出し。アンダーフロー監視 | ✅ 修正済み（2026-05-17） |
| **M3** | `src/audioengine/AudioEngine.Processing.*.cpp` | 各所の `diagLog` | RTスレッドに混入した場合、`juce::Logger::writeToLog` の mutex/heap/I/O でXRUN発生リスク | RTパスからのロギングを完全除去。デバッグ情報は非RTスレッドで集約 | ✅ 対応済み（確認 2026-05-17） |

---

## ⚪ Low（最適化・冗長コード・軽微な改善）

| No | ファイル | 行（目安） | バグ内容 | 修正方向性 | 状態 |
| --- | --- | --- | --- | --- | --- |
| **L1** | `src/AllpassDesigner.cpp` | `designWithCMAES()` 内の `sleep_for` | Workerスレッドの `sleep_for(1ms)`。実害はないが最適化収束の非決定性要因（軽微） | `yield()` のみに変更、または可変スリープに | ✅ 対応済み（確認 2026-05-17） |
| **L2** | `src/MKLNonUniformConvolver.cpp` | `processLayerBlock()` 内の `_mm_prefetch` | ブロック端数で不要プリフェッチ。キャッシュ汚染の可能性（極めて軽微） | 条件を `(i + 8) < numSamples` に厳格化 | ✅ 対応済み（確認 2026-05-17） |
| **L3** | `src/audioengine/AudioEngine.Processing.PrepareToPlay.cpp` | `prepareToPlay()` 内の `latencyBuf` 確保 | 毎回 `aligned_free` + `makeAlignedArray` を実行。メモリ断片化・確保コスト | サイズ変更時のみ再確保する条件分岐を追加 | ✅ 対応済み（確認 2026-05-17） |
| **L4** | `src/SpectrumAnalyzerComponent.cpp` | `timerCallback()` 内の `eqDataDirty` | `lastEqHash` と `lastDirectEqHash` の二重チェック。冗長な分岐 | `lastDirectEqHash` 関連を削除 | ✅ 対応済み（確認 2026-05-17） |
| **L5** | `src/eqprocessor/EQProcessor.h` | `activeBandNodes` | `RefCountedDeferred` + epoch の二重ライフタイムモデル。複雑だが現状は動作可能 | リファクタリング時に単一方式（epoch only）に統一 | ✅ 修正済み（2026-05-17） |
| **L6** | `src/audioengine/AudioEngine.Globals.cpp` | `g_currentEpoch` | `EpochManager` とは別にグローバルエポックが存在。二重管理 | `EpochManager` に統合 | ✅ 対応済み（確認 2026-05-17） |

---

## 📊 重要度別件数サマリ（修正版）

| 重要度 | 件数 |
| ------ | ---- |
| 🔴 Critical | 4 |
| 🟡 High | 5 |
| 🟢 Medium | 3 |
| ⚪ Low | 6 |
| **合計** | **18** |

---

## 🎯 最優先修正項目（推奨順序）

| 順位 | No | 理由 |
| ---- | --- | ---- |
| 1 | **C4** | RTスレッドでのCAS retry loop → 非決定性、最悪停止の可能性 |
| 2 | **C1** | JUCE非同期コールバックの典型的UAF。実クラッシュしやすい |
| 3 | **C3** | CMA-ES NaN伝播 → optimizer完全死亡。音響最適化で発生しやすい |
| 4 | **H3** | shadow atomic → Snapshot architectureの根幹を崩す |
| 5 | **H4** | 直接rebuild要求 → Immutable runtime化の阻害要因 |
| 6 | **C2** | RCUガード順序 → 理論的UAF。上位ガードで実害稀だが規約違反 |
| 7 | **H5** | リスナー直接呼び出し → リエントランシー・境界崩壊リスク |

上記を優先的に修正することで、クラッシュ耐性・リアルタイム性・アーキテクチャ整合性が大幅に向上する。
