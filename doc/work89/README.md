# ConvoPeq バグ発見レポート — 成果サマリ (work89)

## 探索範囲
探索日: 2026-07-26

### カバーしたファイル（work89 新規探索）
- SnapshotFactory (.h/.cpp) — 詳細分析（ハッシュロジック、等価判定）
- SnapshotAssembler (.h/.cpp) — 初回読了
- EQCoeffCache / EQProcessor.ProcessingCache.cpp — 詳細分析
- AudioEngine.Cache.cpp — 全容読了
- EQProcessor.Processing.cpp — 高速パス（process 3引数版）全容読了
- AudioEngine.Snapshot.cpp — 再読了（スナップショット作成パス全体追跡）
- AudioEngine.Processing.PrepareToPlay.cpp — 全容読了
- AudioEngine.Processing.DSPCoreDouble.cpp — 処理パス追跡（eqCacheToUse 経路）
- AudioEngine.h — buildAudioThreadProcessingState 追跡（eqCache 解決経路）
- SnapshotCoordinator (.h/.cpp) — 全容再確認
- SnapshotFadeState.h — 全容再確認
- SnapshotSlotStore.h — 全容読了
- ObservedRuntime.h — 再確認
- RuntimeReaderContext.h — 再確認
- RefCountedDeferred.h — 全容読了
- RuntimeBuildTypes.h — 全容確認
- CustomInputOversampler.cpp — processUp/processDown 全容確認

### 前回からの継続確認
- work88 全バグ (BUG-011〜BUG-046): 修正されていないことを確認

### 今回（後半）カバーしたファイル（第2ラウンド）
- core/EpochDomain.h — 全容徹底分析（メモリ順序 HB 検証、QLock 不変条件、quarantine ロジック）
- core/RCUReader.h — 初回読了（ネスト深度管理、ownerToken CAS、slot acquire/release）
- SafeStateSwapper.h — 再読了（2-step bump 完全性検証、tryReclaim ローテーションパス分析）
- DeferredDeletionQueue.h — 再読了（Vyukov MPMC 動作検証）
- DeferredFreeThread.h — 再読了（ライフサイクル検証）
- audioengine/ISRRetireRouter.h — 全体確認（デリゲート構造）
- audioengine/AtomicAccess.h — 全容確認（ラッパー一貫性検証）

### 今回（第3ラウンド）カバーしたファイル
- DSPTransition.h — 全容読了（publish 完了後の activate/crossfade/retire フロー、transition complete パス）
- ConvolverProcessor.LoadPipeline.cpp — 全容読了（loadIR, applyComputedIR, load/commit pipeline）
- RuntimePublicationOrchestrator.cpp/.h — 全容読了（trySubmit, spec-building, deferred publish）
- PublicationExecutor.cpp/.h — 読了（publish → commitRuntimePublication 連携）
- DSPLifetimeManager.h — 読了（activate/retire authority）
- ISRDSPHandle.h/.cpp — 全容読了（DSPHandle レジストリ、state 遷移、crossfade tracking）
- AudioEngine.Learning.cpp — 全容読了（learning command processing, coeff bank管理）
- AudioEngine.h — 関連部分再読了（commitRuntimePublication, exchangeFadingRuntimeDSP, registerDSPHandleForRuntime）

### 今回（第4ラウンド）カバーしたファイル
- CrossfadeAuthority.h/.cpp — 全容読了（crossfade 判定ロジック、dspProjection 値のみで判断）
- CrossfadeRuntime.h — 全容読了（crossfade 実行状態管理、gain ramp, SPSC completed event queue）
- AudioEngine.Commit.cpp — 全容読了（runPublicationPrecheckNonRt, onRuntimePublishedNonRt, onRuntimeRetiredNonRt）
- AudioEngine.Transition.cpp — 読了（publishIdleWorldOnly）
- SnapshotFadeState.h — 全容再確認（全メンバ atomic、正しい seqlock pattern）
- RuntimeBuilder.cpp — 一部読了（createBootstrapWorld, buildRuntimePublishWorld）
- ConvolverState.h — 全容確認（軽量化メタデータのみ、バグなし）
- FrozenRuntimeWorld.cpp — 確認完了

### グラフ分析（graphify）
- ナレッジグラフ構築完了: 7,043 nodes, 11,657 edges, 321 communities
- 主要クロスコミュニティ接続:
  - Community 0 (ConvolverProcessor) ↔ 128 (DeferredFreeThread): 1-hop direct reference
  - Community 39 (EpochDomain) ↔ 128 (DeferredFreeThread): 2-hop via ConvolverProcessor
  - Community 101 (RCUReader) ↔ 39 (EpochDomain): 2-hop via AudioEngine.h
  - Community 36 (ISRRetireRouter) ↔ 106 (IEpochProvider): delegation layer
  - Community 36 ↔ 198 (StuckReaderInfo): shared diagnostic data type

## 発見バグ: 9件（今回、通算）

| ID | ファイル | タイトル | 重大度 |
|----|---------|---------|--------|
| BUG-047 | EQProcessor.ProcessingCache.cpp:24-44 | EQCoeffCache ハッシュに sampleRate が含まれず、サンプルレート変更後に古い EQ 係数が使われる | **HIGH** |
| BUG-048 | core/EpochDomain.h:397-473 | detectStuckReaders 最初の一致で break し重度 Stuck Reader を見逃す | LOW |
| BUG-049 | core/EpochDomain.h:255-281 | quarantineFlags への並行 store 競合（即時隔離 vs 遅延隔離） | MEDIUM |
| BUG-050 | core/EpochDomain.h:106-129 | enterReader の HB 順序 — epoch store が depth++ の後で stale epoch 読み取りの可能性 | MEDIUM |
| BUG-051 | DSPTransition.h:93-94,126-127 | exchangeFadingRuntimeDSP sentinel mask デッドコード（常に false） | LOW |
| BUG-052 | RuntimePublicationOrchestrator.h:consumeDeferredRequest | デッドコードの non-atomic アクセス（呼び出し元不在） | LOW |
| BUG-053 | AudioEngine.Learning.cpp:40-62 | stopNoiseShaperLearning で queue 投入＋直接 stopLearning() の二重呼び出し | MEDIUM |
| BUG-054 | DSPTransition.h:81-88 | onPublishCompleted の crossfade handle 不一致 — commitRuntimePublication の activate 後で oldHandle が newDSP を指す | MEDIUM |
| BUG-055 | AudioEngine.Commit.cpp:221-224 | runPublicationPrecheckNonRt の到達不能な else if 分岐（hasFadingRuntime 削除時のクリーンアップ漏れ） | LOW |

## 調査済み（バグ確定せず）

### SnapshotFactory ハッシュロジック
- `computeContentHash()` は全パラメータを網羅（ただし generation は意図的に除外 — no-op 最適化の正常動作）
- `areSnapshotsEquivalent()` は generation 以外の全パラメータを epsilon 比較
- ハッシュと等価判定の二段構えは正しい

### EQCoeffCache ライフタイム管理
- `RefCountedDeferred` による参照カウント：正常
- `CacheMap` のコピーコンストラクタで `addRef()`：正常
- `CacheMap` のデストラクタで `release()` または `releaseDirect()`：正常
- Audio Thread は Raw Pointer を使用するが、`CacheMap` のコピー＆リリース機構により参照カウントが常に 1 以上に保たれる：**安全**
- `releaseCache()` は **未使用のデッドコード**（ただし機能バグではない）

### フェードシステム再確認
- `advanceFade` は `AudioEngine.Processing.AudioBlock.cpp:475` で配線済み ✅
- `updateFade` は依然スタブ（`updateAudioThreadSnapshotFade` 未実装、BUG-031）
- `resetFadeStateAndRetireTarget` の `publishEpoch()` は Audio Thread からは未呼び出し（updateFade 自体がスタブのため）
- `completeFade` → `enqueueWithRetry` で `tryReclaim` を使用するが、Timer (NonRT) からのみ呼ばれるため問題なし

### クロスフェード RuntimeDSPSlot
- `exchangeFadingRuntimeDSP` の競合は BUG-029/030 として既報告
- 現状、`buildAudioThreadProcessingState` での `eqCache` 解決パスに改善点なし

### LoadPipeline の sampleRate 一貫性
- `applyComputedIR` と `executePendingCommit` の両方で `currentSampleRate` を publish しているが、両者とも Message Thread 上の writer であるため競合なし
- sampleRate 変更時の generation mismatch チェックで不正な IR は拒否される

### RuntimePublicationOrchestrator publish フロー
- commitRuntimePublication → onPublishCompleted の sequence は正しい（activate 後の DSPTransition 呼び出し）
- oldDSP resolve は事前に行われ DSPCore* ポインタで渡されるため、audio crossfade は正しく動作する
- 問題は DSPHandle ベースの tracking のみ（BUG-054）

### AdaptiveCoeffBank seqlock
- `getCurrentAdaptiveCoefficients` の retry loop (generation前後比較) は正しい seqlock pattern
- `selectAdaptiveCoeffBankForCurrentSettings` の TOCTOU は「best effort」設計上許容範囲

## 重要な所見

### RCU 実装の健全性（第2ラウンド発見）
1. **SafeStateSwapper の RCU は完全に正しい**: 2-step bump + epoch-only 設計によりメモリ順序のギャップがない。getMinReaderEpoch の minEpoch 計算も currentEpoch を起点とする正しい設計。
2. **EpochDomain の HB ギャップ（BUG-050）**: depth++ → epoch store の2段階で、release-acquire synchronize が epoch まで及ばない。しかし enterReader 中の reader は未だポインタを取得していないため UAF には至らない。SafeStateSwapper の単一段階設計が優れている。
3. **quarantineFlags の並行 store 競合（BUG-049）**: 2 Coordinator の同時 quarantine で 8-bit store が競合しうる。Debug ビルドでは assert 発火。Release では depth==0 チェックがクッションになる。
4. **DeferredDeletionQueue の Vyukov MPMC**: 正しい実装。`scanned` 変数はデッドコード（常に 0）。
5. **RCUReader の ownerToken CAS**: 正しいマルチスレッド排除設計。exit() の owner チェックは TOCTOU の懸念があるが、single-thread 契約下では問題なし。

### DSPHandle crossfade tracking の問題（第3ラウンド発見 — BUG-054）
- `commitRuntimePublication()` が publish 成功後に `dspHandleRuntime_.activate(rollbackHandle)` を呼び、`activeRuntimeDSPHandle_` を newDSP の handle に設定する。
- その後 `DSPTransition::onPublishCompleted()` が `getActiveRuntimeDSPHandle()` を呼ぶが、既に newDSP の handle が返る。
- Crossfade が newDSP→newDSP（同一 DSP）として登録され、oldDSP の handle は一切 crossfade tracking に関与しない。
- **実際のオーディオクロスフェードは DSPCore* ポインタで正しく動作する**ため、音声品質への影響はなし。

### LoadPipeline 信頼性
- `currentSampleRate` の TOCTOU 懸念は Message Thread 単一 writer により無効
- Progressive upgrade path は正しい（lowRes から target へ段階的 FFT upgrade）
- 終了スレッドの cleanup は `loaderTrashBin` で適切に管理

### CrossfadeAuthority 設計の健全性（第4ラウンド）
- `evaluate()` は dspProjection 投影値 + 静的 Policy のみで判断し、HealthState など実行時状態に依存しない — 設計通り
- `pending_` release/acquire による LinearRamp 保護: `start()` → `pending_=true` (release) → Audio Thread が `pending_` (acquire) → `gain_.getNextValue()` の HB 順序が成立
- `CrossfadeRuntime::gain_` と `LinearRamp` の NonRT/AudioThread 間 concurrent access は `pending_` の release/acquire により保護されている（表面化しない潜在リスクもなし）

### AudioEngine.Commit.cpp 検証ロジック（第4ラウンド）
- `validateSemanticCompleteness()`: generation, sequence, epoch, version の一貫性チェック — 正常
- `validateRuntimeGraphAuthorityContract()`: activeNode/fadingNode と topology UUID の一致検証 — 正常
- `runPublicationPrecheckNonRt()`: Semantic Transaction State 機械、shutdown guard、monotonicity、closure validation — 正常動作
- BUG-055: 唯一の問題は `else if` デッドコード

### 総括
1. **BUG-047** はサンプルレート変更後に EQ フィルタ周波数が設計値から乖離する確実なバグ。発現条件が「サンプルレート変更」であり、頻度は低いが影響は広範囲（全 20 バンドの SVF カットオフが不正確になる）。
2. **BUG-054** は DSPHandle ベースの crossfade tracking を破壊するが、実際のオーディオ品質に影響しない。診断・デバッグ情報の正確性に影響。
3. **BUG-053** は `stopNoiseShaperLearning` で `stopLearning()` が無駄に二重呼び出しされる。NoiseShaperLearner 側の再入可能性に依存。
4. **BUG-055** は到達不能分岐のデッドコード。実害なし。
5. **EQCacheManager の lifecycle は堅牢**: RefCountedDeferred + CacheMap コピーセマンティクスにより、Audio Thread 参照中の EQCoeffCache が解放されることはない。
6. **SafeStateSwapper の RCU は堅牢**: 2つの独立した RCU 実装（SafeStateSwapper, EpochDomain）の独立検証により、いずれも UAF に至るロジック上のバグは存在しないことを確認。
7. 第4ラウンドでは新規領域（CrossfadeAuthority、Commit検証パイプライン、SnapshotFadeState）をスキャンし、1件の LOW バグ（デッドコード）を発見。コード品質は高く、未発見の重大バグの可能性は低い。
