# ConvoPeq: Immutable Snapshot Runtime 詳細実装計画（rule2準拠）

作成日: 2026-05-17

---

## 1. 前提と強制制約

- 準拠規約: `doc/rule2.md`（IR-1〜IR-7、章3/4/7/8/12/13/15を最優先）
- 上位計画: `doc/plan2.md` の Phase 順序を厳守
- 編集禁止: `JUCE/**`, `r8brain-free-src/**` は変更しない
- Audio Thread で禁止:
  - runtime mutation
  - `RuntimeCommandQueue` drain / command execution
  - RMW atomic (`exchange/compare_exchange/fetch_add` 等)
  - lock/alloc/logging/libm/UIアクセス

---

## 2. 現状コードへの対応マップ（主要）

### 2.1 直近で除去すべき mutable runtime 要素

- `RuntimeCommandQueue`
  - 定義: `src/audioengine/RuntimeCommandQueue.h`
  - 保持: `src/audioengine/AudioEngine.h` (`m_runtimeCommandQueue`)
  - 実行: `src/audioengine/AudioEngine.Processing.AudioBlock.cpp` (`processAudioThreadRuntimeCommands`)
  - 供給: `src/audioengine/AudioEngine.RebuildDispatch.cpp` (`m_runtimeCommandQueue.enqueue`, `drainCoalesced`)
- `DSPExecutionState`
  - 定義: `src/audioengine/DSPExecutionState.h`
  - 保持: `src/audioengine/AudioEngine.h` (`dspExecutionStateCurrent/Fading`)
  - 使用: `AudioEngine.Processing.*`, `ConvolverProcessor.*`, `EQProcessor.*`
- sync API
  - `syncEqAgcTableViewFromRuntimeGraph` in `AudioEngine.h`
- currentXXX atomic cache
  - `m_current*` 群 in `AudioEngine.h`（SoT重複）
- snapshot fade state machine
  - `src/core/SnapshotCoordinator.h/.cpp`（`m_fadeState`, `m_fadeAlpha`, `m_fadeRemainingSamples`, `m_fadeCompleted`）
- snapshot 更新の複線状態機械
  - `src/EQEditProcessor.h/.cpp`（`pendingSnapshot` + Timer debounce）
  - `src/core/WorkerThread.h/.cpp`（`pendingFlush`, `hasPending`, debounce集約）
- runtime近傍 cache/shadow
  - `src/audioengine/AudioEngine.Cache.cpp`（`EQCacheManager::CacheMap`）
  - `src/ConvolverProcessor.h`, `src/convolver/ConvolverProcessor.StateAndUI.cpp`（`cachedLatency`, `irCache` 等）
  - `src/eqprocessor/EQProcessor.*`（`m_rtBypassShadow`, `rtAgc*Shadow`, `rtActiveStructureShadow`）

### 2.3 追加監査で確定した要改修項目（既知6カテゴリ外）

- `SnapshotCoordinator` の mutable fade は `plan2` Phase 6 対象として明示的に除去する
- `EQEditProcessor + WorkerThread + CommandBuffer` の複線更新経路は単一路化する
- runtime近傍 cache/shadow は builder/materialize 側へ責務移管し、runtime側 mutable を削減する
- `CommitStaging/deferredCommitQueue` は single publish path へ収束するよう簡素化する

### 2.2 既存の移行資産（活用）

- `GlobalSnapshot` 系:
  - `src/core/GlobalSnapshot.h`
  - `src/core/SnapshotCoordinator.h`
- Runtime build:
  - `src/audioengine/RuntimeBuilder.h/.cpp`
- Runtime publish world（過渡）:
  - `AudioEngine::RuntimePublishWorld` in `AudioEngine.h`

---

## 3. 実装方針（rule2適合）

- SoT は `GlobalSnapshot` のみ（IR-4）
- publish 単位を `RuntimeWorld` 単一ポインタへ収束
- Audio Thread は `load(acquire)` → `process(const RuntimeWorld&)` のみ
- crossfade は mutable フラグではなく immutable transition plan で表現
- retire/reclaim は Epoch 契約を厳守（reader enter/exit 対称）

---

## 4. フェーズ別詳細計画（実装順固定）

> 順序は `rule2.md` 13章に従い固定。逆順・並列実施は禁止。

### Phase 0: ガードレール先行（変更前）

1. 監査用チェック追加/更新
   - `Strict Atomic Dot-Call Scan` を必須ゲート化
   - `AudioEngine` RT経路に `RuntimeCommandQueue`/`processAudioThreadRuntimeCommands`/`processV2` の参照禁止ルールを追加
2. 可視化
   - Runtime publish/reclaim カウンタに「旧経路利用回数」を追加してゼロ化を追跡

#### Phase 0 完了条件

- 旧経路検知ルールがCI/ローカルで再現可能

---

### Phase 1 (P0): RuntimeCommandQueue 全廃

1. UI API 切替
   - `AudioEngine::setConvolverMix/setConvolverSmoothingTime` を enqueue 方式から snapshot更新+rebuild要求へ置換
2. Audio Thread から command 実行を撤去
   - `getNextAudioBlock` 先頭の `processAudioThreadRuntimeCommands()` 呼び出し削除
   - `processAudioThreadRuntimeCommands()` 関数本体削除
3. Rebuild thread から queue 排水を撤去
   - `AudioEngine.RebuildDispatch.cpp` の `drainCoalesced` 依存除去
4. 構造体削除
   - `AudioEngine.h` の `m_runtimeCommandQueue` 削除
   - `RuntimeCommandQueue.h` / `RuntimeCommand.h` の段階的未使用化（最終Phaseで完全削除）

#### Phase 1 主編集ファイル

- `src/audioengine/AudioEngine.h`
- `src/audioengine/AudioEngine.Processing.AudioBlock.cpp`
- `src/audioengine/AudioEngine.RebuildDispatch.cpp`
- `src/ConvolverControlPanel.cpp`（必要に応じ API 接続調整）

#### Phase 1 完了条件

- Audio Thread 上で command queue の参照ゼロ

---

### Phase 2 (P0): runtime mutation setter/sync API 削除

1. `syncEqAgcTableViewFromRuntimeGraph` を廃止
   - AGC table は build時に `RuntimeWorld` へ固定化
2. RT setter 経路の削除
   - `setMixRT/setSmoothingTimeRT` を AudioThread経由で呼ばない
3. `RuntimeGraph` の「後追い同期のための可変参照」を整理

#### Phase 2 主編集ファイル

- `src/audioengine/AudioEngine.h`
- `src/audioengine/AudioEngine.Processing.AudioBlock.cpp`
- `src/audioengine/AudioEngine.Processing.BlockDouble.cpp`
- `src/audioengine/AudioEngine.Processing.Snapshot.cpp`

#### Phase 2 完了条件

- publish後sync API呼び出しゼロ（IR-1/4.3）

---

### Phase 3 (P0): DSPExecutionState 廃止 + process API 統一

1. API統一
   - `DSPCore::processV2/processDoubleV2` を `process/processDouble` に統一
   - 目標シグネチャ: `process(const RuntimeWorld&, AudioBlock...)`
2. `DSPExecutionState` 参照の段階的除去
   - `ConvolverProcessor::bindExecutionState`
   - `EQProcessor::bindExecutionState`
3. `DSPExecutionState.h` の責務分解
   - 実行時可変stateが必要なら `RuntimeWorld` 非依存のRTローカル最小stateへ限定

#### Phase 3 主編集ファイル

- `src/audioengine/AudioEngine.h`
- `src/audioengine/AudioEngine.Processing.*.cpp`
- `src/audioengine/DSPExecutionState.h`
- `src/convolver/ConvolverProcessor.Runtime.cpp`
- `src/eqprocessor/EQProcessor.Core.cpp`

#### Phase 3 完了条件

- `processV2` / `processDoubleV2` / `DSPExecutionState` 参照ゼロ

---

### Phase 4 (P1): immutable transition への移行

1. mutable crossfade フラグ撤去
   - `dspCrossfadePending` 等を transition plan publishへ集約
2. `EngineRuntime/RuntimeGraph` 2重表現を縮退
   - `PublishedRuntimeWorld`（active/fading/transition）へ一本化
3. Audio Thread は transition を read-only 実行のみ

#### Phase 4 主編集ファイル

- `src/audioengine/RuntimeTransition.h`
- `src/audioengine/AudioEngine.h`
- `src/audioengine/AudioEngine.Commit.cpp`
- `src/audioengine/AudioEngine.Processing.AudioBlock.cpp`
- `src/audioengine/AudioEngine.Processing.BlockDouble.cpp`

#### Phase 4 完了条件

- mutable fade state machine 消滅（IR-6）

> 注: ここでいう fade は **runtime crossfade** を指す。
> `SnapshotCoordinator` 側の fade state machine は Phase 6 で撤去する。

---

### Phase 5 (P1): currentXXX atomic cache 廃止（SoT一元化）

1. `m_current*` 系を `GlobalSnapshot` 参照へ置換
2. UI取得系は `UISnapshotView` 経由で取得
3. `AudioEngine.Init.cpp` などの debounce key 生成元を snapshot化

#### Phase 5 主編集ファイル

- `src/audioengine/AudioEngine.h`
- `src/audioengine/AudioEngine.Init.cpp`
- `src/audioengine/AudioEngine.StateIO.cpp`
- `src/audioengine/AudioEngine.Parameters.cpp`

#### Phase 5 完了条件

- SoT重複削除（IR-4）

---

### Phase 6 (P2): snapshot fade machine 廃止 + RuntimeBuilder 分離強化 + RuntimeWorld 完全統合

1. `SnapshotCoordinator` mutable state machine を撤去
   - `m_fadeState/m_fadeAlpha/m_fadeRemainingSamples/m_fadeCompleted` 依存を廃止
   - transition は `PublishedRuntimeWorld` 内 immutable plan へ統合
2. snapshot 更新経路の単一路化
   - `EQEditProcessor::pendingSnapshot` と `WorkerThread::pendingFlush/hasPending` の二重集約を整理
   - 「更新 → snapshot build → runtime build → publish」の一本道に収束
3. `RuntimeBuilder` 出力を `RuntimeWorld` 型へ昇格
4. publish は単一 atomic ptr のみ
5. build complete before publish を静的に担保（Builder完了判定）
6. runtime近傍 cache/shadow の責務移管
   - `cachedLatency/irCache/rt*Shadow` を runtime mutable として残さない構成へ段階移行
7. `CommitStaging/deferredCommitQueue` を single publish path 観点で簡素化

#### Phase 6 主編集ファイル

- `src/audioengine/RuntimeBuilder.h/.cpp`
- `src/audioengine/AudioEngine.Commit.cpp`
- `src/audioengine/AudioEngine.Snapshot.cpp`
- `src/audioengine/RuntimeGraph.h`（最終的に統合/縮小）
- `src/core/SnapshotCoordinator.h/.cpp`
- `src/core/WorkerThread.h/.cpp`
- `src/EQEditProcessor.h/.cpp`
- `src/convolver/ConvolverProcessor.StateAndUI.cpp`
- `src/eqprocessor/EQProcessor.Processing.cpp`

#### Phase 6 完了条件

- `currentWorld.load(acquire)` 以外で runtime 実体参照しない
- `SnapshotCoordinator` の mutable fade フィールド運用が消滅
- snapshot 更新が単一路（複線 pending/debounce なし）

---

### Phase 7 (P3): Legacy cleanup

1. 未使用型/関数/フラグ削除
   - `RuntimeCommandQueue.h`, `RuntimeCommand.h`, `RuntimeGraph.h`（不要部分）
   - `CommitStaging/deferredCommitQueue` で不要化した補助状態
   - snapshot複線化で不要になった pending/debounce フラグ
2. ドキュメント同期
   - `ARCHITECTURE.md`, `doc/*` を最終構成へ更新

#### Phase 7 完了条件

- 旧mutable runtimeレイヤ完全撤去

---

## 5. 変更管理単位（推奨PR分割）

- PR1: Phase0 + Phase1
- PR2: Phase2
- PR3: Phase3（大）
- PR4: Phase4
- PR5: Phase5
- PR6: Phase6
- PR7: Phase7 + 文書更新

各PRは「1目的・可逆」を維持する。

---

## 6. 検証計画（各PR共通）

1. Build
   - `Release`
   - `Debug`
2. 静的/規約
   - `Strict Atomic Dot-Call Scan`
3. 重点コードレビュー観点
   - IR-1: publish後 mutate の有無
   - IR-2/3: Audio Thread read-only か
   - IR-4: SoT 重複が残っていないか
   - IR-6: transition mutable state 排除
   - SnapshotCoordinator の mutable fade state 排除
   - snapshot build 経路が単一路か（EQEditProcessor/WorkerThread 重複なし）
   - IR-7/8章: retire/reclaim 順序、reader対称性
4. 回帰
   - IRロード直後の音切れ/ピッチずれ/クロスフェード破綻
   - 学習機能（NoiseShaperLearner）開始/停止/再開

---

## 7. リスクと先回り対策

- リスク: Phase3（`DSPExecutionState` 廃止）で影響範囲が広い
  - 対策: `ConvolverProcessor` と `EQProcessor` を先に adapter 化して段階削除
- リスク: transition統合でクリックノイズ
  - 対策: 旧挙動比較ログを一時併走し、遷移差分を可視化
- リスク: retire漏れ/UAF
  - 対策: retire経路を1箇所に集約し、epoch drain テスト追加

---

## 8. 最終受け入れ条件

- `RuntimeCommandQueue` 非存在
- `DSPExecutionState` 非存在
- `processV2` 非存在
- `sync*FromRuntime*` 非存在
- `m_current*` など SoT重複非存在
- snapshot fade machine 非存在（`SnapshotCoordinator` mutable state運用なし）
- Audio Thread で runtime mutation 非存在
- publish単位が `RuntimeWorld` 単一ポインタ
- `rule2.md` 14章の完成条件を全て満たす
