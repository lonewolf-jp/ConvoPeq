# Mutableコード再監査レポート（plan2/rule2準拠観点）

作成日: 2026-05-17
対象: `src/**`（`JUCE/**`, `r8brain-free-src/**` は除外）

## 1. 監査方法

以下の観点で全体検索 + 主要ファイル精読を実施。

- Runtime mutation 経路
  - `RuntimeCommandQueue`, `processAudioThreadRuntimeCommands`, `EngineCommand`
- mutable実行状態
  - `DSPExecutionState`, `processV2/processDoubleV2`, `bindExecutionState`
- post-publish sync
  - `syncEqAgcTableViewFromRuntimeGraph`, `syncStateFrom`, `syncGlobalStateFrom`
- Source of Truth 重複
  - `m_current*` 系 atomic
- mutable transition state machine
  - `dspCrossfadePending`, `firstIrDryCrossfade*`, `runtimeTransitionState`
- rule2 12.2 禁止語系
  - `runtimeShadow`, `shadow`, `currentValue`, `cachedValue`, `incremental`

---

## 2. 主要結論（漏れ有無）

結論: **改修対象のmutableコードは複数残存**。`plan2.md`/`rule2.md`の到達条件には未達。

特に P0 ブロッカーは以下。

1. RuntimeCommandQueue経路が現存（UI→RT command実行）
2. `DSPExecutionState` + `processV2`系が現存
3. `syncEqAgcTableViewFromRuntimeGraph` が現存（post-publish sync）
4. `m_current*` 系のSoT重複が現存
5. mutable crossfade state machine が現存

---

## 3. 証跡（カテゴリ別）

## 3.1 RuntimeCommandQueue 経路（P0）

- `src/audioengine/AudioEngine.h`
  - L70: `#include "RuntimeCommandQueue.h"`
  - L721/L724, L729/L732: `setConvolverMix/setConvolverSmoothingTime` が `m_runtimeCommandQueue.enqueue(...)`
  - L2667: `processAudioThreadRuntimeCommands()` 宣言
  - L2683: `m_runtimeCommandQueue` 保持
- `src/audioengine/AudioEngine.Processing.AudioBlock.cpp`
  - L26: `getNextAudioBlock` 冒頭で `processAudioThreadRuntimeCommands()` 呼出
  - L266-L270: queue drain 実装（`tryDequeue` ループ）
- `src/audioengine/AudioEngine.RebuildDispatch.cpp`
  - L286: `m_runtimeCommandQueue.enqueue(runtimeCommand)`
  - L389: `drainCoalesced(...)`
- `src/audioengine/RuntimeCommandQueue.h`
  - queue実装本体（`enqueue/tryDequeue/drainCoalesced`）

評価: rule2 3.1/3.3, plan2 Phase1 に未達。

---

## 3.2 DSPExecutionState + processV2 系（P0）

- `src/audioengine/DSPExecutionState.h`
  - `struct DSPExecutionState` が現存
- `src/audioengine/AudioEngine.h`
  - `processV2/processDoubleV2` 宣言
  - `dspExecutionStateCurrent/Fading` メンバ
- `src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp`
  - `DSPCore::processV2(...)` 実装
  - `convolverRt().bindExecutionState(&executionState)`
  - `eqRt().bindExecutionState(&executionState)`
- `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp`
  - `processDoubleV2(...)` 同様
- `src/convolver/ConvolverProcessor.Runtime.cpp`
  - `bindExecutionState(...)` 実装
- `src/eqprocessor/EQProcessor.Core.cpp`
  - `bindExecutionState(...)` 実装

評価: plan2 Phase2/Phase3, rule2 7.1/7.2 に未達。

---

## 3.3 post-publish sync API 残存（P0）

- `src/audioengine/AudioEngine.h`
  - `syncEqAgcTableViewFromRuntimeGraph(...)` 定義
- 呼び出し
  - `src/audioengine/AudioEngine.Processing.AudioBlock.cpp` (複数箇所)
  - `src/audioengine/AudioEngine.Processing.BlockDouble.cpp` (複数箇所)
  - `src/audioengine/AudioEngine.Processing.Snapshot.cpp`

評価: rule2 IR-1/4.3, plan2 Phase3 に未達。

---

## 3.4 SoT重複（m_current* 系）残存（P1）

### 保持

- `src/audioengine/AudioEngine.h`
  - L2695-L2706: `m_currentInputHeadroomDb`, `m_currentEqBypass`, `m_currentOversamplingType` 等

### 実利用（write/read）

- `src/audioengine/AudioEngine.Parameters.cpp`
  - 複数 setter で `m_current*` に publish
- `src/audioengine/AudioEngine.Init.cpp`
  - debounce key 生成で `m_current*` を参照
- `src/audioengine/AudioEngine.Snapshot.cpp`
  - `createSnapshotFromCurrentState` が `m_current*` から Snapshot組立

評価: rule2 IR-4（SoT単一）に未達、plan2 Phase4/5 対象。

---

## 3.5 mutable transition state machine 残存（P1）

- `src/audioengine/AudioEngine.h`
  - `dspCrossfadePending`, `firstIrDryCrossfadePending`, `firstIrDryCrossfadeDone`, `dspCrossfadeUseDryAsOld`
  - `runtimeTransitionState` へ代入更新
  - `publishRuntimeSnapshots(...)` + `publishRuntimeTransitionState(...)` + 補助フラグ更新
- `src/audioengine/RuntimeGraph.h`
  - `dspCrossfadePending`, `dspCrossfadeUseDryAsOld` を保持
- `src/audioengine/RuntimeTransition.h`
  - `EngineRuntime` に同等フラグ保持

評価: rule2 IR-6, plan2 Phase5/6 未達。

---

## 3.6 追加で要注意なmutable/sync/shadow（優先度: 中）

### Convolver

- `src/ConvolverProcessor.h`
  - `syncStateFrom(...)` 公開API
  - `beginIncrementalRebuild/advanceIncrementalRebuild` 系
- `src/convolver/ConvolverProcessor.StateAndUI.cpp`
  - `syncStateFrom` が他インスタンス状態を取り込み + engine swap

### EQ

- `src/eqprocessor/EQProcessor.h/.Core.cpp/.Processing.cpp`
  - `syncStateFrom`, `syncGlobalStateFrom`
  - `m_rtBypassShadow`, `rtAgc*Shadow`, `rtActiveStructureShadow` など shadow state

評価: rule2 12.2（shadow/cache追加禁止）観点で、最終形に残すか再設計判断が必要。

---

## 4. 誤検知として除外したもの

- debug/diagnostics用カウンタの `fetch_add` は多数存在するが、
  これは「mutable runtime設計漏れ」とは別問題（RT strictness監査項目）。
- `currentSampleRate/currentProcessingOrder` など `current*` 命名でも、`m_current*` SoT重複群とは性質が異なるものは分離評価。

---

## 5. 影響評価（優先度）

- **P0（直ちに対応）**
  1) RuntimeCommandQueue廃止
  2) processV2/DSPExecutionState廃止
  3) syncEqAgcTableViewFromRuntimeGraph廃止
- **P1（続行）**
  4) m_current* 廃止（GlobalSnapshot一本化）
  5) mutable transition state machine 廃止
- **P2（最終統合）**
  6) Convolver/EQ の sync/shadow 設計を RuntimeWorld 方針に統合

---

## 6. 監査結論

「改修対象のmutableコードの漏れがないか」という問いに対する結論は、
**漏れはまだ多数存在する（未改修箇所あり）**。

特に `AudioEngine` 周辺は、`plan2` の Phase1〜5 を順守して実装すれば、
今回抽出した主要残存点を段階的に解消可能。

---

## 7. 既知6カテゴリ以外の追加mutable候補（今回追加検出）

以下は、ユーザー指定の既知対象（Queue/Sync/V2/DSPExecutionState/currentXXX/crossfade mutable）以外で、
`rule2.md`（特に IR-1/IR-4/IR-6/12.2）照合上、改修検討が必要な候補。

### 7.1 SnapshotCoordinator の mutable fade state（追加P1）

- `src/core/SnapshotCoordinator.h`
  - `m_target`, `m_fadeAlpha`, `m_fadeState`, `m_fadeRemainingSamples`, `m_fadeCompleted` を atomic で保持
- `src/core/SnapshotCoordinator.cpp`
  - `startFade/advanceFade/tryCompleteFade/completeFade` で遷移状態を逐次更新

評価:

`rule2` の「mutable transition 禁止（IR-6）」に照らすと、
snapshot遷移も最終的には immutable transition plan への統合対象。

### 7.2 EQ/Convolver の runtime cache・cachedValue 系（追加P1）

- `src/audioengine/AudioEngine.Cache.cpp`
  - `EQCacheManager::CacheMap` を交換・遅延破棄する mutable cache 運用
- `src/ConvolverProcessor.h`
  - `cachedLatency`, `irCache`, `maxCacheEntries`, `activeCacheKey` など runtime近傍キャッシュ
- `src/convolver/ConvolverProcessor.StateAndUI.cpp`
  - `cachedLatency` を `exchangeAtomic` で差し替え

評価:

`rule2 12.2`（`cachedValue`/mutable cache）観点で、
RuntimeWorld外キャッシュの役割分離（builder専用化 or snapshot内materialize）の再設計候補。

### 7.3 CommitStaging/deferredCommitQueue（追加P2）

- `src/audioengine/AudioEngine.h`
  - `CommitStaging`, `deferredCommitQueue`, `deferredCommitMutex`
- `src/audioengine/AudioEngine.Commit.cpp`
  - `prepareCommit/executeCommit` が staged commit を再投入・繰り延べ

評価:

非RT制御層の実装だが、publish前後の状態遷移が多段化しており、
「single publish path」へ収束させるなら簡素化余地あり。

### 7.4 EQEditProcessor + WorkerThread の debounce/pending 経路（追加P2）

- `src/EQEditProcessor.h/.cpp`
  - `pendingSnapshot` + Timer debounce + `enqueueSnapshotCommand()`
- `src/core/CommandBuffer.h`, `src/core/WorkerThread.cpp`
  - ParameterCommandキュー + `hasPending/latestCommandGeneration/pendingFlush`

評価:

非RT経路だが、snapshot生成の遅延・集約ロジックが別状態機械として存在。
最終形で「更新→snapshot build」の単一路を目指すなら統合対象。

### 7.5 Adaptive coeff bank の二重バンク/書込ロック状態（追加P2）

- `src/audioengine/AudioEngine.h`
  - `AdaptiveCoeffBankSlot` に `activeIndex/generation/writeLock/stateMutex` を保持
  - `currentAdaptiveCoeffBankIndex` でバンク切替
  - `CoeffSetWriteLockGuard`（CAS + generation更新）

評価:

学習機能専用でRT本線とは分離されるが、
SoT単一化方針（IR-4）と整合させるには snapshot/builder への接続設計を明確化すべき。

---

## 8. 追加監査の結論

既知6カテゴリ以外にも、**改修検討すべきmutable箇所は存在**する。

優先順は以下を推奨。

1. SnapshotCoordinator mutable fade state（transition統一の観点で先行）
2. runtime近傍 cache/cachedLatency/cachedInputRMS 群
3. commit/debounce/学習バンクの補助状態機械

---

## 9. 監査範囲拡張結果（「mutable検索した範囲外」への回答）

### 9.1 `src`外の独自C++コード有無

リポジトリ全体の C/C++ 拡張子列挙では、独自実装は実質 `src/**` に集約。

- 第三者コード: `JUCE/**`, `r8brain-free-src/**`（編集禁止対象）
- 生成物/ビルド成果物: `build/**`（監査対象外）

したがって、「前回の`src`限定監査の外側」に独自mutable実装が潜む可能性は低く、
今回の追加監査は **`src/core/**` と未精読インフラ層**へ拡張して実施した。

### 9.2 `src/core/**` の追加判定

#### 改修候補（本体）

- `src/core/SnapshotCoordinator.h/.cpp`
  - fade状態をatomicで保持し段階更新（`m_fadeState`, `m_fadeAlpha`, `m_fadeRemainingSamples`, `m_fadeCompleted`）
  - `startFade/advanceFade/tryCompleteFade/completeFade` の mutable state machine
  - `rule2 IR-6` 観点で immutable transition への統合候補

- `src/core/WorkerThread.h/.cpp`
  - `pendingFlush` + `hasPending` + debounce window による更新集約状態
  - 既存の `EQEditProcessor::pendingSnapshot` と合わせ、snapshot生成の経路が複線化

#### 監視・基盤として除外（直ちに改修対象ではない）

- `src/core/SnapshotFactory.cpp`
  - `g_liveSnapshotCount` は `_DEBUG` 限定の生存数カウンタ
- `src/core/RCUReader.h`, `src/core/EpochCore.h`, `src/core/EpochManager.h`
  - reclamation契約のための同期状態（インフラ）
- `src/core/CommandBuffer.h`, `src/core/DeletionQueue.*`
  - 非RT制御経路の輸送/回収基盤

### 9.3 結論（範囲外調査の回答）

「mutable検索した範囲外でもmutable箇所があるか」に対しては、

- `src`外の独自実装には有意な追加候補は見つからず
- ただし `src/core/**`（前回相対的に薄かった領域）で、
  **SnapshotCoordinator/WorkerThread の mutable state machine** を追加検出

よって、改修計画上は以下を追補するのが妥当。

1. transition統一対象に `SnapshotCoordinator` を明示追加
2. snapshot生成経路（EQEditProcessor + WorkerThread）の単一路化方針を定義
