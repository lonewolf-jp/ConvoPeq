# 修正版詳細設計：Immutable Runtime Graph + Per-thread DSP State

> 基準文書：doc/basic_rule.md（移行計画）、doc/terms.md（実装規律）
> 精査文書：前回詳細設計レビュー（2026-05-12）
> 本文書は精査で発覚した11カテゴリの漏れ・違反をすべて反映した修正版である。

---

## 1. 設計原則（terms.md 再確認）

```
// Immutable after publish. Audio thread may access lock-free.
struct RuntimeGraph { ... };

// Thread-confined DSP execution state. Never shared across audio threads.
struct DSPExecutionState { ... };

// RT-safe:
// - no allocation
// - no locks
// - no ownership mutation
void process(const RuntimeGraph& g, DSPExecutionState& s, AudioBlock& b);
```

**最重大禁止事項（本設計で全て解消する）**：

| 禁止内容 | 現行違反箇所 | 解消フェーズ |
|---|---|---|
| 音声スレッドで `retireDSP()` を呼ぶ | `finalizeCrossfadeMixPath()` AudioEngine.h:1979 | Phase 6 |
| 音声スレッドで `retireDSP()` を呼ぶ | `cleanupCrossfadeDirectPath()` AudioEngine.h:2004 | Phase 6 |
| `thread_local` 依存 | `EQProcessor.Processing.cpp:15` | Phase 4 |
| DSPCore 内にスレッド付きオブジェクト | `PsychoacousticDither.rngProducerThread` | Phase 1 |

---

## 2. 最終目標構造

```
AudioEngine
 ├ atomic<const RuntimeGraph*>        ← 単一の graph publish
 ├ DSPExecutionStatePool              ← current/fading 2本分確保
 ├ BuilderThread                      ← graph 構築専用
 ├ EBRQueue / DeferredDeletionQueue   ← graph retire のみ
 └ RNGWorker                          ← PsychoacousticDither RNG補充（分離後）

RuntimeGraph  ← IMMUTABLE after publish
 ├ runtimeUuid / generation
 ├ sampleRate / ditherBitDepth / noiseShaperType
 ├ oversamplingFactor / oversamplingType / osNumStages
 ├ osStageCoeffs[n]   (FIRタップ、IMMUTABLE)
 ├ fixedNsCoeffs      (FixedNoiseShaper 係数プリセット)
 ├ fixed15NsCoeffs    (Fixed15TapNoiseShaper 係数プリセット)
 ├ outputFilterCoeffs (BiquadCoeff × 4)
 ├ IRBank             (shared_ptr<const IRBank>、FDL partition layout も含む)
 ├ EQCoeffBank        (coeffs / bandActive / modes / filterStructure)
 ├ eqAgcCoeffs        (SR毎係数テーブル)
 ├ eqTotalGain        (double)
 ├ processingOrder / eqBypassed / convBypassed
 ├ softClipEnabled / saturationAmount
 ├ inputHeadroomGain / outputMakeupGain / convInputTrimGain
 ├ filterModes (HC/LC/LPF)
 ├ adaptiveCoeffSnapshot  (const CoeffSet* raw, graph寿命に束縛)
 ├ adaptiveCoeffGeneration / adaptiveCoeffBankIndex
 ├ captureSessionId / captureQueueRef
 ├ fixedLatencySamples / maxSamplesPerBlock / maxInternalBlockSize
 └ transitionMetadata (fade時間・latencyDelta)

DSPExecutionState  ← DSP_THREAD_STATE, audio thread confined
 ├ runtimeUuid  (attachedGraphのUUID、互換性確認用)
 ├ conv          (ConvolverDSPState)
 │   ├ nucStates     (FDL partition history × partitions)
 │   ├ bypassDelayBuf[2][DELAY_SIZE]
 │   ├ dryBuf / oldDryBuf / wetBuf / smoothingBuf
 │   ├ crossfadeGain (LinearRamp)
 │   ├ mixSmoother   (LinearRamp)
 │   ├ latencySmoother / oldDelay
 │   └ latencyAlign  (writePos, delaySamples)
 ├ eq            (EQDSPState)
 │   ├ filterState[NUM_BANDS][2]   (SVF/biquad z)
 │   ├ agcGain / agcEnvInput / agcEnvOutput
 │   ├ smoothTotalGain
 │   ├ bypassFadeGain / bypassed
 │   ├ scratchBuf / dryBypassBuf
 │   └ parallelBufs / structureXfadeBufs
 ├ dc            (DCBlockerDSPState)
 │   ├ outputL/R / inputL/R / oversampledL/R  (UltraHighRateDCBlocker)
 ├ os            (OversamplingDSPState)
 │   ├ stages[n].upHistory / downHistory / centerDelayInput
 │   └ workBuffers / corruptionDetected
 ├ fixedNsState  (errors[][], writePos, rngState, needsReset)
 ├ fixed15NsState (同上)
 ├ adaptiveNsState (states[][], rngState, coeffs_copy)
 ├ ditherState   (shaperStateBuffer, rngRingRef → RNGWorkerから)
 ├ outputFilterState (BiquadState × 4)
 ├ ramp          (fadeInSamplesLeft, bypassFadeGain, bypassed)
 ├ history       (fixedLatBufL/R, writePos, softClipPrev)
 ├ crossfade     (gainRamp, dryScaleRamp, mixBufs)
 ├ latencyAlign  (bufs[4][size], writePos, delaySamples[2])
 ├ scratch       (alignedL/R, dryBypassBufL/R)
 └ captureQueueRef (void*, nullable)
```

---

## 3. Ownership Table（完全版）

| Object | Owner | Thread | Lifetime | Mutability |
|---|---|---|---|---|
| RuntimeGraph | RuntimeManager (Builder publish) | Builder構築→Audioread | generation単位、epoch後retire | immutable |
| DSPExecutionState[current] | AudioEngine | Audio only | device setup単位 | mutable, thread-confined |
| DSPExecutionState[fading] | AudioEngine | Audio only（crossfade期間） | crossfade終了まで | mutable, thread-confined |
| IRBank | Builder / CacheManager | Builder/Worker | graph世代を超えて共有可 | immutable |
| EQCoeffBank | Builder | Builder | graph単位 | immutable |
| OsStageCoeffs | Builder | Builder | graph単位 | immutable |
| AdaptiveCoeffBankSlot | AudioEngine | NoiseShaperLearner / AudioEngine | engine寿命 | WORKER_ONLY write, Audio read-only |
| RNGWorker（分離後） | AudioEngine | 専用ワーカー | engine寿命 | WORKER_ONLY |
| DeferredDeletionQueue | convo::global | Audio enqueue / Timer dequeue | engine寿命 | lock-free MPMC |

---

## 4. Mutation Table（完全版）

> 詳細は doc/phase0_member_classification.md を参照。

| 分類 | 所在 | Mutable | Thread |
|---|---|---|---|
| RuntimeGraph 全フィールド | graph | no | N/A（publish後変更禁止） |
| DSPExecutionState.conv.nucStates | state | yes | audio only |
| DSPExecutionState.eq.filterState | state | yes | audio only |
| DSPExecutionState.eq.agcGain/Env | state | yes | audio only |
| DSPExecutionState.os.stages.upHistory | state | yes | audio only |
| DSPExecutionState.fixedNsState.errors | state | yes | audio only |
| DSPExecutionState.adaptiveNsState.states | state | yes | audio only |
| DSPExecutionState.adaptiveNsState.coeffs | yes（coeffs更新パス） | audio / learner間で原子的切替 |
| DSPExecutionState.ditherState.shaperStateBuffer | state | yes | audio only |
| DSPExecutionState.crossfade.gainRamp | state | yes | audio only |
| DSPExecutionState.latencyAlign | state | yes | audio only |
| EQCoeffCache.parallelBufs（→要分離） | **分離後はstate** | yes | audio only |
| AdaptiveCoeffBankSlot.coeffSetA/B | WORKER_ONLY write | NoiseShaperLearner |
| EBRQueue.retired | non-RT enqueue/dequeue | Builder/Timer |

---

## 5. RT Safety Table（完全版）

| Function | RT Safe | 根拠 / 違反現状 |
|---|---|---|
| `process(graph, state, block)` | yes（最終形） | no alloc, no lock, no ownership mutation |
| `graph atomic pointer load` | yes | 単純 atomic raw read |
| `finalizeCrossfadeMixPath()` | **現行NO** | audio thread で `retireDSP()` を呼ぶ（Phase 6で修正） |
| `cleanupCrossfadeDirectPath()` | **現行NO** | 同上 |
| `EQProcessor::process()` 内 `tls_rcuReader` | **現行NO** | thread_local（Phase 4で修正） |
| `retireDSP()` → `retireObject()` → `EBRQueue.retire()` | non-RT | mutex lock を含む |
| `DeferredDeletionQueue.enqueue()` | yes | ロックフリー MPMC（音声スレッドからの安全な代替） |
| `DeferredDeletionQueue.reclaim()` | non-RT（Timer側） | Timerから呼ぶ |
| crossfade mix（state2本同時実行） | yes（Phase 6以降） | ownership不変、stateのみmutable |
| `IRBank::process()` | yes | immutable data への read-only アクセス |
| `AdaptiveCoeffBankSlot` coeffs切替（轻量パス） | yes（atomic swap後） | DSPExecutionState内coeffsコピーを更新 |

---

## 6. Destruction Table（完全版）

| Path | 破棄対象 | 実行スレッド | 安全条件 |
|---|---|---|---|
| crossfade完了（Phase 6以降） | old DSPExecutionState → reset or pool return | non-RT制御 | audio block境界後 |
| graph switch complete | old RuntimeGraph → **DeferredDeletionQueue.enqueue()** | **音声スレッドからenqueue、Timerでreclaim** ← B-1修正 | epoch到達後delete |
| queued graph discard | 未採用 RuntimeGraph → enqueue/retire | non-RT | current参照と独立確認済み |
| releaseResources | 全DSPExecutionState clear → graph retire | non-RT | callback停止後 |
| shutdown | 全graph/state final reclaim | non-RT | audio thread停止済み |
| PsychoacousticDither.rngProducerThread停止 | engine停止時 | RNGWorker stop | engine lifecycle |

**B-1（最重大違反）の修正設計**：

```cpp
// 修正前（現行 - 音声スレッドで retireDSP を呼ぶ: terms 1.4節違反）
inline void finalizeCrossfadeMixPath(bool resetDryScaleGain) noexcept {
    if (!dspCrossfadeGain.isSmoothing()) {
        if (auto* done = sanitizeRawPtr(fadingOutDSP.exchange(nullptr, ...)))
            retireDSP(done);  // ← RT UNSAFE: EBRQueue::retire() が mutex を取る
    }
}

// 修正後（Phase 6適用後 - DeferredDeletionQueue.enqueue() はロックフリー）
inline void finalizeCrossfadeMixPath(bool resetDryScaleGain) noexcept {
    if (!dspCrossfadeGain.isSmoothing()) {
        if (auto* done = sanitizeRawPtr(fadingOutDSP.exchange(nullptr, ...))) {
            // RT-safe: DeferredDeletionQueue は lock-free MPMC
            // enqueue に失敗した場合（Queue Full）は engineRetireOverflow フラグを立て
            // Timer callback で再処理する。
            const uint64_t epoch = EpochManager::instance().currentEpoch();
            if (!g_deletionQueue.enqueue(
                    done,
                    [](void* p) { delete static_cast<DSPCore*>(p); },
                    epoch)) {
                // fallback: Timer側で処理するための退避
                retireOverflowFlag.store(true, std::memory_order_release);
                retireOverflowPtr.store(done, std::memory_order_release);
            }
        }
    }
}
```

**H-1（thread_local RCUReader）の修正設計**：

```cpp
// 修正前（EQProcessor.Processing.cpp:15 - terms 3.3節違反）
static thread_local convo::RCUReader tls_rcuReader;

// 修正後：DSPExecutionState に明示的に保持（Phase 4）
struct EQDSPState {
    convo::RCUReader rcuReader;  // audio thread confined → thread_local 不要
    // ...
};
// process 呼び出し側：
// void process(const RuntimeGraph& g, DSPExecutionState& s, ...)
// 内部で s.eq.rcuReader を直接使用
```

---

## 7. Allocation Table（完全版）

| Allocation | Thread | Timing | メモリ要件 |
|---|---|---|---|
| RuntimeGraph 本体 | Builder | rebuild時 | new（non-RT） |
| IRBank / partition layout | Builder/Worker | IR更新時 | mkl_malloc 64byte align推奨 |
| DSPExecutionState × 2（current/fading） | AudioEngine | prepareToPlay / rebuild | mkl_malloc 64byte align必須（MKL使用箇所） |
| conv.nucStates（FDL history） | prepareToPlay | non-RT | mkl_malloc 64byte align |
| eq.filterState / scratchBuf 等 | prepareToPlay | non-RT | 64byte align（AVX2使用） |
| os.stages.upHistory / downHistory | prepareToPlay | non-RT | 通常 new または mkl_malloc |
| fixedNsState.errors / rngState 等 | prepareToPlay | non-RT | 固定サイズ、stack or 64byte align |
| ditherState.shaperStateBuffer | prepareToPlay | non-RT | mkl_malloc（MKL使用） |
| RNGWorker.rngRing | engine init | 一度のみ | mkl_malloc |
| crossfade.mixBufs | prepareToPlay | non-RT | mkl_malloc 64byte align |
| latencyAlign.bufs | prepareToPlay | non-RT | mkl_malloc 64byte align |
| scratch.alignedL/R | prepareToPlay | non-RT | mkl_malloc 64byte align |

**注記**：process() 中の allocation は **ゼロ** でなければならない。
EQCoeffCache.parallelBufs 群は現行 EQCoeffCache 内に存在するが、
Phase 4 で DSPExecutionState.eq.parallelBufs として prepareToPlay 時に確保する形に移行する。

---

## 8. Phase別実装計画（修正版）

### Phase 0（メンバ分類台帳 - 完了）

→ doc/phase0_member_classification.md

### Phase 1（型追加・PsychoacousticDither スレッド分離）

**対象ファイル**：

- `src/audioengine/RuntimeTransition.h` - RuntimeGraph 型定義追加
- `src/audioengine/AudioEngine.h` - DSPExecutionState 型定義追加
- `src/PsychoacousticDither.h/.cpp` - rngProducerThread を RNGWorker として切り出し
- `src/audioengine/AudioEngine.h` - RNGWorker メンバ追加

**作業内容**：

1. `RuntimeGraph` 構造体新設（§2の定義に従う）。まず空骨格のみ。
2. `DSPExecutionState` 構造体新設。まず空骨格のみ。
3. `PsychoacousticDither::rngProducerThread` を `AudioEngine::rngWorkerThread` として切り出す。
   - `shaperStateBuffer` / `scale` / `invScale` / `coeffs` は `PsychoacousticDither` 残置。
   - `rngRing` / `rngReadPos` / `rngWritePos` を AudioEngine 保持の共有リングへ移動。
   - `PsychoacousticDither::fillChunkForChannel` は AudioEngine の RNGWorker から呼ぶ。

**制約**：この Phase では既存処理フローを変えない。型追加のみ。

### Phase 2（dual-write 強化・EngineRuntime 中心化）

既存 A-6 作業の継続。EngineRuntime を RuntimeGraph の前身として位置づける。
`publishRuntimePublishState` が dual-write している状態を維持しつつ、
読み取りを EngineRuntime 優先に固める（既実施の A-6 Phase 2 完了相当）。

### Phase 3（process API 変更・adapter 層導入）

**対象ファイル**：

- `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp`
- `src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp`（存在確認要）

**作業内容**：

1. `DSPCore::processDouble(buffer)` に adapter 関数を追加：

   ```cpp
   // Adapter: 既存シグネチャを維持しつつ内部を graph/state 分離へ
   void DSPCore::processDouble(AudioBlock& block) noexcept {
       processV2(*attachedGraph, executionState, block);
   }
   void DSPCore::processV2(const RuntimeGraph& g, DSPExecutionState& s, AudioBlock& b) noexcept {
       // 段階的に移行
   }
   ```

2. crossfade 実行 (`process(graphA, stateA)` + `process(graphB, stateB)` + mix) の骨格作成。

### Phase 4（EQ state split + thread_local 除去）

**対象ファイル**：

- `src/eqprocessor/EQProcessor.h/.cpp`
- `src/eqprocessor/EQProcessor.Processing.cpp`

**作業内容**：

1. `EQCoeffCache.parallelInputBuffer` 等 3 バッファを EQCoeffCache から分離。
   DSPExecutionState.eq.parallelBufs に移動し、prepareToPlay 時に確保。
2. `filterState` / `agcCurrentGain` 等全 DSP_THREAD_STATE フィールドを EQDSPState に移動。
3. `static thread_local convo::RCUReader tls_rcuReader` を削除。
   DSPExecutionState.eq.rcuReader に移動（process 呼び出し時に明示渡し）。
4. AGC 係数テーブル（`agcAttackCoeffTable` 等）を RuntimeGraph.eqAgcCoeffTables に移動。

### Phase 5（IRBank immutable 化・Convolver state 分離）

**対象ファイル**：

- `src/ConvolverProcessor.h/.cpp` および convolver/ 配下全 cpp
- `src/ConvolverState.h`（snapshotRefCount 移行要）

**作業内容**：

1. `StereoConvolver.irData` / `partitionLayout` / `nucConvolvers.immutablePart` を
   `IRBank` として `shared_ptr<const IRBank>` で管理。
2. `nucConvolvers` の FDL history 部分を `ConvolverDSPState.nucStates` に分離。
3. `ConvolverProcessor.dryBuffer` 等の DSP_THREAD_STATE 群を `ConvolverDSPState` に移動。
4. `ConvolverState.snapshotRefCount` の移行：Phase 8 の RCU 縮小後に廃止する
   （interim: DeletionQueue 側の `ConvolverState` 特別処理を IRBank の refcount に統合）。

### Phase 6（Crossfade 再設計・B-1 修正）

**対象ファイル**：

- `src/audioengine/AudioEngine.h`（`finalizeCrossfadeMixPath`, `cleanupCrossfadeDirectPath`）
- `src/audioengine/AudioEngine.Processing.AudioBlock.cpp`（crossfade 呼び出し元）

**作業内容（最優先・terms 1.4節違反修正）**：

1. `finalizeCrossfadeMixPath()` / `cleanupCrossfadeDirectPath()` 内の
   `retireDSP(done)` を `DeferredDeletionQueue.enqueue()` に置き換え。

   ```cpp
   // Before（RT unsafe）:
   retireDSP(done);

   // After（RT safe）:
   const uint64_t epoch = epochCore.currentEpoch();
   if (!g_deletionQueue.enqueue(done,
           [](void* p){ delete static_cast<DSPCore*>(p); }, epoch)) {
       // Queue full: Timer で retry するため overflow 退避
       retireOverflowSlots.push_back(done);  // non-RT fallback（design for Phase 6.1）
   }
   ```

2. crossfade を「graph 切替」ではなく「state 複製実行」に変更（Phase 3 骨格の本実装）：
   - `currentDSP` / `fadingOutDSP` の ownership 役割を縮小。
   - `DSPExecutionState[current]` + `DSPExecutionState[fading]` の 2 本を音声処理に使用。
   - crossfade 終了時は fading state を reset して pool に戻す（delete 不要）。
3. `queuedOldDSP` を廃止方向へ（Phase 6 では still 保持、Phase 8 で完全廃止）。

### Phase 7（RuntimePublishState 廃止）

**廃止順序**：

1. まず `runtimePublishState` の書き込みを停止（EngineRuntime 一本化）。
2. 読み取り箇所をすべて EngineRuntime / RuntimeGraph atomic ptr に切り替え。
3. `runtimePublishState` フィールド削除。
4. `runtimePublishRevision` → RuntimeGraph.runtimeUuid + generation で代替。

### Phase 8（Epoch/RCU 縮小）

1. `EBRQueue` の retire 対象を RuntimeGraph only に縮小。
2. `DeletionQueue` の `ConvolverState` 特別処理（snapshotRefCount チェック）を削除。
   （Phase 5 で IRBank の refcount 管理に統合済みのため）
3. `queuedOldDSP` フィールド廃止。
4. `tls_readerSlot` (static thread_local) を明示的 EpochManager 登録 API に置き換え。

---

## 9. 実装レビュー必須チェックリスト（terms.md 6節完全版）

### 6.1 mutable leakage check

- [ ] RuntimeGraph のすべてのフィールドに `// IMMUTABLE_RUNTIME` コメントがあるか
- [ ] RuntimeGraph 内に `mutable` キーワードが存在しないか
- [ ] RuntimeGraph 内に `static` 変数が存在しないか
- [ ] RuntimeGraph 内に `thread_local` が存在しないか

### 6.2 RT allocation check

- [ ] `process()` 経路内で `new` / `malloc` / `mkl_malloc` が呼ばれないか
- [ ] `process()` 経路内で `vector::resize` / `push_back` が呼ばれないか
- [ ] `process()` 経路内で `std::string` 操作が発生しないか

### 6.3 ownership mutation check

- [ ] 音声スレッドで `shared_ptr::reset` / `unique_ptr::reset` が呼ばれないか
- [ ] 音声スレッドで `retireDSP()` / `retireObject()` / `EBRQueue::retire()` が呼ばれないか
- [ ] 音声スレッドで `DeferredDeletionQueue::reclaim()` が呼ばれないか（enqueue のみ許可）
- [ ] `finalizeCrossfadeMixPath()` / `cleanupCrossfadeDirectPath()` が B-1 修正済みか

### 6.4 reclamation correctness check

- [ ] `DeferredDeletionQueue::enqueue()` が失敗した場合の fallback が設計されているか
- [ ] retire 後のポインタに audio thread がアクセスしないか（epoch 保護確認）
- [ ] `ConvolverState.snapshotRefCount` が残存する場合、削除前の参照チェックが機能しているか

### 6.5 offline rendering check

- [ ] `static thread_local` がすべて削除または明示的引数渡しに置き換えられているか
- [ ] `DSPExecutionState` を別スレッドから実行しても破綻しないか（nested/offline考慮）
- [ ] `PsychoacousticDither.rngProducerThread` が分離済みか（H-2 修正確認）

### 6.6 host reentrancy check

- [ ] `process()` が再入しても内部 static/global mutable state が壊れないか
- [ ] `RNGWorker` の rngRing が SPSC 設計で競合しないか

### 6.7 追加（本設計固有）

- [ ] `EQCoeffCache.parallelInputBuffer` 等が DSPExecutionState に分離済みか（Phase 4）
- [ ] `AdaptiveNoiseShaper.coeffs` が DSPExecutionState 内のコピー経由で更新されているか
- [ ] `LatticeNoiseShaper.states` が DSPExecutionState に移行済みか
- [ ] `ProcessingState` の全フィールドが RuntimeGraph / DSPExecutionState に振り分け済みか

---

## 10. 参照ドキュメント対応表

| 本設計の節 | basic_rule.md 対応 | terms.md 対応 |
|---|---|---|
| §2 最終目標構造 | 3.1〜4.4節 | 1.1〜1.5節 |
| §3 Ownership Table | 2.1〜2.3節 | 5.1節 |
| §4 Mutation Table | 2.1節（根本問題） | 5.2節 |
| §5 RT Safety Table | 8節（Audio thread安全性） | 1.3〜1.4節、3節 |
| §6 Destruction Table | Phase 8（RCU縮小） | 5.4節 |
| §7 Allocation Table | 10.2節（process内allocation禁止） | 5.5節 |
| §8 Phase別計画 | 5〜12節 | 7節（推奨ワークフロー） |
| §9 チェックリスト | 10節（注意事項） | 6節（レビュー検査項目） |
| doc/phase0_member_classification.md | Phase 0 | 2節（必須分類） |
