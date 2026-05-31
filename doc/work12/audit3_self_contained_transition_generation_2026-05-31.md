# 追補レポート：監査3点版（work12）

作成日: 2026-05-31
前報告: `authority_topology_validation_and_blockers_2026-05-31.md`
監査3軸:

1. RuntimeWorld Self-Contained 達成度
2. TransitionState の実行権威性
3. Generation Authority Singularity

---

## 1. RuntimeWorld Self-Contained 達成度

### 判定: **未達（World外状態が独立して残存）**

Authority問題とは別に、`RuntimePublishWorld` の外に状態を持つコンポーネントが稼働している。

### 根拠（機械抽出）

#### ConvolverProcessor 側

| シンボル | ファイル:行 | 説明 |
|---|---|---|
| `PendingParams pendingOverride` | `src/ConvolverProcessor.h:982` | 宣言。明示的に「唯一の Source of Truth」とコメントされている |
| `pendingOverride.{mix,bypassed,smoothingTimeSec,...}` | `src/convolver/ConvolverProcessor.Runtime.cpp:678-1019` | Audio/Message Thread で頻繁に読み書き |
| `PreparedIRState` | `src/ConvolverProcessor.h:230`, `src/convolver/ConvolverProcessor.LoadPipeline.cpp:312` | IR 準備状態を World外で保持 |
| `GenerationManager convolverStateGeneration` | `src/ConvolverProcessor.h:1157` | 世代管理が ConvolverProcessor ローカル |
| `DeferredFreeThread deferredFreeThread` | `src/ConvolverProcessor.h:1155` | 解放スレッドが ConvolverProcessor に帰属 |
| `DeferredDeletionQueue` | `src/core/EpochDomain.h:220` | EpochDomain 内に内包 |

#### AudioEngine 側

| シンボル | ファイル:行 | 説明 |
|---|---|---|
| `GenerationManager m_generationManager` | `src/audioengine/AudioEngine.h:3392` | ParameterCommand 用世代管理。`bumpGeneration()` を UI/StateIO が呼ぶ |
| `deferredDeleteFallbackQueue` | `src/audioengine/AudioEngine.h` | mutex 保護の fallback queue。World の外で GC 管理 |

### コメント証拠

```cpp
// src/ConvolverProcessor.h:853
// H3: bypassed / mixTarget shadow atomics 廃止済み。pendingOverride が唯一の Source of Truth。
```

```cpp
// src/ConvolverProcessor.h:67
// authoritative source-of-truth は PendingParams (pendingOverride)。
```

### 結論

`ConvolverProcessor::pendingOverride` が ConvolverProcessor ローカルに存在する間は、
RuntimePublishWorld だけを読めば実行できる状態にはない。
**Authority問題と Self-Contained問題は別軸**というレビュー指摘は正確。

---

## 2. TransitionState の実行権威性

### 判定: **`active` は実行分岐に実使用（権威性あり）。`policy` / `latencyDeltaSamples` は投影用途寄り。**

### 根拠（証跡行）

#### `transition.active` が実行分岐に使用される箇所

```cpp
// src/audioengine/AudioEngine.Processing.AudioBlock.cpp:188-190
DSPCore* fading = runtimePublishView.transition.active
    ? static_cast<DSPCore*>(runtimePublishView.transition.next)
    : nullptr;
```

```cpp
// src/audioengine/AudioEngine.Processing.BlockDouble.cpp:145-147
DSPCore* fading = runtimePublishView.transition.active
    ? static_cast<DSPCore*>(runtimePublishView.transition.next)
    : nullptr;
```

```cpp
// src/audioengine/AudioEngine.Processing.Snapshot.cpp:27-29
DSPCore* dsp = isFadingTarget
    ? (runtimePublishView.transition.active
        ? static_cast<DSPCore*>(runtimePublishView.transition.next)
        : nullptr)
    : static_cast<DSPCore*>(runtimePublishView.transition.current);
```

```cpp
// src/audioengine/AudioEngine.Timer.cpp:55-57
auto* fadingRuntime = runtimePublishView.transition.active
    ? static_cast<DSPCore*>(runtimePublishView.transition.next)
    : nullptr;
```

```cpp
// src/audioengine/AudioEngine.Timer.cpp:160-162
auto* fadingDspForRuntime = runtimePublishView.transition.active
    ? static_cast<DSPCore*>(runtimePublishView.transition.next)
    : nullptr;
```

```cpp
// src/audioengine/AudioEngine.h:2374-2376
if (!transition.active)
    return nullptr;
```

```cpp
// src/audioengine/AudioEngine.h:2784-2786
auto* fading = (transition.active && transition.next != nullptr)
    ? static_cast<DSPCore*>(transition.next)
    : nullptr;
```

#### `transition.policy` の状況

grep 結果（2件）はいずれも代入・投影のみ。実行 `if/switch` 分岐は確認されず。

```cpp
// src/audioengine/AudioEngine.h:2164
runtime.transition.policy = policy;

// src/audioengine/AudioEngine.h:2553
worldOwner->execution.transitionPolicy = static_cast<int>(engineState.transition.policy);
```

#### `transition.latencyDeltaSamples` の状況

grep 結果（3件）も代入・投影のみ。

```cpp
// src/audioengine/AudioEngine.h:2166
runtime.transition.latencyDeltaSamples = runtime.latencyDelayOld - runtime.latencyDelayNew;

// src/audioengine/AudioEngine.h:2554
worldOwner->execution.latencyCompensationSamples = engineState.transition.latencyDeltaSamples;

// src/audioengine/AudioEngine.h:2580
worldOwner->latency.latencyDeltaSamples = engineState.transition.latencyDeltaSamples;
```

#### `TransitionState` 定義（証跡）

```cpp
// src/audioengine/RuntimeTransition.h:17-26
struct TransitionState
{
    // AuthorityClass::Derived (mirrors authoritative transition fact for observation)
    void* current = nullptr;
    void* next = nullptr;
    TransitionPolicy policy = TransitionPolicy::SmoothOnly;
    double fadeTimeSec = 0.0;
    int latencyDeltaSamples = 0;
    bool active = false;
};
```

コメントに `AuthorityClass::Derived` とあり、設計意図としては Derived（観測用）だが、
`active` は実際に Audio Thread の DSP 選択を決定しており **実行権威の成分** になっている。

### 結論

レビュー「TransitionState 全体が権威か否か追加解析が必要」に対し:

- `transition.active` → **実行権威（DSP 選択分岐を直接決定）**
- `transition.policy` → 現状は **非分岐（投影のみ）**。将来の分岐ポイント候補。
- `transition.latencyDeltaSamples` → **非分岐（値の投影のみ）**

---

## 3. Generation Authority Singularity

### 判定: **Singularity 未達（世代ソースが複数ドメインで並存）**

### 根拠（ドメイン別）

| ドメイン | シンボル | ファイル:行 | 説明 |
|---|---|---|---|
| Runtime publish 世代 | `runtimeGenerationGenerator_.next()` | `src/audioengine/AudioEngine.h`（`reserveNextRuntimeGraphGeneration`） | Runtime world 公開時の authority 世代 |
| Runtime publish 世代 | `world->generation` | `src/audioengine/AudioEngine.Commit.cpp:36,43,44,...` | World 単位の generation チェック |
| Rebuild 世代 | `rebuildGeneration` | `src/audioengine/AudioEngine.h:1669` | `std::atomic<int>`。rebuild 競合防止 |
| Rebuild 世代 | `++rebuildGeneration` | `src/audioengine/AudioEngine.RebuildDispatch.cpp:561` | 唯一のインクリメント箇所 |
| Build snapshot 世代 | `RuntimeBuildSnapshot.generation` | `src/audioengine/RuntimeBuildTypes.h:28` | `captureRuntimeBuildSnapshot` → `finalizeRuntimeBuildSnapshot` |
| Snapshot 世代 | `GlobalSnapshot.generation` | `src/core/GlobalSnapshot.h:48`, `src/core/SnapshotParams.h:37` | `m_generationManager.bumpGeneration()` から設定 |
| 係数バンク世代 | `AdaptiveCoeffBankSlot::generation` | `src/audioengine/AudioEngine.h:1851` | `CoeffSetWriteLockGuard::commit()` で `fetch_add` |
| ISR DSP Handle 世代 | `ISRDSPRegistry::generation` | `src/audioengine/ISRDSPHandle.h:92` | ABA 防止用スロット世代 |

### 主要な generation フロー（機械抽出）

```
requestRebuild
    ↓
++rebuildGeneration  (src/audioengine/AudioEngine.RebuildDispatch.cpp:561)
    ↓ task.generation に格納
prepareCommit(dsp, task.generation)
    ↓
commitNewDSP(dsp, generation)
    ↓
buildRuntimePublishWorld → reserveNextRuntimeGraphGeneration
    → runtimeGenerationGenerator_.next()  ← 別カウンタ
    ↓
world->generation = nextGraphGeneration  ← 上記と等価
    ↓
publishState / commitRuntimePublication
```

```
WorkerThread / onSnapshotRequired
    ↓
m_generationManager.bumpGeneration()  ← さらに別カウンタ
    ↓
GlobalSnapshot.generation             ← World generation とは独立
```

### 結論

- `rebuildGeneration` (int atomic) と `runtimeGenerationGenerator_` (64bit) と
  `m_generationManager` (64bit) の3者が並存しており、
  レビュー指摘の「generation source が複数」は **正確**。
- 設計上は各カウンタが異なる観点（rebuild 競合防止 / publish ID / snapshot デバウンス）で使われており
  単純に統合できないが、**Runtime Authority の単一源**という観点では **Singularity 未達**。

---

## 総合まとめ

| 監査軸 | 判定 | 主要シンボル |
|---|---|---|
| RuntimeWorld Self-Contained | **未達** | `pendingOverride`, `PreparedIRState`, `GenerationManager`, `DeferredFreeThread` |
| TransitionState 実行権威性 | **`active` は権威成分** / `policy`, `latencyDeltaSamples` は投影寄り | `transition.active` (6箇所で分岐), `RuntimeTransition.h` |
| Generation Singularity | **未達** | `rebuildGeneration`, `runtimeGenerationGenerator_`, `m_generationManager`, `BuildSnapshot.generation` |

前報告の P0 群（Authority経路）とこの3軸（World閉包性・transition権威・generation分裂）は
**独立した未達条件**であり、「Authority単一路化は達成したが RuntimeWorld が閉じていない」
というケースも識別できる状態になった。

---

## 参考：前報告との対応

| 前報告 | 今報告 |
|---|---|
| Authority Topology（P0/P1/P2 群） | → 独立軸として据え置き |
| Snapshot 系 198 hit（Authority か否か未判定） | → `transition.active` は実行権威、`policy` は現状非分岐と判明 |
| Generation 系（触れていない） | → 3カウンタ並存による Singularity 未達を確認 |
