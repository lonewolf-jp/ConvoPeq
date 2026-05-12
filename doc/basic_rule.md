# ConvoPeq

## 「Immutable Runtime Graph + Per-thread DSP State」移行計画書

（現行コード解析ベース・詳細版）

---

# 1. 総括

現行 ConvoPeq は、既に以下を部分的に達成しています。

* Runtime rebuild thread
* Runtime publish snapshot
* SnapshotCoordinator
* RCU / Epoch reclamation
* command queue
* dual-write migration
* build/runtime separation
* 일부 immutable snapshot 化

これは「完全 immutable runtime architecture」への移行途中段階です。

しかし現状はまだ、

* DSPCore が mutable DSP object を内部所有
* process 中に mutable state を直接変更
* runtime object と DSP execution state が未分離
* crossfade/runtime transition が ownership-aware
* Convolver/EQ が “processor object” として mutable lifetime を持つ
* thread-local であるべき state が runtime object に混在

しており、設計上の限界があります。

現行構造は「RCU＋mutable DSP instance」です。
最終安定構造は「Immutable Runtime Graph + Per-thread DSP State」です。

これは単なるリファクタではなく、

> “DSPアルゴリズム定義” と “DSP実行状態” を完全分離する

アーキテクチャ移行です。

---

# 2. 現行アーキテクチャ解析

---

# 2.1 現行 DSPCore の本質

現状 DSPCore は以下を同時に持っています。

```cpp
struct DSPCore
{
    ConvolverProcessor convolver;
    EQProcessor eq;
    ...
}
```



これは：

| 種類                 | 内容                          |
| ------------------ | --------------------------- |
| Graph Definition   | 処理構造                        |
| Runtime Parameters | snapshot                    |
| DSP Internal State | delay/history/filter memory |
| Resource Ownership | FFT/IR/buffers              |
| Thread State       | processing state            |

を単一オブジェクトに混在させています。

これが最終安定化を阻害しています。

---

# 2.2 現在すでに immutable 化されている部分

以下は既に良い方向です。

## RuntimeBuilder

```cpp
BuildResult RuntimeBuilder::build(...)
```



これは：

* rebuild thread で構築
* audio thread publish
* prepare 完了後のみ publish

を達成しています。

---

## SnapshotCoordinator

```cpp
const GlobalSnapshot* getCurrent()
```



GlobalSnapshot は immutable snapshot に近い。

---

## RuntimePublishState

```cpp
struct RuntimePublishState
```



これは：

* current
* fading
* queued
* transition

を immutable publish state として見せています。

これは Phase-2 migration 途中です。

---

# 2.3 現行構造の根本問題

## 問題1

DSP object が mutable

例えば：

```cpp
eqRt().process(processBlock);
convolverRt().process(processBlock);
```



内部では：

* FIR history
* overlap-add
* delay line
* filter memory
* smoothing
* DC blocker
* oversampling state

が mutate されています。

つまり：

```text
Runtime Graph == Runtime State
```

になっています。

---

## 問題2

crossfade が runtime ownership を伴う

現状：

```cpp
currentDSP
fadingOutDSP
queuedOldDSP
```



複数 DSPCore を並列保持しています。

これは：

* ownership
* fade lifecycle
* destruction timing
* epoch reclaim

を複雑化しています。

本来 crossfade は：

```text
Graph immutable
State only duplicated
```

であるべきです。

---

## 問題3

ConvolverProcessor が mutable resource owner

```cpp
cachedFFTBuffer
fftHandle
dryBuffer
wetBuffer
```



これらは：

* runtime shared
* process mutable
* thread affinity dependent

です。

特に：

* FFT scratch
* temporary buffers
* overlap buffers

は per-thread state に分離すべきです。

---

## 問題4

prepare/reset/process lifecycle が object に埋め込まれている

```cpp
prepare()
reset()
process()
```



これは JUCE 的には自然ですが、

immutable runtime architecture では：

```text
prepare graph
allocate state
process(state)
```

へ変わります。

---

# 3. 最終目標アーキテクチャ

---

# 3.1 最終構造

最終的には：

```text
UI Thread
   ↓
Command Queue
   ↓
Builder Thread
   ↓
Immutable RuntimeGraph
   ↓ atomic publish
Audio Thread
   ↓
PerThreadDSPState
   ↓
DSP execution
```

になります。

---

# 3.2 RuntimeGraph の責務

RuntimeGraph は immutable。

含むもの：

* routing
* node graph
* coefficient snapshots
* IR immutable blobs
* latency metadata
* fade metadata
* oversampling topology

含まないもの：

* delay history
* overlap-add
* FFT scratch
* smoothing memory
* DC history
* temporary buffers

---

# 3.3 PerThreadDSPState の責務

mutable execution state を集約。

例：

```cpp
struct PerThreadDSPState
{
    ConvolverState conv;
    EQState eq;
    OversamplingState os;
    DCBlockerState dc;
    ScratchBuffers scratch;
};
```

audio thread 毎に独立。

---

# 4. 推奨最終構造

---

# 4.1 RuntimeGraph

```cpp
struct RuntimeGraph
{
    uint64_t runtimeUuid;

    GraphTopology topology;

    std::shared_ptr<const IRBank> irBank;

    std::shared_ptr<const EQCoeffBank> eqBank;

    ProcessingConfig config;

    RuntimeLatency latency;

    RuntimeTransition transition;
};
```

重要：

```text
完全 immutable
```

---

# 4.2 DSPNodeDefinition

Convolver/EQ を node 化。

```cpp
struct DSPNodeDefinition
{
    NodeType type;

    const void* immutableData;
};
```

---

# 4.3 PerThreadDSPState

```cpp
struct DSPExecutionState
{
    std::vector<NodeState> nodeStates;

    ScratchArena arena;

    uint64_t attachedRuntimeUuid;
};
```

---

# 4.4 NodeState

```cpp
struct ConvolverNodeState
{
    OverlapAddHistory history;
    FFTScratch scratch;
};

struct EQNodeState
{
    BiquadHistory filters;
};
```

---

# 5. 段階的移行計画

---

# Phase 0

## 現行 mutable state 全洗い出し

最重要。

分類表を作る。

| 分類                  | immutable化可否 |
| ------------------- | ------------ |
| coefficient         | immutable    |
| IR                  | immutable    |
| FFT scratch         | per-thread   |
| overlap history     | per-thread   |
| smoothing           | per-thread   |
| latency buffers     | per-thread   |
| visualization cache | UI-only      |

これを誤ると破綻します。

---

# Phase 1

## DSPCore 分解

現状：

```cpp
DSPCore
 ├ ConvolverProcessor
 ├ EQProcessor
 ├ state
 └ buffers
```

↓

```text
RuntimeGraph
DSPExecutionState
```

へ分離。

---

## 具体的作業

### 新規

```cpp
RuntimeGraph.h
DSPExecutionState.h
DSPNode.h
NodeState.h
```

追加。

---

# Phase 2

## immutable IR bank 化

現在：

```cpp
ConvolverProcessor
  owns IR state
```

↓

```cpp
shared_ptr<const IRBank>
```

へ変更。

IRBank は：

* FFT済
* immutable
* refcount only

にする。

---

## 重要

現状：

```cpp
updateConvolverState()
```



は mutable publish。

最終的には：

```cpp
publish(new RuntimeGraph)
```

のみになる。

---

# Phase 3

## process API 変更

現状：

```cpp
process(buffer)
```

↓

最終：

```cpp
process(
    const RuntimeGraph& graph,
    DSPExecutionState& state,
    AudioBlock& block)
```

---

# Phase 4

## EQ 完全 stateless 化

現状 EQ は：

* coeff cache
* filter memory
* smoothing

混在。

これを：

| 種類           | 移行先               |
| ------------ | ----------------- |
| coeffs       | RuntimeGraph      |
| histories    | DSPExecutionState |
| temp buffers | ScratchArena      |

へ分離。

---

# Phase 5

## Convolver 完全分離

最難関。

---

## immutable 側

```cpp
IRBank
PartitionLayout
FFTPlan
```

---

## mutable 側

```cpp
FDL history
OverlapAdd
Temp FFT
Wet buffer
```

---

## 特に重要

現在：

```cpp
cachedFFTBuffer
```



これは完全に per-thread 化対象。

---

# Phase 6

## Crossfade 再設計

現在：

```text
currentDSP
fadingDSP
queuedDSP
```

を保持。

---

## 最終

```text
currentGraph
nextGraph

stateA
stateB
```

だけ保持。

crossfade は：

```text
process(graphA,stateA)
process(graphB,stateB)
mix
```

に変える。

ownership lifecycle が激減します。

---

# Phase 7

## RuntimePublishState 廃止

現在：

```cpp
RuntimePublishState
EngineRuntime
```

dual-write 中。



最終：

```cpp
atomic<RuntimeGraph*>
```

のみ。

---

# Phase 8

## Epoch/RCU 縮小

現在：

* DSP object retire
* runtime retire
* fading retire

を行っている。

最終：

```text
immutable graph retire only
```

になる。

UAF リスクが激減。

---

# 6. ConvolverProcessor の最重要改修点

---

# 6.1 現在の問題

ConvolverProcessor は：

```text
UI object
Runtime object
DSP state
Thread scratch
```

全部混在。

---

# 6.2 分離後

## UI Layer

```cpp
ConvolverModel
```

---

## Immutable Runtime

```cpp
ConvolverGraphNode
```

---

## DSP Mutable

```cpp
ConvolverDSPState
```

---

## Worker Resource

```cpp
FFTPlanCache
```

---

# 7. EQProcessor の最重要改修点

---

# 現状

EQProcessor は：

```text
parameter
coeff
history
processing
```

混在。

---

# 最終

```cpp
EQGraphNode
```

immutable。

```cpp
EQDSPState
```

mutable。

---

# 8. Audio Thread 安全性改善効果

---

# 現在

audio thread が：

* object lifetime
* ownership
* reclamation
* fade transition

を意識。

---

# 最終

audio thread は：

```text
graph ptr read
↓
thread state process
```

のみ。

完全 lock-free。

---

# 9. 期待される改善

---

| 項目                     | 改善   |
| ---------------------- | ---- |
| UAF                    | 大幅低減 |
| lifetime complexity    | 激減   |
| crossfade complexity   | 激減   |
| reclaim bugs           | 激減   |
| rebuild safety         | 向上   |
| cache locality         | 向上   |
| RT determinism         | 向上   |
| DSP scalability        | 向上   |
| SIMD optimization      | 容易   |
| future multithread DSP | 容易   |

---

# 10. 最重要注意事項

---

# 10.1 「immutable object」に mutable cache を入れない

最重要。

禁止：

```cpp
mutable FFTScratch scratch;
```

---

# 10.2 process 内 allocation 完全禁止

per-thread arena を事前確保。

---

# 10.3 graph rebuild と state rebuild を分離

graph が変わっても：

```text
compatible state reuse
```

できるようにする。

---

# 10.4 thread_local 多用禁止

thread_local は：

* host thread migration
* offline render
* future parallel DSP

で破綻しやすい。

明示的 `DSPExecutionState` を推奨。

---

# 10.5 JUCE object を audio thread graph に残さない

JUCE class は mutable 内部状態を持つ。

immutable graph に置くべきではない。

---

# 11. 推奨最終構成

```text
AudioEngine
 ├ atomic<RuntimeGraph*>
 ├ BuilderThread
 ├ RuntimeManager
 └ DSPExecutionPool

RuntimeGraph
 ├ NodeGraph
 ├ ImmutableCoeffBank
 ├ IRBank
 ├ LatencyMetadata
 └ TransitionMetadata

DSPExecutionState
 ├ ConvolverState
 ├ EQState
 ├ ScratchArena
 ├ OversamplingState
 └ MeterState
```

---

# 12. 現実的移行順序（推奨）

最適順：

1. immutable IRBank
2. EQ state split
3. DSPCore split
4. process API redesign
5. Convolver state split
6. crossfade redesign
7. RuntimePublishState 廃止
8. RCU簡素化
9. graph/node architecture
10. execution state pooling

---

# 13. 最終評価

現行 ConvoPeq は既に：

* snapshot architecture
* rebuild isolation
* publish transition
* epoch reclamation
* runtime generation

まで到達しています。

しかし現在は：

```text
“mutable DSP object を RCU で安全化している”
```

段階です。

最終安定構造は：

```text
“immutable graph + externalized DSP state”
```

です。

現行コードベースは、この移行を実施できる段階まで既に成熟しています。
特に：

* RuntimeBuilder
* SnapshotCoordinator
* RuntimePublishState
* rebuild thread

は非常に重要な土台になっています。
