# ConvoPeq

# Immutable Snapshot Runtime 完全移行計画書

## 対象

現在の ConvoPeq コードベース全体
（RCU/Epoch 導入済み・部分 immutable 化済み状態）

---

# 0. 最終到達目標

最終的に ConvoPeq を以下へ到達させる。

```text
UI Thread
    ↓
Immutable Snapshot Build
    ↓
Immutable RuntimeWorld Build
    ↓
Atomic Publish (single ptr)
    ↓
Audio Thread
    read-only access only
    no runtime mutation
    no command drain
    no sync propagation
    no partial rebuild
    no runtime cache update
    no runtime state machine
```

つまり:

```text
Runtime publish 後 mutate 完全禁止
```

を達成する。

---

# 1. 現在の問題構造

現状は:

```text
Immutable Snapshot Layer
+
Mutable Runtime Layer
```

である。

特に以下が mutable runtime を形成している。

| 問題                                 | 状態                       |
| ---------------------------------- | ------------------------ |
| RuntimeCommandQueue                | RT mutation              |
| DSPExecutionState                  | mutable cache            |
| syncEqAgcTableViewFromRuntimeGraph | post-publish sync        |
| currentXXX atomics                 | live mutable parameters  |
| runtime crossfade state            | mutable transition       |
| snapshot fade state machine        | mutable audio-side state |
| partial rebuild                    | runtime patching         |
| processV2(runtime, state)          | dual-state runtime       |

---

# 2. 最終アーキテクチャ

---

## 2.1 RuntimeWorld 中心構造へ統一

現在:

```text
GlobalSnapshot
RuntimeGraph
DSPExecutionState
CrossfadeState
CurrentAtomicCaches
```

を統合し、最終的に:

```cpp
struct RuntimeWorld final
{
    RuntimeVersion version;

    ImmutableDSPGraph graph;

    ImmutableDSPTables tables;

    ImmutableConvolverSet convolvers;

    ImmutableParameterBlock params;

    ImmutableCrossfadePlan transition;

    ImmutableMeterConfig meters;

    ImmutableLatencyInfo latency;

    ImmutableAGCState agc;

    ImmutableSmootherSeeds smoothers;
};
```

へ統合。

---

# 3. 段階的移行戦略

移行は必ず:

```text
Bridge Phase
↓
Dual Runtime Phase
↓
Immutable RuntimeWorld Phase
↓
Legacy Removal Phase
```

で進める。

一気に置換すると RT 崩壊リスクが高い。

---

# Phase 1

# Runtime Mutation 禁止フェーズ

最優先。

---

## 3.1 RuntimeCommandQueue 廃止

## 現状

```text
UI
↓
enqueue command
↓
RT drains queue
↓
runtime mutate
```

---

## 改修後

```text
UI param change
↓
new Snapshot build
↓
new RuntimeWorld build
↓
atomic publish
```

---

## 削除対象

### 完全削除

```text
RuntimeCommandQueue
EngineCommand
processAudioThreadRuntimeCommands()
setMixRT()
setSmoothingTimeRT()
runtime mutation setters
```

---

## 必要な変更

### UI parameter update

現状:

```cpp
enqueueRuntimeCommand(...)
```

↓

変更後:

```cpp
requestRuntimeRebuild(...)
```

---

## RT thread

現状:

```cpp
processAudioThreadRuntimeCommands();
```

↓

完全削除。

RT thread は:

```cpp
const RuntimeWorld* world = currentWorld.load(...);
```

のみ。

---

# Phase 2

# DSPExecutionState 廃止

---

## 4.1 問題

現在:

```cpp
processV2(runtimeGraph, dspExecutionState)
```

構造。

これは:

```text
runtime
+
mutable execution cache
```

である。

---

## 4.2 改修

すべて:

```cpp
process(const RuntimeWorld&)
```

へ変更。

---

## RuntimeWorld に内包すべきもの

### 現在 DSPExecutionState が持つもの

| 項目             | 処理              |
| -------------- | --------------- |
| coeff cache    | build時固定        |
| AGC table      | build時固定        |
| lookup table   | build時固定        |
| smoother seeds | immutable seed化 |
| filter config  | immutable化      |

---

## 4.3 process API 最終形

```cpp
void process(
    const RuntimeWorld& world,
    AudioBlock<float>& block) noexcept;
```

のみ。

---

# Phase 3

# sync 系 API 全廃

最重要。

---

## 5.1 削除対象

```text
syncEqAgcTableViewFromRuntimeGraph
syncRuntimeGraphToDSP
syncXXXFromRuntime
updateRuntimeCaches
```

---

## 原則

```text
publish後 synchronization 禁止
```

---

## 5.2 AGC/EQ table

現状:

```text
RuntimeGraph
↓
sync
↓
DSP mutable table
```

↓

最終:

```text
Builder
↓
fully materialized immutable tables
↓
RuntimeWorld.tables
↓
RT reads only
```

---

# Phase 4

# currentXXX Atomic Cache 廃止

---

## 6.1 現状

```cpp
m_currentInputHeadroomDb
m_currentEqBypass
...
```

---

## 問題

source of truth が複数存在。

---

## 改修

UI は:

```text
latest GlobalSnapshot
```

のみ参照。

---

## 必要構造

```cpp
struct UISnapshotView
{
    std::shared_ptr<const GlobalSnapshot> snapshot;
};
```

---

# Phase 5

# Crossfade 完全 immutable 化

現状は非常に危険。

---

## 7.1 現状

```text
runtimeCrossfadePending
armCrossfadeIfPending()
resolveFadingDSP...
```

など mutable transition machine がある。

---

## 7.2 最終構造

```cpp
struct ImmutableTransitionPlan
{
    RuntimeWorldPtr from;

    RuntimeWorldPtr to;

    FadeCurve curve;

    uint64_t durationSamples;

    uint64_t startSample;
};
```

これを publish。

---

## RT thread

```cpp
processTransition(
    transitionPlan,
    currentSample);
```

のみ。

状態変更禁止。

---

# Phase 6

# Snapshot Fade State Machine 廃止

---

## 現状

```text
snapshotFrom
snapshotTo
snapshotAlpha
```

mutable state machine。

---

## 改修

fade そのものを immutable 化。

```cpp
struct PublishedRuntimeWorld
{
    RuntimeWorldPtr active;
    RuntimeWorldPtr fading;
    ImmutableTransitionPlan transition;
};
```

---

# Phase 7

# Builder Thread 分離

---

## 9.1 専用 RuntimeBuilder 導入

```cpp
class RuntimeBuilder final
{
public:
    RuntimeWorldPtr build(
        const GlobalSnapshot&) noexcept;
};
```

---

## Builder thread responsibilities

* coeff generation
* AGC table generation
* FIR plan generation
* latency plan
* crossfade plan
* smoothing seed bake

すべてここで完了。

---

## RT thread は build 禁止

以下禁止:

| 禁止                    |
| --------------------- |
| coeff generation      |
| vector resize         |
| table update          |
| smoothing setup       |
| AGC update            |
| parameter propagation |

---

# Phase 8

# Publication モデル統一

---

## 10.1 単一 publish ptr

```cpp
std::atomic<PublishedRuntimeWorld*> currentWorld;
```

のみ。

---

## 10.2 RCU/Epoch

publish 時:

```text
new world publish
↓
old world retire
↓
epoch reclaim
```

のみ。

---

## 10.3 Audio thread

```cpp
const auto* world =
    currentWorld.load(std::memory_order_acquire);
```

以外の mutation 禁止。

---

# Phase 9

# Memory Ownership 完全固定化

---

## 11.1 Runtime object ownership

全 runtime object を:

```cpp
std::unique_ptr<const T>
```

へ。

---

## 11.2 mutable 禁止

以下禁止:

```cpp
mutable
non-const table ptr
shared mutable cache
lazy initialization
```

---

# Phase 10

# Deterministic RT Enforcement

---

## 12.1 RT static analysis

RT path に以下禁止。

| 禁止                 |
| ------------------ |
| mutex              |
| compare_exchange   |
| exchange           |
| fetch_add(acq_rel) |
| allocation         |
| std::function      |
| virtual dispatch   |
| libm               |
| exception          |
| logging            |

---

## 12.2 compile-time guard

導入推奨:

```cpp
ASSERT_AUDIO_THREAD()
ASSERT_NON_AUDIO_THREAD()
ASSERT_IMMUTABLE_RUNTIME()
```

---

# Phase 11

# Legacy Layer Removal

最終段階。

---

## 完全削除対象

| 削除                       |
| ------------------------ |
| RuntimeCommandQueue      |
| DSPExecutionState        |
| runtime sync API         |
| currentXXX atomics       |
| partial rebuild          |
| runtime mutation setters |
| snapshot fade machine    |
| rebuild dispatch         |
| runtime patching         |

---

# 13. 最終完成形

---

## Audio thread

最終的に RT thread は:

```cpp
void audioCallback(...)
{
    const RuntimeWorld* world =
        currentWorld.load(std::memory_order_acquire);

    process(world);
}
```

のみ。

---

## Build thread

```cpp
GlobalSnapshot
↓
RuntimeBuilder
↓
Immutable RuntimeWorld
↓
publish
↓
retire old
```

のみ。

---

# 14. 最重要移行順序

絶対順序。

| 優先 | 作業                       |
| -- | ------------------------ |
| P0 | RuntimeCommandQueue 廃止   |
| P0 | sync API 廃止              |
| P0 | DSPExecutionState 廃止     |
| P1 | Crossfade immutable 化    |
| P1 | currentXXX atomic 廃止     |
| P1 | snapshot fade machine 廃止 |
| P2 | process API 統一           |
| P2 | RuntimeBuilder 分離        |
| P2 | RuntimeWorld 完全統合        |
| P3 | legacy cleanup           |

---

# 15. 最終的に得られるもの

この移行完了後:

```text
RT deterministic
No runtime mutation
No partial visibility
No stale cache
No sync propagation
No ABA-like runtime mismatch
No parameter race
No mutable DSP graph
```

になる。

ConvoPeq は最終的に:

```text
Deterministic Immutable DSP Runtime System
```

へ到達する。
