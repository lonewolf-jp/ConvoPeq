# メモリ占有調査のためのインストルメンテーション改修案 v43（最終版）

---

**目的**: 2.33GB メモリ占有の原因特定。推定ではなく実測で確定させる。
**v42 からの変更**: ソースコード調査で確定した 6 項目を反映。設計変更なし、実装確定事項のみ。

---

## 0. ソースコード調査で確定した未確定事項

| # | 調査項目 | 結果 | 反映 |
|:--|:--------|:-----|:-----|
| 1 | `StereoConvolver()` コンストラクタ | L645: **`StereoConvolver() = default;`** → `liveCount.fetch_add(1)` 追加のためデフォルト削除＋本体定義 | ctor 変更 |
| 2 | `~StereoConvolver()` デストラクタ | L677: 本体は空（jassert のみ） → `liveCount.fetch_sub(1)` 追加 | dtor 変更 |
| 3 | `StereoConvolver::liveCount` 静的定義 | どの `.cpp` にも未定義 → **新規追加必要** | `ConvolverProcessor.Lifecycle.cpp` に追加 |
| 4 | `DSPCore::DSPCore()` コンストラクタ実装 | `AudioEngine.Processing.DSPCoreLifecycle.cpp` L51 に実装 → `liveCount.fetch_add(1)` 追加 | ctor 変更 |
| 5 | `DSPCore::~DSPCore()` | L815 付近に実装 → `liveCount.fetch_sub(1)` 追加 | dtor 変更 |
| 6 | `DSPCore::liveCount` 静的定義 | どの `.cpp` にも未定義 → **新規追加必要** | `DSPCoreLifecycle.cpp` に追加 |

---

## 1. 確定した全変更箇所

### 1-1. StereoConvolver コンストラクタ（ConvolverProcessor.h L645）

```cpp
// Before:
StereoConvolver() = default;

// After:
StereoConvolver() {
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    liveCount.fetch_add(1, std::memory_order_relaxed);
#endif
}
```

### 1-2. StereoConvolver デストラクタ（ConvolverProcessor.h ~L677）

```cpp
// Before:
~StereoConvolver() {
#if JUCE_DEBUG
    jassert(nucConvolvers[0] == nullptr && nucConvolvers[1] == nullptr);
    jassert(irData[0] == nullptr && irData[1] == nullptr);
#endif
}

// After:
~StereoConvolver() {
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    liveCount.fetch_sub(1, std::memory_order_relaxed);
#endif
#if JUCE_DEBUG
    jassert(nucConvolvers[0] == nullptr && nucConvolvers[1] == nullptr);
    jassert(irData[0] == nullptr && irData[1] == nullptr);
#endif
}
```

### 1-3. StereoConvolver::liveCount 静的定義（ConvolverProcessor.Lifecycle.cpp）

```cpp
// ★ 追加
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
std::atomic<uint32_t> ConvolverProcessor::StereoConvolver::liveCount { 0 };
#endif
```

### 1-4. DSPCore コンストラクタ（AudioEngine.Processing.DSPCoreLifecycle.cpp L51）

```cpp
AudioEngine::DSPCore::DSPCore()
    : dcBlockerState(new DCBlockerRuntimeState())
    , convolverState(new ConvolverRuntimeState())
    , eqState(new EQRuntimeState())
    , rampState(new RampRuntimeState())
    , historyState(new HistoryRuntimeState())
    , runtimeUuid(reserveNextRuntimeUuid())
{
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    liveCount.fetch_add(1, std::memory_order_relaxed);
#endif
    convolverState->bind(convolver);
    eqState->bind(eq);
}
```

### 1-5. DSPCore デストラクタ（AudioEngine.h ~L815）

```cpp
~DSPCore()
{
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    liveCount.fetch_sub(1, std::memory_order_relaxed);
#endif
    convolver.forceCleanup();
}
```

### 1-6. DSPCore::liveCount 静的定義（DSPCoreLifecycle.cpp または AudioEngine.cpp）

```cpp
// ★ 追加
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
std::atomic<uint32_t> AudioEngine::DSPCore::liveCount { 0 };
#endif
```

---

## 2. 全ログフォーマット（v43 確定版）

### IR_RELEASE
```
[IR_RELEASE] NUC#%p seq=%llu MKL: before=%lluMB after=%lluMB delta=%lldMB LayersBefore=%d TotalBefore=%.0fMB(persistent) lostFree=%u(+%d) liveBefore=%u | OS: beforePrivate=%lluMB afterPrivate=%lluMB
```

### IR_LOAD
```
[IR_LOAD] NUC#%p seq=%llu irLen=%d blockSize=%d Layers=%d L0Part=%d L1Part=%d L2Part=%d directTaps=%d ringSize=%d MKL: before=%lluMB after=%lluMB delta=%lldMB lostFree=%u(+%d) live=%u
```

### IR_LAYOUT
```
[IR_LAYOUT] NUC#%p seq=%llu IRFreq=%.0fMB FDL=%.0fMB Accum=%.0fMB Tail=%.0fMB Direct=%.0fMB Ring=%.0fMB Total=%.0fMB(persistent data buffers only) | L0=%.0fMB L1=%.0fMB L2=%.0fMB
```

### MEM_SNAP
```
[MEM_SNAP] PUBLISH gen=%d seq=%d | NUC(MKL only): live=%u alloc=%.0fMB Peak(since reset)=%.0fMB totalA=%.0fGB totalF=%.0fGB lostFree=%u zeroAllocSize=%u(delta=%+d) | Stereo=%u DSPCore=%u | Retire: pending=%u trackedPendingBytes=%.1fMB(diag only) trackedPending=%u/%u (%.0f%%) overflow=%llu reclaim=%llu | OS: Private=%lluMB WorkingSet=%lluMB | OtherPrivate=%.0fMB(JUCE/CRT/IPP/threads/...)
```

---

## 3. 出力例（v43 最終版）

```text
[IR_RELEASE] NUC#001 seq=105 MKL: before=820MB after=110MB delta=-710MB LayersBefore=3 TotalBefore=820MB(persistent) lostFree=18(+0) liveBefore=8 | OS: beforePrivate=2330MB afterPrivate=1620MB
[IR_LOAD]    NUC#001 seq=105 irLen=327680 blockSize=4096 Layers=3 L0Part=4096 L1Part=32768 L2Part=262144 directTaps=32 ringSize=8192 MKL: before=110MB after=812MB delta=+702MB lostFree=18(+0) live=8
[IR_LAYOUT]  NUC#001 seq=105 IRFreq=256MB FDL=420MB Accum=96MB Tail=16MB Direct=24MB Ring=8MB Total=820MB(persistent data buffers only) | L0=8MB L1=64MB L2=720MB
[MEM_SNAP]   PUBLISH gen=21 seq=5 | NUC(MKL only): live=8 alloc=820MB Peak(since reset)=1200MB totalA=28.0GB totalF=27.4GB lostFree=0 zeroAllocSize=0(delta=+0) | Stereo=4 DSPCore=4 | Retire: pending=232 trackedPendingBytes=12.8MB(diag only) trackedPending=8/232 (3%) overflow=0 reclaim=12 | OS: Private=2330MB WorkingSet=2400MB | OtherPrivate=1510MB(JUCE/CRT/IPP/threads/...)
```

---

## 4. 最終変更ファイル一覧（v43 確定版、11 ファイル）

| # | ファイル | 変更内容 |
|:--|:--------|:--------|
| 1 | `src/core/IRetireProvider.h` | `virtual uint64_t pendingRetireBytes() const noexcept { return 0; }` 追加 |
| 2 | `src/DiagnosticsConfig.h` | MklAllocStats + diagMklMalloc/Free + freeTracked + addIfAlive + updateAtomicMaximum64 + DIAG_MKL_* マクロ + computeOtherPrivate（**JUCE 非依存**） |
| 3 | `src/MKLNonUniformConvolver.h` | **新規型**: LayerAllocSizes + NucDiagnosticsSnapshot + `enum : uint64_t { kDiagSeqReserved, kDiagSeqFirstRuntime }` + `liveCount(uint32_t)` + `globalDiagSeq(uint64_t)` + `getDiagnostics()` |
| 4 | `src/MKLNonUniformConvolver.cpp` | ctor/dtor liveCount + globalDiagSeq定義 + 全28箇所 mkl_malloc→DIAG_MKL_MALLOC + 15箇所 allocSizes保存 + freeAll freeTracked + releaseAllLayers freeTracked + 無名名前空間 logIrRelease + diagLogNonRt + IR_RELEASE/IR_LOAD/IR_LAYOUT + `#include "DiagnosticsConfig.h"` |
| 5 | `src/DeferredDeletionQueue.h` | DeletionEntry に `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` で `objectBytes` |
| 6 | `src/audioengine/ISRRetireRouter.h` | `pendingRetireBytes()` override + `m_pendingRetireBytes_` + `m_trackedPendingEntries_` + `trackedRatio()` |
| 7 | `src/audioengine/ISRRetireRouter.cpp` | enqueueRetire/tryReclaim での m_pendingRetireBytes_ 更新 |
| 8 | `src/audioengine/AudioEngine.Timer.cpp` | MEM_SNAP ログ |
| 9 | `src/audioengine/AudioEngine.Processing.DSPCoreLifecycle.cpp` | **DSPCore::liveCount 定義** + ctor `fetch_add(1)` + 確保量ログ |
| 10 | `src/audioengine/AudioEngine.h` | **DSPCore::liveCount(uint32_t) 追加** + DSPCore dtor `fetch_sub(1)` + `DSPCore(const DSPCore&) = delete;` 確認済み |
| 11 | `src/ConvolverProcessor.h` | **StereoConvolver::liveCount(uint32_t) 追加** + ctor `fetch_add(1)` + dtor `fetch_sub(1)` |
| 12 | `src/convolver/ConvolverProcessor.Lifecycle.cpp` | **StereoConvolver::liveCount 定義** |
