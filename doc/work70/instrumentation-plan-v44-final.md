# メモリ占有調査のためのインストルメンテーション改修案 v44（最終版）

---

**目的**: 2.33GB メモリ占有の原因特定。推定ではなく実測で確定させる。
**v43 からの変更**: 3 点の修正。

---

## 0. v43 の問題点と v44 での修正方針

| # | v43 の問題 | v44 の修正 |
|:--|:----------|:----------|
| 1 | 変更ファイル数が 11 と記載されているが実際は 12 | **「12 ファイル」に修正** |
| 2 | DSPCore デストラクタの実装位置が表と本文で矛盾（`AudioEngine.h` vs `DSPCoreLifecycle.cpp`） | **デストラクタは `AudioEngine.h` L815 のインライン実装と統一** |
| 3 | `StereoConvolver` / `DSPCore` の `liveCount.fetch_sub()` で `jassert(old > 0)` なし | **NUC と同様に `const auto old = liveCount.fetch_sub(1); jassert(old > 0);` に統一** |

---

## 1. Patch D/E: liveCount fetch_sub — jassert 統一

### StereoConvolver デストラクタ（ConvolverProcessor.h）

```cpp
~StereoConvolver()
{
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    const uint32_t oldLive = liveCount.fetch_sub(1, std::memory_order_relaxed);
    jassert(oldLive > 0);
#endif
#if JUCE_DEBUG
    jassert(nucConvolvers[0] == nullptr && nucConvolvers[1] == nullptr);
    jassert(irData[0] == nullptr && irData[1] == nullptr);
#endif
}
```

### DSPCore デストラクタ（AudioEngine.h）

```cpp
~DSPCore()
{
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    const uint32_t oldLive = liveCount.fetch_sub(1, std::memory_order_relaxed);
    jassert(oldLive > 0);
#endif
    convolver.forceCleanup();
}
```

---

## 2. 確定した DSPCore デストラクタ位置

- **デストラクタ宣言・実装**: `src/audioengine/AudioEngine.h` L815（インライン）
- **コンストラクタ宣言**: `src/audioengine/AudioEngine.h` L811（宣言のみ）
- **コンストラクタ実装**: `src/audioengine/AudioEngine.Processing.DSPCoreLifecycle.cpp` L51（本体）
- **`liveCount` 静的定義**: `src/audioengine/AudioEngine.Processing.DSPCoreLifecycle.cpp`（新規追加）

---

## 3. 最終変更ファイル一覧（v44 確定版、12 ファイル）

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
| 9 | `src/audioengine/AudioEngine.h` | **DSPCore::liveCount(uint32_t) 追加** + **dtor `fetch_sub(1)` + `jassert(old>0)`** |
| 10 | `src/audioengine/AudioEngine.Processing.DSPCoreLifecycle.cpp` | **ctor `liveCount.fetch_add(1)`** + **`DSPCore::liveCount` 静的定義** + 確保量ログ |
| 11 | `src/ConvolverProcessor.h` | **StereoConvolver::liveCount(uint32_t) 追加** + **ctor `fetch_add(1)`** + **dtor `fetch_sub(1)` + `jassert(old>0)`** |
| 12 | `src/convolver/ConvolverProcessor.Lifecycle.cpp` | **StereoConvolver::liveCount 静的定義** |

---

## 4. 全ログフォーマット（v44 確定版、v43 から変更なし）

### IR_RELEASE / IR_LOAD / IR_LAYOUT / MEM_SNAP（v43 と同一）

---

## 5. 出力例（v44 最終版）

```text
[IR_RELEASE] NUC#001 seq=105 MKL: before=820MB after=110MB delta=-710MB LayersBefore=3 TotalBefore=820MB(persistent) lostFree=18(+0) liveBefore=8 | OS: beforePrivate=2330MB afterPrivate=1620MB
[IR_LOAD]    NUC#001 seq=105 irLen=327680 blockSize=4096 Layers=3 L0Part=4096 L1Part=32768 L2Part=262144 directTaps=32 ringSize=8192 MKL: before=110MB after=812MB delta=+702MB lostFree=18(+0) live=8
[IR_LAYOUT]  NUC#001 seq=105 IRFreq=256MB FDL=420MB Accum=96MB Tail=16MB Direct=24MB Ring=8MB Total=820MB(persistent data buffers only) | L0=8MB L1=64MB L2=720MB
[MEM_SNAP]   PUBLISH gen=21 seq=5 | NUC(MKL only): live=8 alloc=820MB Peak(since reset)=1200MB totalA=28.0GB totalF=27.4GB lostFree=0 zeroAllocSize=0(delta=+0) | Stereo=4 DSPCore=4 | Retire: pending=232 trackedPendingBytes=12.8MB(diag only) trackedPending=8/232 (3%) overflow=0 reclaim=12 | OS: Private=2330MB WorkingSet=2400MB | OtherPrivate=1510MB(JUCE/CRT/IPP/threads/...)
```

---

## 6. v43 からの改善点一覧

| # | v43（問題） | v44（修正） |
|:--|:----------|:-----------|
| 1 | 変更ファイル数が 11 と記載 | **「12 ファイル」に修正** |
| 2 | DSPCore デストラクタの実装位置が表と本文で矛盾 | **デストラクタは `AudioEngine.h` L815 のインラインで統一。`liveCount` 静的定義のみ `DSPCoreLifecycle.cpp` に追加** |
| 3 | `StereoConvolver` / `DSPCore` の `liveCount.fetch_sub()` で `jassert(old > 0)` なし | **NUC と同様に `const auto old = fetch_sub(1); jassert(old > 0);` に統一** |
