# メモリ占有調査のためのインストルメンテーション改修案 v41（最終版）

---

**目的**: 2.33GB メモリ占有の原因特定。推定ではなく実測で確定させる。
**v40 からの変更**: 3 点の修正。

---

## 0. v40 の問題点と v41 での修正方針

| # | v40 の問題 | v41 の修正 |
|:--|:----------|:----------|
| 1 | `diagLogNonRt(const char*)` → 呼び出し元で `toRawUTF8()` → 一時 `juce::String` の寿命に依存しダングリングポインタ | **`DiagnosticsConfig.h` から削除。`MKLNonUniformConvolver.cpp` の無名名前空間で `const juce::String&` 版を定義** |
| 2 | `DiagnosticsConfig.h` が `juce::Logger` を使用 → JUCE 非依存を維持できていない | **`DiagnosticsConfig.h` から JUCE 参照を完全除去** |
| 3 | `StereoConvolver` / `DSPCore` の copy/move 制約未確認 | **StereoConvolver: `JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR` 確認。DSPCore: `DSPCore(const DSPCore&) = delete;` 確認。いずれも liveCount は ctor/dtor のみで OK** |

---

## 1. Patch A: DiagnosticsConfig.h — JUCE 参照を完全除去

```cpp
// DiagnosticsConfig.h — final version.
// ★ v41: JUCE 型への参照を完全に排除。
//   ログ出力ラッパーが必要な場合は、各 .cpp の無名名前空間で定義すること。

#pragma once

#include <cstdint>
#include <atomic>
#include <cassert>
#include <windows.h>
#include <psapi.h>

// ... MklAllocStats, diagMklMalloc, diagMklFree, freeTracked, addIfAlive,
//   updateAtomicMaximum64, resetDiagnostics, computeOtherPrivate ...
```

**`diagLogNonRt()` は DiagnosticsConfig.h から削除。**

---

## 2. Patch B: MKLNonUniformConvolver.cpp — 無名名前空間に diagLogNonRt

```cpp
// MKLNonUniformConvolver.cpp — #include の後

#include "DiagnosticsConfig.h"

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS

namespace {

/// ★ v41: 非 RT スレッドからの診断ログ出力（MKLNonUniformConvolver 専用）。
///   Never call from audio callback (RT).
///   juce::String& で受け取るためダングリングポインタのリスクはない。
inline void diagLogNonRt(const juce::String& message) noexcept
{
    juce::Logger::writeToLog(message);
}

} // namespace

#endif
```

---

## 3. Patch C: StereoConvolver — copy 禁止確認済み

```cpp
// ConvolverProcessor.h — struct StereoConvolver

struct StereoConvolver : public convo::AlignedBase {
    // ★ v41: 既に JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR により
    //   コピー・ムーブが禁止されている。liveCount は ctor/dtor のみで OK。
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(StereoConvolver)

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    static std::atomic<int> liveCount;
#endif

    StereoConvolver() {
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
        liveCount.fetch_add(1, std::memory_order_relaxed);
#endif
    }

    ~StereoConvolver() {
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
        liveCount.fetch_sub(1, std::memory_order_relaxed);
#endif
    }
    // ...
};
```

---

## 4. Patch D: DSPCore — copy/move 禁止確認済み

```cpp
// AudioEngine.h — struct DSPCore

struct DSPCore {
    // ★ v41: 既に DSPCore(const DSPCore&) = delete; によりコピー禁止。
    //   ムーブも暗黙定義なし。liveCount は ctor/dtor のみで OK。

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    static std::atomic<int> liveCount;
#endif

    DSPCore()
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

    ~DSPCore() {
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
        liveCount.fetch_sub(1, std::memory_order_relaxed);
#endif
        // ...
    }
    // ...
};
```

---

## 5. 全ログフォーマット（v41 確定版、v39 から変更なし）

### IR_RELEASE / IR_LOAD / IR_LAYOUT / MEM_SNAP（v39 と同一）

---

## 6. 出力例（v41 最終版）

```text
[IR_RELEASE] NUC#001 seq=105 MKL: before=820MB after=110MB delta=-710MB LayersBefore=3 TotalBefore=820MB(persistent) lostFree=18(+0) live=8 | OS: beforePrivate=2330MB afterPrivate=1620MB
[IR_LOAD]    NUC#001 seq=105 irLen=327680 blockSize=4096 Layers=3 L0Part=4096 L1Part=32768 L2Part=262144 directTaps=32 ringSize=8192 MKL: before=110MB after=812MB delta=+702MB lostFree=18(+0) live=8
[IR_LAYOUT]  NUC#001 seq=105 IRFreq=256MB FDL=420MB Accum=96MB Tail=16MB Direct=24MB Ring=8MB Total=820MB(persistent data buffers only) | L0=8MB L1=64MB L2=720MB
[MEM_SNAP]   PUBLISH gen=21 seq=5 | NUC(MKL only): live=8 alloc=820MB Peak(since reset)=1200MB totalA=28.0GB totalF=27.4GB lostFree=0 zeroAllocSize=0(delta=+0) | Stereo=4 DSPCore=4 | Retire: pending=232 trackedPendingBytes=12.8MB(diag only: sizeof tracked entries, not actual heap) trackedPending=8/232 (3%) overflow=0 reclaim=12 | OS: Private=2330MB WorkingSet=2400MB | OtherPrivate=1510MB(JUCE/CRT/IPP/threads/...)
```

---

## 7. v40 からの改善点一覧

| # | v40（問題） | v41（修正） |
|:--|:----------|:-----------|
| 1 | `diagLogNonRt(const char*)` → 一時 `juce::String` の `toRawUTF8()` に依存しダングリングポインタリスク | **`DiagnosticsConfig.h` から削除。`MKLNonUniformConvolver.cpp` の無名名前空間で `const juce::String&` 版を定義** |
| 2 | `DiagnosticsConfig.h` が `juce::Logger`/`juce::String` を使用 → JUCE 非依存を維持できていない | **`DiagnosticsConfig.h` から JUCE 参照を完全除去。純粋な診断ユーティリティに徹する** |
| 3 | `StereoConvolver` / `DSPCore` の copy/move 制約未確認 | **StereoConvolver: `JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR` 確認。DSPCore: `DSPCore(const DSPCore&) = delete;` 確認。いずれも liveCount は ctor/dtor のみで安全** |
