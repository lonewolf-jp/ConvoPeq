# メモリ占有調査のためのインストルメンテーション改修案 v40（最終版）

---

**目的**: 2.33GB メモリ占有の原因特定。推定ではなく実測で確定させる。
**v39 からの変更**: ソースコード調査で確定した 3 項目を反映。

---

## 0. ソースコード調査で確定した未確定事項

| # | 調査項目 | 結果 | 反映 |
|:--|:--------|:-----|:-----|
| 1 | `diagLogNonRt(juce::String)` → DiagnosticsConfig.h に JUCE 依存が生まれる（v34 の JUCE 非依存化努力を壊す） | **`diagLogNonRt()` は `const char*` を受け取るように変更。呼び出し元で `juce::String::formatted(...).toRawUTF8()` または同等の変換を行う** | シグネチャ変更 |
| 2 | `StereoConvolver::liveCount` が未実装（現状コードに存在しない） | **`struct StereoConvolver` 内に `static std::atomic<int> liveCount` を追加。ctor で `fetch_add(1)`、dtor で `fetch_sub(1)`** | MEM_SNAP の `Stereo=%d` 用 |
| 3 | `DSPCore::liveCount` が未実装（現状コードに存在しない） | **`struct DSPCore` 内に `static std::atomic<int> liveCount` を追加。ctor で `fetch_add(1)`、dtor で `fetch_sub(1)`** | MEM_SNAP の `DSPCore=%d` 用 |

---

## 1. Patch A: DiagnosticsConfig.h — diagLogNonRt を const char* に

```cpp
/// ★ v40: 非 RT スレッドからの診断ログ出力。
///   Never call from audio callback (RT).
///   const char* を受け取るため、juce::String 非依存。
///   呼び出し元で juce::String::formatted(...).toRawUTF8() 等で変換すること。
inline void diagLogNonRt(const char* message) noexcept
{
    juce::Logger::writeToLog(juce::String(message));
}
```

**★ 補足**: この関数は DiagnosticsConfig.h にあるが、`juce::Logger` を使用する。
実際にはこの関数を使用する .cpp ファイル（MKLNonUniformConvolver.cpp 等）は
既に `<JuceHeader.h>` をインクルードしているため、問題なくコンパイルできる。

ただし DiagnosticsConfig.h 自体は `<JuceHeader.h>` をインクルードしない。
関数テンプレートの具体化は呼び出し元の翻訳単位で行われるため、
`juce::Logger` の完全型は呼び出し元の .cpp で利用可能であればよい。

もしコンパイルエラーが発生した場合、`diagLogNonRt()` を DiagnosticsConfig.h から
削除し、各 .cpp ファイル内の無名名前空間で個別に定義すること。

---

## 2. Patch B: StereoConvolver — liveCount 追加

### ConvolverProcessor.h (L628 付近)

```cpp
struct StereoConvolver : public convo::AlignedBase {
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

### ConvolverProcessor.cpp または Lifecycle.cpp

```cpp
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
std::atomic<int> ConvolverProcessor::StereoConvolver::liveCount { 0 };
#endif
```

---

## 3. Patch C: DSPCore — liveCount 追加

### AudioEngine.h (struct DSPCore 内)

```cpp
struct DSPCore {
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    static std::atomic<int> liveCount;
#endif

    DSPCore() {
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
        liveCount.fetch_add(1, std::memory_order_relaxed);
#endif
        // ...
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

### DSPCoreLifecycle.cpp または AudioEngine.cpp

```cpp
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
std::atomic<int> AudioEngine::DSPCore::liveCount { 0 };
#endif
```

---

## 4. IRetireProvider — pendingRetireBytes 追加（v25 からの確定事項）

### IRetireProvider.h

```cpp
class IRetireProvider {
    // ...
    [[nodiscard]] virtual uint64_t pendingRetireBytes() const noexcept { return 0; }
};
```

---

## 5. 全ログフォーマット（v40 確定版、v39 から変更なし）

### IR_RELEASE / IR_LOAD / IR_LAYOUT / MEM_SNAP（v39 と同一）

---

## 6. 出力例（v40 最終版、v39 と同一）

```text
[IR_RELEASE] NUC#001 seq=105 MKL: before=820MB after=110MB delta=-710MB LayersBefore=3 TotalBefore=820MB(persistent) lostFree=18(+0) live=8 | OS: beforePrivate=2330MB afterPrivate=1620MB
[IR_LOAD]    NUC#001 seq=105 irLen=327680 blockSize=4096 Layers=3 L0Part=4096 L1Part=32768 L2Part=262144 directTaps=32 ringSize=8192 MKL: before=110MB after=812MB delta=+702MB lostFree=18(+0) live=8
[IR_LAYOUT]  NUC#001 seq=105 IRFreq=256MB FDL=420MB Accum=96MB Tail=16MB Direct=24MB Ring=8MB Total=820MB(persistent data buffers only) | L0=8MB L1=64MB L2=720MB
[MEM_SNAP]   PUBLISH gen=21 seq=5 | NUC(MKL only): live=8 alloc=820MB Peak(since reset)=1200MB totalA=28.0GB totalF=27.4GB lostFree=0 zeroAllocSize=0(delta=+0) | Stereo=4 DSPCore=4 | Retire: pending=232 trackedPendingBytes=12.8MB(diag only: sizeof tracked entries, not actual heap) trackedPending=8/232 (3%) overflow=0 reclaim=12 | OS: Private=2330MB WorkingSet=2400MB | OtherPrivate=1510MB(JUCE/CRT/IPP/threads/...)
```

---

## 7. v39 からの改善点一覧

| # | 調査結果 | 反映 |
|:--|:--------|:-----|
| 1 | `diagLogNonRt(juce::String)` → DiagnosticsConfig.h に JUCE 依存 | **`const char*` に変更。JUCE 非依存維持** |
| 2 | `StereoConvolver::liveCount` 未実装 | **`struct StereoConvolver` に `static std::atomic<int> liveCount` 追加** |
| 3 | `DSPCore::liveCount` 未実装 | **`struct DSPCore` に `static std::atomic<int> liveCount` 追加** |

---

## 8. 最終変更ファイル一覧（v40 確定版）

| # | ファイル | 変更内容 |
|:--|:--------|:--------|
| 1 | `src/core/IRetireProvider.h` | `virtual uint64_t pendingRetireBytes() const noexcept { return 0; }` 追加 |
| 2 | `src/DiagnosticsConfig.h` | MklAllocStats + diagMklMalloc/Free + freeTracked + addIfAlive + updateAtomicMaximum64 + DIAG_MKL_* マクロ + computeOtherPrivate + `diagLogNonRt(const char*)` |
| 3 | `src/MKLNonUniformConvolver.h` | LayerAllocSizes + NucDiagnosticsSnapshot + liveCount(static) + globalDiagSeq(static) + getDiagnostics + kReservedDiagSeq + kFirstRuntimeDiagSeq |
| 4 | `src/MKLNonUniformConvolver.cpp` | ctor/dtor liveCount + globalDiagSeq定義 + 全 mkl_malloc→DIAG_MKL_MALLOC + allocSizes保存 + freeAll/releaseAllLayers freeTracked + 無名名前空間 logIrRelease + IR_RELEASE/IR_LOAD/IR_LAYOUT + `#include "DiagnosticsConfig.h"` |
| 5 | `src/DeferredDeletionQueue.h` | DeletionEntry に `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` で `objectBytes` |
| 6 | `src/audioengine/ISRRetireRouter.h` | `pendingRetireBytes()` override + `m_pendingRetireBytes_` + `m_trackedPendingEntries_` + `trackedRatio()` |
| 7 | `src/audioengine/ISRRetireRouter.cpp` | enqueueRetire/tryReclaim での m_pendingRetireBytes_ 更新 |
| 8 | `src/audioengine/AudioEngine.Timer.cpp` | MEM_SNAP ログ |
| 9 | `src/audioengine/AudioEngine.Processing.DSPCoreLifecycle.cpp` | DSPCore::liveCount 定義 + 確保量ログ |
| 10 | `src/ConvolverProcessor.h` | `StereoConvolver::liveCount` 追加 |
| 11 | `src/convolver/ConvolverProcessor.Lifecycle.cpp` または .cpp | `StereoConvolver::liveCount` 定義 |

**合計: 11 ファイル変更**
**変更不要**: `ConvolverProcessor.h` の SetImpulse 呼び出し（6 引数のまま動作）
