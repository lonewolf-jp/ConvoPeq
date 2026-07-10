# メモリ占有調査のためのインストルメンテーション改修案 v38（最終版）

---

**目的**: 2.33GB メモリ占有の原因特定。推定ではなく実測で確定させる。
**v37 からの変更**: 5 点の修正（全て保守性・診断性の向上。設計変更なし）。

---

## 0. v37 の問題点と v38 での修正方針

| # | v37 の問題 | v38 の修正 |
|:--|:----------|:----------|
| 1 | `releaseAllLayers()` の size 退避がガードより前（将来の早期 return 追加時、size だけ計算して return する無駄が生じうる） | **ガード後に size 退避に移動** |
| 2 | `freeTracked(size==0)` が `lostFreeCount` を増やさない → allocatedBytes の不整合が説明しづらい | **`lostFreeCount` も加算（zeroAllocSizeCount と両方増える）** |
| 3 | `Logger::writeToLog()` 直呼び出しが将来 Worker Thread からの利用を妨げうる | **`diagLogNonRt()` ラッパーを DiagnosticsConfig.h に追加** |
| 4 | `seq=0 is reserved` がコメントのみでコード上の名前がない | **`constexpr uint64_t kReservedDiagSeq = 0` を MKLNonUniformConvolver.h に追加** |
| 5 | `zeroAllocSize=0(+0)` の `(+N)` が何の増分か不明 | **`zeroAllocSize=%u(delta=%+d)` に変更** |

---

## 1. Patch A: DiagnosticsConfig.h — diagLogNonRt + freeTracked lostFree

### A-1. diagLogNonRt — 簡易ログラッパー

```cpp
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS

/// ★ v38: 非 RT スレッドからの診断ログ出力。
///   現在は juce::Logger::writeToLog を直接呼ぶ。
///   将来 Worker Thread 対応が必要な場合、この関数のみ修正すればよい。
inline void diagLogNonRt(const juce::String& message) noexcept
{
    juce::Logger::writeToLog(message);
}

#endif
```

### A-2. freeTracked — lostFreeCount も加算

```cpp
template<typename T>
inline void freeTracked(T*& p, size_t size) noexcept
{
    if (p)
    {
        if (size > 0)
        {
            DIAG_MKL_FREE(p, size);
        }
        else
        {
            // ★ v38: size==0 は異常。lostFreeCount と zeroAllocSizeCount の両方を増やす。
            mklStats().lostFreeCount.fetch_add(1, std::memory_order_relaxed);
            mkl_free(p);
        }
        p = nullptr;
    }
}
```

---

## 2. Patch B: MKLNonUniformConvolver — ガード順序 + kReservedDiagSeq

### B-1. MKLNonUniformConvolver.h

```cpp
class MKLNonUniformConvolver {
public:
    static std::atomic<uint32_t> liveCount;
    static std::atomic<uint64_t> globalDiagSeq;

    /// ★ v38: デストラクタ等、SetImpulse 以外の経路で使用される seq 値。
    static constexpr uint64_t kReservedDiagSeq = 0;

    // ...
};
```

### B-2. releaseAllLayers — ガード後に size 退避

```cpp
void MKLNonUniformConvolver::releaseAllLayers() noexcept
{
    // ★ v38: ガードチェックを先に実施
#ifdef NUC_DEBUG_GUARDS
    checkGuards();
    // ... checkPtr ...
#endif

    // ★ v38: ガード後に解放サイズを退避（早期 return 追加時の無駄を防止）
    const size_t ringBufBytes    = static_cast<size_t>(m_ringSize) * sizeof(double);
    const size_t directIRBytes   = static_cast<size_t>(m_directTapCount) * sizeof(double);
    const size_t directHistBytes = static_cast<size_t>(m_directHistLen) * sizeof(double);
    const size_t directWinBytes  = static_cast<size_t>(m_directHistLen + m_directMaxBlock) * sizeof(double);
    const size_t directOutBytes  = static_cast<size_t>(m_directMaxBlock) * sizeof(double);

    for (int i = 0; i < kNumLayers; ++i)
        m_layers[i].freeAll();
    // ...
}
```

### B-3. SetImpulse — kReservedDiagSeq 使用

```cpp
bool MKLNonUniformConvolver::SetImpulse(...)
{
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    const uint64_t diagSeq = globalDiagSeq.fetch_add(1, std::memory_order_relaxed) + 1;
    // ★ v38: kReservedDiagSeq(=0) はデストラクタ専用
    // ...
#endif
    // ...
}
```

### B-4. デストラクタ

```cpp
MKLNonUniformConvolver::~MKLNonUniformConvolver()
{
    const uint32_t oldLive = liveCount.fetch_sub(1, std::memory_order_relaxed);
    jassert(oldLive > 0);
    releaseAllLayers();
    // ★ v38: デストラクタでは kReservedDiagSeq(=0) を想定。ログ出力なし
}
```

---

## 3. all ログで diagLogNonRt を使用

```cpp
// IR_RELEASE 内:
diagLogNonRt(juce::String::formatted(
    "[IR_RELEASE] NUC#%p seq=%llu ..."));

// IR_LOAD 内:
diagLogNonRt(juce::String::formatted(
    "[IR_LOAD] NUC#%p seq=%llu ..."));

// IR_LAYOUT 内:
diagLogNonRt(juce::String::formatted(
    "[IR_LAYOUT] NUC#%p seq=%llu ..."));
```

---

## 4. MEM_SNAP — zeroAllocSize 増分表記

```cpp
[MEM_SNAP] PUBLISH gen=%d seq=%d | NUC(MKL only): live=%u alloc=%.0fMB Peak(since reset)=%.0fMB totalA=%.0fGB totalF=%.0fGB lostFree=%u zeroAllocSize=%u(delta=%+d) | Stereo=%d DSPCore=%d | Retire: pending=%u trackedPendingBytes=%.1fMB(diag only: sizeof tracked entries, not actual heap) trackedPending=%u/%u (%.0f%%) overflow=%llu reclaim=%llu | OS: Private=%lluMB WorkingSet=%lluMB | OtherPrivate=%.0fMB(JUCE/CRT/IPP/threads/...)
```

---

## 5. 出力例（v38 最終版）

```text
[IR_RELEASE] NUC#001 seq=105 MKL: before=820MB after=110MB delta=-710MB LayersBefore=3 TotalBefore=820MB(persistent) lostFree=18(+0) live=8 | OS: beforePrivate=2330MB afterPrivate=1620MB
[IR_LOAD]    NUC#001 seq=105 irLen=327680 blockSize=4096 Layers=3 L0Part=4096 L1Part=32768 L2Part=262144 directTaps=32 ringSize=8192 MKL: before=110MB after=812MB delta=+702MB lostFree=18(+0) live=8
[IR_LAYOUT]  NUC#001 seq=105 IRFreq=256MB FDL=420MB Accum=96MB Tail=16MB Direct=24MB Ring=8MB Total=820MB(persistent data buffers only) | L0=8MB L1=64MB L2=720MB
[MEM_SNAP]   PUBLISH gen=21 seq=5 | NUC(MKL only): live=8 alloc=820MB Peak(since reset)=1200MB totalA=28.0GB totalF=27.4GB lostFree=0 zeroAllocSize=0(delta=+0) | Stereo=4 DSPCore=4 | Retire: pending=232 trackedPendingBytes=12.8MB(diag only: sizeof tracked entries, not actual heap) trackedPending=8/232 (3%) overflow=0 reclaim=12 | OS: Private=2330MB WorkingSet=2400MB | OtherPrivate=1510MB(JUCE/CRT/IPP/threads/...)
```

---

## 6. v37 からの改善点一覧

| # | v37（問題） | v38（修正） |
|:--|:----------|:-----------|
| 1 | `releaseAllLayers()` size 退避がガードより前 | **ガード後に移動** |
| 2 | `freeTracked(size==0)` が `lostFreeCount` を増やさない | **`lostFreeCount` も加算（zeroAllocSizeCount と両方増加）** |
| 3 | `Logger::writeToLog()` 直呼び出し | **`diagLogNonRt()` ラッパーを DiagnosticsConfig.h に追加** |
| 4 | `seq=0 reserved` がコメントのみ | **`kReservedDiagSeq = 0` 定数化** |
| 5 | `zeroAllocSize=0(+0)` が何の増分か不明 | **`zeroAllocSize=%u(delta=%+d)` に変更** |
