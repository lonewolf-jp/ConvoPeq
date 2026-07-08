# メモリ占有調査のためのインストルメンテーション改修案 v36（最終版）

---

**目的**: 2.33GB メモリ占有の原因特定。推定ではなく実測で確定させる。
**v35 からの変更**: 3 点の修正。

---

## 0. v35 の問題点と v36 での修正方針

| # | v35 の問題 | v36 の修正 |
|:--|:----------|:----------|
| 1 | `freeTracked(size==0)` が `mkl_free` だけで `lostFreeCount` を増やさない → 統計の不整合を説明できない | **`lostFreeCount` にも加算** |
| 2 | `zeroAllocSizeCount` が累積のみ → いつ増えたか不明。`resetDiagnostics` でリセットされない | **`resetDiagnostics()` でリセット + MEM_SNAP に `(+N)` 増分表示** |
| 3 | `releaseAllLayers()` のサイズ退避位置がガードチェックより前 → コード可読性 | **ガードチェック後に移動** |

---

## 1. Patch A: DiagnosticsConfig.h — freeTracked + zeroAllocSize + reset

### A-1. freeTracked — lostFreeCount 加算（★ v36）

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
            // ★ v36: size 不明でも lostFreeCount を増やすことで、
            //   allocatedBytes が減らない理由を統計的に説明可能
            convo::diag::mklStats().lostFreeCount.fetch_add(1, std::memory_order_relaxed);
            mkl_free(p);
        }
        p = nullptr;
    }
}
```

### A-2. MklAllocStats — zeroAllocSizeCount（v36 は前回値保存方式）

```cpp
struct MklAllocStats {
    std::atomic<uint64_t> allocatedBytes { 0 };
    std::atomic<uint64_t> peakBytes      { 0 };
    std::atomic<uint64_t> totalAllocBytes{ 0 };
    std::atomic<uint64_t> totalFreedBytes{ 0 };
    std::atomic<uint32_t> lostFreeCount  { 0 };
    std::atomic<uint32_t> zeroAllocSizeCount { 0 };  // addIfAlive 検出用
};
```

### A-3. resetDiagnostics — zeroAllocSizeCount もリセット

```cpp
inline void resetDiagnostics() noexcept
{
    mklStats().peakBytes.store(
        mklStats().allocatedBytes.load(std::memory_order_relaxed),
        std::memory_order_relaxed);
    mklStats().totalAllocBytes.store(0, std::memory_order_relaxed);
    mklStats().totalFreedBytes.store(0, std::memory_order_relaxed);
    mklStats().lostFreeCount.store(0, std::memory_order_relaxed);
    mklStats().zeroAllocSizeCount.store(0, std::memory_order_relaxed);  // ★ v36
}
```

---

## 2. Patch B: MKLNonUniformConvolver — releaseAllLayers 順序修正

```cpp
void MKLNonUniformConvolver::releaseAllLayers() noexcept
{
    // ★ v36: ガードチェック（ポインタ異常や NUC_DEBUG_GUARDS）を先に実施
#ifdef NUC_DEBUG_GUARDS
    checkGuards();
    // ... 既存の checkPtr ...
#endif

    // ★ v36: ガードチェック後に解放サイズを退避（可読性向上）
    //   状態リセット前に確定させ、将来のリセット位置変更に耐える。
    const size_t ringBufBytes    = static_cast<size_t>(m_ringSize) * sizeof(double);
    const size_t directIRBytes   = static_cast<size_t>(m_directTapCount) * sizeof(double);
    const size_t directHistBytes = static_cast<size_t>(m_directHistLen) * sizeof(double);
    const size_t directWinBytes  = static_cast<size_t>(m_directHistLen + m_directMaxBlock) * sizeof(double);
    const size_t directOutBytes  = static_cast<size_t>(m_directMaxBlock) * sizeof(double);

    for (int i = 0; i < kNumLayers; ++i)
        m_layers[i].freeAll();
    m_numActiveLayers = 0;
    m_latency         = 0;

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    freeTracked(m_ringBuf,       ringBufBytes);
    freeTracked(m_directIRRev,   directIRBytes);
    freeTracked(m_directHistory, directHistBytes);
    freeTracked(m_directWindow,  directWinBytes);
    freeTracked(m_directOutBuf,  directOutBytes);
#else
    // ... 既存の mkl_free ...
#endif

    // ... 既存の状態リセット ...
}
```

---

## 3. MEM_SNAP — zeroAllocSize 増分表示（★ v36）

```cpp
// MEM_SNAP 内で zeroAllocSize の増分を計算
static uint32_t lastZeroAllocSize = 0;
const uint32_t curZeroAllocSize = convo::diag::mklStats().zeroAllocSizeCount.load(std::memory_order_relaxed);
const int32_t deltaZero = static_cast<int32_t>(curZeroAllocSize) - static_cast<int32_t>(lastZeroAllocSize);
lastZeroAllocSize = curZeroAllocSize;
```

```
[MEM_SNAP] PUBLISH ... | NUC(MKL only): live=%u alloc=%.0fMB Peak(since reset)=%.0fMB totalA=%.0fGB totalF=%.0fGB lostFree=%u zeroAllocSize=%u(+%d) | Stereo=%d DSPCore=%d | ...
```

---

## 4. 全ログフォーマット（v36 確定版）

### IR_RELEASE（live=%u 追加済み）
```
[IR_RELEASE] NUC#%p seq=%llu MKL: before=%lluMB after=%lluMB delta=%lldMB LayersBefore=%d TotalBefore=%.0fMB(persistent) lostFree=%u(+%d) live=%u | OS: beforePrivate=%lluMB afterPrivate=%lluMB
```

### IR_LOAD（v35 から変更なし）
```
[IR_LOAD] NUC#%p seq=%llu irLen=%d blockSize=%d Layers=%d L0Part=%d L1Part=%d L2Part=%d directTaps=%d ringSize=%d MKL: before=%lluMB after=%lluMB delta=%lldMB lostFree=%u(+%d) live=%u
```

### IR_LAYOUT（v35 から変更なし）
```
[IR_LAYOUT] NUC#%p seq=%llu IRFreq=%.0fMB FDL=%.0fMB Accum=%.0fMB Tail=%.0fMB Direct=%.0fMB Ring=%.0fMB Total=%.0fMB(persistent data buffers only) | L0=%.0fMB L1=%.0fMB L2=%.0fMB
```

### MEM_SNAP（zeroAllocSize 増分表示）
```
[MEM_SNAP] PUBLISH gen=%d seq=%d | NUC(MKL only): live=%u alloc=%.0fMB Peak(since reset)=%.0fMB totalA=%.0fGB totalF=%.0fGB lostFree=%u zeroAllocSize=%u(+%d) | Stereo=%d DSPCore=%d | Retire: pending=%u trackedPendingBytes=%.1fMB(diag only) trackedPending=%u/%u (%.0f%%) overflow=%llu reclaim=%llu | OS: Private=%lluMB WorkingSet=%lluMB | OtherPrivate=%.0fMB(JUCE/CRT/IPP/threads/...)
```

---

## 5. 出力例（v36 最終版）

```text
[IR_RELEASE] NUC#001 seq=105 MKL: before=820MB after=110MB delta=-710MB LayersBefore=3 TotalBefore=820MB(persistent) lostFree=18(+0) live=8 | OS: beforePrivate=2330MB afterPrivate=1620MB
[IR_LOAD]    NUC#001 seq=105 irLen=327680 blockSize=4096 Layers=3 L0Part=4096 L1Part=32768 L2Part=262144 directTaps=32 ringSize=8192 MKL: before=110MB after=812MB delta=+702MB lostFree=18(+0) live=8
[IR_LAYOUT]  NUC#001 seq=105 IRFreq=256MB FDL=420MB Accum=96MB Tail=16MB Direct=24MB Ring=8MB Total=820MB(persistent data buffers only) | L0=8MB L1=64MB L2=720MB
[MEM_SNAP]   PUBLISH gen=21 seq=5 | NUC(MKL only): live=8 alloc=820MB Peak(since reset)=1200MB totalA=28.0GB totalF=27.4GB lostFree=0 zeroAllocSize=0(+0) | Stereo=4 DSPCore=4 | Retire: pending=232 trackedPendingBytes=12.8MB(diag only) trackedPending=8/232 (3%) overflow=0 reclaim=12 | OS: Private=2330MB WorkingSet=2400MB | OtherPrivate=1510MB(JUCE/CRT/IPP/threads/...)
```

---

## 6. v35 からの改善点一覧

| # | v35（問題） | v36（修正） |
|:--|:----------|:-----------|
| 1 | `freeTracked(size==0)` が `lostFreeCount` を増やさない → allocatedBytes 減らない理由を統計的に説明不可 | **`lostFreeCount` も加算。統計の整合性が向上** |
| 2 | `zeroAllocSizeCount` が累積のみ + `resetDiagnostics()` でリセットされない | **`resetDiagnostics()` でリセット + MEM_SNAP で `(+N)` 増分表示** |
| 3 | `releaseAllLayers()` のサイズ退避がガードチェックより前 → コード可読性 | **ガードチェック後に移動** |
