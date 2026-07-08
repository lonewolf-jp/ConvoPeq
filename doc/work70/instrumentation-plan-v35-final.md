# メモリ占有調査のためのインストルメンテーション改修案 v35（最終版）

---

**目的**: 2.33GB メモリ占有の原因特定。推定ではなく実測で確定させる。
**v34 からの変更**: 3 点の修正。

---

## 0. v34 の問題点と v35 での修正方針

| # | v34 の問題 | v35 の修正 |
|:--|:----------|:----------|
| 1 | `releaseAllLayers()` で NUC バッファ解放サイズを後から再計算 → 状態リセット位置の変更で不整合リスク | **解放サイズを先にローカル変数に退避** |
| 2 | `freeTracked()` 内の `assert(size != 0)` → Release で `DIAG_MKL_FREE(ptr,0)` が呼ばれる | **size==0 なら mkl_free して nullptr。統計は更新しない。assert は全ビルドで有効に** |
| 3 | `addIfAlive()` の `DBG` が大量出力リスク | **カウンタ集計方式に変更（`zeroAllocSizeCount` in MklAllocStats）** |

---

## 1. Patch A: DiagnosticsConfig.h — freeTracked safe 化 + zeroAllocSizeCount

### A-1. MklAllocStats — zeroAllocSizeCount 追加

```cpp
namespace convo::diag {

struct MklAllocStats {
    std::atomic<uint64_t> allocatedBytes { 0 };
    std::atomic<uint64_t> peakBytes      { 0 };
    std::atomic<uint64_t> totalAllocBytes{ 0 };
    std::atomic<uint64_t> totalFreedBytes{ 0 };
    std::atomic<uint32_t> lostFreeCount  { 0 };
    // ★ v35: addIfAlive で allocSizes 保存漏れを検出した回数
    std::atomic<uint32_t> zeroAllocSizeCount { 0 };
};

// ... 既存の accessors, resetDiagnostics ...

} // namespace convo::diag
```

### A-2. freeTracked — size==0 でも safe に動作

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
            // size 不明でもリーク防止のため mkl_free を呼ぶ
            mkl_free(p);
        }
        p = nullptr;
    }
}
```

**★ `assert(size != 0)` は削除。size=0 でも確実に mkl_free を呼び、統計には影響しない。**
**★ Release でも確実に解放。診断コードの不整合が本体に波及しない。**

### A-3. addIfAlive — カウンタ集計方式

```cpp
static uint64_t addIfAlive(const double* ptr, size_t allocSize, const char* /*name*/) noexcept
{
    if (ptr)
    {
        if (allocSize == 0)
        {
            // ★ v35: ログ出力ではなくカウンタ集計
            mklStats().zeroAllocSizeCount.fetch_add(1, std::memory_order_relaxed);
        }
        return allocSize;
    }
    return 0;
}
```

**★ zeroAllocSizeCount は MEM_SNAP で出力可能。大量ログ出力リスクなし。**

---

## 2. Patch B: MKLNonUniformConvolver — releaseAllLayers サイズ退避

### B-1. releaseAllLayers — サイズを先にローカル変数に退避

```cpp
void MKLNonUniformConvolver::releaseAllLayers() noexcept
{
    // ★ v35: 状態リセット前に全解放サイズをローカル変数に退避
    //   将来 m_ringSize 等のリセット位置が変わっても正しいサイズで解放可能。
    const size_t ringBufBytes    = static_cast<size_t>(m_ringSize) * sizeof(double);
    const size_t directIRBytes   = static_cast<size_t>(m_directTapCount) * sizeof(double);
    const size_t directHistBytes = static_cast<size_t>(m_directHistLen) * sizeof(double);
    const size_t directWinBytes  = static_cast<size_t>(m_directHistLen + m_directMaxBlock) * sizeof(double);
    const size_t directOutBytes  = static_cast<size_t>(m_directMaxBlock) * sizeof(double);

    // ... 既存の guard チェック ...

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
    if (m_ringBuf) { mkl_free(m_ringBuf); m_ringBuf = nullptr; }
    if (m_directIRRev)   { mkl_free(m_directIRRev);   m_directIRRev = nullptr; }
    if (m_directHistory) { mkl_free(m_directHistory); m_directHistory = nullptr; }
    if (m_directWindow)  { mkl_free(m_directWindow);  m_directWindow = nullptr; }
    if (m_directOutBuf)  { mkl_free(m_directOutBuf);  m_directOutBuf = nullptr; }
#endif

    // ... 既存の状態リセット ...
}
```

---

## 3. 全ログフォーマット（v35 確定版、v34 から変更なし）

### IR_RELEASE（v35: live=%u 追加 — 推奨追加）
```
[IR_RELEASE] NUC#%p seq=%llu MKL: before=%lluMB after=%lluMB delta=%lldMB LayersBefore=%d TotalBefore=%.0fMB(persistent) lostFree=%u(+%d) live=%u | OS: beforePrivate=%lluMB afterPrivate=%lluMB
```

### IR_LOAD（v34 から変更なし）
```
[IR_LOAD] NUC#%p seq=%llu irLen=%d blockSize=%d Layers=%d L0Part=%d L1Part=%d L2Part=%d directTaps=%d ringSize=%d MKL: before=%lluMB after=%lluMB delta=%lldMB lostFree=%u(+%d) live=%u
```

### IR_LAYOUT（v34 から変更なし）
```
[IR_LAYOUT] NUC#%p seq=%llu IRFreq=%.0fMB FDL=%.0fMB Accum=%.0fMB Tail=%.0fMB Direct=%.0fMB Ring=%.0fMB Total=%.0fMB(persistent data buffers only) | L0=%.0fMB L1=%.0fMB L2=%.0fMB
```

### MEM_SNAP（zeroAllocSize 追加）
```
[MEM_SNAP] PUBLISH gen=%d seq=%d | NUC(MKL only): live=%u alloc=%.0fMB Peak(since reset)=%.0fMB totalA=%.0fGB totalF=%.0fGB lostFree=%u zeroAllocSize=%u | Stereo=%d DSPCore=%d | Retire: pending=%u trackedPendingBytes=%.1fMB(diag only: sizeof tracked entries, not actual heap) trackedPending=%u/%u (%.0f%%) overflow=%llu reclaim=%llu | OS: Private=%lluMB WorkingSet=%lluMB | OtherPrivate=%.0fMB(JUCE/CRT/IPP/threads/...)
```

---

## 4. 出力例（v35 最終版）

```text
[IR_RELEASE] NUC#001 seq=105 MKL: before=820MB after=110MB delta=-710MB LayersBefore=3 TotalBefore=820MB(persistent) lostFree=18(+0) live=8 | OS: beforePrivate=2330MB afterPrivate=1620MB
[IR_LOAD]    NUC#001 seq=105 irLen=327680 blockSize=4096 Layers=3 L0Part=4096 L1Part=32768 L2Part=262144 directTaps=32 ringSize=8192 MKL: before=110MB after=812MB delta=+702MB lostFree=18(+0) live=8
[IR_LAYOUT]  NUC#001 seq=105 IRFreq=256MB FDL=420MB Accum=96MB Tail=16MB Direct=24MB Ring=8MB Total=820MB(persistent data buffers only) | L0=8MB L1=64MB L2=720MB
[MEM_SNAP]   PUBLISH gen=21 seq=5 | NUC(MKL only): live=8 alloc=820MB Peak(since reset)=1200MB totalA=28.0GB totalF=27.4GB lostFree=0 zeroAllocSize=0 | Stereo=4 DSPCore=4 | Retire: pending=232 trackedPendingBytes=12.8MB(diag only) trackedPending=8/232 (3%) overflow=0 reclaim=12 | OS: Private=2330MB WorkingSet=2400MB | OtherPrivate=1510MB(JUCE/CRT/IPP/threads/...)
```

**IR_RELEASE に live=8**: Crossfade 中かどうかを即座に判断可能。

---

## 5. v34 からの改善点一覧

| # | v34（問題） | v35（修正） |
|:--|:----------|:-----------|
| 1 | `releaseAllLayers()` の NUC バッファサイズを後から参照 → 状態リセット位置変更で不整合リスク | **解放サイズを先にローカル変数に退避** |
| 2 | `freeTracked()` の `assert(size != 0)` → Release で `DIAG_MKL_FREE(ptr,0)` | **size=0 でも mkl_free 呼び出し + nullptr 代入。統計更新なしで安全動作** |
| 3 | `addIfAlive()` の DBG が大量出力リスク | **カウンタ `zeroAllocSizeCount` 集計方式に変更。MEM_SNAP で zeroAllocSize=%u を出力** |
