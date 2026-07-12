# メモリ占有調査のためのインストルメンテーション改修案 v13（最終版）— DiagnosticsSnapshot + ポインタ生存確認 + LostFree 追跡

---

**目的**: 2.33GB メモリ占有の原因特定。推定ではなく実測で確定させる。
**v12 からの変更**: 6 点の実装上の問題を修正。

---

## 0. v12 の問題点と v13 での修正方針

| # | v12 の問題 | v13 の修正 |
|:--|:----------|:----------|
| 1 | `CategoryStats` が `allocSizes` のみ集計 → 実際のポインタ生存状態を反映していない | **ポインタ nullptr チェック後に集計** + ログ名を `LayerBuf` に変更 |
| 2 | `diagMklFree(size==0)` で統計更新スキップ → 実リークと統計漏れが区別不可 | **`lostFreeCount` / `lostFreeBytes` 専用カウンタ追加** |
| 3 | `computeNucCategoryStats()` が `Layer` 内部構造に依存 | **`DiagnosticsSnapshot` 構造体を公開 API として分離** |
| 4 | FFT プラン / IPP 内部 / CRT 等のカテゴリがない | **`HeapStats` 構造体で大まかなカテゴリ分類を追加** |
| 5 | カテゴリ別に Peak / Current がない | **`CategoryStats` に `peakBytes` 追加** |
| 6 | `computeNucCategoryStats()` でポインタ nullptr を無視 | **`if (ptr)` ガードを追加** |

---

## 1. Patch A: DiagnosticsConfig.h — lostFree 追跡 + HeapStats + DiagnosticsSnapshot

### A-1. MklAllocStats — lostFree カウンタ追加（★ v13）

```cpp
namespace convo::diag {

struct MklAllocStats {
    std::atomic<uint64_t> allocatedBytes { 0 };
    std::atomic<uint64_t> peakBytes      { 0 };
    std::atomic<uint64_t> totalAllocBytes{ 0 };
    std::atomic<uint64_t> totalFreedBytes{ 0 };
    // ★ v13: size==0 で diagMklFree が呼ばれた回数と推定未追跡バイト数
    std::atomic<uint32_t> lostFreeCount   { 0 };
    std::atomic<uint64_t> lostFreeBytes   { 0 };
};

// ... diagMklMalloc, accessors, resetDiagnostics ...

inline void diagMklFree(void* ptr, size_t size) noexcept
{
    if (ptr)
    {
        mkl_free(ptr);

        if (size > 0)
        {
            mklStats().allocatedBytes.fetch_sub(
                static_cast<uint64_t>(size), std::memory_order_relaxed);
            mklStats().totalFreedBytes.fetch_add(
                static_cast<uint64_t>(size), std::memory_order_relaxed);
        }
        else
        {
            // ★ v13: size 不明でもカウントのみ記録（統計漏れと実リークの区別用）
            mklStats().lostFreeCount.fetch_add(1, std::memory_order_relaxed);
            DBG("[DIAG] diagMklFree called with size=0, ptr=" << ptr);
        }
    }
}

// ★ v13: 失われた解放情報
[[nodiscard]] inline uint32_t lostFreeCount() noexcept
{
    return mklStats().lostFreeCount.load(std::memory_order_relaxed);
}
[[nodiscard]] inline uint64_t lostFreeBytes() noexcept
{
    return mklStats().lostFreeBytes.load(std::memory_order_relaxed);
}

```cpp
inline void resetDiagnostics() noexcept
{
    mklStats().peakBytes.store(
        mklStats().allocatedBytes.load(std::memory_order_relaxed),
        std::memory_order_relaxed);
    mklStats().totalAllocBytes.store(0, std::memory_order_relaxed);
    mklStats().totalFreedBytes.store(0, std::memory_order_relaxed);
    mklStats().lostFreeCount.store(0, std::memory_order_relaxed);   // ★ v13
    mklStats().lostFreeBytes.store(0, std::memory_order_relaxed);   // ★ v13
}
```

### A-2. CategoryStats — peakBytes 追加 + ポインタ生存確認（★ v13）

```cpp
/// ★ v13: NUC カテゴリ別集計（生存ポインタのみ集計、peak 付き）。
struct NucCategoryStats {
    uint64_t layerBufs[3] = { 0, 0, 0 }; // Layer 0/1/2 の MKL 永続バッファ
    uint64_t direct      = 0;            // Direct FIR
    uint64_t ring        = 0;            // 出力リング
    uint64_t fftPlan     = 0;            // IPP FFT plan / work (概算)
    uint64_t peakLayerBufs[3] = { 0, 0, 0 };
    // ...
};
```

### A-3. HeapStats — 大まかなカテゴリ分類（★ v13）

```cpp
/// ★ v13: ヒープカテゴリ別大まかな集計（Publish ログ用）。
///   各カテゴリの値は概算。正確値はツール（VMMap等）で確認のこと。
struct HeapCategoryStats {
    uint64_t mklBytes    = 0;  // convo::diag::allocatedBytes()
    uint64_t ippBytes    = 0;  // IPP FFT spec/work (概算、NUC Category から算出)
    uint64_t retireBytes = 0;  // ISRRetireRouter::pendingRetireBytes()
    uint64_t osPrivateMB = 0;  // GetProcessMemoryInfo().privateUsageMB
    // 以下は概算で算出:
    uint64_t untrackedMB  = 0; // osPrivate - mkl - ipp - retire
};
```

### A-4. DiagnosticsSnapshot — NUC 診断用公開 API（★ v13）

```cpp
/// ★ v13: NUC 診断用スナップショット。
///   MKLNonUniformConvolver::getDiagnostics() が返す。
///   外部は Layer 構造体を知る必要がない。
struct NucDiagnosticsSnapshot {
    uint64_t layerBufs[3] = { 0, 0, 0 }; // 生存 Layer バッファの合計
    uint64_t layerPeaks[3]= { 0, 0, 0 };
    uint64_t directBytes  = 0;
    uint64_t ringBytes    = 0;
    uint64_t fftPlanBytes = 0;  // IPP 概算
    uint64_t totalMklBytes= 0;  // allocatedBytes()
    uint64_t peakMklBytes = 0;
};
```

---

## 2. Patch B: MKLNonUniformConvolver — DiagnosticsSnapshot API

### B-1. getDiagnostics() 追加（★ v13）

```cpp
class MKLNonUniformConvolver {
public:
    static std::atomic<uint32_t> liveCount;

    /// ★ v13: 診断用スナップショット（Layer 内部構造を公開しない）。
    [[nodiscard]] NucDiagnosticsSnapshot getDiagnostics() const noexcept
    {
        NucDiagnosticsSnapshot snap{};
        snap.totalMklBytes = convo::diag::allocatedBytes();
        snap.peakMklBytes  = convo::diag::peakBytes();

        for (int li = 0; li < kNumLayers; ++li)
        {
            const Layer& l = m_layers[li];
            uint64_t layerTotal = 0;
            layerTotal += addIfAlive(l.irFreqDomain,  l.allocSizes.irFreqDomain);
            layerTotal += addIfAlive(l.irFreqReal,    l.allocSizes.irFreqReal);
            layerTotal += addIfAlive(l.irFreqImag,    l.allocSizes.irFreqImag);
            layerTotal += addIfAlive(l.fdlBuf,        l.allocSizes.fdlBuf);
            layerTotal += addIfAlive(l.fdlReal,       l.allocSizes.fdlReal);
            layerTotal += addIfAlive(l.fdlImag,       l.allocSizes.fdlImag);
            layerTotal += addIfAlive(l.fftTimeBuf,    l.allocSizes.fftTimeBuf);
            layerTotal += addIfAlive(l.fftOutBuf,     l.allocSizes.fftOutBuf);
            layerTotal += addIfAlive(l.prevInputBuf,  l.allocSizes.prevInputBuf);
            layerTotal += addIfAlive(l.accumBuf,      l.allocSizes.accumBuf);
            layerTotal += addIfAlive(l.accumReal,     l.allocSizes.accumReal);
            layerTotal += addIfAlive(l.accumImag,     l.allocSizes.accumImag);
            layerTotal += addIfAlive(l.inputAccBuf,   l.allocSizes.inputAccBuf);
            layerTotal += addIfAlive(l.tailOutputBuf, l.allocSizes.tailOutputBuf);
            snap.layerBufs[li] = layerTotal;

            // FFT plan 概算: fftTimeBuf と fftOutBuf が生きていれば +α
            if (l.fftTimeBuf && l.fftOutBuf)
                snap.fftPlanBytes += l.allocSizes.fftTimeBuf / 4; // 概算
        }

        // Direct + Ring
        snap.directBytes = addIfAlive(m_directIRRev,
            static_cast<size_t>(m_directTapCount) * sizeof(double));
        snap.ringBytes   = addIfAlive(m_ringBuf,
            static_cast<size_t>(m_ringSize) * sizeof(double));
        // ...
        return snap;
    }
```

### B-2. 内部ヘルパ（private）

```cpp
private:
    /// ★ v13: ポインタ生存確認付き加算。
    static uint64_t addIfAlive(const double* ptr, size_t allocSize) noexcept
    {
        return ptr ? allocSize : 0;
    }
```

### B-3. NUC_MEM ログ — getDiagnostics 使用

```cpp
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
{
    const auto snap = getDiagnostics();
    const uint64_t curBytes  = convo::diag::allocatedBytes();
    const uint64_t peakBytes = convo::diag::peakBytes();
    const uint64_t totalA    = convo::diag::totalAllocBytes();
    const uint64_t totalF    = convo::diag::totalFreedBytes();
    const uint32_t lostFree  = convo::diag::lostFreeCount();
    const auto osMem = getProcessMemoryInfo();
    const uint64_t retireBytes = /* pendingRetireBytes() */;
    const int64_t untracked = (int64_t)osMem.privateUsageMB * 1024 * 1024
                            - (int64_t)curBytes - (int64_t)retireBytes;

    diagLog(juce::String::formatted(
        "[NUC_MEM] NUC#%p | "
        "LayerBuf: L0=%.0fMB L1=%.0fMB L2=%.0fMB "
        "Direct=%.0fMB Ring=%.0fMB FFTPlan~=%.0fMB | "
        "MKL: cur=%.0fMB Peak(reset)=%.0fMB "
        "totalA=%.0fGB totalF=%.0fGB lostFree=%u live=%d | "
        "OS: Private=%lluMB WorkingSet=%lluMB | "
        "Untracked(other)=%.0fMB",
        (void*)this,
        snap.layerBufs[0] / (1024.0*1024.0),
        snap.layerBufs[1] / (1024.0*1024.0),
        snap.layerBufs[2] / (1024.0*1024.0),
        snap.directBytes / (1024.0*1024.0),
        snap.ringBytes   / (1024.0*1024.0),
        snap.fftPlanBytes/ (1024.0*1024.0),
        curBytes / (1024.0*1024.0),
        peakBytes / (1024.0*1024.0),
        totalA / (1024.0*1024.0*1024.0),
        totalF / (1024.0*1024.0*1024.0),
        (unsigned)lostFree,
        (unsigned)liveCount.load(std::memory_order_relaxed),
        (unsigned long long)osMem.privateUsageMB,
        (unsigned long long)osMem.workingSetMB,
        std::max(0LL, untracked) / (1024.0 * 1024.0)));
}
#endif
```

---

## 3. MEM_SNAP ログ — HeapCategoryStats 追加（★ v13）

```cpp
    const auto osMem = getProcessMemoryInfo();
    const uint64_t nucBytes    = convo::diag::allocatedBytes();
    const uint64_t retireBytes = m_retireRouter ? m_retireRouter->pendingRetireBytes() : 0;
    const uint32_t lostFree    = convo::diag::lostFreeCount();
    const uint64_t ippEstimate = /* 概算: NUC から FFT plan 分を加算 */;

    juce::Logger::writeToLog(juce::String::formatted(
        "[MEM_SNAP] PUBLISH gen=%d seq=%d | "
        "NUC(MKL only): live=%d alloc=%.0fMB Peak(reset)=%.0fMB "
        "totalA=%.0fGB totalF=%.0fGB lostFree=%u | "
        "Stereo=%d DSPCore=%d | "
        "Retire: pending=%u objBytes=%.1fMB(sizeof) tracked=%u/%u (%.0f%%) "
        "overflow=%llu reclaim=%llu | "
        "OS: Private=%lluMB WorkingSet=%lluMB | "
        "Heap: MKL=%.0fMB Retire=%.0fMB IPP~=%.0fMB | "
        "Untracked(other)=%.0fMB(JUCE/CRT/threads/...)", ...));
```

---

## 4. 出力例（v13）

```text
[NUC_MEM] NUC#0000001234 | LayerBuf: L0=8MB L1=64MB L2=0MB Direct=1MB Ring=2MB FFTPlan~=3MB | MKL: cur=78MB Peak(reset)=142MB totalA=3.2GB totalF=3.14GB lostFree=0 live=2 | OS: Private=88MB WorkingSet=140MB | Untracked(other)=10MB

[MEM_SNAP] PUBLISH gen=8 | NUC(MKL only): live=8 alloc=832MB Peak(reset)=1200MB totalA=28.0GB totalF=27.4GB lostFree=3 | Stereo=4 DSPCore=4 | Retire: pending=232 objBytes=12.8MB(sizeof) tracked=8/232 (3%) overflow=0 reclaim=12 | OS: Private=2330MB WorkingSet=2400MB | Heap: MKL=832MB Retire=12MB IPP~=24MB | Untracked(other)=1462MB(JUCE/CRT/threads/...)
```

**lostFree=3** → 3回の解放でサイズが不明だったことが分かる（統計漏れと実リークを区別可能）。

---

## 5. v12 からの改善点一覧

| 項目 | v12（問題） | v13（修正） |
|:-----|:----------|:-----------|
| CategoryStats のポインタ確認 | `allocSizes` を無条件集計 | **`addIfAlive()` でポインタ nullptr チェック後集計** |
| ログ名の曖昧さ | `Cat: L0=...` | **`LayerBuf: L0=...`** |
| diagMklFree(size==0) 統計漏れ | 警告のみ（実リークと区別不可） | **`lostFreeCount` / `lostFreeBytes` カウンタ追加** |
| 公開 API の設計 | `computeNucCategoryStats()` + friend | **`NucDiagnosticsSnapshot` + `getDiagnostics()`** |
| IPP/FFT plan の概算 | なし | **`fftPlanBytes` 概算追加** |
| ヒープカテゴリ分類 | なし | **MEM_SNAP に `Heap: MKL= Retire= IPP~=` 追加** |
