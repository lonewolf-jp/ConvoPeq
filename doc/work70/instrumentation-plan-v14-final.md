# メモリ占有調査のためのインストルメンテーション改修案 v14（最終版）— 責務分離完了版

---

**目的**: 2.33GB メモリ占有の原因特定。推定ではなく実測で確定させる。
**v13 からの変更**: 7 点の修正。インスタンス統計とグローバル統計の責務を完全分離。

---

## 0. v13 の問題点と v14 での修正方針

| # | v13 の問題 | v14 の修正 |
|:--|:----------|:----------|
| 1 | `lostFreeBytes` が未更新で意味がない | **削除（`lostFreeCount` のみ保持）** |
| 2 | `fftPlanBytes` が完全な推定値（実測と誤認される） | **削除（NUC の責務外）** |
| 3 | `peakLayerBufs` が未更新 | **削除（peak はグローバル統計のみ）** |
| 4 | `getDiagnostics()` がグローバル統計を含む（責務混在） | **インスタンス固有情報のみ返す。グローバル統計はログ側で取得** |
| 5 | `addIfAlive()` に `ptr!=nullptr && size==0` のアサーションなし | **`jassert` 追加** |
| 6 | HeapCategoryStats が MKL/IPP/Retire のみ | **CRT ヒープ概算を追加（Windows API）** |
| 7 | 責務分離が不完全 | **Snapshot はインスタンス情報のみ、グローバル統計との取得箇所を分離** |

---

## 1. Patch A: DiagnosticsConfig.h — lostFreeBytes 削除 + CRT 統計 + 責務整理

### A-1. MklAllocStats — lostFreeBytes 削除

```cpp
namespace convo::diag {

struct MklAllocStats {
    std::atomic<uint64_t> allocatedBytes { 0 };
    std::atomic<uint64_t> peakBytes      { 0 };
    std::atomic<uint64_t> totalAllocBytes{ 0 };
    std::atomic<uint64_t> totalFreedBytes{ 0 };
    std::atomic<uint32_t> lostFreeCount  { 0 };  // size=0 で呼ばれた回数
};

// ... diagMklMalloc, diagMklFree, accessors, ...
```

`diagMklFree` から `lostFreeBytes` の参照を削除。

```cpp
    else
    {
        mklStats().lostFreeCount.fetch_add(1, std::memory_order_relaxed);
        DBG("[DIAG] diagMklFree size=0, ptr=" << ptr);
    }
```

### A-2. グローバル統計アクセッサ（責務明確化のコメント追加）

```cpp
/// ★ グローバル MKL 統計（全 NUC インスタンス合計）。
///   インスタンス単位の集計には NucDiagnosticsSnapshot を使用すること。
[[nodiscard]] inline uint64_t allocatedBytes() noexcept { ... }
[[nodiscard]] inline uint64_t peakBytes() noexcept { ... }
[[nodiscard]] inline uint64_t totalAllocBytes() noexcept { ... }
[[nodiscard]] inline uint64_t totalFreedBytes() noexcept { ... }
[[nodiscard]] inline uint32_t lostFreeCount() noexcept { ... }
```

### A-3. resetDiagnostics（lostFreeBytes のリセット行を削除）

```cpp
inline void resetDiagnostics() noexcept
{
    mklStats().peakBytes.store(
        mklStats().allocatedBytes.load(std::memory_order_relaxed),
        std::memory_order_relaxed);
    mklStats().totalAllocBytes.store(0, std::memory_order_relaxed);
    mklStats().totalFreedBytes.store(0, std::memory_order_relaxed);
    mklStats().lostFreeCount.store(0, std::memory_order_relaxed);
}
```

### A-4. CRT ヒープ概算（Windows API）

```cpp
/// ★ v14: CRT ヒープ使用量概算（malloc/free の差）。
///   SetImpulseなどメッセージスレッドからのみ呼び出すこと。
inline uint64_t getCrtHeapEstimate() noexcept
{
    _HEAPINFO hinfo = { 0 };
    int heapStatus;
    uint64_t totalAllocated = 0;
    uint64_t totalFree = 0;

    // 注意: _heapwalk はデバッグ用途・低速。診断時のみ。
    // 実際の実装では _heapwalk ではなく GetProcessHeaps の方が安全。
    // ここでは方向性のみ示す。
    return 0;  // ★ 本実装時には適切な Windows API を使用すること
}
```

---

## 2. Patch B: MKLNonUniformConvolver — 責務分離した DiagnosticsSnapshot

### B-1. NucDiagnosticsSnapshot（インスタンス情報のみ）

```cpp
/// ★ v14: NUC インスタンス単位の診断スナップショット。
///   グローバル統計（allocatedBytes 等）は含まない。
///   グローバル統計が必要な呼び出し元は convo::diag::XXX() を別途取得すること。
struct NucDiagnosticsSnapshot {
    // Layer 別生存バッファ合計（ポインタ生存確認後）
    uint64_t layerBufs[3] = { 0, 0, 0 };
    // Direct FIR + 出力リング
    uint64_t directBytes  = 0;
    uint64_t ringBytes    = 0;
    // 生存 Layer 数 / IR 状態
    int      numActiveLayers = 0;
    bool     isReady         = false;
};
```

**★ `totalMklBytes`/`peakMklBytes`/`fftPlanBytes`/`peakLayerBufs` は削除。**

### B-2. getDiagnostics() — インスタンス情報のみ

```cpp
class MKLNonUniformConvolver {
public:
    static std::atomic<uint32_t> liveCount;

    [[nodiscard]] NucDiagnosticsSnapshot getDiagnostics() const noexcept
    {
        NucDiagnosticsSnapshot snap{};
        snap.numActiveLayers = m_numActiveLayers;
        snap.isReady  = convo::consumeAtomic(m_ready, std::memory_order_acquire);

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
        }

        snap.directBytes = addIfAlive(m_directIRRev,
            static_cast<size_t>(m_directTapCount) * sizeof(double));
        snap.ringBytes   = addIfAlive(m_ringBuf,
            static_cast<size_t>(m_ringSize) * sizeof(double));
        return snap;
    }
```

### B-3. addIfAlive — デバッグアサーション追加

```cpp
private:
    /// ★ v14: ポインタ生存確認付き加算。
    ///   ptr!=nullptr && size==0 はバグ（allocSizes 保存漏れ）を検出。
    static uint64_t addIfAlive(const double* ptr, size_t allocSize) noexcept
    {
        jassert(ptr == nullptr || allocSize != 0);  // size==0 なら allocSizes 保存漏れ
        return ptr ? allocSize : 0;
    }
```

### B-4. NUC_MEM ログ（責務分離後の形）

```cpp
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
{
    const auto snap = getDiagnostics();  // ★ インスタンス情報
    const uint64_t curBytes  = convo::diag::allocatedBytes();  // ★ グローバル統計
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
        "Direct=%.0fMB Ring=%.0fMB | "
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

## 3. MEM_SNAP ログ（v14）

```cpp
    const uint64_t nucBytes    = convo::diag::allocatedBytes();
    const uint64_t nucPeak     = convo::diag::peakBytes();
    const uint64_t nucTotalA   = convo::diag::totalAllocBytes();
    const uint64_t nucTotalF   = convo::diag::totalFreedBytes();
    const uint32_t lostFree    = convo::diag::lostFreeCount();
    const uint64_t retireBytes = m_retireRouter ? m_retireRouter->pendingRetireBytes() : 0;

    juce::Logger::writeToLog(juce::String::formatted(
        "[MEM_SNAP] PUBLISH gen=%d seq=%d | "
        "NUC(MKL only): live=%d alloc=%.0fMB Peak(reset)=%.0fMB "
        "totalA=%.0fGB totalF=%.0fGB lostFree=%u | "
        "Stereo=%d DSPCore=%d | "
        "Retire: pending=%u objBytes=%.1fMB(sizeof) tracked=%u/%u (%.0f%%) "
        "overflow=%llu reclaim=%llu | "
        "OS: Private=%lluMB WorkingSet=%lluMB | "
        "Untracked(other)=%.0fMB(JUCE/CRT/threads/...)",
        gen, seq,
        (int)MKLNonUniformConvolver::liveCount.load(std::memory_order_relaxed),
        nucBytes / (1024.0*1024.0), nucPeak / (1024.0*1024.0),
        nucTotalA / (1024.0*1024.0*1024.0), nucTotalF / (1024.0*1024.0*1024.0),
        (unsigned)lostFree,
        // ...
        untrackedMB));
```

---

## 4. 出力例（v14）

```text
[NUC_MEM] NUC#0000001234 | LayerBuf: L0=8MB L1=64MB L2=0MB Direct=1MB Ring=2MB | MKL: cur=78MB Peak(reset)=142MB totalA=3.2GB totalF=3.14GB lostFree=0 live=2 | OS: Private=88MB WorkingSet=140MB | Untracked(other)=10MB

[MEM_SNAP] PUBLISH gen=8 | NUC(MKL only): live=8 alloc=832MB Peak(reset)=1200MB totalA=28.0GB totalF=27.4GB lostFree=3 | Stereo=4 DSPCore=4 | Retire: pending=232 objBytes=12.8MB(sizeof) tracked=8/232 (3%) overflow=0 reclaim=12 | OS: Private=2330MB WorkingSet=2400MB | Untracked(other)=1462MB(JUCE/CRT/threads/...)
```

---

## 5. v13 からの改善点一覧

| # | v13（問題） | v14（修正） |
|:--|:----------|:-----------|
| 1 | `lostFreeBytes` 未更新 | **削除（`lostFreeCount` のみ）** |
| 2 | `fftPlanBytes` が完全な推定値 | **削除（NUC の責務外）** |
| 3 | `peakLayerBufs` 未更新 | **削除（peak はグローバル統計のみ）** |
| 4 | `getDiagnostics()` がグローバル統計を含む | **インスタンス情報のみ返す。`totalMklBytes`/`peakMklBytes` 削除** |
| 5 | `addIfAlive()` にアサーションなし | **`jassert(ptr==nullptr \|\| allocSize!=0)` 追加** |
| 6 | HeapCategoryStats が IPP 推定値を含む | **推定値系は削除、CRT 概算は方針のみ記載** |
| 7 | 責務分離が不完全 | **Snapshot=インスタンス情報、グローバル統計=ログ側、と完全分離** |
