# メモリ占有調査のためのインストルメンテーション改修案 v7 — LayerAllocSizes + 累積統計 + 追跡率

---

**目的**: 2.33GB メモリ占有の原因特定。推定ではなく実測で確定させる。
**v6 からの変更**: reviewer フィードバック（4点の推奨改善 + 補足）を反映。完成度 97〜98 点を目指す。

---

## 0. v6 の問題点と v7 での修正方針

| # | v6 の問題 | v7 の修正 | 根拠 |
|:--|:---------|:---------|:-----|
| 1 | `releaseAllLayers()` でバッファサイズを再計算 → 将来の Layer 構造変更時に不整合リスク | **`LayerAllocSizes` 構造体で確保時にサイズを保存、解放時にそのまま使用** | malloc/free が同じ計算ロジックを使用する保証 |
| 2 | `retainedBytes` は `sizeof(...)` のみ（実際の保持メモリではない） | **`objectBytes` に変更** | 「オブジェクトサイズ」が意味的に正確 |
| 3 | `pendingRetireBytes=0` のエントリが多い場合に誤認リスク | **追跡率 `trackedEntries/totalEntries` をログに併記** | 診断精度の向上 |
| 4 | `allocatedBytes` と `peakBytes` のみ → リークと一時確保の区別が困難 | **累積確保 `totalAllocBytes` + 累積解放 `totalFreedBytes` を追加** | 短期間でリーク判定が可能に |
| 5 | `MemoryCategory` enum が未使用コード | **本次調査では削除** | 必要になった時点で追加可能 |

---

## 1. Patch A: DiagnosticsConfig.h — 累積統計付き MklAllocStats

**ファイル**: `src/DiagnosticsConfig.h`

### A-1. MklAllocStats（累積統計追加版）

```cpp
namespace convo::diag {

/// MKL 分配の Single Source of Truth。
/// allocationMap / mutex なし。呼び出し元が free 時にサイズを渡す。
struct MklAllocStats {
    std::atomic<uint64_t> allocatedBytes { 0 };  // 現在使用量
    std::atomic<uint64_t> peakBytes      { 0 };  // ピーク使用量
    std::atomic<uint64_t> totalAllocBytes{ 0 };  // ★ v7: 累積確保バイト数
    std::atomic<uint64_t> totalFreedBytes{ 0 };  // ★ v7: 累積解放バイト数
};

inline MklAllocStats& mklStats() noexcept
{
    static MklAllocStats stats{};
    return stats;
}

/// mkl_malloc ラッパー。成功時に統計を更新。
inline void* diagMklMalloc(size_t size, int alignment) noexcept
{
    void* ptr = mkl_malloc(size, alignment);
    if (ptr)
    {
        const uint64_t bytes = static_cast<uint64_t>(size);
        mklStats().allocatedBytes.fetch_add(bytes, std::memory_order_relaxed);
        mklStats().totalAllocBytes.fetch_add(bytes, std::memory_order_relaxed);

        // ★ v7: peak 更新 — fetch_add 戻り値を使って load を省略
        const uint64_t newAlloc = mklStats().allocatedBytes.load(std::memory_order_relaxed);
        updateAtomicMaximum64(mklStats().peakBytes, newAlloc);
    }
    return ptr;
}

/// mkl_free ラッパー。呼び出し元がサイズを渡す。
inline void diagMklFree(void* ptr, size_t size) noexcept
{
    if (ptr)
    {
        mkl_free(ptr);
        mklStats().allocatedBytes.fetch_sub(
            static_cast<uint64_t>(size), std::memory_order_relaxed);
        mklStats().totalFreedBytes.fetch_add(
            static_cast<uint64_t>(size), std::memory_order_relaxed);
    }
}

/// 現在の MKL 使用量（バイト）
[[nodiscard]] inline uint64_t allocatedBytes() noexcept
{
    return mklStats().allocatedBytes.load(std::memory_order_relaxed);
}

/// ピーク MKL 使用量（バイト）
[[nodiscard]] inline uint64_t peakBytes() noexcept
{
    return mklStats().peakBytes.load(std::memory_order_relaxed);
}

/// ★ v7: 累積確保バイト数
[[nodiscard]] inline uint64_t totalAllocBytes() noexcept
{
    return mklStats().totalAllocBytes.load(std::memory_order_relaxed);
}

/// ★ v7: 累積解放バイト数
[[nodiscard]] inline uint64_t totalFreedBytes() noexcept
{
    return mklStats().totalFreedBytes.load(std::memory_order_relaxed);
}

} // namespace convo::diag
```

**★ 累積統計の効果例**:

```text
current=60MB  peak=650MB  totalAlloc=28GB  totalFreed=27.94GB
```

この 1 行で:
- **リークなし**: totalAlloc ≈ totalFreed + current
- **一時確保**: peak ≫ current（IR リロード時の一時的確保）
- **リークあり**: totalFreed ≪ totalAlloc（解放が追いついていない）

### A-2. updateAtomicMaximum64

```cpp
inline void updateAtomicMaximum64(std::atomic<uint64_t>& target, uint64_t value) noexcept
{
    uint64_t expected = target.load(std::memory_order_relaxed);
    while (value > expected && !target.compare_exchange_weak(expected, value,
        std::memory_order_relaxed, std::memory_order_relaxed)) {}
}
```

### A-3. コンパイル時切替マクロ

```cpp
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
  #define DIAG_MKL_MALLOC(size, align) convo::diag::diagMklMalloc((size), (align))
  #define DIAG_MKL_FREE(ptr, size)     convo::diag::diagMklFree((ptr), (size))
#else
  #define DIAG_MKL_MALLOC(size, align) mkl_malloc((size), (align))
  #define DIAG_MKL_FREE(ptr, size)     mkl_free(ptr)
#endif
```

---

## 2. Patch B: MKLNonUniformConvolver — LayerAllocSizes + DIAG_MKL 置換

### B-1. LayerAllocSizes 構造体（確保時にサイズを保存）

**ファイル**: `src/MKLNonUniformConvolver.h`
**位置**: `struct Layer` の隣

```cpp
/// ★ v7: レイヤーの全 MKL バッファサイズ。
/// SetImpulse() で確保時に一度計算・保存し、
/// releaseAllLayers() で DIAG_MKL_FREE にそのまま渡す。
/// malloc/free が同じサイズを使用する保証。
struct LayerAllocSizes {
    size_t irFreqDomain = 0;   // partStride * sizeof(double)
    size_t irFreqReal   = 0;   // irSoaSize * sizeof(double)
    size_t irFreqImag   = 0;   // irSoaSize * sizeof(double)
    size_t fdlBuf       = 0;   // fdlBufSize * sizeof(double)
    size_t fdlReal      = 0;   // fdlSoaSize * sizeof(double)
    size_t fdlImag      = 0;   // fdlSoaSize * sizeof(double)
    size_t fftTimeBuf   = 0;   // fftSize * sizeof(double)
    size_t fftOutBuf    = 0;   // fftSize * sizeof(double)
    size_t prevInputBuf = 0;   // partSize * sizeof(double)
    size_t accumBuf     = 0;   // partStride * sizeof(double)
    size_t accumReal    = 0;   // complexSize * sizeof(double)
    size_t accumImag    = 0;   // complexSize * sizeof(double)
    size_t inputAccBuf  = 0;   // partSize * sizeof(double)
    size_t tailOutputBuf= 0;   // partSize * sizeof(double) (非 Immediate のみ)
};
```

**★ Layer 構造体に追加**:

```cpp
struct Layer {
    // ... 既存メンバ ...
    void freeAll() noexcept;
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    LayerAllocSizes allocSizes;  // ★ v7: 確保時に計算・保存
#endif
};
```

### B-2. MKLNonUniformConvolver.h — static カウンタ

```cpp
class MKLNonUniformConvolver {
public:
    static std::atomic<int> liveCount;
    // ... 既存のメンバ（変更なし） ...
```

### B-3. MKLNonUniformConvolver.cpp — SetImpulse で allocSizes を保存

```cpp
// SetImpulse 内、バッファ確保の直後:

const size_t irBufSize  = static_cast<size_t>(l.partStride);
const size_t fdlBufSize = static_cast<size_t>(l.partStride) * 2;
const size_t irSoaSize  = static_cast<size_t>(l.numParts) * static_cast<size_t>(l.complexSize);
const size_t fdlSoaSize = static_cast<size_t>(l.numParts) * 2 * static_cast<size_t>(l.complexSize);

l.irFreqDomain = static_cast<double*>(DIAG_MKL_MALLOC(irBufSize  * sizeof(double), 64));
l.irFreqReal   = static_cast<double*>(DIAG_MKL_MALLOC(irSoaSize  * sizeof(double), 64));
l.irFreqImag   = static_cast<double*>(DIAG_MKL_MALLOC(irSoaSize  * sizeof(double), 64));
l.fdlBuf       = static_cast<double*>(DIAG_MKL_MALLOC(fdlBufSize * sizeof(double), 64));
l.fdlReal      = static_cast<double*>(DIAG_MKL_MALLOC(fdlSoaSize * sizeof(double), 64));
l.fdlImag      = static_cast<double*>(DIAG_MKL_MALLOC(fdlSoaSize * sizeof(double), 64));
l.fftTimeBuf   = static_cast<double*>(DIAG_MKL_MALLOC(l.fftSize   * sizeof(double), 64));
l.fftOutBuf    = static_cast<double*>(DIAG_MKL_MALLOC(l.fftSize   * sizeof(double), 64));
l.prevInputBuf = static_cast<double*>(DIAG_MKL_MALLOC(l.partSize  * sizeof(double), 64));
l.accumBuf     = static_cast<double*>(DIAG_MKL_MALLOC(l.partStride * sizeof(double), 64));
l.accumReal    = static_cast<double*>(DIAG_MKL_MALLOC(l.complexSize * sizeof(double), 64));
l.accumImag    = static_cast<double*>(DIAG_MKL_MALLOC(l.complexSize * sizeof(double), 64));
l.inputAccBuf  = static_cast<double*>(DIAG_MKL_MALLOC(l.partSize  * sizeof(double), 64));

if (!l.isImmediate)
    l.tailOutputBuf = static_cast<double*>(DIAG_MKL_MALLOC(l.partSize * sizeof(double), 64));

// ★ v7: 確保サイズを allocSizes に保存（free 時にそのまま使用）
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
l.allocSizes.irFreqDomain = irBufSize  * sizeof(double);
l.allocSizes.irFreqReal   = irSoaSize  * sizeof(double);
l.allocSizes.irFreqImag   = irSoaSize  * sizeof(double);
l.allocSizes.fdlBuf       = fdlBufSize * sizeof(double);
l.allocSizes.fdlReal      = fdlSoaSize * sizeof(double);
l.allocSizes.fdlImag      = fdlSoaSize * sizeof(double);
l.allocSizes.fftTimeBuf   = l.fftSize   * sizeof(double);
l.allocSizes.fftOutBuf    = l.fftSize   * sizeof(double);
l.allocSizes.prevInputBuf = l.partSize  * sizeof(double);
l.allocSizes.accumBuf     = l.partStride * sizeof(double);
l.allocSizes.accumReal    = l.complexSize * sizeof(double);
l.allocSizes.accumImag    = l.complexSize * sizeof(double);
l.allocSizes.inputAccBuf  = l.partSize  * sizeof(double);
l.allocSizes.tailOutputBuf= l.isImmediate ? 0 : l.partSize * sizeof(double);
#endif
```

**★ 確保と保存が同一スコープ** → サイズ不整合のリスクゼロ。

### B-4. releaseAllLayers() で allocSizes を使用して DIAG_MKL_FREE

```cpp
void MKLNonUniformConvolver::releaseAllLayers() noexcept
{
    for (int i = 0; i < kNumLayers; ++i)
    {
        Layer& l = m_layers[i];
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
        // ★ v7: allocSizes からサイズを取得して DIAG_MKL_FREE
        // malloc 時に保存したサイズをそのまま使用（再計算不要）
        if (l.irFreqDomain)  DIAG_MKL_FREE(l.irFreqDomain,  l.allocSizes.irFreqDomain);
        if (l.irFreqReal)    DIAG_MKL_FREE(l.irFreqReal,    l.allocSizes.irFreqReal);
        if (l.irFreqImag)    DIAG_MKL_FREE(l.irFreqImag,    l.allocSizes.irFreqImag);
        if (l.fdlBuf)        DIAG_MKL_FREE(l.fdlBuf,        l.allocSizes.fdlBuf);
        if (l.fdlReal)       DIAG_MKL_FREE(l.fdlReal,       l.allocSizes.fdlReal);
        if (l.fdlImag)       DIAG_MKL_FREE(l.fdlImag,       l.allocSizes.fdlImag);
        if (l.fftTimeBuf)    DIAG_MKL_FREE(l.fftTimeBuf,    l.allocSizes.fftTimeBuf);
        if (l.fftOutBuf)     DIAG_MKL_FREE(l.fftOutBuf,     l.allocSizes.fftOutBuf);
        if (l.prevInputBuf)  DIAG_MKL_FREE(l.prevInputBuf,  l.allocSizes.prevInputBuf);
        if (l.accumBuf)      DIAG_MKL_FREE(l.accumBuf,      l.allocSizes.accumBuf);
        if (l.accumReal)     DIAG_MKL_FREE(l.accumReal,     l.allocSizes.accumReal);
        if (l.accumImag)     DIAG_MKL_FREE(l.accumImag,     l.allocSizes.accumImag);
        if (l.inputAccBuf)   DIAG_MKL_FREE(l.inputAccBuf,   l.allocSizes.inputAccBuf);
        if (l.tailOutputBuf) DIAG_MKL_FREE(l.tailOutputBuf, l.allocSizes.tailOutputBuf);
#else
        if (irFreqDomain)  { mkl_free(irFreqDomain);  irFreqDomain  = nullptr; }
        if (irFreqReal)    { mkl_free(irFreqReal);    irFreqReal    = nullptr; }
        if (irFreqImag)    { mkl_free(irFreqImag);    irFreqImag    = nullptr; }
        if (fdlBuf)        { mkl_free(fdlBuf);         fdlBuf        = nullptr; }
        if (fdlReal)       { mkl_free(fdlReal);       fdlReal       = nullptr; }
        if (fdlImag)       { mkl_free(fdlImag);       fdlImag       = nullptr; }
        if (fftTimeBuf)    { mkl_free(fftTimeBuf);     fftTimeBuf    = nullptr; }
        if (fftOutBuf)     { mkl_free(fftOutBuf);      fftOutBuf     = nullptr; }
        if (prevInputBuf)  { mkl_free(prevInputBuf);   prevInputBuf  = nullptr; }
        if (accumBuf)      { mkl_free(accumBuf);       accumBuf      = nullptr; }
        if (accumReal)     { mkl_free(accumReal);      accumReal     = nullptr; }
        if (accumImag)     { mkl_free(accumImag);      accumImag     = nullptr; }
        if (inputAccBuf)   { mkl_free(inputAccBuf);    inputAccBuf   = nullptr; }
        if (tailOutputBuf) { mkl_free(tailOutputBuf);  tailOutputBuf = nullptr; }
#endif
        // ... 既存のポインタ NULL 化 + 状態リセット ...
    }

    // ... 既存の m_ringBuf, m_direct* 解放（v6 と同一） ...
}
```

**★ malloc 時と free 時が同じ `LayerAllocSizes` を使用。**
将来バッファ追加・変更時は `LayerAllocSizes` と確保コードの 2 箇所のみ修正。

### B-5. デストラクタ

```cpp
MKLNonUniformConvolver::~MKLNonUniformConvolver()
{
    liveCount.fetch_sub(1, std::memory_order_relaxed);
    releaseAllLayers();
}
```

### B-6. NUC_MEM ログ

```cpp
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
{
    const uint64_t curBytes  = convo::diag::allocatedBytes();
    const uint64_t peakBytes = convo::diag::peakBytes();
    const uint64_t totalA    = convo::diag::totalAllocBytes();
    const uint64_t totalF    = convo::diag::totalFreedBytes();
    const auto osMem = getProcessMemoryInfo();
    const uint64_t retireBytes = /* pendingRetireBytes() */;
    const int64_t untracked = (int64_t)osMem.privateUsageMB * 1024 * 1024
                            - (int64_t)curBytes - (int64_t)retireBytes;

    diagLog(juce::String::formatted(
        "[NUC_MEM] NUC#%p | MKL: cur=%.0fMB peak=%.0fMB "
        "totalA=%.0fGB totalF=%.0fGB live=%d | "
        "OS: Private=%lluMB WorkingSet=%lluMB | "
        "Untracked=%.0fMB",
        (void*)this,
        curBytes / (1024.0*1024.0), peakBytes / (1024.0*1024.0),
        totalA / (1024.0*1024.0*1024.0), totalF / (1024.0*1024.0*1024.0),
        (int)liveCount.load(std::memory_order_relaxed),
        (unsigned long long)osMem.privateUsageMB,
        (unsigned long long)osMem.workingSetMB,
        std::max(0LL, untracked) / (1024.0 * 1024.0));
}
#endif
```

---

## 3. Patch C: ISRRetireRouter — objectBytes + 追跡率

### C-1. DeletionEntry に objectBytes フィールド追加（条件付き）

**ファイル**: `src/DeferredDeletionQueue.h`

```cpp
struct DeletionEntry {
    void* ptr = nullptr;
    void (*deleter)(void*) = nullptr;
    uint64_t epoch = 0;
    DeletionEntryType type = DeletionEntryType::Generic;
    uint64_t publicationSequenceId{0};
    uint64_t generation{0};
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    size_t objectBytes{0};  // ★ v7: Retire 対象のオブジェクトバイト数（sizeof）
#endif
};
```

### C-2. ISRRetireRouter — 追跡率カウンタ

```cpp
class ISRRetireRouter : public convo::IEpochProvider {
    // ... 既存メンバ ...
    std::atomic<uint64_t> m_pendingRetireBytes_ { 0 };
    std::atomic<uint32_t> m_trackedRetireEntries_ { 0 };  // ★ v7: objectBytes > 0 のエントリ数

    // ★ v7: objectBytes 付きオーバーロード
    RetireEnqueueResult enqueueRetire(void* ptr,
                                      void (*deleter)(void*),
                                      uint64_t epoch,
                                      DeletionEntryType type,
                                      size_t objectBytes) noexcept;

    [[nodiscard]] uint64_t pendingRetireBytes() const noexcept {
        return convo::consumeAtomic(m_pendingRetireBytes_, std::memory_order_acquire);
    }

    /// ★ v7: 追跡率 (0.0〜1.0)。objectBytes が設定されたエントリの割合。
    [[nodiscard]] double trackedRatio() const noexcept {
        const uint32_t tracked = convo::consumeAtomic(
            m_trackedRetireEntries_, std::memory_order_acquire);
        const uint32_t total = pendingRetireCount();
        return total > 0 ? static_cast<double>(tracked) / static_cast<double>(total) : 0.0;
    }
};
```

### C-3. enqueueRetire 実装

```cpp
RetireEnqueueResult ISRRetireRouter::enqueueRetire(void* ptr,
                                                    void (*deleter)(void*),
                                                    uint64_t epoch,
                                                    DeletionEntryType type,
                                                    size_t objectBytes) noexcept
{
    assert(provider_ != nullptr);
    if (ptr == nullptr || deleter == nullptr)
        return RetireEnqueueResult::Success;

    if (provider_->enqueueRetire(ptr, deleter, epoch, type, objectBytes))
    {
        m_pendingRetireBytes_.fetch_add(objectBytes, std::memory_order_relaxed);
        if (objectBytes > 0)
            m_trackedRetireEntries_.fetch_add(1, std::memory_order_relaxed);
        return RetireEnqueueResult::Success;
    }
    // フォールバック...
}

// 既存の 4 パラメータ版
RetireEnqueueResult ISRRetireRouter::enqueueRetire(void* ptr,
                                                    void (*deleter)(void*),
                                                    uint64_t epoch,
                                                    DeletionEntryType type) noexcept
{
    return enqueueRetire(ptr, deleter, epoch, type, 0);
}
```

### C-4. tryReclaim でのデクリメント

```cpp
void ISRRetireRouter::tryReclaim() noexcept
{
    // reclaim 成功時にエントリの objectBytes を取得してデクリメント
    // tracked エントリの場合のみ trackedRetireEntries_ をデクリメント
}
```

### C-5. Publish ログへの追跡率併記

```cpp
juce::Logger::writeToLog(juce::String::formatted(
    "[MEM_SNAP] PUBLISH gen=%d | "
    "NUC: live=%d alloc=%.0fMB peak=%.0fMB totalA=%.0fGB totalF=%.0fGB | "
    "Retire: pending=%u (%.1fMB) tracked=%u/%u (%.0f%%) | "
    "OS: Private=%lluMB | Untracked=%.0fMB",
    // ...
    pending, retireBytes / (1024.0*1024.0),
    tracked, pending, trackedRatio * 100.0,
    // ...
    untrackedMB));
```

**★ 追跡率の効果例**:

```text
Retire: pending=500 (0.0MB) tracked=12/500 (2%)
```

→ 「500個のエントリのうち、サイズが分かっているのは12個のみ」
→ `pendingRetireBytes` が 0 でも「追跡できていないエントリが多い」と分かる。

---

## 4. Patch D: DSPCore（v6 から変更なし）

- DSPCore liveCount（`fetch_add/fetch_sub` 使用）
- alignedL/R 容量
- dryBypass 容量
- OS PrivateUsage 併記

---

## 5. Patch E: Publish フルスナップショット

```cpp
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
{
    const uint64_t nucBytes    = convo::diag::allocatedBytes();
    const uint64_t nucPeak     = convo::diag::peakBytes();
    const uint64_t nucTotalA   = convo::diag::totalAllocBytes();
    const uint64_t nucTotalF   = convo::diag::totalFreedBytes();
    const int      nucLive     = (int)MKLNonUniformConvolver::liveCount.load(
                                     std::memory_order_relaxed);
    const int      stereoLive  = (int)ConvolverProcessor::StereoConvolver::liveCount.load(
                                     std::memory_order_relaxed);
    const int      dspLive     = (int)AudioEngine::DSPCore::liveCount.load(
                                     std::memory_order_relaxed);
    const uint32_t pending     = m_retireRouter ? m_retireRouter->pendingRetireCount() : 0;
    const uint64_t retireBytes = m_retireRouter ? m_retireRouter->pendingRetireBytes() : 0;
    const uint32_t tracked     = m_retireRouter ? /* trackedRetireEntries */ 0 : 0;
    const double   trackedRatio= m_retireRouter ? m_retireRouter->trackedRatio() : 0.0;
    const uint64_t reclaim     = m_retireRouter ? m_retireRouter->reclaimAttemptCount() : 0;
    const uint64_t overflow    = m_retireRouter ? m_retireRouter->overflowCount() : 0;
    const auto osMem = getProcessMemoryInfo();
    const int64_t untracked    = (int64_t)osMem.privateUsageMB * 1024 * 1024
                               - (int64_t)nucBytes - (int64_t)retireBytes;
    const double untrackedMB   = std::max(0LL, untracked) / (1024.0 * 1024.0);

    juce::Logger::writeToLog(juce::String::formatted(
        "[MEM_SNAP] PUBLISH gen=%d seq=%d | "
        "NUC: live=%d alloc=%.0fMB peak=%.0fMB totalA=%.0fGB totalF=%.0fGB | "
        "Stereo=%d DSPCore=%d | "
        "Retire: pending=%u (%.1fMB) tracked=%u/%u (%.0f%%) overflow=%llu reclaim=%llu | "
        "OS: Private=%lluMB WorkingSet=%lluMB | "
        "Untracked=%.0fMB",
        gen, seq,
        nucLive,
        nucBytes / (1024.0*1024.0), nucPeak / (1024.0*1024.0),
        nucTotalA / (1024.0*1024.0*1024.0), nucTotalF / (1024.0*1024.0*1024.0),
        stereoLive, dspLive,
        pending, retireBytes / (1024.0*1024.0),
        tracked, pending, trackedRatio * 100.0,
        (unsigned long long)overflow, (unsigned long long)reclaim,
        (unsigned long long)osMem.privateUsageMB,
        (unsigned long long)osMem.workingSetMB,
        untrackedMB));
}
#endif
```

---

## 6. Patch F: イベント駆動ログ（v6 から変更なし）

- Publish 時（Patch E）
- IR Reload 時（B-6）
- Retire イベント検出時
- 5秒定期ログ（変化時のみ）

---

## 7. 出力例（v7）

### 正常時

```text
[NUC_MEM] NUC#0000001234 | MKL: cur=62MB peak=142MB totalA=3.2GB totalF=3.14GB live=2 | OS: Private=68MB WorkingSet=120MB | Untracked=6MB
[MEM_SNAP] PUBLISH gen=8 seq=5 | NUC: live=2 alloc=62MB peak=142MB totalA=3.2GB totalF=3.14GB | Stereo=1 DSPCore=1 | Retire: pending=0 (0.0MB) tracked=0/0 (0%) overflow=0 reclaim=47 | OS: Private=68MB WorkingSet=120MB | Untracked=6MB
```

### リーク検出例

```text
[NUC_MEM] NUC#0000001234 | MKL: cur=630MB peak=640MB totalA=28.0GB totalF=27.4GB live=8 | OS: Private=2330MB WorkingSet=2400MB | Untracked=1520MB
[MEM_SNAP] PUBLISH gen=8 | NUC: live=8 alloc=630MB peak=640MB totalA=28.0GB totalF=27.4GB | Stereo=4 DSPCore=4 | Retire: pending=232 (180.5MB) tracked=8/232 (3%) overflow=0 reclaim=12 | OS: Private=2330MB WorkingSet=2400MB | Untracked=1520MB
```

**解釈**:
- NUC=630MB, totalA=28GB, totalF=27.4GB → 累積差 600MB ≈ current（リークの可能性）
- Retire: 232個中8個のみ追跡 → `pendingRetireBytes` は過小評価
- Untracked=1520MB → 未計測領域

---

## 8. 実装コスト見積もり（v7）

| Patch | 変更ファイル数 | 追加行数 | 備考 |
|:------|:-------------|:---------|:-----|
| A: DiagnosticsConfig.h | 1 | ~55行 | totalAllocBytes/totalFreedBytes + ヘルパー |
| B: MKLNonUniformConvolver | 2 | ~55行 | LayerAllocSizes + allocSizes 保存 + releaseAllLayers |
| C: ISRRetireRouter + DeletionEntry | 5 | ~45行 | objectBytes + trackedRetireEntries + trackedRatio |
| D: DSPCore | 2 | ~18行 | v6 から変更なし |
| E: Publish スナップショット | 1 | ~25行 | 累積統計 + 追跡率追加 |
| F: イベント駆動ログ | 2 | ~18行 | v6 から変更なし |
| **合計** | **8〜10ファイル** | **~216行** | |

---

## 9. v6 からの改善点一覧

| 項目 | v6（問題） | v7（修正） |
|:-----|:----------|:----------|
| バッファサイズ計算 | `releaseAllLayers()` で再計算 | **`LayerAllocSizes` で確保時に保存、解放時にそのまま使用** |
| RetireEntry フィールド名 | `retainedBytes` | **`objectBytes`（sizeof 相当、意味的に正確）** |
| Retire 追跡率 | なし | **`trackedEntries/totalEntries` + `trackedRatio()`** |
| 累積統計 | `allocatedBytes` / `peakBytes` のみ | **`totalAllocBytes` + `totalFreedBytes` を追加** |
| peak 更新 | `load` + `CAS` | **`fetch_add` 戻り値から `load` を省略** |
| MemoryCategory | 未使用コードとして存在 | **本次調査では削除** |
| freeAll() の DIAG_MKL_FREE | `releaseAllLayers()` 再計算に依存 | **`LayerAllocSizes` から直接取得（再計算不要）** |
