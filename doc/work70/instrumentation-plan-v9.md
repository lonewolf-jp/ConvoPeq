# メモリ占有調査のためのインストルメンテーション改修案 v9 — allocSizes 個別保存 + freeTracked ヘルパ

---

**目的**: 2.33GB メモリ占有の原因特定。推定ではなく実測で確定させる。
**v8 からの変更**: reviewer フィードバック（5点）を全て反映。完成度 99 点を目指す。

---

## 0. v8 の問題点と v9 での修正方針

| # | v8 の問題 | v9 の修正 | 重要度 |
|:--|:---------|:---------|:------|
| 1 | `allocSizes` を全確保完了後にまとめて保存 → 部分失敗時に未保存 | **各 `DIAG_MKL_MALLOC` 成功直後にサイズを保存** | ★★★ |
| 2 | `freeAll()` で `DIAG_MKL_FREE` と `nullptr` 代入が分離 | **`freeTracked()` ヘルパで `free + nullptr` を一括化** | ★★ |
| 3 | `resetDiagnostics()` の意味論が曖昧 | **コメントに `best-effort reset` を追記** | ★ |
| 4 | `Untracked` が「未追跡」の意味だが誤解の可能性 | ** `Untracked(other)` に変更、annotated** | ★ |
| 5 | `trackedRatio()` で `tracked > pending` が起こりうる | **`min(tracked, pending)` でクランプ** | ★ |

---

## 1. Patch A: DiagnosticsConfig.h — resetDiagnostics コメント + freeTracked ヘルパ

### A-1. MklAllocStats（v8 から変更なし）

```cpp
namespace convo::diag {

struct MklAllocStats {
    std::atomic<uint64_t> allocatedBytes { 0 };
    std::atomic<uint64_t> peakBytes      { 0 };
    std::atomic<uint64_t> totalAllocBytes{ 0 };
    std::atomic<uint64_t> totalFreedBytes{ 0 };
};

inline MklAllocStats& mklStats() noexcept
{
    static MklAllocStats stats{};
    return stats;
}

inline void* diagMklMalloc(size_t size, int alignment) noexcept
{
    void* ptr = mkl_malloc(size, alignment);
    if (ptr)
    {
        const uint64_t bytes = static_cast<uint64_t>(size);
        const uint64_t prev = mklStats().allocatedBytes.fetch_add(
            bytes, std::memory_order_relaxed);
        mklStats().totalAllocBytes.fetch_add(bytes, std::memory_order_relaxed);
        updateAtomicMaximum64(mklStats().peakBytes, prev + bytes);
    }
    return ptr;
}

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

[[nodiscard]] inline uint64_t allocatedBytes() noexcept
{
    return mklStats().allocatedBytes.load(std::memory_order_relaxed);
}

[[nodiscard]] inline uint64_t peakBytes() noexcept
{
    return mklStats().peakBytes.load(std::memory_order_relaxed);
}

[[nodiscard]] inline uint64_t totalAllocBytes() noexcept
{
    return mklStats().totalAllocBytes.load(std::memory_order_relaxed);
}

[[nodiscard]] inline uint64_t totalFreedBytes() noexcept
{
    return mklStats().totalFreedBytes.load(std::memory_order_relaxed);
}

/// ★ v9: best-effort reset。他のスレッドの同時アロケーションで
/// peak が一瞬だけ不正確になることがあるが、診断用途では問題なし。
inline void resetDiagnostics() noexcept
{
    mklStats().peakBytes.store(
        mklStats().allocatedBytes.load(std::memory_order_relaxed),
        std::memory_order_relaxed);
    mklStats().totalAllocBytes.store(0, std::memory_order_relaxed);
    mklStats().totalFreedBytes.store(0, std::memory_order_relaxed);
}

} // namespace convo::diag
```

### A-2. freeTracked ヘルパ（★ v9: free + nullptr 一括化）

```cpp
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
/// ★ v9: DIAG_MKL_FREE + nullptr 代入を一括化。
///   二重解放防止 + dangling pointer 解消。
///   テンプレートなのでポインタ型を自動推論。
template<typename T>
inline void freeTracked(T*& p, size_t size) noexcept
{
    if (p)
    {
        DIAG_MKL_FREE(p, size);
        p = nullptr;
    }
}
#endif
```

**★ freeTracked の効果**:

```cpp
// v8:
if (irFreqDomain)  DIAG_MKL_FREE(irFreqDomain, allocSizes.irFreqDomain);
// ... 後で nullptr 代入 ...

// v9:
freeTracked(irFreqDomain, allocSizes.irFreqDomain);
// free + nullptr が一発で完了
```

---

## 2. Patch B: MKLNonUniformConvolver — allocSizes 個別保存 + freeTracked

### B-1. LayerAllocSizes + Layer 構造体（v8 から変更なし）

```cpp
struct LayerAllocSizes {
    size_t irFreqDomain = 0;
    size_t irFreqReal   = 0;
    size_t irFreqImag   = 0;
    size_t fdlBuf       = 0;
    size_t fdlReal      = 0;
    size_t fdlImag      = 0;
    size_t fftTimeBuf   = 0;
    size_t fftOutBuf    = 0;
    size_t prevInputBuf = 0;
    size_t accumBuf     = 0;
    size_t accumReal    = 0;
    size_t accumImag    = 0;
    size_t inputAccBuf  = 0;
    size_t tailOutputBuf= 0;
};

struct Layer {
    // ... 既存メンバ ...
    void freeAll() noexcept;
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    LayerAllocSizes allocSizes;
#endif
};
```

### B-2. SetImpulse — allocSizes を各確保直後に保存（★ v9 の核心修正）

```cpp
bool MKLNonUniformConvolver::SetImpulse(...)
{
    // ... 既存 ...

    for (int li = 0; li < kNumLayers; ++li)
    {
        // ... 既存のレイヤー初期化 ...

        Layer& l = m_layers[m_numActiveLayers];

        // ★ v9: allocSizes をゼロ初期化（部分確保失敗時の安全保証）
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
        l.allocSizes = {};
#endif

        // ... IPP FFT 初期化 ...

        const size_t irBufSize  = static_cast<size_t>(l.partStride);
        const size_t fdlBufSize = static_cast<size_t>(l.partStride) * 2;
        const size_t irSoaSize  = static_cast<size_t>(l.numParts)
                                * static_cast<size_t>(l.complexSize);
        const size_t fdlSoaSize = static_cast<size_t>(l.numParts) * 2
                                * static_cast<size_t>(l.complexSize);

        // ★ v9: 各 DIAG_MKL_MALLOC 成功直後に allocSizes に保存。
        //   途中の確保が失敗しても、確保済みのバッファは正しいサイズで解放される。
        l.irFreqDomain = static_cast<double*>(DIAG_MKL_MALLOC(irBufSize * sizeof(double), 64));
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
        l.allocSizes.irFreqDomain = irBufSize * sizeof(double);
#endif

        l.irFreqReal = static_cast<double*>(DIAG_MKL_MALLOC(irSoaSize * sizeof(double), 64));
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
        l.allocSizes.irFreqReal = irSoaSize * sizeof(double);
#endif

        l.irFreqImag = static_cast<double*>(DIAG_MKL_MALLOC(irSoaSize * sizeof(double), 64));
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
        l.allocSizes.irFreqImag = irSoaSize * sizeof(double);
#endif

        l.fdlBuf = static_cast<double*>(DIAG_MKL_MALLOC(fdlBufSize * sizeof(double), 64));
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
        l.allocSizes.fdlBuf = fdlBufSize * sizeof(double);
#endif

        l.fdlReal = static_cast<double*>(DIAG_MKL_MALLOC(fdlSoaSize * sizeof(double), 64));
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
        l.allocSizes.fdlReal = fdlSoaSize * sizeof(double);
#endif

        l.fdlImag = static_cast<double*>(DIAG_MKL_MALLOC(fdlSoaSize * sizeof(double), 64));
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
        l.allocSizes.fdlImag = fdlSoaSize * sizeof(double);
#endif

        l.fftTimeBuf = static_cast<double*>(DIAG_MKL_MALLOC(l.fftSize * sizeof(double), 64));
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
        l.allocSizes.fftTimeBuf = l.fftSize * sizeof(double);
#endif

        l.fftOutBuf = static_cast<double*>(DIAG_MKL_MALLOC(l.fftSize * sizeof(double), 64));
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
        l.allocSizes.fftOutBuf = l.fftSize * sizeof(double);
#endif

        l.prevInputBuf = static_cast<double*>(DIAG_MKL_MALLOC(l.partSize * sizeof(double), 64));
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
        l.allocSizes.prevInputBuf = l.partSize * sizeof(double);
#endif

        l.accumBuf = static_cast<double*>(DIAG_MKL_MALLOC(l.partStride * sizeof(double), 64));
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
        l.allocSizes.accumBuf = l.partStride * sizeof(double);
#endif

        l.accumReal = static_cast<double*>(DIAG_MKL_MALLOC(l.complexSize * sizeof(double), 64));
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
        l.allocSizes.accumReal = l.complexSize * sizeof(double);
#endif

        l.accumImag = static_cast<double*>(DIAG_MKL_MALLOC(l.complexSize * sizeof(double), 64));
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
        l.allocSizes.accumImag = l.complexSize * sizeof(double);
#endif

        l.inputAccBuf = static_cast<double*>(DIAG_MKL_MALLOC(l.partSize * sizeof(double), 64));
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
        l.allocSizes.inputAccBuf = l.partSize * sizeof(double);
#endif

        if (!l.isImmediate)
        {
            l.tailOutputBuf = static_cast<double*>(DIAG_MKL_MALLOC(l.partSize * sizeof(double), 64));
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
            l.allocSizes.tailOutputBuf = l.partSize * sizeof(double);
#endif
        }

        // ... 既存の確保失敗チェック（releaseAllLayers 呼び出し）...
    }
    // ... 既存 ...
}
```

**★ 確保と保存が 1:1 対応** — 途中の確保が失敗しても、確保済みのバッファは
正しいサイズで `freeAll()` から解放される。

### B-3. Layer::freeAll() — freeTracked 使用

```cpp
void MKLNonUniformConvolver::Layer::freeAll() noexcept
{
    // ... 既存の fftPlanOwner, fftWorkBuf 解放 ...

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    // ★ v9: freeTracked() で free + nullptr を一括化
    freeTracked(irFreqDomain,  allocSizes.irFreqDomain);
    freeTracked(irFreqReal,    allocSizes.irFreqReal);
    freeTracked(irFreqImag,    allocSizes.irFreqImag);
    freeTracked(fdlBuf,        allocSizes.fdlBuf);
    freeTracked(fdlReal,       allocSizes.fdlReal);
    freeTracked(fdlImag,       allocSizes.fdlImag);
    freeTracked(fftTimeBuf,    allocSizes.fftTimeBuf);
    freeTracked(fftOutBuf,     allocSizes.fftOutBuf);
    freeTracked(prevInputBuf,  allocSizes.prevInputBuf);
    freeTracked(accumBuf,      allocSizes.accumBuf);
    freeTracked(accumReal,     allocSizes.accumReal);
    freeTracked(accumImag,     allocSizes.accumImag);
    freeTracked(inputAccBuf,   allocSizes.inputAccBuf);
    freeTracked(tailOutputBuf, allocSizes.tailOutputBuf);
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

    // ... 既存の状態リセット ...

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    allocSizes = {};
#endif
}
```

---

## 3. Patch C: ISRRetireRouter — trackedRatio クランプ

```cpp
[[nodiscard]] double trackedRatio() const noexcept
{
    const uint32_t tracked = convo::consumeAtomic(
        m_trackedRetireEntries_, std::memory_order_acquire);
    const uint32_t total = pendingRetireCount();
    if (total == 0) return 0.0;
    // ★ v9: tracked > pending にならないようクランプ
    const uint32_t clamped = std::min(tracked, total);
    return static_cast<double>(clamped) / static_cast<double>(total);
}
```

---

## 4. Patch D: Publish ログ — Untracked 注記更新

```cpp
    juce::Logger::writeToLog(juce::String::formatted(
        "[MEM_SNAP] PUBLISH gen=%d seq=%d | "
        "NUC(MKL only): live=%d alloc=%.0fMB peak=%.0fMB "
        "totalA=%.0fGB totalF=%.0fGB | "
        "Stereo=%d DSPCore=%d | "
        "Retire: pending=%u objBytes=%.1fMB(sizeof) tracked=%u/%u (%.0f%%) "
        "overflow=%llu reclaim=%llu | "
        "OS: Private=%lluMB WorkingSet=%lluMB | "
        "Untracked(other)=%.0fMB(JUCE/IPP/CRT/threads/...)",
        // ...
        untrackedMB));
```

**★ `Untracked(other)` に変更** — 「JUCE/IPP/CRT/threads のうちのどれか」ではなく
「これら全てを含む未計測領域」であることを明確化。

---

## 5. 実装コスト見積もり（v9）

| Patch | 変更ファイル数 | 追加行数 | 備考 |
|:------|:-------------|:---------|:-----|
| A: DiagnosticsConfig.h | 1 | ~65行 | freeTracked ヘルパ + reset コメント |
| B: MKLNonUniformConvolver | 2 | ~75行 | allocSizes 個別保存 + freeTracked 使用 |
| C: ISRRetireRouter | 1 | ~3行 | trackedRatio クランプ |
| D: Publish ログ | 1 | ~2行 | Untracked 注記更新 |
| **合計** | **4〜5ファイル** | **~145行** | v8 から大幅に簡素化 |

---

## 6. v8 からの改善点一覧

| 項目 | v8（問題） | v9（修正） |
|:-----|:----------|:----------|
| allocSizes 保存タイミング | 全確保完了後にまとめて保存 | **各 DIAG_MKL_MALLOC 成功直後に個別保存** |
| freeAll() の free + nullptr | 分離（dangling pointer リスク） | **`freeTracked()` で free + nullptr を一括化** |
| resetDiagnostics() | 意味論が曖昧 | **コメントに `best-effort reset` 追記** |
| Untracked 名称 | `Untracked` のみ | **`Untracked(other)` に変更** |
| trackedRatio() | `tracked > pending` が起こりうる | **`min(tracked, total)` でクランプ** |
