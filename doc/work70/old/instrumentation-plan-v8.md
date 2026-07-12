# メモリ占有調査のためのインストルメンテーション改修案 v8 — freeAll 集約 + リセットカウンタ + ログ注記

---

**目的**: 2.33GB メモリ占有の原因特定。推定ではなく実測で確定させる。
**v7 からの変更**: reviewer フィードバック（5点）を全て反映。完成度 97〜99 点を目指す。

---

## 0. v7 の問題点と v8 での修正方針

| # | v7 の問題 | v8 の修正 | 根拠 |
|:--|:---------|:---------|:-----|
| 1 | peak 更新の説明とコードが不一致（load が残存） | **`fetch_add` 戻り値から `updateAtomicMaximum64` へ直接渡す** | load 削減、説明とコードの一致 |
| 2 | `allocSizes` が確保途中で未初期化のまま `DIAG_MKL_FREE` されるリスク | **SetImpulse 冒頭で `allocSizes = {}` をゼロ初期化** | 部分確保失敗時の安全性 |
| 3 | `releaseAllLayers()` が Layer の内部構造を知っている（結合度が高い） | **`Layer::freeAll()` 内に `DIAG_MKL_FREE` を集約** | 責務分離（Layer が「自分が何を確保したか」を知る） |
| 4 | `objectBytes` / `Untracked` のログが誤解を招く可能性 | **ログに注記を追加**（`sizeof only` / `incl. JUCE/IPP/etc.`） | 診断ログの明確性 |
| 5 | 長時間試験向けに診断カウンタをリセットできない | **`resetDiagnostics()` 追加** | 試験開始時のリセット需要 |

---

## 1. Patch A: DiagnosticsConfig.h — fetch_add 修正 + リセット関数

### A-1. MklAllocStats

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

/// mkl_malloc ラッパー。★ v8: fetch_add 戻り値で load を省略。
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

/// ★ v8: 診断カウンタをリセット（長時間試験の区切り用）。
///        allocatedBytes は現在の使用量を維持し、peak/累積のみリセット。
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

**★ `diagMklMalloc` の peak 更新ロジック**:

```cpp
// v7 (不一致):
const uint64_t newAlloc = allocatedBytes.load(...);  // ← load が残存
updateAtomicMaximum64(peakBytes, newAlloc);

// v8 (修正):
const uint64_t prev = allocatedBytes.fetch_add(bytes);  // ← load 不要
updateAtomicMaximum64(peakBytes, prev + bytes);
```

### A-2. resetDiagnostics の効果例

```text
// テスト開始時
convo::diag::resetDiagnostics();

// 30分間のIR Reload試験
// ...

// 結果
current=80MB  peak=600MB  totalA=18GB  totalF=17.92GB
// → peak=600MB は試験中のピーク、totalA/F は試験中の累積
```

### A-3. コンパイル時切替マクロ（変更なし）

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

## 2. Patch B: MKLNonUniformConvolver — Layer::freeAll() に DIAG_MKL_FREE 集約

### B-1. LayerAllocSizes（v7 から変更なし）

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

### B-2. SetImpulse — allocSizes ゼロ初期化 + 保存

```cpp
bool MKLNonUniformConvolver::SetImpulse(...)
{
    // ... 既存 ...

    for (int li = 0; li < kNumLayers; ++li)
    {
        // ... 既存のレイヤー初期化 ...

        Layer& l = m_layers[m_numActiveLayers];

        // ★ v8: allocSizes をゼロ初期化（部分確保失敗時の安全保証）
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
        l.allocSizes = {};
#endif

        // ... 既存のバッファ確保 ...

        const size_t irBufSize  = static_cast<size_t>(l.partStride);
        const size_t fdlBufSize = static_cast<size_t>(l.partStride) * 2;
        const size_t irSoaSize  = static_cast<size_t>(l.numParts)
                                * static_cast<size_t>(l.complexSize);
        const size_t fdlSoaSize = static_cast<size_t>(l.numParts) * 2
                                * static_cast<size_t>(l.complexSize);

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

        // ★ v8: allocSizes に保存（確保完了後）
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

        // ... 既存の確保失敗チェック ...
    }
    // ... 既存 ...
}
```

### B-3. Layer::freeAll() — DIAG_MKL_FREE 集約（v8 の核心変更）

```cpp
void MKLNonUniformConvolver::Layer::freeAll() noexcept
{
    // ... 既存の fftPlanOwner, fftWorkBuf 解放 ...

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    // ★ v8: allocSizes からサイズを取得して DIAG_MKL_FREE。
    //   Layer 自身が「自分が何を確保したか」を知り、解放する。
    //   releaseAllLayers() から Layer の内部構造を知る必要がない。
    if (irFreqDomain)  DIAG_MKL_FREE(irFreqDomain,  allocSizes.irFreqDomain);
    if (irFreqReal)    DIAG_MKL_FREE(irFreqReal,    allocSizes.irFreqReal);
    if (irFreqImag)    DIAG_MKL_FREE(irFreqImag,    allocSizes.irFreqImag);
    if (fdlBuf)        DIAG_MKL_FREE(fdlBuf,        allocSizes.fdlBuf);
    if (fdlReal)       DIAG_MKL_FREE(fdlReal,       allocSizes.fdlReal);
    if (fdlImag)       DIAG_MKL_FREE(fdlImag,       allocSizes.fdlImag);
    if (fftTimeBuf)    DIAG_MKL_FREE(fftTimeBuf,    allocSizes.fftTimeBuf);
    if (fftOutBuf)     DIAG_MKL_FREE(fftOutBuf,     allocSizes.fftOutBuf);
    if (prevInputBuf)  DIAG_MKL_FREE(prevInputBuf,  allocSizes.prevInputBuf);
    if (accumBuf)      DIAG_MKL_FREE(accumBuf,      allocSizes.accumBuf);
    if (accumReal)     DIAG_MKL_FREE(accumReal,     allocSizes.accumReal);
    if (accumImag)     DIAG_MKL_FREE(accumImag,     allocSizes.accumImag);
    if (inputAccBuf)   DIAG_MKL_FREE(inputAccBuf,   allocSizes.inputAccBuf);
    if (tailOutputBuf) DIAG_MKL_FREE(tailOutputBuf, allocSizes.tailOutputBuf);
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
    fftSize = partSize = numParts = numPartsIR = 0;
    fdlMask = complexSize = partStride = 0;
    // ...

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    allocSizes = {};  // ★ v8: 解放後にゼロ初期化
#endif
}
```

**★ 責務分離**: `releaseAllLayers()` は `m_layers[i].freeAll()` を呼ぶだけで、
Layer の内部構造を一切知らなくなる。`m_ringBuf` と `m_direct*` のみ NUC 側で解放。

### B-4. releaseAllLayers() — 簡素化

```cpp
void MKLNonUniformConvolver::releaseAllLayers() noexcept
{
    // ... 既存の guard チェック ...

    // ★ v8: Layer の freeAll() に委譲（内部構造を知る必要なし）
    for (int i = 0; i < kNumLayers; ++i)
        m_layers[i].freeAll();
    m_numActiveLayers = 0;
    m_latency         = 0;

    // NUC レベルのバッファのみ解放
    if (m_ringBuf) { mkl_free(m_ringBuf); m_ringBuf = nullptr; }
    m_ringSize = m_ringMask = m_ringWrite = m_ringRead = m_ringAvail = 0;

    if (m_directIRRev)   { mkl_free(m_directIRRev);   m_directIRRev = nullptr; }
    if (m_directHistory) { mkl_free(m_directHistory); m_directHistory = nullptr; }
    if (m_directWindow)  { mkl_free(m_directWindow);  m_directWindow = nullptr; }
    if (m_directOutBuf)  { mkl_free(m_directOutBuf);  m_directOutBuf = nullptr; }

    // ... 既存の状態リセット ...
}
```

**★ `m_ringBuf` / `m_direct*` は NUC レベル**（Layer ではない）ため、
`releaseAllLayers()` で `mkl_free` のまま。DIAG_MKL_FREE への置換は任意。

---

## 3. Patch C: ISRRetireRouter — objectBytes + 追跡率（v7 から変更なし）

v7 の内容をそのまま採用。

---

## 4. Patch D: DSPCore（v7 から変更なし）

---

## 5. Patch E: Publish ログ — 注記追加

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

    // ★ v8: log に注記を追加
    juce::Logger::writeToLog(juce::String::formatted(
        "[MEM_SNAP] PUBLISH gen=%d seq=%d | "
        "NUC(MKL only): live=%d alloc=%.0fMB peak=%.0fMB "
        "totalA=%.0fGB totalF=%.0fGB | "
        "Stereo=%d DSPCore=%d | "
        "Retire: pending=%u objBytes=%.1fMB(sizesof) tracked=%u/%u (%.0f%%) "
        "overflow=%llu reclaim=%llu | "
        "OS: Private=%lluMB WorkingSet=%lluMB | "
        "Untracked=%.0fMB(incl JUCE/IPP/threads/etc.)",
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

**★ ログ内の注記**:
- `MKL only` — NUC が MKL バッファのみ計測していることを明記
- `objBytes=%.1fMB(sizesof)` — `objectBytes` が `sizeof` ベースであり、内部ヒープを含まないことを明記
- `Untracked=%.0fMB(incl JUCE/IPP/threads/etc.)` — Untracked に含まれるカテゴリを列挙

---

## 6. 出力例（v8）

### 正常時

```text
[NUC_MEM] NUC#0000001234 | MKL: cur=62MB peak=142MB totalA=3.2GB totalF=3.14GB live=2 | OS: Private=68MB WorkingSet=120MB | Untracked=6MB(incl JUCE/IPP/etc.)
[MEM_SNAP] PUBLISH gen=8 seq=5 | NUC(MKL only): live=2 alloc=62MB peak=142MB totalA=3.2GB totalF=3.14GB | Stereo=1 DSPCore=1 | Retire: pending=0 objBytes=0.0MB(sizesof) tracked=0/0 (0%) overflow=0 reclaim=47 | OS: Private=68MB WorkingSet=120MB | Untracked=6MB(incl JUCE/IPP/threads/etc.)
```

### リーク検出例

```text
[MEM_SNAP] PUBLISH gen=8 | NUC(MKL only): live=8 alloc=630MB peak=640MB totalA=28.0GB totalF=27.4GB | Stereo=4 DSPCore=4 | Retire: pending=232 objBytes=12.8MB(sizesof) tracked=8/232 (3%) overflow=0 reclaim=12 | OS: Private=2330MB WorkingSet=2400MB | Untracked=1520MB(incl JUCE/IPP/threads/etc.)
```

---

## 7. 実装コスト見積もり（v8）

| Patch | 変更ファイル数 | 追加行数 | 備考 |
|:------|:-------------|:---------|:-----|
| A: DiagnosticsConfig.h | 1 | ~60行 | fetch_add修正 + resetDiagnostics |
| B: MKLNonUniformConvolver | 2 | ~60行 | freeAll集約 + allocSizes初期化 + ログ注記 |
| C: ISRRetireRouter | 5 | ~40行 | v7 から変更なし |
| D: DSPCore | 2 | ~18行 | v7 から変更なし |
| E: Publish ログ | 1 | ~25行 | 注記追加 |
| **合計** | **8〜10ファイル** | **~203行** | |

---

## 8. v7 からの改善点一覧

| 項目 | v7（問題） | v8（修正） |
|:-----|:----------|:----------|
| peak 更新 | `load` が残存 | **`fetch_add` 戻り値から直接 `updateAtomicMaximum64`** |
| `allocSizes` 未初期化 | 部分確保失敗時に未初期化のまま | **SetImpulse 冒頭で `allocSizes = {}` ゼロ初期化** |
| freeAll の責務 | `releaseAllLayers()` が Layer 内部構造を知る | **`Layer::freeAll()` 内に DIAG_MKL_FREE 集約** |
| ログの明確性 | `objectBytes` / `Untracked` が誤解を招く | **`(sizeof only)` / `(incl JUCE/IPP/etc.)` 注記追加** |
| リセットカウンタ | なし | **`resetDiagnostics()` 追加（長時間試験向け）** |
