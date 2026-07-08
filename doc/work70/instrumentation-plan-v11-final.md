# メモリ占有調査のためのインストルメンテーション改修案 v11（最終版）— nullptr 漏れ修正 + 一時バッファ除外明確化

---

**目的**: 2.33GB メモリ占有の原因特定。推定ではなく実測で確定させる。
**v10 からの変更**: 最終2点の修正。完成度 99+ 点。

---

## 0. v10 の問題点と v11 での修正方針

| # | v10 の問題 | v11 の修正 | 重要度 |
|:--|:----------|:----------|:------|
| 1 | `releaseAllLayers()` の `DIAG_MKL_FREE` 後に `nullptr` 代入がなく、dangling pointer リスク | **`freeTracked()` を流用して `free + nullptr` を一括化** | ★★★ |
| 2 | 一時バッファ（`impulseForFft` / `tempTime` / `tempFreq` / `swapSoA`）の診断対象が曖昧 | **明示的に DIAG 対象外とし、`mkl_malloc/mkl_free` のまま維持** | ★★★ |
| 3 | `diagMklFree()` にデバッグアサーションなし（`freeTracked` 非経由の呼び出しでのサイズ漏れ検出不可） | **`diagMklFree()` に `jassert(size != 0)` 追加** | ★ |

---

## 1. Patch A: DiagnosticsConfig.h — diagMklFree にアサーション追加

### A-1. MklAllocStats + diagMklMalloc/diagMklFree

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
        jassert(size != 0);  // ★ v11: サイズ指定漏れ検出（freeTracked非経由も含む）
        mkl_free(ptr);
        mklStats().allocatedBytes.fetch_sub(
            static_cast<uint64_t>(size), std::memory_order_relaxed);
        mklStats().totalFreedBytes.fetch_add(
            static_cast<uint64_t>(size), std::memory_order_relaxed);
    }
}

// ... accessors, resetDiagnostics, updateAtomicMaximum64 ...

} // namespace convo::diag
```

### A-2. freeTracked ヘルパ

```cpp
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS

/// DIAG_MKL_FREE + nullptr 代入を一括化。
/// デバッグビルドでは size == 0 をアサート（保存漏れ検出）。
template<typename T>
inline void freeTracked(T*& p, size_t size) noexcept
{
    if (p)
    {
        jassert(size != 0);  // allocSizes 保存漏れ or バッファ追加時の実装漏れ検出
        DIAG_MKL_FREE(p, size);
        p = nullptr;
    }
}

#endif
```

---

## 2. Patch B: MKLNonUniformConvolver — 全変更箇所確定版

### B-1. 設計方針（★ v11 で確定）

| バッファ種別 | 診断対象 | 理由 |
|:-----------|:--------|:-----|
| Layer 永続バッファ（14種） | **✅ DIAG_MKL_MALLOC** + `allocSizes` 保存 + `freeTracked()` | 全生存期間にわたってメモリを占有 |
| NUC 永続バッファ（m_ringBuf, m_direct*） | **✅ DIAG_MKL_MALLOC** + `freeTracked()` | 同上 |
| **一時バッファ（impulseForFft, tempTime, tempFreq, swapSoA, gainReal）** | **❌ 対象外（mkl_malloc/mkl_free のまま）** | SetImpulse 終了時に解放、生存時間が短く current への影響なし |

### B-2. 全変更箇所

#### MKLNonUniformConvolver.h

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

class MKLNonUniformConvolver {
public:
    static std::atomic<int> liveCount;
    // ...
};
```

#### MKLNonUniformConvolver.cpp — コンストラクタ/デストラクタ

```cpp
MKLNonUniformConvolver::MKLNonUniformConvolver()
{
    mkl_set_num_threads(1);
    liveCount.fetch_add(1, std::memory_order_relaxed);  // ★ 診断
}

MKLNonUniformConvolver::~MKLNonUniformConvolver()
{
    liveCount.fetch_sub(1, std::memory_order_relaxed);
    releaseAllLayers();
}
```

#### MKLNonUniformConvolver.cpp — releaseAllLayers()

```cpp
void MKLNonUniformConvolver::releaseAllLayers() noexcept
{
    // ... 既存の guard チェック ...

    for (int i = 0; i < kNumLayers; ++i)
        m_layers[i].freeAll();
    m_numActiveLayers = 0;
    m_latency         = 0;

    // ★ v11: freeTracked() で free + nullptr を一括化
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    freeTracked(m_ringBuf,
        static_cast<size_t>(m_ringSize) * sizeof(double));
    freeTracked(m_directIRRev,
        static_cast<size_t>(m_directTapCount) * sizeof(double));
    freeTracked(m_directHistory,
        static_cast<size_t>(m_directHistLen) * sizeof(double));
    freeTracked(m_directWindow,
        static_cast<size_t>(m_directHistLen + m_directMaxBlock) * sizeof(double));
    freeTracked(m_directOutBuf,
        static_cast<size_t>(m_directMaxBlock) * sizeof(double));
#else
    if (m_ringBuf) { mkl_free(m_ringBuf); m_ringBuf = nullptr; }
    if (m_directIRRev)   { mkl_free(m_directIRRev);   m_directIRRev = nullptr; }
    if (m_directHistory) { mkl_free(m_directHistory); m_directHistory = nullptr; }
    if (m_directWindow)  { mkl_free(m_directWindow);  m_directWindow = nullptr; }
    if (m_directOutBuf)  { mkl_free(m_directOutBuf);  m_directOutBuf = nullptr; }
#endif

    m_ringSize = m_ringMask = m_ringWrite = m_ringRead = m_ringAvail = 0;
    m_directTapCount = 0;
    m_directHistLen  = 0;
    m_directMaxBlock = 0;
    m_directPendingSamples = 0;
    m_directEnabled  = false;
    m_tailEnabled = true;
    m_tailStrength = 1.0;
    for (int i = 0; i < kNumLayers; ++i)
        m_tailLayerGain[i] = 1.0;
}
```

#### MKLNonUniformConvolver.cpp — SetImpulse()（永続バッファのみ DIAG）

**NUC レベル永続バッファ（4箇所 → DIAG_MKL_MALLOC + freeTracked）**:

```cpp
// ★ m_directIRRev, m_directHistory, m_directWindow, m_directOutBuf は
//   releaseAllLayers() で freeTracked() により解放されるため DIAG_MKL_MALLOC。
m_directIRRev = static_cast<double*>(DIAG_MKL_MALLOC(
    static_cast<size_t>(m_directTapCount) * sizeof(double), 64));
m_directHistory = (m_directHistLen > 0)
    ? static_cast<double*>(DIAG_MKL_MALLOC(
        static_cast<size_t>(m_directHistLen) * sizeof(double), 64))
    : nullptr;
m_directWindow = static_cast<double*>(DIAG_MKL_MALLOC(
    static_cast<size_t>(m_directHistLen + m_directMaxBlock) * sizeof(double), 64));
m_directOutBuf = static_cast<double*>(DIAG_MKL_MALLOC(
    static_cast<size_t>(m_directMaxBlock) * sizeof(double), 64));
```

**一時バッファ（★ v11: 明示的に診断対象外、mkl_malloc のまま）**:

```cpp
// ★ v11: 一時バッファ — SetImpulse 終了時に解放されるため DIAG 対象外。
//   peak には影響するが current には影響せず、2.33GB 調査の目的に不要。
convo::ScopedAlignedPtr<double> impulseForFft(
    static_cast<double*>(mkl_malloc(static_cast<size_t>(irLen) * sizeof(double), 64)));

// レイヤーループ内:
double* tempTime = static_cast<double*>(mkl_malloc(l.fftSize          * sizeof(double), 64));
double* tempFreq = static_cast<double*>(mkl_malloc((l.fftSize + 2)    * sizeof(double), 64));
// ...
double* swapSoA = static_cast<double*>(mkl_malloc(
    static_cast<size_t>(l.complexSize) * sizeof(double), 64));
// ...
convo::ScopedAlignedPtr<double> gainReal(
    static_cast<double*>(mkl_malloc(static_cast<size_t>(l.complexSize) * sizeof(double), 64)));
```

**Layer 永続バッファ（15箇所 → DIAG_MKL_MALLOC + allocSizes 個別保存）**:

```cpp
const size_t irBufSize  = static_cast<size_t>(l.partStride);
const size_t fdlBufSize = static_cast<size_t>(l.partStride) * 2;
const size_t irSoaSize  = static_cast<size_t>(l.numParts) * static_cast<size_t>(l.complexSize);
const size_t fdlSoaSize = static_cast<size_t>(l.numParts) * 2 * static_cast<size_t>(l.complexSize);

l.irFreqDomain = static_cast<double*>(DIAG_MKL_MALLOC(irBufSize * sizeof(double), 64));
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
l.allocSizes.irFreqDomain = irBufSize * sizeof(double);
#endif

l.irFreqReal = static_cast<double*>(DIAG_MKL_MALLOC(irSoaSize * sizeof(double), 64));
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
l.allocSizes.irFreqReal = irSoaSize * sizeof(double);
#endif

// ... 以下同様に全15バッファ（irFreqImag, fdlBuf, fdlReal, fdlImag,
//     fftTimeBuf, fftOutBuf, prevInputBuf, accumBuf, accumReal, accumImag,
//     inputAccBuf, tailOutputBuf）...
```

**m_ringBuf（1箇所 → DIAG_MKL_MALLOC）**:

```cpp
m_ringBuf = static_cast<double*>(DIAG_MKL_MALLOC(finalSize * sizeof(double), 64));
```

#### MKLNonUniformConvolver.cpp — Layer::freeAll()

```cpp
void MKLNonUniformConvolver::Layer::freeAll() noexcept
{
    // ... 既存の fftPlanOwner, fftWorkBuf 解放 ...

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    // ★ v11: freeTracked() で free + nullptr を一括化
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

## 3. Patch C: チートシート（v11 確定版）

### 置換ルール

| バッファ | 診断対象 | 確保 | 解放 | サイズ保存 |
|:--------|:--------|:----|:----|:----------|
| Layer: irFreqDomain / irFreqReal / irFreqImag / fdlBuf / fdlReal / fdlImag / fftTimeBuf / fftOutBuf / prevInputBuf / accumBuf / accumReal / accumImag / inputAccBuf / tailOutputBuf | **✅** | `DIAG_MKL_MALLOC` | `freeTracked()` | `allocSizes` に各確保直後保存 |
| NUC: m_directIRRev / m_directHistory / m_directWindow / m_directOutBuf | **✅** | `DIAG_MKL_MALLOC` | `freeTracked()` | N/A（freeAll 時サイズ計算） |
| NUC: m_ringBuf | **✅** | `DIAG_MKL_MALLOC` | `freeTracked()` | N/A |
| 一時: impulseForFft / tempTime / tempFreq / swapSoA / gainReal | **❌** | `mkl_malloc`（変更なし） | `mkl_free`（変更なし） | N/A |

---

## 4. コンパイル時切替マクロ（v11 確定版）

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

## 5. v10 からの改善点一覧

| 項目 | v10（問題） | v11（修正） |
|:-----|:----------|:-----------|
| `releaseAllLayers()` DIAG_MKL_FREE 後の nullptr | 代入なし（dangling pointer リスク） | **`freeTracked()` で free + nullptr を一括化** |
| 一時バッファの診断対象 | 曖昧（DIAG対象外でも可） | **`mkl_malloc/mkl_free` のまま確定。仕様として完全に明確化** |
| `diagMklFree()` のデバッグアサーション | なし | **`jassert(size != 0)` 追加（freeTracked 非経由でも検出）** |
