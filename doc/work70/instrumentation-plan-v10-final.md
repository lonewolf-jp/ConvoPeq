# メモリ占有調査のためのインストルメンテーション改修案 v10（最終版）— DIAG_MKL_FREE 統一 + デバッグアサーション

---

**目的**: 2.33GB メモリ占有の原因特定。推定ではなく実測で確定させる。
**v9 からの変更**: 最終2点の改善を反映。完成度 99〜100 点。

---

## 0. v9 の問題点と v10 での修正方針

| # | v9 の問題 | v10 の修正 | 重要度 |
|:--|:---------|:----------|:------|
| 1 | `releaseAllLayers()` の `m_ringBuf` / `m_direct*` が `mkl_free` のまま → 診断対象外 | **`DIAG_MKL_FREE` に統一** | ★★★ |
| 2 | `freeTracked()` にデバッグアサーションなし → 将来の保守ミスを検出できない | **`jassert(p == nullptr \|\| size != 0)` 追加** | ★★ |

---

## 1. Patch A: DiagnosticsConfig.h — freeTracked にアサーション追加

**ファイル**: `src/DiagnosticsConfig.h`

### A-1. MklAllocStats + diagMklMalloc/Free（v9 から変更なし）

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

// ... accessors, resetDiagnostics ...

} // namespace convo::diag
```

### A-2. freeTracked ヘルパ（デバッグアサーション付き）

```cpp
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS

/// ★ v10: DIAG_MKL_FREE + nullptr 代入を一括化。
///   デバッグビルドでは size == 0 をアサート（保存漏れ検出）。
template<typename T>
inline void freeTracked(T*& p, size_t size) noexcept
{
    if (p)
    {
        jassert(size != 0);  // ★ allocSizes 保存漏れ or バッファ追加時の実装漏れ検出
        DIAG_MKL_FREE(p, size);
        p = nullptr;
    }
}

#endif
```

**★ `jassert` は JUCE のデバッグアサーション。Release では消滅。**
コンストラクタの `allocSizes = {}` で全フィールドが 0 初期化されるため、
確保されなかったバッファの `size` は 0 のまま → アサートは通らない。
確保されたのに `allocSizes` に保存し忘れた場合のみアサートが発火。

---

## 2. Patch B: MKLNonUniformConvolver — releaseAllLayers も DIAG_MKL_FREE 統一

### B-1. LayerAllocSizes + Layer（v9 から変更なし）

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
```

### B-2. releaseAllLayers() — 全 MKL バッファを DIAG_MKL_FREE に統一（★ v10）

```cpp
void MKLNonUniformConvolver::releaseAllLayers() noexcept
{
    // ... 既存の guard チェック ...

    for (int i = 0; i < kNumLayers; ++i)
        m_layers[i].freeAll();
    m_numActiveLayers = 0;
    m_latency         = 0;

    // ★ v10: m_ringBuf / m_direct* も DIAG_MKL_FREE に統一。
    //   診断対象を漏れなくし、allocatedBytes との差異を最小化する。
    //   サイズは NUC のメンバ変数から取得（SetImpulse で設定済み）。
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    if (m_ringBuf)       DIAG_MKL_FREE(m_ringBuf,
                              static_cast<size_t>(m_ringSize) * sizeof(double));
    if (m_directIRRev)   DIAG_MKL_FREE(m_directIRRev,
                              static_cast<size_t>(m_directTapCount) * sizeof(double));
    if (m_directHistory) DIAG_MKL_FREE(m_directHistory,
                              static_cast<size_t>(m_directHistLen) * sizeof(double));
    if (m_directWindow)  DIAG_MKL_FREE(m_directWindow,
                              static_cast<size_t>(m_directHistLen + m_directMaxBlock) * sizeof(double));
    if (m_directOutBuf)  DIAG_MKL_FREE(m_directOutBuf,
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
    // ...
}
```

**★ 注意**: `m_ringBuf` と `m_direct*` は NUC のメンバ変数（Layer ではない）ため、
`releaseAllLayers()` 側で解放する。サイズは NUC が保持するメンバから取得。

**★ `DIAG_MKL_FREE` は `ptr` を `nullptr` に設定しない**（`mkl_free` と同じ動作）。
そのため、v9 同様に `releaseAllLayers()` の**既存のポインタリセットコードは維持**する。
実際には `m_ringSize = 0` 等の状態変数のリセットのみでポインタリセットは不要だが、
安全のため既存コードをそのまま残す。

### B-3. SetImpulse — allocSizes 個別保存（v9 から変更なし）

```cpp
l.irFreqDomain = static_cast<double*>(DIAG_MKL_MALLOC(irBufSize * sizeof(double), 64));
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
l.allocSizes.irFreqDomain = irBufSize * sizeof(double);
#endif
```

### B-4. Layer::freeAll() — freeTracked 使用（v9 から変更なし）

```cpp
freeTracked(irFreqDomain,  allocSizes.irFreqDomain);
freeTracked(irFreqReal,    allocSizes.irFreqReal);
// ...
```

### B-5. デストラクタ

```cpp
MKLNonUniformConvolver::~MKLNonUniformConvolver()
{
    liveCount.fetch_sub(1, std::memory_order_relaxed);
    releaseAllLayers();
}
```

---

## 3. Patch C: ISRRetireRouter — trackedRatio クランプ（v9 から変更なし）

```cpp
[[nodiscard]] double trackedRatio() const noexcept
{
    const uint32_t tracked = convo::consumeAtomic(
        m_trackedRetireEntries_, std::memory_order_acquire);
    const uint32_t total = pendingRetireCount();
    if (total == 0) return 0.0;
    const uint32_t clamped = std::min(tracked, total);
    return static_cast<double>(clamped) / static_cast<double>(total);
}
```

---

## 4. Patch D: Publish ログ（v9 から変更なし）

```cpp
    "[MEM_SNAP] PUBLISH gen=%d seq=%d | "
    "NUC(MKL only): live=%d alloc=%.0fMB peak=%.0fMB "
    "totalA=%.0fGB totalF=%.0fGB | "
    "Stereo=%d DSPCore=%d | "
    "Retire: pending=%u objBytes=%.1fMB(sizeof) tracked=%u/%u (%.0f%%) "
    "overflow=%llu reclaim=%llu | "
    "OS: Private=%lluMB WorkingSet=%lluMB | "
    "Untracked(other)=%.0fMB(JUCE/IPP/CRT/threads/...)",
```

---

## 5. 実装コスト見積もり（v10・最終）

| Patch | 変更ファイル数 | 追加行数 | 備考 |
|:------|:-------------|:---------|:-----|
| A: DiagnosticsConfig.h | 1 | ~68行 | freeTracked + jassert |
| B: MKLNonUniformConvolver | 2 | ~80行 | allocSizes + freeTracked + releaseAllLayers統一 |
| C: ISRRetireRouter | 1 | ~3行 | trackedRatio クランプ |
| D: Publish ログ | 1 | ~2行 | Untracked(other) |
| **合計** | **4〜5ファイル** | **~153行** | |

---

## 6. チートシート（実装時に確認すべき全変更箇所）

### 変更するファイル一覧

| # | ファイル | 変更内容 |
|:--|:--------|:--------|
| 1 | `src/DiagnosticsConfig.h` | `MklAllocStats` + `diagMklMalloc` + `diagMklFree` + `freeTracked` + `updateAtomicMaximum64` + `resetDiagnostics` + マクロ |
| 2 | `src/MKLNonUniformConvolver.h` | `LayerAllocSizes` 構造体追加 + `Layer::allocSizes` メンバ + `liveCount` static |
| 3 | `src/MKLNonUniformConvolver.cpp` | 全 `mkl_malloc` → `DIAG_MKL_MALLOC` + allocSizes 保存 + `freeTracked` + `releaseAllLayers` DIAG_MKL_FREE |
| 4 | `src/DeferredDeletionQueue.h` | `DeletionEntry` に `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` で `objectBytes` 追加 |
| 5 | `src/audioengine/ISRRetireRouter.h` | `m_pendingRetireBytes_` + `m_trackedRetireEntries_` + `pendingRetireBytes()` + `trackedRatio()` |
| 6 | `src/audioengine/ISRRetireRouter.cpp` | `enqueueRetire` の `objectBytes` オーバーロード実装 |
| 7 | `src/core/IEpochProvider.h` | `enqueueRetire` に `objectBytes` パラメータ追加 |
| 8 | `src/core/EpochDomain.h` | `enqueueRetire` 実装で `entry.objectBytes = objectBytes` |
| 9 | `src/audioengine/AudioEngine.Processing.DSPCoreLifecycle.cpp` | DSPCore::prepare 確保量ログ |
| 10 | `src/audioengine/AudioEngine.Timer.cpp` | Publish フルスナップショット + OS PrivateUsage |

### mkl_malloc → DIAG_MKL_MALLOC 置換一覧（28箇所）

**SetImpulse 内（NUC レベル、4箇所）**:
- `m_directIRRev`: `m_directTapCount * sizeof(double)`
- `m_directHistory`: `m_directHistLen * sizeof(double)` （条件付き）
- `m_directWindow`: `(m_directHistLen + m_directMaxBlock) * sizeof(double)`
- `m_directOutBuf`: `m_directMaxBlock * sizeof(double)`

**SetImpulse 内（一時バッファ、4箇所 — DIAG 対象外でも可）**:
- `impulseForFft`: `irLen * sizeof(double)`
- `tempTime`: `l.fftSize * sizeof(double)`
- `tempFreq`: `(l.fftSize + 2) * sizeof(double)`
- `swapSoA`: `l.complexSize * sizeof(double)`

**SetImpulse 内（Layer バッファ、15箇所 → allocSizes 保存あり）**:
- `irFreqDomain`, `irFreqReal`, `irFreqImag`, `fdlBuf`, `fdlReal`, `fdlImag`
- `fftTimeBuf`, `fftOutBuf`, `prevInputBuf`
- `accumBuf`, `accumReal`, `accumImag`, `inputAccBuf`
- `tailOutputBuf` （条件付き）

**SetImpulse 内（ringBuf、1箇所）**:
- `m_ringBuf`: `finalSize * sizeof(double)`

### mkl_free → DIAG_MKL_FREE 置換一覧

| 場所 | 対象 | サイズ取得元 |
|:----|:-----|:------------|
| `Layer::freeAll()` | 14 Layer バッファ | `allocSizes.XXX` |
| `releaseAllLayers()` | `m_ringBuf` | `m_ringSize * sizeof(double)` |
| `releaseAllLayers()` | `m_directIRRev` | `m_directTapCount * sizeof(double)` |
| `releaseAllLayers()` | `m_directHistory` | `m_directHistLen * sizeof(double)` |
| `releaseAllLayers()` | `m_directWindow` | `(m_directHistLen + m_directMaxBlock) * sizeof(double)` |
| `releaseAllLayers()` | `m_directOutBuf` | `m_directMaxBlock * sizeof(double)` |

---

## 7. v9 からの改善点一覧

| 項目 | v9（問題） | v10（修正） |
|:-----|:----------|:-----------|
| `releaseAllLayers()` m_ringBuf/direct | `mkl_free` のまま（診断対象外） | **`DIAG_MKL_FREE` に統一** |
| `freeTracked()` デバッグアサーション | なし | **`jassert(size != 0)` 追加（保存漏れ検出）** |
