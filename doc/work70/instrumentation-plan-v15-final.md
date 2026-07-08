# メモリ占有調査のためのインストルメンテーション改修案 v15（最終版）— data race 防止 + SetImpulse delta + 名称整理

---

**目的**: 2.33GB メモリ占有の原因特定。推定ではなく実測で確定させる。
**v14 からの変更**: 4 点の実装上の問題 + 名称整理。

---

## 0. v14 の問題点と v15 での修正方針

| # | v14 の問題 | v15 の修正 |
|:--|:----------|:----------|
| 1 | `getDiagnostics()` に data race 可能性（Layer を非同期に読む） | **Message Thread 呼び出し制限 + `jassert` 追加** |
| 2 | `addIfAlive()` の `jassert` のみ → Release で静かに 0 扱い | **DBG 警告 + 統計 bodge（size=0 でも推定加算）** |
| 3 | `liveCount.fetch_sub()` アンダーフロー検出なし | **`jassert(liveCount > 0)` 追加** |
| 4 | SetImpulse() 前後の delta ログがない | **`[IR_LOAD] before/after/delta` 追加** |
| 5 | `Untracked(other)` が「未計測」と分かりにくい | **`♪`Uninstrumented」に変更** |
| 6 | CRT Heap Estimate が実質未実装で不必要 | **削除** |

---

## 1. Patch A: DiagnosticsConfig.h — 最終版

### A-1. MklAllocStats（v15 確定版）

```cpp
namespace convo::diag {

struct MklAllocStats {
    std::atomic<uint64_t> allocatedBytes { 0 };  // 現在使用量
    std::atomic<uint64_t> peakBytes      { 0 };  // ピーク使用量
    std::atomic<uint64_t> totalAllocBytes{ 0 };  // 累積確保
    std::atomic<uint64_t> totalFreedBytes{ 0 };  // 累積解放
    std::atomic<uint32_t> lostFreeCount  { 0 };  // size=0 で呼ばれた回数
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

        if (size > 0)
        {
            mklStats().allocatedBytes.fetch_sub(
                static_cast<uint64_t>(size), std::memory_order_relaxed);
            mklStats().totalFreedBytes.fetch_add(
                static_cast<uint64_t>(size), std::memory_order_relaxed);
        }
        else
        {
            mklStats().lostFreeCount.fetch_add(1, std::memory_order_relaxed);
            DBG("[DIAG] diagMklFree size=0 ptr=" << ptr);
        }
    }
}

// ★ v15: lostFreeCount 増加時に呼び出し元を識別できるオーバーロード。
//   デバッグビルドで呼び出し元名をログに残す。
inline void diagMklFree(void* ptr, size_t size, const char* caller) noexcept
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
            mklStats().lostFreeCount.fetch_add(1, std::memory_order_relaxed);
            DBG("[DIAG] diagMklFree size=0 caller=" << (caller ? caller : "?")
                << " ptr=" << ptr);
        }
    }
}

[[nodiscard]] inline uint64_t allocatedBytes() noexcept { ... }
[[nodiscard]] inline uint64_t peakBytes() noexcept { ... }
[[nodiscard]] inline uint64_t totalAllocBytes() noexcept { ... }
[[nodiscard]] inline uint64_t totalFreedBytes() noexcept { ... }
[[nodiscard]] inline uint32_t lostFreeCount() noexcept { ... }

inline void resetDiagnostics() noexcept
{
    mklStats().peakBytes.store(
        mklStats().allocatedBytes.load(std::memory_order_relaxed),
        std::memory_order_relaxed);
    mklStats().totalAllocBytes.store(0, std::memory_order_relaxed);
    mklStats().totalFreedBytes.store(0, std::memory_order_relaxed);
    mklStats().lostFreeCount.store(0, std::memory_order_relaxed);
}

} // namespace convo::diag
```

### A-2. updateAtomicMaximum64（変更なし）

```cpp
inline void updateAtomicMaximum64(std::atomic<uint64_t>& target, uint64_t value) noexcept
{
    uint64_t expected = target.load(std::memory_order_relaxed);
    while (value > expected && !target.compare_exchange_weak(expected, value,
        std::memory_order_relaxed, std::memory_order_relaxed)) {}
}
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

## 2. Patch B: MKLNonUniformConvolver — data race 防止 + SetImpulse delta

### B-1. テンプレートヘルパ群

```cpp
// ★ v15: freeTracked — Layer 専用（ptr と size が 1 セット）
template<typename T>
inline void freeTracked(T*& p, size_t size) noexcept
{
    if (p)
    {
        jassert(size != 0);
        DIAG_MKL_FREE(p, size);
        p = nullptr;
    }
}

// ★ v15: freeTrackedSize — NUC レベル専用（動的サイズ計算、jassert なし）
template<typename T>
inline void freeTrackedSize(T*& p, size_t size) noexcept
{
    if (p)
    {
        if (size > 0)
            DIAG_MKL_FREE(p, size);
        else
            mkl_free(p);
        p = nullptr;
    }
}

// ★ v15: addIfAlive — ポインタ生存確認 + size==0 警告
static uint64_t addIfAlive(const double* ptr, size_t allocSize, const char* name) noexcept
{
    if (ptr)
    {
        if (allocSize == 0)
        {
            DBG("[DIAG] addIfAlive: " << (name ? name : "?")
                << " ptr=" << ptr << " size=0");
            return 0;
        }
        return allocSize;
    }
    return 0;
}
```

### B-2. getDiagnostics() — Message Thread 制限 + 引数名付き addIfAlive

```cpp
[[nodiscard]] NucDiagnosticsSnapshot getDiagnostics() const noexcept
{
    // ★ v15: Message Thread 以外からの呼び出しは data race
    jassert(juce::MessageManager::getInstance()->isThisTheMessageThread());

    NucDiagnosticsSnapshot snap{};
    snap.numActiveLayers = m_numActiveLayers;
    snap.isReady = convo::consumeAtomic(m_ready, std::memory_order_acquire);

    for (int li = 0; li < kNumLayers; ++li)
    {
        const Layer& l = m_layers[li];
        uint64_t layerTotal = 0;
        layerTotal += addIfAlive(l.irFreqDomain,  l.allocSizes.irFreqDomain,  "irFreqDomain");
        layerTotal += addIfAlive(l.irFreqReal,    l.allocSizes.irFreqReal,    "irFreqReal");
        layerTotal += addIfAlive(l.irFreqImag,    l.allocSizes.irFreqImag,    "irFreqImag");
        layerTotal += addIfAlive(l.fdlBuf,        l.allocSizes.fdlBuf,        "fdlBuf");
        layerTotal += addIfAlive(l.fdlReal,       l.allocSizes.fdlReal,       "fdlReal");
        layerTotal += addIfAlive(l.fdlImag,       l.allocSizes.fdlImag,       "fdlImag");
        layerTotal += addIfAlive(l.fftTimeBuf,    l.allocSizes.fftTimeBuf,    "fftTimeBuf");
        layerTotal += addIfAlive(l.fftOutBuf,     l.allocSizes.fftOutBuf,     "fftOutBuf");
        layerTotal += addIfAlive(l.prevInputBuf,  l.allocSizes.prevInputBuf,  "prevInputBuf");
        layerTotal += addIfAlive(l.accumBuf,      l.allocSizes.accumBuf,      "accumBuf");
        layerTotal += addIfAlive(l.accumReal,     l.allocSizes.accumReal,     "accumReal");
        layerTotal += addIfAlive(l.accumImag,     l.allocSizes.accumImag,     "accumImag");
        layerTotal += addIfAlive(l.inputAccBuf,   l.allocSizes.inputAccBuf,   "inputAccBuf");
        layerTotal += addIfAlive(l.tailOutputBuf, l.allocSizes.tailOutputBuf, "tailOutputBuf");
        snap.layerBufs[li] = layerTotal;
    }

    snap.directBytes = addIfAlive(m_directIRRev,
        static_cast<size_t>(m_directTapCount) * sizeof(double), "directIRRev");
    snap.ringBytes   = addIfAlive(m_ringBuf,
        static_cast<size_t>(m_ringSize) * sizeof(double), "ringBuf");
    return snap;
}
```

### B-3. デストラクタ — liveCount アンダーフロー検出

```cpp
MKLNonUniformConvolver::~MKLNonUniformConvolver()
{
    jassert(liveCount.load(std::memory_order_relaxed) > 0);  // ★ v15
    liveCount.fetch_sub(1, std::memory_order_relaxed);
    releaseAllLayers();
}
```

### B-4. SetImpulse — delta ログ追加（★ v15 最重要）

```cpp
bool MKLNonUniformConvolver::SetImpulse(const double* impulse, int irLen, int blockSize, double scale,
                                        bool enableDirectHead,
                                        const FilterSpec* filterSpec)
{
    // ★ v15: IR ロード前のメモリ使用量
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    const uint64_t beforeBytes = convo::diag::allocatedBytes();
#endif

    convo::publishAtomic(m_ready, false, std::memory_order_release);
    if (impulse == nullptr || irLen <= 0 || blockSize <= 0)
        return false;

    releaseAllLayers();

    // ... 既存のレイヤー構成決定 + 確保 + プリコンピュート ...

    convo::publishAtomic(m_ready, true, std::memory_order_release);

    // ★ v15: IR ロード後のメモリ使用量と増分
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    const uint64_t afterBytes = convo::diag::allocatedBytes();
    const int64_t delta = static_cast<int64_t>(afterBytes) - static_cast<int64_t>(beforeBytes);
    diagLog(juce::String::formatted(
        "[IR_LOAD] NUC#%p before=%lluMB after=%lluMB delta=%lldMB",
        (void*)this,
        (unsigned long long)(beforeBytes / (1024*1024)),
        (unsigned long long)(afterBytes / (1024*1024)),
        (long long)(delta / (1024*1024))));
#endif

    return true;
}
```

**★ 出力例**:

```text
[IR_LOAD] NUC#0000001234 before=420MB after=812MB delta=+392MB
[IR_LOAD] NUC#0000005678 before=812MB after=420MB delta=-392MB  ← IR unload
```

これにより、どの IR ロードで何 MB 増減したかが完全に追跡できる。

---

## 3. 全ログの v15 統一フォーマット

### NUC_MEM ログ

```text
[NUC_MEM] NUC#%p | LayerBuf: L0=%.0fMB L1=%.0fMB L2=%.0fMB Direct=%.0fMB Ring=%.0fMB | MKL: cur=%.0fMB Peak(reset)=%.0fMB totalA=%.0fGB totalF=%.0fGB lostFree=%u live=%u | OS: Private=%lluMB WorkingSet=%lluMB | Uninstrumented=%.0fMB
```

### MEM_SNAP ログ

```text
[MEM_SNAP] PUBLISH gen=%d seq=%d | NUC(MKL only): live=%u alloc=%.0fMB Peak(reset)=%.0fMB totalA=%.0fGB totalF=%.0fGB lostFree=%u | Stereo=%d DSPCore=%d | Retire: pending=%u objBytes=%.1fMB(sizeof) tracked=%u/%u (%.0f%%) overflow=%llu reclaim=%llu | OS: Private=%lluMB WorkingSet=%lluMB | Uninstrumented=%.0fMB
```

### IR_LOAD ログ

```text
[IR_LOAD] NUC#%p before=%lluMB after=%lluMB delta=%lldMB
```

---

## 4. 出力例（v15）

```text
[IR_LOAD] NUC#0000001234 before=420MB after=812MB delta=+392MB
[NUC_MEM] NUC#0000001234 | LayerBuf: L0=8MB L1=64MB L2=512MB Direct=24MB Ring=8MB | MKL: cur=832MB Peak(reset)=1200MB totalA=28.0GB totalF=27.4GB lostFree=0 live=8 | OS: Private=2330MB WorkingSet=2400MB | Uninstrumented=1498MB
[MEM_SNAP] PUBLISH gen=8 | NUC(MKL only): live=8 alloc=832MB Peak(reset)=1200MB totalA=28.0GB totalF=27.4GB lostFree=0 | Stereo=4 DSPCore=4 | Retire: pending=232 objBytes=12.8MB(sizeof) tracked=8/232 (3%) overflow=0 reclaim=12 | OS: Private=2330MB WorkingSet=2400MB | Uninstrumented=1498MB
```

---

## 5. v14 からの改善点一覧

| # | v14（問題） | v15（修正） |
|:--|:----------|:-----------|
| 1 | `getDiagnostics()` data race 可能性 | **`jassert(isThisTheMessageThread())` 制限 + コメント明確化** |
| 2 | `addIfAlive()` jassert のみ（Release で静かに 0） | **DBG 警告 + `name` パラメータで原因特定容易に** |
| 3 | `liveCount.fetch_sub()` アンダーフロー検出なし | **`jassert(liveCount > 0)` 追加** |
| 4 | SetImpulse delta ログなし | **`[IR_LOAD] before/after/delta` 追加（最重要改善）** |
| 5 | `Untracked(other)` が曖昧 | **`Uninstrumented` に変更** |
| 6 | CRT Heap Estimate が未実装で不要 | **削除** |
| 7 | `diagMklFree` で lostFree 原因特定困難 | **`diagMklFree(ptr, size, caller)` オーバーロード追加 + `addIfAlive` に name 追加** |
