# メモリ占有調査のためのインストルメンテーション改修案 v4 — SetImpulse 差分追跡 + diagMklMalloc + OS PrivateUsage

---

**目的**: 2.33GB メモリ占有の原因特定。推定ではなく実測で確定させる。
**v3 からの変更**: reviewer フィードバックに基づく設計改修（3重大問題の修正 + 5改善点の反映）。

---

## 0. v3 レビュー結果と v4 での修正方針

### ★ 重大問題①②③（統合修正）: `layerAllocated` + `freeAll()` 減算設計の撤廃

**v3 の問題点**:
- `Layer::freeAll()` で `globalAllocated.fetch_sub(layerAllocated)` すると、
  `SetImpulse()` の中で `releaseAllLayers()` → 新規確保の途中で `globalAllocated` が **一時的に 0 MB に低下**する。
- `thisAllocated` は `freeAll()` で減算されず、増える一方になる（60MB → IR変更 → 120MB）。
- `Layer` に `size_t layerAllocated` メンバを追加する意味が薄い（+8byte/Layer の無駄）。

**v4 の修正: SetImpulse 差分追跡方式**

```
SetImpulse() 開始
  ├─ oldBytes = globalAllocated.load()
  ├─ releaseAllLayers()      ← 既存のまま（内部で 0 になってもOK）
  ├─ 新規 Layer 生成         ← 既存のまま
  ├─ m_ringBuf 確保          ← 既存のまま
  ├─ newBytes = globalAllocated.load()
  ├─ delta = newBytes - oldBytes
  ├─ globalAllocated += delta
  └─ peak 更新（CAS or compare_exchange）
```

**利点**:
- `Layer` 構造体への `layerAllocated` メンバ追加不要（+0 byte / Layer）。
- `freeAll()` は既存のまま変更不要。
- 再構築中の一時的な値に左右されない。
- `globalAllocated` は常に「現在の実使用量」を正確に反映。

### 問題④: `DIAG_ALLOC` ×28箇所 → `diagMklMalloc` / `diagMklFree` ラッパー

**v3 の問題点**: `DIAG_ALLOC` マクロを 28 箇所に手動で挿入する設計は保守性が悪い。
将来 `mkl_malloc` が 1 個増えたら `DIAG_ALLOC` も忘れ込む。

**v4 の修正**: ラッパー関数を 1 か所に定義し、全 `mkl_malloc` / `mkl_free` を差し替える。

### 問題⑤: peak 更新

CAS ループは現状の `updateAtomicMaximum` パターンで十分。`uint64_t` 版を追加。

### 改善点: Retire bytes

`pendingRetireCount()` のみでは「100個 = 100KB か 1GB か」が不明。
**`pendingRetireBytes()` 追加**（ISRRetireRouter に `m_pendingRetireBytes_` カウンタ追加）。

### 改善点: Publish ログの重点化

Publish / Retire / IR Reload の **3イベント** を中心に。
定期ログは補助程度。

### 改善点: DSPCore 処理バッファ

DSPCore 数だけでなく `alignedL/R`、`dryBypassBuffer`、`oversampling` の容量も計測。

### ★ 最重要: OS Private Usage との比較

`DiagnosticsConfig.h` に既存の `getProcessMemoryInfo()` を Publish ログに含める。
追跡済みメモリ vs OS Private Usage の差分が即座に分かる。

---

## 1. Patch A: DiagnosticsConfig.h — diagMklMalloc / diagMklFree + uint64_t peak ヘルパー

**ファイル**: `src/DiagnosticsConfig.h`
**位置**: `updateAtomicMaximum` の隣

### A-1. uint64_t 版 `updateAtomicMaximum64` 追加

```cpp
// ---- updateAtomicMaximum64 : uint64_t 版 ----
inline void updateAtomicMaximum64(std::atomic<uint64_t>& target, uint64_t value) noexcept
{
    uint64_t expected = target.load(std::memory_order_relaxed);
    while (value > expected && !target.compare_exchange_weak(expected, value,
        std::memory_order_relaxed, std::memory_order_relaxed)) {}
}
```

### A-2. diagMklMalloc / diagMklFree / グローバルカウンタ

```cpp
// ---- MKL 分配トラッキング ----
// 全 NUC インスタンスで共有するグローバルカウンタ。
// SetImpulse() の差分追跡方式により、再構築中の一時的 0 は発生しない。
namespace convo::diag {

struct MklAllocStats {
    std::atomic<uint64_t> allocatedBytes { 0 };   // 現在使用量
    std::atomic<uint64_t> peakBytes      { 0 };   // ピーク使用量
    std::atomic<uint32_t> allocCount     { 0 };   // 確保回数
    std::atomic<uint32_t> freeCount      { 0 };   // 解放回数
};

inline MklAllocStats& mklStats() noexcept
{
    static MklAllocStats stats{};
    return stats;
}

/// mkl_malloc のラッパー。成功時のみ allocatedBytes を増加させる。
inline void* diagMklMalloc(size_t size, int alignment) noexcept
{
    void* ptr = mkl_malloc(size, alignment);
    if (ptr)
    {
        const uint64_t bytes = static_cast<uint64_t>(size);
        mklStats().allocatedBytes.fetch_add(bytes, std::memory_order_relaxed);
        mklStats().allocCount.fetch_add(1, std::memory_order_relaxed);
        updateAtomicMaximum64(mklStats().peakBytes,
            mklStats().allocatedBytes.load(std::memory_order_relaxed));
    }
    return ptr;
}

/// mkl_free のラッパー。ポインタが非 null の場合のみ allocatedBytes を減少させる。
inline void diagMklFree(void* ptr, size_t expectedSize) noexcept
{
    if (ptr)
    {
        mkl_free(ptr);
        mklStats().allocatedBytes.fetch_sub(
            static_cast<uint64_t>(expectedSize), std::memory_order_relaxed);
        mklStats().freeCount.fetch_add(1, std::memory_order_relaxed);
    }
}

} // namespace convo::diag
```

**設計判断**:
- `diagMklFree` は `expectedSize` パラメータを受け取る。呼び出し元が確保サイズを知っているため、
  `layerAllocated` メンバを Layer に持たせる必要がない。
- `peakBytes` は `allocatedBytes` の read-modify-write 内で更新。
  アロケーション頻度は低いため（SetImpulse 時のみ）、競合は無視できる。

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

## 2. Patch B: MKLNonUniformConvolver — SetImpulse 差分追跡 + DIAG_MKL_MALLOC 置換

### B-1. MKLNonUniformConvolver.h — static カウンタ宣言

**ファイル**: `src/MKLNonUniformConvolver.h`
**位置**: `class MKLNonUniformConvolver` の public セクション

```cpp
class MKLNonUniformConvolver {
public:
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    // [DIAG v4] SetImpulse 差分追跡方式のグローバルカウンタ。
    // layerAllocated メンバは不要（SetImpulse の old/new 差分で管理）。
    static std::atomic<uint64_t> globalAllocated;
    static std::atomic<uint64_t> peakAllocated;
    static std::atomic<int>      liveCount;
#endif
    // ... 既存のメンバ ...
```

**★ Layer 構造体への `layerAllocated` 追加は不要**（v3 から撤廃）。

### B-2. MKLNonUniformConvolver.cpp — static 変数の実体

```cpp
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
std::atomic<uint64_t> MKLNonUniformConvolver::globalAllocated { 0 };
std::atomic<uint64_t> MKLNonUniformConvolver::peakAllocated { 0 };
std::atomic<int>      MKLNonUniformConvolver::liveCount { 0 };
#endif
```

### B-3. MKLNonUniformConvolver.cpp — 全 mkl_malloc → DIAG_MKL_MALLOC 置換

**対象**: SetImpulse 内の全 25 箇所の `mkl_malloc` + releaseAllLayers 内の `mkl_free`

置換パターン（全箇所共通）:

```cpp
// ★ Before (v3):
l.irFreqDomain = static_cast<double*>(mkl_malloc(irBufSize * sizeof(double), 64));

// ★ After (v4):
l.irFreqDomain = static_cast<double*>(DIAG_MKL_MALLOC(irBufSize * sizeof(double), 64));
```

**SetImpulse 内の確保箇所一覧** (25箇所):

| # | 用途 | 確保サイズ式 | free サイズ |
|:--|:----|:-----------|:-----------|
| 1 | m_directIRRev | `m_directTapCount * sizeof(double)` | 同 |
| 2 | m_directHistory | `m_directHistLen * sizeof(double)` | 同（条件付き） |
| 3 | m_directWindow | `(m_directHistLen + m_directMaxBlock) * sizeof(double)` | 同 |
| 4 | m_directOutBuf | `m_directMaxBlock * sizeof(double)` | 同 |
| 5 | impulseForFft | `irLen * sizeof(double)` | 同（一時） |
| 6-19 | Layer メンバ ×14 | 各レイヤーの `size * sizeof(double)` | 各 freeAll で解放 |
| 20 | tempTime | `l.fftSize * sizeof(double)` | 同（一時） |
| 21 | tempFreq | `(l.fftSize + 2) * sizeof(double)` | 同（一時） |
| 22 | swapSoA | `l.complexSize * sizeof(double)` | 同（一時） |
| 23 | m_ringBuf | `finalSize * sizeof(double)` | 同 |
| 24 | gainReal | `l.complexSize * sizeof(double)` | 同（一時） |

**★ 重要**: 一時バッファ（impulseForFft, tempTime, tempFreq, swapSoA, gainReal）も `DIAG_MKL_MALLOC` を使用する。
確保→即使用→即解放のため、`allocatedBytes` への影響は一時的であり、
`SetImpulse()` 終了後の差分には含まれない。

### B-4. SetImpulse 差分追跡の実装

**ファイル**: `src/MKLNonUniformConvolver.cpp`
**位置**: SetImpulse の冒頭と末尾

```cpp
bool MKLNonUniformConvolver::SetImpulse(const double* impulse, int irLen, int blockSize, double scale,
                                        bool enableDirectHead,
                                        const FilterSpec* filterSpec)
{
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    // ★ v4: 差分追跡 — 開始前の使用量を記録
    const uint64_t allocBefore = globalAllocated.load(std::memory_order_relaxed);
#endif

    convo::publishAtomic(m_ready, false, std::memory_order_release);

    if (impulse == nullptr || irLen <= 0 || blockSize <= 0)
        return false;

    releaseAllLayers();

    // ... 既存のレイヤー構成決定・確保ロジック（変更なし）...

    // ... mkl_malloc → DIAG_MKL_MALLOC 置換（全 25 箇所）...

    // ... applySpectrumFilter など ...

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    // ★ v4: 差分追跡 — 終了後の使用量から差分を計算
    const uint64_t allocAfter = globalAllocated.load(std::memory_order_relaxed);
    if (allocAfter >= allocBefore)
        globalAllocated.fetch_add(allocAfter - allocBefore, std::memory_order_relaxed);
    else
        globalAllocated.fetch_sub(allocBefore - allocAfter, std::memory_order_relaxed);
    updateAtomicMaximum64(peakAllocated, globalAllocated.load(std::memory_order_relaxed));
#endif

    convo::publishAtomic(m_ready, true, std::memory_order_release);
    return true;
}
```

**★ 解説**:
- `releaseAllLayers()` 内で `DIAG_MKL_FREE` が呼ばれ、`allocatedBytes` が減少する。
- 新規確保で `DIAG_MKL_MALLOC` が呼ばれ、`allocatedBytes` が増加する。
- SetImpulse 終了時の `allocatedBytes` が「このインスタンスが現在保有する合計」。
- `globalAllocated`（NUC 静的カウンタ）は、差分を反映して更新する。
- 一時バッファ（tempTime, tempFreq 等）は SetImpulse 中に確保→解放されるため、
  allocAfter には含まれない。

### B-5. freeAll() での DIAG_MKL_FREE 置換

```cpp
void MKLNonUniformConvolver::Layer::freeAll() noexcept
{
    // ... 既存の fftPlanOwner, fftWorkBuf 解放（IPP バッファ、DIAG 対象外）...

    // ★ v4: mkl_free → DIAG_MKL_FREE 置換（サイズを明示）
    if (irFreqDomain)  { DIAG_MKL_FREE(irFreqDomain,  irBufSize  * sizeof(double));  irFreqDomain  = nullptr; }
    if (irFreqReal)    { DIAG_MKL_FREE(irFreqReal,    irSoaSize  * sizeof(double));  irFreqReal    = nullptr; }
    if (irFreqImag)    { DIAG_MKL_FREE(irFreqImag,    irSoaSize  * sizeof(double));  irFreqImag    = nullptr; }
    if (fdlBuf)        { DIAG_MKL_FREE(fdlBuf,        fdlBufSize * sizeof(double));  fdlBuf        = nullptr; }
    if (fdlReal)       { DIAG_MKL_FREE(fdlReal,       fdlSoaSize * sizeof(double));  fdlReal       = nullptr; }
    if (fdlImag)       { DIAG_MKL_FREE(fdlImag,       fdlSoaSize * sizeof(double));  fdlImag       = nullptr; }
    if (fftTimeBuf)    { DIAG_MKL_FREE(fftTimeBuf,    fftSize    * sizeof(double));  fftTimeBuf    = nullptr; }
    if (fftOutBuf)     { DIAG_MKL_FREE(fftOutBuf,     fftSize    * sizeof(double));  fftOutBuf     = nullptr; }
    if (prevInputBuf)  { DIAG_MKL_FREE(prevInputBuf,  partSize   * sizeof(double));  prevInputBuf  = nullptr; }
    if (accumBuf)      { DIAG_MKL_FREE(accumBuf,      partStride * sizeof(double));  accumBuf      = nullptr; }
    if (accumReal)     { DIAG_MKL_FREE(accumReal,     complexSize * sizeof(double)); accumReal     = nullptr; }
    if (accumImag)     { DIAG_MKL_FREE(accumImag,     complexSize * sizeof(double)); accumImag     = nullptr; }
    if (inputAccBuf)   { DIAG_MKL_FREE(inputAccBuf,   partSize   * sizeof(double));  inputAccBuf   = nullptr; }
    if (tailOutputBuf) { DIAG_MKL_FREE(tailOutputBuf, partSize   * sizeof(double));  tailOutputBuf = nullptr; }

    // ... 既存の状態リセット ...
}
```

**★ freeAll() 内で DIAG_MKL_FREE が `allocatedBytes` を自動的に減算**する。
`globalAllocated` の直接操作は不要（SetImpulse の差分で管理）。

**★ freeAll() のレイヤーサイズ**: `irBufSize`, `irSoaSize`, `fdlBufSize` 等は
SetImpulse() のレイヤー初期化ループ内で計算される。freeAll() から参照するには
`Layer` のメンバに保存するか、`freeAll()` の呼び出し元（`releaseAllLayers()`）
から渡す必要がある。

**推奨アプローチ**: freeAll() にサイズパラメータを渡す代わりに、
Layer 構造体に `size_t allocatedBytes` メンバを SetImpulse 時にセットし、
freeAll() で `DIAG_MKL_FREE` に使用する。ただし `layerAllocated` のような
「累積管理用」ではなく、**freeAll() 内の解放サイズ指定用**としてのみ使用する。

```cpp
struct Layer {
    // ... 既存メンバ ...

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    // freeAll() 内の DIAG_MKL_FREE に使用する解放サイズ。
    // SetImpulse() で全レイヤー初期化後に一度だけ計算・セットする。
    // globalAllocated の管理には使用しない（SetImpulse 差分方式）。
    size_t freeAllBytes = 0;
#endif
};
```

### B-6. releaseAllLayers() — m_ringBuf / m_direct* の DIAG_MKL_FREE

```cpp
void MKLNonUniformConvolver::releaseAllLayers() noexcept
{
    // ... 既存の guard チェック ...

    for (int i = 0; i < kNumLayers; ++i)
        m_layers[i].freeAll();
    m_numActiveLayers = 0;
    m_latency         = 0;

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    // ★ v4: ringBuf / direct バッファも DIAG_MKL_FREE で解放
    if (m_ringBuf)   { DIAG_MKL_FREE(m_ringBuf, m_ringSize * sizeof(double)); m_ringBuf = nullptr; }
    if (m_directIRRev)   { DIAG_MKL_FREE(m_directIRRev,   m_directTapCount * sizeof(double));                m_directIRRev = nullptr; }
    if (m_directHistory) { DIAG_MKL_FREE(m_directHistory,  m_directHistLen * sizeof(double));                 m_directHistory = nullptr; }
    if (m_directWindow)  { DIAG_MKL_FREE(m_directWindow,  (m_directHistLen + m_directMaxBlock) * sizeof(double)); m_directWindow = nullptr; }
    if (m_directOutBuf)  { DIAG_MKL_FREE(m_directOutBuf,  m_directMaxBlock * sizeof(double));                m_directOutBuf = nullptr; }
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

### B-7. デストラクタ

```cpp
MKLNonUniformConvolver::~MKLNonUniformConvolver()
{
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    liveCount--;
#endif
    releaseAllLayers();
    // ★ v4: thisAllocated の減算は不要
    //   → freeAll() 内の DIAG_MKL_FREE が globalAllocated を自動更新済み。
    //   → SetImpulse 差分方式により、releaseAllLayers() 後の globalAllocated は
    //     「他のインスタンスの使用量」を正確に反映している。
}
```

### B-8. NUC 合計ログ（SetImpulse 成功直後）

**ファイル**: `src/MKLNonUniformConvolver.cpp` — SetImpulse 末尾

```cpp
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
{
    // ★ v4: OS メモリ情報を含むログ
    const auto osMem = getProcessMemoryInfo();
    diagLog(juce::String::formatted(
        "[NUC_MEM] NUC#%p L0:%zuMB L1:%zuMB L2:%zuMB | "
        "globalAlloc=%lluMB peak=%lluMB live=%d | "
        "OS: Private=%lluMB WorkingSet=%lluMB",
        (void*)this,
        m_numActiveLayers >= 1 ? m_layers[0].freeAllBytes / (1024*1024) : 0,
        m_numActiveLayers >= 2 ? m_layers[1].freeAllBytes / (1024*1024) : 0,
        m_numActiveLayers >= 3 ? m_layers[2].freeAllBytes / (1024*1024) : 0,
        (unsigned long long)globalAllocated.load() / (1024*1024),
        (unsigned long long)peakAllocated.load() / (1024*1024),
        (int)liveCount.load(),
        (unsigned long long)osMem.privateUsageMB,
        (unsigned long long)osMem.workingSetMB));
}
#endif
```

---

## 3. Patch C: ISRRetireRouter — pendingRetireBytes 追加

**ファイル**: `src/audioengine/ISRRetireRouter.h` + `.cpp`

### C-1. メンバ変数追加 (.h)

```cpp
class ISRRetireRouter : public convo::IEpochProvider {
    // ... 既存メンバ ...

    // ★ v4: pending retire バイト数カウンタ
    std::atomic<uint64_t> m_pendingRetireBytes_ { 0 };
};
```

### C-2. API 追加 (.h)

```cpp
    /// 退役キュー滞留バイト数（「100個 = 100KB か 1GB か」を判別するため）
    [[nodiscard]] uint64_t pendingRetireBytes() const noexcept override {
        return convo::consumeAtomic(m_pendingRetireBytes_, std::memory_order_acquire);
    }
```

### C-3. enqueueRetire / tryReclaim での更新 (.cpp)

```cpp
// enqueueRetire 成功時:
m_pendingRetireBytes_.fetch_add(entrySize, std::memory_order_relaxed);

// tryReclaim 成功時 (reclaim されたエントリのサイズ分減算):
m_pendingRetireBytes_.fetch_sub(reclaimedSize, std::memory_order_relaxed);
```

**★ 注意**: `DeletionEntryType` に `entrySize` フィールドが既にあればそのまま使用。
無ければ、`enqueueRetire` の呼び出し元から size を渡すか、
Entry 構造体に size フィールドを追加する。

### C-4. IEpochProvider インターフェースへの追加

```cpp
// core/IEpochProvider.h
[[nodiscard]] virtual uint64_t pendingRetireBytes() const noexcept = 0;
```

---

## 4. Patch D: DSPCore 処理バッファ計測

**ファイル**: `src/audioengine/AudioEngine.Processing.DSPCoreLifecycle.cpp`

### D-1. DSPCore liveCount

```cpp
// AudioEngine.h — struct DSPCore
static std::atomic<int> liveCount;
DSPCore() { liveCount++; }
~DSPCore() { liveCount--; }
```

### D-2. DSPCore::prepare 確保量ログ

**ファイル**: `src/audioengine/AudioEngine.Processing.DSPCoreLifecycle.cpp`
**位置**: `alignedL`, `alignedR` 確保後

```cpp
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
{
    const auto osMem = getProcessMemoryInfo();
    diagLog(juce::String::formatted(
        "[MEM_BUF] DSPCore#%llu alignedL/R: %d doubles = %.1f MB | "
        "dryBypass: %d doubles = %.1f MB | "
        "maxInternalBlock=%d | oversampling=%zu | "
        "live=%d | OS: Private=%lluMB WorkingSet=%lluMB",
        (unsigned long long)runtimeUuid,
        alignedCapacity, (alignedCapacity * sizeof(double) * 2) / (1024.0 * 1024.0),
        dryBypassCapacityDouble,
        (dryBypassCapacityDouble * sizeof(double) * 2) / (1024.0 * 1024.0),
        maxInternalBlockSize,
        oversamplingFactor,
        (int)liveCount.load(),
        (unsigned long long)osMem.privateUsageMB,
        (unsigned long long)osMem.workingSetMB));
}
#endif
```

---

## 5. Patch E: Publish 時フルスナップショット（OS メモリ含む）

**ファイル**: `src/audioengine/AudioEngine.Timer.cpp`
**位置**: `publishWorld()` 成功直後

```cpp
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
{
    const uint64_t nucAlloc    = convo::MKLNonUniformConvolver::globalAllocated.load();
    const uint64_t nucPeak     = convo::MKLNonUniformConvolver::peakAllocated.load();
    const int      nucLive     = (int)convo::MKLNonUniformConvolver::liveCount.load();
    const int      stereoLive  = (int)ConvolverProcessor::StereoConvolver::liveCount.load();
    const int      dspLive     = (int)AudioEngine::DSPCore::liveCount.load();
    const uint32_t pending     = m_retireRouter ? m_retireRouter->pendingRetireCount() : 0;
    const uint64_t pendingBytes= m_retireRouter ? m_retireRouter->pendingRetireBytes() : 0;
    const uint64_t reclaim     = m_retireRouter ? m_retireRouter->reclaimAttemptCount() : 0;
    const uint64_t overflow    = m_retireRouter ? m_retireRouter->overflowCount() : 0;
    const auto osMem = getProcessMemoryInfo();

    juce::Logger::writeToLog(juce::String::formatted(
        "[MEM_SNAP] PUBLISH gen=%d seq=%d | "
        "NUC: live=%d alloc=%.0fMB peak=%.0fMB | "
        "Stereo=%d DSPCore=%d | "
        "Retire: pending=%u(%.1fMB) overflow=%llu reclaim=%llu | "
        "OS: Private=%lluMB WorkingSet=%lluMB",
        gen, seq,
        nucLive,
        nucAlloc / (1024.0 * 1024.0), nucPeak / (1024.0 * 1024.0),
        stereoLive, dspLive,
        pending, pendingBytes / (1024.0 * 1024.0),
        (unsigned long long)overflow,
        (unsigned long long)reclaim,
        (unsigned long long)osMem.privateUsageMB,
        (unsigned long long)osMem.workingSetMB));
}
#endif
```

**★ 最重要ポイント**: `OS Private=%lluMB` と `NUC alloc=%.0fMB` を同時にログに出力することで、
「OS が 2.33GB と言っているのに、追跡できているのは 150MB しかない」というギャップが即座に分かる。

---

## 6. Patch F: イベント駆動ログ（Publish/Retire/IR Reload の 3 イベント中心）

定期ログ（5秒）は補助程度。主要なログイベントは以下の 3 つ:

### F-1. Publish 時ログ（上記 Patch E で実装済み）

### F-2. IR Reload 時ログ（SetImpulse 成功直後、Patch B-8 で実装済み）

### F-3. Retire イベント検出ログ

**ファイル**: `src/audioengine/AudioEngine.Retire.cpp` — `tryReclaim()` 成功時

```cpp
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
// ★ v4: Retire イベントログ（reclaim 成功時のみ）
{
    const uint32_t pending = m_retireRouter->pendingRetireCount();
    const uint64_t pendingBytes = m_retireRouter->pendingRetireBytes();
    if (pending > 50 || pendingBytes > 10 * 1024 * 1024)  // 50個以上 or 10MB超
    {
        const auto osMem = getProcessMemoryInfo();
        juce::Logger::writeToLog(juce::String::formatted(
            "[MEM_RETIRE] pending=%u(%.1fMB) reclaim=%llu | "
            "OS: Private=%lluMB WorkingSet=%lluMB",
            pending, pendingBytes / (1024.0 * 1024.0),
            (unsigned long long)m_retireRouter->reclaimAttemptCount(),
            (unsigned long long)osMem.privateUsageMB,
            (unsigned long long)osMem.workingSetMB));
    }
}
#endif
```

### F-4. 5秒定期ログ（変化時のみ、優先度低）

```cpp
static juce::uint32 lastMemLog = 0;
const juce::uint32 nowMs = juce::Time::getApproximateMillisecondTimer();
if (nowMs - lastMemLog > 5000) {
    lastMemLog = nowMs;
    static uint64_t lastAlloc = 0;
    const uint64_t curAlloc = convo::MKLNonUniformConvolver::globalAllocated.load();
    if (curAlloc == lastAlloc) return;
    lastAlloc = curAlloc;
    const auto osMem = getProcessMemoryInfo();
    juce::Logger::writeToLog(juce::String::formatted(
        "[MEM] NUC=%d alloc=%.0fMB peak=%.0fMB | Retire pending=%u | OS: Private=%lluMB",
        (int)convo::MKLNonUniformConvolver::liveCount.load(),
        curAlloc / (1024.0 * 1024.0),
        convo::MKLNonUniformConvolver::peakAllocated.load() / (1024.0 * 1024.0),
        m_retireRouter ? m_retireRouter->pendingRetireCount() : 0,
        (unsigned long long)osMem.privateUsageMB));
}
```

---

## 7. 出力例（v4）

### 正常時

```text
[NUC_MEM] NUC#0000001234 L0:30MB L1:0MB L2:0MB | globalAlloc=62MB peak=142MB live=2 | OS: Private=68MB WorkingSet=120MB
[MEM_BUF] DSPCore#1 alignedL/R: 65536 doubles = 1.0 MB | dryBypass: 32768 doubles = 0.5 MB | maxInternalBlock=4096 | oversampling=8 | live=1 | OS: Private=68MB WorkingSet=120MB
[MEM_SNAP] PUBLISH gen=8 seq=5 | NUC: live=2 alloc=62MB peak=142MB | Stereo=1 DSPCore=1 | Retire: pending=0(0.0MB) overflow=0 reclaim=47 | OS: Private=68MB WorkingSet=120MB
```

### 異常検出例

```text
[NUC_MEM] NUC#0000001234 L0:30MB L1:200MB L2:400MB | globalAlloc=630MB peak=640MB live=8 | OS: Private=2330MB WorkingSet=2400MB
[MEM_SNAP] PUBLISH gen=8 | NUC: live=8 alloc=630MB peak=640MB | Stereo=4 DSPCore=4 | Retire: pending=232(180.5MB) overflow=0 reclaim=12 | OS: Private=2330MB WorkingSet=2400MB
```

**解釈**: NUC=630MB + Retire=180MB = 810MB 追跡済み。OS Private=2330MB → **約 1.5GB が未計測領域**。

---

## 8. 診断フロー（v4）

| globalAlloc | liveCount | pendingRetire | OS Private | 診断 |
|:------------|:----------|:--------------|:-----------|:-----|
| ~60MB | 2 | 0 | ~70MB | **正常。2.33GB は NUC 外部** → DSPCore/ProcessingBuffer 調査 |
| ~600MB | 8 | 0 | ~610MB | **NUC リーク** → StereoConvolver の退役不備 |
| ~60MB | 2 | ≥100 | ~600MB | **RetireQueue 滞留** → ISR EpochDomain の reclaim 不備 |
| ~60MB | 2 | 0 | 2330MB | **NUC 外の未計測領域 2.17GB** → EQ/ProcessingBuffer/OS固有調査 |
| peak ≫ current | — | — | — | **一瞬のピークが原因** → ProgressiveUpgrade の中間生成調査 |

---

## 9. 実装コスト見積もり（v4）

| Patch | 変更ファイル数 | 追加行数 | 備考 |
|:------|:-------------|:---------|:-----|
| A: DiagnosticsConfig.h | 1 | ~40行 | diagMklMalloc/Free + uint64_t peak ヘルパー |
| B: MKLNonUniformConvolver 3階層カウンタ + 差分追跡 | 2 | ~45行 | layerAllocated 廃止、DIAG_MKL_MALLOC 置換 ×25 |
| C: ISRRetireRouter pendingRetireBytes | 3 | ~15行 | .h + .cpp + IEpochProvider |
| D: DSPCore liveCount + 確保量ログ | 2 | ~18行 | |
| E: Publish フルスナップショット | 1 | ~20行 | OS PrivateUsage 含む |
| F: イベント駆動ログ | 2 | ~20行 | Retire イベント + 5秒ログ |
| **合計** | **6〜8ファイル** | **~158行** | |

---

## 10. v3 からの改善点一覧

| 項目 | v3（問題） | v4（修正） |
|:-----|:----------|:----------|
| 減算方式 | `layerAllocated` → `freeAll()` で `globalAllocated.fetch_sub` | **SetImpulse 開始/終了の差分で一括更新** |
| 再構築中の一時値 | `globalAllocated` が 0 になるリスク | **差分方式のため発生しない** |
| `thisAllocated` 減算 | 増える一方（デストラクタで一括） | **不要（SetImpulse 差分で管理）** |
| `Layer::layerAllocated` | +8byte/Layer | **廃止（freeAllBytes のみ）** |
| `DIAG_ALLOC` マクロ | 28箇所に手動挿入 | **`diagMklMalloc()` ラッパーで自動追跡** |
| `mkl_free` サイズ特定 | Layer に保存が必要 | **呼び出し元がサイズを渡す（既知）** |
| peak 更新 | CAS ループ（手動） | **`updateAtomicMaximum64` ヘルパー** |
| Retire bytes | なし | **`pendingRetireBytes()` 追加** |
| OS Private Usage | なし | **`getProcessMemoryInfo()` を全ログに含む** |
| DSPCore バッファ | liveCount のみ | **alignedL/R 容量 + dryBypass 容量も計測** |
| Publish ログ | NUC のみ | **OS Private/WorkingSet 含む** |
| ログ優先度 | 5秒定期を重視 | **Publish/Retire/IR Reload の 3 イベント中心** |
