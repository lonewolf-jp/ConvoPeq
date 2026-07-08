# メモリ占有調査のためのインストルメンテーション改修案 v5 — Single Source of Truth + Retire Bytes

---

**目的**: 2.33GB メモリ占有の原因特定。推定ではなく実測で確定させる。
**v4 からの変更**: reviewer フィードバックに基づく設計再構築（MKL計測の二重管理解消 + RetireEntry サイズ取得方式の確定）。

---

## 0. v4 の設計問題と v5 での修正方針

### ★ 二重加算問題（最大の問題）

v4 では `diagMklMalloc()` が `allocatedBytes` を更新しつつ、SetImpulse 差分方式で
`globalAllocated` も更新する。**同じ情報を 2 か所で加算**するため、値が 2 倍になる。

```
diagMklMalloc  → allocatedBytes += size   (自動更新)
SetImpulse差分  → globalAllocated += delta (手動更新)
```

**→ 二重加算。v5 では SetImpulse 差分方式を完全撤廃。**

### ★ 二重管理問題

`convo::diag::MklAllocStats::allocatedBytes` と
`MKLNonUniformConvolver::globalAllocated` が同一情報を保持。
**Single Source of Truth が 2 つ** → 状態不整合のリスク。

**→ `globalAllocated`/`peakAllocated` を NUC から削除。唯一の集計元は `mklStats()`。**

### ★ freeAllBytes では DIAG_MKL_FREE ができない

各 Layer のバッファはサイズが異なる（irBufSize, irSoaSize, fdlBufSize, complexSize, partStride…）。
`freeAllBytes` 1 つでは全バッファの解放サイズを特定できない。

**→ `diagMklFree` 内部でポインタ→サイズのマップを保持。呼び出し元はサイズ不要。**

### ★ pendingRetireBytes: DeletionEntry に size フィールド追加が必要

`DeletionEntry` は ptr/deleter/epoch/type のみ。**サイズ情報がない**。
`DeferredRetireFallbackEntry` には `estimatedSize` があるが、メインキューにはない。

**→ `DeletionEntry` に `size_t estimatedSize` フィールドを追加。**

---

## 1. Patch A: DiagnosticsConfig.h — 唯一の MKL 計測ソース

**ファイル**: `src/DiagnosticsConfig.h`

### A-1. MklAllocStats（改修版）

```cpp
namespace convo::diag {

/// MKL 分配の Single Source of Truth。
/// 全 NUC インスタンス・全スレッドから使用。
/// ロック: std::mutex（SetImpulse 時のみアクセス、Audio Thread はアクセスしない）。
struct MklAllocStats {
    std::atomic<uint64_t> allocatedBytes { 0 };  // 現在使用量
    std::atomic<uint64_t> peakBytes      { 0 };  // ピーク使用量

    // 内部用: ptr → size マップ（diagMklFree でのサイズ特定用）
    std::mutex           mapMutex;
    std::unordered_map<void*, size_t> allocationMap;
};

inline MklAllocStats& mklStats() noexcept
{
    static MklAllocStats stats{};
    return stats;
}

/// mkl_malloc のラッパー。成功時に allocatedBytes を増加し、ptr→size を記録する。
inline void* diagMklMalloc(size_t size, int alignment) noexcept
{
    void* ptr = mkl_malloc(size, alignment);
    if (ptr)
    {
        const uint64_t bytes = static_cast<uint64_t>(size);
        mklStats().allocatedBytes.fetch_add(bytes, std::memory_order_relaxed);

        // peak 更新
        uint64_t cur = mklStats().allocatedBytes.load(std::memory_order_relaxed);
        uint64_t pk  = mklStats().peakBytes.load(std::memory_order_relaxed);
        while (cur > pk && !mklStats().peakBytes.compare_exchange_weak(
            pk, cur, std::memory_order_relaxed)) {}

        // ptr → size 記録（SetImpulse 時のみ、ロック不要な単発操作）
        {
            std::lock_guard<std::mutex> lock(mklStats().mapMutex);
            mklStats().allocationMap[ptr] = size;
        }
    }
    return ptr;
}

/// mkl_free のラッパー。内部マップからサイズを取得し、allocatedBytes を減少させる。
/// ★ 呼び出し元はサイズを渡す必要がない（マップから自動取得）。
inline void diagMklFree(void* ptr) noexcept
{
    if (!ptr) return;

    size_t size = 0;
    {
        std::lock_guard<std::mutex> lock(mklStats().mapMutex);
        auto it = mklStats().allocationMap.find(ptr);
        if (it != mklStats().allocationMap.end())
        {
            size = it->second;
            mklStats().allocationMap.erase(it);
        }
    }

    mkl_free(ptr);

    if (size > 0)
        mklStats().allocatedBytes.fetch_sub(static_cast<uint64_t>(size),
                                             std::memory_order_relaxed);
}

/// 現在の MKL 使用量（MB）
inline double currentMklAllocMB() noexcept
{
    return mklStats().allocatedBytes.load(std::memory_order_relaxed) / (1024.0 * 1024.0);
}

/// ピーク MKL 使用量（MB）
inline double peakMklAllocMB() noexcept
{
    return mklStats().peakBytes.load(std::memory_order_relaxed) / (1024.0 * 1024.0);
}

} // namespace convo::diag
```

### A-2. uint64_t 版 updateAtomicMaximum64

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
  #define DIAG_MKL_FREE(ptr)           convo::diag::diagMklFree((ptr))
#else
  #define DIAG_MKL_MALLOC(size, align) mkl_malloc((size), (align))
  #define DIAG_MKL_FREE(ptr)           mkl_free(ptr)
#endif
```

**★ diagMklFree はサイズ不要** — 内部マップから自動取得。呼び出し元の保守コストゼロ。

---

## 2. Patch B: MKLNonUniformConvolver — globalAllocated 廃止 + DIAG_MKL_MALLOC 置換

### B-1. MKLNonUniformConvolver.h — static カウンタの完全撤廃

```cpp
class MKLNonUniformConvolver {
public:
    // ★ v5: globalAllocated / peakAllocated / liveCount は全て廃止。
    // MKL 計測は convo::diag::mklStats() に一元化（Single Source of Truth）。
    // liveCount は Publish ログ用に残す（メモリ計測ではなく生存数のみ）。
    static std::atomic<int> liveCount;
    // ... 既存のメンバ（変更なし） ...
```

**★ Layer 構造体には一切の変更不要**（layerAllocated も freeAllBytes も追加しない）。

### B-2. MKLNonUniformConvolver.cpp — static 変数

```cpp
std::atomic<int> MKLNonUniformConvolver::liveCount { 0 };
// ★ globalAllocated / peakAllocated は削除
```

### B-3. 全 mkl_malloc → DIAG_MKL_MALLOC 置換

SetImpulse 内の全 25 箇所 + releaseAllLayers 内の全 mkl_free を置換:

```cpp
// ★ Before:
l.irFreqDomain = static_cast<double*>(mkl_malloc(irBufSize * sizeof(double), 64));

// ★ After:
l.irFreqDomain = static_cast<double*>(DIAG_MKL_MALLOC(irBufSize * sizeof(double), 64));
```

```cpp
// ★ Before:
if (irFreqDomain) { mkl_free(irFreqDomain); irFreqDomain = nullptr; }

// ★ After:
if (irFreqDomain) { DIAG_MKL_FREE(irFreqDomain); irFreqDomain = nullptr; }
```

**★ diagMklFree はサイズパラメータ不要** — 内部マップから自動取得。

### B-4. SetImpulse — 差分追跡の完全撤廃

```cpp
bool MKLNonUniformConvolver::SetImpulse(...)
{
    // ★ v5: allocBefore/allocAfter 差分追跡は不要。
    //   → diagMklMalloc/diagMklFree が自動的に allocatedBytes を管理。
    //   → SetImpulse 終了時に mklStats().allocatedBytes が「NUC の現在使用量」。

    convo::publishAtomic(m_ready, false, std::memory_order_release);
    if (impulse == nullptr || irLen <= 0 || blockSize <= 0) return false;

    releaseAllLayers();

    // ... 既存のレイヤー構成決定・確保ロジック（ DIAG_MKL_MALLOC に変更済み） ...

    // ... applySpectrumFilter など ...

    convo::publishAtomic(m_ready, true, std::memory_order_release);
    return true;
}
```

### B-5. デストラクタ

```cpp
MKLNonUniformConvolver::~MKLNonUniformConvolver()
{
    liveCount--;
    releaseAllLayers();
    // ★ v5: globalAllocated の減算は不要（mklStats が自動管理）。
}
```

### B-6. NUC 合計ログ（SetImpulse 成功直後）

```cpp
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
{
    const auto osMem = getProcessMemoryInfo();
    const double nucMB = convo::diag::currentMklAllocMB();
    const double nucPeakMB = convo::diag::peakMklAllocMB();
    const double retireMB = /* pendingRetireBytes() / (1024.0*1024.0) */;
    const double unknownMB = std::max(0.0,
        (double)osMem.privateUsageMB - nucMB - retireMB);

    diagLog(juce::String::formatted(
        "[NUC_MEM] NUC#%p L0:%dMB L1:%dMB L2:%dMB | "
        "MKL: alloc=%.0fMB peak=%.0fMB live=%d | "
        "OS: Private=%lluMB WorkingSet=%lluMB | "
        "Unknown=%.0fMB",
        (void*)this,
        /* 各 Layer の解放サイズ（freeAll 後なので 0） */,
        nucMB, nucPeakMB, (int)liveCount.load(),
        (unsigned long long)osMem.privateUsageMB,
        (unsigned long long)osMem.workingSetMB,
        unknownMB));
}
#endif
```

**★ Unknown = OS Private - NUC - Retire で「未計測領域」が即座に分かる。**

---

## 3. Patch C: ISRRetireRouter — pendingRetireBytes 追加

### C-1. DeletionEntry に size フィールド追加

**ファイル**: `src/DeferredDeletionQueue.h`

```cpp
struct DeletionEntry {
    void* ptr = nullptr;
    void (*deleter)(void*) = nullptr;
    uint64_t epoch = 0;
    DeletionEntryType type = DeletionEntryType::Generic;
    uint64_t publicationSequenceId{0};
    uint64_t generation{0};
    size_t estimatedSize{0};  // ★ v5: Retire 対象の推定バイト数
};
static_assert(std::is_trivially_copyable_v<DeletionEntry>,
    "DeletionEntry must be trivially copyable");
// ★ estimatedSize は size_t (POD) のため、trivially_copyable 保持。
```

**★ 注意**: `DeletionEntry` は既に `void*` と `function pointer` を含むため
実質的にプラットフォーム依存だが、MSVC/x64 では trivially copyable を維持。

### C-2. enqueueRetire の size 受け取り型オーバーロード追加

**ファイル**: `src/audioengine/ISRRetireRouter.h`

```cpp
    // ★ v5: size パラメータ付きオーバーロード
    RetireEnqueueResult enqueueRetire(void* ptr,
                                      void (*deleter)(void*),
                                      uint64_t epoch,
                                      DeletionEntryType type,
                                      size_t estimatedSize) noexcept;
```

**ファイル**: `src/audioengine/ISRRetireRouter.cpp`

```cpp
RetireEnqueueResult ISRRetireRouter::enqueueRetire(void* ptr,
                                                    void (*deleter)(void*),
                                                    uint64_t epoch,
                                                    DeletionEntryType type,
                                                    size_t estimatedSize) noexcept
{
    assert(provider_ != nullptr);
    if (ptr == nullptr || deleter == nullptr)
        return RetireEnqueueResult::Success;

    // ★ v5: メインキューへ size 付きで enqueue
    if (provider_->enqueueRetire(ptr, deleter, epoch, type, estimatedSize))
    {
        m_pendingRetireBytes_.fetch_add(estimatedSize, std::memory_order_relaxed);
        return RetireEnqueueResult::Success;
    }
    // フォールバック
    // ...
}
```

**★ 既存の 4 パラメータ版は size=0 で委譲**:

```cpp
RetireEnqueueResult ISRRetireRouter::enqueueRetire(void* ptr,
                                                    void (*deleter)(void*),
                                                    uint64_t epoch,
                                                    DeletionEntryType type) noexcept
{
    return enqueueRetire(ptr, deleter, epoch, type, 0);
}
```

### C-3. IEpochProvider / EpochDomain への size パラメータ伝播

**ファイル**: `src/core/IEpochProvider.h`

```cpp
    virtual bool enqueueRetire(void* ptr, void (*deleter)(void*),
                               uint64_t epoch, DeletionEntryType type,
                               size_t estimatedSize = 0) noexcept = 0;
```

**ファイル**: `src/core/EpochDomain.h` — `enqueueRetire` 実装で `entry.estimatedSize = estimatedSize;`

### C-4. ISRRetireRouter — m_pendingRetireBytes_ メンバ

```cpp
class ISRRetireRouter : public convo::IEpochProvider {
    // ... 既存メンバ ...
    std::atomic<uint64_t> m_pendingRetireBytes_ { 0 };
};
```

### C-5. tryReclaim / drainAll でのデクリメント

```cpp
void ISRRetireRouter::tryReclaim() noexcept
{
    // reclaim 成功時にエントリの estimatedSize を取得してデクリメント
    // 具体的には EpochDomain::tryReclaim が返す reclaimed エントリから size を読む
}
```

### C-6. 呼び出し元への size 引渡し

| 呼び出し元 | 対象 | estimatedSize |
|:----------|:-----|:-------------|
| AudioEngine.h L3751 | Generic deleter | `0`（不明） |
| DSPLifetimeManager.h L45 | DSPCore | `sizeof(AudioEngine::DSPCore)` |
| RuntimePublicationCoordinator | StereoConvolver | `sizeof(ConvolverProcessor::StereoConvolver)` |
| SnapshotRetireManager | RuntimeWorld | `sizeof(RuntimeWorld)` |

**★ 既存呼び出しの多くは size=0 のまま。** 少しずつ size を追加していく。
0 の場合でも `pendingRetireBytes` はデクリメントされないため、
正確なバイト数は得られないが、**count と併せて傾向は把握可能**。

### C-7. ISRRetireRouter の公開 API

```cpp
    [[nodiscard]] uint64_t pendingRetireBytes() const noexcept {
        return convo::consumeAtomic(m_pendingRetireBytes_, std::memory_order_acquire);
    }
```

---

## 4. Patch D: DSPCore 処理バッファ計測（変更なし）

v4 の内容をそのまま採用:
- DSPCore liveCount
- alignedL/R 容量
- dryBypass 容量
- OS PrivateUsage 併記

---

## 5. Patch E: Publish フルスナップショット（Unknown メモリ追加）

```cpp
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
{
    const double nucMB     = convo::diag::currentMklAllocMB();
    const double nucPeakMB = convo::diag::peakMklAllocMB();
    const int    nucLive   = (int)MKLNonUniformConvolver::liveCount.load();
    const int    stereoLive= (int)ConvolverProcessor::StereoConvolver::liveCount.load();
    const int    dspLive   = (int)AudioEngine::DSPCore::liveCount.load();
    const uint32_t pending = m_retireRouter ? m_retireRouter->pendingRetireCount() : 0;
    const double retireMB  = m_retireRouter
        ? m_retireRouter->pendingRetireBytes() / (1024.0 * 1024.0) : 0.0;
    const uint64_t reclaim = m_retireRouter ? m_retireRouter->reclaimAttemptCount() : 0;
    const uint64_t overflow= m_retireRouter ? m_retireRouter->overflowCount() : 0;
    const auto osMem = getProcessMemoryInfo();
    const double unknownMB = std::max(0.0,
        (double)osMem.privateUsageMB - nucMB - retireMB);

    juce::Logger::writeToLog(juce::String::formatted(
        "[MEM_SNAP] PUBLISH gen=%d seq=%d | "
        "NUC: live=%d alloc=%.0fMB peak=%.0fMB | "
        "Stereo=%d DSPCore=%d | "
        "Retire: pending=%u (%.1fMB) overflow=%llu reclaim=%llu | "
        "OS: Private=%lluMB WorkingSet=%lluMB | "
        "Unknown=%.0fMB",
        gen, seq,
        nucLive, nucMB, nucPeakMB,
        stereoLive, dspLive,
        pending, retireMB,
        (unsigned long long)overflow, (unsigned long long)reclaim,
        (unsigned long long)osMem.privateUsageMB,
        (unsigned long long)osMem.workingSetMB,
        unknownMB));
}
#endif
```

---

## 6. Patch F: イベント駆動ログ（v4 から変更なし）

- Publish 時（Patch E）
- IR Reload 時（B-6）
- Retire イベント検出時
- 5秒定期ログ（変化時のみ、補助）

---

## 7. 出力例（v5）

### 正常時

```text
[NUC_MEM] NUC#0000001234 L0:30MB L1:0MB L2:0MB | MKL: alloc=62MB peak=142MB live=2 | OS: Private=68MB WorkingSet=120MB | Unknown=6MB
[MEM_SNAP] PUBLISH gen=8 seq=5 | NUC: live=2 alloc=62MB peak=142MB | Stereo=1 DSPCore=1 | Retire: pending=0 (0.0MB) overflow=0 reclaim=47 | OS: Private=68MB WorkingSet=120MB | Unknown=6MB
```

### 異常検出例

```text
[NUC_MEM] NUC#0000001234 L0:30MB L1:200MB L2:400MB | MKL: alloc=630MB peak=640MB live=8 | OS: Private=2330MB WorkingSet=2400MB | Unknown=1520MB
[MEM_SNAP] PUBLISH gen=8 | NUC: live=8 alloc=630MB peak=640MB | Stereo=4 DSPCore=4 | Retire: pending=232 (180.5MB) overflow=0 reclaim=12 | OS: Private=2330MB WorkingSet=2400MB | Unknown=1520MB
```

**解釈**: NUC=630MB + Retire=180MB = 810MB 追跡済み。OS Private=2330MB → **Unknown=1520MB が未計測領域**。

---

## 8. 診断フロー（v5）

| MKL alloc | liveCount | pendingRetire | OS Private | Unknown | 診断 |
|:----------|:----------|:--------------|:-----------|:--------|:-----|
| ~60MB | 2 | 0MB | ~70MB | ~10MB | **正常。2.33GB は NUC 外部** |
| ~600MB | 8 | 0MB | ~610MB | ~10MB | **NUC リーク** |
| ~60MB | 2 | >100MB | ~600MB | ~440MB | **RetireQueue 滞留** |
| ~60MB | 2 | 0MB | 2330MB | **2270MB** | **NUC 外の未計測領域** → EQ/ProcessingBuffer/OS固有調査 |

---

## 9. 実装コスト見積もり（v5）

| Patch | 変更ファイル数 | 追加行数 | 備考 |
|:------|:-------------|:---------|:-----|
| A: DiagnosticsConfig.h | 1 | ~55行 | diagMklMalloc/Free + マップ管理 + ヘルパー |
| B: MKLNonUniformConvolver | 2 | ~35行 | globalAllocated 廃止 + DIAG_MKL_MALLOC 置換 |
| C: ISRRetireRouter + DeletionEntry | 5 | ~40行 | estimatedSize 追加 + pendingRetireBytes |
| D: DSPCore | 2 | ~18行 | v4 から変更なし |
| E: Publish スナップショット | 1 | ~25行 | Unknown メモリ追加 |
| F: イベント駆動ログ | 2 | ~20行 | v4 から変更なし |
| **合計** | **8〜10ファイル** | **~193行** | |

---

## 10. v4 からの改善点一覧

| 項目 | v4（問題） | v5（修正） |
|:-----|:----------|:----------|
| MKL計測の Single Source of Truth | `mklStats.allocatedBytes` + `globalAllocated` の二重管理 | **`mklStats()` のみ。`globalAllocated`/`peakAllocated` 廃止** |
| SetImpulse 差分追跡 | `diagMklMalloc` と二重加算 | **差分追跡完全撤廃。`diagMklMalloc/Free` のみ** |
| `diagMklFree(expectedSize)` | 呼び出し元がサイズを渡す必要あり | **内部マップから自動取得。呼び出し元はサイズ不要** |
| `Layer::freeAllBytes` | 全バッファの個別サイズを保存する必要あり | **不要。Layer への変更ゼロ** |
| allocCount/freeCount | 意味が薄い | **廃止（allocatedBytes のみ）** |
| pendingRetireBytes | DeletionEntry に size なし | **`DeletionEntry` に `estimatedSize` フィールド追加** |
| Unknown メモリ | OS と NUC を並列表示のみ | **`Unknown = OS - NUC - Retire` を計算・表示** |
| NUC static 変数 | 3つ（globalAllocated, peakAllocated, liveCount） | **1つ（liveCount のみ）** |
