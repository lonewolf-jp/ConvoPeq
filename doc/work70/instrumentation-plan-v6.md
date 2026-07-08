# メモリ占有調査のためのインストルメンテーション改修案 v6 — 軽量 DIAG_MKL_FREE + Untracked メモリ + カテゴリ別計測

---

**目的**: 2.33GB メモリ占有の原因特定。推定ではなく実測で確定させる。
**v5 からの変更**: reviewer フィードバック（9点）を全て反映。診断コードとしての完成度 95 点以上を目指す。

---

## 0. v5 の問題点と v6 での修正方針

| # | v5 の問題 | v6 の修正 | 根拠 |
|:--|:---------|:---------|:-----|
| 1 | `allocationMap` + `mutex` が重すぎる | **`DIAG_MKL_FREE(ptr, size)` に回帰** | 診断コードは最小限が原則。呼び出し元がサイズを知っているため不要 |
| 2 | `Unknown` という名称 | **`Untracked` に変更** | 「未追跡」が意味的に正確 |
| 3 | `estimatedSize` という名称 | **`retainedBytes` に変更** | 「保持バイト数」が意味的に明確 |
| 4 | `DeletionEntry` に無条件でフィールド追加 | **`#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` で条件付き** | Release ビルドのキューレイアウトを変更しない |
| 5 | `liveCount++` | **`liveCount.fetch_add(1, relaxed)`** | 一貫性 |
| 6 | `currentMklAllocMB()` | **`allocatedBytes()` / `peakBytes()` のみ公開** | API 汎用性。表示側で単位変換 |
| 7 | peak 更新が毎回 while ループ | **`updateAtomicMaximum64()` ヘルパー統一** | コード重複排除 |
| 8 | NUC_MEM に L0/L1/L2 表示 | **Layer 別表示を削除** | Layer サイズ情報を持たない設計と矛盾 |
| 9 | MKL のみ計測 | **将来的にカテゴリ拡張可能な構造を用意** | プロセス全体の内訳把握に必要 |

---

## 1. Patch A: DiagnosticsConfig.h — 軽量 MKL 計測 + ヘルパー

**ファイル**: `src/DiagnosticsConfig.h`

### A-1. MklAllocStats（allocationMap なし版）

```cpp
namespace convo::diag {

/// MKL 分配の Single Source of Truth。
/// allocationMap/mutex なし。呼び出し元が free 時にサイズを渡す。
struct MklAllocStats {
    std::atomic<uint64_t> allocatedBytes { 0 };
    std::atomic<uint64_t> peakBytes      { 0 };
};

inline MklAllocStats& mklStats() noexcept
{
    static MklAllocStats stats{};
    return stats;
}

/// mkl_malloc ラッパー。成功時に allocatedBytes を増加し、peak を更新。
inline void* diagMklMalloc(size_t size, int alignment) noexcept
{
    void* ptr = mkl_malloc(size, alignment);
    if (ptr)
    {
        mklStats().allocatedBytes.fetch_add(
            static_cast<uint64_t>(size), std::memory_order_relaxed);
        updateAtomicMaximum64(mklStats().peakBytes,
            mklStats().allocatedBytes.load(std::memory_order_relaxed));
    }
    return ptr;
}

/// mkl_free ラッパー。呼び出し元がサイズを渡す（マップ不要）。
inline void diagMklFree(void* ptr, size_t size) noexcept
{
    if (ptr)
    {
        mkl_free(ptr);
        mklStats().allocatedBytes.fetch_sub(
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

} // namespace convo::diag
```

**★ allocationMap / mutex / unordered_map は全て撤廃。**
呼び出し元（SetImpulse / freeAll）は各バッファのサイズを既に知っているため、
`DIAG_MKL_FREE(ptr, size)` で十分。ロックゼロ、追加メモリゼロ。

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
  #define DIAG_MKL_FREE(ptr, size)     convo::diag::diagMklFree((ptr), (size))
#else
  #define DIAG_MKL_MALLOC(size, align) mkl_malloc((size), (align))
  #define DIAG_MKL_FREE(ptr, size)     mkl_free(ptr)
#endif
```

### A-4. MemoryCategory（将来的カテゴリ拡張用）

```cpp
/// ★ v6: カテゴリ別メモリ集計（将来拡張用。今回の調査では MKL のみ計測）。
/// 最終ログでカテゴリ別内訳を出す基盤。
enum class MemoryCategory : uint8_t {
    MKL = 0,            ///< MKL non-uniform convolution buffers
    IPP,                ///< IPP FFT spec / work buffers
    JUCE,               ///< JUCE heap allocations
    ProcessingBuffer,   ///< DSPCore alignedL/R, dryBypass
    RuntimeWorld,       ///< RuntimeWorld instance + subtree
    ThreadStack,        ///< Thread stack (estimated)
    Other,              ///< その他
    Count
};

struct CategoryStats {
    std::atomic<uint64_t> bytes { 0 };
};

inline CategoryStats& categoryStats(MemoryCategory cat) noexcept
{
    static CategoryStats stats[static_cast<size_t>(MemoryCategory::Count)]{};
    return stats[static_cast<size_t>(cat)];
}
```

**★ 今回の調査では MKL のみ。カテゴリ拡張は将来の段階で。**
構造を用意しておくことで、将来的に DSPCore/IPP/RuntimeWorld 等を追加する際に
設計変更が不要になる。

---

## 2. Patch B: MKLNonUniformConvolver — globalAllocated 廃止 + DIAG_MKL 置換

### B-1. MKLNonUniformConvolver.h — static カウンタ

```cpp
class MKLNonUniformConvolver {
public:
    // ★ v6: MKL 計測は diag::mklStats() に一元化。
    // liveCount のみ残す（Publish ログ用）。
    static std::atomic<int> liveCount;
    // ... 既存のメンバ（変更なし） ...
```

Layer 構造体には**一切の変更不要**。

### B-2. MKLNonUniformConvolver.cpp — static 変数

```cpp
std::atomic<int> MKLNonUniformConvolver::liveCount { 0 };
// ★ globalAllocated / peakAllocated は削除済み
```

### B-3. 全 mkl_malloc → DIAG_MKL_MALLOC 置換（28箇所）

```cpp
// ★ Before:
l.irFreqDomain = static_cast<double*>(mkl_malloc(irBufSize * sizeof(double), 64));

// ★ After:
l.irFreqDomain = static_cast<double*>(DIAG_MKL_MALLOC(irBufSize * sizeof(double), 64));
```

### B-4. 全 mkl_free → DIAG_MKL_FREE 置換（26箇所）

```cpp
// ★ Before:
if (irFreqDomain) { mkl_free(irFreqDomain); irFreqDomain = nullptr; }

// ★ After:
if (irFreqDomain) { DIAG_MKL_FREE(irFreqDomain, irBufSize * sizeof(double)); irFreqDomain = nullptr; }
```

**★ freeAll() 内の各バッファサイズ一覧**:

| バッファ | freeAll() での解放サイズ |
|:---------|:----------------------|
| irFreqDomain | `irBufSize * sizeof(double)` |
| irFreqReal | `irSoaSize * sizeof(double)` |
| irFreqImag | `irSoaSize * sizeof(double)` |
| fdlBuf | `fdlBufSize * sizeof(double)` |
| fdlReal | `fdlSoaSize * sizeof(double)` |
| fdlImag | `fdlSoaSize * sizeof(double)` |
| fftTimeBuf | `fftSize * sizeof(double)` |
| fftOutBuf | `fftSize * sizeof(double)` |
| prevInputBuf | `partSize * sizeof(double)` |
| accumBuf | `partStride * sizeof(double)` |
| accumReal | `complexSize * sizeof(double)` |
| accumImag | `complexSize * sizeof(double)` |
| inputAccBuf | `partSize * sizeof(double)` |
| tailOutputBuf | `partSize * sizeof(double)` |

**★ 注意**: freeAll() 内の `DIAG_MKL_FREE` には上記のサイズ式が必要。
`freeAll()` は `releaseAllLayers()` 内の `for` ループから呼ばれるため、
サイズは `Layer` のメンバ（`irBufSize` 等）から直接参照可能。
ただし、`irBufSize` は SetImpulse 内のローカル変数であるため、
freeAll() から直接は参照できない。

**解決策**: `freeAll()` は `DIAG_MKL_FREE` を使わず、
`releaseAllLayers()` 側でサイズを計算して `DIAG_MKL_FREE` を呼ぶ：

```cpp
void MKLNonUniformConvolver::releaseAllLayers() noexcept
{
    for (int i = 0; i < kNumLayers; ++i)
    {
        Layer& l = m_layers[i];
        if (l.irFreqDomain) {
            const size_t irBufSize  = static_cast<size_t>(l.partStride);
            const size_t irSoaSize  = static_cast<size_t>(l.numParts) * static_cast<size_t>(l.complexSize);
            const size_t fdlBufSize = static_cast<size_t>(l.partStride) * 2;
            const size_t fdlSoaSize = static_cast<size_t>(l.numParts) * 2 * static_cast<size_t>(l.complexSize);

            DIAG_MKL_FREE(l.irFreqDomain,  irBufSize  * sizeof(double));
            DIAG_MKL_FREE(l.irFreqReal,    irSoaSize  * sizeof(double));
            DIAG_MKL_FREE(l.irFreqImag,    irSoaSize  * sizeof(double));
            DIAG_MKL_FREE(l.fdlBuf,        fdlBufSize * sizeof(double));
            DIAG_MKL_FREE(l.fdlReal,       fdlSoaSize * sizeof(double));
            DIAG_MKL_FREE(l.fdlImag,       fdlSoaSize * sizeof(double));
            DIAG_MKL_FREE(l.fftTimeBuf,    l.fftSize   * sizeof(double));
            DIAG_MKL_FREE(l.fftOutBuf,     l.fftSize   * sizeof(double));
            DIAG_MKL_FREE(l.prevInputBuf,  l.partSize  * sizeof(double));
            DIAG_MKL_FREE(l.accumBuf,      l.partStride * sizeof(double));
            DIAG_MKL_FREE(l.accumReal,     l.complexSize * sizeof(double));
            DIAG_MKL_FREE(l.accumImag,     l.complexSize * sizeof(double));
            DIAG_MKL_FREE(l.inputAccBuf,   l.partSize  * sizeof(double));
            DIAG_MKL_FREE(l.tailOutputBuf, l.partSize  * sizeof(double));
            l.irFreqDomain = l.irFreqReal = l.irFreqImag = nullptr;
            l.fdlBuf = l.fdlReal = l.fdlImag = nullptr;
            l.fftTimeBuf = l.fftOutBuf = l.prevInputBuf = nullptr;
            l.accumBuf = l.accumReal = l.accumImag = nullptr;
            l.inputAccBuf = l.tailOutputBuf = nullptr;
        }
        // ... 既存の状態リセット ...
    }
    // ... 既存の m_ringBuf, m_direct* 解放 ...
}
```

**★ freeAll() 自体は `DIAG_MKL_FREE` を呼ばず、ポインタを nullptr にするだけ。
サイズ計算と `DIAG_MKL_FREE` は `releaseAllLayers()` 側で実行。**
これにより、`Layer` にサイズフィールドを追加する必要がない。

### B-5. SetImpulse — 差分追跡の完全撤廃

```cpp
bool MKLNonUniformConvolver::SetImpulse(...)
{
    // ★ v6: allocBefore/allocAfter 差分追跡は不要。
    //   diagMklMalloc/diagMklFree が自動的に allocatedBytes を管理。

    convo::publishAtomic(m_ready, false, std::memory_order_release);
    if (impulse == nullptr || irLen <= 0 || blockSize <= 0) return false;

    releaseAllLayers();
    // ... 既存の確保ロジック（ DIAG_MKL_MALLOC に変更済み） ...
    convo::publishAtomic(m_ready, true, std::memory_order_release);
    return true;
}
```

### B-6. デストラクタ

```cpp
MKLNonUniformConvolver::~MKLNonUniformConvolver()
{
    liveCount.fetch_sub(1, std::memory_order_relaxed);
    releaseAllLayers();
}
```

### B-7. NUC_MEM ログ（Layer 別表示を削除）

```cpp
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
{
    const uint64_t nucBytes = convo::diag::allocatedBytes();
    const uint64_t nucPeak  = convo::diag::peakBytes();
    const auto osMem = getProcessMemoryInfo();
    const uint64_t retireBytes = /* pendingRetireBytes() */;
    const int64_t untracked = (int64_t)osMem.privateUsageMB * 1024 * 1024
                            - (int64_t)nucBytes - (int64_t)retireBytes;
    const double untrackedMB = std::max(0LL, untracked) / (1024.0 * 1024.0);

    diagLog(juce::String::formatted(
        "[NUC_MEM] NUC#%p | MKL: alloc=%.0fMB peak=%.0fMB live=%d | "
        "OS: Private=%lluMB WorkingSet=%lluMB | "
        "Untracked=%.0fMB",
        (void*)this,
        nucBytes / (1024.0 * 1024.0), nucPeak / (1024.0 * 1024.0),
        (int)liveCount.load(),
        (unsigned long long)osMem.privateUsageMB,
        (unsigned long long)osMem.workingSetMB,
        untrackedMB));
}
#endif
```

**★ L0/L1/L2 の Layer 別表示は削除**（Layer サイズ情報を保持しない設計）。
`MKL: alloc=62MB` のみ。Untracked で未計測領域を表示。

---

## 3. Patch C: ISRRetireRouter — pendingRetireBytes + retainedBytes

### C-1. DeletionEntry に retainedBytes フィールド追加（条件付き）

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
    size_t retainedBytes{0};  // ★ v6: Retire 対象の保持バイト数（診断のみ）
#endif
};
```

**★ Release ビルドでは `retainedBytes` は存在しない。**
キューのレイアウトは Release で変更されない。
`static_assert(trivially_copyable)` も `size_t` は POD のため問題なし。

### C-2. ISRRetireRouter — retainedBytes 引き渡し

**ファイル**: `src/audioengine/ISRRetireRouter.h`

```cpp
class ISRRetireRouter : public convo::IEpochProvider {
    // ... 既存メンバ ...
    std::atomic<uint64_t> m_pendingRetireBytes_ { 0 };

    // ★ v6: retainedBytes 付きオーバーロード
    RetireEnqueueResult enqueueRetire(void* ptr,
                                      void (*deleter)(void*),
                                      uint64_t epoch,
                                      DeletionEntryType type,
                                      size_t retainedBytes) noexcept;

    [[nodiscard]] uint64_t pendingRetireBytes() const noexcept {
        return convo::consumeAtomic(m_pendingRetireBytes_, std::memory_order_acquire);
    }
};
```

### C-3. enqueueRetire 実装（retainedBytes 付き）

```cpp
RetireEnqueueResult ISRRetireRouter::enqueueRetire(void* ptr,
                                                    void (*deleter)(void*),
                                                    uint64_t epoch,
                                                    DeletionEntryType type,
                                                    size_t retainedBytes) noexcept
{
    assert(provider_ != nullptr);
    if (ptr == nullptr || deleter == nullptr)
        return RetireEnqueueResult::Success;

    if (provider_->enqueueRetire(ptr, deleter, epoch, type, retainedBytes))
    {
        m_pendingRetireBytes_.fetch_add(retainedBytes, std::memory_order_relaxed);
        return RetireEnqueueResult::Success;
    }
    // フォールバック...
}

// 既存の 4 パラメータ版は retainedBytes=0 で委譲
RetireEnqueueResult ISRRetireRouter::enqueueRetire(void* ptr,
                                                    void (*deleter)(void*),
                                                    uint64_t epoch,
                                                    DeletionEntryType type) noexcept
{
    return enqueueRetire(ptr, deleter, epoch, type, 0);
}
```

### C-4. tryReclaim でのデクリメント

reclaim 成功時にエントリの `retainedBytes` を取得してデクリメント。
**注意**: `DeletionEntry` は `#if` 付きなので、Release では `retainedBytes` が存在しない。
`tryReclaim` 内でも `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` で分岐。

### C-5. 呼び出し元での retainedBytes 引渡し

| 呼び出し元 | 対象 | retainedBytes |
|:----------|:-----|:-------------|
| AudioEngine.h L3751 | Generic deleter | `0`（不明） |
| DSPLifetimeManager.h L45 | DSPCore | `sizeof(DSPCore)` |
| RuntimePublicationCoordinator | StereoConvolver | `sizeof(StereoConvolver)` |

**★ 既存呼び出しの多くは `retainedBytes=0` のまま。**
少しずつ `retainedBytes` を追加していく。0 の場合でも `pendingRetireBytes` は
デクリメントされないが、**count と併せて傾向は把握可能**。

---

## 4. Patch D: DSPCore 処理バッファ計測（v5 から変更なし）

- DSPCore liveCount（`fetch_add/fetch_sub` 使用）
- alignedL/R 容量
- dryBypass 容量
- OS PrivateUsage 併記

---

## 5. Patch E: Publish フルスナップショット

```cpp
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
{
    const uint64_t nucBytes   = convo::diag::allocatedBytes();
    const uint64_t nucPeak    = convo::diag::peakBytes();
    const int      nucLive    = (int)MKLNonUniformConvolver::liveCount.load(
                                    std::memory_order_relaxed);
    const int      stereoLive = (int)ConvolverProcessor::StereoConvolver::liveCount.load(
                                    std::memory_order_relaxed);
    const int      dspLive    = (int)AudioEngine::DSPCore::liveCount.load(
                                    std::memory_order_relaxed);
    const uint32_t pending    = m_retireRouter ? m_retireRouter->pendingRetireCount() : 0;
    const uint64_t retireBytes= m_retireRouter ? m_retireRouter->pendingRetireBytes() : 0;
    const uint64_t reclaim    = m_retireRouter ? m_retireRouter->reclaimAttemptCount() : 0;
    const uint64_t overflow   = m_retireRouter ? m_retireRouter->overflowCount() : 0;
    const auto osMem = getProcessMemoryInfo();
    const int64_t untracked = (int64_t)osMem.privateUsageMB * 1024 * 1024
                            - (int64_t)nucBytes - (int64_t)retireBytes;
    const double untrackedMB = std::max(0LL, untracked) / (1024.0 * 1024.0);

    juce::Logger::writeToLog(juce::String::formatted(
        "[MEM_SNAP] PUBLISH gen=%d seq=%d | "
        "NUC: live=%d alloc=%.0fMB peak=%.0fMB | "
        "Stereo=%d DSPCore=%d | "
        "Retire: pending=%u (%.1fMB) overflow=%llu reclaim=%llu | "
        "OS: Private=%lluMB WorkingSet=%lluMB | "
        "Untracked=%.0fMB",
        gen, seq,
        nucLive, nucBytes / (1024.0*1024.0), nucPeak / (1024.0*1024.0),
        stereoLive, dspLive,
        pending, retireBytes / (1024.0*1024.0),
        (unsigned long long)overflow, (unsigned long long)reclaim,
        (unsigned long long)osMem.privateUsageMB,
        (unsigned long long)osMem.workingSetMB,
        untrackedMB));
}
#endif
```

---

## 6. Patch F: イベント駆動ログ（Publish/Retire/IR Reload の 3 イベント中心）

### F-1. Publish 時（Patch E）
### F-2. IR Reload 時（B-7）
### F-3. Retire イベント検出時

```cpp
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
{
    const uint32_t pending = m_retireRouter->pendingRetireCount();
    const uint64_t retireBytes = m_retireRouter->pendingRetireBytes();
    if (pending > 50 || retireBytes > 10 * 1024 * 1024)  // 50個以上 or 10MB超
    {
        const auto osMem = getProcessMemoryInfo();
        const uint64_t nucBytes = convo::diag::allocatedBytes();
        const int64_t untracked = (int64_t)osMem.privateUsageMB * 1024 * 1024
                                - (int64_t)nucBytes - (int64_t)retireBytes;

        juce::Logger::writeToLog(juce::String::formatted(
            "[MEM_RETIRE] pending=%u (%.1fMB) reclaim=%llu | "
            "OS: Private=%lluMB | Untracked=%.0fMB",
            pending, retireBytes / (1024.0*1024.0),
            (unsigned long long)m_retireRouter->reclaimAttemptCount(),
            (unsigned long long)osMem.privateUsageMB,
            std::max(0LL, untracked) / (1024.0 * 1024.0)));
    }
}
#endif
```

### F-4. 5秒定期ログ（変化時のみ、優先度低）

---

## 7. 出力例（v6）

### 正常時

```text
[NUC_MEM] NUC#0000001234 | MKL: alloc=62MB peak=142MB live=2 | OS: Private=68MB WorkingSet=120MB | Untracked=6MB
[MEM_SNAP] PUBLISH gen=8 seq=5 | NUC: live=2 alloc=62MB peak=142MB | Stereo=1 DSPCore=1 | Retire: pending=0 (0.0MB) overflow=0 reclaim=47 | OS: Private=68MB WorkingSet=120MB | Untracked=6MB
```

### 異常検出例

```text
[NUC_MEM] NUC#0000001234 | MKL: alloc=630MB peak=640MB live=8 | OS: Private=2330MB WorkingSet=2400MB | Untracked=1520MB
[MEM_SNAP] PUBLISH gen=8 | NUC: live=8 alloc=630MB peak=640MB | Stereo=4 DSPCore=4 | Retire: pending=232 (180.5MB) overflow=0 reclaim=12 | OS: Private=2330MB WorkingSet=2400MB | Untracked=1520MB
```

**解釈**: NUC=630MB + Retire=180MB = 810MB 追跡済み。OS Private=2330MB → **Untracked=1520MB**。

---

## 8. 診断フロー（v6）

| MKL alloc | liveCount | pendingRetire | OS Private | Untracked | 診断 |
|:----------|:----------|:--------------|:-----------|:----------|:-----|
| ~60MB | 2 | 0MB | ~70MB | ~10MB | **正常。2.33GB は NUC 外部** |
| ~600MB | 8 | 0MB | ~610MB | ~10MB | **NUC リーク** |
| ~60MB | 2 | >100MB | ~600MB | ~440MB | **RetireQueue 滞留** |
| ~60MB | 2 | 0MB | 2330MB | **2270MB** | **NUC 外の未計測領域** |

---

## 9. 実装コスト見積もり（v6）

| Patch | 変更ファイル数 | 追加行数 | 備考 |
|:------|:-------------|:---------|:-----|
| A: DiagnosticsConfig.h | 1 | ~45行 | 軽量 MklAllocStats + ヘルパー + MemoryCategory |
| B: MKLNonUniformConvolver | 2 | ~40行 | globalAllocated 廃止 + releaseAllLayers で DIAG_MKL_FREE |
| C: ISRRetireRouter + DeletionEntry | 5 | ~35行 | retainedBytes(#if付き) + pendingRetireBytes |
| D: DSPCore | 2 | ~18行 | v5 から変更なし |
| E: Publish スナップショット | 1 | ~22行 | Untracked メモリ追加 |
| F: イベント駆動ログ | 2 | ~18行 | v5 から変更なし |
| **合計** | **8〜10ファイル** | **~178行** | |

---

## 10. v5 からの改善点一覧

| 項目 | v5（問題） | v6（修正） |
|:-----|:----------|:----------|
| `diagMklFree` 実装 | `unordered_map<void*,size_t>` + `mutex` | **`DIAG_MKL_FREE(ptr, size)` 呼び出し元指定（マップ不要、ロック不要）** |
| `freeAll()` 内の DIAG_MKL_FREE | Layer にサイズ保存が必要 | **`releaseAllLayers()` 側でサイズ計算 → `DIAG_MKL_FREE` 呼び出し** |
| 未計測メモリ名称 | `Unknown` | **`Untracked`（意味的に正確）** |
| RetireEntry フィールド名 | `estimatedSize` | **`retainedBytes`（保持バイト数、意味的に明確）** |
| `DeletionEntry` 変更 | 無条件フィールド追加 | **`#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` で条件付き（Release 影響ゼロ）** |
| `liveCount` 更新 | `liveCount++` | **`liveCount.fetch_add(1, relaxed)`** |
| API デザイン | `currentMklAllocMB()` (double) | **`allocatedBytes()` / `peakBytes()` (uint64_t)** |
| peak 更新 | 毎回 while ループ | **`updateAtomicMaximum64()` ヘルパー統一** |
| NUC_MEM ログ | L0/L1/L2 Layer 別表示 | **Layer 別表示削除（`MKL: alloc=XXMB` のみ）** |
| カテゴリ計測 | なし | **`MemoryCategory` enum + `CategoryStats` 構造を用意（将来拡張用）** |
