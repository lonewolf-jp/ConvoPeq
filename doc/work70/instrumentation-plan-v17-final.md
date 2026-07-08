# メモリ占有調査のためのインストルメンテーション改修案 v17（最終版）— m_isBuilding フラグ + IR_LAYOUT + liveCount CAS

---

**目的**: 2.33GB メモリ占有の原因特定。推定ではなく実測で確定させる。
**v16 からの変更**: 4 点の問題を修正。

---

## 0. v16 の問題点と v17 での修正方針

| # | v16 の問題 | v17 の修正 |
|:--|:----------|:----------|
| 1 | `getDiagnostics()` が MessageThread 制約のみ → SetImpulse 実行中の診断呼び出しを防止できない | **`m_isBuilding` atomic フラグ追加。SetImpulse 前後に設定。jassert で検証** |
| 2 | `liveCount.fetch_sub()` アンダーフロー時 `UINT_MAX` になる | **`compare_exchange` 方式で Release でも安全** |
| 3 | `IR_LOAD` に partition 構成 / direct taps / ring size がない | **`L0Part/L1Part/L2Part directTaps ringSize` を追加** |
| 4 | バッファ種別ごとのメモリ内訳ログがない | **`[IR_LAYOUT]` ログ追加（IRFreq/FDL/Accum/TailOutput/Direct/Ring）** |
| 5 | `Uninstrumented` が実態より広い意味に | **`OtherPrivate` に変更** |

---

## 1. Patch A: DiagnosticsConfig.h — 最終版

### A-1. MklAllocStats（v17 確定版、変更なし）

```cpp
namespace convo::diag {

struct MklAllocStats {
    std::atomic<uint64_t> allocatedBytes { 0 };
    std::atomic<uint64_t> peakBytes      { 0 };
    std::atomic<uint64_t> totalAllocBytes{ 0 };
    std::atomic<uint64_t> totalFreedBytes{ 0 };
    std::atomic<uint32_t> lostFreeCount  { 0 };
};

// ... diagMklMalloc, diagMklFree, accessors, resetDiagnostics ...
```

### A-2. DIAG_MKL_FREE マクロ（__FILE_NAME__ 優先）

```cpp
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
  #define DIAG_MKL_MALLOC(size, align) convo::diag::diagMklMalloc((size), (align))

  // ★ v17: __FILE_NAME__ (MSVC) が使えればそちらを優先（パスが短い）
  #ifdef _MSC_VER
    #define DIAG_MKL_FREE(ptr, size) \
        convo::diag::diagMklFree((ptr), (size), __FILE_NAME__, __LINE__, __func__)
  #else
    #define DIAG_MKL_FREE(ptr, size) \
        convo::diag::diagMklFree((ptr), (size), __FILE__, __LINE__, __func__)
  #endif
#else
  #define DIAG_MKL_MALLOC(size, align) mkl_malloc((size), (align))
  #define DIAG_MKL_FREE(ptr, size)     mkl_free(ptr)
#endif
```

---

## 2. Patch B: MKLNonUniformConvolver — m_isBuilding + liveCount CAS + IR_LAYOUT

### B-1. MKLNonUniformConvolver.h — m_isBuilding 追加

```cpp
class MKLNonUniformConvolver {
public:
    static std::atomic<uint32_t> liveCount;
    // ...

    [[nodiscard]] NucDiagnosticsSnapshot getDiagnostics() const noexcept;

private:
    std::atomic<bool> m_isBuilding { false };  // ★ v17: SetImpulse 実行中フラグ

    // ... 既存メンバ ...
};
```

### B-2. デストラクタ — compare_exchange 方式（★ v17）

```cpp
MKLNonUniformConvolver::~MKLNonUniformConvolver()
{
    // ★ v17: compare_exchange 方式で Release でも安全。
    //   liveCount が 0 の場合はデクリメントしない（アンダーフロー防止）。
    uint32_t expected = liveCount.load(std::memory_order_relaxed);
    while (expected > 0 &&
           !liveCount.compare_exchange_weak(expected, expected - 1,
               std::memory_order_relaxed, std::memory_order_relaxed))
    {}
    jassert(expected > 0);  // Debug ではアンダーフロー検出

    releaseAllLayers();
}
```

### B-3. getDiagnostics() — m_isBuilding チェック（★ v17）

```cpp
[[nodiscard]] NucDiagnosticsSnapshot getDiagnostics() const noexcept
{
    // ★ v17: MessageThread + ビルド中でないことを両方チェック
    jassert(juce::MessageManager::getInstance()->isThisTheMessageThread());
    jassert(!m_isBuilding.load(std::memory_order_acquire));

    // ... 既存の集計ロジック ...
}
```

### B-4. SetImpulse — m_isBuilding + IR_LAYOUT（★ v17）

```cpp
bool MKLNonUniformConvolver::SetImpulse(const double* impulse, int irLen, int blockSize, double scale,
                                        bool enableDirectHead,
                                        const FilterSpec* filterSpec)
{
    // ★ v17: ビルド中フラグ設定
    m_isBuilding.store(true, std::memory_order_release);

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    const uint64_t beforeBytes = convo::diag::allocatedBytes();
#endif

    convo::publishAtomic(m_ready, false, std::memory_order_release);
    if (impulse == nullptr || irLen <= 0 || blockSize <= 0)
    {
        m_isBuilding.store(false, std::memory_order_release);
        return false;
    }

    releaseAllLayers();

    // ... 既存のレイヤー構成決定 + 確保 + プリコンピュート ...

    convo::publishAtomic(m_ready, true, std::memory_order_release);

    // ★ v17: ビルド中フラグ解除（この後は診断安全）
    m_isBuilding.store(false, std::memory_order_release);

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    const uint64_t afterBytes = convo::diag::allocatedBytes();
    const int64_t delta = static_cast<int64_t>(afterBytes) - static_cast<int64_t>(beforeBytes);

    // IR_LOAD: 構成情報
    diagLog(juce::String::formatted(
        "[IR_LOAD] NUC#%p irLen=%d blockSize=%d "
        "L0Part=%d directTaps=%d ringSize=%d "
        "before=%lluMB after=%lluMB delta=%lldMB live=%u",
        (void*)this, irLen, blockSize, m_layers[0].partSize,
        m_directTapCount, m_ringSize,
        (unsigned long long)(beforeBytes / (1024*1024)),
        (unsigned long long)(afterBytes / (1024*1024)),
        (long long)(delta / (1024*1024)),
        (unsigned)liveCount.load(std::memory_order_relaxed)));

    // IR_LAYOUT: バッファ種別ごとのメモリ内訳（★ v17 追加）
    //   現在の Layer 状態を静的集計（ビルド完了後のため安全）
    const auto snap = getDiagnostics();
    uint64_t irFreqTotal = 0, fdlTotal = 0, accumTotal = 0, tailTotal = 0;
    for (int li = 0; li < m_numActiveLayers; ++li)
    {
        const Layer& l = m_layers[li];
        irFreqTotal += addIfAlive(l.irFreqDomain, l.allocSizes.irFreqDomain, "irFreqDomain");
        irFreqTotal += addIfAlive(l.irFreqReal,   l.allocSizes.irFreqReal,   "irFreqReal");
        irFreqTotal += addIfAlive(l.irFreqImag,   l.allocSizes.irFreqImag,   "irFreqImag");
        fdlTotal    += addIfAlive(l.fdlBuf,       l.allocSizes.fdlBuf,       "fdlBuf");
        fdlTotal    += addIfAlive(l.fdlReal,      l.allocSizes.fdlReal,      "fdlReal");
        fdlTotal    += addIfAlive(l.fdlImag,      l.allocSizes.fdlImag,      "fdlImag");
        accumTotal  += addIfAlive(l.fftTimeBuf,   l.allocSizes.fftTimeBuf,   "fftTimeBuf");
        accumTotal  += addIfAlive(l.fftOutBuf,    l.allocSizes.fftOutBuf,    "fftOutBuf");
        accumTotal  += addIfAlive(l.prevInputBuf, l.allocSizes.prevInputBuf, "prevInputBuf");
        accumTotal  += addIfAlive(l.accumBuf,     l.allocSizes.accumBuf,     "accumBuf");
        accumTotal  += addIfAlive(l.accumReal,    l.allocSizes.accumReal,    "accumReal");
        accumTotal  += addIfAlive(l.accumImag,    l.allocSizes.accumImag,    "accumImag");
        accumTotal  += addIfAlive(l.inputAccBuf,  l.allocSizes.inputAccBuf,  "inputAccBuf");
        tailTotal   += addIfAlive(l.tailOutputBuf,l.allocSizes.tailOutputBuf,"tailOutputBuf");
    }
    diagLog(juce::String::formatted(
        "[IR_LAYOUT] NUC#%p IRFreq=%.0fMB FDL=%.0fMB "
        "Accum=%.0fMB Tail=%.0fMB Direct=%.0fMB Ring=%.0fMB",
        (void*)this,
        irFreqTotal / (1024.0*1024.0),
        fdlTotal   / (1024.0*1024.0),
        accumTotal / (1024.0*1024.0),
        tailTotal  / (1024.0*1024.0),
        snap.directBytes / (1024.0*1024.0),
        snap.ringBytes   / (1024.0*1024.0)));
#endif

    return true;
}
```

**★ 注意**: `getDiagnostics()` を SetImpulse 完了後（`m_isBuilding=false` の後）に
呼んでいる。これにより、データ競合もライフサイクル不整合も発生しない。

---

## 3. 全ログの v17 統一フォーマット

### IR_LOAD

```text
[IR_LOAD] NUC#%p irLen=%d blockSize=%d L0Part=%d directTaps=%d ringSize=%d before=%lluMB after=%lluMB delta=%lldMB live=%u
```

### IR_LAYOUT

```text
[IR_LAYOUT] NUC#%p IRFreq=%.0fMB FDL=%.0fMB Accum=%.0fMB Tail=%.0fMB Direct=%.0fMB Ring=%.0fMB
```

### NUC_MEM

```text
[NUC_MEM] NUC#%p | LayerBuf: L0=%.0fMB L1=%.0fMB L2=%.0fMB Direct=%.0fMB Ring=%.0fMB | MKL: cur=%.0fMB Peak(since reset)=%.0fMB totalA=%.0fGB totalF=%.0fGB lostFree=%u live=%u | OS: Private=%lluMB WorkingSet=%lluMB | OtherPrivate=%.0fMB
```

### MEM_SNAP

```text
[MEM_SNAP] PUBLISH gen=%d seq=%d | NUC(MKL only): live=%u alloc=%.0fMB Peak(since reset)=%.0fMB totalA=%.0fGB totalF=%.0fGB lostFree=%u | Stereo=%d DSPCore=%d | Retire: pending=%u objBytes=%.1fMB(sizeof) tracked=%u/%u (%.0f%%) overflow=%llu reclaim=%llu | OS: Private=%lluMB WorkingSet=%lluMB | OtherPrivate=%.0fMB
```

---

## 4. 出力例（v17）

```text
[IR_LOAD] NUC#0000001234 irLen=327680 blockSize=4096 L0Part=4096 directTaps=32 ringSize=8192 before=420MB after=812MB delta=+392MB live=8
[IR_LAYOUT] NUC#0000001234 IRFreq=256MB FDL=420MB Accum=96MB Tail=16MB Direct=24MB Ring=8MB
[NUC_MEM] NUC#0000001234 | LayerBuf: L0=8MB L1=64MB L2=512MB Direct=24MB Ring=8MB | MKL: cur=832MB Peak(since reset)=1200MB totalA=28.0GB totalF=27.4GB lostFree=0 live=8 | OS: Private=2330MB WorkingSet=2400MB | OtherPrivate=1498MB
[MEM_SNAP] PUBLISH gen=8 | NUC(MKL only): live=8 alloc=832MB Peak(since reset)=1200MB totalA=28.0GB totalF=27.4GB lostFree=0 | ... | OtherPrivate=1498MB
```

**解釈**:
- `IR_LOAD`: IR長=327680 samples, blockSize=4096, L0Part=4096, 先頭32 tap direct, ring=8192 → delta=+392MB
- `IR_LAYOUT`: FDL=420MB が支配的（FDL は partition 数に比例）
- `OtherPrivate`=1498MB → MKL+Retire 以外のメモリ

---

## 5. v16 からの改善点一覧

| # | v16（問題） | v17（修正） |
|:--|:----------|:-----------|
| 1 | `getDiagnostics()` MessageThread 制約のみ → SetImpulse 実行中の不整合 | **`m_isBuilding` atomic フラグ + jassert** |
| 2 | `liveCount.fetch_sub()` アンダーフロー時 UINT_MAX | **`compare_exchange` 方式で Release でも安全** |
| 3 | `IR_LOAD` に partition 構成がない | **`L0Part/directTaps/ringSize` 追加** |
| 4 | バッファ種別ごとの内訳ログがない | **`[IR_LAYOUT] IRFreq/FDL/Accum/Tail/Direct/Ring` 追加** |
| 5 | `Uninstrumented` が実態より広い | **`OtherPrivate` に変更** |
| 6 | `__FILE__` が長いフルパス | **MSVC では `__FILE_NAME__` 優先** |
