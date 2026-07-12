# メモリ占有調査のためのインストルメンテーション改修案 v16（最終版）— マクロ自動付加 + ライフサイクル制約 + IR_LOAD 詳細

---

**目的**: 2.33GB メモリ占有の原因特定。推定ではなく実測で確定させる。
**v15 からの変更**: 5 点の実装上の問題を修正。

---

## 0. v15 の問題点と v16 での修正方針

| # | v15 の問題 | v16 の修正 |
|:--|:----------|:----------|
| 1 | `getDiagnostics()` MessageThread 制約のみ → SetImpulse 完了前の一貫性問題 | **「SetImpulse/Publish 完了後のみ」とライフサイクル制約を明文化** |
| 2 | `addIfAlive()` 仕様と実装が不一致（DBG警告のみなのに統計bodgeと記載） | **仕様を「DBG警告のみ、統計更新しない」に統一** |
| 3 | `diagMklFree()` の caller 識別が文字列引数 | **`__FILE__`/`__LINE__`/`__func__` をマクロ経由で自動付加** |
| 4 | `liveCount` アンダーフロー検出が race condition 含み | **`fetch_sub()` 戻り値で検出** |
| 5 | `IR_LOAD` が before/after/delta のみ | **IR長/blockSize/oversampling を追加** |

---

## 1. Patch A: DiagnosticsConfig.h — マクロ自動付加 + 仕様統一

### A-1. DIAG_MKL_FREE マクロ — `__FILE__`/`__LINE__` 自動付加（★ v16）

```cpp
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
  #define DIAG_MKL_MALLOC(size, align) convo::diag::diagMklMalloc((size), (align))

  // ★ v16: __FILE__, __LINE__, __func__ を自動付加。
  //   呼び出し元は DIAG_MKL_FREE(ptr, size) と書くだけで良い。
  #define DIAG_MKL_FREE(ptr, size) \
      convo::diag::diagMklFree((ptr), (size), __FILE__, __LINE__, __func__)

#else
  #define DIAG_MKL_MALLOC(size, align) mkl_malloc((size), (align))
  #define DIAG_MKL_FREE(ptr, size)     mkl_free(ptr)
#endif
```

### A-2. diagMklFree — file/line/func パラメータ版

```cpp
inline void diagMklFree(void* ptr, size_t size,
                         const char* file, int line, const char* func) noexcept
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
            DBG("[DIAG] diagMklFree size=0 at "
                << (file ? file : "?") << ":" << line
                << " " << (func ? func : "?")
                << " ptr=" << ptr);
        }
    }
}
```

**★ v15 の `caller` 文字列版は削除。マクロ経由で自動的に `__FILE__`/`__LINE__`/`__func__` が付く。**

### A-3. addIfAlive — DBG警告のみ（★ v16 仕様統一）

```cpp
/// ★ v16: ポインタ生存確認 + size==0 の警告。
///   仕様: 「警告のみ、統計更新しない」。
///   size==0 は allocSizes 保存漏れを意味するが、診断統計を狂わせないため 0 扱い。
static uint64_t addIfAlive(const double* ptr, size_t allocSize, const char* name) noexcept
{
    if (ptr)
    {
        if (allocSize == 0)
        {
            DBG("[DIAG] addIfAlive: " << (name ? name : "?")
                << " ptr=" << ptr << " size=0 (allocSizes missing)");
            // ★ size==0 は 0バイト扱い（統計を狂わせない）
        }
        return allocSize;
    }
    return 0;
}
```

---

## 2. Patch B: MKLNonUniformConvolver — ライフサイクル制約 + liveCount race safe

### B-1. getDiagnostics() — ライフサイクル制約を明文化（★ v16）

```cpp
/// ★ v16: 診断用スナップショット。
///   制約:
///     1. Message Thread からのみ呼び出し可能。
///     2. SetImpulse() / releaseAllLayers() 実行中は呼ばないこと。
///        推奨: Publish 直後または SetImpulse 完了後のみ。
///   データ競合: Layer のメンバは Message Thread からのみ書き込まれる。
///   本メソッドも Message Thread から読むため、C++ memory model 上安全。
///   ただし、SetImpulse 実行途中で呼ぶと不整合値を読む可能性がある。
[[nodiscard]] NucDiagnosticsSnapshot getDiagnostics() const noexcept
{
    jassert(juce::MessageManager::getInstance()->isThisTheMessageThread());
    // ★ コメントで「SetImpulse 完了後のみ」を明記

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

### B-2. デストラクタ — fetch_sub 戻り値でアンダーフロー検出（★ v16）

```cpp
MKLNonUniformConvolver::~MKLNonUniformConvolver()
{
    // ★ v16: fetch_sub の戻り値でアンダーフロー検出（race condition に強い）
    const uint32_t oldLive = liveCount.fetch_sub(1, std::memory_order_relaxed);
    jassert(oldLive > 0);  // oldLive==0 ならアンダーフロー
    releaseAllLayers();
}
```

### B-3. SetImpulse — IR_LOAD 詳細ログ（★ v16）

```cpp
bool MKLNonUniformConvolver::SetImpulse(const double* impulse, int irLen, int blockSize, double scale,
                                        bool enableDirectHead,
                                        const FilterSpec* filterSpec)
{
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    const uint64_t beforeBytes = convo::diag::allocatedBytes();
#endif

    convo::publishAtomic(m_ready, false, std::memory_order_release);
    if (impulse == nullptr || irLen <= 0 || blockSize <= 0)
        return false;

    releaseAllLayers();

    // ... 既存のレイヤー構成決定 + 確保 + プリコンピュート ...

    convo::publishAtomic(m_ready, true, std::memory_order_release);

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    const uint64_t afterBytes = convo::diag::allocatedBytes();
    const int64_t delta = static_cast<int64_t>(afterBytes) - static_cast<int64_t>(beforeBytes);
    // ★ v16: IR長/blockSize/liveCount も併記（ログ単体で解析可能）
    const int l0Part = /* 既存の計算 */;
    const int oversamplingFactor = /* 呼び出し元から取得 or 保持 */;
    diagLog(juce::String::formatted(
        "[IR_LOAD] NUC#%p irLen=%d blockSize=%d L0Part=%d"
        " before=%lluMB after=%lluMB delta=%lldMB live=%u",
        (void*)this,
        irLen, blockSize, l0Part,
        (unsigned long long)(beforeBytes / (1024*1024)),
        (unsigned long long)(afterBytes / (1024*1024)),
        (long long)(delta / (1024*1024)),
        (unsigned)liveCount.load(std::memory_order_relaxed)));
#endif

    return true;
}
```

---

## 3. 全ログの v16 統一フォーマット

### IR_LOAD

```text
[IR_LOAD] NUC#%p irLen=%d blockSize=%d L0Part=%d before=%lluMB after=%lluMB delta=%lldMB live=%u
```

### NUC_MEM

```text
[NUC_MEM] NUC#%p | LayerBuf: L0=%.0fMB L1=%.0fMB L2=%.0fMB Direct=%.0fMB Ring=%.0fMB | MKL: cur=%.0fMB Peak(since reset)=%.0fMB totalA=%.0fGB totalF=%.0fGB lostFree=%u live=%u | OS: Private=%lluMB WorkingSet=%lluMB | Uninstrumented=%.0fMB
```

### MEM_SNAP

```text
[MEM_SNAP] PUBLISH gen=%d seq=%d | NUC(MKL only): live=%u alloc=%.0fMB Peak(since reset)=%.0fMB totalA=%.0fGB totalF=%.0fGB lostFree=%u | Stereo=%d DSPCore=%d | Retire: pending=%u objBytes=%.1fMB(sizeof) tracked=%u/%u (%.0f%%) overflow=%llu reclaim=%llu | OS: Private=%lluMB WorkingSet=%lluMB | Uninstrumented=%.0fMB
```

---

## 4. 出力例（v16）

```text
[IR_LOAD] NUC#0000001234 irLen=327680 blockSize=4096 L0Part=4096 before=420MB after=812MB delta=+392MB live=8
[NUC_MEM] NUC#0000001234 | LayerBuf: L0=8MB L1=64MB L2=512MB Direct=24MB Ring=8MB | MKL: cur=832MB Peak(since reset)=1200MB totalA=28.0GB totalF=27.4GB lostFree=0 live=8 | OS: Private=2330MB WorkingSet=2400MB | Uninstrumented=1498MB
[MEM_SNAP] PUBLISH gen=8 | NUC(MKL only): live=8 alloc=832MB Peak(since reset)=1200MB totalA=28.0GB totalF=27.4GB lostFree=0 | Stereo=4 DSPCore=4 | Retire: pending=232 objBytes=12.8MB(sizeof) tracked=8/232 (3%) overflow=0 reclaim=12 | OS: Private=2330MB WorkingSet=2400MB | Uninstrumented=1498MB
```

---

## 5. v15 からの改善点一覧

| # | v15（問題） | v16（修正） |
|:--|:----------|:-----------|
| 1 | `getDiagnostics()` MessageThread 制約のみ → SetImpulse 実行中の不整合リスク | **「SetImpulse/Publish 完了後のみ」とライフサイクル制約をコメントに明記** |
| 2 | `addIfAlive()` 仕様「bodge」と実装「return 0」が不一致 | **仕様を「DBG警告のみ、統計更新しない」に統一** |
| 3 | `diagMklFree()` caller 識別が文字列引数 | **`__FILE__`/`__LINE__`/`__func__` をマクロ経由で自動付加** |
| 4 | `liveCount` アンダーフロー検出が race condition 含み | **`fetch_sub()` 戻り値で検出** |
| 5 | `IR_LOAD` が before/after/delta のみ | **`irLen`/`blockSize`/`L0Part`/`live` を追加** |
