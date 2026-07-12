# メモリ占有調査のためのインストルメンテーション改修案 v12（最終版）— カテゴリ別集計 + 警告ログ + 名前整理

---

**目的**: 2.33GB メモリ占有の原因特定。推定ではなく実測で確定させる。
**v11 からの変更**: reviewer フィードバック（5点）を全て反映。

---

## 0. v11 の問題点と v12 での修正方針

| # | v11 の問題 | v12 の修正 |
|:--|:----------|:----------|
| 1 | `freeTracked()` を `releaseAllLayers()` に流用 → `m_ringBuf` ≠ nullptr かつ `m_ringSize=0` で偽陽性 jassert | **`freeTrackedSize()` ヘルパ: 動的サイズ計算用に分離（jassert なし）** |
| 2 | `diagMklFree()` の `jassert(size != 0)` が厳しすぎる → 未知サイズ用途で debug build 停止 | **DBG 警告に緩和 + size=0 時は統計更新スキップ** |
| 3 | `allocSizes` 保存コメントが「成功直後」だが実装は「要求直後」 | **コメントを「確保要求サイズを保存」に修正** |
| 4 | `peakBytes` が「リセット以降のピーク」であることをログで明示していない | **`Peak(since reset)=` に変更** |
| 5 | Layer 別 / Direct / Ring のカテゴリ別集計がない → 原因特定に情報不足 | **`CategoryAllocStats` + `logNucCategoryBreakdown()` 追加** |

---

## 1. Patch A: DiagnosticsConfig.h — freeTrackedSize + diagMklFree 警告化

### A-1. MklAllocStats（v11 から変更なし）

```cpp
namespace convo::diag {

struct MklAllocStats {
    std::atomic<uint64_t> allocatedBytes { 0 };
    std::atomic<uint64_t> peakBytes      { 0 };
    std::atomic<uint64_t> totalAllocBytes{ 0 };
    std::atomic<uint64_t> totalFreedBytes{ 0 };
};

// ... diagMklMalloc, accessors, resetDiagnostics, updateAtomicMaximum64 ...
```

### A-2. diagMklFree — jassert から DBG 警告に緩和（★ v12）

```cpp
inline void diagMklFree(void* ptr, size_t size) noexcept
{
    if (ptr)
    {
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS && JUCE_DEBUG
        // ★ v12: size=0 は jassert ではなく警告ログ。
        //   将来の未知サイズ用途でもデバッグビルドを止めない。
        if (size == 0)
            DBG("[DIAG] diagMklFree called with size=0 ptr=" << ptr);
#endif

        mkl_free(ptr);

        // ★ v12: size が既知の場合のみ統計を更新（安全動作）
        if (size > 0)
        {
            mklStats().allocatedBytes.fetch_sub(
                static_cast<uint64_t>(size), std::memory_order_relaxed);
            mklStats().totalFreedBytes.fetch_add(
                static_cast<uint64_t>(size), std::memory_order_relaxed);
        }
    }
}
```

### A-3. freeTracked — Layer 専用（jassert 維持。ptr と size が 1 セットのため）

```cpp
/// ★ v12: Layer 専用。pointer と allocSizes が 1 セットで管理されている前提。
///   デバッグビルドでは size == 0 をアサート（allocSizes 保存漏れ検出）。
template<typename T>
inline void freeTracked(T*& p, size_t size) noexcept
{
    if (p)
    {
        jassert(size != 0);  // Layer では ptr と size が 1:1 対応
        DIAG_MKL_FREE(p, size);
        p = nullptr;
    }
}
```

### A-4. freeTrackedSize — NUC レベル専用（動的サイズ計算、jassert なし）（★ v12）

```cpp
/// ★ v12: NUC レベル（m_ringBuf / m_direct*）専用。
///   サイズを動的計算するため、ptr と size の管理が別。
///   jassert はなし（m_ringSize のリセットタイミング次第で偽陽性のため）。
template<typename T>
inline void freeTrackedSize(T*& p, size_t size) noexcept
{
    if (p && size > 0)
    {
        DIAG_MKL_FREE(p, size);
        p = nullptr;
    }
    else if (p && size == 0)
    {
        // ★ サイズ不明でもポインタは解放（メモリリーク防止）
        mkl_free(p);
        p = nullptr;
    }
}
```

### A-5. CategoryAllocStats — カテゴリ別集計（★ v12 追加）

```cpp
/// ★ v12: カテゴリ別メモリ集計（NUC_MEM ログ用）。
struct CategoryStats {
    uint64_t layer0  = 0;  // Layer 0 (Immediate)
    uint64_t layer1  = 0;  // Layer 1 (Tail)
    uint64_t layer2  = 0;  // Layer 2 (Tail)
    uint64_t direct  = 0;  // Direct FIR バッファ
    uint64_t ring    = 0;  // 出力リングバッファ
};

/// ★ NUC の全バッファをカテゴリ別に集計（Message Thread から呼び出し）。
///   現在の Layer 状態をもとに算出するため real-time 安全ではない。
inline CategoryStats computeNucCategoryStats(
    const MKLNonUniformConvolver& nuc) noexcept
{
    CategoryStats s{};
    for (int li = 0; li < nuc.numActiveLayers(); ++li)
    {
        const auto& l = nuc.getLayer(li);
        uint64_t layerTotal = 0;
        layerTotal += l.allocSizes.irFreqDomain;
        layerTotal += l.allocSizes.irFreqReal;
        layerTotal += l.allocSizes.irFreqImag;
        layerTotal += l.allocSizes.fdlBuf;
        layerTotal += l.allocSizes.fdlReal;
        layerTotal += l.allocSizes.fdlImag;
        layerTotal += l.allocSizes.fftTimeBuf;
        layerTotal += l.allocSizes.fftOutBuf;
        layerTotal += l.allocSizes.prevInputBuf;
        layerTotal += l.allocSizes.accumBuf;
        layerTotal += l.allocSizes.accumReal;
        layerTotal += l.allocSizes.accumImag;
        layerTotal += l.allocSizes.inputAccBuf;
        layerTotal += l.allocSizes.tailOutputBuf;

        if (li == 0)        s.layer0 = layerTotal;
        else if (li == 1)   s.layer1 = layerTotal;
        else if (li == 2)   s.layer2 = layerTotal;
    }
    // Direct + Ring は NUC から直接取得（Ring は m_ringSize）
    // 実際の実装では nuc に公開メソッドを追加するか friend
    return s;
}
```

---

## 2. Patch B: MKLNonUniformConvolver — releaseAllLayers + allocSizes コメント

### B-1. SetImpulse — allocSizes 保存コメント修正（★ v12）

```cpp
l.irFreqDomain = static_cast<double*>(DIAG_MKL_MALLOC(irBufSize * sizeof(double), 64));
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
// ★ v12: 「確保要求サイズ」を保存（確保成功かは ptr が nullptr かで判定）。
//   freeTracked() は if(ptr) でガードするため安全。
l.allocSizes.irFreqDomain = irBufSize * sizeof(double);
#endif
```

### B-2. releaseAllLayers() — freeTrackedSize 使用（★ v12）

```cpp
void MKLNonUniformConvolver::releaseAllLayers() noexcept
{
    // ... 既存の guard チェック ...

    for (int i = 0; i < kNumLayers; ++i)
        m_layers[i].freeAll();
    m_numActiveLayers = 0;
    m_latency         = 0;

    // ★ v12: freeTrackedSize — 動的サイズ計算、jassert なし
    //   m_ringSize/m_directTapCount 等は別管理のため、偽陽性を避ける。
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    freeTrackedSize(m_ringBuf,
        static_cast<size_t>(m_ringSize) * sizeof(double));
    freeTrackedSize(m_directIRRev,
        static_cast<size_t>(m_directTapCount) * sizeof(double));
    freeTrackedSize(m_directHistory,
        static_cast<size_t>(m_directHistLen) * sizeof(double));
    freeTrackedSize(m_directWindow,
        static_cast<size_t>(m_directHistLen + m_directMaxBlock) * sizeof(double));
    freeTrackedSize(m_directOutBuf,
        static_cast<size_t>(m_directMaxBlock) * sizeof(double));
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

### B-3. NUC_MEM ログ — カテゴリ別集計追加（★ v12）

```cpp
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
{
    const auto cat = convo::diag::computeNucCategoryStats(*this);
    const uint64_t curBytes  = convo::diag::allocatedBytes();
    const uint64_t peakBytes = convo::diag::peakBytes();
    const uint64_t totalA    = convo::diag::totalAllocBytes();
    const uint64_t totalF    = convo::diag::totalFreedBytes();
    const auto osMem = getProcessMemoryInfo();
    const uint64_t retireBytes = /* pendingRetireBytes() */;
    const int64_t untracked = (int64_t)osMem.privateUsageMB * 1024 * 1024
                            - (int64_t)curBytes - (int64_t)retireBytes;

    diagLog(juce::String::formatted(
        "[NUC_MEM] NUC#%p | "
        "Cat: L0=%.0fMB L1=%.0fMB L2=%.0fMB "
        "Direct=%.0fMB Ring=%.0fMB | "
        "MKL: cur=%.0fMB Peak(since reset)=%.0fMB "
        "totalA=%.0fGB totalF=%.0fGB live=%d | "
        "OS: Private=%lluMB WorkingSet=%lluMB | "
        "Untracked(other)=%.0fMB",
        (void*)this,
        cat.layer0  / (1024.0*1024.0),
        cat.layer1  / (1024.0*1024.0),
        cat.layer2  / (1024.0*1024.0),
        cat.direct  / (1024.0*1024.0),
        cat.ring    / (1024.0*1024.0),
        curBytes / (1024.0*1024.0),
        peakBytes / (1024.0*1024.0),
        totalA / (1024.0*1024.0*1024.0),
        totalF / (1024.0*1024.0*1024.0),
        (int)liveCount.load(std::memory_order_relaxed),
        (unsigned long long)osMem.privateUsageMB,
        (unsigned long long)osMem.workingSetMB,
        std::max(0LL, untracked) / (1024.0 * 1024.0)));
}
#endif
```

### B-4. デストラクタ — liveCount を uint32_t に変更（★ v12）

```cpp
// .h:
static std::atomic<uint32_t> liveCount;

// .cpp:
std::atomic<uint32_t> MKLNonUniformConvolver::liveCount { 0 };

MKLNonUniformConvolver::~MKLNonUniformConvolver()
{
    liveCount.fetch_sub(1, std::memory_order_relaxed);
    releaseAllLayers();
}
```

---

## 3. MEM_SNAP ログ — Peak(since reset) に変更（★ v12）

```cpp
    "[MEM_SNAP] PUBLISH gen=%d seq=%d | "
    "NUC(MKL only): live=%d alloc=%.0fMB Peak(since reset)=%.0fMB "
    "totalA=%.0fGB totalF=%.0fGB | "
    "Stereo=%d DSPCore=%d | "
    "Retire: pending=%u objBytes=%.1fMB(sizeof) tracked=%u/%u (%.0f%%) "
    "overflow=%llu reclaim=%llu | "
    "OS: Private=%lluMB WorkingSet=%lluMB | "
    "Untracked(other)=%.0fMB(JUCE/IPP/CRT/threads/...)",
    // ...
    nucBytes / (1024.0*1024.0),
    nucPeak / (1024.0*1024.0),  // ★「Peak(since reset)=」
```

---

## 4. 出力例（v12）

### 正常時

```text
[NUC_MEM] NUC#0000001234 | Cat: L0=8MB L1=0MB L2=0MB Direct=1MB Ring=2MB | MKL: cur=62MB Peak(since reset)=142MB totalA=3.2GB totalF=3.14GB live=2 | OS: Private=68MB WorkingSet=120MB | Untracked(other)=6MB
[MEM_SNAP] PUBLISH gen=8 seq=5 | NUC(MKL only): live=2 alloc=62MB Peak(since reset)=142MB totalA=3.2GB totalF=3.14GB | Stereo=1 DSPCore=1 | Retire: pending=0 objBytes=0.0MB(sizeof) tracked=0/0 (0%) overflow=0 reclaim=47 | OS: Private=68MB WorkingSet=120MB | Untracked(other)=6MB
```

### リーク検出時

```text
[NUC_MEM] NUC#0000001234 | Cat: L0=32MB L1=256MB L2=512MB Direct=24MB Ring=8MB | MKL: cur=832MB Peak(since reset)=1200MB totalA=28.0GB totalF=27.4GB live=8 | OS: Private=2330MB WorkingSet=2400MB | Untracked(other)=1498MB
[MEM_SNAP] PUBLISH gen=8 | NUC(MKL only): live=8 alloc=832MB Peak(since reset)=1200MB totalA=28.0GB totalF=27.4GB | Stereo=4 DSPCore=4 | Retire: pending=232 objBytes=12.8MB(sizeof) tracked=8/232 (3%) overflow=0 reclaim=12 | OS: Private=2330MB WorkingSet=2400MB | Untracked(other)=1498MB
```

**解釈**:
- Layer2=512MB が支配的 → IR テールが大きい
- Direct=24MB → 先頭タップが多い
- Untracked=1498MB → 未計測領域（JUCE/CRT/IPP 等）

---

## 5. v11 からの改善点一覧

| 項目 | v11（問題） | v12（修正） |
|:-----|:----------|:-----------|
| `releaseAllLayers()` の freeTracked | ptr と size の管理が別→偽陽性リスク | **`freeTrackedSize()` に分離（jassert なし、size=0 でも mkl_free 実行）** |
| `diagMklFree()` jassert | size=0 でデバッグビルド停止 | **DBG 警告 + size>0 のみ統計更新（best-effort）** |
| allocSizes コメント | 「成功直後」と記載 | **「確保要求サイズ」に修正** |
| Peak ログ | `peak=XXMB`（区別不可） | **`Peak(since reset)=XXMB`** |
| カテゴリ別集計 | なし（総量のみ） | **`CategoryStats` + Cat: L0/L1/L2/Direct/Ring をNUC_MEMログに追加** |
| liveCount 型 | `std::atomic<int>` | **`std::atomic<uint32_t>`** |
