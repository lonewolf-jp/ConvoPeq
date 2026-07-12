# メモリ占有調査のためのインストルメンテーション改修案 v24（最終版）— freeTracked 一本化 + getDiagnostics 拡張

---

**目的**: 2.33GB メモリ占有の原因特定。推定ではなく実測で確定させる。
**v23 からの変更**: 6 点の修正。

---

## 0. v23 の問題点と v24 での修正方針

| # | v23 の問題 | v24 の修正 |
|:--|:----------|:----------|
| 1 | `SetImpulse()` に `diagnosticGeneration` 追加 → API 変更の割にメリット小 | **削除。Generation は MEM_SNAP のみで管理** |
| 2 | `IR_LAYOUT` が `getDiagnostics()` とは別に Layer を再走査 | **`getDiagnostics()` を拡張して `IRFreq/FDL/Accum/Tail` も返す** |
| 3 | `freeTracked()` / `freeTrackedSize()` が 2 種類 | **一本化。size=0 でも mkl_free 実行 + lostFreeCount 更新** |
| 4 | `IR_RELEASE` に解放前の Layer 数なし | **`LayersBefore=%d` 追加** |
| 5 | `IR_LAYOUT` に Layer 別合計なし | **`L0/L1/L2` 追加** |
| 6 | `OtherPrivate` の意味が不明瞭 | **コメントで「全未計測領域」と明記** |

---

## 1. Patch A: DiagnosticsConfig.h — freeTracked 一本化 + NucDiagnosticsSnapshot 拡張

### A-1. freeTracked（一本化版）

```cpp
/// ★ v24: 統一 freeTracked。
///   - size>0 → DIAG_MKL_FREE（統計更新あり）
///   - size==0 → mkl_free（裸の解放）+ lostFreeCount 増加
///   - 常に ptr=nullptr
template<typename T>
inline void freeTracked(T*& p, size_t size) noexcept
{
    if (p)
    {
        if (size > 0)
        {
            DIAG_MKL_FREE(p, size);
        }
        else
        {
            mkl_free(p);
            convo::diag::mklStats().lostFreeCount.fetch_add(1, std::memory_order_relaxed);
            DBG("[DIAG] freeTracked size=0 ptr=" << p);
        }
        p = nullptr;
    }
}
```

**★ v23 の `freeTrackedSize()` は削除。全て `freeTracked()` に統合。**

### A-2. NucDiagnosticsSnapshot 拡張（IR_LAYOUT 情報を含む）

```cpp
/// ★ v24: 診断用スナップショット。
///   グローバル統計は含まない。種別別の内訳も保持（二重走査防止）。
struct NucDiagnosticsSnapshot {
    // Layer 別生存バッファ合計（ポインタ生存確認後）
    uint64_t layerBufs[3] = { 0, 0, 0 };
    // 種別別内訳
    uint64_t irFreqBytes  = 0;  // irFreqDomain + irFreqReal + irFreqImag
    uint64_t fdlBytes     = 0;  // fdlBuf + fdlReal + fdlImag
    uint64_t accumBytes   = 0;  // fftTimeBuf + fftOutBuf + prevInputBuf + accumBuf + accumReal + accumImag + inputAccBuf
    uint64_t tailBytes    = 0;  // tailOutputBuf
    uint64_t directBytes  = 0;  // Direct FIR バッファ
    uint64_t ringBytes    = 0;  // 出力リングバッファ
    // 状態
    int      numActiveLayers = 0;
    bool     isReady         = false;
    // 合計（種別別の合計）
    [[nodiscard]] uint64_t totalBytes() const noexcept {
        return layerBufs[0] + layerBufs[1] + layerBufs[2] + directBytes + ringBytes;
    }
};
```

### A-3. OtherPrivate コメント（DiagnosticsConfig.h）

```cpp
/// ★ v24: OS Private Usage のうち MKL + Retire 以外の全メモリ。
///   以下を含む（非網羅）:
///   - JUCE heap allocations
///   - CRT heap (malloc/new)
///   - IPP FFT spec / work buffers
///   - Thread stacks
///   - DLL mappings
///   - VirtualAlloc (Windows heap)
///   - std::vector / std::string internal allocations
///   - MKL FFT plan workspaces (MKL 内部管理)
inline uint64_t computeOtherPrivate(uint64_t osPrivateMB,
                                    uint64_t mklBytes,
                                    uint64_t retireBytes) noexcept
{
    const int64_t other = static_cast<int64_t>(osPrivateMB) * 1024 * 1024
                        - static_cast<int64_t>(mklBytes)
                        - static_cast<int64_t>(retireBytes);
    return static_cast<uint64_t>(std::max<int64_t>(0, other));
}
```

---

## 2. Patch B: MKLNonUniformConvolver — getDiagnostics 拡張 + gen 削除

### B-1. getDiagnostics() 拡張（IR_LAYOUT 情報を含む）

```cpp
[[nodiscard]] NucDiagnosticsSnapshot getDiagnostics() const noexcept
{
    jassert(juce::MessageManager::getInstance()->isThisTheMessageThread());

    NucDiagnosticsSnapshot snap{};
    snap.numActiveLayers = m_numActiveLayers;
    snap.isReady = convo::consumeAtomic(m_ready, std::memory_order_acquire);

    for (int li = 0; li < kNumLayers; ++li)
    {
        const Layer& l = m_layers[li];
        uint64_t irFreq = 0, fdl = 0, accum = 0, tail = 0;

        irFreq += addIfAlive(l.irFreqDomain, l.allocSizes.irFreqDomain);
        irFreq += addIfAlive(l.irFreqReal,   l.allocSizes.irFreqReal);
        irFreq += addIfAlive(l.irFreqImag,   l.allocSizes.irFreqImag);
        fdl    += addIfAlive(l.fdlBuf,       l.allocSizes.fdlBuf);
        fdl    += addIfAlive(l.fdlReal,      l.allocSizes.fdlReal);
        fdl    += addIfAlive(l.fdlImag,      l.allocSizes.fdlImag);
        accum  += addIfAlive(l.fftTimeBuf,   l.allocSizes.fftTimeBuf);
        accum  += addIfAlive(l.fftOutBuf,    l.allocSizes.fftOutBuf);
        accum  += addIfAlive(l.prevInputBuf, l.allocSizes.prevInputBuf);
        accum  += addIfAlive(l.accumBuf,     l.allocSizes.accumBuf);
        accum  += addIfAlive(l.accumReal,    l.allocSizes.accumReal);
        accum  += addIfAlive(l.accumImag,    l.allocSizes.accumImag);
        accum  += addIfAlive(l.inputAccBuf,  l.allocSizes.inputAccBuf);
        tail   += addIfAlive(l.tailOutputBuf,l.allocSizes.tailOutputBuf);

        snap.layerBufs[li]  = irFreq + fdl + accum + tail;
        snap.irFreqBytes   += irFreq;
        snap.fdlBytes      += fdl;
        snap.accumBytes    += accum;
        snap.tailBytes     += tail;
    }

    snap.directBytes = addIfAlive(m_directIRRev,
        static_cast<size_t>(m_directTapCount) * sizeof(double));
    snap.ringBytes   = addIfAlive(m_ringBuf,
        static_cast<size_t>(m_ringSize) * sizeof(double));
    return snap;
}
```

### B-2. SetImpulse シグネチャ（gen 削除、v23 から元に戻す）

```cpp
// ★ v24: シグネチャ変更なし（v22 の diagnosticGeneration は削除）
bool SetImpulse(const double* impulse, int irLen, int blockSize,
                double scale = 1.0,
                bool enableDirectHead = false,
                const FilterSpec* filterSpec = nullptr);
```

### B-3. SetImpulse() — IR_LOAD（gen 削除、getDiagnostics で IR_LAYOUT）

```cpp
bool MKLNonUniformConvolver::SetImpulse(...)
{
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    const uint64_t beforeMkl = convo::diag::allocatedBytes();
    const uint32_t beforeLost = convo::diag::lostFreeCount();
    const auto beforeOs = getProcessMemoryInfo();
#endif

    convo::publishAtomic(m_ready, false, std::memory_order_release);
    if (impulse == nullptr || irLen <= 0 || blockSize <= 0) return false;
    releaseAllLayers();

    // ... 既存の確保 + プリコンピュート ...

    convo::publishAtomic(m_ready, true, std::memory_order_release);

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    const uint64_t afterMkl = convo::diag::allocatedBytes();
    const uint32_t afterLost = convo::diag::lostFreeCount();
    const auto afterOs = getProcessMemoryInfo();
    const int64_t deltaMkl = static_cast<int64_t>(afterMkl) - static_cast<int64_t>(beforeMkl);
    const int32_t deltaLost = static_cast<int32_t>(afterLost) - static_cast<int32_t>(beforeLost);

    // IR_LOAD — gen なし（Generation は MEM_SNAP のみで管理）
    const int l0Part = m_numActiveLayers >= 1 ? m_layers[0].partSize : 0;
    const int l1Part = m_numActiveLayers >= 2 ? m_layers[1].partSize : 0;
    const int l2Part = m_numActiveLayers >= 3 ? m_layers[2].partSize : 0;
    diagLog(juce::String::formatted(
        "[IR_LOAD] NUC#%p irLen=%d blockSize=%d "
        "Layers=%d L0Part=%d L1Part=%d L2Part=%d "
        "directTaps=%d ringSize=%d "
        "MKL: before=%lluMB after=%lluMB delta=%lldMB "
        "lostFree=%u(+%d) | "
        "OS: beforePrivate=%lluMB afterPrivate=%lluMB live=%u",
        (void*)this, irLen, blockSize,
        m_numActiveLayers, l0Part, l1Part, l2Part,
        m_directTapCount, m_ringSize,
        (unsigned long long)(beforeMkl / (1024*1024)),
        (unsigned long long)(afterMkl / (1024*1024)),
        (long long)(deltaMkl / (1024*1024)),
        (unsigned)afterLost, (int)deltaLost,
        (unsigned long long)beforeOs.privateUsageMB,
        (unsigned long long)afterOs.privateUsageMB,
        (unsigned)liveCount.load(std::memory_order_relaxed)));

    // IR_LAYOUT — getDiagnostics は 1 回だけ（二重走査なし）
    const auto snap = getDiagnostics();
    diagLog(juce::String::formatted(
        "[IR_LAYOUT] NUC#%p "
        "IRFreq=%.0fMB FDL=%.0fMB Accum=%.0fMB Tail=%.0fMB "
        "Direct=%.0fMB Ring=%.0fMB Total=%.0fMB | "
        "L0=%.0fMB L1=%.0fMB L2=%.0fMB",
        (void*)this,
        snap.irFreqBytes / (1024.0*1024.0),
        snap.fdlBytes    / (1024.0*1024.0),
        snap.accumBytes  / (1024.0*1024.0),
        snap.tailBytes   / (1024.0*1024.0),
        snap.directBytes / (1024.0*1024.0),
        snap.ringBytes   / (1024.0*1024.0),
        snap.totalBytes() / (1024.0*1024.0),
        snap.layerBufs[0] / (1024.0*1024.0),
        snap.layerBufs[1] / (1024.0*1024.0),
        snap.layerBufs[2] / (1024.0*1024.0)));
#endif

    return true;
}
```

### B-4. releaseAllLayers() — IR_RELEASE（LayersBefore 追加）

```cpp
void MKLNonUniformConvolver::releaseAllLayers() noexcept
{
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    const uint64_t beforeMkl = convo::diag::allocatedBytes();
    const uint32_t beforeLost = convo::diag::lostFreeCount();
    const int layersBefore = m_numActiveLayers;  // ★ v24
    const auto beforeOs = getProcessMemoryInfo();
#endif

    // ... 解放ロジック（freeTracked 一本化）...

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    const uint64_t afterMkl = convo::diag::allocatedBytes();
    const uint32_t afterLost = convo::diag::lostFreeCount();
    const auto afterOs = getProcessMemoryInfo();
    const int64_t deltaMkl = static_cast<int64_t>(afterMkl) - static_cast<int64_t>(beforeMkl);
    const int32_t deltaLost = static_cast<int32_t>(afterLost) - static_cast<int32_t>(beforeLost);
    diagLog(juce::String::formatted(
        "[IR_RELEASE] NUC#%p "
        "MKL: before=%lluMB after=%lluMB delta=%lldMB "
        "LayersBefore=%d lostFree=%u(+%d) | "
        "OS: beforePrivate=%lluMB afterPrivate=%lluMB",
        (void*)this,
        (unsigned long long)(beforeMkl / (1024*1024)),
        (unsigned long long)(afterMkl / (1024*1024)),
        (long long)(deltaMkl / (1024*1024)),
        layersBefore,
        (unsigned)afterLost, (int)deltaLost,
        (unsigned long long)beforeOs.privateUsageMB,
        (unsigned long long)afterOs.privateUsageMB));
#endif
}
```

---

## 3. 全ログの v24 統一フォーマット

### IR_RELEASE

```
[IR_RELEASE] NUC#%p MKL: before=%lluMB after=%lluMB delta=%lldMB LayersBefore=%d lostFree=%u(+%d) | OS: beforePrivate=%lluMB afterPrivate=%lluMB
```

### IR_LOAD（gen なし）

```
[IR_LOAD] NUC#%p irLen=%d blockSize=%d Layers=%d L0Part=%d L1Part=%d L2Part=%d directTaps=%d ringSize=%d MKL: before=%lluMB after=%lluMB delta=%lldMB lostFree=%u(+%d) | OS: beforePrivate=%lluMB afterPrivate=%lluMB live=%u
```

### IR_LAYOUT（Layer 別合計追加）

```
[IR_LAYOUT] NUC#%p IRFreq=%.0fMB FDL=%.0fMB Accum=%.0fMB Tail=%.0fMB Direct=%.0fMB Ring=%.0fMB Total=%.0fMB | L0=%.0fMB L1=%.0fMB L2=%.0fMB
```

### MEM_SNAP（v23 から変更なし）

---

## 4. 出力例（v24）

```text
[IR_RELEASE] NUC#001 MKL: before=820MB after=110MB delta=-710MB LayersBefore=3 lostFree=18(+0) | OS: beforePrivate=2330MB afterPrivate=1620MB
[IR_LOAD]    NUC#001 irLen=327680 blockSize=4096 Layers=3 L0Part=4096 L1Part=32768 L2Part=262144 directTaps=32 ringSize=8192 MKL: before=110MB after=812MB delta=+702MB lostFree=18(+0) | OS: beforePrivate=1620MB afterPrivate=2330MB live=8
[IR_LAYOUT]  NUC#001 IRFreq=256MB FDL=420MB Accum=96MB Tail=16MB Direct=24MB Ring=8MB Total=820MB | L0=8MB L1=64MB L2=720MB
[MEM_SNAP]   PUBLISH gen=21 seq=5 | NUC(MKL only): live=8 alloc=820MB Peak(since reset)=1200MB totalA=28.0GB totalF=27.4GB lostFree=0 | Stereo=4 DSPCore=4 | Retire: pending=232 objBytes=12.8MB(sizeof) tracked=8/232 (3%) overflow=0 reclaim=12 | OS: Private=2330MB WorkingSet=2400MB | OtherPrivate=1510MB(CRT/JUCE/IPP/stacks/...)
```

**IR_LAYOUT の L2=720MB** → Layer2（Tail）が支配的と一目で分かる。

---

## 5. v23 からの改善点一覧

| # | v23（問題） | v24（修正） |
|:--|:----------|:-----------|
| 1 | `diagnosticGeneration` で API 変更 | **削除。Generation は MEM_SNAP のみで管理** |
| 2 | `IR_LAYOUT` で Layer を二重走査 | **`getDiagnostics()` に内訳フィールド追加 → 一回の走査で完了** |
| 3 | `freeTracked()` / `freeTrackedSize()` の 2 種類 | **`freeTracked()` 一本化（size=0 も対応）** |
| 4 | `IR_RELEASE` に解放前 Layer 数なし | **`LayersBefore=%d` 追加** |
| 5 | `IR_LAYOUT` に Layer 別合計なし | **`L0/L1/L2` 追加** |
| 6 | `OtherPrivate` の意味が不明瞭 | **コメントで全未計測領域と明記** |
