# メモリ占有調査のためのインストルメンテーション改修案 v42（最終版）

---

**目的**: 2.33GB メモリ占有の原因特定。推定ではなく実測で確定させる。
**v41 からの変更**: 5 点の改善（設計変更なし、保守性・明確性向上のみ）。

---

## 0. v41 の問題点と v42 での修正方針

| # | v41 の問題 | v42 の修正 |
|:--|:----------|:----------|
| 1 | `liveCount` が `std::atomic<int>` → 負数はバグ。NUC は `uint32_t` | **`std::atomic<uint32_t>` に統一（StereoConvolver / DSPCore）** |
| 2 | `IR_RELEASE` の `live=%u` が解放後か解放前か不明 | **`liveBefore=%u` に変更。解放前にスナップショットから取得** |
| 3 | `computeOtherPrivate()` が `uint64_t` で負値になりうる | **`std::max` で下限保護（既に v31 で実装済み。確認のみ）** |
| 4 | `kReservedDiagSeq` / `kFirstRuntimeDiagSeq` が個別定数 | **`enum : uint64_t` に集約** |
| 5 | `lostFreeCount` と `zeroAllocSizeCount` の役割（v39 で解決済み。確認のみ） | **設計書に明確に記載** |

---

## 1. Patch A: liveCount — uint32_t 統一

### MKLNonUniformConvolver.h（既存）

```cpp
static std::atomic<uint32_t> liveCount;  // ★ 既に uint32_t
```

### ConvolverProcessor.h（修正）

```cpp
struct StereoConvolver : public convo::AlignedBase {
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    static std::atomic<uint32_t> liveCount;  // ★ v42: int → uint32_t
#endif
    // ...
};
```

### AudioEngine.h（修正）

```cpp
struct DSPCore {
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    static std::atomic<uint32_t> liveCount;  // ★ v42: int → uint32_t
#endif
    // ...
};
```

---

## 2. Patch B: IR_RELEASE liveBefore に変更

### logIrRelease — liveBefore 追加

```cpp
namespace {

static void logIrRelease(
    const MKLNonUniformConvolver* nuc,
    uint64_t diagSeq,
    uint64_t beforeMkl,
    uint32_t beforeLost,
    const ProcessMemoryInfo& beforeOs,
    const NucDiagnosticsSnapshot& beforeSnap,
    const ProcessMemoryInfo& afterReleaseOs,
    uint32_t liveBefore) noexcept  // ★ v42: 解放前 liveCount
{
    // ...
    diagLogNonRt(juce::String::formatted(
        "[IR_RELEASE] NUC#%p seq=%llu "
        "MKL: before=%lluMB after=%lluMB delta=%lldMB "
        "LayersBefore=%d TotalBefore=%.0fMB(persistent) "
        "lostFree=%u(+%d) liveBefore=%u | "
        "OS: beforePrivate=%lluMB afterPrivate=%lluMB",
        // ...
        (unsigned)liveBefore));
}

} // namespace
```

### SetImpulse — liveBefore を解放前に取得

```cpp
bool MKLNonUniformConvolver::SetImpulse(...)
{
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    const uint64_t diagSeq = globalDiagSeq.fetch_add(1, ...) + kFirstRuntimeDiagSeq;
    const uint64_t beforeMkl = convo::diag::allocatedBytes();
    const uint32_t beforeLost = convo::diag::lostFreeCount();
    const auto beforeOs = getProcessMemoryInfo();
    const auto beforeSnap = getDiagnostics();
    const uint32_t liveBefore = liveCount.load(std::memory_order_relaxed);  // ★ v42: 解放前
#endif

    convo::publishAtomic(m_ready, false, ...);
    if (impulse == nullptr || ...) return false;

    releaseAllLayers();

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    const auto afterReleaseOs = getProcessMemoryInfo();
    logIrRelease(diagSeq, beforeMkl, beforeLost, beforeOs, beforeSnap, afterReleaseOs, liveBefore);
#endif

    // ...
}
```

### IR_RELEASE フォーマット（liveBefore）

```
[IR_RELEASE] NUC#%p seq=%llu MKL: before=%lluMB after=%lluMB delta=%lldMB LayersBefore=%d TotalBefore=%.0fMB(persistent) lostFree=%u(+%d) liveBefore=%u | OS: beforePrivate=%lluMB afterPrivate=%lluMB
```

---

## 3. Patch C: seq 定数を enum に

### MKLNonUniformConvolver.h

```cpp
class MKLNonUniformConvolver {
public:
    static std::atomic<uint32_t> liveCount;
    static std::atomic<uint64_t> globalDiagSeq;

    /// ★ v42: 診断シーケンス番号の定数。
    enum : uint64_t {
        kDiagSeqReserved = 0,      ///< デストラクタ等、SetImpulse 以外の経路
        kDiagSeqFirstRuntime = 1   ///< SetImpulse の最初の seq 値 (fetch_add(1) + kDiagSeqFirstRuntime)
    };

    // ...
};
```

### SetImpulse

```cpp
const uint64_t diagSeq = globalDiagSeq.fetch_add(1, std::memory_order_relaxed)
                       + kDiagSeqFirstRuntime;
```

---

## 4. 診断フローの完全版（v42 確定）

```
SetImpulse() 開始
├─ globalDiagSeq.fetch_add(1) + kDiagSeqFirstRuntime → seq=105
├─ beforeMkl = allocatedBytes()
├─ beforeSnap = getDiagnostics() → TotalBefore=820MB, LayersBefore=3
├─ liveBefore = liveCount.load() → 8
├─ releaseAllLayers()
│  ├─ Layer::freeAll() → freeTracked(allocSizes)
│  └─ releaseAllLayers() → freeTracked(ringBuf/direct*)
├─ afterReleaseOs = getProcessMemoryInfo()
├─ logIrRelease(seq, beforeMkl, beforeLost, beforeOs, beforeSnap, afterReleaseOs, liveBefore)
│  → [IR_RELEASE] seq=105 MKL:-710MB LayersBefore=3 TotalBefore=820MB liveBefore=8 | OS:-710MB
├─ 新規確保 (DIAG_MKL_MALLOC + allocSizes 保存)
│  └─ エラー時 → releaseAllLayers() で安全に解放
├─ m_ready = true
├─ afterMkl / afterLost / afterLoadOs
├─ [IR_LOAD] seq=105 Layers=3 L0Part=4096 L1Part=32768 L2Part=262144 delta=+702MB live=8
└─ [IR_LAYOUT] seq=105 IRFreq=256 FDL=420 Accum=96 Tail=16 Direct=24 Ring=8 Total=820 | L0=8 L1=64 L2=720
```

---

## 5. 全ログフォーマット（v42 確定版）

### IR_RELEASE（liveBefore）
```
[IR_RELEASE] NUC#%p seq=%llu MKL: before=%lluMB after=%lluMB delta=%lldMB LayersBefore=%d TotalBefore=%.0fMB(persistent) lostFree=%u(+%d) liveBefore=%u | OS: beforePrivate=%lluMB afterPrivate=%lluMB
```

### IR_LOAD（v41 から変更なし）
```
[IR_LOAD] NUC#%p seq=%llu irLen=%d blockSize=%d Layers=%d L0Part=%d L1Part=%d L2Part=%d directTaps=%d ringSize=%d MKL: before=%lluMB after=%lluMB delta=%lldMB lostFree=%u(+%d) live=%u
```

### IR_LAYOUT（v41 から変更なし）
```
[IR_LAYOUT] NUC#%p seq=%llu IRFreq=%.0fMB FDL=%.0fMB Accum=%.0fMB Tail=%.0fMB Direct=%.0fMB Ring=%.0fMB Total=%.0fMB(persistent data buffers only) | L0=%.0fMB L1=%.0fMB L2=%.0fMB
```

### MEM_SNAP（v41 から変更なし）
```
[MEM_SNAP] PUBLISH gen=%d seq=%d | NUC(MKL only): live=%u alloc=%.0fMB Peak(since reset)=%.0fMB totalA=%.0fGB totalF=%.0fGB lostFree=%u zeroAllocSize=%u(delta=%+d) | Stereo=%u DSPCore=%u | Retire: pending=%u trackedPendingBytes=%.1fMB(diag only) trackedPending=%u/%u (%.0f%%) overflow=%llu reclaim=%llu | OS: Private=%lluMB WorkingSet=%lluMB | OtherPrivate=%.0fMB(JUCE/CRT/IPP/threads/...)
```

---

## 6. 出力例（v42 最終版）

```text
[IR_RELEASE] NUC#001 seq=105 MKL: before=820MB after=110MB delta=-710MB LayersBefore=3 TotalBefore=820MB(persistent) lostFree=18(+0) liveBefore=8 | OS: beforePrivate=2330MB afterPrivate=1620MB
[IR_LOAD]    NUC#001 seq=105 irLen=327680 blockSize=4096 Layers=3 L0Part=4096 L1Part=32768 L2Part=262144 directTaps=32 ringSize=8192 MKL: before=110MB after=812MB delta=+702MB lostFree=18(+0) live=8
[IR_LAYOUT]  NUC#001 seq=105 IRFreq=256MB FDL=420MB Accum=96MB Tail=16MB Direct=24MB Ring=8MB Total=820MB(persistent data buffers only) | L0=8MB L1=64MB L2=720MB
[MEM_SNAP]   PUBLISH gen=21 seq=5 | NUC(MKL only): live=8 alloc=820MB Peak(since reset)=1200MB totalA=28.0GB totalF=27.4GB lostFree=0 zeroAllocSize=0(delta=+0) | Stereo=4 DSPCore=4 | Retire: pending=232 trackedPendingBytes=12.8MB(diag only) trackedPending=8/232 (3%) overflow=0 reclaim=12 | OS: Private=2330MB WorkingSet=2400MB | OtherPrivate=1510MB(JUCE/CRT/IPP/threads/...)
```

---

## 7. v41 からの改善点一覧

| # | v41（問題） | v42（修正） |
|:--|:----------|:-----------|
| 1 | `liveCount` が `std::atomic<int>`（負数はバグ） | **`std::atomic<uint32_t>` に統一（NUC / StereoConvolver / DSPCore）** |
| 2 | `IR_RELEASE` の `live=%u` が解放前か後か不明 | **`liveBefore=%u` に変更。解放前に `liveCount.load()` で取得** |
| 3 | `computeOtherPrivate()` の uint64_t 負値対策（v31 で `std::max<int64_t>` 済み） | **確認のみ。設計書に明記** |
| 4 | `kReservedDiagSeq` / `kFirstRuntimeDiagSeq` が個別定数 | **`enum : uint64_t` に集約** |
| 5 | `lostFreeCount` / `zeroAllocSizeCount` の役割（v39 で解決済み） | **確認のみ。設計書に明記** |
