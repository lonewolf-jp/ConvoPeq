# メモリ占有調査のためのインストルメンテーション改修案 v33（最終版）

---

**目的**: 2.33GB メモリ占有の原因特定。推定ではなく実測で確定させる。
**v32 からの変更**: 5 点の修正。

---

## 0. v32 の問題点と v33 での修正方針

| # | v32 の問題 | v33 の修正 | 優先度 |
|:--|:----------|:----------|:------|
| 1 | `afterOs` / `afterOs2` が何のタイミングか不明瞭 | **`afterReleaseOs` / `afterLoadOs` に改名** | ★★★ |
| 2 | `logIrRelease()` が private static member（`this` 不使用） | **無名名前空間の free function に変更** | ★★☆ |
| 3 | `trackedPendingEntries_` の増減ルールが不明 | **設計書に明記: enqueue で +1、reclaim で -1、`pendingRetireBytes` と同期** | ★★☆ |
| 4 | `pendingRetireBytes()` のコメントが不十分 | **"Does NOT include allocator overhead. Does NOT represent process heap usage." まで明記** | ★☆☆ |
| 5 | IR_RELEASE TotalBefore に `(persistent)` なし | **`TotalBefore=820MB(persistent)` に変更** | ★☆☆ |

---

## 1. Patch: 最終コード確定

### 1-1. SetImpulse — afterReleaseOs / afterLoadOs（★ 改名）

```cpp
bool MKLNonUniformConvolver::SetImpulse(...)
{
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    const uint64_t diagSeq = globalDiagSeq.fetch_add(1, std::memory_order_relaxed) + 1;
    const uint64_t beforeMkl = convo::diag::allocatedBytes();
    const uint32_t beforeLost = convo::diag::lostFreeCount();
    const auto beforeOs = getProcessMemoryInfo();
    const auto beforeSnap = getDiagnostics();  // ← 解放前スナップショット
#endif

    convo::publishAtomic(m_ready, false, std::memory_order_release);
    if (impulse == nullptr || irLen <= 0 || blockSize <= 0) return false;

    releaseAllLayers();

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    // ★ 解放直後の OS メモリ（IR_RELEASE 用）
    const auto afterReleaseOs = getProcessMemoryInfo();
    logIrRelease(diagSeq, beforeMkl, beforeLost, beforeOs, beforeSnap, afterReleaseOs);
#endif

    // ... 既存の確保 + プリコンピュート ...

    convo::publishAtomic(m_ready, true, std::memory_order_release);

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    const uint64_t afterMkl = convo::diag::allocatedBytes();
    const uint32_t afterLost = convo::diag::lostFreeCount();

    // ★ 確保完了後の OS メモリ（IR_LOAD / IR_LAYOUT 用）
    const auto afterLoadOs = getProcessMemoryInfo();

    // IR_LOAD
    diagLog(juce::String::formatted(...));

    // IR_LAYOUT
    const auto snap = getDiagnostics();
    diagLog(juce::String::formatted(...));
#endif

    return true;
}
```

### 1-2. logIrRelease — 無名名前空間の free function

```cpp
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS

namespace {  // ★ anonymous namespace

// IR_RELEASE ログ出力。this を必要としない。
static void logIrRelease(
    const MKLNonUniformConvolver* nuc,
    uint64_t diagSeq,
    uint64_t beforeMkl,
    uint32_t beforeLost,
    const ProcessMemoryInfo& beforeOs,
    const NucDiagnosticsSnapshot& beforeSnap,
    const ProcessMemoryInfo& afterReleaseOs) noexcept
{
    const uint64_t afterMkl = convo::diag::allocatedBytes();
    const uint32_t afterLost = convo::diag::lostFreeCount();
    const int64_t deltaMkl = static_cast<int64_t>(afterMkl)
                             - static_cast<int64_t>(beforeMkl);
    const int32_t deltaLost = static_cast<int32_t>(afterLost)
                              - static_cast<int32_t>(beforeLost);

    diagLog(juce::String::formatted(
        "[IR_RELEASE] NUC#%p seq=%llu "
        "MKL: before=%lluMB after=%lluMB delta=%lldMB "
        "LayersBefore=%d TotalBefore=%.0fMB(persistent) "
        "lostFree=%u(+%d) | "
        "OS: beforePrivate=%lluMB afterPrivate=%lluMB",
        (void*)nuc,
        (unsigned long long)diagSeq,
        (unsigned long long)(beforeMkl / (1024*1024)),
        (unsigned long long)(afterMkl / (1024*1024)),
        (long long)(deltaMkl / (1024*1024)),
        beforeSnap.numActiveLayers,
        beforeSnap.totalBytes() / (1024.0 * 1024.0),
        (unsigned)afterLost, (int)deltaLost,
        (unsigned long long)beforeOs.privateUsageMB,
        (unsigned long long)afterReleaseOs.privateUsageMB));
}

} // namespace
#endif
```

### 1-3. ISRRetireRouter — trackedPendingEntries_ 増減ルール

```cpp
class ISRRetireRouter : public convo::IEpochProvider {
    /// ★ trackedPendingEntries_: objectBytes > 0 のエントリ数。
    ///   enqueueRetire() で objectBytes != 0 のエントリがキューに入るたびに +1。
    ///   tryReclaim() でそのエントリが reclaim されるたびに -1。
    ///   pendingRetireCount() と整合性を保つため、増減は必ずペアで行う。
    std::atomic<uint32_t> m_trackedPendingEntries_{0};
};
```

### 1-4. pendingRetireBytes() コメント（拡充）

```cpp
/// ★ Diagnostic estimate only.
///   Returns the sum of object sizes for which a non-zero objectBytes
///   was provided at enqueue time.
///   Does NOT include allocator overhead (malloc bookkeeping, alignment padding).
///   Does NOT represent process heap usage of the retire queue.
///   See also trackedRatio() for coverage.
[[nodiscard]] uint64_t pendingRetireBytes() const noexcept override
{
    return convo::consumeAtomic(m_pendingRetireBytes_, std::memory_order_acquire);
}
```

---

## 2. 全ログフォーマット（v33 確定版）

### IR_RELEASE（TotalBefore に (persistent) 追記）
```
[IR_RELEASE] NUC#%p seq=%llu MKL: before=%lluMB after=%lluMB delta=%lldMB LayersBefore=%d TotalBefore=%.0fMB(persistent) lostFree=%u(+%d) | OS: beforePrivate=%lluMB afterPrivate=%lluMB
```

### IR_LOAD（v32 から変更なし）
```
[IR_LOAD] NUC#%p seq=%llu irLen=%d blockSize=%d ...
```

### IR_LAYOUT（v32 から変更なし）
```
[IR_LAYOUT] NUC#%p seq=%llu IRFreq=%.0fMB ... Total=%.0fMB(persistent data buffers only) | L0=...
```

### MEM_SNAP（v32 から変更なし）
```
[MEM_SNAP] PUBLISH gen=%d seq=%d | ... trackedPendingBytes=... trackedPending=%u/%u (%.0f%%) ...
```

---

## 3. 出力例（v33 最終版）

```text
[IR_RELEASE] NUC#001 seq=105 MKL: before=820MB after=110MB delta=-710MB LayersBefore=3 TotalBefore=820MB(persistent) lostFree=18(+0) | OS: beforePrivate=2330MB afterPrivate=1620MB
[IR_LOAD]    NUC#001 seq=105 irLen=327680 blockSize=4096 Layers=3 L0Part=4096 L1Part=32768 L2Part=262144 directTaps=32 ringSize=8192 MKL: before=110MB after=812MB delta=+702MB lostFree=18(+0) live=8
[IR_LAYOUT]  NUC#001 seq=105 IRFreq=256MB FDL=420MB Accum=96MB Tail=16MB Direct=24MB Ring=8MB Total=820MB(persistent data buffers only) | L0=8MB L1=64MB L2=720MB
[MEM_SNAP]   PUBLISH gen=21 seq=5 | NUC(MKL only): live=8 alloc=820MB Peak(since reset)=1200MB totalA=28.0GB totalF=27.4GB lostFree=0 | Stereo=4 DSPCore=4 | Retire: pending=232 trackedPendingBytes=12.8MB(diag only: sizeof tracked entries, not actual heap) trackedPending=8/232 (3%) overflow=0 reclaim=12 | OS: Private=2330MB WorkingSet=2400MB | OtherPrivate=1510MB(JUCE/CRT/IPP/threads/...)
```

---

## 4. v32 からの改善点一覧

| # | v32（問題） | v33（修正） |
|:--|:----------|:-----------|
| 1 | `afterOs` / `afterOs2` がタイミング不明瞭 | **`afterReleaseOs` / `afterLoadOs` に改名** |
| 2 | `logIrRelease()` が private static member（this 不使用） | **無名名前空間の free function に変更** |
| 3 | `trackedPendingEntries_` の増減ルールが不明 | **enqueue で +1、reclaim で -1、pendingRetireBytes と同期することを明記** |
| 4 | `pendingRetireBytes()` のコメントが不十分 | **"Does NOT include allocator overhead. Does NOT represent process heap usage." を追加** |
| 5 | IR_RELEASE TotalBefore に注釈なし | **`TotalBefore=820MB(persistent)` に変更** |
