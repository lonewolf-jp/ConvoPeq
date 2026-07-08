# メモリ占有調査のためのインストルメンテーション改修案 v32（最終版）— 確定実装版

---

**目的**: 2.33GB メモリ占有の原因特定。推定ではなく実測で確定させる。
**v31 からの変更**: 設計書の不備（`logIrRelease` の未記載、`TotalBefore` の取得タイミング不整合など）を修正。レビュー指摘 9 点を反映。

---

## 0. v31 の問題点と v32 での修正方針

| # | v31 の問題 | v32 の修正 |
|:--|:----------|:----------|
| 1 | `logIrRelease()` の実装と SetImpulse の流れが設計書から欠落 → `TotalBefore` の取得タイミングが確認できない | **`logIrRelease()` と SetImpulse の完全な流れを明記。解放前に `beforeSnap = getDiagnostics()` でスナップショット取得** |
| 2 | `logIrRelease()` が `NucDiagnosticsSnapshot` を受け取らず内部で状態を読む | **シグネチャを `static void logIrRelease(uint64_t seq, ..., const NucDiagnosticsSnapshot& beforeSnap)` に** |
| 3 | `globalDiagSeq` の fetch_add(1)+1 の意図が不明瞭 | **コメント追加: "seq=0 is reserved for non-SetImpulse events (destructor)"** |
| 4 | NUC レベルバッファのサイズが再計算方式と allocSizes 方式で混在 | **コメントで明確に区別: Layer→allocSizes, NUC→動的計算** |
| 5 | `pendingRetireBytes()` が「全滞留バイト数」と誤解されうる | **コメント強化: "tracked object bytes only, not actual heap usage"** |
| 6 | MEM_SNAP の `tracked=8/232` が何の ratio か不明 | **`trackedPending=8/232` に変更** |
| 7 | IR_LAYOUT Total が何を含むか不明 | **コメント追加: "Persistent data buffers only"** |
| 8 | freeAll() の allocSizes={} の順序確認 | **確保。freeTracked 後に allocSizes={}** |
| 9 | `DSPCoreLifecycle.cpp` のパスが不完全 | **`src/audioengine/AudioEngine.Processing.DSPCoreLifecycle.cpp` に統一** |

---

## 1. 確定実装コード（全流れ）

### 1-1. SetImpulse — 完全な流れ（★ 最重要: TotalBefore は解放前に取得）

```cpp
bool MKLNonUniformConvolver::SetImpulse(const double* impulse, int irLen, int blockSize, double scale,
                                        bool enableDirectHead, const FilterSpec* filterSpec)
{
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    const uint64_t diagSeq = globalDiagSeq.fetch_add(1, std::memory_order_relaxed) + 1;
    // seq=0 is reserved for non-SetImpulse events (e.g. destructor)
    const uint64_t beforeMkl = convo::diag::allocatedBytes();
    const uint32_t beforeLost = convo::diag::lostFreeCount();
    const auto beforeOs = getProcessMemoryInfo();

    // ★★★ 解放前スナップショット（これが正しい TotalBefore / LayersBefore）★★★
    const auto beforeSnap = getDiagnostics();
#endif

    convo::publishAtomic(m_ready, false, std::memory_order_release);
    if (impulse == nullptr || irLen <= 0 || blockSize <= 0) return false;

    releaseAllLayers();  // 純粋な解放処理（診断ログなし）

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    // ★ 解放後のOSメモリを取得（afterOs は logIrRelease で使用）
    const auto afterOs = getProcessMemoryInfo();
    logIrRelease(diagSeq, beforeMkl, beforeLost, beforeOs, beforeSnap, afterOs);
#endif

    // ... 既存の確保 + プリコンピュート（全 mkl_malloc→DIAG_MKL_MALLOC、allocSizes 保存）...

    convo::publishAtomic(m_ready, true, std::memory_order_release);

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    const uint64_t afterMkl = convo::diag::allocatedBytes();
    const uint32_t afterLost = convo::diag::lostFreeCount();
    const auto afterOs2 = getProcessMemoryInfo();

    // IR_LOAD
    diagLog(juce::String::formatted(
        "[IR_LOAD] NUC#%p seq=%llu irLen=%d blockSize=%d ...", ...));

    // IR_LAYOUT（解放後スナップショット）
    const auto snap = getDiagnostics();
    diagLog(juce::String::formatted(
        "[IR_LAYOUT] NUC#%p seq=%llu IRFreq=%.0fMB ... | L0=%.0fMB L1=%.0fMB L2=%.0fMB", ...));
#endif

    return true;
}
```

### 1-2. logIrRelease — static 関数（解放前スナップショットを受け取る）

```cpp
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
/// IR_RELEASE ログ出力。
/// 呼び出し元は解放前のスナップショット beforeSnap を引数で渡す。
/// 本関数は getDiagnostics() を呼ばない（呼び出し元で取得済み）。
static void logIrRelease(
    const MKLNonUniformConvolver* nuc,
    uint64_t diagSeq,
    uint64_t beforeMkl,
    uint32_t beforeLost,
    const ProcessMemoryInfo& beforeOs,
    const NucDiagnosticsSnapshot& beforeSnap,
    const ProcessMemoryInfo& afterOs) noexcept
{
    const uint64_t afterMkl = convo::diag::allocatedBytes();
    const uint32_t afterLost = convo::diag::lostFreeCount();
    const int64_t deltaMkl = static_cast<int64_t>(afterMkl) - static_cast<int64_t>(beforeMkl);
    const int32_t deltaLost = static_cast<int32_t>(afterLost) - static_cast<int32_t>(beforeLost);

    diagLog(juce::String::formatted(
        "[IR_RELEASE] NUC#%p seq=%llu "
        "MKL: before=%lluMB after=%lluMB delta=%lldMB "
        "LayersBefore=%d TotalBefore=%.0fMB "
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
        (unsigned long long)afterOs.privateUsageMB));
}
#endif
```

### 1-3. releaseAllLayers — 純粋な解放処理

```cpp
void MKLNonUniformConvolver::releaseAllLayers() noexcept
{
    // ... 既存の guard チェック ...

    for (int i = 0; i < kNumLayers; ++i)
        m_layers[i].freeAll();  // freeAll 内で freeTracked（allocSizes 使用）
    m_numActiveLayers = 0;
    m_latency         = 0;

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    // ★ NUC レベルバッファ（サイズはメンバ変数から動的計算）
    freeTracked(m_ringBuf, static_cast<size_t>(m_ringSize) * sizeof(double));
    freeTracked(m_directIRRev, static_cast<size_t>(m_directTapCount) * sizeof(double));
    freeTracked(m_directHistory, static_cast<size_t>(m_directHistLen) * sizeof(double));
    freeTracked(m_directWindow, static_cast<size_t>(m_directHistLen + m_directMaxBlock) * sizeof(double));
    freeTracked(m_directOutBuf, static_cast<size_t>(m_directMaxBlock) * sizeof(double));
#else
    // ... 既存の mkl_free ...
#endif

    // ... 既存の状態リセット ...
}
```

### 1-4. Layer::freeAll — allocSizes 使用

```cpp
void MKLNonUniformConvolver::Layer::freeAll() noexcept
{
    // ... 既存の fftPlanOwner, fftWorkBuf 解放 ...

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    // ★ allocSizes からサイズ取得（確保時に保存した値）
    freeTracked(irFreqDomain,  allocSizes.irFreqDomain);
    freeTracked(irFreqReal,    allocSizes.irFreqReal);
    // ... 全 14 個 ...
    freeTracked(tailOutputBuf, allocSizes.tailOutputBuf);
#endif

    // ... 既存の状態リセット ...

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    allocSizes = {};  // ★ 解放後にゼロ初期化（freeTracked 実行後のため安全）
#endif
}
```

### 1-5. MEM_SNAP の trackedPending 表記

```
[MEM_SNAP] PUBLISH gen=%d seq=%d | NUC(MKL only): live=%u alloc=%.0fMB Peak(since reset)=%.0fMB totalA=%.0fGB totalF=%.0fGB lostFree=%u | Stereo=%d DSPCore=%d | Retire: pending=%u trackedPendingBytes=%.1fMB(diag only: sizeof tracked entries, not actual heap) trackedPending=%u/%u (%.0f%%) overflow=%llu reclaim=%llu | OS: Private=%lluMB WorkingSet=%lluMB | OtherPrivate=%.0fMB(JUCE/CRT/IPP/threads/...)
```

### 1-6. IR_LAYOUT Total の注記

IR_LAYOUT の `Total` は永続バッファ（IRFreq / FDL / Accum / Tail / Direct / Ring）の合計。
以下は含まない: IPP FFT plan workspaces, MKL 内部管理メモリ, CRT heap, VirtualAlloc.

```
[IR_LAYOUT] NUC#%p seq=%llu IRFreq=%.0fMB FDL=%.0fMB Accum=%.0fMB Tail=%.0fMB Direct=%.0fMB Ring=%.0fMB Total=%.0fMB(persistent data buffers only) | L0=%.0fMB L1=%.0fMB L2=%.0fMB
```

---

## 2. 全ログフォーマット（v32 確定版）

### IR_RELEASE
```
[IR_RELEASE] NUC#%p seq=%llu MKL: before=%lluMB after=%lluMB delta=%lldMB LayersBefore=%d TotalBefore=%.0fMB lostFree=%u(+%d) | OS: beforePrivate=%lluMB afterPrivate=%lluMB
```

### IR_LOAD
```
[IR_LOAD] NUC#%p seq=%llu irLen=%d blockSize=%d Layers=%d L0Part=%d L1Part=%d L2Part=%d directTaps=%d ringSize=%d MKL: before=%lluMB after=%lluMB delta=%lldMB lostFree=%u(+%d) live=%u
```

### IR_LAYOUT
```
[IR_LAYOUT] NUC#%p seq=%llu IRFreq=%.0fMB FDL=%.0fMB Accum=%.0fMB Tail=%.0fMB Direct=%.0fMB Ring=%.0fMB Total=%.0fMB(persistent data buffers only) | L0=%.0fMB L1=%.0fMB L2=%.0fMB
```

### MEM_SNAP
```
[MEM_SNAP] PUBLISH gen=%d seq=%d | NUC(MKL only): live=%u alloc=%.0fMB Peak(since reset)=%.0fMB totalA=%.0fGB totalF=%.0fGB lostFree=%u | Stereo=%d DSPCore=%d | Retire: pending=%u trackedPendingBytes=%.1fMB(diag only: sizeof tracked entries, not actual heap) trackedPending=%u/%u (%.0f%%) overflow=%llu reclaim=%llu | OS: Private=%lluMB WorkingSet=%lluMB | OtherPrivate=%.0fMB(JUCE/CRT/IPP/threads/...)
```

---

## 3. 出力例（v32 最終版）

```text
[IR_RELEASE] NUC#001 seq=105 MKL: before=820MB after=110MB delta=-710MB LayersBefore=3 TotalBefore=820MB lostFree=18(+0) | OS: beforePrivate=2330MB afterPrivate=1620MB
[IR_LOAD]    NUC#001 seq=105 irLen=327680 blockSize=4096 Layers=3 L0Part=4096 L1Part=32768 L2Part=262144 directTaps=32 ringSize=8192 MKL: before=110MB after=812MB delta=+702MB lostFree=18(+0) live=8
[IR_LAYOUT]  NUC#001 seq=105 IRFreq=256MB FDL=420MB Accum=96MB Tail=16MB Direct=24MB Ring=8MB Total=820MB(persistent data buffers only) | L0=8MB L1=64MB L2=720MB
[MEM_SNAP]   PUBLISH gen=21 seq=5 | NUC(MKL only): live=8 alloc=820MB Peak(since reset)=1200MB totalA=28.0GB totalF=27.4GB lostFree=0 | Stereo=4 DSPCore=4 | Retire: pending=232 trackedPendingBytes=12.8MB(diag only: sizeof tracked entries, not actual heap) trackedPending=8/232 (3%) overflow=0 reclaim=12 | OS: Private=2330MB WorkingSet=2400MB | OtherPrivate=1510MB(JUCE/CRT/IPP/threads/...)
```

**TotalBefore=820MB** → 解放前に取得した正しい値。IR_LAYOUT Total=820MB と一致。
**trackedPending=8/232** → 232 エントリ中 8 エントリのみサイズ追跡中。
**trackedPendingBytes=12.8MB(diag only: sizeof tracked entries, not actual heap)** → 誤解防止。
**persistent data buffers only** → IPP/CRT 等は含まないことを明記。

---

## 4. 実装手順（確定版）

| 手順 | ファイル | 変更内容 | 行数目安 |
|:----|:--------|:--------|:---------|
| 1 | `src/core/IRetireProvider.h` | `virtual uint64_t pendingRetireBytes() const noexcept { return 0; }` 追加 | +2行 |
| 2 | `src/DiagnosticsConfig.h` | MklAllocStats + diagMklMalloc/Free + freeTracked + addIfAlive + updateAtomicMaximum64 + DIAG_MKL_* マクロ + computeOtherPrivate | ~75行 |
| 3 | `src/MKLNonUniformConvolver.h` | LayerAllocSizes + NucDiagnosticsSnapshot(拡張版) + liveCount + globalDiagSeq + getDiagnostics + logIrRelease(friend宣言不要: static private) | ~25行 |
| 4 | `src/MKLNonUniformConvolver.cpp` | ctor/dtor liveCount + globalDiagSeq定義 + 全28箇所 mkl_malloc→DIAG_MKL_MALLOC + allocSizes保存 + freeAll freeTracked + releaseAllLayers freeTracked + logIrRelease + IR_RELEASE/IR_LOAD/IR_LAYOUT | ~160行 |
| 5 | `src/DeferredDeletionQueue.h` | DeletionEntry に `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` で `objectBytes` | ~5行 |
| 6 | `src/audioengine/ISRRetireRouter.h` | `pendingRetireBytes()` override + `m_pendingRetireBytes_` + `trackedPendingEntries_` + `trackedRatio()` | ~15行 |
| 7 | `src/audioengine/ISRRetireRouter.cpp` | enqueueRetire/tryReclaim での m_pendingRetireBytes_ 更新 | ~15行 |
| 8 | `src/audioengine/AudioEngine.Timer.cpp` | MEM_SNAP ログ | ~25行 |
| 9 | `src/audioengine/AudioEngine.Processing.DSPCoreLifecycle.cpp` | DSPCore liveCount + 確保量ログ | ~15行 |

**合計: 9 ファイル変更 / 約 337 行追加**
**変更不要**: `ConvolverProcessor.h`（6 引数のまま動作）

---

## 5. v31 からの改善点一覧

| # | v31（問題） | v32（修正） |
|:--|:----------|:-----------|
| 1 | `logIrRelease()` と SetImpulse の流れが設計書から欠落 | **完全なコードを記載。解放前に beforeSnap 取得、logIrRelease に渡す流れを明示** |
| 2 | `logIrRelease()` が `getDiagnostics()` を内部で呼ぶ可能性 | **`const NucDiagnosticsSnapshot& beforeSnap` を引数で受け取る static 関数に** |
| 3 | globalDiagSeq fetch_add(1)+1 の意図不明 | **コメント追加: "seq=0 is reserved for non-SetImpulse events"** |
| 4 | Layer/NUC のサイズ管理方式が異なる旨の注釈なし | **コメントで明確に区別: "allocSizes 使用" / "メンバ変数から動的計算"** |
| 5 | pendingRetireBytes の意味が誤解されうる | **コメント強化 + MEM_SNAP の trackedPendingBytes(diag only) に説明文追加** |
| 6 | MEM_SNAP の tracked=8/232 が何の比率か不明 | **`trackedPending=8/232` に変更** |
| 7 | IR_LAYOUT Total が何を含むか不明 | **`(persistent data buffers only)` を追加** |
| 8 | freeAll() の allocSizes={} の順序確認 | **freeTracked 後に allocSizes={} であることを確認** |
| 9 | DSPCoreLifecycle.cpp のパス不備 | **`src/audioengine/AudioEngine.Processing.DSPCoreLifecycle.cpp` に統一** |
