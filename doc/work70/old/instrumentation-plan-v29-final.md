# メモリ占有調査のためのインストルメンテーション改修案 v29（最終版）— 責務分離完了版

---

**目的**: 2.33GB メモリ占有の原因特定。推定ではなく実測で確定させる。
**v28 からの変更**: 4 点の修正。

---

## 0. v28 の問題点と v29 での修正方針

| # | v28 の問題 | v29 の修正 |
|:--|:----------|:----------|
| 1 | `releaseAllLayers(uint64_t diagSeq)` が診断コンテキストを解放処理に持ち込む（責務混在） | **`releaseAllLayers()` は純粋な解放処理に戻す。診断ログは呼び出し元が個別に実行** |
| 2 | `globalDiagSeq` が 0 開始 → Destructor の seq=0 と区別不可 | **`fetch_add(1)` の結果に +1 して 1 開始。Destructor は明示的に 0** |
| 3 | `ready` が常に 1 → 診断価値なし | **削除** |
| 4 | `IR_RELEASE` に解放前の合計がない → IR_LAYOUT の Total と直接対応できない | **`TotalBefore=%.0fMB` 追加（`getDiagnostics().totalBytes()` から取得）** |

---

## 1. Patch B: MKLNonUniformConvolver — 責務分離 + seq 1 開始

### B-1. releaseAllLayers() — 純粋な解放処理に戻す

```cpp
/// すべての Layer / ring / direct バッファを解放する。
/// ★ v29: 診断ログは含まない。ログは呼び出し元が個別に logIrRelease() などで行う。
void MKLNonUniformConvolver::releaseAllLayers() noexcept
{
    // ... 既存の解放ロジック（変更なし）...
}
```

**★ v28 の `uint64_t diagSeq` パラメータは削除。**

### B-2. logIrRelease() — 診断ログを分離

```cpp
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
/// ★ v29: IR_RELEASE ログ出力。releaseAllLayers() とは別責務。
///   呼び出し元は解放前後に allocatedBytes / lostFreeCount 等を取得して渡す。
void MKLNonUniformConvolver::logIrRelease(uint64_t diagSeq,
                                           uint64_t beforeMkl,
                                           uint32_t beforeLost,
                                           const ProcessMemoryInfo& beforeOs) const noexcept
{
    const uint64_t afterMkl = convo::diag::allocatedBytes();
    const uint32_t afterLost = convo::diag::lostFreeCount();
    const auto afterOs = getProcessMemoryInfo();
    const int64_t deltaMkl = static_cast<int64_t>(afterMkl) - static_cast<int64_t>(beforeMkl);
    const int32_t deltaLost = static_cast<int32_t>(afterLost) - static_cast<int32_t>(beforeLost);
    // ★ v29: TotalBefore 追加（IR_LAYOUT の Total と直接対応）
    const uint64_t totalBefore = getDiagnostics().totalBytes();

    diagLog(juce::String::formatted(
        "[IR_RELEASE] NUC#%p seq=%llu "
        "MKL: before=%lluMB after=%lluMB delta=%lldMB "
        "LayersBefore=%d TotalBefore=%.0fMB "
        "lostFree=%u(+%d) | "
        "OS: beforePrivate=%lluMB afterPrivate=%lluMB",
        (void*)this,
        (unsigned long long)diagSeq,
        (unsigned long long)(beforeMkl / (1024*1024)),
        (unsigned long long)(afterMkl / (1024*1024)),
        (long long)(deltaMkl / (1024*1024)),
        m_numActiveLayers,
        totalBefore / (1024.0 * 1024.0),
        (unsigned)afterLost, (int)deltaLost,
        (unsigned long long)beforeOs.privateUsageMB,
        (unsigned long long)afterOs.privateUsageMB));
}
#endif
```

### B-3. globalDiagSeq — 1 開始（★ v29）

```cpp
// ★ v29: fetch_add(1) の結果に +1 して 1 開始。
//   Destructor では明示的に seq=0。
const uint64_t diagSeq = globalDiagSeq.fetch_add(1, std::memory_order_relaxed) + 1;
```

### B-4. SetImpulse() — 責務分離後の流れ

```cpp
bool MKLNonUniformConvolver::SetImpulse(...)
{
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    const uint64_t diagSeq = globalDiagSeq.fetch_add(1, std::memory_order_relaxed) + 1;
    const uint64_t beforeMkl = convo::diag::allocatedBytes();
    const uint32_t beforeLost = convo::diag::lostFreeCount();
    const auto beforeOs = getProcessMemoryInfo();
#endif

    convo::publishAtomic(m_ready, false, std::memory_order_release);
    if (impulse == nullptr || irLen <= 0 || blockSize <= 0) return false;

    // ★ v29: releaseAllLayers() は純粋な解放処理（ログなし）
    releaseAllLayers();

    // ★ v29: ログは呼び出し元で個別に出力
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    logIrRelease(diagSeq, beforeMkl, beforeLost, beforeOs);
#endif

    // ... 既存の確保 + プリコンピュート ...

    convo::publishAtomic(m_ready, true, std::memory_order_release);

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    const uint64_t afterMkl = convo::diag::allocatedBytes();
    const uint32_t afterLost = convo::diag::lostFreeCount();

    // IR_LOAD
    diagLog(juce::String::formatted(
        "[IR_LOAD] NUC#%p seq=%llu irLen=%d blockSize=%d ...", ...));

    // IR_LAYOUT（★ v29: ready 削除）
    const auto snap = getDiagnostics();
    diagLog(juce::String::formatted(
        "[IR_LAYOUT] NUC#%p seq=%llu "
        "IRFreq=%.0fMB FDL=%.0fMB Accum=%.0fMB Tail=%.0fMB "
        "Direct=%.0fMB Ring=%.0fMB Total=%.0fMB | "
        "L0=%.0fMB L1=%.0fMB L2=%.0fMB", ...));
#endif

    return true;
}
```

### B-5. デストラクタ

```cpp
MKLNonUniformConvolver::~MKLNonUniformConvolver()
{
    const uint32_t oldLive = liveCount.fetch_sub(1, std::memory_order_relaxed);
    jassert(oldLive > 0);
    // ★ v29: デストラクタでは seq=0。診断ログは出力しない（プロセス終了中のため不安定）
    releaseAllLayers();
}
```

---

## 2. 全ログの v29 統一フォーマット

### IR_RELEASE（TotalBefore 追加、ready 削除済み）
```
[IR_RELEASE] NUC#%p seq=%llu MKL: before=%lluMB after=%lluMB delta=%lldMB LayersBefore=%d TotalBefore=%.0fMB lostFree=%u(+%d) | OS: beforePrivate=%lluMB afterPrivate=%lluMB
```

### IR_LOAD（v28 から変更なし）
```
[IR_LOAD] NUC#%p seq=%llu irLen=%d blockSize=%d Layers=%d L0Part=%d L1Part=%d L2Part=%d directTaps=%d ringSize=%d MKL: before=%lluMB after=%lluMB delta=%lldMB lostFree=%u(+%d) live=%u
```

### IR_LAYOUT（ready 削除）
```
[IR_LAYOUT] NUC#%p seq=%llu IRFreq=%.0fMB FDL=%.0fMB Accum=%.0fMB Tail=%.0fMB Direct=%.0fMB Ring=%.0fMB Total=%.0fMB | L0=%.0fMB L1=%.0fMB L2=%.0fMB
```

---

## 3. 出力例（v29 最終版）

```text
[IR_RELEASE] NUC#001 seq=105 MKL: before=820MB after=110MB delta=-710MB LayersBefore=3 TotalBefore=820MB lostFree=18(+0) | OS: beforePrivate=2330MB afterPrivate=1620MB
[IR_LOAD]    NUC#001 seq=105 irLen=327680 blockSize=4096 Layers=3 L0Part=4096 L1Part=32768 L2Part=262144 directTaps=32 ringSize=8192 MKL: before=110MB after=812MB delta=+702MB lostFree=18(+0) live=8
[IR_LAYOUT]  NUC#001 seq=105 IRFreq=256MB FDL=420MB Accum=96MB Tail=16MB Direct=24MB Ring=8MB Total=820MB | L0=8MB L1=64MB L2=720MB
```

**TotalBefore=820MB** → `IR_LAYOUT Total=820MB` と直接対応。解放前に 820MB 保持していたことが一目で分かる。
**seq=105** → （v28 と異なり 1 開始）
**ready 削除** → 常に 1 で診断価値のないフィールドを排除。

---

## 4. v28 からの改善点一覧

| # | v28（問題） | v29（修正） |
|:--|:----------|:-----------|
| 1 | `releaseAllLayers(uint64_t diagSeq)` が診断コンテキストを解放処理に持ち込む | **`releaseAllLayers()` は純粋な解放処理に。`logIrRelease()` を別関数に分離** |
| 2 | `globalDiagSeq` が 0 開始 | **`fetch_add(1) + 1` で 1 開始。Destructor は明示的に seq=0** |
| 3 | `ready` が常に 1 で診断価値なし | **削除** |
| 4 | `IR_RELEASE` に解放前の合計なし | **`TotalBefore=%.0fMB` 追加（`getDiagnostics().totalBytes()` から取得）** |
