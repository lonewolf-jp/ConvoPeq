# メモリ占有調査のためのインストルメンテーション改修案 v30（最終版）— 解放前スナップショット

---

**目的**: 2.33GB メモリ占有の原因特定。推定ではなく実測で確定させる。
**v29 からの変更**: 2 点の修正。

---

## 0. v29 の問題点と v30 での修正方針

| # | v29 の問題 | v30 の修正 |
|:--|:----------|:----------|
| 1 | `logIrRelease()` 内で `getDiagnostics().totalBytes()` を読んでいる → 呼び出し時点で `releaseAllLayers()` が完了しており、`TotalBefore=0MB` になる | **解放前に `getDiagnostics()` でスナップショットを取得し、`totalBefore`/`layersBefore` を引数で渡す** |
| 2 | `logIrRelease()` がインスタンスメソッドだが `this` の状態を読まなくなった | **`static` 関数にして責務を明確化** |
| 3 | `LayersBefore` も同様に解放後は `0` | **解放前スナップショットから `numActiveLayers` を取得して引数で渡す** |

---

## 1. Patch B: MKLNonUniformConvolver — 解放前スナップショット（★ 最重要）

### B-1. logIrRelease() — static 化 + パラメータ化

```cpp
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
/// ★ v30: IR_RELEASE ログ出力。static 関数。
///   呼び出し元は解放前のスナップショット（MKL/OS/Layer 情報）を引数で渡す。
///   これにより releaseAllLayers() 前後の状態変化を正しく記録できる。
static void logIrRelease(
    const MKLNonUniformConvolver* nuc,
    uint64_t diagSeq,
    uint64_t beforeMkl,
    uint32_t beforeLost,
    const ProcessMemoryInfo& beforeOs,
    uint64_t totalBefore,       // ★ v30: 解放前の getDiagnostics().totalBytes()
    int layersBefore,           // ★ v30: 解放前の m_numActiveLayers
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
        layersBefore,
        totalBefore / (1024.0 * 1024.0),
        (unsigned)afterLost, (int)deltaLost,
        (unsigned long long)beforeOs.privateUsageMB,
        (unsigned long long)afterOs.privateUsageMB));
}
#endif
```

### B-2. SetImpulse() — 解放前にスナップショットを取得

```cpp
bool MKLNonUniformConvolver::SetImpulse(...)
{
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    const uint64_t diagSeq = globalDiagSeq.fetch_add(1, std::memory_order_relaxed) + 1;
    const uint64_t beforeMkl = convo::diag::allocatedBytes();
    const uint32_t beforeLost = convo::diag::lostFreeCount();
    const auto beforeOs = getProcessMemoryInfo();

    // ★ v30: 解放前にスナップショットを取得（これが正しい TotalBefore / LayersBefore）
    const auto beforeSnap = getDiagnostics();
    const uint64_t totalBefore = beforeSnap.totalBytes();
    const int layersBefore = beforeSnap.numActiveLayers;
#endif

    convo::publishAtomic(m_ready, false, std::memory_order_release);
    if (impulse == nullptr || irLen <= 0 || blockSize <= 0) return false;

    releaseAllLayers();

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    const auto afterOs = getProcessMemoryInfo();
    logIrRelease(this, diagSeq, beforeMkl, beforeLost, beforeOs,
                 totalBefore, layersBefore, afterOs);
#endif

    // ... 既存の確保 + プリコンピュート ...

    convo::publishAtomic(m_ready, true, std::memory_order_release);

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    const uint64_t afterMkl = convo::diag::allocatedBytes();
    const uint32_t afterLost = convo::diag::lostFreeCount();

    // IR_LOAD
    diagLog(juce::String::formatted(
        "[IR_LOAD] NUC#%p seq=%llu irLen=%d blockSize=%d ...", ...));

    // IR_LAYOUT（解放後の getDiagnostics で問題ない）
    const auto snap = getDiagnostics();
    diagLog(juce::String::formatted(
        "[IR_LAYOUT] NUC#%p seq=%llu IRFreq=%.0fMB ...", ...));
#endif

    return true;
}
```

### B-3. デストラクタ（seq=0、診断ログ出力なし）

```cpp
MKLNonUniformConvolver::~MKLNonUniformConvolver()
{
    const uint32_t oldLive = liveCount.fetch_sub(1, std::memory_order_relaxed);
    jassert(oldLive > 0);
    releaseAllLayers();
    // ★ v30: デストラクタでは診断ログ出力なし
}
```

---

## 2. B-1 の補足: `static void logIrRelease` は `MKLNonUniformConvolver` の private static メソッドとして定義

```cpp
class MKLNonUniformConvolver {
    // ...
private:
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    static void logIrRelease(
        const MKLNonUniformConvolver* nuc,
        uint64_t diagSeq,
        uint64_t beforeMkl,
        uint32_t beforeLost,
        const ProcessMemoryInfo& beforeOs,
        uint64_t totalBefore,
        int layersBefore,
        const ProcessMemoryInfo& afterOs) noexcept;
#endif
};
```

---

## 3. 全ログフォーマット（v30 確定版、変更なし）

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
[IR_LAYOUT] NUC#%p seq=%llu IRFreq=%.0fMB FDL=%.0fMB Accum=%.0fMB Tail=%.0fMB Direct=%.0fMB Ring=%.0fMB Total=%.0fMB | L0=%.0fMB L1=%.0fMB L2=%.0fMB
```

---

## 4. 出力例（v30 最終版）

```text
[IR_RELEASE] NUC#001 seq=105 MKL: before=820MB after=110MB delta=-710MB LayersBefore=3 TotalBefore=820MB lostFree=18(+0) | OS: beforePrivate=2330MB afterPrivate=1620MB
[IR_LOAD]    NUC#001 seq=105 irLen=327680 blockSize=4096 Layers=3 L0Part=4096 L1Part=32768 L2Part=262144 ...
[IR_LAYOUT]  NUC#001 seq=105 IRFreq=256MB FDL=420MB Accum=96MB Tail=16MB Direct=24MB Ring=8MB Total=820MB | L0=8MB L1=64MB L2=720MB
```

**TotalBefore=820MB** → v29 では `releaseAllLayers()` 後に取得して 0MB になっていた可能性がある。
v30 では解放前に `getDiagnostics()` で正しく取得しているため、**IR_LAYOUT Total=820MB と一致**。

---

## 5. v29 からの改善点一覧

| # | v29（問題） | v30（修正） |
|:--|:----------|:-----------|
| 1 | `logIrRelease()` 内で `getDiagnostics().totalBytes()` → 解放後は 0 | **解放前に `beforeSnap = getDiagnostics()` でスナップショット取得し、`totalBefore`/`layersBefore` を引数で渡す** |
| 2 | `logIrRelease()` がインスタンスメソッド（`this` 使用せず） | **`static` 関数に変更** |
| 3 | `LayersBefore` も解放後は 0 だった | **解放前スナップショットの `numActiveLayers` を渡す** |
