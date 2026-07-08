# メモリ占有調査のためのインストルメンテーション改修案 v28（最終版）— グローバル seq + ready

---

**目的**: 2.33GB メモリ占有の原因特定。推定ではなく実測で確定させる。
**v27 からの変更**: 3 点の追加改善。

---

## 0. v27 の問題点と v28 での修正方針

| # | v27 の問題 | v28 の修正 |
|:--|:----------|:----------|
| 1 | IR_RELEASE seq=5, IR_LOAD seq=6 と別番号 → 同一 SetImpulse 内のイベントと認識しづらい | **SetImpulse 開始時に一度だけ採番し、IR_RELEASE / IR_LOAD / IR_LAYOUT の 3 ログで共通の seq を使用** |
| 2 | seq が NUC インスタンスごと → Crossfade で複数 NUC が混在時に混乱 | **`static std::atomic<uint64_t>` グローバル採番に変更。全 NUC 共通で単調増加** |
| 3 | IR_LAYOUT に準備状態なし | **`ready=%d` 追加（`getDiagnostics().isReady` をそのまま出力）** |

---

## 1. Patch B: MKLNonUniformConvolver — グローバル seq

### B-1. MKLNonUniformConvolver.h

```cpp
class MKLNonUniformConvolver {
public:
    static std::atomic<uint32_t> liveCount;
    static std::atomic<uint64_t> globalDiagSeq;  // ★ v28: 全 NUC 共通の診断シーケンス

    // ... 既存メンバ ...
};
```

### B-2. MKLNonUniformConvolver.cpp

```cpp
std::atomic<uint32_t> MKLNonUniformConvolver::liveCount { 0 };
std::atomic<uint64_t> MKLNonUniformConvolver::globalDiagSeq { 0 };  // ★ v28
```

### B-3. SetImpulse — 先頭で一度だけ seq を採番

```cpp
bool MKLNonUniformConvolver::SetImpulse(...)
{
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    // ★ v28: この SetImpulse のシーケンス番号。
    //   以降の IR_RELEASE / IR_LOAD / IR_LAYOUT で共通して使用される。
    //   全 NUC インスタンスで共有される atomic カウンタのため、
    //   クロスフェード中でも時系列順に並ぶ。
    const uint64_t diagSeq = globalDiagSeq.fetch_add(1, std::memory_order_relaxed);

    const uint64_t beforeMkl = convo::diag::allocatedBytes();
    const uint32_t beforeLost = convo::diag::lostFreeCount();
#endif

    // ... 以降は diagSeq をそのまま使い回す ...
```

### B-4. releaseAllLayers() — 引数で seq を受け取る

```cpp
// ★ v28: seq を引数で受け取る。SetImpulse からは diagSeq を渡す。
void MKLNonUniformConvolver::releaseAllLayers(uint64_t diagSeq /*= 0*/) noexcept
{
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    const uint64_t beforeMkl = convo::diag::allocatedBytes();
    const uint32_t beforeLost = convo::diag::lostFreeCount();
    const int layersBefore = m_numActiveLayers;
    const auto beforeOs = getProcessMemoryInfo();
#endif

    // ... 解放ロジック ...

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    const uint64_t afterMkl = convo::diag::allocatedBytes();
    const uint32_t afterLost = convo::diag::lostFreeCount();
    const auto afterOs = getProcessMemoryInfo();
    const int64_t deltaMkl = static_cast<int64_t>(afterMkl) - static_cast<int64_t>(beforeMkl);
    const int32_t deltaLost = static_cast<int32_t>(afterLost) - static_cast<int32_t>(beforeLost);
    diagLog(juce::String::formatted(
        "[IR_RELEASE] NUC#%p seq=%llu "
        "MKL: before=%lluMB after=%lluMB delta=%lldMB "
        "LayersBefore=%d lostFree=%u(+%d) | "
        "OS: beforePrivate=%lluMB afterPrivate=%lluMB",
        (void*)this,
        (unsigned long long)diagSeq,
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

### B-5. デストラクタからの releaseAllLayers 呼び出し

```cpp
MKLNonUniformConvolver::~MKLNonUniformConvolver()
{
    const uint32_t oldLive = liveCount.fetch_sub(1, std::memory_order_relaxed);
    jassert(oldLive > 0);
    releaseAllLayers(0);  // ★ v28: デストラクタでは seq=0（特殊な意味はない）
}
```

### B-6. SetImpulse — IR_LOAD / IR_LAYOUT（同じ seq）

```cpp
    // IR_LOAD（OS Private なし）
    diagLog(juce::String::formatted(
        "[IR_LOAD] NUC#%p seq=%llu irLen=%d blockSize=%d "
        "Layers=%d L0Part=%d L1Part=%d L2Part=%d "
        "directTaps=%d ringSize=%d "
        "MKL: before=%lluMB after=%lluMB delta=%lldMB "
        "lostFree=%u(+%d) live=%u",
        (void*)this,
        (unsigned long long)diagSeq,
        irLen, blockSize,
        m_numActiveLayers, l0Part, l1Part, l2Part,
        m_directTapCount, m_ringSize,
        (unsigned long long)(beforeMkl / (1024*1024)),
        (unsigned long long)(afterMkl / (1024*1024)),
        (long long)(deltaMkl / (1024*1024)),
        (unsigned)afterLost, (int)deltaLost,
        (unsigned)liveCount.load(std::memory_order_relaxed)));

    // IR_LAYOUT（1 回の getDiagnostics から生成）
    const auto snap = getDiagnostics();
    diagLog(juce::String::formatted(
        "[IR_LAYOUT] NUC#%p seq=%llu "
        "IRFreq=%.0fMB FDL=%.0fMB Accum=%.0fMB Tail=%.0fMB "
        "Direct=%.0fMB Ring=%.0fMB Total=%.0fMB "
        "ready=%d | "
        "L0=%.0fMB L1=%.0fMB L2=%.0fMB",
        (void*)this,
        (unsigned long long)diagSeq,
        snap.irFreqBytes / (1024.0*1024.0),
        snap.fdlBytes    / (1024.0*1024.0),
        snap.accumBytes  / (1024.0*1024.0),
        snap.tailBytes   / (1024.0*1024.0),
        snap.directBytes / (1024.0*1024.0),
        snap.ringBytes   / (1024.0*1024.0),
        snap.totalBytes() / (1024.0*1024.0),
        (int)snap.isReady,
        snap.layerBufs[0] / (1024.0*1024.0),
        snap.layerBufs[1] / (1024.0*1024.0),
        snap.layerBufs[2] / (1024.0*1024.0)));
```

---

## 2. 全ログの v28 統一フォーマット

### IR_RELEASE（seq は diagSeq）
```
[IR_RELEASE] NUC#%p seq=%llu MKL: before=%lluMB after=%lluMB delta=%lldMB LayersBefore=%d lostFree=%u(+%d) | OS: beforePrivate=%lluMB afterPrivate=%lluMB
```

### IR_LOAD（seq は同一 diagSeq）
```
[IR_LOAD] NUC#%p seq=%llu irLen=%d blockSize=%d Layers=%d L0Part=%d L1Part=%d L2Part=%d directTaps=%d ringSize=%d MKL: before=%lluMB after=%lluMB delta=%lldMB lostFree=%u(+%d) live=%u
```

### IR_LAYOUT（seq は同一 diagSeq、ready 追加）
```
[IR_LAYOUT] NUC#%p seq=%llu IRFreq=%.0fMB FDL=%.0fMB Accum=%.0fMB Tail=%.0fMB Direct=%.0fMB Ring=%.0fMB Total=%.0fMB ready=%d | L0=%.0fMB L1=%.0fMB L2=%.0fMB
```

### MEM_SNAP（v27 から変更なし）

---

## 3. 出力例（v28 最終版）

```text
[IR_RELEASE] NUC#001 seq=105 MKL: before=820MB after=110MB delta=-710MB LayersBefore=3 lostFree=18(+0) | OS: beforePrivate=2330MB afterPrivate=1620MB
[IR_LOAD]    NUC#001 seq=105 irLen=327680 blockSize=4096 Layers=3 L0Part=4096 L1Part=32768 L2Part=262144 directTaps=32 ringSize=8192 MKL: before=110MB after=812MB delta=+702MB lostFree=18(+0) live=8
[IR_LAYOUT]  NUC#001 seq=105 IRFreq=256MB FDL=420MB Accum=96MB Tail=16MB Direct=24MB Ring=8MB Total=820MB ready=1 | L0=8MB L1=64MB L2=720MB
[IR_RELEASE] NUC#002 seq=106 MKL: before=812MB after=420MB delta=-392MB LayersBefore=2 lostFree=18(+0) | OS: beforePrivate=2330MB afterPrivate=1620MB
[IR_LOAD]    NUC#002 seq=106 irLen=65536 blockSize=4096 Layers=2 L0Part=4096 L1Part=32768 L2Part=0 directTaps=32 ringSize=4096 MKL: before=420MB after=520MB delta=+100MB lostFree=18(+0) live=8
[IR_LAYOUT]  NUC#002 seq=106 IRFreq=40MB FDL=60MB Accum=12MB Tail=0MB Direct=24MB Ring=4MB Total=140MB ready=1 | L0=40MB L1=100MB L2=0MB
```

**seq=105** → IR_RELEASE / IR_LOAD / IR_LAYOUT が同一 SetImpulse 内のイベントと即座に識別可能。
**ready=1** → SetImpulse 完了後で publish 可能状態。
**seq=106** → NUC#002 の SetImpulse。NUC#001 とは別のイベント。

---

## 4. v27 からの改善点一覧

| # | v27（問題） | v28（修正） |
|:--|:----------|:-----------|
| 1 | IR_RELEASE/IR_LOAD で seq が別値 → 同一 SetImpulse 内のイベントと認識しづらい | **SetImpulse 開始時に一度だけ採番。3 ログで共通の seq** |
| 2 | seq が NUC インスタンスごと → Crossfade で複数 NUC 混在時に混乱 | **`static std::atomic<uint64_t>` グローバル採番。全 NUC 共通で時系列順** |
| 3 | IR_LAYOUT に準備状態なし | **`ready=%d` 追加（`isReady` をそのまま出力）** |
