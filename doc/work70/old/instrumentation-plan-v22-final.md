# メモリ占有調査のためのインストルメンテーション改修案 v22（最終版）— gen を外部から注入 + IR_LAYOUT Total

---

**目的**: 2.33GB メモリ占有の原因特定。推定ではなく実測で確定させる。
**v21 からの変更**: 2 点の修正 + 1 点の追加。

---

## 0. v21 の問題点と v22 での修正方針

| # | v21 の問題 | v22 の修正 |
|:--|:----------|:----------|
| 1 | Generation を NUC の責務として扱う方向 | **`SetImpulse()` のパラメータとして外部から注入（診断用コンテキスト）** |
| 2 | `gen=0` が「初期状態」か「取得失敗」か区別不可 | **既定値を `-1` に変更（未取得を明示）** |
| 3 | `IR_LAYOUT` に合計がない → `allocatedBytes` との照合に暗算が必要 | **`Total=%.0fMB` 追加** |

---

## 1. Patch B: MKLNonUniformConvolver — gen を外部注入 + IR_LAYOUT Total

### B-1. SetImpulse シグネチャ変更（★ v22）

```cpp
/// ★ v22: diagnosticGeneration — 診断用 Generation 番号（呼び出し元の
///   RuntimeWorld/ConvolverProcessor から渡す）。NUC はこの値を保持せず、
///   ログ出力にのみ使用する。既定値 -1 は「未指定」を意味する。
bool SetImpulse(const double* impulse, int irLen, int blockSize,
                double scale = 1.0,
                bool enableDirectHead = false,
                const FilterSpec* filterSpec = nullptr,
                int diagnosticGeneration = -1);
```

### B-2. SetImpulse() — IR_LOAD（gen はパラメータから）

```cpp
bool MKLNonUniformConvolver::SetImpulse(..., int diagnosticGeneration)
{
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    const uint64_t beforeMkl = convo::diag::allocatedBytes();
    const uint32_t beforeLost = convo::diag::lostFreeCount();
    const auto beforeOs = getProcessMemoryInfo();
#endif

    convo::publishAtomic(m_ready, false, std::memory_order_release);
    if (impulse == nullptr || irLen <= 0 || blockSize <= 0)
        return false;

    releaseAllLayers();

    // ... 既存の SetImpulse ロジック ...

    convo::publishAtomic(m_ready, true, std::memory_order_release);

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    const uint64_t afterMkl = convo::diag::allocatedBytes();
    const uint32_t afterLost = convo::diag::lostFreeCount();
    const auto afterOs = getProcessMemoryInfo();
    const int64_t deltaMkl = static_cast<int64_t>(afterMkl) - static_cast<int64_t>(beforeMkl);
    const int32_t deltaLost = static_cast<int32_t>(afterLost) - static_cast<int32_t>(beforeLost);

    const int l0Part = m_numActiveLayers >= 1 ? m_layers[0].partSize : 0;
    const int l1Part = m_numActiveLayers >= 2 ? m_layers[1].partSize : 0;
    const int l2Part = m_numActiveLayers >= 3 ? m_layers[2].partSize : 0;

    diagLog(juce::String::formatted(
        "[IR_LOAD] NUC#%p irLen=%d blockSize=%d "
        "Layers=%d L0Part=%d L1Part=%d L2Part=%d "
        "directTaps=%d ringSize=%d gen=%d "
        "MKL: before=%lluMB after=%lluMB delta=%lldMB "
        "lostFree=%u(+%d) | "
        "OS: beforePrivate=%lluMB afterPrivate=%lluMB live=%u",
        (void*)this, irLen, blockSize,
        m_numActiveLayers, l0Part, l1Part, l2Part,
        m_directTapCount, m_ringSize,
        diagnosticGeneration,  // ★ v22: NUC は保持せず、そのままログへ
        (unsigned long long)(beforeMkl / (1024*1024)),
        (unsigned long long)(afterMkl / (1024*1024)),
        (long long)(deltaMkl / (1024*1024)),
        (unsigned)afterLost, (int)deltaLost,
        (unsigned long long)beforeOs.privateUsageMB,
        (unsigned long long)afterOs.privateUsageMB,
        (unsigned)liveCount.load(std::memory_order_relaxed)));
#endif

    return true;
}
```

### B-3. IR_LAYOUT — Total 追加（★ v22）

```cpp
    // ★ v22: IR_LAYOUT に Total を追加
    const double totalMB = irFreqTotal / (1024.0*1024.0)
                         + fdlTotal   / (1024.0*1024.0)
                         + accumTotal / (1024.0*1024.0)
                         + tailTotal  / (1024.0*1024.0)
                         + snap.directBytes / (1024.0*1024.0)
                         + snap.ringBytes   / (1024.0*1024.0);
    diagLog(juce::String::formatted(
        "[IR_LAYOUT] NUC#%p IRFreq=%.0fMB FDL=%.0fMB "
        "Accum=%.0fMB Tail=%.0fMB Direct=%.0fMB Ring=%.0fMB "
        "Total=%.0fMB",
        (void*)this,
        irFreqTotal / (1024.0*1024.0),
        fdlTotal   / (1024.0*1024.0),
        accumTotal / (1024.0*1024.0),
        tailTotal  / (1024.0*1024.0),
        snap.directBytes / (1024.0*1024.0),
        snap.ringBytes   / (1024.0*1024.0),
        totalMB));
```

---

## 2. 全ログの v22 統一フォーマット

### IR_RELEASE（v21 から変更なし）

```text
[IR_RELEASE] NUC#%p MKL: before=%lluMB after=%lluMB delta=%lldMB lostFree=%u(+%d) | OS: beforePrivate=%lluMB afterPrivate=%lluMB
```

### IR_LOAD（gen の既定値 -1）

```text
[IR_LOAD] NUC#%p irLen=%d blockSize=%d Layers=%d L0Part=%d L1Part=%d L2Part=%d directTaps=%d ringSize=%d gen=%d MKL: before=%lluMB after=%lluMB delta=%lldMB lostFree=%u(+%d) | OS: beforePrivate=%lluMB afterPrivate=%lluMB live=%u
```

### IR_LAYOUT（Total 追加）

```text
[IR_LAYOUT] NUC#%p IRFreq=%.0fMB FDL=%.0fMB Accum=%.0fMB Tail=%.0fMB Direct=%.0fMB Ring=%.0fMB Total=%.0fMB
```

### MEM_SNAP（v21 から変更なし）

---

## 3. 出力例（v22）

```text
[IR_RELEASE] NUC#001 MKL: before=820MB after=110MB delta=-710MB lostFree=18(+0) | OS: beforePrivate=2330MB afterPrivate=1620MB
[IR_LOAD]    NUC#001 irLen=327680 blockSize=4096 Layers=3 L0Part=4096 L1Part=32768 L2Part=262144 directTaps=32 ringSize=8192 gen=21 MKL: before=110MB after=812MB delta=+702MB lostFree=18(+0) | OS: beforePrivate=1620MB afterPrivate=2330MB live=8
[IR_LAYOUT]  NUC#001 IRFreq=256MB FDL=420MB Accum=96MB Tail=16MB Direct=24MB Ring=8MB Total=820MB
[MEM_SNAP]   PUBLISH gen=21 seq=5 | NUC(MKL only): live=8 alloc=820MB Peak(since reset)=1200MB totalA=28.0GB totalF=27.4GB lostFree=0 | Stereo=4 DSPCore=4 | Retire: pending=232 objBytes=12.8MB(sizeof) tracked=8/232 (3%) overflow=0 reclaim=12 | OS: Private=2330MB WorkingSet=2400MB | OtherPrivate=1510MB
```

**IR_LAYOUT Total=820MB** → `MEM_SNAP alloc=820MB` と即座に一致確認可能。

`gen=-1` の場合:

```text
[IR_LOAD] NUC#002 irLen=65536 blockSize=4096 Layers=1 L0Part=4096 L1Part=0 L2Part=0 directTaps=32 ringSize=4096 gen=-1 MKL: before=110MB after=120MB delta=+10MB ...
```

**gen=-1** → 呼び出し元から Generation が渡されていないことが明確。

---

## 4. 呼び出し元（ConvolverProcessor）での gen 注入例

```cpp
// ConvolverProcessor.cpp — SetImpulse 呼び出し時
const int diagGen = (currentWorld != nullptr)
    ? static_cast<int>(currentWorld->getGeneration())
    : -1;

nuc.SetImpulse(impulse, irLen, blockSize, scale,
               enableDirectHead, &filterSpec,
               diagGen);  // ★ v22: 診断用 generation を外部から注入
```

---

## 5. v21 からの改善点一覧

| # | v21（問題） | v22（修正） |
|:--|:----------|:-----------|
| 1 | Generation が NUC の責務として扱われる方向 | **`SetImpulse()` パラメータとして外部注入（責務分離維持）** |
| 2 | `gen=0` が「初期状態」か「取得失敗」か区別不可 | **既定値を `-1` に変更** |
| 3 | `IR_LAYOUT` に合計なし → 暗算が必要 | **`Total=%.0fMB` 追加** |
