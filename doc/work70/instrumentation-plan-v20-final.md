# メモリ占有調査のためのインストルメンテーション改修案 v20（最終版）— lostFree + Layers 追記

---

**目的**: 2.33GB メモリ占有の原因特定。推定ではなく実測で確定させる。
**v19 からの変更**: 2 点の追加。

---

## 0. v19 の問題点と v20 での修正方針

| # | v19 の問題 | v20 の修正 |
|:--|:----------|:----------|
| 1 | `IR_RELEASE` に `lostFree` がない → 解放不足がサイズ不明起因か判断不可 | **`lostFree=%u` 追加 |
| 2 | `IR_LOAD` に Layer 数がない → パーティション構成が不明瞭 | **`Layers=%d` 追加 |

---

## 1. Patch B: MKLNonUniformConvolver — IR_RELEASE / IR_LOAD 最終版

### B-1. releaseAllLayers() — IR_RELEASE（lostFree 追加）

```cpp
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    const uint64_t beforeMkl = convo::diag::allocatedBytes();
    const auto beforeOs = getProcessMemoryInfo();
#endif

    // ... 既存の解放ロジック ...

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    const uint64_t afterMkl = convo::diag::allocatedBytes();
    const auto afterOs = getProcessMemoryInfo();
    const int64_t deltaMkl = static_cast<int64_t>(afterMkl) - static_cast<int64_t>(beforeMkl);

    diagLog(juce::String::formatted(
        "[IR_RELEASE] NUC#%p "
        "MKL: before=%lluMB after=%lluMB delta=%lldMB lostFree=%u | "
        "OS: beforePrivate=%lluMB afterPrivate=%lluMB "
        "beforeWS=%lluMB afterWS=%lluMB",
        (void*)this,
        (unsigned long long)(beforeMkl / (1024*1024)),
        (unsigned long long)(afterMkl / (1024*1024)),
        (long long)(deltaMkl / (1024*1024)),
        (unsigned)convo::diag::lostFreeCount(),
        (unsigned long long)beforeOs.privateUsageMB,
        (unsigned long long)afterOs.privateUsageMB,
        (unsigned long long)beforeOs.workingSetMB,
        (unsigned long long)afterOs.workingSetMB));
#endif
```

### B-2. SetImpulse() — IR_LOAD（Layers 追加）

```cpp
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    const uint64_t beforeMkl = convo::diag::allocatedBytes();
    const auto beforeOs = getProcessMemoryInfo();
#endif

    // ... 既存の SetImpulse ロジック ...

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    const uint64_t afterMkl = convo::diag::allocatedBytes();
    const auto afterOs = getProcessMemoryInfo();
    const int64_t deltaMkl = static_cast<int64_t>(afterMkl) - static_cast<int64_t>(beforeMkl);

    const int l0Part = m_numActiveLayers >= 1 ? m_layers[0].partSize : 0;
    const int l1Part = m_numActiveLayers >= 2 ? m_layers[1].partSize : 0;
    const int l2Part = m_numActiveLayers >= 3 ? m_layers[2].partSize : 0;

    diagLog(juce::String::formatted(
        "[IR_LOAD] NUC#%p irLen=%d blockSize=%d "
        "Layers=%d L0Part=%d L1Part=%d L2Part=%d "
        "directTaps=%d ringSize=%d "
        "MKL: before=%lluMB after=%lluMB delta=%lldMB lostFree=%u | "
        "OS: beforePrivate=%lluMB afterPrivate=%lluMB live=%u",
        (void*)this, irLen, blockSize,
        m_numActiveLayers, l0Part, l1Part, l2Part,
        m_directTapCount, m_ringSize,
        (unsigned long long)(beforeMkl / (1024*1024)),
        (unsigned long long)(afterMkl / (1024*1024)),
        (long long)(deltaMkl / (1024*1024)),
        (unsigned)convo::diag::lostFreeCount(),
        (unsigned long long)beforeOs.privateUsageMB,
        (unsigned long long)afterOs.privateUsageMB,
        (unsigned)liveCount.load(std::memory_order_relaxed)));
#endif
```

---

## 2. 全ログの v20 統一フォーマット

### IR_RELEASE

```text
[IR_RELEASE] NUC#%p MKL: before=%lluMB after=%lluMB delta=%lldMB lostFree=%u | OS: beforePrivate=%lluMB afterPrivate=%lluMB beforeWS=%lluMB afterWS=%lluMB
```

### IR_LOAD

```text
[IR_LOAD] NUC#%p irLen=%d blockSize=%d Layers=%d L0Part=%d L1Part=%d L2Part=%d directTaps=%d ringSize=%d MKL: before=%lluMB after=%lluMB delta=%lldMB lostFree=%u | OS: beforePrivate=%lluMB afterPrivate=%lluMB live=%u
```

### IR_LAYOUT / NUC_MEM / MEM_SNAP（v19 から変更なし）

---

## 3. 出力例（v20）

### 正常な解放＋再ロード

```text
[IR_RELEASE] NUC#001 MKL: before=820MB after=110MB delta=-710MB lostFree=0 | OS: beforePrivate=2330MB afterPrivate=1620MB beforeWS=2400MB afterWS=1400MB
[IR_LOAD]    NUC#001 irLen=327680 blockSize=4096 Layers=3 L0Part=4096 L1Part=32768 L2Part=262144 directTaps=32 ringSize=8192 MKL: before=110MB after=812MB delta=+702MB lostFree=0 | OS: beforePrivate=1620MB afterPrivate=2330MB live=8
[IR_LAYOUT]  NUC#001 IRFreq=256MB FDL=420MB Accum=96MB Tail=16MB Direct=24MB Ring=8MB
```

### 解放漏れ — lostFree で原因判別

```text
// ケースA: lostFree=0 で解放なし → 本当に解放漏れ（releaseAllLayers のバグ）
[IR_RELEASE] NUC#005 MKL: before=812MB after=812MB delta=+0MB lostFree=0 | OS: beforePrivate=2330MB afterPrivate=2330MB

// ケースB: lostFree=18 で解放なし → allocSizes 保存漏れが原因（統計ミス）
[IR_RELEASE] NUC#006 MKL: before=812MB after=812MB delta=+0MB lostFree=18 | OS: beforePrivate=2330MB afterPrivate=2330MB
```

**ケースA**: `lostFree=0` → 統計は正確。`releaseAllLayers()` が解放に失敗している。
**ケースB**: `lostFree=18` → 18 回の `diagMklFree` でサイズ不明。`allocSizes` 保存漏れが原因で統計が狂っている。

### 解放されたが lostFree が増えている

```text
[IR_RELEASE] NUC#007 MKL: before=820MB after=110MB delta=-710MB lostFree=5 | OS: beforePrivate=2330MB afterPrivate=1620MB
```

OS Private は 710MB 減っているため解放自体は正しい。しかし `lostFree=5` は「5回の解放でサイズ不明」を意味する。`allocSizes` の保存漏れがあるが、メモリ解放には影響していない（統計のみの問題）。

---

## 4. v19 からの改善点一覧

| # | v19（問題） | v20（修正） |
|:--|:----------|:-----------|
| 1 | `IR_RELEASE` に `lostFree` なし → 解放不足が統計ミスか判断不可 | **`lostFree=%u` 追加 |
| 2 | `IR_LOAD` に Layer 数なし → パーティション構成が不明瞭 | **`Layers=%d` 追加 |
