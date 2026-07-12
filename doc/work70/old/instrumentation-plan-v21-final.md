# メモリ占有調査のためのインストルメンテーション改修案 v21（最終版）— lostFree delta + Generation

---

**目的**: 2.33GB メモリ占有の原因特定。推定ではなく実測で確定させる。
**v20 からの変更**: 2 点の追加。これ以上ログを増やす必要はなし。

---

## 0. v20 の問題点と v21 での修正方針

| # | v20 の問題 | v21 の修正 |
|:--|:----------|:----------|
| 1 | `lostFree` が累積値のみ → どのイベントで増えたか不明 | **`beforeLost/afterLost/deltaLost` で増分も記録** |
| 2 | `IR_LOAD` に Generation なし → MEM_SNAP との対応が不明瞭 | **`gen=%d` 追加（Publish generation）** |

---

## 1. Patch B: MKLNonUniformConvolver — 最終版

### B-1. releaseAllLayers() — IR_RELEASE（lostFree delta 追加）

```cpp
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    const uint64_t beforeMkl = convo::diag::allocatedBytes();
    const uint32_t beforeLost = convo::diag::lostFreeCount();  // ★ v21
    const auto beforeOs = getProcessMemoryInfo();
#endif

    // ... 既存の解放ロジック ...

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    const uint64_t afterMkl = convo::diag::allocatedBytes();
    const uint32_t afterLost = convo::diag::lostFreeCount();   // ★ v21
    const auto afterOs = getProcessMemoryInfo();
    const int64_t deltaMkl = static_cast<int64_t>(afterMkl) - static_cast<int64_t>(beforeMkl);
    const int32_t deltaLost = static_cast<int32_t>(afterLost) - static_cast<int32_t>(beforeLost);

    // ★ v21: lostFree は累積値と増分を併記（例: lostFree=18(+2)）
    diagLog(juce::String::formatted(
        "[IR_RELEASE] NUC#%p "
        "MKL: before=%lluMB after=%lluMB delta=%lldMB "
        "lostFree=%u(+%d) | "
        "OS: beforePrivate=%lluMB afterPrivate=%lluMB",
        (void*)this,
        (unsigned long long)(beforeMkl / (1024*1024)),
        (unsigned long long)(afterMkl / (1024*1024)),
        (long long)(deltaMkl / (1024*1024)),
        (unsigned)afterLost, (int)deltaLost,
        (unsigned long long)beforeOs.privateUsageMB,
        (unsigned long long)afterOs.privateUsageMB));
#endif
```

### B-2. SetImpulse() — IR_LOAD（Generation 追加）

```cpp
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    const uint64_t beforeMkl = convo::diag::allocatedBytes();
    const uint32_t beforeLost = convo::diag::lostFreeCount();  // ★ v21
    const auto beforeOs = getProcessMemoryInfo();
#endif

    // ... 既存の SetImpulse ロジック ...

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    const uint64_t afterMkl = convo::diag::allocatedBytes();
    const uint32_t afterLost = convo::diag::lostFreeCount();   // ★ v21
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
        /* 呼び出し元から gen を取得 */,
        (unsigned long long)(beforeMkl / (1024*1024)),
        (unsigned long long)(afterMkl / (1024*1024)),
        (long long)(deltaMkl / (1024*1024)),
        (unsigned)afterLost, (int)deltaLost,
        (unsigned long long)beforeOs.privateUsageMB,
        (unsigned long long)afterOs.privateUsageMB,
        (unsigned)liveCount.load(std::memory_order_relaxed)));
#endif
```

**★ gen の取得**: SetImpulse の呼び出し元（ConvolverProcessor 等）から
現在の Runtime の generation を渡す。あるいは NUC 自身が保持する generation
カウンタがあればそれを使用。なければ `gen=0` とする。

---

## 2. 全ログの v21 統一フォーマット

### IR_RELEASE

```text
[IR_RELEASE] NUC#%p MKL: before=%lluMB after=%lluMB delta=%lldMB lostFree=%u(+%d) | OS: beforePrivate=%lluMB afterPrivate=%lluMB
```

### IR_LOAD

```text
[IR_LOAD] NUC#%p irLen=%d blockSize=%d Layers=%d L0Part=%d L1Part=%d L2Part=%d directTaps=%d ringSize=%d gen=%d MKL: before=%lluMB after=%lluMB delta=%lldMB lostFree=%u(+%d) | OS: beforePrivate=%lluMB afterPrivate=%lluMB live=%u
```

### IR_LAYOUT / NUC_MEM / MEM_SNAP（v20 から変更なし）

---

## 3. 出力例（v21）

```text
[IR_RELEASE] NUC#001 MKL: before=820MB after=110MB delta=-710MB lostFree=18(+0) | OS: beforePrivate=2330MB afterPrivate=1620MB
[IR_LOAD]    NUC#001 irLen=327680 blockSize=4096 Layers=3 L0Part=4096 L1Part=32768 L2Part=262144 directTaps=32 ringSize=8192 gen=21 MKL: before=110MB after=812MB delta=+702MB lostFree=18(+0) | OS: beforePrivate=1620MB afterPrivate=2330MB live=8
[IR_LAYOUT]  NUC#001 IRFreq=256MB FDL=420MB Accum=96MB Tail=16MB Direct=24MB Ring=8MB
[MEM_SNAP]   PUBLISH gen=21 seq=5 | NUC(MKL only): live=8 alloc=832MB ... | OtherPrivate=1498MB
```

**`gen=21` で IR_LOAD と MEM_SNAP が対応**: どの Publish の IR ロードだったか完全に追跡可能。

**`lostFree=18(+0)`**: 累積18回のサイズ不明解放。今回のイベントでは増加なし。

```text
[IR_RELEASE] NUC#002 MKL: before=812MB after=812MB delta=+0MB lostFree=18(+3) | OS: beforePrivate=2330MB afterPrivate=2330MB
```

**`lostFree=18(+3)`**: 今回の `releaseAllLayers()` で `lostFree` が 3 増加。
→ `allocSizes` 保存漏れが新たに発生。解放は正しく行われたが（OS Private は減少）、
統計が 3 つのバッファで追跡不能。

---

## 4. v20 からの改善点一覧

| # | v20（問題） | v21（修正） |
|:--|:----------|:-----------|
| 1 | `lostFree` が累積値のみ → どのイベントで増えたか不明 | **`beforeLost/afterLost/deltaLost` で増分を記録（`lostFree=%u(+%d)`）** |
| 2 | `IR_LOAD` に Generation なし → MEM_SNAP との対応不明瞭 | **`gen=%d` 追加** |
