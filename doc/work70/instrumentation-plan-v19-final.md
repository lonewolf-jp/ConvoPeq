# メモリ占有調査のためのインストルメンテーション改修案 v19（最終版）— OS Private 併記

---

**目的**: 2.33GB メモリ占有の原因特定。推定ではなく実測で確定させる。
**v18 からの変更**: IR_RELEASE / IR_LOAD に OS メモリ情報を追加。

---

## 0. v18 の問題点と v19 での修正方針

| # | v18 の問題 | v19 の修正 |
|:--|:----------|:----------|
| 1 | `IR_RELEASE` が `allocatedBytes` のみ → OS Private との比較ができない | **`beforePrivate/afterPrivate/beforeWS/afterWS` 追加** |
| 2 | `IR_LOAD` が `allocatedBytes` のみ → 同様 | **`beforePrivate/afterPrivate` 追加** |

---

## 1. DiagnosticsConfig.h — 既存の `getProcessMemoryInfo()` を使用（新規コード不要）

`DiagnosticsConfig.h` には既に以下が存在するため、変更は不要。

```cpp
struct ProcessMemoryInfo {
    uint64_t privateUsageMB = 0;
    uint64_t workingSetMB = 0;
    // ...
};

inline ProcessMemoryInfo getProcessMemoryInfo() noexcept { ... }
```

---

## 2. Patch B: MKLNonUniformConvolver — IR_RELEASE / IR_LOAD に OS Private 追加

### B-1. releaseAllLayers() — IR_RELEASE に OS Private 併記

```cpp
void MKLNonUniformConvolver::releaseAllLayers() noexcept
{
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    const uint64_t beforeMkl = convo::diag::allocatedBytes();
    const auto beforeOs = getProcessMemoryInfo();  // ★ v19
#endif

    // ... 既存の解放ロジック ...

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    const uint64_t afterMkl = convo::diag::allocatedBytes();
    const auto afterOs = getProcessMemoryInfo();   // ★ v19
    const int64_t deltaMkl = static_cast<int64_t>(afterMkl) - static_cast<int64_t>(beforeMkl);

    diagLog(juce::String::formatted(
        "[IR_RELEASE] NUC#%p "
        "MKL: before=%lluMB after=%lluMB delta=%lldMB | "
        "OS: beforePrivate=%lluMB afterPrivate=%lluMB "
        "beforeWS=%lluMB afterWS=%lluMB",
        (void*)this,
        (unsigned long long)(beforeMkl / (1024*1024)),
        (unsigned long long)(afterMkl / (1024*1024)),
        (long long)(deltaMkl / (1024*1024)),
        (unsigned long long)beforeOs.privateUsageMB,
        (unsigned long long)afterOs.privateUsageMB,
        (unsigned long long)beforeOs.workingSetMB,
        (unsigned long long)afterOs.workingSetMB));
#endif
}
```

### B-2. SetImpulse() — IR_LOAD に OS Private 併記

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
        "L0Part=%d L1Part=%d L2Part=%d "
        "directTaps=%d ringSize=%d "
        "MKL: before=%lluMB after=%lluMB delta=%lldMB | "
        "OS: beforePrivate=%lluMB afterPrivate=%lluMB live=%u",
        (void*)this, irLen, blockSize,
        l0Part, l1Part, l2Part,
        m_directTapCount, m_ringSize,
        (unsigned long long)(beforeMkl / (1024*1024)),
        (unsigned long long)(afterMkl / (1024*1024)),
        (long long)(deltaMkl / (1024*1024)),
        (unsigned long long)beforeOs.privateUsageMB,
        (unsigned long long)afterOs.privateUsageMB,
        (unsigned)liveCount.load(std::memory_order_relaxed)));
#endif
```

### B-3. IR_LAYOUT（v18 から変更なし）

---

## 3. 全ログの v19 統一フォーマット

### IR_RELEASE

```text
[IR_RELEASE] NUC#%p MKL: before=%lluMB after=%lluMB delta=%lldMB | OS: beforePrivate=%lluMB afterPrivate=%lluMB beforeWS=%lluMB afterWS=%lluMB
```

### IR_LOAD

```text
[IR_LOAD] NUC#%p irLen=%d blockSize=%d L0Part=%d L1Part=%d L2Part=%d directTaps=%d ringSize=%d MKL: before=%lluMB after=%lluMB delta=%lldMB | OS: beforePrivate=%lluMB afterPrivate=%lluMB live=%u
```

### IR_LAYOUT / NUC_MEM / MEM_SNAP（v18 から変更なし）

---

## 4. 出力例（v19）

### 解放＋再ロード（正常）

```text
[IR_RELEASE] NUC#001 MKL: before=820MB after=110MB delta=-710MB | OS: beforePrivate=2330MB afterPrivate=1620MB beforeWS=2400MB afterWS=1400MB
[IR_LOAD]    NUC#001 irLen=327680 blockSize=4096 L0Part=4096 L1Part=32768 L2Part=262144 directTaps=32 ringSize=8192 MKL: before=110MB after=812MB delta=+702MB | OS: beforePrivate=1620MB afterPrivate=2330MB live=8
[IR_LAYOUT]  NUC#001 IRFreq=256MB FDL=420MB Accum=96MB Tail=16MB Direct=24MB Ring=8MB
```

**解釈**:
- MKL -710MB → OS Private も -710MB（MKL が主因）
- MKL +702MB → OS Private も +710MB（MKL が主因）
- → **MKL が支配的。NUC 外部の要因ではない。**

### 解放漏れ（異常）

```text
[IR_RELEASE] NUC#005 MKL: before=812MB after=812MB delta=+0MB | OS: beforePrivate=2330MB afterPrivate=2330MB
```

**解釈**: MKL も OS Private も全く変化なし → `releaseAllLayers()` が全く機能していない。

### MKL 以外が原因の場合

```text
[IR_RELEASE] NUC#001 MKL: before=820MB after=110MB delta=-710MB | OS: beforePrivate=2330MB afterPrivate=2320MB beforeWS=2400MB afterWS=2390MB
```

**解釈**: MKL が 710MB 解放されたのに OS Private は 10MB しか減っていない。
→ **2.33GB の原因は MKL ではない。** JUCE / CRT / スレッドスタック等他の要因が支配的。

---

## 5. 診断フロー（v19）

| ログの組合せ | 診断 |
|:-----------|:-----|
| MKL delta ≈ OS Private delta | **MKL が支配的。NUC 外部の要因ではない。** |
| MKL delta = 0, OS Private delta = 0 | **解放漏れ（releaseAllLayers 未実行）** |
| MKL delta ≫ 0, OS Private delta ≈ 0 | **MKL 以外が原因（JUCE/CRT/IPP/Retire）** |
| MKL delta = 0, OS Private delta ≫ 0 | **NUC 外でメモリ増加（タイマー/UI/スレッド）** |

---

## 6. v18 からの改善点一覧

| # | v18（問題） | v19（修正） |
|:--|:----------|:-----------|
| 1 | `IR_RELEASE` に OS Private/WorkingSet なし | **`beforePrivate/afterPrivate/beforeWS/afterWS` 追加** |
| 2 | `IR_LOAD` に OS Private なし | **`beforePrivate/afterPrivate` 追加** |
