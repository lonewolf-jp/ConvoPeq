# メモリ占有調査のためのインストルメンテーション改修案 v37（最終版）

---

**目的**: 2.33GB メモリ占有の原因特定。推定ではなく実測で確定させる。
**v36 からの変更**: ソースコード調査で確定した 3 項目を反映。設計変更なし、実装上の確定事項のみ。

---

## 0. ソースコード調査で確定した未確定事項

| # | 調査項目 | 結果 | 設計への反映 |
|:--|:--------|:-----|:------------|
| 1 | `releaseAllLayers()` に早期 return 経路はあるか？ | **なし**。常に全 Layer 解放 + 状態リセットを実行。`IR_RELEASE` ログ出力の前提として安全 | 確認のみ（変更不要） |
| 2 | `diagLog` 関数の定義場所 | **`DSPCoreLifecycle.cpp` の無名名前空間内のみ**。`MKLNonUniformConvolver.cpp` では未定義 → `juce::Logger::writeToLog()` を直接使用するか、ローカル定義が必要 | **`juce::Logger::writeToLog()` を直接使用（既存コードと整合）** |
| 3 | `DiagnosticsConfig.h` のインクルードパス | **`AudioEngine.h` 経由ではインクルードされない。`MKLNonUniformConvolver.cpp` に直接 `#include "DiagnosticsConfig.h"` が必要** | **`#include "DiagnosticsConfig.h"` を `MKLNonUniformConvolver.cpp` に追加** |

---

## 1. インクルードの追加

### MKLNonUniformConvolver.cpp

```cpp
#include <JuceHeader.h>
#include "MKLNonUniformConvolver.h"
#include "DiagnosticsConfig.h"  // ★ 追加（getProcessMemoryInfo, convo::diag, DIAG_MKL_* を使用するため）
#include "AlignedAllocation.h"
// ... 既存のインクルード ...
```

### MKLNonUniformConvolver.h（変更不要）

`NucDiagnosticsSnapshot` は本ヘッダで定義するが、`convo::diag::` の使用は `.cpp` 側に閉じる。

---

## 2. ログ出力方法の統一

`MKLNonUniformConvolver.cpp` 内では、`juce::Logger::writeToLog()` を直接使用する。
ローカルの `diagLog` ラッパーは追加しない（既存コードとの一貫性のため）。

```cpp
// ★ 使用例（全ログで統一）
juce::Logger::writeToLog(juce::String::formatted("[IR_RELEASE] ..."));
juce::Logger::writeToLog(juce::String::formatted("[IR_LOAD] ..."));
juce::Logger::writeToLog(juce::String::formatted("[IR_LAYOUT] ..."));
```

---

## 3. 全ログフォーマット（v37 確定版、v36 から変更なし）

### IR_RELEASE
```
[IR_RELEASE] NUC#%p seq=%llu MKL: before=%lluMB after=%lluMB delta=%lldMB LayersBefore=%d TotalBefore=%.0fMB(persistent) lostFree=%u(+%d) live=%u | OS: beforePrivate=%lluMB afterPrivate=%lluMB
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
[MEM_SNAP] PUBLISH gen=%d seq=%d | NUC(MKL only): live=%u alloc=%.0fMB Peak(since reset)=%.0fMB totalA=%.0fGB totalF=%.0fGB lostFree=%u zeroAllocSize=%u(+%d) | Stereo=%d DSPCore=%d | Retire: pending=%u trackedPendingBytes=%.1fMB(diag only) trackedPending=%u/%u (%.0f%%) overflow=%llu reclaim=%llu | OS: Private=%lluMB WorkingSet=%lluMB | OtherPrivate=%.0fMB(JUCE/CRT/IPP/threads/...)
```

---

## 4. 出力例（v37 最終版）

```text
[IR_RELEASE] NUC#001 seq=105 MKL: before=820MB after=110MB delta=-710MB LayersBefore=3 TotalBefore=820MB(persistent) lostFree=18(+0) live=8 | OS: beforePrivate=2330MB afterPrivate=1620MB
[IR_LOAD]    NUC#001 seq=105 irLen=327680 blockSize=4096 Layers=3 L0Part=4096 L1Part=32768 L2Part=262144 directTaps=32 ringSize=8192 MKL: before=110MB after=812MB delta=+702MB lostFree=18(+0) live=8
[IR_LAYOUT]  NUC#001 seq=105 IRFreq=256MB FDL=420MB Accum=96MB Tail=16MB Direct=24MB Ring=8MB Total=820MB(persistent data buffers only) | L0=8MB L1=64MB L2=720MB
[MEM_SNAP]   PUBLISH gen=21 seq=5 | NUC(MKL only): live=8 alloc=820MB Peak(since reset)=1200MB totalA=28.0GB totalF=27.4GB lostFree=0 zeroAllocSize=0(+0) | Stereo=4 DSPCore=4 | Retire: pending=232 trackedPendingBytes=12.8MB(diag only) trackedPending=8/232 (3%) overflow=0 reclaim=12 | OS: Private=2330MB WorkingSet=2400MB | OtherPrivate=1510MB(JUCE/CRT/IPP/threads/...)
```

---

## 5. v36 からの改善点一覧

| # | 調査結果 | 反映 |
|:--|:--------|:-----|
| 1 | `releaseAllLayers()` 早期 return なし → IR_RELEASE 常時出力可能 | 確認のみ |
| 2 | `diagLog` は `DSPCoreLifecycle.cpp` ローカルのみ → `MKLNonUniformConvolver.cpp` では `juce::Logger::writeToLog()` を直接使用 | **実装方針確定** |
| 3 | `DiagnosticsConfig.h` は `AudioEngine.h` 経由でインクルードされない → `MKLNonUniformConvolver.cpp` に直接追加 | **`#include "DiagnosticsConfig.h"` を追加** |
