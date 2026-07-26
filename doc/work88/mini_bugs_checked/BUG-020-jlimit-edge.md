# BUG-020: `juce::jlimit` 下限 > 上限時の未定義動作（LoaderThread.cpp:198）

**発見日**: 2026-07-26
**カテゴリ**: エッジケース / 論理バグ
**リスク**: LOW（通常運用では発現せず、発現時も影響限定的）

---

## 概要

`ConvolverProcessor.LoaderThread.cpp:198` において、`juce::jlimit(0, targetLength - 1, ...)` の `targetLength` が 0 の場合、`lowerLimit=0 > upperLimit=-1` となり、`jlimit` が常に `upperLimit=-1` を返す。これにより `estimatePeakLatencySamples` が **-1** を返し、`conv.irLatency = -1` が保存される。

---

## 該当箇所

```cpp
// ConvolverProcessor.LoaderThread.cpp:146-201
int ConvolverProcessor::LoaderThread::estimatePeakLatencySamples(
    const juce::AudioBuffer<double>& trimmed, int targetLength) const
{
    int irPeakLatency = 0;
    if (trimmed.getNumChannels() > 0)
    {
        // ...（energyBuffer の計算）...
        irPeakLatency = static_cast<int>(std::floor(maxCentroid + 0.5));
        irPeakLatency = juce::jlimit(0, targetLength - 1, irPeakLatency);
        //                                  ↑ targetLength=0 のとき upper = -1
        //                                    jlimit(0, -1, val) は常に -1 を返す！
    }
    return irPeakLatency;
}
```

## `juce::jlimit` の実装

```cpp
template <typename Type>
constexpr Type jlimit(Type lowerLimit, Type upperLimit, Type value) noexcept
{
    return value < lowerLimit ? lowerLimit : (value > upperLimit ? upperLimit : value);
}
```

`targetLength=0` のとき:

| 引数 | 値 |
|------|-----|
| `lowerLimit` | 0 |
| `upperLimit` | -1 |
| `value` | 0 (通常) |
| 結果 | `0 > -1` → **-1** |

---

## 影響

`conv.irLatency` が -1 になる（本来 0 であるべき）。その後:

```cpp
// ConvolverProcessor.Runtime.cpp:143
const int irPeakLatency = juce::jmax(0, conv.irLatency);
```

で -1 → 0 に補正されるため、**実際のオーディオ処理には影響しない**。ただし:

1. `conv.irLatency = -1` が永続的に保存される（セマンティクス不正）
2. デバッグログ・テレメトリで `irLatency=-1` が記録される
3. 将来のコードリファクタリングで `jmax` ガードが削除された場合に顕在化

---

## 発現条件

`targetLength = computeTargetIRLength(...)` が 0 を返すこと。これは元の IR のサンプル数が 0 の場合に発生しうる。通常の IR（WAV ファイル）読み込みでは 0 にはならないが、異常ファイルまたはプラグイン状態の整合性崩れで起こりうる。

---

## 修正案

```cpp
// ガードを追加
if (targetLength <= 0)
    return 0;

irPeakLatency = juce::jlimit(0, targetLength - 1, irPeakLatency);
```

または jlimit の upper を適切に調整:
```cpp
const int maxLatency = std::max(0, targetLength - 1);
irPeakLatency = juce::jlimit(0, maxLatency, irPeakLatency);
```
