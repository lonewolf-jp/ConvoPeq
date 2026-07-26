# BUG-001: Float処理パスにPeak Limiterが未実装

- **発見日**: 2026-07-26
- **カテゴリ**: 信号処理 / 音質劣化
- **関連**: [P1-1] SimplePeakLimiter, DSPCoreIO, DSPCoreDouble
- **リスク**: HIGH
- **修正**: 未

## 概要

Double処理パス (`DSPCoreDouble.cpp`) では出力直前に `SimplePeakLimiter` による
ソフトニー・ピークリミッティングを適用してからハードクリップしているが、
Float処理パス (`DSPCoreIO.cpp`) ではソフトニーリミッターを経ずに直接ハードクリップしている。

## 該当箇所

### Doubleパス（正しい実装）

**`AudioEngine.Processing.DSPCoreDouble.cpp:700-729`**

```cpp
// ★ [P1-1] Simple Peak Limiter: Hard Clamp (Safety Net) の前段で動作
//   threshold = kOutputHeadroom - 0.5dB, knee = 1.0dB
constexpr double kPLThreshold = 0.8413951287507587;  // -1.5dBFS
constexpr double kPLKnee = 0.108748;
peakLimiter.processBlock(dataL, dataR, numSamples, kPLThreshold, kPLKnee);

// その後、ハードクリップ
const __m256d vLimit = _mm256_set1_pd(kOutputHeadroom); // -1.0dBFS
// ... min/max clamp ...
```

### Floatパス（バグ）

**`AudioEngine.Processing.DSPCoreIO.cpp:506-514`**

```cpp
applyFixedLatencyDelay(dataL, dataR, numSamples);

for (int i = 0; i < numSamples; ++i)
    dstL[i] = static_cast<float>(juce::jlimit(-kOutputHeadroom, kOutputHeadroom, dataL[i]));

if (dstR)
    for (int i = 0; i < numSamples; ++i)
        dstR[i] = static_cast<float>(juce::jlimit(-kOutputHeadroom, kOutputHeadroom, dataR[i]));
```

Floatパスでは `peakLimiter.processBlock()` の呼び出しが存在せず、
信号が-1.0dBFSを超えた場合に**ハードクリップ（矩形波的歪み）** が発生する。

## 再現条件

- `EngineParameterSnapshot::processingOrder == ConvolverThenEQ` floatパスの場合
- 入力信号のピークが -1.5dBFS を超える場合

## 影響

- Doubleパス: ソフトニー（1dB幅）で徐々にゲインリダクションがかかり、自然な圧縮感
- Floatパス: 閾値を超えた瞬間にハードクリップ → 高調波歪み・エイリアシングノイズ
- ユーザーはFloat/Double切替で音質差を体感する可能性が高い
- サチュレーション/ディストーション系IRとの組み合わせで顕著

## 修正案

Floatパス (`DSPCoreIO.cpp`) の `processOutput()` 内で、ハードクリップ前に
`peakLimiter.processBlock()` を呼び出す：

```cpp
// DSPCoreIO.cpp: processOutput(), NaN/Inf scrub → Dither の後、Hard Clamp の前
constexpr double kPLThreshold = 0.8413951287507587;
constexpr double kPLKnee = 0.108748;
peakLimiter.processBlock(dataL, dataR, numSamples, kPLThreshold, kPLKnee);
```

ただし、`DSPCoreIO.cpp` は `DSPCoreDouble.cpp` とは異なり `peakLimiter` メンバを
持たない可能性がある。メンバ追加または共通化の設計判断が必要。
