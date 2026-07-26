# BUG-018: Floating-point `!= 1.0` exact comparison（FP等価比較）

**発見日**: 2026-07-26
**カテゴリ**: 数値計算 / コード品質
**リスク**: LOW（実害は限定的だが、設計原則違反）

---

## 概要

浮動小数点数の値が `1.0` と「等しいか」を `!= 1.0` という**完全一致比較**で判定している箇所が3箇所存在する。これらの値は sqrt/除算/decibelsToGain 等のFP演算結果であり、丸め誤差により理論上 `1.0` と等価な値が `1.000000000000001` 等になる可能性がある。

---

## 該当箇所

### Site 1: `ConvolverProcessor.LoadPipeline.cpp:347`

```cpp
if (prepared->hasScaleFactor && prepared->scaleFactor != 1.0)
```

`scaleFactor` は `computeEnergyScale()` で計算される:
```cpp
// IRConverter.cpp:36-37
constexpr double safetyMargin = 0.5011872336272722;  // -6dB
return (1.0 / std::sqrt(maxChannelEnergy)) * safetyMargin;
```

`maxChannelEnergy ≈ safetyMargin² = 0.251187...` のとき `scaleFactor ≈ 1.0` だが、IEEE 754 の丸めにより完全一致しない可能性がある。

### Site 2: `AudioEngine.Processing.DSPCoreDouble.cpp:440`

```cpp
if (state.convolverInputTrimGain != 1.0)
```

`convolverInputTrimGain` は `juce::Decibels::decibelsToGain(clampedDb)` の結果。`clampedDb` が `0.0f` なら `decibelsToGain(0.0) = 1.0`（正確）。しかしパラメータ復元・オートメーション・プラグイン状態読み込みで微小誤差が混入した場合、`1.0` からずれる可能性がある。

### Site 3: `MKLNonUniformConvolver.cpp:1048`

```cpp
if (scale != 1.0)
    cblas_dscal(l.complexSize * 2, scale, l.irFreqDomain, 1);
```

`scale` は `SetImpulse` の引数として LoadPipeline から渡される値で、IRConverter の `computeEnergyScale()` → `computeScaleFactor()` 経由。Site 1 と同根。

---

## 影響

| Site | 誤動作 | 実害 |
|------|--------|------|
| 1 | `scaleFactor=1.0+ε` でスケーリングが不要なのに実施される | 無視できる（乗算1.0+εの影響は < 1e-12 dB） |
| 2 | `convolverInputTrimGain=1.0+ε` で不要なゲイン適用 | CPU1回の余分な `scaleBlockFallback` → 無視できる |
| 3 | `scale=1.0+ε` で不要な `cblas_dscal` | 同上 |

**最大のリスク**: 実害が小さいこと自体が問題ではなく、コードベース全体で**同一パターンが増殖する文化を生む**こと。将来の修正で `!= 1.0` の箇所が増え、閾値判断の一貫性が失われる。

---

## 修正案

全ての非整数値比較に epsilon を導入する:

```cpp
constexpr double kUnityEpsilon = 1.0e-12;
if (prepared->hasScaleFactor && std::abs(prepared->scaleFactor - 1.0) > kUnityEpsilon)
```

ただし以下の点に注意:
- `kUnityEpsilon` の値は使用コンテキストに応じて調整（IRConverter の `1.0e-18` 相当が適切か否か）
- パフォーマンス影響: `std::abs` + subtraction → 1-2 cycle、問題なし
