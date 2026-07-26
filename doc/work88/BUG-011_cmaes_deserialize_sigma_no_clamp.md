# BUG-011: CmaEsOptimizer::deserializeFrom が sigma をクランプしない（除算-by-ゼロリスク）

- **発見日**: 2026-07-26
- **カテゴリ**: 数値計算 / 除算-by-ゼロ
- **関連**: CmaEsOptimizer.h, CmaEsOptimizer.cpp, NoiseShaperLearner.cpp, DeviceSettings.cpp
- **リスク**: HIGH
- **修正**: 未

## 概要

`CmaEsOptimizer::deserializeFrom()` は外部から渡された `inSigma` をそのまま `sigma` メンバに代入するが、
`[params.sigmaMin, params.sigmaMax]` の範囲にクランプしていない。

`DeviceSettings.cpp:942` では `sanitizeFiniteClamped(..., 0.0, 10.0)` を使用しており、
sigma=0.0 が許容されている。この値が `NoiseShaperLearner::setState()` →
`optimizer.deserializeFrom(..., inState.sigma)` を経て渡されると、
`CmaEsOptimizer::update()` の共分散更新で `sigma` による除算が発生し、
**除算-by-ゼロ** が引き起こされる。

## 該当箇所

### 1. deserializeFrom（クランプなし）

**`CmaEsOptimizer.h:75-80`**

```cpp
void deserializeFrom(const double* inMean9, const double* inCov45, double inSigma) noexcept
{
    std::copy(inMean9, inMean9 + kDim, mean);
    deserializeCovUpperTriangle(inCov45);
    sigma = inSigma;  // ← クランプなし！
}
```

### 2. 呼び出しチェーン

```
DeviceSettings.cpp:942
  state.sigma = sanitizeFiniteClamped(..., 0.0, 10.0);  // sigma=0.0 が許容される
  ↓
NoiseShaperLearner.cpp:384
  optimizer.deserializeFrom(inState.mean, inState.covarianceUpperTriangle, inState.sigma);
  ↓
CmaEsOptimizer::update() 行 162-163
  const double yRow = (candidates[candidateIndex][row] - oldMean[row]) / sigma;   // /0
  const double yColumn = (candidates[candidateIndex][column] - oldMean[column]) / sigma; // /0
```

### 3. update 内の除算

**`CmaEsOptimizer.h:162-163`**

```cpp
const double yRow = (candidates[candidateIndex][row] - oldMean[row]) / sigma;
const double yColumn = (candidates[candidateIndex][column] - oldMean[column]) / sigma;
```

sigma=0 の場合、分子が非零なら **inf**、分子も 0 なら **NaN** が生成される。
これにより共分散行列が破壊され、Cholesky 分解が失敗する。

## 再現条件

1. プラグインの設定ファイル（または状態ファイル）に `sigma=0.0` が含まれている
2. `DeviceSettings.cpp:942` の `sanitizeFiniteClamped(..., 0.0, 10.0)` により sigma=0.0 が許容される
3. `NoiseShaperLearner::setState()` が呼び出され、`deserializeFrom` に sigma=0.0 が渡される
4. 次回 `CmaEsOptimizer::update()` 呼び出しで除算-by-ゼロ

## 影響

- **NoiseShaperLearner** の CMA-ES 最適化が完全に破壊される
- 共分散行列に inf/NaN が淬入 → Cholesky 分解失敗 → 最適化停止
- ユーザーはノイズシェーパーの学習が機能しないと感じる
- 設定ファイルの互換性の問題（旧バージョンで sigma=0 が保存されていた場合）

## 修正案

`deserializeFrom` で sigma をクランプする：

```cpp
void deserializeFrom(const double* inMean9, const double* inCov45, double inSigma) noexcept
{
    std::copy(inMean9, inMean9 + kDim, mean);
    deserializeCovUpperTriangle(inCov45);
    sigma = std::clamp(inSigma, params.sigmaMin, params.sigmaMax);  // クランプ追加
}
```

また、`DeviceSettings.cpp:942` の `sanitizeFiniteClamped` の下限値を `0.0` から
`params.sigmaMin`（デフォルト 0.03）以上に変更することも併せて推奨。
