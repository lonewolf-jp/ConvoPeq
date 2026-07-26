# BUG-013: CmaEsOptimizerDynamic::deserializeFrom が sigma をクランプしない（除算-by-ゼロリスク）

- **発見日**: 2026-07-26
- **カテゴリ**: 数値計算 / 除算-by-ゼロ
- **関連**: CmaEsOptimizerDynamic.cpp, CmaEsOptimizerDynamic.h, AllpassDesigner.cpp
- **リスク**: HIGH
- **修正**: 未

## 概要

`CmaEsOptimizerDynamic::deserializeFrom()` は外部から渡された `inSigma` をそのまま
`sigma` メンバに代入するが、`[params.sigmaMin, params.sigmaMax]` の範囲にクランプしていない。

`CmaEsOptimizerDynamic` は `AllpassDesigner` から使用されており、
最適化状態のシリアライズ/デシリアライズがサポートされている。
デシリアライズ時に sigma=0 が渡されると、`update()` 内の除算-by-ゼロが発生する。

## 該当箇所

### 1. deserializeFrom（クランプなし）

**`CmaEsOptimizerDynamic.cpp:195-205`**

```cpp
void CmaEsOptimizerDynamic::deserializeFrom(const double* inMean, const double* inCov, double inSigma) {
    std::copy(inMean, inMean + dim, mean.begin());
    int idx = 0;
    for (int r = 0; r < dim; ++r)
        for (int c = r; c < dim; ++c) {
            covariance[matrixIndex(r, c, dim)] = inCov[idx];
            covariance[matrixIndex(c, r, dim)] = inCov[idx];
            ++idx;
        }
    sigma = inSigma;  // ← クランプなし！
}
```

### 2. update 内の除算

**`CmaEsOptimizerDynamic.cpp:145-146`**

```cpp
double yRow = (candidate[row] - oldMean[row]) / sigma;     // /0 リスク
double yCol = (candidate[col] - oldMean[col]) / sigma;     // /0 リスク
```

### 3. setSigma も同様（BUG-012 参照）

**`CmaEsOptimizerDynamic.h:29`**

```cpp
void setSigma(double s) noexcept { sigma = s; }  // ← クランプなし！
```

## 再現条件

1. `CmaEsOptimizerDynamic` の状態がシリアライズされ、sigma=0.0 が含まれている
   （設定ファイルの破損、旧バージョンの互換性問題など）
2. `deserializeFrom` が呼び出され、sigma=0.0 が設定される
3. `update()` が呼び出され、除算-by-ゼロ

## 影響

- AllpassDesigner の CMA-ES 最適化が破壊される
- 共分散行列に inf/NaN が淬入 → Cholesky 分解失敗 → 最適化停止
- ユーザーはオールパスフィルタ設計が機能しないと感じる

## 修正案

`deserializeFrom` で sigma をクランプする：

```cpp
void CmaEsOptimizerDynamic::deserializeFrom(const double* inMean, const double* inCov, double inSigma) {
    std::copy(inMean, inMean + dim, mean.begin());
    int idx = 0;
    for (int r = 0; r < dim; ++r)
        for (int c = r; c < dim; ++c) {
            covariance[matrixIndex(r, c, dim)] = inCov[idx];
            covariance[matrixIndex(c, r, dim)] = inCov[idx];
            ++idx;
        }
    sigma = std::clamp(inSigma, params.sigmaMin, params.sigmaMax);  // クランプ追加
}
```

## 補足

`CmaEsOptimizer`（固定次元版）も同じ問題を抱えている（BUG-011）。
両クラスの `deserializeFrom` と `setSigma`（Dynamic版のみ）は
すべて `[params.sigmaMin, params.sigmaMax]` でクランプする必要がある。
