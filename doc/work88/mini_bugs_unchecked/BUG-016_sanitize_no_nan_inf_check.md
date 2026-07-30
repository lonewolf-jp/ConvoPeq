# BUG-016: CmaEsOptimizer / CmaEsOptimizerDynamic の sanitize 関数が NaN/Infinity を処理しない

- **発見日**: 2026-07-26
- **カテゴリ**: 数値計算 / エラー伝播
- **関連**: CmaEsOptimizer.h, CmaEsOptimizerDynamic.h
- **リスク**: MEDIUM
- **修正**: 未

## 概要

`CmaEsOptimizer` と `CmaEsOptimizerDynamic` の両方にある `sanitize()` 関数は、
`std::abs(x) < 1e-15` で 0.0 に丸めるが、**NaN や Infinity のチェックを行わない**。

NaN や Infinity が `sanitize()` を通過すると、共分散行列や平均ベクトルに
淡入し、Cholesky 分解の失敗や最適化の発散を引き起こす。

## 該当箇所

### 1. CmaEsOptimizer::sanitize

**`CmaEsOptimizer.h:201-204`**

```cpp
static inline double sanitize(double x) noexcept
{
    return (std::abs(x) < 1e-15) ? 0.0 : x;  // ← NaN/Inf をそのまま通す！
}
```

### 2. CmaEsOptimizerDynamic::sanitize

**`CmaEsOptimizerDynamic.h:50`**

```cpp
static double sanitize(double x) { return (std::abs(x) < 1e-15) ? 0.0 : x; }
```

### 3. sanitize の使用箇所

**CmaEsOptimizer:**
- `sample()` 行 122: `candidates[populationIndex][dim] = sanitize(mean[dim] + sigma * correlated);`
- `update()` 行 167: `covariance[row * kDim + column] = sanitize(...);`
- `update()` 行 184: `mean[dim] = sanitize(newMean[dim]);`
- `toParcor()` 行 192: `parcor[i] = sanitize(std::tanh(unconstrained[i]));`

**CmaEsOptimizerDynamic:**
- `sample()` 行 84: `candidate[d] = sanitize(mean[d] + sigma * correlated);`
- `update()` 行 149: `covariance[...] = sanitize(...);`

## 問題の詳細

### NaN の伝播

```cpp
// sanitize(NaN) の結果
std::abs(NaN) < 1e-15  →  false  (NaN はどんな比較でも false)
→ NaN がそのまま返される
```

### Infinity の伝播

```cpp
// sanitize(Infinity) の結果
std::abs(Infinity) < 1e-15  →  false
→ Infinity がそのまま返される
```

### 具体的なシナリオ

1. `sigma = 0`（BUG-011/013 による）で `update()` が呼び出される
2. `(candidate[row] - oldMean[row]) / sigma` → **Infinity** または **NaN**
3. `sanitize(Infinity)` → **Infinity**（そのまま）
4. 共分散行列に Infinity が淬入
5. `computeCholesky()` で `std::sqrt(Infinity)` → **Infinity**
6. 次回 `sample()` で `sigma * Infinity` → **Infinity**
7. 候補点が全て Infinity → 最適化完全破綻

## 影響

- NaN/Infinity が共分散行列に淬入 → Cholesky 分解失敗
- 最適化が発散または停止
- ユーザーはオールパスフィルタ設計やノイズシェーパー学習が機能しないと感じる

## 修正案

`std::isfinite()` チェックを追加する：

```cpp
static inline double sanitize(double x) noexcept
{
    if (!std::isfinite(x) || std::abs(x) < 1e-15)
        return 0.0;
    return x;
}
```

## 補足

- `std::isfinite()` は NaN、+Inf、-Inf のすべてを `false` として返す
- この修正は BUG-011/013（sigma=0 による除算-by-ゼロ）の二次的な防御線
- 1e-15 の閾値自体も疑問（BUG-016 として別途検討可能）が、
  このバグレポートでは NaN/Infinity の処理を優先する
