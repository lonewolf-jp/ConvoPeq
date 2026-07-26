# BUG-012: CmaEsOptimizerDynamic::setSigma が sigma をクランプしない

- **発見日**: 2026-07-26
- **カテゴリ**: 数値計算 / パラメータ検証
- **関連**: CmaEsOptimizerDynamic.h, CmaEsOptimizerDynamic.cpp, AllpassDesigner.cpp, ConvolverProcessor.MixedPhase.cpp
- **リスク**: HIGH
- **修正**: 未

## 概要

`CmaEsOptimizerDynamic::setSigma()` は外部から渡された値をそのまま `sigma` メンバに代入するが、
`[params.sigmaMin, params.sigmaMax]` の範囲にクランプしていない。

この関数は AllpassDesigner と ConvolverProcessor.MixedPhase から呼び出され、
ユーザー設定可能な `cmaesInitialSigma` が直接渡される。
値が 0 または負の場合、`update()` 内の除算-by-ゼロが発生する。
値が非常に大きい場合、最適化が不安定になる。

## 該当箇所

### 1. setSigma（クランプなし）

**`CmaEsOptimizerDynamic.h:29`**

```cpp
void setSigma(double s) noexcept { sigma = s; }  // ← クランプなし！
```

### 2. 呼び出しチェーン

```
AllpassDesigner.cpp:303-304
  if (config.cmaesInitialSigma > 0.0)
      optimizer.setSigma(config.cmaesInitialSigma);  // 0.0チェックはあるが上限なし

ConvolverProcessor.MixedPhase.cpp:484
  designer_config.cmaesInitialSigma = 1.0;  // この値が setSigma に渡される
```

### 3. update 内の除算

**`CmaEsOptimizerDynamic.cpp:145-146`**

```cpp
double yRow = (candidate[row] - oldMean[row]) / sigma;     // /0 リスク
double yCol = (candidate[col] - oldMean[col]) / sigma;     // /0 リスク
```

### 4. パラメータ設定

**`AllpassDesigner.h:76-85`**

```cpp
double cmaesInitialSigma = 0.3;  // デフォルト
// ...
AllpassDesignerConfig() {
    cmaesParams.sigmaMin = 1e-6;
    cmaesParams.sigmaMax = 2.0;
    // ...
}
```

`cmaesInitialSigma` はユーザー設定可能であり、0.0 より小さい値や非常に大きい値が
設定される可能性がある。`> 0.0` のチェックはあるが、`sigmaMax` へのクランプはない。

## 再現条件

1. `cmaesInitialSigma` が 0.0（チェックで除外されるが負の値は除外されない）
   または非常に大きい値（例: 100.0）として設定される
2. `setSigma()` が呼び出され、クランプなしで sigma が設定される
3. `update()` が呼び出され、除算-by-ゼロまたは不安定な最適化が発生

## 影響

- sigma=0: `update()` で除算-by-ゼロ → inf/NaN が共分散行列に淬入
- sigma=負: 同様の問題 + ステップサイズ適応が逆方向に働く
- sigma=非常に大きい: 候補点が飛び先を逸脱 → 最適化発散

## 修正案

`setSigma` で sigma をクランプする：

```cpp
void setSigma(double s) noexcept {
    sigma = std::clamp(s, params.sigmaMin, params.sigmaMax);
}
```

## 補足

`AllpassDesigner.cpp:273-278` では、`setParams` を呼ぶ前に `sigmaMin`/`sigmaMax` を
`cmaesInitialSigma` に合わせて調整しているが、`setSigma` 自体はクランプを行わない。
このため、`setParams` で設定された範囲内であっても、`setSigma` が範囲外の値を受け入れてしまう。
