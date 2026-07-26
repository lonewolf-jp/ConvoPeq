# BUG-019: 整数オーバーフローリスク in TruePeakDetector バッファオフセット計算

**発見日**: 2026-07-26
**カテゴリ**: 整数演算 / 潜在バグ
**リスク**: LOW（発現条件は極めて稀だが、発現時はUB）

---

## 概要

`TruePeakDetector.cpp:102-103` で `numSamples * 2` / `numSamples * 4` の乗算結果を `int`（32-bit）に格納しており、`numSamples > 536,870,911` の場合に符号付き整数オーバーフロー → 未定義動作を引き起こす。

また、これらの値はオフセット計算（line 109-111）に連鎖使用されるため、発現時の影響範囲が広い。

---

## 該当箇所

```cpp
// TruePeakDetector.cpp:96-111
double TruePeakDetector::processBlock(const double* dataL, const double* dataR, int numSamples) noexcept
{
    if (numSamples <= 0 || !upsampleBuffer)
        return 0.0;

    double* work = upsampleBuffer.get();
    const int up1Samples = numSamples * 2;       // ← オーバーフロー可能性
    const int up2Samples = numSamples * 4;       // ← オーバーフロー可能性

    constexpr int kStage0LOffset = 0;
    const int   kStage0ROffset = up1Samples;
    const int   kStage1LOffset = up1Samples * 2;    // ← 連鎖
    const int   kStage1ROffset = up1Samples * 2 + up2Samples;  // ← 連鎖
```

---

## 発現条件

| 条件 | 値 | 現実性 |
|------|-----|--------|
| `up2Samples = numSamples * 4` のオーバーフロー | `numSamples > 536,870,911` | ❌ 通常のオーディオ処理では発生しない |
| `up1Samples * 2` のオーバーフロー | `numSamples > 1,073,741,823` | ❌ 同上 |
| `up1Samples * 2 + up2Samples` | 両方が正でも和でオーバーフロー | ❌ 同上 |

**現実的なシナリオ**: なし。TruePeakDetector は通常 32〜8192 samples/block で呼ばれる。

---

## それでも修正すべき理由

1. **C++ 符号付き整数オーバーフローは UB**: コンパイラの最適化パスに依存した予測不能動作を引き起こしうる（例: 分岐削除、ループ最適化の破綻）
2. **コードの移植性**: 現行では 32-bit int だが、将来の変更で `numSamples` が巨大になるパスが追加された場合に不意に顕在化
3. **予防的プログラミング**: 同一ファイルの他の箇所では適切に `static_cast<size_t>` が使用されている

---

## 修正案

```cpp
const long long up1Samples = static_cast<long long>(numSamples) * 2;
const long long up2Samples = static_cast<long long>(numSamples) * 4;
```

またはヘルパー関数:
```cpp
const size_t up1Samples = static_cast<size_t>(numSamples) * 2;
const size_t up2Samples = static_cast<size_t>(numSamples) * 4;
```

後者の場合、`kStage0ROffset` 等も `size_t` に変更する必要がある。
