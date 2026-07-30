# BUG-039: CustomInputOversampler::processDown — passthrough 時に出力サイズ分だけ入力を読み過ぎる

## 発見日
2026-07-26

## ファイル
`src/CustomInputOversampler.cpp:836-841`

## 問題
`processDown()` で `upsampleRatio <= 1` または `numStages == 0` のパススルー経路:

```cpp
if (upsampleRatio <= 1 || numStages == 0)
{
    for (int ch = 0; ch < channels; ++ch)
    {
        double* dst = outputBlock.getChannelPointer(ch);
        const double* src = upsampledBlock.getChannelPointer(ch);
        std::memcpy(dst, src, static_cast<size_t>(targetSamples) * sizeof(double));
    }
    return;
}
```

- `targetSamples = outputBlock.getNumSamples()` で決まる
- `dst` は出力ブロック → `targetSamples` は正当
- `src` は **入力（アップサンプル）ブロック**
- upsampleRatio == 1 の場合、入力ブロックのサイズは出力ブロックより**小さい可能性がある**（呼び出し側が出力バッファを大きめに確保する場合）
- `memcpy(dst, src, targetSamples * sizeof(double))` は `src` を `targetSamples` 分だけ読み、入力ブロックの末尾を超過する

### 影響
- 入力バッファの後続メモリ（出力バッファや履歴バッファ）を読み出す → ノイズ/クラックル
- まれにクラッシュ（ページ境界付近の場合）
- 条件: upsampleRatio==1 かつ upsampledBlock のサンプル数 < outputBlock のサンプル数

### リスク評価
- **重大度**: HIGH — 条件次第で常時ノイズ
- **発生頻度**: upsampleRatio==1 時に常時（呼び出し側のバッファサイズ設計依存）
- **検出性**: 通常の使用では発見困難

### 修正方針
コピー量を `min(targetSamples, upsampledBlock.getNumSamples())` に制限する。
