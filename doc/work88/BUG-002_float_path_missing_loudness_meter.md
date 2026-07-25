# BUG-002: Float処理パスでLoudness Meterが未実行

- **発見日**: 2026-07-26
- **カテゴリ**: 信号処理 / 計測機能欠落
- **関連**: DSPCoreDouble, DSPCoreIO, LoudnessMeter
- **リスク**: MEDIUM
- **修正**: 未

## 概要

Double処理パスでは出力直前に `loudnessMeter.processBlock()` を呼び出して
LUFSブロック平均電力を更新しているが、Float処理パスでは同呼び出しが存在しない。
その結果、Float処理モードではLUFSラウドネスメーターが正確に動作しない。

## 該当箇所

### Doubleパス（正しい実装）
**`AudioEngine.Processing.DSPCoreDouble.cpp:700-702`**

```cpp
// LUFSブロック平均電力（BS.1770-4/5 + EBU R128）
loudnessMeter.processBlock(dataL, dataR, numSamples);

// ★ [P1-1] Simple Peak Limiter
peakLimiter.processBlock(dataL, dataR, numSamples, kPLThreshold, kPLKnee);
```

### Floatパス（バグ）
**`AudioEngine.Processing.DSPCoreIO.cpp:375` — 以下に`loudnessMeter.processBlock()`の呼び出しがない**

```cpp
auto& dc = dcBlockers();
dc.outputL.process(dataL, numSamples);
if (dataR) dc.outputR.process(dataR, numSamples);

// ★ loudnessMeter.processBlock() がここで呼ばれていない

{
    const __m256d vInf = _mm256_set1_pd(1.0e300);
    // NaN/Inf scrub ...
}
```

## 影響

- Float処理モードでLUFSメーター値が常に0または未更新状態となる
- ユーザーがFloat/Doubleを切り替えた際にラウドネス表示が不正確になる
- EBU R128準拠のラウドネス測定が必要な業務利用で問題となる可能性

## 修正案

`DSPCoreIO.cpp` の `processOutput()` 内、DC Blocker適用後・NaN/Inf Scrub前に
`loudnessMeter.processBlock()` を追加する：

```cpp
auto& dc = dcBlockers();
dc.outputL.process(dataL, numSamples);
if (dataR) dc.outputR.process(dataR, numSamples);

loudnessMeter.processBlock(dataL, dataR, numSamples); // 追加
```
