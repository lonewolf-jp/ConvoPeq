# BUG-003: Float処理パスでTruePeak Detectorが未実行

- **発見日**: 2026-07-26
- **カテゴリ**: 信号処理 / 計測機能欠落
- **関連**: DSPCoreDouble, DSPCoreIO, TruePeakDetector
- **リスク**: MEDIUM
- **修正**: 未

## 概要

Double処理パス (`DSPCoreDouble.cpp:698`) ではTruePeak検出（BS.1770-4/5準拠）が
実行されているが、Float処理パス (`DSPCoreIO.cpp`) の `processOutput` には
`truePeakDetector.processBlock()` の呼び出しが存在しない。

## 該当箇所

### Doubleパス（正しい実装）
**`AudioEngine.Processing.DSPCoreDouble.cpp:695-698`**

```cpp
// ★ [P1-2] TruePeak/LUFS 計測を kOutputHeadroom + ディザ後に移動
// TruePeak検出（BS.1770-4/5準拠）
truePeakDetector.processBlock(dataL, dataR, numSamples);

// LUFSブロック平均電力（BS.1770-4/5 + EBU R128）
loudnessMeter.processBlock(dataL, dataR, numSamples);
```

### Floatパス（バグ）
**`AudioEngine.Processing.DSPCoreIO.cpp:376-378`**

```cpp
auto& dc = dcBlockers();
dc.outputL.process(dataL, numSamples);
if (dataR) dc.outputR.process(dataR, numSamples);

// ★ truePeakDetector.processBlock() がここで呼ばれていない
// ★ loudnessMeter.processBlock() もここで呼ばれていない

// NaN/Inf scrub ...
```

## 影響

- Float処理モードでTruePeakメーターが常に0または未更新状態となる
- BS.1770準拠のTruePeak測定が機能しない
- ユーザーがFloat/Doubleを切り替えた際にTruePeak表示が不正確になる

## 修正案

`DSPCoreIO.cpp` の `processOutput()` 内で、NaN/Inf Scrubの後、
Hard Clampの前に以下を追加：

```cpp
truePeakDetector.processBlock(dataL, dataR, numSamples);
loudnessMeter.processBlock(dataL, dataR, numSamples);
```
