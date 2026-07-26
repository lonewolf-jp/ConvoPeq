# BUG-009: 4ファイルで `_mm256_zeroupper()` が AVX2 hot path 後に不足

- **発見日**: 2026-07-26
- **カテゴリ**: パフォーマンス / AVX-SSE 遷移ペナルティ
- **関連**: ConvolverProcessor.Runtime.cpp, CustomInputOversampler.cpp, DSPCoreFloat.cpp, AudioEngine.EQResponse.cpp
- **リスク**: MEDIUM
- **修正**: 未

## 概要

`__m256d` 系 AVX2 命令を用いたホットパスを持つファイルのうち、4ファイルは
関数末尾に `_mm256_zeroupper()` を呼んでいない。これにより、

- 後続のスカラー / SSE 命令が「YMM 上位レーンが汚染された」状態で実行され、
- AVX → SSE 遷移ペナルティが発生する：
  - **Haswell/Broadwell 系**: 1命令あたり最大 60 サイクルのペナルティ
  - **Skylake 以降**: 軽減されたが完全な除去はされない
- `ScopedNoDenormals` ブロック内の後続処理も影響を受け、
- バッファサイズが小さい（～32 samples=> ~0.7ms @48kHz）ケースで
  演奏時間が相対的に短く、バッファオーバーランが顕在化しやすい

## 該当ファイル（`_mm256_zeroupper()` 不在のホットパス）

### 1. `ConvolverProcessor.Runtime.cpp`
- **RISK**: MEDIUM
- **AVX2 使用箇所**: lines 465-480, 522-530, 617-645
  （`applyGainRamp_AVX2`, AVX ウェット/ドライクロスフェード等）
- **後続**: 多数の JUCE `FloatVectorOperations::copy`,
  `activeDelaySmoother`, `applyFixedLatencyDelay`等のスカラー / SSEコード

### 2. `CustomInputOversampler.cpp`
- **RISK**: MEDIUM
- **AVX2 使用箇所**: `isBadSampleV`, `loadStride2`, big-tap FIR
- **後続**: コンパイル単位は [[gnu::target("avx2")]] により AVX2 ターゲット
  限定だが、`process()` 内同一関数で何度も AVX2 → SSE 遷移を繰り返す

### 3. `AudioEngine.Processing.DSPCoreFloat.cpp` (Bug 001 と類似，新規指摘)
- **RISK**: MEDIUM  
- **AVX2 使用箇所**: `applyGainRampBlockAVX2`, `softClipBlockAVX2`
  (lines 102-132, 184-?)
- **比較**: `DSPCoreDouble.cpp:742` では明示的に `_mm256_zeroupper()`
  を呼んでいる（BUG-004）

### 4. `AudioEngine.EQResponse.cpp`
- **RISK**: MEDIUM
- **AVX2 使用箇所**: ファイル全体で `__m256d` を多用
- **後続**: `__m128` / `__m256` float 演算

## 修正案

各ホットパスのスカラー / SSE ループ遷移直前に `_mm256_zeroupper()` を
呼び出す：

```cpp
// 例: DSPCoreFloat.cpp の softClipBlockAVX2 末尾
_mm256_zeroupper();
```

オーバーヘッドは数サイクル以下なので、データ依存分岐ミスのペナルティ（～60 cycle）
を考えると net プラスになる。

## 補足

`LoudnessMeter.cpp:93`, `TruePeakDetector.cpp:181`, `DSPCoreDouble.cpp:742` は
正しく `_mm256_zeroupper()` を呼んでおり、コードベース全体のルールとして
確立されている。BUG-009 ファイル群は漏れである。
