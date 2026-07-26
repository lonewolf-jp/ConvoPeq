# BUG-004: Float処理パスに `_mm256_zeroupper()` が欠落

- **発見日**: 2026-07-26
- **カテゴリ**: パフォーマンス / AVX-SSE遷移ペナルティ
- **関連**: DSPCoreDouble, DSPCoreIO, TruePeakDetector, LoudnessMeter
- **リスク**: LOW
- **修正**: 未

## 概要

Double処理パスではAVX2命令を使用した処理の後、レガシーSSE/スカラーコードへの
移行前に `_mm256_zeroupper()` を呼び出しているが、Float処理パスでは
AVX2命令（NaN/Inf Scrubの`__m256d`演算）を使用しているにも関わらず
`_mm256_zeroupper()` が存在しない。

## 該当箇所

### Doubleパス（正しい実装）
**`AudioEngine.Processing.DSPCoreDouble.cpp:741-742`**

```cpp
    // AVX→legacy SSE 境界: _mm256_zeroupper() を配置
    _mm256_zeroupper();

    juce::FloatVectorOperations::copy(buffer.getWritePointer(0, 0), dataL, numSamples);
```

### Floatパス（バグ）
**`AudioEngine.Processing.DSPCoreIO.cpp:505-514`**

```cpp
    // NaN/Inf scrub（ここで __m256d を使用）
    // ... AVX2 命令実行 ...

    applyFixedLatencyDelay(dataL, dataR, numSamples);  // スカラー

    for (int i = 0; i < numSamples; ++i)
        dstL[i] = static_cast<float>(juce::jlimit(...));  // スカラー
```

Floatパスでは第2のNaN/Inf Scrubブロック（lines 470-505）で
`__m256d`を使用したAVX2命令を実行した後、`_mm256_zeroupper()` なしで
スカラーコードに遷移している。

## 影響

- Haswell/Broadwell世代CPU: AVX→SSE遷移ペナルティ（～60 cycle/命令）が
  スカラーコード全体に適用される
- Skylake以降: ペナルティは軽減されているが、完全には除去されていない
- JUCEの`FloatVectorOperations`内部でもSSE命令が使用される可能性があり、
  間接的なペナルティが発生しうる
- 高サンプリングレート・小バッファ（～32 samples）での影響が相対的に大きい

## 修正案

`DSPCoreIO.cpp` の `processOutput()` 内、NaN/Inf Scrubブロックの後かつ
`applyFixedLatencyDelay()` の前に追加：

```cpp
    _mm256_zeroupper();

    applyFixedLatencyDelay(dataL, dataR, numSamples);
```
