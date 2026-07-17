# Denormal 数値対策 設計ガイドライン

> 作成日: 2026-07-17
> 適用範囲: 全 DSPCore パス（新規 IIR フィルタ追加時）
> 関連Issue: work74 FIX-03

---

## 概要

IEEE 754 の単精度（float）浮動小数点では、絶対値が非常に小さい値（≈ 1.4e-45 未満）は **非正規化数（denormal）** となり、CPU の演算ユニットで著しい性能低下（数十〜数百サイクル）を引き起こす。オーディオ処理の無音区間で IIR フィルタの状態値が denormal 領域に達すると、リアルタイム処理に支障をきたす可能性がある。

## 現状の対策（確認済み）

ConvoPeq では以下の対策が実施されている:

### 1. プロセス全体 FTZ/DAZ 設定

`MKLRealTimeSetup.cpp` でプロセス開始時に FTZ/DAZ を有効化:

```cpp
_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
```

これにより x86/x64 CPU は denormal 値を自動的に 0 にフラッシュする。この設定は全スレッドに適用される。

### 2. 個別 killDenormal（MKLNonUniformConvolver）

`MKLNonUniformConvolver::processLayerBlock()` の IFFT 前に `killDenormalV` 関数で明示的に denormal を除去:

```cpp
// AVX2 版 denormal 除去
inline __m256d killDenormalV(__m256d v) noexcept {
    const __m256d threshold = _mm256_set1_pd(kDenormThreshold);
    const __m256d signMask = _mm256_set1_pd(-0.0);
    const __m256d absV = _mm256_andnot_pd(signMask, v);
    const __m256d mask = _mm256_cmp_pd(absV, threshold, _CMP_GE_OQ);
    return _mm256_and_pd(v, mask);
}
```

### 3. 入力サニタイズ（InputBitDepthTransform）

`InputBitDepthTransform.h` の `sanitizeAndLimit()` で入力段で denormal/NaN を除去:

```cpp
inline void sanitizeAndLimit(double* data, int numSamples) noexcept {
    // NaN/Inf チェック + |v| < kDenormThreshold → 0.0
    // [-1, 1] クランプ
}
```

### 4. NUC 直接処理部

`MKLNonUniformConvolver::processDirectBlock()` の denormal ガード:

```cpp
if (!isFiniteAndAboveThresholdMask(y, kDenormalThreshold))
    y = 0.0;
```

---

## 設計規約（新規 IIR フィルタ追加時）

新規に IIR フィルタを追加する場合は、以下のいずれかの対策を必ず実装すること。

### 推奨パターン A: プロセス全体 FTZ/DAZ に依存する

最もシンプル。`MKLRealTimeSetup.cpp` の FTZ/DAZ 設定が有効である限り、CPU が自動的に denormal を処理する。

**適用条件**: x86/x64 環境で FTZ/DAZ が設定されていること（ConvoPeq では常時設定済み）。

**制限**:
- `float` 型の IIR フィルタでは FTZ が有効でも一部で denormal 性能低下が発生する場合がある（CPU の挙動依存）
- 他のプラットフォーム（ARM 等）では FTZ/DAZ が利用できない可能性がある

### 推奨パターン B: 明示的な killDenormal（推奨）

各 IIR フィルタの「1サンプル処理後」または「ブロック処理後」に denormal 除去を適用する。

```cpp
// double 版（スカラー）
inline double killDenormal(double value) noexcept {
    constexpr double threshold = 1e-15; // 用途に応じて調整
    return (std::abs(value) < threshold) ? 0.0 : value;
}

// float 版（スカラー）
inline float killDenormalF(float value) noexcept {
    constexpr float threshold = 1e-15f;
    return (std::abs(value) < threshold) ? 0.0f : value;
}

// AVX2 版（ダブル4ベクタ）
inline __m256d killDenormalV(__m256d v) noexcept {
    const __m256d threshold = _mm256_set1_pd(1e-15);
    const __m256d signMask = _mm256_set1_pd(-0.0);
    const __m256d absV = _mm256_andnot_pd(signMask, v);
    const __m256d mask = _mm256_cmp_pd(absV, threshold, _CMP_GE_OQ);
    return _mm256_and_pd(v, mask);
}
```

**適用例**:
```cpp
// IIR フィルタの状態更新後
for (int i = 0; i < numSamples; ++i) {
    output[i] = processBiquad(input[i], state);
    output[i] = killDenormal(output[i]); // または状態変数を killDenormal
}
// ブロック終了時に状態をまとめて処理
for (int i = 0; i < stateSize; ++i)
    state[i] = killDenormal(state[i]);
```

### パターン C: 微小値加算

一部の実装で見られるが、本プロジェクトでは推奨しない（演算精度に影響するため）:
```cpp
output[i] += 1e-20; // ← 非推奨。FTZ/DAZ または killDenormal を使用すること
```

---

## 参考: 既存実装箇所

| ファイル | 関数 | 方式 | 備考 |
|---------|------|------|------|
| `MKLRealTimeSetup.cpp` | `setup()` | FTZ/DAZ プロセス全体 | 全スレッド対象 |
| `MKLNonUniformConvolver.cpp` | `processLayerBlock()` | `killDenormalV` AVX2 | IFFT 前 |
| `MKLNonUniformConvolver.cpp` | `processDirectBlock()` | `isFiniteAndAboveThresholdMask` | 直接 FIR 畳み込み後 |
| `InputBitDepthTransform.h` | `sanitizeAndLimit()` | FTZ/DAZ 前提クランプ | 入力段 |

---

## 特記事項

- **`ScopedNoDenormals` は冗長だが無害**: JUCE の `ScopedNoDenormals` は FTZ/DAZ をスコープ単位で設定する。ConvoPeq ではプロセス開始時に一度設定すれば十分だが、二重設定による悪影響はない。既存コードの `ScopedNoDenormals` は維持してよい。
- **`double` の IIR**: double の指数部幅が大きいため、FTZ/DAZ が有効であれば denormal 問題はほぼ発生しない。float の IIR では積極的に `killDenormal` を使用すること。
