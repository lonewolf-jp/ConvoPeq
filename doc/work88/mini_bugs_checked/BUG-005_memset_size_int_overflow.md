# BUG-005: memset サイズの int 溢れリスク

- **発見日**: 2026-07-26
- **カテゴリ**: 整数オーバーフロー / メモリ破壊
- **関連**: MKLNonUniformConvolver, ConvolverProcessor.Runtime
- **リスク**: LOW
- **修正**: 未

## 概要

`MKLNonUniformConvolver::ringRead()` および `Get()` 内で、`memset` のサイズ計算が
`int` 型で行われている箇所がある。非常に大きな `n`（要求サンプル数）が渡された場合、
`n * sizeof(double)` が `int` の範囲を超えてオーバーフローし、意図よりも小さい
サイズでメモリがゼロクリアされる可能性がある。

## 該当箇所

### 1. ringRead() — 未クリア領域のゼロ埋め

**`MKLNonUniformConvolver.cpp:1488`**

```cpp
if (dst) memset(dst, 0, n * sizeof(double));
```

`n` は `int` 型。`n * sizeof(double)` は `int * size_t` の演算だが、C++の整数昇格規則により、
`n` が `size_t` に昇格される前に `int` として乗算される可能性がある（実装定義）。
MSVCでは `int * size_t` は `size_t` に昇格してから乗算されるため安全だが、
他のコンパイラでは未定義動作となり得る。

### 2. ringRead() — 読み出し不足分のゼロ埋め

**`MKLNonUniformConvolver.cpp:1501`**

```cpp
if (toRead < n)
    memset(dst + toRead, 0, (n - toRead) * sizeof(double));
```

同様の問題。`(n - toRead)` は `int`、`sizeof(double)` は `size_t`。
`int * size_t` の乗算結果は実装定義。

### 3. Get() — 未クリア領域のゼロ埋め

**`MKLNonUniformConvolver.cpp:1663`**

```cpp
memset(output, 0, static_cast<size_t>(numSamples) * sizeof(double));
```

この箇所は `static_cast<size_t>` で安全にキャストされている。**問題なし。**

## 影響

- `n` が `INT_MAX / sizeof(double)` ≈ 268,435,455 を超える必要がある
- オーディオ処理でこのサイズのバッファを要求することは極めて稀
- MSVCでは安全（整数昇格で `size_t` に昇格）
- 他のコンパイラ（GCC/Clang）でも通常は `size_t` に昇格するが、標準では未定義

## 修正案

`n * sizeof(double)` を `static_cast<size_t>(n) * sizeof(double)` に統一：

```cpp
memset(dst, 0, static_cast<size_t>(n) * sizeof(double));
```

## 類似箇所（既に修正済み）

`Get()` の line 1663, 1714, 1837, 1839 は既に `static_cast<size_t>` を使用。
`ConvolverProcessor.Runtime.cpp:1165` も `static_cast<size_t>` を使用。
