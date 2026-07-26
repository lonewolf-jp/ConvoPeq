# BUG-006: clone() での memcpy サイズ計算の int 溢れリスク

- **発見日**: 2026-07-26
- **カテゴリ**: 整数オーバーフロー / メモリ破壊
- **関連**: ConvolverProcessor.h, StereoConvolver
- **リスク**: LOW
- **修正**: 未

## 概要

`StereoConvolver::clone()` 内でIRデータをコピーする `std::memcpy` のサイズ計算が
`irDataLength * sizeof(double)` となっており、`irDataLength` は `int` 型である。
非常に大きなIR（`irDataLength` が `INT_MAX / sizeof(double)` ≈ 268,435,455 を超える）
の場合、乗算結果が `int` の範囲を超えてオーバーフローし、
意図よりも小さいサイズでメモリコピーが行われる。

## 該当箇所

**`ConvolverProcessor.h:821-822`**

```cpp
std::memcpy(l.get(), irData[0], irDataLength * sizeof(double));
std::memcpy(r.get(), irData[1], irDataLength * sizeof(double));
```

`irDataLength` は `int` 型（`StereoConvolver::irDataLength` メンバ）。
`irDataLength * sizeof(double)` は `int * size_t` の演算。

C++の整数昇格規則により、`int` が `size_t` に昇格されるのは乗算**前**であるため、
乗算自体は `int` として行われる可能性がある（実装定義）。
MSVCでは `int * size_t` は `size_t` に昇格してから乗算されるが、
標準では未定義動作となり得る。

## 影響

- `irDataLength` が ~268M samples を超える必要がある（48kHzで ~5,590秒 = ~93分のモノラルIR）
- 実用的なIRファイルサイズでは到達しないが、非常に大きなIRや悪意的な入力で問題になり得る
- オーバーフローが発生した場合、コピーが不完全となり、
  `init()` に不完全なIRデータが渡される → 異常なオーディオ出力

## 修正案

`static_cast<size_t>` を追加：

```cpp
std::memcpy(l.get(), irData[0], static_cast<size_t>(irDataLength) * sizeof(double));
std::memcpy(r.get(), irData[1], static_cast<size_t>(irDataLength) * sizeof(double));
```

## 類似箇所

- `MKLNonUniformConvolver.cpp:1488,1501` — `memset` サイズ計算（BUG-005）
- `ConvolverProcessor.Runtime.cpp:388,390` — `memcpy` サイズ計算（同様のパターン）
- `ConvolverProcessor.Runtime.cpp:1165,1174,1179` — 既に `static_cast<size_t>` 済み
