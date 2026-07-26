# BUG-008: MKLNonUniformConvolver.cpp の memcpy/memset でsize_t昇格UB (13箇所)

- **発見日**: 2026-07-26
- **カテゴリ**: 整数オーバーフロー / メモリ破壊
- **関連**: MKLNonUniformConvolver.cpp
- **リスク**: HIGH
- **修正**: 未

## 概要

`MKLNonUniformConvolver.cpp` 全体で、FFTパーティションセットアップ・リング
バッファ操作・FDL 操作時において、`memcpy`/`memset` のサイズ引数が
`int` 整数または `int * size_t` の混合となっており、`static_cast<size_t>` が
抜けている。Loaders（非RT）/RTの双方で影響あり。

## 該当箇所 (13箇所)

| 行 | コード |
|----|--------|
| 1030 | `memset(tempTime, 0, l.fftSize * sizeof(double));` |
| 1037 | `memcpy(tempTime, irSrc + copyStart, copyLen * sizeof(double));` |
| 1045 | `memcpy(l.irFreqDomain, tempFreq, l.complexSize * 2 * sizeof(double));` |
| 1080-89 | `memcpy(swapSoA, realF, l.complexSize * sizeof(double));` 周辺 |
| 1156 | `memcpy(scratch, src, scratchSize * sizeof(double));` |
| 1384 | `memcpy(mirrorFDLSlot, currentFDLSlot, l.partStride * sizeof(double));` |
| 1417 | `memset(m_ringBuf, 0, finalSize * sizeof(double));` |
| 1488 | `memset(dst, 0, n * sizeof(double));` |
| 1501 | `memset(dst + toRead, 0, (n - toRead) * sizeof(double));` ← **アンダーフローも** |
| 1538, 1540 | `memcpy` 関連 |
| 1628, 1637 | `memcpy`/`memset` 関連 |

## 影響

- 通常の計算範囲では正常動作
- Line 1501 の特殊リスク: `ringRead()` では `toRead = std::min(n, m_ringAvail)`
  で `toRead <= n` が保証されるが、何らかの race/corruption で `toRead > n` が
  起きた場合、`(n - toRead)` は負の `int` となり、`* sizeof(double)` 後に
  `size_t` に昇格して**巨大な unsigned 値**として `memset` に渡る
  → buffer overflow で defense-ngate 級メモリ破壊

## 修正案

すべて `static_cast<size_t>(...)` でキャスト：

```cpp
memset(tempTime, 0, static_cast<size_t>(l.fftSize) * sizeof(double));
memcpy(tempTime, irSrc + copyStart, static_cast<size_t>(copyLen) * sizeof(double));
memcpy(l.irFreqDomain, tempFreq, static_cast<size_t>(l.complexSize) * 2 * sizeof(double));
// ... 他の行も同様

if (toRead < n)
    memset(dst + toRead, 0,
           static_cast<size_t>(n - toRead) * sizeof(double)); // toRead > n の場合の defence も追加
```

修正と同時に `toRead > n` の場合は早期 return する defence コードを追加すべき
（`ringRead` の事前条件で `n <= toRead` となった時に呼び出し側でリセット）。
