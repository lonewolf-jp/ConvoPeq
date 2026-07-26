# BUG-007: ConvolverProcessor.Runtime.cpp の memcpy サイズ計算が int 昇格UB

- **発見日**: 2026-07-26
- **カテゴリ**: 整数オーバーフロー / メモリ破壊
- **関連**: ConvolverProcessor.Runtime.cpp
- **リスク**: HIGH
- **修正**: 未

## 概要

`ConvolverProcessor::process()` 内、ドライ信号をディレイバッファに書き込む
`std::memcpy` のサイズ引数が `samplesFirst * sizeof(double)` および
`samplesSecond * sizeof(double)` となっており、`int * size_t` の暗黙昇格に依存している。

## 該当箇所

**`ConvolverProcessor.Runtime.cpp:383-391`**

```cpp
for (int ch = 0; ch < procChannels; ++ch)
{
    const double* src = block.getChannelPointer(ch);
    double* buf = delayBuf[ch];

    int samplesFirst = std::min(numSamples, DELAY_BUFFER_SIZE - wPos);
    int samplesSecond = numSamples - samplesFirst;

    std::memcpy(buf + wPos, src, samplesFirst * sizeof(double));   // ← UB
    if (samplesSecond > 0)
        std::memcpy(buf, src + samplesFirst, samplesSecond * sizeof(double));   // ← UB
}
```

`samplesFirst`, `samplesSecond` は `int` 型。`samplesFirst * sizeof(double)` は
`int × size_t` の乗算。

## 問題の詳細

`sizeof(double)` の型は `size_t` (64-bit Windows で 8 byte)。
`samplesFirst` (int) は乗算前に `size_t` に暗黙昇格されるが、
その動作は C++ 標準上「実装定義(implementation-defined)」である。

- **If**: 暗黙昇格が乗算**前**に行われた場合、結果は正しく `size_t` で表現される
  (MSVC などはこの動作)
- **If**: 暗黙昇格が乗算**後**に行われた場合、`int * size_t` の結果は
  符号付きオーバーフローを起こす可能性があり UB

`ConvolverProcessor.Runtime.cpp:159,161` (同じファイル内別関数) では
既に `static_cast<size_t>(samplesFirst) * sizeof(double)` を使用しており、
コードベース全体で `static_cast<size_t>` を必須としている方針と矛盾する。

## 影響

- 通常時: 動作する（昇格前の size_t 動作が起きるため）
- 異常入力: `samplesFirst` または `samplesSecond` が負値やゼロの場合、
  暗黙昇格の結果が `memcpy` サイズの誤りとなり、バッファを越えて書き込み、
  セグフォルトまたは出所不明の音出力が発生し得る

## 修正案

```cpp
std::memcpy(buf + wPos, src, static_cast<size_t>(samplesFirst) * sizeof(double));
if (samplesSecond > 0)
    std::memcpy(buf, src + samplesFirst, static_cast<size_t>(samplesSecond) * sizeof(double));
```

## 関連箇所

同じパターン（`static_cast<size_t>` 未使用）が他にも多数存在するため、
別途 BUG 報告を作成予定:
- `ConvolverProcessor.LoaderThread.cpp:220,221`
- `ConvolverProcessor.Lifecycle.cpp:248,249`
- `MKLNonUniformConvolver.cpp:1030,1037,1045,1080-1089,1156,1384,1417,1488,1501,1538,1540,1628,1637`
- `IRDSP.cpp:75,83,101`
