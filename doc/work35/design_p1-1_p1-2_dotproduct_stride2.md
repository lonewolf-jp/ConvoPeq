# P1-1 / P1-2 詳細設計書

> 作成日: 2026-06-13
> 対象ファイル: `src/CustomInputOversampler.cpp`
> 関連構造体: `CustomInputOversampler::Stage`
> 関連関数: `decimateStage`, `dotProductAvx2`, `interpolateStage`

---

## 目次

1. [P1-1: バリデーションチェックループ外出し](#p1-1-バリデーションチェックループ外出し)
2. [P1-2: dotProductAvx2 stride-2版追加](#p1-2-dotproductavx2-stride-2版追加)
3. [統合設計: decimateStage 全体像](#統合設計-decimatestage-全体像)
4. [性能試算](#性能試算)
5. [リスク評価](#リスク評価)

---

## P1-1: バリデーションチェックループ外出し

### 現状の問題

現在の `decimateStage` では、出力サンプル `n` ごとに**ループ内で**以下のチェックを実行している:

```
for (int n = 0; n < outSamples; ++n)
{
    // ★ ループ内チェック1: バウンドチェック (base < centerTap || base >= capacity)
    // ★ ループ内チェック2: centerCoeff * history[...] の bad sample チェック
    // ★ ループ内チェック3 (AVX2パス): 各タップのバウンドチェック × 4
    // ★ ループ内チェック4 (AVX2パス): 各タップの isBadSampleV × 4
    //         ↓ bad 検出で全AVX2計算を破棄しスカラーフォールバック
    // ★ ループ内チェック5 (スカラーパス): 各タップのバウンド + isBadSample
}
```

**問題点**:

- チェック3〜4の `bad == true` によりAVX2経路で計算した部分結果が完全に破棄される
- スカラーフォールバックは `centerCoeff` から再計算するため二重計算
- チェックがループ内にあるため、コンパイラのソフトウェアパイプライニングを阻害

### 改善設計

**基本戦略**: 出力サンプル `n` ごとに、convolution の全タップ範囲に対するバリデーションを事前（ループ外）に1回だけ実施する。

#### Step 1a: nループ外部へのバウンドチェック完全外出し（Review 3 改善案）

`base = keep + (n << 1)` は `n` に対して**単調増加**する性質を利用し、nループ全体の最小・最大インデックスをループ**外**で一括チェックする。これにより nループ内の分岐を完全に排除できる。

```cpp
// nループ全体の最小base（n=0）と最大base（n=outSamples-1）
const int baseMin = keep;
const int baseMax = keep + ((outSamples - 1) << 1);

// centerTap用バウンドチェック（nループ全体）
// centerCoeff が history[base - centerTap] にアクセスするため:
//   base - centerTap >= 0 → base >= centerTap → keep >= centerTap
//   base < capacity → baseMax < capacity
const bool centerTapOk = (keep >= stage.centerTap)
                       && (baseMax < capacity);

// convolutionタップ範囲のバウンドチェック（nループ全体）
// 最小convIndex: n=0, r=convCount-1 → baseMin - convParity - ((convCount-1) << 1)
// 最大convIndex: n=outSamples-1, r=0 → baseMax - convParity
const int globalMinConvIdx = keep - stage.convParity - ((stage.convCount - 1) << 1);
const int globalMaxConvIdx = baseMax - stage.convParity;
const bool convTapOk = (globalMinConvIdx >= 0)
                     && (globalMaxConvIdx < capacity);

if (!centerTapOk || !convTapOk)
{
    // ブロック全体が境界違反 → 全出力を0クリアしてcorruptionマーク
    std::memset(output, 0, static_cast<size_t>(outSamples) * sizeof(double));
    markCorruptionDetected();
    return;
}

// ── ここまできたら、nループ内は100%境界安全が保証される ──
for (int n = 0; n < outSamples; ++n)
{
    const int base = keep + (n << 1);

    // [残るチェックは centerCoeff * history の bad sample のみ]
    // history の値は動的に変化するため事後チェックで対応
    const double centerSample = history[base - stage.centerTap];
    double acc = stage.centerCoeff * centerSample;
    if (isBadSample(acc))
    {
        output[n] = 0.0;
        markCorruptionDetected();
        continue;
    }

    // ── AVX2パス（完全に分岐のないストレートコード） ──
    // またはスカラーパス
    ...

    // 事後 isBadSample チェック
    if (isBadSample(acc))
    {
        output[n] = 0.0;
        markCorruptionDetected();
    }
    else
    {
        if (fastAbs(acc) < kDenormThreshold) acc = 0.0;
        output[n] = acc;
    }
}
```

**効果**:

- nループ内の条件分岐が激減（バウンドチェックが完全消失）
- コンパイラ（MSVC）が nループの自動ベクトル化やソフトウェアパイプライニングを展開しやすくなる
- corruption 検出がブロック単位になるが、バッファ異常時は全サンプルが同様に異常であるため問題なし

#### Step 1b: 出力サンプル n の動的事前バリデーション（バックアップ案）

上記の完全外出しが適用できないケース（個別サンプル単位で異なるバリデーションが必要な場合）のバックアップ:

```cpp
for (int n = 0; n < outSamples; ++n)
{
    const int base = keep + (n << 1);

    // 1-c: centerCoeff の bad sample チェック（動的値のためnループ内で必要）
    const double centerSample = history[base - stage.centerTap];
    double acc = stage.centerCoeff * centerSample;
    if (isBadSample(acc))
    {
        output[n] = 0.0;
        markCorruptionDetected();
        continue;
    }

    // ── AVX2パス（バリデーション済みなのでチェックなし） ──
    ...
}
```

#### Step 2: AVX2パスの簡略化（チェック削除）

事前バリデーションにより以下が保証される:

- 全タップインデックスが `[0, capacity)` 範囲内
- `centerCoeff` は bad sample でない

これにより、AVX2パス内では以下を削除できる:

- バウンドチェック: `if (idx0 < 0 || idx0 >= capacity || ...)`  → **削除**
- `isBadSampleV` チェック: `if (isBadSampleV(vSamples))` → **削除**（ただし後述の注意点あり）

```cpp
// 事前バリデーション後のAVX2パス（チェックなし）
if (stage.convCount >= 4)
{
    usedAvxPath = true;
    __m256d vAcc = _mm256_setzero_pd();
    int r = 0;
    const int simdEnd = (stage.convCount / 4) * 4;
    for (; r < simdEnd; r += 4)
    {
        // バウンドチェック: 削除（事前バリデーション済み）
        // isBadSampleV: 削除（事前バリデーション済み）
        const double s0 = history[base - stage.convParity - ((r + 0) << 1)];
        const double s1 = history[base - stage.convParity - ((r + 1) << 1)];
        const double s2 = history[base - stage.convParity - ((r + 2) << 1)];
        const double s3 = history[base - stage.convParity - ((r + 3) << 1)];
        const __m256d vSamples = _mm256_set_pd(s3, s2, s1, s0);
        const __m256d vCoeffs = _mm256_loadu_pd(coeffs + r);
        vAcc = _mm256_fmadd_pd(vSamples, vCoeffs, vAcc);
    }
    // 水平加算（既存の vextractf128 + hadd）
    ...
    // スカラー剰余タップ
    for (; r < stage.convCount; ++r)
    {
        // バウンドチェック: 削除
        // isBadSample: 削除
        acc += coeffs[r] * history[base - stage.convParity - (r << 1)];
    }
}
```

#### Step 3: NaN/Inf の事後チェック

事前バリデーションで `centerCoeff` のbad sampleはチェックするが、historyバッファ内の NaN/Inf の混入は事前検出できない（履歴データは動的に変化する）。

→ **事後チェック** で対応:

```cpp
// AVX2パスまたはスカラーパスの後
if (isBadSample(acc))
{
    output[n] = 0.0;
    markCorruptionDetected();
    // continue は不要（既に値決定済み）
}
else
{
    if (fastAbs(acc) < kDenormThreshold) acc = 0.0;
    output[n] = acc;
}
```

**注**: 従来の `bad` フラグによるAVX2放棄 + スカラー再計算のパスは完全に削除される。NaN/Infが検出された場合は出力を0.0にするだけで、再計算は行わない（NaNが伝搬した convolution 結果を使っても意味がないため）。

### P1-1 による影響

| 影響 | 詳細 |
|------|------|
| **出力値の変化** | なし。事前バリデーションは従来と同じ条件をチェックする |
| **corruption 検出** | 従来より少なくなる場合がある（バウンド違反は事前検出、bad sampleは事後検出） |
| **コード量** | decimateStage から `bad` フラグの分岐構造が消失し可読性向上 |
| **パフォーマンス** | 通常経路（非corruption）でAVX2パスが一切中断しない |

---

## P1-2: dotProductAvx2 stride-2版追加

### 背景

現在の `dotProductAvx2` は **連続メモリアクセス**（stride 1）を前提として設計されている:

```cpp
// 現状の dotProductAvx2（連続アクセス用）
for (; i <= n - 16; i += 16)
{
    acc0 = _mm256_fmadd_pd(_mm256_loadu_pd(x + i),      _mm256_load_pd(coeffs + i),      acc0);
    acc1 = _mm256_fmadd_pd(_mm256_loadu_pd(x + i + 4),  _mm256_load_pd(coeffs + i + 4),  acc1);
    ...
}
```

`interpolateStage` では history から **連続したウィンドウ** を切り出すため、上記関数がそのまま使用できる。

一方 `decimateStage` では halfband フィルタ構造により **stride-2 アクセス** となる:

```
idx = base - convParity - (r << 1)    // r=0,1,2,... で 2 ずつ減少
```

このため `decimateStage` では `dotProductAvx2` を呼べず、インラインで個別ロード + `_mm256_set_pd` を使用している。

### 設計方針

**新しい関数 `dotProductDecimateAvx2`** を追加する:

```cpp
/// @brief Stride-2 半帯域畳み込み用ドット積（AVX2）
///
/// decimateStage 用。history から stride 2 でサンプルを収集し、
/// 連続配置された係数との FMA を4アキュムレータunrollで計算する。
///
/// @param history    履歴バッファへのポインタ（ベース位置を指す）
/// @param coeffs     連続配置された係数（convCoeffs）
/// @param convCount  タップ数（4の倍数を推奨、__assume で伝達）
/// @return           畳み込み結果（スカラー）
///
/// @note 呼び出し側で history ポインタの有効範囲は事前保証すること。
///       本関数内ではバウンドチェック・bad sampleチェックを行わない。
static double dotProductDecimateAvx2(const double* __restrict history,
                                     const double* __restrict coeffs,
                                     int convCount) noexcept;
```

### メモリアクセスパターン

```
history バッファ:  [ -6 | -5 | -4 | -3 | -2 | -1 |  0 | +1 | ... ]
                      ↑                  ↑             ↑
                     r=3                r=2        r=1  r=0

係数配列:  [ coeffs[0] | coeffs[1] | coeffs[2] | coeffs[3] | ... ]
```

目的の演算:

```
acc = coeffs[0]*history[0] + coeffs[1]*history[-2] + coeffs[2]*history[-4] + coeffs[3]*history[-6] + ...
```

### SIMD実装

4要素 (r=0,1,2,3) を1バッチとして処理する。

**Step 1**: 4つのstride-2サンプルを256-bitレジスタにロードする

半帯域フィルタではアクセスが2刻みだが、偶数の相対位置 (0, -2, -4, -6) はメモリ上で:

```
position -6: 要素[0]
position -5: 不要
position -4: 要素[1]
position -3: 不要
position -2: 要素[2]
position -1: 不要
position  0: 要素[3]  ← history[base-convParity]（現在位置）
```

要素 [0], [1] は lower lane、[2], [3] は upper lane に入るよう、以下のように挿入する:

```cpp
// history が base-convParity を指すとすると:
// 必要な要素: history[0], history[-2], history[-4], history[-6]
// = { [0], [-2], [-4], [-6] } in ascending address order

// 2つの128-bit load（各2要素）
__m128d vLow  = _mm_loadu_pd(history - 6);  // = { [-6], [-5] } → [-6] を取得
__m128d vLow2 = _mm_loadu_pd(history - 4);  // = { [-4], [-3] } → [-4] を取得
__m128d vHi   = _mm_loadu_pd(history - 2);  // = { [-2], [-1] } → [-2] を取得
__m128d vHi2  = _mm_loadu_pd(history);       // = { [0],  [1] } → [0] を取得

// 各ペアから最初の要素だけを抽出して256-bitに結合
// 方法A: _mm256_set_m128d / _mm256_insertf128_pd
__m256d vSamples = _mm256_insertf128_pd(
    _mm256_castpd128_pd256(vLow),  // low 128 = { [-6], [-5] }
    vHi,                           // high 128 = { [-2], [-1] }
    1
);
// vSamples = { [-6], [-5], [-2], [-1] } - not what we want!

// 方法B: 各ペアから必要な要素だけを取り出して個別にセット
// _mm256_set_pd だが、今回はコンパイラが単なるレジスタ転送に最適化可能
// （4回の独立ロードよりは効率が良いかもしれない）
__m256d vSamples = _mm256_set_pd(
    _mm_cvtsd_f64(_mm_load_sd(history)),       // [0]
    _mm_cvtsd_f64(_mm_load_sd(history - 2)),   // [-2]
    _mm_cvtsd_f64(_mm_load_sd(history - 4)),   // [-4]
    _mm_cvtsd_f64(_mm_load_sd(history - 6))    // [-6]
);
// = { [-6], [-4], [-2], [0] } ✓
```

ただし `_mm256_set_pd` + `_mm_load_sd` では4回の別々の load が発生する。より効率的な方法は以下。

**方法C**: 64bit-gather（`_mm256_i64gather_pd`）を使用

```cpp
// history をベースアドレスとして、オフセットで gather
// 欲しい要素: history[-6], history[-4], history[-2], history[0]
// オフセット（バイト）: -48, -32, -16, 0
const __m256i vOffsets = _mm256_set_epi64x(
    0LL,          // history[0] へのバイトオフセット
    -16LL,        // history[-2] へのバイトオフセット
    -32LL,        // history[-4] へのバイトオフセット
    -48LL         // history[-6] へのバイトオフセット
);
__m256d vSamples = _mm256_i64gather_pd(history, vOffsets, 8);
// gather_pd(base, offsets, scale=8) → base[offset/8]
// = { history[-6], history[-4], history[-2], history[0] } ✓
```

ただし gather 命令にはレイテンシペナルティがある（通常 ~5-8 cycle, スループット ~2-3要素/cycle）。

**方法D**（推奨）: 2つの128-bit load + shuffle による deinterleave

```cpp
// 連続8要素を2つの256-bit load で取得
__m256d vA = _mm256_loadu_pd(history - 6);  // = { [-6], [-5], [-4], [-3] }
__m256d vB = _mm256_loadu_pd(history - 2);  // = { [-2], [-1], [0],  [1] }

// 各128-bit lane 内で shuffle_pd を使用
// mask: lane0=00, lane1=00 → 各laneで a[0], b[0] を選択
__m256d vShuf = _mm256_shuffle_pd(vA, vB, 0b0000);
// lane0: { vA[0]=[-6], vB[0]=[-2] }
// lane1: { vA[2]=[-4], vB[2]=[0] }
// = { [-6], [-2], [-4], [0] }
```

惜しい！順序が `[-6], [-2], [-4], [0]` で係数の順序と合わない。
係数は `coeffs[r+0], coeffs[r+1], coeffs[r+2], coeffs[r+3]` なので、
サンプルは `history[0], history[-2], history[-4], history[-6]` の順が必要。

つまり permute で要素を入れ替える:

```cpp
// vShuf = { [-6], [-2], [-4], [0] }
// 欲しい: { [0], [-2], [-4], [-6] } → r=0,1,2,3 の順

// 方法1: _mm256_permute_pd でペア内スワップ + permute2f128 でレーンスワップ
// これは複雑すぎる...

// 方法2: 負のオフセットで gather（シンプルで確実）
// ただし gather は現状の set_pd よりは効率が良いはず

// 方法3（推奨）: load 位置を調整して shuffle 結果が正しい順序になるようにする
```

**方法D改**（推奨: レビュー修正版）: 調整済み load + shuffle

> **⚠️ 初版のバグ**: 初版設計書では `_mm_unpacklo_pd(vEvenLow, vEvenHigh)` で `{ [-6], [-4], [-2], [0] }` を生成していたが、これはFMAで係数 `{ c[0],c[1],c[2],c[3] }` と乗算した際に `c[0]*h[-6] + c[1]*h[-4] + c[2]*h[-2] + c[3]*h[0]` となり、正しい畳み込み順序 `c[0]*h[0] + c[1]*h[-2] + c[2]*h[-4] + c[3]*h[-6]` と逆転する致命的な論理エラー。レビューにより修正。

**正しい順序: `{ ptr[0], ptr[-2], ptr[-4], ptr[-6] }`** — 係数 `c[r]` が `history[base-convParity - r*2]` と乗算される順序に合わせる。

```cpp
// ptr は history のベース位置を指す（呼出側: history + (base - convParity)）
// 必要な要素: ptr[0], ptr[-2], ptr[-4], ptr[-6]
// FMAでの乗算: c[0]*ptr[0] + c[1]*ptr[-2] + c[2]*ptr[-4] + c[3]*ptr[-6]
// → ベクトルは { ptr[0], ptr[-2], ptr[-4], ptr[-6] } の順が必要

// 4回の128-bit load（L1キャッシュの帯域2 loads/cycleを活用）
__m128d v0 = _mm_loadu_pd(ptr - 6);    // = { ptr[-6], ptr[-5] }
__m128d v1 = _mm_loadu_pd(ptr - 4);    // = { ptr[-4], ptr[-3] }
__m128d v2 = _mm_loadu_pd(ptr - 2);    // = { ptr[-2], ptr[-1] }
__m128d v3 = _mm_loadu_pd(ptr);        // = { ptr[0],  ptr[1]  }

// _mm_unpacklo_pd(a,b): 各レジスタの low 要素を { a[0], b[0] } に配置
// v3 と v2 の low 要素 → { ptr[0], ptr[-2] }（low 128-bit）
// v1 と v0 の low 要素 → { ptr[-4], ptr[-6] }（high 128-bit）
__m128d vLow  = _mm_unpacklo_pd(v3, v2);  // = { ptr[0], ptr[-2] }
__m128d vHigh = _mm_unpacklo_pd(v1, v0);  // = { ptr[-4], ptr[-6] }

// 256-bit に結合
__m256d vSamples = _mm256_castpd128_pd256(vLow);
vSamples = _mm256_insertf128_pd(vSamples, vHigh, 1);
// = { ptr[0], ptr[-2], ptr[-4], ptr[-6] } ✓
```

FMA検証:

```
FMA(vSamples, vCoeffs, vAcc)
= c[0]*ptr[0] + c[1]*ptr[-2] + c[2]*ptr[-4] + c[3]*ptr[-6]
これは for (int r=0; r<4; ++r) acc += coeffs[r] * history[base-convParity - r*2] と一致 ✓
```

この方法の利点:

- 4回の128-bit load（`_mm_loadu_pd`）で構成 → 連続アドレス領域のためキャッシュ効率が高い
- 2回の `unpacklo` + 1回の `insertf128` でシャッフル → gather命令（~6-8 cycle）より低レイテンシ
- 合計: 4 load + 3 shuffle → 良好なスループット

**Step 2**: FMA + アキュムレータ unroll

`dotProductAvx2` と同様に4つのアキュムレータでunroll:

```cpp
__m256d acc0 = _mm256_setzero_pd();
__m256d acc1 = _mm256_setzero_pd();
__m256d acc2 = _mm256_setzero_pd();
__m256d acc3 = _mm256_setzero_pd();

int r = 0;
// 16要素（4バッチ×4要素）を1セットに unroll
const int simdEnd = (convCount / 16) * 16;
for (; r < simdEnd; r += 16)
{
    // r+0..3
    __m256d vS0 = loadStride2(history - (r << 1));
    __m256d vC0 = _mm256_loadu_pd(coeffs + r);
    acc0 = _mm256_fmadd_pd(vS0, vC0, acc0);

    // r+4..7
    __m256d vS1 = loadStride2(history - ((r + 4) << 1));
    __m256d vC1 = _mm256_loadu_pd(coeffs + r + 4);
    acc1 = _mm256_fmadd_pd(vS1, vC1, acc1);

    // r+8..11
    __m256d vS2 = loadStride2(history - ((r + 8) << 1));
    __m256d vC2 = _mm256_loadu_pd(coeffs + r + 8);
    acc2 = _mm256_fmadd_pd(vS2, vC2, acc2);

    // r+12..15
    __m256d vS3 = loadStride2(history - ((r + 12) << 1));
    __m256d vC3 = _mm256_loadu_pd(coeffs + r + 12);
    acc3 = _mm256_fmadd_pd(vS3, vC3, acc3);
}
// 4要素単位の残り
for (; r <= convCount - 4; r += 4)
{
    __m256d vS = loadStride2(history - (r << 1));
    __m256d vC = _mm256_loadu_pd(coeffs + r);
    acc0 = _mm256_fmadd_pd(vS, vC, acc0);
}
// 水平加算 + スカラー剰余
...
```

### マクロ化によるload操作の共通化

stride-2 load パターンをインライン関数として共通化:

> **修正**: v3/v2 → low, v1/v0 → high の順序で `_mm_unpacklo_pd` に渡すことで係数順序と一致させる（レビュー指摘）。

```cpp
#if defined(__AVX2__)
/// @brief Stride-2 で4要素を半帯域履歴バッファからロード
/// @param ptr  history のベースアドレス（呼出側: history + (base - convParity)）
/// @return     { ptr[0], ptr[-2], ptr[-4], ptr[-6] } の順
/// @note       係数 coeffs[r+0..r+3] との FMA で正しい畳み込み順序になる
static inline __m256d loadStride2(const double* ptr) noexcept
{
    __m128d v0 = _mm_loadu_pd(ptr - 6);    // { ptr[-6], ptr[-5] }
    __m128d v1 = _mm_loadu_pd(ptr - 4);    // { ptr[-4], ptr[-3] }
    __m128d v2 = _mm_loadu_pd(ptr - 2);    // { ptr[-2], ptr[-1] }
    __m128d v3 = _mm_loadu_pd(ptr);        // { ptr[0],  ptr[1]  }
    // low:  ptr[0], ptr[-2]  ← v3 と v2 の low element
    __m128d vLow  = _mm_unpacklo_pd(v3, v2);
    // high: ptr[-4], ptr[-6] ← v1 と v0 の low element
    __m128d vHigh = _mm_unpacklo_pd(v1, v0);
    return _mm256_insertf128_pd(
        _mm256_castpd128_pd256(vLow), vHigh, 1);
}
#endif
```

### 参考: 256-bit load + permute4x64 版（Review 3 改善案）

128-bit load ×4（ロードポート消費4）の代わりに、256-bit load ×2（ロードポート消費2） + `_mm256_shuffle_pd` + `_mm256_permute4x64_pd` を使用する最適化案。近代的Intel CPU（Sunny Cove/Golden Cove以降）では1サイクルあたり2回の256-bit loadが可能なため、ロードポート占有率を半減できる。

```cpp
#if defined(__AVX2__)
/// @brief 256-bit load 版 loadStride2（ロードポート消費2）
static inline __m256d loadStride2_256(const double* ptr) noexcept
{
    // 256-bit load ×2: 連続する64バイトを一気に回収
    __m256d vA = _mm256_loadu_pd(ptr - 6);  // { ptr[-6], ptr[-5], ptr[-4], ptr[-3] }
    __m256d vB = _mm256_loadu_pd(ptr - 2);  // { ptr[-2], ptr[-1], ptr[0],  ptr[1]  }

    // 各128-bit lane内で偶数要素を抽出: { ptr[-6], ptr[-2], ptr[-4], ptr[0] }
    __m256d vShuf = _mm256_shuffle_pd(vA, vB, 0b0000);

    // _mm256_permute4x64_pd で一発並び替え
    // vShuf = { [-6]@0, [-2]@1, [-4]@2, [0]@3 }
    // 欲しい: { [0], [-2], [-4], [-6] }
    // _MM_SHUFFLE(0, 2, 1, 3) → { src[3], src[1], src[2], src[0] }
    return _mm256_permute4x64_pd(vShuf, _MM_SHUFFLE(0, 2, 1, 3));
    // = { ptr[0], ptr[-2], ptr[-4], ptr[-6] } ✓
}
#endif
```

**比較**:

| 方式 | load命令 | ロードポート消費 | シャッフル命令 | 合計命令数 |
|------|---------|----------------|--------------|----------|
| 128-bit ×4（設計書案） | `_mm_loadu_pd` ×4 | **4** | `unpacklo`×2 + `insertf128` | 7 |
| 256-bit ×2（改善案） | `_mm256_loadu_pd` ×2 | **2** | `shuffle_pd` + `permute4x64` | **4** |

メモリ帯域負荷が高いオーディオエンジンにおいて、ロード命令の半減は明確な効果が見込める。ただし `_mm256_permute4x64_pd` のレイテンシ（3 cycle）が `_mm_unpacklo_pd`（1 cycle）より大きいため、convCountが小さい（< 16）場合は128-bit版の方が有利な可能性がある。実機プロファイルに基づき選択すること。

### 参考: 8重アンロール（FMAパイプライン完全隠蔽）

Intel CPUのFMA命令はレイテンシ4〜5 cycle、スループット0.5 cycle（＝1 cycleに2命令）である。パイプラインを完全に飽和させるには理論上 5×2 = 10 本、現実的には **8本のアキュムレータ（acc0〜acc7）による32要素単位のアンロール** が最も効率的。

```cpp
// 8重アンロール（32要素/iteration）
__m256d acc0 = _mm256_setzero_pd();
// ... acc1..acc7 同様 ...

int r = 0;
const int unrollEnd = (convCount / 32) * 32;
for (; r < unrollEnd; r += 32)
{
    acc0 = _mm256_fmadd_pd(loadStride2(history - (r << 1)),       _mm256_loadu_pd(coeffs + r),       acc0);
    acc1 = _mm256_fmadd_pd(loadStride2(history - ((r +  4) << 1)), _mm256_loadu_pd(coeffs + r +  4), acc1);
    acc2 = _mm256_fmadd_pd(loadStride2(history - ((r +  8) << 1)), _mm256_loadu_pd(coeffs + r +  8), acc2);
    acc3 = _mm256_fmadd_pd(loadStride2(history - ((r + 12) << 1)), _mm256_loadu_pd(coeffs + r + 12), acc3);
    acc4 = _mm256_fmadd_pd(loadStride2(history - ((r + 16) << 1)), _mm256_loadu_pd(coeffs + r + 16), acc4);
    acc5 = _mm256_fmadd_pd(loadStride2(history - ((r + 20) << 1)), _mm256_loadu_pd(coeffs + r + 20), acc5);
    acc6 = _mm256_fmadd_pd(loadStride2(history - ((r + 24) << 1)), _mm256_loadu_pd(coeffs + r + 24), acc6);
    acc7 = _mm256_fmadd_pd(loadStride2(history - ((r + 28) << 1)), _mm256_loadu_pd(coeffs + r + 28), acc7);
}
// 4要素単位の残り (16要素)
...
// アキュムレータ統合 (acc0+acc1, acc2+acc3, ...)
...
```

convCountが32以上のステージ（例えば tap数64相当）がある場合、4重→8重への拡張で数%の追加改善が見込める。

**アーキテクチャ世代による最適アンロール数の違い**:

| CPU世代 | アーキテクチャ | FMAレイテンシ | FMAスループット | 最適アンロール数 |
|---------|--------------|-------------|---------------|----------------|
| 第4〜5世代 | Haswell / Broadwell | **5 cycle** | 0.5 cycle (2/cycle) | 10 (5÷0.5) |
| **第6世代〜第10世代** | **Skylake〜Comet Lake** | **4 cycle** | 0.5 cycle (2/cycle) | **8** (4÷0.5) |
| 第11世代〜 | Tiger Lake〜以降 | 4 cycle | 0.5 cycle (2/cycle) | 8 |

開発環境が第6世代Core（Skylake）以降であることから、**8重アンロールはFMAパイプラインを100%飽和させる最適な選択**である。初回実装から8重アンロールを採用することを推奨する。Haswell/Broadwell世代でも8重アンロールは有効であり、10重と比較した差はごくわずかである。

### decimateStage への統合

P1-1（事前バリデーション）後のAVX2パスで `dotProductDecimateAvx2` を呼び出す:

```cpp
if (stage.convCount >= 8)  // stride-2版は8タップ以上で効果的
{
    usedAvxPath = true;
    // historyBase は base - convParity を指す
    const double* histBase = history + (base - stage.convParity);
    const double* coeffs = stage.convCoeffs.get();
    acc += dotProductDecimateAvx2(histBase, coeffs, stage.convCount);
}
else if (stage.convCount >= 4)
{
    // convCount < 8 の場合は簡易版（set_pd 使用、16× unroll なし）
    usedAvxPath = true;
    __m256d vAcc = _mm256_setzero_pd();
    int r = 0;
    const int simdEnd = (stage.convCount / 4) * 4;
    for (; r < simdEnd; r += 4)
    {
        __m256d vS = loadStride2(histBase - (r << 1));
        __m256d vC = _mm256_loadu_pd(coeffs + r);
        vAcc = _mm256_fmadd_pd(vS, vC, vAcc);
    }
    // 水平加算
    __m128d vLo = _mm256_castpd256_pd128(vAcc);
    __m128d vHi = _mm256_extractf128_pd(vAcc, 1);
    __m128d vSum = _mm_add_pd(vLo, vHi);
    vSum = _mm_hadd_pd(vSum, vSum);
    acc += _mm_cvtsd_f64(vSum);
}
```

convCount >= 8 の閾値を設定した理由:

- stride-2 load にはストア転送 + shuffle のオーバーヘッドがある
- 4要素（convCount=4）ではオーバーヘッドが支配的
- convCount >= 8 でアキュムレータ unroll の恩恵が上回る
- convCount=4〜7 は簡易AVX2パスまたはスカラー

---

## 統合設計: decimateStage 全体像

P1-1 + P1-2 適用後の `decimateStage` の完全なフロー:

```
decimateStage(stage, input, inputSamples, output, channel):
  history = stage.downHistory[channel]
  keep = stage.historyDownKeep
  capacity = stage.downHistorySize

  # [既存] サイレンスチェック（変更なし）
  inputSilent チェック → クリアして return

  # [既存] 履歴にコピー（変更なし）
  copy(history + keep, input, inputSamples)

  outSamples = inputSamples >> 1
  coeffs = stage.convCoeffs

  for n in 0..outSamples:
    base = keep + (n << 1)

    # ── P1-1: 事前バリデーション ──
    if base out of range:
      output[n] = 0; markCorruption; continue

    minTapIdx = base - convParity - ((convCount-1) << 1)
    maxTapIdx = base - convParity
    if minTapIdx < 0 || maxTapIdx >= capacity:
      output[n] = 0; markCorruption; continue

    acc = centerCoeff * history[base - centerTap]
    if isBadSample(acc):
      output[n] = 0; markCorruption; continue

    # ── P1-2: stride-2 dot product（AVX2） ──
    if convCount >= 8:
      acc += dotProductDecimateAvx2(
          history + base - convParity, coeffs, convCount)
    elif convCount >= 4:
      # 簡易AVX2（loadStride2 + FMA、16×unrollなし）
      ...
    else:
      # スカラー（バリデーション済みなのでチェックなし）
      for r in 0..convCount:
        acc += coeffs[r] * history[base - convParity - (r << 1)]

    # ── 事後処理 ──
    if isBadSample(acc):
      output[n] = 0.0; markCorruptionDetected()
    else:
      if fastAbs(acc) < kDenormThreshold: acc = 0.0
      output[n] = acc

  # 履歴シフト（変更なし）
  memmove(history, history + inputSamples, keep * sizeof(double))
```

### 改善点まとめ

| 項目 | 現状 | 改善後 |
|------|------|--------|
| `bad` フラグ | あり（AVX2放棄 + スカラー再計算） | **削除** |
| AVX2経路の中断 | `bad=true` で任意のタイミングで中断 | **中断なし** |
| バウンドチェック | ループ内で毎回実行 | **事前に1回** |
| サンプルロード | `_mm256_set_pd` + 個別ロード×4 | **128-bit load×4 + shuffle** |
| アキュムレータ unroll | なし | **4重 unroll** |
| スカラーフォールバック | centerCoeff から再計算 | **不要（削除）** |

---

## 性能試算

### P1-1 による改善

`decimateStage` の hot loop（L571）の self time: **0.762s**

内訳推定:

- convolution 本体: ~80%（0.610s）
- バウンドチェック + bad sample チェック: ~15%（0.114s）
- `bad` フラグ分岐 + 水平加算: ~5%（0.038s）

P1-1 で削減できるのは「bad フラグ分岐」と「スカラーフォールバックの二重計算」。
実際の corruption 発生頻度が低い（プロファイル上、corruptionなしで動作）と仮定すると、削減効果は主に:

- 分岐予測ミスの削減
- ソフトウェアパイプライニングの向上

→ 推定 **5-10%** 削減（0.038〜0.076s）

### P1-2 による改善

stride-2 load（4×128-bit load + shuffle）vs 現状（4×scalar load + set_pd）:

| 方式 | load 命令数 | shuffle 命令数 | 推定レイテンシ |
|------|-----------|-------------|--------------|
| 現状: set_pd + 個別load | 4 scalar + 1 set_pd | 0 | ~8 cycle |
| 改善: 128-bit load×4 + shuffle | 4 vector (128-bit) | 3 | ~5 cycle |
| 改善: gather | 1 gather | 0 | ~6-8 cycle |
| 改善: 128-bit load×2 + _mm256_set_pd(m128d) | 2 vector | 1 | ~3 cycle |

※ ただし上記は1バッチ（4タップ）あたりの比較。アキュムレータ4重unrollで16タップ/iterationになる。

convCount=40（典型的なhalfband tap数）の場合:

- 現状: 10 iteration × ~8 cycle = 80 cycle
- 改善後: 2.5 iteration × ~5 cycle = 12.5 cycle

→ 理論上のスピードアップ: **3〜6倍**

Advisor の hot loop（L571）self time 0.762s に対して:

- 現状のAVX2パス vs スカラーフォールバックの比率は不明
- `convCount >= 4` でAVX2を使用するため、大半のケースでAVX2パスが稼働していると想定
- P1-2 の効果は AVX2パス自体の高速化（3-6倍）+ スカラーフォールバックの完全撲滅
- 推定: **0.762s → 0.3〜0.5s**（35〜60%削減）

---

## リスク評価

| リスク | 影響 | 対策 |
|-------|------|------|
| 事前バリデーションで大きな範囲をチェックするオーバーヘッド | 低。min/max 計算は O(1) で1回のみ | 定数時間であることをアサーションで確認 |
| 事後 isBadSample での corruption 検出漏れ | 低。NaN/Infは事後チェックで捕捉。バウンド違反は事前チェックで捕捉 | 念のためスカラーフォールバックパスを残す（ASSERTで検証のみ） |
| 数値的オーバーフローの中間検出 | 低。convCountが小さい（最大数十）ためオーバーフローリスクは低い | `acc` の事後 isBadSample で捕捉 |
| `loadStride2` の係数順序ミス | ⛔ **致命的**（レビューで修正済み） | v3/v2→low, v1/v0→high の正しい並びで実装すること |
| Gather命令のレイテンシペナルティ | 中。Haswell世代ではgatherが低速 | 推奨案ではgather不使用、load+shuffle方式を採用 |
| `_mm_loadu_pd` のアライメント違反 | 低。履歴バッファはアライン保証外だが `loadu` 使用 | 既存コードと同一 |
| memory ordering / false sharing | なし。履歴バッファはスレッドローカル | — |
| convCount < 4 のケース | なし。条件分岐でカバー | convCount >= 8 で stride-2 dot product、4〜7 は簡易AVX2、<4 はスカラー |
| コンパイラ互換性 | 低。`_mm256_insertf128_pd`, `_mm_unpacklo_pd` はMSVC/Clang/GCC共通 | static_assert で **AVX2** を確認 |

### 実装上の注意点（Review 2より）

1. **`convCount` が 0 でないことの確認**: 事前バリデーションで `minTapIdx` / `maxTapIdx` を計算する前に `stage.convCount > 0` を暗黙的に前提している。現在のコードでは convCount は常に正だが、念のため安全側に倒すなら早期リターンを入れる。
2. **事後 `isBadSample(acc)` で `markCorruptionDetected()` を確実に呼ぶ**: 設計書のコードスケッチには記載があるが、実装時に漏れないよう注意。
3. **`fastAbs(acc) < kDenormThreshold` によるゼロ化**: 従来通り維持。`isBadSample` が先に true ならゼロ化は不要（出力が 0.0 に設定されるため）。
4. **サイレンス最適化パスの維持**: decimateStage 上部の「入力が無音なら履歴をクリアして早期リターン」のパスは変更の影響を受けない。既存のまま維持すること。
5. **`convCount` の値**: `(taps - convParity + 1) / 2` で算出される。通常は偶数になるが4の倍数とは限らない。スカラー剰余ループが存在するため任意の値で安全に動作する。

### 実装上の注意点（Review 3より）

1. **`__assume` のスコープ**: 呼出側（`decimateStage`）で `__assume(stage.convCount >= 8)` と記述しても、`dotProductDecimateAvx2` が別翻訳単位（またはインライン展開されない関数）である場合、その内部最適化に伝搬しない。確実を期すため、`dotProductDecimateAvx2` 関数の先頭内部にも `__assume(convCount >= 8 && convCount % 4 == 0)` を直接記述すること。関数を `inline` にするか、`__forceinline` を指定するのも有効。

2. **`/fp:fast` と `isBadSample` の互換性**: プロジェクトのビルドオプションで `/fp:fast` が有効な場合、MSVCは「NaNは発生しない」と仮定して `x != x` や `std::isnan(x)` によるNaN判定を最適化で消去する危険性がある。現状の `isBadSample` は `std::bit_cast` を用いたビットマスク判定（`exp == 0x7FF`）のため `/fp:fast` の影響を受けない。`isBadSampleV` も `_mm256_cmp_pd` intrinsic を使用しており、こちらも `/fp:fast` の影響を受けない（intrinsics はコンパイラの浮動小数点モデルをバイパスする）。変更時もこのビット演算方式を維持すること。

### 推奨実装ステップ

1. `loadStride2` インライン関数を `CustomInputOversampler.cpp` の先頭付近（`#if defined(__AVX2__)` 内）に追加。
2. `dotProductDecimateAvx2` 関数を追加（同じく条件コンパイル）。
3. `decimateStage` を設計書のフローに従って書き換え（事前バリデーション → AVX2分岐 → 事後処理）。
4. プロファイル（Intel Advisor / perf）で効果を計測。特に `convCount` の閾値（8）は実機でチューニングしてもよい。
5. 必要に応じて `interpolateStage` にも同様の最適化を検討するが、現状の `dotProductAvx2` が連続アクセスで十分なため優先度は低い。

### コードサイズ増加

| 追加要素 | 推定行数 |
|---------|---------|
| `loadStride2` inline関数 | ~15行 |
| `dotProductDecimateAvx2` | ~60行 |
| 事前バリデーション（P1-1） | ~15行 |
| decimateStage の再構成 | ~30行（削減も含む） |
| **正味増加** | **~60行** |

---

## 付録: コードスケッチ（レビュー反映版）

> **⚠️ 重要**: 初版の `loadStride2` は係数順序逆転のバグを含んでいました。以下のコードはレビューにより修正済みです。

### `loadStride2` inline関数

```cpp
#if defined(__AVX2__)
/// @brief Stride-2 で4要素を半帯域履歴バッファからロード
/// @param ptr  history のベースアドレス（呼出側: history + (base - convParity)）
/// @return     { ptr[0], ptr[-2], ptr[-4], ptr[-6] } の順
/// @note       係数 coeffs[r+0..r+3] との FMA で正しい畳み込み順序になる
///             FMA = c[0]*ptr[0] + c[1]*ptr[-2] + c[2]*ptr[-4] + c[3]*ptr[-6]
///             これは for(r=0..3) acc += coeffs[r] * h[base-convParity - r*2] と一致
static inline __m256d loadStride2(const double* ptr) noexcept
{
    __m128d v0 = _mm_loadu_pd(ptr - 6);    // { ptr[-6], ptr[-5] }
    __m128d v1 = _mm_loadu_pd(ptr - 4);    // { ptr[-4], ptr[-3] }
    __m128d v2 = _mm_loadu_pd(ptr - 2);    // { ptr[-2], ptr[-1] }
    __m128d v3 = _mm_loadu_pd(ptr);        // { ptr[0],  ptr[1]  }
    // low:  ptr[0], ptr[-2]  ← v3 と v2 の low element
    __m128d vLow  = _mm_unpacklo_pd(v3, v2);
    // high: ptr[-4], ptr[-6] ← v1 と v0 の low element
    __m128d vHigh = _mm_unpacklo_pd(v1, v0);
    return _mm256_insertf128_pd(
        _mm256_castpd128_pd256(vLow), vHigh, 1);
    // = { ptr[0], ptr[-2], ptr[-4], ptr[-6] }
}
#endif
```

### `dotProductDecimateAvx2` 関数（レビュー修正版）

> **修正点**: `loadStride2` の戻り値の順序が `{ ptr[0], ptr[-2], ptr[-4], ptr[-6] }` に変わったことに伴い、呼出側の `history` ポインタと `r` の関係を確認。
> `loadStride2(history - (r << 1))` の history は `history + (base - convParity)` = ptr を指す。
> `loadStride2(ptr - (r << 1))` = `loadStride2(ptr - r*2)` = `{ ptr[-r*2], ptr[-r*2-2], ptr[-r*2-4], ptr[-r*2-6] }`。
> 係数 `coeffs[r+0..r+3]` とのFMAで `c[r]*ptr[-r*2] + c[r+1]*ptr[-r*2-2] + c[r+2]*ptr[-r*2-4] + c[r+3]*ptr[-r*2-6]` となり正しい。

```cpp
#if defined(__AVX2__) && defined(__FMA__)
double CustomInputOversampler::dotProductDecimateAvx2(
    const double* __restrict history,  // = history + (base - convParity)
    const double* __restrict coeffs,   // = stage.convCoeffs
    int convCount) noexcept
{
    __m256d acc0 = _mm256_setzero_pd();
    __m256d acc1 = _mm256_setzero_pd();
    __m256d acc2 = _mm256_setzero_pd();
    __m256d acc3 = _mm256_setzero_pd();

    int r = 0;
    // 16要素（4バッチ）unroll
    const int unrollEnd = (convCount / 16) * 16;
    for (; r < unrollEnd; r += 16)
    {
        // バッチ0: r+0..3
        __m256d vS0 = loadStride2(history - (r << 1));
        __m256d vC0 = _mm256_loadu_pd(coeffs + r);
        acc0 = _mm256_fmadd_pd(vS0, vC0, acc0);

        // バッチ1: r+4..7
        __m256d vS1 = loadStride2(history - ((r + 4) << 1));
        __m256d vC1 = _mm256_loadu_pd(coeffs + r + 4);
        acc1 = _mm256_fmadd_pd(vS1, vC1, acc1);

        // バッチ2: r+8..11
        __m256d vS2 = loadStride2(history - ((r + 8) << 1));
        __m256d vC2 = _mm256_loadu_pd(coeffs + r + 8);
        acc2 = _mm256_fmadd_pd(vS2, vC2, acc2);

        // バッチ3: r+12..15
        __m256d vS3 = loadStride2(history - ((r + 12) << 1));
        __m256d vC3 = _mm256_loadu_pd(coeffs + r + 12);
        acc3 = _mm256_fmadd_pd(vS3, vC3, acc3);
    }

    // 4要素単位の残り
    const int simdEnd = (convCount / 4) * 4;
    for (; r < simdEnd; r += 4)
    {
        __m256d vS = loadStride2(history - (r << 1));
        __m256d vC = _mm256_loadu_pd(coeffs + r);
        acc0 = _mm256_fmadd_pd(vS, vC, acc0);
    }

    // アキュムレータ統合
    acc0 = _mm256_add_pd(acc0, acc1);
    acc2 = _mm256_add_pd(acc2, acc3);
    acc0 = _mm256_add_pd(acc0, acc2);

    // 水平加算（vextractf128 + hadd）
    __m128d vLo = _mm256_castpd256_pd128(acc0);
    __m128d vHi = _mm256_extractf128_pd(acc0, 1);
    __m128d vSum = _mm_add_pd(vLo, vHi);
    vSum = _mm_hadd_pd(vSum, vSum);
    double result = _mm_cvtsd_f64(vSum);

    // スカラー剰余（convCount が4の倍数でない場合も安全）
    for (; r < convCount; ++r)
        result += coeffs[r] * history[-(r << 1)];

    return result;
}
#endif
```

### decimateStage の新しいAVX2分岐（レビュー反映版）

> **注意**: `convCount` の値は `(taps - convParity + 1)/2` で算出される。通常は偶数になるが保証はされていない。スカラー剰余ループが存在するため、任意の convCount で安全に動作する。`__assume` はコンパイラへの最適化ヒントであり、4の倍数でなくても正しく動作する。

```cpp
#if defined(__AVX2__) && defined(__FMA__)
        // P1-2: stride-2 dot product を使用（convCount >= 8 で効果的）
        if (stage.convCount >= 8)
        {
            usedAvxPath = true;
            __assume(stage.convCount >= 8);
            acc += dotProductDecimateAvx2(
                history + (base - stage.convParity),
                coeffs,
                stage.convCount);
        }
        else if (stage.convCount >= 4)
        {
            // convCount 4〜7: 簡易AVX2（unrollなしのloadStride2）
            usedAvxPath = true;
            __m256d vAcc = _mm256_setzero_pd();
            int r = 0;
            const int simdEnd = (stage.convCount / 4) * 4;
            for (; r < simdEnd; r += 4)
            {
                __m256d vS = loadStride2(
                    history + (base - stage.convParity) - (r << 1));
                __m256d vC = _mm256_loadu_pd(coeffs + r);
                vAcc = _mm256_fmadd_pd(vS, vC, vAcc);
            }
            // 水平加算（vextractf128 + hadd）
            __m128d vLo = _mm256_castpd256_pd128(vAcc);
            __m128d vHi = _mm256_extractf128_pd(vAcc, 1);
            __m128d vSum = _mm_add_pd(vLo, vHi);
            vSum = _mm_hadd_pd(vSum, vSum);
            acc += _mm_cvtsd_f64(vSum);
            // スカラー剰余
            for (; r < stage.convCount; ++r)
                acc += coeffs[r] * history[base - stage.convParity - (r << 1)];
        }
#endif
```
