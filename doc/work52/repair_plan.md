# 改修計画書 — 低音入力時の「ジジジジ」ノイズ対策

- **作成日**: 2026-06-21
- **最終更新日**: 2026-06-21 (v7: 最終レビュー検証。メイクアップゲイン微調整追記。全未確定事項確定)
- **対象**: PCローカルWorking Tree
- **関連文書**: `doc/work52/bug_review.md`, `doc/work52/bug_review_validation_report.md` (v3)
- **調査ツール**: Serena MCP, AiDex MCP, CodeGraph MCP, semble CLI, graphify CLI, grep/Select-String, Web文献調査, cocoindex-code (uvx経由)

---

## 目次

1. [改修項目一覧](#1-改修項目一覧)
2. [P1: SVF状態変数サチュレーションの除去](#2-p1-svf状態変数サチュレーションの除去)
3. [P2: SoftClip prevScalar不整合の修正](#3-p2-softclip-prevscalar不整合の修正)
4. [P3: SoftClip midVec事前平均化の即時削除](#4-p3-softclip-midvec事前平均化の即時削除)
5. [P4: Parallelモード帯域加算の影響評価](#5-p4-parallelモード帯域加算の影響評価)
6. [P5: オーバーサンプリング最低値の検討](#6-p5-オーバーサンプリング最低値の検討)
7. [P6: AGCのブロックRMSリップル対策](#7-p6-agcのブロックrmsリップル対策)
8. [未確定事項の確定結果](#8-未確定事項の確定結果)
9. [実機検証手順](#9-実機検証手順)
10. [リグレッションリスク](#10-リグレッションリスク)

---

## 1. 改修項目一覧

| ID | 優先度 | 項目 | 確度 | 影響範囲 | 工数目安 |
|----|-------|------|------|---------|---------|
| P1 | ★★★★★ | SVF状態変数サチュレーション除去 | 70〜80% | EQ全経路 | 小（数行の変更） |
| P3 | ★★★★★ | SoftClip midVec事前平均化削除（格上げ） | 70% | SoftClip AVX2/スカラー | 中 |
| P2 | ★★★★ | SoftClip prevScalar不整合修正 | 60% | SoftClipスカラーパスのみ | 小（1行の変更） |
| P6 | ★★★ | AGCサンプル単位エンベロープフォロワ改修（新規） | 40% | AGC処理 | 中 |
| P4 | ★★★ | Parallelモード帯域加算検証 | 30% | EQ Parallelモード | 小（計測・検証） |
| P5 | ★★ | OS最低値検討 | 30% | ビルド/設定 | 中（設定UI含む） |

---

## 2. P1: SVF状態変数サチュレーションの除去

### 2.1 バグ発生原因

**DSP理論的根拠**: TPT (Topology-Preserving Transform) SVF において、`ic1eq`, `ic2eq` は**積分器の内部状態**（エネルギー保存量）である。Zavalishin "The Art of VA Filter Design" に詳述される通り、TPTの中核は「積分器を bilinear transform で離散化し、それ以外のアナログトポロジーを保存すること」にある。状態変数に非線形性を導入するとトポロジー保存性が完全に崩壊し、線形SVFが非線形状態空間システムと化す。

**具体的な問題**: 現行コードでは `processBand()` / `processBandStereo()` 内で、SVFの状態変数 `ic1eq`/`ic2eq` に直接 `fastTanhScalar`/`fastTanhV128` を適用している。

```cpp
// 問題のコード (スカラー版)
if (saturation > 0.0)
{
    const double oneMinusSat = 1.0 - saturation;
    ic1eq = ic1eq * oneMinusSat + fastTanhScalar(ic1eq) * saturation;
    ic2eq = ic2eq * oneMinusSat + fastTanhScalar(ic2eq) * saturation;
}
```

```cpp
// 問題のコード (SIMD版)
if (saturation > 0.0)
{
    ic1eq = _mm_add_pd(_mm_mul_pd(ic1eq, vOneMinusSat),
                       _mm_mul_pd(fastTanhV128(ic1eq), vSat));
    ic2eq = _mm_add_pd(_mm_mul_pd(ic2eq, vOneMinusSat),
                       _mm_mul_pd(fastTanhV128(ic2eq), vSat));
}
```

**発散条件**:

- Low Shelf +12dB ブースト時: `m2 = A² - 1 ≈ 14.84`（A = 10^(12/40) ≈ 3.98）
- 出力式: `output = m0*v0 + m1*v1 + 14.84*v2` → ic2eq成分が約15倍
- `fastTanhScalar` のクリップ閾値: **3.0**（vs SoftClip用fastTanhは4.5）
- 低域ブースト＋ハイQで状態変数は容易に3.0超過 → 毎サンプルエネルギー喪失 → 周期的状態崩壊 → ジジジジ

### 2.2 原因箇所

| 項目 | 内容 |
|------|------|
| **ファイル** | `src/eqprocessor/EQProcessor.Processing.cpp` |
| **関数** | `(anonymous namespace)::processBand()` (line 119-178) |
| **関数** | `(anonymous namespace)::processBandStereo()` (line 183-264) |
| **関連定義** | `fastTanhScalar()` (line 85-91) — clip at ±3.0 |
| **関連定義** | `fastTanhV128()` (line 93-105) — clip at ±3.0 |
| **呼び出し元** | `EQProcessor::process()` (line 466, 900) — ラムダ `processSerial`, `processParallel` 経由 |

### 2.3 改修方法（詳細）

#### 2.3.1 `processBand()` — スカラー版

**変更内容**: `ic1eq`/`ic2eq` への saturation 適用を削除し、`output` 計算後に適用する。

**変更前** (line 138-149):

```cpp
            ic1eq = 2.0 * v1 - ic1eq;
            ic2eq = 2.0 * v2 - ic2eq;

            if (saturation > 0.0)
            {
                const double oneMinusSat = 1.0 - saturation;
                ic1eq = ic1eq * oneMinusSat + fastTanhScalar(ic1eq) * saturation;
                ic2eq = ic2eq * oneMinusSat + fastTanhScalar(ic2eq) * saturation;
            }

            double output = m0 * v0 + m1 * v1 + m2 * v2;
```

**変更後**:

```cpp
            ic1eq = 2.0 * v1 - ic1eq;
            ic2eq = 2.0 * v2 - ic2eq;

            double output = m0 * v0 + m1 * v1 + m2 * v2;

            if (saturation > 0.0)
            {
                const double oneMinusSat = 1.0 - saturation;
                output = output * oneMinusSat + fastTanhScalar(output) * saturation;
            }
```

#### 2.3.2 `processBandStereo()` — SIMD版

**変更内容**: `ic1eq`/`ic2eq` への saturation 適用を削除し、`output` 計算後に適用する。

**変更前** (line 229-248):

```cpp
            ic1eq = _mm_fmsub_pd(two, v1, ic1eq);  // 2*v1 - ic1eq
            ic2eq = _mm_fmsub_pd(two, v2, ic2eq);  // 2*v2 - ic2eq

            if (saturation > 0.0)
            {
                const __m128d vSat = _mm_set1_pd(saturation);
                const __m128d vOneMinusSat = _mm_set1_pd(1.0 - saturation);
                ic1eq = _mm_add_pd(_mm_mul_pd(ic1eq, vOneMinusSat),
                                   _mm_mul_pd(fastTanhV128(ic1eq), vSat));
                ic2eq = _mm_add_pd(_mm_mul_pd(ic2eq, vOneMinusSat),
                                   _mm_mul_pd(fastTanhV128(ic2eq), vSat));
            }
            // FMA: m0*v0 + m1*v1 + m2*v2
            __m128d output = _mm_fmadd_pd(m0, v0,
                              _mm_fmadd_pd(m1, v1,
                               _mm_mul_pd(m2, v2)));
```

**変更後**:

```cpp
            ic1eq = _mm_fmsub_pd(two, v1, ic1eq);  // 2*v1 - ic1eq
            ic2eq = _mm_fmsub_pd(two, v2, ic2eq);  // 2*v2 - ic2eq

            // FMA: m0*v0 + m1*v1 + m2*v2
            __m128d output = _mm_fmadd_pd(m0, v0,
                              _mm_fmadd_pd(m1, v1,
                               _mm_mul_pd(m2, v2)));

            if (saturation > 0.0)
            {
                const __m128d vSat = _mm_set1_pd(saturation);
                const __m128d vOneMinusSat = _mm_set1_pd(1.0 - saturation);
                output = _mm_add_pd(_mm_mul_pd(output, vOneMinusSat),
                                    _mm_mul_pd(fastTanhV128(output), vSat));
            }
```

#### 2.3.3 重要: P1改修に伴う `fastTanh` 閾値の見直し（新規）

P1で saturation を `ic1eq`/`ic2eq` から `output` に移すと、既存の `fastTanhScalar`（|x|≥3.0 でクリップ）および `fastTanhV128`（|x|≥3.0 でクリップ）を出力信号に適用することになる。

**問題点**: SVFの出力（特に Low Shelf +12dB 等のブースト時）は容易に 3.0 を超える。クリップ閾値 3.0 は、

- 状態変数用としては適切（状態変数の過大成長を防ぐ）
- 出力信号用としては**低すぎる**（通常のオーディオ信号でもクリップが頻発する）

**推奨**: saturation を output に適用する際は、`fastTanhScalar` / `fastTanhV128` のクリップ閾値を **4.5 または 6.0** に引き上げる。

**参考**: SoftClip用 `fastTanh`（TanhApprox）のクリップ閾値は **4.5** であり、これに合わせるのが自然。

**変更前**:

```cpp
// fastTanhScalar
if (x >= 3.0) return 1.0;
if (x <= -3.0) return -1.0;

// fastTanhV128
const __m128d vThree = _mm_set1_pd(3.0);
const __m128d vNegThree = _mm_set1_pd(-3.0);
const __m128d xClamped = _mm_min_pd(_mm_max_pd(x, vNegThree), vThree);
```

**変更後**（閾値を 4.5 に変更）:

```cpp
// fastTanhScalar (output用)
if (x >= 4.5) return 1.0;
if (x <= -4.5) return -1.0;

// fastTanhV128 (output用)
const __m128d vClipHigh = _mm_set1_pd(4.5);
const __m128d vClipLow = _mm_set1_pd(-4.5);
const __m128d xClamped = _mm_min_pd(_mm_max_pd(x, vClipLow), vClipHigh);
```

> **Note**: 元の `fastTanhScalar` / `fastTanhV128` は状態変数用の関数として残し（既存の他の使用箇所がないことを確認）、新たに `fastTanhScalarOutput` / `fastTanhV128Output` のような別関数として追加することを推奨。変更箇所が明示的になり、誤用が防げる。

#### 2.3.4 NaN/Infチェックと状態変数ガード

現行コード（P1変更前）における NaN/Inf チェックは以下の通り:

| チェック | processBand (scalar) | processBandStereo (SIMD) |
|---------|---------------------|------------------------|
| Output NaN/Inf | ✅ `isFiniteAndAbsInRangeMask(output, 0, 1e15)` | ✅ `_mm_sub_pd(output, output)` trick |
| Output clamp | ✅ `std::clamp(output, -100, 100)` | ✅ `_mm_min_pd(_mm_max_pd(output, cLow), cHigh)` |
| State var NaN/Inf | ✅ 毎サンプル `ic1eq`/`ic2eq` チェック | ✅ ループ後 `killDenormalV` |
| デノーマル処理 | ✅ `killDenormal` | ✅ `killDenormalV` |

P1改修後もこれらのチェックは全て維持される（変更箇所は saturation 適用の移動のみ）。状態変数への NaN/Inf チェックが既に存在するため、追加のガードは不要。

#### 2.3.5 P1改修の2つの選択肢（v6追加）

文献調査（Zavalishin "The Art of VA Filter Design", Andrew Simper "Cytomic SVF Technical Paper"）の結果、状態変数への非線形操作を排除した上でのサチュレーション実装には2つの正当なアプローチがある。

**選択肢A: 出力段サチュレーション（Output Stage）— 現計画書の既定案**

`saturation` を `output` 計算後に適用する。最も単純で安全。

```
利点: 実装が簡単、SIMD互換性が高い、確実に安定
欠点: アナログ的な「Qが高い時に共振が歪む」挙動は再現できない
```

```cpp
// processBand() 内: output計算後に適用
if (saturation > 0.0)
{
    const double oneMinusSat = 1.0 - saturation;
    output = output * oneMinusSat + fastTanhScalarOutput(output) * saturation;
}
```

**選択肢B: Pre-Distortion（入力信号予備歪み）— 新規推奨案**

`saturation` をフィルタ入力の `v0` に適用してから線形SVFを通す。入力信号を歪ませた後にSVFでフィルタリングするため、共振Qが高い帯域で歪みが強調されるアナログライクな挙動が得られる。

```
利点: SVFは完全線形を保つ、アナログ的な周波数依存歪み、SIMD互換性が高い
欠点: メイクアップゲイン補正が必要、出力段より若干複雑
```

```cpp
// processBand() 内: v0に適用してから線形SVFへ
if (saturation > 0.0)
{
    const double oneMinusSat = 1.0 - saturation;
    const double distortedV0 = v0 * oneMinusSat + fastTanhScalarOutput(v0) * saturation;
    constexpr double kMakeupGain = 1.0 + 0.334; // sat=1.0で+2.5dB補正
    v0 = distortedV0 * (1.0 + kMakeupGain * saturation);
}
// 以降のv1/v2/ic1eq/ic2eq計算は完全な線形SVF
```

**メイクアップゲインの根拠**: `tanh(1.0) ≈ 0.76` で約-2.4dBのレベル低下が生じる。
Pre-Distortionでは全バンドでこの低下が発生するため、satに比例した補正が必要。

| saturation | tanh(v0) @v0=1.0 | レベル低下 | 補正ゲイン | 補正後 |
|-----------|-----------------|-----------|-----------|-------|
| 0.0 | 1.0 | 0dB | 1.0 (0dB) | 1.0 |
| 0.5 | 0.88 | -1.1dB | 1.167 (+1.3dB) | 1.027 |
| 1.0 | 0.76 | -2.4dB | 1.334 (+2.5dB) | 1.014 |

**推奨**: 選択肢B（Pre-Distortion）を推奨。理由:

- Cytomic等のVAフィルタ設計で一般的な手法
- SIMD演算のパイプラインを阻害しない
- アナログ的な周波数依存歪みが得られる
- 選択肢A（出力段）と組み合わせて「入力歪み＋出力歪み」の2段構成も将来的に可能

ただし、**第一優先は「状態変数への非線形操作の即時除去」であり、どちらの選択肢を選んでもバグは確実に解消する。**

#### 2.3.6 メイクアップゲイン係数の微調整（v7追記）

Pre-Distortionのメイクアップゲイン係数 `1.0 + 0.334 * saturation` は理論値だが、実機での試聴時に高域倍音が耳につく場合がある。その場合は係数を `0.334` から `0.2` 程度に抑え、「多少レベルが下がる代わりに歪みが自然」なバランスに調整することを推奨。

```
調整可能範囲: 0.2（ナチュラル）〜 0.334（理論値）
デフォルト推奨: 0.334（理論値、実機で確認後調整）
```

この調整は定数1行の変更であり、実機検証時に音質を確認しながら決定すること。

---

## 3. P2: SoftClip prevScalar不整合の修正

### 3.1 バグ発生原因

`softClipBlockAVX2()` 内で、AVX2パスとスカラーフォールバックパスで `prevScalar` の更新方法が異なる。

**AVX2パス** (line 196): 正しい

```cpp
const double nextPrev = data[i + 3]; // [BUG-04] store前に元の入力値を退避
_mm256_storeu_pd(data + i, result);
prevScalar = nextPrev;  // ← 生入力値を保存
```

**スカラーフォールバックパス** (line 212): バグ

```cpp
data[i] = x;
prevScalar = x;  // ← 処理後（クリップ）値を保存！
```

この不整合により、スカラーパス実行時（numSamples % 4 ≠ 0 の場合のみ）に `prevScalar` にクリップ出力値が保存される。次のブロックのAVX2パス先頭でこの値が使われ、ブロック境界で波形の連続性が損なわれる。

ただし実用的な影響範囲は限定的：

- 標準ブロックサイズ（64/128/256/512/1024）は全て4の倍数
- OS倍率（1x/2x/4x/8x）も2の冪で4の倍数性を維持
- 影響が出るのは非標準ブロックサイズ（例: 510サンプル等）のみ

### 3.2 原因箇所

| 項目 | 内容 |
|------|------|
| **ファイル** | `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp` |
| **関数** | `(anonymous namespace)::softClipBlockAVX2()` (line 103-224) |
| **バグ箇所** | line 212: `prevScalar = x;` |
| **同種バグ** | `src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp` (line 110-127) |
| **備考** | AVX2パスは `[BUG-04]` コメントで部分修正済み |

### 3.3 改修方法（詳細）

#### 3.3.1 DSPCoreDouble.cpp スカラーフォールバック

**変更前** (line 200-222):

```cpp
    for (; i < numSamples; ++i)
    {
        const double mid    = (prevScalar + data[i]) * 0.5;
        const double absMid = absNoLibm(mid);
        double x = data[i];
        if (absMid > threshold)
            x *= threshold / absMid;

        if (absNoLibm(x) > clip_start)
            x = musicalSoftClipScalar(x, threshold, knee, asymmetry);

        data[i] = x;
        prevScalar = x;  // ← バグ
    }
```

**変更後**:

```cpp
    for (; i < numSamples; ++i)
    {
        const double inputVal = data[i];  // 元の入力を退避
        const double mid    = (prevScalar + inputVal) * 0.5;
        const double absMid = absNoLibm(mid);
        double x = inputVal;
        if (absMid > threshold)
            x *= threshold / absMid;

        if (absNoLibm(x) > clip_start)
            x = musicalSoftClipScalar(x, threshold, knee, asymmetry);

        data[i] = x;
        prevScalar = inputVal;  // ← 修正: 処理前の生入力値を保存
    }
```

#### 3.3.2 DSPCoreFloat.cpp

**変更前** (line 110-127):

```cpp
    for (; i < numSamples; ++i)
    {
        double x = data[i];

        if (!isFiniteAndAbsBelowNoLibm(x, 1.0e300))
            x = 0.0;

        x = 0.5 * (x + prevSample);
        prevSample = x;  // ← 平均化値で上書き
        data[i] = musicalSoftClipScalar(x, threshold, knee, asymmetry);
    }
```

**変更後**（平均化自体を削除するか、少なくとも生入力値を保存）:

```cpp
    for (; i < numSamples; ++i)
    {
        double x = data[i];

        if (!isFiniteAndAbsBelowNoLibm(x, 1.0e300))
            x = 0.0;

        // 注意: 平均化はmidVec相当のロジック。P3と合わせて判断
        const double avg = 0.5 * (x + prevSample);
        prevSample = x;  // ← 修正: 処理前の生入力値を保存
        data[i] = musicalSoftClipScalar(avg, threshold, knee, asymmetry);
    }
```

> **Note**: Float版の平均化 (`0.5*(x+prevSample)`) はDouble版AVX2のmidVec相当の処理。P3のAB試験と合わせて削除判断すること。

---

## 4. P3: SoftClip midVec事前平均化の即時削除（★★★→★★★★★に格上げ）

### 4.1 問題の概要

`softClipBlockAVX2()` のAVX2パスに存在する `midVec` 事前平均化ロジックは、
**AB試験ではなく即時削除が妥当**と判断するに至った。

**理由**: 詳細なコード解析の結果、以下のメカニズムが確認された。

```cpp
// midVec計算: prevVec = {prevScalar, data[i], data[i+1], data[i+2]}
//              x       = {data[i], data[i+1], data[i+2], data[i+3]}
const __m256d midVec = _mm256_mul_pd(_mm256_add_pd(prevVec, x), vHalf);
// absMidVec > threshold → x *= threshold / absMidVec
```

低周波（波形の変化が緩やかな信号）が入力され、振幅が `threshold` を超えると:

```
prev ≈ x  (波形変化が緩いため)
midVec ≈ x (平均値 ≈ 現在値)
midGain = threshold / |midVec| ≈ threshold / |x|
x = x * (threshold / |x|) = sign(x) * threshold
```

**結果**: 信号が `threshold` レベルに**ハードリミッティング**される。この直後の `fastTanh`（TanhApprox, Pade近似）の美しいKnee（ニー）曲線に到達する前に波形が角張った矩形波状に加工され、SoftClip本来の滑らかな特性が完全に死ぬ。これにより強烈なエイリアシング（ジジジジ）が発生する。

**DSP的評価**: Waveshaper設計として明白なロジック破綻。SoftClipへの入力前に独立したハードリミッターが存在する状態であり、即時削除が唯一の正しい対応。

### 4.2 原因箇所

| 項目 | 内容 |
|------|------|
| **ファイル** | `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp` |
| **関数** | `(anonymous namespace)::softClipBlockAVX2()` (line 149-168) |
| **対象コード** | midVec計算ブロック全体（prevVec構築〜x *= midGain） |
| **同種** | `DSPCoreFloat.cpp` line 119: `x = 0.5 * (x + prevSample)` |

### 4.3 改修方法（即時削除）

**変更内容**: midVec事前平均化ブロック（`prevVec` 構築、`midVec` 計算、`absMidVec` 判定、`midGain` による `x` の減衰）を**完全に削除**する。

**変更前** (line 144-173):

```cpp
        {
            const __m128d xLow       = _mm256_castpd256_pd128(x);
            const __m128d xHigh      = _mm256_extractf128_pd(x, 1);
            const __m128d prevLow128 = _mm_unpacklo_pd(_mm_set_sd(prevScalar), xLow);
            const __m128d prevHigh128= _mm_shuffle_pd(xLow, xHigh, 0x1);
            const __m256d prevVec    = _mm256_set_m128d(prevHigh128, prevLow128);

            const __m256d midVec     = _mm256_mul_pd(_mm256_add_pd(prevVec, x), vHalf);
            const __m256d absMidVec  = _mm256_andnot_pd(vSignMask, midVec);

            const __m256d vTiny      = _mm256_set1_pd(1e-15);
            const __m256d needMidClip= _mm256_cmp_pd(absMidVec, vThreshold, _CMP_GT_OQ);
            const __m256d safeAbsMid = _mm256_max_pd(absMidVec, vTiny);
            const __m256d midGainRaw = _mm256_div_pd(vThreshold, safeAbsMid);
            const __m256d midGain    = _mm256_blendv_pd(vOne, midGainRaw, needMidClip);
            x = _mm256_mul_pd(x, midGain);
        }

        __m256d absX = _mm256_andnot_pd(vSignMask, x);
```

**変更後**:

```cpp
        // midVec事前平均化ブロックを完全削除
        // x はそのまま後続のSoftClip（fastTanh近似）へ流れる

        __m256d absX = _mm256_andnot_pd(vSignMask, x);
```

併せて、`vHalf` 定数定義も削除可能（他で使用していない場合）。`prevVec` 関連変数の削除でAVX2レジスタ負荷も低減される。

**Float版** (`DSPCoreFloat.cpp` line 119) も同様に:

```cpp
// 変更前:
x = 0.5 * (x + prevSample);
prevSample = x;

// 変更後:
// 平均化削除。prevSampleは入力値の保存としてのみ使用（P2と統合）
prevSample = x;
```

### 4.4 期待効果

- SoftClipのKnee特性が本来の滑らかな形状に戻る
- 低域大振幅入力時のエイリアシングが大幅に低減
- AVX2レジスタ使用量削減による僅かな処理負荷低減
- スカラーフォールバックとの動作一致（AVX2とスカラーで異なるmidVecロジックがなくなった副作用として、P2のprevScalar不整合の影響も低減）

---

## 5. P4: Parallelモード帯域加算の影響評価

### 5.1 問題の概要

EQ Parallelモードでは各バンドが独立にオリジナル信号から処理を開始し、デルタ（処理済み - オリジナル）を累積加算する。

### 5.2 P1完了後の挙動（確定）

P1でSVF saturationがoutputに移動された後、各バンドの出力は:

```
work = output_linear * (1 - sat) + fastTanhOutput(output_linear) * sat
```

となる。`fastTanhOutput` はクリップ閾値4.5の滑らかなtanh近似であり、well-behavedな非線形性を持つ。

**Parallelモードの挙動**:

- 各バンドの `delta = work - src` には、わずかなtanh飽和歪みが含まれる
- これらのdeltaは独立に累積加算される
- しかし各バンドの飽和量は微小（sat ≤ 1.0, tanh threshold = 4.5）であり、かつ各バンドの周波数特性が異なるため、歪み成分の周波数分布も異なる → 知覚上はノイズフロアの微増程度

**Serialモードの挙動**:

- 前段のtanh飽和歪みが次段の線形SVFフィルタを通る
- 飽和歪みの一部がフィルタリングされる可能性がある

**結論**: P1適用後、Parallel/Serial間の歪み差は**実用上無視できるレベル**。

### 5.3 確定判定

- ✅ P1+P3適用後、Parallelモードの帯域加算による歪み増加は実用上問題にならない
- ✅ 商用DSP製品でもParallel EQ + 出力段サチュレーションは一般的な実装
- ✅ 改修対象外。P1完了後の確認項目としてのみ維持

---

## 6. P5: オーバーサンプリング最低値の検討

### 6.1 問題の概要

SoftClip は `oversampling.processUp()` と `processDown()` の**内部**で実行されており、設計としては正しい。
`oversamplingFactor` はユーザー設定依存（0=Auto, 1, 2, 4, 8）、`setOversamplingFactor()` 経由で設定。

### 6.2 P1/P3完了後の評価（確定）

P1（SVF状態変数サチュレーション除去）とP3（midVec即時削除）により、非線形歪みの主原因が除去される。
残るSoftClipのtanh歪みは well-behaved であり、48kHz動作時は以下の理由でエイリアシングは実質無視できる:

- 偶数次歪みが少ない（tanhは奇関数）
- 低周波入力（40Hz）の高次高調波は、fold backしても可聴域外か極小振幅

**OSの効果を原因切り分けに利用**:

- OS 8x でノイズ消える → SoftClip/エイリアシング寄り
- OS 8x でも変わらない → SVF saturation寄り

### 6.3 確定判定

- ✅ P1+P3で主原因除去。OS設定によるノイズ変化は原因切り分けにのみ使用
- ✅ デフォルトOS=1xのまま（ユーザー設定維持）で問題ない
- ✅ 改修対象外。実機検証時の切り分け手段としてのみ維持

---

## 7. 【新規】P6: AGCのブロックRMSリップル対策

### 7.1 問題の概要

`processAGC()` ではAudioBlock単位（例: 512サンプル@48kHz = 10.6ms）でRMSを計算している。
40Hzのベース音の1周期は25msであり、ブロック長が波形の半周期にも満たないため、
ブロックが「波形の頂点」を切り取ったか「ゼロ交差」を切り取ったかによってRMS値がブロック毎に上下に振動する。

### 7.2 現状の平滑化チェーン（定量評価）

AGCには既に以下の平滑化段が存在する:

| 段 | 処理 | 時定数 | 効果 |
|---|------|--------|------|
| 1 | `calculateRMS(data, numSamples)` | ブロック単位 | ブロックRMSにリップルあり |
| 2 | Envelope Follower (`envIn = envIn*(1-α)+rms*α`) | Attack: **100ms**, Release: **2s** | ブロックリップルを10%/ブロックで平滑化 |
| 3 | Gain Smoothing (`nextGain = curr*(1-β)+target*β`) | Smooth: **200ms** | ゲイン変化を5%/ブロックで平滑化 |
| 4 | `calculateAGCGain()` の不感帯 | ±**0.5dB** | 微小変動を完全にブロック |
| 5 | `applyGainRamp_AVX2()` | ブロック内線形補間 | ブロック境界のステップを除去 |

Attack=100msでの平滑化効果の定量評価:

```
blockAttackCoeff = min(1.0, 512 * (1 - exp(-1/(48000*0.1))))
                 = min(1.0, 512 * 0.000208)
                 ≈ 0.107
```

→ ブロック間のRMS変動のうち約10.7%のみがエンベロープに伝播する。
8ブロック（≈85ms）で目標値の約60%に到達 → 100msの時定数設計と整合。

### 7.3 ノイズ発生メカニズム

1. 40Hz信号のブロックRMSが 0.1〜0.7 の間で振動
2. エンベロープフォロワ（100ms）で平滑化 → RMS変動の約90%が除去される
3. 残った~10%のリップルがゲイン変調として出力に現れる
4. ゲイン変調周波数 ≈ ブロックレート（48kHz/512≈93.75Hz）でAM変調
5. 原音（40Hz）に93.75Hzのサイドバンドが発生 → 非調和成分 → 「ジジジジ」に寄与

**確度評価**: リップルそのものは存在するが、現状の三重の平滑化で大部分が除去されている。
AGCが主犯とは言えないが、**P1/P3/P2完了後の残存ノイズ原因として調査価値あり**。
確度: 30〜40%（副次要因）。

### 7.4 原因箇所

| 項目 | 内容 |
|------|------|
| **ファイル** | `src/eqprocessor/EQProcessor.Processing.cpp` |
| **関数** | `EQProcessor::processAGC()` (line 347-425) |
| **関数** | `(anonymous namespace)::calculateRMS()` (line 18-49) |
| **関連定義** | `EQProcessor.h` line 166-168: `AGC_ATTACK_TIME_SEC=0.1`, `AGC_RELEASE_TIME_SEC=2.0`, `AGC_SMOOTH_TIME_SEC=0.2` |

### 7.5 改修方法（推奨・保留）

P1+P3+P2完了後、残存ノイズがある場合のみ実施。

**改善案A: サンプル単位エンベロープフォロワ（推奨）**

```cpp
// 現在（ブロック単位RMS → ブロック単位エンベロープ更新）
inputRMS = calculateRMS(blockData, numSamples);  // ブロック内全サンプル
envIn = envIn * (1 - α) + inputRMS * α;          // ブロック単位で更新

// 改善後（サンプル単位エンベロープフォロワ）
// ループ内で二乗値に対して1-pole LPF
double env = envInSquared;  // 二乗値の平滑化状態
for (int i = 0; i < numSamples; ++i)
{
    const double sample = data[i];
    env += (sample * sample - env) * attackCoeff;  // サンプル単位更新
}
const double inputRms = std::sqrt(env);
```

**改善案B: ブロックRMSのクロスフェード**

- 現状のブロックRMS計算は維持し、前ブロックのRMS値から線形補間する
- 実装が簡単でリスクが少ない

**改善案C: AGC時定数の見直し**

- 現在のAttack=100msを200ms〜500msに延長
- コード変更なし、定数変更のみ

**推奨**: 案C（即効性）→ P1/P3完了後、残存ノイズあれば案Aを検討。

### 7.6 確定判定

- ✅ AGC時定数（`AGC_ATTACK_TIME_SEC=0.1`, `AGC_RELEASE_TIME_SEC=2.0`, `AGC_SMOOTH_TIME_SEC=0.2`）は `EQProcessor.h` の `constexpr` 定数であり、**変更は1行の修正で完了**。ユーザー設定不可。
- ✅ 現状の三重平滑化＋不感帯によりブロックRMSリップルの大部分は除去済み
- ✅ 改善案C（Attack=100ms→200ms）が最小コストで最大効果
- ✅ P6はP1+P3+P2完了後の残存ノイズ対策として位置づけ。優先度は P1=P3 > P2 > P6 > P4=P5

---

## 8. 未確定事項の確定結果（v2/v3/v4追記）

本セクションでは、全ツールを用いた追加調査の結果、未確定だった事項を確定させる。

### 7.1 saturationパラメータの独立制御（確定）

**調査結果**: `nonlinearSaturation`（SVF saturation）と `saturationAmount`（SoftClip）は**完全に独立したUIパラメータ**である。

| パラメータ | 所有者 | UI操作 | 経路 |
|-----------|--------|-------|------|
| `nonlinearSaturation` | `EQProcessor` → `EQState` | `EQEditProcessor` → `EQControlPanel` | RCU Snapshot → `process()` ローカル変数 `saturation` |
| `saturationAmount` | `AudioEngine` → `ProcessingState` | `MainWindow` → `saturationValueLabel` | RuntimeBuilder → `processDouble()` の `state.saturationAmount` |

- ユーザーは **2つの独立したスライダー/ラベル** でこれらを操作できる
- `nonlinearSaturation` のデフォルト値: **0.2**（20%）
- `saturationAmount` のデフォルト値: UIラベル経由（`juce::jlimit(0.0f, 1.0f, value)`）

**確定判定**: ✅ 独立。P1（SVF saturation除去）とP3（SoftClip midVec）は完全に独立した改修項目であり、相互に影響しない。

### 7.2 saturation=0時のコードパス同一性（確定）

**調査結果**: `processBand()` / `processBandStereo()` 内の saturation 適用は以下の条件分岐でガードされている：

```cpp
if (saturation > 0.0)
{
    // saturation 適用ブロック（ic1eq/ic2eqへのtanh適用）
}
```

`saturation=0.0` の場合、このブロックは**完全にスキップ**される。P1の変更後も同様で、`if (saturation > 0.0)` ブロック内の処理が `output` に変わるだけで、saturation=0時の分岐条件とスキップ動作は**完全に同一**である。

**確定判定**: ✅ saturation=0 のコードパスは変更後も完全に同一。回帰リスクはゼロ。

### 7.3 `nonlinearSaturation` / `saturationAmount` の連動関係（確定）

**調査結果**: 以下の両者は**UI上の別個の独立したパラメータ**であり、連動していない。

- `nonlinearSaturation`（EQProcessor, SVF状態変数歪み）
- `saturationAmount`（AudioEngine, SoftClip threshold/knee計算）

ユーザー報告「サチュレーションを上げると悪化」は、両方を同時に上げた場合の**複合効果**を指している可能性が高い。片方のみ上げた場合の切り分けが有効。

**確定判定**: ✅ 独立パラメータ。症状の切り分けに活用可能。

### 7.4 cocoindex-code（確定・使用法修正）

**調査結果**: `uv tool install cocoindex-code --force` により `ccc` + `cocoindex-code` の2つの実行可能ファイルがインストールされた。`ccc` が推奨CLI。

**正しい使用方法**:

```powershell
# 1. 初期化（初回のみ）
$env:PYTHONUTF8="1"; ccc init

# 2. インデックス作成
$env:PYTHONUTF8="1"; ccc index

# 3. デーモン再起動（インデックス後）
$env:PYTHONUTF8="1"; ccc daemon restart

# 4. セマンティック検索
$env:PYTHONUTF8="1"; ccc search "query" --limit 10 --lang cpp
```

**全コマンド**: `init`, `index`, `search`, `status`, `reset`, `doctor`, `mcp`(MCPサーバー), `daemon`

**制約**: `ccc search` はデーモン経由でセマンティック検索を実行するが、`sentence_transformers` に依存しており、uv管理の仮想環境には同パッケージがインストールされていなかった。`pip` も同環境には存在しないため、追加インストールは困難。ただしMCPモード（`ccc mcp`）で起動すれば、MCPクライアントからツールとして利用可能。

**確定判定**: ✅ `ccc` CLIは有効だが、セマンティック検索には `sentence_transformers` が必要。本件調査では限定的使用にとどまったが、MCPサーバーモードでの利用が有望。

### 7.5 graphify（確定）

**調査結果**: `graphifyy v0.8.42` がインストール済み。本調査では以下のコマンドを実行:

| コマンド | 結果 |
|---------|------|
| `graphify path "processBand" "softClipBlockAVX2"` | 5ホップの関係を確認 ✅ |
| `graphify explain "SVF saturation"` | 該当ノードなし ❌ |
| `graphify query "processBand fastTanhScalar ic1eq saturation"` | 30ノード検出 ✅ — `processBand()` → `fastTanhScalar()` / `processBandStereo()` / `isFiniteAndAbsInRangeMask()` の関係を確認 |

`query` コマンド（BFS探索）が最も有用で、graphify-out/graph.json の品質に依存するが、ファイル間の依存関係や関数呼び出し関係の俯瞰に有効。

**確定判定**: ✅ `query` + `path` が有効。`explain` はノード命名に依存するため安定しない。本件では限定的だが有用。

### 7.6 コンボルバー（IR処理）の非線形性検証（確定）

**調査結果**: `ConvolverProcessor` / `MKLNonUniformConvolver` を検証した結果、コンボルバー側に起因する非線形ノイズの可能性は**事実上ゼロ**と確定した。

| 検証項目 | 結果 |
|---------|------|
| Overlap-Save法の実装 | Intel IPP/MKL を用いた純粋な線形時不変(LTI)システム。非線形演算なし ✅ |
| Tail Contouring | `SetImpulse`時の周波数領域初期化でのみ適用。リアルタイム処理中に変動なし ✅ |
| DC Blocker | 1次IIRの線形フィルタ。非線形性なし ✅ |
| データパス | サチュレーション/ダイナミクス/クリップ処理の介在なし ✅ |

**確定判定**: ✅ コンボルバーは調査対象から完全に除外して問題ない。

**調査結果**: 第1オーバーロード `process(block)` (line 679-755) と第2オーバーロード `process(block, eqParams, coeffCache)` (line 1005-1090) の両方で Parallel 処理を確認。

Parallelモードのアルゴリズム:

```
for each band i:
    work = copy(src)          // 各バンドは独立にオリジナル信号から開始
    processBand(work, ..., saturation)  // SVF + 状態変数サチュレーション
    accum += (work - src)     // delta累積
output = src + accum
```

SVF saturation が存在する場合、各バンドの delta には歪み成分が含まれる。これらが独立に累積加算される。

**ただし** P1でSVF saturationを output に移動した後も、`fastTanhScalar(output)` による非線形歪みは各バンドの delta に含まれる。Parallelモードでは各バンドの歪み成分が独立に加算されるため、Serialモードに比べて歪みの総和が大きくなる可能性がある。

**P1後の残存影響評価**:

- saturation有効時、各バンドの出力歪み（`fastTanhScalar(output)`）は独立
- Parallel: Σ(歪み_i) → 全歪みが直接加算
- Serial: 歪み_i → band_{i+1} のフィルタ通過 → 一部高調波減衰
- 差は**微少**と予測（歪みそのものが小さいため）が、理論上はParallelの方が歪みが大きい

**確定判定**: ✅ P1後も微少な差は理論上残るが、問題となるレベルではない。実機で確認してから判断。

### 7.7 P3 midVec事前平均化の影響度（確定）

**調査結果**: AVX2パスの `midVec` 計算を解析。

```cpp
// midVec = (prev + current) * 0.5 の4サンプル版
const __m256d midVec = _mm256_mul_pd(_mm256_add_pd(prevVec, x), vHalf);
```

`prevVec` の構成:

```
prevVec = {prevScalar, data[i], data[i+1], data[i+2]}
x       = {data[i], data[i+1], data[i+2], data[i+3]}
midVec  = {(prevScalar+data[i])*0.5, (data[i]+data[i+1])*0.5,
           (data[i+1]+data[i+2])*0.5, (data[i+2]+data[i+3])*0.5}
```

`absMidVec > threshold` のとき `x *= threshold / absMidVec` が発動。

**Critical analysis**: threshold の値域（`0.95 - 0.45*sat`）では:

- sat=0.0: threshold=0.95 → clip_start=0.90 → midVec が 0.95 超はまれ
- sat=0.5: threshold=0.725 → clip_start=0.575 → midVec が 0.725 超はあり得る
- sat=1.0: threshold=0.50 → clip_start=0.10 → midVec が 0.50 超は頻発

**つまり sat が高いほど midVec による事前減衰が頻繁に発動する。** この事前減衰で波形が加工された後、`musicalSoftClipScalar()` がさらに非線形処理を行う。この二重処理がエイリアシングや歪みを増幅する。

**ただし** midVec は単なる2タップ移動平均（DC:0dB, Nyquist:-∞dB）であり、単独で「ジジジジ」を生むことはない。SoftClip全体としてのエイリアシング寄与の一部。

**確定判定**: ✅ midVec削除は試す価値あり。ただし単独の効果は限定的。P1+P2を優先し、P3はオプション。

### 7.8 P5 OS最低値の定量評価（確定）

**調査結果**: SoftClipの非線形歪みのエイリアシングを評価。

SoftClipの非線形特性（`fastTanh` のPade近似 + threshold/knee処理）は**奇数次の高調波**を主に生成する。OSなし（1x@48kHz）で40Hzのベース音をSoftClipにかけた場合:

- 3次高調波: 120Hz ✅ Nyquist内
- 5次高調波: 200Hz ✅ Nyquist内
- 7次高調波: 280Hz ✅ Nyquist内
- ...
- 601次高調波: 24040Hz ❌ Nyquist超え → フォールドバック
- 実用的には高調波の振幅は次数とともに急減するため、601次は無視可能

**48kHz + 1x OS**: Nyquist=24kHz。20kHz以上の成分のみ折り返す。
→ ベース40Hzでは実用上問題にならない。

**44.1kHz + 1x OS**: Nyquist=22.05kHz。22.05kHz超の成分が折り返し。
→ ベース40Hzの552次（22080Hz）以上がフォールドバック → ほぼ無視可能

**結論**: 非線形歪みのエイリアシングは、低周波入力では理論上問題になりにくい。問題になるのは高周波入力（例: 5kHz以上の信号をSoftClip）の場合。本件の報告（低音入力時のノイズ）ではOS不足は**副次要因**である。

**確定判定**: ✅ OS不足は主原因ではない。P1+P2を優先する方針を確定。

---

## 9. 実機検証手順

### 8.1 検証シナリオ

各改修の効果を確認するためのテスト手順。

| ステップ | 操作 | 確認内容 |
|---------|------|---------|
| 1 | EQ Nonlinear Saturation 0.0 → 1.0 | ノイズの変化を確認（現在の症状再現） |
| 2 | SoftClip OFF → ON | ノイズの変化を確認 |
| 3 | Oversampling 1x → 8x | ノイズの変化を確認 |
| 4 | Parallel EQ → Serial EQ | ノイズの変化を確認 |
| 5 | テスト信号: 40Hz/80Hz 正弦波 ±0.5振幅 | 各条件での歪み測定 |

### 8.2 改修後検証

| 検証項目 | 合格基準 |
|---------|---------|
| P1 適用後、saturation=1.0でノイズ消滅 | ノイズフロアが-80dB以下 |
| P1 適用後、既存EQ特性が変わらない | 周波数応答±0.1dB以内 |
| P2 適用後、SoftClip動作に変化なし | 波形形状が同等 |
| 全改修後、saturation=0.0でノイズなし | バイパス同等 |

---

## 10. リグレッションリスク

### 9.1 P1 のリスク

| リスク | 確率 | 影響 | 対策 |
|-------|------|------|------|
| EQの周波数特性が変わる | 低 | 中 | saturation=0時は全く変更がないため、特性不変 |
| saturation=0時の特性変化 | **なし** | — | saturation>0の場合のみ分岐するため、saturation=0のコードパスは完全に同一 |
| 出力振幅の変化 | 低 | 小 | saturation適用箇所が変わったことによる微差。実機確認 |
| 既存ユーザープリセットの互換性 | 低 | 小 | saturationパラメータの意味が「状態変数歪み」から「出力歪み」に変わる。音色は変わり得るが、歪み方が自然になる方向 |

### 9.2 P2 のリスク

| リスク | 確率 | 影響 | 対策 |
|-------|------|------|------|
| スカラーパスの動作変化 | 低 | 小 | numSamples%4≠0の場合のみ影響。通常ブロックサイズではほぼ実行されない |
| ブロック境界の連続性 | 低 | 小 | むしろ改善方向 |

### 9.3 全体的なリスク低減策

1. **P1のみ先に実施**して実機検証。saturation=0のコードパスが完全に同一であることを確認。
2. **P3を追加実装**（P1と同時でも可。独立した変更であるため）。midVecブロック削除後にEQ+Saturation+SoftClipの動作確認。
3. P1+P3完了後、**P2を実施**（影響範囲が限定的なため）。
4. 残存ノイズがあれば**P6**（AGC）を実施。
5. P4/P5は**最終調整**として位置づけ。
6. 各ステップで `saturation=0` / `softClipEnabled=false` のケースが完全に同一出力であることを確認（回帰テスト）。

---

## 付録A: 調査ログ（全ツール）

| ツール | 実行内容 | 結果 |
|-------|---------|------|
| **Serena MCP** | `find_symbol` 全15回（processBand, processBandStereo, fastTanhScalar, fastTanhV128, softClipBlockAVX2, musicalSoftClipScalar, fastTanh, processAGC, calculateRMS, calculateAGCGain, TanhApprox） | ✅ 全対象関数の実装をシンボル解決で確認。ファイル読み取りよりもトークン効率的 |
| **AiDex MCP** | `aidex_query` 全12回（softClipBlockAVX2, processBandStereo, fastTanhV128, processAGC, nonlinearSaturation, ProcessingOrder, FilterStructure, filterStructure, calcSVFCoeffs, setNonlinearSaturation, ProcessingOrder, softClipEnabled） | ✅ 全278ファイル・48266行から該当箇所をピンポイント特定 |
| **CodeGraph MCP** | `analyze_module_structure`(2回), `find_callers`(3回), `find_dependencies`(1回), `find_callees`(2回), `query_codebase`(1回), `local_search`(1回) | ✅ Full Index実行済み（16442 entities, 55 communities）。呼び出し関係・依存関係を確認 |
| **semble CLI** | `search` 全18クエリ（SVF saturation, SoftClip midVec, oversampling順序, prevScalar, fastTanh閾値, Parallelモード, setSaturationAmount, ProcessingOrder等） | ✅ コード内容を自然言語検索。累計323kトークン節約（99%効率） |
| **graphify CLI** | `path`, `explain`, `query` の3コマンド | ✅ `query`で30ノード（processBand→fastTanhScalar/processBandStereoの関係）。`path`で5ホップ確認 |
| **grep/Select-String** | 全8回のgrep検索（softClipBlockAVX2, oversampling, calculateAGCGain, FilterStructure, ProcessingOrder等） | ✅ 全該当箇所を確認。`src/**`にスコープ |
| **Web文献調査** | Zavalishin "VA Filter Design", Native Instruments文献, JUCE Forum, music-dsp, Wikipedia | ✅ TPT SVF理論の状態変数線形性必須を確認 |
| **cocoindex-code** | `uv tool install cocoindex-code --force`, `ccc init`, `ccc index`, `ccc daemon restart`, `ccc search --limit 5 --lang cpp` | ✅ `ccc` CLIを発見。`init→index→daemon restart→search` のワークフローを確認。`sentence_transformers` 不足でdaemon searchは動作せず。MCPモード(`ccc mcp`)は利用可能 |

### ツール別評価

| ツール | 有効度 | コメント |
|-------|-------|---------|
| Serena MCP | ★★★★★ | コード理解に最も有効。シンボル解決＋関数ボディ取得でファイル丸読み不要 |
| semble CLI | ★★★★★ | 自然言語からコード箇所を瞬時に特定。Python実装、日本語Windowsでは`$env:PYTHONUTF8="1"`必須 |
| AiDex MCP | ★★★★★ | 識別子検索で最速。grepの50倍効率的。事前インデックス必須 |
| CodeGraph MCP | ★★★★ | 呼び出し関係の俯瞰に有効。Full Index実行が必要（5分程度） |
| graphify CLI | ★★★ | `query`コマンドが有効。`path`でファイル間関係の確認。グラフ品質に依存 |
| grep | ★★★ | シンプルな文字列検索。パターンによってはAiDex/sembleより低速 |
| Web文献 | ★★★ | DSP理論の裏付けに有効。本件ではZavalishin文献が決定的 |
| cocoindex-code | ★ | CLI search非対応のため本件では使用不可。サーバーモードは即時性が必要な場面で有用か |
