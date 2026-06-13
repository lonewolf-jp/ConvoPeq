# Intel Vtune Profiler 分析結果 詳細レビューレポート

> **作成日**: 2026-06-13
> **対象**: ConvoPeq.exe (Debug build)
> **分析ツール**: Intel Vtune Profiler (Event-based sampling)
> **分析担当**: GitHub Copilot (DeepSeek V4 Flash)

---

## 目次

1. [計測概要](#1-計測概要)
2. [CPU利用率の重大な問題](#2-cpu利用率の重大な問題)
3. [ホットスポット詳細分析](#3-ホットスポット詳細分析)
4. [マルチスレッド分析](#4-マルチスレッド分析)
5. [改善提案](#5-改善提案)
6. [総評](#6-総評)

---

## 1. 計測概要

| 項目 | 値 |
|---|---|
| Elapsed Time | 62.225s |
| CPU Time | 84.148s (CPU利用率 1.35x) |
| Instructions Retired | 736,908,144,182 (7369億命令) |
| Microarchitecture Usage | 50.7% (やや低い) |
| Total Thread Count | 80 |
| Paused Time | 0.025s |
| Logical CPU Count | 16 (Rocket Lake) |
| 収集時間 | 10:02:30 - 10:03:32 UTC (約62秒) |
| ビルド種別 | Debug ビルド |
| Finalization | Fast mode (サンプル間引きあり) |

### Top Hotspots サマリー

| 順位 | 関数 | CPU時間 | 割合 | 呼出命令数 |
|---|---|---|---|---|
| 1 | `isBadSample` | 10.144s | 12.1% | 135,324,476,705 |
| 2 | `std::clamp<double>` | 8.280s | 9.8% | 68,496,981,446 |
| 3 | `decimateStage` loop@555 | 7.223s | 8.6% | 98,734,802,802 |
| 4 | `std::bit_cast<uint64_t,double>` | 5.750s | 6.8% | 71,249,969,157 |
| 5 | `advanceState` loop@251 | 3.152s | 3.7% | 25,233,884,358 |
| | **上位5合計** | **34.549s** | **41.0%** | 399,040,114,468 |

### 関数別CPU時間完全リスト（主要）

| 関数 | CPU Time (Total) | 割合 | μUsage | Source File |
|---|---|---|---|---|
| isBadSample | 10.144s | 12.1% | 63.9% | CustomInputOversampler.cpp |
| std::clamp<double> | 8.280s | 9.8% | 66.7% | algorithm (STL) |
| decimateStage loop@555 | 7.223s | 8.6% | 73.7% | CustomInputOversampler.cpp |
| std::bit_cast<uint64_t,double> | 5.750s | 6.8% | 41.2% | bit (STL) |
| advanceState loop@251 | 3.152s | 3.7% | 69.5% | LatticeNoiseShaper.h |
| dotProductAvx2 loop@139 | 1.996s | 2.4% | 45.5% | CustomInputOversampler.cpp |
| PixelRGB::blend<PixelARGB> | 1.525s | 1.8% | 87.9% | juce_PixelFormats.h |
| memset_repstos | 1.483s | 1.8% | 5.6% | memset.asm |
| computeFeedback | 1.389s | 1.7% | 33.7% | LatticeNoiseShaper.h |
| processSample (self) | 1.172s | 1.4% | 41.9% | LatticeNoiseShaper.h |
| computeMaskingEnergyStable loop@714 | 1.082s | 1.3% | 61.7% | MklFftEvaluator.h |
| fillPath | 10.841s | 12.9% | 46.8% | juce_RenderingHelpers.h |
| paint (SpectrumAnalyzer) | 17.884s | 21.3% | 43.3% | SpectrumAnalyzerComponent.cpp |

---

## 2. CPU利用率の重大な問題

CPU Utilization Histogram が極めて深刻な状態を示しています：

| 同時利用論理CPU | 経過時間 | 評価 |
|---|---|---|
| 0 (Idle) | 2.26s | — |
| **1 (Poor)** | **49.48s (79.5%)** | ❌ **深刻** |
| 2 (Poor) | 4.26s | |
| 3 (Poor) | 0.61s | |
| 4 (Poor) | 4.01s | |
| 5 (Poor) | 1.57s | |
| 6-12 (Poor) | 0s | |
| 13-15 (Ok) | 0s | |
| 16 (Ideal) | 0s | |

**62秒中49.5秒（79.5%）が1コアしか使われていない**。

### 原因分析

- **オーディオ処理スレッド（~31.4s）**: 単一のリアルタイムスレッドで逐次実行
  - ASIO callback chain → AudioEngine::processBlockDouble → DSPCore::processDouble → 各DSPモジュール
- **UIレンダリング（~22.8s）**: メインスレッドでGDIソフトウェア描画が同期的に実行
- **NoiseShaperLearner ワーカー（~16.4s）**: 唯一のマルチスレッド実行だが、全体に対する割合が低い

**根本原因**: オーディオ処理パイプラインが本質的にシングルスレッド設計。マルチコアの利活用がほとんどできていない。

---

## 3. ホットスポット詳細分析

### 🔥 Hotspot #1: `isBadSample` (10.144s, 12.1%) — CustomInputOversampler.cpp

```cpp
inline bool isBadSample(double x) noexcept
{
    const uint64_t bits = std::bit_cast<uint64_t>(x);
    const uint64_t exp = bits & 0x7FF0000000000000ULL;
    if (exp == 0x7FF0000000000000ULL) return true; // NaN/Inf判定
    constexpr uint64_t limit = 0x4340000000000000ULL;
    return (bits & 0x7FFFFFFFFFFFFFFFULL) > limit;  // |x| > 1e20 判定
}
```

**コスト**: `isBadSample` 10.1s + `std::bit_cast` 5.75s (内部呼出) = **約15.9s (CPU時間の18.9%)**

**呼び出しコンテキスト**:

- `decimateStage` の **AVX2パス**（line ~500-520）: 4サンプルを個別スカラーisBadSampleで検査後、_mm256_set_pd で再パック
- `decimateStage` の **スカラーフォールバックパス**（line ~555）: 各サンプルごとに isBadSample + 積和
- 呼び出し深さ: decimateStage → processDown (3段) → DSPCore::processDouble → 各ブロック

**問題点**:

1. AVX2パス内で **4回のスカラーisBadSample呼出** → SIMDの利点を相殺
2. スカラーフォールバックでも各サンプルガード
3. 関数内でさらに `std::bit_cast` を呼出（別ホットスポット）

---

### 🔥 Hotspot #2: `std::clamp<double>` (8.280s, 9.8%)

**呼び出し元**: `LatticeNoiseShaper::advanceState` (LatticeNoiseShaper.h)

```cpp
inline void advanceState(std::array<double, kOrder>& channelState,
                         double error, const double* activeCoeffs) const noexcept
{
    double forward = error;
    double prev_backward = error;
    double* state = channelState.data();
    constexpr double kLatticeStateLimit = 2.0;

    for (int i = 0; i < kOrder; ++i)    // kOrder = 9
    {
        const double backward = state[i];
        const double nextForward = forward + activeCoeffs[i] * backward;
        const double nextBackward = activeCoeffs[i] * forward + backward;
        state[i] = std::clamp(prev_backward, -kLatticeStateLimit, kLatticeStateLimit); // ← 8.3s
        forward = nextForward;
        prev_backward = nextBackward;
    }
}
```

**問題点**: 9次元の格子フィルタ状態ベクトルの各要素を毎サンプル `std::clamp`（分岐あり実装）でクランプ。`_mm256_min_pd` / `_mm256_max_pd`（分岐なし）で置換可能。

---

### 🔥 Hotspot #3: `decimateStage` loop at line 555 (7.223s, 8.6%)

```cpp
// AVX2フォールバック後のスカラー積和ループ
for (int r = 0; r < stage.convCount; ++r) {
    const int idx = base - stage.convParity - (r << 1);
    // bounds check + isBadSample check
    const double x = history[idx];
    if (isBadSample(x)) { bad = true; break; }
    acc += coeffs[r] * x;
}
```

- **命令数: 987億**, CPI: 0.357, μUsage: 73.7%
- タップ数511-1023、halfband実効タップ数256-512
- 各サンプル出力あたり~256-512回のスカラー積和

---

### 🔥 Hotspot #4: `std::bit_cast<uint64_t,double>` (5.750s, 6.8%)

- **命令数: 712億**, CPI: 0.399, μUsage: 41.2% (低い)
- **用途**: isBadSample, fastAbs, killDenormal, isFiniteNoLibm の内部で使用
- Debugビルドでは関数呼出オーバーヘッドが増加しうる

---

### 🔥 Hotspot #5: `LatticeNoiseShaper::advanceState` (11.447s / loop@251: 3.152s)

- 全体: 11.4s (命令数42億, CPI 0.577, μUsage 52.8%)
- loop@251: 3.15s (命令数252億, CPI 0.582, μUsage 69.5%)

格子フィルタ漸化式の逐次依存（forward/backward更新）によりSIMD化が困難。

---

### 🔥 Hotspot #6: `SpectrumAnalyzerComponent::paint` (17.884s, 17.8%)

**内訳**:

- `fillPath` (JUCE GDI): 10.84s
- `paintSpectrum` 内 fillRect × 512本: 個別描画
- `memset_repstos`: 1.48s
- `EdgeTable` 関連: ~0.5s
- `PixelRGB::blend<PixelARGB>`: 1.53s (87.9% μUsage — 高効率だが回数が多い)

**JUCE内部の深い呼出**:

```
paint
  → paintSpectrum (fillRect × NUM_DISPLAY_BARS)
  → paintComponentAndChildren
     → handlePaint
        → performPaint (GDIRenderContext)
           → handlePaintMessage (WindowProc)
```

---

### Hotspot #7: NoiseShaperLearner 評価パス (16.4s, 19.5%)

| 関数 | CPU時間 | 備考 |
|---|---|---|
| `evaluateCandidateMapped` | 22.457s | (ループ@1270, 1276) |
| `runEvaluationJobsForWorker` | 19.712s | ワーカーメインループ@621 |
| `evaluationWorkerMain` | 16.449s | λ→thread::_Invoke |
| `computeMaskingEnergyStable` loop@714 | 1.082s | O(2049 × maskerCount) |
| `computeSfm` loop@650 | 0.447s | 各ノイズマスカーで再スキャン |
| `buildNoiseMaskersFixed` loop@679 | 0.426s | |

**問題点**: `computeSfm` が各ノイズマスカーで呼ばれ、内部で再度全ビンをスキャン。二重ループ構造によりO(n²)的挙動。

---

## 4. マルチスレッド分析

| スレッド種別 | CPU時間 | アクティブスレッド数 | 特性 |
|---|---|---|---|
| メインスレッド (UI) | ~22.8s | 1 | メッセージループ + GDIペイント |
| ASIOオーディオスレッド | ~31.4s | 1 | リアルタイムオーディオ処理 |
| NoiseShaperLearner workers | ~16.4s | 可変 | CMA-ES学習ワーカー（唯一の並列） |
| OSサービス/アイドル | 残り | — | Sleep, カーネル処理 |

「Total Thread Count: 80」はOS管理スレッド総数であり、アクティブに計算しているのは上記のみ。

---

## 5. 改善提案

### 🔧 優先度: 最高 — isBadSample の排除・効率化

**提案①: SIMD版 isBadSample の実装**

現状のAVX2パス:

```cpp
// 現在: 4回のスカラーisBadSample + 4回のスカラーロード → _mm256_set_pd
const double s0 = history[idx0];
const double s1 = history[idx1];
const double s2 = history[idx2];
const double s3 = history[idx3];
if (isBadSample(s0) || isBadSample(s1) || isBadSample(s2) || isBadSample(s3)) { ... }
const __m256d vSamples = _mm256_set_pd(s3, s2, s1, s0);
```

**改善案**:

```cpp
// 改善: 1回のSIMDロード + 1回のSIMDチェック
const __m256d vSamples = _mm256_loadu_pd(&history[idx0]);
if (isBadSampleV(vSamples)) { ... }  // 1 SIMD命令で4要素チェック
```

`isBadSampleV` は `_mm256_cmp_pd` で絶対値 > limit を一括判定。

**推定削減効果**: isBadSample 呼出を75%削減 → **~4s 削減**

---

**提案②: isBadSample の間引きチェック**

リアルタイムオーディオパスにおいて、定常状態のFIRフィルタ演算中にNaN/Infが突発的に発生することはほぼない。以下の戦略でチェック頻度を低減：

- 通常時: 64サンプルまたは128サンプルに1回のsparse checkに間引く
- バッファクリア後/リセット直後のみ: 全サンプルチェック
- 不正値検出時: `markCorruptionDetected()` で事後検出

**推定削減効果**: isBadSample + bit_cast のほぼ全削減 → **~16s 削減**

---

### 🔧 優先度: 高 — decimateStage AVX2 パスの改善

**提案③: スカラーロード+再パックを `_mm256_loadu_pd` に統一**

現状の `history[idx0]` ... `history[idx3]` + `_mm256_set_pd` による4回の個別ロード+再パックは、`_mm256_loadu_pd(&history[idx0])` の1命令に置換可能。これにより：

- メモリアクセス4回 → 1回
- isBadSample チェック4回 → 1回のSIMDチェック

**推定削減効果**: decimateStage 全体の ~2-3s 削減

---

**提案④: advanceState の std::clamp を SIMD に置換**

```cpp
// 現状（9回のスカラーstd::clamp）
state[i] = std::clamp(prev_backward, -kLatticeStateLimit, kLatticeStateLimit);

// 改善案①: 前半8要素をSIMD + 残り1要素スカラー
for (int i = 0; i < kOrder; i += 4) {
    __m256d v = _mm256_loadu_pd(&prevBackwardV[i]);
    v = _mm256_max_pd(v, vNegLimit);
    v = _mm256_min_pd(v, vLimit);
    _mm256_storeu_pd(&state[i], v);
}
// ただし格子フィルタの逐次依存があるため、現状のスカラーループを保ちつつ
// std::clamp の箇所のみビット演算で書き換え:
state[i] = prev_backward < -kLatticeStateLimit ? -kLatticeStateLimit
         : (prev_backward > kLatticeStateLimit ? kLatticeStateLimit : prev_backward);
```

**注意**: `prev_backward` の値がループ搬运依存であるため完全SIMD化は困難。しかし `std::clamp` の関数呼出オーバーヘッド（Debugビルドで顕著）を排除するだけでも効果大。

**推定削減効果**: **~8s 削減**（std::clamp 8.3s 相当の大部分）

---

**提案⑤: computeFeedback 結果レジスタの再利用**

`processSample` 内:

```
feedback = computeFeedback(state, coeffs)   // SIMD dot → スカラー
shapedInputClean = input * headroom + feedback
quantized = quantize(shapedInputClean)
error = quantized - shapedInputClean
clampedError = clamp(error)
advanceState(state, clampedError, coeffs)   // スカラー漸化
```

`computeFeedback` で計算したSIMDレジスタ `v0*c0 + v1*c1` の結果を `advanceState` に渡せないか検討。ただし格子フィルタの漸化式は `forward` をスカラーで持ち越すため、単純なベクトル化は困難。

---

### 🔧 優先度: 中 — UI レンダリングの高速化

**提案⑥: GDI から Direct2D に切替（JUCE 8.0.12対応）**

現在は JUCE の `GDIRenderContext`（ソフトウェアラスタライザ）を使用。JUCE 8.0.12 では `Direct2D` / `OpenGL` バックエンドをサポート。

```cpp
// App initialization で:
juce::OpenGLContext openGLContext;
openGLContext.attachTo(*getTopLevelComponent());
// または Direct2D の場合:
// juce::RuntimePermissions::registerSDK(juce::RuntimePermissions::direct2D, true);
```

**推定削減効果**: fillPath 10.8s → GPU処理で ~1s 未満に激減

---

**提案⑦: SpectrumAnalyzer の描画を Image キャッシュ＋差分更新に変更**

現状: `paint()` 毎に全512本の `fillRect` を再描画 + グリッド + 枠線

改善案:

1. 背景+グリッドを `Image` として事前レンダリング（`resized()` 時のみ再生成）
2. スペクトラムバーのみ `Image` にレンダリングして `drawImageAt()`
3. `repaint(plotArea)` で描画領域を限定

---

**提案⑧: Release ビルドでの再計測を推奨**

Debug ビルドでは：

- STL関数（`std::clamp`, `std::bit_cast`）のインライン化が抑制
- `RTC_CheckStackVars` + `RTC_CheckStackVars` ループ: 1.08s
- `heap_alloc_dbg_internal`: 0.35s + `free_dbg_nolock`: 0.11s
- デバッグアサーション/CRCチェック: 0.06s

Release ビルドでは CPI が 0.5-0.9 → 0.3-0.5 程度に改善見込み（特に STL 関数）。

---

**提案⑨: `computeMaskingEnergyStable` の O(n²) 的構造改善**

```cpp
for (int i = 0; i < kSpectrumBins; ++i)        // 2049 bins
    for (int j = 0; j < maskers.size; ++j)      // ~100 maskers
        // deltaBark チェック + spreadingFunctionAnnexD + push
```

各ビンで全マスカーをスキャン。`kSpreadMaxDeltaBark` でフィルタリングしているが、事前にマスカーをBark領域でソート＋範囲限定探索にすれば計算量削減可能。

---

**提案⑩: `computeSfm` の統合**

`buildNoiseMaskersFixed` 内で各バンドごとに `computeSfm` を呼出 → 内部で全ビン再スキャン → この二重ループを単一パスに統合。

---

### 🔧 優先度: 低 — その他

**提案⑪: スカラー `killDenormal` をSIMD処理内に統合**

`killDenormal` (0.12s) は `_mm256_and_pd` でSIMD化済みのバージョンが存在する。スカラー版が使われている箇所をSIMD版に統一。

**提案⑫: NoiseShaperLearner の評価ループベクトル化**

`evaluateCandidateMapped` 内のレベル別ループで各セグメント処理。セグメント間は独立しているため、アンロール＋SIMD化の余地あり。

---

## 6. 総評

### 改善インパクトマトリクス

| 提案 | カテゴリ | 推定削減時間 | 難易度 | リスク |
|---|---|---|---|---|
| ① SIMD isBadSample | オーディオ | ~4s | 中 | 低 |
| ② isBadSample間引き | オーディオ | ~10-16s | 低 | 中（デバッグ容易性低下） |
| ③ AVX2ロード統一 | オーディオ | ~2-3s | 低 | 低 |
| ④ std::clamp置換 | オーディオ | ~8s | 中 | 低（等価変換） |
| ⑤ レジスタ再利用 | オーディオ | ~0.5-1s | 高 | 中 |
| ⑥ Direct2D化 | UI | ~10-15s | 中 | 低（JUCE標準機能） |
| ⑦ 描画キャッシュ | UI | ~5-10s | 低 | 低 |
| ⑧ Release再計測 | 計測 | — | 低 | なし（推奨） |
| ⑨ マスカーO(n²)改善 | 学習 | ~0.5s | 中 | 低 |
| ⑩ computeSfm統合 | 学習 | ~0.3s | 中 | 低 |
| ⑪ killDenormal統一 | 全般 | ~0.1s | 低 | 低 |
| ⑫ 評価ループSIMD化 | 学習 | ~1s | 高 | 中 |

### 優先実行推奨パッケージ

**Phase 1（即効性・低リスク）**:

1. ⑧ Releaseビルドで再計測（正確な現状把握）
2. ③ AVX2 ロード統一 + SIMD isBadSample（①）
3. ④ std::clamp → インラインmin/max
4. ⑦ 描画キャッシュ

**Phase 2（中程度の工数・効果大）**:
5. ⑥ Direct2D 化
6. ② isBadSample sparse check

**Phase 3（構造的改善・高工数）**:
7. ⑨ マスカー探索改善
8. ⑩ computeSfm 統合
9. ⑤ レジスタ再利用
10. ⑫ 評価ループベクトル化

### 重要注意点

- **Debugビルドで計測されている**ため、ReleaseビルドではSTL関数のオーバーヘッドが大幅に低減される（特に `std::clamp`, `std::bit_cast`）
- **1コア集中（79.5%）** はアーキテクチャレベルでの課題。オーディオ処理のマルチスレッド化は信号処理の逐次依存性から困難だが、UIとNoiseShaperLearnerの並列度は向上可能
- `isBadSample` + `std::clamp` だけで全CPU時間の **21.9%** を消費しており、ここに手をつけるだけで飛躍的な改善が期待できる

---

*以上*
