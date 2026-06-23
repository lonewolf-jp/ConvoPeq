# CMA-ES 学習戦略解析

**日付**: 2026-06-23
**対象**: `NoiseShaperLearner` + `MklFftEvaluator`

---

## 概要

CMA-ES 学習器は、**聴覚心理学モデル（psychoacoustic model）に基づく目的関数**を使用してノイズシェーパー係数を最適化している。単純な「最小二乗誤差」や「FIR vs IIR」の枠組みをはるかに超える、**プロフェッショナル向けの高度な最適化**が実装されている。

## 3層の重み付け構造

### 第1層: 信号レベル重み

`NoiseShaperLearner.h` 定義:

| レベル (dBFS) | 重み | 備考 |
|---------------|------|------|
| -40 | **0.4** | 最も重要な低レベル帯域 |
| -30 | **0.3** | |
| -20 | **0.2** | |
| -10 | **0.1** | 高レベルは低重み |

→ **低レベル信号のノイズ品質が最優先**（高レベルではマスキング効果でノイズが知覚されにくいため）。

#### αブレンド（各レベル内）

```
Low level (-40/-30dBFS): alpha = 0.3 → 時間領域RMS重視 (70%)
High level (-20/-10dBFS): alpha = 0.7 → 周波数領域重視 (70%)
```

低レベルでは広帯域ノイズの総量を抑え、高レベルでは周波数特異的なノイズを抑える戦略。

### 第2層: 聴覚心理学的周波数重み

`MklFftEvaluator::evaluate()` 内の psychoacoustic スコアリング:

#### 2a. バーク帯域マスキングモデル

- 24 バーク帯域で信号のマスキングカーブを計算
- **トーナルマスカー検出**: 隣接ビンより 6dB 以上大きいピークをトーナル成分として検出
- **ノイズマスカー検出**: トーナルでない成分をノイズマスカーとして処理
- マスキングエネルギーから各ビンの**最小可聴閾値**を計算

#### 2b. ATH (Absolute Threshold of Hearing)

- ISO 規格に基づく最小可聴曲線
- `computeAthSplDb()` で周波数ごとに算出
- マスキング閾値の下限として適用

#### 2c. 周波数重み係数 `bandWeightForHz()`

```
       f²
W(f) = ──────────────
       √(h1² + h2²)
```

- ISO 聴感補正カーブを模擬
- **中音域 (2-4kHz) が最大重み**
- 18kHz 以上は 12dB/oct でロールオフ減衰

#### 2d. JND (Just Noticeable Difference) 重み

```
jndWeight = 1.0 / JND(f)
```

- 周波数ごとの最小可聴レベル差の逆数
- 感度の高い周波数帯域（中音域）がより大きな重みを持つ

#### 2e. ペナルティ項

| ペナルティ | 対象範囲 | 重み | 効果 |
|-----------|---------|------|------|
| Spectral Flatness | flatnessStartBin〜flatnessEndBin (≈12-18kHz) | 0.35 | トーナルノイズを強くペナルティ |
| HF Penalty | ultraHighStartBin〜Nyquist (≈Nyquist×0.85〜) | 0.05-0.20 | 超高域の過剰エネルギーを抑制 |
| Tonal Penalty | 全帯域（ピーク検出） | 可変 | トーナルスパイクを抑圧 |

#### 2f. compositeScore の計算式

```
noisePower = psychoWeighted / psychoWeightSum * kFftLength
  (心理音響重み付きノイズパワー)

compositeScore = noisePower
               × (1 + 0.35 × spectralFlatnessPenalty
                      + hfPenaltyWeight × hfPenalty
                      + tonalPenalty)
```

### 第3層: 学習プロセス全体の制御

```
evaluateCandidate():
  candidateCoeffs → tanh() → clampCoeff(margin)  // CMA-ES 探索空間の制約
  ↓
evaluateCandidateMapped():
  isStable() ? |k_i| < 1  // 安定性フィルタ
  shaper.processStereoBlock()  // 実コードで音声処理
  fftEvaluator.evaluate()  // 心理音響スコアリング
  levelWeightedAverage()  // 4レベルの加重平均
  return score
```

## 学習方針の定性的解釈

```
CMA-ES が「良い」と判断するノイズシェーパー:
┌─────────────────────────────────────────────┐
│  1. 低レベル信号 (-40dBFS) でのノイズが最小  │ ← 最重要
│  2. 中音域 (2-4kHz) のノイズが最小           │ ← 聴感上最も敏感
│  3. マスキング閾値以下のノイズが最小          │ ← 心理音響的不可聴が理想
│  4. 高域 (12-18kHz) がホワイトノイズ的        │ ← トーナルノイズ回避
│  5. 超高域が過剰でない                       │ ← 帯域外エネルギー抑制
│  6. トーナルスパイクがない                   │
└─────────────────────────────────────────────┘
```

## 「現在の advanceState() 構造が CMA-ES に与える影響」に関する考察

現在の IIR 積分器構造は、低域のノイズシェーピングを増強する方向に働く。
CMA-ES はこの特性を「良いスコアを出す手段」として利用している可能性がある。

仮に CMSIS-DSP 準拠の純粋 FIR に変更した場合:

- CMA-ES は低域ノイズを同程度に抑制するために**異なる係数セット**を学習する必要がある
- 純粋 FIR の方が目的関数の landscape が滑らか（clamp 非線形性がない）ため、**収束が速くなる可能性**がある
- しかし最終的な到達スコアの優劣は事前には判断できない（NTF 実測が必要）

## 結論

CMA-ES 学習器は「理想的 NTF からの乖離」ではなく「**心理音響スコアの最小化**」を目的として設計されている。現在の advanceState() が CMSIS-DSP 準拠でないことは、この目的に対して**直接的には問題にならない**。重要なのは「現在の構造で CMA-ES が十分に低いスコアを達成できるか」であり、その評価には NTF 実測比較が必要。
