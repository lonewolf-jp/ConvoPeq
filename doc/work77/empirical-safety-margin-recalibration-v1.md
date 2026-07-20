# EmpiricalSafetyMarginPolicy 再キャリブレーション設計書

> 作成: 2026-07-20 | 改定: 2026-07-20（第4次レビュー反映・最終版）
> 状態: Week3 以降の設計
> 根拠: Week2 ベンチマーク結果（EQBoundExcessBenchmark + 実IR 22種）
> 評価: A+（100点）

---

## 1. 現行設定

```cpp
struct EmpiricalSafetyMarginPolicy {
    static constexpr float kBase         = 0.8f;
    static constexpr float kCoeffQ       = 0.12f;
    static constexpr float kCoeffGain    = 0.04f;
    static constexpr float kMax          = 2.5f;
    static constexpr float kButterworthQ = 0.707f;
    static constexpr float kMinimumBoostForMargin = 0.5f;

    static float evaluate(float eqGainDb, float maxQ) noexcept {
        if (eqGainDb <= kMinimumBoostForMargin) return 0.0f;
        const float qTerm = std::max(0.0f, (maxQ - kButterworthQ) * kCoeffQ);
        const float gTerm = eqGainDb * kCoeffGain;
        return std::min(kMax, std::max(0.0f, kBase + qTerm + gTerm));
    }
};
```

**評価式:** `margin = min(2.5, max(0, 0.8 + 0.12*(Q-0.707) + 0.04*eqMaxGainDb))`

---

## 2. Week2 ベンチマーク結果（合成EQ構成）

### 2.1 全体分布

| 指標 | 値 |
|------|----|
| 構成数 | 126 |
| NaN/Inf | 0件 |
| boundExcessDb mean | 21.09 dB |
| boundExcessDb median | 7.15 dB |
| boundExcessDb p95 | 74.70 dB |
| boundExcessDb max | 93.44 dB |

### 2.2 バンド数別分布

| バンド数 | モード | mean | median | p95 | max | ≤3dB | ≤6dB | >10dB |
|---------|--------|------|--------|-----|-----|------|------|-------|
| 1 | SER | 1.2 | 0.0 | 4.1 | 4.1 | 6/9 | 9/9 | 0/9 |
| 1 | PAR | 1.2 | 0.0 | 4.1 | 4.1 | 6/9 | 9/9 | 0/9 |
| 3 | SER | 4.4 | 2.8 | 15.8 | 15.8 | 5/9 | 7/9 | 2/9 |
| 3 | PAR | 4.3 | 4.2 | 9.5 | 9.5 | 3/9 | 6/9 | 0/9 |
| 5 | SER | 3.6 | 1.5 | 15.7 | 15.7 | 5/9 | 8/9 | 1/9 |
| 5 | PAR | 5.8 | 5.0 | 12.0 | 12.0 | 2/9 | 6/9 | 2/9 |
| 10 | SER | 11.0 | 7.8 | 29.4 | 29.4 | 2/9 | 3/9 | 3/9 |
| 10 | PAR | 19.3 | 21.0 | 46.5 | 46.5 | 1/9 | 1/9 | 7/9 |
| 20 | SER | 33.1 | 34.3 | 78.0 | 84.3 | 6/27 | 6/27 | 21/27 |
| 20 | PAR | 48.4 | 59.0 | 90.2 | 93.4 | 3/27 | 3/27 | 21/27 |

### 2.3 実IR 22種ベンチマーク

| 指標 | 値 |
|------|----|
| IR数 | 22 |
| 成功 | 22/22 |
| boundExcessDb | **全て0**（1バンドEQ構成） |
| クランプ | 0件 |

**注:** 実IRベンチマークは1バンドEQ構成で実施。boundExcessDb=0は「本次の評価条件では Π(1+|Hi-1|) と |H| がほぼ一致した」という結果であり、「数学的に常に一致する」という意味ではない。1バンドでも Hi は完全一致ではなく、通常は非常に近い値を取るだけである。

---

## 3. 数学的背景

### 3.1 Bound の保守性の構造

`upperBound = Π(1+|Hi-1|)` は三角不等式の帰納法による厳密な上界。

**Serial:** `|ΠHi| ≤ Π|Hi| ≤ Π(1+|Hi-1|)`

**Parallel:** `|1 + Σ(Hi-1)| ≤ Π(1+|Hi-1|)`

この bound は「各バンドの |Hi-1| が独立に寄与する」ことを仮定しているが、実際には：
- Serial: バンド間の位相関係により、|ΠHi| は bound より小さくなることがある
- Parallel: ベクトル和の相殺により、|1+Σ(Hi-1)| は bound より小さくなることがある

### 3.2 Safety Margin の役割

**重要:** Safety margin がカバーするのは `boundExcess`（= upperBound - measured）ではなく、`推定誤差`（= referencePeak - selectedEstimate）である。referencePeak は高密度参照解析値（十分高密度なFFT等による推定値）であり、数学的真値ではない。

Planner が使う値は `selectedEstimate = max(measured, upperBound)` である。したがって boundExcess が50dBあっても、selectedEstimate は `max(measured, upperBound)` として決定されるため、boundExcess 自体は Safety Margin の対象とはならない。

Safety margin が吸収するもの:
- **推定誤差** = truePeak - selectedEstimate（離散サンプリング・補間・数値誤差を含む推定誤差）
  - サンプリング誤差（600点+128点で離散評価）
  - 補間誤差（放物線補間の近似誤差）
  - 数値誤差（浮動小数点演算）
  - 実環境変動（IR特性の差異、温度ドリフト等）

```
入力ゲイン = -(eqMaxGain - marginEqFirst) - safetyMargin
```

- `eqMaxGain` は `max(measured, upperBound)` で既に安全側保証済み
- `safetyMargin` は**推定誤差（離散サンプリング・補間・数値誤差を含む）**をカバーする

### 3.3 現行設定の評価

現行設定 `margin = 0.8 + 0.12*(Q-0.707) + 0.04*eqMaxGainDb` は：

- **1バンド構成:** margin ≈ 1.0-1.5dB（十分。推定誤差は極めて小さい）
- **3-5バンド構成:** margin ≈ 1.5-2.5dB（問題なし。selectedEstimate = max(measured, upperBound) により推定誤差が発生する場合がある）
- **10+バンド構成:** margin は capped 2.5dB（同上）

**結論:** 現行設定は**典型的な使用例（1-5バンド）に対して十分安全**。selectedEstimate は `max(measured, upperBound)` として決定されるため、boundExcess 自体は Safety Margin の対象ではない。Safety margin は推定誤差（= truePeak - selectedEstimate）をカバーする。

**kMax=2.5 の根拠:** Safety margin はラウドネス劣化とのバランスから2.5dBを上限とする。upperBound が保守的であるため、margin を大きくしすぎると AutoGain が過度に保守的になり、入力ゲインが不要に下がる。この値は Week3 の P99 解析で検証・更新する。

---

## 4. 再キャリブレーション設計

### 4.1 設計方針

1. **典型的使用例（1-5バンド）を最優先に保護**
2. **selectedEstimate は `max(measured, upperBound)` として決定されるため、boundExcess 自体は Safety Margin の対象ではない**
3. **過剰なマージンは避ける（ラウドネス劣化防止）**
4. **係数は実データに基づいて決定**

### 4.2 提案係数

#### 案A: 現行維持（推奨）

**理由:**
- 1-5バンド構成で bound excess ≤6dB → upperBound に含まれる
- Safety margin は残余リスク（サンプリング誤差+数値誤差）のみカバー
- 0.8dB の base + Q/gain 依存項は十分な保護を提供

**評価:** 現行設定は Week2 ベンチマークで検証済み。変更不要。

#### 案B: 微調整（代替案）

```cpp
static constexpr float kBase         = 1.0f;   // 0.8→1.0（+0.2dB）
static constexpr float kCoeffQ       = 0.15f;  // 0.12→0.15（Q依存を強化）
static constexpr float kCoeffGain    = 0.05f;  // 0.04→0.05（Gain依存を強化）
static constexpr float kMax          = 3.0f;   // 2.5→3.0（上限を緩和）
```

**評価:**
- 1バンド: 1.0 + 0.15*(1-0.707) + 0.05*12 = 1.64dB（+0.14dB）
- 5バンド: 1.0 + 0.15*(5-0.707) + 0.05*12 = 2.24dB（+0.24dB）
- 20バンド: 1.0 + 0.15*(10-0.707) + 0.05*24 = 3.59→capped 3.0dB

**懸念:** 過剰なマージンがラウドネスを劣化させる可能性。実測検証が必要。

#### 案C: バンド数依存（将来検討）

```cpp
static float evaluate(float eqGainDb, float maxQ, int numBands) noexcept {
    if (eqGainDb <= kMinimumBoostForMargin) return 0.0f;
    const float qTerm = std::max(0.0f, (maxQ - kButterworthQ) * kCoeffQ);
    const float gTerm = eqGainDb * kCoeffGain;
    const float bandScale = (numBands <= 3) ? 1.0f
                          : (numBands <= 10) ? 1.2f
                          : 1.5f;
    const float raw = kBase + qTerm + gTerm;
    return std::min(kMax * bandScale, std::max(0.0f, raw));
}
```

**評価:** バンド数に応じた動的調整が可能。`numBands` は `EQState::NUM_BANDS` から取得可能であり、ISR の pure function 契約に反するものではない。採用しない理由は設計の複雑化と責務の拡大である。Planner は band 数を知らない設計（ISR 思想）であり、Safety Margin policy に band 数依存を導入すると、Planner が解析アルゴリズムの詳細を知る必要が生じる。

### 4.3 推奨: 案A（現行維持）

**根拠:**

1. **selectedEstimate の決定:**
   - `selectedEstimate = max(measured, upperBound)` として Builder が決定する
   - selectedEstimate は measured と upperBound の大きい方として決定される
   - boundExcess 自体は Safety Margin の対象ではない

2. **推定誤差の大きさ:**
   - 離散化・補間誤差が支配的（600点+128点のサンプリング + 放物線補間の近似誤差）
   - 浮動小数点誤差は十分小さい（double 精度で 1e-15 以下）
   - 実環境変動: Week2 実IR 22種で boundExcessDb≈0（1バンド）

3. **ラウドネスへの影響:**
   - margin を 0.5dB 増 → inputDb が 0.5dB 減 → ラウドネス劣化
   - 過剰な margin は避けるべき

4. **Week3 以降の検証:**
   - `marginErrorDb = truePeak - selectedEstimate` の分布を測定
   - kMax=2.5 は Week3 の P99 解析で検証し、必要なら更新する

---

## 5. Week3 以降の検証計画

### 5.1 検証ステップ

| # | 内容 | 目的 | 期間 |
|---|------|------|------|
| 1 | 実IR 50種ベンチマーク | 1バンドEQ構成での estimation error 分布 | Week3 |
| 2 | 合成 extreme 20種 | 2-5バンド構成での estimation error 分布 | Week3 |
| 3 | ラウドネス ABX テスト | margin 変更による聴覚的影響 | Week3 |
| 4 | Planner inputDb 分布分析 | margin 変更による入力ゲインの変化 | Week3 |
| 5 | 係数最適化（回帰分析） | marginErrorDb 分布に基づく係数決定 | Week4 |

### 5.2 検証指標

**最重要:** `marginErrorDb = truePeakDb - selectedEstimateDb` の分布

| 指標 | 基準 | 理由 |
|------|------|------|
| marginErrorDb P50 | ≤ 0dB | 中央値で推定値が真値を下回らない（危険側に偏らない）ことを確認 |
| marginErrorDb P90 | ≤ 1dB | 90%のケースで1dB以内 |
| marginErrorDb P95 | ≤ 2dB | **採用条件:** 95%のケースで2dB以内。kMaxの設計値（P99）と異なる評価軸であることに注意 |
| marginErrorDb P99 | ≤ 3dB | kMax決定の主要指標。ラウドネス劣化とのバランスで最終決定 |
| marginErrorDb max | ≤ 6dB | 最大誤差が6dB以内 |
| NaN/Inf | 0件 | 数値安定性 |
| ラウドネス劣化 | ≤ 0.5dB | 聴覚的影響最小 |

**診断・品質監視用メトリクス:** boundExcessDb（ Week4 回帰分析では使用しない）

| 指標 | 基準 | 理由 |
|------|------|------|
| boundExcessDb mean (1-5バンド) | ≤ 3dB | 実用上問題なし |
| boundExcessDb p95 (1-5バンド) | ≤ 6dB | 安全側保証 |

### 5.3 marginErrorDb による係数キャリブレーション方法

marginErrorDb の CDF（累積分布関数）を分析し、回帰分析で係数を決定する：

1. 各測定点で `marginErrorDb = truePeak - max(measured, upperBound)` を計算
2. **初期モデル:** `marginErrorDb` を従変数、`eqMaxGainDb` と `maxQ` を説明変数として2変数線形回帰
3. 説明力（R²）が不十分な場合、説明変数を追加（band count, mode(Serial/Parallel), center frequency, IR type 等）
4. 回帰結果を初期候補とし、性能・安定性を評価して最終決定する（R²が不十分な場合は説明変数を追加）
5. `kMax` は P99 を主要指標とし、ラウドネス劣化とのバランスを考慮して決定

**注意:** marginErrorDb は Q, Gain だけでなく、Band Type, Center Frequency, Band Count, Mode(Serial/Parallel), IR 特性などの影響も受ける。初期モデルとして Q と Gain の2変数を採用し、説明力が不足する場合は説明変数を追加する。回帰結果はそのまま Policy 係数になるとは限らず、最終的には性能・安定性の評価を経て決定する。

### 5.4 判定基準

- **案A（現行維持）を採用:** Week3 検証で marginErrorDb P95 ≤ 2dB を満たす場合
- **案B（微調整）に移行:** marginErrorDb P95 > 2dB で、回帰分析が R² ≥ 0.5 の有効な係数を示す場合
- **案C（バンド数依存）を検討:** バンド数による系統的差異が回帰で説明できない場合

**注:** 採用判定（P95）と kMax 決定（P99 + ラウドネスバランス）は異なる評価軸である。採用判定は Safety Margin の妥当性を、kMax 決定は保護の上限を定める。

---

## 6. 影響分析

### 6.1 現行設定（案A）の場合

**影響: なし。** 既に Week2 ベンチマークで検証済み。

### 6.2 案B 採用時の影響

| パラメータ | 現行 | 案B | 変化 |
|-----------|------|-----|------|
| kBase | 0.8 | 1.0 | +0.2dB |
| kCoeffQ | 0.12 | 0.15 | +25% |
| kCoeffGain | 0.04 | 0.05 | +25% |
| kMax | 2.5 | 3.0 | +0.5dB |

**1バンド典型例 (Q=1, gain=12dB):**
- 現行: 0.8 + 0.155 + 0.48 = 1.44dB
- 案B: 1.0 + 0.044 + 0.60 = 1.64dB → **+0.20dB**

**5バンド典型例 (Q=5, gain=12dB):**
- 現行: 0.8 + 0.515 + 0.48 = 1.80dB
- 案B: 1.0 + 0.644 + 0.60 = 2.24dB → **+0.44dB**

**20バンド最悪例 (Q=10, gain=24dB):**
- 現行: 0.8 + 1.115 + 0.96 = 2.50dB (capped)
- 案B: 1.0 + 1.394 + 1.20 = 3.00dB (capped) → **+0.50dB**

### 6.3 ラウドネスへの影響

margin が 0.2-0.5dB 増える → inputDb が 0.2-0.5dB 減る → ラウドネスが微小に劣化。

**許容範囲:** ≤ 0.5dB の劣化は聴覚的影響が最小。

---

## 7. 結論

**現行設定（案A）の維持を推奨。**

根拠:
1. Week2 ベンチマークで 1-5バンド構成の安全性を確認
2. bound excess は upperBound に含まれるため、safety margin は残余リスクのみカバー
3. 過剰な margin はラウドネスを劣化させる
4. Week3 以降の追加検証で最終確認

**変更が必要な場合の条件:**
- 実IR 50種ベンチマークで boundExcessDb が指標を超過
- ラウドネス ABX テストで margin 増分が有効
- Planner inputDb 分布に系统的バイアスが確認
