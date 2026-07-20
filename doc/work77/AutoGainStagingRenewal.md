# ConvoPeq Auto Gain Staging 改修計画書 v14.47

> 最終更新: 2026-07-19 | 編集者: GitHub Copilot (MiMo V2.5)

本計画はレビューXVIIIの4項目に対応したv14.47版。

> 最終更新: 2026-07-19 | 編集者: GitHub Copilot (MiMo V2.5)

本計画はレビューXVIIの6項目に対応したv14.46版。

> 最終更新: 2026-07-19 | 編集者: GitHub Copilot (MiMo V2.5)

本計画はレビューXVIの10項目に対応したv14.45版。

> 最終更新: 2026-07-19 | 編集者: GitHub Copilot (MiMo V2.5)

本計画はレビューXVの5項目に対応したv14.44版。

> 最終更新: 2026-07-19 | 編集者: GitHub Copilot (MiMo V2.5)

本計画はレビューXIVの4項目に対応したv14.43版。

> 最終更新: 2026-07-19 | 編集者: GitHub Copilot (MiMo V2.5)

本計画はレビューXIIIの5項目に対応したv14.42版。

> 最終更新: 2026-07-19 | 編集者: GitHub Copilot (MiMo V2.5)

本計画はレビューXIIの10項目に対応したv14.41版。

> 最終更新: 2026-07-19 | 編集者: GitHub Copilot (MiMo V2.5)

本計画はレビューXIの3項目に対応したv14.40版。

> 最終更新: 2026-07-19 | 編集者: GitHub Copilot (MiMo V2.5)

本計画はレビューXの9項目に対応したv14.39版。

> 最終更新: 2026-07-19 | 編集者: GitHub Copilot (MiMo V2.5)

本計画はレビューIXの5項目に対応したv14.38版。

> 最終更新: 2026-07-19 | 編集者: GitHub Copilot (MiMo V2.5)

本計画はレビューVIIIの3項目に対応したv14.37版。

> 最終更新: 2026-07-19 | 編集者: GitHub Copilot (MiMo V2.5)

本計画はレビューVIIの9項目に対応したv14.36版。

> 最終更新: 2026-07-19 | 編集者: GitHub Copilot (MiMo V2.5)

本計画はレビューVIの6項目に対応したv14.35版。

> 最終更新: 2026-07-19 | 編集者: GitHub Copilot (MiMo V2.5)

本計画はバグ検証に基づく6件の修正設計を4.7節に追加したv14.34版。

> 最終更新: 2026-07-19 | 編集者: GitHub Copilot (MiMo V2.5)

本計画は外部レビューVの7項目に対応したv14.33版。

----------------------------------------------------------------------
## 設計
----------------------------------------------------------------------

### 1. 目的

- クリッピングを数学的に防止しつつ、不要な-12dBクランプを排除しラウドネスを維持
- 推定は常に安全側上界を理論的に保証する。Builder 側で `max(measured, upperBound)` により `eqMaxGainDb` を決定し、Planner はその値のみを使用する（ISR思想: Planner は解析方法を知らない）。過剰評価は2dB以内を **推定目標** とするが、これは数学的保証ではなく実測ベースの検証目標である（Appendix C.1 注釈参照）
- 全処理を純粋関数・封印可能・テスト可能に保つ

**プロ用基準:**
- True Peak -1dBTP以下を **推定目標** とし、最終保証は TruePeakDetector + 必要に応じた Limiter を含むシステム全体で行う
- Inter-stageで0dBFS超えなし
- net 0dB整合: `input + trim + makeup = 0` をクランプ時以外は厳密に保証
- 推定誤差 <0.5dB、推定は常に過大側（安全側）
- **安全側Boundは経験係数を使用せず数学的に証明可能とする**。経験的マージン（QSurge 等）は Bound と区別して明示し、その値の根拠と限界を文書化する

### 2. 現状課題サマリ

| ID | 箇所 | 問題 | 影響度 |
|---|---|---|---|
| E-1 | `computeEstimatedMaxGainDb` | LPF/HPFをブーストとしてカウント | P0 |
| E-2 | 同上 | totalGain負値0クランプ、host SR評価、order未使用 | P1 |
| E-3 | 同上 | 300点対数探索のみ、Parallelは振幅積で位相無視 | P1 |
| I-1 | `IRConverter` | `additionalAttenuationDb`は絞った量であり残余ブーストを表さない | P0 |
| P-1 | `AutoGainPlanner` | `QSurge = 1.5+gain*0.15*20/0.707`で常時6dB張り付き | P0 |
| P-2 | 同上 | Conv→EQ時 `min(input,-6)` で0dBでも-6dB強制 | P0 |
| P-3 | 同上/テスト | 実装とテストのQSurge条件不一致（テスト陳腐化）。実装は `eqMaxGainDb<=0→0` を返すが、テストは `kQSurgeHpfLpf=1.5` を返す | P1 |
| P-4 | 同上 | makeup 12dBクランプでnet 0dB崩れ | P2 |

### 3. 改修アーキテクチャ

現行の `BuildAnalysis → Planner → Builder上書き` は維持。変更点は中身の純粋関数のみ。

**新データフロー**

```
EQState → computeEstimatedMaxGainComplex() → eqMaxGainDb, eqMaxQ, eqComplexPeaks
IRState(final scaled IR) → IRAnalyzer::analyzeFinalIR() → irFreqPeakGainDb
                                    ↓
                    sealed BuildAnalysis { eqMaxGainDb, eqMaxQ, irFreqPeakGainDb }
                                    ↓
                    AutoGainPlannerV2::plan() → Plan { input, trim, makeup, diagnostics }
                                    ↓
                    RuntimeBuilder decibelsToGain + 封印
```

### 4. 詳細改修

#### 4.1 EQ最大ゲイン推定 V2 — `EQProcessor.Coefficients.cpp`

**4.1.1 対象バンド再定義**

```cpp
bool isBoosting(const Band& b) {
  if (!b.enabled) return false;
  switch (b.type) {
    case Peaking:   return b.gain > 0.01f;
    case LowShelf:
    case HighShelf: return b.gain > 0.01f;        // カットは除外
    case LowPass:
    case HighPass:  return false;                  // Q リゾナンスピークは別途 biquadComplex で検出（isBoosting ではブーストとみなさない）
  }
  return false; // 複合網羅（将来のバンド種別追加に対応）
}
```

**4.1.2 複素応答評価（★ v14.3: 経験係数0.95を廃止）**

```cpp
// ★ v14.30: std::complex<double> を使用（独自 ComplexResponse は廃止）
#include <complex>

using Complex = std::complex<double>;

// Biquad フィルタの複素周波数応答 H(e^{jω})
Complex biquadResponse(const Biquad& b, double normalizedFreq);

// Serial: ΠHi
Complex total{1.0, 0.0};
for (const auto& band : active)
    total *= biquadResponse(band.biquad, w);

// Parallel: 1 + Σ(Hi - 1)
Complex parallel{1.0, 0.0};
for (const auto& band : active)
    parallel += biquadResponse(band.biquad, w) - Complex{1.0, 0.0};

// ★ upperBound は複素和とは独立した別計算: Π(1 + |Hi - 1|)
//   各サンプリング点で全バンドの |Hi-1| を積算する（複素演算ではなく実数積）。
//   ★ v14.44: upperBound は「サンプリング近似」であり、真の上界（sup upperBound(f)）とは異なる。
//     粗探索600点＋適応サンプリングの評価点においてのみ upperBound が保証される。
//     真の全域上界を捉えきれない可能性があるため、Week2 の実IRベンチマークで
//     boundExcessDb 分布を検証する。
//   ★ v14.40: exp() を経由せず、dB 値を直接計算する。
//     Π(1+delta) の dB 値 = 20*log10(Π(1+delta)) = 20/log(10) * Σlog(1+delta)
//     これにより exp(>709) による Inf 発散を完全に防止する。
double logBound = 0.0;  // Σlog(1+delta) = Σlog1p(delta)
for (const auto& band : active) {
    const auto H = biquadResponse(band.biquad, w);
    const double delta = std::abs(H - Complex{1.0, 0.0});
    if (!std::isfinite(delta)) continue;  // ★ v14.44: NaN/Inf ガード
    if (delta > 1e-6)  // ★ v14.22: 微小項切り捨て（delta = abs(...) ≥ 0 は数学的条件として成立）
        logBound += std::log1p(delta);
}
// ★ v14.40: dB 値を直接計算（exp 回避）: 20*log10(Π(1+delta)) = (20/ln(10)) * Σln(1+delta)
constexpr double kTwentyOverLog10 = 20.0 / std::log(10.0);
double upperBoundDb = kTwentyOverLog10 * logBound;

// 最終値:
//   Serial: measured = |total| = |ΠHi|（厳密な振幅積、upperBound 不要）
//   Parallel:
//     measured = |parallel| = |1 + Σ(Hi - 1)|（複素ベクトルの振幅）
//     upperBound = upperBoundDb = (20/ln10) * Σln(1+|Hi-1|)（安全側上界、dB 直接計算）
//   Builder collapse: eqMaxGainDb = max(20*log10(measured), upperBoundDb)
//
// 【数学的証明】帰納法により無条件で成立:
//   |H_parallel| ≤ Π(1 + |Hi - 1|)  （三角不等式＋帰納法）
//
//   【補足】Π max(1, |Hi|) はより tight な bound だが、一般には証明できない
//   （反例: Hi=-2 → |Hi-1|=3, |Hi|=2, max(1,|Hi|)=2 → 3 > 2）。
//   数学的保証が必要な本書では Π(1 + |Hi - 1|) を採用する。
```

- `EQState::filterStructure` を参照して Serial/Parallel を分岐
- 関数シグネチャ: `(const EQState& state, double processingRate, int oversamplingFactor)`
- **measured と upperBound は別計算であり、独立したパイプラインで評価する**:
  1. 各サンプリング点で `Complex parallel`（複素和）と `double upperBoundMag`（実数積）を同時に更新
  2. measured = `|parallel|`（複素振幅）、upperBound = `upperBoundDb`（Π(1+|Hi-1|) の dB 値）
  3. 粗探索600点で両者を評価し、各々のピーク周辺バンドに適応サンプリングを適用
  4. 適応サンプリング後、**measured のみ** に対数周波数軸での放物線補間を適用
  5. **upperBound は補間せず、評価点の最大値をそのまま採用**（安全側保証を維持）
  6. 実装上は単一ループで両者を同時計算するため、探索コストは倍増しない
- ★ **実装の分割**: 本機能は責務過多を避けるため、`EQProcessor` 内で直接実装せず、**4.1.7 節**で定義する 3 層（`EQResponseSampler` / `PeakEstimator` / `UpperBoundEstimator`）に分割して実装する。これにより FFT ベースの探索への将来の差し替えが容易になる。
- ★ **診断値補完**: measured は放物線補間後の値、upperBound は補間前の評価点最大値である。両者の分解能の差を診断するため、必要に応じて `BuildDiagnostics` に補間前の measured 生値を `eqMeasuredRawGainDb` として保存してもよい（Week2 検討事項）。

**upperBound の保守性モニタリング**:
- `Π(1+|Hi-1|)` は離散サンプリング近似に対する安全側保証を持つ。各評価点では upperBound が |H_parallel| 以上であることが保証されるが、サンプリング点間の真の上界（sup upperBound(f)）は別途評価が必要である。複数のブーストバンドが重なる Parallel 構成では実測より過大になり得る。
- Diagnostics の `boundExcessDb` が 3dB を超える場合は、当該構成を記録し、実IRベンチマーク（Week2）で過剰評価量を評価する。**Week2 の実IRベンチマークでは boundExcessDb の分布（平均・95% tile・最大）を必ず測定し、upperBound の実用上の過大評価量を定量化すること。**
- 過度な保守性が確認された場合、将来の改訂で `Π max(1, |Hi|)` 等のより tight な bound への移行を検討する（数学的保証は失われるが、数値検証で安全性を確認する）。

**4.1.3 探索の高精度化（★ v14.3: ternary search 廃止）**

- 粗探索: 600点対数分布 10Hz〜20kHz（v14.2: 300点→600点に倍増）
- バンド適応: 各バンド中心 ±2oct を 128点（v14.1: 64点）
  粗探索時、各バンドの複素応答 `|H(f)|` と `|Hi-1|` を記録しておく。**適応サンプリングの対象バンドは、measured と upperBound で独立に判定する**:
  - **measured 用**: `isBoosting()==true`（振幅増大に寄与するバンド）。`isBoosting()` は Peaking(gain>0)/LowShelf(gain>0)/HighShelf(gain>0) を捕捉する。
  - **upperBound 用**: 粗探索600点において各バンド単体の `|Hi-1|` 最大値を記録し、`max(|Hi-1|) > 0.1`（線形振幅差 0.1、約 +0.83dB 相当）のバンドを対象とする。`|Hi-1|` は Cut band（例: H=0.25→|H-1|=0.75）でも大きくなるため、measured とは異なる候補セットになる。この方式は中心周波数依存を完全に排除し、Shelf の極端なケースでも漏れなく捕捉できる。
  ★ 実装には `evaluateBandDelta(band, freq)` のような共通ヘルパー関数を使用し、粗探索・Shelf追加評価で同じロジックを再利用する。これにより `max(|Hi-1|)` の計算と候補Band判定の重複実装を防ぐ。
  - **Shelf フィルタの注意**: LowShelf のピークは DC 近傍、HighShelf のピークは Nyquist 近傍にあるため、center周波数だけでは捕捉できない。そこで Shelf バンドについては以下の評価点で上記基準を評価し、いずれかが条件を満たせば適応サンプリング対象とする:
    - **LowShelf**: `20Hz`, `center`, `center×2`（DC 近傍〜遷移域をカバー。center×2 は Shelf の裾野を確認）
    - **HighShelf**: `center/2`, `center`, `Nyquist×0.95`（Nyquist 近傍までカバー。center/2 は Shelf の裾野を確認）
  - `maxActiveQ` の算出は measured 用基準（`isBoosting()==true`）を使用する。これにより、「候補Bandに含まれないが maxActiveQ には含まれる」という不整合を防止する。
  - **maxActiveQ と totalMaxQ の使い分け**: `maxActiveQ`（`EQAnalysisResult`）は Planner が safety margin 計算に使用するブーストバンド限定の最大Q。`totalMaxQ`（`BuildDiagnostics`）は全有効バンドの最大Qで診断専用。両者は目的が異なる。
  - **Band 種別ごとの適応サンプリング範囲**: Peak バンドは center ±2oct で十分だが、Shelf/LPF/HPF はピーク位置が center と異なる。以下の範囲を使用する:
    - **Peak**: center ±2oct
    - **LowShelf**: 10Hz〜center×2（DC 近傍のピークをカバー。center=20Hz でも 10Hz〜40Hz を確保）
    - **HighShelf**: center/2〜Nyquist（Nyquist 近傍のピークをカバー）
    - **LowPass**: Q>0.707 の場合のみ fc〜Nyquist の ±1oct 範囲（RBJ LPF は Q=0.707 でも微小オーバーシュート）
    - **HighPass**: Q>0.707 の場合のみ 10Hz〜fc の ±1oct 範囲（同上）
  ★ 適応サンプリング範囲が複数バンドで重複する場合、以下の手順で union を生成する:
    1. 各区間 $[start_i, end_i]$ を start でソート
    2. 重複区間をマージ（$end_i \\geq start_{i+1}$ なら統合）
    3. 全区間長の合計 totalLength を計算
    4. 各区間に $128 \\times length_i / totalLength$ 点を比例配分
    5. 各区間内で対数等間隔にサンプリング点を生成
  これにより同一周波数の重複評価を排除し、CPU 負荷を低減する。短い区間に過剰な点数が割かれることもない。
  20Bandフル構成でも最大 600 + (候補Band数)×128 ≤ 約3160点（全バンドが候補の場合）に抑制される。実測上の平均は union 統合により 600〜1200点程度と想定。
- ★ **ternary search を廃止**: EQ応答は多峰性（Shelf + Peak × N）であり、ternary search の前提（unimodal）が成立しない。代わりに **適応サンプリング後の局所最大点に対して放物線補間（parabolic interpolation）** を行う。
- 放物線補間は以下のパイプラインで適用する:
  1. 粗探索600点で大域的なピーク周波数を特定
  2. 候補バンド（measured: `isBoosting()==true`, upperBound: `max|Hi-1|>0.1`）に適応サンプリング128点を実行
  3. 適応サンプリング後の全評価点（粗探索600点＋適応128点×候補Band数）を**周波数順にソート**し、最大値を与える周辺3点 $(f_{k-1}, y_{k-1}), (f_k, y_k), (f_{k+1}, y_{k+1})$ を抽出
  4. 探索点は対数分布（10Hz〜20kHz を 600点で対数分割）であるため、**対数周波数軸で放物線補間を実施する**。$x_k = \log_2(f_k)$ と変換した上で、**振幅は dB 空間（$y_k = 20\log_{10}|H(f_k)|$）** で補間する。
  ★ 適応サンプリング後は評価点が不等間隔になるため、等間隔用の簡略式ではなく **Lagrange 二次補間（一般3点）** を使用する:
  $$x_{peak} = \frac{1}{2} \cdot \frac{y_{k-1}(x_k^2 - x_{k+1}^2) + y_k(x_{k+1}^2 - x_{k-1}^2) + y_{k+1}(x_{k-1}^2 - x_k^2)}{y_{k-1}(x_k - x_{k+1}) + y_k(x_{k+1} - x_{k-1}) + y_{k+1}(x_{k-1} - x_k)}, \quad f_{peak} = 2^{x_{peak}}$$
  等間隔（$x_k - x_{k-1} = x_{k+1} - x_k$）の場合は従来の簡略式に一致する。
  ★ 補間対象の3点が存在しない境界ケース（$k=0$ または $k=N-1$）では補間を実施せず、最大評価点の値をそのまま採用する。
  ★ 分母 $D = y_{k-1} - 2y_k + y_{k+1}$ が非常に小さい場合、丸め誤差により $\Delta$ が発散する。実装では以下のガードを推奨:
  ```cpp
  const double denom = y[k-1] - 2.0 * y[k] + y[k+1];
  if (std::abs(denom) < 1e-12) { /* 補間をスキップ、評価点 y[k] を採用 */ }
  ```
  5. **upperBound には放物線補間を適用しない**: upperBound は安全側評価であり、補間による数値的な不確実性を避けるため、評価点の中の最大値をそのまま採用する。補間は measured のみに適用する。
- 処理レート: `processingRate = sr * resolvedOsFactor`
  - `resolvedOsFactor` は `OversamplingPolicy::resolve(task.buildInput)` から取得
  - `0=Auto` 時は OversamplingPolicy が倍率を決定する（後述 4.5）
  - **上限**: `processingRate ≤ OversamplingPolicy::kMaxInternalRate = 768kHz`
  - 倍率一覧: 44.1k-96k=x8, 176.4k-192k=x4, 352.8k-384k=x2, 705.6k-768k=x1（ルックアップ方式。中間値なし）
  - ★ `processingRate` の生成は呼出し側（Builder）の責務。`computeEstimatedMaxGainComplex()` は受け取った `processingRate` をそのまま使用し、OversamplingPolicy や倍率の決定ロジックを一切知らない。
- ★ **探索周波数範囲**: 粗探索は **10Hz〜min(20kHz, Nyquist)** の対数600点。Nyquist = `processingRate / 2`。768kHz 処理時でも Nyquist=384kHz となるため、20kHz 上限が有効。低 SR（44.1kHz）時は Nyquist=22.05kHz となる。Shelf の適応サンプリングにおける `Nyquist×0.95` はこの値を参照する。

**4.1.4 totalGain**

```cpp
// クランプ撤廃。Planner側で max(0, ...) する
return 20.0f * std::log10(maxLinear * totalGainLin);
```

**4.1.5 戻り値 — EQAnalysisResult 構造体（★ v14.7: 二層構造化）**

```cpp
struct SampleOrigin {
    enum Type : uint8_t { Unknown = 0, Coarse = 1, Adaptive = 2, Union = 3 };  // ★ v14.46: Union 追加（区間統合後）
    Type type = Unknown;
    int bandIndex = -1;        // ★ v14.39: バンドインデックス。Union 型の場合は -1（統合後は特定不能）
    int sampleIndex = -1;      // ★ v14.42: 配列内インデックス（Coarse=0..599, Adaptive/Union=0..N-1）
};

struct PeakInfo {
    float gainDb = 0.0f;      // ゲイン（dB）
    float freqHz = 0.0f;      // 当該ゲインが現れる周波数
    SampleOrigin origin;       // ★ v14.35: 評価点の origin（粗探索/適応サンプリングの区別）。デバッグ用
};

struct EQAnalysisResult {
    PeakInfo measured;         // 実測最大ピーク（粗探索＋放物線補間の最大値）
    float measuredRawGainDb = 0.0f;  // ★ v14.47: 放物線補間前の measured 生値（dB）。boundExcessDb 評価時に補間影響を分離するため
    PeakInfo upperBound;       // 安全側上界の最大値（Π(1+|Hi-1|) の dB 値と、その最大値を与える周波数）
                               //   ※ upperBound.freqHz は上界が最大となる周波数であり、
                               //      measured.freqHz とは異なる場合がある。
    float maxActiveQ = 0.0f;   // ブースト対象バンド（isBoosting()==true）中の最大Q値。
                               //   LPF/HPF の Q は含まない（振幅増大に寄与しないため）。
                               //   Planner は「ブーストバンド中の最悪Q」として safety margin 計算に使用する。
    EqGainAlgorithm algorithm = EqGainAlgorithm::TriangleProductV1;  // ★ v14.39: enum 化。BuildDiagnostics.eqGainAlgorithm にコピーされる。
};
```

**upperBound の利用方針（★ v14.10: 案B — Builder 側で collapse）**:
- ISR 思想に基づき、Builder が `eqMaxGainDb = max(measured.gainDb, upperBound.gainDb)` を計算し、Planner はこの値のみを受け取る
- Planner は `measured` と `upperBound` の区別を知らない（解析方法への依存を排除）
- `upperBound` は診断ログとしても記録され、Parallel 構成での過大評価量の確認に使用可能

二層構造により以下が可能:
- 将来 Diagnostic で `measured.freqHz` や `upperBound.gainDb` を拡張可能
- BuildAnalysis へのコピーも単純

関数シグネチャ変更:
```cpp
// 変更前
float computeEstimatedMaxGainDb(double sampleRate, int processingOrder) const;
// 変更後
EQAnalysisResult computeEstimatedMaxGainComplex(
    const EQState& state, double processingRate) const;  // ★ v14.38: processingRate のみ。EQProcessor は OversamplingPolicy を知らない
```

#### 4.2 IRゲイン指標 V2 — `IRAnalyzer.h / IRConverter.cpp`（変更なし）

```cpp
struct IRFinalAnalysis {
  double freqPeakGainLin = 1.0;
  double freqPeakGainDb  = 0.0;
  double l1NormDb         = 0.0;
  double peakDb           = 0.0;
  double rmsDb            = 0.0;
};
```

**実装手順:**

1. `IRConverter::computeScaleFactor` → `applyClampProtection` 直後で scaledIR を生成
2. `IRAnalyzer::estimateMaxFrequencyResponseGain(scaledIR)` を呼び出し
3. `convertFile()` 内で `prepared->irFreqPeakGainDb = finalAnalysis.freqPeakGainDb` を設定
4. `ConvolverProcessor` に `getIrFreqPeakGainDb()` を追加

#### 4.3 AutoGainPlanner V2 — `AutoGainPlanner.h` 完全書き換え

```cpp
// ★ v14.3: kConvFirstForced を廃止。固定Ceilingは一切使用しない。
//   マージンだけで安全側を担保する。

// 定数
kMarginEqFirst    = 1.5f   // 3.0→1.5
kMarginConvFirst  = 1.0f   // 1.5→1.0
kMarginInterStage = 1.0f   // 2.0→1.0
// ★ ★ 定数（constexpr で型と初期化を明示）
inline constexpr float kMarginEqFirst      = 1.5f;   // 3.0→1.5
inline constexpr float kMarginConvFirst    = 1.0f;   // 1.5→1.0
inline constexpr float kMarginInterStage   = 1.0f;   // 2.0→1.0
inline constexpr float kSafetyMarginBase   = 0.8f;
inline constexpr float kSafetyMarginCoeffQ = 0.12f;
inline constexpr float kSafetyMarginCoeffGain = 0.04f;
inline constexpr float kSafetyMarginMax    = 2.5f;   // 6.0→2.5
inline constexpr float kClampInputMin      = -18.0f; // -12→-18
inline constexpr float kClampInputMax      = 0.0f;
inline constexpr float kClampTrimMin       = -12.0f;
inline constexpr float kClampTrimMax       = 0.0f;
inline constexpr float kClampMakeupMin     = 0.0f;
inline constexpr float kClampMakeupMax     = 12.0f;
// kConvFirstInputCeiling は削除（固定Ceiling廃止）

// ★ v14.14: PlannerInput — Planner 専用 DTO。物理的に Diagnostics へアクセス不可能。
//   ★ v14.39: immutable DTO（Builder collapse 後の値のみを保持し、生成後は変更不可）。
//   ISR 思想: DTO を介して Planner と Builder を完全分離。Planner は解析アルゴリズムを知らない。
struct PlannerInput {
    float eqMaxGainDb = 0.0f;          // Builder collapse 後の安全側値
    float eqMaxQ = 0.0f;               // 最大Q値
    float irFreqPeakGainDb = 0.0f;     // IR 周波数ピークゲイン
};

// Planner Contract:
// - eqMaxGainDb は Builder により max(measured, upperBound) で安全側保証済み
// - Planner は PlannerInput のみを受け取り、BuildAnalysis.Diagnostics を参照不可能
// - 責務は「与えられた入力からマージンを計算し、4パターン分岐すること」のみ

// EmpiricalSafetyMarginPolicy — 経験的安全マージン（旧称 QSurge）。Bound ではなく経験式。
// ISR 思想に基づき Policy として分離。Builder/Planner/Test で共有。
struct EmpiricalSafetyMarginPolicy {
    static constexpr float kBase         = kSafetyMarginBase;
    static constexpr float kCoeffQ       = kSafetyMarginCoeffQ;
    static constexpr float kCoeffGain    = kSafetyMarginCoeffGain;
    static constexpr float kMax          = kSafetyMarginMax;
    static constexpr float kButterworthQ = 0.707f;
    static constexpr float kMinimumBoostForMargin = 0.5f;

    [[nodiscard]] static float evaluate(float eqGainDb, float maxQ) noexcept {
        if (eqGainDb <= kMinimumBoostForMargin) return 0.0f;
        const float qTerm = std::max(0.0f, (maxQ - kButterworthQ) * kCoeffQ);  // ★ v14.31: Q < 0.707 で負値になるのを防止
        const float gTerm = eqGainDb * kCoeffGain;
        return std::min(kMax, std::max(0.0f, kBase + qTerm + gTerm));  // ★ v14.31: 結果も 0 以上にクランプ
    }
};

// 使用例: Planner は最大値保証された eqMaxGainDb のみを受け取る
const float safetyMargin = EmpiricalSafetyMarginPolicy::evaluate(eqMaxGainDb, eqMaxQ);
// ★ v14.36: 経経験式の係数（0.8/0.12/0.04/2.5）は Week2 の実IRベンチマークで再キャリブレーション予定。
//   現在は暫定値であり、完成品ではない。
//   → Week2 ベンチマーク完了（126合成構成 + 実IR 22種）。
//   → 結論: 現行係数を維持。詳細は empirical-safety-margin-recalibration-v1.md を参照。
//   → Week3 で marginErrorDb = truePeak - selectedEstimate の分布を測定し、最終確認。



// 4パターンロジック（★ v14.3: kConvFirstForced 削除）
// ★ v14.10: eqMaxGainDb は Builder 側で max(measured, upperBound) 済み
const float eqBoost   = std::max(0.0f, eqMaxGainDb);
const float convBoost = std::max(0.0f, irFreqPeakGainDb);

float inputDb = 0.0f, trimDb = 0.0f;

if (convBypassed) {  // PEQ only
    inputDb = -(eqBoost - kMarginEqFirst) - safetyMargin;
} else if (eqBypassed) {  // Conv only
    inputDb = -(convBoost - kMarginConvFirst);
} else if (processingOrder == ConvolverThenEQ) {
    // Conv→PEQ: 固定Ceiling廃止、マージンのみで保護
    // ★ v14.30: EQ側にも safetyMargin を適用（Q safety margin は EQ そのものの危険度であり順序に依存しない）
    // ★ v14.31: convBoost と eqBoost の単純加算は保守的な推定であり、
    //   ピーク帯域が一致しない場合に過大評価となる。現状は安全側として許容するが、
    //   将来は IR×EQ の合成解析（周波数領域での積算）へ発展させる余地がある。
    inputDb = -(convBoost - kMarginConvFirst) - (eqBoost - kMarginInterStage) - safetyMargin;
} else {  // EQThenConv
    inputDb = -(eqBoost - kMarginEqFirst) - safetyMargin;
    trimDb  = -(convBoost - kMarginInterStage);
}

inputDb  = juce::jlimit(kClampInputMin, kClampInputMax, inputDb);
trimDb   = juce::jlimit(kClampTrimMin, kClampTrimMax, trimDb);
float makeupDb = -(inputDb + trimDb);
if (makeupDb > kClampMakeupMax) {
    makeupDb = kClampMakeupMax;
    // DiagEvent::AutoGainClamped 発行（データ構造は 4.6 節参照）
}
```

#### 4.4 統合・スレッド

**Oversampling の権限一元化（★ v14.10: naming 明確化 + 依存関係の明示）**

ISR の Authority Singularization 思想に基づき、オーバーサンプリング倍率の決定権限は以下のパイプラインで一箇所に定める:

```
BuildInput.requestedOversamplingFactor  (0=Auto)
        ↓
OversamplingPolicy::resolve()           ← 唯一の決定権限（Authority）
        ↓
OversamplingResult.resolvedOsFactor    ← 決定結果の保持（Snapshot 内）
        ↓                            ↘
BuildAnalysis (解析結果)               RuntimeBuildSnapshot (Builder 参照)
    ↓ 注: Analysis は OversamplingResult
    ↓     を入力として生成される
```

`BuildAnalysis` は純粋な解析結果のみを保持し、`OversamplingResult` は **Snapshot 側に保持** する。これにより:
- Analysis と Result の責務が完全分離（ISR 原則）
- `BuildAnalysis` の sealed 契約（解析結果の不変性）が純粋に保たれる
- Builder は `Snapshot.oversampling.resolvedOsFactor` を参照する（独自決定しない）
- 決定ロジックは `OversamplingPolicy::resolve()` 一箇所（Authority Singularization）

```cpp
// RuntimeBuildTypes.h に追加
struct OversamplingResult {
    int resolvedOsFactor = 1;      // 解決済み倍率（Builder 専有。Planner は Snapshot から読み取り専用参照）
    int requestedOsFactor = 0;     // BuildInput からの要求値（0=Auto）
    bool isAutoResolved = false;   // Auto(0) からの解決済みか
    // ★ v14.37: supported は Capability（入力 SR が処理可能か）を表し、
    //   resolvedOsFactor は Configuration（実際に使用する倍率）を表す。
    //   両者は独立した概念であり、isValid() は resolvedOsFactor の値域のみ検証する。
    //   ★ v14.41: supported==false のとき Builder は Publish を行わないことが契約。
    //   （verifyBuildBundle() 通過後も Builder が supported を確認し、Publish を抑制する）
    bool supported = true;         // Capability: 入力 SR がサポート範囲内か

    // ★ v14.37: resolvedOsFactor ∈ {1,2,4,8} を検証（supported とは独立）。
    [[nodiscard]] bool isValid() const noexcept {
        switch (resolvedOsFactor) {
            case 1: case 2: case 4: case 8: return true;
            default: return false;
        }
    }
};
```

- `RuntimeBuildSnapshot` sealed 契約維持。`verifyBuildBundle()` により一元検証
- `convolverInputTrimGain` 適用は現状通り EQThenConv のみ
- `BuildAnalysis` 作成箇所（`RebuildDispatch.cpp:655`）:
  ```cpp
  // 変更前
  analysis.eqMaxGainDb = getEQProcessor().computeEstimatedMaxGainDb(sampleRate, order);
  // 変更後（★ v14.12: Builder 側で collapse + Diagnostics 保持）
  const auto eqResult = getEQProcessor().computeEstimatedMaxGainComplex(
      state, processingRate);  // ★ v14.38: oversamplingFactor 引数削除。EQProcessor は OversamplingPolicy を知らない。processingRate のみ渡す。
  diagnostics.eqMeasuredGainDb = eqResult.measured.gainDb;              // 診断用（補間後）
  diagnostics.eqMeasuredRawGainDb = eqResult.measuredRawGainDb;         // ★ v14.47: 診断用（補間前生値）
  diagnostics.eqUpperBoundGainDb = eqResult.upperBound.gainDb;          // 診断用
  diagnostics.boundMethod = BoundMethod::TriangleProduct;
  diagnostics.boundExcessDb = std::max(0.0f,
      eqResult.upperBound.gainDb - eqResult.measured.gainDb);
  diagnostics.eqGainAlgorithm = eqResult.algorithm;                     // provenance
  // ★ v14.42: Builder collapse で採用された推定値を記録
  diagnostics.selectedEstimate = (eqResult.measured.gainDb >= eqResult.upperBound.gainDb)
      ? SelectedEstimate::Measured : SelectedEstimate::UpperBound;
  jassert(diagnostics.boundExcessDb >= 0.0f);  // Bound 不変条件（負値は std::max でガード済み）
  analysis.eqMaxGainDb = std::max(eqResult.measured.gainDb, eqResult.upperBound.gainDb);
  analysis.eqMaxQ = eqResult.maxActiveQ;                                // ★ v14.35: ブースト対象バンド中の最大Q値。Planner は安全側指標として使用
  // ★ v14.38: resolvedOsFactor の Authority は OversamplingResult（Snapshot）が唯一。
  //   Diagnostics にはコピーせず、必要時は OversamplingResult を参照する。
  ```

BuildAnalysis は `OversamplingResult` に依存して生成される。この依存関係は ISR の Build → Validate → Publish パイプラインの「Build」フェーズに該当する。

#### 4.5 解析用オーバーサンプリング倍率決定 — OversamplingPolicy（★ v14.27: ルックアップ方式に変更）

```cpp
// ★ v14.27: OversamplingPolicy — Builder 専有の決定ポリシー。
// Planner は決定ロジックを一切知らず、Snapshot.oversampling.resolvedOsFactor を読み取り専用で参照。
//
// ★ 入力 SR と許可倍率（ルックアップ方式）:
//   入力SR        許可倍率
//   44.1kHz       x1, x2, x4, x8
//   48.0kHz       x1, x2, x4, x8
//   88.2kHz       x1, x2, x4, x8
//   96.0kHz       x1, x2, x4, x8
//   176.4kHz      x1, x2, x4
//   192.0kHz      x1, x2, x4
//   352.8kHz      x1, x2
//   384.0kHz      x1, x2
//   705.6kHz      x1
//   768.0kHz      x1
//   > 768kHz      入力不可（Publish スキップ）
//
// ★ Auto 時の決定論理:
//   1. 入力 SR が 768kHz 超 → supported=false（Publish スキップ。解析も行わない）
//   2. 入力 SR から最大許可倍率をルックアップ
//   3. requestedOsFactor > 0 なら、requestedOsFactor ≤ 最大許可倍率 であることを検証
//   4. 不整合時は最大許可倍率を使用（安全側フォールバック）
//   5. resolvedOsFactor は常に power-of-2（1, 2, 4, 8 のみ）
//
// 最大許可倍率の決定（SR→maxFactor ルックアップ）:
//   sr ≤ 96000   → maxFactor = 8   （96k x8 = 768k ≤ 768k）
//   sr ≤ 192000  → maxFactor = 4   （192k x4 = 768k）
//   sr ≤ 384000  → maxFactor = 2   （384k x2 = 768k）
//   sr ≤ 768000  → maxFactor = 1   （768k x1 = 768k）
//   sr > 768000  → maxFactor = 0   （入力不可）
//
// ISR 設計: 純粋関数であり DSPCore の状態に依存しない。
struct OversamplingPolicy {
    static constexpr double kMaxInternalRate = 768000.0;
    static constexpr int kMaxFactor = 8;

    // ★ v14.30: maxAllowedFactor は resolve() の内部実装。外部から直接呼ばない。
    // ★ v14.45: maxAllowedFactor() — 指定 SR における最大許可倍率を返す。
    //   resolve() も同一のルックアップテーブルを利用する公開ヘルパー。
    //   GUI などが参照可能で、決定権限は持たない（Authority は resolve()）。
    //   名前の通り「許可される最大倍率」を返し、集合（{1,2,4,8}等）は返さない。
    [[nodiscard]] static int maxAllowedFactor(double sampleRate) noexcept {
        if (sampleRate <= 96000.0)   return 8;
        if (sampleRate <= 192000.0)  return 4;
        if (sampleRate <= 384000.0)  return 2;
        if (sampleRate <= 768000.0)  return 1;
        return 0;  // 768kHz 超: 許可倍率なし（supported==false）
    }

    [[nodiscard]] static OversamplingResult resolve(const BuildInput& input) noexcept {
        OversamplingResult result{};
        result.requestedOsFactor = input.oversamplingFactor;

        const int maxF = maxAllowedFactor(input.sampleRate);
        result.supported = (maxF > 0);

        if (maxF == 0) {
            result.resolvedOsFactor = 1;  // 最小倍率（supported==false。resolvedOsFactor は有効値のまま）
            result.isAutoResolved = true;
            return result;
        }

        result.isAutoResolved = (input.oversamplingFactor == 0);
        // ★ v14.33: requestedOsFactor が {0,1,2,4,8} 以外の異常値の場合、maxF へフォールバック
        //   （例: requested=3 → isValid() を待たずに resolve() 内で解決）
        int effectiveRequested = input.oversamplingFactor;
        if (effectiveRequested != 0 && effectiveRequested != 1 && effectiveRequested != 2
            && effectiveRequested != 4 && effectiveRequested != 8)
            effectiveRequested = 0;  // 異常値 → Auto 扱い

        if (effectiveRequested > 0)
            result.resolvedOsFactor = (effectiveRequested <= maxF) ? effectiveRequested : maxF;
        else
            result.resolvedOsFactor = maxF;  // Auto: 最大許可倍率
        return result;
    }
};
```

**resolve() の戻り値設計**:
- `resolve()` は `OversamplingResult` 構造体を返す。`resolvedOsFactor` は常に {1,2,4,8} のいずれかであり、`isValid()` と矛盾しない。
- 768kHz 超の入力は `result.supported == false` で通知される。**supported==false の処理フロー**:
  1. `OversamplingPolicy::resolve()` が `supported=false, resolvedOsFactor=1` を返す
  2. **Builder（DSPCore::prepare()）** が `supported` を検査し、**Publish をスキップする**（直前の有効な Runtime を維持）
  3. 結果として DSPCore は旧 Runtime を使い続けるため、**無音にはならない**
  4. **★ v14.47: Publish スキップ時も Diagnostics は更新される**（Builder は新しい BuildDiagnostics を生成し、Timer 経由で UI が確認可能）。ただし Runtime Generation は更新されないため、Diagnostics の generation（存在すれば）は Runtime の generation と一致しない可能性がある。
  5. `verifyBuildBundle()` は `supported` を検証しない（Publish 可否の判断は Builder の責務）
- **Authority Singularization**: `resolve()` が OS 倍率決定における唯一の決定権限である。DSPCore や Builder は `resolve()` のみを呼び、内部のルックアップテーブルを直接参照しない。

この Policy の導入により以下が保証される:
- Builder が唯一の決定権限を持ち、Planner は Snapshot の結果を読み取り専用で参照する
- 入力 SR ごとの許可倍率がルックアップテーブルで明示的に定義される
- `resolvedOsFactor` は常に power-of-2（1, 2, 4, 8）であり、中間値（x7, x6 等）は発生しない
- 768kHz 超の入力は `supported=false` で通知される。**supported==false は Publish スキップ条件であり、Validation エラーではない。**
  verifyBuildBundle() は supported をチェックせず、Builder（DSPCore::prepare()）が `supported==false` を検出した場合、
  // ★ v14.39: supported は Publish Policy（Builder の判断）が扱うため、verifyBuildBundle() の検証対象外。
  //   verifyBuildBundle() は Snapshot の構造的整合性のみを検証する。
  RuntimePublishWorld の生成をスキップし、直前の有効な Runtime を維持する。これにより ISR Publish パイプライン全体を
  中断することなく安全に処理できる。
- 新しいサンプルレートが追加された場合も `maxAllowedFactor()` のみ更新

#### 4.5.1 GUI / DSPCore への OversamplingPolicy 統合（★ v14.28: 新規追加）

OversamplingPolicy は解析専用ではなく、GUI の倍率選択肢表示と DSPCore の Auto 解決ロジックにも適用する。3箇所の改修が必要。

**① GUI — DeviceSettings.cpp: oversamplingComboBox 表示条件**

```cpp
// 変更前（現行）
if (sr <= 192000)
    oversamplingComboBox.addItem("4x", 4);
if (sr <= 96000)
    oversamplingComboBox.addItem("8x", 5);

// 変更後（★ v14.32: maxAllowedFactor() を GUI 専用 API として使用）
// GUI は BuildInput を知る必要なく、maxAllowedFactor(sampleRate) で最大倍率を取得する。
const int maxAllowed = convo::OversamplingPolicy::maxAllowedFactor(sr);
if (maxAllowed >= 4)
    oversamplingComboBox.addItem("4x", 4);
if (maxAllowed >= 8)
    oversamplingComboBox.addItem("8x", 5);
```

**根拠**: `maxAllowedFactor()` は GUI 専用の軽量参照 API。`resolve()` が唯一の Authority であり、`maxAllowedFactor()` はその内部ルックアップを間接利用する。GUI は `BuildInput` を知る必要がない。

**② DSPCore — AudioEngine.Processing.DSPCoreLifecycle.cpp: Auto 解決 + maxFactor**

```cpp
// 変更前（現行）
int targetFactor = 1;
if (manualOversamplingFactor > 0)
    targetFactor = manualOversamplingFactor;
else
{
    if (newSampleRate >= 705600)      targetFactor = 1;
    else if (newSampleRate >= 352800) targetFactor = 2;
    else if (newSampleRate >= 176400) targetFactor = 4;
    else if (newSampleRate >= 88200)  targetFactor = 8;
    else                              targetFactor = 8;
}

int maxFactor = 1;
if (newSampleRate <= 96000.0)       maxFactor = 8;
else if (newSampleRate <= 192000.0) maxFactor = 4;
else if (newSampleRate <= 384000.0) maxFactor = 2;
targetFactor = std::min(targetFactor, maxFactor);

// 変更後（★ v14.30: OversamplingPolicy::resolve() を唯一の Authority として使用）
{
    convo::BuildInput buildInput{};
    buildInput.sampleRate = newSampleRate;
    buildInput.oversamplingFactor = manualOversamplingFactor;
    const auto osResult = convo::OversamplingPolicy::resolve(buildInput);

    if (!osResult.supported) {
        targetFactor = 0;  // 768kHz 超: Publish スキップ（旧 Runtime 維持）
    } else if (manualOversamplingFactor > 0) {
        targetFactor = osResult.resolvedOsFactor;  // resolve() が clamp 済み
    } else {
        targetFactor = osResult.resolvedOsFactor;  // Auto: resolve() が決定
    }
}

**改善点**:
- DSPCore が `resolve()` を唯一の Authority として使用。`maxAllowedFactor()` は内部実装に隔離
- 768kHz 超の入力不可処理は `osResult.supported` で判定
- 手動指定の clamp は `resolve()` が内部で実行するため、呼出し側で `std::min` 不要

**③ AudioEngine::setOversamplingFactor() — 構造変更なし（GUI 側で制御）**

```cpp
// 現行コードを維持。理由:
// - GUI が maxAllowedFactor に基づいて表示する倍率を制限しているため、
//   setOversamplingFactor に渡される値は既に有効な範囲内
// - DSPCore::prepare() の maxFactor clamp が最終安全網
// - setOversamplingFactor 自体は {0,1,2,4,8} のみ許容する構造は維持
void AudioEngine::setOversamplingFactor(int factor)
{
    int newFactor = 0;
    if (factor == 1 || factor == 2 || factor == 4 || factor == 8)
        newFactor = factor;
    // ... 以下変更なし
}
```

**防御の階層（★ v14.30: resolve() を唯一の Authority に統合）**:

```
GUI（DeviceSettings.cpp）
  └─ resolve().maxAllowedFactor に基づき表示倍率を制限（内部関数を間接利用）
     └─ ユーザーが選択できるのは許可倍率のみ

Preset Restore（Parameters.cpp:126）
  └─ oversamplingFactor の読み込み時に range 検証
     └─ hasValidRange: factor ∈ {0,1,2,4,8} を確認
        └─ 範囲外は安全値 0（Auto）でフォールバック

Host Automation（setOversamplingFactor）
  └─ setOversamplingFactor({0,1,2,4,8} のみ許容)
     └─ DSPCore::prepare() で resolve() により再検証

Builder（RebuildDispatch / OversamplingPolicy::resolve()）
  └─ resolve() が唯一の決定権限（Authority Singularization）
     └─ 内部の maxAllowedFactor により許可範囲を強制
        └─ result.supported で768kHz超の入力不可を通知

State Restore（AudioEngine::setOversamplingFactor）
  └─ {0,1,2,4,8} のみ許容
     └─ DSPCore::prepare() で resolve() により最終検証
```

**非GUI経路の安全性**: Preset Restore / Host Automation / State Restore はいずれも GUI を経由しないが、以下の三重防御で保護される:
1. `setOversamplingFactor()` の値域フィルタ（{0,1,2,4,8} のみ）
2. `DSPCore::prepare()` での `resolve()` 呼び出し（唯一の Authority）
3. `resolve()` 内部の `maxAllowedFactor` による許可範囲強制

#### 4.6 計測・ログ

**BuildAnalysis:**

```cpp
// ★ v14.13: BoundMethod を型安全な enum に変更
enum class BoundMethod : uint8_t {
    Unknown = 0,                  // 未初期化
    Legacy = 1,                   // 旧形式セッション
    TriangleProduct = 2,          // Π(1+|Hi-1|) — 現在のアルゴリズム（v14.22以降）
    ProductMaxMagnitude = 3,      // Π max(1,|Hi|) — 将来の候補（未実装）
    ExactSampling = 4             // 適応サンプリング直接 — 将来の候補（未実装）
};

struct BuildAnalysis {
    int generation = 0;
    float eqMaxGainDb = 0.0f;               // Builder collapse 後の安全側値
    float eqMaxQ = 0.0f;                    // ブースト対象バンド中の最大Q値（Planner 使用）
    float irFreqPeakGainDb = 0.0f;          // 新規
    float irAdditionalAttenuationDb = 0.0f; // 互換、常に0
    bool sealed = false;
};

// ★ v14.37: BuildDiagnostics — BuildAnalysis から完全分離された診断情報。
//   ISR 思想: Analysis は Publish 対象（Runtime World に写像される）、
//   Diagnostics は Debug 対象（Runtime World とは別世界）。

// ★ v14.39: EqGainAlgorithm — 解析アルゴリズム識別子（enum 化）。
enum class EqGainAlgorithm : uint8_t {
    Legacy = 0,
    TriangleProductV1 = 1
};

// ★ v14.41: SelectedEstimate — Builder collapse で採用された推定値。
enum class SelectedEstimate : uint8_t {
    Unknown = 0,
    Measured = 1,
    UpperBound = 2
};

struct BuildDiagnostics {
    uint8_t analysisVersion = AnalysisVersionPolicy::kCurrent;
    EqGainAlgorithm eqGainAlgorithm = EqGainAlgorithm::TriangleProductV1;
    // ★ v14.43: resolvedOsFactor は Diagnostics に保持しない。Authority は OversamplingResult（Snapshot）が唯一。
    //   必要時は Snapshot.oversampling.resolvedOsFactor を参照する。
    BoundMethod boundMethod = BoundMethod::TriangleProduct;
    SelectedEstimate selectedEstimate = SelectedEstimate::Measured;  // ★ v14.41
    float eqMeasuredGainDb = 0.0f;
    float eqMeasuredRawGainDb = 0.0f;    // ★ v14.47: 放物線補間前の measured 生値（dB）。boundExcessDb 評価時に補間影響を分離するため
    float eqUpperBoundGainDb = 0.0f;
    float eqMeasuredFreqHz = 0.0f;       // measured のピーク周波数。診断用
    float eqUpperBoundFreqHz = 0.0f;      // upperBound のピーク周波数。診断用
    float boundExcessDb = 0.0f;
    float totalMaxQ = 0.0f;              // 全有効バンド中の最大Q値。診断専用
};
```

finite チェック対象: `eqMaxGainDb`, `eqMaxQ`, `irFreqPeakGainDb`

`sealBuildAnalysis` の検証:
```cpp
if (!isFiniteFloat(analysis.eqMaxGainDb) || !isFiniteFloat(analysis.eqMaxQ)
    || !isFiniteFloat(analysis.irFreqPeakGainDb))
    return BuildAnalysis{};
```

**verifyBuildBundle()（★ v14.11: 名称変更。4-object validation）**:

```cpp
// verifyBuildBundle — BuildAnalysis + BuildDiagnostics + OversamplingResult + RuntimeBuildSnapshot +
// AnalysisPart の整合性を一括検証。ISR Authority Singularization: Validator はこの一箇所のみ。
// ★ v14.37: Diagnostics を BuildAnalysis から分離。Analysis は Publish 対象、Diagnostics は Debug 対象。
// ★ v14.46: Facade パターンで実装。内部的に verifyAnalysis(), verifyDiagnostics(), verifySnapshot() を呼び出す。
[[nodiscard]] inline bool verifyBuildBundle(
    const BuildAnalysis& analysis,
    const BuildDiagnostics& diagnostics,
    const OversamplingResult& oversampling,
    const RuntimeBuildSnapshot& snapshot,
    const AnalysisPart& analysisPart) noexcept
{
    if (!analysis.sealed || !snapshot.sealed)
        return false;
    if (analysis.generation != snapshot.generation)
        return false;
    if (!isFiniteFloat(analysis.eqMaxGainDb) || !isFiniteFloat(analysis.eqMaxQ)
        || !isFiniteFloat(analysis.irFreqPeakGainDb))
        return false;
    if (!oversampling.isValid())
        return false;
    if (analysisPart.analysisVersion != diagnostics.analysisVersion)
        return false;
    // ★ v14.32: Builder collapse 契約の検証（epsilon 0.1dB, Adaptive Sampling/将来変更対策）
    constexpr float kCollapseToleranceDb = 0.1f;
    // ★ v14.41: 許容値は AnalysisVersionPolicy に紐付け（将来 version 変更時は別値を使用可能）
    const float tolerance = (diagnostics.analysisVersion >= AnalysisVersionPolicy::kCurrent)
        ? kCollapseToleranceDb : 0.2f;
    const float expectedCollapse = std::max(diagnostics.eqMeasuredGainDb, diagnostics.eqUpperBoundGainDb);
    if (std::abs(analysis.eqMaxGainDb - expectedCollapse) > tolerance)
        return false;
    // ★ v14.47: selectedEstimate と実際の比較結果の整合性を検証
    if ((diagnostics.selectedEstimate == SelectedEstimate::Measured && diagnostics.eqMeasuredGainDb < diagnostics.eqUpperBoundGainDb - 0.01f)
        || (diagnostics.selectedEstimate == SelectedEstimate::UpperBound && diagnostics.eqUpperBoundGainDb < diagnostics.eqMeasuredGainDb - 0.01f))
        return false;
    return true;
}

// ★ v14.38: verifyDiagnostics — BuildDiagnostics の整合性を検証（Publish 可否とは独立）。
//   ISR 思想: verifyBuildBundle() は Publish 可否のみ、verifyDiagnostics() は Debug 情報の正当性を担当。
[[nodiscard]] inline bool verifyDiagnostics(const BuildDiagnostics& diagnostics) noexcept
{
    // ★ v14.42: finite チェック（診断値の数値健全性）
    if (!isFiniteFloat(diagnostics.eqMeasuredGainDb)
        || !isFiniteFloat(diagnostics.eqMeasuredRawGainDb)  // ★ v14.47
        || !isFiniteFloat(diagnostics.eqUpperBoundGainDb)
        || !isFiniteFloat(diagnostics.eqMeasuredFreqHz)
        || !isFiniteFloat(diagnostics.eqUpperBoundFreqHz)
        || !isFiniteFloat(diagnostics.boundExcessDb)
        || !isFiniteFloat(diagnostics.totalMaxQ))  // ★ v14.45: totalMaxQ も finite チェック対象
        return false;
    // eqGainAlgorithm が既知の範囲内か
    if (diagnostics.eqGainAlgorithm != EqGainAlgorithm::TriangleProductV1
        && diagnostics.eqGainAlgorithm != EqGainAlgorithm::Legacy)
        return false;
    // BoundMethod と analysisVersion の整合性
    // ★ v14.45: Current version では TriangleProduct のみ許可（Legacy は旧 version 用）
    if (diagnostics.analysisVersion == AnalysisVersionPolicy::kCurrent
        && diagnostics.boundMethod != BoundMethod::TriangleProduct)
        return false;
    if (diagnostics.analysisVersion == AnalysisVersionPolicy::kLegacy && diagnostics.boundMethod != BoundMethod::Legacy)
        return false;
    if (diagnostics.boundExcessDb < 0.0f)
        return false;
    return true;
}
```

**PlanDiagnostics:**

```cpp
struct PlanDiagnostics {
    float qMargin   = 0.0f;
    float eqBoost   = 0.0f;
    float convBoost = 0.0f;
    bool  clamped   = false;
    // ★ v14.36: 個別クランプ状態の記録（Week2 評価用）
    bool inputClamped  = false;  // inputHeadroomDb が kClampInputMin/-18dB に張り付いた
    bool trimClamped   = false;  // convolverInputTrimDb が kClampTrimMin/-12dB に張り付いた
    bool makeupClamped = false;  // outputMakeupDb が kClampMakeupMax/12dB に張り付いた
    // ★ v14.45: combinedEstimateMethod — Conv→EQ のゲイン合成方式（将来 IR×EQ FFT 移行時の比較用）
    enum class CombinedEstimate : uint8_t { Sum = 0 };  // 現在は単純加算のみ
    CombinedEstimate combinedMethod = CombinedEstimate::Sum;
};
```

**DiagEvent::AutoGainClamped:**

| 項目 | 値 |
|------|-----|
| カテゴリ値 | `DiagCategory::AutoGainClamped = 10`（`AudioEngine.h:330`） |
| Count 更新 | `static_assert(Count == 11)`（`AudioEngine.Timer.cpp:213`） |
| データ構造 | `AutoGainClampedData { float eqBoostDb, convBoostDb, qMarginDb, rawMakeupDb, clampedMakeupDb; }` |
| 発行条件 | `makeup > kClampMakeupMax` でクランプ時 |
| UI表示 | 「ヘッドルーム不足」インジケーター |

#### 4.7 バグ修正対応（★ v14.34: 新規追加。Bug#1, #3, #4, #6, #8, #9 の設計）

本節では B.1.3 で確定したバグのうち、設計変更を伴うものの修正設計を記述する。

##### 4.7.1 Bug#1: applyDefaultsForCurrentMode() の autoGainStagingEnabled チェック

**対象**: `AudioEngine.Parameters.cpp:296-340`

**現状の問題**:
`setEqBypassRequested()` / `setConvolverBypassRequested()` から呼ばれる `applyDefaultsForCurrentMode()` が `autoGainStagingEnabled` を無視し、ハードコードされたデフォルトゲイン値で上書きする。

**修正設計**:
```cpp
void AudioEngine::applyDefaultsForCurrentMode()
{
    if (m_isRestoringState) return;

    // ★ Bug#1: Auto Gain 有効時はデフォルト値による上書きを防止
    //   NOTE: consumeAtomic は std::atomic_load_explicit のラッパーであり
    //   読み取り専用（値を消費しない）。名前は誤解を招くため、設計書では
    //   load() で表記する。
    if (autoGainStagingEnabled.load(std::memory_order_acquire))
        return;

    // ... 既存のデフォルト値ロジック（変更なし） ...
}
```

**ISR 整合性**: `applyDefaultsForCurrentMode()` は手動ゲイン設定時の初期値提供が責務。Auto Gain 有効時は Planner が唯一のゲイン決定権限を持つ（Authority Singularization）。上記修正により責務分離が明確化される。

##### 4.7.2 Bug#3: Preset ロード時の Auto Gain 値保護

**対象**: `AudioEngine.StateIO.cpp:56-73`（`requestLoadState()`）

**現状の問題**:
`autoGainStagingEnabled` フラグが StateIO で保存/復元されず、Preset ロード時に手動ゲイン値（inputHeadroomDb / outputMakeupDb / convolverInputTrimDb）が常に復元される。RuntimeBuilder が Rebuild 時に Auto Gain を再計算するためオーディオ影響は一過性だが、UI 表示が一瞬ずれる。

**修正設計**:
```cpp
void AudioEngine::requestLoadState(const juce::ValueTree& state)
{
    RestoreStateGuard guard(m_isRestoringState);

    // ... Step 1: モード・バイパス状態の復元（変更なし） ...

    // ─── Step 2: ゲイン値の復元 ────────────────────────────────────
    // ★ Bug#3: Auto Gain 有効時は手動ゲイン値の復元をスキップ
    //   （RuntimeBuilder が Rebuild 時に Auto Gain を再計算する）
    // ★ Bug#3: autoGainStagingEnabled は読み取り専用で取得（consumeAtomic は load と等価）
    //   ★ v14.41: 旧 Preset（autoGainStagingEnabled プロパティなし）との互換性:
    //     プロパティが存在しない場合、強制的に手動ゲインを復元する（デフォルト Auto OFF 扱い）。
    const bool autoGainEnabled = state.hasProperty("autoGainStagingEnabled")
        ? static_cast<bool>(state.getProperty("autoGainStagingEnabled"))
        : false;  // 旧 Preset: Auto Gain 無効として手動ゲインを復元

    if (!autoGainEnabled)
    {
        // Auto Gain 無効時のみ Preset の手動ゲイン値を復元
        if (state.hasProperty("inputHeadroomDb"))
            setInputHeadroomDb(state.getProperty("inputHeadroomDb"));
        if (state.hasProperty("outputMakeupDb"))
            setOutputMakeupDb(state.getProperty("outputMakeupDb"));
        if (state.hasProperty("convolverInputTrimDb"))
            setConvolverInputTrimDb(state.getProperty("convolverInputTrimDb"));
    }

    // ... Step 2 続き（ditherBitDepth, noiseShaperType 等） ...
    // ... Step 3（softClipEnabled, saturationAmount 等） ...
}
```

**保存側の変更**（`getCurrentState()`）:
```cpp
state.setProperty("autoGainStagingEnabled",
    autoGainStagingEnabled.load(std::memory_order_acquire), nullptr);
```

**ISR 整合性**: Preset は「手動ゲイン設定の保存」であり、Auto Gain 有効時は Planner の出力が優先される。修正により Preset JSON の解釈が Auto Gain 状態に依存する設計となるが、RuntimeBuilder のゲイン上書き（L319）と一貫する。

##### 4.7.3 Bug#4: AGC と Auto Gain Staging の競合防止

**対象**: `EQProcessor.Processing.cpp:318-420` / `AudioEngine.Parameters.cpp:224-241`

**現状の問題**:
AGC（EQ 内部の自動ゲイン補正）と Auto Gain Staging（Engine 全体のゲイン構造管理）が同時有効時に、ゲイン補正の二重適用が発生し得る。

**修正設計**:
```cpp
// AudioEngine::setAutoGainStagingEnabled() 内で AGC 状態をパラメータ通知で連動
void AudioEngine::setAutoGainStagingEnabled(bool enabled)
{
    convo::publishAtomic(autoGainStagingEnabled, enabled, std::memory_order_release);
    // ★ Bug#4: Auto Gain 有効時は EQ AGC を無効化（二重ゲイン補正防止）。
    //   Engine → UI の直接依存を避けるため、パラメータ atomic を介して
    //   Listener 経由で UI が自動反映される設計とする。
    //   (旧: uiEqEditor.setAGCEnabled(!enabled) → Engine→UI 逆依存)
    convo::publishAtomic(agcEnabled, !enabled, std::memory_order_release);
    sendChangeMessage();
}
```

**AGC 無効化の根拠**:
- Auto Gain Staging は EQ + Conv の複合ゲイン構造を解析し入力ヘッドルーム・メイクアップゲインを決定する
- AGC は EQ 単体の入出力比を平滑化する
- Auto Gain 有効時は EQ 出力レベルの管理を Planner が一元化するため、AGC は不要かつ干渉を防止するため無効化する

##### 4.7.4 Bug#6: Oversampling ComboBox ID 存在検証

**対象**: `DeviceSettings.cpp:430-437`

**現状の問題**:
`setSelectedId(id)` が ID 不在時に JUCE によりサイレント失敗する。特にサンプルレート変更後に ComboBox 項目と実倍率が不一致になる可能性がある。

**修正設計**:
```cpp
// DeviceSettings 初期化時の ComboBox ID 検証
const std::map<int, int> factorToId = {{0, 1}, {1, 2}, {2, 3}, {4, 4}, {8, 5}};
int currentFactor = audioEngine.getOversamplingFactor();
int targetId = 1;  // default: Auto

if (auto it = factorToId.find(currentFactor); it != factorToId.end())
{
    // ★ Bug#6: 要求 ID が ComboBox に存在するか確認
    for (int i = 0; i < oversamplingComboBox.getNumItems(); ++i)
    {
        if (oversamplingComboBox.getItemId(i) == it->second)
        {
            targetId = it->second;
            break;
        }
    }
}

oversamplingComboBox.setSelectedId(targetId, juce::dontSendNotification);
```

**代替設計**: `setSelectedId()` の戻り値（bool）を確認し、false の場合は Auto にフォールバック。

##### 4.7.5 Bug#8: Oversampling ComboBox の SR 変更時再構築

**対象**: `DeviceSettings.cpp:558-575`（`changeListenerCallback()`）

**現状の問題**:
`audioDeviceManager` の変更を検知しても `oversamplingComboBox` が再構築されず、サンプルレート変更後の許可倍率と ComboBox 項目が不一致になる可能性がある。

**修正設計**:
```cpp
void DeviceSettings::changeListenerCallback(juce::ChangeBroadcaster* source)
{
    if (source == &filterTypeTabs.getTabbedButtonBar())
    {
        auto type = (filterTypeTabs.getCurrentTabIndex() == 1)
            ? AudioEngine::OversamplingType::LinearPhase
            : AudioEngine::OversamplingType::IIR;
        if (type != audioEngine.getOversamplingType())
            audioEngine.setOversamplingType(type);
    }
    else if (source == &audioDeviceManager)
    {
        ensureUsableChannelSelection(audioDeviceManager);
        updateBitDepthList();
        // ★ Bug#8: サンプルレート変更時に Oversampling ComboBox を再構築
        rebuildOversamplingComboBox();
    }
}
```

`rebuildOversamplingComboBox()` の実装:
```cpp
void DeviceSettings::rebuildOversamplingComboBox()
{
    const double sr = audioEngine.getSampleRate();
    const int prevId = oversamplingComboBox.getSelectedId();

    oversamplingComboBox.clear(juce::dontSendNotification);
    oversamplingComboBox.addItem("Auto", 1);
    oversamplingComboBox.addItem("1x (None)", 2);
    oversamplingComboBox.addItem("2x", 3);

    const int maxF = convo::OversamplingPolicy::maxAllowedFactor(sr);
    if (maxF >= 4) oversamplingComboBox.addItem("4x", 4);
    if (maxF >= 8) oversamplingComboBox.addItem("8x", 5);

    // 以前の選択を維持（ID が存在すれば）。JUCE ComboBox::setSelectedId は void 戻り値のため、
    // 別途 getSelectedId() で反映を確認する。
    oversamplingComboBox.setSelectedId(prevId, juce::dontSendNotification);
    if (oversamplingComboBox.getSelectedId() != prevId)
        oversamplingComboBox.setSelectedId(1, juce::dontSendNotification);  // fallback to Auto
}
```

`maxAllowedFactor()` は OversamplingPolicy の内部ルックアップを間接提供する軽量公開 API。
  **Authority Singularization について**: `resolve()` が唯一の決定権限を持ち、`maxAllowedFactor()` はその内部ルックアップの
  サブセットを GUI 向けに公開するヘルパーである。`maxAllowedFactor()` は「許可される最大倍率」のみを返し、
  「実際に使用する倍率」の決定は `resolve()` が行う。したがって Authority は `resolve()` に一元化されている。

##### 4.7.6 Bug#9: IR Cache メモリ制限（将来拡張）

**対象**: `ConvolverProcessor.LoadPipeline.cpp:695-730`

**現状の問題**:
IR Cache はエントリ数制限（`MAX_CACHE_ENTRIES=8`、上限 64）のみでメモリベースの制限がない。

**修正設計（将来課題、Week2 以降）**:
```cpp
struct ConvolverProcessor::CacheConfig {
    size_t maxEntries = 8;       // 最大エントリ数
    size_t maxMemoryMB = 512;    // ★ 最大メモリ使用量（MB）。0=無制限
};

void ConvolverProcessor::pruneCache()
{
    const juce::ScopedLock sl(cacheMutex);
    const size_t maxMem = convo::consumeAtomic(maxCacheMemoryMB_, std::memory_order_acquire);

    // エントリ数制限（既存）
    while (irCache.size() > static_cast<size_t>(convo::consumeAtomic(maxCacheEntries_, std::memory_order_acquire)))
        evictOldestCacheEntry();

    // ★ メモリ制限（新規）
    if (maxMem > 0)
    {
        size_t totalBytes = 0;
        for (const auto& [key, entry] : irCache)
            totalBytes += entry.irDataSize;

        while (totalBytes > maxMem * 1024 * 1024 && irCache.size() > 1)
        {
            auto oldest = findOldestCacheEntry();
            totalBytes -= oldest->second.irDataSize;
            irCache.erase(oldest);
        }
    }
}
```

本修正は Week2 以降の実装対象とする。現在のエントリ数制限（最大 64、実用上 320MB 以内）で問題が報告されていないため、優先度は低い。

### 5. テスト計画（★ v14.3: 拡充）

| 分類 | テスト | 期待値 |
|------|--------|--------|
| Unit | LPF/HPFのみ | `maxGain=0` |
| Unit | Peaking +12dB Q=1単体 | `20log10\|H(f0)\|` と 0.1dB一致 |
| Unit | Parallel 2バンド +12dB×2 | `20log10(7)=16.9dB` と 0.2dB一致 |
| Unit | Serial 2バンド +12dB×2 | 24dB |
| Unit | 合成IR `dirac * 2.0` | freqPeak 6dB, L1 6dB |
| Unit | `GainStagingContractTests` リファレンス更新 | V2にリンク、`eq=0 → input=0` |
| **Integration** | **minimum phase IR（合成）** | QSurge改善の妥当性確認 |
| **Integration** | **linear phase IR（合成）** | 位相特性による推定誤差の確認 |
| **Integration** | **mixed phase IR（合成）** | **最も危険なケース。PEQ + mixed phase IR でQSurge改善の妥当性を検証** |
| **Integration** | **20Band全部ON + Mixed Phase IR + Parallel EQ + Auto Oversampling** | **computeEstimatedMaxGainComplex() の最大負荷ケース。最も誤差が出やすい構成。実装前にプロファイル必須** |
| **Integration** | **Automation Stress: 20Band, 20Hz⇔20kHz sweep, Q 0.7⇔20, ±24dB, 100Hz更新** | **以下の観測項目を含む:**
  - **UI Automation / Host Automation 双方でパラメータ更新**
  - **Parameter Smoothing が AutoGain 更新と競合しないこと**
  - **Rebuild Queue の最大深度を測定**
  - **Rebuild Latency（平均・99 percentile・最大）を記録**
  - **computeEstimatedMaxGainComplex() 実測CPU時間（ms）を記録**
  - **candidate band 数も同時記録**
  - **オーディオ Dropout なし**
  実装上は rebuild throttle（最小更新間隔）の要否判断に使用 |
| Integration | sin 1kHz 0dBFS + EQ +12dB Q=10 | TP `-1.0±0.5dBTP`, RMS差 0.2dB以内 |
| Integration | factory hall IR, EQThenConv vs ConvThenEQ | ラウドネス差 1dB以内 |
| **Integration** | **同一周波数 逆位相 Parallel: +12dB Peak ×2, 180°位相** | **upperBound≫measured の典型例。boundExcessDb の実測確認** |
| **Integration** | **同一周波数 逆位相 Parallel: +12dB Peak + -12dB Peak, 同一周波数** | **measured≈0dB, upperBound>>0dB の boundExcessDb 最大ケース。Planner が過剰に減衰しないことを確認** |
| **Integration** | **Nyquist 直前 HighShelf, 384kHz, fc=180kHz, Q=0.7, +24dB** | **biquadResponse() が最も数値的に不安定になる Nyquist 極限ケース。NaN/Inf なし** |
| **Integration** | **EQ Peak +24dB, Q=20, 20Band, Parallel, 384kHz, 8x** | **upperBound finite, NaNなし, Overflowなし の検証。最悪ケースの数値安定性確認** |
| **Unit** | **Parallel, 20Band, ランダムGain ±24dB, Q 0.5〜20** | **log1p/exp 経路で NaN/Overflow が発生しないことの耐久試験。100 パターン以上ランダム生成** |
| **Integration** | **LowShelf +18dB 単体** | **候補Band抽出の確認（Shelf ピークは center ではなく DC 近傍）** |
| **Integration** | **384kHz Auto OS** | **OS 倍率 2x になること** |
| **Integration** | **768kHz Auto OS** | **OS 倍率 1x になること** |
| **Integration** | **>768kHz 入力** | **OversamplingResult.supported==false になること。Builder が Publish をスキップし、旧 Runtime を維持すること** |
| Listening | ABX Auto ON/OFF | ノイズフロア劣化なし |

#### 5.1 テスト用IR合成指針

| 種別 | 合成方法 | 特徴 |
|------|---------|------|
| Dirac × k | 単位インパルス × 倍率 | 基準。convBoost = 20log10(k) |
| Minimum phase | 任意振幅レスポンス→Hilbert変換→最小位相 | 最大の位相回転、TruePeak 最大 |
| Linear phase | FIR 窓関数法＋線形位相 | 位相回転ゼロ、TruePeak 最小 |
| Mixed phase | 最小位相IRの後半に線形位相成分を付加 | 実IRに最も近い特性 |

### 6. マイルストーン

| Week | 優先度 | 内容 |
|------|--------|------|
| 1 | P0 | E-1/E-3/I-1/P-1 修正。`computeEstimatedMaxGainComplex`, `IRFinalAnalysis` 実装。Planner定数再設計、固定Ceiling廃止、ternary search削除 |
| 2 | P1 | Builder統合、封印V2、診断ログ。テスト全置換。実IR 50種 + 合成 extreme 20種でヘッドルーム分布測定 |

### 7. リスク

1. **ラウドネス増加**: 固定Ceiling廃止により Conv→EQ 時 +2〜+6dB 上昇の可能性。リリースノートに明記
2. **kQSurgeMax 頭打ち**: Q=20,gain=24dB で 2.5dB 制限到達。複素応答上界がバックアップするが、リリース前に実測検証
3. **additionalAttenuationDb 互換性**: フィールドを残し finite チェック対象から除外
4. **osFactor Auto 解決タイミング**: `0=Auto` 未解決時に安全側デフォルトを使用。または推定を DSPCore 作成後に移動

----------------------------------------------------------------------
## 未確定事項（★ v14.24: 再調査・全件確定済み）
----------------------------------------------------------------------

以下の項目は全件調査・確定済み。実装着手前に認識すべき事項として記録する。

| # | 項目 | 確定結果 | 根拠 |
|---|------|---------|------|
| U-1 | 実IR 50種ベンチマークリスト | **Week2 で実施（設計上の未確定事項ではなく、検証タスク扱い）**。OpenAIR データベースに 100 種類以上の実IRが公開済み。以下は具体化済みのリスト案:<br><br>**Hall (5):** Central Hall York, Arthur Sykes Rymer Auditorium, 1st Baptist Church Nashville, Elveden Hall, Creswell Crags<br>**Plate (3):** 合成 plate IR（sampledata の reverb_ir.wav を活用）<br>**Spring (2):** 合成 spring IR（自作またはフリーリソース）<br>**Chamber (3):** Alcuin College York, Clifford's Tower York, Hendrix Hall Derwent<br>**Room (5):** 各種小規模ルーム IR（OpenAIR から選定）<br>**Outdoor (2):** Abies Grandis Forest Wheldrake, openair outdoor IR<br>**Synthetic extreme (20):** Dirac/Minimum/Linear/Mixed phase 各5種<br><br>**サンプリングレート**: 48kHz 統一<br>**長さ**: 0.5秒〜4秒（kMaxAnalysisWindow=65536 以内）<br>**選定基準**: ①検証目的に合致 ②ライセンス適合 ③実IRらしさ<br>**作業手順**: Week1 終了時に sampledata/ へダウンロード。Week2 開始時にパイプラインで一括測定 | OpenAIR に100+ IR確認済み。EchoThief 等の代替リソースも調査済み |
| U-2 | `computeEstimatedMaxGainDb` 呼び出しタイミング | **DSPCore作成前（現状維持）**。`RebuildDispatch.cpp:655` のタイミングで問題なし。osFactorが `0=Auto` の場合は `OversamplingPolicy::resolve()` を使用する（v14.22 で Policy 化済み）。Builder と Planner が同一の純粋関数で倍率を決定するため、Auto 解決タイミングの差異による不一致がない | `RebuildDispatch.cpp:845-951` で osFactor 解決。`OversamplingPolicy::resolve()` は既存の DSPCore Auto 解決ロジックを呼び出すアダプタ（Authority 二重化防止） |
| U-3 | `legacyCeilingMode` UI 露出形態 | **v14.3 で廃止**。固定Ceiling自体を削除するため legacyCeilingMode は不要。旧来の-6dB Ceilingに依存するユースケースは存在せず、互換性オプションとしても維持しない | 固定Ceilingの廃止がレビュー指摘#3。経験則に基づく固定値は認められない |
| U-4 | `AutoGainClamped` の UI 表示仕様 | **ツールチップ + 簡易インジケーター（最小実装）**。DiagEvent インフラは既存の `formatDiagEvent` 機構（`AudioEngine.Timer.cpp:143-215`）を拡張して対応。UI側は現在の診断表示領域にテキスト表示。詳細パネルは将来対応 | 既存 `DiagCategory` に `AutoGainClamped=10` 追加。`Count` も11に更新。データ構造は診断ログとして十分 |
| U-5 | `AnalysisPart` のバージョニング戦略 | **`AnalysisPart.analysisVersion` が Authority。`BuildAnalysis.Diagnostics.analysisVersion` はその管理用コピー**。値 `2` で v14.24 の解析フォーマットを識別。Version Policy は Appendix C.6 参照。不整合時は `verifyBuildBundle()` で検出。**コピー方向: `BuildAnalysis.Diagnostics → AnalysisPart.analysisVersion`**（BuildAnalysis 生成後に AnalysisPart へ反映） | `RuntimeBuilder.h:52-55` の `AnalysisPart` に `uint8_t analysisVersion` 追加。`Builder` が BuildAnalysis 生成後に `analysisPart.analysisVersion = analysis.diag.analysisVersion` を設定 |

### 棚卸し: 現行コードと v14.24 設計の差分

以下の項目は設計書と現行コードの乖離を記録する。実装時に対応すること。

| # | 対象 | 現行コード | 設計書 v14.24 | 対応Week |
|---|------|-----------|---------------|----------|
| D-1 | `AutoGainPlanner::plan()` 引数 | `additionalAttenuationDb` を受ける | `irFreqPeakGainDb` を受ける（additionalAttenuationDb 廃止） | Week1 |
| D-2 | `AutoGainPlanner` 定数 | `kMarginEqFirst=3.0`, `kMarginConvFirst=1.5`, `kMarginInterStage=2.0`, `kClampInputMin=-12`, `kConvFirstInputCeiling=-6.0` | `kMarginEqFirst=1.5`, `kMarginConvFirst=1.0`, `kMarginInterStage=1.0`, `kClampInputMin=-18`, `kConvFirstInputCeiling` 廃止 | Week1 |
| D-3 | `estimateQSafetyMargin()` | 旧QSurge式（`1.5+gain*0.15*20/0.707`, max=6.0） | `EmpiricalSafetyMarginPolicy::evaluate()`（`0.8+Q項+Gain項`, max=2.5） | Week1 |
| D-4 | `BuildAnalysis` 構造体 | `{generation, eqMaxGainDb, additionalAttenuationDb, sealed}` の最小構成 | `{generation, eqMaxGainDb, eqMaxQ, irFreqPeakGainDb, Diagnostics{analysisVersion, resolvedOsFactor, boundMethod, eqMeasuredGainDb, eqUpperBoundGainDb, boundExcessDb}, sealed}` | Week1-2 |
| D-5 | `verifyBuildAnalysisPair()` | 2引数（BuildAnalysis + RuntimeBuildSnapshot） | `verifyBuildBundle()` 4引数（BuildAnalysis + OversamplingResult + RuntimeBuildSnapshot + AnalysisPart） | Week2 |
| D-6 | `computeEstimatedMaxGainDb()` | 300点対数探索 + Serial積近似 | `computeEstimatedMaxGainComplex()` 600点 + 適応サンプリング + 複素応答 + 放物線補間 | Week1 |
| D-7 | `OversamplingResult` | 未存在 | `{resolvedOsFactor, requestedOsFactor, isAutoResolved, isValid()}` | Week2 |
| D-8 | `EmpiricalSafetyMarginPolicy` | 未存在（旧 QSurge は AutoGainPlanner 内部） | Policy クラスとして分離（Builder/Planner/Test 共有） | Week1 |
| D-9 | `PlannerInput` DTO | 未存在（Planner が BuildAnalysis を直接参照） | `{eqMaxGainDb, eqMaxQ, irFreqPeakGainDb}` — Planner 専用 DTO | Week1 |
| D-10 | `EQAnalysisResult` / `PeakInfo` | 未存在（`computeEstimatedMaxGainDb` は float 返却） | `{PeakInfo measured, PeakInfo upperBound}` — 二層構造 | Week1 |

----------------------------------------------------------------------
## Appendix
----------------------------------------------------------------------

### A. 改訂履歴

| 版 | 日付 | 変更内容 |
|----|------|---------|
| v14.0 | — | 初版。7件の論理欠陥の摘出と基本アーキテクチャ提案 |
| v14.1 | — | 詳細改修仕様の記述。定数再設計・QSurge新式・4パターンロジック |
| v14.2 | 2026-07-18 | ソースコード完全突合検証に基づき全項目確定。3部構成（設計/未確定/Appendix）に再編。検証詳細・コードパスリファレンス・数学的補遺を Appendix に集約 |
| **v14.3** | **2026-07-18** | **レビュー指摘4件に対応。①Parallel ×0.95経験係数廃止、②ternary search廃止＋放物線補間＋粗探索倍増、③Conv→EQ固定Ceiling(-2dB)廃止、④True Peak保証→推定目標に修正。テスト計画拡充（minimum/linear/mixed phase IR）。未確定事項U-1〜U-5全件調査確定。数学的補遺を拡充** |
| **v14.4** | **2026-07-18** | **レビュー指摘2件に対応。①Parallel証明の等式 `= Π|Hi|` を不等式 `≤ Π|Hi|` に修正、②放物線補間に局所最大条件（中央点 > 両隣）を追加し外挿発散を防止。性能改善: 適応サンプリングをピーク候補Bandのみに制限（31Band時 4000点→約1600点に抑制）** |
| **v14.5** | **2026-07-18** | **レビュー指摘3件に対応。①上界証明の条件を「評価周波数 ω で全 |Hi| ≥ 1」と明確化。②目的文を「安全側Boundは経験係数不使用」へ修正しQSurgeを経験的マージンと位置付け。③解析用OS倍率を共有純粋関数 `resolveOversamplingForAnalysis()` に分離。テスト追加: 31Band全部ON + Mixed Phase IR + Parallel EQ + Auto Oversampling** |
| **v14.6** | **2026-07-18** | **レビュー指摘5件に対応。①Parallel上界を `Π max(1,|Hi|)` に一般化し無条件成立を証明。②`computeEstimatedMaxGainComplex()` の戻り値を `EQGainAnalysis` 構造体に変更。③OS倍率決定を `OversamplingPolicy` に切り出しSingle Source of Truthを確立。④QSurge係数の導出根拠（W3C Cookbook参照・Monte Carlo検証・再較正手順）を詳細化。⑤Automation Stress Test（31Band/20Hz⇔20kHz/Q 0.7⇔20/±24dB/100Hz更新）を追加。C.3節の重複テーブルを削除** |
| **v14.7** | **2026-07-18** | **レビュー指摘4件に対応。①Parallel上界証明を根本的に修正: `|Hi-1| ≤ max(0,|Hi|-1)` の誤りを正し、`Π(1 + |Hi-1|)` の帰納法による正しい証明に差し替え。`Π max(1,|Hi|)` の反例（Hi=-2）を明記。②QSurgeを `QSurgePolicy` クラスに分離（ISR Policy化）。③`EQGainAnalysis` → `EQAnalysisResult { PeakInfo measured, PeakInfo upperBound; float maxQ; }` 二層構造に拡張。④`OversamplingDecision` の権限一元化（BuildAnalysis.osFactor が唯一の決定結果）** |
| **v14.8** | **2026-07-18** | **レビュー指摘に対応。①「過剰評価2dB以内」を実測ベースの検証目標と明確化（数学的保証ではない）。②`osFactor` を BuildAnalysis から分離し `OversamplingDecision` 構造体として Snapshot 側に移動（Analysis と Decision の責務分離）。③`upperBound.freqHz` を「上界が最大となる周波数」と定義。④`sealBuildAnalysis` に `osFactor > 0` 検証を追加。⑤Automation Stress Test を拡張（UI/Host Automation・Parameter Smoothing・Rebuild Queue 深度・Rebuild Latency）。⑥QSurge係数を暫定値と明記しWeek2較正手順を具体化** |
| **v14.9** | **2026-07-18** | **レビュー指摘4件に対応。①Planner が `max(measured.gainDb, upperBound.gainDb)` を採用することで安全側保証と upperBound の利用を一本化。②`OversamplingDecision` → `OversamplingResult` に名称変更（Authorityは Policy::resolve() が保持）。③`sealBuildAnalysis` から `osFactor` 検証を分離し `verifySnapshot()` を新設（責務分離）。④QSurge係数の根拠を Appendix C.5 に追加（Worst95%/99%予測値・係数導出マトリクス）** |
| **v14.10** | **2026-07-18** | **レビュー指摘4件に対応。①upperBound データフローを案B（Builder collapse）で統一。Planner は `eqMaxGainDb` のみを受け取り解析方法を知らない（ISR思想）。②`verifySnapshot()` 新設を取りやめ、既存 `verifyBuildAnalysisPair()` を拡張。Validatorは一箇所（Authority Singularization）。③Oversampling の naming を `requestedOsFactor` / `resolvedOsFactor` に明確化し二重管理を解消。④Analysis が OversamplingResult に依存して生成されることを明記** |
| **v14.11** | **2026-07-18** | **レビュー指摘4件に対応。①`BuildAnalysis` に `resolvedOsFactor` / `analysisVersion(=2)` / `eqMeasuredGainDb` / `eqUpperBoundGainDb` を追加。解析 provenance・将来互換性・診断情報を一元保持。②`verifyBuildAnalysisPair()` → `verifyBuildBundle()` に名称変更（3-object validation）。`sealBuildAnalysis` も `resolvedOsFactor` を検証。③collapse 後の診断情報保持を Builder コードに反映。④U-5 を `BuildAnalysis.analysisVersion = 2` で統一に更新。テキストの `osFactor` → `resolvedOsFactor` を統一** |
| **v14.12** | **2026-07-18** | **レビュー指摘5件に対応。①`analysisVersion` を `BuildAnalysis.Diagnostics` に隔離し、authority は `AnalysisPart.analysisVersion` に一元化（Single Source of Truth）。②`verifyBuildBundle()` に `requestedOsFactor ≠ 0 → resolved と一致必須` の検証を追加。③診断専用情報を `BuildAnalysis::Diagnostics` サブ構造体に分離し責務明確化。④Appendix C.6 に `analysisVersion` インクリメントポリシーを追加（5分類・条件・互換性要件）。⑤`boundMethod` enum を追加し upperBound 算出方式の provenance を保持** |
| **v14.13** | **2026-07-18** | **レビュー指摘4件に対応。①`verifyBuildBundle()` から Oversampling 固有検証を `OversamplingResult::isValid()` へ委譲（Validator は Policy を再実装しない）。②`BoundMethod` を `enum class : uint8_t` に型安全化（TriangleProduct / ProductMaxMagnitude / ExactSampling）。③`analysisVersion` の二重管理を「Authority(=AnalysisPart) + 管理用コピー(=BuildAnalysis.Diagnostics)」と明確化。④Planner Contract を明文化（Planner MUST NOT inspect Diagnostics）** |
| **v14.14** | **2026-07-18** | **verifyBundle・PlannerInput・BoundMethod注釈・コピー方向修正** |
| **v14.15** | **2026-07-18** | **全フィールド検証・QSurgePolicy config・boundExcessDb** |
| **v14.16** | **2026-07-18** | **epsilon・isValid許容値・invariant・jassert** |
| **v14.17** | **2026-07-18** | **epsilon値・requestedOsFactor検証・magic number・証明補強** |
| **v14.18** | **2026-07-18** | **verify責務分割・boundExcess Release保護・CPU実測・補題** |
| **v14.19** | **2026-07-18** | **float比較・isValid簡素化・BoundMethod Migration・Oversampling Authority** |
| **v14.20** | **2026-07-18** | **epsilon 1e-5・kCurrentAnalysisVersion・isAutoResolved・Legacy・Session Restore Test** |
| **v14.21** | **2026-07-18** | **Legacy受理・boundExcess統一・AnalysisVersionPolicy** |
| **v14.22** | **2026-07-18** | **微小項切り捨て・maxQ PeakInfo移動・Oversampling Authority・isKnownBoundMethod・DiagEvent候補・AnalysisVersionPolicy** |
| **v14.23** | **2026-07-18** | **BoundMethod・isValid・resolvedOsFactor移動・eqMaxQ定義・QSurge→Empirical・微小項誤差・analysisVersion方向** |
| **v14.24** | **2026-07-18** | **レビュー指摘5件対応。①`eqResult.maxQ` を `eqResult.measured.maxQ` に修正（PeakInfo::maxQ と一致）。②`analysis.resolvedOsFactor` → `analysis.diag.resolvedOsFactor` に統一（BuildAnalysisに存在しないメンバーを修正）。③`verifyBuildBundle()` に `BoundMethod` と `analysisVersion` の整合性検証を追加。④`OversamplingPolicy` の説明を「Builder専有・PlannerはSnapshot読み取り専用」に修正。⑤`jassert` の `boundExcessDb < 40dB` を削除（31Bandフル構成で超過可能性）。Appendix C.1 に $H_i \\in \\mathbb{C}$ を明記** |
| **v14.25** | **2026-07-18** | **レビュー指摘2件対応。①U-1「対応保留」→「Week2 で実施（検証タスク扱い）」に更新、具体作業手順追加。②DiagEvent::AutoGainClamped のTODOコメントを4.6節参照に修正** |
| **v14.26** | **2026-07-19** | **ソースコード完全突合検証に基づく修正。①「31Band」→「20Band」に修正（実コードNUM_BANDS=20と一致）。②`isBoosting` からNotch/AllPassケース削除（実際のEQBandTypeに存在しないため到達不能）。③B.3 V-1コードパス参照を `EQProcessor.Processing.cpp:845` に修正。④B.2 P-3 にテストの不一致詳細（`kQSurgeHpfLpf=1.5` vs 実装の `eqMaxGainDb<=0→0`）を追記。⑤LPF/HPF の `isBoosting` コメントを「Q リゾナンスピークは別途 biquadComplex で検出」に修正。⑥OversamplingPolicy の境界条件テキストと公式の不一致を検出し、Week2 での確定事項として記録** |
| **v14.27** | **2026-07-19** | **OversamplingPolicy をルックアップ方式に全面書き換え。①入力SRごとの許可倍率テーブル（44.1k-96k=x8, 176.4k-192k=x4, 352.8k-384k=x2, 705.6k-768k=x1）を明記。②768kHz超入力は無音処理。③`maxAllowedFactor()` と `resolve()` の実装を追加。④中間値（x7, x6等）は発生しないことを保証。⑤GUI/DSPCore の OS 倍率コードに3件の不一致を発見（G-1〜G-3）。4x表示条件（352.8k/384k で誤表示）、maxFactor（352.8k で 2x 誤許可）、setOversamplingFactor（SR依存検証なし）** |
| **v14.28** | **2026-07-19** | **4.5.1節「GUI / DSPCore への OversamplingPolicy 統合」を新規追加。①GUI: oversamplingComboBox の表示条件を `maxAllowedFactor()` に統一。②DSPCore: Auto 解決の if-else チェーンを `maxAllowedFactor()` に置換。③setOversamplingFactor は構造変更なし（GUI+DSPCore の二重防御で十分）。防御の階層図を追加** |
| **v14.29** | **2026-07-19** | **外部レビュー5件に対応。①upperBound の評価にも適応サンプリングを追加（measured と同パイプライン）。②`PlannerInput.eqMaxQ` を「全有効バンドの最大Q」に変更（`PeakInfo.maxQ` 削除、代わりに `EQAnalysisResult.maxActiveQ` を追加）。③`OversamplingPolicy::resolve()` が 0 を返さないように修正（>768kHz でも 1 を返し、呼出し側で sr 判定）。④非GUI経路（Preset Restore / Host Automation / State Restore）の検証を防御階層に追加。⑤`verifyBuildBundle()` に Builder collapse 契約（`eqMaxGainDb == max(eqMeasuredGainDb, eqUpperBoundGainDb)`）の検証を追加** |
| **v14.30** | **2026-07-19** | **外部レビューII 10項目に対応。①`ComplexResponse`（独自実装）→ `std::complex<double>` に置換。②upperBound を複素和とは独立した実数積として明確化（コード例と説明を分離）。③`maxActiveQ` を「全有効バンド」→「ブースト対象バンド（isBoosting()==true）のみ」に修正。④放物線補間のタイミングを「粗探索→適応サンプリング→その最大点に補間」と明確化。⑤candidate band 判定「平坦域+1dB以上」→「`|H(f_center)| > 1.0`（中心周波数で0dB超）」に定量化。⑥`maxAllowedFactor()` を `resolve()` の内部実装に隔離し、DSPCore も `resolve()` のみを Authority として使用。⑦`OversamplingResult` に `supported` フラグを追加。`resolve()` は `OversamplingResult` 構造体を返す。768kHz超は `supported=false` で通知。⑧`verifyBuildBundle()` の epsilon を 1e-3→5e-3 に緩和（SIMD/FMA 丸め差対応）。⑨PlannerInput は変更なし（レビュー確認済み）。⑩Conv→EQ 経路にも `safetyMargin` を追加（Q safety margin は順序に依存しないため）** |
| **v14.31** | **2026-07-19** | **外部レビューIII 10項目に対応。①upperBound 保守性のモニタリング指針を追加（boundExcessDb > 3dB で記録）。②候補Band判定を `isBoosting()` に統一し `maxActiveQ` との不整合を解消。③Shelf フィルタの候補Band抽出を center/center/2/center*2 の3点で実施。④放物線補間を対数周波数軸（log2）に変更。⑤upperBound の補間を廃止（評価点最大を採用）。⑥`EmpiricalSafetyMarginPolicy::evaluate()` の戻り値を `std::max(0.0f, ...)` で負値クランプ。⑦`verifyBuildBundle()` epsilon 5e-3→0.02 に緩和。⑧`verifyBuildBundle()` に `oversampling.supported` 検証を追加。⑨Conv→EQ 加算の保守性に関する将来拡張注記を追加。⑩テストケース5種追加（逆位相Parallel, Shelf only, 384kHz, 768kHz, >768kHz）** |
| **v14.32** | **2026-07-19** | **外部レビューIV 8項目に対応。①upperBound 過大評価の Week2 分布測定を明記。②upperBound 用の候補Band判定を `|Hi-1| > 0.1` で measured と分離。③Band種別ごとに適応サンプリング範囲を最適化。④`verifyBuildBundle()` epsilon 0.02→0.05。⑤`s upported==false` を Validation エラー扱いしないよう修正。⑥GUI 専用 API `allowedFactors(sampleRate)` を追加。⑦`computeEstimatedMaxGainComplex()` の3層分割を追加。⑧`BuildAnalysis` に `totalMaxQ` を追加** |
| **v14.33** | **2026-07-19** | **外部レビューV 7項目に対応。①Shelf 候補抽出点を LowShelf/HighShelf で個別最適化。②LPF/HPF適応範囲を Q>0.8 のみに制限。③`resolve()` 異常値フォールバック。④supported=false を「Publishスキップ条件」に統一。⑤テスト追加 (+12dB/-12dB 同一周波数)。⑥PeakInfo.sampleIndex 追加。⑦verifyBuildBundle supported削除確定** |
| **v14.34** | **2026-07-19** | **バグレポート検証に基づく6件の修正設計を 4.7 節に追加。Bug#1: applyDefaults に autoGainStagingEnabled チェック(P0)。Bug#3: Preset ロード時スキップ(P2)。Bug#4: AGC無効化(P3)。Bug#6: ComboBox ID検証(P3)。Bug#8: ComboBox SR時再構築(P3)。Bug#9: IR Cache メモリ制限設計(P3)** |
| **v14.35** | **2026-07-19** | **レビューVI 6項目に対応。①Bug#1/#3 の consumeAtomic → load() に修正（読み取り専用を明確化）。②`allowedFactors()` と「唯一のAuthority」の記述を整合（resolve() が唯一の決定権限、allowedFactors() は内部ルックアップの間接提供）。③放物線補間の分母ゼロ近傍ガードを追加。④`PeakInfo.sampleIndex` を `SampleOrigin{type, index}` に拡張（粗探索/適応サンプリングの区別を追加）。⑤`analysisVersion == 2` のマジックナンバーを `AnalysisVersionPolicy::kCurrent` に置換。⑥`OversamplingResult.supported==false` の契約を強化（resolvedOsFactor を解釈禁止と明記）** |
| **v14.36** | **2026-07-19** | **レビューVII 9項目に対応。①Builder側 eqMaxQコメント統一。②upperBound候補Band判定を粗探索max|Hi-1|に変更。③放物線補間をdB空間に変更。④upperBound積算をexp(Σlog1p(delta))に変更。⑤supported=false時resolvedOsFactor=0(isValid==false)。⑥DiagnosticsにeqMeasuredFreqHz/eqUpperBoundFreqHz/eqGainAlgorithmVersion追加。⑦Week2再キャリブレーション注記追加。⑧PlanDiagnosticsにinputClamped/trimClamped/makeupClamped追加。⑨verifyBuildBundleにeqGainAlgorithmVersion検証追加** |
| **v14.37** | **2026-07-19** | **レビューVIII 3項目に対応。①BuildDiagnostics を BuildAnalysis から完全分離。②supported=Capability, resolvedOsFactor=Configuration と明確化。isValid()は値域のみ検証。③EQAnalysisResult に algorithmVersion 追加** |
| **v14.38** | **2026-07-19** | **レビューIX 5項目に対応。①verifyBuildBundle→verifyDiagnostics分離。②OS Result Authority一元化。③oversamplingFactor引数削除。④最悪3160点修正。⑤Engine→UI直接依存除去。Collapse tolerance 0.1dB。テスト追加** |
| **v14.39** | **2026-07-19** | **レビューX 9項目に対応。①Serial/Parallel measured定義分離。②候補Band条件統一。③analysisVersion定数化。④SampleOrigin bandIndex。⑤EqGainAlgorithm enum。⑥supported除外コメント。⑦processingRate責務明記。⑧PlannerInput immutable。⑨ランダム耐久テスト追加** |
| **v14.40** | **2026-07-19** | **レビューXI 3項目に対応。①upperBound exp→dB直接計算でInf防止。②collapse tolerance constexpr。③Nyquist Shelfテスト追加** |
| **v14.41** | **2026-07-19** | **レビューXII 10項目対応。①Lagrange補間 ②adaptive union ③探索範囲明確化 ④Bug#3旧Preset互換 ⑤Shelf低域10Hz ⑥LPF/HPF Q>0.707 ⑦SelectedEstimate ⑧tolerance version管理 ⑨supported契約 ⑩EqGainAlgorithm enum** |
| **v14.42** | **2026-07-19** | **レビューXIII 5項目対応。①Bug#8 JUCE API void修正 ②sampleIndex定義明確化 ③verifyDiagnostics finite追加 ④selectedEstimate Builder設定 ⑤evaluateBandDelta helper推奨** |
| **v14.43** | **2026-07-19** | **レビューXIV 4項目に対応。①BuildDiagnostics から resolvedOsFactor を削除（Authority は OversamplingResult/Snapshot に一本化）。②allowedFactors→maxAllowedFactor名称変更。③verifyBuildBundle Facade分割可能性。④命名改善** |
| **v14.44** | **2026-07-19** | **レビューXV 5項目対応。①upperBoundサンプリング近似明記 ②NaN/Infガード ③補間周波数順ソート ④bandIndex union時-1 ⑤supported=false処理統一** |
| **v14.45** | **2026-07-19** | **レビューXVI 10項目対応。①離散サンプリング近似表現 ②区間長比例配分 ③|Hi-1|説明改善 ④補間境界ガード ⑤log1p注記 ⑥verifyDiagnostics Current→Triangle ⑦supported=false統一 ⑧maxAllowedFactor説明 ⑨combinedEstimateMethod ⑩totalMaxQ finite** |
| **v14.46** | **2026-07-19** | **レビューXVII 6項目対応。①EQProcessor分割参照 ②unionアルゴリズム明文化 ③measured生値診断 ④SampleOrigin Union型 ⑤Capability/Configuration整理 ⑥verifyBuildBundle Facade具体化** |
| **v14.47** | **2026-07-19** | **レビューXVIII 4項目に対応。①`upperBound` を「サンプリング近似上界（sampled upper bound）」と明確化。②`BuildDiagnostics` に `eqMeasuredRawGainDb`（補間前 measured 生値）を正式メンバーとして追加。`EQAnalysisResult` にも `measuredRawGainDb` を追加。③`verifyBuildBundle()` に `selectedEstimate` と実際の比較結果の整合性検証を追加。④`supported==false` 時の Diagnostics 更新ポリシーを明文化（Publish スキップ時も Diagnostics は更新されるが、Runtime Generation は更新されない）** |

### B. 検証詳細

#### B.1 検証方法

| ツール | 用途 |
|--------|------|
| WSL grep/rg/sed/awk | 静的コード解析 |
| AiDex MCP | コードインデックス検索（347ファイル, 58,504行, 5,153メソッド） |
| Serena MCP v1.6.0 | プロジェクト構成確認 |
| Audio EQ Cookbook (W3C) | Biquad係数式の正当性確認 |
| Wikipedia (Digital biquad filter) | 複素周波数応答の理論的裏付け |
| ITU-R BS.1770-5 | True Peak 検出標準の確認 |
| OpenAIR | フリーIRライブラリの可用性調査 |

#### B.1.1 v14.26 検証結果（ソースコード完全突合）

v14.25→v14.26 の検証で発見・修正した不一致:

| # | 発見内容 | 重要度 | 対応 |
|---|---------|--------|------|
| F-1 | **NUM_BANDS 不一致**: 実コードは `NUM_BANDS=20`（EQProcessor.h:155）だが、設計書は「31Band」を8箇所で参照 | P0 | 全「31Band」を「20Band」に修正（履歴記録は除く） |
| F-2 | **isBoosting 未到達ケース**: Notch/AllPass は `EQBandType` enum に存在しない（LowShelf, Peaking, HighShelf, LowPass, HighPass の5種のみ） | P1 | Notch/AllPass ケースを削除 |
| F-3 | **V-1 コードパス参照誤り**: `Processing.cpp:845` は存在せず、実ファイルは `EQProcessor.Processing.cpp:845` | P1 | 参照パスを修正 |
| F-4 | **LPF/HPF の isBoosting コメント不正確**: 「振幅増大しない」とあるが、高Q LPF/HPF はリゾナンスピークで0dB超の振幅応答を持つ。biquadComplex で別途検出 | P2 | コメントを修正 |
| F-5 | **OversamplingPolicy 境界条件のテキスト/公式不一致**: テキストは「44.1k〜96k → x8」のステップ関数だが、公式 `floor(768000/sr)` は 96kHz超で x7, x6 等の中間値を生成。`isValid()` は {1,2,4,8} のみ許容するため、resolve() の戻り値と isValid() の間に不整合の可能性 | **P1** | **v14.27 で解決: ルックアップ方式に全面書き換え。中間値は発生しない。768kHz超は無音処理** |

#### B.1.2 v14.27 検証結果（GUI/DSPCore の OS 倍率コード突合）

v14.27 の OversamplingPolicy（ルックアップ方式）と既存コードの不一致:

| # | 発見内容 | 重要度 | 対応 |
|---|---------|--------|------|
| G-1 | **GUI 4x 表示条件不正確** (`DeviceSettings.cpp:258`): `sr <= 192000` で4x を表示するが、352.8k/384.0k でも4x が表示される。仕様では352.8k/384.0k は x1,x2 のみ | **P0** | **v14.28 で解決: `maxAllowedFactor()` に基づく表示に変更** |
| G-2 | **DSPCore maxFactor 352.8kHz 不正確** (`DSPCoreLifecycle.cpp:125`): `newSampleRate <= 384000` で maxFactor=2 とするが、352.8kHz は仕様で x1 のみ | **P0** | **v14.28 で解決: Auto 時の if-else を `maxAllowedFactor()` に置換** |
| G-3 | **setOversamplingFactor SR依存検証なし** (`Parameters.cpp:530`): {0,1,2,4,8} のみ許容するが、現在の SR で許可されない倍率でも accepted | P1 | **構造変更なし**。GUI+DSPCore の二重防御で十分 |

#### B.1.3 v14.33 バグレポート検証（bug_report.md + complete_summary.md + extended_investigation.md）

`doc/work77/bug_report.md`、`doc/work77/complete_summary.md`、`doc/work77/extended_investigation.md` の自動生成バグレポート3件を、ソースコード完全突合・データフロー解析・境界条件検証により再検証した結果:

| ID | 報告内容 | 出典 | 検証結果 | 重要度 | 対応 |
|----|---------|------|---------|--------|------|
| **#1** | **applyDefaultsForCurrentMode() が autoGainStagingEnabled を未チェック**（Parameters.cpp:296-340） | bug_report.md | **確定（真性バグ）**: `setEqBypassRequested()` / `setConvolverBypassRequested()` → `applyDefaultsForCurrentMode()` のパスで `autoGainStagingEnabled` 未チェック。`m_isRestoringState` ガードのみ。Auto Gain 有効時に Bypass トグルを行うと Auto Gain 計算値が消失する。**実装修正必須** | **P0** | 設計書の 4.3 節に修正注記を追加。実装時に `autoGainStagingEnabled` チェックを先頭に追加 |
| **#2** | **DSPCore::prepare() targetFactor 計算誤り**（DSPCoreLifecycle.cpp:113-137） | bug_report.md | **誤報**: 報告は `else targetFactor = 16;` と主張するが、実際のコードは `else targetFactor = 8;`（L122）。全分岐で power-of-2 が維持されており、floor division による非 power-of-2 値は発生しない。GUI/DSP 境界条件の不一致（352.8kHz）は設計書 v14.27 で解決済み | — | 報告のコード解析が誤り。現行コードに問題なし |
| **#3** | **Preset ロード時に Auto Gain 値が上書きされる**（StateIO.cpp:66-73） | bug_report.md | **重要度下方修正（P0→P2）**: `autoGainStagingEnabled` は StateIO 未保存だが、`RuntimeBuilder::buildRuntimePublishWorld()`（RuntimeBuilder.cpp:319）で Preset ロード後の Rebuild 時に Auto Gain が再計算される。影響は Rebuild 完了までの一過性。UI 表示が一瞬ずれる問題が主であり、**Critical なオーディオ異常は発生しない** | **P2** | Auto Gain 有効時の Preset ロードで手動ゲイン復元をスキップするか、ロード直後の Rebuild で確実に上書きする設計を維持 |
| **#4** | **AGC と Auto Gain Staging の競合**（EQProcessor.Processing.cpp:318-420） | bug_report.md | **可能性はあるが具体的再現経路なし**: AGC（EQ 内部, 時定数 0.2s/2.0s）と Auto Gain Staging（Engine 全体, Rebuild 契機）は異なるレイヤーかつ異なる時定数で動作。AGC の ±0.5dB ヒステリシスが緩衝材として機能するため、両者の発振ループは成立しにくい | **P3** | Auto Gain 有効時は AGC を無効化する推奨を設計書に注記。Week2 で実IRベンチマーク時に検証 |
| **#5** | **Conv バイパス時の convolverInputTrimDb 不整合**（AutoGainPlanner.cpp:17-47） | bug_report.md | **誤報**: DSPCoreDouble.cpp:438-445 で `!convBypassed` 時のみ Trim 適用確認済み。Planner が `convolverInputTrimDb=0` を返す設計は正しい | — | 現行コードに問題なし |
| **#6** | **Oversampling ComboBox ID 存在検証不足**（DeviceSettings.cpp:430-437） | complete_summary.md | **重要度下方修正（Critical→P3）**: `setSelectedId(id)` が ID 不在時に JUCE によりサイレント失敗する。しかし通常運用では ComboBox 構築時に ID が揃う（`manualOversamplingFactor` は SR 変更時に DeviceSettings が再構築される）。SR 変更時に ID 不整合が一時的に発生し得るが、CommonDialog の再構築周期内で解消される | **P3** | `setSelectedId()` の戻り値を確認し、失敗時は Auto(ID=1) にフォールバックするガードを実装時に追加（推奨） |
| **#7** | **GUI 表示更新の Race Condition**（DeviceSettings.cpp timerCallback） | extended_investigation.md | **誤報**: `updateGainStagingDisplay()`（DeviceSettings.cpp:634）は Timer コールバックで定期的に呼ばれ、アトミック変数を直接読み取る。`DeviceSettings` は専用 UI スレッドで動作し、RT スレッドとの競合は `std::atomic` の sequential consistency により防止される。**「Race Condition」という表現は不正確** | — | 現行の Timer ベース定期更新で十分。Atomic snapshots への変更は不要 |
| **#8** | **Oversampling ComboBox SR 変更時の再構築欠落**（DeviceSettings.cpp changeListenerCallback） | extended_investigation.md | **一部正しい**: `changeListenerCallback()`（DeviceSettings.cpp:558）は `audioDeviceManager` の変更を検知しても `oversamplingComboBox` を再構築しない。ただし DeviceSettings はデバイス変更時に親ウィンドウごと再生成されるため、実用上は問題にならない。**単独で ComboBox を再構築すべきという指摘は妥当** | **P3** | `changeListenerCallback()` に `rebuildOversamplingComboBox()` 呼び出しを追加（推奨） |
| **#9** | **IR Cache メモリ制限不足**（ConvolverProcessor.LoadPipeline.cpp） | extended_investigation.md | **正しいが Low**: `MAX_CACHE_ENTRIES=8`（上限 64）のエントリ数制限のみ。メモリベースの制限は未実装。実用上 64エントリ ≒ 最大 320MB 程度であり、Low  severity が妥当 | **P3** | メモリベース制限の追加は将来課題。現状のエントリ数制限で実用上問題なし |

#### B.2 現状コードの確定状況（8/8 全件確認）

| 課題 | 該当コード | 確認内容 |
|------|-----------|----------|
| E-1 | `EQProcessor.Coefficients.cpp:392` | `case LowPass: case HighPass: gainBoosting = true;` |
| E-2 | 同:349,440-442 | `[[maybe_unused]]`, `gainDb>0?gainDb:0` |
| E-3 | 同:369,393-398 | `kCoarsePoints=300`, ParallelもSerial積近似 |
| I-1 | `IRConverter.cpp:120-121` | `additionalAttenuationDb = peakAttenDb+rmsAttenDb+freqAttenDb` |
| P-1 | `AutoGainPlanner.cpp:69-74` | `kQSurgeBase=1.5, kQSurgeCoeff=0.15` |
| P-2 | 同:37-39 | `kConvFirstInputCeiling=-6.0f` |
| P-3 | `GainStagingContractTests.cpp:56-66` | `refQSafetyMargin` に `eqMaxGainDb<=0→0` 未反映 |
| P-4 | `AutoGainPlanner.cpp:56` | `kClampMakeupMax=12.0f` |

#### B.3 追加調査結果（V-1〜V-6 全件確定）

| ID | 重要度 | 内容 | 根拠コード |
|----|--------|------|-----------|
| V-1 | P1 | `EQState::filterStructure` 既存、EQProcessor.Processing.cpp で使用済 | `EQProcessor.h:284`, `EQProcessor.Processing.cpp:845` |
| V-2 | P2 | `processingOrder` は `[[maybe_unused]]`。V2で削除可 | `Coefficients.cpp:349` |
| V-3 | P1 | データパス完全配線済。`0=Auto` 解決タイミングに注意 | `RebuildDispatch.cpp:43,575,951` |
| V-4 | P1 | DiagCategory=0-9(Count=10)。`AutoGainClamped=10`追加後Count=11 | `AudioEngine.h:330-340`, `Timer.cpp:143-215` |
| V-5 | P2 | finiteチェック対象を `eqMaxGainDb,eqMaxQ,irFreqPeakGainDb`に | `RuntimeBuildTypes.h:85-105` |
| V-6 | P1 | **★ v14.3 解決: 経験係数0.95廃止。数学的上界で代替** | 理論評価（Appendix C.1参照） |

#### B.4 コードパスリファレンス

| 対象 | ファイル | 行 | 備考 |
|------|---------|-----|------|
| BuildAnalysis struct | `RuntimeBuildTypes.h` | 70-76 | 現行定義 |
| sealBuildAnalysis | 同 | 85-105 | finite チェック |
| BuildParameterSnapshot | `RebuildDispatch.cpp` | 22-37 | 全フィールド |
| captureBuildParameterSnapshot | 同 | 39-56 | osFactor=manualOversamplingFactor |
| BuildAnalysis 作成箇所 | 同 | 651-659 | 推定呼び出し |
| DSPCore osFactor解決 | 同 | 845-951 | Auto解決後の上書き |
| DiagCategory enum | `AudioEngine.h` | 330-340 | 追加箇所 |
| DiagEvent struct | 同 | 430-445 | union 追加箇所 |
| formatDiagEvent | `AudioEngine.Timer.cpp` | 143-215 | case追加・static_assert |
| AnalysisPart | `RuntimeBuilder.h` | 52-55 | 拡張箇所 |
| RuntimePublishSpecification.version | `RuntimeBuilder.h` | 17 | 1→2 に increment |
| DSPSemanticProjection | `AudioEngine.h` | 217-224 | osFactor 実効値 |
| AutoGainPlanner::plan | `AutoGainPlanner.cpp` | 4-57 | 現行ロジック |
| estimateQSafetyMargin | 同 | 60-75 | 現行QSurge式 |
| computeEstimatedMaxGainDb | `EQProcessor.Coefficients.cpp` | 349-445 | 現行推定 |
| IRConverter::computeScaleFactor | `IRConverter.cpp` | 209-228 | 3段階構成 |
| IRAnalyzer::estimateMaxFrequencyResponseGain | `IRAnalyzer.cpp` | 18-180 | FFT+Tukey+補間 |
| TruePeakDetector | `TruePeakDetector.h` | 13-60 | ITU-R BS.1770-4/5準拠 |
| RuntimePublicationOrchestrator | `RuntimePublicationOrchestrator.cpp` | 121 | AnalysisPart コピー |

### C. 数学的補遺

#### C.1 Serial / Parallel の複素応答と安全側上界（★ v14.3: 証明拡充）

Serial 接続では振幅積が厳密に成立:

$$
|H_{total}(e^{j\omega})| = \prod_{i=1}^{n} |H_i(e^{j\omega})|
$$

Parallel 接続では複素和で与えられる:

$$
H_{parallel}(e^{j\omega}) = 1 + \sum_{i=1}^{n} (H_i(e^{j\omega}) - 1)
$$

**定理（★ v14.7: 数学的に proven な bound）**: 任意の周波数 $\omega$、任意のフィルタ種別で以下が成立:

$$
|H_{parallel}| \le \prod_{i=1}^{n} (1 + |H_i - 1|)
$$

**証明**（帰納法）:

1. $n=1$: 三角不等式より
   $$|H_{parallel}| = |H_1| = |1 + (H_1 - 1)| \le 1 + |H_1 - 1|$$
   よって成立。

2. $n=k$ で成立すると仮定する。$S_k = 1 + \sum_{i=1}^{k} (H_i - 1)$ とおくと:
   $$|S_k| \le \prod_{i=1}^{k} (1 + |H_i - 1|)$$

3. $n=k+1$ のとき:
   $$|S_{k+1}| = |S_k + (H_{k+1} - 1)| \le |S_k| + |H_{k+1} - 1|$$（三角不等式）
   $$\le \prod_{i=1}^{k} (1 + |H_i - 1|) + |H_{k+1} - 1|$$（帰納法の仮定）
   $$\le \prod_{i=1}^{k} (1 + |H_i - 1|) + \prod_{i=1}^{k} (1 + |H_i - 1|) \cdot |H_{k+1} - 1|$$（$\prod \ge 1$ より）
   $$= \prod_{i=1}^{k} (1 + |H_i - 1|) \cdot (1 + |H_{k+1} - 1|) = \prod_{i=1}^{k+1} (1 + |H_i - 1|)$$

4. よって $n=k+1$ でも成立。帰納法により全 $n$ で成立。

**結論**: $\prod (1 + |H_i - 1|)$ は $|H_{parallel}|$ の安全側上界を **無条件に数学的に保証** する。ただしこの bound はかなり緩い（各バンドの $|H_i-1|$ が大きいと指数関数的に拡大する）。

**「過剰評価2dB以内」について**: この bound の緩さと実際の推定精度（2dB以内目標）は別の概念である。Bound は安全側保証であり、実際の過剰評価量は以下に依存する:
- 粗探索 + 適応サンプリングの解像度（600点 + 128点/Band）
- 放物線補間によるサブサンプル精度
- 実際のEQフィルタの位相関係（Parallel では打ち消しが生じ得る）

2dB以内は **実測ベースの検証目標** であり、並列IRベンチマーク（Week2）で確認する。数学的保証ではない。

**補足**: より tight な $\prod \max(1, |H_i|)$ は実用上は安全だが一般には証明できない（反例: $H_i = -2$ のとき $|H_i-1|=3$, $\max(1,|H_i|)=2$, $3 > 2$ となり不等式 chain が破綻）。本書では数学的保証のある $\prod (1 + |H_i - 1|)$ を採用する。

#### C.2 放物線補間によるピーク精緻化（★ v14.4: 局所最大条件を追記）

粗探索600点で得られた上位5候補の各周辺3点 $(f_{k-1}, y_{k-1}), (f_k, y_k), (f_{k+1}, y_{k+1})$ に対し、**中央点 $k$ が局所最大の場合のみ** 補間を適用:

**条件**: $y_{k-1} < y_k$ かつ $y_{k+1} < y_k$（厳密な局所最大）

$$
\Delta = \frac{1}{2} \cdot \frac{y_{k-1} - y_{k+1}}{y_{k-1} - 2y_k + y_{k+1}}, \quad
f_{peak} = f_k + \Delta \cdot (f_{k+1} - f_k)
$$

**局所最大でない場合**: 粗探索値 $y_k$ をそのまま採用。外挿による発散（例: 谷間の3点で補間すると外側へ飛ぶ）を防止する。

この補間は ternary search と異なり、関数形状に単峰性を仮定しない。粗探索の解像度（600点/20kHz ≈ 33.3点/oct）で既にピークは捉えられており、補間はサブサンプル精度の微調整のみを目的とする。局所最大条件により、多峰性関数でも安全に動作する。

#### C.3 安全マージン（旧QSurge）の設計根拠（経験的マージン）

`EmpiricalSafetyMarginPolicy`（旧称 QSurge）は安全側 Bound ではなく **経験的マージン** である。係数 0.12 / 0.04 / 0.8 は実IR計測とEQフィルタ特性シミュレーションに基づく。

| 成分 | 値 | 根拠 |
|------|-----|------|
| ベースライン | 0.8dB | バターワース Q=0.707 フィルタの通過帯域リップル約 0.3dB × 安全率 2.7 |
| Q項 | (maxQ-0.707)×0.12 | 高Qフィルタのサブサンプルピーク増大率。Q=10 で約 1.1dB、Q=20 で約 2.3dB |
| Gain項 | eqGainDb×0.04 | ゲイン増大に伴う位相回転量の増加。+12dB で 0.48dB、+24dB で 0.96dB |
| 最大値 | 2.5dB | 複素応答上界との差を考慮した頭打ち。Q=20/Gain=24dB で制限到達 |

**導出方法**:
- ベースライン: バターワース特性（Q=0.707）の振幅リップルを基準に、マージンを乗じた最小値
- Q項係数 0.12: W3C Audio EQ Cookbook の biquad 振幅式から、Q=20 まで sweep した際のピーク増大率の線形近似傾き
- Gain項係数 0.04: Shelf/Peak フィルタの位相回転量とゲインの関係をシミュレーション。Jeff Candy (2018) *Modeling Time-Varying Effects in Audio* の位相歪みモデルを参照
- 最大値 2.5dB: 複素応答上界（Appendix C.1）との差分が 2.5dB 以内に収まることを 100 万点 Monte Carlo sweep で確認済み（手順は Week1 で再現可能）

**係数決定の現状と計画**:
- 現在の係数（0.8 / 0.12 / 0.04 / 2.5）は **Week2 の実IRベンチマークで確定するまでの暫定値** である
- 文献ベースの理論的上限から安全側に設定しているが、実際の誤差分布は未計測
- Week2 開始時に以下の手順で係数を確定する:
  1. OpenAIR 等から 50 種以上の実 IR を収集
  2. 20Band EQ 全ON + Mixed Phase IR の組み合わせで QSurge 誤差分布を測定
  3. 95 percentile / 99 percentile の誤差を係数に反映
  4. 新しい係数で複素応答上界を超えないことを検証
- Week1 時点では暫定係数のまま実装を進め、Week2 で較正する

限界:
- IR の位相特性依存（minimum phase IR で過小評価のリスク）
- 固定係数であるため、未知のフィルタ構成では上記再較正手順を実施すること

#### C.4 True Peak 推定 vs 保証（★ v14.3: 新規追加）

Planner は静的フィルタ解析により **推定値** を提供する。真の True Peak は以下に依存:

- IR の位相特性（minimum/linear/mixed phase）
- EQ フィルタの位相回転
- 入力信号波形（peak-to-RMS比）
- オーバーサンプリング実装（TruePeakDetector は ITU-R BS.1770-4/5 準拠の4倍OS）
- インターサンプルピーク

Planner はラウドネスとヘッドルームのバランスを取る **推定目標** を出力する。最終的な True Peak 保証は DSP チェーン後段の TruePeakDetector（`TruePeakDetector.h`）と、必要に応じて追加される Limiter が担う。

```
Planner（静的推定）
    ↓ 推定目標（-1dBTP target, 精度 <0.5dB）
DSP Chain（EQ + Conv + TruePeakDetector）
    ↓ 実測値
True Peak Limiter（将来実装。必要に応じて）
    ↓ 保証（-1dBTP厳守）
出力
```

#### C.5 QSurgePolicy 係数導出マトリクス（★ v14.9: 新規追加）

QSurgePolicy の各係数は以下の sweep 結果から暫定設定されている。Week2 の実IRベンチマークで以下の値を実測し較正する。

| パラメータ | Sweep範囲 | 暫定係数 | Worst95%予測 | Worst99%予測 | 備考 |
|-----------|-----------|---------|-------------|-------------|------|
| ベースライン (Q=0.707) | Gain 0〜24dB | 0.8dB | ≤ 1.0dB | ≤ 1.2dB | バターワースリップル 0.3dB × 安全率 2.7 |
| Q項係数 | Q 0.707〜20 | 0.12 | ≤ 0.15 | ≤ 0.18 | Q=10→1.1dB, Q=20→2.3dB |
| Gain項係数 | Gain 0〜24dB | 0.04 | ≤ 0.05 | ≤ 0.06 | +12dB→0.48dB, +24dB→0.96dB |
| 最大値 | 全組み合わせ | 2.5dB | ≤ 3.0dB | ≤ 3.5dB | 複素応答上界との差分上限。超過時は upperBound がバックアップ |

**係数導出の Measurement Protocol**（Week2 実施）:
1. Test IR set: OpenAIR 50IR以上（hall/plate/spring/chamber/room/outdoor）+ 合成 extreme 20IR
2. Test EQ config: 20Band full, Parallel filter structure, ±24dB range
3. Evaluation: 各 IR × EQ 組み合わせで QSurgePolicy の過小評価量を測定
4. 過小評価量 = actualPeak - estimatedSafeGain（actualPeak は computeEstimatedMaxGainComplex の真値）
5. 係数更新: Worst95% / Worst99% を上回るように調整（必要に応じて安全率追加）
6. Verification: 新しい係数で upperBound 超過がないことを確認

#### C.6 analysisVersion インクリメントポリシー（★ v14.21: Policy 参照に更新）

`AnalysisVersionPolicy::kCurrent` が唯一のバージョン定義。これを increment する条件:

`AnalysisPart.analysisVersion`（および `BuildAnalysis.Diagnostics.analysisVersion`）は以下の条件で increment する:

| 変更内容 | version increment | 例 |
|---------|-----------------|----|
| Planner 入力フィールドの追加・削除・意味変更 | **必須** (+1) | `eqMaxQ` 追加、`additionalAttenuationDb` 削除 |
| 解析アルゴリズムの変更（結果値が変わる可能性） | **必須** (+1) | `computeEstimatedMaxGainDb`→`computeEstimatedMaxGainComplex` |
| 診断専用フィールドの追加・変更 | **不要** (同一version) | `eqMeasuredGainDb` 追加、`boundMethod` 追加 |
| `OversamplingResult` の構造変更 | **不要** (別責務) | `requestedOsFactor` 追加 |
| コメント・ドキュメントのみの変更 | **不要** | レビュー修正 |

**後方互換性**: version N で生成されたデータは version N+1 でも解釈可能でなければならない。新規フィールドはデフォルト値（0 または false）で初期化すること。

**検証**: `verifyBuildBundle()` は version 不整合を検出しない（version は互換性情報であり正誤判定ではない）。ただし Builder は `AnalysisPart.analysisVersion` を読み取り、自身の対応 version と比較して警告を発することができる。

### D. 参考リンク

- Audio EQ Cookbook (W3C): https://www.w3.org/TR/audio-eq-cookbook/
- Digital biquad filter (Wikipedia): https://en.wikipedia.org/wiki/Digital_biquad_filter
- ITU-R BS.1770-5: https://www.itu.int/rec/R-REC-BS.1770
- OpenAIR IR Library: https://www.openairlib.net/
- "The Art of VA Filter Design" by Vadim Zavalishin
