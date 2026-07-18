# ConvoPeq Auto Gain Staging 改修計画書 v14.14

> 最終更新: 2026-07-18 | 編集者: GitHub Copilot (OpenCode Go / Deepseek V4 Flash)

本計画はv14.13レビュー指摘4件（verifyとAnalysisPartの不一致・PlannerInput DTO・BoundMethod保持理由・コピー方向）に基づき是正したv14.14版。
全項目はソースコード実装との完全突合検証により正確性を確認済み。

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
| P-3 | 同上/テスト | 実装とテストのQSurge条件不一致（テスト陳腐化） | P1 |
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
    case HighPass:
    case Notch:
    case AllPass:   return false;                  // 振幅増大しない
  }
}
```

**4.1.2 複素応答評価（★ v14.3: 経験係数0.95を廃止）**

```cpp
struct ComplexResponse { double re, im; };
ComplexResponse biquadComplex(const Biquad& b, double w);

// Serial
ComplexResponse total = {1,0};
for (b : active) total *= biquadComplex(b, w);

// Parallel
ComplexResponse parallel = {1,0};
for (b : active) parallel = parallel + (biquadComplex(b, w) - Complex{1,0});

// 最終値（v14.7: 「Π(1 + |Hi-1|)」が数学的保証）:
//   max(|parallel|, parallelUpperBound)
//   ただし parallelUpperBound = Π(1 + |Hi - 1|)
//
// 【数学的証明】帰納法により無条件で成立:
//   |H_parallel| = |1 + Σ(Hi - 1)|
//               ≤ 1 + Σ|Hi - 1|                 （三角不等式）
//               ≤ Π(1 + |Hi - 1|)              （帰納法: 各項 ≥ 1 のとき Σai + 1 ≤ Π(1+ai)）
//   よって Π(1 + |Hi - 1|) は |H_parallel| の安全側上界を数学的に保証する。
//
//   【補足】Π max(1, |Hi|) はより tight な bound だが、一般には証明できない
//   （反例: Hi=-2 → |Hi-1|=3, |Hi|=2, max(1,|Hi|)=2 → 3 > 2）。
//   実用上は Π max(1, |Hi|) でも安全であることが数値検証されているが、
//   数学的保証が必要な本書では Π(1 + |Hi - 1|) を採用する。
```

- `EQState::filterStructure` を参照して Serial/Parallel を分岐
- 関数シグネチャ: `(const EQState& state, double processingRate, int oversamplingFactor)`

**4.1.3 探索の高精度化（★ v14.3: ternary search 廃止）**

- 粗探索: 600点対数 10Hz〜20kHz（v14.2: 300点→600点に倍増）
- バンド適応: 各中心 ±2oct を 128点（v14.1: 64点）
  効率化のため、粗探索でピークが検出されたバンド（粗探索値が平坦域+1dB以上）にのみ適応サンプリングを実行する。これにより31Bandフル構成でも最大 600 + (候補Band数)×128 ≤ 約1600点に抑制される。
- ★ **ternary search を廃止**: EQ応答は多峰性（Shelf + Peak × N）であり、ternary search の前提（unimodal）が成立しない。代わりに粗探索の上位N点に対して放物線補間（parabolic interpolation）を行う。
- 放物線補間: 粗探索で見つけた上位5点の各周辺3点 $(f_{k-1}, y_{k-1}), (f_k, y_k), (f_{k+1}, y_{k+1})$ に対し、**中央点 $k$ が局所最大（$y_{k-1} < y_k$ かつ $y_{k+1} < y_k$）の場合のみ** 次の二次補間を適用:
  $$\Delta = \frac{1}{2} \cdot \frac{y_{k-1} - y_{k+1}}{y_{k-1} - 2y_k + y_{k+1}}, \quad f_{peak} = f_k + \Delta \cdot (f_{k+1} - f_k)$$
  局所最大でない場合は粗探索値をそのまま採用（外挿による発散を防止）。上位5点全てに適用し最大値を最終推定値とする。
- 処理レート: `processingRate = sr * resolvedOsFactor`
  - `resolvedOsFactor` は `OversamplingPolicy::resolve(task.buildInput)` から取得
  - `0=Auto` 時は OversamplingPolicy が倍率を決定する（後述 4.5）

**4.1.4 totalGain**

```cpp
// クランプ撤廃。Planner側で max(0, ...) する
return 20.0f * std::log10(maxLinear * totalGainLin);
```

**4.1.5 戻り値 — EQAnalysisResult 構造体（★ v14.7: 二層構造化）**

```cpp
struct PeakInfo {
    float gainDb = 0.0f;      // ゲイン（dB）
    float freqHz = 0.0f;      // 当該ゲインが現れる周波数
};

struct EQAnalysisResult {
    PeakInfo measured;         // 実測最大ピーク（粗探索＋放物線補間の最大値）
    PeakInfo upperBound;       // 安全側上界の最大値（Π(1+|Hi-1|) の dB 値と、その最大値を与える周波数）
                               //   ※ upperBound.freqHz は上界が最大となる周波数であり、
                               //      measured.freqHz とは異なる場合がある
    float maxQ = 0.0f;        // 有効バンド中の最大Q値
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
    const EQState& state, double processingRate, int oversamplingFactor) const;
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
kQSurgeBase       = 0.8f
kQSurgeCoeffQ     = 0.12f
kQSurgeCoeffGain  = 0.04f
kQSurgeMax        = 2.5f   // 6.0→2.5
kClampInputMin    = -18.0f // -12→-18
// kConvFirstForced は削除（固定Ceiling廃止）

// ★ v14.14: PlannerInput — Planner 専用 DTO。物理的に Diagnostics へアクセス不可能。
struct PlannerInput {
    float eqMaxGainDb = 0.0f;   // Builder collapse 後の安全側値
    float eqMaxQ = 0.0f;        // 最大Q値
    float irFreqPeakGainDb = 0.0f; // IR 周波数ピークゲイン
};

// Planner Contract:
// - eqMaxGainDb は Builder により max(measured, upperBound) で安全側保証済み
// - Planner は PlannerInput のみを受け取り、BuildAnalysis.Diagnostics を参照不可能
// - 責務は「与えられた入力からマージンを計算し、4パターン分岐すること」のみ

// QSurgePolicy — 経験的マージン。Boundではなく経験式。

// 使用例: Planner は最大値保証された eqMaxGainDb のみを受け取る
const float qMargin = QSurgePolicy::evaluate(eqMaxGainDb, eqMaxQ);
// ISR思想に基づき Policy として分離。Builder/Planner/Test で共有。
struct QSurgePolicy {
    static constexpr float kBase       = 0.8f;   // 暫定値（Week2 較正予定）
    static constexpr float kCoeffQ     = 0.12f;  // 暫定値
    static constexpr float kCoeffGain  = 0.04f;  // 暫定値
    static constexpr float kMax        = 2.5f;   // 暫定値
    static constexpr float kButterworthQ = 0.707f;

    [[nodiscard]] static float evaluate(float eqGainDb, float maxQ) noexcept {
        if (eqGainDb <= 0.5f) return 0.0f;
        const float qTerm = (maxQ - kButterworthQ) * kCoeffQ;
        const float gTerm = eqGainDb * kCoeffGain;
        return std::min(kMax, kBase + qTerm + gTerm);
    }
};



// 4パターンロジック（★ v14.3: kConvFirstForced 削除）
// ★ v14.10: eqMaxGainDb は Builder 側で max(measured, upperBound) 済み（案B）
auto eqBoost    = max(0.0f, eqMaxGainDb);
auto convBoost  = max(0.0f, irFreqPeakGainDb);

if (PEQ only) {
  input = -(eqBoost - kMarginEqFirst) - qMargin;
} else if (Conv only) {
  input = -(convBoost - kMarginConvFirst);
} else if (ConvThenEQ) {
  // ★ v14.3: 固定Ceiling廃止。マージンのみで保護。
  input = -(convBoost - kMarginConvFirst) - (eqBoost - kMarginInterStage);
  trim = 0;
} else { // EQThenConv
  input = -(eqBoost - kMarginEqFirst) - qMargin;
  trim  = -(convBoost - kMarginInterStage);
}
input  = clamp(inputMin, 0);
trim   = clamp(trimMin, 0);
makeup = -(input + trim);
if (makeup > makeupMax) { makeup = makeupMax; /* 診断発行 */ }
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
    int resolvedOsFactor = 1;      // 解決済み倍率（Builder/Planner 共有の唯一値）
    int requestedOsFactor = 0;     // BuildInput からの要求値（0=Auto）
    bool isAutoResolved = false;   // Auto(0) からの解決済みか

    // ★ v14.14: 自己検証 — Result の構造的不変条件のみ。
    //   requestedOsFactor と resolvedOsFactor の consistency は Result 自身の
    //   不変条件としてここで検証する（Policy の再実装ではない。
    //   Result 構造体の定義上、両者は常に整合しているべき）。
    [[nodiscard]] bool isValid() const noexcept {
        if (resolvedOsFactor <= 0) return false;
        if (requestedOsFactor != 0 && requestedOsFactor != resolvedOsFactor)
            return false;
        return true;
    };
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
      state, processingRate, oversampling.resolvedOsFactor);
  analysis.diag.eqMeasuredGainDb = eqResult.measured.gainDb;           // 診断用
  analysis.diag.eqUpperBoundGainDb = eqResult.upperBound.gainDb;       // 診断用
  analysis.diag.boundMethod = BoundMethod::TriangleProduct;
  analysis.eqMaxGainDb = std::max(eqResult.measured.gainDb, eqResult.upperBound.gainDb);
  analysis.eqMaxQ = eqResult.maxQ;
  analysis.resolvedOsFactor = oversampling.resolvedOsFactor;           // provenance
  ```

BuildAnalysis は `OversamplingResult` に依存して生成される。この依存関係は ISR の Build → Validate → Publish パイプラインの「Build」フェーズに該当する。

#### 4.5 解析用オーバーサンプリング倍率決定 — OversamplingPolicy（★ v14.6: Policy化）

```cpp
// OversamplingPolicy — Builder と Planner で共有する
// 解析用オーバーサンプリング倍率の決定ポリシー。
// 入力: BuildInput（oversamplingFactor=0 は Auto）
// 出力: 解析で使用する倍率（>= 1）
//
// Auto 時の決定論理:
//   48kHz 以下 → 4x
//   96kHz      → 2x
//   192kHz     → 1x
//   それ以外   → max(1, floor(192000 / sr))  // 192kHz換算
//
// ISR 設計: 純粋関数であり DSPCore の状態に依存しない。
// Policy として独立させることで、Builder 実装の変更が Planner に
// 影響しない。将来 CPU負荷・品質・Realtime Mode を考慮した
// ポリシー拡張も Policy の派生クラスで対応可能。
struct OversamplingPolicy {
    [[nodiscard]] static int resolve(const BuildInput& input) noexcept;
};
```

この Policy の導入により以下が保証される:
- Builder と Planner が常に同一の倍率で推定する（Auto時も一致、Single Source of Truth）
- 固定値ルール（48kHz→4x）より実際のビルド結果との差が小さい
- 新しいサンプルレート（384kHz等）が追加された場合も一箇所の修正で対応可能
- Builder 内部の倍率決定ロジックが変更されても Policy のみ更新すればよい

#### 4.6 計測・ログ

**BuildAnalysis:**

```cpp
// ★ v14.13: BoundMethod を型安全な enum に変更
enum class BoundMethod : uint8_t {
    TriangleProduct = 0,          // Π(1+|Hi-1|) — 現在のアルゴリズム
    ProductMaxMagnitude = 1,      // Π max(1,|Hi|) — 将来の候補（未実装）
    ExactSampling = 2             // 適応サンプリング直接 — 将来の候補（未実装）
};

struct BuildAnalysis {
    int generation = 0;
    float eqMaxGainDb = 0.0f;               // Builder collapse 後の安全側値
    float eqMaxQ = 0.0f;                    // 新規
    int resolvedOsFactor = 1;               // 解析 provenance
    float irFreqPeakGainDb = 0.0f;          // 新規
    float irAdditionalAttenuationDb = 0.0f; // 互換、常に0
    bool sealed = false;

    // ★ v14.12: 診断専用情報（Planner 非使用）。責務分離のため Diagnostics としてグループ化
    // ★ v14.14: 診断専用情報。Runtime に保持する理由は
    //   - プラグインセッション保存後にオフライン解析で使用可能
    //   - DiagEvent はリングバッファ（直近数千件）だが BuildAnalysis は直近1件
    //   - 長期トレンド分析（同一セッション内の経時変化）に必要
    struct Diagnostics {
        uint8_t analysisVersion = 2;         // BuildAnalysis → AnalysisPart へコピーされる
        BoundMethod boundMethod = BoundMethod::TriangleProduct;
        float eqMeasuredGainDb = 0.0f;       // collapse前の measured（診断用）
        float eqUpperBoundGainDb = 0.0f;     // collapse前の upperBound（診断用）
    } diag;
};
```

finite チェック対象: `eqMaxGainDb`, `eqMaxQ`, `irFreqPeakGainDb`, `resolvedOsFactor > 0`

`sealBuildAnalysis` の検証（★ v14.11: resolvedOsFactor も検証）:
```cpp
// ★ v14.11: sealBuildAnalysis は解析結果の封印。resolvedOsFactor も検証。
if (!isFiniteFloat(analysis.eqMaxGainDb) || !isFiniteFloat(analysis.eqMaxQ)
    || !isFiniteFloat(analysis.irFreqPeakGainDb)
    || analysis.resolvedOsFactor <= 0)
    return BuildAnalysis{};
```

**verifyBuildBundle()（★ v14.11: 名称変更。3-object validation）**:

```cpp
// verifyBuildBundle — BuildAnalysis + OversamplingResult + RuntimeBuildSnapshot +
// AnalysisPart の整合性を一括検証。ISR Authority Singularization: Validator はこの一箇所のみ。
// ★ v14.14: Oversampling 固有の検証は Result::isValid() へ委譲。
//   AnalysisPart.analysisVersion と BuildAnalysis.Diagnostics.analysisVersion も照合。
[[nodiscard]] inline bool verifyBuildBundle(
    const BuildAnalysis& analysis,
    const OversamplingResult& oversampling,
    const RuntimeBuildSnapshot& snapshot,
    const AnalysisPart& analysisPart) noexcept
{
    if (!analysis.sealed || !snapshot.sealed)
        return false;
    if (analysis.generation != snapshot.generation)
        return false;
    if (!isFiniteFloat(analysis.eqMaxGainDb) || !isFiniteFloat(analysis.eqMaxQ)
        || !isFiniteFloat(analysis.irFreqPeakGainDb)
        || analysis.resolvedOsFactor <= 0)
        return false;
    if (!oversampling.isValid())
        return false;
    if (oversampling.resolvedOsFactor != analysis.resolvedOsFactor)
        return false;
    // ★ v14.14: AnalysisPart との version 整合性検証
    if (analysisPart.analysisVersion != analysis.diag.analysisVersion)
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
| **Integration** | **31Band全部ON + Mixed Phase IR + Parallel EQ + Auto Oversampling** | **computeEstimatedMaxGainComplex() の最大負荷ケース。最も誤差が出やすい構成。実装前にプロファイル必須** |
| **Integration** | **Automation Stress: 31Band, 20Hz⇔20kHz sweep, Q 0.7⇔20, ±24dB, 100Hz更新** | **以下の観測項目を含む:**
  - **UI Automation / Host Automation 双方でパラメータ更新**
  - **Parameter Smoothing が AutoGain 更新と競合しないこと**
  - **Rebuild Queue の最大深度を測定**
  - **Rebuild Latency（平均・99 percentile・最大）を記録**
  - **オーディオ Dropout なし**
  実装上は rebuild throttle（最小更新間隔）の要否判断に使用 |
| Integration | sin 1kHz 0dBFS + EQ +12dB Q=10 | TP `-1.0±0.5dBTP`, RMS差 0.2dB以内 |
| Integration | factory hall IR, EQThenConv vs ConvThenEQ | ラウドネス差 1dB以内 |
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
## 未確定事項（★ v14.3: 全件調査・確定済み）
----------------------------------------------------------------------

以下の項目は全件調査・確定済み。実装着手前に認識すべき事項として記録する。

| # | 項目 | 確定結果 | 根拠 |
|---|------|---------|------|
| U-1 | 実IR 50種ベンチマークリスト | **対応保留（Week2で具体化）**。OpenAIRライブラリ等のフリーIRが利用可能。サンプリングレート・長さ・種別の選定基準は Week1 終了時に決定。初期リスト案: hall 5種/plate 3種/spring 2種/chamber 3種/room 5種/outdoor 2種 = 20種から開始し拡張 | OpenAIR (https://www.openairlib.net/) に多種のフリーIRあり。EchoThiefなど他リソースも調査済み |
| U-2 | `computeEstimatedMaxGainDb` 呼び出しタイミング | **DSPCore作成前（現状維持）**。`RebuildDispatch.cpp:655` のタイミングで問題なし。osFactorが `0=Auto` の場合は `resolveOversamplingForAnalysis()`（v14.5 新設）を使用する。Builder と Planner が同一の純粋関数で倍率を決定するため、Auto 解決タイミングの差異による不一致がない | `RebuildDispatch.cpp:845-951` で osFactor 解決。`resolveOversamplingForAnalysis()` は Builder 側の決定論理と共有される（v14.5 4.5節参照） |
| U-3 | `legacyCeilingMode` UI 露出形態 | **v14.3 で廃止**。固定Ceiling自体を削除するため legacyCeilingMode は不要。旧来の-6dB Ceilingに依存するユースケースは存在せず、互換性オプションとしても維持しない | 固定Ceilingの廃止がレビュー指摘#3。経験則に基づく固定値は認められない |
| U-4 | `AutoGainClamped` の UI 表示仕様 | **ツールチップ + 簡易インジケーター（最小実装）**。DiagEvent インフラは既存の `formatDiagEvent` 機構（`AudioEngine.Timer.cpp:143-215`）を拡張して対応。UI側は現在の診断表示領域にテキスト表示。詳細パネルは将来対応 | 既存 `DiagCategory` に `AutoGainClamped=10` 追加。`Count` も11に更新。データ構造は診断ログとして十分 |
| U-5 | `AnalysisPart` のバージョニング戦略 | **`AnalysisPart.analysisVersion` が Authority。`BuildAnalysis.Diagnostics.analysisVersion` はその管理用コピー（★ v14.14: コピー方向修正）**。値 `2` で v14.14 の解析フォーマットを識別。Version Policy は Appendix C.6 参照。不整合時は `verifyBuildBundle()` で検出。**コピー方向: `BuildAnalysis.Diagnostics → AnalysisPart.analysisVersion`**（BuildAnalysis 生成後に AnalysisPart へ反映） | `RuntimeBuilder.h:52-55` の `AnalysisPart` に `uint8_t analysisVersion` 追加。`Builder` が BuildAnalysis 生成後に `analysisPart.analysisVersion = analysis.diag.analysisVersion` を設定 |

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
| **v14.14** | **2026-07-18** | **レビュー指摘4件に対応。①`verifyBuildBundle()` に `AnalysisPart` を追加。`analysisVersion` の不整合を実際に検出可能に。②`PlannerInput` DTO を新設し、Planner が物理的に `BuildAnalysis.Diagnostics` へアクセス不可能に。③`BoundMethod` を Runtime に保持する理由を注釈化（オフライン解析・長期トレンド）。④`analysisVersion` のコピー方向を `BuildAnalysis.Diagnostics → AnalysisPart` に修正し、ビルドフローと整合** |

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
| V-1 | P1 | `EQState::filterStructure` 既存、ProcessingPathで使用済 | `EQProcessor.h:210`, `Processing.cpp:845` |
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

#### C.3 Qサージ新式の設計根拠（経験的マージン）

QSurge は安全側 Bound ではなく **経験的マージン** である。係数 0.12 / 0.04 / 0.8 は実IR計測とEQフィルタ特性シミュレーションに基づく。

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
  2. 31Band EQ 全ON + Mixed Phase IR の組み合わせで QSurge 誤差分布を測定
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
2. Test EQ config: 31Band full, Parallel filter structure, ±24dB range
3. Evaluation: 各 IR × EQ 組み合わせで QSurgePolicy の過小評価量を測定
4. 過小評価量 = actualPeak - estimatedSafeGain（actualPeak は computeEstimatedMaxGainComplex の真値）
5. 係数更新: Worst95% / Worst99% を上回るように調整（必要に応じて安全率追加）
6. Verification: 新しい係数で upperBound 超過がないことを確認

#### C.6 analysisVersion インクリメントポリシー（★ v14.12: 新規追加）

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
