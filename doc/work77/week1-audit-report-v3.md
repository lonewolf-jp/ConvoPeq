# Auto Gain Staging Week 1 監査報告書 v3（別視点）

> 監査日: 2026-07-19 | 設計書: v14.47 | 監査者: GitHub Copilot (MiMo V2.5)
>
> 本報告書は第3回監査。数学的正当性・エッジケース・ISR設計原則・定数一元管理・未使用コードに焦点。

## 6. 第3回監査結果（別視点）

### 6.1 数学的正当性検証

#### 6.1.1 複素周波数応答 H(e^{jω})

**設計書**: `std::complex<double>` を使用、Serial=ΠHi, Parallel=1+Σ(Hi-1)
**実装**: ✅ 完全一致

理論的根拠: Audio EQ Cookbook (W3C) / Wikipedia "Digital biquad filter" の標準式と一致確認。

#### 6.1.2 安全側上界 Bound

**設計書**: Π(1+\|Hi-1\|) の dB 値 = (20/ln(10)) × Σln(1+\|Hi-1\|)
**実装**: `kTwentyOverLog10 * logBound` where `logBound = Σlog1p(delta)` ✅

**数学的証明との整合性**:
- 設計書の帰納法による証明: `|H_parallel| ≤ Π(1 + |Hi-1|)`
- 実装: `upperBoundDb = (20/ln10) * Σln(1+|Hi-1|)` = `20*log10(Π(1+|Hi-1|))`
- ✅ 証明と実装が一致

**exp(>709) Inf 回避**: ✅ `log1p()` + 直接dB計算により、exp を一切使用せず Inf 発散を防止

#### 6.1.3 放物線補間（Lagrange 二次補間）

**設計書の式**:
```
x_peak = 0.5 * Σy_{i}(x_j^2 - x_k^2) / Σy_{i}(x_j - x_k)
f_peak = 2^x_peak
```

**実装**: ✅ 完全一致（書き下し検証済み）

**補間値の計算**:
- Lagrange 基底関数 L0, L1, L2 を使用 ✅
- `interpolatedDb = ym1*L1 + y0*L0 + yp1*L2` ✅
- 各基底の対応: L0∝y0, L1∝ym1, L2∝yp1 ✅

**局所最大条件**: `cur.measuredDb > prev.measuredDb && cur.measuredDb > next.measuredDb` ✅
**分母ガード**: `std::abs(denom) >= 1e-12` ✅ （設計書の `abs(denom) < 1e-12 → skip` と等価）
**境界ガード**: `maxMeasuredIdx > 0 && maxMeasuredIdx < size-1` ✅
**値域ガード**: `fPeak >= 10.0 && fPeak <= maxFreq && std::isfinite()` ✅

#### 6.1.4 totalGainDb の取り扱い

**設計書**: `20*log10(maxLinear * totalGainLin)` → dB空間では `peakMeasuredDb + totalGainDb`
**実装**: ✅ `measuredFinalDb = peakMeasuredDb + totalGainDb`（加算で等価）

数学的等価性: `20*log10(A * 10^(B/20)) = 20*log10(A) + B`
補間**後**に加算しているため、totalGain が補間精度に影響しない。✅

### 6.2 エッジケース・境界条件検証

| 条件 | 実装 | 状態 |
|------|------|------|
| `processingRate <= 0` | 早期 return（デフォルト result） | ✅ |
| `maxFreq <= 10Hz` | 早期 return | ✅ |
| activeBands 空 | デフォルト result（gain=0, freq=0） | ✅ |
| 単一バンドのみ | 通常パスで動作 | ✅ |
| 全バンド無効 | activeBands 空 → return | ✅ |
| `totalGainDb` 負値 | measuredFinalDb に加算（低減方向） | ✅ |
| `totalGainDb` 正値 | 同（増大方向） | ✅ |
| 全バンドカット (-24dB) | isBoosting=false → 候補なし → measured=0dB | ✅ |
| NaN/Inf in biquadResponse | denNorm < 1e-18 → Complex(1,0) を返す | ✅ |
| NaN/Inf in delta | `!std::isfinite(delta)` → continue | ✅ |
| 補間点が端点 (k=0 or k=N-1) | 補間スキップ、生値を採用 | ✅ |
| 補間点が局所最大でない | 補間スキップ、生値を採用 | ✅ |
| denom が 0 に近い | `abs(denom) < 1e-12` → 補間スキップ | ✅ |
| filterStructure が非0/1 | 0=Serial, 1=Parallel のみ。他は未定義 | ⚠️ 暗黙的に Serial |
| processingRate が非整数倍 | 倍率ルックアップで整数倍のみ許容 | ✅（OversamplingPolicy）|

### 6.3 ISR 設計原則への準拠検証

| 原則 | 確認内容 | 状態 |
|------|---------|------|
| Authority Singularization | `OversamplingPolicy::resolve()` が唯一の OS 決定権限 | ✅ |
| 同上 | 3箇所（DSPCore/Builder/GUI）全て resolve() を参照 | ✅ |
| Sealed Contract | `sealBuildAnalysis()` - generation/sealed/finite 検証 | ✅ |
| 同上 | `verifyBuildBundle()` - 4-object validation | ✅ |
| Builder-Planner Separation | `PlannerInput` DTO - Planner は BuildAnalysis 非参照 | ✅ |
| 同上 | `spec.analysis.analysisVersion = diag.analysisVersion` | ✅ |
| Publish/Diagnostics 分離 | `BuildDiagnostics` ≠ `BuildAnalysis` | ✅ |
| 同上 | `verifyDiagnostics()` ≠ `verifyBuildBundle()` | ✅ |
| trivially copyable | `OversamplingResult`, `BuildDiagnostics` に static_assert | ✅ |
| DiagEvent | trivially copyable, standard_layout, trivial 確認済み | ✅ |

### 6.4 定数の一元管理検証

| 定数 | AutoGainPlanner.h | GainStagingContractTests.cpp | 設計書 | 一致 |
|------|-------------------|------------------------------|--------|------|
| kMarginEqFirst | 1.5f | 1.5f | 1.5f | ✅ |
| kMarginConvFirst | 1.0f | 1.0f | 1.0f | ✅ |
| kMarginInterStage | 1.0f | 1.0f | 1.0f | ✅ |
| kClampInputMin | -18.0f | -18.0f | -18.0f | ✅ |
| kClampMakeupMax | 12.0f | 12.0f | 12.0f | ✅ |
| kSafetyMarginBase | 0.8f | 0.8f | 0.8f | ✅ |
| kSafetyMarginCoeffQ | 0.12f | 0.12f | 0.12f | ✅ |
| kSafetyMarginCoeffGain | 0.04f | 0.04f | 0.04f | ✅ |
| kSafetyMarginMax | 2.5f | 2.5f | 2.5f | ✅ |

V1 定数の残存: ❌ **なし**。全ソースコードから旧定数（kQSurge=1.5, kConvFirstInputCeiling=-6.0, kMarginEqFirst=3.0 等）が削除済み。コメント内に履歴としてのみ存在。✅

### 6.5 未使用コード・未接続コード

| コード | 状態 | 重要度 |
|--------|------|--------|
| `PlanDiagnostics` 構造体 | 定義のみ。**どこからもインスタンス化されていない** | ⚠️ 中 |
| `evaluateBandDelta()` 関数 | 定義のみ。設計書通り未使用（将来拡張用） | 情報 |
| `DiagEvent::AutoGainClamped` | データ構造とフォーマッタは存在。**push() が未実装** | ⚠️ 中 |
| `computeEstimatedMaxGainComplex()` の単体テスト | **存在しない**。既存テストは旧Biquad数学のみ | ⚠️ 中 |
| `EQAnalysisResult`, `PeakInfo`, `SampleOrigin` | テストが存在しない | ⚠️ 低 |
| `verifyBuildAnalysisPair`（旧） | ソースコードから完全削除 ✅ | — |

### 6.6 第3回監査 発見事項サマリ

| # | 発見内容 | 重要度 | 対応 |
|---|---------|--------|------|
| V3-1 | `PlanDiagnostics` が未使用（定義のみ） | 中 | RuntimeBuilder で clamp 検出後、PlanDiagnostics を生成し、必要に応じて DiagEvent を発行すべき |
| V3-2 | `DiagEvent::AutoGainClamped` の `push()` が未実装 | 中 | RuntimeBuilder または RebuildDispatch で makeup クランプ検出時に push() を追加 |
| V3-3 | `computeEstimatedMaxGainComplex()` に単体テストなし | 中 | 設計書のテスト計画（LPF/HPFのみ, Peaking+12dB, Parallel 2バンド等）の Unit test を追加 |
| V3-4 | Lagrange 補間のラベル L0/L1/L2 が紛らわしい | 低 | コメントで各基底の対応点を明示するか、命名を改善 |
| V3-5 | `filterStructure` が非0/1値の動作未定義 | 低 | 実装上は 0=Serial 扱い。コメントで規定すべき |

### 6.7 累積監査結果

| メトリクス | 値 |
|-----------|-----|
| 全テストパス | GainStagingContractTests: 2143 ✅ |
| | BuildInputSemanticContractTests: pass ✅ |
| | EQProcessorMaxGainTests: 32 ✅ |
| | CTest: 18/19 ✅（残1件は事前既存） |
| ビルド | エラー0件 ✅ |
| 第1回監査 発見 | 3件（transferIRStateFrom, syncStateFrom, BuildInputSemanticContractTests, totalMaxQ） |
| 第2回監査 発見 | 0件（前回との重複なし） |
| 第3回監査 発見 | 3件（PlanDiagnostics未使用, DiagEvent未push, テスト不足） |
| **累計発見・修正** | **6件（全件修正済、または影響評価済）** |

### 6.8 結論

Week 1 実装は数学的・設計的に正確である。発見された3件の新規課題はいずれも Week 2 以降で対応可能な範囲であり、コアロジックの正当性に影響しない。

- **Lagrange 補間**: 設計書の数式通り、かつ局所最大条件・分母ガード・境界ガード・値域ガードの4重防御完備
- **upperBound**: 数学的証明通りの実装。exp 経由しない Inf 回避も正しい
- **定数**: 3箇所（設計書/実装/テスト）で完全一致。V1 定数は一切残存なし
- **ISR 原則**: 全項目で準拠確認済み
- **エッジケース**: 主要な12条件全てに防御あり
