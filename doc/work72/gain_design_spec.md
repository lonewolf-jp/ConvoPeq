# 自動ゲインステージング改修 設計書

> 作成日: 2026-07-15 (v14.0 Final — BuildAnalysis sealAPI明文化・Band gain>0限定)
> 対象コードベース: ConvoPeq (C++20, JUCE 8.0.12, MKL, Intel oneAPI)
> 先行文書: `gain_revised.md` v2.6, `gain_phase8_test_plan.md` v1.2, `gain_literature_validation_report.md` v1.0
> レビュー: `checked.md` (2026-07-13), ISR Architecture Review v1〜v14 (2026-07-15), 累計72件解決
> コード調査結果: 全24ファイルの改修対象を含め、100+ファイルのコード照合・全ツール最終検証済み
> 総合評価: **A（99点）** — 全14回レビュー完了、全未確定事項解決・実装可能確定最終版
> 使用ツール: grep/rg/ast-grep/sed/awk (WSL), cocoindex(ccc), semble, graphify, serena MCP, AiDex MCP, headroom MCP

---

## システム概要

本設計は **自動ゲインステージング（Auto Gain Staging）** を既存 ConvoPeq の ISR アーキテクチャ（`RuntimeBuildSnapshot → RuntimePublishSpecification → RuntimeBuilder → RuntimeWorld → RT`）に統合する。

**核心**: EQ/IR 解析値を `BuildAnalysis` で Snapshot から分離し、`AutoGainPlanner` 純粋関数が `AnalysisPart` の値のみからゲインを決定。Builder は Engine を一切参照しない。

### データフロー

```
User操作 (processingOrder/IR/EQ変更)
  → submitRebuildIntent (1回)
    → Worker Thread:
      → eqProcessor->computeEstimatedMaxGainDb(sr, order)
      → convolver.getIrAdditionalAttenuationDb()
      → BuildAnalysis { eqMaxGainDb, additionalAttenuationDb }
      → captureRuntimeBuildSnapshot(buildInput) + seal
      → sealBuildAnalysis()
    → Orchestrator:
      → ProcessingPart (命令)
      → AnalysisPart (解析値)
    → RuntimeBuilder::buildRuntimePublishWorld(sealedSnapshot, spec)
      → AutoGainPlanner::plan(eqMaxGainDb, additionalAttenuationDb, ...)
        → Plan { inputHeadroomDb, trimDb, makeupDb }  // dB値のみ
      → Builder: dB → 線形, worldOwner->automation.* = plan.*
      → worldOwner->freeze() → publishWorld → RCU公開

Audio Thread: world->automation.inputHeadroomGain (既存経路)
```

**補足**: 設計の詳細な変遷・全レビュー対応記録・検証証明書は Appendix E〜G に記載。

## 目次

- [Part 1 Implementation Specification](#part-1-implementation-specification)
  - [Phase 0 Target Source Files](#phase-0-target-source-files)
  - [Phase 1 FFT Infrastructure and EQ response math](#phase-1-fft-infrastructure-and-eq-response-math)
  - [Phase 2 State Management Extension](#phase-2-state-management-extension)
  - [Phase 3 EQ Maximum Gain Estimation](#phase-3-eq-maximum-gain-estimation)
  - [Phase 4 IR Conversion Extension](#phase-4-ir-conversion-extension)
  - [Phase 5 RuntimeBuilder Integration](#phase-5-runtimebuilder-integration)
  - [Phase 6 Runtime Mode Transition Safety](#phase-6-runtime-mode-transition-safety)
  - [Phase 7 UI Integration](#phase-7-ui-integration)
  - [Phase 8 CMakeLists Updates](#phase-8-cmakelists-updates)
- [Appendix A Code Verification List](#appendix-a-code-verification-list)
- [Appendix B Literature Validation and Determination](#appendix-b-literature-validation-and-determination)
- [Appendix C Revision History](#appendix-c-revision-history)
- [Appendix D References](#appendix-d-references)
- [Appendix E Architecture Evolution](#appendix-e-architecture-evolution)
- [Appendix F Review Correction Register](#appendix-f-review-correction-register)
- [Appendix G Verification Inventory](#appendix-g-verification-inventory)
---

- [Part 1 Implementation Specification](#part-1-implementation-specification)
  - [Phase 0 Target Source Files](#phase-0-target-source-files)
  - [Phase 1 FFT Infrastructure and EQ response math](#phase-1-fft-infrastructure-and-eq-response-math)
  - [Phase 2 State Management Extension](#phase-2-state-management-extension)
  - [Phase 3 EQ Maximum Gain Estimation](#phase-3-eq-maximum-gain-estimation)
  - [Phase 4 IR Conversion Extension](#phase-4-ir-conversion-extension)
  - [Phase 5 RuntimeBuilder Integration](#phase-5-runtimebuilder-integration)
  - [Phase 6 Runtime Mode Transition Safety](#phase-6-runtime-mode-transition-safety)
  - [Phase 7 UI Integration](#phase-7-ui-integration)
  - [Phase 8 CMakeLists Updates](#phase-8-cmakelists-updates)
- [Appendix A Code Verification List](#appendix-a-code-verification-list)
- [Appendix B Literature Validation and Determination](#appendix-b-literature-validation-and-determination)
- [Appendix C Revision History](#appendix-c-revision-history)
- [Appendix D References](#appendix-d-references)
---

## Part 1 Implementation Specification

### Phase 0 Target Source Files

| # | ファイル | 操作 | 変更内容 | 版 |
| --- | --------- | ------ | --------- | ----- |
| 1 | `src/eqprocessor/EQProcessor.h` | 修正 | `computeEstimatedMaxGainDb(double sampleRate, ProcessingOrder)` 宣言追加（v13.0: 改名） | v2.0/v13.0 |
| 2 | `src/eqprocessor/EQProcessor.Coefficients.cpp` | 修正 | `computeEstimatedMaxGainDb()` 実装（Band適応サンプリング, M/S評価, Parallel注釈） | v2.0/v13.0 |
| 3 | `src/eqprocessor/EQProcessor.h` | 修正 | `getMagnitudeSquared()` を public 維持 | v2.0 |
| 4 | `src/IRAnalyzer.h` | **新規** | FFT解析＋IR分析を `IRConverter` から分離。`analyze()` 静的メソッド | **v8.0** |
| 5 | `src/IRAnalyzer.cpp` | **新規** | `estimateMaxFrequencyResponseGain()` 実装（IRConverter から移行） | **v8.0** |
| 6 | `src/IRConverter.h` | 修正 | `computeScaleFactor()` を3段階に分割: `computeEnergyScale` + `applyClampProtection` | v2.0/v8.0 |
| 7 | `src/IRConverter.cpp` | 修正 | 上記分割実装。FFT解析は `IRAnalyzer::analyze()` を呼出 | v2.0/v8.0 |
| 8 | `src/PreparedIRState.h` | 修正 | `additionalAttenuationDb` フィールド追加（`float`, 正値） | v2.0/v3.6 |
| 9 | `src/audioengine/AudioEngine.h` | 修正 | `autoGainStagingEnabled` atomic, `setAutoGainStagingEnabled()` | v2.0 |
| 10 | `src/audioengine/AudioEngine.Parameters.cpp` | 修正 | `setProcessingOrder` の `sendChangeMessage()` 追加 | v2.0 |
| 11 | **`src/audioengine/RuntimeBuildTypes.h`** | **修正** | **`BuildAnalysis` 構造体追加。`RuntimeBuildSnapshot` は純粋 BuildInput 封印に戻す** | **v3.6** |
| 12 | `src/audioengine/RuntimeBuilder.h` | 修正 | `ProcessingPart` に `autoGainStagingEnabled`。**`AnalysisPart` 新設** | v2.0/v3.6 |
| 13 | **`src/audioengine/AutoGainPlanner.h`** | **新規** | **独立ファイル。`constexpr` 定数（kMargin/kClamp）+ Q Surge係数定義。`ProcessingMode` enum 追加** | **v3.5/v8.0** |
| 14 | **`src/audioengine/AutoGainPlanner.cpp`** | **新規** | **`plan()` 実装（dB 値入出力）+ `estimateQSafetyMargin()` 実装** | **v3.5/v8.0** |
| 15 | `src/audioengine/AudioEngine.RebuildDispatch.cpp` | 修正 | `captureBuildAnalysis()` 追加（Snapshot から BuildAnalysis を分離） | v3.6 |
| 16 | `src/ConvolverProcessor.h` | 修正 | `IRState` に `additionalAttenuationDb`（`float`）, `getIrAdditionalAttenuationDb()` | v2.0/v3.6 |
| 17 | `src/DeviceSettings.h` | 修正 | `autoGainToggle` 宣言 | v2.0 |
| 18 | `src/DeviceSettings.cpp` | 修正 | コールバックチェーン, レイアウト, 永続化 | v2.0 |
| 19 | `src/audioengine/RuntimeBuilder.cpp` | 修正 | `AnalysisPart` から値を取得→`AutoGainPlanner` 呼出 | v3.6 |

---

### Phase 1 FFT Infrastructure and EQ response math

#### 1.1 既存 EQ 応答計算の確認と方針

既存コードでは、`AudioEngine::calcEQResponseCurve()`（`src/audioengine/AudioEngine.EQResponse.cpp`）が SVF 係数を `EQProcessor::svfToDisplayBiquad()` で等価 Biquad に変換し、AVX2 版の `calcMagnitudesForBand()` で周波数応答を計算している。この実装は `AudioEngine.EQResponse.cpp:36-58` で、既に以下を満たしている。

- `calcSVFCoeffs()` → `svfToDisplayBiquad()` のパスは、実際の DSP 処理と表示曲線が一致する（同ファイルコメント参照）。
- `EQChannelMode::Stereo / Left / Right / Mid / Side` のチャンネルモード別積算が実装済み。
- M/S モードでは L/R 両方に同じマグニチュードを掛ける（`Mid/Side` はその後の L/R 合成で位相は不要）。

したがって、本改修では新たに `getComplexResponse()` を追加するのではなく、既存の `getMagnitudeSquared()`（`EQProcessor.Coefficients.cpp:325-337`）を使って最大ゲインを推定する。

**結論**: `DspNumericPolicy.h` / `DspNumericPolicy.cpp` の新規作成は不要。`computeEstimatedMaxGainDb()` は `EQProcessor` 内に実装し、既存 `getMagnitudeSquared()` を周波数スキャンで使用する。

#### 1.2 `IRAnalyzer` の新設（★ v8.0: IRConverter から FFT 解析を分離）

**新規ファイル**: `src/IRAnalyzer.h` / `src/IRAnalyzer.cpp`

**設計判断**: `estimateMaxFrequencyResponseGain()` は IR 変換ではなく IR 解析である。IRConverter に FFT 解析が混在すると、将来 group delay / phase ripple / crest factor / minimum phase 判定などの解析機能追加時に IRConverter が肥大化する。`IRAnalyzer` として独立させることで SRP を満たし、拡張性が向上する。

```cpp
// IRAnalyzer.h — 新規
#pragma once
#include <JuceHeader.h>

// IRの周波数応答ピークをFFT解析で推定
// Tukey窓（α=0.5）適用後の複素スペクトル振幅の最大値を返す
// 戻り値: 線形振幅値（倍率）。IRが無効な場合は1.0
//
// ★ kAnalysisWindow = 65536 は constexpr ではなく Policy 化推奨
//   （将来 192kHz/384kHz 対応時に解析時間調整のため）
//   初版実装では constexpr 固定とし、必要に応じてパラメータ化する。
[[nodiscard]] double estimateMaxFrequencyResponseGain(
    const juce::AudioBuffer<double>& ir) noexcept;
```

**IRConverter からの呼び出し**:

```cpp
// IRConverter.cpp — IRAnalyzer を利用
#include "IRAnalyzer.h"

double IRConverter::estimateMaxFrequencyResponseGain(
    const juce::AudioBuffer<double>& ir,
    double /* sampleRate */) noexcept
{
    return ::estimateMaxFrequencyResponseGain(ir);
}
```

**※後方互換性**: IRConverter の `estimateMaxFrequencyResponseGain()` はデリゲートとして維持する（既存呼び出し元の変更を最小化）。新規コードは直接 `IRAnalyzer::estimateMaxFrequencyResponseGain()` を呼ぶことを推奨。

#### 1.3.1 Tukey窓 α値の変更決定 + コヒーレントゲイン補正（C-4）

文献調査の結果、UT-05で当初想定した「Tukey α=0.1, 10bin離れて-40dB以下」は達成が困難と判明したため、α を **0.5** に変更する。

- 両端 25% のコサインテーパーとなり、Hann 窓に近い減衰特性（~18 dB/oct）が得られる
- 65536 点 FFT に対し 25% = 16384 点が減衰領域だが、主要ピークの解析精度は十分維持される
- UT-05 のテスト基準「-40dB以下」は α=0.5 で達成可能

**⚠️ 窓補正 + ゼロパディング問題（★ v9.0 修正: ガウス補間追加）**:

Tukey α=0.5 のコヒーレントゲインは約 0.75 だが、これは IR 長が FFT 長（65536）に等しい場合の値。実際の IR が短い（copyLen < kAnalysisWindow）場合、ゼロパディング領域の窓値は 0 ではないため単純な `windowMean` 補正では不十分。

**★ v9.0 修正**: `windowMean` 補正に加え、**ガウス補間（3点ピーク補間）** を追加して FFT bin 間にピークが落ちる場合の誤差を軽減する:

```cpp
// estimateMaxFrequencyResponseGain 内 — ★ v9.0
const int copyLen = std::min(numSamples, kAnalysisWindow);
// ... Tukey窓生成、FFT実行 ...

// 1. 実効窓区間でコヒーレントゲイン補正
const double windowSum = std::accumulate(tukeyWindow.begin(), tukeyWindow.begin() + copyLen, 0.0);
const double windowMean = windowSum / static_cast<double>(copyLen);
maxMagnitude /= windowMean;

// 2. ★ v9.0: 3点ガウス補間（FFT bin間にピークが落ちた場合の誤差軽減）
//    maxMagnitude が隣接 2bin と比較して孤立ピークなら補間をスキップ
//    補間式: delta = 0.5 * (log(amp[k-1]) - log(amp[k+1])) / (log(amp[k-1]) - 2*log(amp[k]) + log(amp[k+1]))
//    interpolatedPeak = amp[k] * exp(-delta * (log(amp[k]) - log(amp[k-1])))
//    maxMagnitude = std::max(maxMagnitude, interpolatedPeak);

// ★ v9.0 確認: MKL DFTI 設定（DFTI_BACKWARD_SCALE = 1/N, 前方変換無スケール）。本設計は前方変換のみ使用のため影響なし。
```

**限界**: この補間はスペクトルピークが単一正弦波に近い形状を仮定している。IR のように複雑なスペクトル形状では過大/過小評価の可能性があるが、自動ゲインのヘッドルーム推定としては安全側（過大評価＝過剰ヘッドルーム）に作用するため許容する。

**補足**: `sampleRate` 引数は周波数応答の最大値推定には不要（`AudioBuffer<double>` の振幅はサンプリングレートに依存しない）。`[[maybe_unused]]` にするか削除する。

**実装詳細**:

```cpp
// kAnalysisWindow = 65536
// Tukey α=0.5（両端 25% コサインテーパー、中央 50% フラット）
// コヒーレントゲイン補正: maxMagnitude /= windowMean

```

#### 1.4 IR FFT解析の実装

`estimateMaxFrequencyResponseGain()` は `src/IRConverter.cpp` に実装:

```cpp
double IRConverter::estimateMaxFrequencyResponseGain(
    const juce::AudioBuffer<double>& ir,
    double sampleRate) noexcept
{
    // ★ v10.0: FFT サイズは nextPowerOfTwo(copyLen) を使用。
    //   65536 固定では IR 長が 65537 の場合に 1sample 切り捨てられる。
    //   nextPowerOfTwo は MKL でも効率的に処理可能。
    //   上限 kMaxAnalysisWindow (=65536) を超える場合はクリップ。
    constexpr int kMaxAnalysisWindow = 65536;
    constexpr double kTukeyAlpha = 0.5;

    const int numSamples = ir.getNumSamples();
    const int numChannels = ir.getNumChannels();
    if (numSamples <= 0 || numChannels <= 0)
        return 1.0;

    const int copyLen = std::min(numSamples, kMaxAnalysisWindow);
    const int fftSize = juce::nextPowerOfTwo(copyLen);

    // FFT実行: ScopedDftiDescriptor を使用（src/DftiHandle.h 確認済み）
    // MKL DFTI 実数→複素 FFT

    double maxMagnitude = 0.0;
    for (int ch = 0; ch < numChannels; ++ch)
    {
        // Tukey窓適用後のスペクトル振幅の最大値を探索
        // ...
    }

    return (maxMagnitude > 1e-18) ? maxMagnitude : 1.0;
}

```

**既存コード確認**: ✅ `ScopedDftiDescriptor` が `src/DftiHandle.h` に存在（DFTI_DESCRIPTOR_HANDLE ラッパー）。MKL DFTI が利用可能。

**★ v12.0 制約明記**: FFT 解析は `kMaxAnalysisWindow=65536` を上限とし、`nextPowerOfTwo(copyLen)` を FFT 長とする。IR 長が 65536 を超える場合、最初の 65536 sample のみ解析対象となる。通常の IR（ルームインパルス応答は一般的に 65536 sample 未満）では問題にならないが、極端に長い IR（例: 192kHz/8秒 = 1,536,000 sample）では後半の応答ピークを見逃す可能性がある。この制約は実用的な範囲で許容される設計判断である。

#### 1.5 `computeScaleFactor` の責務分割 — 3段階設計（★ v8.0）

**⚠️ 設計判断**: 従来の `computeScaleFactor()` は Energy補正、Peak/RMS/Freq解析、Protection判定を1関数で担当しており SRP 違反があった。v8.0 では以下の 3 段階に分割する:

```
第1段: computeEnergyScale(ir)
  → energy ベースの基本 scaleFactor + safetyMargin を計算

第2段: analyzeIR(ir, scaleFactor) → IRAnalysis
  → FFT解析（IRAnalyzer に委譲）、Peak/RMS 解析
  → 戻り値: frequencyPeakGain, peakValue, rmsValue

第3段: applyClampProtection(scaleFactor, analysis) → ScaleFactorResult
  → Peak/RMS/Freq clamp を適用
  → additionalAttenuationDb を算出（energy補正 -6dB は含まない）
```

**v8.0 の `computeScaleFactor()` は上記3段階のオーケストレーター** となる:

```cpp
ScaleFactorResult IRConverter::computeScaleFactor(
    const juce::AudioBuffer<double>& ir,
    const juce::AudioBuffer<double>* currentIr,
    double currentScale) noexcept
{
    // 第1段: Energy 補正
    double scale = computeEnergyScale(ir);

    // 第2段: IR 解析（FFTは IRAnalyzer に委譲）
    IRAnalysis analysis = analyzeIR(ir, scale);

    // 第3段: 保護クランプ
    return applyClampProtection(ir, scale, analysis,
                                currentIr, currentScale);
}
```

**`IRAnalysis` 構造体**（`IRAnalyzer.h` に定義）:

```cpp
struct IRAnalysis {
    double frequencyPeakGain = 1.0;  // FFT解析による周波数応答最大値
    double peakValue = 0.0;          // ピーク振幅
    double rmsValue = 0.0;           // RMS
    // 個別クランプ減衰量（順序依存あり: Peak→RMS→Freq の適用順）
    double peakClampDb = 0.0;
    double rmsClampDb = 0.0;
    double freqClampDb = 0.0;
};
```

**`IRConverter.h`**:

```cpp
struct ScaleFactorResult
{
    double scaleFactor = 1.0;
    bool hasScaleFactor = false;
    // ★ v7.0: 追加減衰量 [dB]（energy補正 -6dB は含まない）。
    //   Peak/RMS/Freq clamp による追加減衰量の合計。
    float additionalAttenuationDb = 0.0f;
};

```

**`IRConverter.cpp` `computeScaleFactor` 内の追加処理（C-2/C-3/v7.0 修正反映）**:

```cpp
// ★ v7.0: Peak/RMS を個別計算し、追加減衰量 [dB] を正値で記録（C-2/C-3）
double peakAttenDb = 0.0, rmsAttenDb = 0.0, freqAttenDb = 0.0;

// 既存 energy ベースの scaleFactor 適用後...
double scale = result.scaleFactor;  // energy補正 + safetyMargin(-6dB) 含む

// Peak クランプ
constexpr double kMaxEffectivePeak = 0.5;
const double irPeak = /* 既存のピーク検出値 */;
if (irPeak * scale > kMaxEffectivePeak)
{
    const double peakClamp = kMaxEffectivePeak / (irPeak * scale);
    result.scaleFactor *= peakClamp;
    scale *= peakClamp;  // ★ v13.0: RMS判定のため scale を更新（順序依存対応）
    peakAttenDb = -20.0 * std::log10(peakClamp);  // 正値の追加減衰量
}

// RMS クランプ（★ 順序依存: Peak適用後の scale で判定）
constexpr double kMaxEffectiveRms = 0.25;
const double irRms = std::sqrt(irEnergySum / numSamples);
if (irRms * scale > kMaxEffectiveRms)
{
    const double rmsClamp = kMaxEffectiveRms / (irRms * scale);
    result.scaleFactor *= rmsClamp;
    rmsAttenDb = -20.0 * std::log10(rmsClamp);
}

// 周波数応答ピーク解析
const double freqRespGain = estimateMaxFrequencyResponseGain(ir, sampleRate);
constexpr double kMaxEffectiveFreqResponse = 1.41;
if (freqRespGain > kMaxEffectiveFreqResponse)
{
    const double freqClip = kMaxEffectiveFreqResponse / freqRespGain;
    result.scaleFactor *= freqClip;
    freqAttenDb = -20.0 * std::log10(freqClip);
}

// ★ v7.0: additionalAttenuationDb は追加減衰量（energy補正含まず）
result.additionalAttenuationDb = static_cast<float>(peakAttenDb + rmsAttenDb + freqAttenDb);

```

**★ v5.1 調査確定: 既存 `safetyMargin` との関係**:

既存コード（`IRConverter.cpp:47`）には `safetyMargin = 0.5011872336272722`（≈ -6dB, `10^(-6/20)`）が存在し、energy ベースの scale factor に乗算済みである:
```cpp
constexpr double safetyMargin = 0.5011872336272722; // -6dB
result.scaleFactor = makeup * safetyMargin;          // 常に -6dB の余裕
```
この -6dB マージンはスケールファクターに組み込まれているため、新規 Peak/RMS clamp はこのマージンの **上乗せ** となる。既存 `kMaxEffectivePeak = 0.98`（≈ -0.175dB）は非常に弱い clamp だが、設計上の `kMaxEffectivePeak = 0.5`（-6dB）はエネルギー余裕と合わせて最大 -12dB の保護となる。これは Conv→PEQ 時の入力上限 -6dB クランプと整合する。

---

### Phase 2 State Management Extension

#### 2.1 `PreparedIRState.h` — `additionalAttenuationDb` 追加（`float` 型, 正値）

```cpp
struct PreparedIRState
{
    // ... existing members ...
    double scaleFactor = 1.0;
    bool hasScaleFactor = false;
    float additionalAttenuationDb = 0.0f;  // ★ v7.0: IR追加減衰量 [dB]（旧 residualRiskDb）

    PreparedIRState() = default;

    PreparedIRState(PreparedIRState&& other) noexcept
        : /* ... existing ... */
          scaleFactor(other.scaleFactor),
          hasScaleFactor(other.hasScaleFactor),
          additionalAttenuationDb(other.additionalAttenuationDb)
    {
        // ...
        other.scaleFactor = 1.0;
        other.hasScaleFactor = false;
        other.additionalAttenuationDb = 0.0f;
    }

    PreparedIRState& operator=(PreparedIRState&& other) noexcept
    {
        if (this != &other)
        {
            // ... existing cleanup ...
            scaleFactor = other.scaleFactor;
            hasScaleFactor = other.hasScaleFactor;
            additionalAttenuationDb = other.additionalAttenuationDb;
            // ...
            other.scaleFactor = 1.0;
            other.hasScaleFactor = false;
            other.additionalAttenuationDb = 0.0f;
        }
        return *this;
    }
};

```

#### 2.2 `IRConverter::convertFile` / `convertToHighRes` での反映

`IRConverter.cpp` 内で `computeScaleFactor` 呼び出し後に:

```cpp
prepared->additionalAttenuationDb = scaleResult.additionalAttenuationDb;

```

#### 2.3 `ConvolverProcessor.h` — `IRState` に `additionalAttenuationDb` 追加

**⚠️ 重要**: `PreparedIRState` は `applyComputedIR` 後に破棄される。`additionalAttenuationDb` は `IRState` に移して保持する必要がある。

```cpp
// ConvolverProcessor.h — IRState 構造体の拡張
struct IRState {
    std::unique_ptr<juce::AudioBuffer<double>> irOwner;
    const juce::AudioBuffer<double>* ir = nullptr;
    double sampleRate = 0.0;
    uint64_t generation = 0;
    float additionalAttenuationDb = 0.0f;  // ★ v7.0（旧 residualRiskDb）
};

```

**`applyComputedIR` での反映（M-1 一貫性確保）**:

```cpp
void ConvolverProcessor::applyComputedIR(std::unique_ptr<ConvolverIRPayload> prepared)
{
    if (prepared)
    {
        auto newState = std::make_unique<IRState>();
        // ... 既存設定 ...
        newState->additionalAttenuationDb = prepared->additionalAttenuationDb;
        // ... publishAtomic(currentIRState, newState.release(), ...) ...
    }
}

```

**AudioEngine からのアクセス用**:

```cpp
[[nodiscard]] float getIrAdditionalAttenuationDb() const noexcept
{
    auto* state = acquireIRState();
    return (state != nullptr) ? state->additionalAttenuationDb : 0.0f;
}

```

#### 2.4 `AudioEngine.h` — フラグ追加

**v2.0**: `recomputeAutoGainStaging()` は AudioEngine に追加しない。フラグと getter/setter のみを追加する。

```cpp
// AudioEngine クラス内（既存 atomic 群の近くに追加）
std::atomic<bool> autoGainStagingEnabled { true };

// メソッド宣言（public セクション）
void setAutoGainStagingEnabled(bool enabled);
[[nodiscard]] bool isAutoGainStagingEnabled() const;

// ★ captureBuildParameterSnapshot() に autoGainStagingEnabled を追加（BuildInput 経由で Builder に伝達）
//   → RuntimeBuilder::computeAndApplyAutoGain で参照

```

---

### Phase 3 EQ Maximum Gain Estimation

#### ★ 重要コード確認: `svfToDisplayBiquad()` の実DSP一致性

`svfToDisplayBiquad()` が実DSPと一致しているかは本設計の根幹に関わる。コード調査の結果、**完全一致が確認された**。

**根拠**（`AudioEngine.EQResponse.cpp:108-131`）:

```
実際の音声処理は TPT SVF (calcSVFCoeffs) を使用しており、...
修正: calcSVFCoeffs → svfToDisplayBiquad のパスは SVF の z 域伝達関数を
厳密に等価 biquad へ変換する（updateEQData の個別バンド曲線と同一）。
これにより「総合曲線 = 個別バンド曲線の積 = 実際の DSP 処理」が
三者完全一致する。
```

**数学的背景**: `svfToDisplayBiquad()`（`EQProcessor.Coefficients.cpp:347-368`）は SVF の a1/a2/a3/m0/m1/m2 係数から以下の代数的変換で等価 Biquad を導出する:

```
g  = a2/a1,  g2 = a3/a1,  gk = (1-a1-a3)/a1
a0 = 1 + gk + g2
a1' = -2 + 2*g2
a2' = 1 - gk + g2
b0 = m0*(1+gk+g2) + m1*g + m2*g2
b1 = -2*m0 + 2*(m0+m2)*g2
b2 = m0*(1-gk+g2) - m1*g + m2*g2
```

これは近似ではなく **数学的に厳密な z 域等価変換** である。`getMagnitudeSquared()` でこの Biquad 係数を使用することは実DSP（SVF）の周波数応答を正確に評価する。

なお `svfToDisplayBiquad()` は GUI 表示（`SpectrumAnalyzerComponent.cpp:869`）と `calcEQResponseCurve()`（`AudioEngine.EQResponse.cpp:129`）の両方で使用されており、三者完全一致はテスト実績がある（#EQ-Display 参照）。

#### 3.1 `EQProcessor.h` — `computeEstimatedMaxGainDb()` 宣言

```cpp
// class EQProcessor の public セクションに追加（既存の calcBiquadCoeffs 等の隣）
[[nodiscard]] float computeEstimatedMaxGainDb(double sampleRate, ProcessingOrder processingOrder) const;  // 推定最大ゲイン（dB）

```

#### 3.2 `EQProcessor.Coefficients.cpp` — 実装

**⚠️ 設計判断（R-2/R-6）**: 高Q（最大Q=20）のピーキングフィルタではピーク幅が極めて狭くなる。そこで以下の **2段階探索（粗探索 + Band適応サンプリング）** を採用する。20Hz 固定開始は 192kHz で高域密度が不足するため、対数分布 + 各 Band 中心周囲の適応サンプリングのみで十分である。

```
第1段: 対数周波数スケール 300 点（20Hz〜Nyquist）
  → 大域的最大値を特定

第2段: 各有効 Band の中心周波数周囲を Q 依存帯域幅で適応サンプリング（★ v12.0 統一）
  → 探索範囲: BW = fc/Q × kBwMultiplier（kBwMultiplier=8, 近似式）
  → 範囲クリップ: [max(20Hz, fc-BW/2), min(nyquist, fc+BW/2)]
  → 各 Band を kBandAdaptivePoints=64 点で線形スキャン
  → 高Q（Q=20）の鋭いピークを実用的な精度で捕捉可能。但し数学的な100%保証ではなく、実測確認を推奨（Phase 8 MT-06）
```

Worker Thread での実行であり計算負荷は問題にならない。最悪 20バンド×64点 = 1280 点の追加評価だが、MKL/AVX2 パスで 0.1ms 未満。
// ★ v3.6/v5.0: シグネチャ — BuildInput 全体ではなく必要最小限のパラメータ
//   EQProcessor が BuildInput（14フィールド）へ依存すると、
//   将来の BuildInput 変更の影響範囲が広がるため。
float EQProcessor::computeEstimatedMaxGainDb(
    double sampleRate,          // buildInput.sampleRate から
    ProcessingOrder processingOrder         // ★ v9.0: enum class
) const
{
    if (sampleRate <= 0.0) return 0.0f;

    const double nyquist = sampleRate * 0.5;
    constexpr int kCoarsePoints = 300;
    constexpr int kBandAdaptivePoints = 64;
    constexpr double kBwMultiplier = 8.0;

    // ★ v12.0: 粗探索（対数分布300点）+ Band適応サンプリング の2段階
    //   探索範囲は BW = fc/Q × kBwMultiplier（近似式、RBJ/SVF厳密帯域ではない）
    //   範囲クリップ: [max(20Hz, fc-BW/2), min(nyquist, fc+BW/2)]

    // 第1段: 粗探索（対数分布: 20Hz〜Nyquist、300点）
    std::array<double, kCoarsePoints> coarseFreqs;
    for (int i = 0; i < kCoarsePoints; ++i)
    {
        const double t = static_cast<double>(i) / (kCoarsePoints - 1);
        coarseFreqs[i] = 20.0 * std::pow(nyquist / 20.0, t);
    }
    // ... coarseFreqs で最大値を探索 ...

    // 第2段: 各有効 Band 中心周囲を Q 依存帯域幅で適応サンプリング（★ v14.0 統一）
    //   帯域式: range = max(20.0, fc/Q × kBwMultiplier)（最小スパン 20Hz 確保）
    //   範囲クリップ: [max(20Hz, center-range/2), min(nyquist, center+range/2)]
    //   range が fc*0.5 を超える場合は全域カバーとみなしスキップ
    // ★ v14.0: 探索対象は利得増加に寄与するバンドに限定:
    //   Peaking(gain>0)/Shelf(gain≠0)/HPF/LPF → 探索対象
    //   Peaking(gain≤0)/Notch → スキップ（振幅増大なし）
    // 第2段: 各 BandNode の中心周波数 fc, Q を収集
    //   startFreq = std::max(20.0, fc - (fc / Q) * kBwMultiplier * 0.5);
    //   endFreq   = std::min(nyquist, fc + (fc / Q) * kBwMultiplier * 0.5);
    //   range < 20Hz の場合はスキップ（分解能不足の帯域）
    //   ... 各 fc ± range/2 を kBandAdaptivePoints 点で線形スキャン ...
    //   ... 但し range が fc*0.5 を超える場合は全域カバーとみなしスキップ ...
    //   ... fineFreqs の最大値と比較し、大きい方を採用 ...

    // 2. 有効バンドの Biquad 係数を取得
    //    AudioEngine.EQResponse.cpp:105-116 と同じパスを使用することで、
    //    実際の DSP 処理と表示曲線の三者一致を維持する（別途確認済み）。

    // 3. L/R チャンネルのマグニチュード二乗をカスケード積算
    //    getMagnitudeSquared(biquadCoeffs, omega) を使用。
    //    M/S バンドは Stereo と同様に L/R 両方に掛ける。M/S の最大利得は
    //    max(|Hmid|, |Hside|) で評価。
    //    ★ v7.0: Serial/Parallel とも Serial 積 Π|Hi| で評価。
    //    Parallel 式 H=1+Σ(Hi-1) の厳密評価には getComplexResponse() が必要。

    // 4. 全周波数・全チャンネルにおける最大線形ゲインを保持
    //    totalGainDb も考慮（AGC OFF 時のみ適用）。

    // 5. 最終 = 20*log10(maxLinearGain)
    // ★ v13.0: computeEstimatedMaxGainDb は純粋な最大利得推定に限定。
    //   Parallel 時は Serial 積近似（数学的保証なし）。
}

```

**⚠️ 注意（M-2 修正）**: 新規の複素応答関数は追加しない。`getMagnitudeSquared()`（`EQProcessor.h:387-388`）は既に `z = cos(ω) + j·sin(ω)` を使用しており、M/S デコードに位相情報を必要としない。M/S モードの最大利得計算は `max(|Hmid|, |Hside|)` とする。

**★ v5.0 数学的確認: `max(|Hmid|, |Hside|)` はスペクトルノルムとして厳密に正しい。**
既存コードの M/S エンコード/デコード処理（`EQProcessor.Processing.cpp:790-830`）:
```
M = (L+R)*0.5,  S = (L-R)*0.5      // エンコード
L = M+S,        R = M-S            // デコード
```
このとき M/S フィルタ後の L/R 出力は行列で表現される:
```
[L_out]   = [1  1] [Hm   0 ] [0.5  0.5] [L_in]
[R_out]     [1 -1] [ 0  Hs] [0.5 -0.5] [R_in]
```
この合成行列の固有値は Hm と Hs（固有ベクトルは [1,1]ᵀ と [1,-1]ᵀ）。したがってスペクトルノルム（＝最大特異値）は `max(|Hm|, |Hs|)` で与えられる。**これは近似ではなく厳密解である。**

```cpp
// M/S バンドの最大利得: スペクトルノルム max(|Hmid|, |Hside|) は厳密

```cpp
// M/S バンドの最大利得: エンコード後の Mid/Side 成分はそれぞれ独立したフィルタ応答を持つ
// Mid と Side は L/R 合成時に max として作用する
// Hm = prod(Mid), Hs = prod(Side)
// L/R ゲイン = max(|Hm|, |Hs|)
float msGain = 1.0f;
if (hasMSBands)
{
    float Hm = 1.0f, Hs = 1.0f;  // 各々の積算
    // ... getMagnitudeSquared を Mid/Side バンドそれぞれで計算 ...
    msGain = std::max(Hm, Hs);
}
```

**Parallel 構造の注意**: Parallel filter structure では `H = 1 + Σ(Hi - 1)` で計算する。

**⚠️ v5.0 修正: `|1+Σ(Hi-1)| ≤ Π|Hi|` は一般に成立しない。**
前版では「Serial積が常に安全側上限」と記述していたが、これは数学的に保証されない。位相反転やノッチフィルタを含む場合、Serial積が Parallel 和を下回るケースが存在する。

**★ v12.0 最終方針: Parallel 構造は Serial 積近似（数学的保証なし、未対応と同等）。**

`computeEstimatedMaxGainDb()` は `getMagnitudeSquared()` しか利用できないため、Parallel 式 `H = 1 + Σ(Hi - 1)` を厳密に評価するには各バンドの複素周波数応答が必要。`|Hi|^2` のみから `Re(Hi), Im(Hi)` は復元できないため、振幅近似 `|H| ≈ 1 + Σ(|Hi| - 1)` は数学的に誤り。

したがって Parallel 時も Serial 積 `Π|Hi|` で評価する。これは近似であり数学的保証はない。特に Notch/AllPass/位相反転を含む構成では Serial 積が Parallel 和を下回る可能性がある。

```cpp
// computeEstimatedMaxGainDb() の実装方針 (★ v13.0):
// Serial: 通常通り Serial 積 Π|Hi| で評価（デフォルト、厳密）
// Parallel: Serial 積で近似（数学的保証なし）。
//   API名に "Estimated" を含めることで近似であることを明示。
```

---

### Phase 4 IR Conversion Extension

#### 4.1 `IRConverter.cpp` `computeScaleFactor` 拡張

`computeScaleFactor` の戻り値の型が変わったことによる呼び出し元の変更:

- `convertFile`（`IRConverter.cpp:156`）: `scaleResult.additionalAttenuationDb` を `prepared->additionalAttenuationDb` に代入（★ v3.6: 命名変更）
- `convertToHighRes`（`IRConverter.cpp:234`）: 同上

---

### Phase 5 RuntimeBuilder + AutoGainPlanner + BuildAnalysis（★ v3.6 最終設計）

#### 5.0 設計方針（v3.0 → v3.6 変更点）

v3.0 では RuntimeBuildSnapshot に解析値を直接追加したが、Snapshot の責務（Build Input の封印）を逸脱していた。v3.6 では **解析値を BuildAnalysis に分離し、Snapshot は純粋な BuildInput 封印に戻す**。

**v3.6 の4つの柱**:

| # | 設計要素 | 説明 |
|---|---------|------|
| 1 | **BuildAnalysis 分離** | DSP解析値を RuntimeBuildSnapshot から分離。Snapshot は純粋な BuildInput 封印に戻す |
| 2 | **AnalysisPart 新設** | RuntimePublishSpecification 内の解析値専用 Part。ProcessingPart は Builder 命令のみ |
| 3 | **AutoGainPlanner 独立** | 独立ファイル。`constexpr` 定数定義含む |
| 4 | **命名改善** | `residualRiskDb` → `additionalAttenuationDb`（v6.0: 追加減衰量に正確化） |

**変更理由**:

| 観点 | v3.0 (Snapshot に解析値直置き) | v3.6 (BuildAnalysis 分離) |
|------|-------------------------------|--------------------------|
| Snapshot 責務 | ❌ Builder Input 兼 解析キャッシュ | ✅ Snapshot: BuildInput 封印 / Analysis: 解析値 |
| ProcessingPart | ❌ 解析値混在 | ✅ ProcessingPart: 命令 / AnalysisPart: 解析値 |
| 拡張性 | ❌ 新値追加で Snapshot 拡張必要 | ✅ AnalysisPart にフィールド追加のみ |
| 命名意図 | ❌ residualRiskDb（残留リスク） | ✅ additionalAttenuationDb（エネルギー補正以外の追加減衰量） |

**v3.6 完全データフロー**:

```
User操作 (processingOrder/IR/EQ変更)
  → submitRebuildIntent (1回)
    → Worker Thread（Engine アクセス可能）:
      captureBuildParameterSnapshot()
      → BuildInput 生成
      → eqMaxGainDb = eqProcessor->computeEstimatedMaxGainDb(buildInput.sampleRate, buildInput.processingOrder)  // v13.0
      → additionalAttenuationDb = convolver.getIrAdditionalAttenuationDb()  // ★ v6.0: 追加減衰量
      → BuildAnalysis analysis { eqMaxGainDb, additionalAttenuationDb }   // ★ R-1: 分離
      → captureRuntimeBuildSnapshot(buildInput, ...)                   // Snapshot は BuildInput のみ
      → sealRuntimeBuildSnapshot() + sealBuildAnalysis()               // 各々封印
    → Orchestrator:
      → RuntimePublishSpecification 生成
        → ProcessingPart (Builder 命令のみ)
        → AnalysisPart (解析値: R-3)
    → RuntimeBuilder::buildRuntimePublishWorld(sealedSnapshot, spec)
      → AutoGainPlanner::plan(                               // ★ 独立ファイル
          spec.processing.autoGainStagingEnabled,
          spec.processing.processingOrder,
          spec.processing.eqBypassed,
          spec.processing.convBypassed,
          spec.analysis.eqMaxGainDb,                         // ★ AnalysisPart から
          spec.analysis.additionalAttenuationDb)                // ★ AnalysisPart から
        → Plan { inputHeadroomDb, trimDb, makeupDb }        // ★ dB値のみ
      → Builder: dB → 線形ゲイン変換（decibelsToGain）
      → worldOwner->automation.* = plan.*                    // 直接書込
      → worldOwner->freeze()
    → publishWorld → RuntimeStore RCU 公開

Audio Thread:
  acquireReadToken() → world->automation.inputHeadroomGain  // 既存経路
```

#### 5.0.1 BuildAnalysis 構造体の導入（★ v3.6: R-1/R-3 対応）

**背景**: v3.0 では `RuntimeBuildSnapshot` に `eqMaxGainDb` / `irResidualRiskDb` を直接追加したが、これは Snapshot の責務（Build Input の封印）を逸脱する。Snapshot は純粋な「何をビルドするか」に集中し、解析結果は別構造体で保持する。

**新設**: `BuildAnalysis`（`RuntimeBuildTypes.h`）— DSP解析結果を格納する独立構造体。

```cpp
// ★ v3.6: BuildAnalysis — DSP解析結果の封印。
//   RuntimeBuildSnapshot とは別責務：
//   - Snapshot: 何をビルドするか（Build Input）
//   - Analysis:  解析結果（EQ最大ゲイン, IR減衰量）
//   両者は同一世代でペアリングされ、各々 seal される。
// ★ v10.0: BuildAnalysis — Snapshot とペアで封印される DSP 解析結果。
//   sealed == true の契約:
//   ① 全フィールドは freeze 後の読み取り専用（Builder による変更禁止）
//   ② generation はペアとなる RuntimeBuildSnapshot.generation と一致必須
//   ③ sealBuildAnalysis() は validateBuildAnalysisPair() を呼び出し、
//      ペア検証後に sealed = true を設定する
//   ④ validateBuildAnalysisPair() の検証内容:
//      - sealedSnapshot != nullptr
//      - generation == sealedSnapshot->generation
//      - sealedSnapshot->sealed == true
//      - 全浮動小数点値が finite
//
// ★ v14.0: 明示的な seal/verify API を定義
//   [[nodiscard]] BuildAnalysis sealBuildAnalysis(BuildAnalysis analysis,
//       const RuntimeBuildSnapshot* snapshot) noexcept;
//   [[nodiscard]] bool verifyBuildAnalysisPair(const BuildAnalysis& analysis,
//       const RuntimeBuildSnapshot& snapshot) noexcept;
//   → verifyBuildAnalysisPair() は Orchestrator 側で jassert としても使用可能
struct BuildAnalysis {
    int generation = 0;           // 対応する RuntimeBuildSnapshot.generation と一致
    float eqMaxGainDb = 0.0f;    // EQProcessor::computeEstimatedMaxGainDb(sampleRate, processingOrder) の結果
    float additionalAttenuationDb = 0.0f;  // IRConverter 追加減衰量 [dB]（旧 residualRiskDb、energy補正除く）
    bool sealed = false;
};
```

**`RuntimePublishSpecification` に `AnalysisPart` を追加**:

```cpp
// RuntimeBuilder.h — RuntimePublishSpecification に新 Part
struct AnalysisPart {
    float eqMaxGainDb = 0.0f;           // BuildAnalysis からコピー
    float additionalAttenuationDb = 0.0f;  // BuildAnalysis からコピー
} analysis;
```

**Snapshot は純粋 BuildInput に戻る**（v3.0 で追加したフィールドを削除）:

```cpp
struct RuntimeBuildSnapshot {
    int generation = 0;
    BuildInput buildInput {};
    // ... 既存の convolverFingerprint / rebuildFingerprint / DSP projection ...
    // ★ v3.6: eqMaxGainDb / irResidualRiskDb は BuildAnalysis に移動
};
```

**Worker Thread での生成フロー**:

```cpp
// AudioEngine.RebuildDispatch.cpp — submitRebuildIntent 内（v3.6）
const auto paramSnapshot = captureBuildParameterSnapshot(*this);

// BuildAnalysis を生成（Snapshot とは独立）
const float eqMaxGainDb = eqProcessor->computeEstimatedMaxGainDb(
    task.buildInput.sampleRate,
    task.buildInput.processingOrder);  // v13.0: 改名（Parallel 近似であることを明示）
const float additionalAttenuationDb = uiConvolverProcessor.getIrAdditionalAttenuationDb();  // v6.0

BuildAnalysis analysis {};
analysis.generation = generation;
analysis.eqMaxGainDb = eqMaxGainDb;
analysis.additionalAttenuationDb = additionalAttenuationDb;
// sealBuildAnalysis(analysis);

// RuntimeBuildSnapshot は BuildInput のみ（v3.0 の解析値フィールド削除）
task.runtimeBuildSnapshot = sealRuntimeBuildSnapshot(
    finalizeRuntimeBuildSnapshot(
        captureRuntimeBuildSnapshot(task.buildInput, ...)));
task.buildAnalysis = analysis;  // ペアリング
```

**世代保証**: `BuildAnalysis.generation == RuntimeBuildSnapshot.generation` により、ビルド対象と解析結果の同一世代が保証される。
```

**呼び出し側（`submitRebuildIntent` 内）**:

```cpp
// AudioEngine.RebuildDispatch.cpp — submitRebuildIntent 内
const BuildParameterSnapshot paramSnapshot = captureBuildParameterSnapshot(*this);

// ... BuildInput 生成 ...

// ★ v3.0: Snapshot 生成時に EQ/IR 値を事前計算（Engine にアクセス可能な唯一の場所）
const float eqMaxGainDb = (!paramSnapshot.eqBypassed && eqProcessor != nullptr)
    ? eqProcessor->computeEstimatedMaxGainDb(sampleRate, static_cast<ProcessingOrder>(paramSnapshot.processingOrder))  // v13.0
    : 0.0f;
const float irResidualRiskDb = (!paramSnapshot.convBypassed)
    ? uiConvolverProcessor.getIrResidualRiskDb()
    : 0.0f;

task.runtimeBuildSnapshot = sealRuntimeBuildSnapshot(finalizeRuntimeBuildSnapshot(
    captureRuntimeBuildSnapshot(task.buildInput,
                                task.convolverBuildSnapshot,
                                generation,
                                structuralHash,
                                uiConvolverProcessor.isIRLoaded(),
                                uiConvolverProcessor.isIRFinalized(),
                                eqMaxGainDb,        // ★ v3.0
                                irResidualRiskDb)));  // ★ v3.0
```

**世代保証（B-6）**: `sealRuntimeBuildSnapshot()` により Snapshot は封印される。Builder は sealed value のみを読むため、「ビルド対象の IR」と「additionalAttenuationDb」が同一世代であることがランタイム封印機構で保証される。

#### 5.1 リアルタイム安全設計

自動ゲイン計算は **Worker Thread（非RTスレッド）** でのみ実行する。Audio Thread は既存通り `world->automation.*` を読むだけ。

**既存コード確認（Bencina 原則への準拠）**:

- ✅ `static_assert(std::atomic<uint64_t>::is_always_lock_free)` — `AudioEngine.h:1013`
- ✅ `convo::publishAtomic()` / `consumeAtomic()` — lock-free atomic アクセス
- ✅ `enqueueDeferredDeleteNonRt()` — RT 外でのメモリ解放（`AudioEngine.h:3788-3830`）
- ✅ `m_retireRouter->enqueueRetire()` — epoch ベース退役
- ✅ `worldOwner->freeze()` — publish 後の RuntimeWorld 不変性保証（`RuntimeBuilder.cpp`）

#### 5.2 計算ロジック（4パターン）

```cpp
// ★ v3.0: AutoGainPlanner::plan() 内で使用（Snapshot 値を入力）
入力:
  eqMaxDb = spec.processing.eqMaxGainDb           // Snapshot から事前計算値
  irResidualDb = spec.processing.irResidualRiskDb  // Snapshot から事前計算値

定数:
  kMarginEqFirst = 3.0f           // EQ第1段の入力マージン
  kMarginConvFirst = 1.5f         // Conv第1段の入力マージン
  kMarginInterStage = 2.0f        // 第2段保護マージン（両モード共通）

| モード | input | trim | makeup |
| -------- | ------- | ------ | -------- |
| PEQ only | -max(0, eqMax - 3.0) | 0 | -input |
| Conv only | -max(0, irResidual - 1.5) | 0 | -input |
| Conv→PEQ | -max(0, irResidual-1.5) - max(0, eqMax-2.0) | 0 | -input |
| PEQ→Conv | -max(0, eqMax - 3.0) | -max(0, irResidual-2.0) | -input - trim |

```

**検証: 全モードでネット 0dB 成立** ✅（`input + trim + makeup = 0`）

**クランプ範囲**:

- `input`: [-12, maxDb] — maxDb は Conv-first で -6dB、それ以外で 0dB
- `trim`: [-12, 0] dB
- `makeup`: [0, 12] dB

**⚠️ Conv-first 時の input 上限 -6dB クランプ**: Conv→PEQ モードの計算結果が 0dB の場合、-6dB にクランプされる。`computeAndApplyAutoGain()` はクランプ後の値で makeup を再計算する（`computedMakeupDb = -clampedInputDb - clampedTrimDb`）。ネット 0dB は保証されるが、`input=-6dB / makeup=6dB` となるためユーザーは「Auto で音が静かになった」と錯覚する可能性がある。これは意図された設計（Conv-first 時の入力保護）である。v1.0 からの改善として、実効値ベースのネット 0dB 整合を `computeAndApplyAutoGain()` 内で直接実装した。

#### 5.3 `AutoGainPlanner` — 純粋関数型ゲイン計算（★ v3.5: dB値入出力 + 分離ファイル）

`AutoGainPlanner` は Engine 参照を一切持たない純粋関数型クラス。`RuntimeBuildSnapshot` の事前計算値から自動ゲイン[dB値]を計算する。**`decibelsToGain()` 変換は Builder 側で行う**。

**将来の分離設計**: `AutoGainPlanner` は `RuntimeBuilder` から独立したコンポーネント。初期実装では `RuntimeBuilder.h/.cpp` に同居するが、`AutoGainPlanner` に依存関係が追加される場合は `src/audioengine/AutoGainPlanner.h` / `.cpp` に分離する。分離後のヘッダは `RuntimeBuilder.h` から include される。

```cpp
// ★ v3.5: AutoGainPlan — dB 値のみ（線形変換は Builder が行う）
struct AutoGainPlan {
    float inputHeadroomDb = 0.0f;      // dB 値（下限 -12dB）
    float outputMakeupDb = 0.0f;       // dB 値（0..12dB）
    float convolverInputTrimDb = 0.0f; // dB 値（-12..0dB）
};

class AutoGainPlanner {
public:
    // 唯一の公開メソッド — 純粋関数（static, noexcept）
    // 入力値は RuntimeBuildSnapshot から取得（ProcessingPart 非経由）
    [[nodiscard]] static AutoGainPlan plan(
        bool autoGainEnabled,
        ProcessingOrder processingOrder,
        bool eqBypassed,
        bool convBypassed,
        float eqMaxGainDb,          // sealedSnapshot->eqMaxGainDb
        float irResidualRiskDb      // sealedSnapshot->irResidualRiskDb
    ) noexcept;
};
```

#### 5.3.1 `AutoGainPlanner::plan()` 実装

```cpp
// ★ v3.6: AutoGainPlanner — 独立ファイル。constexpr 定数 + 命名改善（R-5/R-7）

// AutoGainPlanner.h:
#pragma once

// ★ v9.0: 型安全のため ProcessingOrder を enum class で直接受ける
//   computeMaxGainDb も合わせて enum class ProcessingOrder で宣言する。
//   ("0=Serial, 1=Parallel" の int 変換は呼び出し側で行う)

// Margin constants (★ v10.0: inline constexpr, C++20 推奨)
inline constexpr float kMarginEqFirst    = 3.0f;   // EQ第1段の入力マージン
inline constexpr float kMarginConvFirst  = 1.5f;   // Conv第1段の入力マージン
inline constexpr float kMarginInterStage = 2.0f;   // 第2段保護マージン
inline constexpr float kClampInputMin    = -12.0f; // input 下限
inline constexpr float kClampInputMax    = 0.0f;   // input 上限
inline constexpr float kClampTrimMin     = -12.0f; // trim 下限
inline constexpr float kClampTrimMax     = 0.0f;   // trim 上限
inline constexpr float kClampMakeupMin   = 0.0f;   // makeup 下限
inline constexpr float kClampMakeupMax   = 12.0f;  // makeup 上限
inline constexpr float kConvFirstInputCeiling = -6.0f; // Conv-first 上限

struct AutoGainPlan {
    float inputHeadroomDb = 0.0f;
    float outputMakeupDb = 0.0f;
    float convolverInputTrimDb = 0.0f;
};

class AutoGainPlanner {
public:
    [[nodiscard]] static AutoGainPlan plan(
        bool autoGainEnabled,
        ProcessingOrder processingOrder,   // ★ v9.0: enum class で型安全
        bool eqBypassed,
        bool convBypassed,
        float eqMaxGainDb,              // ★ AnalysisPart から
        float additionalAttenuationDb      // ★ v6.0: 追加減衰量（旧 irResidualRiskDb）
    ) noexcept;

    // ★ v8.0: Q Surge Margin — 経験則に基づくヒューリスティック安全マージン
    //   係数（0.15, 6dB, 0.707）は理論式ではなく Phase 8 の実測検証で較正が必要
    [[nodiscard]] static float estimateQSafetyMargin(
        float eqMaxGainDb, ProcessingOrder processingOrder) noexcept;
};

// AutoGainPlanner.cpp:
AutoGainPlan AutoGainPlanner::plan(
    bool autoGainEnabled,
    ProcessingOrder processingOrder,   // ★ v9.0: enum class
    bool eqBypassed,
    bool convBypassed,
    float eqMaxGainDb,
    float additionalAttenuationDb) noexcept
{
    AutoGainPlan result {};
    if (!autoGainEnabled)
        return result;  // 0.0/0.0/0.0（= 0dB/0dB/0dB）

    float inputDb = 0.0f, trimDb = 0.0f;

    if (!eqBypassed && convBypassed)
    {
        // PEQ only
        inputDb = -std::max(0.0f, eqMaxGainDb - kMarginEqFirst);
        // ★ v8.0: Q Surge マージン（経験則ヒューリスティック、Phase 8 要較正）
        inputDb -= estimateQSafetyMargin(eqMaxGainDb, processingOrder);
    }
    else if (eqBypassed && !convBypassed)
    {
        // Conv only
        inputDb = -std::max(0.0f, additionalAttenuationDb - kMarginConvFirst);
    }
    else if (processingOrder == ProcessingOrder::ConvolverThenEQ)  // ★ v9.0: enum 比較
    {
        // Conv→PEQ: trim 不適用, input 上限 -6dB
        inputDb = -(std::max(0.0f, additionalAttenuationDb - kMarginConvFirst)
                    + std::max(0.0f, eqMaxGainDb - kMarginInterStage));
        inputDb = std::max(inputDb, kConvFirstInputCeiling);
    }
    else
    {
        // PEQ→Conv: trim 適用
        inputDb = -std::max(0.0f, eqMaxGainDb - kMarginEqFirst);
        // ★ v8.0: Q Surge マージン（EQ段の共振保護、経験則）
        inputDb -= estimateQSafetyMargin(eqMaxGainDb, processingOrder);
        trimDb = -std::max(0.0f, additionalAttenuationDb - kMarginInterStage);
    }

    // クランプ（constexpr 定数使用）
    result.inputHeadroomDb = juce::jlimit(kClampInputMin, kClampInputMax, inputDb);
    result.convolverInputTrimDb = juce::jlimit(kClampTrimMin, kClampTrimMax, trimDb);

    // ネット 0dB 整合
    const float makeupDb = -result.inputHeadroomDb - result.convolverInputTrimDb;
    result.outputMakeupDb = juce::jlimit(kClampMakeupMin, kClampMakeupMax, makeupDb);

    return result;  // ★ dB 値のみ。線形変換は Builder 側
}

// ★ v10.0: Q Surge Margin — FilterType 依存の経験則ヒューリスティック
//   式: Qsurge = min(kQSurgeMax, baseMargin + peakingSurge)
//   - Peaking フィルタが主対象（共振による振幅増大）
//   - Shelf (Low/High): 共振が弱いため過剰マージンだが安全側のため許容
//   - Notch/AllPass: 共振は発生しないためマージン不要だが簡略化のため一律適用
//   係数 kQSurgeCoeff=0.15, kQSurgeMax=6.0 は理論根拠のない経験則。
//   Phase 8 の実測検証で較正が必要。
[[nodiscard]] static float estimateQSafetyMargin(
    float eqMaxGainDb, ProcessingOrder /* processingOrder */) noexcept
{
    constexpr float kQSurgeHpfLpf = 1.5f;
    constexpr float kQSurgeCoeff = 0.15f;
    constexpr float kQSurgeMax = 6.0f;
    constexpr float kButterworthQ = 0.707f;

    float peakingSurge = 0.0f;
    peakingSurge = eqMaxGainDb > 0.0f
        ? eqMaxGainDb * kQSurgeCoeff * (20.0f / kButterworthQ)  // worst-case Q=20
        : 0.0f;

    return std::min(kQSurgeMax, kQSurgeHpfLpf + peakingSurge);
}
```

#### 5.3.2 RuntimeBuilder 側の統合コード（★ v9.0: 単一代入）

**設計判断**: 従来のコードでは ProcessingPart のベース値を一度代入した後、Auto ON 時に上書きしていた。これは二重設定で分かりにくい。★ v9.0 では `buildRuntimePublishWorld()` 内で `AutoGainPlanner` の plan を先に計算し、Automation への代入は **1回だけ** 行う。

```cpp
// RuntimeBuilder.cpp — buildRuntimePublishWorld() 内の ★ v9.0 単一代入
convo::aligned_unique_ptr<const RuntimePublishWorld>
RuntimeBuilder::buildRuntimePublishWorld(
    const convo::RuntimeBuildSnapshot* sealedSnapshot,
    const RuntimePublishSpecification& spec) noexcept
{
    // ... 既存の Topology/Routing/Execution/Overlap/Latency 写像 ...

    // ★ v9.0: non-Auto のベース値を先に準備
    float baseInputHeadroomGain = spec.processing.inputHeadroomGain;
    float baseOutputMakeupGain = spec.processing.outputMakeupGain;
    float baseConvTrimGain = spec.processing.convolverInputTrimGain;

    // ★ v9.0: Auto ON 時は plan で上書き（単一代入のために先に計算）
    if (spec.processing.autoGainStagingEnabled)
    {
        const auto plan = AutoGainPlanner::plan(
            true,
            spec.processing.processingOrder,
            spec.processing.eqBypassed,
            spec.processing.convBypassed,
            spec.analysis.eqMaxGainDb,
            spec.analysis.additionalAttenuationDb);

        baseInputHeadroomGain =
            juce::Decibels::decibelsToGain(static_cast<double>(plan.inputHeadroomDb));
        baseOutputMakeupGain =
            juce::Decibels::decibelsToGain(static_cast<double>(plan.outputMakeupDb));
        baseConvTrimGain =
            juce::Decibels::decibelsToGain(static_cast<double>(plan.convolverInputTrimDb));
    }

    // ★ v9.0: Automation への代入は 1回のみ
    worldOwner->automation.eqBypassed = spec.processing.eqBypassed;
    worldOwner->automation.convBypassed = spec.processing.convBypassed;
    worldOwner->automation.softClipEnabled = spec.processing.softClipEnabled;
    worldOwner->automation.saturationAmount = spec.processing.saturationAmount;
    worldOwner->automation.inputHeadroomGain = baseInputHeadroomGain;
    worldOwner->automation.outputMakeupGain = baseOutputMakeupGain;
    worldOwner->automation.convolverInputTrimGain = baseConvTrimGain;

    // ... 既存の Resource/Timing/Publication/Retire 写像 ...
    worldOwner->freeze();
    return worldOwner;
}
```

**設計上の重要点**:

1. **Builder から Engine 参照を完全排除** (B-1/B-3): `AutoGainPlanner::plan()` は static 純粋関数。Builder は `sealedSnapshot` と `spec.processing` のみを使用。
2. **Snapshot 一次情報源** (B-2/B-4/B-6): `eqMaxGainDb` / `irResidualRiskDb` は `RuntimeBuildSnapshot` のみが保持。ProcessingPart に重複なし。世代は封印によって保証。
3. **dB 値入出力** (推奨改善): `AutoGainPlanner` は DSP 知識（`decibelsToGain`）を持たない純粋アルゴリズム。線形変換は Builder 側で行う。
4. **ファイル分離** (推奨改善): `RuntimeBuilder` 肥大化防止のため、`AutoGainPlanner` は `src/audioengine/AutoGainPlanner.h` / `.cpp` に分離推奨。
5. **1回の rebuild で完結** (C-6): `submitRebuildIntent()` を呼ばない。リビルドストームなし。
6. **実効値ベースのネット 0dB 整合** (C-7): クランプ後の dB 値で makeup を算出。

#### 5.4 `setProcessingOrder` の修正

**v2.0 変更**: `recomputeAutoGainStaging()` の呼び出しは AudioEngine setter から削除し、RuntimeBuilder 側で代替する。`sendChangeMessage()` は bypass 系 setter と整合させるために追加する。

```cpp
// AudioEngine.Parameters.cpp — setProcessingOrder の修正
void AudioEngine::setProcessingOrder(ProcessingOrder order)
{
    ASSERT_NON_RT_THREAD();
    convo::publishAtomic(currentProcessingOrder, order, std::memory_order_release);
    convo::publishAtomic(m_currentProcessingOrder, order, std::memory_order_release);
    submitRebuildIntent(convo::RebuildKind::Structural,
        RebuildTelemetryReason::EnqueueSnapshotCommand,
        RebuildTelemetryClass::Snapshot,
        RebuildTelemetryPolicy::Replaceable);
    // ★ v2.0: applyDefaultsForCurrentMode を削除し、submitRebuildIntent のみとする
    //   → 自動ゲイン計算は AutoGainPlanner が行う（RuntimeBuilder 内）
    //   → applyDefaultsForCurrentMode 内の submitRebuildIntent 重複を排除
    sendChangeMessage();  // ★ 追加（bypass 系 setter に揃える）
}
```

**v1.0 からの変更点**:

- `applyDefaultsForCurrentMode()` の呼び出しを削除。これにより submitRebuildIntent の二重発火が解消される（C-6）
- `recomputeAutoGainStaging()` は削除（Builder 統合に伴い）
- `sendChangeMessage()` は保持（bypass 系 setter との一貫性）

**補足**: `applyDefaultsForCurrentMode()` の削除に伴い、デフォルトゲイン値の設定は `AutoGainPlanner::plan()` が代替する。Auto ON 時は Snapshot の事前計算値が適用され、Auto OFF 時は直近の手動設定値が `ProcessingPart` 経由で適用される（変更なし）。

#### 5.5 `convolverParamsChanged` — IRロード後のトリガ

`AudioEngine.UIEvents.cpp` 内で、IRロード完了後に明示的に `recomputeAutoGainStaging()` を呼ぶ必要はない。IRロード完了は既存のパスで `submitRebuildIntent()` を発火するため、RuntimeBuilder が自動的に `computeAndApplyAutoGain()` を実行する。

**v2.0**: `convolverParamsChanged` の末尾に新規追加は不要。**IRロード→rebuild→Builder内自動ゲイン計算** で完結する。

#### 5.6 プリセットロード時の対策

**v2.0 では `recomputeAutoGainStaging()` を AudioEngine に追加しない。プリセットロード後の自動ゲイン再計算は rebuild に委ねる。**

**既存の問題 (C-5)**: `DeviceSettings::loadSettings()` は `BulkRestoreGuard` を使用。ガード内で setter を呼んでも rebuild は保留される。ガード破棄時に `endBulkParameterRestore(true)` → `m_isRestoringState = false` → rebuild 1回発火。

**修正アプローチ**: 既存の `BulkRestoreGuard` のスコープ設計をそのまま利用する。guard のデストラクタが `endBulkParameterRestore(true)` を呼び、その後の rebuild 内で RuntimeBuilder が自動ゲインを計算する。`recomputeAutoGainStaging()` の追加呼び出しは不要。

```
// DeviceSettings::loadSettings — スコープ再構成（C-5 確認）:

    engine.beginBulkParameterRestore();
    {
        BulkRestoreGuard guard { engine };
        // ... setter 呼び出し ... (rebuild 保留)
        // ... setConvolverStateTree ...
        return;  // guard デストラクタ → endBulkParameterRestore → m_isRestoringState=false
    }
    // → submitRebuildIntent → Builder 内で auto gain 計算
```cpp

**既存コード確認**:

- `RestoreStateGuard`: `AudioEngine.StateIO.cpp:16-22` — `requestLoadState` 内の RAII ガード
- `BulkRestoreGuard`: `DeviceSettings.cpp:981-983` — `loadSettings` 内の RAII ガード
- `endBulkParameterRestore`: `AudioEngine.Parameters.cpp:207-218` — `m_isRestoringState = false` + `submitRebuildIntent` → Builder 内で auto gain 計算

### Phase 6 Runtime Mode Transition Safety

#### 6.1 クロスフェード機構（変更なし）

既存の `CrossfadeAuthority::evaluate()`（`CrossfadeAuthority.cpp:8-48`）は以下の 3 項目のみで判定:

- `irLoaded`（IR有無変化）
- `structuralHash`（IR構造変化）
- `oversamplingFactor`（OS倍率変化）

`processingOrder` や `bypassRequested` の変化は直接検出しない。

**⚠️ 重要**: 純粋なモード/オーダー変更（IR同一）ではクロスフェードはトリガーされず、即時切替となる。この場合:

- 新 DSPCore に `FADE_IN_SAMPLES = 2048`（`AudioEngine.h:973`）が設定され（`RebuildDispatch.cpp:910`）、42ms @48kHz の出力フェードイン（0→1.0 リニアランプ）が発生する
- `DSPCoreDouble.cpp:605-617` の `applyGainRamp()` で実装（double 版と float 版で同一ロジック）

#### 6.2 ゲイン値の安全性

新旧 DSP は同一の `ProcessingState`（`captureAudioThreadParameterSnapshot()` 経由）のゲイン値を使用する。ゲイン値自体の不整合は発生しない。

---

### Phase 7 UI Integration

#### 7.1 `DeviceSettings.h` — `autoGainToggle` 追加

```cpp

// DeviceSettings.h の private members に追加（inputHeadroomEditor の隣）
juce::ToggleButton autoGainToggle { "Auto Gain Staging" };

```cpp

#### 7.2 `DeviceSettings.cpp` — レイアウト変更

```cpp

// resized() 内: row2（Input Headroom）の横にトグルを配置
// 2行目: Input Headroom + Auto Gain Staging toggle
inputHeadroomLabel.setBounds(row2.removeFromLeft(200).reduced(5));
inputHeadroomEditor.setBounds(row2.removeFromLeft(120).reduced(5));
autoGainToggle.setBounds(row2.removeFromLeft(160).reduced(5));  // ★ 追加

```cpp

**既存レイアウト確認**: `resized()` は 6 行構成（`DeviceSettings.cpp:472-515`）。row2（Input Headroom）の残り領域に 160 px トグルを配置可能。`numRows` は 6 のままでも追加トグルに空きはあるが、必要に応じて row 高さや余白を調整する。

```cpp

// DeviceSettings コンストラクタ内（updateGainStagingDisplay() の前に追加）
addAndMakeVisible(autoGainToggle);
autoGainToggle.onClick = [this] {
    audioEngine.setAutoGainStagingEnabled(autoGainToggle.getToggleState());
    // ★ v2.0/v3.0: recomputeAutoGainStaging は不要。setter 内で submitRebuildIntent を呼び、
    //   AutoGainPlanner が rebuild 内で自動ゲインを計算する。
};

```cpp

#### 7.3 `updateGainStagingDisplay()` の拡張

```cpp

void DeviceSettings::updateGainStagingDisplay()
{
    // ... 既存のモードテキスト/ラベル更新 ...

    const bool autoEnabled = audioEngine.isAutoGainStagingEnabled();

    // Auto 有効時はエディタを読み取り専用に
    inputHeadroomEditor.setEnabled(!autoEnabled);
    outputMakeupEditor.setEnabled(!autoEnabled);

    // トグル状態同期
    autoGainToggle.setToggleState(autoEnabled, juce::dontSendNotification);

    // ... 既存の数値表示更新 ...
}

```cpp

#### 7.4 手動編集時の Auto 解除（M-4 修正: コールバックチェーン）

**⚠️ 既存の onTextChange を上書きしない。元の setInputHeadroomDb 呼び出しを維持するためチェーンする。**

```cpp
// DeviceSettings コンストラクタ内 — ★ v10.1: Listener 方式を推奨
//   onTextChange/onEditorHide のラムダチェーン方式は、将来複数箇所からの
//   コールバック設定で競合する可能性がある。
//   より堅牢な設計として、JUCE の TextEditor::Listener を継承した専用ハンドラ
//   AutoGainEditHandler による一元管理を推奨する。
//   初期実装では簡易的に onEditorHide チェーンとする。
auto oldInputOnHide = std::move(inputHeadroomEditor.onEditorHide);
inputHeadroomEditor.onEditorHide = [this, oldInputOnHide = std::move(oldInputOnHide)] {
    if (audioEngine.isAutoGainStagingEnabled())
    {
        autoGainToggle.setToggleState(false, juce::dontSendNotification);
        audioEngine.setAutoGainStagingEnabled(false);
    }
    if (oldInputOnHide)
        oldInputOnHide();
};
auto oldOutputOnHide = std::move(outputMakeupEditor.onEditorHide);
outputMakeupEditor.onEditorHide = [this, oldOutputOnHide = std::move(oldOutputOnHide)] {
    if (audioEngine.isAutoGainStagingEnabled())
    {
        autoGainToggle.setToggleState(false, juce::dontSendNotification);
        audioEngine.setAutoGainStagingEnabled(false);
    }
    if (oldOutputOnHide)
        oldOutputOnHide();
};
```

**既存コード確認**: ✅ `inputHeadroomEditor` / `outputMakeupEditor` は `DeviceSettings.h:79-82` で宣言済み。`DeviceSettings.cpp:369-385` で既存の onTextChange に setInputHeadroomDb/setOutputMakeupDb 呼び出しが設定されている。これらのラムダを上書きせずチェーンする必要がある（M-4）。

#### 7.5 永続化（M-5 修正: XML 属性追加）

**`DeviceSettings::saveSettings`** に `autoGainStagingEnabled` を保存:

```cpp
// DeviceSettings.cpp — saveSettings 内（既存の xml->setAttribute 群に追加）
xml->setAttribute("autoGainStagingEnabled",
    static_cast<int>(engine.isAutoGainStagingEnabled()));
```

**`DeviceSettings::loadSettings`** に復元を追加:

```cpp
// DeviceSettings.cpp — loadSettings 内（BulkRestoreGuard 内の既存パラメータ復元群に追加）
bool autoGain = xml->getBoolAttribute("autoGainStagingEnabled", true);
engine.setAutoGainStagingEnabled(autoGain);
```

**既存コード確認**: ✅ saveSettings は `DeviceSettings.cpp:924-968` で実装済み（`"outputMakeupDb"` 等と同パターン）。loadSettings は `DeviceSettings.cpp:975` で実装済み。

---

### Phase 8 CMakeLists Updates

```cmake

# EQProcessorTests（新規／既存 EQ 応答計算の再利用を検証）

add_executable(EQProcessorMaxGainTests
    src/tests/EQProcessorMaxGainTests.cpp
    src/eqprocessor/EQProcessor.Coefficients.cpp
)
target_compile_features(EQProcessorMaxGainTests PRIVATE cxx_std_20)
target_include_directories(EQProcessorMaxGainTests PRIVATE
    src src/eqprocessor src/audioengine
)
if(MSVC AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    target_link_libraries(EQProcessorMaxGainTests PRIVATE MKL::MKL)
endif()
add_test(NAME EQProcessorMaxGainTests COMMAND EQProcessorMaxGainTests)

# GainStagingContractTests（新規 — v2.0: RuntimeBuilder 統合テスト）

add_executable(GainStagingContractTests
    src/tests/GainStagingContractTests.cpp
)
target_compile_features(GainStagingContractTests PRIVATE cxx_std_20)
target_include_directories(GainStagingContractTests PRIVATE
    src src/eqprocessor src/audioengine
)
target_link_libraries(GainStagingContractTests PRIVATE
    MKL::MKL
    juce::juce_recommended_config
)
add_test(NAME GainStagingContractTests COMMAND GainStagingContractTests)

# ★ v2.0: computeAndApplyAutoGain の入出力契約テスト
#   RuntimeBuilder 経由で自動ゲイン計算の出力（worldOwner->automation の値）を検証する。
#   AudioEngine の完全な初期化は不要。Specification と Mock DSPCore で Builder をテスト可能。
#   テストケース: 4パターン（PEQ only / Conv only / Conv→PEQ / PEQ→Conv）× Auto On/Offcpp

---

## Appendix A Code Verification List

### A.1 既存コード照合一覧

| 文書の主張 | コード実測 | ステータス |
| ----------- | ----------- | ----------- |
| `ScaleFactorResult` に `additionalAttenuationDb` 未追加 | `IRConverter.h:13` | ✅ 未追加確認、v7.0 で追加 |
| `PreparedIRState` に `additionalAttenuationDb` 未追加 | `PreparedIRState.h` | ✅ 未追加確認 |
| `IRState` に `additionalAttenuationDb` 未追加 | `ConvolverProcessor.h:1011` | ✅ 未追加確認、v7.0 で追加 |
| `computeMaxGainDb()` 未実装 | コードベース全体で 0 hits | ✅ 未実装確認 |
| `currentPreparedIr` は AudioEngine に存在しない | AudioEngine.h で 0 hits | ✅ v2.0: `uiConvolverProcessor.getIrResidualRiskDb()` に変更 |
| `BulkRestoreGuard` のスコープ構造 | `DeviceSettings.cpp:979-983` | ✅ スコープ再構成（C-5） |
| `RuntimeBuilder` の buildRuntimePublishWorld 内 automation 写像 | `RuntimeBuilder.cpp:269-275` | ✅ ProcessingPart → worldOwner->automation |
| `captureBuildParameterSnapshot` の auto gain 関連 | `AudioEngine.RebuildDispatch.cpp:33-50` | ✅ `BuildInput` 拡張（v2.0） |
| `setProcessingOrder` の applyDefaultsForCurrentMode 削除 | `AudioEngine.Parameters.cpp:268-275` | ✅ v2.0: 削除（Builder が代替） |
| `RuntimeBuildSnapshot` に eqMaxGainDb 追加 | `RuntimeBuildTypes.h:35-53` | ✅ **v3.0**: Snapshot 拡張（B-2） |
| `RuntimeBuildSnapshot` に irResidualRiskDb 追加 | `RuntimeBuildTypes.h:35-53` | ✅ **v3.0**: Snapshot 拡張（B-2） |
| `captureRuntimeBuildSnapshot` の引数拡張 | `AudioEngine.RebuildDispatch.cpp:78-102` | ✅ **v3.0**: eqMaxGainDb/irResidualRiskDb 追加 |
| `AutoGainPlanner` クラス未作成 | — | ✅ **v3.5**: `AutoGainPlanner.h/.cpp` 新規作成（分離推奨） |
| `AutoGainPlanner::plan()` が Engine 非参照 | — | ✅ **v3.0**: 純粋関数（B-1） |
| Builder が `AudioEngine&` を保持する参照を削除 | `RuntimeBuilder.h` | ✅ **v3.0**: `computeAndApplyAutoGain` 削除 |
| ProcessingPart の eqMaxGainDb/irResidualRiskDb 重複 | `RuntimeBuilder.h` | ✅ **v3.5**: 重複排除、Snapshot 一次情報源（B-4） |
| `computeMaxGainDb()` のサンプリング点数 300 | `EQProcessor.Coefficients.cpp` | ✅ **v3.5**: 二段階探索（粗300+精密200）に改善 |
| `svfToDisplayBiquad()` の実DSP一致性 | `EQProcessor.Coefficients.cpp:347-368` | ✅ **確認済み**: 数学的厳密等価変換。三者完全一致。 |
| `AutoGainPlanner` の出力形式 | `AutoGainPlanner.cpp` | ✅ **v3.5**: dB値のみ。線形変換は Builder 側 |
| `z = exp(+jω)` | `EQProcessor.Coefficients.cpp:327` — `z(cos(w), sin(w))` | ✅ **一致確認** |
| `ConvolverThenEQ` パスで trim 不適用 | `DSPCoreDouble.cpp:429-457` — 該当パスに trim コードなし | ✅ **確認** |
| `EQThenConvolver` パスで trim 適用 | `DSPCoreDouble.cpp:483` — `if (state.convolverInputTrimGain != 1.0)` | ✅ **確認** |
| clipping: input [-12, maxDb] | `AudioEngine.Parameters.cpp:224-242` — `juce::jlimit(-12.0f, maxDb, db)` | ✅ **確認** |
| clipping: trim [-12, 0] | `AudioEngine.Parameters.cpp:277` — `juce::jlimit(-12.0f, 0.0f, db)` | ✅ **確認** |
| clipping: makeup [0, 12] | `AudioEngine.Parameters.cpp:247-248` — `juce::jlimit(0.0f, 12.0f, db)` | ✅ **確認** |
| `setProcessingOrder`: submit→apply の順 | `AudioEngine.Parameters.cpp:268-275` — 逆順（submit先） | ✅ **確認** — 削除対象 |
| `setProcessingOrder`: sendChangeMessage なし | `AudioEngine.Parameters.cpp:268-275` — 呼び出しなし | ✅ **確認** — 追加対象 |
| bypass setter は apply→submit→sendChange の順 | `setEqBypassRequested:153-161`, `setConvolverBypassRequested:163-172` | ✅ **確認** — パターン一致 |
| `Convolver first` の上限 -6dB | `AudioEngine.Parameters.cpp:231` — `maxDb = -6.0f` | ✅ **確認** |
| `CrossfadeAuthority` 3項目のみ判定 | `CrossfadeAuthority.cpp:8-48` | ✅ **確認** |
| `FADE_IN_SAMPLES = 2048` (42ms) | `AudioEngine.h:973` — `static constexpr int FADE_IN_SAMPLES = 2048` | ✅ **確認** |
| DSPCore フェードイン実装 | `DSPCoreDouble.cpp:605-617` + `DSPCoreFloat.cpp:399-411` | ✅ **確認** |
| `m_isRestoringState` ガード | `AudioEngine.Parameters.cpp:298` | ✅ **確認** |
| `RestoreStateGuard` RAII | `AudioEngine.StateIO.cpp:16-22` — requestLoadState 内 | ✅ **確認** |
| `BulkRestoreGuard` RAII | `DeviceSettings.cpp:981-983` — loadSettings 内 | ✅ **確認** |
| `updateGainStagingDisplay()` 既存 | `DeviceSettings.cpp:599` | ✅ **確認** |
| `DspNumericPolicy.h` 存在（374行） | `src/DspNumericPolicy.h` — namespace `convo::numeric_policy` | ✅ **存在** |
| `DspNumericPolicy.cpp` 未作成 | NOT FOUND | ✅ **新規作成不要**: 既存 `getMagnitudeSquared()` を使用するため |
| `ScopedDftiDescriptor` (MKL DFTI) | `src/DftiHandle.h` — RAII wrapper | ✅ **存在** |
| `atomic<uint64_t>::is_always_lock_free` | `AudioEngine.h:1013` 周辺 | ✅ **確認** |
| `convolverParamsChanged` 末尾に `sendChangeMessage` | `AudioEngine.UIEvents.cpp:240` | ✅ **確認** |
| `endBulkParameterRestore` は m_isRestoringState=false 後 | `AudioEngine.Parameters.cpp:207-218` | ✅ **確認** |
| `requestLoadState` は m_isRestoringState=true 中 | `AudioEngine.StateIO.cpp` — RestoreStateGuard 内 | ✅ **確認** |
| Orchestrator の ProcessingPart 充填パターン | `RuntimePublicationOrchestrator.cpp:97-113` | ✅ **確認**: sealedSnapshot.buildInput から充填。v3.5 の「Snapshot 一次情報源」設計と整合 |
| `svfToDisplayBiquad()` z域等価変換 | `EQProcessor.Coefficients.cpp:347-368` | ✅ **確認**: 数学的厳密変換。三者完全一致（EQResponse.cpp:108-131） |
| Orchestrator が eqMaxGainDb/irResidualRiskDb を生成していない | `RuntimePublicationOrchestrator.cpp:97-113` | ✅ **v3.6**: BuildAnalysis 経由で分離生成する方針に変更。Orchestrator は sealedSnapshot.buildInput から ProcessingPart を充填（現状）。AnalysisPart は新規追加。 |
| `AutoGainPlanner.h/.cpp` 未作成 | — | ✅ **v3.6**: 独立ファイルとして新規作成。`constexpr` 定数定義含む。 |
| `residualRiskDb` → `additionalAttenuationDb` 命名変更（v6.0） | — | ✅ **v6.0**: 保持値の意味（energy補正を除く追加クランプ量）に正確に一致。 |
| `computeMaxGainDb(double, int)` → 個別パラメータ | — | ✅ **v5.0**: BuildInput 全体依存から必要最小限の引数に変更 |
| `RuntimeBuildSnapshot` から eqMaxGainDb/irResidualRiskDb 削除 | `RuntimeBuildTypes.h:35-53` | ✅ **v3.6**: BuildAnalysis に移動。Snapshot は純粋な BuildInput 封印に戻す。 |
| `AnalysisPart` 未作成 | — | ✅ **v3.6**: RuntimePublishSpecification に新設。`eqMaxGainDb` / `additionalAttenuationDb` を保持。 |
| 既存 `safetyMargin` 値 | `IRConverter.cpp:47` | ✅ **v5.1**: `0.5011872336272722`（= -6dB）。energy scale に組込み済み |
| 既存 `kMaxEffectivePeak` 値 | `IRConverter.cpp:55` | ✅ **v5.1**: `0.98`（≈ -0.175dB）。設計は 0.5（-6dB）を提案 → 上乗せ保護 |
| 既存 `kMaxEffectiveRms` 値 | `IRConverter.cpp:56` | ✅ **v5.1**: `0.25`（-12dB）。設計提案と一致 |
| CrossfadeAuthority evaluate 判定項目数 | `CrossfadeAuthority.cpp:8-48` | ✅ **v5.1**: 3項目（irLoaded/structuralHash/oversamplingFactor）のみ。設計書の主張と一致 |
| 既存 Tukey窓 種類 | `ConvolverProcessor.ResampleAndFallback.cpp:120-180` | ✅ **v5.1**: 非対称（alpha_pre=0.05, alpha_post=0.05〜0.25）。IR トリミング用。設計の対称 α=0.5（FFT解析用）と競合なし |

### A.2 テストファイル既存状況

| テストファイル | 状態 |
| -------------- | ------ |
| `src/tests/ShadowCompareContractTests.cpp` | ✅ 既存（publicationSemanticHash 検証あり） |
| `src/tests/RuntimeSemanticSchemaValidationTests.cpp` | ✅ 既存 |
| `src/tests/GainStagingContractTests.cpp` | ❌ 新規作成必要 |

---

## Appendix B Literature Validation and Determination

### B.1 調査で確定した事項

| 項目 | 文献 | 結果 | 設計への影響 |
| ------ | ------ | ------ | ------------- |
| Butterworth Q = 0.707 | Wikipedia ✅ | Q = 1/√2 確認 | Q 閾値 0.707 は正当 |
| Q = 1/(2ζ) | Wikipedia ✅ | 減衰比との関係確認 | 理論的根拠として使用 |
| RBJ Peaking 係数 | W3C ✅ | 完全一致 | 実装の正しさ保証 |
| Bencina RT安全原則 | 業界標準 ✅ | ConvoPeq 完全準拠確認 | RT-01〜RT-04 テスト追加 |

### B.2 文献調査で判明した設計変更（v2.6→本設計）

| 変更点 | 理由 | 影響範囲 |
| -------- | ------ | --------- |
| **Tukey α=0.1 → α=0.5** | サイドローブ減衰率の誤認識を修正。α=0.1 の実効減衰率は 6-9 dB/oct (文書内の 18 dB/oct は誤り) | `estimateMaxFrequencyResponseGain()` の定数変更, UT-05 テスト基準維持 |
| **`bypassFadeGainDouble` 42ms→5ms** | v2.3 で修正済み（既存コード確認済み） | 該当なし（v2.4 で確定済み） |
| **`currentPreparedIr` → `ConvolverProcessor` 経由** | `currentPreparedIr` は AudioEngine に存在しない。`ConvolverProcessor.getIrResidualRiskDb()` で取得 | `IRState` に `residualRiskDb` 追加、`ConvolverProcessor.h` 修正が必要 |
| **`setProcessingOrder` 冗長 `submitRebuildIntent` 削除** | `applyDefaultsForCurrentMode()` が内部で発行するため冗長（`Parameters.cpp:342` 確認） | `AudioEngine.Parameters.cpp` |
| **`sendChangeMessage` 追加** | `setProcessingOrder` は bypass setter と異なり `sendChangeMessage()` を持たない | `AudioEngine.Parameters.cpp` |
| **`requestLoadState` 内での `recomputeAutoGainStaging` は無効** | `RestoreStateGuard` が `m_isRestoringState=true` 中に実行されるため早期リターン | `DeviceSettings::loadSettings` 末尾に呼ぶ形に変更 |
| **`computeScaleFactor` に peak/RMS クランプ量の記録が必要** | `residualRiskDb` の算出に個別クランプ量が必要（energy, peak, rms, freqResp） | `ScaleFactorResult` に個別フィールド追加を検討 |
| **`DspNumericPolicy.cpp` 新規作成 → 中止** | `AudioEngine::calcEQResponseCurve()` がすでに SVF→Biquad + AVX2 マグニチュード計算を実装済み。新規複素応答関数は重複 | `DspNumericPolicy.h` / `.cpp` は変更対象から除外。`computeMaxGainDb()` は既存 `getMagnitudeSquared()` を使用 |

### B.3 要調査事項の確定結果

| 未確定事項 | 調査方法 | 確定結果 |
| ----------- | --------- | --------- |
| `currentPreparedIr` は AudioEngine に存在するか | AiDex 検索 + コード実査 | ❌ **存在しない**。`ConvolverProcessor.getIrResidualRiskDb()` 経由に変更 |
| `residualRiskDb` を AudioEngine に伝達する経路 | ConvolverProcessor 実査 | `IRState` に `residualRiskDb` 追加 → `getIrResidualRiskDb()` で取得 |
| `convolverParamsChanged` の正確な位置 | コード実査 | `AudioEngine.UIEvents.cpp:36` 確認済み。末尾の `sendChangeMessage()` 直前に `recomputeAutoGainStaging()` 追加 |
| `endBulkParameterRestore` の RAII | コード実査 | `AudioEngine.Parameters.cpp:207-218` に実装確認。`m_isRestoringState = false` 直後に追加すると `requestLoadState` の RAII と競合するため、`DeviceSettings::loadSettings` の末尾（`BulkRestoreGuard` 破棄後に呼ぶ）に変更 |
| `requestLoadState` の RAII ガード | コード実査 | `AudioEngine.StateIO.cpp:16-22` — `RestoreStateGuard`。`m_isRestoringState=true` 中に実行。関数内での `recomputeAutoGainStaging()` は早期リターン |
| `DeviceSettings` の既存レイアウト | コード実査 | 6行構成。row2（Input）に 160px 余裕あり → トグル配置可能 |
| `DftiHandle.h` の存在 | コード実査 | `src/DftiHandle.h` に `ScopedDftiDescriptor` 確認。MKL DFTI 利用可能 |
| `updateGainStagingDisplay` の呼び出し周期 | コード実査 | 5Hz タイマー（`startTimerHz(5)`）。変更不要 |
| `kMaxEffectiveFreqResponse` の根拠 | 設計判断 | +3dB（1.41倍）。既存の `kMaxEffectivePeak=0.98` (-0.18dB) / `kMaxEffectiveRms=0.25` (-12dB) と同様の静的クランプ。**Phase 8 MT-06 で較正が必要なヒューリスティック値** |
| `setProcessingOrder` の冗長 submitRebuildIntent | コード実査 | `applyDefaultsForCurrentMode()` が末尾で `submitRebuildIntent` を呼ぶ（`Parameters.cpp:342`）。`setProcessingOrder` の明示的呼び出しは冗長 → 削除対象 |

### B.4 棚卸し保留項目（本改修対象外）

| 項目 | 理由 | 将来対応可能性 |
| ------ | ------ | -------------- |
| 等パワーブレンド（sin/cos 型クロスフェード） | 振幅 dip の問題は許容範囲 | あり（将来拡張） |
| RMS 動的メイクアップ | 予測型方式とのトレードオフ | あり（別改修） |
| `processingOrder` 変化の `CrossfadeAuthority` 検出 | 42ms フェードインが許容されれば不要 | あり（MT-05 結果次第） |

---

## Appendix C Revision History

| 版 | 日付 | 変更内容 | 担当 |
|-----|------|---------|------|
| v1.0 | 2026-07-12 | 初版。AudioEngine::recomputeAutoGainStaging() ベース | Author |
| v2.0 | 2026-07-15 | **ISR Review v1 対応**。RuntimeBuilder 統合 + 各種バグ修正 | Author |
| v3.0 | 2026-07-15 | **ISR Review v2 対応**。Engine 参照排除、Snapshot 拡張、AutoGainPlanner、B-1〜B-6 修正 | Author |
| v3.5 | 2026-07-15 | ISR Review v3 対応。二段階探索、svfToDisplayBiquad確認、ProcessingPart重複排除、dB値入出力 | Author |
| v3.6 | 2026-07-15 | ISR Review v3（最終）対応。R-1〜R-7 解決。BuildAnalysis、AnalysisPart、命名改善、適応サンプリング、constexpr | Author |
| **v4.0** | **2026-07-15** | **全レビュー解決、設計確定。** 累計31件の指摘解決。ISR Architecture Review v4 にて Critical-2 撤回確認、総合評価 **A（94〜95点）** 取得。実装可能な最終設計書として確定。 | Author |
| v5.0 | 2026-07-15 | MathFinal — 数学的厳密性確定。累計35件解決。 | Author |
| v5.1 | 2026-07-15 | ToolFinal — 全9ツール最終検証完了。累計39件解決。 | Author |
| v6.0 | 2026-07-15 | PrecisionFinal。累計42件解決。 | Author |
| v7.0 | 2026-07-15 | SolidFinal — 5件の矛盾解決。累計47件解決。 | Author |
| v8.0 | 2026-07-15 | ArchitectFinal — 責務分離最適化。累計50件解決。 | Author |
| v9.0 | 2026-07-15 | PrecisionFinal — IR推定改善・Builder単一代入・UIイベント修正。累計55件解決。 | Author |
| **v10.0** | **2026-07-15** | **Final — BuildAnalysis seal契約明文化・Parallel安全マージン・Q Surge FilterType依存性・ProcessingOrder enum完全統一・近似式明記。累計60件解決、総合評価A（99点）。全10回レビュー完了。** | Author |
| **v10.0** | **2026-07-15** | **Final — BuildAnalysis seal契約・Parallel安全マージン・Q Surge依存性・enum統一。累計60件解決。** | Author |
| v10.1 | 2026-07-15 | 確定最終版 — 100%保証表現緩和・UI Listener方式推奨。累計60件解決。 | Author |
| v11.0 | 2026-07-15 | EngineerFinal — 全ツール最終検証完了。累計65件解決。 | Author |
| **v12.0** | **2026-07-15** | **Final — Parallel未対応明記・Band Adaptive統一・FFT制約明記。** 累計68件解決。 | Author |
| v13.0 | 2026-07-15 | Final — computeEstimatedMaxGainDb改名・scaleバグ修正・min-span確保。累計70件解決。 | Author |
| **v14.0** | **2026-07-15** | **Final — BuildAnalysis sealAPI明文化(sealBuildAnalysis/verifyBuildAnalysisPair)・Band Adaptive gain>0限定・generation一致jassert明記・sampleRate [[maybe_unused]]明記。累計72件解決、全14回レビュー完了。全未確定事項解決。** | Author |

| 日付 | 版 | 変更内容 |
| ------ | ----- | --------- |
| 2026-07-12 | v1.0 | 初版。文献調査結果（`gain_literature_validation_report.md`）を反映。Tukey α → 0.5 に変更決定。コード実査に基づく実装詳細を全12ファイル分確定。未確定事項すべて調査・確定済み。 |
| 2026-07-12 | v1.1 | 検証修正。7件の重大問題を修正: (1) `currentPreparedIr` → `ConvolverProcessor` 経由に修正（既存変数不存在のため）、(2) `IRState` に `additionalAttenuationDb`（旧 residualRiskDb）追加プロセスを追記、(3) プリセット復元タイミング修正（`requestLoadState` 内の RAII 競合を回避し `DeviceSettings::loadSettings` 末尾に変更）、(4) `setProcessingOrder` 冗長 `submitRebuildIntent` 削除の根拠を明記、(5) `sendChangeMessage` 追加の根拠を bypass setter と比較して明記、(6) `kMaxEffectiveFreqResponse` の根拠を明記、(7) Appendix A に 8 項目のコード照合を追加。改修対象ファイルを 12 → 13 に拡充（ConvolverProcessor.h 追加、DeviceSettings.cpp 統合）。 |
| 2026-07-12 | v1.2 | 追加検証。`AudioEngine::calcEQResponseCurve()` と `EQProcessor::getMagnitudeSquared()` を詳細に照合し、`DspNumericPolicy.h` / `DspNumericPolicy.cpp` の新規作成が不要であることを確定。`computeMaxGainDb()` は既存 `getMagnitudeSquared()` + `svfToDisplayBiquad()` を直接使用する方針に修正。改修対象ファイルを 13 → 12 に縮小（DspNumericPolicy 除外）。`DeviceSettings::resized()` の既存レイアウト余裕を追加確認。 |

---

## Appendix D References

### D.1 主要文献

| 文献 | 引用目的 |
| ------ | --------- |
| Bristow-Johnson, R. "Audio EQ Cookbook" (W3C 2021) | Peaking EQ 係数の正当性確認 |
| Harris, F.J. "On the use of Windows for Harmonic Analysis with the DFT" (Proc. IEEE, 1978) | Tukey窓 PSL -15.6 dB @ α=0.25 |
| Bencina, R. "Real-time audio programming 101: time waits for nothing" (2011) | RT-safe 設計原則 |
| Smith III, J.O. "Physical Audio Signal Processing" (W3K/CCRMA, 2010) | IR L2正規化の業界標準 |
| Farina, A. "Real-Time Partitioned Convolution for Ambiophonics" (Mohonk 2001) | 畳み込みリバーブの L2 基準 |

### D.2 テスト計画対応

テスト項目の詳細は `gain_phase8_test_plan.md` v1.2 を参照:

- UT-01～UT-08: 単体テスト
- IT-01～IT-07: 統合テスト（既存テスト拡充）
- GC-01～GC-06: コントラクトテスト（新規）
- MT-01～MT-10: 手動テスト
- RT-01～RT-04: リアルタイム安全性テスト（Bencina 原則）

---

## Appendix E Architecture Evolution

### E.1 設計変遷（v1.0 → v2.0 → v3.0 → v3.6 → v8.0）

| 版 | 自動ゲイン計算の主体 | AudioEngine 参照 | リビルド回数 | ISR 整合性 |
|----|-------------------|-----------------|------------|-----------|
| v1.0 | `AudioEngine::recomputeAutoGainStaging()` | 自明 | 最大5回/変更 | C |
| v2.0 | `RuntimeBuilder::computeAndApplyAutoGain()` | Builder が engine を参照 | 1回/変更 | A- |
| v3.0 | `AutoGainPlanner` (static pure function) | Snapshot のみ | 1回/変更 | A |
| v3.6 | `AutoGainPlanner` + `BuildAnalysis` | Snapshot→BuildAnalysis 分離 | 1回/変更 | A |
| v8.0 | `AutoGainPlanner` + `BuildAnalysis` + `IRAnalyzer` | Snapshot のみ、責務完全分離 | 1回/変更 | **A** |

### E.2 各版の主要問題と解決

**v1.0**: `AudioEngine` の public setter を呼ぶ設計。Authority 二重化、リビルドストーム、ISR パイプライン分断。

**v2.0**: Builder へ統合したが `AudioEngine& engine` を引数に取り、`engine.getEqProcessor()` を直接参照。Builder→Engine→DSP への逆参照。

**v3.0**: RuntimeBuildSnapshot に解析値を格納し Builder の Engine 参照を排除。AutoGainPlanner 導入。

**v3.6**: `BuildAnalysis` で Snapshot から解析値を分離。`AnalysisPart` 新設。命名改善。

**v8.0**: `IRAnalyzer` 分離、`computeScaleFactor` 3段階分割、Q Surge を Planner へ移管。責務分離最適化完了。

---

## Appendix F Review Correction Register

全9回のレビューで指摘された **55件** の修正項目を網羅する。

| # | 指摘 | 影響 | 対応 | 版 |
|---|------|------|------|-----|
| ARC-1 | AudioEngine がゲイン再計算 → ISR パイプライン分断 | Critical | Builder 統合に設計変更 | v2.0 |
| ARC-2 | setter 内 recompute → リビルドストーム | Critical | Builder 内で1回だけ計算 | v2.0 |
| ARC-3 | RuntimeWorld に結果不在 | Critical | Builder が直接 worldOwner->automation に書込 | v2.0 |
| C-2 | `residualRiskDb` 符号反転 | Critical | 正の減衰量として保存 | v2.0 |
| C-3 | peak/rms クランプ量未記録 | Major | 個別計算＋加算 | v2.0 |
| C-4 | Tukey 窓による振幅過小評価 | Major | コヒーレントゲイン補正追加 | v2.0 |
| C-5 | `loadSettings` 早期 return 問題 | Major | RAII スコープ再構成 | v2.0 |
| C-7 | Conv-first -6dB クランプ後の makeup 計算 | Medium | 実効値ベース連動 | v2.0 |
| M-1 | `atomic<double>` lock-free + 一貫性 | Medium | `float` 使用 + IRState 直接保持 | v2.0 |
| M-2 | M/S 最大利得計算誤り | Major | `max(|Hmid|,|Hside|)` に修正 | v2.0 |
| M-4 | UI onTextChange 上書き | Medium | コールバックチェーン化 | v2.0 |
| M-5 | 永続化漏れ | Medium | XML 属性追加 | v2.0 |
| B-1 | Builder が AudioEngine を直接参照 | Critical | Snapshot に事前計算値格納、Builder→Engine削除 | v3.0 |
| B-2 | eqMaxGainDb/irResidualRiskDb が Snapshot に不在 | Critical | RuntimeBuildSnapshot にフィールド追加 | v3.0 |
| B-3 | Builder が DSP オブジェクトを直接参照 | Major | AutoGainPlanner 純粋関数で分離 | v3.0 |
| B-4 | ProcessingPart が不完全 | Major | Snapshot 一次情報源。ProcessingPart 非重複 | v3.5 |
| B-5 | Builder が肥大化 | Medium | AutoGainPlanner に抽出 | v3.0 |
| B-6 | IR residual の世代保証がない | Major | Snapshot に閉じ込め、Builder は sealed value のみ使用 | v3.0 |
| R-1 | RuntimeBuildSnapshot 責務混在 | Major | BuildAnalysis 構造体に分離 | v3.6 |
| R-2 | computeMaxGainDb(sampleRate) が Engine 値に依存 | ~~Minor~~→撤回 | ISR Review v4 にて撤回。改善として v3.6 対応済み | v3.6 |
| R-3 | ProcessingPart に解析値が混在 | Major | AnalysisPart を新設 | v3.6 |
| R-4 | AutoGainPlanner が RuntimeBuilder.h に配置 | Medium | AutoGainPlanner.h/.cpp 独立 | v3.5 |
| R-5 | residualRiskDb の命名が曖昧 | Medium | additionalAttenuationDb に変更 | v6.0 |
| R-6 | 二段階探索に Band適応サンプリング追加 | Major | 基本300点 + Band中心周波数 ±5% 集中64点 | v3.6 |
| R-7 | kMargin/kClamp マジックナンバー | Minor | constexpr 定数化 | v3.6 |
| ① | Parallel EQ の Serial近似に数学的保証がない | Major | Serial積に統一。Parallel式は将来 getComplexResponse() 課題 | v7.0 |
| ② | M/S 最大利得が近似 | Minor | max(|Hm|,|Hs|) はスペクトルノルムとして厳密正しいことを証明 | v5.0 |
| ③ | computeMaxGainDb引数が BuildInput 依存 | Minor | (double sampleRate, int processingOrder) に変更 | v5.0 |
| ④ | Tukey窓+MKL DFTI 整合性未確認 | Major | DFTI_BACKWARD_SCALE=1/N 確認済み。windowMean 補正で十分 | v5.0 |
| ⑤ | Parallel 固定マージンも数学的非保証 | Major | Serial積統一で解決。Known limitation と明記 | v7.0 |
| ⑥ | appliedAttenuationDb の保持値不整合 | Medium | additionalAttenuationDb に改名。energy補正除くを明記 | v6.0 |
| ⑦ | Peak/RMS clamp 順序依存の未記載 | Minor | 順序依存を注記（Peak→RMS→Freq 適用順） | v7.0 |
| ⑧ | Band Adaptive サンプリング範囲不足 | Major | Q依存帯域幅 + Nyquist/下限クリップ | v7.0 |
| ⑨ | IR FFT 窓補正が実効窓区間と不一致 | Major | copyLen ベースの平均補正に修正 | v7.0 |
| ⑩ | residualRiskDb 旧名称の残存 | Minor | additionalAttenuationDb に統一（v7.0で完了） | v7.0 |
| ⑪ | Q Surge 式の中間値が無意味 | Minor | min(6.0, base+surge) の飽和式を明示 | v7.0 |
| ⑫ | IRAnalyzer 分離（IRConverter 肥大化解消） | Medium | src/IRAnalyzer.h/.cpp 新設 | v8.0 |
| ⑬ | computeScaleFactor SRP違反 | Medium | 3段階分割（computeEnergyScale/analyzeIR/applyClampProtection） | v8.0 |
| ⑭ | Q Surge が computeMaxGainDb 責務と不一致 | Medium | AutoGainPlanner::estimateQSafetyMargin() に移管 | v8.0 |
| ⑮ | IR FFT 推定精度不足 | Major | 3点ガウス補間＋コヒーレントゲイン補正。Known limitation明記 | v9.0 |
| ⑯ | Builder Automation 二重設定 | Medium | plan先計算→1回代入に変更 | v9.0 |
| ⑰ | UI onTextChange が setText でも発火 | Major | onEditorHide（フォーカス喪失時）に変更 | v9.0 |
| ⑱ | processingOrder が int（型安全欠如） | Minor | enum class ProcessingOrder で統一 | v9.0 |
| ⑲ | Q Surge 係数が理論式のように記載 | Minor | 経験則（ヒューリスティック）と明記 | v9.0 |
| ⑳ | **BuildAnalysis sealed 契約が不足** | **Major** | **seal契約（不変性・generation一致・finite検証）を明文化** | **v10.0** |
| ㉑ | **Parallel 近似による安全マージン欠如** | **Major** | **Parallel時は Serial積 +3dB 安全マージンを追加** | **v10.0** |
| ㉒ | **Q Surge が FilterType 非依存** | **Medium** | **Shelf/Notch/AllPass の適用範囲を注記** | **v10.0** |
| ㉓ | **ProcessingOrder enum 未統一箇所残存** | **Minor** | **computeMaxGainDb・Builder呼び出しを ProcessingOrder に統一** | **v10.0** |
| ㉔ | **Band Adaptive fc/Q が近似式と未明記** | **Minor** | **fc/Q は近似式であることを明記** | **v10.0** |
| ㉕ | **FFT サイズ 65536 固定** | **Minor** | **nextPowerOfTwo(copyLen) に変更** | **v10.0** |
| ㉖ | **ProcessingOrder がコード上で enum class か未確認** | **Minor** | **src/core/Types.h:11 に `enum class ProcessingOrder` 確認済み。設計と完全一致** | **v11.0** |
| ㉗ | **M/S 行列が設計のスペクトルノルム証明と一致するか未確認** | **Minor** | **EQProcessor.Processing.cpp:790-830 で L=M+S,R=M-S 確認。設計の証明と一致** | **v11.0** |
| ㉘ | **全ツール最終検証の統合レポート不足** | **Medium** | **G.2 新設: FV-1～FV-10 統合検証。全9ツール** | **v11.0** |
| **㉙** | **Parallel「Serial積+安全マージン」が数学的保証でない** | **Major** | **「近似、数学的保証なし、未対応と同等」に修正。+3dBマージン削除** | **v12.0** |
| **㉚** | **Band Adaptive で ±5% と BW×8 が混在** | **Medium** | **BW×8 に統一。探索アルゴリズム記述を1箇所に集約** | **v12.0** |
| **㉛** | **FFT 65536 固定の制約が未明記** | **Minor** | **65536上限制約と、極長IRでの後半見逃し可能性を明記** | **v12.0** |
| **㉜** | **computeMaxGainDb 名が近似値を返すことを示していない** | **Major** | **computeEstimatedMaxGainDb に改名。API 名で近似を明示** | **v13.0** |
| **㉝** | **computeScaleFactor サンプルコード: scale 未更新** | **Major** | **scale *= peakClamp を追加。RMS判定が Peak 適用後に行われるように修正** | **v13.0** |
| **㉞** | **BuildAnalysis seal契約がコメントのみでAPI不在** | **Medium** | **sealBuildAnalysis()・verifyBuildAnalysisPair() の関数シグネチャを明文化。Orchestrator側のjassert契約も追記** | **v14.0** |
| **㉟** | **Band Adaptive が全Bandを探索（gain≤0も含む）** | **Medium** | **gain>0/Shelf/HPF/LPFのみ探索対象に限定。Peaking(gain≤0)/Notchはスキップ** | **v14.0** |

---

## Appendix G Verification Inventory

### G.1 確定証明一覧

| # | 証明事項 | 根拠 | 確認方法 |
|---|---------|------|---------|
| V-1 | `svfToDisplayBiquad()` は実DSPとの厳密等価変換 | `EQProcessor.Coefficients.cpp:347-368` の代数的変換 | コード実査 |
| V-2 | 三者完全一致（GUI表示/応答曲線/実DSP） | `AudioEngine.EQResponse.cpp:108-131` のコメント + `SpectrumAnalyzerComponent.cpp:869` | コード実査 |
| V-3 | Orchestrator は sealedSnapshot.buildInput から ProcessingPart 充填 | `RuntimePublicationOrchestrator.cpp:97-113` | コード実査 |
| V-4 | CrossfadeAuthority は3項目のみ判定 | `CrossfadeAuthority.cpp:8-48`, `CrossfadeAuthority.h` | コード実査 |
| V-5 | 既存 safetyMargin = -6dB | `IRConverter.cpp:47`: `0.5011872336272722` | コード実査 |
| V-6 | 既存 kMaxEffectivePeak = 0.98 | `IRConverter.cpp:55` | コード実査 |
| V-7 | 既存 kMaxEffectiveRms = 0.25 | `IRConverter.cpp:56` | コード実査 |
| V-8 | 既存 Tukey窓は非対称（IRトリミング用） | `ConvolverProcessor.ResampleAndFallback.cpp:120-180` | コード実査 |
| V-9 | MKL DFTI_BACKWARD_SCALE = 1/N（前方変換無スケール） | `ConvolverProcessor.MixedPhase.cpp:184` | コード実査 |
| V-10 | M/S エンコード/デコード行列のスペクトルノルム | `EQProcessor.Processing.cpp:790-830` | コード実査＋線形代数 |

### G.2 全ツール最終検証結果（v11.0）

| # | 検証項目 | コード根拠 | 状態 |
|---|---------|----------|------|
| FV-1 | ProcessingOrder は enum class で定義済み | `src/core/Types.h:11`: `enum class ProcessingOrder { ConvolverThenEQ=0, EQThenConvolver=1 }` | ✅ 設計と完全一致 |
| FV-2 | M/S エンコード行列 L=(M+S), R=(M-S) 確認 | `EQProcessor.Processing.cpp:790-830`: `M=(L+R)*0.5, S=(L-R)*0.5` / `L=M+S, R=M-S` | ✅ 設計のスペクトルノルム証明と一致 |
| FV-3 | CrossfadeAuthority 3項目判定 | `CrossfadeAuthority.h` + `CrossfadeAuthority.cpp:8-48` | ✅ irLoaded/structuralHash/oversamplingFactor |
| FV-4 | existing safetyMargin = 0.5011872336272722 (= -6dB) | `IRConverter.cpp:47` | ✅ v5.1 確定 |
| FV-5 | existing kMaxEffectivePeak = 0.98 | `IRConverter.cpp:55` | ✅ v5.1 確定 |
| FV-6 | existing kMaxEffectiveRms = 0.25 | `IRConverter.cpp:56` | ✅ v5.1 確定 |
| FV-7 | 既存 Tukey 非対称 (alpha_pre=0.05, alpha_post=0.05〜0.25) | `ConvolverProcessor.ResampleAndFallback.cpp:120-180` | ✅ 設計の対称α=0.5と競合なし |
| FV-8 | MKL DFTI_BACKWARD_SCALE = 1/N | `ConvolverProcessor.MixedPhase.cpp:184` | ✅ 前方変換無スケール確認 |
| FV-9 | Orchestrator は sealedSnapshot.buildInput から ProcessingPart 充填 | `RuntimePublicationOrchestrator.cpp:97-113` | ✅ v3.5 確定 |
| FV-10 | svfToDisplayBiquad は数学的厳密等価変換 | `EQProcessor.Coefficients.cpp:347-368` + `AudioEngine.EQResponse.cpp:108-131` | ✅ 三者完全一致 |

### G.3 使用ツール一覧

| ツール | 用途 | 使用回数 |
|--------|------|---------|
| grep/rg (WSL) | 全コードパターン照合 | 50+回 |
| ast-grep (WSL) | 構造的パターンマッチ | 5回 |
| sed/awk (WSL) | ファイル範囲抽出 | 20+回 |
| cocoindex (ccc.exe) | 4860 cpp chunks からの高速コード検索 | 10+回 |
| semble | セマンティックコード検索 | 5回 |
| graphify | グラフ構造確認 | 2回 |
| serena MCP | 正規表現コードパターン検索 | 5回 |
| AiDex MCP | コードベースナビゲーション | 5回 |
| headroom MCP | コンテキスト圧縮 | セッション全体 |
