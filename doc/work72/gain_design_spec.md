# 自動ゲインステージング改修 設計書

> 作成日: 2026-07-15 (v2.0 — ISR Architecture Review 反映)
> 対象コードベース: ConvoPeq (C++20, JUCE 8.0.12, MKL, Intel oneAPI)
> 先行文書: `gain_revised.md` v2.6, `gain_phase8_test_plan.md` v1.2, `gain_literature_validation_report.md` v1.0
> レビュー: `checked.md` (2026-07-13), ISR Architecture Review (2026-07-15)
> コード調査結果: 全16ファイルの改修対象を含め、60+ファイルのコード照合・確認済み

---

## 検証結果サマリ

### アーキテクチャ上の重要設計判断

本設計の最大の設計判断は、**自動ゲイン計算を AudioEngine の setter 連鎖から RuntimeBuilder のビルド工程へ移管した** 点にある。

**背景**: 現行 ConvoPeq の ISR (Immediate State Runtime) アーキテクチャは、
```
Intent → RuntimePublicationSpecification → RuntimeBuilder → RuntimeWorld → RT
```
という一方向パイプラインを確立している。Automation 値（`inputHeadroomGain`, `outputMakeupGain`, `convolverInputTrimGain`）も `RuntimeBuilder::buildRuntimePublishWorld()` 内で `ProcessingPart` から `worldOwner->automation` へ写像される（`RuntimeBuilder.cpp:269-275`）。Audio Thread は `world->automation.inputHeadroomGain` を読み取る（`AudioEngine.h:2820`）。

v1.0 設計書では `AudioEngine::recomputeAutoGainStaging()` が public setter を呼ぶ設計だったが、これは以下を引き起こす:
- **Authority の二重化**: AudioEngine の setter 経路と RuntimeBuilder の Spec→World 経路が並立する
- **リビルドストーム**: 単一のパラメータ変更から最大5回の `submitRebuildIntent` が発火する
- **ISR パイプライン分断**: `Intent → Specification → Builder → World` の流れを setter 内の即時 atomic 更新が迂回する

**v2.0 の設計方針**: 自動ゲイン計算を `RuntimeBuilder::buildRuntimePublishWorld()` 内のビルド工程に統合する。これにより:
- 自動ゲイン計算は rebuild ごとに1回だけ実行される（リビルドストーム排除）
- 計算結果は直接 `worldOwner->automation` に書き込まれ、Audio Thread は従来どおり RuntimeWorld を読むだけ
- `RuntimePublishSpecification::ProcessingPart` に `autoGainStagingEnabled` フラグを追加し、Orchestrator が Builder に伝達する

### コード照合確定事項

- 既存コードには `AudioEngine` 側の `currentPreparedIr` 変数は存在せず、IR リスク値の取得経路は `ConvolverProcessor` 経由であることが確定した。設計上は `getIrResidualRiskDb()` を新規追加し、`IRState` / `PreparedIRState` / `ScaleFactorResult` へ `residualRiskDb` を伝搬する構成とする。
- `RuntimePublishSpecification::ProcessingPart`（`RuntimeBuilder.h:40-49`）は Orchestrator が sealedSnapshot/engine atomic から収集したゲイン値の DTO。これに `autoGainStagingEnabled` を追加し、Builder が自動ゲイン計算の要否を判定する。
- `EQProcessor` の `getMagnitudeSquared()` を使用し、M/S デコード時も含めて実 DSP と整合した最大ゲイン推定を行う方針は妥当である。ただし M/S 評価は `max(|Hmid|, |Hside|)` で行う（詳細は Phase 3）。
- `kMaxEffectiveFreqResponse = 1.41` (+3 dB) は現行コードの既存 clamp ルールに整合する静的ヒューリスティック。実装後の手動検証で再校正する前提とする。
- `setProcessingOrder()` の `sendChangeMessage()` 追加は bypass 系 setter と整合するが、`recomputeAutoGainStaging()` は AudioEngine に追加せず Builder が代替する。

### レビュー修正事項 (checked.md / ISR Review 反映)

| # | 指摘 | 影響 | 対応 |
|---|------|------|------|
| ARC-1 | AudioEngine がゲイン再計算 → ISR パイプライン分断 | **Critical** | Builder 統合に設計変更 |
| ARC-2 | setter 内 recompute → リビルドストーム | **Critical** | Builder 内で1回だけ計算 |
| ARC-3 | RuntimeWorld に結果不在 | **Critical** | Builder が直接 worldOwner->automation に書込 |
| C-2 | `residualRiskDb` 符号反転 | **Critical** | 正の減衰量として保存 |
| C-3 | peak/rms クランプ量未記録 | **Major** | 個別計算＋加算 |
| C-4 | Tukey 窓による振幅過小評価 | **Major** | コヒーレントゲイン補正追加 |
| C-5 | `loadSettings` 早期 return 問題 | **Major** | RAII スコープ再構成 |
| C-7 | Conv-first -6dB クランプ後の makeup 計算 | **Medium** | 実効値ベース連動を確認済み |
| M-1 | `atomic<double>` lock-free + 一貫性 | **Medium** | `float` 使用 + 同一 atomic での一貫更新 |
| M-2 | M/S 最大利得計算誤り | **Major** | `max(|Hmid|, |Hside|)` に修正 |
| M-4 | UI onTextChange 上書き | **Medium** | コールバックチェーン化 |
| M-5 | 永続化漏れ | **Medium** | XML 属性追加 |

### 本設計の未確定要素（保留）

- `computeMaxGainDb()` の最終マージン係数 (`0.15` など) は文献根拠よりも実測補正型のヒューリスティック。Phase 8 の手動試験で最終確定する。
- `CrossfadeAuthority` のモード切替検出は現状 3 項目のみ。`processingOrder` 変更による即時切替が許容される前提で設計する。必要に応じて MT-05 で評価する。
- Tukey 窓 α=0.5 の UT-05 テスト基準「-40dB以下」の達成確認は未実施。実装後に側波帯減衰率を実測検証する。

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
---

## Part 1 Implementation Specification

### Phase 0 Target Source Files

| # | ファイル | 操作 | 変更内容 |
| --- | --------- | ------ | --------- |
| 1 | `src/eqprocessor/EQProcessor.h` | 修正 | `computeMaxGainDb()` 宣言追加 |
| 2 | `src/eqprocessor/EQProcessor.Coefficients.cpp` | 修正 | `computeMaxGainDb()` 実装（M/S 評価は `max(|Hmid|,|Hside|)`） |
| 3 | `src/eqprocessor/EQProcessor.h` | 修正 | `getMagnitudeSquared()` を public 維持（`computeMaxGainDb` から利用） |
| 4 | `src/IRConverter.h` | 修正 | `estimateMaxFrequencyResponseGain()` 宣言, `ScaleFactorResult` に `residualRiskDb`（正値減衰量） |
| 5 | `src/IRConverter.cpp` | 修正 | 上記関数実装（Tukey α=0.5 + コヒーレントゲイン補正）, `computeScaleFactor` 拡張（peak/rms 個別計算） |
| 6 | `src/PreparedIRState.h` | 修正 | `residualRiskDb` フィールド追加（`float`, 正値）, ムーブ演算子更新 |
| 7 | `src/audioengine/AudioEngine.h` | 修正 | `autoGainStagingEnabled` atomic フラグ, `setAutoGainStagingEnabled()` 宣言 |
| 8 | `src/audioengine/AudioEngine.Parameters.cpp` | 修正 | `setProcessingOrder` に `sendChangeMessage()` 追加（`recomputeAutoGainStaging` は追加しない） |
| 9 | `src/audioengine/RuntimeBuilder.h` | 修正 | `RuntimePublishSpecification::ProcessingPart` に `autoGainStagingEnabled` 追加 |
| 10 | `src/audioengine/RuntimeBuilder.cpp` | 修正 | `buildRuntimePublishWorld()` 内で自動ゲイン計算を実行（`computeAndApplyAutoGain`） |
| 11 | `src/audioengine/AudioEngine.RebuildDispatch.cpp` | 修正 | `captureBuildParameterSnapshot()` / `BuildInput` に `autoGainStagingEnabled` 追加 |
| 12 | `src/ConvolverProcessor.h` | 修正 | `IRState` に `residualRiskDb` 追加（`float`）, `getIrResidualRiskDb()` 追加 |
| 13 | `src/DeviceSettings.h` | 修正 | `autoGainToggle` 宣言, `gainDisplaySignature` 拡張 |
| 14 | `src/DeviceSettings.cpp` | 修正 | トグル・エディタ連携（コールバックチェーン）, レイアウト変更, `loadSettings` スコープ再構成, `saveSettings` に `autoGainStagingEnabled` 永続化 |

---

### Phase 1 FFT Infrastructure and EQ response math

#### 1.1 既存 EQ 応答計算の確認と方針

既存コードでは、`AudioEngine::calcEQResponseCurve()`（`src/audioengine/AudioEngine.EQResponse.cpp`）が SVF 係数を `EQProcessor::svfToDisplayBiquad()` で等価 Biquad に変換し、AVX2 版の `calcMagnitudesForBand()` で周波数応答を計算している。この実装は `AudioEngine.EQResponse.cpp:36-58` で、既に以下を満たしている。

- `calcSVFCoeffs()` → `svfToDisplayBiquad()` のパスは、実際の DSP 処理と表示曲線が一致する（同ファイルコメント参照）。
- `EQChannelMode::Stereo / Left / Right / Mid / Side` のチャンネルモード別積算が実装済み。
- M/S モードでは L/R 両方に同じマグニチュードを掛ける（`Mid/Side` はその後の L/R 合成で位相は不要）。

したがって、本改修では新たに `getComplexResponse()` を追加するのではなく、既存の `getMagnitudeSquared()`（`EQProcessor.Coefficients.cpp:325-337`）を使って最大ゲインを推定する。

**結論**: `DspNumericPolicy.h` / `DspNumericPolicy.cpp` の新規作成は不要。`computeMaxGainDb()` は `EQProcessor` 内に実装し、既存 `getMagnitudeSquared()` を周波数スキャンで使用する。

#### 1.2 `IRConverter.h` — `estimateMaxFrequencyResponseGain` 宣言追加

```cpp
// IRConverter.h の class IRConverter 内に追加（public 静的メソッドとして）

// IRの周波数応答ピークをFFT解析で推定
// Tukey窓（α=0.5 変更推奨）適用後の複素スペクトル振幅の最大値を返す
// 戻り値: 線形振幅値（倍率）。IRが無効な場合は1.0
static double estimateMaxFrequencyResponseGain(
    const juce::AudioBuffer<double>& ir,
    double sampleRate) noexcept;

```

#### 1.3.1 Tukey窓 α値の変更決定 + コヒーレントゲイン補正（C-4）

文献調査の結果、UT-05で当初想定した「Tukey α=0.1, 10bin離れて-40dB以下」は達成が困難と判明したため、α を **0.5** に変更する。

- 両端 25% のコサインテーパーとなり、Hann 窓に近い減衰特性（~18 dB/oct）が得られる
- 65536 点 FFT に対し 25% = 16384 点が減衰領域だが、主要ピークの解析精度は十分維持される
- UT-05 のテスト基準「-40dB以下」は α=0.5 で達成可能

**⚠️ コヒーレントゲイン補正（C-4 修正）**:

Tukey α=0.5 の平均ゲイン（コヒーレントゲイン）は約 0.75。窓関数を適用すると FFT の DC 成分が `0.75 * N` になり、周波数応答ピークを **約 -2.5 dB 過小評価** する。このままでは安全側ではなく「過小評価 = 過剰なヘッドルーム」になる。

**修正**: FFT ピーク検出後にコヒーレントゲインで除算する。

```cpp
// estimateMaxFrequencyResponseGain 内
const double windowSum = std::accumulate(tukeyWindow.begin(), tukeyWindow.end(), 0.0);
const double windowMean = windowSum / static_cast<double>(kAnalysisWindow);
// ... FFT 実行後 ...
maxMagnitude /= windowMean;  // コヒーレントゲイン補正
```

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
    constexpr int kAnalysisWindow = 65536;
    // 変更: Tukey α = 0.5（文献調査結果に基づき決定）
    constexpr double kTukeyAlpha = 0.5;

    const int numSamples = ir.getNumSamples();
    const int numChannels = ir.getNumChannels();
    if (numSamples <= 0 || numChannels <= 0)
        return 1.0;

    const int copyLen = std::min(numSamples, kAnalysisWindow);

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

#### 1.5 `computeScaleFactor` 拡張 — `ScaleFactorResult` に `residualRiskDb` 追加（C-2/C-3）

**⚠️ 符号設計（C-2 修正）**: `residualRiskDb` は **正の減衰量 [dB]** として保存する。元設計では `peakClipDb + rmsClipDb + freqClipDb` を負値（`20*log10(clip) < 0`）としていたが、`recomputeAutoGainStaging` → `computeAndApplyAutoGain` 内で `-max(0, irResidualDb - 1.5f)` と使用するため、正値の減衰量として保存しないと期待通りに動作しない。

**⚠️ Peak/RMS 個別計算（C-3 修正）**: 既存の `computeScaleFactor` は `absoluteClamp = min(peakClamp, rmsClamp)` で一括計算しているため、どちらが効いたか分からない。以下の修正で peak/rms を分離して記録する。

**`IRConverter.h`**:

```cpp
struct ScaleFactorResult
{
    double scaleFactor = 1.0;
    bool hasScaleFactor = false;
    // ★ v2.0: 正の減衰量 [dB]（IR のピーク/RMS/周波数応答による減衰量の合計）
    float residualRiskDb = 0.0f;
};

```

**`IRConverter.cpp` `computeScaleFactor` 内の追加処理（C-2/C-3 修正反映）**:

```cpp
// ★ v2.0: Peak/RMS を個別計算し、減衰量 [dB] を正値で記録（C-2/C-3）
double peakAttenDb = 0.0, rmsAttenDb = 0.0, freqAttenDb = 0.0;

// 既存 energy ベースの scaleFactor 適用後...
double scale = result.scaleFactor;

// Peak クランプ（既存の kMaxEffectivePeak を分離計算）
constexpr double kMaxEffectivePeak = 0.5;
const double irPeak = /* 既存のピーク検出値 */;
if (irPeak * scale > kMaxEffectivePeak)
{
    const double peakClamp = kMaxEffectivePeak / (irPeak * scale);
    result.scaleFactor *= peakClamp;
    peakAttenDb = -20.0 * std::log10(peakClamp);  // 正値の減衰量
}

// RMS クランプ（既存の kMaxEffectiveRms を分離計算）
constexpr double kMaxEffectiveRms = 0.25;
const double irRms = std::sqrt(irEnergySum / numSamples);
if (irRms * scale > kMaxEffectiveRms)
{
    const double rmsClamp = kMaxEffectiveRms / (irRms * scale);
    result.scaleFactor *= rmsClamp;
    rmsAttenDb = -20.0 * std::log10(rmsClamp);  // 正値の減衰量
}

// 周波数応答ピーク解析（Tukey α=0.5 + コヒーレントゲイン補正）
const double freqRespGain = estimateMaxFrequencyResponseGain(ir, sampleRate);
constexpr double kMaxEffectiveFreqResponse = 1.41; // +3dB
if (freqRespGain > kMaxEffectiveFreqResponse)
{
    const double freqClip = kMaxEffectiveFreqResponse / freqRespGain;
    result.scaleFactor *= freqClip;
    freqAttenDb = -20.0 * std::log10(freqClip);  // 正値の減衰量
}

// residualRiskDb = 正の減衰量合計 [dB]（C-2 符号修正）
result.residualRiskDb = static_cast<float>(peakAttenDb + rmsAttenDb + freqAttenDb);

```

---

### Phase 2 State Management Extension

#### 2.1 `PreparedIRState.h` — `residualRiskDb` 追加（`float` 型, 正値）

```cpp
struct PreparedIRState
{
    // ... existing members ...
    double scaleFactor = 1.0;
    bool hasScaleFactor = false;
    float residualRiskDb = 0.0f;  // ★ 新規追加（IR解析結果の正値減衰量 [dB]）

    PreparedIRState() = default;

    PreparedIRState(PreparedIRState&& other) noexcept
        : /* ... existing ... */
          scaleFactor(other.scaleFactor),
          hasScaleFactor(other.hasScaleFactor),
          residualRiskDb(other.residualRiskDb)  // ★ 追加（float）
    {
        // ...
        other.scaleFactor = 1.0;
        other.hasScaleFactor = false;
        other.residualRiskDb = 0.0f;  // ★ 追加（float リテラル）
    }

    PreparedIRState& operator=(PreparedIRState&& other) noexcept
    {
        if (this != &other)
        {
            // ... existing cleanup ...
            scaleFactor = other.scaleFactor;
            hasScaleFactor = other.hasScaleFactor;
            residualRiskDb = other.residualRiskDb;  // ★ 追加（float）
            // ...
            other.scaleFactor = 1.0;
            other.hasScaleFactor = false;
            other.residualRiskDb = 0.0f;  // ★ 追加（float リテラル）
        }
        return *this;
    }
};

```

#### 2.2 `IRConverter::convertFile` / `convertToHighRes` での反映

`IRConverter.cpp` 内で `computeScaleFactor` 呼び出し後に:

```cpp
prepared->residualRiskDb = scaleResult.residualRiskDb;

```

#### 2.3 `ConvolverProcessor.h` — `IRState` に `residualRiskDb` 追加

**⚠️ 重要**: `PreparedIRState` は `applyComputedIR` 後に破棄される。`residualRiskDb` は `IRState` に移して保持する必要がある。

```cpp
// ConvolverProcessor.h — IRState 構造体の拡張
struct IRState {
    std::unique_ptr<juce::AudioBuffer<double>> irOwner;
    const juce::AudioBuffer<double>* ir = nullptr;
    double sampleRate = 0.0;
    uint64_t generation = 0;
    float residualRiskDb = 0.0f;  // ★ 新規追加（float, 正値減衰量）
};

```

**`applyComputedIR` での反映（M-1 一貫性確保）**:

`IRState` に `residualRiskDb` を直接保持し、`acquireIRState()` 経由で読み取る。これにより別途 `atomic<double>` を管理する必要がなくなり、IRState 全体の一貫性が保証される。

```cpp
void ConvolverProcessor::applyComputedIR(std::unique_ptr<ConvolverIRPayload> prepared)
{
    // ... 既存の IRState 生成 ...

    // ★ residualRiskDb を IRState に直接保存（M-1 一貫性）
    if (prepared)
    {
        auto newState = std::make_unique<IRState>();
        // ... 既存の irOwner/ir/sampleRate/generation 設定 ...
        newState->residualRiskDb = prepared->residualRiskDb;
        // ... publishAtomic(currentIRState, newState.release(), ...) ...
    }
}

```

**AudioEngine からのアクセス用**:

```cpp
// ConvolverProcessor.h に追加（public メソッド）
// IRState から residualRiskDb を読み取る（acquireIRState 経由で一貫性保証）
[[nodiscard]] float getIrResidualRiskDb() const noexcept
{
    auto* state = acquireIRState();
    return (state != nullptr) ? state->residualRiskDb : 0.0f;
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

#### 3.1 `EQProcessor.h` — `computeMaxGainDb()` 宣言

```cpp
// class EQProcessor の public セクションに追加（既存の calcBiquadCoeffs 等の隣）
[[nodiscard]] float computeMaxGainDb(double sampleRate) const;  // 最大ゲイン（dB）

```

#### 3.2 `EQProcessor.Coefficients.cpp` — 実装

```cpp
float EQProcessor::computeMaxGainDb(double sampleRate) const
{
    // 1. 対数周波数スケールで 300 点を生成
    constexpr int kNumPoints = 300;
    std::array<double, kNumPoints> omegas;
    // 20Hz〜Nyquist、対数分布

    // 2. 有効バンドの Biquad 係数を取得
    //    calcSVFCoeffs(type, freq, gain, q, sr) → svfToDisplayBiquad()
    //    AudioEngine.EQResponse.cpp:105-116 と同じパスを使用することで、
    //    実際の DSP 処理と表示曲線の三者一致を維持する。

    // 3. L/R チャンネルのマグニチュード二乗をカスケード積算
    //    getMagnitudeSquared(biquadCoeffs, omega) を使用。
    //    M/S バンドは Stereo と同様に L/R 両方に掛ける（Mid/Side は最終的な L/R 合成の前段）。

    // 4. 全周波数・全チャンネルにおける最大線形ゲインを保持
    //    totalGainDb も考慮（AGC OFF 時のみ適用）。

    // 5. Q Surge Margin の算出
    //    - HPF/LPF: +1.5dB 固定
    //    - Peaking(gain>0 && Q>0.707): gain × 0.15 × (Q/0.707)
    //    - 合計を 6.0dB でクリップ
    // 閾値 0.707 は Butterworth Q = 1/√2（Wikipedia 確認済み）
    // 0.15 係数はヒューリスティック値（文献に理論根拠なし、Phase 8 で実測検証）

    // 6. 最終 = 20*log10(maxLinearGain) + qSurgeMarginDb
}

```

**⚠️ 注意（M-2 修正）**: 新規の複素応答関数は追加しない。`getMagnitudeSquared()`（`EQProcessor.h:387-388`）は既に `z = cos(ω) + j·sin(ω)` を使用しており、M/S デコードに位相情報を必要としない。ただし、M/S モードの最大利得計算は `max(|Hmid|, |Hside|)` とする。単純な `L = M+S` は振幅を過大評価するため、正しい M/S 評価は以下の式を用いる:

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

**Parallel 構造の注意**: Parallel filter structure では `H = 1 + Σ(Hi - 1)` で計算する。本設計は Serial を主要想定とするが、Parallel 時も Serial 積を安全な上限として使用可能。`computeMaxGainDb()` の実装コメントに明記する。

`calcEQResponseCurve()` と同じ `svfToDisplayBiquad()` 経由で計算することで、実 DSP と整合した最大ゲイン推定が可能である。

---

### Phase 4 IR Conversion Extension

#### 4.1 `IRConverter.cpp` `computeScaleFactor` 拡張

`computeScaleFactor` の戻り値の型が変わったことによる呼び出し元の変更:

- `convertFile`（`IRConverter.cpp:156`）: `scaleResult.residualRiskDb` を `prepared->residualRiskDb` に代入
- `convertToHighRes`（`IRConverter.cpp:234`）: 同上

---

### Phase 5 RuntimeBuilder Integration（★ v2.0 全面改訂）

#### 5.0 設計方針（v1.0 → v2.0 変更点）

v1.0 では `AudioEngine::recomputeAutoGainStaging()` を新設し、setter 内で呼び出す設計だった。v2.0 ではこれを **RuntimeBuilder のビルド工程に統合**する。

**変更理由**:

| 観点 | v1.0 (AudioEngine setter 経由) | v2.0 (RuntimeBuilder 統合) |
|------|-------------------------------|---------------------------|
| Authority | AudioEngine と RuntimeBuilder で二重化 | RuntimeBuilder に一本化 |
| リビルド回数 | 5回/変更 (C-6) | 1回/変更 |
| データフロー | atomic → setter → atomic → Spec → World | Spec → Builder → World（一方向） |
| Audio Thread | atomics fallback or World | World のみ（既存通り） |
| テスト容易性 | AudioEngine 全体の結合が必要 | Builder ユニットテスト可能 |

**v2.0 データフロー**:

```
User操作 (processingOrder/IR/EQ変更)
  → submitRebuildIntent (1回)
    → Worker Thread:
      captureBuildParameterSnapshot()  // autoGainStagingEnabled 含む
      → BuildInput.autoGainStagingEnabled = ...
      → RuntimeBuilder::build(input, convolverSnapshot)
        → DSPCore 生成・準備
      → Orchestrator:
        → RuntimePublishSpecification 生成
          → ProcessingPart.autoGainStagingEnabled = ...
          → ProcessingPart.inputHeadroomGain = 1.0 (仮)
        → RuntimeBuilder::buildRuntimePublishWorld(spec)
          → computeAndApplyAutoGain()  // ★ ここで自動ゲイン計算
            → EQProcessor::computeMaxGainDb() を呼出
            → ConvolverProcessor::getIrResidualRiskDb() を呼出
            → worldOwner->automation.inputHeadroomGain = ...  // 直接書込
            → worldOwner->automation.outputMakeupGain = ...
            → worldOwner->automation.convolverInputTrimGain = ...
          → worldOwner->freeze()
      → publishWorld(worldOwner)
        → RuntimeStore に RCU 公開

Audio Thread:
  acquireReadToken()
  → world->automation.inputHeadroomGain  // 既存経路で読取
```

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
// ★ v2.0: RuntimeBuilder::computeAndApplyAutoGain 内で使用
入力:
  eqMaxDb = engine.getEqProcessor()->computeMaxGainDb(currentSampleRate)
  irResidualDb = engine.uiConvolverProcessor.getIrResidualRiskDb()  // IRState から一貫性読取

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

#### 5.3 `computeAndApplyAutoGain()` — RuntimeBuilder 内での自動ゲイン計算

`RuntimeBuilder::buildRuntimePublishWorld()` 内で、`ProcessingPart` の写像直後に呼び出す。AudioEngine の public setter は使用せず、直接 `worldOwner->automation` に書き込む。

```cpp
// RuntimeBuilder.cpp — buildRuntimePublishWorld() 内の拡張
convo::aligned_unique_ptr<const RuntimePublishWorld>
RuntimeBuilder::buildRuntimePublishWorld(
    const convo::RuntimeBuildSnapshot* sealedSnapshot,
    const RuntimePublishSpecification& spec) noexcept
{
    // ... 既存の Topology/Routing/Execution/Overlap/Latency 写像 ...

    // ★ v9.4 P0: ProcessingPart から読み取り（Orchestrator が収集済み）
    worldOwner->automation.eqBypassed = spec.processing.eqBypassed;
    worldOwner->automation.convBypassed = spec.processing.convBypassed;
    worldOwner->automation.softClipEnabled = spec.processing.softClipEnabled;
    worldOwner->automation.saturationAmount = spec.processing.saturationAmount;
    worldOwner->automation.inputHeadroomGain = spec.processing.inputHeadroomGain;
    worldOwner->automation.outputMakeupGain = spec.processing.outputMakeupGain;
    worldOwner->automation.convolverInputTrimGain = spec.processing.convolverInputTrimGain;

    // ★ v2.0: 自動ゲインステージング — ProcessingPart 値の上書き
    if (spec.processing.autoGainStagingEnabled)
    {
        computeAndApplyAutoGain(worldOwner, spec, sealedSnapshot,
                                engine, nextGraphGeneration);
    }

    // ... 既存の Resource/Timing/Publication/Retire 写像 ...
    worldOwner->freeze();
    return worldOwner;
}
```

```cpp
// ★ v2.0 新規: 自動ゲイン計算（RuntimeBuilder 内 private メソッド）
void RuntimeBuilder::computeAndApplyAutoGain(
    RuntimePublishWorld* worldOwner,
    const RuntimePublishSpecification& spec,
    const convo::RuntimeBuildSnapshot* sealedSnapshot,
    AudioEngine& engine,
    std::uint64_t generation) noexcept
{
    ASSERT_NON_RT_THREAD();

    const bool eqBypassed = spec.processing.eqBypassed;
    const bool convBypassed = spec.processing.convBypassed;
    const int order = spec.processing.processingOrder;
    const double sr = (sealedSnapshot != nullptr)
        ? sealedSnapshot->buildInput.sampleRate
        : 48000.0;

    // EQ最大ゲイン: DSPCore から EQProcessor を取得
    float eqMaxDb = 0.0f;
    if (!eqBypassed && engine.getEqProcessor() != nullptr)
    {
        eqMaxDb = engine.getEqProcessor()->computeMaxGainDb(sr);
    }

    // IR残存リスク: ConvolverProcessor から取得
    float irResidualDb = 0.0f;
    if (!convBypassed)
    {
        irResidualDb = engine.uiConvolverProcessor.getIrResidualRiskDb();
    }

    // 4パターン判定（ProcessingOrder × bypass の組み合わせ）
    float newInputDb = 0.0f, newTrimDb = 0.0f, newMakeupDb = 0.0f;

    if (!eqBypassed && convBypassed)
    {
        // PEQ only
        newInputDb = -std::max(0.0f, eqMaxDb - 3.0f);
    }
    else if (eqBypassed && !convBypassed)
    {
        // Conv only
        newInputDb = -std::max(0.0f, irResidualDb - 1.5f);
    }
    else if (order == static_cast<int>(ProcessingOrder::ConvolverThenEQ))
    {
        // Conv→PEQ: trim 不適用, input 上限 -6dB
        newInputDb = -(std::max(0.0f, irResidualDb - 1.5f)
                       + std::max(0.0f, eqMaxDb - 2.0f));
        newInputDb = std::max(newInputDb, -6.0f);  // Conv-first 安全上限
    }
    else
    {
        // PEQ→Conv: trim 適用
        newInputDb = -std::max(0.0f, eqMaxDb - 3.0f);
        newTrimDb = -std::max(0.0f, irResidualDb - 2.0f);
    }

    // ★ 実効値ベースのネット 0dB 整合（クランプ適用後の値で makeup を算出）
    const float clampedInputDb = juce::jlimit(-12.0f, 0.0f, newInputDb);
    const float clampedTrimDb  = juce::jlimit(-12.0f, 0.0f, newTrimDb);
    const float computedMakeupDb = -clampedInputDb - clampedTrimDb;
    const float clampedMakeupDb = juce::jlimit(0.0f, 12.0f, computedMakeupDb);

    // 直接 worldOwner->automation に書込（setter / submitRebuildIntent は呼ばない）
    worldOwner->automation.inputHeadroomGain =
        juce::Decibels::decibelsToGain(static_cast<double>(clampedInputDb));
    worldOwner->automation.outputMakeupGain =
        juce::Decibels::decibelsToGain(static_cast<double>(clampedMakeupDb));
    worldOwner->automation.convolverInputTrimGain =
        juce::Decibels::decibelsToGain(static_cast<double>(clampedTrimDb));

    // ★ Diagnostic: 自動ゲイン計算結果を rebuild 診断ログに出力
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    diagLog("[AutoGain] gen=" + juce::String(static_cast<juce::int64>(generation))
        + " eq=" + juce::String(eqMaxDb) + " ir=" + juce::String(irResidualDb)
        + " in=" + juce::String(clampedInputDb)
        + " trim=" + juce::String(clampedTrimDb)
        + " makeup=" + juce::String(clampedMakeupDb));
#endif
}
```

**設計上の重要点**:

1. **1回の rebuild で完結**: `computeAndApplyAutoGain` は `buildRuntimePublishWorld()` 内で呼ばれ、`worldOwner` に直接書き込む。`submitRebuildIntent()` は呼ばない → リビルドストーム完全排除 (C-6)
2. **public setter を呼ばない**: `setInputHeadroomDb()` / `setOutputMakeupDb()` / `setConvolverInputTrimDb()` は経由しない → atomic 二重更新排除
3. **クランプ後の実効値で makeup 計算**: `clampedInputDb` / `clampedTrimDb` を `jlimit` 後に決定し、`computedMakeupDb = -clampedInputDb - clampedTrimDb` でネット 0dB を保証 (C-7)
4. **AudioEngine のフラグのみ**: `autoGainStagingEnabled` は atomic フラグとして AudioEngine に残し、UI トグルで操作。`captureBuildParameterSnapshot()` が `BuildInput` に含めて Orchestrator 経由で Builder に伝達する

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
    //   → 自動ゲイン計算は RuntimeBuilder::computeAndApplyAutoGain が行う
    //   → applyDefaultsForCurrentMode 内の submitRebuildIntent 重複を排除
    sendChangeMessage();  // ★ 追加（bypass 系 setter に揃える）
}
```

**v1.0 からの変更点**:

- `applyDefaultsForCurrentMode()` の呼び出しを削除。これにより submitRebuildIntent の二重発火が解消される（C-6）
- `recomputeAutoGainStaging()` は削除（Builder 統合に伴い）
- `sendChangeMessage()` は保持（bypass 系 setter との一貫性）

**補足**: `applyDefaultsForCurrentMode()` の削除に伴い、デフォルトゲイン値の設定は RuntimeBuilder の `computeAndApplyAutoGain()` が代替する。Auto ON 時は計算値が適用され、Auto OFF 時は直近の手動設定値が `ProcessingPart` 経由で適用される（変更なし）。

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
    // ★ v2.0: recomputeAutoGainStaging は不要。setter 内で submitRebuildIntent を呼び、
    //   RuntimeBuilder::computeAndApplyAutoGain が自動ゲインを計算する。
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
// DeviceSettings コンストラクタ内 — ★ M-4 コールバックチェーン化
auto oldInputOnChange = std::move(inputHeadroomEditor.onTextChange);
inputHeadroomEditor.onTextChange = [this, oldInputOnChange = std::move(oldInputOnChange)] {
    // Auto 有効時は disable して UI 同期
    if (audioEngine.isAutoGainStagingEnabled())
    {
        autoGainToggle.setToggleState(false, juce::dontSendNotification);
        audioEngine.setAutoGainStagingEnabled(false);
    }
    // 元の setInputHeadroomDb 呼び出しを維持（上書き防止）
    if (oldInputOnChange)
        oldInputOnChange();
};
auto oldOutputOnChange = std::move(outputMakeupEditor.onTextChange);
outputMakeupEditor.onTextChange = [this, oldOutputOnChange = std::move(oldOutputOnChange)] {
    if (audioEngine.isAutoGainStagingEnabled())
    {
        autoGainToggle.setToggleState(false, juce::dontSendNotification);
        audioEngine.setAutoGainStagingEnabled(false);
    }
    if (oldOutputOnChange)
        oldOutputOnChange();
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
| `ScaleFactorResult` に `residualRiskDb` 未追加 | `IRConverter.h:13` — `scaleFactor`/`hasScaleFactor` のみ | ✅ 未追加確認、追加が必要 |
| `PreparedIRState` に `residualRiskDb` 未追加 | `PreparedIRState.h` — 同様に未追加 | ✅ 未追加確認 |
| `IRState` に `residualRiskDb` 未追加 | `ConvolverProcessor.h:1011` — `ir`/`sampleRate`/`generation` のみ | ✅ 未追加確認、追加が必要 |
| `computeMaxGainDb()` 未実装 | コードベース全体で 0 hits | ✅ 未実装確認 |
| `currentPreparedIr` は AudioEngine に存在しない | AudioEngine.h で 0 hits | ✅ v2.0: `uiConvolverProcessor.getIrResidualRiskDb()` に変更 |
| `BulkRestoreGuard` のスコープ構造 | `DeviceSettings.cpp:979-983` | ✅ スコープ再構成（C-5） |
| `RuntimeBuilder` の buildRuntimePublishWorld 内 automation 写像 | `RuntimeBuilder.cpp:269-275` | ✅ ProcessingPart → worldOwner->automation |
| `captureBuildParameterSnapshot` の auto gain 関連 | `AudioEngine.RebuildDispatch.cpp:33-50` | ✅ `BuildInput` 拡張（v2.0） |
| `setProcessingOrder` の applyDefaultsForCurrentMode 削除 | `AudioEngine.Parameters.cpp:268-275` | ✅ v2.0: 削除（Builder が代替） |
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
| v1.0 | 2026-07-12 | 初版。AudioEngine::recomputeAutoGainStaging() ベースの設計 | Author |
| v2.0 | 2026-07-15 | **ISR Architecture Review 対応 全面改訂**。RuntimeBuilder 統合、C-2/C-3/C-4/C-5/C-6/M-1/M-2/M-4/M-5 修正 | Author |

| 日付 | 版 | 変更内容 |
| ------ | ----- | --------- |
| 2026-07-12 | v1.0 | 初版。文献調査結果（`gain_literature_validation_report.md`）を反映。Tukey α → 0.5 に変更決定。コード実査に基づく実装詳細を全12ファイル分確定。未確定事項すべて調査・確定済み。 |
| 2026-07-12 | v1.1 | 検証修正。7件の重大問題を修正: (1) `currentPreparedIr` → `ConvolverProcessor.getIrResidualRiskDb()` に修正（既存変数不存在のため）、(2) `IRState` に `residualRiskDb` 追加プロセスを追記、(3) プリセット復元タイミング修正（`requestLoadState` 内の RAII 競合を回避し `DeviceSettings::loadSettings` 末尾に変更）、(4) `setProcessingOrder` 冗長 `submitRebuildIntent` 削除の根拠を明記、(5) `sendChangeMessage` 追加の根拠を bypass setter と比較して明記、(6) `kMaxEffectiveFreqResponse` の根拠を明記、(7) Appendix A に 8 項目のコード照合を追加。改修対象ファイルを 12 → 13 に拡充（ConvolverProcessor.h 追加、DeviceSettings.cpp 統合）。 |
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
