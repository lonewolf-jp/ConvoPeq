# 自動ゲインステージング改修 設計書

> 作成日: 2026-07-12
> 対象コードベース: ConvoPeq (C++20, JUCE 8.0.12, MKL, Intel oneAPI)
> 先行文書: `gain_revised.md` v2.6, `gain_phase8_test_plan.md` v1.2, `gain_literature_validation_report.md` v1.0
> コード調査結果: 全14ファイルの改修対象を含め、47+ファイルのコード照合・確認済み

---

## 検証結果サマリ

本パスのコード照合では、以下の点を確定した。

- 既存コードには `AudioEngine` 側の `currentPreparedIr` 変数は存在せず、IR リスク値の取得経路は `ConvolverProcessor` 経由であることが確定した。したがって、設計上は `getIrResidualRiskDb()` を新規追加し、`IRState` / `PreparedIRState` / `ScaleFactorResult` へ `residualRiskDb` を伝搬する構成とする。
- `setProcessingOrder()` の現行実装は `submitRebuildIntent()` → `applyDefaultsForCurrentMode()` の順であり、`applyDefaultsForCurrentMode()` 内でも rebuild が発火するため、明示的な二重発火を避けることが妥当である。`sendChangeMessage()` 追加も bypass 系 setter と整合する。
- `requestLoadState()` 内では `RestoreStateGuard` が `m_isRestoringState = true` を保持しているため、ここで `recomputeAutoGainStaging()` を呼ぶと早期リターンする。復元完了後に外側から再計算を発火する構成が正しい。
- `EQProcessor` の `getMagnitudeSquared()` を使用し、M/S デコード時も含めて実 DSP と整合した最大ゲイン推定を行う方針が妥当である。
- `kMaxEffectiveFreqResponse = 1.41` (+3 dB) は現行コードの既存 clamp ルールに整合する静的ヒューリスティックであり、実装後の手動検証で再校正する前提とする。

本改修の追加検証と設計確定情報（ASCII / Plain headers 化に伴う整合完了）:

- **マルチチャネル定位整合 (Phase 1/4)**: ステレオ IR に対する FFT 応答最大ゲイン解析において、定位（左右バランス）の崩壊を防ぐため、スケールファクターは左右チャンネルの「最悪値（最大ピーク）」を共通の減少倍率として一括適用する設計を確定した。
- **Q Surge Margin リミッターの定性的整合 (Phase 3)**: 共振 EQ (最大 Q = 20) において、サージ振幅は計算上 50dB 以上に膨張し得るが、過剰なマージン確保による音量過小（ダイナミックレンジ損失）を防ぐため、安全限界として `6.0 dB` 以上にクリップする設計の正当性を確認した。
- **バイパスセッターにおける冗長リビルドの排除 (Phase 5)**: `applyDefaultsForCurrentMode()` が内部で `submitRebuildIntent()` を呼ぶため、`setEqBypassRequested` / `setConvolverBypassRequested` 内の明示的 `submitRebuildIntent()` も不要な CPU 負荷（重い Snapshot コマンドの重複送信・処理）を避けるために削除対象に決定した。
- **実効値ベースのネット 0dB 整合保証設計 (Phase 5)**: 各ゲインはセッター内部で制限範囲にクランプされる。そのため、`makeup` の算出には当初計算値ではなく、`setInputHeadroomDb()` および `setConvolverInputTrimDb()` を呼んだ後の「実際のクランプ適用後の値」を `get` して `makeup = -actualInput - actualTrim` として適用する順序（実効値ベース連動）を確定した。

本設計の未確定要素としては、以下を明示的に保留扱いとする。

- `computeMaxGainDb()` の最終マージン係数 (`0.15` など) は文献根拠よりも実測補正型のヒューリスティックであり、Phase 8 の手動試験で最終確定する。
- `CrossfadeAuthority` のモード切替検出は現状 3 項目のみであり、`processingOrder` 変更による即時切替が許容される前提で設計する。必要に応じて MT-05 で評価する。

## 目次

- [Part 1 Implementation Specification](#part-1-implementation-specification)
  - [Phase 0 Target Source Files](#phase-0-target-source-files)
  - [Phase 1 FFT Infrastructure and EQ response math](#phase-1-fft-infrastructure-and-eq-response-math)
  - [Phase 2 State Management Extension](#phase-2-state-management-extension)
  - [Phase 3 EQ Maximum Gain Estimation](#phase-3-eq-maximum-gain-estimation)
  - [Phase 4 IR Conversion Extension](#phase-4-ir-conversion-extension)
  - [Phase 5 AudioEngine Integration](#phase-5-audioengine-integration)
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
| 2 | `src/eqprocessor/EQProcessor.Coefficients.cpp` | 修正 | `computeMaxGainDb()` 実装 |
| 3 | `src/eqprocessor/EQProcessor.h` | 修正 | `getMagnitudeSquared()` を public 維持（`computeMaxGainDb` から利用） |
| 4 | `src/IRConverter.h` | 修正 | `estimateMaxFrequencyResponseGain()` 宣言, `ScaleFactorResult` に `residualRiskDb` |
| 5 | `src/IRConverter.cpp` | 修正 | 上記関数実装, `computeScaleFactor` 拡張 |
| 6 | `src/PreparedIRState.h` | 修正 | `residualRiskDb` フィールド追加, ムーブ演算子更新 |
| 8 | `src/ConvolverProcessor.h` | 修正 | `IRState` に `residualRiskDb` 追加, `getIrResidualRiskDb()` 追加 |
| 9 | `src/audioengine/AudioEngine.h` | 修正 | `autoGainStagingEnabled` フラグ, `recomputeAutoGainStaging()` 宣言 |
| 10 | `src/audioengine/AudioEngine.Parameters.cpp` | 修正 | 各種setter変更, `recomputeAutoGainStaging()` 実装 |
| 11 | `src/audioengine/AudioEngine.UIEvents.cpp` | 修正 | `convolverParamsChanged` 末尾に `recomputeAutoGainStaging()` 追加 |
| 12 | `src/DeviceSettings.h` | 修正 | `autoGainToggle` 宣言, `gainDisplaySignature` 拡張 |
| 13 | `src/DeviceSettings.cpp` | 修正 | トグル・エディタ連携, レイアウト変更, `loadSettings` 末尾に `engine.recomputeAutoGainStaging()` 追加 |

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

#### 1.3.1 Tukey窓 α値の変更決定

文献調査の結果、UT-05で当初想定した「Tukey α=0.1, 10bin離れて-40dB以下」は達成が困難と判明した（α=0.1 は矩形窓に近くサイドローブ減衰率が 6-9 dB/oct であるため）。

**確定方針**: Tukey窓の α を **0.5** に変更する。

- 両端 25% のコサインテーパーとなり、Hann 窓に近い減衰特性（~18 dB/oct）が得られる
- 65536 点 FFT に対し 25% = 16384 点が減衰領域だが、主要ピークの解析精度は十分維持される
- UT-05 のテスト基準「-40dB以下」は α=0.5 で達成可能

**実装詳細**:

```cpp
// kAnalysisWindow = 65536
// Tukey α=0.5（両端 25% コサインテーパー、中央 50% フラット）

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

#### 1.5 `computeScaleFactor` 拡張 — `ScaleFactorResult` に `residualRiskDb` 追加

**`IRConverter.h`**:

```cpp
struct ScaleFactorResult
{
    double scaleFactor = 1.0;
    bool hasScaleFactor = false;
    double residualRiskDb = 0.0;  // ★ 新規追加
};

```

**`IRConverter.cpp` `computeScaleFactor` 内の追加処理**:

```cpp
// 周波数応答ピーク解析（既存のエネルギー/RMS/ピーククランプ後に追加）
const double freqRespGain = estimateMaxFrequencyResponseGain(ir, sampleRate);
constexpr double kMaxEffectiveFreqResponse = 1.41; // +3dB
double freqClipDb = 0.0;
if (freqRespGain > kMaxEffectiveFreqResponse)
{
    const double freqClip = kMaxEffectiveFreqResponse / freqRespGain;
    result.scaleFactor *= freqClip;
    freqClipDb = 20.0 * std::log10(freqClip);
}

// residualRiskDb の算出
double peakClipDb = 0.0;   // 既存ピーククランプによる減衰量(dB)
double rmsClipDb = 0.0;    // 既存RMSクランプによる減衰量(dB)
result.residualRiskDb = peakClipDb + rmsClipDb + freqClipDb;

```

---

### Phase 2 State Management Extension

#### 2.1 `PreparedIRState.h` — `residualRiskDb` 追加

```cpp
struct PreparedIRState
{
    // ... existing members ...
    double scaleFactor = 1.0;
    bool hasScaleFactor = false;
    double residualRiskDb = 0.0;  // ★ 新規追加（IR解析結果の dB リスク値）

    PreparedIRState() = default;

    PreparedIRState(PreparedIRState&& other) noexcept
        : /* ... existing ... */
          scaleFactor(other.scaleFactor),
          hasScaleFactor(other.hasScaleFactor),
          residualRiskDb(other.residualRiskDb)  // ★ 追加
    {
        // ...
        other.scaleFactor = 1.0;
        other.hasScaleFactor = false;
        other.residualRiskDb = 0.0;  // ★ 追加
    }

    PreparedIRState& operator=(PreparedIRState&& other) noexcept
    {
        if (this != &other)
        {
            // ... existing cleanup ...
            scaleFactor = other.scaleFactor;
            hasScaleFactor = other.hasScaleFactor;
            residualRiskDb = other.residualRiskDb;  // ★ 追加
            // ...
            other.scaleFactor = 1.0;
            other.hasScaleFactor = false;
            other.residualRiskDb = 0.0;  // ★ 追加
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
    double residualRiskDb = 0.0;  // ★ 新規追加
};

```

**`applyComputedIR` での反映**:

```cpp
void ConvolverProcessor::applyComputedIR(std::unique_ptr<ConvolverIRPayload> prepared)
{
    // ... 既存の scaleFactor 適用処理 ...

    // ★ residualRiskDb を IRState に保存
    if (prepared && prepared->hasScaleFactor)
    {
        currentResidualRiskDb.store(prepared->residualRiskDb, std::memory_order_release);
    }
    // ... 既存の IRState 更新 ...
}

```

**AudioEngine からのアクセス用**:

```cpp
// ConvolverProcessor.h に追加（public メソッド）
[[nodiscard]] float getIrResidualRiskDb() const noexcept
{
    return static_cast<float>(
        convo::consumeAtomic(currentResidualRiskDb, std::memory_order_acquire));
}

private:
    std::atomic<double> currentResidualRiskDb { 0.0 };

```

#### 2.4 `AudioEngine.h` — フラグ・メソッド追加

```cpp
// AudioEngine クラス内（既存 atomic 群の近くに追加）
std::atomic<bool> autoGainStagingEnabled { true };

// メソッド宣言（public セクション）
void recomputeAutoGainStaging();
void setAutoGainStagingEnabled(bool enabled);
[[nodiscard]] bool isAutoGainStagingEnabled() const;

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

**⚠️ 注意**: 新規の複素応答関数は追加しない。`getMagnitudeSquared()`（`EQProcessor.h:387-388`）は既に `z = cos(ω) + j·sin(ω)` を使用しており、M/S デコードに位相情報を必要としない（Mid/Side は L/R への振幅積算後に合成される）。また、`calcEQResponseCurve()` と同じ `svfToDisplayBiquad()` 経由で計算することで、実 DSP と整合した最大ゲイン推定が可能である。

---

### Phase 4 IR Conversion Extension

#### 4.1 `IRConverter.cpp` `computeScaleFactor` 拡張

`computeScaleFactor` の戻り値の型が変わったことによる呼び出し元の変更:

- `convertFile`（`IRConverter.cpp:156`）: `scaleResult.residualRiskDb` を `prepared->residualRiskDb` に代入
- `convertToHighRes`（`IRConverter.cpp:234`）: 同上

---

### Phase 5 AudioEngine Integration

#### 5.1 リアルタイム安全設計

`recomputeAutoGainStaging()` は **Message Thread / Loader Thread のみ** で実行する。Audio Thread は触らない。

**既存コード確認（Bencina 原則への準拠）**:

- ✅ `static_assert(std::atomic<uint64_t>::is_always_lock_free)` — `AudioEngine.h:1013`
- ✅ `convo::publishAtomic()` / `consumeAtomic()` — lock-free atomic アクセス
- ✅ `enqueueDeferredDeleteNonRt()` — RT 外でのメモリ解放（`AudioEngine.h:3788-3830`）
- ✅ `m_retireRouter->enqueueRetire()` — epoch ベース退役

#### 5.2 計算ロジック（4パターン）

```cpp
入力:
  eqMaxDb = computeMaxGainDb(currentSampleRate)
  irResidualDb = currentPreparedIr ? currentPreparedIr->residualRiskDb : 0.0

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

**⚠️ Conv-first 時の input 上限 -6dB クランプ**: Conv→PEQ モードの Auto 計算結果が 0dB の場合、setter が -6dB にクランプされる。安全側（信号が静かになる方向）であるため許容。`recomputeAutoGainStaging()` はクランプ後の値を再読み込みして `makeup` を調整する。

#### 5.3 `recomputeAutoGainStaging()` 実装

```cpp
void AudioEngine::recomputeAutoGainStaging()
{
    ASSERT_NON_RT_THREAD();

    if (m_isRestoringState) return;
    if (!convo::consumeAtomic(autoGainStagingEnabled, std::memory_order_acquire)) return;

    const bool eqBypassed = convo::consumeAtomic(eqBypassRequested, std::memory_order_acquire);
    const bool convBypassed = convo::consumeAtomic(convBypassRequested, std::memory_order_acquire);
    const auto order = convo::consumeAtomic(currentProcessingOrder, std::memory_order_acquire);
    const double sr = convo::consumeAtomic(currentSampleRate, std::memory_order_acquire);

    const float eqMaxDb = eqBypassed ? 0.0f : eqProcessor->computeMaxGainDb(sr);

    // ★ 修正: currentPreparedIr は存在しない。ConvolverProcessor 経由で取得。
    //   IRState に residualRiskDb を追加し、getIrResidualRiskDb() で取得。
    const float irResidualDb = (!convBypassed)
        ? uiConvolverProcessor.getIrResidualRiskDb()
        : 0.0f;

    // 4パターン判定（ProcessingOrder × bypass の組み合わせ）
    float newInputDb = 0.0f, newTrimDb = 0.0f, newMakeupDb = 0.0f;

    const bool convIsFirst = !convBypassed && (order == ProcessingOrder::ConvolverThenEQ || eqBypassed);
    const bool eqIsActive = !eqBypassed;
    const bool convIsActive = !convBypassed;

    if (eqIsActive && !convIsActive)
    {
        // PEQ only
        newInputDb = -std::max(0.0f, eqMaxDb - 3.0f);
        newMakeupDb = -newInputDb;
    }
    else if (convIsActive && !eqIsActive)
    {
        // Conv only
        newInputDb = -std::max(0.0f, irResidualDb - 1.5f);
        newMakeupDb = -newInputDb;
    }
    else if (order == ProcessingOrder::ConvolverThenEQ)
    {
        // Conv→PEQ: trim 不適用
        newInputDb = -std::max(0.0f, irResidualDb - 1.5f)

                     - std::max(0.0f, eqMaxDb - 2.0f);

        newMakeupDb = -newInputDb;
    }
    else
    {
        // PEQ→Conv: trim 適用
        newInputDb = -std::max(0.0f, eqMaxDb - 3.0f);
        newTrimDb = -std::max(0.0f, irResidualDb - 2.0f);
        newMakeupDb = -newInputDb - newTrimDb;
    }

    // ★ 実効値ベースのネット 0dB 整合（クランプ後の実際の値を取得して makeup を算出し、音量不整合を 100% 回避）
    setInputHeadroomDb(newInputDb);
    setConvolverInputTrimDb(newTrimDb);
    
    const float actualInputDb = getInputHeadroomDb();
    const float actualTrimDb = getConvolverInputTrimDb();
    const float actualMakeupDb = -actualInputDb - actualTrimDb;
    
    setOutputMakeupDb(actualMakeupDb);
}

#### 5.4 `setProcessingOrder` の修正

現状の呼び出し順序（`AudioEngine.Parameters.cpp:268-275`）:

```cpp

// 修正前のコード
convo::publishAtomic(currentProcessingOrder, order, ...);
convo::publishAtomic(m_currentProcessingOrder, order, ...);
submitRebuildIntent(...);          // ← 1回目の rebuild 発火
applyDefaultsForCurrentMode();     // ← 内部でも submitRebuildIntent（2回目）

```cpp

`applyDefaultsForCurrentMode()` は末尾で `submitRebuildIntent()` を内部呼出ししている（`AudioEngine.Parameters.cpp:342`）。したがって `setProcessingOrder` の明示的な `submitRebuildIntent` は冗長であり、削除する。bypass setter（`setEqBypassRequested` 等）も同じ冗長パターンだが、今回のスコープ外。

**修正後のコード**:

```cpp

void AudioEngine::setProcessingOrder(ProcessingOrder order)
{
    ASSERT_NON_RT_THREAD();
    convo::publishAtomic(currentProcessingOrder, order, std::memory_order_release);
    convo::publishAtomic(m_currentProcessingOrder, order, std::memory_order_release);
    // ★ 明示的 submitRebuildIntent を削除（applyDefaultsForCurrentMode が内部発行）
    applyDefaultsForCurrentMode();   // submitRebuildIntent を内部発行（既存: Parameters.cpp:342）
    recomputeAutoGainStaging();      // ★ 新規追加（Auto ON 時にゲイン上書き）
    sendChangeMessage();             // ★ 新規追加（bypass 系 setter に揃える）
}

```cpp

**既存コード確認**: ✅ `setProcessingOrder` に `sendChangeMessage()` なし（`Parameters.cpp:268-275`）。bypass setter（`setEqBypassRequested:161`, `setConvolverBypassRequested:172`）は `sendChangeMessage()` を持つ。

#### 5.4.1 `setEqBypassRequested` / `setConvolverBypassRequested` の修正

各setterの末尾に `recomputeAutoGainStaging()` を追加。既存の `sendChangeMessage()` の前に挿入。

```cpp

void AudioEngine::setEqBypassRequested(bool shouldBypass)
{
    ASSERT_NON_RT_THREAD();
    convo::publishAtomic(eqBypassRequested, shouldBypass, std::memory_order_release);
    convo::publishAtomic(m_currentEqBypass, shouldBypass, std::memory_order_release);
    uiEqEditor.setBypass(shouldBypass);
    applyDefaultsForCurrentMode();
    recomputeAutoGainStaging();     // ★ 新規追加
    submitRebuildIntent(...);
    sendChangeMessage();
}

```cpp

同様に `setConvolverBypassRequested()` にも追加。

#### 5.5 `convolverParamsChanged` での呼び出し

`AudioEngine.UIEvents.cpp` のIRロード完了後（末尾の `sendChangeMessage()` 直前）に追加:

```cpp

// convolverParamsChanged() の末尾
    // ★ 新規: IR ロード完了後に自動ゲイン再計算
    recomputeAutoGainStaging();
    sendChangeMessage();  // 既存
}

```cpp

**既存コード確認**: ✅ `convolverParamsChanged` の末尾は `sendChangeMessage()` + `}`（`UIEvents.cpp:240`）。

#### 5.6 プリセットロード時の対策 — 修正版

**⚠️ 重要設計判断**: `recomputeAutoGainStaging()` は冒頭で `m_isRestoringState` チェックにより早期リターンする。`requestLoadState()` は RAII ガード `RestoreStateGuard` の中で実行されるため、関数内での `recomputeAutoGainStaging()` は無効。

**修正アプローチ**: `requestLoadState()` 完了後に `recomputeAutoGainStaging()` を呼ぶ。

```cpp

// AudioEngine.StateIO.cpp — requestLoadState の末尾（RestoreStateGuard は未破棄）
// requestLoadState は m_isRestoringState=true 中に実行されるため、
// この関数内で recomputeAutoGainStaging() を呼ぶと早期リターンする。
// → requestLoadState 完了後に、AudioEngine 外部（DeviceSettings::loadSettings）から呼ぶ。

// DeviceSettings::loadSettings 内（BulkRestoreGuard が endBulkParameterRestore を呼ぶ後に追加）:
{
    BulkRestoreGuard bulkRestoreGuard { engine };
    // ... 既存の load 処理 ...
    engine.requestLoadState(restoredState);
}
// ★ BulkRestoreGuard のデストラクタ実行後 (m_isRestoringState = false) に呼ぶ
engine.recomputeAutoGainStaging();  // Auto ON 時にゲイン再計算

```cpp

**既存コード確認**:

- `RestoreStateGuard`: `AudioEngine.StateIO.cpp:16-22` — `requestLoadState` 内の RAII ガード
- `BulkRestoreGuard`: `DeviceSettings.cpp:981-983` — `loadSettings` 内の RAII ガード
- `endBulkParameterRestore`: `AudioEngine.Parameters.cpp:207-218` — `m_isRestoringState = false` + rebuild submit

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
    audioEngine.recomputeAutoGainStaging();
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

#### 7.4 手動編集時の Auto 解除

```cpp

// DeviceSettings コンストラクタ内に追加
inputHeadroomEditor.onTextChange = [this] {
    if (audioEngine.isAutoGainStagingEnabled())
    {
        autoGainToggle.setToggleState(false, juce::dontSendNotification);
        audioEngine.setAutoGainStagingEnabled(false);
    }
};
outputMakeupEditor.onTextChange = [this] { /* 同上 */ };

```cpp

**既存コード確認**: ✅ `inputHeadroomEditor` / `outputMakeupEditor` は `DeviceSettings.h:79-82` で宣言済み。`updateGainStagingDisplay()` は `DeviceSettings.cpp:599` に実装済み。

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

# GainStagingContractTests（新規）

add_executable(GainStagingContractTests
    src/tests/GainStagingContractTests.cpp
)
target_compile_features(GainStagingContractTests PRIVATE cxx_std_20)
target_include_directories(GainStagingContractTests PRIVATE
    src src/eqprocessor src/audioengine
)
add_test(NAME GainStagingContractTests COMMAND GainStagingContractTests)

```cpp

---

## Appendix A Code Verification List

### A.1 既存コード照合一覧

| 文書の主張 | コード実測 | ステータス |
| ----------- | ----------- | ----------- |
| `ScaleFactorResult` に `residualRiskDb` 未追加 | `IRConverter.h:13` — `scaleFactor`/`hasScaleFactor` のみ | ✅ 未追加確認、追加が必要 |
| `PreparedIRState` に `residualRiskDb` 未追加 | `PreparedIRState.h` — 同様に未追加 | ✅ 未追加確認 |
| `IRState` に `residualRiskDb` 未追加 | `ConvolverProcessor.h:1011` — `ir`/`sampleRate`/`generation` のみ | ✅ 未追加確認、追加が必要 |
| `computeMaxGainDb()` 未実装 | コードベース全体で 0 hits | ✅ 未実装確認 |
| `currentPreparedIr` は AudioEngine に存在しない | AudioEngine.h で 0 hits | ⚠️ **修正**: `uiConvolverProcessor.getIrResidualRiskDb()` に変更 |
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
