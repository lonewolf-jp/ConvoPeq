# Audio Thread libm 規約準拠チェックレポート

更新日: 2026-03-16

## 監査対象

- `src/AudioEngine.cpp`
- `src/AudioEngine.h`
- `src/EQProcessor.cpp`
- `src/EQProcessor.h`
- `src/MKLNonUniformConvolver.cpp`
- `src/MKLNonUniformConvolver.h`

## 判定基準

- **準拠**: Audio Thread 実行経路で `std::abs / std::isfinite / std::pow / std::sqrt / std::exp / std::sin / std::cos / std::tan` などの libm 呼び出しが無い。
- **非準拠**: 上記呼び出しが Audio Thread 実行経路に残存。
- **対象外**: Message/UI/Rebuild Thread 限定経路。

## 結論（要約）

- **総合判定: 準拠（監査対象の Audio Thread 経路）**
  - `AudioEngine` / `EQProcessor` / `MKLNonUniformConvolver` の監査対象 Audio Thread 経路で、`std::abs` / `std::isfinite` などの libm 呼び出し残存は確認されなかった。

## 詳細一覧

### 1) AudioEngine 系

#### Audio Thread 経路

- `AudioEngine::getNextAudioBlock()` → `DSPCore::process()` / `processOutputDouble()`
  - 監査結果: 今回確認した範囲で **libm 呼び出し残存なし**（直近対応により `std::abs/std::isfinite` 置換済み）。

#### Audio Thread 対象外（スレッド上許容）

- `AudioEngine::prepareToPlay(...)`
  - `std::isfinite`, `std::abs` 使用あり（Message Thread 想定）。
- `AudioEngine::rebuildThreadLoop()`
  - `std::abs` 使用あり（Rebuild Worker Thread）。
- `AudioEngine::calcEQResponseCurve(...)`
  - `std::abs`, `std::isfinite`, `std::sqrt` 使用あり（表示用計算、Audio Thread 直結ではない）。
- `AudioEngine::UltraHighRateDCBlocker::init(...)` (`AudioEngine.h`)
  - `std::exp`, `std::isfinite` 使用あり（コメントで Audio Thread 外呼び出しを明示）。

### 2) EQProcessor 系

#### Audio Thread 経路

- `EQProcessor::process(juce::dsp::AudioBlock<double>&)`
  - `target` 比較の `std::abs` を libm 非依存判定へ置換済み。
- `EQProcessor::processAGC(...)`（`process()` から呼び出し）
  - `std::isfinite(inputRMS/outputRMS/envIn/envOut/currentGain)` を libm 非依存判定へ置換済み。
  - 監査結果: **libm 呼び出し残存なし**。

#### Audio Thread 対象外（スレッド上許容）

- `prepareToPlay(...)`: `std::abs`
- `createBandNode(...)`: `std::abs`
- 係数計算群（`calc*SVF`, `calc*Biquad`, `calcBiquadCoeffs`, `getMagnitudeSquared(freq, ...)`）
  - `std::pow`, `std::tan`, `std::sin`, `std::cos`, `std::sqrt`, `std::isfinite`, `std::abs`
  - 主用途は係数生成/表示計算であり、Audio Thread 直接経路ではない。

### 3) MKLNonUniformConvolver 系

#### Audio Thread 経路

- `processDirectBlock(...)`, `processLayerBlock(...)` ほか RT 処理
  - 監査結果: 今回確認した範囲で **libm 呼び出し残存なし**。

#### Audio Thread 対象外（スレッド上許容）

- `applySpectrumFilter(...)`（Message Thread 明示）
  - `std::round`, `std::sqrt`, `std::pow`, `std::cos`, `std::exp`

## 対応済み項目

1. `EQProcessor::process()` の `std::abs` を libm 非依存比較へ置換。
2. `EQProcessor::processAGC()` の `std::isfinite` を SSE 判定ヘルパーへ置換。
3. 置換後、`Audio Thread` 経路で `std::abs/std::isfinite` 残存ゼロを再確認。

## 備考

- 本レポートは「Audio Thread 経路の規約準拠」を目的とするため、Message/UI/Rebuild Thread での libm 使用は違反として扱っていない。
- 監査は静的な呼び出し経路確認に基づく。実行時スレッド逸脱がないことは別途運用（呼び出し規約）で担保する。
