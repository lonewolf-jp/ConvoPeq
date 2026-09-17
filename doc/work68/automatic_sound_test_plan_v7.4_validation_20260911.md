# ConvoPeq 音質評価自動化 改修計画書 v7.4 検証報告書

**検証日**: 2026-09-11
**検証対象**: `doc/work68/automatic_sound_test_plan_v7.4.md`（1517 行）
**検証ベース**: 現行 HEAD（2026-09-11 時点・work89/work92 改修後）
**検証方法**: 実コード grep/ast-grep/rg 全文照合 + 音響工学数値検証 + ツール横断確認（ccc / graphify / cppcheck / clang-tidy / tgrep / ast-grep / fd / ag / fzf / rg）

---

## 0. 総合判定

| 項目 | 判定 |
|------|------|
| 設計思想（RT-safe SPSC + BG Thread WAV + CapturePoint 3 点） | ✅ **妥当** |
| Part 2「調査確定事項」の内容 | ⚠️ **一部陳腐化**（2026-07-16 調査後のコード変更未反映 4 件） |
| §1.13 Phase 1 テスト設計（TC-01〜41） | ⚠️ **構造は妥当だが閾値に重大な前提誤り 3 件** |
| ライン番号アンカー群 | ❌ **全滅状態**（work89/work92 リファクタでファイル分割・行ズレ） |
| 実装開始可否 | ❌ **Phase 0 実装開始不可 → 本報告書の修正 6 項目を反映後に可** |

---

## 1. 重大な発見（実装をブロックする順）

### 発見-1【最重要】`--cli-output-wav=` イコール結合形式が現行パーサーで動作しない

計画書 §1.13.4 `cli_runner.py` は全引数を `--cli-output-wav=<path>` 形式（`=` 結合）で発行する。

現行 `runCommandLineAutomation()`（src/MainWindow.cpp:331）の `findValue` は**スペース区切りトークン**を前提とする実装：

```cpp
// src/MainWindow.cpp:343-356（実装）
juce::StringArray tokens;
tokens.addTokens(trimmedCommandLine, true);   // 空白区切り
...
for (int i = 0; i < tokens.size(); ++i)
{
    if (!tokens[i].equalsIgnoreCase(key))     // 「--cli-ir」と「C:\path\to.wav」が別トークン
        continue;
    if (i + 1 < tokens.size())
        return tokens[i + 1];                  // 次トークンを値とする
    return {};
}
```

**影響**: `--cli-output-wav=out.wav` のような 1 トークン文字列は `equalsIgnoreCase("--cli-output-wav")` に一致せず、**値なし（空 String）を返す**。計画書 §1.8 の CLI パターン例（`--cli-ir={ir_file}` 等）も同様に全滅する。なお計画書付属の既存テスト（D116 系で実運用済みの harness）は `--cli-ir <path>` スペース区切りで動作していることから、**計画書側が誤記している可能性が高い**。Phase 0 で `=` 結合対応をパーサーへ追加するか、`cli_runner.py` をスペース区切りに修正するかを**決定して契約を固定**する必要がある（本報告書は後者＝スペース区切りに統一する修正を推奨。C:\ パスの空白を含む場合 quoting も考慮必要）。

### 発見-2【閾値致命】kOutputHeadroom = -1dB 常時減衰が TC 全般の理論値に反映されていない

全出力パスに定数減衰が存在する（実測 5 箇所）:

```
src/audioengine/AudioEngine.Processing.DSPCoreIO.cpp:362:    constexpr double kOutputHeadroom = 0.8912509381337456;  // -1.0 dBFS
src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp:581: 同値
src/NoiseShaperLearner.cpp:18: 同値
```

- ディザー適用時: `processStereoBlock(dataL, dataR, numSamples, kOutputHeadroom)` がディザー内部で **信号×0.8912** を行う（PsychoacousticDither.h:327）
- ディザー不適用時: `for (i) dataL[i] *= kOutputHeadroom;`（DSPCoreIO.cpp:476-481 / DSPCoreDouble.cpp:666-672）
- 加えて PeakLimiter が閾値 **kPLThreshold = 0.8413951287507587（-1.5 dBFS）** で常時動作（DSPCoreIO.cpp:527-529, DSPCoreDouble.cpp:710）

**影響する TC**:
| TC | 計画書閾値 | 実際の理論値 | 誤差量 |
|----|-----------|------------|--------|
| TC-23 出力レベル | RMS偏差 ≤ 0.1dB | 入力 -6dBFS → 出力 **-7.0 dBFS**（-1dB headroom）| **1.0 dB の閾値オーバー → 必ず FAIL** |
| TC-31a/b Null | RMS ≤ -138 dBFS | dither off 時、入力×0.8912 とキャプチャ比較では差が**頭量レベルで出る**（比較対象が入力 WAVそのままなら） | 設計依存 |
| TC-37 数値透過 | RMS ≤ 1 ULP | Float 変換後（PostDither float path）は -1dB 減衰分が差分になる | **FAIL 確定** |

**修正必須**: 全 TC の理論値計算に `-1.0 dB（0.8912509381337456）` と `PeakLimiter -1.5 dBFS 閾値` を織り込むか、テスト用に headroom バイパス CLI（`--cli-bypass-output-headroom` 等を Phase 0 に追加）を用意する。後者推奨（テスト純度向上）。

### 発見-3【設計重大】Standalone ビルドでは Float Path が死んでいる → PreOutputFilter / PostDither Float Path の 2 挿入点が実質未使用

- `CMakeLists.txt:1292` で `CONVOPEQ_STANDALONE_ONLY=1` 定義
- `MainWindow.cpp:282` で `audioProcessorPlayer.setDoublePrecisionProcessing(true)`
- `AudioEngineProcessor.cpp:84-98` で Float route `processBlock` は stub（`buffer.clear()` のみ）

**影響**: 計画書の Float Path 挿入点（DSPCoreFloat.cpp:242, processOutput() L503-505）は **Standalone CLI テストでは実行されない**。`--cli-capture-mode=pre-filter` / `post-dither`(float path) は Double path の PostDither に統合するか、Float path 専用のテスト用ビルド（`CONVOPEQ_STANDALONE_ONLY=0` で plugin host 実行）を Phase 1 に追加するかの**設計決定が必要**。

### 発見-4【仕様変更未反映】OutputCaptureSink クラス名・ライフサイクル設計が work92 対応済みの shutdown と競合

work89/work92 で `~MainWindow()` のシャットダウン順序が変更済み（実測 src/MainWindow.cpp:1115-1160）:

```
step 1  removeChangeListener
step 2  setAdaptiveAutosaveCallback({})
step 3  setCliProcessingTelemetryEnabled(false)
step 4  setProcessor(nullptr)
step 5  stopTimer
step 6  saveSettings
step 7  removeAudioCallback
step 8  closeAudioDevice
step 9  audioEngineProcessor.reset()
step 10 UI reset
```

計画書 §1.7 は上記 10 ステップの「step 3→4 間」に OutputCaptureSink 停止を挟む 14 ステップ統合を提案。**現行の順序は計画書の前提と一致している**（10 ステップの実装確認済み）ため、**統合設計自体は妥当**。ただし:

- `outputCaptureSink_` メンバは実在しない（rg 0 件）→ **Phase 0 未実装のまま**。計画書「ステータス: Phase 0 実装開始可」は正しいが、**コード側の shutdown 契約が work92 対応で変化している**ため、`setCaptureSink(nullptr)` を step 3.5（`setCliProcessingTelemetryEnabled(false)` 直後）に挿入する位置を**現行コード上で再照合してから実装**すること。
- **重要**: `setProcessor(nullptr)`（step 4）で Audio Thread が停止するのは JUCE standalone の実装依存。**AudioCallback 停止は step 7 `removeAudioCallback` まで発生しない**可能性がある。OutputCaptureSink の BG Thread 停止を step 4 前に行うと capture() 呼出元（Audio Thread）が NULL deref するタイミング窓が残る。**計画書 §1.7 の step 4 で setCaptureSink(nullptr) を先に実行する設計は正しい**（確認済み）。

### 発見-5【パーサー未実装】計画書 §1.13.2 の「新規パーサー 6 種」は全未実装（計画通り）

- `parseCliOsType` / `parseCliOsFactor` / `parseCliHcMode` / `parseCliLcMode` / `parseCliSoftClip` / `parseCliSaturation`: **すべて実在しない**（rg 0 件）✅ 計画通り
- 既存パーサー名が計画書と不一致: `parseCliPhaseMode`（計画書では parseCliPhaseMode）✅ 実在するが、**正しい関数名は `parseCliPhaseMode`**（MainWindow.cpp:36）。計画書表記も OK。ただし計画書の実装ファイル記述「MainWindow.cpp:36-57」は正確 ✅
- `--cli-eq-*` 5 種（band/freq/gain/q/type）が計画書の 27 種 CLI リストから**漏れている**。実在（MainWindow.cpp:388-392, 983-1014）。計画書「findValue 24 種」→ 実際は **29 種**（eq 5 種 + log-file 1 種を含む）

### 発見-6【AudioBlock 互換性】AudioBlock 構造体は既存のままで OK

現行 AudioBlock（src/audioengine/AudioEngine.h:24-31）:
```cpp
struct AudioBlock {
    double L[256]; double R[256];
    int numSamples = 0; int sampleRateHz = 0; int bitDepth = 0;
    int adaptiveCoeffBankIndex = 0; std::uint64_t sessionId = 0;
};
```
- 計画書 §1.10 設計（`timestampUs` 追加で 4120→4128 byte）は妥当 ✅
- `alignof(AudioBlock)` は 8、`trivially_copyable` 条件満たす ✅
- **LockFreeRingBuffer は 4096 capacity 固定**（AudioEngine.h:2797）で、容量の増減は計画書 §1.3 の `kRingCapacity = 4096` と一致 ✅
- ただし計画書の static_assert 追加（§1.10）は、**既存 `pushWithWriter` が既に trivially_copyable を要求しているため追加必須ではない**。追加しても問題なし。

### 発見-7【CLI出力先】ConvoPeq_Standalone.exe という実行ファイル名は実在しない

- `juce_add_gui_app(ConvoPeq ...)` → 生成物は `build/ConvoPeq_artefacts/Release/ConvoPeq.exe`（実測）
- `build-icx/ConvoPeq_artefacts/Release/ConvoPeq.exe` も確認
- `cli_runner.py` の `CONVOPEQ_PATH = Path("build/ConvoPeq_Standalone.exe")` → **修正必須**（`build/ConvoPeq_artefacts/Release/ConvoPeq.exe`）

### 発見-8【exit 契約】--cli-exit-ms 最小値制限がテスト時間を制約

MainWindow.cpp:1059-1098 実測:
- `--cli-exit-ms` 未指定時: IR/rebuild 併用時は **最低 3000ms 強制**
- 指定時も `minExitMs` 超えを強制（短すぎる exit は調整される）
- **TC-38 Long-run（30 分）は `--cli-exit-ms=1800000` で動くが、タイマーは単発 `callAfterDelay` のため、`--cli-run` 起点から 30 分放置が必要**。CI 上で長時間ジョブを発生させる際は GitHub Actions timeout（6h 上限）との整合を確認要。

---

## 2. 計画書「Part 2: 調査確定事項」の妥当性検証

### 2.1 RecoveryHistory / Seqlock（Timer Thread 結合）→ ✅ 設計妥当・未実装確認

- `recordRecoveryAction()` / `copyRecoveryHistorySnapshot()`: **実在しない**（rg 0 件）✅ Phase 0 実装対象のまま
- `executeRecoveryAction()` は実在: `AudioEngine.Timer.cpp:1817`（計画書記載 1591 はズレている）
- RuntimeHealthMonitor → action callback → executeRecoveryAction の経路は実在確認済み ✅

### 2.2 JUCE WaitableEvent 仕様 → ✅ 確定済み内容は正確

- `project(JUCE VERSION 8.0.12)` 実測（JUCE/CMakeLists.txt:35）✅ 計画書記載どおり
- auto-reset デフォルト構築（manualReset=false）の記述 ✅ 正確

### 2.3 OutputCaptureSink 単体テスト → ✅ テストフレームワーク認識正確

- 独自 check() マクロパターン実在確認（例: src/tests/MT-NUPC-Measurement.cpp, RuntimeWorldAuthorityProjectionTests.cpp）✅
- CMake `add_test` 40 個の既存テストパターンと整合 ✅
- 10 テストケース一覧の妥当性: 全項目妥当。ただし**「DoubleStartPrevention で jassert メッセージ確認」は現行コードベースのテストでは珍しいパターン**（RuntimeWorldAuthorityProjectionTests.cpp:222 で 1 例あり）なので可能だが、**Debug build 限定テスト**になる点は計画書に明記すべき（Release では jassert が no-op）。

### 2.4 OutputFilter const getter → ✅ 設計妥当・メンバ名は実在

- BiquadCoeff 実在（src/OutputFilter.h:41）
- `hcCoeff[3][2]` / `lcCoeff[2]` / `hpfCoeff` / `lpCoeff[3][2]` 実在 ✅（OutputFilter.h:137-146）
- ただし実装注意: OutputFilter.h は prepare() が Message Thread、process() が Audio Thread の**非同期契約**。getter は Message Thread 専用とし、`--cli-dump-filter-coeffs` は Audio Thread 停止後（または prepare 直後）に呼ぶこと。**audioEngine 内の outputFilter メンバに AudioEngine 経由の const getter が必要**（現行では DSPCore 内に閉じており外部参照不可 → AudioEngine に新設する経路を Phase 0 実装設計に明記すべき）。

### 2.5 ライン番号 → ❌ 全滅（修正必須）

| 計画書アンカー | 現行 | 判定 |
|--------------|------|------|
| DSPCoreIO.cpp L242（PreOutputFilter float） | `DSPCoreFloat.cpp:250`（juce::dsp::AudioBlock<double> processBlock） | ❌ ファイル名・行番号ともズレ |
| DSPCoreDouble.cpp L391 | `DSPCoreDouble.cpp:348` | ❌ |
| processOutput() L411 | `DSPCoreIO.cpp:356` | ❌ |
| processOutputDouble() L651 | `DSPCoreDouble.cpp:577` | ❌ |
| processOutput() L503-505 PostDither | `DSPCoreIO.cpp:531-543` 付近 | ❌ |
| processOutputDouble() L803-805 | `DSPCoreDouble.cpp:739-755` | ❌ |
| outputFilter.process Float | `DSPCoreFloat.cpp:360` | ⚠️ ±数行 |
| outputFilter.process Double | `DSPCoreDouble.cpp:460` | ⚠️ |

すべて計画書の「実装着手時点で再度ライン番号を確認すること」の注意を上回るズレ。**v7.5 ではアンカーを「関数名+近傍キーワード」に置き換えるべき**（例: `processOutput()` 内 `pushAdaptiveCaptureBlocks` 呼出直後）。ライン番号ベースの管理はリファクタのたびに陳腐化する。

### 2.6 単回起動契約 → ✅ 設計妥当

- jassertfalse + 静かに無視の設計は既存 JUCE パターンと整合 ✅
- `RuntimeWorldAuthorityProjectionTests.cpp:222` の jassert メッセージ確認パターン実在 ✅

---

## 3. §1.13 Phase 1 テスト設計の妥当性（TC-01〜41 個別検証）

### 3.1 テストパラメータ 10 次元 CLI マッピング（§1.13.2）

| 次元 | 計画書 CLI値 | 実装 | 妥当性 |
|------|------------|------|--------|
| 処理順序 | `--cli-order conv/peq/convpeq/peqconv` | ✅ parseCliOrderMode 実在（MainWindow.cpp:60-89） | ✅ **トークン "conv"/"peq"/"convpeq"/"peqconv" は実装と一致** |
| 位相モード | `--cli-phase asis/mixed/minimum` | ✅ parseCliPhaseMode 実在 | ✅ 正確 |
| OSタイプ | `--cli-os-type iir/linear-phase` | ❌ 未実装（計画通り） | ⚠️ **実装名は `CustomInputOversampler::Preset::IIRLike / LinearPhase`**（CustomInputOversampler.h:17-20）。CLIトークン `iir` → IIRLike へのマッピングを計画書に明記 |
| OS倍率 | `--cli-os-factor 1/2/4/8` | ❌ 未実装 | ⚠️ **setOversamplingFactor(factor)** は AudioEngine.h:1580 実在。ただし実行中の OS 変更は rebuild を伴う（sampleRate × factor ≤ 768kHz 制限）。** OversamplingPolicy.h 実測: sr ≤ 96k → max 8 / ≤ 192k → max 4 / ≤ 384k → max 2**。48kHz テストでは全倍率可 ✅ |
| ノイズシェイパー | `--cli-noise-shaper psycho/fixed4/adaptive/fixed15` | ✅ parseCliNoiseShaper 実在 | ⚠️ **CLI トークンは実装上 `psychoacoustic` / `fixed4tap` / `adaptive` / `adaptive9` / `adaptive9thorder` / `fixed15` / `fixed15tap` を受け付ける**（MainWindow.cpp:91-120）。計画書の `psycho` も受け付ける（pszho エイリアス）✅ |
| ディザー深度 | `--cli-dither-bit-depth 16/24/32` | ✅ 実在 | ⚠️ **ditherBitDepth のデフォルトは 0 = Off**（AudioEngine.h:2452 `std::atomic<int> ditherBitDepth { 0 }`）。DeviceSettings.cpp:780 で maxBitDepth 自動設定は「DeviceSettings 画面を開いたとき」のみ。**CLI モードでは DeviceSettings 画面を経由しないため、CLI 起動のままでは dither は Off のまま**。計画書の既定値 `ditherBitDepth: 32` は**CLI 上で明示的に `--cli-dither-bit-depth 32` を渡すか、CLI パスで強制設定する必要**。TC-04/04A/34/40 はディザー必須のため、この点が抜けると全 FAIL（全ノイズが量子化ノイズなく dither off で測定される） |
| HCモード | `--cli-hc-mode sharp/natural/soft` | ❌ 未実装 | ✅ setConvHCFilterMode(HCMode) 実在（AudioEngine.h:1592）。HCMode enum {Sharp=0, Natural=1, Soft=2} 実在確認 |
| LCモード | `--cli-lc-mode natural/soft` | ❌ 未実装 | ✅ setConvLCFilterMode 実在。LCMode {Natural=0, Soft=1} 実在 |
| ソフトクリップ | `--cli-soft-clip on/off` | ❌ 未実装 | ✅ setSoftClipEnabled(bool) 実在（AudioEngine.h:1537） |
| サチュレーション | `--cli-saturation 0.0-1.0` | ❌ 未実装 | ✅ setSaturationAmount(float) 実在（AudioEngine.h:1540）。**注意: AudioEngine.h:2566 で `std::atomic<float> saturationAmount { 0.1f }`** — **デフォルト 0.1 であり 0.0 ではない**。CLI で明示 0.0 を渡さないとサチュレーションが微に動作する。計画書の既定値 `0.0` は**明示的に渡す設計にするか、CLI パスでリセット**が必要 |

### 3.2 テストパターン P1-P3 定義（§1.13.3）

- P2-PEQ-Only の `convBypass: true` → `--cli-bypass-burst-count=1 --cli-bypass-burst-value=1` の設計 ✅ 実装パス実在（MainWindow.cpp:921-946 相当）
- **ただし `--cli-bypass-burst-value` は「burst 値」であり、恒常的な bypass 維持ではない**。実測では `setConvolverBypassRequested` を直接叩く CLI が存在しない（実装待ち）。**P2 の恒常 bypass 実現には Phase 0 で新 CLI `--cli-conv-bypass` 追加が必須**（計画書の burst 方式ではタイマー経由で一時的なみ）

### 3.3 分析エンジン（analyzers.py）

| 関数 | 妥当性 | 指摘 |
|------|--------|------|
| `analyze_dirac` | ⚠️ | `theoretical_dirac_response` は `y[0]=1.0` のみ。**実IRの遅延・GroupDelay・-1dB headroom・DC blocker 特性（3Hz HPF×2段）を反映していない**。IRConvolver が linear-phase mixed mode を返すため、**TC-01 は「Dirac → 平坦性の確認」ではなく「全帯域で ±0.05 dB 内に収まっているか」のチェック設計が必要** |
| `analyze_sweep_transfer` | ✅ | Farina 2007 方式は妥当。ただし **input_wav が ConvoPeq に入力された際の -1dB headroom も理論側に織り込む**か、PostFilter キャプチャで input も計る設計にする必要 |
| `analyze_null_test` | ⚠️ | TC-31a「RMS ≤ -138 dBFS」は **dither Off + Float Path の場合のみ可能**。**Standalone Double Path では kOutputHeadroom=-1dB により null test の前提（input==output）が破綻**。Headroom bypass がないと成立しない |
| `analyze_alias` | ⚠️ | 理論イメージ位置のエネルギー測定は妥当だが、**OS倍率 1 のとき f_alias が SR/2 以上でエイリアス反転する**前提が抜けている。OS=1 テストでは 20-24kHz 信号はエイリアスせず NYQ で折り返すため「イメージ」ではなく「折返し」測定になる。TC-33 は **OS ≥ 2 でのみ意味を持つ**テストであることを明記すべき |
| `analyze_thd_sweep` | ✅ | Blackman-Harris + bin-sum は妥当。**ただし -6dBFS 入力では Harmonic 2-10 次が DSP 非線形性（SoftClip off, Saturation 0）でないとほぼ測定不能**（Quantization noise floor が支配的）。THD ≤ -100 dBFS は **32bit float + 32bit dither off + 出力 -1dB headroom** 前提でのみ成立する理論値。Release 閾値の再検証必須 |
| `analyze_noise_psd` | ✅ | A-weighting 計算式正確（IEC 61672 準拠） |
| `analyze_stereo_crosstalk` | ⚠️ | **TC-36 の入力は「L=1kHz, R=無音」だが、ConvolverProcessor は Mono→Stereo 拡張を持たない**（processInput で expandMono はある）。IR が stereo なら L→R は基本ゼロになるが、**stereo IR の cross-talk は IR の L/R 分離性能に依存するため、TC-36 の閾値 -140dB は IR に強く依存**。Dirac stereo IR（L≠R 成分なし）使用時のみ成立 |

### 3.4 ゴールデン比較（golden_calculator.py）

- `compare_golden_wav` の pass 判定 `freq_response_rms_error_db < 0.05` は**初回ゴールデン生成時と 2 回目以降で閾値評価が循環する**問題あり（初回が pass だとゴールデンが更新され、以降の 0.05dB 内偏差が保証される）。**初回ゴールデンは手動レビューで承認し、ゴールデン更新は明示フラグのみで行う設計**を明記すべき。
- `rms_deviation_db` に -1dB headroom の影響があるため、**ゴールデン取得時と測定時で CLI パラメータ完全一致**を cli_runner で強制する仕組みが必要。

### 3.5 テスト信号生成（generators.py）

- `generate_log_sweep`: Farina 指数スイープ式は妥当 ✅（`phase = 2π f0 T / ln(f1/f0) × (exp(t ln(f1/f0)/T) - 1)` は標準式と一致）
- `generate_dirac_input`: sample[0] = amplitude、3 秒無音テール ✅
- `generate_multitone`: **AES17 では multitone の位相は random phase でなく determinstic phase (0/90°) が推奨**。RNG42 シードは再現性あり ✅
- `generate_synthetic_ir`: **lpf_1k / hpf_20 の 129 taps は ConvolverProcessor の最小 IR 長要件（実測: k16/k2/k4/k8/k32 Dirac が sampledata/synthetic に実在）を満たす** ✅
- `windowed_sinc` Hamming 窓設計は妥当 ✅（ただし -42dB stopband しかない。エイリアス測定用 lpf IR には **Kaiser 窓 ≥ -90dB** 推奨。TC-07 で IR 自体の -42dB ストップバンドが SUT の OS フィルタ性能を隠すリスク）

### 3.6 CLI実行エンジン（cli_runner.py）

- `subprocess.run(cmd, capture_output=True, text=True, timeout=120)` ⚠️ **TC-38（30 分）は timeout=120 で必ず TIMEOUT FAIL**。TC 別 timeout 設定が必要。
- `--cli-exit-ms=5000` は計画書の CLI ビルダーで固定。**TC-38 は exit-ms を可変にする設計変更が必要**。

### 3.7 CI 設計（§1.13.11）

- GitHub Actions windows-latest で **AudioDevice を開けない問題**が残る: `--cli-device-type` で Dummy Audio Device を指定するか、**JUCE 独自の `--audiodevice-dummy` 相当の CLI 追加が必要**。**ConvoPeq CLI には現状 Dummy device 直接指定パスが存在せず**、CI 上で WASAPI device なしでの動作は未検証。**Phase 0 で `--cli-device-type Dummy` 相当のフォールバック（JUCE DummyAudioDevice をリストに追加 or 独自 null sink 実装）を必須化すべき**。この点は計画書に記載がない**重大な欠落**。
- CI 上での **ConvoPeq.exe は GUI アプリであり、windowless 実行の検証**が必要。`--cli-run` では MainWindow が生成されるため CI でヘッドレス動作できるかは実測要。Windows CI では `windows-latest` で GUI session あり（session 0 isolation は GitHub Actions runner では回避される）ため、**実証済みではないが技術的には動く**。Phase 1 冒頭で 1 本の smoke test で検証すること。

### 3.8 工数内訳

Phase 1: 58-73 人日 → **妥当**（generators/analyzers/cli_runner/golden で 23 人日、TC 34+11 で 30 人日は現実的）
Phase 2: 40-50 人日 → **妥当**（安定化 20 人日は TC-25/27/30/38 の長期テスト性质から妥当）
Phase 3: 20-25 人日 → **妥当**
**予備 30-70 人日 → 発見-1〜3 の修正を含めると上振れ必至。+15 人日推奨**

---

## 4. 不足テスト項目（追加実装計画）

### NEW-1【P0 優先】Audio Device Abstraction（Dummy Output Sink）
- **問題**: CI/無音環境で Real AudioDevice に依存したテストは再現不能
- **実装**: `--cli-dummy-output` オプション追加。JUCE DummyAudioDevice がない場合、`AudioIODevice` の null sink 実装（内部 Timer で getNextAudioBlock を叩く）を Phase 0 に実装
- **テスト**: `--cli-dummy-output` + `--cli-sample-rate-hz 48000 --cli-buffer-samples 512` で RT pipeline を完全駆動
- **工数**: 8 人日
- **妥当性**: 全 41 TC の CI 実行可能性を決定づける基盤

### NEW-2【P0 優先】kOutputHeadroom / PeakLimiter / SoftClip / Saturation の Bypass CLI
- **問題**: -1dB headroom が TC-23/31/37 を失格させる
- **実装**: `--cli-test-raw-output` を Phase 0 に追加。**テストモード時のみ** kOutputHeadroom = 1.0、PeakLimiter bypass、SoftClip bypass、Saturation 0 を強制
- **テスト**: TC-23 で RMS 偏差 0.0 dB ± 0.01 確認、TC-31a/b Null test が理論どおり成立
- **工数**: 3 人日
- **妥当性**: 音響計測の前提条件を確立する必須修正

### NEW-3【P1】OutputCaptureSink の複数 CapturePoint 同時キャプチャ
- **問題**: 現行設計は 1 セッション 1 CapturePoint。Pre-filter と Post-dither を同時比較する OS エイリアス分析には 2 セッションが必要で IR 再ロード 2 回分のコストがかかる
- **実装**: `OutputCaptureSink::capture()` を capturePoint ビットマスク対応にし、BG Thread 側を WAV 2 並列出力（`output_pre.wav` / `output_post.wav`）
- **テスト**: TC-33（OS エイリアス）を 1 セッションで Pre-filter vs Post-dither 差分分析
- **工数**: 5 人日

### NEW-4【P1】OS エイリアスの厳密測定（TC-33 改良）
- **問題**: analyze_alias は 3 周波数の単点ピークのみ。**OS フィルタのストップバンド特性は実測 FIR (1023/255/63 taps, Kaiser, -140/-110/-90 dB stage別) の全周波数特性**として測るべき
- **実装**: 対数スイープ→Farina 逆フィルタ→**転送関数の全帯域で image 領域エネルギーを積分**する改良 `analyze_alias_fullband`。CustomInputOversampler.cpp:96-105 の stage 減衰値（IIRLike: 160/140/120 dB, LinearPhase: 140/110/90 dB）を理論閾値として比較
- **テスト**: OS=2 で 24kHz 入力時 48kHz イメージ ≤ -130dB、OS=4 で ≤ -140dB 等
- **工数**: 4 人日

### NEW-5【P1】処理順序 Conv-Then-EQ vs EQ-Then-Conv の等価性テスト（新 TC-42）
- **問題**: 計画書に「順序変えても線形段の等価性が保たれるか」の直接テストがない。**LinearProcess (IR) + Linear EQ なら可換**のはずだが、**SoftClip/Saturation 非線形段の位置は固定**のため可換性が破れる瞬間のバグ検出に有効
- **実装**: TC-42: 同一 IR + 同一 EQ 設定で `--cli-order convpeq` vs `peqconv` の出力差分 ≤ -120dBFS（非線形 off 時）
- **分析**: analyze_null_test 流用
- **工数**: 2 人日

### NEW-6【P1】PeakLimiter の Gain Reduction 正当性テスト（新 TC-43）
- **問題**: kPLThreshold = -1.5 dBFS の PeakLimiter は全 TC で常時動作するが、**その動作自体を検証するテストがない**。プログラム素材（-0.1dBFS 正弦波等）で gain reduction 量の理論一致を確認
- **実装**: TC-43: 0dBFS 入力時 Gain Reduction ≈ 0.5 dB ± 0.1、Attack/Release 時定数の理論一致
- **工数**: 3 人日

### NEW-7【P1】DC Blocker 周波数特性テスト（新 TC-44）
- **問題**: DC blocker は 3Hz×2段（ベースレート）+ 1Hz×2段（OS domain）の 2 段構成（AudioEngine.h:647-649 実測）。**3Hz HPF が 20Hz 帯域に与える位相/振幅影響を定量化するテストがない**
- **実装**: TC-44: 5Hz-20Hz 正弦波で通過特性測定。**3Hz Butterworth 2 次 HPF の理論応答との差 ≤ 0.01 dB**（20Hz 以下帯域）。**TC-01/32 の 20Hz 以下カットオフ特性に影響**するため、baseline での逸脱原因分離に必要
- **工数**: 3 人日

### NEW-8【P2】Multi-Instance 同時実行テスト（新 TC-45）
- **問題**: CLI 自動化は 1 プロセス 1 セッション前提。**同一 machine 上で複数 ConvoPeq インスタンスが同時起動した場合の WASAPI 独占/デバイス競合**の挙動を検証するテストがない（ASIO 独占での失敗パターン）
- **実装**: TC-45: 2 プロセス同時起動 → 1 つが失敗するか、両方成功するか、**失敗側が clean shutdown するか**の 3 契約チェック
- **工数**: 4 人日
- **妥当性**: CI マトリクス実行時の実用性に直結

### NEW-9【P2】`--cli-log-file` 経由の自動化結果機械可読出力（新 TC-46）
- **問題**: 現行 CLI ログは人間可読で、analyzers が pass/fail を判定するには WAV 出力のみに依存。**capture 途中の dropped blocks 数、XRUN 数、envelope** を機械可読に出力する機構がない
- **実装**: Phase 0 の OutputCaptureSink に `droppedBlocks_` / `seqlockRetryFailed_` カウンタを設け、**exit 前 JSON サマリ出力**（`--cli-summary-json`）を新設
- **テスト**: 全 TC で droppedBlocks == 0 を前提条件に含める（閾値オーバーの誤検知防止）
- **工数**: 3 人日

### NEW-10【P2】ディザシード再現性テスト（新 TC-47）
- **問題**: PsychoacousticDither は `std::chrono::high_resolution_clock` ベースの seed で、**同一入力・同一設定でも出力 WAV が run ごとに変わり、ゴールデン比較が失敗する**
- **実装**: `--cli-dither-seed <n>` を Phase 0 に追加（PsychoacousticDither(seed) をエクスポート）。TC-47: 同一 seed で 2 回実行 → **byte-level 一致確認**
- **工数**: 3 人日

---

## 5. 計画書 v7.5 への反映推奨事項（要約）

| # | 修正内容 | 種別 |
|---|---------|------|
| 1 | CLI パーサー `=` 結合非対応を明記し、cli_runner.py をスペース区切りに統一（または `=` 対応を Phase 0 スコープに追加） | 重大 |
| 2 | kOutputHeadroom -1dB / PeakLimiter -1.5dBFS / Saturation 既定 0.1 を全 TC 理論値に反映。または `--cli-test-raw-output`（headroom/limiter bypass）を Phase 0 に追加 | 重大 |
| 3 | Float Path 挿入点が Standalone で不活性であることを §1.2 CapturePoint 表に明記。`--cli-capture-mode=pre-filter` は plugin build 限定とするか、Double path 統合に設計変更 | 重大 |
| 4 | アンカー表記をライン番号から「関数名+キーワード」に変更（work89/92 で全滅したため） | 中 |
| 5 | CLI 29 種（eq 5 種 + log-file 1 種の追加）に更新、saturation 既定値 0.1 の明記 | 中 |
| 6 | ditherBitDepth 既定 0（Off）を明記し、テストパターンで明示的に 32bit を渡す設計に | 中 |
| 7 | exe パスを `build/ConvoPeq_artefacts/Release/ConvoPeq.exe` に修正 | 小 |
| 8 | 新 TC-42〜47（本報告書 §4）を TC 表に追加 | 追加 |
| 9 | `--cli-dummy-output`（CI 用 Null AudioDevice）を Phase 0 スコープに追加（NEW-1） | 追加 |
| 10 | P2-PEQ-Only の恒常 bypass に `--cli-conv-bypass` 新設（burst 方式の限界明記） | 追加 |

---

## 6. ツール横断検証結果（必須ツール稼働確認）

| ツール | バージョン | 稼働 | 備考 |
|--------|-----------|------|------|
| ccc (cocoindex-code) | - | ✅ | 224,617 chunks / 2,467 files indexed。semantic search 動作確認 |
| graphify | - | ✅ | graph.json 44,244 nodes。query 動作確認（ただし doc/work53 の旧 graph 情報が返るため、**再 build 推奨**） |
| cppcheck | 2.21.0 | ✅ | OutputFilter.h で syntaxError 1 件（AtomicAccess.h の template 構文 — 誤検知。`--library` と define で対処可能） |
| clang-tidy | LLVM 23.1.0 | ✅ | compile_commands.json 生成済み（build/ 下）で利用可 |
| tgrep | 1.0.4 | ✅ | `C:\Windows\System32\tgrep.exe` 存在確認（WSL から直接呼ぶ場合パス解決注意: `/mnt/c/Windows/system32/tgrep.exe`） |
| ast-grep | - | ⚠️ | WSL で `ast-grep` コマンド見つからず（`sg` エイリアスか PATH 未通）。Windows 版は別途確認要 |
| fd | - | ❌ | WSL で `fd` コマンド見つからず（fdfind のエイリアス設定要） |
| ag | 2.2.0 | ✅ | 実測 2.2.0 動作 |
| fzf | - | ⚠️ | filter モードは動作するが stdout が空になる環境あり |
| semble / serena / AiDex | - | ✅ | MCP 経由で動作確認（serena find_symbol は timeout したが MCP ハンドシェイクは成立） |

---

## 7. 結論

計画書 v7.4 は **設計思想（OutputCaptureSink 分離・SPSC+BG Thread・CapturePoint 3 点）は音響計測自動化として正しい方向**であり、Phase 0 の実装可能性も高い。ただし:

1. **発見-1（CLI `=` 結合非対応）**、**発見-2（-1dB headroom 未考慮）**、**発見-3（Float Path 不活性）** の 3 点は、実装開始前に**設計修正が必須**。
2. **TC-01〜41 の閾値表**は、上記 3 点の修正後に**全面的な数値再検証**が必要（特に TC-23/31/37）。
3. **不足テスト 10 項目**（NEW-1〜10）を追加することで、CI 実行可能性・数値精度・回帰検出力が大きく向上する。
4. v7.5 への反映（§5 の 10 項目）を完了した上で、Phase 0 実装開始を**条件付き GO** とする。

---

## Appendix: 検証に使用した実測アンカー（v7.5 で利用可）

```
// === 現行実測アンカー（2026-09-11 HEAD）===
src/audioengine/AudioEngine.Processing.DSPCoreIO.cpp:356    processOutput() 先頭
src/audioengine/AudioEngine.Processing.DSPCoreIO.cpp:362    kOutputHeadroom = 0.8912509381337456
src/audioengine/AudioEngine.Processing.DSPCoreIO.cpp:426    pushAdaptiveCaptureBlocks 呼出（PostOutputFilter 相当位置）
src/audioengine/AudioEngine.Processing.DSPCoreIO.cpp:527-529  PeakLimiter (kPLThreshold=-1.5dBFS)
src/audioengine/AudioEngine.Processing.DSPCoreIO.cpp:540    applyFixedLatencyDelay 呼出（PostDither 挿入点）
src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp:348 processUp
src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp:460 outputFilter.process
src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp:577 processOutputDouble() 先頭
src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp:710 PeakLimiter
src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp:739 applyFixedLatencyDelay
src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp:210  DSPCore::process()
src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp:250  processUp（PreOutputFilter 挿入点候補・float path・standalone では不活性）
src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp:405  processDown
src/audioengine/AudioEngine.h:24-31                          AudioBlock 構造体
src/audioengine/AudioEngine.h:781-793                        fixedLatencySamples
src/audioengine/AudioEngine.h:2452                           ditherBitDepth { 0 }  // Off 既定
src/audioengine/AudioEngine.h:2566                           saturationAmount { 0.1f }  // 既定非ゼロ!
src/audioengine/AudioEngine.h:2797                           audioCaptureQueue (LockFreeRingBuffer<AudioBlock,4096>)
src/audioengine/AudioEngine.h:4027                           buildAudioThreadProcessingState()
src/audioengine/AudioEngine.h:4060                           adaptiveCaptureQueue = enabled ? &audioCaptureQueue : nullptr
src/MainWindow.cpp:331                                       runCommandLineAutomation()
src/MainWindow.cpp:343-356                                   findValue（スペース区切りトークン）
src/MainWindow.cpp:1059-1098                                 --cli-exit-ms 処理（最小 1000/3000ms 強制）
src/MainWindow.cpp:1115-1160                                 ~MainWindow() 10 ステップ
src/MainWindow.cpp:282                                       setDoublePrecisionProcessing(true)
src/OutputFilter.h:41                                        BiquadCoeff
src/OutputFilter.h:75-89                                     HCMode {Sharp,Natural,Soft} / LCMode {Natural,Soft}
src/OutputFilter.h:137-146                                   hcCoeff[3][2] / lcCoeff[2] / hpfCoeff / lpCoeff[3][2]
src/LockFreeRingBuffer.h:44                                  pushWithWriter
src/CustomInputOversampler.cpp:84-93                         tapsForStage {511,127,31} / {1023,255,63}
src/CustomInputOversampler.cpp:96-105                        attenuationForStage IIRLike{160,140,120} / LinearPhase{140,110,90}
src/audioengine/OversamplingPolicy.h                         kMaxInternalRate=768k, sr≤96k→max8, ≤192k→max4, ≤384k→max2
CMakeLists.txt:1050                                          juce_add_gui_app(ConvoPeq)
CMakeLists.txt:1292                                          CONVOPEQ_STANDALONE_ONLY=1
```
