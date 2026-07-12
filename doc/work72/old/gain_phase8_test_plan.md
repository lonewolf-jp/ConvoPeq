# Phase 8 テスト計画書 — 自動ゲインステージング改修

> 作成日: 2026-07-11
> 対象: `gain_revised.md` v2.6（詳細文献調査反映済み - Ross Bencina 2011, Robert Bristow-Johnson Cookbook, Harris 1978, Julius O. Smith III CCRMA, Vadim Zavalishin）
> テスト形式: 既存テスト（`ShadowCompareContractTests.cpp`等）の自己完結型`main()`ドライバ形式に準拠
> 改訂: v1.1 → v1.2 — 第5次検証でUT-06自己矛盾修正、IT-04 Conv onlyクランプ考慮追加、IT-06/MT-05拡充、文献ベース RT-01〜RT-04（Bencina 2011 realtime-safe 原則）追加、Cookbook RBJ 公式との数値検証明示

---

## 0. テスト構成概要

| レイヤ | テストファイル | テスト数 | 想定工数 |
|--------|---------------|---------|---------|
| 単体テスト | `src/tests/DspNumericPolicyTests.cpp`（新規） | 8 | 1.5人日 |
| 統合テスト | `src/tests/ShadowCompareContractTests.cpp`（拡充） | 7 | 2.0人日 |
| 統合テスト | `src/tests/GainStagingContractTests.cpp`（新規） | 6 | 1.5人日 |
| 手動テスト | 実機チェックリスト | 10 | 1.0人日 |
| **合計** | | **31** | **5.0人日（Phase 8見積もり内）** |

---

## 1. 単体テスト: `DspNumericPolicyTests.cpp`（新規）

### テスト形式

既存テストに準拠:
- 自己完結型の`main()`ドライバ
- 各テスト関数は`[[nodiscard]] bool testName()`形式
- 失敗時は`throw std::runtime_error("test name failed")`
- CMakeLists.txtに`add_executable` + `add_test`を追加

### テスト項目

#### UT-01: 複素応答計算の精度検証（Peak +12dB, Q=2.0）

```
目的: getComplexResponse()が既知のBiquad係数に対して理論値と一致する
入力: Peak EQ, gain=+12dB, Q=2.0, fc=1000Hz, fs=48000Hz
検証:
  - 周波数1000Hzでの|H(z)|が 20*log10(|H|) ≈ +12.0dB になること（誤差±0.1dB）
  - 周波数100Hzでの|H(z)|が ≈ 0dB になること（誤差±0.5dB）
  - 位相応答が連続であること（不連続ジャンプがない）
期待結果: 全ての周波数点で理論値と±0.1dB以内で一致
```

#### UT-02: AVX2版とスカラー版の一致確認

```
目的: evaluateComplexResponseAVX2()とgetComplexResponse()の結果が一致する
入力: ランダムなBiquad係数10セット × 周波数点300点
検証: 各周波数点での相対誤差 |H_avx2 - H_scalar| / |H_scalar| < 1e-6
期待結果: 全3000点で相対誤差 < 1e-6
```

#### UT-03: z=exp(+jω)の符号確認

```
目的: z = exp(+jω)が使用されていることを確認（exp(-jω)でない）
入力: 1次LPF係数（位相回転が明確なケース）
検証:
  - 既存のEQProcessor.Coefficients.cpp:327のgetMagnitudeSquared()と
    新規getComplexResponse()のマグニチュードが一致すること
  - 位相角の符号が+ω方向であること
期待結果: マグニチュード一致（±1e-10）、位相符号が+
```

#### UT-04: Mid/Sideデコードの数学的検証

```
目的: M/Sバンドのカスケード積算が正しくデコードされる
入力: Midバンド(Peak +6dB, Q=1.0) + Sideバンド(Peak -3dB, Q=1.0)
検証:
  - L = M + S の応答が、Mid単体とSide単体の和と一致
  - R = M - S の応答が、Mid単体とSide単体の差と一致
  - 周波数点300点全てで一致（相対誤差 < 1e-10）
期待結果: L/Rデコードが数学的に正確
```

#### UT-05: Tukey窓スペクトルリーク抑制確認

```
目的: Tukey窓（α=0.1）適用後のスペクトルリークが-40dB以下
入力: 単位インパルス（先頭1サンプル=1.0、残り0）× 65536点
検証:
  - FFT後のメインローブピーク位置から10bin離れた位置での振幅が
    ピークに対して-40dB以下であること
  - Tukey窓未適用の場合と比較して、サイドローブが抑制されていること
期待結果: 10bin離散位置で全て-40dB以下
```

#### UT-06: computeMaxGainDb() — 単一Peakバンド

```
目的: 単一Peakバンドの最大ゲイン計算が正確
入力: Peak +9dB, Q=1.0, fc=1000Hz, fs=48000Hz
検証:
  - Q=1.0 > 0.707のためQ Surge Marginが加算される
  - margin = 9 × 0.15 × (1.0/0.707) ≈ 1.91 dB
  - computeMaxGainDb()の戻り値が ≈ 9.0 + 1.91 ≈ 10.9dBになること
期待結果: eqMax ≈ 10.9dB
```

#### UT-07: computeMaxGainDb() — Q Surge Marginクリップ確認

```
目的: Q Surge Marginが6.0dBでクリップされる
入力: Peak +12dB, Q=10.0, fc=1000Hz（margin = 12*0.15*(10/0.707) = 25.46dB → clip 6.0dB）
検証:
  - computeMaxGainDb()の戻り値が 12.0 + 6.0 = 18.0dBになること
  - クリップが正しく機能していること
期待結果: eqMax = 18.0dB
```

#### UT-08: computeMaxGainDb() — 全バンドバイパス時

```
目的: 全バンドバイパス時にeqMax=0.0dB
入力: 全バンドbypass=true
検証: computeMaxGainDb()の戻り値が 0.0dB
期待結果: eqMax = 0.0dB
```

---

## 2. 統合テスト: `ShadowCompareContractTests.cpp`（拡充）

### テスト項目（既存テストファイルに追加）

#### IT-01: Q Surge Margin Bound Test

```
目的: 高QピーキングフィルタでQ Surge Marginが正しく加算される
入力: EQ設定 Peak +12dB, Q=4.0, fc=1000Hz + IR（フラット応答）
検証:
  - recomputeAutoGainStaging()後のinputHeadroomDbが
    eqMax = 12 + min(12*0.15*(4/0.707), 6.0) = 12 + 6.0 = 18.0dB
    に対して -max(0, 18.0 - 3.0) = -15.0dB になること
  - クランプ範囲[-12, 0]を考慮し、input = -12.0dBになること
期待結果: input = -12.0dB（クランプ）
```

#### IT-02: Spectral Leakage Isolation Test

```
目的: IR周波数応答ピーク解析がTukey窓で正しく動作する
入力: インパルスIR（先頭に鋭いピークを持つIRデータ）
検証:
  - estimateMaxFrequencyResponseGain()の戻り値が、
    Tukey窓未適用の場合より小さいこと（リーク抑制効果）
  - residualRiskDbが正の値になること
期待結果: freqRespGain < 未適用値、residualRiskDb > 0
```

#### IT-03: RCU Transaction Contract — ゲイン更新のpublicationSemanticHash反映

```
目的: ゲイン値更新がpublicationSemanticHashに反映される
入力: inputHeadroomGain = 1.0 → 0.5（-6dB）に変更
検証:
  - RuntimeBuilder.cpp:410-412のbit_castハッシュにinputHeadroomGainが含まれる
  - ハッシュ値が変更前に≠変更後になること
  - RCU worldのautomation.inputHeadroomGainが0.5になること
期待結果: publicationSemanticHashが変化し、新worldに正しいゲイン値が反映
```

#### IT-04: Mode Switch Gain Consistency Test

```
目的: 4モード間切り替え時にゲイン値がモードに対応した正しい値に更新される
入力: eqMax=9.0dB, irResidual=6.0dB

検証パターン:
  1. PEQ only: input = -max(0, 9-3) = -6.0, makeup = +6.0
  2. Conv only: input = -max(0, 6-1.5) = -4.5 → クランプ -6.0, makeup = +6.0
     ※ Conv onlyモードはeqBypassed=trueによりconvIsFirst=true、上限-6dBでクランプ
  3. Conv→PEQ: input = -max(0,6-1.5)-max(0,9-2) = -11.5, trim=0, makeup=+11.5
  4. PEQ→Conv: input = -max(0,9-3) = -6.0, trim=-max(0,6-2)=-4.0, makeup=+10.0

検証:
  - 各モード切り替え後、atomicsから読み取ったゲイン値が上記理論値と一致
  - input + trim + makeup = 0（ネット0dB）が全モードで成立
期待結果: 4モード全てで理論値と一致
```

#### IT-05: Mode Switch Crossfade Safety Test

```
目的: モード切り替え中のクロスフェード期間中にSimplePeakLimiterが過剰動作しない
入力: 音楽信号（-10dBFS RMS、ピーク-3dBFS）を再生中に4モード循環切り替え
検証:
  - クロスフェード期間中の出力ピークが+3dBFS以下であること
  - SimplePeakLimiterのreduction量が2dB以下であること
  - クリップインジケーターが点灯しないこと
期待結果: リミッター過剰動作なし
```

#### IT-06: Bypass比較テスト（LoudnessMeter）

```
目的: 処理前後でラウドネスが大きく変わらない（意図的ブースト/カット除く）
入力: フラットEQ（全バンド0dB）+ フラットIR
検証:
  - 入力信号: ピンクノイズ -20 dBFS RMS（20 秒間）を再生
  - `LoudnessMeter` で測定した入力/出力の Integrated LUFS 差が ±1.0 LU 以内
  - 内部信号: input+makeup = 0（ネット0dB）のため、ラウドネスが保持される
  - Short-term LUFS の時間変動も ±1.5 LU 以内に収まることを確認
期待結果: Integrated LUFS 差 ±1.0 LU 以内、Short-term LUFS 差 ±1.5 LU 以内
注: K-weighting フィルタ（ITU-R BS.1770）は LoudnessMeter 側で適用済み（既存実装、`AudioEngine.LoudnessMeter.cpp`）。本テストは処理チェーンのゲイン整合性検証であり、LoudnessMeter 単体の精度検証ではない。
```

#### IT-07: Auto ON/OFFトグル時の一貫性

```
目的: Auto ON→OFF→ONでゲイン値が正しく復元/再計算される
入力: Auto ON状態でゲイン計算済み → OFF → 手動でinput=-3dB → ON
検証:
  - OFF時に手動値が保持される
  - ON時にrecomputeAutoGainStaging()が再計算し、Auto値に上書き
  - UI表示が正しく更新される
期待結果: トグル後のゲイン値がAuto計算値と一致
```

---

## 3. 統合テスト: `GainStagingContractTests.cpp`（新規）

### テスト形式

`CrossfadeExecutorLocalContractTests.cpp`と同様のソースコード静的解析アプローチ（ファイル読み取り+パターンマッチ）と、構造体レベルの契約テストを組み合わせる。

### テスト項目

#### GC-01: Conv→PEQモードでtrimが適用されない契約

```
目的: ConvolverThenEQパスでconvolverInputTrimGainが適用されないことを確認
方法: ソースコード静的解析
検証:
  - DSPCoreDouble.cppのConvolverThenEQ分岐（line 429-457）に
    "convolverInputTrimGain"が出現しないこと
  - EQThenConvolver分岐（line 458-494）に"convolverInputTrimGain"が出現すること
期待結果: Conv→PEQパスにtrim適用なし
```

#### GC-02: trim=0時のガードスキップ契約

```
目的: convolverInputTrimGain == 1.0時にスケーリング処理がスキップされる
方法: ソースコード静的解析
検証:
  - DSPCoreDouble.cpp:483に `if (state.convolverInputTrimGain != 1.0)` が存在
  - スケーリング処理がガード内にあること
期待結果: ガード条件が存在し、trim=0dB時にスキップされる
```

#### GC-03: setProcessingOrder呼び出し順序契約

```
目的: setProcessingOrderの呼び出し順序が修正後の仕様通り
方法: ソースコード静的解析
検証:
  - applyDefaultsForCurrentMode()の後にrecomputeAutoGainStaging()が呼ばれる
  - submitRebuildIntentの明示的呼び出しが削除されている
  - sendChangeMessage()が末尾に追加されている
期待結果: 修正後の順序通り
```

#### GC-04: PreparedIRStateムーブ演算子にresidualRiskDbが含まれる

```
目的: PreparedIRStateのムーブコンストラクタ/代入演算子にresidualRiskDbのコピーが含まれる
方法: ソースコード静的解析（PreparedIRState.h読み取り）
検証:
  - ムーブコンストラクタにresidualRiskDbの代入がある
  - ムーブ代入演算子にresidualRiskDbの代入がある
期待結果: 両方のムーブ演算子にresidualRiskDbが含まれる
```

#### GC-05: ScaleFactorResultにresidualRiskDbが含まれる

```
目的: IRConverter.hのScaleFactorResult構造体にresidualRiskDbフィールドが追加される
方法: ソースコード静的解析
検証:
  - struct ScaleFactorResultに `double residualRiskDb = 0.0;` が存在
期待結果: フィールド追加済み
```

#### GC-06: recomputeAutoGainStagingのm_isRestoringState早期リターン

```
目的: プリセットロード中にrecomputeAutoGainStagingが早期リターンする
方法: ソースコード静的解析
検証:
  - recomputeAutoGainStaging()の冒頭に `if (m_isRestoringState) return;` が存在
  - autoGainStagingEnabled == false時の早期リターンも存在
期待結果: 両方のガードが存在
```

---

## 4. 手動テスト（実機チェックリスト）

### テスト環境

- DAW: Reaper 7.x または Ableton Live 12
- サンプルレート: 48kHz
- バッファサイズ: 256サンプル
- テスト信号: ピンクノイズ(-12dBFS RMS) + 音楽素材(ボーカル/アコースティック)

### テスト項目

#### MT-01: 各モードでのリミッター発動頻度確認

```
モード: 4モード各々
EQ設定: Peak +9dB, Q=2.0, fc=500Hz（意図的な大ブースト）
IR設定: Hall IR（residualRisk ≈ 6dB）
信号: ピンクノイズ -12dBFS RMS
確認:
  - SimplePeakLimiterのreduction表示が-3dB以下を維持
  - 出力ピークが-1dBFS以下に制御される
  - クリップインジケーターが点灯しない
合格基準: 全モードでリミッターreduction ≤ -3dB、クリップなし
```

#### MT-02: 768kHzアップサンプリング時のTrue Peak確認

```
モード: Conv→PEQ（デフォルト）
EQ設定: Peak +6dB, Q=1.0, fc=2kHz
信号: ピンクノイズ -10dBFS RMS
確認:
  - TruePeakDetectorの測定値が+1dBFSを超えない
  - 768kHzアップサンプリングが正常動作
  - 出力ディザリングが正常
合格基準: True Peak ≤ +1dBFS
```

#### MT-03: プリセットロード/リストア後のAuto ON再計算

```
手順:
  1. Auto ON状態でEQ/IRを設定しゲイン計算
  2. プリセットAを保存
  3. EQ/IRを変更（別設定）
  4. プリセットAをロード
  5. ゲイン値がプリセットAの設定に基づいて再計算されることを確認
確認:
  - m_isRestoringState中にrecomputeAutoGainStagingが実行されない
  - リストア完了後に1回だけ再計算される
  - ゲイン値が手動計算値と一致
合格基準: プリセットロード後に正しいゲイン値が設定
```

#### MT-04: 4モード循環切り替えテスト

```
手順:
  1. 音楽再生中に Conv→PEQ → PEQ→Conv → PEQ only → Conv only → Conv→PEQ の順に切り替え
  2. 各切り替え間隔: 3秒
確認:
  - ノイズ（ポップ/クリック）が発生しない
  - クリップが発生しない
  - 音切れが発生しない
  - ゲイン値が各モードに対応した値に更新される
合格基準: 全切り替えでノイズ/クリップ/音切れなし
```

#### MT-05: 純粋オーダー切替（IR同一）時の即時切替テスト

```
目的: 純粋モード切替（IR同一）時の即時切替となる場合の出力挙動を検証する
手順:
  1. Conv→PEQ で IR ロード済みの状態で音楽再生（テスト信号: ボーカル素材 -10 dBFS RMS）
  2. PEQ→Conv に切り替え（IR は変更しない）
  3. クロスフェードがトリガーされないことを確認 → `crossfadeRuntime_.getGain().isSmoothing()` が false
  4. `simplePeakLimiter` の gain reduction のログを記録
  5. WaveSpectrogram/Waveform で切り替え時点の振幅を記録（A/B 比較用）
確認:
  - `FADE_IN_SAMPLES = 2048`（=42ms @ 48kHz）によるフェードイン（0 → 1.0 リニアランプ）が発生する
    - 開始時点: 切り替え後最初のサンプル
    - 終了時点: 切り替え後 2048 サンプル後
  - フェードイン中の振幅推移
  - 42ms 以内に音量がほぼ回復すること
  - ノイズ/クリップが発生しないこと
合格基準:
  - 即時切替で「気にならない」レベルの音切れ（主観評価）
  - 42ms フェードインの有無、カーブ形状、リカバリ時間を記録
  - クリップなし（limiter の gain reduction が滑らか）
注意:
  - このテストで42msフェードインによる音切れが許容できない場合（主観的に目立つ）、
    `CrossfadeAuthority::evaluate()` に `processingOrder` 変化検出を追加し、
    クロスフェードをトリガーする追加改修の検討が必要。
  - 既存のコンボルバーバイパス（5ms）、EQ バイパス（5ms）のフェードとは独立
```

#### MT-06: 予測値と実測True Peakの一致性評価

```
手順:
  1. 各モードでEQ/IR設定後、予測ヘッドルームを記録
  2. ピンクノイズ(-12dBFS RMS)を再生
  3. TruePeakDetectorの測定値を記録
確認:
  - 予測ヘッドルームと実測True Peakの差が2dB以内
  - 予測値が過小評価（実測が予測を超過）していない
合格基準: |予測 - 実測| ≤ 2dB、かつ実測 ≤ 予測+2dB
```

#### MT-07: UI操作時の応答性確認

```
手順:
  1. Auto ON状態でEQスライダーをドラッグ
  2. デバウンス（50ms）後にゲインが更新されることを確認
確認:
  - スライダードラッグ中に音切れ/ノイズが発生しない
  - デバウンス後のゲイン更新がスムーズ
  - UI表示がリアルタイムに追従する
合格基準: ドラッグ中のノイズなし、デバウンス後にゲイン反映
```

#### MT-08: Auto ON/OFFトグル時のUX確認

```
手順:
  1. Auto ON → inputHeadroomEditor/outputMakeupEditorが読み取り専用になる
  2. Auto OFF → エディタが編集可能になる
  3. Auto OFF時に手動編集 → Auto ONに戻す → 再計算される
確認:
  - UIの有効/無効切り替えが即座に反映
  - 手動値がAuto値に上書きされる
合格基準: UXが直感的で一貫性がある
```

#### MT-09: 極端設定時の安全性確認

```
手順:
  1. EQ: Peak +15dB, Q=10.0, fc=100Hz × 3バンド重ね
  2. IR: 大ホールIR（residualRisk ≈ 12dB）
  3. ピンクノイズ -10dBFS RMS を再生
確認:
  - inputが-12dBクランプされる
  - 最終出力がSimplePeakLimiterで-1dBFS以下に制御される
  - クリップインジケーターが点灯しない（リミッターが捕捉）
  - 音質劣化（ポンピング/ディストーション）が許容範囲内
合格基準: クリップなし、リミッター動作が許容範囲
```

#### MT-10: Audio Threadプロファイリング

```
手順:
  1. AudioEngineのプロファイリング機能を有効化
  2. 4モード各々で5分間再生
  3. Audio Threadの処理時間を記録
確認:
  - 改修前と比較してAudio Thread処理時間が増加していない
  - 新規負荷（atomic ロード以外）がAudio Threadにない
  - ブロック予算内に収まる
合格基準: Audio Thread処理時間の増加 ≤ 5%
```

---

## 5. CMakeLists.txt への追加

### DspNumericPolicyTests

```cmake
add_executable(DspNumericPolicyTests
    src/tests/DspNumericPolicyTests.cpp
    src/DspNumericPolicy.cpp
)
target_compile_features(DspNumericPolicyTests PRIVATE cxx_std_20)
target_include_directories(DspNumericPolicyTests PRIVATE
    src src/eqprocessor src/audioengine
)
if(MSVC AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    target_link_libraries(DspNumericPolicyTests PRIVATE MKL::MKL)
endif()
add_test(NAME DspNumericPolicyTests COMMAND DspNumericPolicyTests)
```

### GainStagingContractTests

```cmake
add_executable(GainStagingContractTests
    src/tests/GainStagingContractTests.cpp
)
target_compile_features(GainStagingContractTests PRIVATE cxx_std_20)
target_include_directories(GainStagingContractTests PRIVATE
    src src/eqprocessor src/audioengine
)
add_test(NAME GainStagingContractTests COMMAND GainStagingContractTests)
```

### テスト実行コマンド

```bash
# 全テスト実行
ctest --output-on-failure

# 個別実行
./DspNumericPolicyTests
./ShadowCompareContractTests
./GainStagingContractTests
```

---

## 6. 承認基準とのマッピング

| 承認基準 | 対応テスト | 合格条件 |
|---------|-----------|---------|
| 単体テスト（DspNumericPolicyTests）が全てパス | UT-01〜UT-08 | 全8テストパス |
| コントラクトテスト（ShadowCompareContractTests）が全てパス | IT-01〜IT-07 | 全7テストパス |
| GainStagingContractTestsが全てパス | GC-01〜GC-06 | 全6テストパス |
| 実機での手動テストでリミッター発動頻度が許容範囲内 | MT-01, MT-09 | リダクション ≤ -3dB |
| Audio Threadのプロファイリングで負荷増加なし | MT-10 | 増加 ≤ 5% |
| UI操作時の応答性が従来と同等以上 | MT-07, MT-08 | ノイズなし、即座に反映 |
| プリセットロード/リストア後にAuto ONで正しく再計算 | MT-03 | 正しいゲイン値が設定 |
| 4モード循環切り替えでノイズ/クリップ/音切れなし | MT-04 | 全切り替えで問題なし |
| 純粋オーダー切替時の即時切替でノイズ/クリップなし | MT-05 | 42msフェードイン記録、クリップなし |
| 予測ヘッドルームと実測True Peakの差が2dB以内 | MT-06 | 差 ≤ 2dB |
| 768kHzでのTrue Peak検出器の挙動 | MT-02 | True Peak ≤ +1dBFS |
| リアルタイム安全性が保証される（malloc / mutex / IO が Audio Callback に無い） | 別途追加 RT-01〜RT-04（本表） | 全4項目パス |

### 6.1 リアルタイム安全性テスト (RT-01〜RT-04)

Ross Bencina "Real-time audio programming 101: time waits for nothing" (2011) に基づき、Audio Callback で禁止される操作の不在を静的解析する。

#### RT-01: Audio Callback 内で malloc/new 禁止

```
目的: Audio Thread callback 内でメモリ確保が起きていないこと
方法: 静的解析（ソースコード読み取り）
検証:
  - `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp` の `process()` 内に `malloc`, `new`, `operator new`, `aligned_alloc` が無いこと
  - `dsplib_processor.process*` 系も同様に無いこと
期待結果: 0 hits
ツール: ripgrep / static regex scan
```

#### RT-02: Audio Callback 内で mutex.lock 禁止

```
目的: Audio Thread で priority inversion リスクのある mutex/lock が無いこと
検証:
  - `std::lock_guard<...>`, `std::unique_lock<...>`, `juce::SpinLock::enter` などが Audio Thread 関数内に無いこと
  - すべての lock 操作が `convo::publishAtomic` / `enqueueDeferredDeleteNonRt` など lock-free API 経由であること
期待結果: 0 hits
```

#### RT-03: Audio Thread と Message Thread 間通信が lock-free FIFO

```
目的: Bencina 推奨の "lock-free FIFO queue for RT/non-RT communication" に準拠
検証:
  - `AudioEngine.h:1013` の `static_assert(std::atomic<uint64_t>::is_always_lock_free)` が lock-free 保証
  - `enqueueRetireEpochBounded`（`AudioEngine.h:1058`）が Audio Thread で呼ばれないこと
期待結果: すべてのRT/non-RT通信が lock-free
```

#### RT-04: ヒープピーク動作の Worst-Case Execution Time 評価

```
目的: Audio Thread のブロック時間予算（@48kHz/256サンプル = 5.33ms）を超えないこと
検証:
  - MT-10 の結果とハロメトリクスから、Audio Thread 処理時間が予算の 70% (3.7ms) を超えないこと
  - extreme mode（4モード × 極端設定）で 99th percentile レイテンシが予算内
期待結果: budget_use < 70%
```

---

## 7. テスト実行スケジュール

| タイミング | テスト | 備考 |
|-----------|--------|------|
| Phase 1完了後 | UT-01〜UT-05 | DspNumericPolicy実装直後 |
| Phase 3完了後 | UT-06〜UT-08 | computeMaxGainDb実装直後 |
| Phase 5完了後 | IT-01〜IT-04, GC-01〜GC-06, RT-01〜RT-04 | AudioEngine統合直後 |
| Phase 6完了後 | IT-05, MT-04, MT-05 | モード切替安全設計完了後 |
| Phase 7完了後 | IT-06, IT-07, MT-07, MT-08 | UI統合完了後 |
| 全フェーズ完了後 | MT-01〜MT-03, MT-06, MT-09, MT-10 | 最終検証 |

---

## 8. リスクと対策

| リスク | 対策 |
|--------|------|
| MT-05で42msフェードインによる音切れが検出される | CrossfadeAuthority::evaluate()にprocessingOrder変化検出を追加しクロスフェードをトリガー（別途改修） |
| RT-01〜RT-04 で Audio Callback 内に禁止操作が混入 | CI に ripgrep チェックを Lint 段階に追加 |
| UT-01〜UT-08 で Cookbook RBJ 公式との誤算出 | CI に既知の Cookbook b₀, b₁, b₂, a₀, a₂ の数値検証を追加 |
| UT-05 で Tukey 窓サイドローブが -40dB 超過 | Harris 1978 の -15.6 dB 1st-sidelobe（10 bin で外部減衰で -40dB 達成可能）、65536点FFT で問題なし |
| UT-02でAVX2環境が利用できない | スカラー版のみでテスト実行、AVX2版はCI環境で検証 |
| MT-06で予測値と実測の差が2dBを超える | Q Surge Marginの0.15係数調整、マージン定数の再調整 |
| IT-03でpublicationSemanticHashにゲインが反映されない | RuntimeBuilder.cpp:410-412のハッシュ計算を確認 |
| MT-10でAudio Thread負荷が増加 | recomputeAutoGainStagingがAudio Thread以外で実行されていることを確認 |
