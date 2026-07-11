# `doc/work72/gain.md` 妥当性検証レポート

検証日: 2026-07-11
検証対象: `doc/work72/gain.md`（394行）

---

## 0. 文書の構造

- **前半（1〜124行）**: プロオーディオの理論的ゲインステージング構造の解説
- **後半（127〜394行）**: ConvoPeqプロジェクト向けの具体的な実装計画書（確定版）

全体として、理論→適用→実装という論理的な流れは明確だが、前半の理論と後半の実装計画の間に**重要な乖離**がある。

---

## 1. 前半（プロオーディオ理論）の妥当性検証

### 1.1 技術的に妥当な記述

- **64bit浮動小数点（double）内部処理** → コードベースで完全に検証済み（`AudioEngine.h:565` `using SampleType = double;`、全DSP経路が`AudioBuffer<double>`で動作）
- **RMS/LUFS計測とトゥルーピーク制限の分離** → プロオーディオの標準的アプローチとして妥当。コードベースにも`LoudnessMeter`（ITU-R BS.1770-4/5準拠）と`TruePeakDetector`（4xオーバーサンプリング）が別個に存在
- **Simple Peak Limiterの特性（Attack 0ms Release 50-200ms）** → `SimplePeakLimiter.h:5`の実装と完全一致。Release-onlyでLookAhead FIFOなし
- **768kHzアップサンプリング** → 検証済み。`SAFE_MAX_SAMPLE_RATE = 768000.0`、192kHz入力時はAutoで4x → 768kHz（`AudioEngine.Processing.Latency.cpp:54-57`）
- **ダウンサンプリングのフィルター膨張（inter-sample peak発生）** → 技術的に正確。ローパスリカウストラクション時の周波数変調でピークが増幅される現象は実在する
- **-0.5 dBFSシーリング / -1.0〜-2.0 dBFS推奨** → 放送基準（EBU R128は-1.0 dBFS、Apple Musicは-1.0 dBFS）と整合。妥当

### 1.2 問題点・不正確な点

#### A.「32bit int」入力/出力の記載が実態と不一致
- **第6行**: `[ 入力: 32bit int / 192kHz ]`、**第46行**: `[ 出力: 32bit int / 192kHz ]`
- **実態**: JUCEホストインターフェースは`AudioBuffer<float>`（32bit浮動小数点）で入出力する。コードベースに`int32_t`のAudioBufferは存在しない
- 32bitはディザ/ノイズシェーピングのターゲット深度としては存在する（`DeviceSettings.cpp:654`で16/24/32を選択可能）が、入出力バスのフォーマットではない

#### B. 自動メイクアップの公式が対象範囲を過小評価
- **第37行**: `G = 20 * log10( RMS_in / RMS_out )` はRMS（ラウドネス）復元として妥当
- しかし第61行で「コンボリューションとPEQによって変化（主に低下）」と記載しつつ、**第95行のIRヘッドルーム自動算出ロジックは周波数ピークのみを考慮**し、RMS低下を補償していない。第37行の公式と第95行のロジックが結合されておらず不整合。実装計画書（後半）でもIRの周波数応答ピークを「residualRiskDb」として扱うのみで、RMSベースのメイクアップは実装されていない

#### C. 「L1正規化」の概念が後半のそれと矛盾している
- **第57行**: 「IR全体のすべてのサンプルの絶対値の合計 ＝ 1.0」または「DC Gainが0dB」 → **L1ノルム正規化**
- **実態（`IRConverter.cpp:12-48`）**: Energy-based normalization（`cblas_ddot`で`Σx²`を計算し`1/sqrt(maxChannelEnergy)`を適用、safety margin = 0.5012 ≈ -6dB）。これは**L2ノルム正規化**（エネルギー正規化）であって、L1正規化ではない
- 後半で提案される`estimateMaxFrequencyResponseGain`は周波数領域のピークを追加クランプする方向だが、L1正規化の方針は維持されていない。前半の理論説明と実装計画で正規化方針が変わっている

#### D. 「位相マージン」概念的な導入理由の説明が不十分
- 第72行で「歪みがゼロ」と主張するが、これは内部処理がdoubleであることを根拠にしている。理論的にはdoubleの精度でも量子化ノイズはゼロではない（`double`の最小正規化数 ≈ `2.2e-308` = `-1233 dB`）。実用上は無視できるが「ゼロ」とは不正確

---

## 2. 後半（実装計画書）の技術的妥当性検証

### 2.1 実装計画が示す明白な事実関係

| 主張 | 検証結果 |
|------|----------|
| `inputHeadroomDb`、`outputMakeupDb`、`convolverInputTrimDb`が既存 | ✅ `AudioEngine.h:2176-2188`ですべて確認 |
| `interStageTrimDb`へリネーム/一般化 | ❌ **まだ提案のみ**。実コードでは`convolverInputTrimDb`のまま |
| `kOutputHeadroom`（固定-1 dBFS、変更なし） | ✅ `0.8912509381337456` ≈ -1dBFS、ローカルconstexprとして`AudioEngine.Processing.DSPCoreDouble.cpp:624`他で確認 |
| `EQProcessor::computeMaxGainDb()`を新規実装 | ❌ **まだ提案のみ**。コードベースに存在しない |
| `IRConverter::estimateMaxFrequencyResponseGain()`を新規実装 | ❌ **まだ提案のみ** |
| `PreparedIRState::residualRiskDb`フィールド追加 | ❌ **まだ提案のみ**。現在の`PreparedIRState.h`には存在しない |
| `autoGainStagingEnabled`フラグと`recomputeAutoGainStaging()` | ❌ **まだ提案のみ** |
| `ScaleFactorResult::residualRiskDb` | ❌ **まだ提案のみ** |
| 4つの処理モード | ⚠️ 論理的派生品。2値の`enum ProcessingOrder { ConvolverThenEQ, EQThenConvolver }`（`core/Types.h:11-14`）×バイパスフラグの組み合わせ |
| `EQEditProcessor::scheduleDebounce()`の50msデバウンス | ✅ `EQEditProcessor.h:79` `kDebounceMs = 50`で確認 |
| `submitRebuildIntent()`の存在 | ✅ `AudioEngine.RebuildDispatch.cpp:144`で確認 |
| `ScopedDftiDescriptor` / `IppFftPlanCache`の流用 | ✅ 両方とも存在（`DftiHandle.h`、`MKLNonUniformConvolver.cpp`内部） |
| `ShadowCompareContractTests.cpp`の存在 | ✅ 検証済み（47行） |
| `DspNumericPolicy.h`、`.cpp`、`DspNumericPolicyTests.cpp` | ⚠️ ヘッダは存在（`src/DspNumericPolicy.h`、374行）。`.cpp`とテストファイルは**不在** |

### 2.2 アーキテクチャ設計の妥当性

#### ✅ 妥当な設計判断

1. **Audio Thread完全受動化**（2.1）
   - すべての計算をMessage Thread / Loader Threadで実行し、Audio Threadは`atomic<float>`のロード→乗算のみ。RT-Safe設計として正しく、`DspNumericPolicy.h`の`ASSERT_AUDIO_THREAD()` / `ASSERT_NON_RT_THREAD()`マクロと整合

2. **RCU / スナップショット機構の流用**（2.1、3.5.2）
   - すべてのゲイン変更は`submitRebuildIntent`を経由し、`publicationSemanticHash`で追跡される。既存の安全な公開経路を一貫して利用する方針は妥当

3. **位相マージンの導入**（3.3.1項 5）
   - 高Qピーキングフィルタは周波数応答ピークの近傍で位相が急激に回転し、実波形では瞬時ピークがマグニチュード表示よりも大きくなる現象が存在する。これを補正量として加算する方針は保守的かつ妥当。6dBクリップ上限も適切

4. **Tukey窓（終端10%コサイン減衰）の選択**（3.1.2項 2）
   - IRの後端をスムーズにフェードアウトし、周期性によるスペクトルリーケージを抑制する選択として妥当

5. **クランプ閾値の設計**（3.4.1項 2）
   - `kMaxEffectiveFreqResponse = 1.41 (+3dB)`、ピーク/RMSクランプの既存値（`IRConverter.cpp:54-55`）との整合を保つ設計

6. **モード別の段間マージン設計**（3.5.1）
   - EQが1段目の場合`kMarginEqFirst = 3.0`、Convが1段目の場合`kMarginConvFirst = 1.5`という非対称設計は、PEQブーストのほうがピーク変動が大きいことと整合

7. **Phase 6 UI統合で`autoGainToggle`を追加し、手動編集時にAuto解除するロジック**（3.6.3）
   - 競合管理の設計として妥当

#### ⚠️ 懸念点・不整合

1. **`applyDefaultsForCurrentMode()`の既存ロジックとの競合**
   - 既存（`AudioEngine.Parameters.cpp:296-340`）では、`ConvBypassed && !eqBypassed`の場合`makeup = 0`、`EQThenConvolver && both active`の場合`makeup = +10, convTrim = -6`、それ以外`input = -6, makeup = +12`
   - 後半計画では`applyDefaultsForCurrentMode()`はAuto OFF時のみ使用と明記（3.5.3項）、Auto ON時は`recomputeAutoGainStaging`が上書き。ロジックは共存可能だが、プリセットロード直後にAutoがONだと`m_isRestoringState`抑制と相互作用で想定外の値が出る可能性がある
   - **3.5.2項の「`setProcessingOrder`、`setEqBypassRequested`、`setConvolverBypassRequested()`の末尾」への呼び出し追加**は必要だが、これらの既存setterが`applyDefaultsForCurrentMode()`を呼んだ後に`recomputeAutoGainStaging`する設計では二重更新が発生する。順序の明示が必要

2. **`DspNumericPolicy.h`のパス不正確**
   - 3.1.1項で`src/audioengine/DspNumericPolicy.h` / `.cpp`と指定しているが、実ファイルは`src/DspNumericPolicy.h`（`audioengine`サブディレクトリにない）。`.cpp`も不在
   - ファイルパスの指定ミスであり、実装時に混乱を招く

3. **AVX2版で「4周波数同時処理」（3.1.1項）**
   - 複素応答は`std::complex<double>`なので実部・虚部それぞれ2倍の計算。`__m256d`は4要素double同時処理可能。しかし`z = exp(-jω)`のcos/sin計算は`libm`依存になり、AVX2の`_mm256_sincos_pd`は存在しない（Intel SVMLまたはSLEEFが必要）。計画では「`libm`を避ける」方針（`DspNumericPolicy.h`の方針）と潜在的に矛盾する

4. **65536点FFTを「IRロード時のみ、バックグラウンドスレッドで実行」（5節）**
   - `IRConverter`はLoader Threadで実行されている想定。計画は正しいが、すでに別途実行されているIRロード処理をブロックしない設計の明示が必要

5. **最大12,000点周波数スキャン**（5節リスク表記：「12,000点」 vs 3.3.1項 1の「300点」）
   - 3.3.1項では「対数スケールで300点」、5節リスクには「12,000点」と記載。**矛盾している**。3.3.1項が正しいと思われる（300点で十分精度が出る）

6. **`ScaleFactorResult`への`residualRiskDb`フィールド追加（3.4.1項 4）**
   - 既存の`ScaleFactorResult`は3フィールド（`scaleFactor`、`hasScaleFactor`）のみ。フィールド追加は後方互換だが、呼び出し元で初期化漏れがあると未定義動作になる。明示的なデフォルト値`0.0`が必要（計画書には明記されている）

7. **`makeup`クランプ範囲`[0, 12]dB` vs コードベースの既存デフォルト`+12`**
   - `AudioEngine.h:2178`のデフォルトは`+12`、`applyDefaultsForCurrentMode`でも最大`+12`。計画通り`[0, 12]`で整合。ただし`setInputHeadroomDb`の実装（`AudioEngine.Parameters.cpp:224`）は`[-12, maxDb]`のクランプで、`maxDb`はConv-first routingで変わる。計画書の`input [-12, 0]`クランプと既存の`[-12, maxDb]`の差異に注意

8. **メイクアップ計算で`-input - trim`となっているが、RMS比ベースの式ではない**
   - 計画書3.5.1項の式は`makeup = -input - trim`のみ。しかし前半第37行は`G = 20 * log10( RMS_in / RMS_out )`。後半では**RMS計測ベースのメイクアップは登場しない**。前半の「自動メイクアップ構造」の核心的アイデアが後半で実装されていない。前半が理想論であり、後半が純粋な予測型安全余白方式へ置き換わっている。文書としてこの差異を明記すべき

9. **「位相マージン」という用語の誤用可能性**
   - 3.3.1項 5で`phaseMarginDb`と命名しているが、制御理論のPhase Margin（位相余裕）とは異なる。ここでは高Qピーキング時のピーク膨張を見込む追加マージンのことであり、「Q surge margin」や「peak swell margin」といった命名が実態に合う

10. **`EQEditProcessor`の「public継承しているためそのまま呼び出し可能」（3.3.2項）**
    - `EQEditProcessor.h:28`で`class EQEditProcessor final : public EQProcessor, private juce::Timer`。protected/privateメンバへのアクセスは可能だが、`computeMaxGainDb`を`const`で実装する場合、`EQProcessor`の`EQState`へのアクセス方法を確認する必要がある

### 2.3 リスクと対策（5節）の妥当性

| リスク | 対策 | 評価 |
|--------|------|------|
| 周波数スキャン（12,000点）がUIスレッドを圧迫 | デバウンス+AVX2 | ⚠️ 12,000点は3.3.1項の300点と矛盾。300点ならそもそも圧迫しない |
| IR解析FFTがLoader Threadで重くなる | バックグラウンド実行 | ✅ 妥当 |
| 自動計算と手動設定の競合 | UI読み取り専用化 + 手動編集でAuto OFF | ✅ 妥当 |
| 位相マージンの過剰評価 | 6dBクリップ + チューニング可能 | ✅ 妥当 |
| `applyDefaultsForCurrentMode`との競合 | Auto ON時は`recomputeAutoGainStaging`優先 | ⚠️ 二重更新の可能性は未解明 |

### 2.4 スケジュール見積もり（6節）

15人日は、AVX2複素応答実装 + 65536 FFT統合 + モード別ロジック + UI統合 + コントラクトテスト拡充を考えると**楽観的**。特にAVX2同期検証、IPP/MKL FFT統合のエッジケース対応で+5〜10人日見込むのが現実的。

---

## 3. 総合評価

### 3.1 全体的妥当性: **概ね妥当だが、重要な不整合を含む**

- プロオーディオ理論としての基本方針（double内部処理、ピーク制限とラウドネス計測の分離、IR正規化）は技術的に堅実
- 実装計画は既存コードベース（`AudioEngine`、`EQProcessor`、`IRConverter`、`PreparedIRState`、RCU機構）を正確に把握しており、拡張方針は技術的に成立する
- ただし、後半の計画は前半の「RMS自動メイクアップ」理論を**実装せず、予測型静的マージン方式にすり替えている**。文書全体としてこの差異が明示されていない

### 3.2 主要な問題点

1. **前半と後半の理論的不連続**: 前半はRMS比ベースの動的メイクアップを主張するが、後半ではRMS計測を使わず静的マージン計算に置換。意図的な簡略化かもしれないが、文書の信用性を損なう
2. **「32bit int」入出力の誤記**: 実コードは32bit float
3. **L1正規化 vs L2正規化の不一致**: 前半はL1を推奨するが、実コードはL2（エネルギー正規化）。後半の計画もL2系を維持
4. **「位相マージン」という用語の不正確さ**: 制御理論のPhase Marginと混同されやすい
5. **周波数スキャン点数の矛盾**（300点 vs 12,000点）
6. **ファイルパスの不正確さ**: `src/audioengine/DspNumericPolicy.h`ではなく実際は`src/DspNumericPolicy.h`
7. **スケジュールの楽観見積もり**: 15人日に対して実際は20〜25人日が現実的

### 3.3 実装計画としての価値

不整合はあるものの、**実装計画自体は実行可能で技術的に妥当**。モード別計算式、クランプ範囲、RCU連携、Audio Threadの非侵襲設計、テスト戦略（コントラクトテスト拡充）はいずれも現実的。前半の理論パートを「参考」と割り切り、後半の実装計画をベースに進めるのであれば、十分に作業指示書として機能する。ただし上記7点の不整合は開始前に修正すべき。
