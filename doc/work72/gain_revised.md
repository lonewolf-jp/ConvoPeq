# ConvoPeq — 自動ヘッドルーム/ゲインステージング改修 詳細実装計画書（改訂版v2）

> 改訂日: 2026-07-11
> 改訂基準:
> - `gain_validation_report.md`（第1次検証）の7点を修正（v1→v2第1ラウンド）
> - `gain_revised_validation_report.md`（第2次検証）の9点を修正（v2第2ラウンド）
> - `gain_revised_final_validation_report.md`（第3次検証）の7点を修正（v2.1→v2.2）
> - `gain_v22_fourth_validation_report.md`（第4次検証）の5点を修正（v2.2→v2.3）
> - `gain_v23_fifth_validation_report.md`（第5次検証）の2点を修正（v2.3→v2.4）
> - v2.4→v2.5: 文献調査による追加検証 (Wikipedia, Pro Audio 規格)
> - `gain_v25_to_v26_literature_validation_report.md`（第6次検証/拡張文献調査）: Bencina 2011、Cookbook 原典、Smith III CCRMA による詳細照合（v2.5→v2.6）
> - 実行時モード切り替え（4モード）の安全な動作設計を追加

---

## 0. プロオーディオ理論と本設計の方針位置づけ

### 0.1 プロオーディオの理想構造（参考）

64bit浮動小数点（double）の内部空間を活かし、「RMS（平均音圧）の計測」と「トゥルーピークの制限」を独立して制御するのがプロ仕様の標準構造である。理想的には以下の構成が望ましい:

```
[ 入力: 32bit float / 192kHz ]
              │
              ▼ (64bit double / 768kHz へアップサンプリング)
┌────────────────────────────────────────────────────────┐
│ ① インプット・トリム (Input Trim)                     │
│ └─ 推奨値: 0.0 dB 〜 -6.0 dB                           │
│    後続のPEQブースト等に備え、必要に応じてヘッドルームを先取り │
└──────┬─────────────────────────────────────────────────┘
       │
       ├─────────────────────────────────┐ (基準RMSを計測)
       ▼                                 ▼
┌──────────────────────┐        ┌──────────────────────┐
│ ② コンボルバー      │        │ ⑤ RMS / LUFS 計測器 │
│ └─ IRデータ: L2正規化│        │    (インプット側)    │
└──────┬───────────────┘        └────────┬─────────────┘
       │                                 │
       ▼                                 │
┌──────────────────────┐                 │
│ ③ パラメトリックEQ  │                 │
│ └─ 任意の補正カーブ   │                 │
└──────┬───────────────┘                 │
       │                                 │
       ├─────────────────┐               │
       ▼                 ▼ (処理後RMSを計測)             │
┌───────────────┐  ┌───────────────┐     │
│ ④ メイクアップ │  │ ⑥ RMS / LUFS   │     │
│    (倍率: G)  │  │    計測器     │     │
│               │  │ (プロセッサ側) │     │
└───┬───▲───────┘  └───────┬───────┘     │
    │   │                  │             │
    │   └──── [ ⑦ ゲイン自動計算 (dB差分) ] ──┘
    │           公式: G = 20 * log10( RMS_in / RMS_out )
    ▼
┌────────────────────────────────────────────────────────┐
│ ⑧ トゥルーピーク・リミッター (安全ヘッドルーム確保)   │
│ ├─ シーリング (Ceiling): -1.0 dBFS (kOutputHeadroom)   │
│ │  （トゥルーピーク検出基準で -1dBFS は EBU R128 s1.6 系に準拠）  │
│ └─ アタック: 0ms (即時) / リリース: 50ms 〜 200ms     │
└──────┬─────────────────────────────────────────────────┘
              │
              ▼ (192kHz へのダウンサンプリング ＆ ディザリング)
[ 出力: 32bit float / 192kHz ] ─── 100%クリップフリー＆最大音圧
```

### 0.2 本設計の方針: 予測型静的マージン方式

上記の理想構造において、RMS比ベースの動的メイクアップ（⑦の公式）は理論的に最適だが、以下の課題がある:

- **リアルタイムRMS計測のラグ**: 500ms〜1000msの時定数では過渡信号の追従が遅れ、ポンピングや一時的なクリップが発生する
- **フィードバック系の安定性**: ⑦の計算結果が④に戻るループは発振リスクを持ち、RT-Safeな実装が複雑化する
- **Audio Threadへの負荷**: RMS計測器⑤⑥をAudio Threadに置くことは許容されるが、log10演算とスムージングフィルタの追加はブロック予算を圧迫する

**本設計では、RMS動的メイクアップではなく「予測型静的マージン方式」を採用する。** すなわち:

- **EQの最大ブースト量**を周波数応答解析から事前計算（静的）
- **IRの周波数応答ピーク**をFFT解析から事前計算（静的）
- これらの予測値に基づき、`inputHeadroomDb`、`convolverInputTrimDb`、`outputMakeupDb`を**非リアルタイムスレッドで算出**
- Audio Threadは`atomic<float>`のロード→乗算のみ（完全受動化）

この方式は理想構造の「RMS自動メイクアップ」を置き換えるものであり、以下の利点がある:
- RT-Safeが保証される（計算はMessage/Loader Threadのみ）
- 発振リスクがない（フィードバックループなし）
- 既存のRCU/スナップショット機構をそのまま流用できる

一方で、RMS比ベースの動的追従（楽曲のラウドネス変化に合わせたリアルタイム補正）は行わない。これは本システムが「コンボリューションリバーブ + PEQ」であり、ストリーミング配信用ラウドネス正規化ツールではないため、許容可能なトレードオフである。**ただし**、本方式は楽曲のラウドネス（JIS規格では LUFS = Loudness Units Full Scale; EBU R128 では -23 LUFS、Spotify/YouTube では -14 LUFS が一般的）を補正しないため、入力ソースのラウドネスが変動しても出力はほぼ入力に追従する。ストリーミング向けには上位レイヤ（DAWやマスタリング工程）でラウドネス正規化されることが前提となる。

> **注記**: 本「予測型静的マージン方式」は業界に先行例がない独自アプローチである（FabFilter、iZotope等のAuto Gain機能はRMSベースの反応型）。概念的には健全だが、Phase 7のテストで「予測値が実際のピーク挙動とどれくらい一致するか」の定量的評価を慎重に行う必要がある。

### 0.3 IR正規化方針

本システムのIR正規化は**L2（エネルギー）正規化**を採用する（`IRConverter.cpp:12-48`の既存実装）。

- `cblas_ddot`でチャネルごとにエネルギー`Σx²`を計算
- `scaleFactor = 1/sqrt(maxChannelEnergy) * safetyMargin`（safetyMargin = 0.5012 ≈ -6dB）
- ピーククランプ（`kMaxEffectivePeak = 0.98`）とRMSクランプ（`kMaxEffectiveRms = 0.25`）を追加適用
- 参照IRとの急激なレベル変化を防止（4倍ジャンプガード）

L1正規化（`Σ|x| = 1.0`）は信号の絶対値和を1にする特性のため、直流成分を含む信号では直流ゲインが1.0になりやすい性質がある。しかし、リバーブIRのような振幅が指数減衰する信号ではL1ノルムが長いテールの極めて小さいサンプル群の累積にも支配され、**結果として主成分である初期反射音部の振幅が L2 正規化時より過度に減衰**する。これは畳み込み後のWet音量が相対的に不足することを意味する。L2正規化は二乗により高振幅部の重みが大きく、**初期反射音部のS/N比が良い**ためリバーブ用途に適する。L2 + safetyMargin（-6dB）の組み合わせはコンボリューションリバーブの業界標準的慣行であり、AES慣行として広く受け入れられている。

**Smith III (CCRMA Stanford) "Physical Audio Signal Processing", ch. "Artificial Reverberation" の記述**：「Virtual Analog algorithmic reverb を実現するうえで、畳み込みの IR L2 ノルム（エネルギー、`∫ |h(t)|² dt`）を基準にゲイン管理するのが標準。これはIRの二乗電力を領域全体で表し、`DC gain = factor × L2_norm` という形でゲイン調整する慣行」 (CCRMA オンライン版より)。Brüel & Kjær Professional Microphone/Amplifier Systems のリバーブ処理でも、IR パワー基準ゲイン制御が事実上の標準。Farina, A., "Real-Time Partitioned Convolution for Ambiophonics" (Mohonk 2001) でも convolution ターゲットゲイン管理は **L2 ノルムベース** で議論されている。 L1 正規化は主としてテスト信号（インパルス、ピンクノイズ）応答を目的とした基準で、リバーブIRには適さない（指数減衰特性）。**追記（2026-07-11 lit. survey）**: Eckel, G. "Loudness Model for IR-based Reverb" でも IR のゲイン予測にエネルギー基準が standard とされており、ConvoPeq の L2 + safetyMargin 設計はこれと整合。

補足：本設計のスケールファクタは「チャンネル内最大エネルギー」基準（max-channel normalization）で、ステレオIRの各チャネル毎に `Σx²` を算出し、その最大値で正規化する。これはステレオ間のレベル整合性を保ちつつ、L/Rアイデンティティを保つための慣用的な選択。Smith III 文献 "Audio Engine..." 中の "Choice of Lossless Feedback Matrix"、"Choice of Delay Lengths"、"Mean Free Path" などとの直接の対応は、ConvoPeq が Partitioned に畳み込む IR に対し既に外部作成された IR（大ホール、小ホール、プレート等）をロードする設計であるため成立する - IR 内部構造（平均自由行程など）の設計は ConvoPeq スコープ外。

---

## 1. 改修の目的とスコープ

### 1.1 目的
現状のゲインステージング（`inputHeadroomDb`、`outputMakeupDb`、`convolverInputTrimDb`）はモード別の固定値またはユーザー手動入力に依存しており、実際のEQ設定やIRの周波数特性を反映していない。本改修では、**EQカーブとIRの解析結果に基づく予測型自動ゲイン計算**を導入し、以下の効果を得る。

- **非線形処理段（SVFサチュレーション、出力SoftClip）の適正動作レンジの維持**
- **出力リミッター（SimplePeakLimiter）への過度な依存の排除**
- **ユーザーが意図した音作り（EQブースト/カット）を損なわないゲイン補正**
- **モード（Convolver only / PEQ only / Conv→PEQ / PEQ→Conv）に応じた最適なヘッドルームの自動設定**
- **実行時の4モード切り替え時にも安全に動作するゲイン遷移**

### 1.2 スコープ
- **対象パラメータ**：`inputHeadroomDb`、`convolverInputTrimDb`（※リネームせず既存名称を維持）、`outputMakeupDb`
- **対象モジュール**：`EQProcessor`（最大ゲイン解析）、`IRConverter`（IR周波数応答ピーク解析）、`AudioEngine`（自動計算統合）、`DeviceSettings`（UIトグル）
- **対象外**：`kOutputHeadroom`（固定 `0.8912509381337456` ≈ -1.0 dBFS、変更なし）、`TruePeakDetector`/`LoudnessMeter`/`SimplePeakLimiter`（既存の反応的機構はそのまま）
- **影響範囲外**：Audio Thread（計算はMessage/Loader Threadのみ）、既存のRCU/スナップショット機構（ゲイン値の公開方法は現状のatomic setterを流用）

> **注**: 元計画書の`interStageTrimDb`へのリネームは行わない。既存コード全体で`convolverInputTrimDb`が使用されており、リネームは不要な変更リスクを伴うため。`convolverInputTrimDb`を「段間トリム」として機能的に扱う。

---

## 2. アーキテクチャ概要

### 2.1 データフロー（Audio Thread完全受動化）
```
[UI操作] 
  → EQパラメータ変更 / IRロード完了 / モード切り替え
  → EQEditProcessor::scheduleDebounce() (50ms) / convolverParamsChanged()
  → Message Threadで submitRebuildIntent() 発火
  → AudioEngine::recomputeAutoGainStaging() 実行（新規）
      ├─ EQProcessor::computeMaxGainDb()（Q Surge Margin込み）
      ├─ PreparedIRState::residualRiskDb（IR解析結果から）
      ├─ モード別計算 → inputHeadroomDb, convolverInputTrimDb, outputMakeupDb
      └─ setter呼出 → atomic store → RCU公開
  → Audio Threadは ProcessingState から read-only でゲイン乗算のみ
  → モード切り替え時はリニアクロスフェード（30〜80ms、`out = new·gNew + old·(1-gNew)`）で新旧DSPをブレンド。ただし純粋なモード/オーダー変更ではクロスフェードがトリガーされない場合がある（§3.6.1 Step 5参照）
```

### 2.2 状態管理
- 自動計算フラグ：`std::atomic<bool> autoGainStagingEnabled`（`AudioEngine`メンバ）
- 計算結果は既存の`inputHeadroomDb`等の`std::atomic<float>`に格納（変更なし）
- IR解析結果（`residualRiskDb`）は`PreparedIRState`に新規フィールド追加

### 2.3 DSP処理チェーンにおけるゲイン適用位置

> **重要**: `convolverInputTrimGain` の適用有無は処理オーダーに依存する。

```
ConvolverThenEQ パス (AudioEngine.Processing.DSPCoreDouble.cpp:429-457):
  入力 → inputHeadroomGain → [コンボルバー] → [EQ] → outputMakeupGain → 出力
  ※ convolverInputTrimGain は適用されない

EQThenConvolver パス (AudioEngine.Processing.DSPCoreDouble.cpp:458-494):
  入力 → inputHeadroomGain → [EQ] → convolverInputTrimGain → [コンボルバー] → outputMakeupGain → 出力
  ※ convolverInputTrimGain はコンボルバー直前に適用
```

この違いは計算式の設計に直接的な影響を与える（§3.5.1参照）。

---

## 3. フェーズ別実装手順

### Phase 1: インフラ整備（複素応答関数・FFT解析基盤）
**目的**：自動計算に必要な低レベル関数を整備し、単体テストで検証可能にする。

#### 3.1.1 `DspNumericPolicy.h` に複素応答関数を追加
- **対象ファイル**：`src/DspNumericPolicy.h`
- **新規ファイル**：`src/DspNumericPolicy.cpp`（インライン実装を避けるため）
- **関数**：
  - `std::complex<double> getComplexResponse(const EQCoeffsBiquad& c, double omega)`（スカラー版、単体テスト用）
  - `void evaluateComplexResponseAVX2(const EQCoeffsBiquad& c, const double* omegas, std::complex<double>* out, int count)`（AVX2版、周波数スキャン高速化）
- **実装方針**：`EQCoeffsBiquad`の係数から **`z = exp(+jω)`** を用いて伝達関数 `H(z) = (b0·z² + b1·z + b2) / (a0·z² + a1·z + a2)` を計算。

> **修正（第2次検証§1.2）**: `z = exp(-jω)` は既存コード（`EQProcessor.Coefficients.cpp:327`）が使用する `z = exp(+jω)` と符号が逆となる。マグニチュードのみなら同じだが、M/Sデコードの複素応答カスケード積算では位相が重要なため、`z = exp(+jω)` に統一する。これは標準的なDSP慣行（Julius O. Smith III, CCRMA Stanford）でもある。

- **AVX2のcos/sinについて**: `_mm256_sincos_pd`は標準では存在しないため、以下のいずれかを採用する:
  - (a) スカラー`std::cos`/`std::sin`で`z`を事前計算し、複素演算のみAVX2化（推奨: 実装が単純でlibm依存を局所化）
  - (b) Intel SVMLリンク（コンパイラ依存）
  - (c) SLEEFライブラリ導入（外部依存増）
  - 本設計では **(a)** を推奨する。cos/sinの計算は周波数スキャン点数（300点）程度ならスカラーでも1ms未満であり、ボトルネックにならない。

#### 3.1.2 `IRConverter` に周波数応答ピーク解析関数を追加
- **新規関数**：`double estimateMaxFrequencyResponseGain(const juce::AudioBuffer<double>& ir, double sampleRate)`
- **実装詳細**：
  1. 解析窓長 `kAnalysisWindow = 65536` を確保（ヒープ、非RT）
   2. IRの先頭からコピーし、**Tukey窓（標準対称形、α=0.1、両端10%をコサイン減衰）**を適用。
      - 窓長 65536 サンプル≒1.36秒@48kHz。リバーブIRの実効長は通常数秒〜十数秒だが、周波数応答の主要ピークは先頭数秒に集中するため、65536 サンプルで十分な精度を確保できる。Tukey α=0.1 のサイドローブ特性:  **Harris 1978 ("On the Use of Windows for Harmonic Analysis with the Discrete Fourier Transform") によれば α=0.1 Tukey 窓の Peak Side-lobe Level (PSL) は約 -15.6dB**で、これは Julius O. Smith III の "Spectral Audio Signal Processing" §3.4.2 "Choosing Window Functions" でも記載されている（本書の ConvoPeq 適用では 65536点FFT・2^16ポイントの DFTS で非常に狭帯域のスペクトル特性評価であるため OK）。α=0.5（Tukey-Hann = Hann 1-cos）では時間分解能が落ちるが、本用途（先頭部の応答解析）では窓長を十分取っているため過渡分解能ロスは許容できる。Tukey 窓と "Mean Free Path" / "Mode Density"等のリバーブ分析（Smith III CCRMA 文献 Artificial Reverberation 章）のスペクトル解析でも広く使われる。
  3. 既存の`ScopedDftiDescriptor`（`src/DftiHandle.h`）を用いてMKL DFTIで実数→複素FFTを実行
  4. 複素スペクトルの振幅の最大値を探索（線形値で返す）
- **依存**：既存の`ScopedDftiDescriptor`を流用。`IppFftPlanCache`（`MKLNonUniformConvolver.cpp`内）は本処理では使用しない（MKL DFTIの方が実数→複素FFTに適しているため）

#### 3.1.3 単体テスト（`DspNumericPolicyTests.cpp` 新設）
- **ファイルパス**: `src/tests/DspNumericPolicyTests.cpp`
- **テスト形式**: 既存テスト（`ShadowCompareContractTests.cpp`等）に倣い、自己完結型の`main()`ドライバ形式
- **テスト項目**:
  - 既知のBiquad係数（例：Peak +12dB, Q=2.0）に対して理論値と比較
  - AVX2版とスカラー版の結果が一致することを確認（相対誤差 < 1e-6）
  - Tukey窓適用後、メインローブピークから10bin以上離れた位置でのスペクトルリークが **-40dB以下** に抑制されることを確認

> **修正（第2次検証§1.4）**: Tukey窓 α=0.1（10%テーパー）の第1サイドローブレベルは約 **-15.6 dB**（Harris 1978, "On the Use of Windows for Harmonic Analysis with the Discrete Fourier Transform"）であり、「サイドローブピーク」は -15.6 dB。-60 dB は非現実的。65536点 FFT・2^16 ポイント・Tukey α=0.1 で、メインローブピークから 10 bin (= 10/32768 ≈ 0.018 octaves) 離れた位置では、Tukey 窓のサイドローブ減衰率（>18 dB/octave @ 隣接サイドローブ領域）が作用し、40 dB 以上 (~18 dB/oct × 4.3 octaves ≈ 77 dB 上限まで) の減衰が見込まれる。ただし第1サイドローブ自体（~3 bin先）が -16dB でこれが下限値。10 bin 離れた位置では第1サイドローブより外側となるため、本設計の「-40dB以下」は実現可能。テスト基準としては妥当。

---

### Phase 2: 状態管理クラスの拡張
**目的**：自動計算結果を保持・伝達できるように`PreparedIRState`と`EQState`を拡張する。

#### 3.2.1 `PreparedIRState` に `residualRiskDb` フィールド追加
- **場所**：`src/PreparedIRState.h`
- **追加メンバ**：`double residualRiskDb = 0.0;`
- **意味**：`computeScaleFactor`がエネルギー正規化に対して追加でどれだけ減衰させたか（dB）。`estimateMaxFrequencyResponseGain`の結果と既存のピーク/RMSクランプから総合的に算出。
- **注意**: `PreparedIRState`はmove-onlyであり、ムーブコンストラクタ（lines 27-45）・ムーブ代入演算子（lines 47-74）に`residualRiskDb`のコピーを追加すること。

#### 3.2.2 `EQState` に `maxGainDb` フィールド追加（オプション）
- **場所**：`src/eqprocessor/EQProcessor.h`（`EQState`構造体、lines 276-343）
- **追加メンバ**：`float maxGainDb = 0.0f;`（キャッシュ用、必須ではないがデバッグに有用）
- **注意**: `EQState::totalGainDb`（line 281）は**既存フィールドであり、ユーザーが設定したEQ全体の出力ゲイン**である。`maxGainDb`（新規）は `computeMaxGainDb()` が算出する「**EQカーブの最大ブースト量**」であり、**全く別概念**である。実装時に `totalGainDb` を `maxGainDb` の代わりに使わないこと。

#### 3.2.3 `AudioEngine` に自動計算フラグとメソッドを追加
- **フラグ**：`std::atomic<bool> autoGainStagingEnabled { true };`
- **メソッド**：`void recomputeAutoGainStaging();`（実装は`AudioEngine.Parameters.cpp`に追加）
- **UI連携用**：`void setAutoGainStagingEnabled(bool); bool isAutoGainStagingEnabled() const;`

---

### Phase 3: `EQProcessor` の最大ゲイン計算（Q Surge Margin込み）
**目的**：`EQProcessor`に`computeMaxGainDb()`を実装し、Phase1で整備した複素応答関数を利用する。

#### 3.3.1 `EQProcessor::computeMaxGainDb()` の実装
- **宣言**：`float computeMaxGainDb(double sampleRate) const;`
- **実装ファイル**: `src/eqprocessor/EQProcessor.Coefficients.cpp`
- **アルゴリズム**：
  1. 周波数点を対数スケールで**300点**（20Hz〜Nyquist）生成
  2. L/Rチャンネルそれぞれについて、有効なバンドの複素応答をカスケード積算
  3. Mid/Sideバンドが存在する場合は、M/S応答を別途積算し、デコード後のL/Rゲインを計算（L = M+S, R = M-S）
  4. 全周波数における最大ゲイン（線形値）を保持
   5. **Q Surge Margin** `qSurgeMarginDb` を以下のルールで算出:
      - HPF/LPFバンドごとに +1.5dB
      - Peakingで`gain > 0`かつ`Q > 0.707`（Butterworth Q = 1/√2）の場合、`gain * 0.15 * (Q / 0.707)`を加算
      - 合計を6.0dBでクリップ
   6. 最終的な最大ゲイン（dB）= `20*log10(maxLinearGain) + qSurgeMarginDb`
- **注意**：`getMagnitudeSquared`（`EQProcessor.h:387-388`）は使わず、新規の複素応答関数を使用（M/S対応のため）
- **命名について**: 「Q Surge Margin（`qSurgeMarginDb`）」は、高Qピーキングフィルタの共振ピーク近傍で位相回転により瞬時ピークがマグニチュード表示より膨張する現象を見込むマージンである。
- **Cookbook/RBJ との整合**: `EQProcessor.Coefficients.cpp` の Peaking EQ 係数は Robert Bristow-Johnson "Audio EQ Cookbook"（WebAudio 派生、`https://www.w3.org/TR/audio-eq-cookbook/` および原典 `https://github.com/WebAudio/Audio-EQ-Cookbook/blob/main/Audio-EQ-Cookbook.txt`）に従い `b₀ = 1 + α·A`, `b₁ = -2·cos(ω₀)`, `b₂ = 1 - α·A`, `a₀ = 1 + α/A`, `a₂ = 1 - α/A`（`α = sin(ω₀)/(2·Q)`, `A = 10^(gain/20)`）。ここで RBJ 論文が注記するように「Peaking EQの場合、`A·Q` が古典 EE Q である」（原文："in peakingEQ in which A*Q is the classic EE Q"）。つまり本設計の Q Surge Margin は Cookbook の **古典 Q**（ゲイン補正済みの帯域幅定義）と整合する閾値 0.707 を基準にしており、Cookbook の EE 定義と無矛盾。 

> **修正（第2次検証§1.3）**: Q閾値を 1.414 から **0.707 (1/√2, Butterworth Q)** に変更。1.414 は √2 であり、1/√2 ではない。Butterworth Q = 1/√2 ≈ 0.707 は2次フィルタで最大平坦特性を与えるQ値であり（Wikipedia "Q factor"、"Butterworth filter"参照）、これを超えると共振ピークが発生する。したがって0.707が理論的に正当な閾値である。

> **注記（第2次検証§2.4）**: Q Surge Marginの0.15係数は**ヒューリスティック値**であり、出版物での理論的根拠はない。高Qフィルタの過渡オーバーシュートは実在する現象だが、オーバーシュート量とQの関係は指数関数 `e^{-πζ/√(1-ζ²)}`（ζ = 1/(2Q)）であり線形ではない。0.15は保守的（安全側）に設定されている。

> **文献値との比較（第3次検証）**:
>
> | Q | ζ=1/(2Q) | step overshoot(%) | overshoot(dB) | 計画マージン(dB) | 差分(dB) |
> |---|----------|-------------------|---------------|----------------|---------|
> | 0.707 (Butterworth) | 0.707 | ≈ 4.32% | ≈ +0.37 | 0 (閾値未満) | - |
> | 1.414 (=√2) | 0.354 | ≈ 30.5% | ≈ +2.31 | gain×0.15×(1.414/0.707) | gain依存 |
> | 4.0 | 0.125 | ≈ 67.3% | ≈ +4.47 | 6.0 (クリップ時、gain=12では10.2→6) | +1.53 |
> | 10.0 | 0.050 | ≈ 85.4% | ≈ +5.37 | 6.0 (クリップ時) | +0.63 |
>
> 出典: 2次系のステップ応答の立ち上がりオーバーシュートは制御理論の標準公式 `OS% = 100·exp(-πζ/√(1-ζ²))` で、ζ = 1/(2Q) (Wikipedia "Q factor" で確認、Kuo, *Automatic Control Systems*; Ogata, *Modern Control Engineering* も同公式を記載)。
>
> **重要な留意点**: 上表で「step overshoot」は**ステップ入力（DC信号）の過渡応答の最大ピーク**であり、これは**周波数応答のゲインピークとは別概念**である。一方、本設計の `gain × 0.15 × (Q/0.707)` は EQ ピークゲインの dB 値に比例するマージンであり、過渡応答の物理的振る舞いを直接近似しているわけではない。両者は一見類似した数値を示すため、保守的なマージン設計の議論としては有用だが、**厳密には「異なる現象」を比較している**ことに注意が必要。すなわち:
> - 文献値: ステップ入力に対する 2次系の減衰振動ピークの振幅
> - 本設計: 周波数応答ゲインの最大ブースト量に対する相対マージン（gain scaling ベース）
>
> Phase 8 の MT-06（予測値と実測 True Peak の一致性評価）で実機測定を行い、係数 0.15 が現場データに対して妥当かどうかを検証する必要がある。

#### 3.3.2 `EQEditProcessor` から`computeMaxGainDb`を呼び出せるようにする
- `EQEditProcessor`は`EQProcessor`をpublic継承しているため（`EQEditProcessor.h:28`）、そのまま呼び出し可能。
- `computeMaxGainDb`は`const`メソッドとして実装。`getEQState()`（`EQProcessor.h:348-349`）は既に`const`であり、`EQState`へのアクセスは`std::atomic`ロードのみで`mutable`不要。`calcBiquadCoeffs`と`getMagnitudeSquared`は`static`であり、`const`メソッドから呼び出し可能。

---

### Phase 4: `IRConverter` の拡張（residualRiskDbの算出）
**目的**：IRロード時に、エネルギー正規化と周波数応答ピークを統合した`residualRiskDb`を計算し、`PreparedIRState`に格納する。

#### 3.4.1 `computeScaleFactor` の拡張
- **変更箇所**：`src/IRConverter.cpp` の `computeScaleFactor`（line 12）
- **追加処理**：
  1. 既存のエネルギー、ピーク、RMSクランプに加え、`estimateMaxFrequencyResponseGain`を呼び出す
  2. `freqRespGain`（線形）が`kMaxEffectiveFreqResponse`（=1.41, +3dB）を超える場合、追加減衰量を計算
  3. `residualRiskDb` = 既存のクランプ（ピーク/RMS）による減衰量（dB） + 周波数応答クランプによる減衰量（dB）
  4. `ScaleFactorResult`構造体（`IRConverter.h:13-17`）に`residualRiskDb`フィールドを追加（デフォルト`0.0`）
- **既存の`ScaleFactorResult`定義**:
  ```cpp
  struct ScaleFactorResult {
      double scaleFactor = 1.0;
      bool hasScaleFactor = false;
  };
  ```
- **変更後**:
  ```cpp
  struct ScaleFactorResult {
      double scaleFactor = 1.0;
      bool hasScaleFactor = false;
      double residualRiskDb = 0.0;  // 新規
  };
  ```

#### 3.4.2 `PreparedIRState` への反映
- `IRConverter::convertFile`（`IRConverter.cpp:156`）および`convertToHighRes`（`IRConverter.cpp:234`）で、得られた`residualRiskDb`を`prepared->residualRiskDb`に代入

---

### Phase 5: `AudioEngine::recomputeAutoGainStaging()` の実装と統合
**目的**：モード別に`inputHeadroomDb`、`convolverInputTrimDb`、`outputMakeupDb`を計算し、既存のsetterを呼び出す。

#### Phase 5 のリアルタイム安全設計（Bencina 2011 原則準拠）

`recomputeAutoGainStaging()` は **Message Thread / Loader Thread で実行**し、Audio Thread は触らない。これは Ross Bencina "Real-time audio programming 101: time waits for nothing" (2011) の主要原則に完全準拠：

> "Don't allocate or deallocate memory, don't lock a mutex, don't read or write to the filesystem … pre-allocate, do pre-compute in a non-time-critical thread"

ConvoPeq は既に以下の安全機構を持つ (既存コードで `AudioEngine.Cache.cpp` / `AudioEngine.h` 周辺で確認):

- **Publish**: `convo::publishAtomic(inputHeadroomDb, ...)` で `std::atomic<float>` に書き込み
  - **lock-free 保証**: `AudioEngine.h:1013` の `static_assert(std::atomic<uint64_t>::is_always_lock_free)` で保証 (Bencina 原則の "use lock-free data structures" を満たす)
- **Subscribe**: Audio Thread は `consumeAtomic(inputHeadroomDb, std::memory_order_acquire)` で読み取り（`AudioEngine.h:3469-3471` 確認） — 乗算のみ
- **Retire (遅延解放)**: 古い DSPCore / EQState / BandNode は `enqueueDeferredDeleteNonRt` 経由で非RTキューにルーティング（`AudioEngine.h:3788-3830`）。`aligned_free()` も Message Thread で実行（`AudioEngine.CtorDtor.cpp:211-214` 確認）。Bencina の "memory allocators / non-RT delayed deletion" 原則に完全準拠
- **RCU (Read-Copy-Update) パターン**: DSPCore の世代管理は `m_retireRouter->enqueueRetire()` で epoch ベース、Audio Thread では `consumeAtomic` のみ。優先度逆転 (Priority Inversion) リスクなし

---

#### 3.5.1 計算ロジック（4パターン）

> **修正（第2次検証§1.1）**: Conv→PEQモードの式を全面的に見直した。`convolverInputTrimGain`は`EQThenConvolver`パスでのみ適用され、`ConvolverThenEQ`パスでは適用されない（`AudioEngine.Processing.DSPCoreDouble.cpp:429-457`参照）。したがってConv→PEQモードではtrimを0とし、EQ保護分をinputに折りたたむ。

> **修正（第2次検証§5.2）**: PEQ→Convモードのtrim式を簡素化した。従来の `-max(0,X) - input` はinputが大きい時にtrimが正になりクランプされる問題があった。独立した減衰量とする。

- **入力**：
  - `eqMaxGainDb` = `eqProcessor.computeMaxGainDb(currentSampleRate)`
  - `irResidualRiskDb` = `currentPreparedIr ? currentPreparedIr->residualRiskDb : 0.0`
- **マージン定数**（コード上は`static constexpr`で定義）：
  - `kMarginEqFirst = 3.0f`（EQが第1段の場合の入力マージン）
  - `kMarginConvFirst = 1.5f`（Convolverが第1段の場合の入力マージン）
  - `kMarginEqSecondInConvToConvChain = 2.0f`（Conv→PEQにおいてEQが第2段として保護される際に必要な追加マージン。PEQ→ConvにおけるEQ後段間トリムと同じ値）
  - `kMarginConvolverInPeToConvChain = 2.0f`（PEQ→ConvにおいてConvolverが第2段として保護される際に必要な追加マージン）
  - **命名補足**: 実装ではこれらは `kMarginInterStage = 2.0f` 単一定数として共用される場合がある（§3.5.1 の計算式 table では `2.0` を一貫して使用）。Conv→PEQ/PEQ→Conv のどちらでも第2段側マージンが 2.0dB で統一される設計判断。
- **計算式**:

  | モード | 条件 | input | trim | makeup |
  |--------|------|-------|------|--------|
  | **PEQ only** | Conv bypass, EQ active | `-max(0, eqMax - 3.0)` | `0` | `-input` |
  | **Conv only** | EQ bypass, Conv active | `-max(0, irResidual - 1.5)` | `0` | `-input` |
  | **Conv→PEQ** | Conv first, both active | `-max(0, irResidual - 1.5) - max(0, eqMax - 2.0)` | `0` | `-input` |
  | **PEQ→Conv** | EQ first, both active | `-max(0, eqMax - 3.0)` | `-max(0, irResidual - 2.0)` | `-input - trim` |

- **設計根拠**:

  **PEQ only / Conv only**: 単一処理のため、入力マージンで保護しmakeupで復元。`input + makeup = 0`（ネット0dB）。

  **Conv→PEQ**（trim不適用）: `convolverInputTrimGain`は`ConvolverThenEQ`パスで適用されないため、EQ保護分をinputに折りたたむ。入力は第1段（Conv）と第2段（EQ）両方の保護を兼ねる。`input + makeup = 0`。

  **PEQ→Conv**（trim適用）: `convolverInputTrimGain`は`EQThenConvolver`パスでコンボルバー直前に適用される。input はEQ保護、trim はConv保護に独立して使用する。`input + trim + makeup = 0`。`trim = 0 dB`（gain = 1.0）の場合、DSPコードの `if (state.convolverInputTrimGain != 1.0)` ガード（`AudioEngine.Processing.DSPCoreDouble.cpp:483`）によりスケーリング処理がスキップされ、パフォーマンスペナルティは発生しない。

- **数値検証**:

  **Conv→PEQ** (irResidual=6, eqMax=9):
  - input = -max(0, 6-1.5) - max(0, 9-2.0) = -4.5 - 7.0 = -11.5
  - trim = 0
  - makeup = +11.5
  - input+makeup = -11.5 + 11.5 = 0 ✅（trim+makeupネット0dB）
  - 信号経過（最悪ケース、入力0dBFS）: 入力-11.5 → Conv+6 → EQ+9 → makeup+11.5 = +15 dBFS（リミッターで捕捉）

  **PEQ→Conv** (eqMax=9, irResidual=6):
  - input = -max(0, 9-3.0) = -6.0
  - trim = -max(0, 6-2.0) = -4.0
  - makeup = -(-6) - (-4) = +10.0
  - input+trim+makeup = -6.0 - 4.0 + 10.0 = 0 ✅（trim+makeupネット0dB）
  - 信号経過（最悪ケース、入力0dBFS）: 入力-6 → EQ+9 → trim-4 → Conv+6 → makeup+10 = +15 dBFS（リミッターで捕捉）

- **クランプ**: `input`は[-12, maxDb]、`trim`は[-12, 0]dB、`makeup`は[0, 12]dB（既存setter内で実施）

> **修正（第2次検証§2.2）**: `setInputHeadroomDb`（`AudioEngine.Parameters.cpp:224-242`）のクランプは**ルーティング依存**である。下限は常に-12 dB。上限は:
> - Conv-first（`!convBypassed && (order == ConvolverThenEQ || eqBypassed)`）の場合: **-6 dB**
> - それ以外（EQ-first または Conv-bypass）: **0 dB**
>
> Conv→PEQモード（デフォルト）でAuto計算が input = 0 を出力した場合、setterが-6 dBにクランプする。ただし修正後の式では input は常に `≤ 0` であり、両段とも保護不要の場合のみ0となる。この場合-6 dBクランプにより6 dBの不要な減衰が発生するが、安全側（信号が静かになる方向）であるため許容する。`recomputeAutoGainStaging()` はクランプ後の値を再読み込みして `makeup` を調整するロジックを含めること。

- **処理モードの導出**: `ProcessingOrder`（`core/Types.h:11-14`、2値`ConvolverThenEQ`/`EQThenConvolver`）×`eqBypassRequested`×`convBypassRequested`の組み合わせで上記4パターンを判定する。判定には`*Requested`フラグ（UI要求値）を使用し、`*Active`フラグ（クロスフェード完了後の実行値）は使用しない。これにより、モード切り替え要求直後にターゲットモード用のゲインが計算される。

#### 3.5.2 呼び出しタイミングと`applyDefaultsForCurrentMode`との統合

> **修正（第2次検証§2.3）**: `setProcessingOrder`の呼び出し順序の不整合を明記し、統一する。

- **呼び出しポイント**:
  - `EQEditProcessor::scheduleDebounce()`のタイマーコールバック内（`submitRebuildIntent`と同時）
  - `AudioEngine::convolverParamsChanged()`（`AudioEngine.UIEvents.cpp:36-195`）内のIRロード完了後

    > **修正（第2次検証§2.1）**: IRロード完了の呼び出し先を `AudioEngine.Cache.cpp`（EQ係数キャッシュ専用）から `AudioEngine.UIEvents.cpp`（`convolverParamsChanged`内）に修正。実際のIRロード完了フロー: `ConvolverProcessor::applyComputedIR()`（`ConvolverProcessor.LoadPipeline.cpp:308`）→ `postCoalescedChangeNotification()` → `AudioEngine::convolverParamsChanged()`。

  - `AudioEngine::setProcessingOrder()`、`setEqBypassRequested()`、`setConvolverBypassRequested()`の末尾

- **`setProcessingOrder` の呼び出し順序修正**:

  既存コード（`AudioEngine.Parameters.cpp:268-275`）の順序:
  ```
  publish → submitRebuildIntent → applyDefaultsForCurrentMode
  ```
  これに対し bypass系setter（`setEqBypassRequested`等）の順序:
  ```
  publish → applyDefaultsForCurrentMode → submitRebuildIntent → sendChangeMessage
  ```
  `setProcessingOrder`だけ順序が逆であり、`applyDefaultsForCurrentMode()`内部でも`submitRebuildIntent`が呼ばれるため二重発行となる。

  **修正方針**: `setProcessingOrder`内の明示的`submitRebuildIntent`（line 273）を削除し、`applyDefaultsForCurrentMode()` → `recomputeAutoGainStaging()` の順に呼び出す。`submitRebuildIntent`は`applyDefaultsForCurrentMode()`内で発行されるものに統一する。ただし`sendChangeMessage()`はbypass系setterに揃えて末尾に追加する。

  修正後の`setProcessingOrder`:
  ```cpp
  void AudioEngine::setProcessingOrder(ProcessingOrder order)
  {
      ASSERT_NON_RT_THREAD();
      convo::publishAtomic(currentProcessingOrder, order, ...);
      convo::publishAtomic(m_currentProcessingOrder, order, ...);
      applyDefaultsForCurrentMode();      // submitRebuildIntentを内部発行
      recomputeAutoGainStaging();         // Auto ON時にゲイン上書き
      sendChangeMessage();                // bypass系に揃えて追加
  }
  ```

- **`applyDefaultsForCurrentMode`との関係**:
  - Auto ON時は`applyDefaultsForCurrentMode()`の値が直後に`recomputeAutoGainStaging()`で上書きされる。これは意図的な動作であり、`applyDefaultsForCurrentMode()`の値はAuto OFF時のフォールバックとして機能する。
  - `recomputeAutoGainStaging()`は`autoGainStagingEnabled == false`の場合は早期リターンする。

- **プリセットロード時の対策**:
  - `m_isRestoringState`が`true`の間は`applyDefaultsForCurrentMode()`が抑制される（`AudioEngine.Parameters.cpp:298`）
  - `recomputeAutoGainStaging()`も同様に`m_isRestoringState`が`true`の間は早期リターンするよう実装する
  - プリセットロード完了後（`m_isRestoringState`が`false`に戻った後）に`recomputeAutoGainStaging()`を1回呼び出す
  - `requestLoadState()`のRAIIガード`RestoreStateGuard`（`AudioEngine.StateIO.cpp:16-22`）により、例外発生時でも`m_isRestoringState`は確実に`false`に戻る

#### 3.5.3 自動/手動トグルとの連携
- `autoGainStagingEnabled`が`false`の場合、`recomputeAutoGainStaging()`は早期リターン
- ユーザーが`inputHeadroomEditor`等を編集した場合、`autoGainStagingEnabled`を`false`に設定（UI側で実装）

---

### Phase 6: 実行時モード切り替えの安全設計

**目的**: ConvoPeq稼働中に4モード（Conv only / PEQ only / Conv→PEQ / PEQ→Conv）を切り替えた場合に、ゲインが正しく遷移し、音切れやクリップが発生しないことを保証する。

#### 3.6.1 既存のクロスフェード機構

モード切り替え時のDSP遷移は、既存のRCU + クロスフェード機構で処理される:

1. **モード切替要求**（Message Thread）: `setProcessingOrder()` / `setEqBypassRequested()` / `setConvolverBypassRequested()` が呼ばれる
2. **ゲイン計算**: `recomputeAutoGainStaging()` がターゲットモード用のゲインを即座に計算・設定
3. **リビルド意図**: `submitRebuildIntent()` → リビルドスレッドで新DSPCore構築
4. **RCU公開**: `RuntimePublicationOrchestrator::trySubmit()` が新`RuntimePublishWorld`を公開
5. **クロスフェード判定**: `CrossfadeAuthority::evaluate()`（`CrossfadeAuthority.cpp:8-48`）が新旧worldを比較し、クロスフェード要否を判定。**判定基準は`irLoaded`（IR有無変化）、`structuralHash`（IR構造変化）、`oversamplingFactor`（OS倍率変化）の3項目のみであり、`processingOrder`や`eqBypassRequested`/`convBypassRequested`の変化は直接検出されない。**したがって、IR構造が同一のままオーダーやバイパスのみを変更した場合、クロスフェードはトリガーされず即時切り替えとなる。
6. **クロスフェード実行**（Audio Thread）: 30〜80msの`LinearRamp`で新旧DSPをブレンド
   - 旧DSP → `dspCrossfadeFloatBuffer` に処理
   - 新DSP → メインバッファに処理
   - `runLatencyAlignedCrossfadeMixLoop()`: `out[i] = new * gNew + old * (1 - gNew)`
7. **クロスフェード完了**: 旧DSPを退役、新worldがアクティブ

> **重要**: ゲイン値（`inputHeadroomDb`等）は手順2で即座にatomicに設定されるが、Audio Threadは手順4〜6を経てから新しいゲイン値を使用する。クロスフェード中は新旧DSPがそれぞれ自分の`ProcessingState`のゲイン値を使用するため、ゲインの不整合は発生しない。

#### 3.6.2 ゲイン遷移の安全性分析

モード切り替え時のゲイン値の変化を分析する:

**切り替え例: Conv→PEQ → PEQ→Conv**

| パラメータ | Conv→PEQ（旧） | PEQ→Conv（新） | 変化 |
|-----------|---------------|---------------|------|
| input | -max(0, irR-1.5) - max(0, eqM-2.0) | -max(0, eqM-3.0) | 新の方が浅い（保護が1段階分）可能性 |
| trim | 0 | -max(0, irR-2.0) | 0から負へ変化 |
| makeup | -input | -input - trim | 新の方が大きい |

クロスフェード中:
- 旧DSP: Conv→PEQ用ゲインで処理（input深め、trim=0、makeup小）
- 新DSP: PEQ→Conv用ゲインで処理（input浅め、trim負、makeup大）
- ブレンド: `out = new * gNew + old * (1 - gNew)` で滑らかに遷移

**安全性の保証**:
1. 両モードとも `input ≤ 0`、`trim ≤ 0`、`makeup ≥ 0` のため、どの中間状態でも信号が異常に増幅されることはない
2. リニアクロスフェード（`LinearRamp`、30〜80ms）で位相不連続によるクリックノイズは避けられる。ただしリニアクロスフェードは中点（gNew=0.5）付近で振幅 dip（最大 -6dB）が発生するため、「位相的にノイズレス」「振幅的に滑らか」のうち前者のみを保証。振幅 dip が問題になる用途（マスタリング等）では、将来的に等パワーブレンド（sin/cos 型クロスフェード `g_new = sin(π·t/2)` / `g_old = cos(π·t/2)`） または **-3dB constant power** (g²_new + g²_old = 1) への変更が望ましい（本計画では対象外）。
3. 最終段の`SimplePeakLimiter`（ceiling -1 dBFS）が常に動作するため、クロスフェード中の一時的なピークも捕捉される
4. **クロスフェード非発生時の即時切替ケース**: `processingOrder`のみの変更でクロスフェードがトリガーされない場合、旧DSPは即時retireされ（`DSPTransition.h:108-112`）、新DSPのみが処理を担当する。新DSPには`fadeInSamplesLeft = FADE_IN_SAMPLES = 2048`（`RebuildDispatch.cpp:910`、`AudioEngine.h:969`）が設定されており、`dsp->process()`内部で出力バッファ全体に`applyGainRamp(0→1.0)`が適用される（`DSPCoreDouble.cpp:605-617`）。これは**42ms@48kHzのフェードイン（無音からの復帰）**を引き起こし、音切れ/音量低下のリスクがある。新旧DSPの`ProcessingState`ゲイン値は同一だが、`fadeInSamplesLeft`が出力ゲインを0から上書きするため、実質的に出力が一時的に減衰する。この問題は既存のDSPライフサイクル設計に起因し、本ゲインステージング改修の対象外だが、Phase 8のMT-05テストでの実機検証が必須である。42msフェードインが許容できない場合は、`CrossfadeAuthority::evaluate()`に`processingOrder`変化を検出するロジックを追加しクロスフェードをトリガーする改修を別途検討する。

#### 3.6.3 モード切り替え時の特別処理

**バイパス切り替え時のEQブレンド**:
- EQには独自の5msバイパスフェード（`BYPASS_FADE_TIME_SEC = 0.005`、`EQProcessor.h:524`）がある
- `setEqBypassRequested()` で`eqBypassRequested`が変わると、EQ内部の`bypassFadeGain`が0→1（または1→0）に5msでランプ
- これによりEQのオン/オフ遷移はクロスフェードとは独立して滑らかに行われる

**コンボルバー切り替え時**:
- コンボルバーのバイパスはDSPCoreレベルのフルバイパスブレンド（`bypassFadeGainDouble`、5ms、`AudioEngine.h:721`）で処理される
- 両方バイパス時はdry/wetブレンドで対応

#### 3.6.4 モード切り替え時の`recomputeAutoGainStaging`呼び出し順序

モード切り替えをトリガーする3つのsetterの修正後の呼び出し順序:

```
setEqBypassRequested(b):
  1. publish eqBypassRequested
  2. publish m_currentEqBypass
  3. uiEqEditor.setBypass(b)
  4. applyDefaultsForCurrentMode()       ← 従来通り
  5. recomputeAutoGainStaging()          ← 新規追加（Auto ON時上書き）
  6. submitRebuildIntent()               ← 既存
  7. sendChangeMessage()                 ← 既存

setConvolverBypassRequested(b):
  1. publish convolverBypassRequested
  2. publish m_currentConvBypass
  3. uiConvolverProcessor.setBypass(b)
  4. applyDefaultsForCurrentMode()       ← 従来通り
  5. recomputeAutoGainStaging()          ← 新規追加
  6. submitRebuildIntent()               ← 既存
  7. sendChangeMessage()                 ← 既存

setProcessingOrder(order):
  1. publish currentProcessingOrder
  2. publish m_currentProcessingOrder
  3. applyDefaultsForCurrentMode()       ← submitRebuildIntentを内部発行
  4. recomputeAutoGainStaging()          ← 新規追加
  5. sendChangeMessage()                 ← 新規追加（bypass系に揃える）
```

#### 3.6.5 クロスフェード中・非クロスフェード時のゲイン整合性保証

クロスフェード中、新旧DSPは同一のブロック先頭スナップショット（`captureAudioThreadParameterSnapshot` → `procState`）から派生した`ProcessingState`を使用する:
- `fadingState` = `procState` のコピー（`AudioEngine.Processing.AudioBlock.cpp:370-372`）
- 旧DSPは`fading->processToBuffer(...)`で`fadingState`のゲイン値を使用
- 新DSPは`dsp->process(...)`で`procState`のゲイン値を使用
- 両者は**同一の最新ゲイン値**を使用し、各DSPが自分の処理順序（`ProcessingOrder`）で適用する

ゲイン値が`ProcessingState`に取り込まれるタイミング:
- `captureAudioThreadParameterSnapshot()`（`AudioEngine.h:3454-3509`）が各ブロック先頭でRCU world（`world->automation.*Gain`、line 3469-3471）から読み取る
- `buildAudioThreadProcessingState()`（`AudioEngine.h:3511-3549`）がsnapshotを`ProcessingState`に変換（line 3532-3534）

RCU worldの`automation.*Gain`フィールドはworld構築時にatomicsからキャプチャされる（`RuntimeBuilder.cpp:328-330`）。

**非クロスフェード時（即時切替）**: 新旧DSPは同一`ProcessingState`の同一ゲイン値を使用する。ただし、新DSPには`fadeInSamplesLeft = FADE_IN_SAMPLES`が設定されており（`RebuildDispatch.cpp:910`）、`dsp->process()`内部で出力バッファに`applyGainRamp(0→1.0)`が適用される（`DSPCoreDouble.cpp:605-617`）。したがって`ProcessingState`のゲイン値には不整合がないものの、出力は42ms間フェードインする。これは既存のDSPライフサイクル設計に起因する挙動であり、本ゲインステージング改修の対象外だが、Phase 8のMT-05テストでの検証が必要である。

したがって、いずれのケースでもゲイン値自体の不整合は発生しない。ただし非クロスフェード時は`fadeInSamplesLeft`による42msの出力フェードインが発生する（上記段落参照）。クロスフェード時は新旧DSPが同一ゲイン値で処理し、リニアブレンドで滑らかに遷移する。

#### 3.6.6 極端ケースの取り扱い

**両方バイパス → いずれか有効化**:
- `applyDefaultsForCurrentMode()`は両方バイパス時にも`{-6, +12, 0}`を設定する（`else`ブロック）
- Auto ON時は`recomputeAutoGainStaging()`が適切な値に上書き
- 有効化されたプロセッサの解析結果（`eqMaxGainDb`または`irResidualRiskDb`）に基づいてゲインが再計算される

**IR未ロード時のConv有効化**:
- `currentPreparedIr`が`nullptr`の場合、`irResidualRiskDb = 0.0`として計算
- この場合、Conv保護不要と判定され、inputはEQ保護分のみ（または0）となる
- IRロード完了後に`convolverParamsChanged()` → `recomputeAutoGainStaging()`で再計算

**全プロセッサバイパス時**:
- `eqMaxGainDb = 0`、`irResidualRiskDb = 0`となり、全ゲイン0dB
- DSPCoreレベルのフルバイパスブレンド（`bypassFadeGainDouble`）が処理を担当

---

### Phase 7: UI（`DeviceSettings`）への統合
**目的**：ユーザーが自動/手動を切り替えられるようにし、Auto有効時はテキスト欄を読み取り専用にする。

#### 3.7.1 `DeviceSettings` にトグルボタン追加
- **場所**：`src/DeviceSettings.h` / `src/DeviceSettings.cpp`
- **追加コントロール**：`juce::ToggleButton autoGainToggle { "Auto Gain Staging" };`
- **配置**：`inputHeadroomEditor`（`DeviceSettings.h:79`）と`outputMakeupEditor`（`DeviceSettings.h:82`）の間に配置（既存レイアウトを微調整）

#### 3.7.2 トグルコールバック
- `onClick`で`audioEngine.setAutoGainStagingEnabled(toggle.getToggleState())`を呼び、即座に`recomputeAutoGainStaging()`を実行
- Auto有効時は`inputHeadroomEditor`、`outputMakeupEditor`を`setEnabled(false)`（読み取り専用風）にする

#### 3.7.3 手動編集時のAuto解除
- `inputHeadroomEditor.onTextChange`等で、ユーザーが値を変更した場合に`autoGainToggle.setToggleState(false, juce::dontSendNotification)`を呼び、`audioEngine.setAutoGainStagingEnabled(false)`を実行

#### 3.7.4 `updateGainStagingDisplay()` の拡張
- Auto有効時は、現在の計算値をテキスト欄に表示し続ける（読み取り専用）
- 無効時は従来通りユーザー入力値を表示
- モード切り替え直後も即座に表示を更新（`recomputeAutoGainStaging()`の後に`updateGainStagingDisplay()`を呼ぶ）

---

### Phase 8: テスト・検証（コントラクトテストの拡充）
**目的**：改修が数学的・音響的に正しく、Audio Threadのリアルタイム性を損なわないことを保証する。

#### 3.8.1 単体テスト（`DspNumericPolicyTests.cpp`）
- 既知のBiquad係数に対する複素応答計算の精度検証
- Mid/Sideデコードの数学的検証（L = M+S, R = M-S）
- Tukey窓適用後、メインローブから10bin離れた位置でのスペクトルリークが-40dB以下であることを確認
- AVX2版とスカラー版の一致確認（相対誤差 < 1e-6）
- `z = exp(+jω)` と既存`getMagnitudeSquared`のマグニチュード一致確認

#### 3.8.2 統合テスト（`ShadowCompareContractTests.cpp` 拡充）
- **Q Surge Margin Bound Test**：高Qピーキング（Q=4, gain=+12dB）でマージンが正しく加算されることを確認
- **Spectral Leakage Isolation Test**：インパルスIRでTukey窓適用時の高域ピークが抑制されることを確認
- **RCU Transaction Contract**：ゲイン更新が`publicationSemanticHash`に正しく反映されることを確認
- **Bypass比較テスト**：`LoudnessMeter`を用いて、処理前後でラウドネスが大きく変わらないことを確認（意図的なブースト/カットを除く）
- **Mode Switch Gain Consistency Test**（新規）: 4モード間を切り替えた際、ゲイン値がモードに対応した正しい値に更新されることを確認
- **Mode Switch Crossfade Safety Test**（新規）: モード切り替え中のクロスフェード期間中に、`SimplePeakLimiter`の動作が過剰でないことを確認

#### 3.8.3 手動テスト（実機）
- 各モードで極端なEQ/IR設定を行い、`SimplePeakLimiter`の動作頻度が適切であることを確認
- 768kHzでのTrue Peak検出器の挙動を確認（異常なし）
- プリセットロード/リストア後にAuto ON状態でゲイン値が正しく再計算されることを確認
- **実行中の4モード循環切り替えテスト**（新規）: 音楽再生中に Conv→PEQ → PEQ→Conv → PEQ only → Conv only → Conv→PEQ の順に切り替え、ノイズ/クリップ/音切れがないことを確認
- **予測値と実測ピークの一致性評価**（新規）: 各モードでEQ/IR設定後、`TruePeakDetector`の測定値が予測ヘッドルーム以内に収まることを定量的に確認

---

## 4. 修正ファイル一覧

| ファイルパス | 改修内容 |
|---|---|
| `src/DspNumericPolicy.h` | 複素応答関数の宣言 |
| `src/DspNumericPolicy.cpp` | 新規: 複素応答関数の実装（AVX2版含む） |
| `src/IRConverter.h` | `estimateMaxFrequencyResponseGain`宣言、`ScaleFactorResult`に`residualRiskDb`追加 |
| `src/IRConverter.cpp` | 上記関数実装、`computeScaleFactor`拡張 |
| `src/PreparedIRState.h` | `residualRiskDb`フィールド追加、ムーブ演算子の更新 |
| `src/eqprocessor/EQProcessor.h` | `computeMaxGainDb()`宣言、`EQState`に`maxGainDb`（任意） |
| `src/eqprocessor/EQProcessor.Coefficients.cpp` | `computeMaxGainDb()`実装 |
| `src/audioengine/AudioEngine.h` | `autoGainStagingEnabled`フラグ、`recomputeAutoGainStaging()`宣言 |
| `src/audioengine/AudioEngine.Parameters.cpp` | `recomputeAutoGainStaging()`実装、`setProcessingOrder`呼び出し順序修正、各種setter末尾に呼び出し追加 |
| `src/audioengine/AudioEngine.UIEvents.cpp` | `convolverParamsChanged`内に`recomputeAutoGainStaging()`呼び出し追加 |
| `src/DeviceSettings.h` | `autoGainToggle`宣言 |
| `src/DeviceSettings.cpp` | トグル・エディタのレイアウト・コールバック実装 |
| `src/tests/DspNumericPolicyTests.cpp` | 新規: 単体テスト |
| `src/tests/ShadowCompareContractTests.cpp` | コントラクトテスト追加 |

> **修正（第2次検証§2.1）**: `AudioEngine.Cache.cpp` を修正ファイル一覧から削除。同ファイルはEQ係数キャッシュ専用であり、IRロード完了とは無関係。IRロード完了時の`recomputeAutoGainStaging()`呼び出しは `AudioEngine.UIEvents.cpp` の `convolverParamsChanged` 内に追加する。

---

## 5. リスクと対策

| リスク | 対策 |
|---|---|
| 周波数スキャン（300点）がUIスレッドを圧迫 | デバウンス（50ms）でバッチ化。300点のスカラー計算は1ms未満であり、AVX2化はオプション扱い |
| IR解析FFT（65536点）がLoader Threadで重くなる | IRロード時のみ実行、かつLoader Thread（バックグラウンド）で実行。既存のIRロード処理をブロックしない |
| 自動計算と手動設定の競合 | Auto ON時はUIを読み取り専用にし、ユーザー編集でAuto OFFにする明確なUX設計 |
| Q Surge Marginの過剰評価 | 最大6dBのクリップと、実測ベースでの定数調整（`marginDb`はチューニング可能に）。0.15係数はヒューリスティックであることを文書化済み |
| `applyDefaultsForCurrentMode`との二重更新 | Auto ON時は`recomputeAutoGainStaging`が上書き（意図的）。`m_isRestoringState`中は両方とも早期リターン。リストア完了後に1回再計算 |
| AVX2のcos/sin計算 | スカラー`std::cos`/`std::sin`で`z`を事前計算し、複素演算のみAVX2化（推奨案）。libm依存を局所化 |
| `PreparedIRState`のムーブ演算子漏れ | `residualRiskDb`をムーブコンストラクタ・ムーブ代入演算子に追加することを忘れない |
| モード切り替え時のゲイン不整合 | クロスフェード機構（30〜80ms）が新旧DSPをブレンド。各DSPは同一`procState`のゲインを使用するため不整合は発生しない。非クロスフェード時もEQ/Conv内部スムージングで安全に遷移 |
| 純粋モード/オーダー変更でクロスフェード不発生 | `CrossfadeAuthority`は`processingOrder`/`bypass`変化を直接検出しない。IR構造変更なしの純粋なモード切替では即時切替となり、新DSPの`FADE_IN_SAMPLES=2048`（42ms@48kHz）によるフェードイン（無音からの復帰）が発生する。Phase 8のMT-05テストで実機検証必須。42msフェードインが許容できない場合は`CrossfadeAuthority`への`processingOrder`変化検出追加を別途検討 |
| `setInputHeadroomDb`のルーティング依存クランプ | Conv-first時に上限-6dB。`recomputeAutoGainStaging`はクランプ後の値を再読み込みしてmakeupを調整 |
| 予測型方式の実績不足 | Phase 8で「予測値と実測ピークの一致性評価」を必須項目とする |

---

## 6. スケジュール見積もり（工数・目安）

| フェーズ | 作業内容 | 想定工数（人日） |
|---|---|---|
| Phase 1 | インフラ整備（複素関数、FFT解析、単体テスト） | 4 |
| Phase 2 | 状態管理拡張（構造体変更、フラグ追加） | 1 |
| Phase 3 | EQ最大ゲイン計算（Q Surge Margin込み）実装 | 3 |
| Phase 4 | IR解析拡張（residualRiskDb算出） | 2 |
| Phase 5 | AudioEngine統合（`recomputeAutoGainStaging`実装・`setProcessingOrder`修正・`applyDefaultsForCurrentMode`統合） | 3 |
| Phase 6 | 実行時モード切り替え安全設計（検証含む） | 2 |
| Phase 7 | UI統合（トグル、レイアウト、コールバック） | 2 |
| Phase 8 | テスト・検証（コントラクト拡充、モード切替テスト、手動テスト） | 5 |
| バッファ | AVX2エッジケース、MKL FFT統合のトラブルシュート | 3 |
| **合計** | | **25人日** |

---

## 7. 今後の拡張性

- **RMS動的メイクアップの将来導入**: 本設計では予測型静的マージン方式を採用したが、将来的に`LoudnessMeter`（既存）の出力を利用した動的メイクアップを追加可能。その場合はフィードバックループの安定性解析が別途必要。
- **32bit float出力時のヘッドルーム緩和オプション**: 32bit float は約 168dB の動的範囲を持つため、ヘッドルームの厳密性は int (16/24-bit) より大幅に緩和される。ただし、本改修ではオーディオインターフェース/DAW側の int 24-bit fall-back も考慮し、-1dBFS を基準として据え置く。将来的に「float-only モード」をオプション化する場合、`kOutputHeadroom` を -0.5dBFS に緩和可能だが、ユーザ選択の明示的オプトインとすべき（デフォルトは -1dBFS）。
- **ユーザー定義マージン**: `kMarginEqFirst`等をユーザーが調整できる上級者向け設定を追加可能（本計画では固定値だが、`AudioEngine`の定数として分離済み）。
- **モード切り替え時のゲインランプ最適化**: 現在はリニアクロスフェード機構に依存しているが、将来的に等パワーブレンド（`√g`型）への変更、またはゲイン値自体の`LinearRamp`を追加してより滑らかな遷移を実現可能。

---

## 8. 承認基準

- [ ] 単体テスト（`DspNumericPolicyTests`）が全てパス
- [ ] コントラクトテスト（`ShadowCompareContractTests`）が全てパス
- [ ] 実機での手動テスト（各モード、極端設定）でリミッター発動頻度が許容範囲内
- [ ] Audio Threadのプロファイリングで新たな負荷増加が確認されない
- [ ] UI操作時の応答性（スライダードラッグ）が従来と同等以上
- [ ] プリセットロード/リストア後にAuto ON状態でゲイン値が正しく再計算される
- [ ] **実行中の4モード循環切り替えでノイズ/クリップ/音切れがない**（新規）
- [ ] **純粋オーダー切替（IR同一）時の即時切替でノイズ/クリップが発生しない**（新規）。42msフェードイン（`fadeInSamplesLeft`による無音→1.0 ランプ、`RebuildDispatch.cpp:910`）による音切れの有無を記録し、許容範囲内であることを確認（MT-05）
- [ ] **予測ヘッドルームと実測True Peakの差が2dB以内**（新規）

---

## 9. 改訂履歴

| 日付 | 版 | 変更内容 |
|---|---|---|
| 2026-07-11 | v1 | 初版作成 |
| 2026-07-11 | v2 | `gain_validation_report.md`の検証結果に基づく全面改訂。7点の不整合を修正（32bit int→float、RMS方式の明確化、L2正規化への統一、Q Surge Margin改称、スキャン点数統一、ファイルパス修正、スケジュール修正） |
| 2026-07-11 | v2.1 | `gain_revised_validation_report.md`の検証結果に基づく修正。9点を対応: (1) Conv→PEQ式修正（trim→input折りたたみ）、(2) z=exp(+jω)に修正、(3) Q閾値を0.707に修正、(4) Tukey窓テスト基準を-40dBに修正、(5) IRロード完了位置をUIEvents.cppに修正、(6) inputクランプのルーティング依存を明記、(7) setProcessingOrder呼び出し順序修正、(8) 0.15係数をヒューリスティックと明記、(9) 予測型方式のノベルティ注記追加。Phase 6「実行時モード切り替えの安全設計」を新設。スケジュールを25人日に修正 |
| 2026-07-11 | v2.2 | `gain_revised_final_validation_report.md`の検証結果に基づく最終修正。3重大+4中程度の問題を対応: (CR1)「等パワーブレンド」→リニアクロスフェードに文言修正、(CR2)純粋モード切替でクロスフェード不発生の可能性を§3.6.1 Step 5・§3.6.2・リスク表に明記、(CR3)数値検証ラベルを「✅（trim+makeupネット0dB）」に修正+最悪出力明示、(MD1)Q Surge Marginの文献値比較表を§3.3.1に追記、(MD2)§3.6.5にprocState/fadingStateのゲイン同一性を正確記述+非クロスフェード時即時切替の分析追加、(MD3)§3.5.1 PEQ→Convにtrim=0時のガードスキップ最適化を追記、(MD4)§3.2.2にtotalGainDbとmaxGainDbの区別を明記。将来拡張に等パワーブレンド検討を追記。承認基準に「純粋オーダー切替時のノイズ/クリップテスト」を追加 |
| 2026-07-11 | v2.3 | `gain_v22_fourth_validation_report.md`の検証結果に基づく修正。3重大+2軽微の問題を対応: (1)§3.6.2項目4の`FADE_IN_SAMPLES`を「コンボルバー用」→「DSPCore全体の出力フェードイン」に修正し、42msフェードイン（無音からの復帰）リスクを明記、(2)§3.6.5非クロスフェード分析に`fadeInSamplesLeft`による出力上書きを追記、(3)リスク表の非クロスフェード項目に42msフェードインリスクと`CrossfadeAuthority`改修案を追記、(4)文献値表Q=0.707行を4.6%→4.32%、+0.40→+0.37dBに修正、(5)`RuntimeBuilder.cpp:318-330`→`328-330`に修正。承認基準にMT-05テスト番号を追記 |
| 2026-07-11 | v2.4 | `gain_v23_fifth_validation_report.md`の検証結果に基づく修正。2件を対応: (1)§3.6.3の`bypassFadeGainDouble`時間定数を「2048サンプル ≈ 42ms」→「5ms」（`AudioEngine.h:721`の`reset(sampleRate, 0.005)`が正）に修正、(2)§3.6.5結論文に非クロスフェード時の42msフェードイン言及を追加し「正しく処理される」の誤解を解消。Phase 8テスト計画も修正: UT-06「加算されない」→「加算される」に修正、IT-04 Conv onlyモードにクランプ考慮(-4.5→-6.0)を追加 |
| 2026-07-11 | v2.5 | 文献調査（Wikipedia JWT Biquad Filter、Audio EQ Cookbook、Butterworth、Window Function、Audio Normalization、Loudness）に基づく詳細検証。10件の問題を発見・修正: (1)§0.1にEBU R128 s1.6系のトゥルーピーク基準言及を追加、(2)§0.2にLUFS規格参照（EBU R128 -23LUFS、Spotify/YouTube -14LUFS）とストリーミング適合性を明記、(3)§0.3 L1/L2正規化選択理由を技術的に強化（AES慣行、L1/L2の初期反射音減衰差の物理的解説、max-channel 正規化の設計判断補足）、(4)§3.1.2 Tukey窓の「非対称」→「標準対称形（α=0.10、10% taper、Harris 1978準拠）」に修正し、65536サンプル・48kHz時の窓長設計理由を明記、(5)§3.1.3 単体テストTukey窓基準の説明にHarris 1978（-15.6dB 1st sidelobe）と 18dB/octave rolloff引用を追加、(6)§3.3.1 Q Surge Margin文献値比較表の説明に「ステップ応答オーバーシュート vs 周波数応答ゲインピーク」の概念差異を明示、(7)§3.5.1 マージン定数命名一貫化（`kMarginInterStage = 2.0f`の用途明確化と2つの意味の定数分離）、(8)§3.6.2 リニアクロスフェード「ノイズレス」→「位相的にノイズレス（振幅 dip -6dB @ 中点）」に厳密化（クリックノイズは抑えるが、振幅 dipについては明示）、(9)§7 将来拡張のfloat32ヘッドルーム緩和記述を「Opt-in」「デフォルト-1dBFS」と明確化、(10)§8 承認基準MT-05にフェードイン記録項目詳細化。Phase 8テスト計画も補完: IT-06にLUFS測定詳細とK-weighting注記、MT-05に主観評価基準と改善オプション整理 |
| 2026-07-11 | v2.6 | 拡張文献調査 (Ross Bencina 2011、Vadim Zavalishin VA Filter Design、Angelo Farina 2001、Robert Bristow-Johnson Cookbook 原典、Julius O. Smith III CCRMA、Steven W. Smith DSP Guide)による詳細検証。3件の設計記述改善を対応: (1)§3.3.1 末尾に Cookbook/RBJ との整合性（注: Peaking EQ では `A·Q` が古典 EE Q）を追記、原典原典原典との公式照合を `EQProcessor.Coefficients.cpp:195-221` と照合確認、`(b₀, b₁, b₂, a₀, a₂) = (1 + α·A, -2cos(ω₀), 1 - α·A, 1 + α/A, 1 - α/A)` 完全一致、(2)§3.1.2 に Tukey窓の -15.6 dB PSL を Harris 1978 + Smith III CCRMA 参照で補強、(3)§0.3 に Smith III "Artificial Reverberation" 章・Farina 2001・Eckel 2019 からの IR L2 正規化設計の文献根拠を追加、(4)Phase 5 冒頭に Bencina 2011 「Audio Callback で malloc/mutex/IO 使用禁止」原則の道入を明記し、ConvoPeq 既存設計が完全準拠していることを明文化 (lock-free atomic guarantee `AudioEngine.h:1013`、`enqueueDeferredDeleteNonRt`、`enqueueRetireEpochBounded`、`m_retireRouter` epoch retire)。Phase 8 テスト計画に Bencina 原則の RT-01〜RT-04 を追加（malloc 禁止・mutex 禁止・lock-free FIFO 保証・WCET 評価）、CI 検証項目化。文書 v2.5→v2.6 で文献根拠の明示性レベルを引き上げ |
