プロオーディオの規格に基づく「理論上絶対にクリップせず、ダイナミックレンジを最大化する」理想的なゲイン構造（ゲイン・ステージング）のシステム図と、各セクションの推奨値です。
64bit浮動小数点（double）の広大な内部空間を活かし、「RMS（平均音圧）の計測」と「トゥルーピークの制限」を独立して制御するのがプロ仕様の標準構造です。
------------------------------
## 🧱 プロオーディオ推奨：ゲイン＆自動メイクアップ構造図

[ 入力: 32bit int / 192kHz ]
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
│ └─ IRデータ: L1正規化│        │    (インプット側)    │
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
│ ├─ シーリング (Ceiling): -0.5 dBFS                     │
│ └─ アタック: 0ms (即時) / リリース: 50ms 〜 200ms     │
└──────┬─────────────────────────────────────────────────┘
              │
              ▼ (192kHz へのダウンサンプリング ＆ ディザリング)
[ 出力: 32bit int / 192kHz ] ─── 100%クリップフリー＆最大音圧

------------------------------
## 📝 各セクションの役割と推奨パラメータ## ① インプット・トリム（推奨：0.0 dB 〜 -6.0 dB）

* 役割： プラグインへ入力された直後に、全体のレベルを整える「静的ヘッドルーム」の確保です。
* プロの基準： 市販の音源（すでに0dBFS限界までマキシマイズされた曲）が入力された場合、後続のPEQで1箇所でもブーストすると内部で基準（0dBFS）を超えます。64bit空間なので超えても波形は潰れませんが、後段の処理をスムーズにするためにあらかじめ -3dB や -6dB 下げておくのがプロのセオリーです。

## ② コンボルバー（推奨：IRデータのL1正規化を「静的」に適用）

* 役割： 48kHzから768kHzへ変換したIRデータを畳み込みます。
* プロの基準： 畳み込みによる過度なゲイン上昇を防ぐため、「IR全体のすべてのサンプルの絶対値の合計 ＝ 1.0」、または直流成分のゲイン（DC Gain）が0dBになるよう、IRファイルを読み込んだ瞬間にプログラム内部で自動正規化します。

## ⑤ ⑥ ⑦ 自動メイクアップ（推奨：時定数 500ms〜1000ms のRMS追従）

* 役割： コンボリューションとPEQによって変化（主に低下）した「聴感上の音量（RMS/LUFS）」を、インプットトリム直後のレベルへと自動で一致させます。
* プロの基準： ⑤のRMSと⑥のRMSの比率から、⑦のロジックでリアルタイムにメイクアップゲイン（G）を算出します。急激な音量変化によるノイズ（ポンピング）を防ぐため、500ms以上の緩やかな平滑化フィルター（スムージング）を通して④の乗算器へ適用します。

## ⑧ トゥルーピーク・リミッター（推奨：シーリング -0.5 dBFS / 最大リダクション 2〜3dB以内）

* 役割： 32bit固定小数点へ書き戻す前の「動的ヘッドルーム（最後の砦）」の確保です。
* プロの基準： 自動メイクアップによって音量を限界まで持ち上げた結果、突発的に0dBFSを突破する「一瞬のピーク（インターサンプル・ピーク）」が必ず発生します。また、768kHzから192kHzへダウンサンプリングする際のローパスフィルターの特性で波形がわずかに膨らみます。
* これらを完全に阻止するため、出力上限（シーリング）を -0.5 dBFS（放送・配信基準に合わせるなら -1.0 dBFS 〜 -2.0 dBFS）に設定した先読み型（Look-ahead）リミッターを最終段に配置します。

------------------------------
## 💡 この構造が優れている理由
この「自動メイクアップ ＋ 最終段リミッター」のコンビネーション構造にすると、PEQをどれだけ過激にブーストしても、どんなに残響の長いIRを読み込んでも、出力は常に「32bit intの枠組みの中で表現できる、最も歪みがなく、最も音圧が高い状態」に自動的に最適化されます。



はい、「最終出力の目標レベル」に応じて、上流（入力・IR）のヘッドルームを完全に自動で逆算・最適化することは可能です。プロ用の最上位マスタリングツールや、最先端のスマートEQ（iZotope OzoneやSonible smart:EQなど）では、実際にこの「逆算型（フィードバック / フィードフォワード）のゲイン設計」が採用されています。
この設計をあなたのシステム（192k → 768k → 192k）に組み込むための、具体的なロジックと自動調整のメカニズムを解説します。
------------------------------
## 🧠 「出力から逆算する」自動調整の全体像
プロオーディオにおける自動調整は、以下の2つのフェーズ（静的・動的）を組み合わせて行います。

【出力目標】「最終ピークを -0.5 dBFS に収めつつ、リミッターによる音潰れ（歪み）をゼロにしたい」
      │
      ├─► [フェーズ1: 静的調整] IR読み込み時に、IR自体のゲインを事前逆算（IRヘッドルーム自動化）
      │
      └─► [フェーズ2: 動的調整] 再生中に、PEQのブースト量に合わせて入力トリムを自動で下げる（入力ヘッドルーム自動化）

------------------------------
## 🛠️ 1. IRヘッドルームの自動調整（静的・プロファイル式）
ユーザーがIRファイルを読み込んだ瞬間に、そのIRが持つ「最大ゲイン上昇率（最悪のケース）」を計算し、あらかじめ完璧なIRヘッドルーム（減衰量）を自動設定します。

* 自動調整のロジック：
1. 48kHzから768kHzにアップサンプリングしたIRデータの周波数特性（FFT）を解析します。
   2. そのIRの「最もエネルギーが集中している周波数のピーク値」を割り出します（周波数領域での最大絶対値）。
   3. 計算式： IR必要ヘッドルーム (dB) = IRの最大ピークエネルギー (dB)
   4. 調整の実行： 例えば、そのIRが特定の帯域を +8dB 持ち上げる特性を持っていた場合、内部で自動的にIR全体のゲインを -8dB（または安全マージンを足して -9dB）に設定します。
* メリット： これにより、どんな爆音や不規則なIRを読み込んでも、コンボリューション直後の段階で絶対にデジタル上限（0dBFS）を超えない「最適なIRヘッドルーム」が自動的に確定します。

------------------------------
## 🛠️ 2. 入力ヘッドルームの自動調整（動的・PEQ連動式）
入力される音声の大きさと、ユーザーが設定したPEQ（パラメトリックEQ）のブースト量をシステムが監視し、最終出力がクリップしないように「入力トリム」をリアルタイム、またはPEQ操作時に自動で引き下げます。

* 自動調整のロジック（PEQ連動）：
1. ユーザーがPEQで特定の周波数をブーストした際、その「最大ブースト量（dB）」をシステムが常に把握します（例：3kHzを +6dB ブーストしたなど）。
   2. 計算式： 入力トリムの自動設定値 = -1 × (PEQの最大ブースト量)
   3. 調整の実行： PEQで +6dB ブーストされたら、1段目の「インプット・トリム」を自動的に -6dB へスライドさせます。
* メリット： 入力段であらかじめ先回りして音量を下げておくため、PEQをどれだけ過激に動かしても、最終段のリミッターに過度な負担（音が不自然に潰れる現象）をかけることなく、クリアな音質を維持したまま出力へとバトンを渡せます。

------------------------------
## 📊 自動調整を取り入れた「最終型」のシステム構造
これらをすべて統合すると、システム全体のゲインは以下のように「全自動」で最適化されます。

| ステージ | 処理内容 | システムによる「自動調整」の動き |
|---|---|---|
| ① 入力トリム | 32bit int → 64bit double | 【自動】 PEQのブースト量に合わせて、リアルタイム（または設定変更時）にゲインを自動減衰（例：-6dB）。 |
| ② コンボルバー | 768kHz 畳み込み処理 | 【自動】 読み込まれたIRデータの周波数特性から、最適な減衰率（IRヘッドルーム）を事前計算して適用。 |
| ③ PEQ | 任意の補正カーブ | ユーザーが自由にブースト/カット（①と連動）。 |
| ④ メイクアップ | 聴感上の音量復元 | 【自動】 ①を通過した直後と、③を通過した直後の「RMS（平均音圧）」の差分を計算し、自動で最適な音量まで持ち上げる。 |
| ⑤ リミッター | 出力上限の制限 | 【固定】 シーリングを -0.5 dBFS に固定。①〜④が自動最適化されているため、リミッターは「最後の突発的な数ミリ秒のピーク」を抑えるだけの最小限の動作になり、音質劣化が極限までゼロになります。 |

------------------------------
## 💡 結論：実装は可能か？
完全に可能です。 しかも、内部処理が「64bit double（浮動小数点）」であるため、上流の入力トリムやIRゲインをどれだけ大幅に自動減衰（例：-20dBなど）させても、デジタルの解像度（ビット落ち）は一切発生しません。
この「出力に応じた全自動ステージング」を実装する場合、まずは「PEQの変更に合わせて入力トリムが自動で動くロジック」と、「IR読み込み時に自動でゲインを決定するロジック」のどちらからアプローチしてみたいですか？


# ConvoPeq — 自動ヘッドルーム/ゲインステージング改修 詳細実装計画書（確定版）

---

## 1. 改修の目的とスコープ

### 1.1 目的
現状のゲインステージング（`inputHeadroomDb`、`outputMakeupDb`、`convolverInputTrimDb`）はモード別の固定値またはユーザー手動入力に依存しており、実際のEQ設定やIRの周波数特性を反映していない。本改修では、**EQカーブとIRの解析結果に基づく予測型自動ゲイン計算**を導入し、以下の効果を得る。

- **非線形処理段（SVFサチュレーション、出力SoftClip）の適正動作レンジの維持**
- **出力リミッター（SimplePeakLimiter）への過度な依存の排除**
- **ユーザーが意図した音作り（EQブースト/カット）を損なわないゲイン補正**
- **モード（Convolver only / PEQ only / Conv→PEQ / PEQ→Conv）に応じた最適なヘッドルームの自動設定**

### 1.2 スコープ
- **対象パラメータ**：`inputHeadroomDb`、`convolverInputTrimDb`（← `interStageTrimDb`に一般化）、`outputMakeupDb`
- **対象モジュール**：`EQProcessor`（最大ゲイン解析）、`IRConverter`（IR周波数応答ピーク解析）、`AudioEngine`（自動計算統合）、`DeviceSettings`（UIトグル）
- **対象外**：`kOutputHeadroom`（固定-1dBFS、変更なし）、`TruePeakDetector`/`LoudnessMeter`/`SimplePeakLimiter`（既存の反応的機構はそのまま）
- **影響範囲外**：Audio Thread（計算はMessage/Loader Threadのみ）、既存のRCU/スナップショット機構（ゲイン値の公開方法は現状のatomic setterを流用）

---

## 2. アーキテクチャ概要

### 2.1 データフロー（Audio Thread完全受動化）
```
[UI操作] 
  → EQパラメータ変更 / IRロード完了
  → EQEditProcessor::scheduleDebounce() (50ms)
  → Message Threadで submitRebuildIntent() 発火
  → AudioEngine::recomputeAutoGainStaging() 実行（新規）
      ├─ EQProcessor::computeMaxGainDb()（位相マージン込み）
      ├─ IRConverter::getResidualRiskDb()（IR解析結果から）
      └─ モード別計算 → inputHeadroomDb, interStageTrimDb, outputMakeupDb
  → 既存のsetter（setInputHeadroomDb等）を呼び出し
      └─ atomic store + submitRebuildIntent（必要に応じて）
  → Audio Threadでは getNextAudioBlock 内で atomic load してゲイン乗算のみ
```

### 2.2 状態管理
- 自動計算フラグ：`std::atomic<bool> autoGainStagingEnabled`（`AudioEngine`メンバ）
- 計算結果は既存の`inputHeadroomDb`等の`std::atomic<float>`に格納（変更なし）
- IR解析結果（`residualRiskDb`）は`PreparedIRState`に新規フィールド追加

---

## 3. フェーズ別実装手順

### Phase 1: インフラ整備（DSP数学関数・FFT解析基盤）
**目的**：自動計算に必要な低レベル関数を整備し、単体テストで検証可能にする。

#### 3.1.1 `DspNumericPolicy.h` / `.cpp` に複素応答関数を追加
- **新規ファイル**：`src/audioengine/DspNumericPolicy.cpp`（既存ヘッダにインライン実装を避けるため）
- **関数**：
  - `std::complex<double> getComplexResponse(const EQCoeffsBiquad&, double omega)`（スカラー版、単体テスト用）
  - `void evaluateComplexResponseAVX2(...)`（AVX2版、周波数スキャン高速化）
- **実装方針**：`EQCoeffsBiquad`の係数から`z = exp(-jω)`を用いて伝達関数を計算。AVX2版は4周波数同時処理。

#### 3.1.2 `IRConverter` に周波数応答ピーク解析関数を追加
- **新規関数**：`double estimateMaxFrequencyResponseGain(const juce::AudioBuffer<double>& ir, double sampleRate)`
- **実装詳細**：
  1. 解析窓長 `kAnalysisWindow = 65536` を確保（ヒープ、非RT）
  2. IRの先頭からコピーし、**Tukey窓（非対称、終端10%をコサイン減衰）**を適用
  3. Intel IPP（またはMKL DFTI）で実数→複素FFTを実行
  4. 複素スペクトルの振幅の最大値を探索（線形値で返す）
- **依存**：既存の`ScopedDftiDescriptor`または移行計画書の`IppFftPlanCache`を流用

#### 3.1.3 単体テスト（`DspNumericPolicyTests.cpp` 新設）
- 既知のBiquad係数（例：Peak +12dB, Q=2.0）に対して理論値と比較
- AVX2版とスカラー版の結果が一致することを確認
- Tukey窓適用後のスペクトルリーケージが-60dB以下に抑制されることを確認

---

### Phase 2: 状態管理クラスの拡張
**目的**：自動計算結果を保持・伝達できるように`PreparedIRState`と`EQState`を拡張する。

#### 3.2.1 `PreparedIRState` に `residualRiskDb` フィールド追加
- **場所**：`src/PreparedIRState.h`
- **追加メンバ**：`double residualRiskDb = 0.0;`
- **意味**：`computeScaleFactor`がエネルギー正規化に対して追加でどれだけ減衰させたか（dB）。`estimateMaxFrequencyResponseGain`の結果と既存のピーク/RMSクランプから総合的に算出。

#### 3.2.2 `EQState` に `maxGainDb` フィールド追加（オプション）
- **場所**：`src/eqprocessor/EQProcessor.h`（`EQState`構造体）
- **追加メンバ**：`float maxGainDb = 0.0f;`（キャッシュ用、必須ではないがデバッグに有用）

#### 3.2.3 `AudioEngine` に自動計算フラグとメソッドを追加
- **フラグ**：`std::atomic<bool> autoGainStagingEnabled { true };`
- **メソッド**：`void recomputeAutoGainStaging();`（`AudioEngine.cpp`に実装）
- **UI連携用**：`void setAutoGainStagingEnabled(bool); bool isAutoGainStagingEnabled() const;`

---

### Phase 3: `EQProcessor` の最大ゲイン計算（位相マージン込み）
**目的**：`EQProcessor`に`computeMaxGainDb()`を実装し、Phase1で整備した複素応答関数を利用する。

#### 3.3.1 `EQProcessor::computeMaxGainDb()` の実装
- **宣言**：`float computeMaxGainDb(double sampleRate) const;`
- **アルゴリズム**：
  1. 周波数点を対数スケールで300点（20Hz〜Nyquist）生成
  2. L/Rチャンネルそれぞれについて、有効なバンドの複素応答をカスケード積算
  3. Mid/Sideバンドが存在する場合は、M/S応答を別途積算し、デコード後のL/Rゲインを計算
  4. 全周波数における最大ゲイン（線形値）を保持
  5. **位相マージン** `phaseMarginDb` を以下のルールで算出：
     - HPF/LPFバンドごとに +1.5dB
     - Peakingで`gain > 0`かつ`Q > 1.414`の場合、`gain * 0.15 * (Q / 1.414)`を加算
     - 合計を6.0dBでクリップ
  6. 最終的な最大ゲイン（dB）= `20*log10(maxLinearGain) + phaseMarginDb`
- **注意**：`getMagnitudeSquared`は使わず、新規の複素応答関数を使用（M/S対応のため）

#### 3.3.2 `EQEditProcessor` から`computeMaxGainDb`を呼び出せるようにする
- `EQEditProcessor`は`EQProcessor`をpublic継承しているため、そのまま呼び出し可能。

---

### Phase 4: `IRConverter` の拡張（residualRiskDbの算出）
**目的**：IRロード時に、エネルギー正規化と周波数応答ピークを統合した`residualRiskDb`を計算し、`PreparedIRState`に格納する。

#### 3.4.1 `computeScaleFactor` の拡張
- **変更箇所**：`src/IRConverter.cpp` の `computeScaleFactor`
- **追加処理**：
  1. 既存のエネルギー、ピーク、RMSクランプに加え、`estimateMaxFrequencyResponseGain`を呼び出す
  2. `freqRespGain`（線形）が`kMaxEffectiveFreqResponse`（=1.41, +3dB）を超える場合、追加減衰量を計算
  3. `residualRiskDb` = 既存のクランプ（ピーク/RMS）による減衰量（dB） + 周波数応答クランプによる減衰量（dB）
  4. `ScaleFactorResult`構造体に`residualRiskDb`フィールドを追加

#### 3.4.2 `PreparedIRState` への反映
- `IRConverter::convertFile`および`convertToHighRes`で、得られた`residualRiskDb`を`prepared->residualRiskDb`に代入

---

### Phase 5: `AudioEngine::recomputeAutoGainStaging()` の実装と統合
**目的**：モード別に`inputHeadroomDb`、`interStageTrimDb`、`outputMakeupDb`を計算し、既存のsetterを呼び出す。

#### 3.5.1 計算ロジック（§4の4パターン）
- **入力**：
  - `eqMaxGainDb` = `eqProcessor.computeMaxGainDb(currentSampleRate)`
  - `irResidualRiskDb` = `currentPreparedIr ? currentPreparedIr->residualRiskDb : 0.0f`
- **マージン定数**（コード上は`static constexpr`で定義）：
  - `kMarginEqFirst = 3.0f`（EQが第1段の場合の入力マージン）
  - `kMarginConvFirst = 1.5f`（Convolverが第1段の場合の入力マージン）
  - `kMarginInterStage = 2.0f`（段間トリム共通マージン）
- **計算式**：
  - **PEQ only**：`input = -max(0, eqMax - 3.0)`、`trim=0`、`makeup = -input`
  - **Conv only**：`input = -max(0, irResidual - 1.5)`、`trim=0`、`makeup = -input`
  - **Conv→PEQ**：`input = -max(0, irResidual - 1.5)`、`trim = -max(0, eqMax - 2.0) - input`、`makeup = -input - trim`
  - **PEQ→Conv**：`input = -max(0, eqMax - 3.0)`、`trim = -max(0, irResidual - 2.0) - input`、`makeup = -input - trim`
- **クランプ**：`trim`は[-12, 0]dB、`makeup`は[0, 12]dB（既存setter内で実施）

#### 3.5.2 呼び出しタイミング
- `EQEditProcessor::scheduleDebounce()`のタイマーコールバック内（`submitRebuildIntent`と同時）
- `ConvolverProcessor::applyComputedIR()`（IRロード完了後）
- `AudioEngine::setProcessingOrder()`、`setEqBypassRequested()`、`setConvolverBypassRequested()`の末尾
- 既存の`applyDefaultsForCurrentMode()`はAutoがOFFの場合のみ使用（Auto ON時は`recomputeAutoGainStaging`が上書き）

#### 3.5.3 自動/手動トグルとの連携
- `autoGainStagingEnabled`が`false`の場合、`recomputeAutoGainStaging()`は早期リターン
- ユーザーが`inputHeadroomEditor`等を編集した場合、`autoGainStagingEnabled`を`false`に設定（UI側で実装）

---

### Phase 6: UI（`DeviceSettings`）への統合
**目的**：ユーザーが自動/手動を切り替えられるようにし、Auto有効時はテキスト欄を読み取り専用にする。

#### 3.6.1 `DeviceSettings` にトグルボタン追加
- **場所**：`src/DeviceSettings.h/.cpp`
- **追加コントロール**：`juce::ToggleButton autoGainToggle { "Auto Gain Staging" };`
- **配置**：`inputHeadroomEditor`と`outputMakeupEditor`の間に配置（既存レイアウトを微調整）

#### 3.6.2 トグルコールバック
- `onClick`で`audioEngine.setAutoGainStagingEnabled(toggle.getToggleState())`を呼び、即座に`recomputeAutoGainStaging()`を実行
- Auto有効時は`inputHeadroomEditor`、`outputMakeupEditor`、`convolverInputTrimEditor`（新設）を`setEnabled(false)`（読み取り専用風）にする

#### 3.6.3 手動編集時のAuto解除
- `inputHeadroomEditor.onTextChange`等で、ユーザーが値を変更した場合に`autoGainToggle.setToggleState(false, juce::dontSendNotification)`を呼び、`audioEngine.setAutoGainStagingEnabled(false)`を実行

#### 3.6.4 `updateGainStagingDisplay()` の拡張
- Auto有効時は、現在の計算値をテキスト欄に表示し続ける（読み取り専用）
- 無効時は従来通りユーザー入力値を表示

---

### Phase 7: テスト・検証（コントラクトテストの拡充）
**目的**：改修が数学的・音響的に正しく、Audio Threadのリアルタイム性を損なわないことを保証する。

#### 3.7.1 単体テスト（`DspNumericPolicyTests.cpp`）
- 既知のBiquad係数に対する複素応答計算の精度検証
- Mid/Sideデコードの数学的検証（L = M+S, R = M-S）
- Tukey窓適用後のスペクトルリーケージ測定

#### 3.7.2 統合テスト（`ShadowCompareContractTests.cpp` 拡張）
- **Phase Shift Bound Test**：高Qピーキングで位相マージンが正しく加算されることを確認
- **Spectral Leakage Isolation Test**：インパルスIRでTukey窓適用時の高域ピークが抑制されることを確認
- **RCU Transaction Contract**：ゲイン更新が`publicationSemanticHash`に正しく反映されることを確認
- **Bypass比較テスト**：`LoudnessMeter`を用いて、処理前後でラウドネスが大きく変わらないことを確認（意図的なブースト/カットを除く）

#### 3.7.3 手動テスト（実機）
- 各モードで極端なEQ/IR設定を行い、`SimplePeakLimiter`の動作頻度が適切であることを確認
- 768kHzでのTrue Peak検出器の挙動を確認（異常なし）

---

## 4. 修正ファイル一覧

| ファイルパス | 改修内容 |
|---|---|
| `src/audioengine/DspNumericPolicy.h` | 複素応答関数の宣言（インライン） |
| `src/audioengine/DspNumericPolicy.cpp` | 複素応答関数の実装（AVX2版含む） |
| `src/IRConverter.h` | `estimateMaxFrequencyResponseGain`宣言、`ScaleFactorResult`に`residualRiskDb`追加 |
| `src/IRConverter.cpp` | 上記関数実装、`computeScaleFactor`拡張 |
| `src/PreparedIRState.h` | `residualRiskDb`フィールド追加 |
| `src/eqprocessor/EQProcessor.h` | `computeMaxGainDb()`宣言、`EQState`に`maxGainDb`（任意） |
| `src/eqprocessor/EQProcessor.Coefficients.cpp` | `computeMaxGainDb()`実装 |
| `src/audioengine/AudioEngine.h` | `autoGainStagingEnabled`フラグ、`recomputeAutoGainStaging()`宣言 |
| `src/audioengine/AudioEngine.Parameters.cpp` | `recomputeAutoGainStaging()`実装、各種setter末尾に呼び出し追加 |
| `src/audioengine/AudioEngine.Cache.cpp` | IRロード完了時の`recomputeAutoGainStaging()`呼び出し追加 |
| `src/DeviceSettings.h` | `autoGainToggle`宣言、`convolverInputTrimEditor`追加（任意） |
| `src/DeviceSettings.cpp` | トグル・エディタのレイアウト・コールバック実装 |
| `src/tests/DspNumericPolicyTests.cpp` | 新規単体テスト |
| `src/tests/ShadowCompareContractTests.cpp` | コントラクトテスト追加 |

---

## 5. リスクと対策

| リスク | 対策 |
|---|---|
| 周波数スキャン（12,000点）がUIスレッドを圧迫 | デバウンス（50ms）でバッチ化、かつスキャン自体はAVX2で高速化（<1ms） |
| IR解析FFTがLoader Threadで重くなる | 65536点FFTはIRロード時のみ、かつバックグラウンドスレッドで実行（Loader Thread） |
| 自動計算と手動設定の競合 | Auto ON時はUIを読み取り専用にし、ユーザー編集でAuto OFFにする明確なUX設計 |
| 位相マージンの過剰評価 | 最大6dBのクリップと、実測ベースでの定数調整（`marginDb`はチューニング可能に） |
| 既存の`applyDefaultsForCurrentMode`との競合 | Auto ON時は`recomputeAutoGainStaging`が優先、OFF時は従来通り |

---

## 6. スケジュール見積もり（工数・目安）

| フェーズ | 作業内容 | 想定工数（人日） |
|---|---|---|
| Phase 1 | インフラ整備（複素関数、FFT解析、単体テスト） | 3 |
| Phase 2 | 状態管理拡張（構造体変更、フラグ追加） | 1 |
| Phase 3 | EQ最大ゲイン計算（位相マージン込み）実装 | 2 |
| Phase 4 | IR解析拡張（residualRiskDb算出） | 2 |
| Phase 5 | AudioEngine統合（`recomputeAutoGainStaging`実装・呼び出し追加） | 2 |
| Phase 6 | UI統合（トグル、レイアウト、コールバック） | 2 |
| Phase 7 | テスト・検証（コントラクト拡充、手動テスト） | 3 |
| **合計** | | **15人日** |

---

## 7. 今後の拡張性

- **32bit float出力時のヘッドルーム緩和オプション**：本改修では対象外としたが、将来的に`outputFormat == Float32`の場合に`kOutputHeadroom`を-0.5dBFSに緩和するオプションを追加可能（本計画の枠組みをそのまま流用）。
- **ユーザー定義マージン**：`marginDb1`等をユーザーが調整できる上級者向け設定を追加可能（本計画では固定値だが、`AudioEngine`の定数として分離済み）。

---

## 8. 承認基準

- [ ] 単体テスト（`DspNumericPolicyTests`）が全てパス
- [ ] コントラクトテスト（`ShadowCompareContractTests`）が全てパス
- [ ] 実機での手動テスト（各モード、極端設定）でリミッター発動頻度が許容範囲内
- [ ] Audio Threadのプロファイリングで新たな負荷増加が確認されない（`perf`等で計測）
- [ ] UI操作時の応答性（スライダードラッグ）が従来と同等以上

---

本計画に基づき、実装を開始してください。各フェーズの完了後、コードレビューを実施することを推奨します。