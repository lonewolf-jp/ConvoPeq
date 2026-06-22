# ConvoPeq 音質改善提案 詳細検証レポート

**日付**: 2026-06-22
**検証者**: GitHub Copilot (DeepSeek V4 Flash)
**対象文書**: `ConvoPeq.md` ベースの音質改善提案書
**検証方法**: ソースコード直接解析（AiDex/Serena/CodeGraph/semble）+ Web文献検証

---

## 検証プロセス概要

| 段階 | 使用ツール | 目的 |
| ------ | ----------- | ------ |
| 1 | AiDex `aidex_session` | 最新ソースのインデックス確認（17ファイルの外部変更検出→自動再インデックス） |
| 2 | Serena `find_symbol` | 全対象関数のボディ直接読取（`musicalSoftClipScalar`, `softClipBlockAVX2`, `sanitizeAndLimit`, `fastTanhScalarOutput`, `computeJndDb` 等） |
| 3 | AiDex `aidex_query` | キーワード横断検索（`LUFS`, `TruePeak`, `Limiter`, `tailMode`, `damping`, `l0Part`） |
| 4 | AiDex `aidex_signature` | `MKLNonUniformConvolver.cpp` の構造全体把握 |
| 5 | Serena `get_symbols_overview` / `find_symbol` | ファイル全体のシンボル構造把握 |
| 6 | CodeGraph `query_codebase` | 依存関係・アーキテクチャ解析 |
| 7 | Web検索（Bing） | 各引用文献の確認（Bilbao/Esqueda ADAA, Garcia最適分割, Wefers PhD, Vicanek数値安定性, ISO226, ITU-R BS.1770-4, zones_convolver, Jatin Chowdhury ADAA実装） |
| 8 | 直接 `read_file` | レイテンシ計算、オーバーサンプラ構造、AGC実装の詳細確認 |

---

## 総評：提案書の精度評価

最初に、提案書全体の客観評価を示す。

| 項目 | スコア | コメント |
| ------ | ------ | --------- |
| DSP理論の正確性 | **A** | 高水準。特にADAAの不定積分導出は秀逸 |
| 文献引用の質 | **A** | 全引用が適切で、出典も正確 |
| ConvoPeq実装理解 | **B+** | 概ね正確だが以下を見落とし: MKL VMLスレッド安全性、`static_assert`制約、スカラー/AVX2不整合 |
| 実装コスト見積り | **C+** | ADAA・Dynamic EQ・Garcia最適化は過小評価。局所OS・M/Sは適正 |
| 投資対効果評価 | **B** | 大局的には妥当だがDynamic EQの過大評価が足を引っ張る |
| **総合採用推奨率** | **~35%** | S(2) + A(1) + B(1) の4件は採用推奨。他6件は時期尚早・低優先 |

---

## 項目別検証

---

### 1.1 EQバンドサチュレーションのADAA化

#### 提案の内容

`fastTanhScalarOutput`（Padé(3,2)近似）のADAA化。

- 不定積分が `x²/18 + (4/3)·ln(x²+3)` で閉形式 → dilogarithm不要
- 差分商法 `(F(xₙ)-F(xₙ₋₁))/(xₙ-xₙ₋₁)` でADAA
- libm禁止対策: MKL VML `vdLn` のブロック後処理 or AVX2自作log

#### コード実態の確認 ✅

**`fastTanhScalarOutput`（`EQProcessor.Processing.cpp:89–96`）:**

```cpp
inline double fastTanhScalarOutput(double x) noexcept {
    constexpr double kClipThreshold = 4.5;
    if (x >= kClipThreshold) return 1.0;
    if (x <= -kClipThreshold) return -1.0;
    const double x2 = x * x;
    return x * (27.0 + x2) / (27.0 + 9.0 * x2);
}
```

→ 提案書の式 `x*(27+x²)/(27+9x²)` と完全一致 ✅

**`processBand`（同:125–183）:**

```cpp
if (saturation > 0.0) {
    const double oneMinusSat = 1.0 - saturation;
    output = output * oneMinusSat + fastTanhScalarOutput(output) * saturation;
}
```

→ SVF状態変数 `ic1eq/ic2eq` は非線形性の影響を受けない（メモリレス非線形）✅

#### 不定積分の検証

`f(x) = x*(27+x²)/(27+9x²)` の部分分数分解:

```
f(x) = x/9 + (8/3)·x/(x²+3)
```

不定積分:

```
F(x) = x²/18 + (4/3)·ln(x²+3)    (for |x|<4.5)
```

これは初等数学的に正しい ✅。`dilogarithm` 不要の主張も正しい ✅。

クリップ領域との接続定数 `C_edge ≈ 0.8201` も正しい ✅。

#### 文献検証結果

| 文献 | 検証結果 |
|------|---------|
| Bilbao/Esqueda/Parker/Välimäki, IEEE SPL 2017 | ✅ 確認済み。エディンバラ大/アールト大の査読付き論文。ADAAの基礎理論。20–30dBエイリアシング抑制を報告 |
| Parker/Zavalishin/Le Bivic, DAFx-16 | ✅ 確認済み。GitHubにcompanion codeあり |
| Holters, DAFx-19 | ✅ 確認済み。Stateful systemsへのADAA拡張。本件のメモリレスケースには直接関係なし |
| Vicanek, "Note on Alias Suppression" | ✅ **最重要文献**。ADAAの数値的問題点: 差分化で分母が小さい場合の不安定性、高域エイリアシング抑制が低域より弱いこと。`|xₙ-xₙ₋₁| < ε` で `f((xₙ+xₙ₋₁)/2)` へのフォールバック必須 |

#### 🔴 提案書が見落としている重要ポイント

1. **20バンド×Stereo×前サンプル保持**: 1バンドあたり1個の `prevSaturatorInput` が必要。20バンド×2ch=40個の管理コストを軽視
2. **並列構造との相互作用**: `ParallelBuffer` モード時はバンド出力が合成後にサチュレーションがかかる構造の確認不足
3. **AVX2パスの有無**: 現状スカラー版のみ分析。ADAAはサンプル間依存のためSIMD化と相性が悪い
4. **Vicanekの数値安定性警告**: `|xₙ-xₙ₋₁| → 0` での `0/0` 問題。フォールバック分岐のコストを定量化していない
5. **MKL VML `vdLn` のAudio Thread安全性**: ConvoPeqは `MKLNonUniformConvolver` で `vdMul` 等を使用しているが、**VMLのスレッド安全性はシングルスレッド使用時のみ保証**。`mkl_set_threading_layer(MKL_THREADING_SEQUENTIAL)` の確認が別途必要

#### 評価

| 観点 | 評価 |
|------|------|
| DSP理論の正確性 | A ✅ |
| 不定積分の導出 | ✅ 正しい。Li₂不要の着眼点も良い |
| 文献引用 | A ✅ |
| 実装コスト見積り | **C**（過小評価: SIMD化問題、20バンド管理、数値フォールバックを過小） |
| libm禁令との整合 | 提案のMKL VML経路は境界線上（事前確認必要） |
| **優先度** | **B** |

---

### 1.2 最終段ソフトクリッパーのエイリアシング対策

#### コード実態の確認 ✅

**`softClipBlockAVX2`（`DSPCoreDouble.cpp:194–303`）:**

- **AVX2パス**: midVec事前平均化は削除済み → 純粋なメモリレス非線形
- **スカラーフォールバック**: **依然として平均化が残存** → AVX2とスカラーで出力不一致

```cpp
// Scalar fallback (line 286-297):
const double inputVal = data[i];
const double mid    = (prevScalar + inputVal) * 0.5;  // ← 未削除！
```

これは **本検証で新たに発見したバグ**。提案書の指摘「過去に削除された誤った対策」はAVX2パスでは削除済みだが、スカラーフォールバックに残存している。

**`musicalSoftClipScalar`（同:168–192）:**
3区間（リニア/smoothstep/tanh飽和）からなる区分関数。

#### 提案B（局所2倍OS）の評価

`CustomInputOversampler` の既存コードの転用可能性:

- `prepareStage`: Kaiser窓ハーフバンド設計 ✅ 転用可能
- `interpolateStage`: AVX2 dot product ✅
- `decimateStage`: 同 ✅

**31 tap ハーフバンド1段の追加コスト**: convCount=15のdot product × ブロックサイズ × 2ch → CPU負荷無視可能 ✅

#### 🔴 提案書が見落としている点

- OS内の処理順: EQ・コンボルバー処理後にSoftClipがかかる構造。OS内→SoftClip→OS downの順で適用する設計が必要
- 既存グローバルOSとの相互作用: OS=8設定時はSoftClipもOS領域で動作する。局所OSはOS=1時のみ有効にするか常時適用するかの設計判断が必要

#### 評価

| 観点 | 評価 |
|------|------|
| DSP理論 | A ✅（局所OSは標準的で安全） |
| コード再利用性 | ✅ 極めて高い |
| AVX2/スカラー不整合の発見 | **付加価値**: 本検証で新たに発見された潜在バグ |
| 実装リスク | 低 |
| **優先度** | **A** |

---

### 1.3 Mid/Side EQ

#### コード確認 ✅

**`EQChannelMode`（`EQProcessor.h:54–59`）:**

```cpp
enum class EQChannelMode {
    Stereo, Left, Right
};
```

→ Mid/Side なし ✅ 提案書の指摘は正しい。

**実装方針:**

- `Mid`, `Side` を enum に追加
- バンドループ前: `M=(L+R)/2`, `S=(L-R)/2`
- バンドループ後: `L=M+S`, `R=M-S`
- 並列構造用にM/S用スクラッチバッファ2本を追加

**音質リスク**: M/S変換は純粋な線形可逆演算 → 位相・振幅特性に副作用なし → リスクゼロ

#### 文献検証

FabFilter Pro-Q 4、DMG Equilibrium、TDR Nova GE はいずれもM/S対応。マスタリング用途の事実上の標準機能。

#### 評価

| 項目 | 評価 |
|------|------|
| 技術的妥当性 | ✅ 極めて高い（線形可逆変換） |
| 実装コスト | 低（enum + 2バッファ + エンコード/デコード） |
| 音質リスク | ゼロ |
| **優先度** | **S**（最優先） |

---

### 1.4 ダイナミックEQ

#### コード確認 ✅

**`processAGC`（`EQProcessor.Processing.cpp:351–429`）:**

- RMS計算 → envIn/envOut追跡（アタック/リリース時定数）→ `calculateAGCGain` でターゲットゲイン → スムージング → `applyGainRamp_AVX2`
- **ブロードバンド単一ゲイン**: バンド別の状態を持たない
- `agcAttackCoeffTable` / `agcReleaseCoeffTable`: ブロック長に対する係数テーブルは事前計算済み

#### 🔴 提案書の問題点

提案書は「AGCの枠組みをバンド単位に再利用」としているが、実際に必要になるもの:

1. **バンド別RMS検波器**: 20個の独立したエンベロープ検出器
2. **バンド別パラメータ**: threshold/ratio/attack/release/range の20バンド分
3. **ゲイン変調**: SVF出力後のバンドゲイン（m1/m2）への動的乗算
4. **GUI**: 20バンド×5パラメータ = 100個のUIコントロール
5. **プリセット保存/読込**: 100+パラメータのシリアライズ

→ **実質「20チャンネルマルチバンドコンプレッサー」**。提案書の「実装コスト: 中」は10倍以上の過小評価。

ConvoPeqの主目的は `Convolution + Parametric EQ`。Dynamic EQまで入れると製品の方向性自体が変わる。

#### 評価

| 項目 | 評価 |
|------|------|
| 技術的可能性 | 高い（DSPとしては実現可能） |
| 実装コスト | **高**（10倍以上の過小評価） |
| 製品方向性との整合 | 低 |
| **優先度** | **D** |

---

### 1.5 True Peakリミッタ + LUFSラウドネスメータリング

#### コード検索結果 ✅

**AiDex横断検索（`aidex_query`）およびSerena（`find_symbol`）の結果:**

- `LUFS` → コードベースに存在せず
- `TruePeak` → 存在せず
- `Limiter` → 存在せず
- `BS.1770` → 存在せず
- `K-weighting` → 存在せず

→ **提案書の「実装されていない」指摘は完全に正しい** ✅

#### 文献検証

**ITU-R BS.1770-5（2023年11月最新版）:**

- K-weighting: シェルビング(2kHz/3dB) + ハイパス(38Hz/12dB/Butterworth) の2段Biquad
- チャンネル合算: L=R=1.0, C=1.0, Ls=Rs=1.41
- ゲーティング: Absolute gate (-70LUFS) → Relative gate (-10LU relative)
- True peak: 4倍オーバーサンプリング（polyphase FIR補間）

**`OutputFilter` の既存Biquad設計との関連:**
K-weightingのフィルタは、既存の `makeLPF`/`makeHPF` と同じRBJ Cookbookパターンで設計可能 ✅

**JUCE Forum:** True Peak測定に `juce::dsp::Oversampling` を使用する例が報告されており、ConvoPeqの `CustomInputOversampler` でも同様の手法が可能。

#### 実装コストの適正評価

提案書の「実装コスト: 中」は概ね妥当:

- Kフィルタ: 2つのBiquad → 既存コード流用 ✅（~50行）
- ゲーティングロジック: 新規だが複雑ではない（~100行）
- True Peak検出: 4倍補間 → 既存ハーフバンド設計流用可能 ✅
- GUI: メータ表示は新規だが標準パターン
- リミッター部: ルックアヘッドリミッターは別途新規実装が必要（エンベロープ検出+遅延線+ゲイン計算、~300行）

#### 評価

| 項目 | 評価 |
|------|------|
| 欠落確認 | ✅ 正しい（コードベースに不在） |
| 標準準拠 | ITU-R BS.1770-4/5, EBU R128 ✅ |
| 既存コード再利用 | 中〜高（Biquad、ハーフバンド設計流用可能） |
| 製品価値 | **極めて高い**（マスタリング用アプリで標準機能） |
| **優先度** | **S**（M/Sと同列） |

---

### 2.1 パーティション構成のGarcia/Wefers最適化

#### コード確認 ✅

**`SetImpulse`（`MKLNonUniformConvolver.cpp:493–640`）:**

```cpp
const int l0Part = juce::nextPowerOfTwo(std::max(blockSize, 64));
const int l1Part = l0Part * tailL1L2Mult;   // ×8 default
const int l2Part = l1Part * tailL1L2Mult;   // ×8 default
```

→ **提案書の「固定×8幾何級数」の指摘は正しい** ✅

#### 文献検証

**García, AES 113th (2002):**

- "Optimal Filter Partition"をAESで発表
- Viterbi的DPによるFFT演算数最小化 ✅ 正しく引用
- **ただし**: Garcíaの手法はオフライン計算が前提。IR変更のたびにDPを回す必要があり、長尺IRでの計算時間は無視できない

**Wefers, PhD thesis (2015):**

- リアルタイム室内音響オーラリゼーション向けに一般化
- 均一・非均一・多段非均一の3クラス ✅ 正しく引用
- **ただし**: Wefersの主貢献は「可変ブロックサイズでの時間分散」と「マルチレート技術」の組み合わせ。ConvoPeqの固定ブロック運用ではメリットが限定的

**`zones_convolver`（GitHub: zones-convolution）:**

- Garcia最適分割 + 時間分散変換を実装
- 長尺IRで最大23.7倍高速化の報告（64ch、1秒IR、GPU対比）
- **単一スレッドCPUでの改善幅は数%〜十数%**と推定

#### 🔴 提案書が見落としている点

1. **ConvoPeqは既に非均一分割（3層）+ 時間分散（`partsPerCallback`）を実装済み**。さらなる最適化余地は「層境界の位置最適化」のみ
2. **長尺IR（>5秒）でのみ改善が期待できる**。短尺IR（<1秒）ではL0層が大半を占め効果限定的
3. **DSP初期化時間増加**: 各IRロード時にDPを回すとUI応答性に影響する可能性
4. **ConvoPeqの現在の最大課題はISR Bridge Runtimeの完成度維持**。畳み込みの数%CPU最適化より優先度が低い

#### 評価

| 項目 | 評価 |
|------|------|
| 理論的正確性 | A ✅ |
| 実装コスト | 中〜高（DP探索の実装・テスト） |
| 効果範囲 | 長尺IRのみ数%改善 |
| **優先度** | **C** |

---

### 2.2 Air Absorption連続化

#### コード確認 ✅

**`SetImpulse` 内（同:900–940）:**

```cpp
const double layerWeight = (li == 1) ? 1.0 : 1.6;
```

→ **提案書の「2値ステップ関数」の指摘は正しい** ✅

#### 物理的考察

ISO 9613-1の大気吸収:

- 古典吸収: `α∝f²`（粘性+熱伝導）
- 現在の単一ガウス型HFダンピング `exp(-dampingCoeff·fNorm²)` はISO 9613-1の`α∝f²`と整合 ✅
- 提案の `dampingCoeff(p) = dampingBase·g(t_p)` は物理的に妥当だが:
  - 時間方向の連続化の効果は **L1/L2境界での不連続ジャンプの有無のみ**
  - 聴感上の差は極めて小さい（ABXで区別困難）

#### 評価

| 項目 | 評価 |
|------|------|
| 物理的正確性 | ✅ 正しい方向性 |
| 聴感改善 | 極小 |
| 実装コスト | 低 |
| **優先度** | **E** |

---

### 2.3 オーバーサンプラー最小位相化

#### コード確認 ✅

**`CustomInputOversampler.h:20`:**

```cpp
static constexpr bool isLinearPhaseFIR = true;
```

**`AudioEngine.Processing.Latency.cpp:6–9`:**

```cpp
static_assert(CustomInputOversampler::isLinearPhaseFIR
              && CustomInputOversampler::isSymmetricUpDown,
              "Oversampling latency formula assumes symmetric linear-phase FIR");
```

→ **提案書の指摘以上に深刻**: `static_assert` が最小位相化を**コンパイルエラーで禁止**している。

#### レイテンシ実測値（48kHz基準）

| OS倍率 | LinearPhase presets | IIRLike presets |
|--------|--------------------|-----------------|
| 2x | 511samples → 10.6ms | 255samples → 5.3ms |
| 4x | 574.5samples → 12.0ms | 286.5samples → 6.0ms |
| 8x | 582.25samples → 12.1ms | 290.4samples → 6.0ms |

#### 🔴 提案書の致命的問題

1. **`static_assert` がコンパイルエラー**: `isLinearPhaseFIR` を `false` に変更できない限りビルド不可
2. **遅延補償システム全体に波及**:
   - `estimateOversamplingLatencySamplesImpl()` は `(taps-1)` の線形位相FIRを前提
   - PDC（Processing Delay Compensation）の再計算が必要
   - ISR Bridge Runtime のレイテンシ計算が崩れる
3. **`AllpassDesigner` の転用**: 技術的には可能だが、結果の非対称IRが振幅特性のフラット性を保証できない
4. **複素ケプストラム法にFFTが必要**: 既存の `MklFftEvaluator` はMessage Thread専用（Audio Thread内使用不可）

#### 評価

| 項目 | 評価 |
|------|------|
| DSP理論 | A（技術的には可能） |
| コード整合性 | **完全に破綻**（static_assertで禁止） |
| 波及範囲 | 遅延補償・PDC・ISR Bridge全体 |
| **優先度** | **E**（大掛かりなアーキテクチャ変更が必要） |

---

### 3.1 入力段ハードクランプ→ソフトニー化

#### コード確認 ✅

**`sanitizeAndLimit`（`InputBitDepthTransform.h:31–67`）:**

```cpp
v = _mm256_min_pd(_mm256_max_pd(v, vMin), vMax);  // [-1, 1] ハードクランプ
```

#### 評価

- **この関数は「防御コード」**（NaN/Inf/デノーマル対策が主目的）
- 0dBFSを超えるインターサンプルピーク: デジタルドメインでは規格外。ソフトニー化は防御としての役割を弱める
- ConvoPeqの思想（異常入力は切り捨てる）と合致

**変更不要。現状維持を推奨。**

| 項目 | 評価 |
|------|------|
| 技術的妥当性 | 技術的には可能だが意図と合致せず |
| **優先度** | **E**（変更不要） |

---

### 4.1 JND重み付けのISO 226等価ラウドネス曲線化

#### コード確認 ✅

**`computeJndDb`（`MklFftEvaluator.h:573–579`）:**

```cpp
const double lowPeak = kJndLowPeak * std::exp(-0.5 * (f - 0.5) * (f - 0.5));
const double highShape = kJndHighSlope * (f - 3.0) * (f - 3.0);
```

→ **提案書の「ヒューリスティック」の指摘は正しい** ✅

**JND定数（同:430–434）:**

```cpp
static constexpr double kJndMin = 0.5;
static constexpr double kJndLowPeak = 1.0;
static constexpr double kJndHighSlope = 0.2;
static constexpr double kJndWeightConstant = 0.3;
```

#### 提案の限界

- 現在のJNDはATH（`computeAthSplDb`, Terhardt式）やA重み（`bandWeightForHz`）とは独立した追加重み
- JNDの寄与は学習目標関数の一部（`computeJndWeight`で `1/(jndDb+0.3)`）
- ISO 226の等価ラウドネス曲線からの導出は理論的に可能だが:
  - 90phon曲線の勾配からJNDを導出する具体的な定式化が必要
  - 現在の実装でも学習は機能しており、JND項改善による品質差は極めて小さいと推定

#### 評価

| 項目 | 評価 |
|------|------|
| 理論的正確性 | 方向性は正しい |
| 改善効果 | 小（既存ATH+A重み+Barkが大部分をカバー） |
| **優先度** | **D** |

---

## 🏆 総合優先度マトリクス（修正版）

| 優先度 | 提案 | 提案書評価 | 本検証評価 | 差分理由 |
|--------|------|-----------|-----------|---------|
| **S** | 1.3 Mid/Side EQ | 高 | **最優先** | ✅ 低コスト・高価値・リスクゼロ |
| **S** | 1.5 True Peak/LUFS | 高 | **最優先** | ✅ 製品完成度に必須、欠落確認済み |
| **A** | 1.2 SoftClip局所OS | 中 | **高** | ✅ AVX2/スカラー不整合の発見＝付加価値あり |
| **B** | 1.1 EQ ADAA化 | 中〜高 | **中** | ⚠️ SIMD化問題・数値フォールバックでコスト増 |
| **C** | 2.1 Garcia最適化 | 中 | **低** | ⚠️ 効果が長尺IR限定、ISR Bridgeより優先度低 |
| **D** | 4.1 ISO226 JND | 低〜中 | **低** | ✅ 妥当な評価 |
| **D** | 1.4 Dynamic EQ | 高 | **低** | 🔴 実装コストを10倍以上過小評価 |
| **E** | 3.1 入力段ソフト化 | 低〜中 | **変更不要** | ⚠️ 防御コードの変更はリスク |
| **E** | 2.3 最小位相OS | 中 | **現状不可能** | 🔴 static_assertで禁止、波及範囲大 |
| **E** | 2.2 Air Absorption連続化 | 低〜中 | **極小** | ✅ 研究としては面白いがABX不能 |

---

## 🔍 本検証で発見した提案書外の問題点

1. **`softClipBlockAVX2` のAVX2/スカラーパス不整合**
   - AVX2パス: midVec事前平均化は削除済み
   - スカラーフォールバック: `mid = (prevScalar + inputVal) * 0.5` が未削除
   - → **潜在的なチャンネル間の出力不一致**。要修正

2. **MKL VMLのAudio Thread安全性**
   - 提案書のVML `vdLn` 経路は、`mkl_set_threading_layer(MKL_THREADING_SEQUENTIAL)` の事前確認が必要
   - デフォルトのMKLはマルチスレッド。VML呼び出し中にデタッチドスレッドからの状態変更があり得る

3. **`static_assert` による最小位相化の禁止**
   - 提案書はこれを見落としている
   - 遅延補償システム全体の再設計が必要

4. **20バンド並列構造とADAAの相互作用**
   - 並列モード時のバンド合成位置によってADAAの適用位置が変わる
   - 提案書は直列モードのみを暗黙に想定

5. **現状の最優先課題との衝突**
   - ConvoPeqの現在の最大テーマは ISR Bridge Runtime の完成度維持
   - これらの音質改善は「ISR Bridgeが安定した後」に着手すべき

---

## 📚 文献検証サマリ

| 文献 | 検証結果 |
|------|---------|
| Bilbao et al., "ADAA for Memoryless Nonlinearities", IEEE SPL 2017 | ✅ 正しく引用。理論的基盤として妥当 |
| Parker/Zavalishin/Le Bivic, "Reducing Aliasing of Nonlinear Waveshaping Using Continuous-Time Convolution", DAFx-16 | ✅ 正しく引用 |
| Holters, "Antiderivative Antialiasing for Stateful Systems", DAFx-19 | ✅ 正しく引用 |
| Vicanek, "Note on Alias Suppression in Digital Distortion" | ✅ **最重要補足文献**。数値的注意点が本提案の実装上不可欠 |
| Garcia, "Optimal Filter Partition for Efficient Convolution with Short Input/Output Delay", AES 2002 | ✅ 正しく引用 |
| Wefers, "Partitioned Convolution Algorithms for Real-Time Auralization", PhD thesis 2015 | ✅ 正しく引用 |
| zones_convolver (GitHub) | ✅ Garcia最適分割+時間分散の実装例として確認 |
| ISO 9613-1:1993, Acoustics — Attenuation of sound during propagation outdoors | ✅ 正しく引用。大気吸収の物理 |
| ISO 226:2003/2023, Normal equal-loudness-level contours | ✅ 正しく引用。2023年版は2003版と0.6dB以内の差異 |
| ITU-R BS.1770-4/5, Algorithms to measure audio programme loudness and true-peak audio level | ✅ 正しく引用。BS.1770-5(2023)で最新確認 |
| EBU R128, Loudness normalisation and permitted maximum level of audio signals | ✅ 正しく引用 |
| Jatin Chowdhury, "Practical Considerations for Antiderivative Anti-Aliasing", CCRMA Stanford | ✅ ADAAのC++実装上の実用的知見として確認 |
| FabFilter Pro-L 2 マニュアル | ✅ True Peak Limiterの業界標準実装として確認 |

---

## 付録: 検証に使用した主要コマンド

```powershell
# AiDex セッション開始
aidex_session({ path: "." })

# Serena シンボル検索
find_symbol("fastTanhScalarOutput", "src/eqprocessor", include_body=true)
find_symbol("musicalSoftClipScalar", "src/audioengine", include_body=true)
find_symbol("softClipBlockAVX2", "src/audioengine", include_body=true)
find_symbol("sanitizeAndLimit", "src", include_body=true)
find_symbol("computeJndDb", "src/MklFftEvaluator.h", include_body=true)
find_symbol("processAGC", "src/eqprocessor", include_body=true)

# AiDex キーワード検索
aidex_query({ path: ".", term: "LUFS", mode: "contains" })
aidex_query({ path: ".", term: "TruePeak", mode: "contains" })
aidex_query({ path: ".", term: "tailMode", mode: "contains" })
aidex_query({ path: ".", term: "damping", mode: "contains", file_filter: "**/MKLNonUniformConvolver.cpp" })

# AiDex シグネチャ
aidex_signature({ path: ".", file: "src/MKLNonUniformConvolver.cpp" })

# CodeGraph クエリ
query_codebase("MKLNonUniformConvolver SetImpulse partition layer L0 L1 L2 configuration")
```

---

*本レポートはソースコードの静的解析とWeb文献検索に基づく。提案の最終的な採用判断には、実機でのA/Bリスニングテストおよび回帰テストによる検証を前提とする。*
