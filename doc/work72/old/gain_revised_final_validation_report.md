# `doc/work72/gain_revised.md` 妥当性検証レポート（第3次）

検証日: 2026-07-11
検証対象: `doc/work72/gain_revised.md`（改訂版v2、645行）
検証方法: コードベース実装確認 + 音響工学文献調査 + 数式検証

---

## 0. 検証サマリー

| 重大度 | 項目 | 数 | 概要 |
|--------|------|----|------|
| 🔴 重大 | 実装すると意図通り動作しない / 文書の主張がコード実装と根本的に矛盾 | 3 |
| 🟡 中程度 | 実装は可能だが文書記述に不整合・不足がある | 4 |
| 🟢 軽微 | 妥当、または軽微な改善点のみ | 9 |

---

## 1. 🔴 重大な問題

### 1.1 「等パワーブレンド（LinearRamp）」の記載が実装と矛盾

**改訂版v2 §3.6.2:**
> `LinearRamp`による等パワーブレンド（30〜80ms）で聴感上ノイズレスな遷移が保証される

**実装確認結果（コードベース検証 Item 7）:**

クロスフェードのブレンド式は:

```cpp
// AudioEngine.Processing.AudioBlock.cpp:422
outL[i] = static_cast<float>(alignedNewL * gNew + dryScaledL * gOld);
// AudioEngine.Processing.BlockDouble.cpp:395
outL[i] = alignedNewL * gNew + alignedOldL * gOld;
```

ここで `gOld = 1.0 - gNew`（`AudioEngine.h:3726`）。**これは純粋なリニアクロスフェードである。**

リニアクロスフェード `out = new·gNew + old·(1-gNew)` は:
- `gNew = 0.5` で振幅加算となり、相関信号では最大 +6 dB、無相関信号でも約 +3 dB の局所的ピーク変動が発生
- 等パワーブレンド `out = new·√gNew + old·√(1-gNew)` なら電力一定だが、これは**実装されていない**

**文献調査結果（Wikipedia "Fade (audio engineering)"）:**
> "When the goal is to have the perceived loudness of the combined mix signal stay fairly constant across the full range of the crossfade, special equal power shapes must be used."
> "An example pair of curves that keep power equal across the mix are √m and √(1−m)"
> "This type of fade [linear] is not very natural sounding... at the halfway point ... the perceived volume drops below 50%."

**影響**: 30〜80msのリニアクロスフェードは短期間だが、特にコンボルバーの残響が新旧DSPで異なる場合、クロスフェード中点でラウドネスのディップ（最大 -6 dB相当）が知覚される可能性がある。80msでは持続音に対して確実に知覚可能。

**修正案**:
- (A) 文書の「等パワーブレンド」の記載を削除し、「リニアクロスフェード」と正確に記載
- (B) 必要であれば等パワーブレンド（√g 型）への変更を将来の改善項目として追記

### 1.2 純粋なモード/オーダー変更がクロスフェードをトリガーしない可能性

**改訂版v2 §3.6.1 Step 5-6:**
> クロスフェード判定: `CrossfadeAuthority::evaluate()` が新旧worldを比較し、クロスフェード要否を判定
> クロスフェード実行（Audio Thread）: 30〜80msの`LinearRamp`で新旧DSPをブレンド

**コードベース検証結果（Item 12）:**

`CrossfadeAuthority::evaluate()`（`CrossfadeAuthority.cpp:8-48`）の判定基準は:
- `irLoaded` の変化（IRの有無変化）
- `structuralHash` の変化（IRの構造ハッシュ変化）
- `oversamplingFactor` の変化（オーバーサンプリング倍率変化）

**`processingOrder` の変化それ自体は判定基準に含まれない。** `eqBypassRequested` や `convBypassRequested` の変化も直接の判定基準ではない。

したがって、IRとオーバーサンプリングが同一のまま`ProcessingOrder`のみを切り替えた場合（例: Conv→PEQ → PEQ→Conv）、**クロスフェードはトリガーされず**、新旧DSPの即時切り替えが発生する。これにより:
- 新旧DSPが同一の`ProcessingState`ゲイン値を使用する（同一スナップショット由来）
- ゲイン値の不整合は発生しない（同一値のため）
- しかし処理順序が突然入れ替わるため、コンボルバーの残響が瞬時に切り替わる可能性がある

**実際の切り替えで何が起きるか:**

同じ`world`内でオーダーが変更された場合:
1. `setProcessingOrder` → `submitRebuildIntent` → リビルド → 新world公開
2. `CrossfadeAuthority::evaluate` は `structuralHash` が同じなら `needsCrossfade = false`
3. `DSPTransition::onPublishCompleted` → クロスフェードなしで即時切り替え
4. Audio Threadは新しいProcessingStateを使用（正しいオーダーとゲイン値）

**この動作は安全か？**
- オーダーの即時切り替えは、コンボルバーの残響末尾で「ポップ」ノイズが発生するリスクがある
- しかしEQとコンボルバーの内部には独自のスムージング機構がある（EQ: 5msバイパスフェード、コンボルバー: `FADE_IN_SAMPLES=2048`）
- ATF内部スムージング機構により、短期的なアーティファクトは抑制される可能性が高い

**評価**: 設計上の**潜在的リスク**だが、即座にクリティカルな問題になる可能性は低い。ただし文書にはこの挙動を明記すべき。クロスフェードが発生しないケースがあることを「保証」と記載するのは不正確。

**修正案**:
- §3.6.1 Step 5に「IR構造ハッシュまたはオーバーサンプリングが変化しない純粋なモード切替ではクロスフェードがトリガーされない場合がある」ことを明記
- §3.6.2に「クロスフェード非発生時のEQ/コンボルバー内部スムージングによる即時切替」の安全性分析を追記

### 1.3 Conv→PEQモードの数値検証「ネット0dB」の記載が誤解を招く

**改訂版v2 §3.5.1 数値検証:**
> **Conv→PEQ** (irResidual=6, eqMax=9):
> - input = -max(0, 6-1.5) - max(0, 9-2.0) = -4.5 - 7.0 = -11.5
> - trim = 0
> - makeup = +11.5
> - 信号経過: 入力-11.5 → Conv+6 → EQ+9 → makeup+11.5 = +15.0 ✅（ネット0dB）

**問題:** +15.0 dBFS（0 dBFS を +15 dB 超過）という出力を「ネット0dB」とラベリングしている。これは正確ではない。

**正しい解釈:**
- `input + makeup = -11.5 + 11.5 = 0 dB`（入力トリムとメイクアップのネットは0 dB）
- しかし実際の信号経過では IR（+6 dB）と EQ（+9 dB）のゲインが加わるため、入力0 dBFS → 出力 +15 dBFS となる
- 「ネット0dB」は「入出力のトリム/メイクアップがネット0dB」という意味であり、**IR/EQゲインを含めたトータル出力が0 dBFSという意味ではない**

**修正案**: 数値検証のラベルを「✅（ネット0dB）」から「✅（input+makeup=0dB）」または「✅（IR+EQゲインを含めた最終出力は+15dBFS、リミッターで捕捉）」に変更する。数値式自体は正しい。

---

## 2. 🟡 中程度の問題

### 2.1 Q Surge Marginのオーバーシュート計算値が文献値と乖離

**改訂版v2 §3.3.1 注記（第2次検証§2.4）:**
> オーバーシュート量とQの関係は指数関数 `e^{-πζ/√(1-ζ²)}`（ζ = 1/(2Q)）であり線形ではない。0.15は保守的（安全側）に設定されている。

文献調査で、標準的な2次系ステップ応答オーバーシュート式から計算した正確な値:

| Q | ζ = 1/(2Q) | オーバーシュート% | dB換算 |
|---|-----------|------------------|--------|
| 0.707 (Butterworth) | 0.707 | ≈ 4.6% | ≈ +0.40 dB |
| 1.414 (=√2) | 0.354 | ≈ 30.5% | ≈ +2.31 dB |
| 4.0 | 0.125 | ≈ 67.3% | ≈ +4.47 dB |
| 10.0 | 0.050 | ≈ 85.4% | ≈ +5.37 dB |

> 出典: 制御理論の標準公式（参考文献: Kuo, *Automatic Control Systems*; Ogata, *Modern Control Engineering*）。Wikipedia "Q factor" でも ζ = 1/(2Q) の関係が確認されている。

**計画書の0.15係数との比較:**

| Q | gain | 計画のマージン | 実際のオーバーシュート(dB) | 差分 |
|---|------|-------------|------------------------|------|
| 4.0 | 12 dB | 12 × 0.15 × (4/0.707) = **10.2 dB** → 6.0 dB クリップ | ≈ 4.47 dB | +1.53 dB 過剰 |
| 2.0 | 6 dB | 6 × 0.15 × (2/0.707) = **2.55 dB** | ≈ 1.27 dB (Q=2, ζ=0.25, overshoot≈44%≈+1.6dB) | ≈ +0.95 dB 過剰 |

計画のマージンは実際のオーバーシュートより**保守的（安全側）**だが、6 dBクリップによりほとんどの実用ケースでクリップされる。Q=4, gain=12 dBのケースでは10.2 dB→クリップ6 dBとなり、実オーバーシュート 4.47 dBを約1.5 dB上回る。

**評価**: ヘッドルーム余裕として妥当。しかし「オーバーシュート量とQの関係は指数関数」との表記は正しいが、計画書の線形式 `gain * 0.15 * (Q / 0.707)` と指数関数 `e^{-π/(4Q√(1-1/(4Q²)))}` の乖離は数値で示されておらず、実装者に誤解を与える可能性がある。

### 2.2 クロスフェード中の新旧DSPゲイン値の記述に微妙な不整合

**改訂版v2 §3.6.5:**
> クロスフェード中、新旧DSPはそれぞれ異なる`ProcessingState`（異なるゲイン値）を使用する。これは**設計上正しい**動作である:
> - 旧DSP: 古いモードに適したゲイン値で処理 → クロスフェードバッファへ
> - 新DSP: 新しいモードに適したゲイン値で処理 → メインバッファへ

**コードベース検証結果（Item 4 Critical Finding）:**

実際のクロスフェード実装では:
- `fadingState` = `procState` の**コピー**（`AudioEngine.Processing.AudioBlock.cpp:370-372`）
- `procState` はそのブロック開始時にキャプチャされた**単一のスナップショット**（同ファイル:320）
- 旧DSPと新DSPは**同一の `procState` から派生した同一のゲイン値**で処理される

ただし、各DSPのRCU world（構築時点のゲインを凍結保持）と `procState`（現在のスナップショット）は異なる値を取りうる。**しかし `procState` と `fadingState` は実質的に同一である**ため、「異なるゲイン値」という主張は実装詳細のレベルでは正確ではない。

**評価**: 「異なるモードに適した値」という設計意図は正しいが、実装上は同一 `procState` から派生したゲイン値が両DSPに渡される。これ自体はゲインが `recomputeAutoGainStaging()` → setter → `submitRebuildIntent` → RCU公開の経路で既にターゲットモード用に更新されているため、**機能的には正しく動作する**。文書の記述を「両DSPは同一の最新ゲイン値を使用する（各DSPは自分の処理順序で適用）」と修正するのがより正確。

### 2.3 PEQ→Conv モード：trimが0ならconvInputTrimGainのガードによりスケーリング処理がスキップされることの未記載

**改訂版v2 §3.5.1 PEQ→Conv 式:**
> `trim = -max(0, irResidual - 2.0)`

irResidual ≤ 2.0 dB の場合、trim = 0 dB（gain = 1.0）。この時、DSPコードの:

```cpp
// AudioEngine.Processing.DSPCoreDouble.cpp:483
if (state.convolverInputTrimGain != 1.0)
```

ガードにより**スケールブロック処理がスキップされる**。これはパフォーマンス上の利点であるが、計画書に明記されていない。

**評価**: 軽微な最適化だが、実装上知っておくべき挙動。追記を推奨。

### 2.4 EQState::totalGainDb（既存）と maxGainDb（新規提案）の区別が不明瞭

**改訂版v2 §3.2.2:**
> `EQState` に `maxGainDb` フィールド追加（オプション）

コードベース確認: `EQState::totalGainDb`（`EQProcessor.h:281`）は**ユーザーが設定したEQ全体の出力ゲイン**であり、`computeMaxGainDb()` が計算する「EQカーブの最大ブースト量」とは**全く異なる概念**である。

計画書では両者の区別を明示していないため、実装時に `totalGainDb` を `maxGainDb` の代わりに使ってしまう誤りが発生するリスクがある。

**修正案**: §3.2.2に「`totalGainDb`（既存、ユーザー設定のEQ出力ゲイン）と `maxGainDb`（新規、`computeMaxGainDb()` が算出するEQカーブの最大ブースト量）は別概念である」ことを明記。

---

## 3. 🟢 確認済み・妥当な項目

### 3.1 複素応答のカスケード積算による伝達関数評価 ✅

文献調査（CCRMA Stanford, Julius O. Smith III "Series and Parallel Transfer Functions"、W3C Audio EQ Cookbook）で確認:
- カスケード接続の伝達関数は各段の積: H_total(z) = Π H_i(z)
- マグニチュードは乗算（dBでは加算）、位相は加算
- IEEE-754 double（53ビット仮数、約15.9桁の10進精度）では、20バンド×300周波数点の複素乗算で精度問題は発生しない

> 出典: https://ccrma.stanford.edu/~jos/filters/Series_Parallel_Transfer_Functions.html

### 3.2 `z = exp(+jω)` の標準DSP慣行との整合 ✅

コードベースの既存実装（`EQProcessor.Coefficients.cpp:327`: `z(std::cos(w), std::sin(w))`）と、CCRMA Smithの定義（`z = e^{jωT}`）が一致。改訂版v2の修正は正しい。

### 3.3 Conv→PEQ式の修正 ✅

コードベース確認（`AudioEngine.Processing.DSPCoreDouble.cpp:429-457`）:
- `ConvolverThenEQ` パスで `convolverInputTrimGain` は参照されない（29行にわたって出現なし）
- `EQThenConvolver` パスの lines 483-488 でのみ適用される

改訂版v2の Conv→PEQ 式 `trim = 0` はコード実装と完全に整合。

### 3.4 PEQ→Conv のtrim簡素化 ✅

`trim = -max(0, irResidual - 2.0)` は:
- `irResidual ≤ 2.0` → trim = 0（gain = 1.0、no-op）
- `irResidual > 2.0` → trim = 2.0 - irResidual ≤ 0

`setConvolverInputTrimDb` のクランプ範囲 `[-12, 0]` dB と完全互換。trimが正になるケースは存在しない。

### 3.5 `setProcessingOrder` の呼出順序修正 ✅

コードベース確認:
- Bypass系setter（`AudioEngine.Parameters.cpp:153-162, 164-173`）: publish → `applyDefaultsForCurrentMode()` → `submitRebuildIntent()` → `sendChangeMessage()`
- `setProcessingOrder`（同ファイル:268-275）: publish → `submitRebuildIntent()` → `applyDefaultsForCurrentMode()`（**逆順**）
- `applyDefaultsForCurrentMode()` 内部でも `submitRebuildIntent` が呼ばれ（line 339）、二重発行

改訂版v2の修正案（`submitRebuildIntent`削除、`applyDefaultsForCurrentMode` → `recomputeAutoGainStaging` → `sendChangeMessage`）はbypass系と統一され、技術的に正しい。

### 3.6 IRロード完了位置の修正 ✅

改訂版v2 §4で `AudioEngine.Cache.cpp` を削除し、`AudioEngine.UIEvents.cpp` を追加した修正はコード実装と一致する:
- IRロード完了: `ConvolverProcessor::applyComputedIR()`（`LoadPipeline.cpp:308`）
- → `AudioEngine::convolverParamsChanged()`（`UIEvents.cpp:36-195`）

### 3.7 Butterworth Q 閾値の修正（0.707） ✅

Q = 1/√2 ≈ 0.707 は Butterworth Q（Wikipedia "Q factor"、"Butterworth filter" で確認）。2次フィルタで最大平坦通過域特性を与えるQ値。ピーキングEQでこれを超えると共振ピークが発生する。理論的に正当な閾値。

### 3.8 Tukey窓テスト基準の修正 ✅

Harris 1978、Heinzel 2002（Max Planck Institute）および Wikipedia "Window function" で確認:
- Tukey α=0.1（10%テーパー）の第1サイドローブ: ≈ −15 dB
- 65536点FFTで10bin以上離れた位置: 矩形窓の遠方サイドローブロールオフ（≈ 6 dB/oct）により ≈ −30〜−40 dBが達成可能
- 改訂版v2のテスト基準「-40dB以下」は達成可能な現実値

> 出典: Harris, "On the Use of Windows for Harmonic Analysis with the Discrete Fourier Transform," Proc. IEEE 66(1):51-83, 1978. https://web.mit.edu/xiphmont/Public/windows.pdf

### 3.9 予測型静的マージン方式のノベルティ注記 ✅

文献調査で、FabFilter、iZotope、Waves等のAuto Gain機能はRMSベース（反応型）であり、EQカーブの事前解析に基づく入力ヘッドルーム自動調整を行う製品は**確認されなかった**。改訂版v2の注記は正確。

---

## 4. マージン定数の文献評価

改訂版v2 §3.5.1 の3つのマージン定数について、文献的裏付けを検証した:

| 定数 | 値 | 文献的根拠 |
|------|----|-----------|
| `kMarginEqFirst = 3.0 dB` | 3.0 | **なし**。特定のEQ段に3 dBマージンを推奨する文献は見つからなかった。ただしEBU R68のアライメントレベル（-18 dBFS）や一般的な2〜6 dBの段間マージンの範囲内。工学的には妥当。 |
| `kMarginConvFirst = 1.5 dB` | 1.5 | **なし**。多くのコンボリューション設計では3〜6 dBの入力マージンが推奨されるが、IR自体が既にL2正規化+safety margin（= −6 dB）されていることを考慮すると、追加1.5 dBは控えめだが妥当。 |
| `kMarginInterStage = 2.0 dB` | 2.0 | **なし**。段間トリムとして2 dBは標準的な慣行範囲内。 |

**結論**: 3つのマージン定数はいずれも文献的根拠がなく、工学的判断（ヒューリスティック）である。Phase 8の「予測値と実測ピークの一致性評価」で実測調整を行う設計は正しい。

---

## 5. 修正Conv→PEQ式の全数値検証テーブル

| irResid | eqMax | input | trim | makeup | net trim+makeup | 最悪出力(dBFS) | 備考 |
|---------|-------|-------|------|--------|----------------|--------------|------|
| 0 | 0 | 0 | 0 | 0 | 0 | 0 | 保護不要 |
| 0 | 6 | -4.0 | 0 | +4.0 | 0 | +6.0 | ✅ |
| 6 | 0 | -4.5 | 0 | +4.5 | 0 | +6.0 | ✅ |
| 6 | 9 | -11.5 | 0 | +11.5 | 0 | +15.0 | makeup12以内 ✅ |
| 3 | 9 | -1.5-7=-8.5 | 0 | +8.5 | 0 | +12.0 | ✅ |
| 15 | 3 | -13.5-1=-14.5→-12 | 0 | +12 | -1.5 | +16.5 | inputクランプ ⚠️ |
| 0 | 15 | -13→-12 | 0 | +12 | -1.0 | +15.0 | inputクランプ ⚠️ |
| 6 | 15 | -4.5-13=-17.5→-12 | 0 | +12 | -5.5 | +21.0 | input+makeup不足 ⚠️ |

**分析**: inputが-12 dBにクランプされる極端ケース（irResidual ≥ 13.5 dB または eqMax ≥ 14 dB）では、makeup+12 dBでも補填不足となり、ネット減衰が発生する。これは保護過剰による信号減衰であり、安全側の挙動。Phase 8テストで極端IR/EQの検証が必要。

---

## 6. クロスフェード機構の詳細分析

### 6.1 新旧DSPのゲイン値ソース

```
captureAudioThreadParameterSnapshot(world) (AudioEngine.h:3454-3509):
  → world->automation.inputHeadroomGain  (line 3469)
  → world->automation.outputMakeupGain  (line 3470)
  → world->automation.convolverInputTrimGain (line 3471)

buildAudioThreadProcessingState(dsp, snapshot) (AudioEngine.h:3511-3549):
  → .inputHeadroomGain = snapshot.inputHeadroomGain (line 3532)
  → .outputMakeupGain = snapshot.outputMakeupGain (line 3533)
  → .convolverInputTrimGain = snapshot.convolverInputTrimGain (line 3534)
```

world の automation フィールドは world 構築時に atomics からキャプチャされる（`RuntimeBuilder.cpp:318-330`）。

### 6.2 クロスフェード時の procState 共有

```cpp
// AudioEngine.Processing.AudioBlock.cpp:370-372
auto fadingState = procState;  // procState のコピー
fadingState.analyzerEnabled = false;
```

旧DSP処理 (`fading->processToBuffer(...)`) と新DSP処理 (`dsp->process(...)`) は**同一の procState/fadingState のゲイン値**を使用する。

### 6.3 モード切替の完全なデータフロー

```
[Message Thread]
  user changes mode (e.g., setProcessingOrder)
    → recomputeAutoGainStaging()
      → setInputHeadroomDb(-11.5) → atomic store → submitRebuildIntent
      → setConvolverInputTrimDb(0) → atomic store → submitRebuildIntent
      → setOutputMakeupDb(+11.5) → atomic store → submitRebuildIntent
    → submitRebuildIntent (via applyDefaultsForCurrentMode or setter)

[Rebuild Thread]
  → rebuilds DSPCore → calls RuntimePublicationOrchestrator::trySubmit()
    → RuntimeBuilder reads atomics → world->automation.*Gain = current atomics (line 318-330)
    → CrossfadeAuthority::evaluate() → needsCrossfade?
    → if structuralHash changed (IR dependency) → yes, crossfade
    → if only order/bypass changed → MAY NOT trigger crossfade

[Publication]
  → RuntimePublicationOrchestrator publishes new world (atomic swap)
  → if crossfade: CrossfadeRuntime starts LinearRamp new gain 0→1

[Audio Thread]
  → captureAudioThreadParameterSnapshot(world) → gains from world->automation.*
  → buildAudioThreadProcessingState() → ProcessingState
  → if crossfade: dual processing + runLatencyAlignedCrossfadeMixLoop
  → if not crossfade: single DSP processing with new ProcessingState
```

---

## 7. 関連数値データ（文献参照）

| 項目 | 値 | 出典 |
|------|----|------|
| IEEE-754 double 仮数精度 | 53 bit (≈ 15.9 decimal digits) | IEEE 754-2008 |
| EBU R128 true peak ceiling | −1 dBTP | EBU R128, ITU-R BS.1770-4 |
| Spotify/Apple Music/YouTube target | −1 dBTP | Wikipedia "Loudness war" |
| EBU R68 alignment level | −18 dBFS | EBU R68 |
| ITU-R BS.1770-4 true peak OS | 4× | ITU-R BS.1770-4 |
| Tukey α=0.1 1st sidelobe | ≈ −15 dB | Harris 1978, Heinzel 2002 |
| Butterworth Q | 1/√2 ≈ 0.707 | Wikipedia "Q factor", "Butterworth filter" |
| 2次系 step overshoot formula | exp(−πζ/√(1−ζ²)) | Standard control theory |
| リバーブIR crest factor (typical) | 12−25 dB (peak-to-RMS) | Wikipedia "Crest factor", empirical |
| プロ音楽のcrest factor | 12−18 dB (processed), 18−20 dB (unprocessed) | Wikipedia "Crest factor" |

---

## 8. 保留中の判断（文献未確認の設計値）

以下の6項目は、文献調査で**特定の値に対する権威的な裏付けが見つからなかった**:

| 項目 | 計画書の値 | 評価 |
|------|-----------|------|
| `kMaxEffectivePeak = 0.98` | ≈ −0.17 dBFS | IRピーク余裕として**非常にタイト**。多くの設計で −3〜−6 dBFS 推奨 |
| `kMaxEffectiveRms = 0.25` | ≈ −12 dBFS | IRのRMS目標として保守的で妥当 |
| `kMarginEqFirst = 3.0 dB` | EQ段のマージン | 工学的に妥当だが文献未確認 |
| `kMarginConvFirst = 1.5 dB` | Conv段のマージン | 文献未確認。IR正規化済みなら許容範囲 |
| `kMarginInterStage = 2.0 dB` | 段間マージン | 工学的標準範囲内 |
| Q Surge Margin 0.15係数 | 線形ヒューリスティック | 理論的根拠なし、保守的（安全側） |

これらの値は Phase 8 の実測検証で妥当性を確認する必要がある。

---

## 9. 総合評価

### 9.1 改訂版v2の改善点（v2からの）

v2で発見された9点の重大・中程度問題のうち:
- ✅ **7点が完全に解消**: Conv→PEQ式、z=exp(+jω)、Q閾値、Tukey窓テスト基準、IRロード完了位置、inputクランプ依存、setProcessingOrder順序
- ✅ **1点が部分的に文書化**: 予測型方式のノベルティ、0.15係数のヒューリスティック性
- ✅ **新たな設計要素が追加**: 実行時モード切替安全設計（Phase 6）

### 9.2 新たに発見された問題

| # | 問題 | 重大度 | 要点 |
|---|------|--------|------|
| 1 | 「等パワーブレンド」の実装不一致 | 🔴 重大 | 実コードはリニアクロスフェード。記述修正必要 |
| 2 | 純粋モード切替でクロスフェード不発生の可能性 | 🔴 重大 | CrossfadeAuthority が order/bypass 変化を直接検出しない。文書にリスク明記必要 |
| 3 | 数値検証「ネット0dB」ラベルの誤解誘発性 | 🔴 重大 | +15 dBFS を「ネット0dB」と記載するのは不正確 |
| 4 | クロスフェード中procState/fadingStateのゲイン同一性の不正確記述 | 🟡 中 | 「異なるゲイン」→ 実装上は同一procStateから派生 |
| 5 | trim = 0時のガードスキップ最適化の未記載 | 🟡 中 | パフォーマンス上の重要挙動 |
| 6 | totalGainDb vs maxGainDb の区別未明示 | 🟡 中 | 実装時の誤用リスク |
| 7 | Q Surge Margin: 線形式と指数関数の数値乖離の未表示 | 🟡 中 | 文献値比較が必要 |

### 9.3 結論

改訂版v2は、**前2回の検証で指摘された技術的欠陥をほぼすべて解消しており、実装計画として高い完成度**に達している。3つの重大問題も**いずれも文書記述の不正確さ**であり、計算式や設計ロジック自体に欠陥はない。

- **問題1（等パワーブレンド）**: 「等パワー」→「リニア」に文言修正のみ。ロジック変更不要
- **問題2（クロスフェード不発生）**: 既存コードの仕様を文書に正確反映するのみ。追加対応は将来課題
- **問題3（ネット0dB）**: 数値検証ラベル修正のみ。計算式は正しい

中程度4件もすべて文書記述の精緻化で対応可能であり、設計変更を必要としない。

**本計画書は、文書記述の修正を行えば実装開始可能な完成度にある。**

---

## 10. 引用文献・参考資料

1. Julius O. Smith III, *Introduction to Digital Filters with Audio Applications*, CCRMA, Stanford University — https://ccrma.stanford.edu/~jos/filters/ (Series and Parallel Transfer Functions, Peaking Equalizers)
2. W3C Audio EQ Cookbook (Robert Bristow-Johnson) — https://www.w3.org/TR/audio-eq-cookbook/
3. Wikipedia — Digital biquad filter — https://en.wikipedia.org/wiki/Digital_biquad_filter
4. Wikipedia — Q factor — https://en.wikipedia.org/wiki/Q_factor
5. Wikipedia — Butterworth filter — https://en.wikipedia.org/wiki/Butterworth_filter
6. Wikipedia — Window function (Tukey window, Harris 1978) — https://en.wikipedia.org/wiki/Window_function
7. Harris, F.J., "On the Use of Windows for Harmonic Analysis with the Discrete Fourier Transform," *Proc. IEEE* 66(1):51-83, 1978 — https://web.mit.edu/xiphmont/Public/windows.pdf
8. Heinzel, G., Rüdiger, A., Schilling, R., "Spectrum and spectral density estimation by the Discrete Fourier transform," Max Planck Institute, 2002 — https://edoc.mpg.de/395068
9. Wikipedia — EBU R 128 — https://en.wikipedia.org/wiki/EBU_R128
10. ITU-R BS.1770-4 — Algorithms to measure audio programme loudness and true-peak audio level — https://www.itu.int/rec/R-REC-BS.1770
11. EBU R 128 — Loudness normalisation and permitted maximum level — https://tech.ebu.ch/publications/r128/
12. Wikipedia — Fade (audio engineering) — https://en.wikipedia.org/wiki/Fade_(audio_engineering)
13. Wikipedia — Headroom (audio signal processing) — https://en.wikipedia.org/wiki/Headroom_(audio_signal_processing)
14. Wikipedia — Crest factor — https://en.wikipedia.org/wiki/Crest_factor
15. Wikipedia — Loudness war (streaming platform targets) — https://en.wikipedia.org/wiki/Loudness_war