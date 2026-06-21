# bug_review3.md 検証レポート v2（修正版）

- **作成日**: 2026-06-21
- **対象**: `doc/work52/bug_review3.md`
- **使用ツール**: grep/Select-String, CodeGraph MCP, AiDex MCP, semble CLI
- **注意**: 本レポートは仮説の提示であり、「確定原因」と断定するものではない。各主張の根拠・反証を併記する。

---

## 1. 信号経路の実コードトレース（最重要検証）

### 1.1 processDouble() → processOutputDouble() 完全トレース

`src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp` の実コードに基づく。

```
[1] processInputDouble()       line 323  → buffer → alignedL/alignedR (inputHeadroomGain倍)
[2] EQ + Convolver 処理       line 373-393 → processBlock (= alignedL/alignedR)
[3] outputFilter (DC+LPF)      line 409-420 → processBlock
[4] ★ outputMakeupGain適用    line 435-439 → alignedL/alignedR ★
[5] SoftClip                   line 449-458 → alignedL/alignedR
[6] Bypass blend + OS down     line 460-504 → alignedL/alignedR
[7] processOutputDouble()      line 513
  [7a] DC Blocker Out          line 551     → alignedL/alignedR
  [7b] NaN/Inf除去              line 553-577 → alignedL/alignedR
  [7c] NoiseShaper             line 596-619 → alignedL/alignedR (kOutputHeadroom=0.891)
  [7d] Clamp ±Headroom         line 621-645 → alignedL/alignedR
  [7e] copy to buffer          line 650-651 → alignedL/alignedR → buffer
```

### 1.2 出力メイクアップゲインの位置（確定）

**【確定】outputMakeupGain (+12dB) は NoiseShaper の前に適用される。**

- `processDouble()` line 439: `scaleBlockFallback(ptr, numSamples, state.outputMakeupGain)`
  - この時点で `alignedL/alignedR` の値が `outputMakeupGain` 倍される
- `processOutputDouble()` line 596-613: `adaptiveNoiseShaper.processStereoBlock(dataL, dataR, ...)`
  - 同じ `alignedL/alignedR` を NoiseShaper が処理する
- 間に他のゲイン変更処理は存在しない

したがって **「NoiseShaper直前に+12dBが存在する」はコード上確定事実** である。

### 1.3 SoftClipとの関係

SoftClip (line 449-458) は outputMakeupGain 適用 (line 439) の**後**、NoiseShaper (line 596-619) の**前**に位置する。

```
aligned → [Makeup +12dB] → [SoftClip] → [Bypass/OS down] → [DC Blocker out] → [NoiseShaper ×0.891]
```

SoftClip は +12dB で増幅された信号に対してクリッピングを行うため、SoftClip の threshold (0.95 at sat=0) を超えて常時クリッピングしている可能性がある。SoftClip後の信号は高調波歪みを含み、これが NoiseShaper に入力される。

---

## 2. 各レビュー主張の検証

### 2.1 「約7倍」計算の検証

**【指摘】単純な乗算ではDSP的に不正確**

```
Net gain = outputMakeupGain / inputHeadroomGain × kOutputHeadroom
         = 3.98 / 0.5 × 0.891
         = 7.09
```

この計算は以下の要因を無視しているため、実際の NoiseShaper 入力比は異なる可能性がある：

| 要因 | 影響 |
|------|------|
| SoftClip | +12dB入力で常時クリッピング → ピークは制限されるがRMSは高いまま |
| オーバーサンプリング | OS処理でピークが変化する可能性 |
| EQ/Convolver の周波数特性 | 特定帯域でのゲイン変化 |
| Dry/Wet ブレンド | bypassFadeGain によるレベル変化 |

**結論**: 単純な「7倍」は誤解を与える。正確には **「+12dBのゲインがNoiseShaper直前に適用される」という事実のみが確実**。実際のNoiseShaper入力レベルは信号・設定依存であり、実測が必要。

### 2.2 プリリンギング説 — ★☆☆☆☆ 可能性極低

**根拠**: 低音持続音でプリリンギングが発生する説明が困難。低音信号には急峻なトランジェントが存在しないため、プリリンギングは発生しない。

### 2.3 Partition境界グリッチ説 — ★★☆☆☆ 可能性低い

**根拠**: `ringRead()` でのゼロ埋めは確認されたが、低音依存性の説明が弱い。リングオーバーフローは信号の振幅よりレイテンシ/バッファサイズに依存する。

### 2.4 IR由来DC/超低域説 — ★★★★☆ 可能性高い

**確認結果**:

| チェック項目 | 結果 |
|------------|------|
| IR読み込み時のDC Blocker | ✅ `UltraHighRateDCBlocker` 適用（LoaderThread.cpp line 595-598） |
| 出力DC Blocker | ✅ `dc.outputL.processStereo()` 存在（processOutputDouble.cpp line 551） |
| DC完全除去の保証 | ❌ FloatVectorOperationsの丸め誤差レベルのDCが残留する可能性あり |
| DC→Lattice相互作用 | ✅ 微小DCでも Lattice NoiseShaper の積分器（state値）に蓄積され得る |

**低音依存性との整合**: 低音はDC/超低域と周波数が近いため、信号とDCの分離が困難。低音が大きいほど DC Blocker の過渡応答が長くなり、残留DCが増える可能性がある。

### 2.5 NoiseShaper共通経路の飽和 — ★★★★☆ 可能性高い

**Fixed4Tapでも発生** というユーザー報告から、Adaptive Lattice 固有の問題（P7）ではないことが明確。

Fixed4Tap のコード:

```cpp
// FixedNoiseShaper.h
const double y = x - fb;
const double yq = quantize(y, rng);
const double error = yq - y;
const double clampedError = std::clamp(error, -2.0 * scale, 2.0 * scale);
```

- +12dB makeup gain 後、SoftClip を通った信号が NoiseShaper に入力
- 量子化誤差 `error` が `±2*scale` の clamp に頻繁にヒット
- clamp による非線形性 → 高調波歪み → 「ジジジジ」

**全ての NoiseShaper に共通する要因** である点でゲイン説と整合。

---

## 3. 切り分けに必要なテスト（確定）

以下の3テストで原因を高精度に切り分け可能：

### テスト1: Output Makeup +12dB → 0dB

```cpp
// AudioEngine.Parameters.cpp line 326 を一時的に変更
newOutputMakeupDb = 12.0f; → newOutputMakeupDb = 0.0f;
```

**予測**:

- ノイズ消滅 → ゲインステージングが主因（最も可能性高い）
- ノイズ不変 → ゲイン以外の要因（DC/IR品質など）

### テスト2: NoiseShaper OFF

`applyDither` が `false` になる条件（`ditherBitDepth <= 0`）にする。
または一時的に `if (applyDither)` → `if (false)` に変更。

**予測**:

- ノイズ消滅 → NoiseShaper 共通経路が原因
- ノイズ不変 → NoiseShaper 以前の段が原因

### テスト3: 20Hz HPF 挿入

Convolver→EQ の経路のどこかに 20Hz ハイパスフィルタを挿入。

**予測**:

- ノイズ低減 → DC/超低域成分が原因
- ノイズ不変 → 低域以外の要因

---

## 4. 総合評価

| 仮説 | 確度 | 根拠 | 反証 |
|:----:|:----:|------|------|
| **ゲインステージング** | **75%** | +12dBがNoiseShaper直前にある（コード確定）。Fixed4Tapでも発生と整合 | 「7倍」計算は単純化。SoftClipによるクリッピングを経由するため実効ゲインは未知数 |
| **DC/超低域** | **40%** | 低音依存性と整合。Latticeの積分器がDCに敏感 | IR読み込み時にDC Blocker適用済み。出力DC Blockerも存在 |
| **Convolverピーク** | **35%** | IRによっては6-15dBのピーク増幅が起き得る | Conv→EQモードに限定されずEQ単独でも起き得る |
| **リングオーバーフロー** | **20%** | ゼロ埋めはNoiseShaperにインパルス刺激 | 低音依存性の説明が困難 |
| **プリリンギング** | **5%** | — | 低音持続音での発生メカニズムが説明不能 |

### 最も確からしいシナリオ

```
Conv→EQ モード
  → inputHeadroom -6dB (0.5x)
  → Convolver (IRによってはピーク増幅)
  → EQ (低域ブースト)
  → outputMakeup +12dB (3.98x)  ← ★ここで信号が増大
  → SoftClip (クリッピング発生→高調波歪み)
  → NoiseShaper (過大入力→内部状態不安定化→「ジジジジ」)
```

SoftClip が +12dB 増幅された信号をクリッピングすることで高調波を生成し、これが NoiseShaper の入力となる。NoiseShaper（Lattice/Fixed4Tap 両方）はこの高調波を含む信号を処理できず、発振的挙動を示す。

---

## 5. 推奨される改修方針

### 優先: outputMakeupGain の NoiseShaper 後への移動（検証が必要）

現状:

```
... → [Makeup +12dB] → [SoftClip] → ... → [NoiseShaper ×0.891]
```

変更案:

```
... → [SoftClip] → ... → [NoiseShaper ×0.891] → [Makeup +12dB] → [output]
```

**これにより**:

- NoiseShaper への入力を適正レベルに保てる
- 出力レベルは維持される
- ただし signal path の変更による副作用（SoftClipの効き方変化等）の検証が必要

### 参考: リングオーバーフロー診断

`m_ringOverflowCount` の値をログ出力することで、Conv のリングバッファが溢れているか確認可能。
