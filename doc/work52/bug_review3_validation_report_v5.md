# bug_review3.md 検証レポート v5（修正版）

- **作成日**: 2026-06-21
- **対象**: `doc/work52/bug_review3.md`
- **使用ツール**: grep/Select-String, CodeGraph MCP, AiDex MCP, semble CLI

---

## 0. はじめに

本レポートは v1〜v4 の誤りを修正した確定版である。v4 の「post-NS ハードクランプ主犯説」はユーザー評価により否定された。本レポートでは「確実に言えること」と「仮説」を明確に区別する。

---

## 1. 確実に言えること（コード確定事実）

### 1.1 Conv→Peq のゲイン設定差

| モード | inputHeadroomDb | outputMakeupDb | convTrimDb | 正味ゲイン差 |
|:-----:|:--------------:|:--------------:|:----------:|:----------:|
| PEQ-only | 0dB | 0dB | 0dB | **0dB** |
| EQ→Conv | 0dB | +10dB | -6dB | +4dB |
| **Conv→Peq** | **-6dB** | **+12dB** | **0dB** | **+6dB** |

**(確定)** `AudioEngine.Parameters.cpp` `applyDefaultsForCurrentMode()` line 325-326。

### 1.2 +12dB outputMakeupGain の適用位置

```
processDouble() [line 439]:
  scaleBlockFallback(ptr, numSamples, state.outputMakeupGain)  ← +12dB on alignedL/R

processOutputDouble() [lines 596-619]:
  NoiseShaper.processStereoBlock(..., kOutputHeadroom)          ← ×0.891 on same alignedL/R
```

**(確定)** outputMakeupGain (+12dB) は NoiseShaper より前に適用される。コード上疑いの余地なし。

### 1.3 kOutputHeadroom は NS 内部で使用される

```cpp
constexpr double kOutputHeadroom = 0.8912509381337456;  // ≈ -1dB
```

| 使用箇所 | 用途 |
|---------|------|
| NS呼び出し (line 609-615) | `processStereoBlock(..., kOutputHeadroom)` の引数 |
| 非NS時 (line 621-623) | `dataL[i] *= kOutputHeadroom` の乗算 |
| post-NS clamp (line 658-680) | `jlimit(-kOutputHeadroom, kOutputHeadroom, ...)` |

**(確定)** kOutputHeadroom は単なる clamp 閾値ではなく、NoiseShaper の **動作基準値** として設計されている。post-NS clamp は最終安全柵。

### 1.4 SoftClip は sat=0 でも完全バイパスされない

```cpp
// DSPCoreDouble.cpp line 442
if (state.softClipEnabled)  // 有効/無効の判定は softClipEnabled フラグ
{
    softClipBlockAVX2(data, ..., clipThreshold, clipKnee, ...);
    // sat=0 でも関数は実行される（threshold=0.95, knee=0.05）
}
```

```cpp
// sat=0 のパラメータ
clipThreshold = 0.95 - 0.45 * 0 = 0.95
clipKnee      = 0.05 + 0.35 * 0 = 0.05
clipAsymmetry = 0.10 * 0 = 0.0
```

**(確定)** sat=0 では threshold=0.95, knee=0.05 の gentle な SoftClip が動作する。**完全バイパスではない。**

---

## 2. v4 の誤りとその修正

### 2.1 「post-NS clamp が直接原因」 ← 誤り

| v4 の主張 | 反証 |
|----------|------|
| 「全NSで発生→共通後段のclampが犯人」 | 全NSに共通なのは clamp だけでなく、Convolver/EQ/OutputFilter/Makeup/SoftClip/OS Down/DCBlocker の**全て**。共通であることと原因であることは無関係 |
| 「clampが歪みを生む」 | 症状「ジジジジ」は発振・状態暴走に近く、-1dBのハードクリップで生じる「バリバリ/ガリガリ」とは異なる |
| 「計算上clamp通過がぎりぎり」 | 自身の計算で「SoftClip有効時は 0.846 < 0.891 で通過」と示している。clampに当たっていない |

### 2.2 v4 の「kOutputHeadroom=1.0変更案」 ← 危険・非推奨

kOutputHeadroom は NoiseShaper 群の動作基準値として内部で使用されている。単純な変更は量子化スケール・ディザ量・ノイズシェイプ量に影響する。

---

## 3. 未確定事項の棚卸し

| # | 項目 | 現状 | 確定に必要な調査 |
|:-:|------|:----:|----------------|
| U1 | post-NS clamp の実際の発動率 | **未知** | 実機で clamp ヒット回数をカウント |
| U2 | Convolver IR の出力ピーク増幅率 | **未知** | IRファイル毎の peak/RMS 測定 |
| U3 | +12dB makeup 有無でのNS入力レベルの差 | **推定のみ** | `measureLevel()` でログ取得 |
| U4 | OS Downsampler 内部状態の影響 | **仮説** | OS=1x 固定テスト |
| U5 | SoftClip 有無での症状変化 | **仮説** | `softClipEnabled=false` テスト |

---

## 4. 現時点で確率的に最も高い原因

### 4.1 確率評価

| 原因 | 確率 | 根拠 |
|:----:|:----:|------|
| **①Conv→Peqゲイン構造** | **50%** | +12dBの存在は確定。PEQ-onlyとの差も確定 |
| **②NS入力過大（①の結果）** | **30%** | 全NSで発生する事実と整合。NS内部状態が過大信号で不安定化 |
| **③Convolver出力ピーク増幅** | **10%** | IR次第。低音IRで crest factor 増大し得る |
| **④OS Downsampler状態** | **5%** | OS=1xテストで検証可能 |
| **⑤SoftClip副作用** | **3%** | sat=0でも処理は実行されるが、影響度は低い |
| **⑥DC/超低域** | **1%** | 二重DC Blockerあり |
| **⑦post-NS clamp** | **1%** | 発動率未確認。症状の質とも不一致 |

### 4.2 最有力シナリオ

```
Conv→Peq モード設定
  ↓
+12dB outputMakeupGain (確定)
  ↓
NoiseShaper 入力信号が PEQ-only より過大 (高い確率)
  ↓
NoiseShaper 内部状態（Lattice/IIRのstate値、またはFixedのerror buffer）が
入力信号の高エネルギーで不安定化 (推定)
  ↓
「ジジジジ」発振的ノイズ (報告症状と一致)
```

**「全 NoiseShaper で発生」の説明**: それぞれ異なるアルゴリズムでも、**過大入力によって内部状態が不安定化する**点は共通。

- Lattice: `kLatticeStateLimit=2.0` に頻繁にヒット → 非線形性
- Fixed4Tap: `clampedError` が `±2*scale` に頻繁にヒット → 非線形性
- Psychoacoustic: 12-tap IIR の state 飽和

---

## 5. 推奨切り分けテスト（優先順）

### テスト1: outputMakeup 0dB（最も重要）

```cpp
// AudioEngine.Parameters.cpp line 326
newOutputMakeupDb = 12.0f; → newOutputMakeupDb = 0.0f;
```

**消える→ゲイン構造が主因。消えない→ゲイン以外の要因。**

### テスト2: NoiseShaper完全OFF

`ditherBitDepth <= 0` 相当の状態（applyDither=false）にする。

**消える→NS内部状態が主因。消えない→NSより前段が主因。**

### テスト3: NS入力レベル測定

`processOutputDouble()` の NS 呼び出し直前で `measureLevel()` またはピーク値をログ出力。PEQ-only と Conv→Peq で比較。

### テスト4: post-NS clamp 発動率測定

```cpp
// processOutputDouble() の clamp 前に挿入（診断用）
static std::atomic<int> clampCount{0};
for (int i = 0; i < numSamples; ++i) {
    if (std::abs(dataL[i]) > kOutputHeadroom) clampCount++;
}
```

1ブロックあたりの clamp 発動サンプル数をカウント。

### テスト5: OS=1x固定

```cpp
manualOversamplingFactor = 1;
```

### テスト6: SoftClip完全OFF

```cpp
softClipEnabled = false;
```

---

## 6. 結論

| 項目 | 評価 |
|------|:----:|
| +12dB makeup の NoiseShaper 前存在 | ✅ 確定 |
| 全NSで発生 | ✅ 確定（バグ報告） |
| post-NS clamp が直接原因 | ❌ 否定（v4の誤り） |
| kOutputHeadroom=1.0変更 | ❌ 危険・非推奨 |
| **最有力: ゲイン構造→NS入力過大→内部状態不安定化** | **推定（テストで検証可能）** |
| post-NS clamp 発動率 | **未測定（最重要データ不足）** |
| OS Downsampler 関与 | **未検証（仮説）** |
