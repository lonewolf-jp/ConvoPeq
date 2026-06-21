# bug_review3.md 検証レポート v3（修正版）

- **作成日**: 2026-06-21
- **対象**: `doc/work52/bug_review3.md` — Conv→Peq 限定「ジジジジ」ノイズ
- **使用ツール**: grep/Select-String, CodeGraph MCP, AiDex MCP, semble CLI

---

## 1. 信号経路完全トレース（確定）

### processDouble() → processOutputDouble() の実コード

```
▼ processDouble() [DSPCoreDouble.cpp:309]
  [1] processInputDouble(buffer, inputHeadroomGain)  ← buffer→alignedL/R (-6dB)
  [2] EQ + Convolver 処理                          ← processBlock(=alignedL/R)
  [3] outputFilter (DC Blocker + LPF)               ← processBlock
  [4] ★ scaleBlockFallback(outputMakeupGain) ← +12dB (3.98x) ON alignedL/R ★
  [5] SoftClip (if state.softClipEnabled)            ← ON alignedL/R
  [6] Bypass blend + Oversampling Down              ← ON alignedL/R
  [7] processOutputDouble(buffer)                    ← 同じalignedL/R

▼ processOutputDouble() [DSPCoreDouble.cpp:530]
  [8]  dc.outputL/R.processStereo (DC Blocker)       ← ON alignedL/R
  [9]  NaN/Inf removal                               ← ON alignedL/R
  [10] NoiseShaper (×kOutputHeadroom=0.891)          ← ON alignedL/R
  [11] Clamp ±kOutputHeadroom                        ← ON alignedL/R
  [12] alignedL/R → buffer コピー
```

**【確定】outputMakeupGain (+12dB) は NoiseShaper より前に位置する。**

- line 439: `scaleBlockFallback(ptr, numSamples, state.outputMakeupGain)` → alignedL/R
- line 596-613: `adaptiveNoiseShaper.processStereoBlock(dataL, dataR, ...)` → 同じ alignedL/R
- 間に他のゲイン調整は存在しない

---

## 2. Oversampling Downsampler の検証（v2の最大の見落とし）

### 2.1 DSPCore での OS 処理順

```cpp
// DSPCoreDouble.cpp line 346-348 (processUp)
if (oversamplingFactor > 1)
    processBlock = oversampling.processUp(originalBlock, ...);

// ... EQ+Conv+Makeup+SoftClip on upsampled block ...

// line 480-485 (processDown)
if (oversamplingFactor > 1)
    oversampling.processDown(processBlock, originalBlock, ...);
```

**OS有効時の全体フロー**:

```
[processUp] → [EQ+Conv] → [Makeup +12dB] → [SoftClip] → [processDown] → [DC Blocker] → [NoiseShaper]
```

### 2.2 Downsampler の内部状態の調査

`CustomInputOversampler` は各ステージの `downHistory[channel]` を内部状態として保持する。

```cpp
// decimateStage() [CustomInputOversampler.cpp:523]
double* history = stage.downHistory[channel].get();
// ... FIR畳み込み： history + keep に新しい入力を追加し、decimate ...
```

**状態のライフサイクル**:

| 操作 | 状態の扱い |
|------|-----------|
| `prepare()` | 全ステージの history を `clear` |
| `reset()` | 全ステージの history を `clear` |
| 通常のブロック処理 | history はブロック間で継続（正しい動作） |
| `clearAllStages()` | corruption 検出時に clear |

### 2.3 Downsampler と低域相互作用の分析

**問題のメカニズム**（仮説）:

1. 低域（40-80Hz）の1周期は 12.5-25ms
2. 典型的なブロックサイズ（512 samples @ 48kHz = 10.6ms）は低域の半周期未満
3. +12dB makeup gain で増幅された Low Shelf 出力が SoftClip を通り、波形が非対称にクリップ
4. クリップされた波形のブロック境界での不連続性が、Downsampler の FIR フィルタ（Kaiser窓 sinc）に「インパルス状の刺激」として入力
5. Downsampler の内部 history に異常値が蓄積 → 次のブロックの decimate 出力に歪み
6. この歪みが NoiseShaper（Lattice / Fixed4Tap 両方）で増幅 → 「ジジジジ」

**この仮説の強み**: 低域のみで発生する理由を説明できる（低域ほど波形のブロック境界不連続が急峻になる）。

**この仮説の弱み**: OS=1x（OS OFF）でも症状が再現するか未確認。OS=1xなら downsampler は完全にバイパスされる。

### 2.4 確認すべきポイント

```cpp
// OS=1xの場合、processDownは呼ばれない
// DSPCoreDouble.cpp line 459-477:
if (oversamplingFactor == 1 && ...) { bypass blend }
// ...
if (oversamplingFactor > 1) {
    oversampling.processDown(...)  // OS>1でのみ呼ばれる
}
```

**OS=1x 固定で症状が...**

- 消える → Downsampler が原因の一部
- 残る → Downsampler は無関係

---

## 3. SoftClip 依存性の検証

### 3.1 SoftClip の有効条件

```cpp
// DSPCoreDouble.cpp line 442
if (state.softClipEnabled)  // ← 条件付き
{
    softClipBlockAVX2(data, numProcSamples, clipThreshold, clipKnee, clipAsymmetry, ...);
}
```

`softClipEnabled` のデフォルト値:

```cpp
// AudioEngine.h line 1740
std::atomic<bool> softClipEnabled { true };  // デフォルト有効
```

`saturationAmount` のデフォルト値:

```cpp
// AudioEngine.h line 1744
std::atomic<float> saturationAmount { 0.1f };  // 10%
```

**SoftClip が有効でも sat=0.1 での影響**:

```cpp
clipThreshold = 0.95 - 0.45 * sat = 0.95 - 0.045 = 0.905
clipKnee      = 0.05 + 0.35 * sat = 0.05 + 0.035 = 0.085
```

→ SoftClip は 0.905 以上の振幅からのみクリッピングを開始。Knee幅は 0.085。
+12dB makeup gain (=3.98x) がかかった信号では、元信号が 0.227 (=0.905/3.98) を超えると SoftClip が効き始める。

**判定**: SoftClip の影響は `saturationAmount` と信号振幅に強く依存する。

- `saturationAmount = 0` なら SoftClip は「threshold = 0.95, knee = 0.05」の gentle なリミッター
- sat=0 でも +12dB 増幅後のピークが 0.95 を超える領域ではクリッピングが発生
- **SoftClip が犯人かどうかは `saturationAmount` 設定と入力信号レベルに依存**

**推奨テスト4**: `softClipEnabled = false`（または `saturationAmount = 0.0`）でノイズが消えるか確認。

---

## 4. 各仮説の確度再評価

| 原因 | 確率 | 根拠 | 反証/不確定要素 |
|:----:|:----:|------|---------------|
| **①ゲイン構造** | **45%** | +12dBがNoiseShaper直前（コード確定）。Fixed4Tapでも発生と整合 | SoftClip非有効時の影響度は未確認 |
| **②NS共通飽和** | **25%** | Fixed4Tapでも発生。全NSが+12dB増幅信号を処理 | NS内部で `±2*scale` clamp はあるが異常値の証拠なし |
| **③OS Downsampler** | **15%** | 低域依存性と整合。ブロック境界不連続→FIR履歴汚染の可能性 | OS=1xでの挙動未確認。推測の域 |
| **④Convピーク増幅** | **10%** | IRによる6-15dBのピーク増幅は現実的 | PEQ単独でもIR経由しないと再現せず |
| **⑤DC/超低域** | **5%** | — | 二重のDC Blockerあり。残留DCは極小 |
| **⑥Partition境界** | **3%** | — | 低音依存性の説明困難 |
| **⑦プリリンギング** | **<1%** | — | 低音持続音では発生しない |

---

## 5. 3者相互作用モデル（新規）

```
低音入力
  │
  ▼
Convolver (IRの低域増幅)
  │
  ▼
EQ Low Shelf Boost (ユーザー設定)
  │
  ▼
★★ outputMakeupGain +12dB ★★  ← 45%
  │
  ├──SoftClip有効──→ 高調波歪み + ブロック境界不連続
  │                     │
  │                     ▼
  │               Oversampling Downsampler  ← 15%
  │               (FIR history 汚染)
  │                     │
  │                     ▼
  └──SoftClip無効──→ 単純増幅信号
                        │
                        ▼
                   DC Blocker (線形、影響小)  ← 5%
                        │
                        ▼
                   NoiseShaper (Lattice/Fixed4Tap)
                   (過大入力→内部状態不安定化)  ← 25%
                        │
                        ▼
                   「ジジジジ」ノイズ
```

**最も可能性の高い相互作用**: ①ゲイン構造で増幅された信号が、③OS Downsamplerの履歴を汚染し、②NoiseShaper共通経路で発振的挙動を示す。SoftClipの有無は二次的。

---

## 6. 切り分けテスト（7項目）

| # | テスト | 操作 | 症状消失時の示唆 |
|:-:|-------|------|---------------|
| 1 | Output Makeup 0dB | `newOutputMakeupDb = 12.0f` → `0.0f` | ①ゲイン構造が主因 |
| 2 | NoiseShaper OFF | `applyDither = false` 相当 | ②NS共通経路が主因 |
| 3 | 20Hz HPF | Convolver→EQ間にHPF挿入 | ⑤DC/超低域が主因 |
| 4 | SoftClip OFF | `softClipEnabled = false` | SoftClipの高調波が主因の一部 |
| 5 | OS 1x固定 | `manualOversamplingFactor = 1` | ③OS Downsamplerが主因 |
| 6 | Convolver OFF | Conv bypass（PEQのみ） | ①④の組み合わせ |
| 7 | Makeup 段階変更 | +12→+6→0→-6dB | ①ゲイン構造と症状強度の相関 |

**優先順位**: テスト1 → テスト2 → テスト5 → テスト4 → テスト7 → テスト3 → テスト6

---

## 7. outputMakeupGain 移動案の評価

**危険性**: v2で提案した「outputMakeupGain を NoiseShaper 後へ移動」は以下の理由で推奨しない：

| 問題点 | 説明 |
|--------|------|
| NoiseShaper前提の崩壊 | NoiseShaper は最終出力レベルを前提に量子化ステップを計算。+12dB後にNSを置くとノイズシェイプ特性が崩れる |
| scale/invScale の不整合 | `quantize()` 内の `scale = 1/invScale` は最終出力レベルの量子化ステップ。後段でゲインが変わると不適切 |
| 副作用の大きさ | 信号経路の変更であり、全出力モードでの検証が必要 |

**代案**: 原因が確定するまでは「診断用途での makeup gain 低減（0dB固定）」を優先。

---

## 8. 結論

| 項目 | 評価 |
|------|------|
| Conv→Peq ゲイン構造異常 | **確認済み**（コード確定） |
| NoiseShaper共通飽和 | **可能性高い**（Fixed4Tapでも発生） |
| OS Downsampler 関与 | **新たな有力仮説**（v2の見落とし） |
| SoftClip犯人説 | **未確定**（sat量依存） |
| DC説 | **可能性低い**（二重除去あり） |
| outputMakeup移動案 | **危険・推奨しない** |

**現時点での最有力シナリオ**: `Conv→Peq専用ゲイン構造(+12dB)` が NoiseShaper 入力を過大にし、さらに OS Downsampler の内部状態との相互作用により低域で顕在化する。SoftClip の関与は二次的。

**確定には上記7テストの実機検証が必要**。
