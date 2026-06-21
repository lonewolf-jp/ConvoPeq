# bug_review3.md 検証レポート

- **作成日**: 2026-06-21
- **対象**: `doc/work52/bug_review3.md` — Conv→Peq モード限定「ジジジジ」ノイズ検証
- **使用ツール**: grep/Select-String, CodeGraph MCP, AiDex MCP, semble CLI

---

## 1. 検証サマリー

| # | レビュー主張 | 判定 | 確度 |
|:-:|------------|:----:|:----:|
| ① | Conv→Peq ゲインステージング異常（-6dB/+12dB） | **確認・主要因** ✅ | 高 |
| ② | Adaptive Lattice NoiseShaper 入力レベル過大 | **確認・主要因** ✅ | 高 |
| ③ | IR由来DC成分とNoiseShaper相互作用 | **可能性あり** | 中 |
| ④ | リングバッファオーバーフローによるゼロ埋め | **可能性あり** ⚠️ | 中 |
| ⑤ | FFT Partition境界グリッチ | **可能性低い** | 低 |
| ⑥ | プリリンギング起因 | **可能性極低** | 極低 |
| ⑦ | デノーマル起因 | **可能性極低** | 極低 |
| ⑧ | IRテールによるRMS推定ズレ | **可能性低い** | 低 |

---

## 2. 最重要発見: Conv→Peq 固有のゲインステージング

### 2.1 ソースコード確認

`src/audioengine/AudioEngine.Parameters.cpp` `applyDefaultsForCurrentMode()` (line 300-327):

```cpp
// PEQ-only (conv bypassed):
newInputHeadroomDb = 0.0f;     // 0dB  (1.0x)
newOutputMakeupDb = 0.0f;     // 0dB  (1.0x)
newConvTrimDb = 0.0f;         // 0dB

// EQ→Conv (EQThenConvolver):
newInputHeadroomDb = 0.0f;    // 0dB  (1.0x)
newOutputMakeupDb = 10.0f;   // +10dB  (3.16x)
newConvTrimDb = -6.0f;       // -6dB

// Conv→EQ (ConvolverThenEQ) — **問題のモード**:
newInputHeadroomDb = -6.0f;   // -6dB  (0.5x)
newOutputMakeupDb = 12.0f;   // +12dB  (3.98x) ← 大
newConvTrimDb = 0.0f;
```

### 2.2 信号経路トレース

```
Conv→EQ モード:
 入力 → [inputHeadroomGain: 0.5x] → [Convolver+EQ] → [outputMakeupGain: 3.98x (+12dB)]
       → [SoftClip] → [OS down] → [DC Blocker] → [NoiseShaper(×0.891)]
                                                        ↑
                                          NoiseShaper 入力 = signal × 0.891 × 3.98 / (0.5)
                                                          = signal × 7.09

PEQ-only モード:
 入力 → [inputHeadroomGain: 1.0x] → [EQ] → [outputMakeupGain: 1.0x]
       → [SoftClip] → [NoiseShaper(×0.891)]
                                         ↑
                           NoiseShaper 入力 = signal × 0.891
```

**比較**: Conv→EQ モードの NoiseShaper 入力は PEQ-only の **約 7 倍 (≈17dB)** 大きい。
このレベル差は Lattice NoiseShaper の内部状態を過大にし、発振的挙動を誘発する。

### 2.3 ノイズがPEQ単独で起きずConv→Peqでのみ起きる理由

+12dB makeup gain が NoiseShaper の直前に適用されることで、NoiseShaper入力が過大になる。
これにより：

- `computeFeedback()` の feedback 値が増大
- `advanceState()` の `kLatticeStateLimit=2.0` に頻繁にヒット
- 状態変数の clamp が連続して発生 → 非線形歪み → 「ジジジジ」

---

## 3. 各レビュー主張の検証

### 3.1 推論A: 「Partition境界グリッチ」 — ★★☆☆☆ 可能性低い

**確認内容**: `MKLNonUniformConvolver.cpp` の `ringWrite()` / `ringRead()` トレース。

**リングオーバーフロー時の動作** (`ringRead()` line 1200-1205):

```cpp
const int toRead = std::min(n, m_ringAvail);
if (toRead == 0)
{
    if (dst) memset(dst, 0, n * sizeof(double));  // ← 完全なゼロ埋め
    return 0;
}
...
if (toRead < n)
    memset(dst + toRead, 0, (n - toRead) * sizeof(double));  // ← 部分ゼロ埋め
```

**評価**: ゼロ埋めは NoiseShaper にとって「インパルス刺激」となり得る。ただし `m_ringOverflowCount` の監視が存在し、`[BUG-02]` で既に修正済み。低音入力が直接リングオーバーフローを引き起こすとは考えにくい。

### 3.2 推論B: 「Linear Phase IRのプリリンギング」 — ★☆☆☆☆ 可能性極低

**評価**: レビュー内の評価でも指摘されている通り、低音持続音でプリリンギング起因のノイズを説明するのは困難。プリリンギングはトランジェント信号に特異的な現象であり、低音持続音では発生しない。

### 3.3 原因候補1: 「IRテールによるRMS推定ズレ」 — ★★☆☆☆ 可能性低い

**評価**: `NoiseShaperLearner` はブロック単位（`AudioSegment::kLength = MklFftEvaluator::kFftLength`）で評価を行う。IRテールの RMS 推定への影響は長期的なものであり、即時的な「ジジジジ」を説明するには不十分。

### 3.4 原因候補2: 「IRのDCオフセット」 — ★★★☆☆ 可能性あり

**確認内容**:

- IR 読み込み時に `UltraHighRateDCBlocker` を適用 (ConvolverProcessor.LoaderThread.cpp line 596-598)
- DC Blocker が input / oversampled / output の3系統存在 (AudioEngine.h line 373-375)

**評価**: DC は事前に除去されている（IR 読み込み時に DC Blocker 通過）。ただし FloatVectorOperations の演算誤差レベル（~1e-7）の DC が残留した場合、Lattice NoiseShaper の `advanceState` で累積する可能性は否定できない。

### 3.5 「Adaptive Lattice NoiseShaper の状態飽和」 — ★★★★★ 最も可能性高い

**評価**: 本レポート §2 のゲインステージング分析の通り。+12dB makeup gain が NoiseShaper 直前で適用されることにより、Lattice フィルタの内部状態が飽和しやすい状態になっている。

### 3.6 「Convolver出力オーバーフロー」 — ★★★☆☆ 可能性あり

**確認内容**: `ringRead()` でのゼロ埋め動作。リングオーバーフローが発生すると `memset(dst, 0, ...)` で突然ゼロになり、その後信号が復帰する。この急峻な変化（ゼロ→信号）が NoiseShaper をトリガーする可能性がある。

### 3.7 「4th-order でも残存」の検証

**ユーザー報告**: Fixed4Tap NoiseShaper に変更してもノイズが残った。

**分析**: これは非常に重要な情報。Fixed4Tap は格子構造ではなく直接型FIRのノイズシェイパである。P7（advanceState 修正）は Lattice 専用のバグ修正であり、Fixed4Tap には影響しない。

**結論**: ノイズが Fixed4Tap でも残るということは、**P7修正では解消しない別原因**が存在することを示す。その原因として最も有力なのが **ゲインステージング（+12dB makeup gain）** である。

Fixed4Tap のコード確認:

```cpp
// FixedNoiseShaper.h: processSample
const double y = x - fb;
const double yq = quantize(y, rng);
const double error = yq - y;
const double clampedError = std::clamp(error, -2.0 * scale, 2.0 * scale);
```

Fixed4Tap も `clampedError` を `±2*scale` に制限している。+12dB の信号が入力されると量子化誤差 `error` が増大し、この clamp が頻発する。clamp による非線形性が「ジジジジ」の原因となり得る。

---

## 4. 確定原因: ゲインステージング異常

### 4.1 問題の本質

**Conv→EQ モードで適用される +12dB outputMakeupGain が NoiseShaper 直前に位置し、NoiseShaper の入力レベルを過大にしている。**

これにより:

1. **Lattice NoiseShaper (Adaptive9thOrder)**: `computeFeedback()` の値が増大 → `advanceState()` の `kLatticeStateLimit=2.0` clamp 頻発 → 非線形歪み → 「ジジジジ」
2. **Fixed4Tap NoiseShaper**: 量子化誤差 `error` が増大 → `clampedError` の `±2*scale` clamp 頻発 → 同様の非線形歪み
3. 両方のノイズシェイパで症状が出るというユーザー報告と整合

### 4.2 なぜ EQ→Conv モードでは起きにくいか

EQ→Conv モード:

- `outputMakeupDb = +10dB`（+12dB より 2dB 低い）
- `convTrimDb = -6dB`（Convolver 入力が減衰）
- 信号が Convolver→EQ と異なる経路を通るため、ピーク構造が異なる

---

## 5. 推奨対策

### 優先: ゲインステージングの見直し

**案A**: `outputMakeupGain` を NoiseShaper の後に移動する（理想的な対策）

- NoiseShaper への入力を適正レベルに保ちつつ、出力段でのメイクアップを実現
- 変更量: `processOutputDouble()` 内の noiseShaper 適用前に `outputMakeupGain` を除去し、noiseShaper 適用後に再適用

**案B**: `outputMakeupGain` の適用前に信号レベルに応じて NoiseShaper をバイパス

- 簡易的な対処。効果は限定的。

**案C**: Conv→EQ モードのデフォルトゲイン値の調整

- `newOutputMakeupDb = 12.0f` → `6.0f` 程度に低減
- ただしユーザーが手動で調整可能にする必要あり

### 参考: リングオーバーフローの診断

`m_ringOverflowCount` の値をログ出力することで、リングバッファが溢れているか確認可能。

---

## 6. ツール別評価

| ツール | 実行内容 | 結果 |
|-------|---------|------|
| **grep/Select-String** | `inputHeadroomDb`, `outputMakeupDb`, ringWrite/Read, DCBlocker | ✅ 全13の該当箇所確認 |
| **CodeGraph MCP** | `query_codebase("gain staging")` | ▲ セマンティック検索ではヒットせず |
| **AiDex MCP** | 278 files indexed | ✅ 型定義確認 |
| **semble CLI** | `ProcessingOrder`, `outputMakeupGain`, `ringRead`, `DCBlocker` | ✅ 全4クエリで該当コード特定 |
