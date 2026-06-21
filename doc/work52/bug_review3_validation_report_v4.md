# bug_review3.md 検証レポート v4（確定版）

- **作成日**: 2026-06-21
- **対象**: `doc/work52/bug_review3.md` — Conv→Peq 限定「ジジジジ」ノイズ
- **使用ツール**: grep/Select-String, CodeGraph MCP, AiDex MCP, semble CLI

---

## 0. バグ内容の整理

| 条件 | ノイズ |
|:----:|:------:|
| Conv→Peq + Adaptive9th | **発生** |
| Conv→Peq + Fixed4Tap | **発生** |
| Conv→Peq + Fixed15Tap | **発生** |
| Conv→Peq + Psychoacoustic | **発生（推定）** |
| PEQ-only + 全NS | **発生せず** |
| トリガー | **大き目の低音 + 低音とかさなって** |

**最重要事実**: 全 NoiseShaper で発生する → NoiseShaper 個別のバグ（P7含む）は原因ではない。

**問題は NoiseShaper より前段に存在する。**

---

## 1. processOutputDouble() 最終段 clamp の発見

### 1.1 実コード

`src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp` (lines 604-664):

```
[applyDither = true]
  NoiseShaper (×0.891 internal headroom)       ← lines 604-613
      ↓  NS出力は最大 ±1.0 （quantize clamp）
  NaN/Inf 除去                                  ← lines 621-645
      ↓
  ★★★ HARD CLAMP at ±kOutputHeadroom (±0.891) ★★★ ← lines 647-664
      ↓
  Output

[applyDither = false]
  dataL[i] *= kOutputHeadroom                   ← lines 615-619
      ↓  ×0.891 適用後も信号が大きい可能性
  NaN/Inf 除去                                  ← lines 621-645
      ↓
  ★★★ HARD CLAMP at ±kOutputHeadroom (±0.891) ★★★ ← lines 647-664
      ↓
  Output
```

### 1.2 Clamp コード

```cpp
// lines 647-664
const __m256d vLimit = _mm256_set1_pd(kOutputHeadroom);     // 0.891
const __m256d vNegLimit = _mm256_set1_pd(-kOutputHeadroom);
// ...
vL = _mm256_min_pd(_mm256_max_pd(vL, vNegLimit), vLimit);  // HARD CLAMP
```

**この clamp は全 NoiseShaper 共通の後段に位置し、全モードで常に実行される。**

---

## 2. 信号レベルトレース（数値検証）

### 2.1 PEQ-only モード（正常）

Input: 正弦波 -6dBFS (peak=0.5), EQ Low Shelf +6dB, sat=0.1

```
inputHeadroom 0dB  → 0.5 × 1.0   = 0.50
EQ +6dB           → 0.5 × 2.0   = 1.00
outputFilter      →              ≈ 1.00
outputMakeup 0dB  → 1.0 × 1.0   = 1.00
SoftClip (sat=0.1)→ threshold=0.905 → clip → ≈0.95
×0.891            → 0.95 × 0.891 = 0.846
NS quantize       →              ≈ 0.846
★★★ CLAMP ±0.891  → 0.846 < 0.891 → 通過 ✅
```

**PEQ-only では clamp に引っかからない。**

### 2.2 Conv→Peq モード（異常）

Input: 正弦波 -6dBFS (peak=0.5), IRノーマライズ (+0dB peak), EQ Low Shelf +6dB, sat=0.1

```
inputHeadroom -6dB → 0.5 × 0.5      = 0.25
Convolver (IR)    → 0.25 × 1.0(+0dB)= 0.25
  ※IRによっては +6〜15dB peak 増幅あり
EQ +6dB           → 0.25 × 2.0     = 0.50
outputFilter      →                 ≈ 0.50
★★ outputMakeup +12dB → 0.50 × 3.98 = 1.99 ★★
SoftClip (sat=0.1)→ threshold=0.905 → clip → ≈0.95
(bass heavy: 3.98x >> 0.905 → 強烈クリップ)
×0.891            → 0.95 × 0.891   = 0.846
NS quantize       →                 ≈ 0.846
★★★ CLAMP ±0.891  → 0.846 < 0.891 → 通過...ぎりぎり ✅
```

**SoftClip 有効時はぎりぎり clamp を通過するが、以下の条件で危険:**

| 条件 | NSF入力推定 | clamp判定 |
|------|:----------:|:---------:|
| sat=0, SoftClip無効 | 1.99×0.891=1.77→NS clamp ±1.0→1.0 | **CLAMP!** ❌ |
| IR +10dB peak | 3.98×0.891=3.55→NS clamp ±1.0→1.0 | **CLAMP!** ❌ |
| EQ +12dB boost | 同上 | **CLAMP!** ❌ |
| 低域RMS高＋SoftClip非線形→NS内部状態増大 | NS出力が1.0付近 | **CLAMP!** ❌ |

### 2.3 IR ピーク増幅の影響

IRファイルの性質:

- 残響系IR: ピーク/平均比が大きい（6-15dB）
- Convolverは線形 → ピークがそのまま出力される
- inputHeadroom -6dB では補償しきれない場合がある

⇒ IRに+6dBのピークがある場合:

```
0.5 × 0.5(inputHeadroom) × 2.0(IR+6dB) = 0.5
× 2.0(EQ+6dB) = 1.0
× 3.98(makeup+12dB) = 3.98
SoftClip → 0.95 (heavy clip)
× 0.891 → 0.846 → clamp通過...ぎりぎり
```

**でも SoftClip が無効または sat=0 なら:**

```
× 0.891 → 3.55
NS quantize clamp → 1.0
★★★ CLAMP ±0.891 → ❌ HARD CLIP!
```

---

## 3. 原因の特定

### 3.1 直接原因: post-NS ハードクランプ

`processOutputDouble()` の最終段 (lines 647-664) で **全信号を ±0.891 にハードクリッピング** している。この clamp は NoiseShaper の**後**、出力の**前**に位置し、全 NoiseShaper に共通。

**全ての NoiseShaper で発生する事実と完全に整合。**

### 3.2 根本原因: Conv→Peq ゲイン構造

Conv→Peq モードでは以下のゲイン差により NS 入力信号が過大になり、post-NS clamp が頻発する：

| モード | inputHeadroom | outputMakeup | 正味ゲイン差 |
|:-----:|:------------:|:------------:|:----------:|
| PEQ-only | 0dB | 0dB | **0dB**（基準） |
| Conv→Peq | **-6dB** | **+12dB** | **+6dB** |

さらに +12dB の適用位置が SoftClip より前（line 439）のため、SoftClip が無効または弱い場合に信号が 3.98x まで増幅され、下流の全段に過大信号が流れる。

### 3.3 低音特化性

| 要因 | 説明 |
|:----:|------|
| 低音の高エネルギー | 低域は中高域より振幅が大きい（音楽信号の性質） |
| SoftClipの非線形性 | 低域の高振幅が SoftClip で高調波歪みに変換 |
| NS内部状態増大 | 低域の高振幅＋歪みが NS の内部状態（Lattice/IIR）を増大 |
| post-NS clamp発動 | NS出力が ±0.891 を超え、ハードクリップ |

---

## 4. 改修提案

### 4.1 提案A: post-NS ハードクランプの除去または緩和（推奨）

`processOutputDouble()` の ±0.891 ハードクランプ（lines 647-664）は、以下の理由で見直しが必要：

| 問題点 | 説明 |
|--------|------|
| 約-1dBでのハードクリップ | ±0.891 はデジタルフルスケールの約 -1dB。ここでのハードクリップは可聴歪みを生む |
| 全NS共通 | どのNSを使っても同じ歪みが発生する（バグ報告と完全一致） |
| 既存の安全策との重複 | NaN/Inf除去(lines 621-645)、NS内部のquantize clamp、SoftClip が既に存在 |

**改修案A-1**: clamp 閾値を ±0.891 から ±1.0 に緩和

```cpp
// 変更前
constexpr double kOutputHeadroom = 0.8912509381337456;
// 変更後
constexpr double kOutputHeadroom = 1.0;
```

**改修案A-2**: clamp を「Inf/NaN 対策のみ」に変更（±1e15 など）

```cpp
// 変更前
vL = _mm256_min_pd(_mm256_max_pd(vL, vNegLimit), vLimit);
// 変更後（セーフティクランプのみ、通常信号は通す）
const __m256d vSafetyLimit  = _mm256_set1_pd(1.0e15);
const __m256d vSafetyNegLimit = _mm256_set1_pd(-1.0e15);
vL = _mm256_min_pd(_mm256_max_pd(vL, vSafetyNegLimit), vSafetyLimit);
```

**リスク**: 出力信号がデジタルフルスケールを超える可能性がある。ただし DAC や後段ファイルフォーマットで対応可能な範囲（+0.1〜+0.5dB 程度）。

### 4.2 提案B: outputMakeupGain の低減（診断・暫定対策）

```cpp
// AudioEngine.Parameters.cpp line 326
newOutputMakeupDb = 12.0f;  →  newOutputMakeupDb = 6.0f;
```

**効果**: +12dB → +6dB で信号の過大増幅を緩和。post-NS clamp の発動頻度が低下。
**副作用**: 出力レベルが約-6dB低下する。

### 4.3 提案C: outputMakeupGain と kOutputHeadroom の一貫性確保

`kOutputHeadroom = 0.891`（-1dB）は「最終出力段のヘッドルーム」として設計されているが、前段の +12dB 増幅と整合していない。outputMakeupGain と kOutputHeadroom の和が 0dB になるよう整合させる：

```cpp
// 設計思想: outputMakeup + kOutputHeadroom = 0dB を維持
// Conv→Peq: inputHeadroom -6dB → makeup +6dB → headroom 0dB (=1.0)
```

---

## 5. 優先テスト手順

### テスト1: post-NS clamp の影響確認

`processOutputDouble()` の clamp ブロック（lines 647-664）を一時的にコメントアウトしてビルド。ノイズが消えるか確認。

**消える → post-NS clamp が直接原因。**

### テスト2: Output Makeup 0dB

`newOutputMakeupDb = 12.0f → 0.0f` でビルド。ノイズが消えるか確認。

**消える → ゲイン構造が根本原因。**

### テスト3: kOutputHeadroom = 1.0

`constexpr double kOutputHeadroom = 0.891... → 1.0` でビルド。ノイズが消えるか確認。

**消える → post-NS clamp が直接原因（テスト1と同様）。**

---

## 6. 結論

| 項目 | 判定 |
|------|:----:|
| NoiseShaper固有バグ | **否（全NSで共通）** |
| P7 advanceState修正 | **無関係** |
| 直接原因 | **post-NS ハードクランプ（±0.891）** |
| 根本原因 | **Conv→Peq の +12dB outputMakeupGain** |
| SoftClipの役割 | **二次的（有効時は緩和、無効時は悪化）** |
| OS Downsampler | **無関係（OS=1xでも再現する可能性が高い）** |
| 推奨対策 | **提案A（clamp緩和）+ 提案B（makeup低減）を実機検証** |

**最も確率の高いシナリオ**:

```
Conv→Peq の +12dB gain → 信号過大 → NoiseShaper(どの方式でも)出力が ±0.891 超過
→ post-NS ハードクランプ発動 → 方形波状歪み → 「ジジジジ」
```

**この説の最大の強み**: 「全 NoiseShaper で発生」というバグ報告と完全に整合する。各 NS の内部処理は関係なく、NS の**後段**にある共通のハードクランプが犯人である。
