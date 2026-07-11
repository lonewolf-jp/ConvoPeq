# #1 / #2 詳細検証報告 — DSPCoreFloat バイパス欠落 & リングバッファ負index

**調査日**: 2026-07-07
**検証基盤**: WSL grep/rg, semble, ast-grep, 実コード読解

---

## バグ#1: DSPCoreFloat.cpp におけるバイパス・ブレンド機構の欠落

### 結論: **VALID。Double版に存在する完全バイパススムーシング機構がFloat版にない。**

従来の報告は「Double版にバイパスブレンドが欠落」と主張していたが、これは**事実と逆方向**である。
実際には:
- ✅ **Double版 (`processDouble`)**: 完全に実装済み
- ❌ **Float版 (`process`)**: 完全に欠落

---

### 1.1 完全バイパス（Full Bypass）とは

- **完全バイパス**: EQ と Convolver の両方が同時にバイパスされた状態
  → プラグインは入力を素通しする
- **問題**: 完全バイパスに**遷移する瞬間**に、処理済み音声から未処理音声への**急峻な切り替え**が発生するとクリック/ノイズになる
- **対策**: 5ms の線形フェード（`LinearRamp`）で**クロスフェード**する

---

### 1.2 Double版（正しく実装済み）のコードフロー

**ファイル**: `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp`

#### Step 1: バイパス判定とフェードトリガー（380-387行目）

```cpp
const bool requestedFullBypass = state.eqBypassed && state.convBypassed;
auto& ramp = ramps();
if (requestedFullBypass != ramp.bypassedDouble)
{
    ramp.bypassFadeGainDouble.setTargetValue(requestedFullBypass ? 0.0 : 1.0);
    ramp.bypassedDouble = requestedFullBypass;
}
```

- `state.eqBypassed && state.convBypassed` で完全バイパスを判定
- **状態変化時のみ** `bypassFadeGainDouble` を target 設定（0.0 = ミュート, 1.0 = 通過）
- `ramp.bypassedDouble` に状態を保存（次回変化検出用）

#### Step 2: ドライ信号の保存（392-396行目）

```cpp
if (dryBypassBufferDoubleL && dryBypassBufferDoubleR && dryBypassCapacityDouble >= numSamples)
{
    juce::FloatVectorOperations::copy(dryBypassBufferDoubleL.get(), alignedL.get(), numSamples);
    juce::FloatVectorOperations::copy(dryBypassBufferDoubleR.get(), alignedR.get(), numSamples);
}
```

- DSP処理の**前**に生の入力を `dryBypassBufferDoubleL/R` に保存
- これがクロスフェードの「ドライ信号」になる

#### Step 3a: OS=1 の場合のクロスフェード（548-568行目）

```cpp
const bool bypassBlendRequested = ramp.bypassFadeGainDouble.isSmoothing() || requestedFullBypass;
if (oversamplingFactor == 1 && dryBypassBufferDoubleL && dryBypassBufferDoubleR
    && dryBypassCapacityDouble >= numSamples && bypassBlendRequested)
{
    // サンプル単位で wet* gWet + dry* gDry を計算
    for (int i = 0; i < numProcSamples; ++i)
    {
        const double gWet = ramp.bypassFadeGainDouble.getNextValue();
        const double gDry = 1.0 - gWet;
        wetL[i] = wetL[i] * gWet + dryL[i] * gDry;
    }
}
```

#### Step 3b: OS>1 の場合のクロスフェード（572-608行目）

```cpp
if (oversamplingFactor > 1)
{
    oversampling.processDown(processBlock, originalBlock, ...);
    processBlock = originalBlock;
    if (bypassBlendRequested)
    {
        // 同様のクロスフェード処理
        // canUseDry の有無でフォールバック
    }
}
```

- OS>1 の場合は**ダウンサンプル後に**クロスフェードを実施
- `dryBypassBufferDouble` が無効でも `wet*gWet` だけでフェード可能

#### Step 4: RampRuntimeState 定義（AudioEngine.h 709-726行目）

```cpp
struct RampRuntimeState {
    convo::LinearRamp bypassFadeGainDouble;  // バイパスフェード用リニアランプ
    bool bypassedDouble = false;             // 現在のバイパス状態キャッシュ

    void prepare(double sampleRate) noexcept {
        bypassFadeGainDouble.reset(sampleRate, 0.005);  // 5ms のフェード時間
        bypassFadeGainDouble.setCurrentAndTargetValue(1.0);
        bypassedDouble = false;
    }
};
```

- フェード時間: **5ms** (`0.005`)
- `prepare()` で初期化（これによりフェードは常に5msで動作）

#### Step 5: dryBypassBufferDouble の確保（DSPCoreLifecycle.cpp）

```cpp
if (newRequired > dryBypassCapacityDouble || !dryBypassBufferDoubleL || !dryBypassBufferDoubleR)
{
    auto newDryL = convo::makeAlignedArray<double>(static_cast<size_t>(newRequired));
    auto newDryR = convo::makeAlignedArray<double>(static_cast<size_t>(newRequired));
    // ...
    dryBypassBufferDoubleL = std::move(newDryL);
    dryBypassBufferDoubleR = std::move(newDryR);
    dryBypassCapacityDouble = newRequired;
}
```

- 内部ブロックサイズと同じ容量でアライメント確保

---

### 1.3 Float版（欠落）のコードフロー

**ファイル**: `src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp` (260-410行目)

```cpp
void DSPCore::process(const juce::AudioSourceChannelInfo& bufferToFill, ...)
{
    // 1) アーリーリターン（ガード）
    // 2) オーバーサンプリングUp
    // 3) Core DSP（Convolver / EQ、各バイパス対応）
    // 4) OutputFilter
    // 5) Makeup gain
    // 6) Soft clip
    // 7) Oversampling Down
    //    ※ bypassBlendRequested / dryBypassBuffer の概念なし
    // 8) Analyzer
    // 9) processOutput  → Float出力変換・ノイズシェイパ・ディザ
    //    ※ バイパスブレンド処理なし
}
```

**欠缺している要素**:

| 要素 | Double版 | Float版 |
|-------|----------|---------|
| `requestedFullBypass` 判定 | ✅ 382行目 | ❌ なし |
| `bypassFadeGainDouble.setTargetValue()` | ✅ 386行目 | ❌ なし |
| ドライバッファへのコピー | ✅ 394行目 | ❌ なし |
| `bypassBlendRequested` | ✅ 548行目 | ❌ なし |
| OS=1 クロスフェード | ✅ 553-568行目 | ❌ なし |
| OS>1 クロスフェード | ✅ 575-588行目 | ❌ なし |

---

### 1.4 影響と深刻度

| シナリオ | 影響 |
|----------|------|
| **32bit float処理時**にEQもConvolverもバイパス | → 状態変化時に音声が不連続になる可能性 |
| ホストが32bit floatパスを使用（多くのDAWのデフォルト） | → ユーザーが体感する可能性が高い |
| 完全バイパス→解除の遷移 | → 5msのクロスフェードがないためクリックノイズ |
| Dry/Wet（Mix）コントロール | → これはConvolverProcessor内の `mixSmoother` が独立して処理するため本件とは別 |

**リスク評価**:
- **発生条件**: 完全バイパス（EQ+Convolver両方OFF）を操作したとき
- **可聴性**: クリック/ポップとして聞こえる。フェード時間5msは短いが、急峻な0→1切り替えよりはるかに良好
- **対象パス**: `process()` を呼ぶ `processToBuffer()` 経路（float出力）
- **非該当パス**: `processDouble()` を呼ぶ `processDoubleToBuffer()` 経路（double出力）は正しい

---

### 1.5 修正方針

Float版 `process()` に以下を追加する必要がある：

1. **完全バイパス判定**: `requestedFullBypass = state.eqBypassed && state.convBypassed`
2. **フェードランプ設定**: `bypassFadeGainDouble.setTargetValue()`
3. **ドライ信号保存**: `alignedL/R` の内容を保持
4. **クロスフェード**: OS=1 と OS>1 の両方のパスで `bypassFadeGainDouble` を使用した wet/dry ブレンド

注意点:
- `bypassFadeGainDouble` は `RampRuntimeState` に**すでに存在**（Float用にも確保済みだが未使用）
- `dryBypassBufferDoubleL/R` も `DSPCore` に**すでに存在**（Float用にも確保済みだが未使用）
- つまり**データ構造は揃っているが、Float版の `process()` が使っていないだけ**

---

## バグ#2: リングバッファ負のindex — `(idx - 1) & DELAY_BUFFER_MASK`

### 結論: **C++17以前では実装定義だが、現実の全x86/x64実装で正しく動作する。C++20では完全に定義済み。実害は極めて低い。**

---

### 2.1 該当箇所

**ファイル**: `src/convolver/ConvolverProcessor.Runtime.cpp` 478-482行目

```cpp
// readInterpolated — Cubic補間のリングバッファ読み出し
// fallback path（iRead < 1 または バッファ終端付近）
for (; i < samplesToRead; ++i)
{
    int idx = iRead + i;
    double p0 = srcBuf[(idx - 1) & DELAY_BUFFER_MASK];
    double p1 = srcBuf[(idx    ) & DELAY_BUFFER_MASK];
    double p2 = srcBuf[(idx + 1) & DELAY_BUFFER_MASK];
    double p3 = srcBuf[(idx + 2) & DELAY_BUFFER_MASK];
    dst[i] = w0 * p0 + w1 * p1 + w2 * p2 + w3 * p3;
}
```

### 2.2 定数定義（ConvolverProcessor.h）

```cpp
static constexpr int DELAY_BUFFER_SIZE = 4194304;  // 2^22
static constexpr int DELAY_BUFFER_MASK = DELAY_BUFFER_SIZE - 1;  // 0x3FFFFF
static_assert((DELAY_BUFFER_SIZE & (DELAY_BUFFER_SIZE - 1)) == 0,
              "must be a power of 2");
```

### 2.3 問題のメカニズム

`iRead` はリングバッファの現在読み取り位置。
このelseブランチは以下の場合に実行される：

```
!(iRead >= 1 && iRead + samplesToRead + 2 < DELAY_BUFFER_SIZE)
```

= **「読み取り位置が先頭すぎる」または「終端に近すぎる」**

### 【ケースA】`iRead == 0` のとき

```
idx(i=0)   = 0
idx-1      = -1
結果: (-1) & 0x3FFFFF
```

#### C++17以前

**実装定義**。しかし、MSVC/GCC/Clang の全x86/x64コンパイラは2の補数表現を使用:
```
-1 = 0xFFFFFFFFFFFFFFFF (64bit 2の補数)
(-1) & 0x3FFFFF = 0x3FFFFF = 4194303
```
→ **リングバッファの最終要素へのアクセス**。これは正しい（cubic補間の過去方向）。

#### C++20以降:
**明確に定義**。`[expr.bit.and]/1` により、`unsigned` への変換ではなく二項 `&` のオペランドとしての負の整数は、2の補数ビット表現での演算が保証される。

### 【ケースB】バッファ終端付近のとき

```
例: iRead = DELAY_BUFFER_SIZE - 3, i = 0
idx+2 = DELAY_BUFFER_SIZE - 1  → OK
i = 1:
idx+2 = DELAY_BUFFER_SIZE      → & MASK = 0 → 先頭にラップ
```
これも**意図されたリングバッファ動作**。cubic補間の前方サンプルがバッファ先頭にラップする。

### 2.4 C++標準の詳細

| C++標準 | `(-1) & MASK` の動作 | 備考 |
|---------|---------------------|------|
| C++11/14/17 | **実装定義** | [expr.bit.and]/1: "The result is an implementation-defined two's complement representation" — ただし現行の全コンパイラが2の補数 |
| C++20 | **明確に定義** | P1236R1: 符号付き整数は2の補数と規定 |
| C++23/26 | 同左 | 変更なし |

**現実の互換性**:
MSVC: `/std:c++14` 以降すべて2の補数
GCC/Clang: `-fwrapv` / `-fno-strict-overflow` の有無にかかわらずビット演算は常に2の補数

### 2.5 改善提案

**実害がない**ため優先度は低いが、ポータブルにする場合:

```cpp
// Before:
double p0 = srcBuf[(idx - 1) & DELAY_BUFFER_MASK];

// After:
// Option A: unsigned で演算
double p0 = srcBuf[(static_cast<unsigned>(idx) - 1u) & DELAY_BUFFER_MASK];

// Option B: 加算で正の値に
double p0 = srcBuf[(idx - 1 + DELAY_BUFFER_SIZE) & DELAY_BUFFER_MASK];

// Option C: ガード条件を強化（最もクリーン）
int idx = (iRead + i) & DELAY_BUFFER_MASK;
double p0 = srcBuf[(idx - 1 + DELAY_BUFFER_SIZE) & DELAY_BUFFER_MASK];
double p1 = srcBuf[idx];
double p2 = srcBuf[(idx + 1) & DELAY_BUFFER_MASK];
double p3 = srcBuf[(idx + 2) & DELAY_BUFFER_MASK];
```

Option C が最も明示的で誤りが少ない。

### 2.6 同一パターンの箇所

同じファイル内で他に `(idx + n) & DELAY_BUFFER_MASK` パターンを使用する箇所:

```
462: 読み取り位置補正 (rPos -= floorNoLibm(rPos / DELAY_BUFFER_SIZE) * DELAY_BUFFER_SIZE)
465: (iRead + 1) & DELAY_BUFFER_MASK     — これも iRead = DELAY_BUFFER_SIZE-1 で安全
470: rPosInt & DELAY_BUFFER_MASK         — 正値のみ
478: (idx - 1) & DELAY_BUFFER_MASK       — ★該当箇所
479: (idx    ) & DELAY_BUFFER_MASK       — 問題なし（idx≧0）
480: (idx + 1) & DELAY_BUFFER_MASK       — idxが終端付近でのみラップ
481: (idx + 2) & DELAY_BUFFER_MASK       — 同上
517: (activeDelayWritePos - delayInt) & DELAY_BUFFER_MASK — rPos < 0 のガードあり
```

「rPos < 0 ガードあり」の箇所（517行目）と比較すると、478行目はガードなしで負値が発生しうる唯一の箇所である。

---

## 総合リスク評価

### #1: DSPCoreFloat バイパス欠落 — 🟡 中リスク

| 項目 | 評価 |
|------|------|
| 発生確率 | 中（完全バイパスをfloatパスで行う頻度に依存） |
| 影響度 | 小〜中（クリックノイズ。音割れやクラッシュではない） |
| 修正難易度 | 低（データ構造は既存、process()にコード追加のみ） |
| 検出可能性 | 低（通常のEQ/Convolver単体バイパスは問題なし） |

### #2: 負のindex — 🟢 低リスク

| 項目 | 評価 |
|------|------|
| 発生確率 | 中（iRead=0のコーナーケース） |
| 影響度 | 極低（現実の全コンパイラで正しく動作） |
| 修正難易度 | 極低（3文字追加 `+ DELAY_BUFFER_SIZE`） |
| C++20以降 | 問題なし（言語仕様で保証） |
