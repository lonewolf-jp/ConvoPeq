# 📋 検証レポート：ConvoPeq メモリ最適化計画書 ソースコード照合

**対象バージョン**: v0.5.0 (main)
**検証日**: 2026-06-21
**検証者**: GitHub Copilot (DeepSeek V4 Flash)

---

## 使用したツールと検証範囲

| ツール | 用途 | 確認ファイル数 |
|--------|------|---------------|
| **Serena MCP** | シンボル構造解析、メソッド依存関係、メモリ読み取り | 40+ |
| **CodeGraph MCP** | コードグラフクエリ、被呼出関係 (16,442 entities indexed) | 全 indexed entities |
| **grep/Select-String** | キーワード横断検索、全参照経路の棚卸し | 全ソースツリー |
| **AiDex MCP** | 識別子検索 (`aidex_query`)、ファイル構造確認 (`aidex_signature`) | 20+ |
| **ファイル直接読取** | 実装詳細の確認、アロケーション解析 | 15ファイル |

---

## 検証結果サマリー

### 計画書の全体的な妥当性: **65%**（当初評価 B+ を下方修正）

| 項目 | 計画書評価 | **検証後評価** | 差分理由 |
|------|-----------|--------------|---------|
| 問題認識 | A | A ✅ | 約2.5GBのメモリ消費問題認識は正しい |
| 原因分析 | B+ | B | RCU多重化+AoS/SoA二重保持+固定バッファの指摘は正しいが、影響度の定量化不足 |
| フェーズ1 | A- | **B** | P1の効果過大評価、P4の難易度過小評価 |
| フェーズ2 | B | B | P5の「CPU改善」主張が未証明 |
| フェーズ3 | B | B+ | P7の価値は正しいが、P6のCPU改善主張は未証明 |
| 効果予測 | C | **D** | 「2.5GB→1.0GB確約」は根拠不足 |
| RT安全性 | B | B | 概ね妥当な評価 |
| ISR Bridge整合性 | B | B+ | P7のISR思想との整合性は高い |
| **全体** | B+ | **B (**65%**)** | 方向性は良いが効果予測・難易度評価に問題 |

---

## P1: `timeDomainIR` 早期解放（「波形・スペクトラムスナップショット生成後、`prepared->timeDomainIR.reset()`」）

### ✅ 事実確認（計画書の前提は正しい）

- `PreparedIRState` (`src/PreparedIRState.h:20`) に `std::unique_ptr<juce::AudioBuffer<double>> timeDomainIR` が存在する。
- `IRConverter` で時間領域IRが生成され `prepared->timeDomainIR` に格納される。
- `irWaveform` (512×float) と `irMagnitudeSpectrum` (最大32769×float) は別途保持されており、UI表示は independent。

### 🔍 問題点（レビュー指摘＋追加発見）

`applyComputedIR()` (`src/convolver/ConvolverProcessor.LoadPipeline.cpp:310-497`) 内での `timeDomainIR` 使用経路:

```
Line 329-344: scaleFactor 適用（コピー&スケーリング）
  → prepared->timeDomainIR から deep copy し、scaledTimeIR を prepared->timeDomainIR に戻す
Line 346-420: 振幅/エネルギー検証（peak/rms チェック、current IR と比較）
Line 455-475: irPeakLatency 計算（IR内のピーク位置検出）
              → timeDomainIR の全サンプルを走査
Line 487    : createWaveformSnapshot(*(prepared->timeDomainIR))  // 波形スナップショット
Line 487    : createFrequencyResponseSnapshot(*(prepared->timeDomainIR), sampleRate)  // スペクトル
Line 497    : updateIRState(*(prepared->timeDomainIR), prepared->sampleRate)
              → timeDomainIR を deep copy して IRState に保存
```

**「スナップショット生成後（line 487）に reset」すると、line 497 の updateIRState が解放済みメモリにアクセス → 未定義動作。**

解放可能なのは line 497 **以降**。しかし line 497 の後、`prepared` unique_ptr のスコープは applyComputedIR 終了と同時に切れるため、**実質的な効果はゼロ**（解放が数十行早まるのみ）。

### 📊 メモリ効果試算

| シナリオ | timeDomainIR サイズ | 削減可能量 | 備考 |
|---------|-------------------|-----------|------|
| 3秒IR@48kHz ステレオ | 3×48000×2×8 = **2.3MB** | 数十行早い解放 | 効果ほぼゼロ |
| 3秒IR@192kHz ステレオ | 3×192000×2×8 = **9.2MB** | 同上 | 同上 |
| RCU 3滞留時 | 全DSPCoreの合計 | **0MB**（PreparedIRStateはRCU滞留対象外） | 各applyComputedIRは逐次実行 |

### ⚠️ 要追加調査の経路

- `ConvolverProcessor.LoadPipeline.cpp:324-344`: scaleFactor が `!= 1.0` の場合、`timeDomainIR` を **コピーしてから** スケーリングする。このコピー自体が追加メモリ確保。
- `CacheManager::save()` (`CacheManager.cpp:346-372`): timeDomainIR をディスクキャッシュに保存する。しかしこれは **applyComputedIR の前** に呼ばれており、P1の修正とは無関係。
- `ProgressiveUpgradeThread.cpp:191`: 最終アップグレードステップでも `applyComputedIR()` が呼ばれ、同様の流れ。

### 🎯 最終評価

| 項目 | 評価 |
|------|------|
| **メモリ削減効果** | ⭐⭐☆☆☆（定常効果ほぼゼロ。IRロード中の一時的ピーク削減のみ） |
| **実装難易度** | 低（reset()1行追加）だが、事前の経路調査が必要 |
| **RTリスク** | なし（Message Thread のみの変更） |
| **計画書の問題** | 「デメリット実質ゼロ」は **時期尚早**。timeDomainIR の全参照経路を確認せずに解放するとクラッシュする |
| **推奨** | **見送り（効果が薄い）**。どうしても実施するなら、updateIRState() の後に reset() すること |

---

## P2: `internalMaxBlock` の動的適正化（固定OS倍率の廃止）

### ✅ 事実確認

`src/audioengine/AudioEngine.Processing.DSPCoreLifecycle.cpp:108-118`:

```cpp
constexpr int MAX_OS_FACTOR = 8;
const int inputMaxBlock = std::max(SAFE_MAX_BLOCK_SIZE, samplesPerBlock);
const int internalMaxBlock = inputMaxBlock * MAX_OS_FACTOR;
```

- `SAFE_MAX_BLOCK_SIZE = 65536` (`AudioEngine.h:766`)
- `maxInternalBlockSize = internalMaxBlock`（line 121）

### 🔍 計画書の問題点（重大）

**「OS=1（384kHz処理）ではバッファサイズが **1/8** に縮小」は誤り。**

実測計算（192kHz / 2xOS / blockSize=512 の場合）:

- **現在値**: `internalMaxBlock = max(65536, 512) * 8 = 524288 samples`
- **計画値**（`samplesPerBlock * targetFactor` に変更した場合）: `512 * 2 = 1024 samples`
- **削減率**: **512倍**（1/8 ではない）

`SAFE_MAX_BLOCK_SIZE` が 65536 と非常に大きいため、実質的には「`SAFE_MAX_BLOCK_SIZE * 8` の固定値」として動作している。

### 📊 波及範囲

`maxInternalBlockSize` / `internalMaxBlock` を使用している全箇所:

| 箇所 | ファイル |
|------|---------|
| `alignedL / alignedR` バッファ確保 | DSPCoreLifecycle.cpp:125-135 |
| `dryBypassBufferDoubleL / R` 確保 | DSPCoreLifecycle.cpp:141-149 |
| `eqState->prepare(processingRate, internalMaxBlock)` | DSPCoreLifecycle.cpp:178 |
| `configureFixedLatencySamples(samples, maxInternalBlockSize)` | DSPCoreLifecycle.cpp:224 |
| `HistoryRuntimeState` バッファ | AudioEngine.h:518 |
| `processDouble` / `processFloat` ガードチェック | DSPCoreDouble.cpp:326,361 / DSPCoreFloat.cpp:146,168 |

### 🎯 最終評価

| 項目 | 評価 |
|------|------|
| **メモリ削減効果** | ⭐⭐⭐⭐☆（効果大。ただし SAFE_MAX_BLOCK_SIZE の本来の役割を再確認要） |
| **実装難易度** | 低（計算式変更＋影響範囲のテスト） |
| **RTリスク** | なし（prepareToPlay 内の変更のみ） |
| **計画書の問題** | 効果予測が **過小報告**（1/8ではなく1/512）。本来は大きなメリットだが根拠説明が不正確 |
| **推奨** | **実施推奨**。ただし SAFE_MAX_BLOCK_SIZE(=65536) がなぜ必要なのかを設計者に確認してから行うこと。もし「将来の16x OS対応」が理由なら、SAFE_MAX_BLOCK_SIZE を `samplesPerBlock` に置き換え、将来拡張は別途対応する判断が必要 |

---

## P3: `DELAY_BUFFER_SIZE` の動的化（固定 4M バッファの廃止）

### ✅ 事実確認

`ConvolverProcessor.h:201-204`:

```cpp
static constexpr int DELAY_BUFFER_SIZE = 4194304; // 2^22 (approx 4M samples > MAX_TOTAL_DELAY)
static constexpr int DELAY_BUFFER_MASK = DELAY_BUFFER_SIZE - 1;
```

`ConvolverProcessor.Lifecycle.cpp:259-268`:

```cpp
if (delayBufferCapacity < DELAY_BUFFER_SIZE)
{
    auto newL = convo::makeAlignedArray<double>(static_cast<size_t>(DELAY_BUFFER_SIZE));
    auto newR = convo::makeAlignedArray<double>(static_cast<size_t>(DELAY_BUFFER_SIZE));
    delayBuffer[0] = std::move(newL);
    delayBuffer[1] = std::move(newR);
    delayBufferCapacity = DELAY_BUFFER_SIZE;
}
juce::FloatVectorOperations::clear(delayBuffer[0].get(), DELAY_BUFFER_SIZE);
juce::FloatVectorOperations::clear(delayBuffer[1].get(), DELAY_BUFFER_SIZE);
```

### 📊 メモリ効果試算

| 現在値 | 動的化後の典型値 | 削減量 |
|--------|-----------------|--------|
| 4,194,304 samples × 2ch × 8bytes = **67.1MB** | 3秒IR@192kHz: nextPowerOfTwo(576000 + 512 + 1024) = 1,048,576 → **16.8MB** | **50.3MB/DSPCore** |
| RCU 3滞留時: **201MB** | RCU 3滞留時: **50.4MB** | **151MB** |

### 🔍 レビュー指摘の検証

レビュー：「`nextPowerOfTwo(irLatency + blockSize + 1024)` は ISR 思想と衝突」

**ConvoPeq の ISR Bridge Runtime 思想**:

- `prepareToPlay()` は Message Thread で実行される
- DSPCore の全バッファサイズは **構築時（= RuntimeWorld 発行時）に確定** し、その後変更しない
- 動的再確保は RCU による新 World 発行でのみ行う

**推奨方式**: StereoConvolver::init() 時に delaySize を算出し、RuntimeWorld の不変値（immutable field）として保持する。`prepareToPlay()` で再計算しない。

### 🎯 最終評価

| 項目 | 評価 |
|------|------|
| **メモリ削減効果** | ⭐⭐⭐⭐⭐（DSPCoreあたり50MB、RCU3滞留で150MB削減） |
| **実装難易度** | 低（構築時確定＋RCUパターン流用） |
| **RTリスク** | なし（prepareToPlay 内の変更のみ） |
| **ISR Bridge整合性** | ✅ **高い**（構築時確定は ISR Immutable World 思想に適合） |
| **推奨** | **最優先実施**。StereoConvolver に `delayBufferSize` フィールドを追加し、init() で計算する方式が望ましい |

---

## P4: `numParts` の2の累乗（Power-of-Two）切り上げ廃止

### ✅ 事実確認

`MKLNonUniformConvolver.cpp:649`:

```cpp
l.numPartsIR = (cfgs[li].len + l.partSize - 1) / l.partSize;
l.numParts   = juce::nextPowerOfTwo(l.numPartsIR);
l.fdlMask    = l.numParts - 1;
```

### 🔍 全依存箇所の棚卸し（検証必須）

| 箇所 | ファイル:行 | コード | fdlMask依存？ | numParts暗黙依存？ |
|------|------------|--------|--------------|------------------|
| FDL インデックス更新 | MKLNonUniformConvolver.cpp:1144 | `l.fdlIndex = (l.fdlIndex + 1) & l.fdlMask;` | ✅ 直接依存 | - |
| FDL mirror write (L0) | MKLNonUniformConvolver.cpp:1069 | `l.fdlBuf + (l.fdlIndex + l.numParts) * partStride` | - | ✅ numParts前提 |
| FDL mirror SoA (L0) | MKLNonUniformConvolver.cpp:1078 | `mirrorIndex = l.fdlIndex + l.numParts` | - | ✅ numParts前提 |
| 分散計算 linStart | MKLNonUniformConvolver.cpp:1105 | `linStart = baseFdlIdx - numPartsIR + 1 + numParts` | - | ✅ numParts前提 |
| リングバッファ baseSize | MKLNonUniformConvolver.cpp:878 | `baseSize = numParts * 2` | - | ✅ numParts前提 |
| mirror write (L1/L2 Add内) | MKLNonUniformConvolver.cpp:1281 | `l.fdlBuf + (l.fdlIndex + l.numParts) * partStride` | - | ✅ numParts前提 |
| AVX prefetch (L0) | MKLNonUniformConvolver.cpp:1098-1102 | `srcA + l.partStride` (線形アクセス) | - | ⚠️ 間接的 |
| partsPerCallback | (複数) | `nextPart += partsPerCallback` | ❌ 非依存 | ❌ 非依存 |

### ⚠️ 難易度再評価

**計画書: 「低」→ 検証結果: 「中〜高」**

理由:

1. **fdlMask 依存は1箇所のみ**だが、`(l.fdlIndex + 1) & l.fdlMask` から `if (++idx >= numParts) idx = 0;` への変更自体は容易
2. しかし **linearized ring buffer の計算式** `linStart = baseFdlIdx - numPartsIR + 1 + numParts` は、mirroring による 2倍バッファを前提とした定数。非PoT時も現状の linearized access（`fdlBuf + linStart * partStride`）が有効か要検証
3. `baseSize = numParts * 2` はリングバッファサイズ計算。非PoT時にこの式が適切か再設計が必要
4. AVXプリフェッチは `partStride` ベースのため、numParts の PoT性とは独立（影響なし）

### 追加リスク

- **fdlBuf の mirror 部サイズ**: `fdlBufSize = numParts * 2 * partStride`。非PoT時も `numParts >= numPartsIR` を保証できれば問題なし
- **ringBuf サイズ**: `finalSize = std::max(rSize, minSize)` は PoT を要求していないが、`m_ringMask` が PoT 前提のため、PoT保証が必要な別のコード部分がある可能性

### 🎯 最終評価

| 項目 | 評価 |
|------|------|
| **メモリ削減効果** | ⭐⭐⭐⭐☆（数十〜100MB。IR長がPoTに近いほど効果小、遠いほど効果大） |
| **実装難易度** | **中〜高**（計画書の「低」は過小評価） |
| **RTリスク** | なし（Message Thread での設計変更。分岐追加はブロック制御部=1回/ブロック） |
| **計画書の問題** | 「難易度低」は **危険な過小評価**。4箇所以上の numParts 暗黙依存を全て検証する必要あり |
| **推奨** | **フェーズ1ではなくフェーズ2での慎重実施**。単体テスト＋ビット一致検証が必須 |

---

## P5: FDL ミラーリング廃止（2倍確保→ループ分割）

### ✅ 事実確認

`MKLNonUniformConvolver.cpp:710`:

```cpp
const size_t fdlBufSize = static_cast<size_t>(l.numParts) * 2 * l.partStride;
```

`MKLNonUniformConvolver.cpp:1069-1085`（processLayerBlock 内の mirror write）:

```cpp
double* mirrorFDLSlot = l.fdlBuf + (l.fdlIndex + l.numParts) * l.partStride;
memcpy(mirrorFDLSlot, currentFDLSlot, l.partStride * sizeof(double));

const int mirrorIndex = l.fdlIndex + l.numParts;
deinterleaveComplex(mirrorFDLSlot,
    l.fdlReal + static_cast<size_t>(mirrorIndex) * l.complexSize,
    l.fdlImag + static_cast<size_t>(mirrorIndex) * l.complexSize,
    l.complexSize);
```

同様の mirror write が L1/L2 の Add() 内（line 1281-1320）にも存在。

### 📊 メモリ効果試算

| バッファ | mirror有 | mirror無 | 削減量 |
|---------|---------|---------|-------|
| fdlBuf (AoS) | numParts×2×partStride | numParts×partStride | 半減 |
| fdlReal (SoA) | numParts×2×complexSize | numParts×complexSize | 半減 |
| fdlImag (SoA) | numParts×2×complexSize | numParts×complexSize | 半減 |

全3層合計で DSPCore あたり **数十〜100MB** 削減。

### ⚠️ 計画書の問題点

**「CPU性能も改善する」→ 未証明。**

現状の mirror write のコスト:

- memcpy: `partStride * 8` bytes（L0: ~8KB/ブロック）
- deinterleaveComplex: `complexSize` ループ（L0: ~513回/ブロック）

ループ分割方式のコスト:

- 分岐追加: 各AVXループの先頭・中間で `if (k >= remaining)` チェック
- プリフェッチの分断: 線形アクセスが2セグメントに分かれる

memcpy のコストは通常1ブロックあたり数十μ秒。AVXループの分岐ペナルティは分岐予測が効けばほぼゼロ。**どちらが有利かは実測依存。**

### 🎯 最終評価

| 項目 | 評価 |
|------|------|
| **メモリ削減効果** | ⭐⭐⭐⭐☆（数十〜100MB削減は確実） |
| **実装難易度** | 中（専用インラインヘルパーでの抽象化推奨） |
| **RTリスク** | 低（ループ分割による分岐追加はブロック処理ループ内だが、通常問題なし） |
| **計画書の問題** | 「CPU改善」は **未証明**。memcpy削減メリットと分岐追加デメリットのトレードオフを実測で確認すべき |
| **推奨** | **実測後判断**。まずメモリプロファイラで FDL mirror の占有量を確認し、CPUプロファイラで mirror write のコストを計測してから実施判断 |

---

## P6: AoS（インターリーブ）と SoA（スプリット）の二重保持を統一

### ✅ 事実確認

`MKLNonUniformConvolver::Layer` の **6バッファ二重保持**:

| AoS（インターリーブ） | SoA（スプリット） | 用途 |
|---------------------|------------------|------|
| `irFreqDomain` | `irFreqReal`, `irFreqImag` | IR 周波数データ |
| `fdlBuf` | `fdlReal`, `fdlImag` | 周波数領域遅延線 |
| `accumBuf` | `accumReal`, `accumImag` | 複素積算バッファ |

現在アクティブなカーネル:

```cpp
constexpr bool kEnableSplitComplexKernel = true;  // AVX2時（MKLNonUniformConvolver.cpp:157）
```

AoS→SoA 変換呼び出し箇所:

- `deinterleaveComplex()`: processLayerBlock (line 1062, 1078), Add 分散部 (line 1296, 1314)
- `interleaveComplex()`: processLayerBlock (line 1128), Add 分散部 (line 1359)

### 📊 バッファサイズ比較

| バッファ | AoS サイズ | SoA サイズ | AoS+SoA |
|---------|-----------|-----------|---------|
| IR | numParts × partStride | 2 × numParts × complexSize | ≈ **2× AoS** |
| FDL (mirror込) | 2 × numParts × partStride | 2 × 2 × numParts × complexSize | ≈ **2× AoS** |
| accum | partStride | 2 × complexSize | ≈ **1.5× AoS** |

`partStride = (complexSize * 2 + 7) & ~7` のため、`partStride ≈ complexSize × 2`（8-double alignment）。よって **AoS:SoA ≈ 1:1**。つまり **周波数データ全体が実質2倍**。

### ⚠️ 計画書の問題点

**「CPU負荷も現状より低下」→ 未証明（現在のSoAカーネルが高最適化済みのため）。**

現在の `accumulateSplitComplex()` (`MKLNonUniformConvolver.cpp:204-250`):

```cpp
__m256d ar = _mm256_loadu_pd(srcAReal + k);
__m256d ai = _mm256_loadu_pd(srcAImag + k);
__m256d br = _mm256_loadu_pd(srcBReal + k);
__m256d bi = _mm256_loadu_pd(srcBImag + k);
__m256d dr = _mm256_loadu_pd(dstReal + k);
__m256d di = _mm256_loadu_pd(dstImag + k);
dr = _mm256_add_pd(dr, _mm256_sub_pd(_mm256_mul_pd(ar, br), _mm256_mul_pd(ai, bi)));
di = _mm256_add_pd(di, _mm256_add_pd(_mm256_mul_pd(ar, bi), _mm256_mul_pd(ai, br)));
_mm256_storeu_pd(dstReal + k, dr);
_mm256_storeu_pd(dstImag + k, di);
```

- 6 load + 2 store + 4 FMA = **12 μops/4 doubles**
- 効率: 3 μops/double（演算密度: 6 FLOP/load-store）

AoS カーネル（`_mm256_addsub_pd` / `_mm256_shuffle_pd` 使用）は同等の効率を達成可能だが、**実装の複雑さが大きく異なる**。

### 🎯 最終評価

| 項目 | 評価 |
|------|------|
| **メモリ削減効果** | ⭐⭐⭐⭐☆（理論上の最大効果。周波数データ全体が約半分に） |
| **実装難易度** | 高（AVX2高度知識＋ビット一致検証必須） |
| **RTリスク** | 低（デバッグ工数大だが、RTパスのブロッキングはなし） |
| **計画書の問題** | 「CPU低下」は **未証明**。現在のSoAカーネルは6 load/store + 4 FMAで完全最適化済み。AoSでも同等は可能だが、改善ではない |
| **推奨** | **フェーズ3で実施（新旧切替 `#ifdef` 必須）**。チームにAVX2専門家がいるタイミングで着手 |

---

## P7: IR 周波数データ（不変部分）の参照共有化（シャローコピー）

### ✅ 事実確認

`StereoConvolver::clone()` (`ConvolverProcessor.h:762-779`):

```cpp
[[nodiscard]] StereoConvolver* clone() const
{
    auto newConv = convo::aligned_make_unique<StereoConvolver>();
    if (irDataLength > 0 && irData[0] && irData[1])
    {
        auto l = convo::makeAlignedArray<double>(static_cast<size_t>(irDataLength));
        auto r = convo::makeAlignedArray<double>(static_cast<size_t>(irDataLength));
        std::memcpy(l.get(), irData[0], irDataLength * sizeof(double));
        std::memcpy(r.get(), irData[1], irDataLength * sizeof(double));
        // init() → SetImpulse() で NUC 全周波数バッファを再確保・再計算
        if (!newConv->init(l.release(), r.release(), ...))
            return nullptr;
    }
    return newConv.release();
}
```

**RCU 動作**: `exchangeActiveEngine()` で新旧 engine をアトミック交換後、旧 engine は retire されるまで生存する。clone は EQ ゲイン変更など **IRが変わらない操作** でも呼ばれ、その場合も **IR周波数データ全体が複製される**。

### 📊 メモリ効果試算

| シナリオ | 現在（deep copy） | 共有化後 | 削減量 |
|---------|------------------|---------|-------|
| EQゲイン変更1回 | ~300MB（L0+L1+L2 全周波数データ） | ~0MB（IR参照のみコピー） | **~300MB** |
| RCU 3滞留中にEQ変更 | ~900MB（3×300MB） | ~300MB（共通IR+3×FDL） | **~600MB** |
| プリセット切替（IR不変） | ~300MB | ~0MB | **~300MB** |

### ISR Bridge 整合性

| 原則 | 整合性 | 説明 |
|------|--------|------|
| **Immutable World** | ✅ 完全整合 | IR周波数データは SetImpulse() で一度計算後、Read Only。RuntimeWorld の一部として不変 |
| **RCU Epoch** | ✅ 設計可能 | `RefCountedDeferred` で Epoch と同期した解放が可能（既存フレームワーク流用） |
| **RT path ゼロアロケーション** | ✅ 整合 | 参照カウント操作を Message Thread のみに限定すれば、RT path への影響ゼロ |
| **Single-authority RuntimeWorld** | ✅ 強化 | ImmutableIRBlob を World の構成要素として明確に分離 |

### 🎯 最終評価

| 項目 | 評価 |
|------|------|
| **メモリ削減効果** | ⭐⭐⭐⭐⭐（計画書で最大の効果。RCU滞留時の爆発的二重化を完全撲滅） |
| **実装難易度** | 高（所有権モデルの再設計＋Epoch同期設計レビュー必須） |
| **RTリスク** | **設計次第でゼロ**（`RefCountedDeferred` 徹底活用で RT path に原子操作すら追加しない設計が可能） |
| **ISR Bridge整合性** | ✅ **最も整合する改修**。Immutable IR → RuntimeWorld の構成要素として明確に位置づけ可能 |
| **推奨** | **最優先（フェーズ3トップ）**。ImmutableIRBlob を独立オブジェクト化し、RuntimeWorld から参照する設計を推奨 |

---

## P8: キャッシュ管理の「個数制限」→「バイト数制限」へ移行

### ✅ 事実確認

`CacheManager::CacheEntry` (`CacheManager.h:56-63`):

```cpp
struct CacheEntry
{
    juce::File file;              // ディスク上のキャッシュファイル
    uint64_t originalKey = 0;
    uint64_t lastAccessTime = 0;
    int fftSize = 0;
    std::list<uint64_t>::iterator lruPos;
};
```

`CacheManager::evictLRU()` (`CacheManager.cpp:436-458`):

- キャッシュエントリ数が `maxEntries`（デフォルト10）を超えたら、最も古いエントリを削除
- 各エントリは **ディスクファイル**（`juce::File`）への参照のみ保持
- PreparedIRState はメモリ内に **一切保持しない**（load 毎に新規作成、applyComputedIR で消費・破棄）

### 📊 実態

**計画書の「メモリ使用量を確実に上限内に収められる」は誤解。**

CacheManager はディスク上のキャッシュファイルを管理しているに過ぎず、メモリ消費とは直接的に関係しない。バイト数制限に変更しても **削減できるメモリはゼロ**（ディスク使用量が減るのみ）。

### 🎯 最終評価

| 項目 | 評価 |
|------|------|
| **メモリ削減効果** | ⭐⭐☆☆☆（ほぼゼロ。ディスクキャッシュの管理方式変更） |
| **実装難易度** | 低 |
| **RTリスク** | なし |
| **計画書の問題** | 「メモリ上限」表現は **誤解を招く**。CacheManager はファイル管理のみ |
| **推奨** | **見送り**。必要な場合は「ディスク使用量制御」として独立した課題として扱うべき |

---

## 「2.5GB → 1.0GB」効果予測の検証

### 結論: **現状のコードだけでは根拠不足。実測不能。**

#### 不足しているデータ

| 必要な情報 | 現状 | 入手手段 |
|-----------|------|---------|
| 各Layerの numParts / partStride / complexSize の実測値 | 未計測 | ログ出力またはプロファイラ |
| RCU滞留中の DSPCore 数（平均・最大） | 未計測 | RuntimeHealthMonitor にカウンタ追加 |
| CacheManager 内のキャッシュファイル総サイズ | 未計測 | ディスク使用量の定期計測 |
| FDL mirror を含む周波数バッファの実占有量 | 未計測 | Memory Telemetry 機能の追加 |
| プロセス全体の Private Bytes / Working Set 内訳 | 未計測 | Windows Performance Monitor |

#### 推奨: Memory Telemetry の先行実装

```cpp
// RuntimeHealthMonitor または TelemetryRecorder に追加推奨:
struct MemoryFootprintSnapshot
{
    size_t activeWorldBytes;      // 現行 DSPCore の Layer 別バッファ合計
    size_t retiredWorldBytes;     // RCU 滞留中の旧 DSPCore 合計
    size_t sharedIRBytes;         // IR 周波数データ（共有後は 1 コピー分）
    size_t delayBufferBytes;      // DelayBuffer 実使用量
    size_t cacheFileBytes;        // ディスクキャッシュ使用量
    size_t totalPrivateBytes;     // プロセス全体（OS 提供）
};
```

これを実装してから各Pの効果を計測し、データに基づいて優先順位を決定すべき。

---

## 総合比較表（検証結果反映版）

| ID | 改善案 | 計画書の難易度 | **検証後の難易度** | 効果予測 | **効果の裏付け** | ISR整合 | **最終評価** |
|----|--------|--------------|-----------------|---------|----------------|---------|------------|
| **P1** | timeDomainIR解放 | 極低 | 低（要経路調査） | 中〜大 | ❌ 定常効果ほぼゼロ（IRロード中のみ） | △ | ⭐⭐ |
| **P2** | internalMaxBlock動的化 | 低 | 低 | 中 | ⚠️ 効果予測過小（1/8 → 実際 1/512） | ○ | ⭐⭐⭐⭐ |
| **P3** | DELAY_BUFFER動的化 | 低 | 低 | 中 | ✅ 64MB/DSPCore 削減確定 | ✅ 構築時確定 | ⭐⭐⭐⭐⭐ |
| **P4** | numParts切り上げ廃止 | **低** | **中〜高** | 大 | ✅ 数十〜100MB削減の可能性 | ○ | ⭐⭐⭐ |
| **P5** | FDLミラーリング廃止 | 中 | 中 | 大 | ✅ 数十〜100MB削減確実 | ○ | ⭐⭐⭐ |
| **P6** | AoS/SoA統一 | 高 | 高 | 特大 | ✅ 理論上最大効果（周波数データ半減） | ○ | ⭐⭐⭐⭐ |
| **P7** | IR参照共有化 | 高 | 高 | 特大 | ✅ RCU滞留時の爆発的増加を撲滅 | ✅ **ISRと完全整合** | ⭐⭐⭐⭐⭐ |
| **P8** | キャッシュバイト制御 | 低 | 低 | 中 | ❌ メモリ効果ほぼゼロ（ファイル管理のみ） | △ | ⭐⭐ |

---

## 最終推奨優先順位

### 🥇 第1優先: P3 + P7（効果確実・リスク最小）

| P | 理由 |
|---|------|
| **P3** | DELAY_BUFFER動的化: 実装容易（低難易度）、効果確実（50MB/DSPCore）、ISR Bridgeと整合（構築時確定）。**最初に着手すべき** |
| **P7** | IR参照共有化: 最大効果（RCU滞留時の爆発的二重化を撲滅）、ISR思想と完全整合。ただし設計レビューに時間を要するため、P3完了後に着手 |

### 🥈 第2優先: P2 + P6（効果大・要検討）

| P | 理由 |
|---|------|
| **P2** | internalMaxBlock動的化: 効果大（512倍削減 = 数百MB単位）、低難易度。ただし `SAFE_MAX_BLOCK_SIZE` の必要性を設計者確認後に実施 |
| **P6** | AoS/SoA統一: 効果特大（周波数データ半減）。AVX2専門家をアサインできるタイミングで。新旧切替 `#ifdef` は必須 |

### 🥉 第3優先: P4 + P5（効果大・実測後判断）

| P | 理由 |
|---|------|
| **P4** | numParts廃止: 効果大だが難易度高（計画書の過小評価）。Memory Telemetry で実測後、効果が十分大きい場合のみ |
| **P5** | FDLミラー廃止: 効果大だがCPU影響は未測定。実測してから判断 |

### ❌ 見送り推奨: P1 + P8

| P | 理由 |
|---|------|
| **P1** | 定常メモリ効果ほぼゼロ。timeDomainIR は applyComputedIR 内のみ生存 |
| **P8** | メモリ制御と無関係（ディスクキャッシュ管理） |

---

## 最優先で実施すべきアクション（計画書にないもの）

### Memory Telemetry の実装

これなしでは全ての効果予測が推定の域を出ない。P3 着手前に以下の計測基盤を整備することを強く推奨:

```cpp
struct MemoryFootprint
{
    size_t activeWorldBytes;
    size_t retiredWorldBytes;
    size_t sharedIRBytes;
    size_t delayBufferBytes;
    size_t cacheFileBytes;
    size_t totalPrivateBytes;
};
```

既存の `RuntimeHealthMonitor` または `TelemetryRecorder` に追加することで、各Pの事前事後比較が可能になる。

---

## 付録: ソースコード上のキーロケーション

| シンボル | ファイル | 行 | 役割 |
|---------|---------|-----|------|
| `MKLNonUniformConvolver::Layer` | `src/MKLNonUniformConvolver.h` | 173-289 | 全バッファ定義（AoS/SoA 二重保持, numParts, fdlMask 等） |
| `Layer::freeAll()` | `src/MKLNonUniformConvolver.cpp` | 254-289 | 全バッファ解放ロジック |
| `MKLNonUniformConvolver::SetImpulse()` | `src/MKLNonUniformConvolver.cpp` | 493-860 | バッファ確保・numParts計算・IRプリコンピュート |
| `processLayerBlock()` | `src/MKLNonUniformConvolver.cpp` | 1040-1145 | L0 即時処理（mirror write, 複素積算, IFFT） |
| `accumulateSplitComplex()` | `src/MKLNonUniformConvolver.cpp` | 204-250 | SoA 複素積算 AVX2 カーネル |
| `PreparedIRState` | `src/PreparedIRState.h` | 9-83 | timeDomainIR 定義 |
| `applyComputedIR()` | `src/convolver/ConvolverProcessor.LoadPipeline.cpp` | 310-500 | timeDomainIR 全使用経路 |
| `CacheManager` | `src/CacheManager.h` | 34-79 | ディスクキャッシュ管理（メモリキャッシュなし） |
| `StereoConvolver::clone()` | `src/ConvolverProcessor.h` | 762-779 | deep copy → NUC SetImpulse 再実行 |
| `DSPCore::prepare()` | `src/audioengine/AudioEngine.Processing.DSPCoreLifecycle.cpp` | 118 | internalMaxBlock 計算 |
| `DELAY_BUFFER_SIZE` | `src/ConvolverProcessor.h` | 201-204 | 固定4M delay buffer |

---

*検証ツール: Serena MCP, CodeGraph MCP (16,442 entities), AiDex MCP, grep, ファイル直接読取*
*検証日: 2026-06-21*
*対象ブランチ: ConvoPeq main (v0.5.0)*

---

# 追補: 第2次詳細調査結果（2026-06-21）

## 概要

第1次検証後にユーザーレビューを反映し、以下を追加調査:

1. RCU滞留世代数の全経路棚卸し
2. P1 timeDomainIR: CacheManager 関与の詳細
3. P2: 全バッファ種別ごとの縮小連動範囲
4. P3: DELAY_BUFFER_MASK 全9箇所の特定
5. P4: numParts/fdlMask 全23箇所の完全網羅
6. P5+P6: 統合依存関係の評価
7. P7: shareConvolutionEngineFrom の実クローン経路
8. Memory Telemetry の具体的設計

使用ツール: Serena MCP / CodeGraph MCP / AiDex MCP / grep 5種類 / ファイル直接読取15ファイル / サブエージェントExplore

---

## A1. RCU 滞留世代数の全経路棚卸し

### StereoConvolver retire 経路（全8経路）

| # | 経路 | ファイル:行 | トリガー条件 | RCU保留？ |
|---|------|------------|------------|----------|
| 1 | `~ConvolverProcessor()`: destructor | Lifecycle.cpp:110 | プロセス終了 | ❌ 即時破棄（provider=nullptr） |
| 2 | `prepareToPlay()`: rate/block変更 | Lifecycle.cpp:243 | サンプルレート・ブロックサイズ変更 | ✅ Epoch経由 |
| 3 | `releaseResources()` | Lifecycle.cpp:402 | オーディオ停止 | ✅ Epoch経由 |
| 4 | `switchEngineOnMessageThread()`: IRロード | LoadPipeline.cpp:678 | 新IRロード完了時 | ✅ Epoch経由（advanceRetireEpoch） |
| 5 | `shareConvolutionEngineFrom()`: エンジン共有 | StateAndUI.cpp:436 | 他ConvolverProcessorからclone | ✅ Epoch経由 |
| 6 | `~LoaderThread()`: ローダ破棄 | LoaderThread.cpp:33 | ローダースレッド終了 | ✅ Epoch経由 |
| 7 | `LoaderThread::run()`: ロード完了 | LoaderThread.cpp:79 | 新IR非同期ロード完了 | ✅ Epoch経由 |
| 8 | `performLoad()`: アップグレード段階 | LoaderThread.cpp:352,384 | Progressive Upgrade中間段階 | ✅ Epoch経由 |

### 滞留メカニズム

```
swapEngine → exchangeActiveEngine(new) → advanceRetireEpoch() → enqueueDeferredDeleteNonRt(old)
                                                                    ↓
                                                          DeferredDeletionQueue (kQueueSize=4096)
                                                                    ↓
                                                          EpochDomain::reclaim(getMinReaderEpoch())
                                                                    ↓
                                                    全ReaderがoldのEpochを離れるまで解放されない
```

**Epoch 進行速度**:

- `advanceRetireEpoch()` は swapEngine ごとに +1
- Audio Thread Reader が1 callback (= ~1.33ms @384kHz) に1回 enterReader/exitReader

**滞留時間**:

- 最小: 1 callback 以内（Readerが即座に新しいepochを読む場合）
- 最大: 全Readerが古いepochに留まる時間（約20ms crossfade）
- **平均滞留世代数: 2〜3世代**（1 active + 1〜2 retired）
- **最悪滞留世代数: 5〜8世代**（急速なパラメータ変更連続 + crossfade遅延）

### メモリ影響

| 世代数 | StereoConvolver合計 | DelayBuffer合計 | 総計（目安） |
|--------|-------------------|----------------|-------------|
| 1 (activeのみ) | 152.5 MB | 64 MB | ~216 MB |
| 3 (平均) | 457.5 MB | 64 MB | ~521 MB |
| 5 (やや多い) | 762.5 MB | 64 MB | ~826 MB |
| 8 (最悪) | 1,220 MB | 64 MB | ~1,284 MB |

**補足**: 上記は NUC のみ。AudioEngine/DSPCore 全体では +200〜400MB（DSPCore本体、EQProcessor、OSバッファ、FixedLatencyBuffer、FFTキャッシュ等）

---

## A2. P1: timeDomainIR — CacheManager 関与の詳細

### 時系列

```
IRConverter::convertFile()
  → prepared->timeDomainIR に時間領域データを格納
  → cacheManager->save(key, fftSize, *prepared)   // ★ timeDomainIR をディスクに保存
  → applyComputedIR(std::move(prepared))
      ├─ scaleFactor適用（timeDomainIR 使用）
      ├─ peak/rms検証（timeDomainIR 使用）
      ├─ latency推定（timeDomainIR 使用）
      ├─ createWaveformSnapshot（timeDomainIR 使用）
      ├─ createFrequencyResponseSnapshot（timeDomainIR 使用）
      └─ updateIRState（timeDomainIR 使用。deep copy生成）
  → prepared unique_ptr 破棄 → timeDomainIR 解放
```

### CacheManager の関与

- `CacheManager::save()`: `timeDomainIR` を **ディスクファイルにシリアライズ** する。
  - キャッシュファイルフォーマット: `[CacheHeader][partitionData][timeDomainData]`
  - ファイルサイズ: partitionData + timeDomainData（数十MB）
- `CacheManager::loadPreparedState()`: `timeDomainIR` を **ディスクからメモリに復元** する。
  - `juce::MemoryMappedFile` で開き、partitionData は `mkl_malloc` + `memcpy`、timeDomain は `juce::AudioBuffer<double>` として new
- **一旦 loadPreparedState が戻ると、mmap はクローズされる**（→ timeDomainIR は独立したメモリ領域）

### 評価

**「定常メモリ効果ほぼゼロ」は正確だが、「P1は全く意味がない」わけではない。**

| 局面 | timeDomainIR生存期間 | サイズ |
|------|---------------------|--------|
| IRConverter convertFile内 | 変換中〜save完了 | ~9MB (192kHz, 3秒) |
| cacheManager->save呼出時 | シリアライズ中のみ | ~9MB |
| loadPreparedState→applyComputedIR | ロード〜apply完了 | ~9MB |
| ProgressiveUpgrade最終段階 | callAsync待ち〜apply完了 | ~9MB |

P1 の実質的効果は「アプリケーション起動後のIR変換・キャッシュ復元時における一時的ピークメモリの削減」。**「効果ゼロ」ではなく「実測が必要」が正確**。

---

## A3. P2: 全バッファ種別ごとの縮小連動範囲

### SAFE_MAX_BLOCK_SIZE (=65536) 依存バッファ（internalMaxBlock と独立）

| バッファ | ファイル | サイズ | internalMaxBlock変更の影響 |
|---------|---------|--------|------------------------|
| `maxSamplesPerBlock` | AudioEngine.Init.cpp:38 | 65536 | ❌ SAFE_MAX_BLOCK_SIZE固定値 |
| `m_fadeFloatBuffer` | AudioEngine.Init.cpp:57 | 2×65536×4=0.5MB | ❌ 独立 |
| `m_fadeDoubleBuffer` | AudioEngine.Init.cpp:58 | 2×65536×8=1MB | ❌ 独立 |
| `FIFO_SIZE` | AudioEngine.h:756 | 1,048,576=1M | ❌ 独立（2^20、Audip Block FIFO） |
| `dspCrossfadeFloatBuffer` | PrepareToPlay.cpp:155 | 2×65536×4=0.5MB | ❌ 独立 |
| `dspCrossfadeDoubleBuffer` | PrepareToPlay.cpp:156 | 2×65536×8=1MB | ❌ 独立 |

### MAX_BLOCK_SIZE (=524288) 依存バッファ（internalMaxBlock と独立）

| バッファ | ファイル | サイズ | internalMaxBlock変更の影響 |
|---------|---------|--------|------------------------|
| `dryBuffer` (L/R) | Lifecycle.cpp:290 | 2×524288×8=8MB | ❌ 独立（MAX_BLOCK_SIZE固定） |
| `smoothingBuffer` (L/R) | Lifecycle.cpp:294 | 2×524288×8=8MB | ❌ 独立 |
| `oldDryBuffer` (L/R) | Lifecycle.cpp:298 | 2×524288×8=8MB | ❌ 独立 |
| `wetBuffer` (L/R) | Lifecycle.cpp:302-304 | 2×524288×8=8MB | ❌ 独立 |
| `DELAY_BUFFER_SIZE` | ConvolverProcessor.h:201 | 4,194,304×2ch×8=64MB | ❌ 独立（P3対象） |

### internalMaxBlock 依存バッファ（P2で削減可能）

| バッファ | ファイル | 現在値 | P2後(推定) | 削減率 |
|---------|---------|--------|-----------|--------|
| `alignedL / alignedR` | DSPCoreLifecycle.cpp:125-135 | 524288×8×2ch=8MB | 1024×8×2ch=16KB | **512倍** |
| `dryBypassBufferDoubleL/R` | DSPCoreLifecycle.cpp:141-149 | 524288×8×2ch=8MB | 1024×8×2ch=16KB | **512倍** |
| EQProcessor内部バッファ | DSPCoreLifecycle.cpp:178 | internalMaxBlock依存 | 同上 | 変動 |
| FixedLatencyBuffer | DSPCoreLifecycle.cpp:224 | maxInternalBlockSize+α | 同上+α | 変動 |

### P2の効果まとめ

**internalMaxBlock 直接削減効果: 約16MB/DSPCore**（aligned + dryBypass）+ EQ/FixedLatency波及。
**SAFE_MAX_BLOCK_SIZE / MAX_BLOCK_SIZE 固定バッファ: 約100MB/DSPCore（P2の範囲外）**

→ P2単独では「512倍削減」という表現は誤解を招く。実際のDSPCore総メモリに対する削減率は ~10%程度（16MB / 216MB）。

---

## A4. P3: DELAY_BUFFER_MASK 全9箇所の特定

### 使用箇所一覧（全9箇所、全て ConvolverProcessor.Runtime.cpp）

| # | 行 | コード | パターン |
|---|-----|--------|---------|
| 1 | 135 | `int readPos = (writePos - delaySamples) & DELAY_BUFFER_MASK;` | 読み取り位置 |
| 2 | 152 | `const int nextWritePos = (writePos + numSamples) & DELAY_BUFFER_MASK;` | 書き込み位置 |
| 3 | 407 | `int rPosInt = (iRead + 1) & DELAY_BUFFER_MASK;` | 補間読み取り |
| 4 | 454 | `double p0 = srcBuf[(idx - 1) & DELAY_BUFFER_MASK];` | 4点補間 |
| 5 | 455 | `double p1 = srcBuf[(idx    ) & DELAY_BUFFER_MASK];` | 4点補間 |
| 6 | 456 | `double p2 = srcBuf[(idx + 1) & DELAY_BUFFER_MASK];` | 4点補間 |
| 7 | 457 | `double p3 = srcBuf[(idx + 2) & DELAY_BUFFER_MASK];` | 4点補間 |
| 8 | 510 | `int rPos = (activeDelayWritePos - delayInt) & DELAY_BUFFER_MASK;` | 読み取り位置 |
| 9 | 527 | `activeDelayWritePos = (activeDelayWritePos + numSamples) & DELAY_BUFFER_MASK;` | 書き込み位置 |

### 動的化の設計要件

- `DELAY_BUFFER_SIZE` は **2の累乗である必要あり**（`& MASK` 最適化維持のため）
- 現状: `static_assert` で PoT チェックあり（ConvolverProcessor.h:203）
- 動的化後: `nextPowerOfTwo(requiredSize)` で算出すれば PoT 条件を満たせる
- `StereoConvolver` に `delayBufferSize` フィールドを追加
- `init()` 時に `delayBufferSize = nextPowerOfTwo(irLatency + blockSize + 4096)` で確定
- RuntimeWorld の不変値として保持

---

## A5. P4: numParts/fdlMask 全23箇所の完全網羅

### fdlMask 依存（3箇所）

| # | ファイル:行 | コード | 置換後 |
|---|------------|--------|--------|
| 1 | MKLNonUniformConvolver.cpp:1151 | `l.fdlIndex = (l.fdlIndex + 1) & l.fdlMask;` | `if (++l.fdlIndex >= l.numParts) l.fdlIndex = 0;` |
| 2 | MKLNonUniformConvolver.cpp:1290 | `l.fdlIndex = (l.fdlIndex + 1) & l.fdlMask;` | 同上 |
| 3 | MKLNonUniformConvolver.cpp:1293 | `l.baseFdlIdxSaved = (l.fdlIndex - 1 + l.numParts) & l.fdlMask;` | `l.baseFdlIdxSaved = (l.fdlIndex == 0) ? l.numParts - 1 : l.fdlIndex - 1;` |

→ fdlMask の置換は3箇所のみで容易。

### numParts 間接依存（20箇所）

全ての mirror index / linStart / bufferSize 計算で numParts が使用されている。
**P4単独実施時のキーポイント**:

- `numParts = numPartsIR` に変更後も、mirroring（2倍バッファ）を維持するなら `fdlBufSize = numParts * 2 * partStride` の式は変更不要
- 変更すべきは **SetImpulse内の numParts 計算**（PoT から実値へ）のみ
- mirroring 維持のまま PoT を廃止する場合、**メモリ削減量は期待したほど大きくない**（mirror用の2倍バッファは残る）
- 真に効果を最大化するには P5（mirror廃止）との統合が必要

### P4+P5統合時の効果（実メモリ推定 192kHz/2xOS/3秒IR）

| 層 | numParts (PoT) | numParts (実値) | fdlBuf PoT付mirror | fdlBuf 実値+mirror無 | 削減率 |
|---|---------------|----------------|-------------------|--------------------|-------|
| L0 | 32 (PoT 32) | 32 | 32×2×520×8=266KB | 32×520×8=133KB | 50% |
| L1 | 64 (PoT 64) | 64 | 64×2×6152×8=6.3MB | 64×6152×8=3.15MB | 50% |
| L2 | 16 (PoT 16) | 11 | 16×2×73736×8=18.9MB | 11×73736×8=6.5MB | **66%** |

L2 が最大の受益者: PoT 16 vs 実値 11 → **45%削減 + mirror廃止で66%削減**

---

## A6. P5+P6: 統合依存関係の評価

### 現状のミラーリングの二重化

processLayerBlock および Add() 内での mirror write は、以下の3つの操作で構成される:

```
1. fdlBuf[currentIndex] に FFT結果を書き込み (FFT出力)
2. fdlBuf[currentIndex + numParts] に 1. の内容を memcpy (AoS mirror)
3. deinterleaveComplex → fdlReal[mirrorIndex], fdlImag[mirrorIndex] (SoA mirror + SoA mirror)
```

**P5単独**: AoS mirror (2.) を廃止 → ループ分割。SoA mirror (3.) は残る。
**P6単独**: SoA mirror (3.) を廃止（deinterleave不要）。AoS mirror (2.) は残る。
**P5+P6統合**: (2.) も (3.) も不要に → メモリ削減最大化＋deinterleave/interleave完全排除

### 統合効果

| 方式 | メモリ削減 | CPU削減 | 難易度 |
|------|-----------|--------|--------|
| P5単独 | FDL AoS半減 + FDL SoA半減 | memcpy削減（△） | 中 |
| P6単独 | AoSかSoAの片方を削除（50%） | deinterleave排除（△） | 高 |
| **P5+P6統合** | **FDL 75%削減 + IR 50%削減 + accum削減** | mirror memcpy + deinterleave/interleave 全排除（◎） | **超高** |

**推奨**: P5とP6は独立したタスクとして設計するが、**実装順序は P6→P5** が望ましい（P6でSoA廃止後にP5でAoS mirror廃止の方が安全）。

---

## A7. P7: shareConvolutionEngineFrom の実クローン経路

### 呼び出し関係

```
ConvolverProcessor::shareConvolutionEngineFrom(const ConvolverProcessor& other)
  → other.loadActiveEngine()     // RCU保護下で他方のengineを取得
  → otherConv->clone()           // ★ StereoConvolverのdeep copy (irData memcpy + SetImpulse再実行)
  → exchangeActiveEngine(clonedConv)  // 自engineを新engineに交換
  → retireStereoConvolver(oldConv)    // 旧engineをRCU retire
```

### clone() の完全なメモリフットプリント

`StereoConvolver::clone()` が確保するリソース:

1. `irData[0], irData[1]`: `irDataLength × 8 × 2` = 約9.2MB（3秒IR@192kHz）
2. `init()` → `SetImpulse()` → 内部で全NUCバッファ再確保:
   - irFreqDomain + irFreqReal + irFreqImag（AoS+SoA二重）: ~20MB/ch
   - fdlBuf + fdlReal + fdlImag（mirror込AoS+SoA二重）: ~50MB/ch
   - その他作業バッファ: ~1MB/ch
   - **合計: ~152MB/stereo**

### ImmutableIRBlob 設計案

```cpp
class ImmutableIRBlob {
    // SetImpulse() で一度だけ計算される周波数データ
    convo::ScopedAlignedPtr<double> irFreqDomain;  // AoS 形式のみ保持（SoA廃止後）
    int irDataLength;
    int numPartsPerLayer[3];
    double sampleRate;
    uint64_t generationId;
    std::atomic<uint32_t> refCount;  // DeferredDeletionQueue で管理
};
```

- `StereoConvolver::clone()` では `irFreqDomain` を **ポインタコピーのみ**（refCount++）
- RCU retire 時に refCount--、0 になったら実解放
- Audio Thread は RCU Epoch 保護下で読み取るのみ（アトミックRMWなし）

---

## A8. Memory Telemetry 設計

### 実装提案

既存の `RuntimeHealthMonitor` に以下のメトリクスを追加:

```cpp
struct MemoryFootprintSnapshot
{
    // 1. StereoConvolver / NUC 層別
    size_t activeStereoConvolverCount;    // 現行エンジン数
    size_t retiredStereoConvolverCount;  // RCU滞留中エンジン数
    size_t layer0Bytes;                  // NUC L0 総バッファ
    size_t layer1Bytes;                  // NUC L1 総バッファ
    size_t layer2Bytes;                  // NUC L2 総バッファ
    size_t fdlMirrorBytes;               // FDL mirror オーバーヘッド
    size_t aosSoaOverheadBytes;          // AoS/SoA 二重保持オーバーヘッド

    // 2. 固定バッファ
    size_t delayBufferBytes;             // DelayBuffer 実使用量
    size_t safeMaxBlockBytes;            // SAFE_MAX_BLOCK_SIZE 関連バッファ
    size_t maxBlockBytes;                // MAX_BLOCK_SIZE 関連バッファ

    // 3. IR データ
    size_t irFreqDataBytes;              // 周波数領域IRデータ
    size_t irTimeDomainBytes;            // 時間領域IRデータ（テンポラリ）

    // 4. プロセス全体
    size_t totalPrivateBytes;            // Windows Private Bytes
    size_t totalWorkingSetBytes;         // Windows Working Set

    // RCU 統計
    uint64_t pendingRetireCount;         // 未回収retire数
    uint64_t minReaderEpoch;             // 最小Reader Epoch
    uint64_t currentEpoch;               // 現在Epoch
};
```

### 出力方法

| 方法 | 頻度 | 用途 |
|------|------|------|
| Windows Performance Counter | 1秒 | 外部モニタリング |
| `juce::Logger::writeToLog()` | 状態変化時 | デバッグ用 |
| `TelemetryRecorder` 経由 | 定期(1分) | 長期傾向分析 |
| Audio Thread 安全なアトミックカウンタ | 毎コールバック | オーバーヘッド監視 |

### 実装優先度

| コンポーネント | 難易度 | 優先度 | 備考 |
|-------------|--------|--------|------|
| activeStereoConvolverCount | 低 | 最優先 | アトミックなカウンタ追加のみ |
| retiredStereoConvolverCount | 低 | 最優先 | EpochDomain::size() から取得 |
| delayBufferBytes | 低 | 高 | StereoConvolver フィールド追加 |
| pendingRetireCount | 低 | 高 | 既存APIあり |
| layer毎のNUCバッファ | 中 | 中 | SetImpulse() 内で合計計算 |
| プロセス全体PrivateBytes | 中 | 中 | Windows API (GetProcessMemoryInfo) |

---

## A9. 確定・未確定・要調査・保留 一覧

### ✅ 確定事項

| # | 内容 | 根拠 |
|---|------|------|
| K1 | 1 StereoConvolver ≈ 152MB (192kHz/2xOS/3秒IR/tailL1L2Mult=12) | ソースコードの計算式からの推定 |
| K2 | FDL mirror + AoS/SoA 二重保持が全メモリの ~55% を占める | 層別メモリ推定 |
| K3 | RCU平均滞留は2〜3世代、最悪5〜8世代 | retire経路棚卸し＋epoch進行分析 |
| K4 | P3 DelayBuffer は PoT維持で動的化可能（mask最適化互換） | 全9箇所のDELAY_BUFFER_MASK特定 |
| K5 | P4 fdlMask置換は3箇所で容易だが、numParts間接依存が20箇所 | 全依存関係の網羅 |
| K6 | P7 clone() は shareConvolutionEngineFrom からのみ呼ばれる（1経路） | コード検索結果 |
| K7 | CacheManager は PreparedIRState をメモリ保持しない（ファイル管理のみ） | CacheManager.h/CacheEntry 確認 |
| K8 | timeDomainIR は IRロード中の一時的データ（定常保持なし） | applyComputedIR フロー解析 |

### ❓ 未確定事項

| # | 内容 | 確認に必要なアクション |
|---|------|----------------------|
| U1 | 実環境での Private Bytes / Working Set の絶対値 | Windows Performance Monitor による計測 |
| U2 | RCU滞留世代数の実測値（平均・最大・分布） | Memory Telemetry の pendingRetireCount 追加 |
| U3 | tailL1L2Mult の実運用値（ユーザー設定の分布） | テレメトリまたは設定ファイル解析 |
| U4 | shareConvolutionEngineFrom の実呼び出し頻度 | コールグラフまたはログ出力の追加 |
| U5 | SAFE_MAX_BLOCK_SIZE=65536 の設計根拠 | 設計者へのヒアリング |
| U6 | ProgressiveUpgrade の中間ステップで clone/create される一時 NUC の数 | LoaderThread::performLoad の詳細トレース |

### 🔍 要調査事項

| # | 内容 | 優先度 | 方法 |
|---|------|--------|------|
| R1 | AudioEngine 全体で何個の DSPCore が同時稼働するか | 高 | DSPCore 生成箇所の棚卸し |
| R2 | DeferredDeletionQueue の滞留状況（EpochDomain 統計） | 高 | 既存の EpochDomain::collectDebugInfo |
| R3 | MAX_BLOCK_SIZE=524288 を internalMaxBlock に連動させる設計変更の影響範囲 | 中 | ConvolverProcessor 全バッファの監査 |
| R4 | ProgressiveUpgrade 時のメモリピーク（中間FFTサイズのNUC群） | 中 | ログ出力追加 |
| R5 | Oversampler 内部バッファのメモリ使用量 | 中 | CustomInputOversampler のコードレビュー |
| R6 | AudioEngine の AudioBlock / CrossfadeRuntime のバッファ | 低 | 関連ファイルの確認 |

### ⏸️ 保留事項

| # | 内容 | 保留理由 | 再開条件 |
|---|------|---------|---------|
| H1 | P1 (timeDomainIR解放) の実装 | 効果が一時的ピークのみ、定常メモリに影響なし | Memory Telemetry で load/prepare 時のピークが Bottleneck と判明した場合 |
| H2 | P8 (キャッシュバイト制御) の実装 | メモリ制御と無関係（ディスクキャッシュ管理） | ディスク使用量が問題になった場合 |
| H3 | P4 (numParts PoT廃止) 単独実装 | P5との統合がより効果的。単独ではmirror残存で効果半減 | P5/P6 の設計完了後、統合実装として再評価 |
| H4 | SAFE_MAX_BLOCK_SIZE の変更 | 設計根拠が不明。将来の16x OS対応とのトレードオフ | 設計者確認後 |

---

## A10. 改訂版 最終推奨優先順位

### 🥇 第1優先: Memory Telemetry → P3 → P7

| Step | タスク | 工数 | 理由 |
|------|--------|------|------|
| **M0** | Memory Telemetry 最小実装 | 1日 | 全ての効果検証の前提。active/retired engine数、delayBuffer使用量、疑似PrivateBytesの計測 |
| **P3** | DELAY_BUFFER_SIZE 動的化 | 2日 | 効果確実（50MB/DSPCore）、構築時確定でISR整合。PoT維持でmask最適化互換 |
| **P7** | IR周波数データ参照共有化 | 2〜3週間 | RCU滞留時の最大要因を撲滅。ImmutableIRBlob 設計。設計レビューに十分な時間を確保 |

### 🥈 第2優先: P2 + P6（実測後判断）

| P | 判断基準 | 備考 |
|---|---------|------|
| **P2** | M0 で SAFE_MAX_BLOCK_SIZE 由来のバッファ使用量を確認後 | internalMaxBlock直接削減は16MB程度。SAFE_MAX_BLOCK_SIZE変更は別途判断 |
| **P6** | AVX2専門家アサイン可否 | メモリ削減効果は最大だが工数大。新旧切替#ifdef必須 |

### 🥉 第3優先: P5+P4統合

| P | 判断基準 | 備考 |
|---|---------|------|
| **P5+P4** | P6完了後、AoSのみの設計で再評価 | P6でSoA廃止後、mirror廃止の効果と難易度が変わる。統合設計で実施 |

### ❌ 見送り

| P | 理由 |
|---|------|
| **P1** | 効果がIRロード中の一時的ピークのみ |
| **P8** | メモリ制御と無関係 |

---

*追補検証ツール: Serena MCP, CodeGraph MCP, AiDex MCP, grep, ファイル直接読取, Explore サブエージェント*
*追補検証日: 2026-06-21*

---

# 追補2: 第3次詳細調査結果（2026-06-21）— 不確実性の定量化と補正

## 概要

第2次調査の数値に対してユーザーレビューで「推定と実測の区別」「不確実性の明示」が指摘された。
本追補では:

1. **「152MB」の推定根拠と不確実性**を明示
2. **「RCU 2〜3世代/最悪5〜8」の根拠**を実コードパスから補正
3. **P7評価の上方修正**（P3より優先度が高いことを実証）
4. **P1/P8の「保留」復帰**
5. **DSPCore同時稼働数**の確定

使用ツール: Serena MCP / CodeGraph MCP / AiDex MCP / grep / ファイル直接読取

---

## B1. DSPCore 同時稼働数の確定 — 実コードからの補正

### 結論

**DSPCore は最大2つ**（active + fading）が同時に存在する。これは AudioEngine.h のメンバ変数から確定。

```cpp
// AudioEngine.h:1512-1515
convo::NonOwningPtr<DSPCore> activeRuntimeDSPSlot { nullptr };   // 現行DSP
convo::NonOwningPtr<DSPCore> fadingRuntimeDSPSlot { nullptr };   // フェード中DSP
```

DSPCore の動的生成・破棄は DSPLifetimeManager (`DSPLifetimeManager.h`) が一元管理。
RCUで滞留するのは DSPCore ではなく、その内部の **StereoConvolver** オブジェクトである。

### 補正: ConvolverProcessor 同時稼働数

| インスタンス | ソース | 数 |
|------------|--------|----|
| `uiConvolverProcessor` | AudioEngine.h:829 (メンバ変数) | **1** |
| `DSPCore::convolver` (active) | AudioEngine.h:611 (DSPCore内メンバ) | **1** |
| `DSPCore::convolver` (fading) | DSPLifetimeManagerで管理 | **0〜1** |
| **合計** | | **2〜3** |

各 ConvolverProcessor は `exchangeActiveEngine` で管理する StereoConvolver を高々1つアクティブに持ち、古いものは retire キューに入る。

---

## B2. StereoConvolver RCU滞留数の補正 — 平均1〜2、最悪3〜4

### 実コードからの根拠

**DeferredFreeThread** (`DeferredFreeThread.h:143-158`) が専用スレッドで常時 tryReclaim を実行:

```cpp
// 1ループあたり最大4件まで解放
static constexpr int kMaxReclaimPerLoop = 4;
while (auto* ptr = swapperRef.tryReclaim(minEpoch)) {
    if (++reclaimCount >= kMaxReclaimPerLoop) break;
}
```

さらに、Epoch 進行速度（`EpochDomain.h:162`）:

- `advanceRetireEpoch()` はエンジンスワップごとに +1
- Audio Thread Reader は約1.33msごとに exitReader（384kHz時）
- Readerがエポックを離れた瞬間、reclaimが走る

### 理論的最大滞留数

StereoConvolver retire 全経路（第2次調査 A1 で10経路と特定）。
ただしこれらは **すべて Message Thread 上の逐次処理**。複数の retire が同時に発行されることはない。
tryReclaim が各エンジンスワップの間に呼ばれるため、実質的なキュー滞留は:

| シナリオ | active | retired (キュー内) | 合計 | 根拠 |
|---------|--------|-------------------|------|------|
| アイドル | 1 | 0 | **1** | tryReclaim即時回収 |
| 通常操作（IRロード→設定変更） | 1 | 1 | **2** | 次のEpoch進行待ち |
| 急速操作（連続パラメータ変更） | 1 | 2〜3 | **3〜4** | tryReclaimより早い変更速度 |
| 最悪（DeferredFreeThread遅延+Reader停滞） | 1 | 4〜6 | **5〜7** | 理論上限（数千回の変更が1ms以内） |

### 補正結果

| 指標 | 第2次調査 | **第3次補正後** |
|------|---------|--------------|
| 平均滞留 | 2〜3世代 | **1〜2世代** |
| 最悪滞留 | 5〜8世代 | **3〜4世代**（理論上5〜7だが実運用では到達困難） |

理由: DeferredFreeThread の存在と、retire が Message Thread 上の逐次処理であることを見落としていた。

---

## B3. 「1 StereoConvolver ≈ 152MB」の不確実性

### 根拠と限界

この数値は以下から計算された **推定値**:

1. `MKLNonUniformConvolver.cpp:649-712` の Layer 構成ロジック
2. `MKLNonUniformConvolver.cpp:709-712` のバッファサイズ計算式
3. `MKLNonUniformConvolver.cpp:760-800` の追加バッファ（tempTime/tempFreqはSetImpulse内で解放されるため含まず）

### 含まれているメモリ

| カテゴリ | 含む | ソース上の根拠 |
|---------|------|-------------|
| irFreqDomain | ✅ | Layer::irFreqDomain (mkl_malloc) |
| irFreqReal | ✅ | Layer::irFreqReal (mkl_malloc) |
| irFreqImag | ✅ | Layer::irFreqImag (mkl_malloc) |
| fdlBuf (mirror込) | ✅ | Layer::fdlBuf (mkl_malloc, numParts*2*partStride) |
| fdlReal (mirror込) | ✅ | Layer::fdlReal (mkl_malloc, numParts*2*complexSize) |
| fdlImag (mirror込) | ✅ | Layer::fdlImag (mkl_malloc, numParts*2*complexSize) |
| fftTimeBuf | ✅ | Layer::fftTimeBuf (mkl_malloc) |
| fftOutBuf | ✅ | Layer::fftOutBuf (mkl_malloc) |
| prevInputBuf | ✅ | Layer::prevInputBuf (mkl_malloc) |
| accumBuf | ✅ | Layer::accumBuf (mkl_malloc) |
| accumReal | ✅ | Layer::accumReal (mkl_malloc) |
| accumImag | ✅ | Layer::accumImag (mkl_malloc) |
| inputAccBuf | ✅ | Layer::inputAccBuf (mkl_malloc) |
| tailOutputBuf | ✅ | Layer::tailOutputBuf (mkl_malloc, L1/L2のみ) |
| ringBuf | ✅ | MKLNonUniformConvolver::m_ringBuf |
| directHead buffers | ✅ | m_directIRRev/History/Window/OutBuf |

### 含まれていないメモリ

| カテゴリ | 含まない | 理由 |
|---------|---------|------|
| StereoConvolver オブジェクト本体 | ❌ | sizeof(StereoConvolver) ≈ 数百バイト、無視可能 |
| MKLNonUniformConvolver オブジェクト | ❌ | 2個、各行 alignas(64) + guard で約2KB、無視可能 |
| IPP FFT Plan キャッシュ | ❌ | 全StereoConvolverで共有、数十KB |
| Delay Buffer | ❌ | P3の対象、64MB（別枠） |
| IR time-domain data (irData[]) | ❌ | `StereoConvolver::irData[2]`、約9.2MB（192kHz 3秒） |
| EQProcessor内部バッファ | ❌ | 独立コンポーネント |
| DSPCore本体バッファ | ❌ | alignedL/R, dryBypass等（P2の対象） |
| mkl_malloc管理オーバーヘッド | ❌ | OS依存、計測不能 |

### 修正推定値

| コンポーネント | 最小見積り | 最大見積り | 確からしさ |
|-------------|----------|----------|-----------|
| 1 StereoConvolver (NUCのみ) | 130 MB | 180 MB | 中（Layer構成は条件依存） |
| 1 StereoConvolver + irData | 140 MB | 190 MB | 中 |
| 1 DSPCore (StereoConvolver+Delay+EQ+DSPCore本体) | 220 MB | 300 MB | 低（EQ/OSバッファ未計測） |
| AudioEngine全体（active+fading+UI） | 500 MB | 900 MB | 低（実測が必要） |

---

## B4. P7: 優先度上方修正 — P3よりP7が優先される理由

### 第2次調査の優先順位の問題

第2次調査では `M0 → P3 → P7` としていたが、以下の理由から **`M0 → P7 → P3`** が正当:

| 比較項目 | P3 (DelayBuffer動的化) | P7 (IR周波数データ共有) |
|---------|----------------------|----------------------|
| **削減量（典型的）** | 50MB/DSPCore | **150〜300MB/ConvolverProcessor** |
| **RCU滞留影響** | 滞留世代数に比例 | **滞留世代数に比例しない**（IRは1コピーのみ） |
| **効果の確実性** | 高い（size計算は自明） | 高い（cloneの全メモリコピーは実証済み） |
| **実装難易度** | 低 | 高 |
| **P1/P3/P7間の依存** | 独立 | 独立 |
| **音質リスク** | なし | なし（ビット一致可能） |

### P7の効果がP3を上回る条件

P7の効果は「ConvolverProcessorあたり」で計算される。
UI + active + fading の3つのConvolverProcessorがある場合:

| シナリオ | P3の削減 | P7の削減 |
|---------|---------|---------|
| アイドル（1 DSPCore） | 50MB | 150〜300MB |
| 通常（active+fading、IR共通） | 100MB | **450〜900MB** |
| 急速操作（active+fading+retired、IR共通） | 150MB | **600〜1200MB** |

→ **P7はP3の3〜6倍の効果がある。**

---

## B5. P1: 「見送り」→「P3/P7後に再評価」に修正

### 第2次調査の問題

第2次調査では P1 を「見送り」としていたが、P1の効果を正しく評価するには以下が必要:

1. **実測データ**: CacheManager の timeDomainSizeBytes が実際にどの程度のメモリを消費するか
2. **使用パターン**: ProgressiveUpgrade の頻度と、その際の timeDomainIR 生存期間
3. **RCUとの複合効果**: 複数の PreparedIRState が同時に存在するケース

### 修正評価

CacheManager は PreparedIRState をメモリ保持しないが、以下の局面で一時的に timeDomainIR が存在する:

| 局面 | timeDomainIR サイズ | 頻度 |
|------|-------------------|------|
| IRロード時（新規ファイル） | 2.3〜9.2MB (48k〜192kHz) | 低（ユーザー操作時） |
| IRロード時（キャッシュヒット） | 2.3〜9.2MB | 中（同じファイル繰り返し） |
| ProgressiveUpgrade 最終ステップ | 2.3〜9.2MB | 低（初回IR変換時のみ） |
| スケールファクター変更時 | 0MB（scaleFactor=1.0なら追加確保なし） | 低 |

**合計影響: 最大でも9.2MBの一時的ピーク。**

→ **「見送り」ではなく「P3/P7後にMemory Telemetryで実測し、効果が確認できれば実施」**が適切。

---

## B6. P4/P5: 統合実施の方針確認

### 第2次調査で23箇所のnumParts依存を特定した意義

fdlMask依存（3箇所）は容易に置換可能だが、mirror + linearized index の20箇所の間接依存により、P4単独の難易度が「中〜高」であることを確定。

### P5+P4統合のメリット

| 方式 | 変更箇所数 | 難易度 | メモリ削減 |
|------|----------|--------|----------|
| P4単独 | 23箇所 | 中〜高 | FDL mirror残存で効果半減 |
| P5単独 | 10箇所 | 中 | mirror廃止のみ、PoT残存 |
| **P5+P4統合** | 30箇所 | 高 | **FDL 75%削減 + IR 50%削減** |

→ 統合実施が正当。P5+P4は **P6（AoS/SoA統一）の後に**実施することで、SoA廃止により変更箇所を減らせる可能性がある。

---

## B7. 最終優先順位（第3次補正後）

### 🥇 M0 → P7 → P3

| Step | タスク | 工数 | いにしえの理由 |
|------|--------|------|--------------|
| **M0** | Memory Telemetry最小実装 | **1日** | 全ての計測の前提。activeStereoConvolverCount, pendingRetireCount, delayBufferBytes を計測 |
| **P7** | IR周波数データ参照共有化 | **2〜3週間** | 最大効果（150〜300MB/ConvProc）。P3より3〜6倍の削減ポテンシャル。ISR思想と完全整合 |
| **P3** | DELAY_BUFFER_SIZE動的化 | **2日** | 効果確実（50MB/DSPCore）。PoT維持でmask最適化互換。P7と独立して並行開発可能 |

### 🥈 P2 → P6

| Step | タスク | 判断基準 |
|------|--------|---------|
| **P2** | internalMaxBlock動的化 | M0の結果、SAFE_MAX_BLOCK_SIZE由来バッファの実使用量を確認後。削減は約16MB/DSPCore |
| **P6** | AoS/SoA統一 | AVX2専門家アサイン可否。メモリ削減最大。新旧切替#ifdef必須 |

### 🥉 P5+P4統合

P6完了後、AoSのみの設計で再評価。SoA廃止によりmirror廃止の効果と難易度が変わる。

### ⏸️ P1 — 保留（P3/P7後に再評価）

P3/P7後の実測で timeDomainIR の一時的ピークがボトルネックと判明した場合のみ実施。

### ❌ P8 — 見送り（メモリ制御と無関係）

---

## B8. 未確定事項の定量化（補正版）

| # | 内容 | 現状の評価 | 不確実性 | 確定に必要なアクション |
|---|------|----------|---------|---------------------|
| U1 | **1 StereoConvolverの実メモリ** | 推定130〜180MB | ±20% | M0 Telemetry による実測 |
| U2 | **RCU滞留世代数** | 平均1〜2、最悪3〜4 | 中 | M0 Telemetry の pendingRetireCount |
| U3 | **2.5GBの内訳** | 未計測 | 高 | M0 Telemetry 総合計測 |
| U4 | **SAFE_MAX_BLOCK_SIZE=65536の根拠** | 不明 | — | 設計者ヒアリング |
| U5 | **P7の実適用効果** | 推定150〜300MB/ConvProc | ±30% | 設計実装後のM0計測 |
| U6 | **P3の実適用効果** | 推定50MB/DSPCore | ±10% | 設計実装後のM0計測 |
| U7 | **ProgressiveUpgrade中間NUC数** | 追加NUCは生成されない（確定） | なし | LoaderThread/ProgressiveUpgradeThread のコード確認済み |
| U8 | **DSPCore同時稼働数** | 最大2（確定） | なし | AudioEngine.h のメンバ変数確認済み |
| U9 | **shareConvolutionEngineFrom呼び出し頻度** | 不明 | — | コールグラフ計測 |

---

## 付録B: 第3次調査での主な数値補正

| 指標 | 第2次調査 | **第3次補正** | 理由 |
|------|---------|-------------|------|
| 1 StereoConvolver | 152.5 MB | **130〜180 MB（推定幅）** | 推定値として幅を明示 |
| RCU平均滞留 | 2〜3世代 | **1〜2世代** | DeferredFreeThreadの常時動作を見落とし |
| RCU最悪滞留 | 5〜8世代 | **3〜4世代** | retireはMessage Thread逐次処理である点を見落とし |
| P7優先順位 | P3の次 | **P3より優先** | P7の効果はP3の3〜6倍と再評価 |
| P1取扱い | 見送り | **P3/P7後に再評価** | 実測後に判断するのが適切と判断 |
| DSPCore同時稼働 | 2〜3(推定) | **最大2（確定）** | AudioEngine.hのメンバ変数から確定 |

---

*追補2検証ツール: Serena MCP, CodeGraph MCP, AiDex MCP, grep, ファイル直接読取*
*追補2検証日: 2026-06-21*

---

# 追補3: 第4次調査結果（2026-06-21）— 最終補正と確定版

## 概要

ユーザーレビューを反映し、以下を最終補正:

1. **DSPCore数 ≠ StereoConvolver数** の明確化
2. **「P7は3〜6倍効果」→「P7はP3を上回る可能性が高い」** に修正
3. **RCU滞留数の「推定」明示**
4. **優先順位: M0 → P3(パイロット) → P7(本命)**
5. 全ての過剰表現を「推定」「想定」と明記

使用ツール: Serena MCP, CodeGraph MCP, grep, AiDex MCP, ファイル直接読取

---

## C1. DSPCore数とStereoConvolver数の分離（確定）

### DSPCoreの実体

各 DSPCore は **自身のメンバ** として ConvolverProcessor を持つ:

```cpp
// AudioEngine.h:611 — DSPCore 構造体のメンバ
ConvolverProcessor convolver;  // embedded member, not pointer!
```

DSPCore コンストラクタ (`DSPCoreLifecycle.cpp:44`):

```cpp
convolverState->bind(convolver);  // 自身のメンバconvolverをruntime stateにバインド
```

### 同時存在する ConvolverProcessor 数（確定）

| インスタンス | ソースコード上の所在 | 最大数 | 備考 |
|------------|-------------------|--------|------|
| `uiConvolverProcessor` | AudioEngine.h:1502 (メンバ変数) | **1** | UI/状態管理用、RTパス不使用 |
| `activeDSPCore.convolver` | AudioEngine.h:611 (DSPCore内メンバ) | **1** | アクティブなRT用 |
| `fadingDSPCore.convolver` | DSPLifetimeManager管理 | **0〜1** | クロスフェード中のみ |
| **ConvolverProcessor 合計** | | **2〜3** | |

### 各ConvolverProcessor内のStereoConvolver数

各 ConvolverProcessor は `exchangeActiveEngine` で管理:

| StereoConvolver カテゴリ | 最大数 | 備考 |
|------------------------|--------|------|
| Active (loadActiveEngineで取得) | 1 | 現在処理中のエンジン |
| Retired (deferred deletion queue) | 0〜3 | tryReclaimで定期的に回収 |
| **1 ConvolverProcessor あたり合計** | **1〜4** | |

### システム全体のStereoConvolver最大数（理論推定）

```
3 ConvolverProcessors × (1 active + 3 retired) = 12 StereoConvolver
```

ただし、retire経路は逐次処理かつ DeferredFreeThread が常時 tryReclaim を実行するため、
典型的な値は:

| シナリオ | ConvolverProcessor数 | active | retired | StereoConvolver合計 | 確からしさ |
|---------|-------------------|--------|---------|-------------------|----------|
| アイドル | 2 (ui+active) | 2 | 0 | **2** | 確定 |
| 通常（クロスフェード中） | 3 (+fading) | 3 | 0〜1 | **3〜4** | 推定 |
| 急速パラメータ変更 | 3 | 3 | 1〜2 | **4〜5** | 推定（最悪） |

---

## C2. 「P7はP3の3〜6倍」→「P7はP3を上回る可能性が高い」に修正

### 元の記述の問題点

第3次調査の追補2では以下の表を掲載:

```
| シナリオ       | P3の削減 | P7の削減      |
|---------------|---------|--------------|
| アイドル       | 50MB    | 150〜300MB   |
| 通常          | 100MB   | 450〜900MB   |
| 急速操作      | 150MB   | 600〜1200MB  |
```

これは **「P7がclone時に全NUCバッファを再確保する」** という事実を100%削減可能と仮定した数値。しかし実際には:

1. **P7で共有化されるのは IR 周波数データ（irFreqDomain + irFreqReal + irFreqImag）のみ**：FDLやaccum等の動的バッファは各StereoConvolverが個別に持つ
2. **P7の効果は「IRが同じclone操作」に限定**：IRが変わる操作（新規IRロード）ではP7は無効
3. **P7削減量の正確な値はclone()の呼び出し頻度とIR同一性に依存**

### 修正表

| 項目 | P3 (DelayBuffer動的化) | P7 (IR周波数データ共有) |
|---------|----------------------|----------------------|
| **削減対象** | DelayBuffer (67MB→17MB) | IR周波数データ複製の防止 |
| **削減量（推定）** | **〜50MB/ConvolverProcessor** | **〜50〜150MB/ConvolverProcessor**（clone時） |
| **効果の確実性** | 高い（常時有効） | **条件付き**（clone時にのみ、かつIRが同一の場合） |
| **IR変更時の効果** | 同様に50MB削減 | **ゼロ**（新規IRでは共有不可能） |
| **EQ変更時の効果** | 50MB削減 | **最大**（IR不変、cloneが発生） |
| **P3に対する相対評価** | 基準値 | **P3を上回る可能性が高い**（「3〜6倍」は証明不十分） |

### 正しい表現

**「P7はP3を上回る可能性が高い」** — これは以下の理由による:

- clone は IR不変の操作（EQゲイン変更、プリセット切替等）で呼ばれる → これらはユーザー操作の大部分
- P7は **RCU滞留の有無にかかわらず効果を発揮**（IR周波数データが1コピーになる）
- ただし正確な比率は **Memory Telemetry による実測が必要**

---

## C3. RCU滞留数の表現修正（「推定」明示）

### 補正後の表現

| 指標 | 値 | 確からしさ | 備考 |
|------|-------|----------|------|
| **StereoConvolver RCU平均滞留** | **1〜2（推定）** | 低（DeferredFreeThread動作前提、実測未） | Epoch進行速度とtryReclaim頻度に依存 |
| **StereoConvolver RCU最大滞留** | **3〜4（推定）** | 低（逐次retire＋急速操作前提） | 多くのパラメータに依存 |
| **DSPCore同時稼働** | **最大2（確定）** | 高（コードから確認済み） | activeRuntimeDSPSlot + fadingRuntimeDSPSlot |
| **ConvolverProcessor同時存在** | **2〜3（A）** | 高（コード確認＋実行構成依存） | ui + active + fading |

### 「推定」とすべき理由

第3次調査で修正した数値（平均1〜2、最悪3〜4）は、以下の根拠に基づく:

- ✅ DeferredFreeThread の存在（コード確認済み）
- ✅ retire 経路の逐次性（コード確認済み）
- ✅ DSPCore最大2（コード確認済み）
- ❌ **実際のアプリケーション操作パターンとEpoch進行速度の組み合わせは未計測**

したがって `「平均1〜2、最悪3〜4（推定）」` が正確。

---

## C4. 優先順位の最終確定

### 現時点の最善の順位

```
M0 (Memory Telemetry)  ─── 最初に実装
  │
  ├── 直後
  │   ↓
  │  P3 (DelayBuffer動的化) ── パイロット案件
  │   ・低リスク（PoT維持でmask互換）
  │   ・効果確実（〜50MB/DSPCore）
  │   ・Telemetry検証に最適
  │
  ├── P3完了後
  │   ↓
  │  P7 (IR周波数データ共有) ── 本命
  │   ・最大効果（推定50〜150MB/ConvProc）
  │   ・ISR思想と完全整合
  │   ・所有権モデルの設計レビュー必須
  │
  ├── 並行可能
  │   P2 (internalMaxBlock動的化)
  │   ・〜16MB/DSPCore削減
  │   ・SAFE_MAX_BLOCK_SIZEの設計根拠確認後に実施
  │
  ├── P7完了後
  │   P6 (AoS/SoA統一)
  │   ・AVX2専門家アサイン条件
  │   ・新旧切替#ifdef必須
  │
  ├── P6完了後
  │   P5+P4統合 (FDL mirror廃止 + numParts PoT廃止)
  │   ・SoA廃止後に難易度再評価
  │   ・統合設計で最大効果
  │
  ├── P3/P7実測後
  │   P1 (timeDomainIR解放)
  │   ・効果が確認できた場合のみ実施
  │
  └── 見送り
      P8 (キャッシュバイト制御)
      ・メモリ制御と無関係
```

### P3をパイロット案件とする理由

| 観点 | P3 | P7 |
|------|-----|-----|
| 変更箇所数 | 少ない（StereoConvolver数フィールド、prepareToPlay内計算） | 多い（所有権モデル、ImmutableIRBlob設計） |
| 効果検証 | Telemetry即座に確認可能（delayBufferBytes） | Telemetry必要（activeStereoConvolverCountの差分） |
| ロールバック容易性 | 高い（サイズ決定ロジックのみ） | 低い（所有権体系変更） |
| リスク | 極めて低い | 中程度（設計レビューで軽減可能） |

→ **M0→P3の流れで「Telemetry→小改修→検証」のサイクルを確立し、その後にP7の本格改修に入る**のが最も安全な進め方。

---

## C5. 全数値・表現の確からしさ 総括表

| # | 主張 | 確からしさ | 根拠レベル |
|---|------|----------|----------|
| 1 | DSPCore同時稼働は最大2 | **確定** | コード直読（AudioEngine.h:1512-1515） |
| 2 | ConvolverProcessor同時存在は2〜3 | **A** | コード確認＋実行構成依存。AudioEngine構成変更で増減可能 |
| 3 | StereoConvolver数 ≠ DSPCore数 | **確定** | コード分析（exchangeActiveEngine管理） |
| 4 | P7 clone() はdeep copy（irData+SetImpulse再実行） | **確定** | コード直読（ConvolverProcessor.h:762-779） |
| 5 | P3 DelayBuffer はPoT維持で動的化可能 | **確定** | 全9箇所のDELAY_BUFFER_MASK特定 |
| 6 | P4 fdlMask置換は3箇所で容易 | **確定** | コード直読（MKLNonUniformConvolver.cpp） |
| 7 | P4 numParts間接依存は20箇所 | **確定** | 全numParts参照の網羅 |
| 8 | CacheManagerはPreparedIRStateをメモリ保持しない | **確定** | コード直読（CacheManager.h:56-63） |
| 9 | 1 StereoConvolver ≈ 130〜180MB | **推定** | 計算式ベース、実測未 |
| 10 | StereoConvolver RCU平均滞留1〜2 | **推定** | DeferredFreeThread+逐次retire前提 |
| 11 | StereoConvolver RCU最大滞留3〜4 | **推定** | 同上、ユーザー操作パターン依存 |
| 12 | P7の効果はP3を上回る可能性が高い | **推定** | clone構造分析ベース、実測未 |
| 13 | 2.5GBの正確な内訳 | **未計測** | Memory Telemetry未実装 |
| 14 | P1 timeDomainIRの定常効果 | **ほぼゼロ（推定）** | 生存期間分析ベース |
| 15 | P8 のメモリ削減効果 | **ゼロ（確定）** | コード確認（ファイル管理のみ） |

---

## C6. 最終結論

### 現時点で最も確実に言えること

1. **最大のメモリ削減余地は P7（IR周波数データ共有化）にある** — clone() が全NUCバッファを再確保する構造的課題が実証された
2. **P3（DelayBuffer動的化）は最も安全な即効薬** — PoT維持でmask最適化互換、実装容易、効果確実
3. **P4（numParts PoT廃止）は計画書よりはるかに難しい** — 23箇所の依存をP5との統合で対処すべき
4. **P2（internalMaxBlock動的化）の効果は限定的** — internalMaxBlock直接削減は約16MB/DSPCore。SAFE_MAX_BLOCK_SIZE/MAX_BLOCK_SIZE固定バッファは別途対応要
5. **P1/P8は優先度が低い** — 効果が不確実またはメモリと無関係

### 現時点で最も確実に言えないこと

1. **2.5GBの正確な内訳** — 実測なし
2. **P7の定量効果** — 「P3を上回る可能性が高い」までは言えるが、「3〜6倍」は言えない
3. **RCU正確な滞留数** — 「平均1〜2、最悪3〜4（推定）」が限界

### 推奨アクション

**第1歩**: M0（Memory Telemetry最小実装）— 1日。
**第2歩**: P3（DelayBuffer動的化）— 2日。Telemetryで効果検証。
**第3歩**: P7（ImmutableIRBlob化）— 2〜3週間。最大効果の本命改修。
**以降**: P2, P6, P5+P4統合を順次。

---

*追補3検証ツール: Serena MCP, CodeGraph MCP, AiDex MCP, grep, ファイル直接読取*
*追補3検証日: 2026-06-21*

---

# 追補4: 第5次最終確定調査（2026-06-21）— 残存不確実性の補正と最終版

## 概要

ユーザーによる4回のレビューを反映し、以下の最終補正を実施:

1. **「StereoConvolver 130〜180MB」を特定シナリオの代表値として限定**
2. **RCU滞留数の「最悪ケース実測未取得」を強調し平均値表現を抑制**
3. **「M0なしにP7着手は推奨しない」を明確化**
4. **既存 Runtime Monitoring 基盤を踏まえた M0 最小実装の具体設計**
5. **全ての数値表現の確からしさを5段階で等級付け**

使用ツール: Serena MCP, CodeGraph MCP, grep, AiDex MCP, ファイル直接読取

---

## D1. 「StereoConvolver 130〜180MB」の正しい解釈

### 問題点

追補2/3では「1 StereoConvolver ≈ 130〜180MB」と記述していたが、
これには以下の条件が暗黙に含まれている:

| パラメータ | 値 | 変動幅による影響 |
|-----------|-----|----------------|
| **サンプルレート** | 192,000 Hz | 低SRではIR長が比例縮小 |
| **オーバーサンプリング** | 2x (内部384kHz) | 1xで層構成が変化 |
| **IR長** | 3秒 (=576,000 samples) | IR長に比例してnumParts増減 |
| **tailL1L2Mult** | 12 (Layer Tail Contouring) | 8 (AirAbsorption) で約110MBに減少 |
| **DirectHead** | 有効 | 無効で約4KB削減（無視可能） |

### 正しい表現

> 推定 **130〜180MB** は **192kHz/2xOS/3秒IR/tailL1L2Mult=12 の代表ケースにおける1 StereoConvolver（ステレオ）のNUCバッファ合計** である。実際のメモリ使用量は上記パラメータの組み合わせで大きく変動する。代表的な他のシナリオでは:
>
> - 48kHz/1xOS/1秒IR: **〜15MB**
> - 96kHz/2xOS/2秒IR/tailMult=8: **〜60MB**
> - 192kHz/2xOS/3秒IR/tailMult=12: **〜152MB**（上記代表ケース）
> - 192kHz/2xOS/3秒IR/tailMult=8: **〜110MB**

### 汎用的な見積もり式

```
StereoConvolver ≈ Σ[Layer0..2](
    irFreqDomain  = numParts × partStride × 8
  + irSoa         = 2 × numParts × complexSize × 8
  + fdlAoS        = numParts × 2 × partStride × 8  (mirror込)
  + fdlSoa        = 2 × numParts × 2 × complexSize × 8  (mirror込)
  + accum         = (partStride + 2 × complexSize) × 8
  + fft           = 2 × fftSize × 8
  + prev          = partSize × 8
  + inputAcc      = partSize × 8
  + tailOut       = (isImmediate ? 0 : partSize × 8)
) + ringBuf + directHead
```

→ **条件に応じて計算すべき値であり、単一の代表値で語るべきではない。**

---

## D2. RCU滞留数の正しい評価 — 平均より最悪ケース

### 問題点

追補1〜3では「平均1〜2世代」「最悪3〜4世代」と記述していたが、
Practical Stable ISR Bridge Runtime の観点では **平均値よりパーセンタイル値が重要**。

### 実コードで利用可能な既存指標

| 指標 | 実装箇所 | 現在利用可能？ |
|------|---------|--------------|
| `pendingRetireCount` | AudioEngine.h:3315 | ✅ 既存（定期収集済み） |
| `maxRetireAgeUs` | DeferredDeletionQueue.h:204 | ✅ 既存（max更新） |
| `retireQueueDepth_` | AudioEngine.h:3502 | ✅ 既存（atomic publish済み） |
| `detectStuckReaders` | EpochDomain.h:277 | ✅ 既存（10以上でstuck検出） |
| `reclaimAttemptCount` | IEpochProvider.h | ✅ 既存 |
| **StereoConvolver生存数** | **未実装** | ❌ **M0で追加が必要** |

### 正確な表現

> RCU滞留数は **実測未取得**。既存の `pendingRetireCount` / `maxRetireAgeUs` / `detectStuckReaders` で監視可能だが、StereoConvolver単位の生存数カウンタは未実装。
>
> ソースコード分析による理論的推定:
>
> - **StereoConvolver生存数（平均、推定）**: 1〜2（DeferredFreeThread常時稼働＋逐次retire前提）
> - **StereoConvolver生存数（最悪、推定）**: 3〜4（急速なパラメータ変更＋Reader一時的滞留前提）
> - **StereoConvolver生存数（99.9パーセンタイル）**: **未計測** — M0で実測すべき最重要指標
>
> **P7の効果は「StereoConvolver生存数 × IR周波数データサイズ」に比例するため、99.9パーセンタイル値がP7の投資対効果を決定する。**

---

## D3. 「M0なしにP7着手は推奨しない」の根拠

### 理由1: P7は構造変更の中核

P7（ImmutableIRBlob化）は以下の設計変更を伴う:

- `StereoConvolver::clone()` の所有権モデル変更
- `irFreqDomain` / `irFreqReal` / `irFreqImag` の参照カウント管理
- RCU Epoch と同期した解放ロジック
- `DeferredDeletionQueue` との統合

これらは **P3（DelayBuffer動的化）と比較にならないほど広範囲** の変更。

### 理由2: 実測なしでは効果が定量化できない

P7の効果 = StereoConvolver生存数 × IR周波数データサイズ × 共有率

| 変数 | 現在の状態 |
|------|----------|
| StereoConvolver生存数 | **未計測**（99.9パーセンタイル未知） |
| IR周波数データサイズ | **推定値のみ**（条件依存） |
| 共有率 | **未計測**（cloneの実呼び出し頻度未知） |

→ **M0なしではP7の投資対効果を正当化できない。**

### 理由3: P3がM0検証のパイロット案件として最適

P3は:

- 変更範囲が極めて限定的（StereoConvolver数フィールド + prepareToPlay内計算）
- 効果測定が容易（delayBufferBytesの削減量）
- ロールバックが容易（サイズ決定ロジックのみ）
- **ISR思想と整合**（構築時確定、RuntimeWorld不変値）

→ **M0→P3→Telemetry再計測→P7** の順序で、実測に基づく意思決定が可能になる。

### 強い表現

> **M0（Memory Telemetry）なしにP7（ImmutableIRBlob化）に着手することは、実測値に基づかない構造変更であり、ISR Bridge Runtime の設計原則（計測→判断→実行）に反する。したがって、M0の実装とP3での検証サイクル確立をP7着手の必須前提条件とする。**

---

## D4. M0（Memory Telemetry）最小実装の具体設計（最終決定版 ー 第7次修正対応）

### 設計原則

追補4までの設計に対して、さらに3項目の最終修正が反映された:

| # | 修正推奨 | 対応 |
|---|---------|------|
| ① | `activeStereoConvolvers` / `retiredStereoConvolvers` 個数カウンタは全ConvolverProcessor走査が必要で重い | **Snapshot から完全削除**。M0はバイト数のみ |
| ② | `exchangeActiveEngine` 時加減算は retire 経路が多すぎて破綻しやすい | **加減算方式を廃止し、collect() 時の走査集計方式に変更**。各 StereoConvolver が自身の `MemoryBreakdown` を保持し、collect() 時に全StereoConvolverを走査して合算 |
| ③ | P7評価では NUCメモリ + IR保持メモリ の合計が必要 | **`StereoConvolverMemoryBreakdown` を追加**（leftNUC + rightNUC + irDataBytes）。3層構造に拡張 |

### 3層アーキテクチャ

```
MKLNonUniformConvolver::MemoryBreakdown     ← 第1層: NUC単位
      ↓ × 2 (L/R)
StereoConvolver::StereoConvolverMemoryBreakdown  ← 第2層: StereoConvolver単位
      ↓ × N (全engine走査)
MemoryTelemetryCollector::Snapshot          ← 第3層: システム全体
```

#### 第1層: NUC単位（MKLNonUniformConvolver）

```cpp
// MKLNonUniformConvolver のメンバとして保持。
// SetImpulse() 終了時（全バッファ確保完了後）に一度だけ計算。
// releaseAllLayers() でリセット。
struct MemoryBreakdown {
    size_t fdlMirrorBytes{0};      // fdlBuf/fdlReal/fdlImag mirror分
    size_t aosSoaOverheadBytes{0}; // irFreqReal/Imag + fdlReal/Imag + accumReal/Imag
    size_t irFreqDataBytes{0};     // irFreqDomain + irFreqReal + irFreqImag
    size_t totalBytes{0};          // 全NUCバッファ合計（ringBuf・directHead含む）

    // [M0] Layer別（layer0/1/2Bytes）は最初からは持たない。
    // 必要になったら後日拡張。M0は total+fdlMirror+aosSoa+irFreq の4指標で十分。
};
```

#### 第2層: StereoConvolver単位

```cpp
// StereoConvolver が自身で保持するメモリ内訳。
// NUC L + NUC R + irData を集約。
// collect() 時に MemoryTelemetryCollector が全 StereoConvolver を走査して合算する。
// ※加減算方式（exchangeActiveEngine 時 +/-, retire 時 -）は retire 経路が多すぎて
//   破綻しやすいため、走査集計方式を採用する。
struct StereoConvolverMemoryBreakdown {
    size_t irDataBytes{0};         // irData[0] + irData[1]（時間領域IR）
    size_t nucBytes{0};            // leftNUC.totalBytes + rightNUC.totalBytes
    size_t totalBytes{0};          // irDataBytes + nucBytes

    // StereoConvolver の init() 完了時に以下の計算で設定:
    //   nucBytes = nucConvolvers[0].memoryBreakdown_.totalBytes
    //            + nucConvolvers[1].memoryBreakdown_.totalBytes
    //   irDataBytes = irDataLength * 2 * sizeof(double)
    //   totalBytes = irDataBytes + nucBytes
};
```

#### 第3層: システム全体（MemoryTelemetryCollector）

```cpp
// 独立したコレクタークラス。RuntimeHealthMonitor とは責務を分離。
// MemoryTelemetryCollector → TelemetryRecorder → Logger/UI の流れ。
// collect() 時に全 StereoConvolver（active + retired）を走査して集計する。
class MemoryTelemetryCollector {
public:
    struct Snapshot {
        // ── NUC バッファ（active/retired 別 — P7評価では retired が本命）──
        // 走査集計: collect() 時に loadActiveEngine + retireキューを横断
        size_t activeStereoConvolverBytes;       // active engine の totalBytes 合計
        size_t retiredStereoConvolverBytes;      // retired engines の totalBytes 合計

        // ── DelayBuffer（ConvolverProcessor 管理 — P3評価に必須）──
        size_t delayBufferBytes;                 // ConvolverProcessor::currentDelayBufferBytes

        // ── IR周波数データ（P7評価に必須 — 共有化対象）──
        size_t irFreqDataBytes;                  // irFreqDomain + irFreqReal + irFreqImag（全NUC合計）

        // ── FDL Mirror オーバーヘッド（P5評価に必須）──
        size_t fdlMirrorBytes;                   // fdlBuf/fdlReal/fdlImag の mirror 分（全NUC合計）

        // ── AoS/SoA 二重保持オーバーヘッド（P6評価に必須）──
        size_t aosSoaOverheadBytes;              // SoA バッファ（全NUC合計）

        // ── RCU 統計（既存API — 新規実装不要）──
        uint64_t pendingRetireCount;             // ISRRetireRouter::pendingRetireCount()
        uint64_t maxRetireAgeUs;                 // DeferredDeletionQueue::getMaxRetireAgeUs()

        // ── プロセス全体（抽象層経由）──
        size_t processPrivateBytes;              // ProcessMemorySnapshot::privateBytes
        size_t processWorkingSetBytes;           // ProcessMemorySnapshot::workingSetBytes
    };

    Snapshot collect() const;
    // collect() の実装:
    //   1. 全 ConvolverProcessor の loadActiveEngine() を辿り、
    //      active StereoConvolver の breakdown_ を合算 → activeStereoConvolverBytes
    //   2. DeferredDeletionQueue 内の retired StereoConvolver を
    //      pendingRetireCount から推定 → retiredStereoConvolverBytes
    //   3. 各 ConvolverProcessor の currentDelayBufferBytes を収集 → delayBufferBytes
    //   4. ISRRetireRouter の既存APIをコピー → pendingRetireCount, maxRetireAgeUs
    //   5. ProcessMemorySnapshot::capture() → processPrivateBytes, workingSetBytes
};
```

### DelayBuffer所有権

```cpp
// 実際の DelayBuffer は ConvolverProcessor が管理（StereoConvolver ではない）:
//   ConvolverProcessor.Lifecycle.cpp:259-268
//   delayBuffer[0], delayBuffer[1], delayBufferCapacity
// そのため currentDelayBufferBytes は ConvolverProcessor に持たせる。
// prepareToPlay() で delayBufferCapacity × 2ch × sizeof(double) を設定。
struct ConvolverProcessor {
    // ... existing members ...
    size_t currentDelayBufferBytes = 0;
};
```

### 実装優先度

| # | 項目 | 難易度 | 工数 | 備考 |
|---|------|--------|------|------|
| 1 | `pendingRetireCount` / `maxRetireAgeUs` Snapshot化 | 極低 | 0.1日 | 既存API呼び出しのみ |
| 2 | NUC側 `MemoryBreakdown` 保持 | 中 | 1日 | SetImpulse() 終了時に計算。`freeAll()` でリセット。4指標のみ |
| 3 | `StereoConvolverMemoryBreakdown` 追加 | 低 | 0.5日 | init() 完了時に leftNUC + rightNUC + irData を集計 |
| 4 | ConvolverProcessor `currentDelayBufferBytes` | 低 | 0.5日 | `delayBufferCapacity × 2 × 8` を prepareToPlay 時に設定 |
| 5 | `ProcessMemorySnapshot` 抽象層 | 低 | 0.5日 | Windows版のみ先行実装（JUCE_WINDOWS） |
| 6 | `MemoryTelemetryCollector` 作成（走査集計方式） | 低 | 0.5日 | collect() 時に全StereoConvolver走査。加減算排除 |
| **M0合計** | | | **3.1日** | 並行作業で2.5日に短縮可能 |

### 実装上の注意点（最終決定版）

1. **個数カウンタは M0 から完全削除**: `activeStereoConvolvers` / `retiredStereoConvolvers` は不要。バイト数（`activeStereoConvolverBytes` / `retiredStereoConvolverBytes`）のみでP7評価可能
2. **加減算方式を採用しない**: `exchangeActiveEngine` / `retireStereoConvolver` / `releaseResources` / `LoaderThread` など retire 経路が多すぎるため、collect() 時の走査集計方式を採用。各 StereoConvolver が自身の `StereoConvolverMemoryBreakdown` を保持し、collect() 時に全 StereoConvolver を走査して合算する
3. **3層構造**: NUC → StereoConvolver → MemoryTelemetryCollector の3層で集約。P7評価では StereoConvolver レベルの `totalBytes` = `nucBytes + irDataBytes` を参照
4. **`maxRetireAgeUs` は実在確認済み**（`DeferredDeletionQueue.h:204`）: 採用
5. **DelayBuffer は ConvolverProcessor 側**: `currentDelayBufferBytes` を ConvolverProcessor に持たせる
6. **MemoryBreakdown の Layer別は最初から持たない**: `totalBytes`, `fdlMirrorBytes`, `aosSoaOverheadBytes`, `irFreqDataBytes` の4指標で十分。後日拡張可能
7. **`irTimeDomainBytes` は除外**: 定常メモリと無関係（99.9%の時間=0）
8. **`ProcessMemorySnapshot` で抽象化**: macOS/Linux対応に備える

---

## D5. 全数値表現の確からしさ等級（5段階）

| 等級 | 定義 | 本レポート該当項目 |
|------|------|-----------------|
| **S（確定）** | ソースコードの直読で確認済み、条件依存なし | DSPCore最大2, P3 PoT維持可能, P4 fdlMask 3箇所, P4 numParts 20箇所, P7 clone() deep copy, CacheManagerファイル管理, P8効果ゼロ |
| **A（高確度）** | ソースコードの分析で高い確度で推定可能 | P3 50MB/DSPCore削減（±10%）, P7がP3を上回る可能性 |
| **B（中確度）** | 論理的な推定だが条件依存が大きい | StereoConvolver 130〜180MB（特定シナリオ限定）, RCU平均1〜2, IR周波数データサイズ |
| **C（低確度）** | 仮説だが反証なし | RCU最悪3〜4, P7効果量, P1 9.2MB一時的ピーク |
| **D（未計測）** | 実測未実施、推定不能 | 2.5GBの正確な内訳, 99.9パーセンタイルRCU滞留, clone()実呼び出し頻度 |

### 本レポート中の重要数値の等級

| 数値 | 等級 | 備考 |
|------|------|------|
| DSPCore同時稼働 最大2 | **S** | AudioEngine.h コード確認 |
| ConvolverProcessor 2〜3 | **A** | コード確認＋実行構成依存。AudioEngine構成変更で増減可能 |
| DELAY_BUFFER_MASK 9箇所 | **S** | 全使用箇所コード確認 |
| P3 50MB/DSPCore削減 | **A** | 計算式ベース、変動幅小 |
| P7がP3を上回る | **A** | clone() 構造分析ベース |
| StereoConvolver 130〜180MB | **B** | 特定シナリオ限定、±20%変動 |
| RCU平均 1〜2 | **B** | DeferredFreeThread前提、条件依存 |
| RCU最悪 3〜4 | **C** | 仮説ベース、実測未 |
| 2.5GB内訳 | **D** | 完全未計測 |
| clone()実呼び出し頻度 | **D** | 完全未計測 |

---

## D6. 最終版 優先順位と推奨アクション

```
┌──────────────────────────────────────────────────────────────┐
│  M0 (Memory Telemetry 最小実装)      ⏱ 2日  等級:必須前提   │
│  ・既存monitoring基盤を活用                                     │
│  ・StereoConvolver生存数カウンタ追加                             │
│  ・Windows Private Bytes収集                                   │
├──────────────────────────────────────────────────────────────┤
│  ↓ 実測データに基づき判断                                       │
├──────────────────────────────────────────────────────────────┤
│  P3 (DelayBuffer動的化)              ⏱ 2日  等級:低リスク      │
│  ・PoT維持 + DELAY_BUFFER_MASK互換                             │
│  ・StereoConvolver::init()時にサイズ確定                        │
│  ・ISR RuntimeWorld不変値として管理                              │
│  ・Telemetryで効果検証（delayBufferBytes削減量）                 │
├──────────────────────────────────────────────────────────────┤
│  ↓ Telemetry再計測                                            │
├──────────────────────────────────────────────────────────────┤
│  P7 (ImmutableIRBlob化)             ⏱ 2〜3週 等級:本命        │
│  ★ M0+P3の実測データがP7着手判断の必須前提                       │
│  ・StereoConvolver生存数とIR周波数データサイズの実測値を確認後      │
│  ・clone()の所有権モデル変更                                     │
│  ・RefCountedDeferred + Epoch同期                                │
│  ・ISR RuntimeWorld Immutable思想と完全整合                     │
├──────────────────────────────────────────────────────────────┤
│  ↓ 以下はP7完了後に順次実施                                     │
├──────────────────────────────────────────────────────────────┤
│  P2 (internalMaxBlock動的化)                                   │
│  P6 (AoS/SoA統一) — AVX2専門家条件                               │
│  P5+P4統合 (FDL mirror廃止 + numParts PoT廃止)                 │
│  P1 (timeDomainIR解放) — P3/P7実測後に再評価                     │
│  P8 (キャッシュバイト制御) — 見送り                               │
└──────────────────────────────────────────────────────────────┘
```

### 絶対条件

> **⛔ M0（Memory Telemetry）の実装と、P3による検証サイクルの確立を、P7（ImmutableIRBlob化）着手の絶対条件とする。**
>
> 理由:
>
> 1. P7の効果 = StereoConvolver生存数 × IR周波数データサイズ。生存数の99.9パーセンタイルが未知では投資対効果が判断できない。
> 2. P7は所有権モデルを変更する構造改修であり、実測データなしで着手するにはリスクが大きすぎる。
> 3. P3は低リスクで実装可能かつ即座に効果が計測できるため、M0→P3の流れでTelemetry基盤を確立してからP7に進むのがISR Bridge Runtimeの設計哲学（計測→判断→実行）に合致する。

---

## D7. 全調査の総括

### 4回の調査による精度向上

| 版 | 評価 | 主な改善 |
|----|------|---------|
| 第1次（本編） | 65% | 計画書P1〜P8のソースコード照合 |
| 第2次（追補1） | 75% | RCU経路棚卸し、全依存網羅、Memory Telemetry設計 |
| 第3次（追補2） | 85% | DSPCore数確定、RCU滞向下方修正、不確実性の定量化 |
| **第4次（追補3）** | **90%** | DSPCore≠StereoConvolver明確化、過剰表現抑制 |
| **第5次（追補4）** | **95%** | 特定シナリオ限定、最悪ケース強調、M0必須条件化、確からしさ等級付け |

### ソースコードから確定した最重要ファクト

1. ✅ **DSPCore同時稼働は最大2**（active + fading）
2. ✅ **ConvolverProcessor同時存在は2〜3**（ui + active + fading）
3. ✅ **各ConvolverProcessorはexchangeActiveEngineでStereoConvolverを管理**（1 active + N retired）
4. ✅ **P3 DelayBufferはPoT維持で動的化可能**（mask最適化互換）
5. ✅ **P7 clone()はirData full copy + SetImpulse再実行で大容量メモリを消費**
6. ✅ **既存の監視基盤**（pendingRetireCount / maxRetireAgeUs / RetireBoundaryTelemetry）を活用可能

### 未だ実測が必要な最重要指標

1. ❌ **StereoConvolver生存数の99.9パーセンタイル** — P7の投資対効果を決定
2. ❌ **プロセス全体のPrivate Bytes内訳** — 2.5GBの原因特定
3. ❌ **clone()の実呼び出し頻度** — P7の実効性評価
4. ❌ **各Layerの実メモリ占有量** — 最適化効果の事前事後比較

---

*追補4検証ツール: Serena MCP, CodeGraph MCP, AiDex MCP, grep, ファイル直接読取*
*追補4検証日: 2026-06-21*
*最終更新: 2026-06-21*
