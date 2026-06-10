# 改訂版 改修計画書

**作成日**: 2026-06-10
**バージョン**: 2.0（ISR Bridge Runtime 観点反映）
**ベース**: `doc/work28/newbug.md`, `doc/work28/newbug_validation_report.md`
**評価基準**: Practical Stable ISR Bridge Runtime（実運用破綻耐性・リアルタイム制約・世代整合性・スレッドモデル整合性）

---

## 0. 設計方針

### 0.1 分類体系

本計画では全項目を以下の4分類で評価する。

| 分類 | 意味 | 扱い |
|:----:|------|------|
| **BUG** | 実装上の欠陥。動作不具合・クラッシュの原因 | **最優先で修正** |
| **SAFETY** | 未定義動作・例外安全・メモリ安全性の改善 | **計画的に修正** |
| **DESIGN** | 責務境界・命名・状態管理の設計明確化 | **優先度低〜検討課題** |
| **OPTIMIZATION** | 性能最適化・デッドコード整理 | **ついでに対応** |

### 0.2 除外項目

以下は本計画の対象外とする。

| ID | 理由 |
|:--:|------|
| **M-02** | DCブロッカーカットオフ1Hz/3Hzの差異はオーバーサンプリング段のサンプルレートを考慮した正当な設計。オーディオ的観点で正しい。 |
| **M-04** | `observeMonotonicRollbackRequested_` フラグは `AudioEngine.Commit.cpp:352` の `exchangeAtomic(..., false, ...)` で正しくリセットされている。誤報告。 |
| **H-04** | 線形クロスフェードは音響設計の選択でありバグではない。等パワークロスフェードへの変更は「音響キャラクタ変更」として正式な仕様決定を経るべき。 |
| **L-01** | レベルメーターが「ヘッドルーム適用前の内部処理レベル」を測定する仕様なら誤りではない。測定定義の問題でありバグではない。 |
| **H-02** | リーク経路はJUCEシャットダウン時のMessageManagerキュー破棄のみで実質的に問題なし。過大評価。 |

---

## 1. 改修一覧（ISR分類別）

| # | ID | ISR分類 | タイトル | 改修難易度 | リスク | 影響ファイル |
|:-:|:--:|:-------:|----------|:----------:|:------:|:-------------|
| 1 | C-03 | **BUG** | ProgressiveUpgradeThread Use-After-Free | ★★☆ | 中 | `ProgressiveUpgradeThread.cpp` |
| 2 | H-03 | **BUG** | IR逆順後 irFreqReal/irFreqImag 不整合 | ★★☆ | 中 | `MKLNonUniformConvolver.cpp` |
| 3 | M-01 | **BUG** | std::abs をAudio Threadで使用 | ★☆☆ | 低 | `MKLNonUniformConvolver.cpp` |
| 4 | C-02 | **SAFETY** | union型パンニング→std::bit_cast | ★★☆ | 低 | `AudioEngine.h`, `EQProcessor.Processing.cpp`, `ConvolverProcessor.Runtime.cpp`, `DspNumericPolicy.h` |
| 5 | H-05 | **SAFETY** | finalizeNUCEngine例外補完 | ★☆☆ | 低 | `ConvolverProcessor.LoadPipeline.cpp` |
| 6 | C-01 | **DESIGN** | LinearRampのRT責務境界違反 | ★★☆ | 中 | `DspNumericPolicy.h`, `ConvolverProcessor.Runtime.cpp` |
| 7 | M-05 | **DESIGN** | wetBuf[0]流用の解消 | ★☆☆ | 低 | `ConvolverProcessor.Runtime.cpp`, `ConvolverProcessor.h` |
| 8 | H-01 | **DESIGN** | m_rtBypassShadow設計明確化（コメントのみ） | ★☆☆ | 低 | `EQProcessor.Processing.cpp` |
| 9 | M-03 | **OPTIMIZATION** | NoiseShaper全状態スキャンの最適化 | ★★☆ | 低 | `Fixed15TapNoiseShaper.h` |
| 10 | L-02 | **OPTIMIZATION** | HC else到達不能ブランチ整理 | ★☆☆ | 低 | `MKLNonUniformConvolver.cpp` |
| 11 | L-03 | **DOC** | 命名/コメント修正 | ★☆☆ | 低 | `Fixed15TapNoiseShaper.h`, `EQProcessor.Coefficients.cpp` |
| 12 | H-02 | **DOC** | callAsync設計意図のコメント化 | ★☆☆ | 低 | `ConvolverProcessor.LoaderThread.cpp` |

---

## 2. 詳細設計

---

### 2.1 C-03: ProgressiveUpgradeThread Use-After-Free 【BUG】

**問題**: `upgradeStep()` 内のラムダが `std::atomic<bool>* cancelledFlag = &cancelled;` でメンバ変数の生ポインタをキャプチャ。オブジェクト破棄後もラムダが実行されると解放済みメモリにアクセスする。

**ISR評価**: **真の破綻点**。lifetime ownership がスレッドに依存しており、スレッド分離境界が崩壊すると回復不能。

**改修方針**: weakOwner 経由で `cancelled` にアクセスする。これによりオブジェクト生存期間とポインタ参照が同期される。

**変更内容** (`src/ProgressiveUpgradeThread.cpp`):

```cpp
// ── Before ──
std::atomic<bool>* cancelledFlag = &cancelled;
const uint64_t expectedGeneration = taskGeneration;

prepared = converter.convertToHighRes(irFile, ...,
    [weakOwner, cancelledFlag, expectedGeneration]() {
        auto* owner = weakOwner.get();
        if (owner == nullptr) return true;
        return juce::Thread::currentThreadShouldExit()
            || convo::consumeAtomic(*cancelledFlag, ...)
            || !owner->isConvolverGenerationCurrent(expectedGeneration);
    });

// ── After ──
const uint64_t expectedGeneration = taskGeneration;

prepared = converter.convertToHighRes(irFile, ...,
    [weakOwner, expectedGeneration]() {
        auto* owner = weakOwner.get();
        if (owner == nullptr) return true;
        return juce::Thread::currentThreadShouldExit()
            || convo::consumeAtomic(owner->cancelled, ...)
            || !owner->isConvolverGenerationCurrent(expectedGeneration);
    });
```

**備考**: より本質的には `cancelled` を owner state に統合し atomic pointer 参照を禁止する設計が望ましいが、スコープ外。本修正で実用上十分な安全性を確保する。

**検証**: スレッド実行中に `ProgressiveUpgradeThread` を破棄してもクラッシュしないことを確認。

---

### 2.2 H-03: IR逆順後 irFreqReal/irFreqImag 不整合 【BUG】

**問題**: `SetImpulse()` 内で `irFreqDomain`（AoS）のみ逆順化され、`irFreqReal/irFreqImag`（SoA）は更新されない。`filterSpec != nullptr` の場合のみ `applySpectrumFilter` 内の再デインターリーブで偶発的に修正されるが、`filterSpec == nullptr` では Split-complex AVX2 パスが逆順IRを使用する。

**ISR評価**: **真の破綻点**。「一部経路だけで整合性が保証される構造」（lazy repair依存）は典型的なデータ整合性バグ。

**改修方針**: 逆順化ループ内で AoS（`irFreqDomain`）と SoA（`irFreqReal/irFreqImag`）を同時に swap する。

**変更内容** (`src/MKLNonUniformConvolver.cpp`):

```cpp
// ── Before: irFreqDomain のみ swap ──
if (l.numPartsIR > 1)
{
    double* swapBuf = static_cast<double*>(mkl_malloc(
        static_cast<size_t>(l.partStride) * sizeof(double), 64));
    if (swapBuf)
    {
        for (int pf = 0; pf < l.numPartsIR / 2; ++pf)
        {
            const int pb = l.numPartsIR - 1 - pf;
            double* slotF = l.irFreqDomain + pf * l.partStride;
            double* slotB = l.irFreqDomain + pb * l.partStride;
            memcpy(swapBuf, slotF, l.partStride * sizeof(double));
            memcpy(slotF,   slotB, l.partStride * sizeof(double));
            memcpy(slotB,   swapBuf, l.partStride * sizeof(double));
        }
        mkl_free(swapBuf);
    }
}

// ── After: irFreqDomain + irFreqReal/Imag 同時 swap ──
if (l.numPartsIR > 1)
{
    double* swapDomain = static_cast<double*>(mkl_malloc(
        static_cast<size_t>(l.partStride) * sizeof(double), 64));
    double* swapSoA = static_cast<double*>(mkl_malloc(
        static_cast<size_t>(l.complexSize) * sizeof(double), 64));
    if (swapDomain && swapSoA)
    {
        for (int pf = 0; pf < l.numPartsIR / 2; ++pf)
        {
            const int pb = l.numPartsIR - 1 - pf;

            // irFreqDomain swap (AoS)
            double* slotF = l.irFreqDomain + pf * l.partStride;
            double* slotB = l.irFreqDomain + pb * l.partStride;
            memcpy(swapDomain, slotF, l.partStride * sizeof(double));
            memcpy(slotF,      slotB, l.partStride * sizeof(double));
            memcpy(slotB,      swapDomain, l.partStride * sizeof(double));

            // irFreqReal swap (SoA)
            double* realF = l.irFreqReal + static_cast<size_t>(pf) * l.complexSize;
            double* realB = l.irFreqReal + static_cast<size_t>(pb) * l.complexSize;
            memcpy(swapSoA, realF, l.complexSize * sizeof(double));
            memcpy(realF,   realB, l.complexSize * sizeof(double));
            memcpy(realB,   swapSoA, l.complexSize * sizeof(double));

            // irFreqImag swap (SoA)
            double* imagF = l.irFreqImag + static_cast<size_t>(pf) * l.complexSize;
            double* imagB = l.irFreqImag + static_cast<size_t>(pb) * l.complexSize;
            memcpy(swapSoA, imagF, l.complexSize * sizeof(double));
            memcpy(imagF,   imagB, l.complexSize * sizeof(double));
            memcpy(imagB,   swapSoA, l.complexSize * sizeof(double));
        }
    }
    if (swapDomain) mkl_free(swapDomain);
    if (swapSoA)    mkl_free(swapSoA);
}
```

**注意**: `applySpectrumFilter` 内の再デインターリーブは削除しない（防御的プログラミング）。ただし「逆順化と同時に SoA も更新済み」のコメントを追記する。

**検証**: `filterSpec == nullptr` の状態を作り、Split-complex パスの出力が正周波数特性を示すことを確認。

---

### 2.3 M-01: std::abs → absNoLibm 【BUG】

**問題**: `MKLNonUniformConvolver::Get()` の `addScaledFallback` ラムダが `std::abs` を Audio Thread で使用。libm呼び出しによりリアルタイム性が低下する可能性。

**ISR評価**: 純粋な置き換えで互換性維持可能な軽微なバグ。

**変更内容** (`src/MKLNonUniformConvolver.cpp`):

```cpp
// Before
if (std::abs(gain - 1.0) < 1.0e-12)

// After
if (absNoLibm(gain - 1.0) < 1.0e-12)
```

**インクルード確認**: `absNoLibm` は `AudioEngine.h:117` に定義。`MKLNonUniformConvolver.cpp` が `DspNumericPolicy.h` を既に include していることを確認し、経路がない場合はインクルードを追加する。

**検証**: デバッグ/リリースビルド成功。数値結果の一致確認。

---

### 2.4 C-02: union 型パンニング → std::bit_cast 【SAFETY】

**問題**: `absNoLibm`, `killDenormal`, `isFiniteNoLibm` 等で C++ UB の `union` 型パンニングを使用。最適化ビルドで予期せぬコード生成のリスク。

**ISR評価**: UB排除は正しい。ただし提案コードの `std::bit_cast` 二重ネストは冗長なため、中間変数形式を採用する。

**変更テンプレート**:

```cpp
// Before (UB)
union { double d; uint64_t u; } v { x };
v.u &= 0x7FFFFFFFFFFFFFFFULL;
return v.d;

// After (defined, ISR推奨: 中間変数形式)
uint64_t bits = std::bit_cast<uint64_t>(x);
bits &= 0x7FFFFFFFFFFFFFFFULL;
return std::bit_cast<double>(bits);

// 非推奨 (2重bit_cast、branch prediction不安定)
// return std::bit_cast<double>(std::bit_cast<uint64_t>(x) & 0x7FFFFFFFFFFFFFFFULL);
```

**該当関数**:

| ファイル | 関数 | パターン |
|----------|------|----------|
| `AudioEngine.h` | `absNoLibm` | double → uint64_t → マスク → double |
| `EQProcessor.Processing.cpp` | `absNoLibm` (匿名名前空間) | 同上 |
| `ConvolverProcessor.Runtime.cpp` | `isFiniteAndAbsBelowNoLibm` | double → uint64_t 指数部検査 |
| `DspNumericPolicy.h` | `killDenormal` | double → uint64_t ビット操作 |
| `DspNumericPolicy.h` | `killDenormalV` (AVX2) | `_mm_and_pd` に置換済みのため対象外 |
| `DspNumericPolicy.h` | `isFiniteNoLibm` (x64版) | double → uint64_t 指数部検査 |

**検証**: デバッグ/リリース両ビルドで `absNoLibm(-1.0) == 1.0`, `killDenormal(0.0) == 0.0` 等の基本アサート確認。全呼び出し箇所の動作回帰テスト。

---

### 2.5 H-05: finalizeNUCEngine 例外補完 【SAFETY】

**問題**: `finalizeNUCEngineOnMessageThread` で `std::bad_alloc` のみ catch。`std::runtime_error` 等が未捕捉で `std::terminate` に至る。

**ISR評価**: catch 追加は安全化。ただし Message Thread 上の処理であり、ISR的には error code 統一が望ましいが段階的対応で可。

**変更内容** (`src/convolver/ConvolverProcessor.LoadPipeline.cpp`):

```cpp
// Before
catch (const std::bad_alloc&)
{
    handleLoadError("Failed to initialize NUC engine (Memory allocation or MKL setup failed).");
}

// After
catch (const std::bad_alloc&)
{
    handleLoadError("Failed to initialize NUC engine (Out of memory).");
}
catch (const std::exception& e)
{
    handleLoadError(juce::String("NUC engine initialization failed: ") + e.what());
}
catch (...)
{
    handleLoadError("NUC engine initialization failed: Unknown error");
}
```

**検証**: ビルド成功確認。例外経路のテストは困難なためコードレビューで担保。

---

### 2.6 C-01: LinearRamp の RT責務境界違反 【DESIGN】

**問題**: `LinearRamp::setCurrentAndTargetValue()` に `ASSERT_NON_RT_THREAD()` が付いているが、`ConvolverProcessor::process()`（Audio Thread）から6箇所で呼ばれている。

**ISR評価**: これは「バグ」ではなく**設計上の責務境界問題**。ランプ = 時間依存状態機械 (state machine) であり、RT thread から状態破壊操作を行う設計そのものが問題。単なるAPI追加では本質は解決しない。

**改修方針**: `LinearRamp` に RT-safe な即時値設定メソッド `applyImmediateValueRT()` を追加する。"force" は ISR において例外経路を増幅する危険な命名であるため使用しない。

**変更内容** (`src/DspNumericPolicy.h`):

```cpp
// LinearRamp に追加
/// Audio Thread から即座に current/target を設定する。
/// 世代カウンターによる同期が呼び出しの安全性を保証している場合に使用。
/// 注意: この操作はランプを無効化し、current = target に設定する。
void applyImmediateValueRT(double v) noexcept
{
    ASSERT_AUDIO_THREAD();
    current = target = v;
    step      = 0.0;
    remaining = 0;
}
```

**呼び出し元修正** (`src/convolver/ConvolverProcessor.Runtime.cpp`):

| 行 | 修正前 | 修正後 |
|:--:|--------|--------|
| 246 | `activeCrossfadeGain.setCurrentAndTargetValue(0.0);` | `activeCrossfadeGain.applyImmediateValueRT(0.0);` |
| 279 | `activeLatencySmoother.setCurrentAndTargetValue(val);` | `activeLatencySmoother.applyImmediateValueRT(val);` |
| 295 | `activeCrossfadeGain.setCurrentAndTargetValue(0.0);` | `activeCrossfadeGain.applyImmediateValueRT(0.0);` |
| 307 | `activeMixSmoother.setCurrentAndTargetValue(...);` | `activeMixSmoother.applyImmediateValueRT(...);` |
| 323 | `activeMixSmoother.setCurrentAndTargetValue(currentVal);` | `activeMixSmoother.applyImmediateValueRT(currentVal);` |
| 491 | `activeLatencySmoother.setCurrentAndTargetValue(...);` | `activeLatencySmoother.applyImmediateValueRT(...);` |

**検証**: デバッグビルドで `jassert` が発火しないことを確認。`ASSERT_AUDIO_THREAD()` が成立していること。

---

### 2.7 M-05: wetBuf[0] 流用の解消 【DESIGN】

**問題**: `double* delayFadeRamp = wetBuf[0];` で Wet 信号バッファをゲイン値格納に流用。コード意図が不明瞭で上書きリスクがある。

**ISR評価**: スタック配列（元案）は `block size` に依存するためオーバーフローの危険がある。`prepareToPlay` で事前確保する allocator cache reuse 方式が安全。

**改修方針**: `ConvolverProcessor.h` に専用バッファを追加し、`prepareToPlay` で `maxBlockSize` に基づき確保する。

**変更内容** (`src/ConvolverProcessor.h`):

```cpp
// メンバ変数追加
convo::ScopedAlignedPtr<double> delayFadeRampBuffer;
int delayFadeRampCapacity = 0;
```

**変更内容** (`src/convolver/ConvolverProcessor.Lifecycle.cpp`):

```cpp
// prepareToPlay 内で確保
const int neededFadeSamples = maxBlockSize; // または別途適切な値
if (delayFadeRampCapacity < neededFadeSamples)
{
    delayFadeRampBuffer.reset(
        static_cast<double*>(convo::aligned_malloc(
            static_cast<size_t>(neededFadeSamples) * sizeof(double), 64)));
    delayFadeRampCapacity = (delayFadeRampBuffer.get() != nullptr) ? neededFadeSamples : 0;
}
```

**変更内容** (`src/convolver/ConvolverProcessor.Runtime.cpp:358`):

```cpp
// Before
double* delayFadeRamp = wetBuf[0];

// After
double* delayFadeRamp = delayFadeRampBuffer.get();
// 安全ガード: 確保失敗時は delayFadeRamp == nullptr となり、以降の memset/memcpy で即座にクラッシュする
if (delayFadeRamp == nullptr || delayFadeRampCapacity < activeDelayCrossfadeSamples)
{
    // バッファ不足: クロスフェードをスキップ
    activeCrossfadeGain.skip(activeDelayCrossfadeSamples);
    activeDelayCrossfadeSamples = 0;
    goto skip_crossfade;
}
```

**検証**: クロスフェード処理が正常に動作することを確認。バッファ確保失敗時の fallback 動作も確認。

---

### 2.8 H-01: m_rtBypassShadow 設計明確化 【DESIGN】

**問題**: 機能的には動作している（`setBypassFromRT()` は `DSPCoreDouble.cpp:382`, `DSPCoreFloat.cpp:189` から呼ばれている）が、初期値と同期経路が不明瞭。

**ISR評価**: バグではなく設計の明確化問題。コメント追記のみで十分。

**変更内容** (`src/eqprocessor/EQProcessor.Processing.cpp`):

```cpp
// 既存コードの直前にコメント追記
// m_rtBypassShadow は AudioEngine::DSPCore::processDoubleToBuffer/
// processFloatToBuffer 内で state.eqBypassed (RuntimeSnapshot由来) から
// setBypassFromRT() 経由で毎ブロック設定される。
// 初期値 false (= 非バイパス) は初回 process() 呼び出し前に DSPCore が
// 設定するまで有効であり、その間に process() が呼ばれることはない。
const bool requestedBypass = m_rtBypassShadow;
```

**検証**: ビルド確認のみ。

---

### 2.9 M-03: NoiseShaper 全状態スキャンの最適化 【OPTIMIZATION】

**問題**: `Fixed15TapNoiseShaper::processSample` 内で毎サンプル `ORDER=16` の全状態配列を走査して最大値を計算。512ブロック×2ch で約16,384回の `absNoLibm` + 比較が発生。

**ISR評価**: インクリメンタル最大値追跡（元案）はリングバッファ上書きにより真の最大値を過小評価し、ノイズシェーパーのリセット判定が不発になるリスクがある。数学的意味が変わる近似変更は「バグ修正」ではなく「アルゴリズム変更」。

**改修方針**: **Decay envelope tracking** 方式を採用する。各サンプルの誤差を envelope で平滑化し、その値が閾値を超えたらリセットする。これにより全状態スキャンを排除しつつ、ノイズ特性を維持する。

**変更内容** (`src/Fixed15TapNoiseShaper.h`):

```cpp
// メンバ変数追加
static constexpr double kEnvelopeAlpha = 0.01; // 時定数調整可能
double errorEnvelope = 0.0;

// processSample 末尾: 全状態スキャンを削除し decay envelope で代替
// Before:
double maxAbs = 0.0;
for (int i = 0; i < ORDER; ++i)
{
    const double absVal = absNoLibm(channelErrors[static_cast<size_t>(i)]);
    if (absVal > maxAbs)
        maxAbs = absVal;
}
if (maxAbs > kErrorStateThreshold)
    convo::publishAtomic(needsReset, true, std::memory_order_release);

// After: decay envelope
errorEnvelope = std::max(absNoLibm(error), errorEnvelope * (1.0 - kEnvelopeAlpha));
// 比較用に tracking max として保持、ブロック終了時にチェック
```

```cpp
// processStereoBlock 終了時に一度だけチェック
if (errorEnvelope > kErrorStateThreshold)
    convo::publishAtomic(needsReset, true, std::memory_order_release);
errorEnvelope = 0.0;
```

**代替案（sampling window）**: 一定間隔（例: 32サンプルごと）で全状態スキャンを実行。毎サンプルよりは1/32に削減され、数学的完全性を維持。

**検証**: ノイズシェーパーのリセット判定タイミングが従来と同等であることを確認。長期間の安定性テスト。

---

### 2.10 L-02: HC else 到達不能ブランチ整理 【OPTIMIZATION】

**問題**: `applySpectrumFilter` の HC 処理で `hcFcEnd = nyquist` のため `k > kEnd` が決して真にならず、`else` ブランチが到達不能。

**改修方針**: 到達不能 `else` を削除し、コメントで理由を記載する。

**変更内容** (`src/MKLNonUniformConvolver.cpp`):

```cpp
// Before
for (int k = 0; k < cSize; ++k)
{
    if (k <= kStart) { /* passband */ }
    else if (k <= kEnd) { /* taper */ }
    else { gain[k] = 0.0; }  // ← 到達不能
}

// After
for (int k = 0; k < cSize; ++k)
{
    if (k <= kStart)
    {
        // パスバンド: ゲイン 1.0
    }
    else
    {
        // テーパー領域 (hcFcEnd = nyquist のため k > kEnd は発生しない)
        // hcFcEnd が nyquist 未満に変更された場合は else-if 分割を再検討すること。
    }
}
```

**検証**: ビルド成功。フィルター周波数特性の同一性確認。

---

### 2.11 L-03: 命名/コメント修正 【DOC】

**1. `Fixed15TapNoiseShaper.h`**:

```cpp
// Before
static constexpr int ORDER = 16;  // クラス名は15-Tap

// After
// 注意: フィルタ次数は16次（ORDER=16）、クラス名 "Fixed15Tap" は
// 従来の命名を維持（タップ数≠次数のため）。次数変更はフィルタ特性を
// 変えるため不可。
static constexpr int ORDER = 16;
```

**2. `EQProcessor.Coefficients.cpp`**:

```cpp
// Before
// SVF係数計算 (Audio Thread用)

// After
// SVF係数計算 (Message Thread専用: std::pow / std::tan を含むためRT不可)
```

**検証**: ビルド確認のみ。

---

### 2.12 H-02: callAsync 設計意図のコメント化 【DOC】

**問題**: 実害のない理論的リーク経路について、意図を文書化する。

**変更内容** (`src/convolver/ConvolverProcessor.LoaderThread.cpp`):

```cpp
// queueFinalizeOnMessageThread 内、callAsync 成功後:
// ラムダ内で unique_ptr / ScopedAlignedPtr にラップしているため、
// weakOwner.get() が nullptr を返してもリソースは解放される。
// 唯一の未解放経路は JUCE シャットダウン時の MessageManager キュー破棄だが、
// これは正常シャットダウンで許容範囲の動作である。
```

**検証**: ビルド確認のみ。

---

## 3. フェーズ分割

### Phase S: 真の破綻点（最優先、1PR）

| 順序 | ID | タスク | 分類 | 工数 | 依存 |
|:----:|:--:|--------|:----:|:----:|:----:|
| S.1 | C-03 | ProgressiveUpgradeThread weakOwner 化 | BUG | 1h | なし |
| S.2 | H-03 | IR逆順 + SoA 同時 swap | BUG | 2h | なし |

**検証**: 両方ともデバッグビルド + `Strict Atomic Dot-Call Scan`。H-03 は `filterSpec=null` 状態での Split-complex 出力確認。

### Phase 1: Safety & Boundary Fix（1PR）

| 順序 | ID | タスク | 分類 | 工数 | 依存 |
|:----:|:--:|--------|:----:|:----:|:----:|
| 1.1 | C-02 | union→std::bit_cast（中間変数形式） | SAFETY | 2h | なし |
| 1.2 | C-01 | `applyImmediateValueRT()` 追加＋6箇所置換 | DESIGN | 1h | なし |
| 1.3 | M-01 | `std::abs`→`absNoLibm` | BUG | 0.5h | C-02完了後（absNoLibmがbit_cast化されるため同一PRで可） |
| 1.4 | H-05 | 例外補完（catch追加） | SAFETY | 0.5h | なし |

**検証**: Debug/Release 両ビルド成功。`jassert` 不发火確認。

### Phase 2: Design & Optimization（1PR）

| 順序 | ID | タスク | 分類 | 工数 | 依存 |
|:----:|:--:|--------|:----:|:----:|:----:|
| 2.1 | M-05 | wetBuf[0]→allocator cache reuse | DESIGN | 1h | なし |
| 2.2 | M-03 | NoiseShaper decay envelope 化 | OPT | 1.5h | なし |
| 2.3 | L-02 | HC else 到達不能ブランチ整理 | OPT | 0.5h | なし |

**検証**: ビルド成功＋ノイズシェーパー長期安定性テスト。

### Phase 3: Documentation（1PR、任意）

| 順序 | ID | タスク | 分類 | 工数 | 依存 |
|:----:|:--:|--------|:----:|:----:|:----:|
| 3.1 | L-03 | 命名/コメント修正 | DOC | 0.5h | なし |
| 3.2 | H-01 | コメント追記 | DESIGN | 0.25h | なし |
| 3.3 | H-02 | コメント追記 | DOC | 0.25h | なし |

**検証**: ビルド確認のみ。

---

## 4. 総合スケジュール

| Phase | 内容 | 項目数 | PR数 | 工数 | リスク |
|:-----:|------|:------:|:----:|:----:|:------:|
| **S** | 真の破綻点（BUG） | 2 | 1 | 3h | 中（H-03のデバッグに時間を要す可能性） |
| **1** | Safety & Boundary Fix | 4 | 1 | 4h | 低（C-02のstd::bit_castは広範囲だが機械的置換） |
| **2** | Design & Optimization | 3 | 1 | 3h | 低（M-03のenvelope定数調整に試行錯誤の可能性） |
| **3** | Documentation | 3 | 1 | 1h | 低（コメントのみ） |
| **合計** | | **12** | **4** | **11h** | |

### 実装順序の根拠

```
Phase S ──→ Phase 1 ──→ Phase 2 ──→ Phase 3
  (BUG)      (SAFETY)     (DESIGN)     (DOC)
                +            +
             (BUG M-01)  (OPT)
```

- Phase S はリリースブロッカー。**真の破綻点**である C-03（UAF）と H-03（IR逆順）を最優先で修正する。
- Phase 1 はリリース前必須。UB 排除（C-02）と RT 安全違反（C-01, M-01）、例外安全（H-05）を含む。
- Phase 2 は計画的対応。設計明確化（M-05）と最適化（M-03, L-02）。
- Phase 3 は任意。コメントとドキュメントのみ。

---

## 5. 各項目の改修前後比較

```
C-03 [BUG]  生ポインタキャプチャ ─→ weakOwner 経由アクセス
H-03 [BUG]  irFreqDomain のみ逆順 ─→ AoS + SoA 同時 swap
M-01 [BUG]  std::abs ─→ absNoLibm
C-02 [SAFETY] union 型パンニング ─→ std::bit_cast（中間変数形式）
H-05 [SAFETY] bad_alloc のみ catch ─→ bad_alloc + exception + ... catch
C-01 [DESIGN] forceSetCurrentAndTargetValueRT ─→ applyImmediateValueRT
M-05 [DESIGN] wetBuf[0] 流用 ─→ allocator cache reuse
H-01 [DESIGN] コメント追記のみ
M-03 [OPT] 毎サンプル全状態スキャン ─→ decay envelope
L-02 [OPT] 到達不能 else 削除 ─→ else-if 統合＋コメント
L-03 [DOC] コメント修正
H-02 [DOC] コメント追記
```

---

## 6. リスクと対策

| リスク | 該当項目 | 影響 | 対策 |
|--------|:--------:|:----:|------|
| `std::bit_cast` による数値変化 | C-02 | C-02の誤りが全 `absNoLibm` 呼び出しに波及 | 単体テスト（既知値での入出力確認） |
| スレッドタイミングによる再現困難 | C-03 | 修正検証が不完全 | コードレビュー＋強制タイミングテスト |
| `filterSpec=null` の実現困難 | H-03 | 修正検証が不完全 | テスト用に一時的に `nullptr` を渡すコードを挿入 |
| ノイズシェーパー特性変化 | M-03 | 音質劣化 | 長期安定性テスト＋envelope時定数の調整 |
| バッファ確保失敗時の fallback | M-05 | クロスフェード未適用 | goto fallback パスの確認 |

---

## 7. CI/CD Workflow バグ分析と改修

### 7.1 概要

`.github/workflows/` 配下の4つのワークフローのうち、ローカル実行で以下の不合格を確認。

| ワークフロー | 実行者 | 結果 | 不合格原因 |
|:------------:|:------:|:----:|-----------|
| `audioengine-lint.yml` | `check-audioengine-lint.ps1` | ✅ PASS | — |
| `list-compliance.yml` | `check-list-compliance.ps1` | ❌ FAIL | 内部で呼ぶ `check-src-atomic-dotcall.ps1` が **37件の違反** を検出 |
| `list-compliance.yml` | `check-src-size-mul-cast.ps1` | ✅ PASS | — |
| `check-work21-epochdomain-gates.ps1` | 手動実行 | ❌ FAIL | P1-19違反 + routerMethods レグレッション |

### 7.2 WF-BUG-01: 原子力ドットコール違反（37件）【BUG】

**問題**: `check-src-atomic-dotcall.ps1` が `.load()`, `.store()`, `.fetch_add()` 等の直接原子力操作を検出。ISR ポリシーでは `convo::consumeAtomic / publishAtomic / exchangeAtomic / fetchAddAtomic` のラッパー使用が必須。

**ISR評価**: 設計方向は正しいが、**機械的置換は危険**。以下の理由により一律置換は実運用破綻リスクを伴う:

| リスク | 原因 | 影響 |
|--------|------|------|
| 過剰同期化 | `.load(relaxed)` → `consumeAtomic(acquire)` は意味が異なる | 不要なCPUバリア → RTジッタ |
| カウンタ破壊 | `fetch_add(relaxed)` → `fetchAddAtomic(acq_rel)` は性能劣化 | lock-free設計の意図崩壊 |
| 抽象化の形骸化 | 強制acq_rel wrapperは「統一」ではなく「制限」 | 意図表現力低下 |

**ISRの本来の設計**: 「メモリ順序の意図的選択を残す抽象化」であって「強制同期API化」ではない。

**違反ファイルと件数**:

| ファイル | 違反数 | 主な違反パターン |
|----------|:------:|-----------------|
| `src/audioengine/TelemetryRecorder.h` | 13 | `.load()`, `.store()`, `.fetch_add()` |
| `src/audioengine/TelemetryRecorder.cpp` | 6 | `.store()`, `.load()` |
| `src/audioengine/RuntimePublicationOrchestrator.h` | 5 | `.load()`, `.store()` |
| `src/audioengine/RuntimePublicationOrchestrator.cpp` | 1 | `.store()` |
| `src/audioengine/AudioEngine.CtorDtor.cpp` | 1 | `.fetch_add()` |
| `src/audioengine/ISRShutdown.cpp` | 2 | `.store()` |
| `src/core/EpochDomain.h` | 4 | `.load()`, `.store()`, `.fetch_add()` |
| `src/core/SnapshotCoordinator.h` | 3 | `.load()`, `.store()` |
| `src/DeferredDeletionQueue.h` | 3 | `.load()`, `.store()` |

**改修方針（ISR補正版）**: 一律置換ではなく、**メモリオーダリングを維持した意図保存型置換**を行う。

**ルール**:

| 現状 | 置換先 | 条件 |
|------|--------|------|
| `.load(acquire)` | `convo::consumeAtomic(..., acquire)` | acquire は維持 |
| `.load(relaxed)` | **直接呼び出し維持** または `convo::consumeAtomic(..., relaxed)` を追加 | relaxed は relaxed のまま。wrapper追加は任意 |
| `.store(release)` | `convo::publishAtomic(..., release)` | release は維持 |
| `.store(relaxed)` | **直接呼び出し維持** | relaxed カウンタ（Telemetry等）は relaxed を保つ |
| `.fetch_add(relaxed)` | **直接呼び出し維持** | 非同期カウンタは relaxed のまま。`fetchAddAtomic` 強制は禁止 |
| `.fetch_add(acq_rel)` | `convo::fetchAddAtomic(..., acq_rel)` | acq_rel は維持 |
| `.store(seq_cst)` | **lint違反対象** | `memory_order_seq_cst` は禁止（別ルール） |

**変更テンプレート（意図保存型）**:

```cpp
// ▼ acquire → consumeAtomic (維持)
// Before
value = someAtomic.load(std::memory_order_acquire);
// After
value = convo::consumeAtomic(someAtomic, std::memory_order_acquire);

// ▼ release → publishAtomic (維持)
// Before
someAtomic.store(newValue, std::memory_order_release);
// After
convo::publishAtomic(someAtomic, newValue, std::memory_order_release);

// ▼ relaxed load → 直接呼び出し維持 (または consumeAtomic(relaxed))
// Before (OK, relaxed is intentional)
uint64_t counter = statsCounter.load(std::memory_order_relaxed);
// After (変更なし。必要なら consumeAtomic(relaxed) を使用)

// ▼ relaxed fetch_add → 直接呼び出し維持 (fetchAddAtomic 強制禁止)
// Before (OK, relaxed counter for telemetry)
auto idx = writePos_.fetch_add(1, std::memory_order_relaxed);
// After (変更なし。acq_rel 化は過剰同期化)

// ▼ acq_rel fetch_add → fetchAddAtomic
// Before
auto old = someAtomic.fetch_add(1, std::memory_order_acq_rel);
// After
auto old = convo::fetchAddAtomic(someAtomic, static_cast<uint64_t>(1), std::memory_order_acq_rel);
```

**ファイル別置換方針**:

| ファイル | relaxed維持 | acquire→consumeAtomic | release→publishAtomic | acq_rel→fetchAddAtomic |
|----------|:-----------:|:---------------------:|:---------------------:|:----------------------:|
| `TelemetryRecorder.h` | `.fetch_add(relaxed)` 4箇所 | `.load(acquire)` 5箇所 | `.store(release)` 4箇所 | — |
| `TelemetryRecorder.cpp` | — | `.load(acquire)` 3箇所 | `.store(release)` 3箇所 | — |
| `RuntimePublicationOrchestrator.h` | `.load(relaxed)` 2箇所 | `.load(acquire)` 1箇所 | `.store(release)` 2箇所 | — |
| `EpochDomain.h` | `.load(relaxed)` 1箇所 | `.load(acquire)` 2箇所 | — | — |
| `SnapshotCoordinator.h` | — | `.load(acquire)` 2箇所 | `.store(release)` 1箇所 | — |
| `DeferredDeletionQueue.h` | — | `.load(acquire)` 2箇所 | `.store(release)` 1箇所 | — |

**注**: `EpochDomain.h` は ISR 基盤コードであり、`convo::consumeAtomic` が定義されている `AtomicAccess.h` を既に include していることを確認。`TelemetryRecorder` 系は `AudioEngine.h` 経由で `AtomicAccess.h` が利用可能。

**検証**: `check-src-atomic-dotcall.ps1` が relaxed 維持箇所を**許容するように改修され**た上で PASS すること。または各 relaxed 箇所に `// NOLINT(atomic-dot-call)` を付与して許容を明示。

**ISR トレードオフ**:

| 観点 | 機械的置換（旧案） | 意図保存型置換（本計画） |
|:----:|:------------------:|:-----------------------:|
| 安全性 | △（過剰同期化リスク） | ○（正しいordering維持） |
| 性能 | ✗（不要バリア増加） | ○（relaxed維持） |
| 意図表現力 | ✗（一律acq_rel化） | ○（ordering選択を残す） |
| 抽象化統一 | ○（全箇所wrapper化） | △（relaxedは直接呼び出し） |
| lint単純性 | ○（全禁止で単純） | △（例外ルール必要） |

**結論**: `relaxed` の直接呼び出しを許容する lint ルール緩和とセットで進める。lint スクリプトに relaxed 例外の明示的許可機構を追加することを推奨。

### 7.3 WF-BUG-02: EpochDomain 公開API違反 + レグレッション【BUG】

**問題**: `check-work21-epochdomain-gates.ps1` が以下を検出:

1. **P1-19 VIOLATION**: `ISRRetireRouter.h:47` で `EpochDomain` 型が公開コンストラクタパラメータとして露出
2. **routerMethods レグレッション**: 前回3→今回4（+1増加）

**ISR評価**: 設計方向は正しいが、**インターフェース過剰抽象化のリスク**がある。以下の点に注意:

- ISR Retire 経路の本質的機能は「epoch参照」「retireトリガ」「snapshot sync」の3機能のみ
- `IEpochProvider`, `AbstractRetireCoordinator`, `IRetireProvider` の3層導入は過剰設計の兆候
- 抽象層を増やすほど RT 依存解決コストが増加する
- 本質的な設計課題は「誰がepochを進めるか」の一方向化であって、インターフェース増殖ではない

**改修方針（ISR補正版）**: 抽象インターフェースの多層導入は避け、**以下の最小単位の修正**に留める:

1. `ISRRetireRouter` のコンストラクタが直接 `EpochDomain&` を受け取る代わりに、**前方宣言＋実装ファイルでのみ依存**する形にする
2. または、既存の `IRetireProvider` インターフェース（存在する場合）に `currentEpoch()` メソッドを追加して再利用する
3. 新規インターフェース導入は**1つまで**に制限する

**変更内容** (`src/audioengine/ISRRetireRouter.h`) - 最小修正案:

```cpp
// Before (EpochDomain が公開ヘッダに露出)
#include "core/EpochDomain.h" // または前置き宣言

explicit ISRRetireRouter(EpochDomain& epochDomain) noexcept
    : epochDomain_(&epochDomain) {}
// ...
EpochDomain* epochDomain_ = nullptr;

// After Option A: 前方宣言＋実装ファイル分離
// ISRRetireRouter.h では前方宣言のみ
class EpochDomain; // 不完全型で十分

explicit ISRRetireRouter(EpochDomain& epochDomain) noexcept;
// ...
EpochDomain* epochDomain_ = nullptr;

// ISRRetireRouter.cpp で完全型を使用
// #include "core/EpochDomain.h" は cpp 側のみ

// After Option B: 既存 IRetireProvider を拡張（推奨、過剰抽象化回避）
// 既存の IRetireProvider (または Iv3Provider) に currentEpoch 取得を追加
// これにより新規インターフェース導入ゼロで EpochDomain 露出を排除
```

**重要**: 以下の設計は**避ける**こと:

- ❌ `IEpochProvider`, `AbstractRetireCoordinator`, `IRetireProvider` の3層導入
- ❌ virtual 基底クラスの追加による vtable コスト増加
- ❌ 単一実装しかないインターフェースの作成
- ❌ EpochDomain を隠すためだけのラッパークラス

**検証**: `check-work21-epochdomain-gates.ps1` が PASS すること（P1-19違反0）。routerMethods レグレッションが解消されていること。

### 7.4 WF-BUG-03: isr-verification.yml のパス問題【DESIGN】

**問題**: `isr-verification.yml` の verifier 実行ステップで以下の問題が確認された:

1. `python $v` で Python スクリプトを実行しているが、`tools/` に当該スクリプトが存在するか事前チェックがない
2. `'tools\generate_publication_manifest.py --verify --repo-root $(pwd)'` は Unix シェル記法。PowerShell では `$(pwd)` が期待通り展開されず、リポジトリルートが空になる
3. Python verifier 群がエラー時に即座に `throw` する設計だが、`$ErrorActionPreference = 'Continue'` との整合性が不明瞭

**ISR評価**: 実運用でのワークフロー破綻リスクは低いが、CI の信頼性を損なう。

**改修方針**:

```yaml
# Before (PowerShell で $(pwd) は未定義)
'tools\generate_publication_manifest.py --verify --repo-root $(pwd)'

# After (PowerShell 互換)
"tools\generate_publication_manifest.py --verify --repo-root $((Get-Location).Path)"
```

また、Python スクリプト実行前に存在確認を追加:

```powershell
# 各 verifier 実行前に存在チェック
if (-not (Test-Path $v.Split(' ')[0])) {
    Write-Host "::warning::Verifier not found: $v (skipping)"
    continue
}
```

**検証**: `isr-verification.yml` をローカルで模擬実行し、パス解決が正しいことを確認。

---

## 8. 追加改修の統合計画

### 新規項目サマリ

| ID | ISR分類 | タイトル | 工数 | 影響ファイル |
|:--:|:-------:|----------|:----:|:-------------|
| WF-BUG-01 | **BUG** | 原子力ドットコール37件違反 | 3h | `TelemetryRecorder.h/cpp`, `RuntimePublicationOrchestrator.h/cpp`, `AudioEngine.CtorDtor.cpp`, `ISRShutdown.cpp`, `EpochDomain.h`, `SnapshotCoordinator.h`, `DeferredDeletionQueue.h` |
| WF-BUG-02 | **BUG** | EpochDomain公開API違反 + レグレッション | 2h | `ISRRetireRouter.h` |
| WF-BUG-03 | **DESIGN** | isr-verification.yml パス問題 | 0.5h | `.github/workflows/isr-verification.yml` |

### 全フェーズ改訂版

| Phase | 内容 | 項目数 | PR数 | 工数 | リスク |
|:-----:|------|:------:|:----:|:----:|:------:|
| **S** | 真の破綻点 (C-03, H-03) | 2 | 1 | 3h | 中 |
| **S-CI** | CI破綻点 (WF-BUG-01, WF-BUG-02) | 2 | 1 | 5h | 中（波及範囲広い） |
| **1** | Safety & Boundary Fix | 4 | 1 | 4h | 低 |
| **1-CI** | CI安全性 (WF-BUG-03) | 1 | (Phase1に含) | 0.5h | 低 |
| **2** | Design & Optimization | 3 | 1 | 3h | 低 |
| **3** | Documentation | 3 | 1 | 1h | 低 |
| **合計** | | **15** | **5-6** | **16.5h** | |

### CIワークフロー検証手順

各Phase完了後、以下のコマンドでCIゲート通過を確認:

```powershell
# Phase S-CI 完了後
pwsh -NoProfile -ExecutionPolicy Bypass .github\scripts\check-src-atomic-dotcall.ps1     # WF-BUG-01
pwsh -NoProfile -ExecutionPolicy Bypass .github\scripts\check-work21-epochdomain-gates.ps1 # WF-BUG-02

# CI全体（各Phase完了後）
pwsh -NoProfile -ExecutionPolicy Bypass .github\scripts\check-audioengine-lint.ps1
pwsh -NoProfile -ExecutionPolicy Bypass .github\scripts\check-list-compliance.ps1
pwsh -NoProfile -ExecutionPolicy Bypass .github\scripts\check-src-size-mul-cast.ps1
```

---

## 9. 参考: 除外判断の詳細

| ID | 元報告 | ISR判断 | 判断理由 |
|:--:|:------:|:-------:|----------|
| H-04 | 音質劣化 (High) | **DESIGN CHANGE** | 線形 vs 等パワーは音響キャラクタの選択。変更する場合は正式な音響設計変更として扱うべき |
| L-01 | 表示精度 (Low) | **DESIGN** | 「内部処理レベル測定」が仕様なら正しい。測定定義の問題 |
| M-02 | 音質劣化 (Medium) | **DESIGN** | オーバーサンプリング段の高サンプルレートでは 1Hz 設定が適切 |
| M-04 | 状態管理 (Medium) | **誤報告** | `exchangeAtomic(..., false)` でリセット済み |
| H-02 | メモリリーク (High) | **過大評価** | シャットダウン時キュー破棄のみ。実害なし |

---

*本計画書は Practical Stable ISR Bridge Runtime の観点に基づき作成されました。*
