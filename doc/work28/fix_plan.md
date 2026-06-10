# newbug.md 改修計画

**作成日**: 2026-06-10
**ベース**: `doc/work28/newbug_validation_report.md` 検証結果
**除外項目**: M-02（オーディオ的観点で正しい）、M-04（誤報告）

---

## 改修一覧

| # | ID | タイトル | 改修難易度 | リスク | ファイル影響範囲 |
|:-:|:--:|----------|:----------:|:------:|:----------------|
| 1 | C-01 | LinearRamp RTスレッドセーフ違反 | ★★☆ | 中 | `DspNumericPolicy.h`, `ConvolverProcessor.Runtime.cpp` |
| 2 | C-02 | union型パンニング→std::bit_cast | ★☆☆ | 低 | `AudioEngine.h`, `EQProcessor.Processing.cpp`, `ConvolverProcessor.Runtime.cpp`, `DspNumericPolicy.h` |
| 3 | C-03 | ProgressiveUpgradeThread Use-After-Free | ★★☆ | 中 | `ProgressiveUpgradeThread.cpp` |
| 4 | H-04 | 線形→等パワークロスフェード | ★☆☆ | 低 | `AudioBlock.cpp`, `BlockDouble.cpp`, `ConvolverProcessor.Runtime.cpp` |
| 5 | H-05 | finalizeNUCEngine例外補完 | ★☆☆ | 低 | `ConvolverProcessor.LoadPipeline.cpp` |
| 6 | M-01 | std::abs→absNoLibm置換 | ★☆☆ | 低 | `MKLNonUniformConvolver.cpp` |
| 7 | H-03 | IR逆順後irFreqReal/Imag不整合 | ★★☆ | 中 | `MKLNonUniformConvolver.cpp` |
| 8 | M-03 | NoiseShaperインクリメンタル最大値追跡 | ★★☆ | 低 | `Fixed15TapNoiseShaper.h` |
| 9 | M-05 | wetBuf[0]流用の解消 | ★☆☆ | 低 | `ConvolverProcessor.Runtime.cpp`, `ConvolverProcessor.h` |
| 10 | L-01 | レベルメーター測定タイミング修正 | ★☆☆ | 低 | `DSPCoreDouble.cpp`, `DSPCoreFloat.cpp` |
| 11 | L-02 | HC else到達不能ブランチ削除 | ★☆☆ | 低 | `MKLNonUniformConvolver.cpp` |
| 12 | L-03 | 命名/コメント修正 | ★☆☆ | 低 | `Fixed15TapNoiseShaper.h`, `EQProcessor.Coefficients.cpp` |
| 13 | H-01 | m_rtBypassShadow設計明確化 | ★☆☆ | 低 | `EQProcessor.h`, `EQProcessor.Processing.cpp`（コメントのみ） |
| 14 | H-02 | callAsyncエラーハンドリング補強 | ★☆☆ | 低 | `ConvolverProcessor.LoaderThread.cpp` |

---

## 各改修の詳細設計

---

### 1. C-01: LinearRamp RTスレッドセーフ違反

**問題**: `LinearRamp::setCurrentAndTargetValue()` に `ASSERT_NON_RT_THREAD()` が付いているが、`ConvolverProcessor::process()`（Audio Thread）から6箇所で呼ばれている。

**改修方針**: `LinearRamp` に Audio Thread 安全な `forceSetCurrentAndTargetValueRT()` を追加。内部でアサーションのみ `ASSERT_AUDIO_THREAD()` に変更し、機能的には同じ（単なる値の代入＋ランプ無効化）。

**変更内容**:

```cpp
// DspNumericPolicy.h - LinearRamp に追加
/// Audio Thread から呼び出す緊急用。current/target を即座に同一値に設定する。
/// 世代カウンターによる同期が呼び出しの安全性を保証している場合に使用。
void forceSetCurrentAndTargetValueRT(double v) noexcept
{
    ASSERT_AUDIO_THREAD();
    current = target = v;
    step      = 0.0;
    remaining = 0;
}
```

**対象ファイル**: `src/DspNumericPolicy.h`

**呼び出し元修正**: `ConvolverProcessor.Runtime.cpp` の該当6箇所を `setCurrentAndTargetValue` → `forceSetCurrentAndTargetValueRT` に置換。

| 行 | 現状 | 修正後 |
|:--:|------|--------|
| 246 | `activeCrossfadeGain.setCurrentAndTargetValue(0.0);` | `activeCrossfadeGain.forceSetCurrentAndTargetValueRT(0.0);` |
| 279 | `activeLatencySmoother.setCurrentAndTargetValue(val);` | `activeLatencySmoother.forceSetCurrentAndTargetValueRT(val);` |
| 295 | `activeCrossfadeGain.setCurrentAndTargetValue(0.0);` | `activeCrossfadeGain.forceSetCurrentAndTargetValueRT(0.0);` |
| 307 | `activeMixSmoother.setCurrentAndTargetValue(...)` | `activeMixSmoother.forceSetCurrentAndTargetValueRT(...);` |
| 323 | `activeMixSmoother.setCurrentAndTargetValue(currentVal);` | `activeMixSmoother.forceSetCurrentAndTargetValueRT(currentVal);` |
| 491 | `activeLatencySmoother.setCurrentAndTargetValue(...)` | `activeLatencySmoother.forceSetCurrentAndTargetValueRT(...);` |

**検証**: デバッグビルドで `jassert` が発火しないことを確認。`Strict Atomic Dot-Call Scan` + `Debug Build` タスク。

---

### 2. C-02: union型パンニング→std::bit_cast

**問題**: `absNoLibm`, `killDenormal`, `isFiniteNoLibm`, `isFiniteAndAbsInRangeMask` 等で C++ UB の `union` 型パンニングを使用。

**改修方針**: 該当関数の `union` パターンを `std::bit_cast` (C++20) に置換。

**テンプレート**:

```cpp
// Before
union { double d; uint64_t u; } v { x };
v.u &= 0x7FFFFFFFFFFFFFFFULL;
return v.d;

// After
return std::bit_cast<double>(std::bit_cast<uint64_t>(x) & 0x7FFFFFFFFFFFFFFFULL);
```

**該当関数一覧**:

| ファイル | 関数 | パターン |
|----------|------|----------|
| `AudioEngine.h` | `absNoLibm` | double→uint64_t masked→double |
| `EQProcessor.Processing.cpp` | `absNoLibm` (匿名NS) | 同上 |
| `ConvolverProcessor.Runtime.cpp` | `isFiniteAndAbsBelowNoLibm` | double→uint64_t 検査 |
| `DspNumericPolicy.h` | `killDenormal` | double→uint64_t ビット操作 |
| `DspNumericPolicy.h` | `isFiniteNoLibm` (x64版) | 同上 |

**注意点**: `isFiniteNoLibm` の非x64版 (fallback) は libm 非依存の整数演算。そちらに統一する選択肢もあり。

**検証**: デバッグ/リリース両ビルドで数値結果が同一であることを確認。`absNoLibm` 使用箇所の広範囲に影響するため、回帰テスト必須。

---

### 3. C-03: ProgressiveUpgradeThread Use-After-Free

**問題**: `cancelledFlag = &cancelled`（生ポインタ）をラムダにキャプチャ。オブジェクト破棄後もラムダが実行される可能性。

**改修方針**: 生ポインタの代わりに `weakOwner` 経由でアクセスする。または `std::shared_ptr` でオブジェクトの寿命を延長。

**変更内容**:

**選択肢A（推奨）**: `weakOwner` 経由でアクセス（既存の weakOwner パターンに統一）

```cpp
// ProgressiveUpgradeThread.cpp:upgradeStep
// Before
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

// After
const uint64_t expectedGeneration = taskGeneration;

prepared = converter.convertToHighRes(irFile, ...,
    [weakOwner, expectedGeneration]() {
        auto* owner = weakOwner.get();
        if (owner == nullptr) return true;
        return juce::Thread::currentThreadShouldExit()
            || convo::consumeAtomic(owner->cancelled, ...)  // weakOwner 経由でアクセス
            || !owner->isConvolverGenerationCurrent(expectedGeneration);
    });
```

**選択肢B**: `shared_from_this()` + `weak_ptr` を使用する場合、`ProgressiveUpgradeThread` を `std::enable_shared_from_this` から継承させる必要があるため、変更範囲が大きい。

**検証**: スレッド実行中に `ProgressiveUpgradeThread` を破棄してもクラッシュしないことを確認。

---

### 4. H-04: 線形→等パワークロスフェード

**問題**: 2箇所で線形クロスフェード（`gOld = 1.0 - gNew`）を使用。中点で -3dB ディップ。

**改修方針**: 既存の `equalPowerSin()` 関数（`ConvolverProcessor.Runtime.cpp` に定義済み）を使用して等パワークロスフェード化。

**変更内容**:

**対象1**: `AudioEngine.Processing.AudioBlock.cpp:267`

```cpp
// Before
const double gOld = 1.0 - gNew;

// After
const double gNew_eq = equalPowerSin(gNew);
const double gOld_eq = equalPowerSin(1.0 - gNew);
// 以降 gNew → gNew_eq, gOld → gOld_eq に置換
```

**対象2**: `AudioEngine.Processing.BlockDouble.cpp:233`

```cpp
// Before
const double gOld = 1.0 - gNew;

// After
const double gNew_eq = equalPowerSin(gNew);
const double gOld_eq = equalPowerSin(1.0 - gNew);
```

**`equalPowerSin` の共通化**: 現在 `ConvolverProcessor.Runtime.cpp` の匿名名前空間に定義されている `equalPowerSin` を共通ヘッダ（例: `DspNumericPolicy.h`）に移動し、AudioBlock.cpp / BlockDouble.cpp からも使用可能にする。

**注意**: `equalPowerSin` は Audio Thread 安全（多項式近似、libm 不使用）。既存の実装をそのまま流用可能。

**検証**: クロスフェード中点での出力レベルが -3dB から -3.01dB（等パワー）に改善することを確認。

---

### 5. H-05: finalizeNUCEngine例外補完

**問題**: `std::bad_alloc` のみ catch。`std::runtime_error` 等が未捕捉。

**変更内容** (`ConvolverProcessor.LoadPipeline.cpp`):

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

**検証**: ビルド成功のみ（例外経路テストは困難なため、コードレビューで確認）。

---

### 6. M-01: std::abs→absNoLibm置換

**問題**: `MKLNonUniformConvolver::Get()` 内で `std::abs` を Audio Thread で使用。

**変更内容** (`MKLNonUniformConvolver.cpp:1403`):

```cpp
// Before
if (std::abs(gain - 1.0) < 1.0e-12)

// After
if (absNoLibm(gain - 1.0) < 1.0e-12)
```

**`absNoLibm` のインクルード確認**: `MKLNonUniformConvolver.cpp` は `AudioEngine.h` を include していない可能性。`DspNumericPolicy.h` 経由で利用可能か確認し、必要に応じてインクルード追加。

**検証**: デバッグ/リリースビルド。数値結果の一致確認。

---

### 7. H-03: IR逆順後irFreqReal/Imag不整合

**問題**: `SetImpulse()` で `irFreqDomain` のみ逆順化され、`irFreqReal/irFreqImag` は更新されない。`filterSpec != nullptr` 時のみ `applySpectrumFilter` の再デインターリーブで回復するが、潜在バグ。

**改修方針**: 逆順化ループ内で `irFreqReal/irFreqImag` も同時に swap する。

**変更内容** (`MKLNonUniformConvolver.cpp:816-834`):

```cpp
// Before: irFreqDomain のみ swap
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

// After: irFreqDomain + irFreqReal/Imag も同時 swap
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

            // irFreqDomain swap
            double* slotF = l.irFreqDomain + pf * l.partStride;
            double* slotB = l.irFreqDomain + pb * l.partStride;
            memcpy(swapDomain, slotF, l.partStride * sizeof(double));
            memcpy(slotF,      slotB, l.partStride * sizeof(double));
            memcpy(slotB,      swapDomain, l.partStride * sizeof(double));

            // irFreqReal swap
            memcpy(swapSoA,
                   l.irFreqReal + static_cast<size_t>(pf) * l.complexSize,
                   l.complexSize * sizeof(double));
            memcpy(l.irFreqReal + static_cast<size_t>(pf) * l.complexSize,
                   l.irFreqReal + static_cast<size_t>(pb) * l.complexSize,
                   l.complexSize * sizeof(double));
            memcpy(l.irFreqReal + static_cast<size_t>(pb) * l.complexSize,
                   swapSoA, l.complexSize * sizeof(double));

            // irFreqImag swap
            memcpy(swapSoA,
                   l.irFreqImag + static_cast<size_t>(pf) * l.complexSize,
                   l.complexSize * sizeof(double));
            memcpy(l.irFreqImag + static_cast<size_t>(pf) * l.complexSize,
                   l.irFreqImag + static_cast<size_t>(pb) * l.complexSize,
                   l.complexSize * sizeof(double));
            memcpy(l.irFreqImag + static_cast<size_t>(pb) * l.complexSize,
                   swapSoA, l.complexSize * sizeof(double));
        }
    }
    if (swapDomain) mkl_free(swapDomain);
    if (swapSoA)    mkl_free(swapSoA);
}
```

**注意**: `applySpectrumFilter` 内の再デインターリーブは冗長になるが、削除しない（防御的プログラミング）。ただしコメントで「逆順化と同時に SoA も更新済み」と追記。

**検証**: `filterSpec == nullptr` の状態を作り、Split-complex パスの出力が正しいことを確認。

---

### 8. M-03: NoiseShaperインクリメンタル最大値追跡

**問題**: `processSample` 内で毎サンプル `ORDER=16` 全エントリを走査。

**改修方針**: ブロック単位でインクリメンタルに最大値を追跡し、ブロック終了時に一度だけチェック。

**変更内容** (`Fixed15TapNoiseShaper.h`):

```cpp
// メンバ変数追加
double currentMaxAbsError = 0.0;

// processSample 内の走査を削除し、代わりにインクリメンタル更新
// Before (processSample 末尾):
double maxAbs = 0.0;
for (int i = 0; i < ORDER; ++i)
{
    const double absVal = absNoLibm(channelErrors[static_cast<size_t>(i)]);
    if (absVal > maxAbs)
        maxAbs = absVal;
}
if (maxAbs > kErrorStateThreshold)
    convo::publishAtomic(needsReset, true, std::memory_order_release);

// After (processSample 末尾):
const double absErr = absNoLibm(error);
if (absErr > currentMaxAbsError)
    currentMaxAbsError = absErr;
```

```cpp
// processStereoBlock 終了時に一度だけチェック（return 直前）
if (currentMaxAbsError > kErrorStateThreshold)
    convo::publishAtomic(needsReset, true, std::memory_order_release);
currentMaxAbsError = 0.0; // リセット
```

**注意**: `channelErrors` 配列はリングバッファであり、過去の最大値が上書きされる可能性がある。インクリメンタル追跡では真の最大値を過小評価しうる。このトレードオフを受け入れるか、完全な正確性が必要なら別手法（例: 間引きチェック）を検討。

**検証**: ノイズシェーパーのリセット判定が正常に動作することを確認。

---

### 9. M-05: wetBuf[0]流用の解消

**問題**: `double* delayFadeRamp = wetBuf[0];` で Wet 信号バッファをゲイン値格納に流用。

**改修方針**: `ConvolverProcessor` に専用の `delayFadeRamp` バッファを追加するか、スタック配列を使用。

**選択肢A（推奨: スタック配列）**:

```cpp
// ConvolverProcessor.Runtime.cpp 内
constexpr int kMaxFadeSamples = 4096; // または他の適切な最大値
// double* delayFadeRamp = wetBuf[0];  // Before
double delayFadeRamp[kMaxFadeSamples]; // After: スタック配列
```

**選択肢B（動的確保）**: `ConvolverProcessor.h` に `convo::ScopedAlignedPtr<double> delayFadeRampBuffer` を追加し、`prepareToPlay` で `maxBlockSize` 分を確保。

**検証**: クロスフェード処理が正しく動作することを確認。

---

### 10. L-01: レベルメーター測定タイミング修正

**問題**: `measureLevel` が `processOutputDouble`（ヘッドルーム適用）の前に呼ばれる。

**改修方針**: `measureLevel` を `processOutputDouble` の後に移動。ただし測定対象バッファがヘッドルーム適用済みであることを考慮。

**変更内容** (`DSPCoreDouble.cpp:520-524`):

```cpp
// Before
const float outputLinear = measureLevel(originalBlock);
if (outputLevelLinear != nullptr)
    convo::publishAtomic(*outputLevelLinear, outputLinear, std::memory_order_release);

processOutputDouble(buffer, numSamples, state);

// After
processOutputDouble(buffer, numSamples, state);

const float outputLinear = measureLevel(originalBlock);
if (outputLevelLinear != nullptr)
    convo::publishAtomic(*outputLevelLinear, outputLinear, std::memory_order_release);
```

**注意**: `originalBlock` が `processOutputDouble` 後に有効なバッファを指していることを確認。`processOutputDouble` は `buffer` を処理するが、`originalBlock` が `buffer` の内部バッファを指している場合は測定値がヘッドルーム適用後になる。

**同様の修正**: `DSPCoreFloat.cpp:279` も同様。

**検証**: レベルメーターの表示値が約 1dB 低下することを確認。

---

### 11. L-02: HC else到達不能ブランチ削除

**問題**: `hcFcEnd = nyquist` のため `k > kEnd` が決して真にならない `else` ブランチが存在。

**変更内容** (`MKLNonUniformConvolver.cpp:352-370`):

```cpp
// Before
for (int k = 0; k < cSize; ++k)
{
    if (k <= kStart)
    {
        // パスバンド: ゲイン 1.0
    }
    else if (k <= kEnd)
    {
        // テーパー
    }
    else
    {
        gain[k] = 0.0;  // ← 到達不能
    }
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
        // テーパー (kStart < k <= kEnd、hcFcEnd=nyquist なので k>kEnd は発生しない)
    }
}
```

**検証**: ビルド成功＋フィルター特性の同一性確認。

---

### 12. L-03: 命名/コメント修正

**1. `Fixed15TapNoiseShaper.h`**:

```cpp
// Before
static constexpr int ORDER = 16;  // クラス名は15-Tap

// After
static constexpr int ORDER = 16;  // 16-Tapフィルタ（タップ数=次数、クラス名は保守的命名のため維持）
// またはクラス名を Fixed16TapNoiseShaper に変更（破壊的変更のため非推奨）
```

※ `ORDER = 15` への変更はフィルタ特性を変えるため不可。コメントのみ修正。

**2. `EQProcessor.Coefficients.cpp`**:

```cpp
// Before
// SVF係数計算 (Audio Thread用)

// After
// SVF係数計算 (Message Thread専用: std::pow/std::tan を含む)
```

---

### 13. H-01: m_rtBypassShadow設計明確化

**問題**: 機能的には動作しているが、`m_rtBypassShadow` の初期値と同期経路が不明瞭。

**変更内容** (`EQProcessor.Processing.cpp:474`):

```cpp
// コメント追記
const bool requestedBypass = m_rtBypassShadow;
// m_rtBypassShadow は DSPCore::processDoubleToBuffer/FloatToBuffer 内で
// state.eqBypassed (RuntimeSnapshot由来) から setBypassFromRT() 経由で設定される。
// 初期値 false (= 非バイパス) は初回 process() 呼び出し前に DSPCore が設定するまで有効。
```

**検証**: コメント変更のみのため、ビルド確認のみ。

---

### 14. H-02: callAsyncエラーハンドリング補強

**問題**: シャットダウン時のキュー破棄による理論上のメモリリーク。実害はほぼないが、予防的対応。

**変更内容** (`ConvolverProcessor.LoaderThread.cpp`):
`queueFinalizeOnMessageThread` 内の `callAsync` 失敗時のクリーンアップコードを確認し、既に適切に処理されていることをコメントで明記。

```cpp
// エラーハンドリング: 既に unique_ptr/ScopedAlignedPtr で解放済み。
// シャットダウン時のキュー破棄によるラムダ未実行は許容範囲。
```

（コード変更不要。コメント追加のみ。）

---

## フェーズ分割と実施順序

### Phase 0: 準備（全フェーズ共通）

- `git branch fix/newbug-phase1` を作成
- デバッグ/リリースビルドの事前確認

### Phase 1: Critical（1PR = 1日）

| 順序 | タスク | 見積時間 |
|:----:|--------|:--------:|
| 1.1 | C-02: union→std::bit_cast（全ファイル） | 2時間 |
| 1.2 | C-01: LinearRamp forceSetCurrentAndTargetValueRT 追加 | 1時間 |
| 1.3 | C-03: ProgressiveUpgradeThread weakOwner 経由アクセス | 1時間 |

### Phase 2: High（1〜2PR、2日）

| 順序 | タスク | 見積時間 |
|:----:|--------|:--------:|
| 2.1 | H-04: 等パワークロスフェード（equalPowerSin共通化含む） | 2時間 |
| 2.2 | H-05: 例外補完 | 0.5時間 |
| 2.3 | M-01: std::abs→absNoLibm | 0.5時間 |
| 2.4 | H-03: IR逆順 + SoA同時swap | 2時間 |

### Phase 3: Medium（1PR、1日）

| 順序 | タスク | 見積時間 |
|:----:|--------|:--------:|
| 3.1 | M-03: NoiseShaperインクリメンタル追跡 | 1.5時間 |
| 3.2 | M-05: wetBuf[0]→スタック配列 | 0.5時間 |
| 3.3 | L-01: レベルメータータイミング修正 | 0.5時間 |

### Phase 4: Low（まとめて1PR、0.5日）

| 順序 | タスク | 見積時間 |
|:----:|--------|:--------:|
| 4.1 | L-03: コメント修正 | 0.5時間 |
| 4.2 | L-02: 到達不能 else 削除 | 0.5時間 |
| 4.3 | H-01: コメント追記 | 0.25時間 |
| 4.4 | H-02: コメント追記 | 0.25時間 |

---

## 総合スケジュール

| Phase | 内容 | PR数 | 工数 | リスク |
|:-----:|------|:----:|:----:|:------:|
| 1 | Critical 3件 | 1 | 4h | 中（std::bit_cast は広範囲） |
| 2 | High 4件 | 1-2 | 5h | 中（H-03 はデバッグ難） |
| 3 | Medium 3件 | 1 | 2.5h | 低 |
| 4 | Low 4件 | 1 | 1.5h | 低 |
| **合計** | **14件** | **4-5** | **13h** | |

---

## 各改修の検証方法

| # | ID | ビルド確認 | 単体テスト | 動作確認 | 注意点 |
|:-:|:--:|:----------:|:----------:|:--------:|--------|
| 1 | C-01 | ✅ Debug/Release | — | Debug起動でアサート消滅 | `ASSERT_AUDIO_THREAD` が正であること |
| 2 | C-02 | ✅ Debug/Release | — | 数値結果一致確認 | 全 `absNoLibm` 呼び出しに影響 |
| 3 | C-03 | ✅ | — | スレッド破棄テスト | タイミング依存のため再現困難 |
| 4 | H-04 | ✅ | — | クロスフェード中点レベル測定 | 等パワーは約 -3.01dB |
| 5 | H-05 | ✅ | — | — | コードレビューで十分 |
| 6 | M-01 | ✅ | — | — | 数値一致確認 |
| 7 | H-03 | ✅ | `filterSpec=null` テスト | IR逆順パスで音声確認 | Split-complex パスのみ影響 |
| 8 | M-03 | ✅ | ✅ リセット判定テスト | — | 最大値過小評価リスクに注意 |
| 9 | M-05 | ✅ | — | クロスフェード動作確認 | — |
| 10 | L-01 | ✅ | — | レベルメーター値確認 | 約 1dB 減少 |
| 11 | L-02 | ✅ | — | — | 機能的影響なし |
| 12 | L-03 | ✅ | — | — | コメントのみ |
| 13 | H-01 | ✅ | — | — | コメントのみ |
| 14 | H-02 | ✅ | — | — | コメントのみ |

---

## 参考: 除外項目

| ID | 理由 |
|:--:|------|
| **M-02** | オーバーサンプリング段の DC ブロッカーは高いサンプルレートで動作するため、1 Hz 設定が適切。オーディオ的観点で正しい設計。 |
| **M-04** | `exchangeAtomic(observeMonotonicRollbackRequested_, false, ...)` で正しくリセットされている。誤報告。 |
