# ConvoPeq 未修正バグ改修設計書

**作成日**: 2026-07-21
**対象**: Part 7〜10 + bug1/bug2 で特定された未修正バグ（11件）
**優先度**: 重大 → 中 → 低 の順で改修

---

## 目次

1. [Bug A/B/D — NaN/Inf 伝播防止（重大）](#bug-abd--naninf-伝播防止)
2. [Bug C — pushBlock 境界チェック（重大）](#bug-c--pushblock-境界チェック)
3. [Bug 2 — bypass delayBuffer null 時未クリア（中）](#bug-2--bypass-delaybuffer-null-時未クリア)
4. [Bug 3/E — copyLatest TOCTOU（中）](#bug-3e--copylatest-toctou)
5. [Finding #9 — emitRetireIntentRT 命名（High 潜在的）](#finding-9--emitretireintentrt-命名)
6. [Finding #10 — MKL バッファ std::vector（Medium）](#finding-10--mkl-バッファ-stdvector)
7. [Bug F — StereoConvolver::init 空ブロック（低）](#bug-f--stereoconvolverinit-空ブロック)
8. [Bug G — 冗長な負値チェック（低）](#bug-g--冗長な負値チェック)
9. [Bug H — StereoConvolver::init 例外安全性（低）](#bug-h--stereoconvolverinit-例外安全性)

---

## Bug A/B/D — NaN/Inf 伝播防止

### 概要

`killDenormal()` は Release ビルドで no-op であり、NaN/Inf を通過させる。`quantize()` は NaN が比較演算を通過するため、フィルタ発散時に音声出力が完全に破綻する。

### 修正対象

| ファイル | 関数 | 変更内容 |
|---------|------|----------|
| `src/DspNumericPolicy.h` | `killDenormal(double)` | NaN/Inf チェック追加 |
| `src/DspNumericPolicy.h` | `killDenormal(float)` | NaN/Inf チェック追加 |
| `src/FixedNoiseShaper.h` | `quantize()` | NaN ガード追加 |
| `src/Fixed15TapNoiseShaper.h` | `processSample()` | fb NaN チェック追加 |
| `src/LatticeNoiseShaper.h` | `processSample()` | fb NaN チェック追加 |

### 修正案 1: `killDenormal` の NaN/Inf 対応

**ファイル**: `src/DspNumericPolicy.h`

```cpp
// ─────────────────────────────────────────────────────────────────
// デノーマル除去ヘルパー関数（libm 非依存・分岐レス）
// ─────────────────────────────────────────────────────────────────

inline double killDenormal(double x) noexcept
{
#if !defined(JUCE_DEBUG) && !defined(_DEBUG) && !defined(CONVOPEQ_DEBUG_DENORMALS)
    // Release ビルド: FTZ/DAZ が全該当スレッドで有効なためデノーマルは自動除去。
    // ただし NaN/Inf は FTZ/DAZ で除去されないため、明示的にチェックする。
    constexpr uint64_t kExpMask = 0x7FF0000000000000ULL;
    constexpr uint64_t kFracMask = 0x000FFFFFFFFFFFFFULL;

    const uint64_t bits = std::bit_cast<uint64_t>(x);
    const bool isSubnormal = ((bits & kExpMask) == 0ULL) && ((bits & kFracMask) != 0ULL);
    const bool isNanOrInf = ((bits & kExpMask) == 0x7FF0000000000000ULL);
    return (isSubnormal || isNanOrInf) ? 0.0 : x;
#else
    constexpr uint64_t kExpMask = 0x7FF0000000000000ULL;
    constexpr uint64_t kFracMask = 0x000FFFFFFFFFFFFFULL;

    const uint64_t bits = std::bit_cast<uint64_t>(x);
    const bool isSubnormal = ((bits & kExpMask) == 0ULL) && ((bits & kFracMask) != 0ULL);
    const bool isNanOrInf = ((bits & kExpMask) == 0x7FF0000000000000ULL);
    return (isSubnormal || isNanOrInf) ? 0.0 : x;
#endif
}

inline float killDenormal(float x) noexcept
{
#if !defined(JUCE_DEBUG) && !defined(_DEBUG) && !defined(CONVOPEQ_DEBUG_DENORMALS)
    constexpr uint32_t kExpMask = 0x7F800000U;
    constexpr uint32_t kFracMask = 0x007FFFFFU;

    const uint32_t bits = std::bit_cast<uint32_t>(x);
    const bool isSubnormal = ((bits & kExpMask) == 0U) && ((bits & kFracMask) != 0U);
    const bool isNanOrInf = ((bits & kExpMask) == 0x7F800000U);
    return (isSubnormal || isNanOrInf) ? 0.0f : x;
#else
    constexpr uint32_t kExpMask = 0x7F800000U;
    constexpr uint32_t kFracMask = 0x007FFFFFU;

    const uint32_t bits = std::bit_cast<uint32_t>(x);
    const bool isSubnormal = ((bits & kExpMask) == 0U) && ((bits & kFracMask) != 0U);
    const bool isNanOrInf = ((bits & kExpMask) == 0x7F800000U);
    return (isSubnormal || isNanOrInf) ? 0.0f : x;
#endif
}
```

### 修正案 2: `quantize()` の NaN ガード

**ファイル**: `src/FixedNoiseShaper.h`

```cpp
inline double quantize(double v, Xoshiro256State& rng) const noexcept
{
    // NaN/Inf ガード: NaN は比較演算を通過するため、最前面で除去
    if (!std::isfinite(v)) v = 0.0;

    const double minV = -1.0;
    const double maxV = 1.0 - (1.0 / invScale);

    // ★ 修正: クランプを先に実行し、その後ディザを加算（Lipshitz/Wannamaker 正規順序）
    if (v < minV)
        v = minV;
    else if (v > maxV)
        v = maxV;

    // TPDF dither
    const double u1 = uniform(rng);
    const double u2 = uniform(rng);
    v += (u1 + u2 - 1.0) * scale;

    __m128d d = _mm_set_sd(v * invScale);
    d = _mm_round_sd(d, d, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const double q = _mm_cvtsd_f64(d);
    return q * scale;
}
```

### 影響範囲

- `FixedNoiseShaper.h` の `quantize()` は `FixedNoiseShaper` と `Fixed15TapNoiseShaper` で共有
- `LatticeNoiseShaper.h` は別実装だが、`killDenormal` を経由するため、修正案 1 で保護される
- パフォーマンス影響: `std::isfinite()` は 1 命令（`v == v` と同等の最適化が可能）

### テスト方法

1. NaN を注入した入力でフィルタを実行し、出力が 0.0 になることを確認
2. Inf を注入した入力でフィルタを実行し、出力が 0.0 になることを確認
3. 通常動作での音質劣化がないことを確認（A/B テスト）

---

## Bug C — pushBlock 境界チェック

### 概要

`AudioSegmentBuffer::pushBlock` で `numSamples > kCapacity` の場合、リングバッファのラップアラウンド計算が破綻し、バッファオーバーフローが発生する。

### 修正対象

| ファイル | 関数 | 変更内容 |
|---------|------|----------|
| `src/AudioSegmentBuffer.h` | `pushBlock()` | 境界チェック追加 |

### 修正案

**ファイル**: `src/AudioSegmentBuffer.h`

```cpp
void pushBlock(const double* left, const double* right, int numSamples) noexcept
{
    if (left == nullptr || right == nullptr || numSamples <= 0)
        return;

    // 境界チェック: kCapacity を超えるサンプル数は切り詰め
    numSamples = std::min(numSamples, kCapacity);

    // acquire: 直前の clear/pushBlock の release と HB し、有効な writePosition を取得。
    const int currentWritePos = convo::consumeAtomic(writePosition, std::memory_order_acquire);
    int first = std::min(numSamples, kCapacity - currentWritePos);
    juce::FloatVectorOperations::copy(leftSamples + currentWritePos, left, first);
    juce::FloatVectorOperations::copy(rightSamples + currentWritePos, right, first);

    if (first < numSamples)
    {
        int second = numSamples - first;
        juce::FloatVectorOperations::copy(leftSamples, left + first, second);
        juce::FloatVectorOperations::copy(rightSamples, right + first, second);
        // release: 更新後の writePosition を読み取りスレッドに可視化。
        convo::publishAtomic(writePosition, second, std::memory_order_release);
    }
    else
    {
        int nextPos = currentWritePos + numSamples;
        if (nextPos >= kCapacity)
            nextPos = 0;
        // release: 次書き込み位置を読み取りスレッドに可視化。
        convo::publishAtomic(writePosition, nextPos, std::memory_order_release);
    }

    // acquire: clear/pushBlock の release と HB し、有効な totalSamples を取得。
    const int currentTotal = convo::consumeAtomic(totalSamples, std::memory_order_acquire);
    // release: 更新後の totalSamples を読み取りスレッドに可視化。
    convo::publishAtomic(totalSamples, std::min(kCapacity, currentTotal + numSamples), std::memory_order_release);
}
```

### 影響範囲

- 通常のブロックサイズ（最大 7680 サンプル）では影響なし
- `kCapacity = 3,840,000` サンプル（5秒 × 768kHz）の上限を事前保証
- パフォーマンス影響: `std::min` の追加（1命令のみ）

### テスト方法

1. `numSamples = kCapacity + 1` で `pushBlock` を呼び出し、クラッシュしないことを確認
2. `numSamples = kCapacity` で正常動作することを確認
3. 通常のブロックサイズで正常動作することを確認

---

## Bug 2 — bypass delayBuffer null 時未クリア

### 概要

`processBypassWithLatencyCompensation` で `delayBuffer` が null の場合、出力バッファをクリアせずに return する。バイパス遷移中に stale data が残留する。

### 修正対象

| ファイル | 関数 | 変更内容 |
|---------|------|----------|
| `src/convolver/ConvolverProcessor.Runtime.cpp` | `processBypassWithLatencyCompensation()` | delayBuffer null 時にバッファクリア追加 |

### 修正案

**ファイル**: `src/convolver/ConvolverProcessor.Runtime.cpp`

```cpp
void ConvolverProcessor::processBypassWithLatencyCompensation(juce::dsp::AudioBlock<double>& block,
                                                              const StereoConvolver& conv) noexcept
{
    const int procChannels = (std::min)(static_cast<int>(block.getNumChannels()), 2);
    const int numSamples = static_cast<int>(block.getNumSamples());
    if (procChannels == 0 || numSamples <= 0)
        return;

    double* delayBuf[2] = { delayBuffer[0].get(), delayBuffer[1].get() };
    int activeDelayCapacity = delayBufferCapacity;

    if (delayBuf[0] == nullptr || delayBuf[1] == nullptr || activeDelayCapacity < DELAY_BUFFER_SIZE)
    {
        // delayBuffer が未確保の場合は無音を出力（stale data 防止）
        for (int ch = 0; ch < procChannels; ++ch)
            juce::FloatVectorOperations::clear(block.getChannelPointer(static_cast<size_t>(ch)), numSamples);
        return;
    }
    // ... 既存の処理 ...
}
```

### 影響範囲

- `prepareToPlay()` 未呼び出し時や `releaseResources()` 後のバイパス遷移中に発生
- 通常運用ではほぼ発生しないが、シャットダウン中のグリッチ防止に有効
- パフォーマンス影響: null チェック時のメモリセット（通常は発生しないパス）

### テスト方法

1. `releaseResources()` 後に `process()` を呼び出し、クラッシュしないことを確認
2. バイパス遷移中に無音が正しく出力されることを確認
3. 通常のバイパス動作で音質劣化がないことを確認

---

## Bug 3/E — copyLatest TOCTOU

### 概要

`AudioSegmentBuffer::copyLatest` で `totalSamples` と `writePosition` を別々に読み取るため、読み取り間に書き込みが発生すると不整合なデータを読み取る可能性がある。

### 修正対象

| ファイル | 関数 | 変更内容 |
|---------|------|----------|
| `src/AudioSegmentBuffer.h` | `copyLatest()` | 読み取り順序の統一 |

### 修正案

**ファイル**: `src/AudioSegmentBuffer.h`

```cpp
int copyLatest(double* outLeft, double* outRight, int requestedSamples) const noexcept
{
    if (outLeft == nullptr || outRight == nullptr || requestedSamples <= 0)
        return 0;

    // acquire: pushBlock の release と HB し、最新の totalSamples/writePosition を取得。
    // ★ 修正: Writer の書き込み順序（writePosition → totalSamples）と一致させる。
    //   Writer は writePosition を先に release し、次に totalSamples を release する。
    //   Reader は writePosition を先に acquire し、次に totalSamples を acquire する。
    //   これにより、totalSamples を読んだ時点で writePosition も最新であることが保証される。
    const int currentWritePos = convo::consumeAtomic(writePosition, std::memory_order_acquire);
    const int currentTotal = convo::consumeAtomic(totalSamples, std::memory_order_acquire);

    const int availableSamples = std::min(requestedSamples,
        currentTotal >= kCapacity ? kCapacity : currentTotal);
    const int start = (currentWritePos - availableSamples + kCapacity) % kCapacity;

    for (int i = 0; i < availableSamples; ++i)
    {
        const int sourceIndex = (start + i) % kCapacity;
        outLeft[i] = leftSamples[sourceIndex];
        outRight[i] = rightSamples[sourceIndex];
    }

    return availableSamples;
}
```

### 重要: Writer 側の順序確認

`pushBlock` の Writer 側は以下の順序で release している：

```cpp
// pushBlock 内
convo::publishAtomic(writePosition, second, std::memory_order_release);  // ①先に release
// ...
convo::publishAtomic(totalSamples, std::min(kCapacity, currentTotal + numSamples), std::memory_order_release);  // ②後に release
```

Reader 側も `writePosition` → `totalSamples` の順で acquire する必要がある。

### 影響範囲

- 通常運用では `totalSamples` が古い値（小さい値）になるため、読み取り量が少なくなるだけ（安全側に倒れる）
- 修正後は Reader が常に最新の Writer 状態を観測できる
- パフォーマンス影響: なし（メモリバリアの順序変更のみ）

### テスト方法

1. Writer スレッドと Reader スレッドを並列で実行し、不整合がないことを確認
2. 高負荷下でもデータの整合性が保たれることを確認
3. 通常動作でのパフォーマンス劣化がないことを確認

---

## Finding #9 — emitRetireIntentRT 命名

### 概要

`emitRetireIntentRT()` は関数名から「RT スレッドから安全に呼べる版」を示唆するが、実装は `emitRetireIntent()` を素通しで、mutex ロック経路を含む。将来 Audio Thread から誤呼び出しを追加するリスクが高い。

### 修正対象

| ファイル | 関数 | 変更内容 |
|---------|------|----------|
| `src/audioengine/ISRRetire.h` | `emitRetireIntentRT()` | 名前変更 + コメント追加 |
| `src/audioengine/ISRRetire.cpp` | `emitRetireIntentRT()` | 名前変更 |
| `src/audioengine/AudioEngine.Commit.cpp` | 呼び出し元 | 名前変更に追従 |

### 修正案（段階的アプローチ）

#### Phase 1: コメント追加（即時対応）

**ファイル**: `src/audioengine/ISRRetire.h`

```cpp
/**
 * Retire runtime
 */
class RetireRuntime
{
public:
    // Generic helper used internally / non-RT paths.
    void emitRetireIntent(const RetireIntent& intent) noexcept;

    /**
     * ★ R-9: 命名注意 — 「RT」は RealTime thread safety を意味しない。
     *   実装は emitRetireIntent() を素通しし、輻輳時に std::mutex をロックする。
     *   現時点では呼び出し元は全て非 RT スレッドであることを確認済み。
     *   将来 Audio Thread から呼び出す場合は、mutex を使わない別実装を用意すること。
     *
     * Preferred API for runtime retire intent publication from commit path.
     */
    void emitRetireIntentRT(const RetireIntent& intent) noexcept;

    // ★ B14: Vyukov MPSC 新 API
    // ...
};
```

#### Phase 2: 名前変更（長期対応）

```cpp
// 変更前
void emitRetireIntentRT(const RetireIntent& intent) noexcept;

// 変更後
void emitRetireIntentFromNonRT(const RetireIntent& intent) noexcept;
```

### 影響範囲

- 呼び出し元は `AudioEngine.Commit.cpp` の `onRuntimeRetiredNonRt()` のみ（`ASSERT_NON_RT_THREAD()` あり）
- 名前変更は ABI 破壊変更のため、バージョンアップ時に実施
- Phase 1 のコメント追加は即時実施可能

### テスト方法

1. コメント追加後、コンパイルエラーがないことを確認
2. 名前変更後、呼び出し元が正しく更新されていることを確認
3. Debug ビルドで `ASSERT_NON_RT_THREAD()` が正しく動作することを確認

---

## Finding #10 — MKL バッファ std::vector

### 概要

MKL DFTI API を直接呼ぶ関数内で `std::vector<double>` / `std::make_unique` が使用されており、 Finding #2（`IRAnalyzer.cpp`）と同型の規約違反。

### 修正対象

| ファイル | 変更内容 |
|---------|----------|
| `src/convolver/ConvolverProcessor.MixedPhase.cpp` | `std::vector<double>` → `convo::makeAlignedArray<double>` |
| `src/convolver/ConvolverProcessor.ResampleAndFallback.cpp` | `std::vector<int>` → 固定サイズ配列 or アライン確保 |

### 修正案: 機械的監査と一括置換

#### Step 1: MKL ファイルの列挙

```bash
# MKL ヘッダを含むファイルを検索
grep -rn "#include <mkl.h>\|#include <mkl_dfti.h>" src/
```

#### Step 2: 各ファイル内の std::vector/std::make_unique の列挙

```bash
# MKL ファイル内の std::vector/std::make_unique を検索
grep -n "std::vector\|std::make_unique" src/convolver/ConvolverProcessor.MixedPhase.cpp
```

#### Step 3: 一括置換（例: `phiMinUnwrapped`）

```cpp
// 変更前
std::vector<double> phiMinUnwrapped(static_cast<size_t>(complexSize));

// 変更後
auto phiMinUnwrapped = convo::makeAlignedArray<double>(static_cast<size_t>(complexSize));
```

#### Step 4: 使用箇所の更新

```cpp
// 変更前
phiMinUnwrapped[0] = std::atan2(...);

// 変更後（makeAlignedArray は std::unique_ptr を返すため）
phiMinUnwrapped[0] = std::atan2(...);
// ポインタとして使用する場合: phiMinUnwrapped.get()[0]
```

### 影響範囲

- 非 RT スレッド専用（IR ロード/変換パイプライン）
- Audio Thread 規約への抵触はないが、一貫性の欠如を解消
- パフォーマンス影響: アライン確保による L1 キャッシュヒット率の向上（可能性あり）

### テスト方法

1. MKL 関数が正しく動作することを確認（FFT 結果の整合性）
2. メモリリークがないことを確認（Valgrind / AddressSanitizer）
3. 通常動作でのパフォーマンス劣化がないことを確認

---

## Bug F — StereoConvolver::init 空ブロック

### 概要

`StereoConvolver::init` で `ownerProcessor != nullptr` 時の処理ブロックが空。意図された処理（例: レイテンシ通知）が欠落している可能性。

### 修正対象

| ファイル | 関数 | 変更内容 |
|---------|------|----------|
| `src/ConvolverProcessor.h` | `StereoConvolver::init()` | 空ブロックの確認・修正 |

### 修正案

**ファイル**: `src/ConvolverProcessor.h`

```cpp
if (ownerProcessor != nullptr)
{
    // 現時点では特別な処理不要（レイテンシ通知は process() 側で処理）
    // 将来 ownerProcessor へのコールバックが必要になった場合はここに追加
}
```

### 影響範囲

- 現時点では機能的な影響なし
- 将来の拡張ポイントとして記録

### テスト方法

1. コンパイルエラーがないことを確認
2. 通常動作で問題がないことを確認

---

## Bug G — 冗長な負値チェック

### 概要

`processBypassWithLatencyCompensation` で `if (readPos < 0)` は C++20 では冗長。C++20 以前のコンパイラでコンパイルした場合に問題になる可能性。

### 修正対象

| ファイル | 関数 | 変更内容 |
|---------|------|----------|
| `src/convolver/ConvolverProcessor.Runtime.cpp` | `processBypassWithLatencyCompensation()` | 冗長チェックの削除 or コメント追加 |

### 修正案

**ファイル**: `src/convolver/ConvolverProcessor.Runtime.cpp`

```cpp
// 2) 遅延した信号を出力へ戻す
// C++20: 二の補数が義務付けられているため、`& DELAY_BUFFER_MASK` は常に非負。
// C++17 以前: コンパイラ依存のため、冗長でも安全のため残す。
int readPos = (writePos - delaySamples) & DELAY_BUFFER_MASK;
// if (readPos < 0)  // C++20 では不要（常に false）
//     readPos += DELAY_BUFFER_SIZE;
```

### 影響範囲

- C++20 環境では変更なし（最適化で自動除去）
- C++17 環境では安全のため残す
- パフォーマンス影響: なし

### テスト方法

1. C++20 でコンパイルし、警告がないことを確認
2. 通常動作で問題がないことを確認

---

## Bug H — StereoConvolver::init 例外安全性

### 概要

`StereoConvolver::init` で `std::bad_alloc` のみキャッチ。他の例外時は `irData` がリークする。

### 修正対象

| ファイル | 関数 | 変更内容 |
|---------|------|----------|
| `src/ConvolverProcessor.h` | `StereoConvolver::init()` | 全例外パターンをキャッチ |

### 修正案

**ファイル**: `src/ConvolverProcessor.h`

```cpp
try
{
    auto nuc0 = convo::aligned_make_unique<convo::MKLNonUniformConvolver>();
    auto nuc1 = convo::aligned_make_unique<convo::MKLNonUniformConvolver>();

    if (nuc0->SetImpulse(irData[0], irDataLength, knownBlockSize, scale, enableDirectHead, filterSpec) &&
        nuc1->SetImpulse(irData[1], irDataLength, knownBlockSize, scale, enableDirectHead, filterSpec))
    {
        destroyNUCConvolver(nucConvolvers[0]);
        destroyNUCConvolver(nucConvolvers[1]);
        nucConvolvers[0] = nuc0.release();
        nucConvolvers[1] = nuc1.release();

        latency   = nucConvolvers[0]->getLatency();
        DBG("Convolver: NUC Engine Active. Latency: " << latency << " samples");
        if (ownerProcessor != nullptr)
        {
        }
        return true;
    }
}
catch (const std::bad_alloc&)
{
    // Fall through to cleanup on memory allocation failure
}
catch (const std::exception&)
{
    // その他の例外もクリーンアップに進む
}
catch (...)
{
    // 不明な例外もクリーンアップに進む
}

// NUC セットアップ失敗 or メモリ確保失敗
destroyNUCConvolver(nucConvolvers[0]);
destroyNUCConvolver(nucConvolvers[1]);
if (irData[0]) { convo::aligned_free(irData[0]); irData[0] = nullptr; }
if (irData[1]) { convo::aligned_free(irData[1]); irData[1] = nullptr; }
irDataLength = 0;
latency = 0;
this->irLatency = 0;
return false;
```

### 影響範囲

- 例外発生時のメモリリーク防止
- 通常運用ではほぼ発生しない
- パフォーマンス影響: なし（例外ハンドリングは通常パスを通過しない）

### テスト方法

1. `std::bad_alloc` 発生時に正しくクリーンアップされることを確認
2. 通常動作で問題がないことを確認
3. メモリリークがないことを確認（Valgrind / AddressSanitizer）

---

## 改修優先度とスケジュール

### 優先度 1: 重大（即時対応）

| Bug | 改修内容 | 工数（推定） |
|-----|----------|-------------|
| Bug A/B/D | `killDenormal` + `quantize` の NaN/Inf 対応 | 0.5 日 |
| Bug C | `pushBlock` 境界チェック | 0.5 日 |

### 優先度 2: 中（1週間以内）

| Bug | 改修内容 | 工数（推定） |
|-----|----------|-------------|
| Bug 2 | `processBypassWithLatencyCompensation` null クリア | 0.5 日 |
| Bug 3/E | `copyLatest` TOCTOU 修正 | 1 日 |
| Finding #10 | MKL バッファ std::vector 置換 | 2 日 |

### 優先度 3: 低（余裕があれば）

| Bug | 改修内容 | 工数（推定） |
|-----|----------|-------------|
| Finding #9 | `emitRetireIntentRT` コメント追加 | 0.5 日 |
| Bug F | 空ブロックの確認 | 0.5 日 |
| Bug G | 冗長チェックの整理 | 0.5 日 |
| Bug H | 例外安全性の改善 | 0.5 日 |

### 総工数推定

- 優先度 1: 1 日
- 優先度 2: 3.5 日
- 優先度 3: 2 日
- **合計: 6.5 日**

---

## テスト計画

### 単体テスト

1. **NaN/Inf テスト**: NaN/Inf を注入した入力で全ノイズシェイパーをテスト
2. **境界テスト**: `pushBlock` で `numSamples > kCapacity` の場合をテスト
3. **null テスト**: `processBypassWithLatencyCompensation` で delayBuffer null の場合をテスト

### 統合テスト

1. **通常動作テスト**: 全修正後に音質劣化がないことを確認
2. **パフォーマンステスト**: 修正前後のパフォーマンス比較
3. **メモリテスト**: メモリリークがないことを確認（Valgrind / AddressSanitizer）

### リグレッションテスト

1. **既存テスト実行**: 全既存テストがパスすることを確認
2. **手動テスト**: 実機での音質確認

---

## リスク評価

### 修正によるリスク

| リスク | 確率 | 影響 | 対策 |
|--------|------|------|------|
| パフォーマンス劣化 | 低 | 中 | ベンチマーク実行 |
| 既存機能の破壊 | 低 | 高 | リグレッションテスト |
| 新規バグの導入 | 中 | 中 | 段階的な修正とレビュー |

### 未修正によるリスク

| リスク | 確率 | 影響 | 対策 |
|--------|------|------|------|
| 音声グリッチ | 中 | 高 | Bug A/B/D の修正 |
| メモリ破損 | 低 | 高 | Bug C の修正 |
| 将来の誤呼び出し | 高 | 中 | Finding #9 の修正 |

---

## 承認

| 役割 | 名前 | 日付 | 署名 |
|------|------|------|------|
| 設計者 | | | |
| レビュアー | | | |
| 承認者 | | | |
