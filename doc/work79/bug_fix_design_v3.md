# ConvoPeq 未修正バグ改修設計書（v3 — 最終レビュー反映版）

**作成日**: 2026-07-21
**レビュー反映**: 2026-07-21（v2 レビュー → v3 反映）
**対象**: Part 7〜10 + bug1/bug2 で特定された未修正バグ（11件）
**優先度**: P0（即時）→ P1（近日）→ P2（改善）→ P3（保留）の順

---

## 変更履歴

| バージョン | 日付 | 変更内容 |
|-----------|------|----------|
| v1 | 2026-07-21 | 初版作成 |
| v2 | 2026-07-21 | v1 レビュー反映：Bug3/E 不採用、sanitizeFinite 分離、BugH RAII 化優先、Finding10 位置付け変更 |
| v3 | 2026-07-21 | v2 レビュー反映：sanitizeFinite 説明修正、NaN 方針の DSP 全体整理、BugC jassert 追加、Finding9 ASSERT 追加、BugG 説明修正、BugH 不一致解消、Finding10 注意事項追加 |

---

## 目次

1. [Bug A/B/D — NaN/Inf 伝播防止（P0 採用）](#bug-abd--naninf-伝播防止)
2. [Bug C — pushBlock 境界チェック（P0 採用）](#bug-c--pushblock-境界チェック)
3. [Bug 2 — bypass delayBuffer null 時未クリア（P0 採用）](#bug-2--bypass-delaybuffer-null-時未クリア)
4. [Bug 3/E — copyLatest TOCTOU（保留 — 将来対応）](#bug-3e--copylatest-toctou)
5. [Finding #9 — emitRetireIntentRT 命名（P1 採用）](#finding-9--emitretireintentrt-命名)
6. [Finding #10 — MKL バッファ std::vector（P2 改善項目）](#finding-10--mkl-バッファ-stdvector)
7. [Bug F — StereoConvolver::init 空ブロック（P3 保留）](#bug-f--stereoconvolverinit-空ブロック)
8. [Bug G — 冗長な負値チェック（P2 整理程度）](#bug-g--冗長な負値チェック)
9. [Bug H — StereoConvolver::init 例外安全性（P3 RAII 化優先）](#bug-h--stereoconvolverinit-例外安全性)

---

## 評価サマリ（v3）

| 項目 | 判定 | 優先度 | 採用/不採用 |
|------|------|--------|------------|
| Bug A/B/D | ★★★★★ | P0 | 採用 |
| Bug C | ★★★★★ | P0 | 採用 |
| Bug 2 | ★★★★★ | P0 | 採用 |
| Bug 3/E | ★☆☆☆☆ | 保留 | **不採用**（将来 64bit atomic または seqlock） |
| Finding 9 | ★★★★★ | P1 | 採用 |
| Finding 10 | ★★★☆☆ | P2 | 改善項目 |
| Bug F | ★★☆☆☆ | P3 | 保留 |
| Bug G | ★★★☆☆ | P2 | 整理程度 |
| Bug H | ★★☆☆☆ | P3 | RAII 化優先 |

---

## NaN/Inf 対策の DSP 全体方針

### 方針

NaN/Inf の伝播を防ぐため、以下の層で防御を行う：

1. **入口**: `sanitizeFinite()` をノイズシェイパーの `processSample()` 入口に配置
2. **中間**: `quantize()` の最前部に `sanitizeFinite()` を配置
3. **出口**: 出力バッファへの書き込み前に最終チェック（既存の `saturateAVX2` で一部保護）

### 適用範囲

| 関数 | 適用理由 |
|------|----------|
| `FixedNoiseShaper::processSample()` | fb（フィードバック値）が NaN の場合、`y = x - fb` で NaN が伝播 |
| `Fixed15TapNoiseShaper::processSample()` | 同上（15 次フィルタは発散リスクが高い） |
| `LatticeNoiseShaper::processSample()` | `computeFeedback()` の返り値が NaN の場合、伝播 |
| `FixedNoiseShaper::quantize()` | NaN は比較演算を通過するため、最前面で除去 |

### 除外範囲

| 関数 | 理由 |
|------|------|
| `killDenormal()` | デノーマル除去専用。NaN/Inf 除去は責務外 |
| `saturateAVX2()` | clamp 演算。NaN は通過するが、上流で除去済みが前提 |
| `UltraHighRateDCBlocker` | 既に `isFiniteAndBelowThresholdMask()` で保護済み |

---

## Bug A/B/D — NaN/Inf 伝播防止

### 概要

`killDenormal()` は Release ビルドで no-op であり、NaN/Inf を通過させる。`quantize()` は NaN が比較演算を通過するため、フィルタ発散時に音声出力が完全に破綻する。

### 修正方針

`killDenormal()` の責務は「デノーマル除去」のみ。NaN/Inf 除去は別関数 `sanitizeFinite()` として分離する。

### 修正対象

| ファイル | 関数 | 変更内容 |
|---------|------|----------|
| `src/DspNumericPolicy.h` | 新規 `sanitizeFinite(double)` | NaN/Inf → 0.0 変換 |
| `src/DspNumericPolicy.h` | 新規 `sanitizeFinite(float)` | NaN/Inf → 0.0f 変換 |
| `src/FixedNoiseShaper.h` | `quantize()` | `sanitizeFinite()` 呼び出し追加 |
| `src/FixedNoiseShaper.h` | `processSample()` | `sanitizeFinite()` 呼び出し追加 |
| `src/Fixed15TapNoiseShaper.h` | `processSample()` | `sanitizeFinite()` 呼び出し追加 |
| `src/LatticeNoiseShaper.h` | `processSample()` | `sanitizeFinite()` 呼び出し追加 |

### 修正案 1: `sanitizeFinite()` の追加

**ファイル**: `src/DspNumericPolicy.h`

```cpp
// ─────────────────────────────────────────────────────────────────
// NaN/Inf 除去ヘルパー関数（libm 非依存、ビット判定）
// ─────────────────────────────────────────────────────────────────

inline double sanitizeFinite(double x) noexcept
{
    constexpr uint64_t kExpMask = 0x7FF0000000000000ULL;
    const uint64_t bits = std::bit_cast<uint64_t>(x);
    const bool isNanOrInf = ((bits & kExpMask) == 0x7FF0000000000000ULL);
    return isNanOrInf ? 0.0 : x;
}

inline float sanitizeFinite(float x) noexcept
{
    constexpr uint32_t kExpMask = 0x7F800000U;
    const uint32_t bits = std::bit_cast<uint32_t>(x);
    const bool isNanOrInf = ((bits & kExpMask) == 0x7F800000U);
    return isNanOrInf ? 0.0f : x;
}
```

### 修正案 2: `quantize()` の NaN ガード

**ファイル**: `src/FixedNoiseShaper.h`

```cpp
inline double quantize(double v, Xoshiro256State& rng) const noexcept
{
    // NaN/Inf ガード: NaN は比較演算を通過するため、最前面で除去
    v = sanitizeFinite(v);

    const double minV = -1.0;
    const double maxV = 1.0 - (1.0 / invScale);

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

### 修正案 3: `processSample()` の fb NaN チェック

**ファイル**: `src/Fixed15TapNoiseShaper.h`

```cpp
inline double processSample(double x, int channel, double& outError) noexcept
{
    auto& channelErrors = errors[static_cast<size_t>(channel)];
    int& idx = writePos[static_cast<size_t>(channel)];

    double fb = 0.0;
    for (int i = 0; i < ORDER; ++i)
    {
        fb += coeffs[i] * get(channelErrors, idx, i);
    }
    fb = killDenormal(fb);
    fb = sanitizeFinite(fb);  // ← NaN/Inf 除去を追加

    const double y = x - fb;
    const double yq = quantize(y, rngState[static_cast<size_t>(channel)]);
    const double error = yq - y;
    outError = error;

    const double clampedError = saturateAVX2(error, -2.0 * scale, 2.0 * scale);
    const double denormalFreeError = killDenormal(clampedError);
    idx = (idx - 1 + ORDER) % ORDER;
    channelErrors[static_cast<size_t>(idx)] = denormalFreeError;

    return yq;
}
```

### 影響範囲

- `killDenormal()` はそのまま（責務不变）
- `sanitizeFinite()` は新規追加（NaN/Inf 専用）
- パフォーマンス影響: `sanitizeFinite()` はビットマスク比較 + 条件分岐（1-2 命令）

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
#if JUCE_DEBUG || defined(_DEBUG)
    jassert(numSamples <= kCapacity);  // Debug 時にアサーション
#endif
    numSamples = std::min(numSamples, kCapacity);  // Release では安全にクランプ

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
        convo::publishAtomic(writePosition, second, std::memory_order_release);
    }
    else
    {
        int nextPos = currentWritePos + numSamples;
        if (nextPos >= kCapacity)
            nextPos = 0;
        convo::publishAtomic(writePosition, nextPos, std::memory_order_release);
    }

    const int currentTotal = convo::consumeAtomic(totalSamples, std::memory_order_acquire);
    convo::publishAtomic(totalSamples, std::min(kCapacity, currentTotal + numSamples), std::memory_order_release);
}
```

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

### テスト方法

1. `releaseResources()` 後に `process()` を呼び出し、クラッシュしないことを確認
2. バイパス遷移中に無音が正しく出力されることを確認
3. 通常のバイパス動作で音質劣化がないことを確認

---

## Bug 3/E — copyLatest TOCTOU

### 概要

`AudioSegmentBuffer::copyLatest` で `totalSamples` と `writePosition` を別々に読み取るため、読み取り間に書き込みが発生すると不整合なデータを読み取る可能性がある。

### レビュー判定: 保留

**理由**: C++ の acquire/release は別 atomic には伝播しない。snapshot は保証されない。

### 現状の評価

- SPSC（Single Producer Single Consumer）条件下では、現状でも実用上は安全
- 将来マルチプロデューサに拡張した場合に問題になる可能性

### 将来対応案

1. **方案 A**: `writePosition` と `totalSamples` を 1 つの 64bit atomic にまとめる
2. **方案 B**: Sequence lock の導入
3. **方案 C**: 現状維持（SPSC 限定、ドキュメント化）

---

## Finding #9 — emitRetireIntentRT 命名

### 概要

`emitRetireIntentRT()` は関数名から「RT スレッドから安全に呼べる版」を示唆するが、実装は `emitRetireIntent()` を素通しで、mutex ロック経路を含む。

### 修正対象

| ファイル | 関数 | 変更内容 |
|---------|------|----------|
| `src/audioengine/ISRRetire.h` | `emitRetireIntentRT()` | コメント追加 + `ASSERT_NON_RT_THREAD()` 追加 |
| `src/audioengine/ISRRetire.cpp` | `emitRetireIntentRT()` | `ASSERT_NON_RT_THREAD()` 追加 |

### 修正案

**ファイル**: `src/audioengine/ISRRetire.cpp`

```cpp
void RetireRuntime::emitRetireIntentRT(const RetireIntent& intent) noexcept
{
    // ★ R-9: 「RT」は RealTime thread safety を意味しない。
    //   実装は emitRetireIntent() を素通しし、輻輳時に std::mutex をロックする。
    //   現時点では呼び出し元は全て非 RT スレッドであることを確認済み。
    //   将来 Audio Thread から呼び出す場合は、mutex を使わない別実装を用意すること。
    ASSERT_NON_RT_THREAD();  // ← Debug ビルドで Audio Thread からの誤呼び出しを検知
    emitRetireIntent(intent);
}
```

### 将来対応

名前変更（ABI 破壊変更のため、バージョンアップ時に実施）:

```cpp
// 変更前
void emitRetireIntentRT(const RetireIntent& intent) noexcept;

// 変更後
void emitRetireIntentFromNonRT(const RetireIntent& intent) noexcept;
```

### テスト方法

1. Debug ビルドで `ASSERT_NON_RT_THREAD()` が正しく動作することを確認
2. Audio Thread から呼び出した場合にアサーションが発火することを確認

---

## Finding #10 — MKL バッファ std::vector

### 概要

MKL DFTI API を直接呼ぶ関数内で `std::vector<double>` / `std::make_unique` が使用されており、 Finding #2（`IRAnalyzer.cpp`）と同型の規約違反。

### レビュー判定: △ 改善項目

**理由**:

- `std::vector` だから悪いという訳ではない
- 問題は MKL へ渡す配列の **Alignment**
- IR ロード専用なら性能差はほぼ誤差
- 設計ポリシーの話であり、Bug ではなく Medium 改善程度

### 注意事項

`makeAlignedArray` へ置き換える場合、`std::vector` が持つ以下の機能が失われる：

| 機能 | 影響 | 対策 |
|------|------|------|
| `size()` | 要素数の取得が必要な場合 | `irDataLength` 等の既存変数で代替 |
| `begin()` / `end()` | レンジベース for | インデックスループに変換 |
| `resize()` | 動的リサイズ | 事前にサイズ確定（IR ロード時は固定） |

**各利用箇所で `std::vector` のコンテナ機能を利用していないことを確認してから置換する。**

### 修正対象

| ファイル | 変更内容 |
|---------|----------|
| `src/convolver/ConvolverProcessor.MixedPhase.cpp` | `std::vector<double>` → `convo::makeAlignedArray<double>` |
| `src/convolver/ConvolverProcessor.ResampleAndFallback.cpp` | `std::vector<int>` → 固定サイズ配列 or アライン確保 |

### 修正案: 機械的監査と一括置換

#### Step 1: MKL ファイルの列挙

```bash
grep -rn "#include <mkl.h>\|#include <mkl_dfti.h>" src/
```

#### Step 2: 各ファイル内の std::vector/std::make_unique の列挙

```bash
grep -n "std::vector\|std::make_unique" src/convolver/ConvolverProcessor.MixedPhase.cpp
```

#### Step 3: 各利用箇所のコンテナ機能使用確認

```bash
# size() / begin() / end() / resize() の使用を確認
grep -n "\.size()\|\.begin()\|\.end()\|\.resize()" src/convolver/ConvolverProcessor.MixedPhase.cpp
```

#### Step 4: 置換（例: `phiMinUnwrapped`）

```cpp
// 変更前
std::vector<double> phiMinUnwrapped(static_cast<size_t>(complexSize));

// 変更後（コンテナ機能未使用を確認済み）
auto phiMinUnwrapped = convo::makeAlignedArray<double>(static_cast<size_t>(complexSize));
```

### テスト方法

1. MKL 関数が正しく動作することを確認（FFT 結果の整合性）
2. メモリリークがないことを確認（Valgrind / AddressSanitizer）
3. 通常動作でのパフォーマンス劣化がないことを確認

---

## Bug F — StereoConvolver::init 空ブロック

### 概要

`StereoConvolver::init` で `ownerProcessor != nullptr` 時の処理ブロックが空。意図された処理（例: レイテンシ通知）が欠落している可能性。

### レビュー判定: △ 保留

**理由**: Bug とは断定できない。将来拡張なのか実装漏れなのか、コードだけでは断定不可。

### 修正案

**ファイル**: `src/ConvolverProcessor.h`

```cpp
if (ownerProcessor != nullptr)
{
    // 現時点では特別な処理不要（レイテンシ通知は process() 側で処理）
    // 将来 ownerProcessor へのコールバックが必要になった場合はここに追加
}
```

---

## Bug G — 冗長な負値チェック

### 概要

`processBypassWithLatencyCompensation` で `if (readPos < 0)` は冗長。

### レビュー判定: ○ 軽微

### 説明の修正

**v2 の誤り**: 「C++20 では二の補数が義務付けられているため」

**正しい説明**: `DELAY_BUFFER_MASK` が `2^n - 1`（例: `0xFFF`）であることから、`& DELAY_BUFFER_MASK` の結果は常に 0 以上。二の補数の保証ではなく、**マスク値の性質**による。

### 修正案

**ファイル**: `src/convolver/ConvolverProcessor.Runtime.cpp`

```cpp
// 2) 遅延した信号を出力へ戻す
// DELAY_BUFFER_MASK = 2^n - 1 であるため、`& DELAY_BUFFER_MASK` の結果は常に非負。
// したがって `if (readPos < 0)` は死コード。
int readPos = (writePos - delaySamples) & DELAY_BUFFER_MASK;
// if (readPos < 0)
//     readPos += DELAY_BUFFER_SIZE;
```

---

## Bug H — StereoConvolver::init 例外安全性

### 概要

`StereoConvolver::init` で `std::bad_alloc` のみキャッチ。他の例外時は `irData` がリークする可能性。

### レビュー判定: △ RAII 化優先

**理由**: 例外安全性は `catch 増加` より `所有権設計` で解決する方が望ましい。

### 修正案（RAII 確認）

**ファイル**: `src/ConvolverProcessor.h`

現在のコードは既に `convo::aligned_make_unique` を使用しており、`std::unique_ptr` で所有権管理されている。`nuc0`, `nuc1` は例外発生時でもスコープ離脱時に自動破棄される。

```cpp
try
{
    auto nuc0 = convo::aligned_make_unique<convo::MKLNonUniformConvolver>();
    auto nuc1 = convo::aligned_make_unique<convo::MKLNonUniformConvolver>();

    if (nuc0->SetImpulse(irData[0], irDataLength, knownBlockSize, scale, enableDirectHead, filterSpec) &&
        nuc1->SetImpulse(irData[1], irDataLength, knownBlockSize, scale, enableDirectHead, filterSpec))
    {
        // 所有権を移動（release() で生ポインタを取得）
        destroyNUCConvolver(nucConvolvers[0]);
        destroyNUCConvolver(nucConvolvers[1]);
        nucConvolvers[0] = nuc0.release();  // nuc0 は null になる
        nucConvolvers[1] = nuc1.release();  // nuc1 は null になる

        latency   = nucConvolvers[0]->getLatency();
        DBG("Convolver: NUC Engine Active. Latency: " << latency << " samples");
        return true;
    }
}
catch (const std::bad_alloc&)
{
    // メモリ確保失敗: nuc0, nuc1 はスコープ離脱時に自動破棄（RAII）
}

// NUC セットアップ失敗
destroyNUCConvolver(nucConvolvers[0]);
destroyNUCConvolver(nucConvolvers[1]);
if (irData[0]) { convo::aligned_free(irData[0]); irData[0] = nullptr; }
if (irData[1]) { convo::aligned_free(irData[1]); irData[1] = nullptr; }
irDataLength = 0;
latency = 0;
this->irLatency = 0;
return false;
```

**要点**: `nuc0`, `nuc1` は `std::unique_ptr` であるため、例外が発生してもスコープ離脱時に自動破棄される。`catch` で明示的にクリーンアップする必要はない。

---

## 改修優先度とスケジュール

### P0: 即時対応（2.5 日）

| Bug | 改修内容 | 工数 |
|-----|----------|------|
| Bug A/B/D | `sanitizeFinite()` 追加 + `quantize()`/`processSample()` 修正 | 1 日 |
| Bug C | `pushBlock` 境界チェック + Debug 用 `jassert` | 0.5 日 |
| Bug 2 | `processBypassWithLatencyCompensation` null クリア | 0.5 日 |
| Finding 9 | `emitRetireIntentRT` コメント + `ASSERT_NON_RT_THREAD()` | 0.5 日 |

### P1: 近日対応

| Bug | 改修内容 | 工数 |
|-----|----------|------|
| （なし） | | |

### P2: 改善項目（3 日）

| Bug | 改修内容 | 工数 |
|-----|----------|------|
| Finding 10 | MKL バッファ std::vector → makeAlignedArray | 3 日 |
| Bug G | 冗長チェックの整理 + 説明修正 | 5 分 |

### P3: 保留（将来対応）

| Bug | 改修内容 | 工数 |
|-----|----------|------|
| Bug 3/E | TOCTOU 修正（sequence lock or 64bit atomic 化） | 2 日 |
| Bug F | 空ブロックにコメント追加 | 5 分 |
| Bug H | RAII 設計の再検討 | 1 日 |

### 総工数推定

- P0: 2.5 日
- P2: 3 日
- P3: 3 日
- **合計: 8.5 日**

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
| 将来の誤呼び出し | 高 | 中 | Finding 9 の修正 |

---

## 承認

| 役割 | 名前 | 日付 | 署名 |
|------|------|------|------|
| 設計者 | | | |
| レビュアー | | | |
| 承認者 | | | |
