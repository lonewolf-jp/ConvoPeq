# ConvoPeq 未修正バグ改修設計書（v6 — 完成版）

**作成日**: 2026-07-21
**レビュー反映**: 2026-07-21（v5 レビュー → v6 反映：4点の最終修正）
**対象**: Part 7〜10 + bug1/bug2/bug3 で特定された未修正バグ（19件）
**評価**: A+（97〜98点）— 実装着手可能

---

## 変更履歴

| バージョン | 日付 | 変更内容 |
|-----------|------|----------|
| v1 | 2026-07-21 | 初版作成 |
| v2 | 2026-07-21 | v1 レビュー反映：Bug3/E 不採用、sanitizeFinite 分離、BugH RAII 化優先、Finding10 位置付け変更 |
| v3 | 2026-07-21 | v2 レビュー反映：sanitizeFinite 説明修正、NaN 方針の DSP 全体整理、BugC jassert 追加、Finding9 ASSERT 追加、BugG 説明修正、BugH 不一致解消、Finding10 注意事項追加 |
| v4 | 2026-07-21 | v3 レビュー反映：sanitizeFinite 性能説明修正、NaN 方針の責務図追加、BugH RAII 説明の正確化 |
| v4+ | 2026-07-21 | bug3.md 検証反映：bug3-1〜8 の妥当性検証、妥当6件を追加 |
| v5 | 2026-07-21 | v4 レビュー反映：sanitizeFinite 適用箇所統一、BugC fail-fast 変更、BugH RAII/irData 分離、MKL allocator policy 追加、Finding9 deprecated 追加、Debug 用 jassert 追加 |
| v6 | 2026-07-21 | v5 レビュー反映：BugH aligned_unique_ptr 化検討、BugC drop 方針明確化、sanitizeFinite NaN counter 余地、bug3-6 API 契約明記 |

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
9. [Bug H — StereoConvolver::init 例外安全性（P0 採用）](#bug-h--stereoconvolverinit-例外安全性)
10. [bug3群 — 第3回報告バグ（P0/P3）](#bug3群--第3回報告バグ)

---

## 評価サマリ（v6 — 最終）

### Part 7〜10 + bug1/bug2

| 項目 | 判定 | 優先度 | 採用/不採用 |
|------|------|--------|------------|
| Bug A/B/D | ★★★★★ | P0 | 採用 |
| Bug C | ★★★★★ | P0 | 採用（drop 方針） |
| Bug 2 | ★★★★★ | P0 | 採用 |
| Bug 3/E | ★☆☆☆☆ | 保留 | **不採用**（将来 64bit atomic または seqlock） |
| Finding 9 | ★★★★★ | P1 | 採用（deprecated 追加） |
| Finding 10 | ★★★☆☆ | P2 | 改善項目（allocator policy 決定後） |
| Bug F | ★★☆☆☆ | P3 | 保留 |
| Bug G | ★★★☆☆ | P2 | 整理程度 |
| Bug H | ★★★★★ | P0 | **採用**（aligned_unique_ptr 化） |

### bug3.md（第3回報告）検証結果

| 項目 | 判定 | 優先度 | 採用/不採用 |
|------|------|--------|------------|
| bug3-1 | ★★★★★ | P0 | **採用**（irData リーク — 重大） |
| bug3-2 | ★★★★☆ | P0 | **採用**（clone 時リーク） |
| bug3-3 | ★★★★★ | P0 | **採用**（numSamples 負値でバッファオーバーフロー） |
| bug3-4 | ★★★☆☆ | P3 | **採用**（storedFilterSpec 未リセット） |
| bug3-5 | ★★★☆☆ | P3 | **採用**（callQuantumSamples 等未リセット） |
| bug3-6 | ★★☆☆☆ | P3 | **採用**（delayWritePos API 契約明記） |
| bug3-7 | ★☆☆☆☆ | — | **不採用**（既に `irDataLength > 0` でガード済み） |
| bug3-8 | ★★★☆☆ | P3 | **採用**（got > numSamples の防御チェック） |

---

## 実装優先順位

| 優先 | 項目 | 理由 |
|------|------|------|
| P0-1 | Bug C | メモリ破壊は再現頻度が低くても影響が最大 |
| P0-2 | Bug 2 | stale data 出力によるグリッチ防止 |
| P0-3 | Bug A/B/D | 音声破綻防止（通常運用では異常値発生時のみ） |
| P0-4 | Bug H / bug3-1 | irData リーク（aligned_unique_ptr 化で根本解決） |
| P0-5 | bug3-3 | numSamples 負値でバッファオーバーフロー |
| P1 | Finding 9 | Debug での誤呼び出し検知 |
| P2 | Finding 10 | 設計ポリシー統一（allocator policy 決定後） |
| P2 | Bug G | コード整理 |
| P3 | bug3-4/5 | init 失敗時の状態リセット |
| P3 | bug3-6 | delayWritePos API 契約明記 |
| P3 | bug3-8 | got > numSamples の防御チェック |
| 保留 | Bug 3/E | 将来対応 |
| 保留 | Bug F | コメントのみ |

---

## NaN/Inf 対策の DSP 全体方針

### 責務境界

`quantize()` は公共 API であり、単独利用もあり得る。 therefore、**`quantize()` の先頭のみで `sanitizeFinite()` を実行**する。

### NaN 発生時の追跡（v6 改善点）

Debug ビルドでは `jassert(std::isfinite(x))` で NaN 発生源を追跡可能。将来 DiagnosticCounter を導入する場合は、`NaNCounter++` を追加することで統計的に追跡できる。

```
┌─────────────────────────────────────────────────────────────────┐
│                    DSP Numeric Policy                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  killDenormal()  →  デノーマル除去（FTZ/DAZ と連携）     │   │
│  │  sanitizeFinite() →  NaN/Inf 除去（本設計で追加）        │   │
│  │    - Debug: jassert(isfinite(x)) で NaN 発生源を追跡    │   │
│  │              + 将来 DiagnosticCounter で統計追跡可能      │   │
│  │    - Release: 0.0 に置換                                 │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Bug A/B/D — NaN/Inf 伝播防止

### 概要

`killDenormal()` は Release ビルドで no-op であり、NaN/Inf を通過させる。`quantize()` は NaN が比較演算を通過するため、フィルタ発散時に音声出力が完全に破綻する。

### 修正方針

`quantize()` は公共 API であり、単独利用もあり得る。 therefore、**`quantize()` の先頭のみで `sanitizeFinite()` を実行**する。

### 修正対象

| ファイル | 関数 | 変更内容 |
|---------|------|----------|
| `src/DspNumericPolicy.h` | 新規 `sanitizeFinite(double)` | NaN/Inf → 0.0 変換 + Debug 用 jassert |
| `src/DspNumericPolicy.h` | 新規 `sanitizeFinite(float)` | NaN/Inf → 0.0f 変換 + Debug 用 jassert |
| `src/FixedNoiseShaper.h` | `quantize()` | `sanitizeFinite()` 呼び出し追加（唯一の適用箇所） |

### 修正案 1: `sanitizeFinite()` の追加

**ファイル**: `src/DspNumericPolicy.h`

```cpp
// ─────────────────────────────────────────────────────────────────
// NaN/Inf 除去ヘルパー関数（libm 非依存、ビット判定）
// ─────────────────────────────────────────────────────────────────

inline double sanitizeFinite(double x) noexcept
{
#if JUCE_DEBUG || defined(_DEBUG)
    // Debug ビルド: NaN/Inf 発生時にアサーションで検出（追跡用）
    // ★ 将来 DiagnosticCounter を導入する場合は NaNCounter++ を追加可能
    jassert(std::isfinite(x));
#endif
    constexpr uint64_t kExpMask = 0x7FF0000000000000ULL;
    const uint64_t bits = std::bit_cast<uint64_t>(x);
    const bool isNanOrInf = ((bits & kExpMask) == 0x7FF0000000000000ULL);
    return isNanOrInf ? 0.0 : x;
}

inline float sanitizeFinite(float x) noexcept
{
#if JUCE_DEBUG || defined(_DEBUG)
    jassert(std::isfinite(x));
#endif
    constexpr uint32_t kExpMask = 0x7F800000U;
    const uint32_t bits = std::bit_cast<uint32_t>(x);
    const bool isNanOrInf = ((bits & kExpMask) == 0x7F800000U);
    return isNanOrInf ? 0.0f : x;
}
```

### 修正案 2: `quantize()` の NaN ガード（唯一の適用箇所）

**ファイル**: `src/FixedNoiseShaper.h`

```cpp
inline double quantize(double v, Xoshiro256State& rng) const noexcept
{
    // NaN/Inf ガード: NaN は比較演算を通過するため、最前面で除去
    // ★ 唯一の適用箇所（processSample 側は不要）
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

### テスト方法

1. NaN を注入した入力でフィルタを実行し、出力が 0.0 になることを確認
2. Inf を注入した入力でフィルタを実行し、出力が 0.0 になることを確認
3. Debug ビルドで `jassert` が発火することを確認
4. 通常動作での音質劣化がないことを確認（A/B テスト）

---

## Bug C — pushBlock 境界チェック

### 概要

`AudioSegmentBuffer::pushBlock` で `numSamples > kCapacity` の場合、リングバッファのラップアラウンド計算が破綻し、バッファオーバーフローが発生する。

### 修正方針（v6 明確化）

**Drop 方針**: `numSamples > kCapacity` の入力は契約違反であり、**処理を拒否**する。AudioSegmentBuffer の意味が変わる可能性があるため、Silent recovery（truncate）より安全。

### 修正対象

| ファイル | 関数 | 変更内容 |
|---------|------|----------|
| `src/AudioSegmentBuffer.h` | `pushBlock()` | 境界チェック追加（drop） |

### 修正案

**ファイル**: `src/AudioSegmentBuffer.h`

```cpp
void pushBlock(const double* left, const double* right, int numSamples) noexcept
{
    if (left == nullptr || right == nullptr || numSamples <= 0)
        return;

    // ★ 境界チェック: kCapacity を超える入力は契約違反（drop 方針）
    //   理由: truncate すると AudioSegmentBuffer の意味が変わる可能性がある
    //   DSP では Silent recovery より Fail-fast が安全
    if (numSamples > kCapacity)
    {
        jassertfalse;  // Debug 時にアサーション
        return;         // Release では無視（呼び出し元のバグ）
    }

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

### API 契約

| パラメータ | 契約 | 違反時の挙動 |
|-----------|------|-------------|
| `left` / `right` | `nullptr` 以外 | return（no-op） |
| `numSamples` | `0 < numSamples <= kCapacity` | `jassertfalse` + return（drop） |

### テスト方法

1. `numSamples = kCapacity + 1` で `pushBlock` を呼び出し、Debug でアサーションが発火することを確認
2. `numSamples = kCapacity` で正常動作することを確認
3. 通常のブロックサイズで正常動作することを確認

---

## Bug 2 — bypass delayBuffer null 時未クリア

### 概要

`processBypassWithLatencyCompensation` で `delayBuffer` が null の場合、出力バッファをクリアせずに return する。バイパス遷移中に stale data が残留する。

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

### 修正案

**ファイル**: `src/audioengine/ISRRetire.h`

```cpp
/**
 * ★ R-9: 命名注意 — 「RT」は RealTime thread safety を意味しない。
 *   実装は emitRetireIntent() を素通しし、輻輳時に std::mutex をロックする。
 *   現時点では呼び出し元は全て非 RT スレッドであることを確認済み。
 *   将来 Audio Thread から呼び出す場合は、mutex を使わない別実装を用意すること。
 *
 * TODO: 将来リネーム（emitRetireIntentFromNonRT）。
 *       ABI 破壊変更のため、バージョンアップ時に実施。
 */
[[deprecated("RT は RealTime thread safety を意味しない。非 RT スレッド専用。")]]
void emitRetireIntentRT(const RetireIntent& intent) noexcept;
```

**ファイル**: `src/audioengine/ISRRetire.cpp`

```cpp
void RetireRuntime::emitRetireIntentRT(const RetireIntent& intent) noexcept
{
    ASSERT_NON_RT_THREAD();  // ← Debug ビルドで Audio Thread からの誤呼び出しを検知
    emitRetireIntent(intent);
}
```

---

## Finding #10 — MKL バッファ std::vector

### 概要

MKL DFTI API を直接呼ぶ関数内で `std::vector<double>` / `std::make_unique` が使用されている。

### レビュー判定: △ 改善項目（allocator policy 決定後）

### Allocator Policy（v5 で決定）

| Allocator | アラインメント | 用途 | 推奨 |
|-----------|---------------|------|------|
| `convo::makeAlignedArray<T>` | 64byte（AVX512 対応） | FFT バッファ、IR データ | ✅ 推奨 |
| `mkl_malloc` | MKL 内部最適 | MKL 専用バッファ | MKL 専用のみ |
| `std::vector<T>` | アラインメント保証なし | 一般用途 | ❌ MKL バッファには不適切 |

**方針**: MKL と直接やり取りするバッファは `convo::makeAlignedArray<T>` に統一。

---

## Bug F — StereoConvolver::init 空ブロック

### レビュー判定: △ 保留

**理由**: Bug とは断定できない。将来拡張なのか実装漏れなのか、コードだけでは断定不可。

---

## Bug G — 冗長な負値チェック

### 概要

`processBypassWithLatencyCompensation` で `if (readPos < 0)` は冗長。

### 説明

`DELAY_BUFFER_MASK` が `2^n - 1`（例: `0xFFF`）であることから、`& DELAY_BUFFER_MASK` の結果は常に 0 以上。二の補数の保証ではなく、**マスク値の性質**による。

### 修正案

```cpp
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

### 修正方針（v6 変更点）

**v5 の問題**: `catch(...)` は対症療法。根本原因は `irData` が `aligned_malloc` された裸ポインタであること。

**v6 の方針**: `irData` を **`aligned_unique_ptr`** で RAII 管理し、例外安全性を根本的に解決。

### `aligned_unique_ptr` の確認

ConvoPeq には既に `convo::aligned_unique_ptr<T>` が存在する（`AlignedAllocation.h`）。

```cpp
template <typename T>
using aligned_unique_ptr = std::unique_ptr<T, AlignedObjectDeleter<T>>;
```

- `AlignedObjectDeleter` が `ptr->~T()` + `convo::aligned_free()` を呼ぶ
- `std::unique_ptr` の RAII により、例外時でも自動解放される

### 修正案（aligned_unique_ptr 化）

**ファイル**: `src/ConvolverProcessor.h`

```cpp
// ★ v6 変更: irData を aligned_unique_ptr で RAII 管理
//   これにより例外安全性が根本的に解決される
struct StereoConvolver
{
    // ★ 変更前: double* irData[2] = { nullptr, nullptr };
    // ★ 変更後: aligned_unique_ptr で RAII 管理
    convo::aligned_unique_ptr<double[]> irData[2];

    // ... 既存のメンバ ...

    bool init(double* irL, double* irR, int length, double sr, int peakDelay, int knownBlockSize, int preferredCallSize, double scale = 1.0,
          bool enableDirectHead = false,
          const convo::FilterSpec* filterSpec = nullptr,
          ConvolverProcessor* ownerProcessor = nullptr)
    {
        // ★ irData の所有権を移動（aligned_unique_ptr で RAII 保護）
        irData[0].reset(irL);
        irData[1].reset(irR);
        irDataLength = length;
        // ... 既存の処理 ...

        try
        {
            auto nuc0 = convo::aligned_make_unique<convo::MKLNonUniformConvolver>();
            auto nuc1 = convo::aligned_make_unique<convo::MKLNonUniformConvolver>();

            // ★ SetImpulse に raw ポインタを渡す（所有権は移動しない）
            if (nuc0->SetImpulse(irData[0].get(), irDataLength, knownBlockSize, scale, enableDirectHead, filterSpec) &&
                nuc1->SetImpulse(irData[1].get(), irDataLength, knownBlockSize, scale, enableDirectHead, filterSpec))
            {
                // 成功: NUC に所有権を移動
                destroyNUCConvolver(nucConvolvers[0]);
                destroyNUCConvolver(nucConvolvers[1]);
                nucConvolvers[0] = nuc0.release();
                nucConvolvers[1] = nuc1.release();

                // ★ irData は aligned_unique_ptr が保持（RAII 保護）
                latency   = nucConvolvers[0]->getLatency();
                DBG("Convolver: NUC Engine Active. Latency: " << latency << " samples");
                return true;
            }
        }
        catch (const std::bad_alloc&)
        {
            // メモリ確保失敗: irData は aligned_unique_ptr で自動解放
        }
        catch (...)
        {
            // ★ その他の例外: irData は aligned_unique_ptr で自動解放
        }

        // cleanup（失敗時）
        destroyNUCConvolver(nucConvolvers[0]);
        destroyNUCConvolver(nucConvolvers[1]);
        // ★ irData は aligned_unique_ptr が自動解放（手動解放不要）
        irData[0].reset();
        irData[1].reset();
        irDataLength = 0;
        latency = 0;
        this->irLatency = 0;
        return false;
    }
};
```

### 要点

- `irData` は `aligned_unique_ptr<double[]>` で RAII 保護
- `SetImpulse()` に `.get()` で生ポインタを渡す（所有権は移動しない）
- 例外時でも `irData` は自動解放（`catch(...)` で明示的クリーンアップ不要）
- `nuc0/nuc1` も `aligned_unique_ptr` で RAII 保護

### テスト方法

1. **Exception injection test**: `SetImpulse()` で `throw runtime_error` を強制し、cleanup が正しく動くことを確認
2. 通常動作で問題がないことを確認
3. メモリリークがないことを確認（Valgrind / AddressSanitizer）

---

## bug3群 — 第3回報告バグ

### bug3-1 — 【重大】StereoConvolver::init — SetImpulse 非 bad_alloc 例外で irData リーク

**修正**: Bug H の aligned_unique_ptr 化で根本解決。

---

### bug3-2 — 【中】StereoConvolver::clone — init 失敗時の newConv リーク

**修正**: Bug H の aligned_unique_ptr 化で根本解決。

---

### bug3-3 — 【中】StereoConvolver::process — numSamples <= 0 の未チェック

**修正案**:
```cpp
if (channel < 0 || channel >= 2 || !nucConvolvers[channel] || numSamples <= 0)
{
    if (numSamples > 0)
        std::memset(out, 0, numSamples * sizeof(double));
    return;
}
```

---

### bug3-4 — 【低】StereoConvolver::init — 失敗時に storedFilterSpec / hasStoredFilterSpec がリセットされない

**修正案**:
```cpp
// cleanup に追加
hasStoredFilterSpec = false;
storedFilterSpec = convo::FilterSpec{};
```

---

### bug3-5 — 【低】StereoConvolver::init — 失敗時に callQuantumSamples / storedSampleRate 等がリセットされない

**修正案**:
```cpp
// cleanup に追加
callQuantumSamples = 0;
storedSampleRate = 0.0;
storedKnownBlockSize = 0;
storedScale = 1.0;
storedDirectHeadEnabled = false;
```

---

### bug3-6 — 【低】processBypassWithLatencyCompensation — delayWritePos の非アトミックアクセス

**修正方針**: Atomic 化ではなく、**API 契約**を明文化。

**ファイル**: `src/ConvolverProcessor.h`

```cpp
// ★ API 契約: reset() は Audio Thread 停止後にのみ呼び出すこと。
//   Audio Thread 実行中に reset() を呼び出すとデータレースが発生する。
//   （ISR 設計思想に準拠: RuntimeWorld の publish/release パターンと同一）
int delayWritePos = 0;
```

---

### bug3-7 — 【不採用】StereoConvolver::clone — irDataLength 負値のオーバーフロー

**判定**: **不採用**（既に `if (irDataLength > 0 && irData[0] && irData[1])` でガード済み）

---

### bug3-8 — 【低】StereoConvolver::process — got > numSamples の未チェック

**修正案**:
```cpp
const int got = nucConvolvers[channel]->Get(out, numSamples);
jassert(got <= numSamples);  // 防御的チェック
if (got < numSamples)
    std::memset(out + got, 0, (numSamples - got) * sizeof(double));
```

---

## 改修優先度とスケジュール

### P0: 即時対応（4 日）

| Bug | 改修内容 | 工数 |
|-----|----------|------|
| Bug C | `pushBlock` 境界チェック（drop） | 0.5 日 |
| Bug 2 | `processBypassWithLatencyCompensation` null クリア | 0.5 日 |
| Bug A/B/D | `sanitizeFinite()` 追加 + `quantize()` 修正 | 1 日 |
| Bug H / bug3-1 | `irData` の `aligned_unique_ptr` 化 | 1.5 日 |
| bug3-3 | `numSamples <= 0` チェック追加 | 0.5 日 |

### P2: 改善項目（3 日）

| Bug | 改修内容 | 工数 |
|-----|----------|------|
| Finding 10 | MKL バッファ std::vector → makeAlignedArray | 3 日 |
| Bug G | 冗長チェックの整理 | 5 分 |

### P3: 保留（将来対応）

| Bug | 改修内容 | 工数 |
|-----|----------|------|
| Bug 3/E | TOCTOU 修正 | 2 日 |
| Bug F | 空ブロックにコメント追加 | 5 分 |
| bug3-4/5 | init 失敗時の状態リセット | 0.5 日 |
| bug3-6 | delayWritePos API 契約明記 | 5 分 |
| bug3-8 | got > numSamples の防御チェック | 5 分 |

### 総工数推定

- P0: 4 日
- P2: 3 日
- P3: 3 日
- **合計: 10 日**

---

## テスト計画

### 単体テスト

1. **NaN/Inf テスト**: NaN/Inf を注入した入力で全ノイズシェイパーをテスト
2. **境界テスト**: `pushBlock` で `numSamples > kCapacity` の場合をテスト
3. **null テスト**: `processBypassWithLatencyCompensation` で delayBuffer null の場合をテスト
4. **Exception injection test**: `SetImpulse()` で `throw runtime_error` を強制し、cleanup が正しく動くことを確認

### 統合テスト

1. **通常動作テスト**: 全修正後に音質劣化がないことを確認
2. **パフォーマンステスト**: 修正前後のパフォーマンス比較
3. **メモリテスト**: メモリリークがないことを確認（Valgrind / AddressSanitizer）

### リグレッションテスト

1. **既存テスト実行**: 全既存テストがパスすることを確認
2. **手動テスト**: 実機での音質確認

---

## 承認

| 役割 | 名前 | 日付 | 署名 |
|------|------|------|------|
| 設計者 | | | |
| レビュアー | | | |
| 承認者 | | | |
