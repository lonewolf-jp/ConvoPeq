# ConvoPeq 未修正バグ改修設計書（v5 — 実装着手可能版）

**作成日**: 2026-07-21
**レビュー反映**: 2026-07-21（v4 レビュー → v5 反映：4点の再検討事項反映）
**対象**: Part 7〜10 + bug1/bug2/bug3 で特定された未修正バグ（19件）
**評価**: A+（95点前後）— 実装着手可能

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
10. [bug3群 — 第3回報告バグ（P0/P3）](#bug3群--第3回報告バグ)

---

## 評価サマリ（v5 — 最終）

### Part 7〜10 + bug1/bug2

| 項目 | 判定 | 優先度 | 採用/不採用 |
|------|------|--------|------------|
| Bug A/B/D | ★★★★★ | P0 | 採用 |
| Bug C | ★★★★★ | P0 | 採用（fail-fast 変更） |
| Bug 2 | ★★★★★ | P0 | 採用 |
| Bug 3/E | ★☆☆☆☆ | 保留 | **不採用**（将来 64bit atomic または seqlock） |
| Finding 9 | ★★★★★ | P1 | 採用（deprecated 追加） |
| Finding 10 | ★★★☆☆ | P2 | 改善項目（allocator policy 決定後） |
| Bug F | ★★☆☆☆ | P3 | 保留 |
| Bug G | ★★★☆☆ | P2 | 整理程度 |
| Bug H | ★★★★☆ | P3 | 採用（RAII/irData 分離） |

### bug3.md（第3回報告）検証結果

| 項目 | 判定 | 優先度 | 採用/不採用 |
|------|------|--------|------------|
| bug3-1 | ★★★★★ | P0 | **採用**（irData リーク — 重大） |
| bug3-2 | ★★★★☆ | P0 | **採用**（clone 時リーク） |
| bug3-3 | ★★★★★ | P0 | **採用**（numSamples 負値でバッファオーバーフロー） |
| bug3-4 | ★★★☆☆ | P3 | **採用**（storedFilterSpec 未リセット） |
| bug3-5 | ★★★☆☆ | P3 | **採用**（callQuantumSamples 等未リセット） |
| bug3-6 | ★★☆☆☆ | P3 | **採用**（delayWritePos データレース） |
| bug3-7 | ★☆☆☆☆ | — | **不採用**（既に `irDataLength > 0` でガード済み） |
| bug3-8 | ★★★☆☆ | P3 | **採用**（got > numSamples の防御チェック） |

---

## 実装優先順位

| 優先 | 項目 | 理由 |
|------|------|------|
| P0-1 | Bug C | メモリ破壊は再現頻度が低くても影響が最大 |
| P0-2 | Bug 2 | stale data 出力によるグリッチ防止 |
| P0-3 | Bug A/B/D | 音声破綻防止（通常運用では異常値発生時のみ） |
| P0-4 | bug3-1 | irData リーク（重大） |
| P0-5 | bug3-3 | numSamples 負値でバッファオーバーフロー |
| P1 | Finding 9 | Debug での誤呼び出し検知 |
| P2 | Finding 10 | 設計ポリシー統一（allocator policy 決定後） |
| P2 | Bug G | コード整理 |
| P3 | Bug H | RAII 設計の再検討 |
| P3 | bug3-4/5 | init 失敗時の状態リセット |
| P3 | bug3-6 | delayWritePos データレース |
| P3 | bug3-8 | got > numSamples の防御チェック |
| 保留 | Bug 3/E | 将来対応 |
| 保留 | Bug F | コメントのみ |

---

## NaN/Inf 対策の DSP 全体方針

### 責務境界（v5 変更点）

**v4 の問題**: `processSample()` と `quantize()` の両方で `sanitizeFinite()` を呼ぶと、2重チェックになる。

**v5 の方針**: `quantize()` は公共 API であり、単独利用もあり得る。 therefore、**`quantize()` の先頭のみで `sanitizeFinite()` を実行**する。`processSample()` 側は不要。

```
┌─────────────────────────────────────────────────────────────────┐
│                    DSP Numeric Policy                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  killDenormal()  →  デノーマル除去（FTZ/DAZ と連携）     │   │
│  │  sanitizeFinite() →  NaN/Inf 除去（本設計で追加）        │   │
│  │    - Debug: jassert(isfinite(x)) で NaN 発生源を追跡    │   │
│  │    - Release: 0.0 に置換                                 │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    NoiseShaper 入口                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  processSample()                                       │   │
│  │    fb = computeFeedback(...)                           │   │
│  │    fb = killDenormal(fb)    ← デノーマル除去            │   │
│  │    ★ sanitizeFinite(fb) は不要（quantize で保護済み）  │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    量子化（公共 API）                             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  quantize()                                            │   │
│  │    ★ sanitizeFinite(v)  ← ここで統一的に NaN/Inf 除去  │   │
│  │    if (v < minV) v = minV;                             │   │
│  │    else if (v > maxV) v = maxV;                        │   │
│  │    ... dither + rounding ...                           │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    出力                                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  出力バッファへの書き込み                               │   │
│  │  （saturateAVX2 で一部保護、上流で除去済みが前提）      │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 適用範囲（v5 変更後）

| 関数 | 適用 | 理由 |
|------|------|------|
| `FixedNoiseShaper::quantize()` | ✅ 適用 | **唯一の適用箇所**（公共 API） |
| `Fixed15TapNoiseShaper::processSample()` | ❌ 不要 | `quantize()` 内で保護済み |
| `LatticeNoiseShaper::processSample()` | ❌ 不要 | `quantize()` 内で保護済み |
| `FixedNoiseShaper::processSample()` | ❌ 不要 | `quantize()` 内で保護済み |

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

### 影響範囲

- `killDenormal()` はそのまま（責務不变）
- `sanitizeFinite()` は新規追加（NaN/Inf 専用）
- `processSample()` 側の `sanitizeFinite()` 呼び出しは**削除**（重複防止）
- Debug ビルドでは `jassert(std::isfinite(x))` で NaN 発生源を追跡可能

### テスト方法

1. NaN を注入した入力でフィルタを実行し、出力が 0.0 になることを確認
2. Inf を注入した入力でフィルタを実行し、出力が 0.0 になることを確認
3. Debug ビルドで `jassert` が発火することを確認
4. 通常動作での音質劣化がないことを確認（A/B テスト）

---

## Bug C — pushBlock 境界チェック

### 概要

`AudioSegmentBuffer::pushBlock` で `numSamples > kCapacity` の場合、リングバッファのラップアラウンド計算が破綻し、バッファオーバーフローが発生する。

### 修正方針（v5 変更点）

**v4 の方針**: `numSamples = std::min(numSamples, kCapacity)` でクランプ

**v5 の方針**: **Fail-fast** で拒否。AudioSegmentBuffer の意味が変わる可能性があるため、Silent recovery より安全。

### 修正対象

| ファイル | 関数 | 変更内容 |
|---------|------|----------|
| `src/AudioSegmentBuffer.h` | `pushBlock()` | 境界チェック追加（fail-fast） |

### 修正案

**ファイル**: `src/AudioSegmentBuffer.h`

```cpp
void pushBlock(const double* left, const double* right, int numSamples) noexcept
{
    if (left == nullptr || right == nullptr || numSamples <= 0)
        return;

    // 境界チェック: kCapacity を超える入力は fail-fast で拒否
    // ★ 理由: クランプすると AudioSegmentBuffer の意味が変わる可能性がある
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

### テスト方法

1. `numSamples = kCapacity + 1` で `pushBlock` を呼び出し、Debug でアサーションが発火することを確認
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
| `src/audioengine/ISRRetire.h` | `emitRetireIntentRT()` | コメント追加 + `ASSERT_NON_RT_THREAD()` 追加 + `[[deprecated]]` |
| `src/audioengine/ISRRetire.cpp` | `emitRetireIntentRT()` | `ASSERT_NON_RT_THREAD()` 追加 |

### 修正案

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
     * TODO: 将来リネーム（emitRetireIntentFromNonRT）。
     *       ABI 破壊変更のため、バージョンアップ時に実施。
     */
    [[deprecated("RT は RealTime thread safety を意味しない。非 RT スレッド専用。")]]
    void emitRetireIntentRT(const RetireIntent& intent) noexcept;

    // ★ B14: Vyukov MPSC 新 API
    // ...
};
```

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
3. `[[deprecated]]` がコンパイル警告として表示されることを確認

---

## Finding #10 — MKL バッファ std::vector

### 概要

MKL DFTI API を直接呼ぶ関数内で `std::vector<double>` / `std::make_unique` が使用されており、 Finding #2（`IRAnalyzer.cpp`）と同型の規約違反。

### レビュー判定: △ 改善項目（allocator policy 決定後）

**理由**:

- `std::vector` だから悪いという訳ではない
- 問題は MKL へ渡す配列の **Alignment**
- IR ロード専用なら性能差はほぼ誤差
- 設計ポリシーの話であり、Bug ではなく Medium 改善程度

### ★ Allocator Policy（v5 追加）

MKL バッファのアラインメント方針を事前に決定する：

| Allocator | アラインメント | 用途 | 推奨 |
|-----------|---------------|------|------|
| `convo::makeAlignedArray<T>` | 64byte（AVX512 対応） | FFT バッファ、IR データ | ✅ 推奨 |
| `mkl_malloc` | MKL 内部最適 | MKL 専用バッファ | MKL 専用のみ |
| `std::vector<T>` | アラインメント保証なし | 一般用途 | ❌ MKL バッファには不適切 |

**方針**: MKL と直接やり取りするバッファは `convo::makeAlignedArray<T>` に統一。`mkl_malloc` は MKL 内部でのみ使用。

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

### レビュー判定: ★★★★☆ 採用（RAII/irData 分離）

**v5 の改善点**: `nuc0/nuc1` の RAII と `irData` の所有権を明確に分離して記述。

### 修正案（RAII/irData 分離）

**ファイル**: `src/ConvolverProcessor.h`

```cpp
// ★ irData の所有権: try ブロック外で管理（RAII で保護されない）
irData[0] = irL;
irData[1] = irR;

try
{
    // ★ nuc0/nuc1 の所有権: unique_ptr で RAII 保護
    auto nuc0 = convo::aligned_make_unique<convo::MKLNonUniformConvolver>();
    auto nuc1 = convo::aligned_make_unique<convo::MKLNonUniformConvolver>();

    // ★ SetImpulse() 中で例外が発生しても:
    //   - nuc0/nuc1: スコープ離脱時に自動破棄（RAII）
    //   - irData: try ブロック外のため、手動クリーンアップが必要
    if (nuc0->SetImpulse(irData[0], irDataLength, knownBlockSize, scale, enableDirectHead, filterSpec) &&
        nuc1->SetImpulse(irData[1], irDataLength, knownBlockSize, scale, enableDirectHead, filterSpec))
    {
        // 成功: 所有権を移動
        destroyNUCConvolver(nucConvolvers[0]);
        destroyNUCConvolver(nucConvolvers[1]);
        nucConvolvers[0] = nuc0.release();
        nucConvolvers[1] = nuc1.release();

        latency   = nucConvolvers[0]->getLatency();
        DBG("Convolver: NUC Engine Active. Latency: " << latency << " samples");
        return true;
    }
}
catch (const std::bad_alloc&)
{
    // メモリ確保失敗: nuc0/nuc1 は RAII で自動破棄、irData は手動解放
}
catch (...)
{
    // ★ v5 追加: その他の例外も捕捉（bug3-1 対応）
    // nuc0/nuc1 は RAII で自動破棄、irData は手動解放
}

// cleanup（全経路で到達）
destroyNUCConvolver(nucConvolvers[0]);
destroyNUCConvolver(nucConvolvers[1]);
if (irData[0]) { convo::aligned_free(irData[0]); irData[0] = nullptr; }
if (irData[1]) { convo::aligned_free(irData[1]); irData[1] = nullptr; }
irDataLength = 0;
latency = 0;
this->irLatency = 0;
return false;
```

### 要点

- `nuc0/nuc1` は `std::unique_ptr` で RAII 保護。例外時でも自動破棄。
- `irData` は try ブロック外で管理。例外時でも手動解放が必要。
- `catch(...)` で全例外を捕捉し、irData のリークを防止。

---

## bug3群 — 第3回報告バグ

### bug3-1 — 【重大】StereoConvolver::init — SetImpulse 非 bad_alloc 例外で irData リーク

**対象**: `src/ConvolverProcessor.h` (`StereoConvolver::init`)

**問題**: `irData[0]` / `irData[1]` は `try` ブロックの**前**で代入されるが、`catch` は `bad_alloc` しか捕捉しない。`SetImpulse` がそれ以外の例外を投げた場合、`irData` が解放されない。

**修正**: Bug H の修正と同一（`catch(...)` で全例外を捕捉）。

---

### bug3-2 — 【中】StereoConvolver::clone — init 失敗時の newConv リーク

**対象**: `src/ConvolverProcessor.h` (`StereoConvolver::clone`)

**問題**: `init` が `l.release()` / `r.release()` の**後**で失敗した場合、`newConv` の `unique_ptr` デストラクタが `~StereoConvolver()` を呼ぶが、デストラクタは `jassert` のみで解放しない。Bug 1 と同じ経路で `irData` が解放されない。

**修正**: Bug H の修正と同一（`catch(...)` で全例外を捕捉）。

---

### bug3-3 — 【中】StereoConvolver::process — numSamples <= 0 の未チェック

**対象**: `src/convolver/ConvolverProcessor.Runtime.cpp` (`StereoConvolver::process`)

**問題**: `numSamples` が 0 または負の場合のチェックがない。`numSamples` が負の場合、`numSamples * sizeof(double)` が `size_t` に変換されて巨大な値になり、`std::memset` でバッファオーバーフローが発生する。

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

**対象**: `src/ConvolverProcessor.h` (`StereoConvolver::init`)

**修正案**:
```cpp
// cleanup に追加
hasStoredFilterSpec = false;
storedFilterSpec = convo::FilterSpec{};
```

---

### bug3-5 — 【低】StereoConvolver::init — 失敗時に callQuantumSamples / storedSampleRate 等がリセットされない

**対象**: `src/ConvolverProcessor.h` (`StereoConvolver::init`)

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

**対象**: `ConvolverProcessor.h` (`processBypassWithLatencyCompensation`)

**修正方針**: Atomic 化ではなく、**呼び出し契約**（`reset()` は Audio Thread 停止後にのみ呼び出す）を保証する。

---

### bug3-7 — 【不採用】StereoConvolver::clone — irDataLength 負値のオーバーフロー

**判定**: **不採用**（既に `if (irDataLength > 0 && irData[0] && irData[1])` でガード済み）

---

### bug3-8 — 【低】StereoConvolver::process — got > numSamples の未チェック

**対象**: `src/convolver/ConvolverProcessor.Runtime.cpp` (`StereoConvolver::process`)

**修正案**:
```cpp
const int got = nucConvolvers[channel]->Get(out, numSamples);
jassert(got <= numSamples);  // 防御的チェック
if (got < numSamples)
    std::memset(out + got, 0, (numSamples - got) * sizeof(double));
```

---

## 改修優先度とスケジュール

### P0: 即時対応（3 日）

| Bug | 改修内容 | 工数 |
|-----|----------|------|
| Bug C | `pushBlock` 境界チェック（fail-fast） | 0.5 日 |
| Bug 2 | `processBypassWithLatencyCompensation` null クリア | 0.5 日 |
| Bug A/B/D | `sanitizeFinite()` 追加 + `quantize()` 修正 | 1 日 |
| Finding 9 | `emitRetireIntentRT` コメント + `ASSERT_NON_RT_THREAD()` + `[[deprecated]]` | 0.5 日 |
| bug3-1 | `catch(...)` 追加（irData リーク防止） | 0.5 日 |

### P2: 改善項目（3 日）

| Bug | 改修内容 | 工数 |
|-----|----------|------|
| Finding 10 | MKL バッファ std::vector → makeAlignedArray（allocator policy 決定後） | 3 日 |
| Bug G | 冗長チェックの整理 | 5 分 |

### P3: 保留（将来対応）

| Bug | 改修内容 | 工数 |
|-----|----------|------|
| Bug 3/E | TOCTOU 修正（sequence lock or 64bit atomic 化） | 2 日 |
| Bug F | 空ブロックにコメント追加 | 5 分 |
| bug3-4/5 | init 失敗時の状態リセット | 0.5 日 |
| bug3-6 | delayWritePos 呼び出し契約の明文化 | 0.5 日 |
| bug3-8 | got > numSamples の防御チェック | 5 分 |

### 総工数推定

- P0: 3 日
- P2: 3 日
- P3: 3 日
- **合計: 9 日**

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
| irData リーク | 中 | 中 | bug3-1 の修正 |

---

## 承認

| 役割 | 名前 | 日付 | 署名 |
|------|------|------|------|
| 設計者 | | | |
| レビュアー | | | |
| 承認者 | | | |
