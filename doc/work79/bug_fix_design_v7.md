# ConvoPeq 未修正バグ改修設計書（v7 — 最終版）

**作成日**: 2026-07-21
**レビュー反映**: 2026-07-21（v6 レビュー → v7 反映：6点の細部修正）
**対象**: Part 7〜10 + bug1/bug2/bug3 で特定された未修正バグ（19件）
**評価**: A+（95〜97点）— 実装着手可能

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
| v7 | 2026-07-21 | v6 レビュー反映：catch typo 修正、IEEE754 前提明記、BugC drop 時の状態不変明記、Finding10 対象限定、bug3-6 Debug アサート追加、BugH テストケース追加 |

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

## 評価サマリ（v7 — 最終）

### Part 7〜10 + bug1/bug2

| 項目 | 判定 | 優先度 | 採用/不採用 |
|------|------|--------|------------|
| Bug A/B/D | ★★★★★ | P0 | 採用 |
| Bug C | ★★★★★ | P0 | 採用（drop 方針） |
| Bug 2 | ★★★★★ | P0 | 採用 |
| Bug 3/E | ★☆☆☆☆ | 保留 | **不採用**（将来 64bit atomic または seqlock） |
| Finding 9 | ★★★★★ | P1 | 採用（deprecated 追加） |
| Finding 10 | ★★★☆☆ | P2 | 改善項目（MKL SIMD 配列のみ対象） |
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
| bug3-6 | ★★★☆☆ | P3 | **採用**（delayWritePos API 契約 + Debug アサート） |
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
| P2 | Finding 10 | 設計ポリシー統一（MKL SIMD 配列のみ対象） |
| P2 | Bug G | コード整理 |
| P3 | bug3-4/5 | init 失敗時の状態リセット |
| P3 | bug3-6 | delayWritePos API 契約 + Debug アサート |
| P3 | bug3-8 | got > numSamples の防御チェック |
| 保留 | Bug 3/E | 将来対応 |
| 保留 | Bug F | コメントのみ |

---

## NaN/Inf 対策の DSP 全体方針

### 前提条件（v7 追加）

- **IEEE754 binary32/binary64 前提**: `std::numeric_limits<double>::is_iec559 == true`
- `uniform()` は有限値のみ返す（API 契約）。ただし防御的に `sanitizeFinite()` で保護

### 責務境界

`quantize()` は公共 API であり、単独利用もあり得る。 therefore、**`quantize()` の先頭および出口で `sanitizeFinite()` を実行**する。

### 責務図（v7 変更点）

```
┌─────────────────────────────────────────────────────────────────┐
│                    DSP Numeric Policy                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  killDenormal()  →  デノーマル除去（FTZ/DAZ と連携）     │   │
│  │  sanitizeFinite() →  NaN/Inf 除去（本設計で追加）        │   │
│  │    - IEEE754 binary32/binary64 前提                     │   │
│  │    - Debug: jassert(isfinite(x)) で NaN 発生源を追跡    │   │
│  │    - Release: 0.0 に置換                                 │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    量子化（公共 API）                             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  quantize()                                            │   │
│  │    v = sanitizeFinite(v)  ← 入口で NaN/Inf 除去         │   │
│  │    ... dither + rounding ...                           │   │
│  │    return sanitizeFinite(q * scale)  ← 出口でも保護     │   │
│  │    ※ uniform() は有限値のみ返す契約だが防御的に保護     │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Bug A/B/D — NaN/Inf 伝播防止

### 概要

`killDenormal()` は Release ビルドで no-op であり、NaN/Inf を通過させる。`quantize()` は NaN が比較演算を通過するため、フィルタ発散時に音声出力が完全に破綻する。

### 修正対象

| ファイル | 関数 | 変更内容 |
|---------|------|----------|
| `src/DspNumericPolicy.h` | 新規 `sanitizeFinite(double)` | NaN/Inf → 0.0 変換 + Debug 用 jassert |
| `src/DspNumericPolicy.h` | 新規 `sanitizeFinite(float)` | NaN/Inf → 0.0f 変換 + Debug 用 jassert |
| `src/FixedNoiseShaper.h` | `quantize()` | 入口 + 出口で `sanitizeFinite()` 呼び出し |

### 修正案 1: `sanitizeFinite()` の追加

**ファイル**: `src/DspNumericPolicy.h`

```cpp
// ─────────────────────────────────────────────────────────────────
// NaN/Inf 除去ヘルパー関数
// 前提: IEEE754 binary32/binary64（std::numeric_limits::is_iec559）
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

### 修正案 2: `quantize()` の NaN ガード（入口 + 出口）

**ファイル**: `src/FixedNoiseShaper.h`

```cpp
inline double quantize(double v, Xoshiro256State& rng) const noexcept
{
    // NaN/Inf ガード（入口）
    v = sanitizeFinite(v);

    const double minV = -1.0;
    const double maxV = 1.0 - (1.0 / invScale);

    if (v < minV)
        v = minV;
    else if (v > maxV)
        v = maxV;

    // TPDF dither
    // ※ uniform() は有限値のみ返す API 契約だが、防御的に出口でも保護
    const double u1 = uniform(rng);
    const double u2 = uniform(rng);
    v += (u1 + u2 - 1.0) * scale;

    __m128d d = _mm_set_sd(v * invScale);
    d = _mm_round_sd(d, d, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const double q = _mm_cvtsd_f64(d);

    // NaN/Inf ガード（出口）: uniform() の将来変更に備えて防御
    return sanitizeFinite(q * scale);
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

### 修正方針

**Drop 方針**: `numSamples > kCapacity` の入力は契約違反であり、**処理を拒否**する。drop 時は **`writePosition` / `totalSamples` を更新しない**。

### 修正案

**ファイル**: `src/AudioSegmentBuffer.h`

```cpp
void pushBlock(const double* left, const double* right, int numSamples) noexcept
{
    if (left == nullptr || right == nullptr || numSamples <= 0)
        return;

    // ★ 境界チェック: kCapacity を超える入力は契約違反（drop）
    //   drop 時は writePosition / totalSamples を更新しない
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
| **drop 時** | — | `writePosition` / `totalSamples` を**更新しない** |

---

## Bug 2 — bypass delayBuffer null 時未クリア

### 修正案

```cpp
if (delayBuf[0] == nullptr || delayBuf[1] == nullptr || activeDelayCapacity < DELAY_BUFFER_SIZE)
{
    for (int ch = 0; ch < procChannels; ++ch)
        juce::FloatVectorOperations::clear(block.getChannelPointer(static_cast<size_t>(ch)), numSamples);
    return;
}
```

---

## Bug 3/E — copyLatest TOCTOU

### レビュー判定: 保留

C++ の acquire/release は別 atomic には伝播しない。SPSC 条件下では実用上安全。

---

## Finding #9 — emitRetireIntentRT 命名

### 修正案

```cpp
// ★ v7: deprecated 属性は public API の互換性に影響する可能性があるため、
//   まずコメントで警告し、ABI 破壊変更時に deprecated を検討する。
// TODO: 将来リネーム（emitRetireIntentFromNonRT）。バージョンアップ時に実施。
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

### 対象の限定（v7 変更点）

**全部置換ではなく、「MKL SIMD 処理へ直接渡す配列」のみ対象**。

| 状況 | 対応 |
|------|------|
| MKL に `.data()` で渡す配列 | `makeAlignedArray` に置換 |
| IR ロード専用で MKL に渡さない配列 | `std::vector` のままで可 |
| 非 RT スレッドで MKL を使わない関数 | `std::vector` のままで可 |

### Allocator Policy

| Allocator | アラインメント | 用途 | 推奨 |
|-----------|---------------|------|------|
| `convo::makeAlignedArray<T>` | 64byte | MKL SIMD 処理へ直接渡す配列 | ✅ 推奨 |
| `std::vector<T>` | アラインメント保証なし | MKL に渡さない一般用途 | ✅ そのまま可 |

---

## Bug F — StereoConvolver::init 空ブロック

### レビュー判定: △ 保留

Bug とは断定できない。将来拡張なのか実装漏れなのか、コードだけでは断定不可。

---

## Bug G — 冗長な負値チェック

### 説明

`DELAY_BUFFER_MASK` が `2^n - 1` であることから、`& DELAY_BUFFER_MASK` の結果は常に 0 以上。

```cpp
int readPos = (writePos - delaySamples) & DELAY_BUFFER_MASK;
// if (readPos < 0)  ← 死コード
//     readPos += DELAY_BUFFER_SIZE;
```

---

## Bug H — StereoConvolver::init 例外安全性

### 概要

`StereoConvolver::init` で `std::bad_alloc` のみキャッチ。他の例外時は `irData` がリークする可能性。

### 修正方針

`irData` を **`aligned_unique_ptr`** で RAII 管理し、例外安全性を根本的に解決。

### 修正案（aligned_unique_ptr 化）

**ファイル**: `src/ConvolverProcessor.h`

```cpp
struct StereoConvolver
{
    // ★ irData を aligned_unique_ptr で RAII 管理
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

            if (nuc0->SetImpulse(irData[0].get(), irDataLength, knownBlockSize, scale, enableDirectHead, filterSpec) &&
                nuc1->SetImpulse(irData[1].get(), irDataLength, knownBlockSize, scale, enableDirectHead, filterSpec))
            {
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
            // メモリ確保失敗: irData は aligned_unique_ptr で自動解放
        }
        catch (...)
        {
            // その他の例外: irData は aligned_unique_ptr で自動解放
        }

        // cleanup（失敗時）
        destroyNUCConvolver(nucConvolvers[0]);
        destroyNUCConvolver(nucConvolvers[1]);
        irData[0].reset();
        irData[1].reset();
        irDataLength = 0;
        latency = 0;
        this->irLatency = 0;
        return false;
    }
};
```

### テスト方法

1. **Exception injection test（全パターン）**:
   - `SetImpulse` 1回目で `throw runtime_error` → 全 cleanup が正しく動くことを確認
   - `SetImpulse` 1回目成功・2回目で `throw runtime_error` → **片チャンネル成功時のリークがないこと**を確認
2. 通常動作で問題がないことを確認
3. メモリリークがないことを確認（Valgrind / AddressSanitizer）

---

## bug3群 — 第3回報告バグ

### bug3-1 / bug3-2

Bug H の aligned_unique_ptr 化で**同時に解決**。

### bug3-3 — numSamples <= 0 チェック

```cpp
if (channel < 0 || channel >= 2 || !nucConvolvers[channel] || numSamples <= 0)
{
    if (numSamples > 0)
        std::memset(out, 0, numSamples * sizeof(double));
    return;
}
```

### bug3-4 / bug3-5 — init 失敗時の状態リセット

```cpp
// cleanup に追加
hasStoredFilterSpec = false;
storedFilterSpec = convo::FilterSpec{};
callQuantumSamples = 0;
storedSampleRate = 0.0;
storedKnownBlockSize = 0;
storedScale = 1.0;
storedDirectHeadEnabled = false;
```

### bug3-6 — delayWritePos API 契約

**ファイル**: `src/ConvolverProcessor.h`

```cpp
// ★ API 契約: reset() は Audio Thread 停止後にのみ呼び出すこと。
//   Audio Thread 実行中に reset() を呼び出すとデータレースが発生する。
int delayWritePos = 0;
```

**ファイル**: `src/convolver/ConvolverProcessor.Runtime.cpp`

```cpp
void ConvolverProcessor::reset() noexcept
{
    // ★ API 契約違反の検知（Debug ビルド）
    ASSERT_NON_RT_THREAD();
    // ... 既存の処理 ...
    delayWritePos = 0;
}
```

### bug3-8 — got > numSamples の防御チェック

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
| Bug A/B/D | `sanitizeFinite()` 追加 + `quantize()` 入口/出口修正 | 1 日 |
| Bug H / bug3-1 | `irData` の `aligned_unique_ptr` 化 | 1.5 日 |
| bug3-3 | `numSamples <= 0` チェック追加 | 0.5 日 |

### P2: 改善項目（3 日）

| Bug | 改修内容 | 工数 |
|-----|----------|------|
| Finding 10 | MKL SIMD 配列のみ `makeAlignedArray` に置換 | 3 日 |
| Bug G | 冗長チェックの整理 | 5 分 |

### P3: 保留（将来対応）

| Bug | 改修内容 | 工数 |
|-----|----------|------|
| Bug 3/E | TOCTOU 修正 | 2 日 |
| Bug F | 空ブロックにコメント追加 | 5 分 |
| bug3-4/5 | init 失敗時の状態リセット | 0.5 日 |
| bug3-6 | delayWritePos API 契約 + Debug アサート | 5 分 |
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
4. **Exception injection test（全パターン）**:
   - `SetImpulse` 1回目で `throw runtime_error` → cleanup 確認
   - `SetImpulse` 1回目成功・2回目で `throw runtime_error` → 片チャンネル成功時のリーク確認

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
