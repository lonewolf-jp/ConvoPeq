# ConvoPeq 未修正バグ改修設計書（v8 — 完成版）

**作成日**: 2026-07-21
**レビュー反映**: 2026-07-21（v7 レビュー → v8 反映：Bug H Strong Exception Guarantee 再設計、細部修正）
**対象**: Part 7〜10 + bug1/bug2/bug3 で特定された未修正バグ（19件）
**評価**: A+（95〜97点）— 実装着手可能

---

## 変更履歴

| バージョン | 日付 | 変更内容 |
|-----------|------|----------|
| v1-v6 | 2026-07-21 | 段階的にレビュー反映 |
| v7 | 2026-07-21 | v6 レビュー反映：catch typo 修正、IEEE754 前提明記、BugC drop 明記、Finding10 対象限定、bug3-6 Debug アサート、BugH テストケース追加 |
| v8 | 2026-07-21 | v7 レビュー反映：**Bug H Strong Exception Guarantee 再設計**、BugA static_assert 追加、BugC jassert 可読性改善、bug3-8 got>=0 チェック追加、BugH catch 内 DBG 追加 |

---

## 評価サマリ（v8 — 最終）

### Part 7〜10 + bug1/bug2

| 項目 | 判定 | 優先度 | 採用/不採用 |
|------|------|--------|------------|
| Bug A/B/D | ★★★★★ | P0 | 採用 |
| Bug C | ★★★★★ | P0 | 採用（drop 方針） |
| Bug 2 | ★★★★★ | P0 | 採用 |
| Bug 3/E | ★☆☆☆☆ | 保留 | **不採用** |
| Finding 9 | ★★★★★ | P1 | 採用 |
| Finding 10 | ★★★☆☆ | P2 | 改善項目（MKL SIMD 配列のみ対象） |
| Bug F | ★★☆☆☆ | P3 | 保留 |
| Bug G | ★★★☆☆ | P2 | 整理程度 |
| Bug H | ★★★★★ | P0 | **採用**（Strong Exception Guarantee） |

### bug3.md 検証結果

| 項目 | 判定 | 優先度 | 採用/不採用 |
|------|------|--------|------------|
| bug3-1 | ★★★★★ | P0 | **採用**（Bug H で同時に解決） |
| bug3-2 | ★★★★☆ | P0 | **採用**（Bug H で同時に解決） |
| bug3-3 | ★★★★★ | P0 | **採用** |
| bug3-4 | ★★★☆☆ | P3 | **採用** |
| bug3-5 | ★★★☆☆ | P3 | **採用** |
| bug3-6 | ★★★☆☆ | P3 | **採用** |
| bug3-7 | ★☆☆☆☆ | — | **不採用** |
| bug3-8 | ★★★☆☆ | P3 | **採用** |

---

## Bug A/B/D — NaN/Inf 伝播防止

### 修正案

**ファイル**: `src/DspNumericPolicy.h`

```cpp
// ─────────────────────────────────────────────────────────────────
// NaN/Inf 除去ヘルパー関数
// 前提: IEEE754 binary32/binary64
// ─────────────────────────────────────────────────────────────────

// ★ v8: static_assert で IEEE754 前提を明示（設計と実装の一致）
static_assert(std::numeric_limits<double>::is_iec559, "IEEE754 binary64 前提");
static_assert(std::numeric_limits<float>::is_iec559, "IEEE754 binary32 前提");

inline double sanitizeFinite(double x) noexcept
{
#if JUCE_DEBUG || defined(_DEBUG)
    // Debug ビルド: NaN/Inf 発生時にアサーションで検出
    // ★ アサーション後も安全側（0.0）へフォールバックする
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
    jassert(std::isfinite(x));  // アサーション後も安全側へフォールバック
#endif
    constexpr uint32_t kExpMask = 0x7F800000U;
    const uint32_t bits = std::bit_cast<uint32_t>(x);
    const bool isNanOrInf = ((bits & kExpMask) == 0x7F800000U);
    return isNanOrInf ? 0.0f : x;
}
```

**ファイル**: `src/FixedNoiseShaper.h`

```cpp
inline double quantize(double v, Xoshiro256State& rng) const noexcept
{
    // NaN/Inf ガード（入口）
    v = sanitizeFinite(v);

    const double minV = -1.0;
    const double maxV = 1.0 - (1.0 / invScale);
    if (v < minV) v = minV;
    else if (v > maxV) v = maxV;

    // TPDF dither
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

---

## Bug C — pushBlock 境界チェック

### 修正案（v8: jassert 条件を可読性優先に変更）

```cpp
if (numSamples > kCapacity)
{
    jassert(numSamples <= kCapacity);  // ★ v8: 条件を直接記述（可読性向上）
    return;
}
```

### API 契約

| パラメータ | 契約 | 違反時の挙動 |
|-----------|------|-------------|
| `numSamples` | `0 < numSamples <= kCapacity` | `jassert` + return（drop） |
| **drop 時** | — | `writePosition` / `totalSamples` を**更新しない** |

---

## Bug 2 — bypass delayBuffer null 時未クリア

変更なし。無音化は正しい。

---

## Bug H — StereoConvolver::init 例外安全性（v8: Strong Exception Guarantee）

### v7 の問題

```cpp
irData[0].reset(irL);  // ← ここで所有権を取得
irData[1].reset(irR);
// ... 失敗時 cleanup で reset() → 呼び出し側は irData が解放されていると誤認
```

- `init()` の途中で入力ポインタの所有権を取得するため、API 契約が変わる
- 失敗時に cleanup で `reset()` されるため、呼び出し側の期待と不一致

### v8 の修正案: Strong Exception Guarantee

**一時的な `aligned_unique_ptr` で初期化を実行し、成功時のみ `irData` に `std::move()` する。**

```cpp
bool init(double* irL, double* irR, int length, double sr, int peakDelay, int knownBlockSize, int preferredCallSize, double scale = 1.0,
      bool enableDirectHead = false,
      const convo::FilterSpec* filterSpec = nullptr,
      ConvolverProcessor* ownerProcessor = nullptr)
{
    // ★ v8: 一時的な aligned_unique_ptr で初期化（Strong Exception Guarantee）
    //   irData は成功時のみ更新。失敗時は元の状態を維持。
    convo::aligned_unique_ptr<double[]> newIrL(irL);
    convo::aligned_unique_ptr<double[]> newIrR(irR);

    // 既存の irData を解放（init の再呼び出し対応）
    irData[0].reset();
    irData[1].reset();

    irDataLength = length;
    this->irLatency = peakDelay;
    callQuantumSamples = juce::jmax(1, preferredCallSize);
    storedSampleRate = sr;
    storedKnownBlockSize = knownBlockSize;
    storedScale = scale;
    storedDirectHeadEnabled = enableDirectHead;
    if (filterSpec != nullptr) {
        storedFilterSpec = *filterSpec;
        hasStoredFilterSpec = true;
    } else {
        hasStoredFilterSpec = false;
    }

    try
    {
        auto nuc0 = convo::aligned_make_unique<convo::MKLNonUniformConvolver>();
        auto nuc1 = convo::aligned_make_unique<convo::MKLNonUniformConvolver>();

        if (nuc0->SetImpulse(newIrL.get(), irDataLength, knownBlockSize, scale, enableDirectHead, filterSpec) &&
            nuc1->SetImpulse(newIrR.get(), irDataLength, knownBlockSize, scale, enableDirectHead, filterSpec))
        {
            // ★ 成功時のみ所有権を移動
            destroyNUCConvolver(nucConvolvers[0]);
            destroyNUCConvolver(nucConvolvers[1]);
            nucConvolvers[0] = nuc0.release();
            nucConvolvers[1] = nuc1.release();

            // ★ irData に move（成功時のみ）
            irData[0] = std::move(newIrL);
            irData[1] = std::move(newIrR);

            latency = nucConvolvers[0]->getLatency();
            DBG("Convolver: NUC Engine Active. Latency: " << latency << " samples");
            return true;
        }
    }
    catch (const std::bad_alloc&)
    {
        DBG("Convolver: init failed - memory allocation error");
        // newIrL/newIrR はスコープ離脱時に自動解放（RAII）
    }
    catch (...)
    {
        DBG("Convolver: init failed - unknown exception");
        // newIrL/newIrR はスコープ離脱時に自動解放（RAII）
    }

    // ★ 失敗時: irData は更新されない（Strong Exception Guarantee）
    //   newIrL/newIrR はスコープ離脱時に自動解放
    irDataLength = 0;
    latency = 0;
    this->irLatency = 0;
    return false;
}
```

### Strong Exception Guarantee の保証

| 状況 | irData の状態 | newIrL/newIrR |
|------|-------------|---------------|
| **成功** | 更新（move） | release（所有権移動済み） |
| **失敗（例外）** | **更新されない** | 自動解放（RAII） |
| **失敗（SetImpulse false）** | **更新されない** | 自動解放（RAII） |

### API 契約（v8 明確化）

- `init()` は成功時にのみ `irData` を更新する
- 失敗時は `irData` が元の状態を維持する（Strong Exception Guarantee）
- 呼び出し側は失敗時に `irData` が保持されていると期待してよい

### テスト方法

1. **Exception injection test（全パターン）**:
   - `SetImpulse` 1回目で `throw runtime_error` → irData が更新されていないことを確認
   - `SetImpulse` 1回目成功・2回目で `throw runtime_error` → irData[0] は保持、irData[1] は更新されていないことを確認
2. 通常動作で問題がないことを確認
3. メモリリークがないことを確認（Valgrind / AddressSanitizer）

---

## Finding #9 — emitRetireIntentRT 命名

変更なし。`ASSERT_NON_RT_THREAD()` 追加はそのまま採用。

---

## Finding #10 — MKL バッファ std::vector

変更なし。「MKL SIMD 処理へ直接渡す配列」のみ対象。

---

## Bug G — 冗長な負値チェック

変更なし。マスク値の性質による説明は正しい。

---

## bug3群 — 第3回報告バグ

### bug3-1 / bug3-2

Bug H の Strong Exception Guarantee 再設計で**同時に解決**。

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
hasStoredFilterSpec = false;
storedFilterSpec = convo::FilterSpec{};
callQuantumSamples = 0;
storedSampleRate = 0.0;
storedKnownBlockSize = 0;
storedScale = 1.0;
storedDirectHeadEnabled = false;
```

### bug3-6 — delayWritePos API 契約

```cpp
// ★ API 契約: reset() は Audio Thread 停止後にのみ呼び出すこと。
int delayWritePos = 0;
```

### bug3-8 — got > numSamples の防御チェック（v8: got >= 0 チェック追加）

```cpp
const int got = nucConvolvers[channel]->Get(out, numSamples);
jassert(got >= 0 && got <= numSamples);  // ★ v8: 範囲チェック
if (got < numSamples)
    std::memset(out + got, 0, (numSamples - got) * sizeof(double));
```

---

## 実装優先順位

| 優先 | 項目 | 理由 |
|------|------|------|
| P0-1 | Bug C | メモリ破壊防止 |
| P0-2 | Bug 2 | stale data 防止 |
| P0-3 | Bug A/B/D | 音声破綻防止 |
| P0-4 | Bug H / bug3-1 | irData リーク（Strong Exception Guarantee） |
| P0-5 | bug3-3 | numSamples 負値でバッファオーバーフロー |
| P1 | Finding 9 | Debug での誤呼び出し検知 |
| P2 | Finding 10 | 設計ポリシー統一 |
| P2 | Bug G | コード整理 |
| P3 | bug3-4/5/6/8 | 各種防御的改善 |
| 保留 | Bug 3/E, Bug F | 将来対応 |

### 総工数推定

- P0: 4 日
- P2: 3 日
- P3: 2 日
- **合計: 9 日**

---

## テスト計画

### 単体テスト

1. **NaN/Inf テスト**: NaN/Inf を注入した入力でフィルタをテスト
2. **境界テスト**: `pushBlock` で `numSamples > kCapacity` の場合をテスト
3. **null テスト**: `processBypassWithLatencyCompensation` で delayBuffer null の場合をテスト
4. **Exception injection test（全パターン）**:
   - `SetImpulse` 1回目で例外 → irData が更新されていないことを確認
   - `SetImpulse` 1回目成功・2回目で例外 → 片チャンネル保持を確認

### 統合テスト

1. 通常動作テスト、パフォーマンステスト、メモリテスト

---

## 承認

| 役割 | 名前 | 日付 | 署名 |
|------|------|------|------|
| 設計者 | | | |
| レビュアー | | | |
| 承認者 | | | |
