# ConvoPeq 未修正バグ改修設計書（v13 — 最終完成版）

**作成日**: 2026-07-21
**レビュー反映**: 2026-07-21（v12 レビュー → v13 反映：MKLNonUniformConvolver デストラクタ noexcept 前提明記、コメント統一）
**対象**: Part 7〜10 + bug1/bug2/bug3 で特定された未修正バグ（19件）
**評価**: A+（97〜98点）— 実装着手可能

---

## 変更履歴

| バージョン | 日付 | 変更内容 |
|-----------|------|----------|
| v1-v11 | 2026-07-21 | 段階的にレビュー反映 |
| v12 | 2026-07-21 | commit 順序改善、sanitizeFinite 命名変更、nullptr テスト修正 |
| v13 | 2026-07-21 | **v12 レビュー反映：MKLNonUniformConvolver デストラクタ noexcept 前提明記、コメント統一（全ての非有限値）** |

---

## 評価サマリ（v13 — 最終）

| 項目 | 判定 | 優先度 | 採用/不採用 |
|------|------|--------|------------|
| Bug A/B/D | ★★★★★ | P0 | 採用 |
| Bug C | ★★★★★ | P0 | 採用（drop 方針） |
| Bug 2 | ★★★★★ | P0 | 採用 |
| Bug 3/E | ★☆☆☆☆ | 保留 | **不採用** |
| Finding 9 | ★★★★★ | P1 | 採用 |
| Finding 10 | ★★★☆☆ | P2 | 改善項目 |
| Bug F | ★★☆☆☆ | P3 | 保留 |
| Bug G | ★★★☆☆ | P2 | 整理程度 |
| Bug H | ★★★★★ | P0 | **採用**（noexcept 前提明記済み） |

---

## Bug A/B/D — NaN/Inf 伝播防止（v13: コメント統一）

### 修正案

**ファイル**: `src/DspNumericPolicy.h`

```cpp
// ─────────────────────────────────────────────────────────────────
// 全ての非有限値（NaN・±Inf）を 0.0 に置換するヘルパー関数
// 前提: IEEE754 binary32/binary64
// ─────────────────────────────────────────────────────────────────

static_assert(std::numeric_limits<double>::is_iec559, "IEEE754 binary64 前提");
static_assert(std::numeric_limits<float>::is_iec559, "IEEE754 binary32 前提");

inline double replaceNonFiniteWithZero(double x) noexcept
{
#if JUCE_DEBUG || defined(_DEBUG)
    // Debug: 全ての非有限値（NaN・±Inf）の発生をアサーションで検出
    // アサーション後も安全側（0.0）へフォールバック
    jassert(std::isfinite(x));
#endif
    constexpr uint64_t kExpMask = 0x7FF0000000000000ULL;
    const uint64_t bits = std::bit_cast<uint64_t>(x);
    // 指数部が全て1 → quiet NaN か ±Inf（全ての非有限値）
    const bool isNonFinite = ((bits & kExpMask) == 0x7FF0000000000000ULL);
    return isNonFinite ? 0.0 : x;
}

inline float replaceNonFiniteWithZero(float x) noexcept
{
#if JUCE_DEBUG || defined(_DEBUG)
    jassert(std::isfinite(x));
#endif
    constexpr uint32_t kExpMask = 0x7F800000U;
    const uint32_t bits = std::bit_cast<uint32_t>(x);
    const bool isNonFinite = ((bits & kExpMask) == 0x7F800000U);
    return isNonFinite ? 0.0f : x;
}
```

### ポイント

- コメントを「全ての非有限値（NaN・±Inf）」に統一（実装と完全に一致）
- ビット判定の説明を「指数部が全て1 → quiet NaN か ±Inf」に追加

---

## Bug C — pushBlock 境界チェック

変更なし。

---

## Bug 2 — bypass delayBuffer null 時未クリア

変更なし。

---

## Bug H — StereoConvolver::init 例外安全性（v13: noexcept 前提明記）

### Phase 2 の noexcept 根拠（v13 明確化）

| 操作 | noexcept 根拠 |
|------|--------------|
| `destroyNUCConvolver(ptr)` | `ptr->~MKLNonUniformConvolver()` + `mkl_free()`。**MKLNonUniformConvolver のデストラクタは `noexcept` であることを前提とする**（C++11 以降、デストラクタは暗黙的に `noexcept`）。`mkl_free` は C 関数で例外を投げない |
| `aligned_unique_ptr::reset()` | `aligned_free()`（`mkl_free` wrapper）を呼ぶだけ。`noexcept` |
| `std::move(unique_ptr)` | `noexcept` |
| `unique_ptr::release()` | 生ポインタを返すだけ。`noexcept` |
| メンバー代入 | 組み込み型・POD 型。`noexcept` |

**結論**: Phase 2 の全操作は `noexcept` である。例外は発生しない。

### 修正案（v13）

```cpp
bool init(double* irL, double* irR, int length, double sr, int peakDelay, int knownBlockSize, int preferredCallSize, double scale = 1.0,
      bool enableDirectHead = false,
      const convo::FilterSpec* filterSpec = nullptr,
      ConvolverProcessor* ownerProcessor = nullptr)
{
    // ============================================================
    // Phase 1: すべてローカル変数で初期化を実行（メンバー未更新）
    // ============================================================

    convo::aligned_unique_ptr<double[]> newIrL(irL);
    convo::aligned_unique_ptr<double[]> newIrR(irR);

    auto newNuc0 = convo::aligned_make_unique<convo::MKLNonUniformConvolver>();
    auto newNuc1 = convo::aligned_make_unique<convo::MKLNonUniformConvolver>();

    if (!newNuc0->SetImpulse(newIrL.get(), length, knownBlockSize, scale, enableDirectHead, filterSpec))
    {
        DBG("Convolver: init failed - SetImpulse ch0 failed");
        return false;
    }
    if (!newNuc1->SetImpulse(newIrR.get(), length, knownBlockSize, scale, enableDirectHead, filterSpec))
    {
        DBG("Convolver: init failed - SetImpulse ch1 failed");
        return false;
    }

    // ============================================================
    // Phase 2: 全成功 — メンバーを一括更新（commit）
    //
    // ★ noexcept 根拠:
    //   - destroyNUCConvolver: ~MKLNonUniformConvolver() (noexcept 前提) + mkl_free (C関数)
    //   - aligned_unique_ptr::reset: aligned_free (mkl_free wrapper, noexcept)
    //   - std::move(unique_ptr): noexcept
    //   - unique_ptr::release: noexcept
    //   - メンバー代入: noexcept
    //
    // ★ 保証対象: StereoConvolver の内部状態のみ。
    //   呼び出し側の所有権状態は保証しない（irL.release() 済み）。
    // ============================================================

    const int newLatency = newNuc0->getLatency();

    destroyNUCConvolver(nucConvolvers[0]);
    destroyNUCConvolver(nucConvolvers[1]);
    irData[0].reset();
    irData[1].reset();

    irData[0] = std::move(newIrL);
    irData[1] = std::move(newIrR);
    nucConvolvers[0] = newNuc0.release();
    nucConvolvers[1] = newNuc1.release();

    irDataLength = length;
    this->irLatency = peakDelay;
    latency = newLatency;
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

    DBG("Convolver: NUC Engine Active. Latency: " << latency << " samples");
    return true;
}
```

### テスト方法

| テストケース | 期待結果 |
|-------------|----------|
| `SetImpulse` 1回目で例外 | 内部状態が元のまま |
| `SetImpulse` 1回目成功・2回目で例外 | 内部状態が元のまま |
| `SetImpulse == false` | 内部状態が元のまま |
| `aligned_make_unique` 失敗 | 内部状態が元のまま |
| `filterSpec == nullptr` | 正常動作 |
| `length == 0` | 正常動作 |
| `preferredCallSize == 0` | 正常動作 |
| `irL == nullptr` | API 契約違反として適切に失敗する |
| `irR == nullptr` | API 契約違反として適切に失敗する |

---

## Finding #9 — emitRetireIntentRT 命名

変更なし。

---

## Finding #10 — MKL バッファ std::vector

変更なし。

---

## Bug G — 冗長な負値チェック

変更なし。

---

## bug3群 — 第3回報告バグ

bug3-1/2 は Bug H で解決。bug3-3/6/8 は変更なし。

---

## 承認

| 役割 | 名前 | 日付 | 署名 |
|------|------|------|------|
| 設計者 | | | |
| レビュアー | | | |
| 承認者 | | | |
