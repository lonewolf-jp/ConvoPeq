# ConvoPeq 未修正バグ改修設計書（v14 — 最終完成版）

**作成日**: 2026-07-21
**レビュー反映**: 2026-07-21（v13 レビュー → v14 反映：NaN 判定コメントの IEEE754 準拠修正）
**対象**: Part 7〜10 + bug1/bug2/bug3 で特定された未修正バグ（19件）
**評価**: A+（97〜98点）— 実装着手可能

---

## 変更履歴

| バージョン | 日付 | 変更内容 |
|-----------|------|----------|
| v1-v12 | 2026-07-21 | 段階的にレビュー反映 |
| v13 | 2026-07-21 | MKLNonUniformConvolver デストラクタ noexcept 前提明記、コメント統一 |
| v14 | 2026-07-21 | **v13 レビュー反映：NaN 判定コメントを IEEE754 準拠に修正（quiet NaN + signaling NaN + ±Inf）** |

---

## 評価サマリ（v14 — 最終）

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
| Bug H | ★★★★★ | P0 | **採用** |

---

## Bug A/B/D — NaN/Inf 伝播防止（v14: IEEE754 準拠コメント）

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
    jassert(std::isfinite(x));
#endif
    constexpr uint64_t kExpMask = 0x7FF0000000000000ULL;
    const uint64_t bits = std::bit_cast<uint64_t>(x);
    // IEEE754: 指数部が全て1 → NaN または ±Inf（全ての非有限値）
    //   - quiet NaN (Mantissa != 0, MSB=1)
    //   - signaling NaN (Mantissa != 0, MSB=0)
    //   - +Inf (Mantissa == 0, Sign=0)
    //   - -Inf (Mantissa == 0, Sign=1)
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

- コメントを IEEE754 の正式な分類に準拠（quiet NaN / signaling NaN / +Inf / -Inf を列挙）
- `quiet NaN か ±Inf` → `NaN または ±Inf（全ての非有限値）` に修正

---

## Bug C — pushBlock 境界チェック

変更なし。

---

## Bug 2 — bypass delayBuffer null 時未クリア

変更なし。

---

## Bug H — StereoConvolver::init 例外安全性

変更なし。

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

変更なし。

---

## 承認

| 役割 | 名前 | 日付 | 署名 |
|------|------|------|------|
| 設計者 | | | |
| レビュアー | | | |
| 承認者 | | | |
