# ConvoPeq 未修正バグ改修設計書（v16 — 最終完成版）

**作成日**: 2026-07-21
**レビュー反映**: 2026-07-21（v15 レビュー → v16 反映：変数名改善、noexcept 表現の正確化）
**対象**: Part 7〜10 + bug1/bug2/bug3 で特定された未修正バグ（19件）
**評価**: A+（98〜99点）— 実装着手可能

---

## 変更履歴

| バージョン | 日付 | 変更内容 |
|-----------|------|----------|
| v1-v14 | 2026-07-21 | 段階的にレビュー反映 |
| v15 | 2026-07-21 | getLatency() noexcept 宣言推奨追加、IEEE754 コメント補強 |
| v16 | 2026-07-21 | **v15 レビュー反映：変数名 kExpMask→kExponentMask 改善、noexcept 表現を正確化** |

---

## 評価サマリ（v16 — 最終）

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

## Bug A/B/D — NaN/Inf 伝播防止（v16: 変数名改善）

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
    constexpr uint64_t kExponentMask = 0x7FF0000000000000ULL;
    const uint64_t bits = std::bit_cast<uint64_t>(x);
    // IEEE754: 指数部が全て1の値は NaN または ±Inf のみ
    //   指数部==all ones && 仮数部==0 → ±Inf
    //   指数部==all ones && 仮数部!=0 → NaN (quiet/signaling)
    const bool isNonFinite = (bits & kExponentMask) == kExponentMask;
    return isNonFinite ? 0.0 : x;
}

inline float replaceNonFiniteWithZero(float x) noexcept
{
#if JUCE_DEBUG || defined(_DEBUG)
    jassert(std::isfinite(x));
#endif
    constexpr uint32_t kExponentMask = 0x7F800000U;
    const uint32_t bits = std::bit_cast<uint32_t>(x);
    const bool isNonFinite = (bits & kExponentMask) == kExponentMask;
    return isNonFinite ? 0.0f : x;
}
```

### 変更点

- `kExpMask` → `kExponentMask`（可読性向上、レビュー時の説明が不要に）
- `(bits & mask) == mask` の比較が直感的に理解できる

---

## Bug C — pushBlock 境界チェック

変更なし。

---

## Bug 2 — bypass delayBuffer null 時未クリア

変更なし。

---

## Bug H — StereoConvolver::init 例外安全性（v16: noexcept 表現の正確化）

### getLatency() noexcept 推奨（v16 表現修正）

```cpp
// ★ 推奨: getLatency() を noexcept 宣言
//   API 契約を型レベルで明示できる。
//   将来の実装変更で例外が伝播した場合は std::terminate() となり、
//   契約違反を早期に顕在化できる。
[[nodiscard]] int getLatency() const noexcept { return latency; }
```

### noexcept の挙動（C++ 仕様に基づく正確な説明）

| 状況 | 挙動 |
|------|------|
| `noexcept` 関数内で例外が発生 | `std::terminate()` が呼ばれる |
| 例外が外部に伝播 | `std::terminate()` が呼ばれる |
| **メリット** | API 契約を型レベルで明示できる |

**注意**: 「コンパイルエラーで検知できる」は不正確。C++ の `noexcept` は例外送出時に `std::terminate()` の対象であり、コンパイルエラーにはならない。

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
