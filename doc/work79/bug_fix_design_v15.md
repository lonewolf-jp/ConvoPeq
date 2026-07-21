# ConvoPeq 未修正バグ改修設計書（v15 — 最終完成版）

**作成日**: 2026-07-21
**レビュー反映**: 2026-07-21（v14 レビュー → v15 反映：getLatency noexcept 推奨追加、ビット判定コメント強化）
**対象**: Part 7〜10 + bug1/bug2/bug3 で特定された未修正バグ（19件）
**評価**: A+（97〜98点）— 実装着手可能

---

## 変更履歴

| バージョン | 日付 | 変更内容 |
|-----------|------|----------|
| v1-v13 | 2026-07-21 | 段階的にレビュー反映 |
| v14 | 2026-07-21 | IEEE754 準拠コメント修正（quiet NaN / signaling NaN / ±Inf 列挙） |
| v15 | 2026-07-21 | **v14 レビュー反映：getLatency() noexcept 宣言推奨追加、ビット判定コメント IEEE754 根拠強化** |

---

## 評価サマリ（v15 — 最終）

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
| Bug H | ★★★★★ | P0 | **採用**（getLatency noexcept 推奨追加） |

---

## Bug A/B/D — NaN/Inf 伝播防止（v15: ビット判定コメント強化）

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
    // IEEE754: 指数部が全て1の値は NaN または ±Inf のみ
    //   指数部==all ones && 仮数部==0 → ±Inf
    //   指数部==all ones && 仮数部!=0 → NaN (quiet/signaling)
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

- 「指数部が全て1の値は NaN または ±Inf のみ」を明記（IEEE754 の正式な仕様）
- 指数部==all ones && 仮数部==0 → ±Inf、指数部==all ones && 仮数部!=0 → NaN を記載

---

## Bug C — pushBlock 境界チェック

変更なし。

---

## Bug 2 — bypass delayBuffer null 時未クリア

変更なし。

---

## Bug H — StereoConvolver::init 例外安全性（v15: getLatency noexcept 推奨追加）

### 実装時の推奨事項（v15 追加）

**`getLatency()` を `noexcept` 宣言することを推奨する。**

```cpp
// ★ 推奨: getLatency() を noexcept 宣言
//   設計書側で「noexcept 操作」と説明する根拠がより強固になる
[[nodiscard]] int getLatency() const noexcept { return latency; }
```

**メリット**:
- 設計書の「Phase 2 は noexcept 操作のみ」という前提がコンパイラで保証される
- 万一将来例外を投げる実装に変更された場合、コンパイルエラーで検知できる
- レビュー時の説明が不要になる

### Phase 2 の noexcept 根拠（v15 更新）

| 操作 | noexcept 根拠 |
|------|--------------|
| `destroyNUCConvolver(ptr)` | `~MKLNonUniformConvolver()`（noexcept 前提）+ `mkl_free`（C 関数） |
| `aligned_unique_ptr::reset()` | `aligned_free`（`mkl_free` wrapper） |
| `std::move(unique_ptr)` | noexcept |
| `unique_ptr::release()` | noexcept |
| メンバー代入 | noexcept |
| `newNuc0->getLatency()` | **noexcept 宣言を推奨**（v15 追加） |

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
