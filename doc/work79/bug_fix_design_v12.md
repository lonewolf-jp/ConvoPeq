# ConvoPeq 未修正バグ改修設計書（v12 — 最終完成版）

**作成日**: 2026-07-21
**レビュー反映**: 2026-07-21（v11 レビュー → v12 反映：commit 順序改善、nullptr テスト修正、sanitizeFinite 命名検討）
**対象**: Part 7〜10 + bug1/bug2/bug3 で特定された未修正バグ（19件）
**評価**: A+（97〜98点）— 実装着手可能

---

## 変更履歴

| バージョン | 日付 | 変更内容 |
|-----------|------|----------|
| v1-v10 | 2026-07-21 | 段階的にレビュー反映 |
| v11 | 2026-07-21 | Bug H noexcept 根拠明記、nullptr テスト修正 |
| v12 | 2026-07-21 | **v11 レビュー反映：commit 順序改善（getLatency を前倒し）、nullptr テスト表現修正、sanitizeFinite 命名検討** |

---

## 評価サマリ（v12 — 最終）

| 項目 | 判定 | 優先度 | 採用/不採用 |
|------|------|--------|------------|
| Bug A/B/D | ★★★★★ | P0 | 採用（`replaceNonFiniteWithZero` 命名推奨） |
| Bug C | ★★★★★ | P0 | 採用（drop 方針） |
| Bug 2 | ★★★★★ | P0 | 採用 |
| Bug 3/E | ★☆☆☆☆ | 保留 | **不採用** |
| Finding 9 | ★★★★★ | P1 | 採用 |
| Finding 10 | ★★★☆☆ | P2 | 改善項目 |
| Bug F | ★★☆☆☆ | P3 | 保留 |
| Bug G | ★★★☆☆ | P2 | 整理程度 |
| Bug H | ★★★★★ | P0 | **採用**（commit 順序改善済み） |

---

## Bug A/B/D — NaN/Inf 伝播防止（v12: 命名検討）

### 命名の選択肢

| 命名 | 意味 | 推奨 |
|------|------|------|
| `sanitizeFinite()` | 有限値にサニタイズ | 現状（簡潔） |
| `replaceNonFiniteWithZero()` | 非有限値を 0 に置換 | ✅ 推奨（意図が明確） |
| `sanitizeFiniteToZero()` | 有限値にサニタイズ（0 へ） | 中間 |

**推奨**: `replaceNonFiniteWithZero()` にリネーム。将来 Inf → 1.0 等の飽和処理へ変更する場合も対応しやすい。

### 修正案

**ファイル**: `src/DspNumericPolicy.h`

```cpp
// ─────────────────────────────────────────────────────────────────
// 非有限値（NaN/Inf）を 0.0 に置換するヘルパー関数
// 前提: IEEE754 binary32/binary64
// ─────────────────────────────────────────────────────────────────

static_assert(std::numeric_limits<double>::is_iec559, "IEEE754 binary64 前提");
static_assert(std::numeric_limits<float>::is_iec559, "IEEE754 binary32 前提");

inline double replaceNonFiniteWithZero(double x) noexcept
{
#if JUCE_DEBUG || defined(_DEBUG)
    // Debug: NaN/Inf 発生をアサーションで検出。アサーション後も安全側（0.0）へフォールバック
    jassert(std::isfinite(x));
#endif
    constexpr uint64_t kExpMask = 0x7FF0000000000000ULL;
    const uint64_t bits = std::bit_cast<uint64_t>(x);
    const bool isNanOrInf = ((bits & kExpMask) == 0x7FF0000000000000ULL);
    return isNanOrInf ? 0.0 : x;
}

inline float replaceNonFiniteWithZero(float x) noexcept
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

### メリット

- 「0 に置換」という仕様が関数名から明白
- 将来飽和処理（Inf → 1.0）へ変更する場合、関数名を変更するだけで対応可能
- コードレビュー時に意図が即座に理解できる

---

## Bug C — pushBlock 境界チェック

変更なし。

---

## Bug 2 — bypass delayBuffer null 時未クリア

変更なし。

---

## Bug H — StereoConvolver::init 例外安全性（v12: commit 順序改善）

### v11 の問題

```cpp
destroyNUCConvolver(nucConvolvers[0]);  // 旧状態破棄
// ...
latency = nucConvolvers[0]->getLatency();  // 新 NUC から取得
```

`getLatency()` が将来例外を投げる場合、旧状態は既に破棄済み。

### v12 の修正案: getLatency を commit 前に取得

```cpp
// ============================================================
// Phase 2: 全成功 — メンバーを一括更新（commit）
// ============================================================

// ★ v12: getLatency は commit 前に取得（将来変更への耐性）
const int newLatency = newNuc0->getLatency();

// 既存リソースを解放
destroyNUCConvolver(nucConvolvers[0]);
destroyNUCConvolver(nucConvolvers[1]);
irData[0].reset();
irData[1].reset();

// 一括コミット
irData[0] = std::move(newIrL);
irData[1] = std::move(newIrR);
nucConvolvers[0] = newNuc0.release();
nucConvolvers[1] = newNuc1.release();

irDataLength = length;
this->irLatency = peakDelay;
latency = newLatency;  // ★ 既に取得済みの値を使用
// ... 残りのメンバー更新 ...
```

### commit 順序（v12）

```
1. newLatency = newNuc0->getLatency()  ← 旧状態破棄前に取得
2. destroy old
3. move new
4. メンバー代入（newLatency を使用）
```

これにより、将来 `getLatency()` が例外を投げるようになっても安全。

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

### テストケース（v12 修正）

| テストケース | 期待結果 | 根拠 |
|-------------|----------|------|
| `irL == nullptr` | **API 契約違反として適切に失敗する** | `SetImpulse` の仕様に依存（assert / throw / false / access violation のいずれか） |
| `irR == nullptr` | **API 契約違反として適切に失敗する** | 同上 |

---

## 承認

| 役割 | 名前 | 日付 | 署名 |
|------|------|------|------|
| 設計者 | | | |
| レビュアー | | | |
| 承認者 | | | |
