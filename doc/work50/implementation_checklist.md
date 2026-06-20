# P0-P3 実装チェックリスト

**作成日**: 2026-06-20
**元文書**: `bug_review_verification_report.md`
**ステータス**: 全件完了

---

## 凡例

| 記号 | 意味 |
|------|------|
| ✅ | 完了 |
| 🔧 | 作業中 |
| ⏳ | 未着手 |
| ❌ | 保留/中断 |

---

## P0（即日修正）— 確定バグ

| # | 項目 | ファイル | 内容 | 状況 | 確認日 |
|---|------|---------|------|------|--------|
| 1 | IRCacheKey::operator< SWO違反 | `src/ConvolverProcessor.h` | `std::abs()` 許容誤差比較を削除し、直接比較に修正 | ✅ 完了 | 2026-06-20 |

**変更内容**:

```diff
- if (std::abs(f1 - other.f1) > 1.0e-6f) return f1 < other.f1;
- if (std::abs(f2 - other.f2) > 1.0e-6f) return f2 < other.f2;
- if (std::abs(tau - other.tau) > 1.0e-6f) return tau < other.tau;
+ if (f1 != other.f1) return f1 < other.f1;
+ if (f2 != other.f2) return f2 < other.f2;
+ if (tau != other.tau) return tau < other.tau;
```

**検証**: `std::map<IRCacheKey, CacheEntry>` のコンパレータとして Strict Weak Ordering を満たすことを確認。

---

## P2（次回リリース）— 実装不整合

| # | 項目 | ファイル | 内容 | 状況 | 確認日 |
|---|------|---------|------|------|--------|
| 2 | CMA-ES θ正規化 | `src/AllpassDesigner.cpp` | `kThetaMax` を名前空間レベル定数化 + `theta/kThetaMax` に修正 | ✅ 完了 | 2026-06-20 |

**変更内容**:

- `kThetaMax` を `unconstrainedToTheta()` 内部から匿名名前空間レベルに切出し
- 初期平均値計算の正規化を `theta / pi` → `theta / kThetaMax` に修正

---

## P3（改善候補）

| # | 項目 | ファイル | 内容 | 状況 | 確認日 |
|---|------|---------|------|------|--------|
| 3 | DeferredDeletionQueue::reclaim 早期脱出 | `src/DeferredDeletionQueue.h` | 先頭エントリ削除不可時に即座に `break` | ✅ 完了 | 2026-06-20 |
| 4 | ディザ順序（3ファイル） | `src/Fixed15TapNoiseShaper.h`, `src/FixedNoiseShaper.h`, `src/LatticeNoiseShaper.h` | クランプ→ディザ→量子化 の正規順序に修正 | ✅ 完了 | 2026-06-20 |

### No.3: reclaim 最適化

**変更内容**: `DeferredDeletionQueue::reclaim()` の else 分岐に早期脱出を追加

```diff
  } else {
+     // ★ 先頭エントリが削除不可の場合、後続も削除不可（FIFO順序）のため即座に脱出
+     if (!canDelete)
+         break;
      if (scanPos - deqPos > static_cast<uint32_t>(kMaxScan)) {
```

**効果**: Reader stall 時に最大1024件の無駄なスキャンを防止。Message/Timer Thread の CPU 負荷低減。

### No.4: ディザ順序修正

**対象3ファイル**: `Fixed15TapNoiseShaper.h`, `FixedNoiseShaper.h`, `LatticeNoiseShaper.h`

**変更内容**: `quantize()` 内の処理順序を以下のとおり修正

```diff
- // ① ディザ加算 → ② クランプ（旧：非正規順序）
+ // ① クランプ → ② ディザ加算（Lipshitz/Wannamaker 正規順序）
+ if (v < minV) v = minV;
+ else if (v > maxV) v = maxV;
+ // TPDF dither
  const double u1 = uniform(rng);
  const double u2 = uniform(rng);
  v += (u1 + u2 - 1.0) * scale;
- if (v < minV) v = minV;
- else if (v > maxV) v = maxV;
```

**理論的根拠**: Lipshitz, Wannamaker, 「Theory of Dithered Quantization」(1992) に基づく。クランプを先に行うことで、ディザノイズの確率分布がクリップ境界で非対称に歪むのを防止。実用上の聴感差は極小だが、DSP理論上の正当性を向上。

---

## Architecture Debt（設計遺産・継続監視）

| # | 項目 | 状況 | 備考 |
|---|------|------|------|
| 5 | ConvolverState 経路の整理 | ⏳ 未着手 | `doc/work50/architecture_insight_convolverstate_vs_stereoconvolver.md` を参照 |
| 6 | partitionData デッドコード整理 | ⏳ 未着手 | ConvolverState 整理と合わせて対応 |

---

## サマリー

| 優先度 | 件数 | 完了 | 未着手 |
|--------|------|------|--------|
| P0 | 1 | ✅ 1 | 0 |
| P2 | 1 | ✅ 1 | 0 |
| P3 | 2 | ✅ 2 | 0 |
| Architecture Debt | 2 | 0 | ⏳ 2 (別案件) |
| **合計** | **6** | **4** | **2 (設計整理)** |
