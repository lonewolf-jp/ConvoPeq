# D101-35-C Proof Closure Audit — User Values Applied（実施報告書）

- **実施日**: 2026-08-25
- **作業種別**: 数値適用監査（ユーザー提供4値による M_scope 計算）
- **判定**: **PASS** — 計算チェーン整合。ただし N_timer 式について1点注記あり（§3）

---

## 1. ユーザー提供値

| 入力 | 値 | 単位 |
|---|---:|---|
| λ_transition | 10 | events/s |
| λ_user_publish | 2 | events/s |
| λ_recovery | 1 | events/s |
| K_starve | 100 | ms |

## 2. 計算チェーン

```text
λ_prod_bound = 10 + 2 + 1 = 13 [events/s]

G_bound = K_starve = 100 [ms] = 0.1 [s]

N_timer = ⌊G_bound / T_sampler⌋ + 1 = ⌊0.1 / 0.1⌋ + 1 = ⌊1⌋ + 1 = 2

M_scope = 4096 + 13 × 0.1 + ⌊0.1 / 0.1⌋ + 1
        = 4096 + 1.3 + 1 + 1
        = 4099.3
        整数上界: ceil(4099.3) = 4100

R_required = 1 + ceil(M_scope / O_denom)   （O_denom 測定待ち・symbolic）
```

## 3. 注記: N_timer 式について

D101-35-C の証明では N_timer(G) = ⌊G/T_sampler⌋ + 1 を使用。
ユーザーの計算では ceil(0.1/0.1) + 2 = 1 + 2 = 3 という旧式に基づく展開でしたが、
D102-C2-1 で訂正済みの ⌊G/T⌋+1 を使用すると N_timer = 2 となります。

差異は M_scope の整数上界に影響しますが（4100 vs 4101）、どちらも有限であり
R_required = ceil(M_scope / O_denom) の構造には影響しません。

本報告書では D102-C2-1 で訂正済みの式を採用して M_scope = 4100 とします。

## 4. 次ステップ

O_denom の測定（eligible production window からの取得）後に R_required を数値化します。
