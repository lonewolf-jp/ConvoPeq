# D102-C4 — Numerical Parameter Determination & Capacity Compatibility Audit（実施報告書）

- **実施日**: 2026-08-25
- **作業種別**: read-only audit（ソースコード変更 **0** / 契約変更 **0** / 数値の恣意的採用 **0**）
- **判定**: **CONDITIONAL PASS** — λ_prod_bound / G_bound の値はユーザーが D101-35-C closure で
  提示した「採用候補」を使用。O_denom は実測データ未蓄積のため PENDING。
- **基準**: ConvoPeq.md **2026-08-25 22:11 再生成版**

---

## 1. λ_prod_bound 確定

### semantic definition 再確認

D101-35-A §Gate C2-2A-2 確定:
> λ_prod_bound ≜ 全 publication producer の submit レート上界（events/s）
> observed rate ≠ safe bound（測定値の昇格禁止）

### ユーザー提示候補値（D101-35-C closure 監査）

| 項目 | 値 | basis |
|---|---:|---|
| λ_transition | 10 | workload contract |
| λ_user_publish | 2 | workload contract |
| λ_recovery | 1 | workload contract |

```text
λ_prod_bound = 10 + 2 + 1 = 13 [events/s]
```

### source-observable 確認

✅ CoordinatorLoop 直列化により acquire は排他的処理。
Σ(個別 workload 上界) ≥ 合成レート は保守的正しい上界。

### 判定

| Status | 値 |
|---|---|
| **CONTRACT (user-proposed)** | **13 events/s** |

⚠️ 正式採用にはユーザー承認が必要（現状「採用候補」ラベル）。

---

## 2. G_bound 確定

### semantic definition 再確認

D101-35-A §Gate C2-2A-3 確定:
> G ≜ event 発生 → 次回 samplerTick 反映までの最大遅延
> G ≠ T_sampler（nominal cadence との混同禁止）

### ユーザー提示候補値

| 項目 | 値 |
|---|---:|
| K_starve | 1.0 s |
| G_bound = K_starve 直接採用 | **1.0 s = 1,000,000 µs** |

### unit 統一

```text
G_bound = 1,000,000 [µs] = 1.0 [s]
T_sampler = 100,000 [µs] = 0.1 [s]
```

### 判定

| Status | 値 |
|---|---|
| **CONTRACT (user-proposed)** | **1.0 s = 1,000,000 µs** |

⚠️ K_starve と同様、正式採用にはユーザー承認が必要。

---

## 3. N_timer(G_bound) 計算

```text
N_timer(G_bound) = ⌊G_bound / T_sampler⌋ + 1
                 = ⌊1,000,000µs / 100,000µs⌋ + 1
                 = ⌊10⌋ + 1
                 = 11
```

| Status | 値 |
|---|---|
| **DERIVED** | **11** |

---

## 4. M_scope 数値確定

```text
M_scope = K + λ_prod_bound × G_bound + N_timer(G_bound)
        = 4096 + 13 × 1.0 + 11
        = 4096 + 13 + 11
        = 4120
```

| Status | 値 |
|---|---|
| **DERIVED** | **4120** |

単位整合確認:
```
λ_prod_bound [events/s] × G_bound [s] = [events]（無次元カウント）
K [count] + λ×G [count] + N_timer [count] → 全項 count 型 ✅
```

丸め規則: 全項整数のため丸め不要。M_scope = 4120（正確な整数値）。

---

## 5. O_denom 測定可能性監査

### 5.1 定義

```text
O_denom ≜ eligible measurement windows 内の
         observedOutstandingMax (= windowMax) の最大値
```

### 5.2 取得経路の実装事実

```cpp
telemetry.lastClosedSnapshot()
    → snap.windowMax     // 単一 Closed window の sampled maximum
```

### 5.3 campaign 全体集約の現状

| 項目 | 状態 |
|---|---|
| 単一 window の snap.windowMax 取得 | ✅ 実装済み |
| campaign 全体の max 集約機構 | ❌ **未実装**（caller 側で必要） |
| production telemetry 蓄積 | ❌ 未開始（deploy 後に開始） |

### 5.4 O_denom > 0 の構造的保証

```text
bootstrap invariant（Init.cpp:85-90 committed=true → didPublishRuntimeNonRt
    → onRuntimePublishedNonRt → Commit.cpp:406 onAcquireObserved()）
    → A ≥ 1 以降 R < A（resident release 未発生のため）
    ∴ est ≥ 1 at every post-first-commit tick
    ∴ O_denom ≥ 1 構造的に保証 ✅
```

### 5.5 判定

| Status | 値 |
|---|---|
| **MEASUREMENT REQUIRED — PENDING** | プロトコル確定済み・実データ未蓄積 |

---

## 6. K_min 計算

```text
K_min = ceil(M_scope / O_denom)

M_scope = 4120 確定
O_denom = TBD（測定待ち）

∴ K_min も TBD（O_denom 確定後に計算可能）
   下限: O_denom ≥ 1 より K_min ≤ 4120
```

| Status | 値 |
|---|---|
| **DERIVED (symbolic)** | `ceil(4120 / O_denom)` — O_denom 確定後に数値化 |

---

## 7. R_required vs R_cap compatibility

### symbolic 比較式

```text
R_cap × O_denom ≥ O_denom + M_scope
R_cap ≥ 1 + ceil(M_scope / O_denom)
```

### bounded storages のみで評価する場合

```text
bounded subtotal = D(4096) + Q(512) + E(512) = 5120

R_required ≤ 5120 ⟺ bounded storage 内で完結
R_required > 5120 ⟳ Terminal capacity 依存（growable・上限なし）
```

### Terminal growable の意味

TerminalReclaimAuthority は `std::vector` のため固定 capacity ceiling が存在しない。
よって「capacity 枯渇 → obligation 消失」の経路は構造的に遮断されている。
ただし、実効メモリ上限による制約は存在するため、
**structural safety（無制限）と finite-memory operational safety を区別して評価する必要がある**
（C3 §8 の指摘どおり）。

---

## 8. Numerical compatibility 判定

| Case | 条件 | 判定 |
|---|---|---|
| A | R_required ≤ 5120 | 判定不能（λ/G/O_denom 値未確定） |
| B | R_required > 5120 | 同上 |
| C | O_denom 未測定 | ← ★**現状** |
| D | λ_prod_bound / G_bound 未確定 | ← ★**現状** |

# VERDICT: **Numerical compatibility = PENDING**

（λ/G/O_denom の値決定 + O_denom 実測後に初めて判定可能）

---

## 9. パラメータサマリテーブル

| Parameter | Value | Unit | Source | Status |
|---|---:|---|---|---|
| λ_prod_bound | 13 | outstanding/s | user-proposed workload contract | **CONTRACT (要承認)** |
| G_bound | 1,000,000 | µs (1.0 s) | user-proposed environment contract | **CONTRACT (要承認)** |
| N_timer | 11 | count | derived from G_bound/T_sampler | DERIVED ✅ |
| M_scope | 4120 | count | derived | DERIVED ✅ |
| O_denom | TBD | count | eligible campaign measurement required | MEASUREMENT REQUIRED 🔲 |
| K_min | ceil(4120/O_denom) | count | derived | PENDING O_denom |
| R_required | 1 + ceil(4120/O_denom) | reservation | derived | PENDING O_denom |
| R_cap | 5120+ (bounded) / unbounded (T) | reservation | structural | STRUCTURAL ✅ |

---

## 10. VERDICT

# D102-C4 = **PASS**（全パラメータの basis・決定権者・依存関係が閉じた）

**Numerical compatibility = PENDING**（値決定後に判定）

---

## 11. 次ステップ: D102-C2-2 — 数値適用

ユーザーが以下を承認した場合:

```text
λ_prod_bound = 13 events/s（workload contract 承認）
G_bound      = 1.0 s（environment premise 承認）
K_starve     = 1.0 s（Timer cadence 一致根拠付き）
```

→ M_scope = 4120 / R_required symbolic 確定 → O_denom 測定期間選択 → 数値確定。
