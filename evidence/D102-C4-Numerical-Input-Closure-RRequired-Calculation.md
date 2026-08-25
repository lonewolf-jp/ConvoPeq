# D102-C4 — Numerical Input Closure & R_required Calculation（実施報告書）

- **実施日**: 2026-08-25
- **作業種別**: read-only 数値適用監査（ソースコード変更 **0** / 契約変更 **0** /
  λ·G·O_denom 具体値の**独自仮定採用 0**）
- **判定**: **Numerical closure = INCOMPLETE**
  （λ_prod_bound / G_bound の workload・environment contract 値がユーザー未決定のため。
   O_denom も実稼働測定データが未蓄積。構造的 compatibility は D102-C3 で PASS 済み。）
- **基準**: ConvoPeq.md **21:13 再生成版**（ローカル実ソースから）

---

## 1. λ_prod_bound

| 項目 | 状態 |
|---|---|
| value | **PENDING_USER_INPUT** |
| unit | obligations/second |
| source | workload contract（運用仕様） |
| status | 🔲 未決定 — ユーザー/製品設計者が宣言する必要あり |

### 構成要素（D102-C2-2D §2 確定済み inventory）

```text
λ_prod_bound = λ_transition + λ_user_publish + λ_recovery

定常状態での加算項:
  λ_transition   = workload contract TBD
  λ_user_publish = workload contract TBD
  λ_recovery     = workload contract TBD

lifecycle-bound producers（P-1/P-2/P-5）は定常状態で発生しないため加算項 = 0
Timer #5 は N_timer 項に分離済み（λ スコープ外）
```

---

## 2. G_bound

| 項目 | 状態 |
|---|---|
| value | **PENDING_USER_INPUT** |
| unit | seconds |
| decomposition | K_starve + T_sampler + δ_processing（δ_processing ≤ K_starve により subsumed） |
| 確定式 | G_bound = K_starve + T_sampler |
| status | 🔲 K_starve 値のユーザー決定待ち |

### 分解詳細

```text
K_starve       = environment contract TBD（例示 5s の昇格禁止遵守）
T_sampler      = 100 ms = 0.1 s（source constant: startTimer(100) Init.cpp:121）
δ_processing   = K_starve premise に包含（独立項廃止・D102-C2-2D Gate 3 確定）
```

---

## 3. O_denom

| 項目 | 状態 |
|---|---|
| campaign | ❌ 未実施（production deploy 後に蓄積開始） |
| window count | N/A |
| per-window maxima | N/A |
| campaign maximum | N/A |
| status | 🔲 **measurement required** |

### 取得 protocol（確定済み・D102-C2-2A §Gate C2-2A-5）

```text
取得経路: telemetry.lastClosedSnapshot().snap.windowMax
eligibility: post-first-commit ∧ wrap-free ∧ 定常含有 ∧ 同一 campaign
O_denom = max(eligible windows' snap.windowMax)
```

⚠️ `lastClosedSnapshot()` は単一 window の値のみ返す。campaign 全体最大値には
caller 側集約が必要（現行コードに集約機構なし → 将来インフラ強化候補）。

---

## 4. M_scope 計算

| 項目 | 状態 |
|---|---|
| baseline | 4096（K = kIntentQueueCapacity、source constant） |
| λ × G | **計算不可**（λ_prod_bound と G_bound が共に PENDING_USER_INPUT） |
| N_timer | ⌊G_bound/T_sampler⌋ + 1（G_bound 確定後に計算可能だが G_bound 自体が TBD） |
| other terms | M_boundary = 0（吸収済み）/ M_jitter ⊆ M_gap（吸収済み） |
| total | **PENDING_USER_INPUT** |
| status | 🔲 λ_prod_bound / G_bound 値決定後に数値化 |

---

## 5. R_required 計算

| 項目 | 状態 |
|---|---|
| formula | `R_required = 1 + ceil(M_scope / O_denom)` ✅ 確定 |
| value | **PENDING_USER_INPUT**（M_scope / O_denom 共に未確定のため） |
| precondition | O_denom > 0 ✅ bootstrap invariant により保証 |

---

## 6. R_cap 二層評価

### 固定 bounded storage

| storage | capacity | source |
|---|---:|---|
| D (DeferredDeletionQueue) | 4096 | kQueueSize, DDQ.h:262 |
| Q (RetireQuarantineStore) | 512 | kMaxQuarantinedEntries, Store.h:65 |
| E (EmergencyQ) | 512 | 同一クラス型別 instance |
| **fixed bounded subtotal** | **5120** | D+Q+E |

### Terminal (TerminalReclaimAuthority)

| 特性 | 状態 |
|---|---|
| storage 型 | growable std::vector<Entry> |
| 固定 capacity ceiling | **不存在**（entries_.push_back 常時成功） |
| 実効メモリ上限 | **未評価**（OS メモリ制約依存） |

### structural interpretation

| 判定 | 意味 |
|---|---|
| R_required ≤ 5120 | bounded stores 内で完結 — Terminal に依存せず収容可能 |
| R_required > 5120 | Terminal capacity に依存 — structural safety は成立するが finite-memory operational bound は未閉鎖 |
| Terminal 固定上限なし | **構造上の固定 capacity failure は証明されない** ✅ |
| 実効メモリ上限 | **未評価** — 明示的に区別を維持 |

---

## 7. Numerical compatibility 判定

```text
Case A: R_required ≤ 5120
    → bounded stores 内で成立
    → 判定不能（M_scope/R_required 数値未確定のため）

Case B: R_required > 5120
    → Terminal 依存
    → structural safety 成立（growable vector）
    → finite-memory operational bound 未閉鎖
    → 判定不能

Case C: O_denom が未測定
    → NO-GO / Numerical closure incomplete ← ★現状

Case D: λ_prod_bound または G_bound が未確定
    → NO-GO / Numerical closure incomplete ← ★現状
```

# VERDICT: **Numerical closure = INCOMPLETE**

（C と D の両方が理由。λ/G 値決定 + O_denom 測定後に初めて判定可能。）

---

## 8. Phase I implication

| 判定 | 条件 |
|---|---|
| GO | λ_prod_bound / G_bound / O_denom の全値確定後 |
| NO-GO | **現状維持**（数値未確定のため） |

Phase I NO-GO 解除の阻害要因は R_cap ではなく **λ_prod_bound / G_bound / O_denom の
3入力の値決定**にあることを本監査で再確認。

---

## 9. ユーザーへの意思決定要求事項

D102-C4 を完了させるために以下のユーザー意思決定が必要です:

| # | 決定事項 | 影響範囲 |
|---|---|---|
| U-1 | λ_transition の workload contract 値 | M_scope の burst 項 |
| U-2 | λ_user_publish の workload contract 値 | M_scope の publish 要求項 |
| U-3 | λ_recovery の workload contract 値 | M_scope の recovery 項 |
| U-4 | K_starve の environment premise 値（bounded-starvation contract 承認） | G_bound / M_gap 項 |
| U-5 | O_denom 測定期間の選択（eligible production campaign の指定） | R_required 分母 |
