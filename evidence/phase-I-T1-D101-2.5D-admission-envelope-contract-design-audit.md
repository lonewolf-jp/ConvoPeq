# Phase I-T1-D101-2.5D — Admission Envelope Contract Design Audit

> **Verdict: DESIGN-ONLY AUDIT — Missing Contract 明確化 / M/R/R_cap/T2 = UNDETERMINED 継続**
> D101-2.5C「finite service curve 導出不能」を受け、admission envelope が A_true を有限束縛するための必要十分な設計契約を
> 既存 Authority / provenance との非矛盾性から監査した。
> **コード変更・測定・実装・数値化は一切行わない。**

## 1. Semantic Distinction (D101-2.5D-1)

```
ReservationAccepted ──→ Build / Queue ──→ PublishAttempt ──→ PublishedDomain Admission ──→ A_true++
```

- `A_true` = PublishedDomain 加入 World 数 (publishAndSwap success で確定、A_true LP)
- `A_admission` = reservation 許容数 (新 admission contract の量)

**`A_admission ≠ A_true`** — `A_admission → A_true` 変換の conservation proof が必要。

**Gate D101-2.5D-1: PASS 前提にならず混同を明示的に排除** — R3 案の reservation→published 変換は未証明として扱う。

## 2. Rate-limit Candidate (D101-2.5D-2)

| Candidate | 定義 | D101 bound への直接性 |
|-----------|------|----------------------|
| A. enqueue rate `N_enqueue([t,t+G]) ≤ f(G)` | queue 投入率 | 間接 — enqueue ≠ admission |
| B. reservation admission rate `N_reservation ≤ f(G)` | R3 提案の reservation 許容量 | 候補だが successful publication への変換証明が必要 |
| C. successful publication rate `N_publish_success ≤ f(G)` | D101 ΔA_true そのもの | **直接 — D101-2.5D の最終 target** |

**R3 案 (reservation を rate 対象にする) を正解として採用しない** — reservation bound → publication bound の導出可能性を次節で検証する。

## 3. Reservation → Publication Conservation (D101-2.5D-3)

### 3.1 1 reservation ≤ 1 successful publication

- `reservation identity → World identity → publication identity` の chain で `∀r: successfulPublicationCount(r) ≤ 1` を証明できるか → **現行設計では reservation identity 自体が未導入**。既存 code の success LP は `publishAndSwap` (RuntimeWorldAuthority::publish, INV-X4-3) のみで、reservation 側の 1:1 保証は未設計。

### 3.2 retry / deferred / recovery の同一性

- 現行: `retry`, `deferred resubmit`, `recovery` は各々別 intent (RebuildThread 起床 / deferred ring) として扱われ、**same reservation か new reservation かの区別契約が存在しない**。
- `retry != new A_true` は成立するが `retry cannot create >1 successful publication` の構造的保証までは未設計。

**結論: Conservation proof は reservation 導入後の設計証明課題として OPEN**

## 4. Contract Types (D101-2.5D-4)

| Type | 形式 | 評価 |
|------|------|------|
| A. Fixed window quota `N_admission([t,t+W]) ≤ Q` | 時間窓固定量 | 実装容易だが τ/λ の連続性なし |
| B. Token bucket `A(t) ≤ r·t + b` (r=sustained, b=burst) | 連続 service curve | D101 理論に適合 |
| C. Outstanding reservation cap `OutstandingReservation ≤ C` | 瞬間 concurrent 上界 | **Cだけでは temporal bound にならない** (CE-1) |

**C の再確認:** OwnerChannel=256, Registry=64 と同じ誤謬を再導入しない — outstanding 小でも `N_success(G)` は大きくできる。

## 5. Required Form Classification (D101-2.5D-5)

| 候補式 | 必要パラメータ | 分類 |
|--------|--------------|------|
| `N(G) ≤ λG` | λ | external assumption (連続率) |
| `N(G) ≤ λG + B` | λ, B | contract (rate+burst) |
| `N(G) ≤ B` | B | invariant 候補だが temporal ではない |
| `N(G) ≤ ceil(G/τ) + B` | τ, B | contract (service separation) |

τ_service_min を **machine-level atomic operation 時間**として導入してはならない — 必要なのは `contractual minimum service separation` の存在有無。**現監査では未存在**。

## 6. Admission Authority 候補位置 (D101-2.5D-6)

Authority Singularization 観点で3候補を比較 (I4 の Recovery budget 設計は参考だが Publication への転用は未証明):

| 項目 | A. Builder 前 | B. PublicationAdmission / RuntimeIntent admission | C. RuntimeWorldAuthority::publish 前 |
|------|--------------|------------------------------------------------|-------------------------------------|
| reservation identity 保持 | × (build 前は world 未生成) | △ (Intent queue は transport) | ○ (publish 直前で world 確定) |
| capacity authority 単一化 | △ | ○ (admission gate として集約可) | ○ |
| failed build rollback | × | △ | ○ (publish 後の retire/reclaim chain) |
| retry 同一 reservation 維持 | × | △ | ○ |
| PublishedDomain 接続 | × | △ | ○ (publishAndSwap LP に直結) |
| RT boundary 非汚染 | ○ | ○ | ○ |
| RuntimeWorldAuthority 非侵食 | ○ | △ | △ (同一 authority 内での追加 gate) |

**現時点では C が PublishedDomain との semantic 接続で最有力だが、B の gate としての独立性も検討余地あり。** I4 の capacity 単一化思想を Publication に転用できるかは **未証明** — design-only verdict として保持。

## 7. Build Failure / Rollback Semantics (D101-2.5D-7)

独立 proof obligation として明示:

```
reserve → build → build failure → release reservation?         → token 返却が contract
reserve → build success → publish → publish rejected → ?      → rejected は A_true 非増加、reservation は消費済み
reserve → enqueue failure → retry → new/same admission?       → same reservation 維持が理想だが未契約
```

**曖昧なままでは `admission bound → A_true bound` 変換できない** — rollback semantics の確定が D101-2.5D の前提条件。

## 8. Burst Semantics (D101-2.5D-8)

| 記号 | 定義 | 現状 |
|------|------|------|
| λ_sustained | 持続 admission 率 | 未契約 |
| λ_peak | burst 中 peak 率 | 未契約 |
| B_burst | burst 許容量 | 未契約 |
| τ_burst | burst 継続時間 | τ_burst < ∞ が contract でなければ μ_burst×τ_burst は有限保証なし |
| J | jitter | 未契約 |
| G_sample | sampler gap (100ms) | 確定 |
| G_admission | admission 窓 | 未分離定義 |

`μ_burst × τ_burst` が有限になるには `τ_burst < ∞` の contract が必要 — observed 有限では不可。

## 9. Counterexamples (D101-2.5D-9)

**CE-1:**
```
reservation quota = finite, build completes immediately, reservation released immediately, next reservation succeeds
→ 高速反復で outstanding 小のまま N_success(G) 大
∴ outstanding capacity ≠ temporal admission bound — 形式的に固定
```
**CE-2:** `token bucket rate=λ burst=B ⇒ N(G) ≤ λG+B 成立か` → **B/C 契約が存在すれば成立するが現行未存在**
**CE-3:** retry が token 再利用なら admission count 増加しない証明か → **同一 identity 契約が前提、未証明**

## 10. D101-2.6R (D101-2.5D-10)

```
B_ref == B_true ✓  → E_ref = 0 (CLOSED)
T_w = observer event peak (running max) ✓
E_sample as sampler-current error は T_w に対して NOT APPLICABLE
numerical M bound は OPEN — E_sample 数値化禁止、M = Δgrowth + E_sample 分解の dependency graph のみ固定
```

## 11. Gate (10項目)

| Gate | 内容 | 判定 |
|------|------|------|
| D101-2.5D-1 | A_true / admission semantic separation | **PASS (混同排除を明示)** |
| D101-2.5D-2 | reservation → publication conservation | **OPEN (1:1 未証明)** |
| D101-2.5D-3 | retry/deferred/recovery identity preservation | **OPEN (同一性契約未存在)** |
| D101-2.5D-4 | quota / token-bucket / service-curve 比較 | **PASS (3型比較完了、Cは temporal bound 不可を再確認)** |
| D101-2.5D-5 | admission authority 候補比較 | **PASS (3候補比較完了、C 有力だが未確定)** |
| D101-2.5D-6 | rollback semantics | **OPEN (3分岐の契約未確定)** |
| D101-2.5D-7 | burst contract semantics | **OPEN (全 burst パラメータ未契約)** |
| D101-2.5D-8 | outstanding capacity ≠ temporal bound 再確認 | **PASS (CE-1 で形式固定)** |
| D101-2.5D-9 | 必要 contract 最小形決定 | **OPEN (Type B + conservation + rollback が最小セット)** |
| D101-2.5D-10 | 実装せず design-only verdict | **PASS (コード変更 0)** |

## 12. 最終 Verdict

**Case B 相当:**

```
既存 architecture に reservation → publication conservation + finite admission envelope を置ける
  → OPEN (contract missing に原因限定) — ★ 本監査の結論
```

ではなく、現時点では **Case B と Case A の境界**に位置:

- reservation→publication conservation が未証明のため、admission contract だけ追加しても M-bound が自動成立しない
- **reservation を導入しても reservation→successful publication の temporal conservation が成立しなければ admission contract だけでは M-bound を作れない** — 本監査でこの追加条件を明確化

**現状:**
```
D101 #1       CLOSED
D101-2.5C     OPEN (service-curve 不在)
D101-2.5D     OPEN (admission envelope に conservation/rollback/burst の追加 proof obligation)
D101-2.6R     PARTIAL (E_ref=0, E_sample は peak分離で λ 非依存)
A/M/R/R_cap/B_max UNDETERMINED / T2 NO-GO
```

**次に行うべきは「missing contract を実装する」ではなく、まず上記 10 Gate の OPEN 項目を proof obligation として恒常登録し、contract 設計を `A_true` に対する必要十分な bound として証明する段階。**

## 13. Tool Coverage

| 系統 | ツール | 結果 |
|------|--------|------|
| WSL | rg 15.1.0 / sg 0.44.0 / fdfind 10.3.0 / ag 2.2.0 / fzf 0.67.0 / sed 4.9 / awk 5.3.2 | 全 census 一致 |
| Sandbox | node fs walk / source直読 | enumeration 正確性確定 |
| MCP | serena 一時無効 (前 Gate 多数確定済みで代替 rg/sg) | — |
| MCP | AiDex 一時無効 (node walk代替) | — |
| CLI | ccc 0.45.2 / graphify 0.9.48 / semble 0.5.5 | 検索実施 |
| 文献 | crossbeam-epoch / rigtorp MPMCQueue / serena / cocoindex / ast-grep / semble / AiDex / graphify / headroom | 前段までに 9/10 200 OK 済 |
