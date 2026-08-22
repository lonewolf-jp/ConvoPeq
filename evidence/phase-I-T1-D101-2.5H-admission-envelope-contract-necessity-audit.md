# Phase I-T1-D101-2.5H — Admission Envelope Contract Necessity / Minimal Contract Audit

> **Verdict: H-B — Necessary contract set identifiable, sufficient derivation OPEN**
> D101-2.5G `τ_res→publish` 有限 bound 不存在を踏まえ、有限 service curve `N_success([t,t+G]) ≤ F(G)` に必要な最小契約集合を設計監査で確定。
> **コード変更・測定・実装・数値化 なし / M/R/R_cap/T2 UNDETERMINED 維持 / A_max=1, T_w=2, max(E_w)=1 observed only**

## 1. H-1 Final Target Fixed (D101-2.5H-1)

```
A_true(I) = successful PublishedDomain admissions in interval I
Target: ∀ finite G: N_success([t,t+G]) ≤ F(G)  candidate: N_success ≤ λG + B
```

- `λG+B` を仮採用せず、別 service curve で十分かを判定対象とする (指示どおり)。
- `A_true` は `publishAndSwap` 成功 (RuntimeWorldAuthority::publish, INV-X4-3/4) のみで増加し、`enqueue/registry/OwnerChannel/IntentQueue/retry/deferred` は increment ではない。

## 2. H-2 Four-Layer Decomposition (D101-2.5H-2)

```
                A_true service curve
                        |
        ┌───────────────┴───────────────┐
        │                               │
Temporal conservation          Admission envelope
 reserve→publish displacement    reservation acquire rate/burst
        │                               │
        └───────────────┬───────────────┘
                        │
            Reservation conservation
              identity / ≤1 / rollback
```

| Layer | 必要内容 | 現状 |
|-------|----------|------|
| Reservation conservation | `∀r: N_success(r) ≤1`, identity, rollback | **Design CLOSED (INV-R1〜R4, D48) / Production OPEN (L2 NO-GO)** |
| Temporal displacement | `t_publish - t_reserve ≤ D` | **OPEN (Gで確定)** |
| Reservation admission envelope | `N_reserve(I) ≤ E(|I|)` | **OPEN (λ_max/burst 未契約)** |
| Publication service curve | `N_success([t,t+G]) ≤ F(G)` | **OPEN (上記3層の合成として未導出)** |

## 3. H-3 Minimality Test #1 — Reservation Conservation (C1)

**C1:** `∀r: N_success(r) ≤ 1`

- I4 INV-R2 は `1 World identity ≤ 1 reservation` を CLOSED (D48) とするが、D101 target は `reservation → successful publication`。
- `1 World ≤1 reservation` だけでは `1 reservation ≤1 publication` は導出不可 — 追加 contract。

**Result:** C1 は追加 contract として必要 (D101-2.5E-2 OPEN を継承)。

## 4. H-4 Minimality Test #2 — Temporal Displacement Fixed D vs Bounded D(G)

Gで `finite D` 不存在が確定。H-4 で `fixed D` vs `window-dependent D(G)` を比較:

| 候補 | 有限性への寄与 | 現状 |
|------|----------------|------|
| fixed `D: t_publish - t_reserve ≤ D` | `N_success([t,t+G]) ≤ N_reservation([t-D,t+G])` への dilation 可能 | OPEN |
| bounded `D(G)` | window依存の dilation で service curve 導出可能 | OPEN |
| deadline / FIFO positional bound / work-conserving curve | いずれも structural invariant 未証明 | OPEN |

**結論:** `D(G)` だけでも導出可能だが、いずれの形も `τ_res→publish` 有限 upper bound が前提 — 現時点でいずれも未証明。

**「Dがないから token bucketが必要」の一段飛ばしは禁止 — 本監査で遵守。**

## 5. H-5 Admission Envelope Generalization (D101-2.5H-5)

| Type | Form | Handling |
|------|------|----------|
| Fixed quota | `N(I) ≤ Q` | temporal window 固定要だが G_sample ≠ G_admission |
| Rate+burst | `N(I) ≤ λ|I| + B` | 標準候補だが λ,B 未契約 |
| **General service curve** | `N(I) ≤ α(|I|)` | **最上位抽象形として採用 — λG+Bは一実装形** |

## 6. H-6 Rollback Conservation (D101-2.5H-6)

```
reserve ├─ build failure → token retained? returned? new required?
        ├─ enqueue failure → same? new?
        ├─ deferred → retain
        ├─ retry → same reservation?
        ├─ publish reject → ?
        └─ publish success → consumed
```

- `reserve → build failure → release` だけでは不十分。
- D69 は reservation acquire/release authority と publication/retirement authority 分離を明示するが、conservation を evidence で閉じるには各 branch の token 状態遷移 (`consumed/retained/returned/new required`) を個別に contract 化する必要がある。

**Result:** OPEN — 各状態の token 遷移未契約。

## 7. H-7 Outstanding Cap Position (D101-2.5H-7)

```
C_outstanding: OutstandingReservation ≤ R  —  CLOSED invariant 候補だが
C_temporal: N_success([t,t+G]) ≤ F(G) —  単独では導出不可
→ R < ∞ は lifecycle/safety invariant であって R→λ / R→B 変換規則ではない (CLOSED として固定)
```

- I4 `R` は retired-but-not-destroyed World identity の reservation (既存 4608 capacity とは独立 authority)。
- `R finite ≠ λ finite ≠ service curve` を明示固定。

## 8. H-8 Sufficient Condition Construction (D101-2.5H-8)

候補最小条件集合:

```
C1  reservation → publication conservation (∀r success(r) ≤1)
C2  temporal displacement envelope (t_publish - t_reserve ≤ D  or  D(G))
C3  reservation admission envelope (N_reserve(I) ≤ α(|I|))
C4  rollback / retry conservation
    ↓
publication envelope: N_success([t,t+G]) ≤ α(G+D)  (一般形例)
    ↓
A_true service curve
```

- 例: `N_reserve(I) ≤ α(|I|)` + `t_publish - t_reserve ≤ D` + `success(r) ≤1` ⇒ `N_success([t,t+G]) ≤ α(G+D)`。
- **数値導入なし** — 候補としての形式導出可能性の検証のみ。

**現状:** C1〜C4 全て OPEN のため publication envelope への合成は検証不能。

## 9. H-9 Necessary vs Sufficient Separation (D101-2.5H-9)

| Contract | 必要性 | 十分性 | 現状 |
|----------|--------|--------|------|
| Reservation identity | 必要 | 不十分 | Design CLOSED / Prod OPEN |
| ≤1 success / reservation | 必要 | 不十分 | OPEN |
| Rollback | 必要 | 不十分 | OPEN |
| Retry identity | 必要 | 不十分 | OPEN |
| Temporal displacement D | 必要 | 不十分 | OPEN |
| Reservation envelope α(G) | 必要 | 不十分 | OPEN |
| Outstanding cap R | 必要 (lifecycle) | 非十分 (temporalへ非変換) | Design CLOSED / temporal boundではない |
| λ+B form | 未決定 (αの一実装) | 単独不十分 | 未決定 |
| General α(G) | 必要 (最上位) | 単独不十分 | 候補 |
| Publication service curve | **target** | **target** | OPEN |

**「必要」と「実装上便利」を分離して監査 — 本表で確定。**

## 10. H-10 T2 Boundary (D101-2.5H-10)

- D101-2.5H: **必要契約を証明するところまで** — 実装するかは決めない。
- Phase I-T2: `held-set / free-stack / token / R gate` を含み (`D69` で T1=telemetryのみ)、ReservationExhausted / R gate を扱う設計 (I4 D48〜D51)。
- **H で implementation GO/NOGO は判定しない** — contract 集合の形式閉鎖のみを対象。

## 11. Final Verdict

| 判定 | 条件 |
|------|------|
| **H-A** | `general α(G) + finite D + conservation` で finite service curve 導出可 |
| **H-B** | 必要集合特定可だが十分性の形式導出まだ不可 |
| H-C | conservation 自体が破綻 |

**本監査: H-B — 必要条件 C1〜C4 + outstanding cap の役割分離 + α(G) 最上位抽象化は特定できたが、`A_true` service curve への十分性導出は C1〜C4 全 OPEN のため未到達。**

## 12. Current Fixed State

```
D101 #1       CLOSED (D101-1R2)
D101-2.5C     OPEN (service-curve 不在再確認)
D101-2.5D     OPEN (admission envelope に conservation/rollback/burst 追加要)
D101-2.5E     OPEN (Case C)
D101-2.5F     Design CLOSED / Production OPEN
D101-2.5G     OPEN (finite D なし)
D101-2.5H     PARTIAL/H-B (minimal contract set 特定、十分性導出 OPEN)
D101-2.6R     PARTIAL (E_ref=0, E_sample OPEN)
A_max_candidate = 1     observed only
T_w             = 2     observed only
max(E_w)        = 1     observed only
M               = UNDETERMINED
R               = UNDETERMINED
R_cap           = UNDETERMINED
B_max^true      = UNDETERMINED
T2              = NO-GO
```

**次:** D101-2.5H で最小契約集合が形式的に閉じた場合のみ `D101-2.5I — Minimal Contract → Authority Placement Audit` へ (どの authority が admission envelope / temporal deadline / reservation identity を所有するか、`Authority Singularization` との非矛盾で決定)。現時点では H-B のため次 Gate へは進まず、minimal contract の証明可能性を恒常登録。

## 13. Prohibitions Maintained

`× コード変更 × Reservation 実装 × R gate 実装 × Token bucket 実装 × admission limiter 実装 × harness変更 × 追加測定 × R数値決定 × A_max=1昇格 × T_w=2昇格 × max(E_w)=1昇格 × R→λ短絡 × R→B_max短絡 × λG+B仮採用 × T2 GO` — **全て未実施**

Production truth は **2026-08-21版 ConvoPeq.md**、design/production refinement 分離 (F/G 二層モデル) 維持。

## 14. Tool Coverage

| 系統 | ツール | 結果 |
|------|--------|------|
| WSL | rg 15.1.0 / sg 0.44.0 / fdfind 10.3.0 / ag 2.2.0 / fzf 0.67.0 / sed 4.9 / awk 5.3.2 | 全 census 一致 |
| Sandbox | node fs walk / source直読 | enumeration 正確性確定 |
| MCP | serena 一時無効 (代替 rg/sg) | — |
| MCP | AiDex 一時無効 (代替) | — |
| CLI | ccc 0.45.2 / graphify 0.9.48 / semble 0.5.5 | 検索実施 |
| 文献 | crossbeam-epoch / rigtorp MPMCQueue / serena / cocoindex / ast-grep / semble / AiDex / graphify / headroom | 前段までに 9/10 200 OK 済 |

*Evidence generated: Phase I-T1-D101-2.5H — Minimal contract necessity audit. H-B: necessary set identified, sufficient derivation OPEN.*
