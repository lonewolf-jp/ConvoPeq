# Phase I-T1-D101-2.5F — Design-to-Production Reservation Refinement Audit

> **Verdict: Provenance 是正完了 — D101-2.5F 全 Gate CLOSED に向けて収束**
> D101-2.5E の「reservation 未導入」表現を是正し、**Design-level CLOSED / Production L2 NO-GO** の二層モデルを確定。
> **コード変更・測定・実装なし / M/R/R_cap/T2 = UNDETERMINED 維持**

## 1. F-1 Design / Production 二層モデル固定

| Layer | 意味 | 現状 |
|-------|------|------|
| L0 | `WorldRetirementReservation` design contract | **DEFINED** (I4 D48/D52) |
| L1 | INV-R1〜R4 design proof | **CLOSED** (I4 D48: acquire=publish()内swap前, release=deferred-delete deleter内, 1 World≤1 reservation, 全path同一authority, context lifetime INV-R6確定) |
| L2 | production code に reservation acquire/release 実装 | **NO-GO / 未実装** (src/ に `WorldRetirementReservation` 型 0件, acquire/release API 0件) |

**是正:** 「reservation identity が未設計」ではなく「**reservation identity は design-level で定義済み。production implementation は未実施** (D53/D69 Phase I implementation = NO-GO)」と修正。一時矛盾を解消。

## 2. F-2 D101-2.5E Level 0/1 再判定

| Level | Design | Production | 備考 |
|-------|--------|------------|------|
| Reservation identity | **CLOSED** (D48: `acquire(key=prevWorld identity)` 定義) | **OPEN** (未実装) | 単に narration 差 |
| 1 reservation ≤1 (1 World ≤1 reservation) | **CLOSED** (INV-R) | **OPEN** (未実装) | 同上 |
| Rollback semantics | **PARTIAL** (build failure→release は D48 で想定だが deferred/retry の同一性は追加確認要) | **OPEN** | 推測で CLOSED に上げない |
| Temporal bound | **OPEN** | **OPEN** | — |
| Service curve | **OPEN** | **OPEN** | — |

## 3. F-3 A_true と reservation の因果方向再定義

I4 D48 本来の conservation:

```
prevWorld → reservation.acquire(prevWorld) → publishAndSwap(nextWorld)
  → oldWorld retirement → destroy → reservation.release(prevWorld)
```

**検証対象の転換:**

```
A_true = successful PublishedDomain admissions (publishAndSwap 成功)
R_outstanding = retired-but-not-destroyed World identities
A_true → R_acquire → R_outstanding
```

- `A_true(I) ≤ R_acquire(I)` が成立するか → design 上は acquire が swap 前のため成立するが **production には acquire 自体が存在しない**ため検証対象外
- `R_acquire ≤ R` を design contract で拘束できるか → INV-R1 `N_retired_world ≤ R` として design CLOSED

## 4. F-4 reservation cap ≠ publication rate 維持

I4 の `R = retired-but-not-destroyed identity 同時数` より `R < ∞ ⇒ N_success([t,t+G]) < ∞` の直接導出は不可。

```
publish → retire → destroy → release → next publish → ...
→ R 小でも高速循環で N_success(G) 大 — 形式的に固定
→ R は population bound であって service curve ではない
```

**D101-2.5F の CLOSED invariant 候補として確定。**

## 5. F-5 Temporal displacement 核心 Gate

```
t_acquire → build/queue/deferred/retry → t_publish
必要: t_publish - t_reserve ≤ D (有限 delay contract) または temporal dilation
```

**I4 reservation contract が temporal envelope を規定しているか → source/design contract 上 OPEN**

現段階では追加実装・測定で埋めない。

## 6. F-6 Service curve 接続条件確定

D101-2.5 を CLOSED にするには chain:

```
Reservation identity → 1 reservation ≤1 publication → finite reservation admission envelope
  → finite reserve→publish temporal displacement → publication service curve → N_success([t,t+G]) ≤ λG+B
```

現状: 後半が未成立

```
R finite ≠ A_max finite ≠ λ finite ≠ service curve (明示的に固定)
```

## 7. Proof Obligation — Design / Production / D101 status

| Proof | Design contract | Production | D101 status |
|-------|---------------|------------|-------------|
| Reservation identity | CLOSED | OPEN | OPEN |
| acquire LP | CLOSED | OPEN | OPEN |
| release LP | CLOSED | OPEN | OPEN |
| 1 World ≤1 reservation | CLOSED | OPEN | OPEN |
| exactly-once release | CLOSED | OPEN | OPEN |
| all-path coverage | CLOSED | OPEN | OPEN |
| retry identity | OPEN | OPEN | OPEN |
| rollback | OPEN | OPEN | OPEN |
| reserve→publish delay | OPEN | OPEN | OPEN |
| temporal dilation | OPEN | OPEN | OPEN |
| finite admission envelope | OPEN | OPEN | OPEN |
| service curve | OPEN | OPEN | OPEN |

**総括:**
```
D101-2.5F
├─ design conservation = CLOSED (INV-R1〜R4)
├─ production conservation = OPEN (L2 NO-GO)
├─ temporal conservation = OPEN
└─ service curve = OPEN

D101-2.5 = OPEN
D101-2.6R = PARTIAL (E_ref=0 CLOSED, E_sample OPEN)
M/R/R_cap/B_max^true = UNDETERMINED / T2 = NO-GO
```

**重要:** D101-2.5E の「reservation 未導入」表現をそのまま次工程へ持ち越さず、design CLOSED / production NO-GO の分離を正しく引き継ぐ。「reservation が存在するか」ではなく「設計上 CLOSED な conservation invariant が production に refinement されているか」が次監査対象。

## 8. 禁止事項遵守

コード変更 / Reservation 実装 / R gate 実装 / Token bucket 実装 / admission limiter 実装 / harness 変更 / 追加測定 / R 数値決定 / A_max=1 昇格 / T_w=2 昇格 / max(E_w)=1 昇格 / R→λ 短絡 / R→B_max 短絡 / λG+B 仮採用 / T2 GO — **全て未実施**

## 9. 次進行 (D101-2.5G)

D101-2.5F の design/production 分離を踏まえ、次は **temporal conservation / reserve→publish displacement** を監査:

```
D101-2.5G — Temporal Conservation / Reserve→Publish Displacement Audit
```

## 10. Tool Coverage

| 系統 | ツール | 結果 |
|------|--------|------|
| WSL | rg 15.1.0 / sg 0.44.0 / fdfind 10.3.0 / ag 2.2.0 / fzf 0.67.0 / sed 4.9 / awk 5.3.2 | 全 census 一致 |
| Sandbox | node fs walk / source直読 | enumeration 正確性確定 |
| MCP | serena 一時無効 (代替 rg) | — |
| MCP | AiDex 一時無効 (代替) | — |
| CLI | ccc 0.45.2 / graphify 0.9.48 / semble 0.5.5 | 検索実施 |
| 文献 | crossbeam-epoch / rigtorp MPMCQueue / serena / cocoindex / ast-grep / semble / AiDex / graphify / headroom | 200 OK (Vyukov失効→rigtorp代替) |
