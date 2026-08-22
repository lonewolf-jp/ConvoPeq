# Phase I-T1-D101-2.5E — Reservation → Successful Publication Conservation Proof Audit

> **Verdict: D101-2.5E = OPEN (Level 0/1 のみ部分到達、Level 2/3 到達不能)**
> **→ D101-2.5 全体 = OPEN 維持 / D101-2.5D の「contract missing」判定を conservation 観点から補強**
> **M / R / R_cap / B_max^true / T2 = UNDETERMINED 継続 / 測定・コード変更なし**

## 0. 前提と方法

- D101-2.5D で `Type B (token-bucket) + conservation + rollback` が最小候補となったが、**reservation→publication の時間的有界性が未証明**。
- 本 audit は **lifecycle 証明 (outstanding reservation の数に関する証明) と service-curve 証明 (時間窓上の A_true bound) を混同しない**ことが核心。
- `ConvoPeq.md` 一次資料 + `src/` 全量を rg/ast-grep/fdfind/ag/fzf/sed/awk、serena代替(rg)、coco/graphify/semble/AiDex(代替)で全件 census。
- コード変更・harness変更・測定・数値化は禁止。

---

## 1. D101-2.5E-1 — Reservation identity の定義

| 候補 | 現行 design / production 上の実体 |
|------|-----------------------------------|
| ReservationId | **未導入** — `WorldRetirementReservation` は design contract (I4/別資料 R3) として存在するが、production の reservation acquire/release mechanism は未実装 |
| WorldId | PublishedDomain 加入後の World を識別するが、`worldId == reservationId` とは仮定不可 |
| PublicationSequence / Generation | `publishAndSwap` 成功時の sequence epoch/gen として存在するが、reservation の同一性根拠ではない |

**将来設計の identity と現行 production identity を混同しない** — 現行 production は `enqueueRuntimePublicationFireAndForget → OwnerChannel/Key → executePublish → publishAndSwap` で reservation 層を経由しない。

**Gate D101-2.5E-1: `reservation identity を一意に定義可能` → PARTIAL (design contract 上は定義可能だが production 実装が存在しないためコード上の census 不可) / 実質 OPEN**

---

## 2. D101-2.5E-2 — 1 reservation → 最大1 success

**要求:** `∀r ∈ Reservation: successfulPublicationCount(r) ≤ 1`

| 経路 | 同一性 | 判定 |
|------|--------|------|
| normal publish | reservation 未実装のため publication 単位で計数 | 未証明 |
| bootstrap | 同上 | 未証明 |
| deferred publish | 再試行時に同一 World を再 enqueue するが reservation 再利用契約なし | 未証明 |
| recovery | Recovery intent は設計上別経路 | 未証明 |
| retry | `same reservation` vs `new reservation` の区別契約が未定義 | **OPEN** |

**禁止誤推論の遵守:**
- `same World ⇒ same reservation` とはしない — World は PublishedDomain の成果物であり reservation 前段階の識別子ではない
- `same generation ⇒ same reservation` とはしない — generation は World 属性であり reservation 層の同一性ではない

**Gate D101-2.5E-2: OPEN — reservation 自体が未実装のため ≤1 証明の前提が不存在**

---

## 3. D101-2.5E-3 — publishAndSwap と reservation の順序 (5ケース)

基準構造: `validate → admission → owner transfer → acquire → publishAndSwap → retire` (I4 / readiness audit より)

| Case | 状態 | reservation state | A_true | release state |
|------|------|-------------------|--------|---------------|
| A. reserve → success | 未実装 | — | +1 (publishAndSwap success) | terminal release で観測 |
| B. reserve → reject/failure | 未実装 | — | 0 | N/A |
| C. reserve → enqueue failure | 未実装 | — | 0 | retain or release 未定義 |
| D. reserve → deferred → retry → publish | 未実装 | 同一性未定義 | 0→+1 | reservation release は publish 成功時に限定 |
| E. reserve → shutdown | 未実装 | — | 0 (D101-1R2 で residual=Generic 確定) | drainAllNonRt で回収 |

**全ケースで reservation state の対応表は設計上の想定に留まり、コード上の不変条件として固定されていない。**

---

## 4. D101-2.5E-4 — Rollback conservation

| 状態 | A_true | Reservation |
|------|:------:|-------------|
| validation reject | 0 | acquire 前 |
| admission reject | 0 | acquire 前 / rollback 未定義 |
| build failure | 0 | release (未実装) |
| enqueue failure | 0 | retain or release — **未定義** |
| deferred | 0 | retain (未実装) |
| publish failure | 0 | release/retain 要定義 — **未定義** |
| publish success | +1 | consumed (未実装) |
| retry (same token) | 0 | 消費しない — **契約なし** |
| terminal release | 0 | reservation release (未実装) |

**`reserve → build failure → release` だけでなく `reserve → enqueue failure → retry` の意味論が閉じていなければ conservation proof は CLOSED にできない → OPEN**

---

## 5. D101-2.5E-5 — Retry identity proof

**要求:**
```
same reservation retry ⇒ no additional admission unit
same reservation ⇒ at most one successful publication
```

- `retry_count` は D101 growth count ではない — 正しい
- しかし retry が別 reservation を生成する場合は `retry → new reservation` として admission envelope で数える必要があり、この区別が未定義
- **曖昧なままでは conservation を閉じられない → OPEN**

---

## 6. D101-2.5E-6 — Temporal displacement (最重要 Gate)

仮に `∀r: success(r) ≤ 1` が証明できても:
```
N_success([t,t+G]) ≤ N_reservationAcquire([t,t+G])
```
は無条件に成立しない。

**例:**
```
reservation: t=0 R1, t=1 R2, t=2 R3
publication: t=100 P1, t=101 P2, t=102 P3
→ window [t,t+G] がずれ、reservation window と publication window が乖離
```

必要条件: `τ_res→publish ≤ D` (有限 envelope) または `N_publish(I) ≤ N_reservation(I ⊕ D)` の temporal dilation

**現行 design contract に有限 D は存在しない → OPEN**

**原因:** Build / queue / deferred / retry による `reservation acquire (t0) → delay → publish success (t1)` の遅延が有界であるという契約が未導入。

---

## 7. D101-2.5E-7 — Reservation count と temporal admission rate の分離

`A_max = outstanding reservation population` (I4/D101-9 定義) は **queue capacity とは異なる**が、**publication rate でもない**。

```
A_max < ∞ が証明されても N_success(G) < ∞ の temporal bound は直ちに出ない
```

**必須 counterexample 固定:**
```
reserve → publish → release reservation → reserve → publish → release → ...
→ OutstandingReservation ≤ 1 でも N_success(G) は時間的に別問題
→ A_max finite ⇒ temporal bound という短絡は禁止
```

**本 audit で形式的に固定: outstanding cap ≠ temporal admission bound**

---

## 8. D101-2.5E-8 — Conservation 強さ 4段階分類

| Level | 内容 | 到達 |
|-------|------|------|
| Level 0 | reservation → publication の semantic 対応 | PARTIAL (design contract 上は対応関係を記述可能だが code 上未実装) |
| Level 1 | `∀r: success(r) ≤ 1` | **OPEN** (reservation 未実装のため前提不存在) |
| Level 2 | `N_success(I) ≤ N_reservation(I')` (I' は I の temporal dilation) | **OPEN** (有限 D 未証明) |
| Level 3 | `N_success([t,t+G]) ≤ λG + B` | **OPEN** (D101-2.5C で service-curve 導出不能を確認済み) |

**D101-2.5 を CLOSED にできるのは Level 3 到達時のみ → 現状 Level 1/2 で OPEN 維持**

---

## 9. D101-2.5E-9 — 現行設計と将来T2の境界

```
D101-9 reservation lifecycle design proof = outstanding reservation の lifecycle proof (acquire=1 → 9 terminal release 到達)
T1 = counting / telemetry (A_true, R_true の観測 — 現行で CLOSED)
T2 = R gate + bounded admission authority + K_terminal + G_contract (将来 — 未実装)
```

- lifecycle 証明 ≠ service-curve 証明 — 前者は outstanding 数の保存、後者は時間窓上の bound
- 本 audit は **T2 が存在すれば Level 3 を理論上提供できる設計になっているかを問うが、T2 実装案を作る段階ではない**

**判定: T2 の admission authority 設計は budget authority として構想されているが、D101-2.5E までの conservation proof が未閉鎖のため T2 実装の前提を満たさない → NO-GO 維持**

---

## 10. D101-2.5E-10 — 最終分類 (3択)

| Case | 条件 | 判定 |
|------|------|------|
| Case A — Conservation + temporal envelope closed | reservation 1:1 + bounded delay → λG+B 導出 | **不成立** |
| Case B — Conservation closed, temporal envelope open | `∀r success(r) ≤1` は証明可能だが delay/burst 未契約 | **不成立** (Case B の前提である Level 1 すら未到達) |
| Case C — Conservation 自体が閉じない | reservation identity / retry identity / rollback 未定義 | **★ 該当 — 現行の状態** |

**→ D101-2.5D の OPEN を継続し、T2 設計には進まない (Case C)**

---

## 11. D101-2.6R は今回変更しない

```
E_ref = 0                         CLOSED (B_ref == B_true)
T_w = observer running peak       CLOSED (event-driven running max)
E_sample = OPEN                   維持 (T_w = observer peak であることを確認した後に sampler cadence で M を導出する方向へ戻らない)
numerical M bound                 OPEN
```

**T_w が observer peak であることを確認した後に `E_sample = 0` とはしない — `M = Δgrowth + E_sample` 分解の dependency graph のみ固定**

---

## 12. 禁止事項遵守

`× コード変更 × harness変更 × 測定 × reservation implementation × A_max 数値化 × A_max → temporal rate bound × OwnerChannel capacity → A_max × Registry capacity → A_max × λ×G 採用 × observed burst → contract × T2 implementation × R/R_cap 決定 × M 数値化` — **全て未実施**

---

## 13. 最終成果物 — Proof obligation 一覧 (15項目)

| Proof obligation | 結果 |
|------------------|:----:|
| reservation identity | OPEN (design contract 上は WorldRetirementReservation として存在するが production 未実装) |
| reservation → World identity | PARTIAL (design 上は 1:1 想定、code 上未実装) |
| World → publication identity | PARTIAL (publishAndSwap が同一性 LP だが reservation との紐付け未実装) |
| 1 reservation → ≤1 success | OPEN |
| retry identity preservation | OPEN |
| deferred identity preservation | OPEN |
| recovery identity preservation | OPEN |
| build-failure rollback | OPEN |
| enqueue-failure rollback | OPEN |
| publish-failure rollback | OPEN |
| reservation → publish delay bound | OPEN |
| window dilation bound | OPEN |
| `N_success(I) ≤ N_reservation(I')` | OPEN |
| `A_max` → temporal bound | OPEN (outstanding cap ≠ temporal bound) |
| `λG+B` derivation | OPEN |

---

## 14. 最終判定

```
D101-2.5E = OPEN (Case C — Conservation 自体が閉じない)
D101-2.5D = OPEN 継続
D101-2.5 overall = OPEN
D101-2.6R = PARTIAL (E_ref=0 のみ CLOSED)
M / R / R_cap / B_max^true / T2 = UNDETERMINED 継続
```

**結論:** D101-2.5D で「reservation contract が必要」と分かった段階で、次は設計するのではなく **reservation が successful publication に対して本当に conservation law を与えるかを先に証明する**必要があったが、本 audit で **lifecycle 証明 (outstanding reservation 数) と service-curve 証明 (時間窓上の bound) は別物**であり、現行では前者すら code 上未実装のため後者の議論に入れないことが確定した。

**現在の状態は固定:**
```
D101 #1       CLOSED
D101-2.5C     OPEN
D101-2.5D     OPEN
D101-2.5E     OPEN (Case C)
D101-2.6R     PARTIAL

A_max_candidate = 1      observed only
T_w             = 2      observed only
max(E_w)        = 1      observed only

M               = UNDETERMINED
R               = UNDETERMINED
R_cap           = UNDETERMINED
B_max^true      = UNDETERMINED
T2              = NO-GO
```

## 15. Tool Coverage

| 系統 | ツール | 結果 |
|------|--------|------|
| WSL | rg 15.1.0 / sg 0.44.0 / fdfind 10.3.0 / ag 2.2.0 / fzf 0.67.0 / sed 4.9 / awk 5.3.2 | 全 census 一致 |
| Sandbox | node fs walk / source直読 (RuntimeWorldAuthority.h, RuntimePublicationCoordinator.h, AudioEngine.h 等) | enumeration 正確性確定 |
| MCP | serena 一時無効 (前 Gate 多数確定済みで rg 代替) | — |
| MCP | AiDex 一時無効 (node walk代替) | — |
| CLI | ccc 0.45.2 / graphify 0.9.48 / semble 0.5.5 | 検索実施 |
| 文献 | crossbeam-epoch / rigtorp MPMCQueue / serena / cocoindex / ast-grep / semble / AiDex / graphify / headroom | 9/10 200 OK (Vyukov失効→rigtorp代替) |

*Evidence generated: Phase I-T1-D101-2.5E — Conservation proof audit. lifecycle ≠ service-curve を混同しないことが核心。*
