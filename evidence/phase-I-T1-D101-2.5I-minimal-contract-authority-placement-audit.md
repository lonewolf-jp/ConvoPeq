# Phase I-T1-D101-2.5I — Minimal Contract → Authority Placement / Non-Circularity Audit

> **Verdict: PLACEMENT-CANDIDATE PARTIAL — C1/C4 可配置、 C2/C3 は新規 Authority 導入が前提**
> H-B（必要集合特定可、十分性導出 OPEN）の下で 4 契約を既存 Authority に仮配置し、循環・越権を監査。
> **コード変更・測定・実装なし / M/R/R_cap/T2 UNDETERMINED 維持**

## I-1 Authority Inventory 固定

| Authority | 役割 | publishAndSwap 所有 |
|-----------|------|---------------------|
| **RuntimeWorldAuthority** (A) | 物理 RuntimeStore の write authority。`publishAndSwap` 唯一性は X4-B で固定 | **Yes** — `RuntimeStore::publishAndSwap` は `RuntimeWorldAuthority-owned WriteAccess` のみ (INV-X4-3) |
| **PublicationAdmission** (B) | publish 可否判定 (generation/sequence stale 等) | No — Admission コンポーネント、Deferred Queue は移設済み |
| **RuntimePublicationCoordinator** (C) | Intent enqueue/dispatch、CoordinatorLoop polling | No — publish 本体は RuntimeWorldAuthority へ委譲 |
| **Builder / RebuildThread** (D) | World 構築 | No — 構築後は Coordinator へ委譲 |

補助層:

| 補助 | 役割 | Authority 誤認リスク |
|------|------|---------------------|
| OwnerChannel | ownership transport (SPSC, key={seq,epoch,mappedGen}) | transport ≠ authority |
| PendingPublishRegistry | enqueue→commit gap の ownership 補助 (kCapacity=64) | transport/lifetime 補助、明示的に Registry≠Authority |

## I-2 C1〜C4 Authority 別配置監査

| Contract | RWA (A) | PublicationAdmission (B) | Coordinator (C) | Builder (D) |
|----------|---------|--------------------------|-----------------|-------------|
| **C1 reservation identity / ≤1 success** | **適合** (D48 design: publish()内 swap前 acquire, deleter内 release — 既存 semantic object と直結) | 越権 (transport/build 補助が admission 意味を所有) | 越権 (scheduling 層) | 越権 (build 層) |
| **C2 temporal envelope** | **未確定** (physical authority と temporal policy の同一化が Singularization と矛盾する可能性) | 適合候補 (admission gate として temporal 契約を所有可) | 越権 (FIFO/wake は temporal contract ではない — 2.5C否定済み) | 越権 |
| **C3 admission envelope α(G)** | **未確定** (rate policy を RWA に埋め込むと physical+temporal 同一化) | **適合候補** (α(G) 所有者として最有力 — 下記詳論) | 越権 | 越権 |
| **C4 rollback / retry identity** | **適合** (publication/admission の成否と同一 authority 内で rollback 契約) | 適合候補 (retry identity は admission 層でも保持可) | 未確定 (deferred/recovery は Coordinator で再 admission — 同一性保証の主体が曖昧) | 越権 |

## I-3 C1/C3 分離

**Invariant 固定:**

```
queue admission ≠ publication admission
PendingPublishRegistry capacity ≠ admission budget
```

- `enqueuePublicationIntent()` の shutdown/queue-full rejection は transport gate であり D101 temporal envelope ではない
- PendingPublishRegistry (64) は enqueue→commit lifetime gap の ownership 補助、admission authority ではない

**C1 (reservation conservation) と C3 (admission envelope) は別契約として分離 — 混同禁止**

## I-4 C1 Reservation Authority 精査

D48 design contract 再利用:

```
RuntimeWorldAuthority::publish() → reservation.acquire(prevWorld) → publishAndSwap → retire → destroy → reservation.release
```

- 問いは「C1 をどこに置くべきか」ではなく「D48 authority を `reservation → successful publication ≤1` conservation に拡張しても Singularization が壊れないか」
- 現時点では production implementation が NO-GO のため、拡張の非矛盾性は **design-level では適合、production refinement としては未検証** — I-4 は適合候補として留める

## I-5 C2 Temporal Envelope 独立監査

```
reservation authority ≠ temporal policy authority の可能性
RWA: reservation identity/lifecycle
Admission: α(G)
Coordinator: execution/scheduling
```

- CoordinatorLoop 1ms wait / FIFO / single consumer / bounded queue を temporal contract に昇格させてはならない (2.5C/G 否定済み) — **再確認**
- C2 の配置は C1 と独立して監査 — RWA に temporal policy を同一化するか、Admission 層に分離するかが論点

## I-6 C3 Admission Envelope 候補比較

| 候補 | 形式 | 所有者判定 |
|------|------|------------|
| α(G) General service curve | `N(I) ≤ α(|I|)` | **最上位抽象 — 所有者は未確定だが B より上位** |
| λG+B Rate+burst | `N(I) ≤ λ|I|+B` | α(G) の一実装形、所有者は α と同一 |
| fixed-window quota | `N(I) ≤ Q` | window 固定要、α の特殊形 |

**RWA へ rate policy 直埋めの Singularization 監査:**

```
physical publication authority + temporal policy authority → 同一化の是非
→ 現時点で「矛盾しない」と断定できない — 未確定として留める
→ 最も安全な分離は RWA=physical, Admission=temporal の分離
```

## I-7 C4 Retry/Deferred/Recovery Identity Ownership

| Path | 同一 logical obligation → same reservation 保証主体 |
|------|---------------------------------------------------|
| normal | RWA (publish 内 acquire) |
| deferred (単一スロット 0/1) | Orchestrator deferred state — 再 admission 時に同一性維持の主体が曖昧 |
| retry | PublicationAdmission の stale判定 ≠ reservation identity (generation/sequence equality ≠ reservation identity) — **同一視禁止** |
| recovery | Builder Work Queue 転送 — reservation との紐付け未定義 |
| bootstrap | RWA |

**I-7 は重要 Gate — deferred/recovery の identity ownership が未確定のため C4 は OPEN**

## I-8 Temporal Contract 3形態比較

| 形態 | 式 | C2 冗長性 |
|------|----|-----------|
| I-A | `∀r: t_publish(r) - t_reserve(r) ≤ D` | fixed D — 最も強い |
| I-B | `∀I: N_publish(I) ≤ N_reservation(I⊕D)` | window dilation — I-A の緩和形 |
| I-C | `N_publish(I) ≤ F(|I|)` 直接 publication service curve | **C2 (reserve→publish delay) が冗長になる可能性** — publication authority で直接 F(G) を契約すれば C2 不要 |

**判定:** C2 を reservation temporal delay として持つ必要があるか、publication authority で直接 `F(G)` を契約すれば冗長になるか — **未確定**。H-B を H-A に近づける分岐として保持。

## I-9 Circularity Check

```
Admission budget → Reservation → Publish → Retire → Reservation release
Publication service curve → Coordinator scheduling → Publish
```

| 循環候補 | 成立有無 |
|----------|----------|
| C3 depends on C2 delay, C2 depends on C3 queue restriction | **循環リスクあり — 十分 proof ではない** |
| Admission → RWA → Admission | 適合配置なら非循環 (RWA が reservation identity を所有、Admission が α を所有) だが分離しないと循環 |
| Coordinator scheduling → Publish → Admission | Coordinator が scheduling authority を持つ場合、admission と scheduling の相互依存が発生 |

**循環ができた契約集合は sufficient proof ではない — I-9 で明示的に固定**

## I-10 Final Classification

| 配置 | 判定 |
|------|------|
| C1 → Authority X (RWA 推定) | **適合候補** — ownership non-overlap, single semantic authority は design 上成立 |
| C2 → Authority Y (Admission 推定) | **未確定** — temporal policy authority の分離要否が残る |
| C3 → Authority Z (Admission 推定) | **未確定** — RWA 直埋めの Singularization 矛盾が未解消 |
| C4 → Authority X/Y | **未確定** — retry/deferred/recovery の同一性主体が曖昧 |

```
D101-2.5I = OPEN (配置自体が未確定 — contract decomposition の精査継続が必要)
D101-2.5C/D/E/F/G/H の OPEN は維持
```

ではなく、本監査の到達点として:

```
D101-2.5I = PLACEMENT-CANDIDATE PARTIAL
  C1: RWA 適合候補として特定
  C2/C3: Admission 層候補だが RWA との分離要否が未確定
  C4: RWA/Admission 境界で未確定
  Circularity: 潜在的循環を特定、配置分離で回避可能だが proof 未完成
```

**実装 GO にはしない — 配置可能性の監査に留める**

## H-B → CLOSED 条件の明文化

H-B を CLOSED にするには:

```
C1 → RWA, C2 → Y, C3 → Z, C4 → X/Y について
ownership non-overlap + no circular dependency + single semantic authority が証明
```

現時点では C2/C3/C4 の non-overlap / no-circularity が未証明のため **CLOSED 不可**。

## Final State Update

```
D101 #1       CLOSED
D101-2.5C     OPEN
D101-2.5D     OPEN
D101-2.5E     OPEN
D101-2.5F     Design CLOSED / Production OPEN
D101-2.5G     OPEN
D101-2.5H     H-B
D101-2.5I     PLACEMENT-CANDIDATE PARTIAL (C1 適合候補、C2/C3/C4 未確定、循環リスク特定)
D101-2.6R     PARTIAL
A/M/R/R_cap/B_max UNDETERMINED / T2 NO-GO
```

## Prohibitions Maintained

`× コード変更 × Reservation 実装 × Admission limiter 実装 × Token bucket 実装 × R gate 実装 × harness変更 × 追加測定 × λ数値化 × B数値化 × D数値化 × observed latency→D × queue capacity→rate × R→λ × R→B × A_max→temporal bound × λG+B仮採用 × T2 GO` — **全て未実施**

**最大論点:** `physical publication authority (RWA) と temporal admission policy authority の同一化可否` — 次 Gate の中心問題として保持。

## Tool Coverage

| 系統 | ツール | 結果 |
|------|--------|------|
| WSL | rg 15.1.0 / sg 0.44.0 / fdfind 10.3.0 / ag 2.2.0 / fzf 0.67.0 / sed 4.9 / awk 5.3.2 | 全 census 一致 |
| Sandbox | node fs walk / source直読 | enumeration 正確性確定 |
| MCP | serena 一時無効 (代替 rg/sg) | — |
| MCP | AiDex 一時無効 (代替) | — |
| CLI | ccc 0.45.2 / graphify 0.9.48 / semble 0.5.5 | 検索実施 (一部 not found は環境差) |
| 文献 | crossbeam-epoch / rigtorp MPMCQueue / serena / cocoindex / ast-grep / semble / AiDex / graphify / headroom | 9/10 200 OK (Vyukov失効→rigtorp代替) |
