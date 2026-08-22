# Phase I-T1-D101-2.5K — C2 Contract Semantics / Authority Placement Closure

> **Verdict: OPEN / CONDITIONAL — C2 semantic fixed to C2-B (latency), but reverse-edge non-circularity unproven**
> **→ H-A/CLOSED への進展不可 / D101-2.5K OPEN 維持 / M/R/R_cap/B_max^true/T2 UNDETERMINED**
> コード変更・測定・新規 Authority 実装なし。C2 を Model B に即時採用せず、provenance と enforcement の循環性を監査。

## K-0 Work Constraints

- コード変更禁止 / 測定禁止 / 新規 Authority 実装禁止 — 遵守
- M/R/R_cap/T2 UNDETERMINED 維持 — 遵守
- C2/C3 数学的十分性を仮定せず — 遵守
- D101-2.5J Model A/B/C を比較するが Model B 即時採用せず — 遵守
- ConvoPeq.md 一次ソースとして扱い、`PublicationAdmission::evaluate()` (generation/shutdown/finalization/health-pressure/fading) と `RuntimeWorldAuthority` sole gateway の構造を確認 — 遵守

## K-1 C2 Semantic Definition Fixed (3 candidates)

| Candidate | Definition | Nature |
|-----------|------------|--------|
| **C2-A** Reservation admission rate `N_reservation([t,t+G]) ≤ β(G)` | reservation 許可数 | rate contract |
| **C2-B** Reservation → publication latency `τ_res→publish ≤ D` | `reservation accepted → publication success` の時間境界 | **本命候補** (J-2 で `τ` 必要と記述) |
| **C2-C** Reservation occupancy `N_reserved_and_unpublished ≤ K` | 同時未 publish reservation 数 | outstanding bound — rate/latency ではない |

**K-1 Invariant: 3つを同一 C2 として扱わない** — 本監査で 3者を分離して固定。C2-B を本命候補とするが未確定として扱う。

## K-2 C2 Input Provenance Complete Census

| input | producer | owner | consumer | observation edge | decision edge |
|-------|----------|-------|----------|------------------|---------------|
| reservation timestamp | BudgetAuthority (future, design L0) | BudgetAuthority | Reservation-side authority | Telemetry (observed τ, 禁止依存) | — |
| publication success timestamp | RWA (`publishAndSwap` success) | RWA (RuntimeStore::current) | Telemetry / Admission (候補) | Telemetry observation | — |
| reservation identity | BudgetAuthority (mint token) | BudgetAuthority | RWA / Admission | — | RWA |
| publication identity | RWA (`publishAndSwap` success LP) | RWA | Telemetry / Admission | Telemetry | — |
| generation | Builder | RuntimeState/publication | Admission | — | Admission |
| sequence | publication (Bake) | RuntimeState/publication | Admission | — | Admission |
| admission state | Coordinator (queue state) | Coordinator | Admission | — | Admission |

**Critical verification — reverse edge candidate:**

```
Admission → RWA (publish path) が既存
RWA → Admission を C2 decision input として追加
⇒ Admission → RWA → Admission 循環の可能性
```

> **C2 の判定に publish completion を必要とするなら、Model B はそのままでは非循環とは言えない** — 本監査で反例として明示的に検証（K-4 で詳論）。

## K-3 Enforcement Point Decision (Most Important)

**C2-B = `reservation → publish ≤ D` の場合:**

### Candidate 1 (循环型 — 却下候補)
```
Reservation → Admission が deadline 設定 → RWA publish → 期限超過なら Admission が拒否
```
- publish 後に Admission が結果を必要とするため **循環発生の可能性** — enforcement point が publish 後の Admission 判定に依存。

### Candidate 2 (分離型 — 推奨候補)
```
Reservation → C2 lease/deadline 発行 (Reservation-side authority)
    ↓
RWA publish (physical execution)
    ↓
期限内/期限外を Telemetry が観測 (observation only)
```
- **Decision Authority = Reservation-side authority**
- **Physical execution = RWA**
- **Observation = Telemetry** — 3者分離により循環回避の可能性。

**K-3 結論: Candidate 2 が非循環性を満たす唯一の構造候補だが、`C2 を enforcement contract として十分に実現できるかは未証明` — OPEN として保持。**

## K-4 Model A/B/C Re-evaluation

### Model A `Admission: C3 / RWA: C1 + C2`

- RWA が `physical publish + temporal policy` 同時所有 → **Authority concentration 発生**
- `publishAndSwap()` は不可逆境界（reversible admission checks はその前に配置）— temporal policy をここに侵入させると NO-GO 候補
- **Verdict: 慎重に扱うべき — concentration リスク**

### Model B `Admission: C2 + C3 / RWA: C1`

- 一見最も分離が明確だが、C2-B の provenance 展開で reverse edge 問題が顕在化:
```
Admission → RWA → publish completion → Admission
```
- **J の「C2/C3 集約で循環減少」評価を C2-B provenance まで展開して再検証が必要** — 本 K-4 で検証中
- **Verdict: 即時採用危険 — reverse edge 検証が前提**

### Model C `RWA: C1 + F(G) / Admission: C3`

- RWA = physical + temporal → temporal authority 侵食が最大
- **最も慎重に扱うべき候補** — 現段階では採用不可

**総合: 3モデルいずれも即時確定不可 — K-3 Candidate 2 型の分離構造が前提**

## K-5 Observation ≠ Decision Separation

```
RWA: publishSuccessCount (physical count)
Telemetry: observed τ_res→publish (observed latency — 禁止依存)
Admission: admission decision
```

- `Telemetry observation → Admission decision input` になる瞬間に新規 dependency 発生
- **J-3 禁止依存 `observed latency → D` を C2 にも適用** — 本監査で明示的に禁止として固定
- `observed latency → C2 decision` を許すか禁止するか → **禁止** (observed → guaranteed への昇格は未証明)

## K-6 Necessary / Sufficient Condition Separation (H-B 維持)

| | 必要条件 | 十分条件 |
|---|---|---|
| **Necessary** | reservation identity, admission timestamp, publication identity, publication success event | — |
| **Sufficient** | — | deadline/lease semantics, enforcement point, terminal disposition |

- `timestamp を記録できる ≠ τ ≤ D を保証できる` — 記録可能性と保証を分離して固定

## K-7 Dependency Matrix (Final Deliverable)

| Contract | Decision Authority | Observation Authority | Execution Authority | Input | Output | Reverse edge |
|----------|-------------------|---------------------|---------------------|-------|--------|--------------|
| **C1** | RWA / Budget Authority | Telemetry | RWA | identity/reservation | publish/retire | **none** |
| **C2** | **Reservation-side (Candidate 2)** | Telemetry (observed τ, not decision) | RWA | reservation timestamp, publication success timestamp | τ ≤ D decision | **must prove none** — publish completion → Admission が必要なら循環 |
| **C3** | Admission (candidate) | Telemetry | Admission | G/window, generation/sequence | α(G) decision | **must prove none** |

**Edge 判定:**

```
C1 → C2: RWA reservation → Admission temporal — dependency あり、方向は RWA → Admission (非循環 if C2 decision は Reservation-side)
C2 → C3: temporal deadline → admission envelope — C2 が Admission にあれば同一 Authority 内、RWA にあれば跨ぎ
C3 → C1: admission envelope → reservation conservation — R finite → C1 への feedback は禁止 (R→λ 短絡禁止)
```

**K-7 結論: C1→C2, C2→C3 は方向性が固定されれば非循環だが、C3→C1 の feedback を許すと循環 — 固定により回避可能だが未証明**

## K-8 Gate Criteria (7項目)

| Gate | 条件 | 判定 |
|------|------|------|
| **K-C1** | C2 semantic が C2-A/B/C のいずれかに一意化 | **CONDITIONAL** — C2-B (latency) を本命候補として固定、ただし A/C の完全排除は未確定 |
| **K-C2** | C2 全入力の producer/owner/consumer 一意 | **PARTIAL** — generation/sequence/admission state は確定、reservation/publication timestamp は BudgetAuthority/RWA で確定だが RWA→Admission の publish completion 依存が未解消 |
| **K-C3** | enforcement point 一意 | **CONDITIONAL** — Candidate 2 (Reservation-side deadline + RWA execution + Telemetry observation) を推奨候補として固定、十分性は未証明 |
| **K-C4** | `Admission → RWA → Admission` reverse edge 非存在証明 or 解消構造提示 | **OPEN** — publish completion を C2 input とする場合に reverse edge が発生。Candidate 2 の分離で解消可能だが proof 未完成 |
| **K-C5** | C1/C2/C3 temporal/identity semantics 非混同 | **PASS** — 3者を分離して固定 |
| **K-C6** | necessary/sufficient 分離 | **PASS** — timestamp 記録 ≠ τ 保証 を明示 |
| **K-C7** | Model A/B/C のうち少なくとも1つで non-circular placement 具体証明 | **CONDITIONAL** — Model B の変形 (Candidate 2 分離) で non-circular 候補を提示、ただし proof 未完成 |

## Prohibitions Maintained

`× C2=τ≤D 仮定 × Model B 即時採用 × RWA publish timestamp 直読 × Telemetry observed latency 保証化 × queue capacity→C2 × reservation count→α(G) × 実装開始` — **全て未実施**

## Current Fixed State & Next

```
D101-2.5C     OPEN (service-curve 不在)
D101-2.5D     OPEN (admission envelope に conservation/rollback/burst 追加要)
D101-2.5E     OPEN (Case C)
D101-2.5F     Design CLOSED / Production OPEN
D101-2.5G     OPEN (finite D なし)
D101-2.5H     H-B
D101-2.5I     PLACEMENT-CANDIDATE PARTIAL
D101-2.5J     PLACEMENT-CANDIDATE CONDITIONAL
D101-2.5K     OPEN/CONDITIONAL — C2-B 本命固定、reverse-edge 解消が残課題

A_max=1 / T_w=2 / max(E_w)=1 observed only / M/R/R_cap/B_max UNDETERMINED / T2 NO-GO
```

**H-A/CLOSED への残条件:**
- K-C4 reverse edge の非循環証明完成 (Candidate 2 の分離構造を形式証明)
- K-C2 全入力の producer/owner/consumer 一意化完成
- K-C3 enforcement point の十分性証明

## Tool Coverage

| 系統 | ツール | 結果 |
|------|--------|------|
| WSL | rg 15.1.0 / sg 0.44.0 / fdfind 10.3.0 / ag 2.2.0 / fzf 0.67.0 / sed 4.9 / awk 5.3.2 | 全 census 一致 |
| Sandbox | node fs walk / source直読 (RuntimeWorldAuthority.h, PublicationAdmission.h 等) | enumeration 正確性確定 |
| MCP | serena 一時無効 (代替 rg/sg) | — |
| MCP | AiDex 一時無効 (代替) | — |
| CLI | ccc 0.45.2 / graphify 0.9.48 / semble 0.5.5 | 検索実施 (一部 not found は環境差) |
| 文献 | crossbeam-epoch / rigtorp MPMCQueue / serena / cocoindex / ast-grep / semble / AiDex / graphify / headroom | 9/10 200 OK (Vyukov失効→rigtorp代替) |
