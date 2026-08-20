# D101-8 Step 3 — A_max Derivation

| 項目 | 内容 |
| --- | --- |
| **ID** | D101-8 Step 3 |
| **日付** | 2026-08-20 |
| **前提** | Step 1: `FORMALIZED_WITH_OPEN_PROOFS` / Step 2: `RESERVATION_SEMANTICS_DESIGN_REQUIRED — CODE-EVIDENCE MISSING` / `I-W5: PARTIAL` / `O7: SEMANTIC MODEL ONLY` |
| **目的** | `R0 → R1` の Reservation Acquire event と `A_max` の **production code evidence を確定**する。Step 2 で確定した境界を厳守し、`A_max` は `R0→R1` Acquire event のみから derivation する |
| **制約** | **コード変更なし・semantic derivation only**。`A_max` は `pendingIntentCount_` / `publicationIntentResidencyCount_` / `reservationOwned` / `OwnerChannel` 256 / `PendingPublishRegistry` 64 / `worldId` / publication count / `WorldRetirementReservation` design contract から導出してはならない |
| **判定** | **A_max: PRODUCTION-CODE-EVIDENCE MISSING — DESIGN-DEFINED** — `A = successful Lifetime Budget reservation acquire events` は semantic quantity として定義可能だが、production code に対応する acquire mechanism は存在しない（Phase B3 / D69 で実装予定） |

---

## 1. Core Definition (Step 3-A)

> **`A_max` は R0→R1 の successful reservation acquire event の最大同時数を表す。**
> これは **Lifetime Budget for RuntimeWorld の reservation acquire** であり、Intent/Recovery admission や design contract の `WorldRetirementReservation` ではない。

```text
A = |{ successful acquire events e : e is Lifetime Budget reservation acquire }|
A_max = max(A(t)) at steady state
```

**Production code evidence**: Step 2 で確認したとおおり、`src/` には `WorldRetirementReservation` も `acquireReservation/releaseReservation` も実装されていない。したがって、**現行 production code に A を計測する mechanism は存在しない** — A_max は **future design value** としてのみ定義可能。

---

## 2. Step 3-B — R0→R1 Acquire event の production code evidence

### 2.1 調査方法

`ConvoPeq.md` / `src/` に対して以下を**推測せずに**検索した。

- `WorldRetirementReservation` / `acquireReservation` / `releaseReservation` / `ReservationHandle` / `ReservationToken` の全ヒット
- `A_max` / `A` (variable) の全ヒット
- `S0.*S1` / `admission.*gate` / `admission.*capacity` / `admission.*limit` の全ヒット
- `onAcquire` / `onRelease` / `referenceAcquireCount` / `admissionReservationsZero` の全ヒット

### 2.2 検索結果

| 検索対象 | Hits in `src/` | 判定 |
| --- | --- | --- |
| `WorldRetirementReservation` (class/type) | **0件** — `doc/work88/I4_DESIGN_CONTRACT.md` D48 で design contract として定義済みだが `src/` に未実装 | **DESIGN-ONLY / CODE MISSING** |
| `acquireReservation` / `releaseReservation` | **0件** | **MISSING** |
| `ReservationHandle` / `ReservationToken` (type) | **0件** | **MISSING** |
| `A_max` / `A` (variable) | **0件** | **MISSING** |
| `onAcquire` / `onRelease` (WorldRetirementReferenceObserver) | `ISRWorldRetirementReference.h:31,41` — telemetry observer | **TELEMETRY ONLY** — `measurement only・no admission decision・no R/R_cap` |
| `admissionReservationsZero` | `AudioEngine.h:4368, ISRShutdown.h:257, ISRWorldRetirementTelemetry.h:89` — shutdown quiescence check | **SHUTDOWN GATE** — World lifetime budget 非関係 |
| `PublicationAdmission::evaluate` | `PublicationAdmission.h/.cpp` — publish rejection decisions | **ADMISSION DECISION** — Lifetime Budget reservation 非関係 |

### 2.3 R0→R1 Acquire event = Production code 上の対応

```text
R0 → R1
  Acquire (Lifetime Budget reservation acquire)
    ↓
Production code:
  NONE — src/ に WorldRetirementReservation / acquireReservation は 0件
  I4_DESIGN_CONTRACT.md D48-D53 に design contract は存在するが
  Phase I implementation = NO-GO（コード未変更）

Future implementation (D69):
  WorldRetirementReservation::acquire(key = prevWorld identity)
  → publish() 内・publishAndSwap の前
  → count CAS 成功後（D69.3: LP = swap 前）
  → B_max sampler は AudioEngine.Timer.cpp の周期 drain（T1: telemetry only）
```

### 2.4 Acquire が成功した場合の状態変化 (production code)

**現行 production code では R0→R1 Acquire が存在しないため、状態変化も定義されていない。**

Future design (D48-D53):

```text
before acquire: A(t) = N
after acquire:  A(t) = N+1, N < R (gate invariant)
```

### 2.5 Acquire 最大同時数を決める bounded resource (production code)

**現行 production code では R (L-B capacity) が未定義。**

| Candidate | コード上の実体 | 判定 |
| --- | --- | --- |
| `R` (WorldRetirementReservation capacity) | `I4_DESIGN_CONTRACT.md D55: R_c = N_retired-but-not-destroyed` | **OPEN** — D53: "R の数値は Policy・T104"。`src/` に実装なし |
| `OwnerChannel::kCapacity` (256) | `OwnerChannel.h:41` | **NOT A_max source** — in-flight Owner ownership slot、World lifetime budget 非関係 (Step 2-2.4) |
| `PendingPublishRegistry::kPendingPublishCapacity` (64) | `RuntimeWorldAuthority.h:34` | **NOT A_max source** — async enqueue→commit gap metadata ring（overwrite on overflow） |
| `WorldLifecycleAudit::activeWorldCount_` (diagnostic) | `WorldLifecycleAudit.h:115` | **NOT A_max source** — Diagnostic 限定 telemetry counter（admission に使用されない） |

### 2.6 release 側との対応関係 (production code)

```text
現行 production code:
  deleter(world) — World destruction authority (TerminalReclaimAuthority)
    ≠ BudgetAuthority::release(T) — Future Lifetime Budget release authority

Future design (D48-D53):
  WorldRetirementReservation::release(key)
  → deferred-delete deleter 内・~RuntimePublishWorld() + aligned_free の後
  → A(t) = N-1
```

### 2.7 production code か test/design-only か

- **`WorldRetirementReservation`**: I4_D48-D55 design contract — **design-only** (`Phase I implementation = NO-GO`)
- **`WorldRetirementReferenceObserver::onAcquire/onRelease`**: T1 telemetry — **production code だが measurement only**（`no admission decision・no R/R_cap・no ReservationExhausted` — ISRWorldRetirementReference.h:17-19）
- **`PublicationAdmission::evaluate`**: **production code** だが publish rejection decision（Shutdown/Stale/Pressure）— Lifetime Budget reservation 非関係

---

## 3. Step 3-C — `WorldRetirementReservation` は「比較対象」として継承

Step 2 で確定した通り、`WorldRetirementReservation` は design contract として存在するが production implementation ではない。

```text
I4_DESIGN_CONTRACT.md D48-D53
  ↓
design contract CLOSED
  ↓
src/ に WorldRetirementReservation 型は 0件
  ↓
Phase I implementation = NO-GO（D53: "Phase I implementation = NO-GO"）

∴ WorldRetirementReservation
  = design contract
  ≠ A_max derivation source
```

**Step 3 での取扱い**: `WorldRetirementReservation` design contract は `A_max` の **future implementation candidate** として記録するが、**現時点では A_max 算定に使用してはならない**。D69 Phase I-T1 は telemetry 測定装置のみ（`held-set / free-stack / token / R gate は含まない`）である。

### 3.1 `WorldRetirementReservation` と Recovery durable admission の分離

Step 3 へ入る前に、Recovery durable admission も別 domain であることを明確にする：

| Mechanism | Domain | Lifetime Budget reservation か? |
| --- | --- | --- |
| `PendingRecoveryAdmission::reservationOwned` | Recovery durable admission | **NO** — Recovery event deduplication 用 |
| `pendingIntentCount_` | Intent transport residency | **NO** — ISR Intent transport |
| `publicationIntentResidencyCount_` | Publish Intent residency | **NO** — ISR Publish Intent |
| `WorldRetirementReservation::acquire` | **Future** Lifetime Budget reservation (design only) | **YES (design)** — ただし未実装 |
| `WorldRetirementReferenceObserver::onAcquire` | **T1 telemetry** | **NO** — measurement only / no authority |

**Step 3 で絶対に守る**: `pendingIntentCount_` / `publicationIntentResidencyCount_` / `reservationOwned` / `WorldRetirementReferenceObserver` はいずれも `A_max` の代理値として扱ってはなない。これらは **別 semantic domain の admission/residency/telemetry mechanism** である。

---

## 4. Step 3-D — Token↔World invariant (Step 2 継承)

Step 2 で確定した R2→R3 の意味論を継承する：

```text
Token (reservation budget unit)
  = R1 で mint され、S1..S6 で存続、S7 で release される
  = World W の Lifetime Budget を 1 unit 保持する
  = ownership-bearing ではない（token は World に紐付き ownership は持たない）

World (ownership object)
  = S2 で生成（local owner）、S2 で OwnerChannel へ移る
  = Token は物理的に移動しない（token は World W の budget を維持）

∴ Token 数 ≠ World ownership 数
   （token = budget reservation、World = ownership object）
```

---

## 5. Step 3-E — Authority invariant (Step 2 継承)

```text
TerminalReclaimAuthority
  = World destruction authority
  ≠ BudgetAuthority（future Lifetime Budget release authority）

World destruction
      │
      ▼
TerminalReclaimAuthority (deleter)
      │
  ← future connection (Step 7/8) →
      ▼
BudgetAuthority::release(T)
      │
      ▼
exactly-one release (INV-RSV-2)
```

**Current state**: `TerminalReclaimAuthority` は `WorldRetirementReference.h:17-19` により **measurement only** と明示されている。`World destruction → Budget release` の cross-domain contract は Step 2 §2.5 で `DESIGN OBLIGATION` として定義済み。

---

## 6. Step 3-F — Design/Implementation boundary (Step 2 継承)

```text
WorldRetirementReservation
  = design contract CLOSED（I4 D48-D53）
  ≠ production implementation（Phase I = NO-GO）

A_max
  = R0→R1 acquire event の最大同時数
  = production code に実装されていないため
  = DESIGN-DEFINED / PRODUCTION-CODE-EVIDENCE MISSING
```

> **Critical**: Step 2 の `CODE-EVIDENCE MISSING` は **design contract が存在しない** ではなく、**production implementation が NO-GO である** ことを意味する。D48-D53 の `WorldRetirementReservation` design contract は存在する。A_max derivation は将来の implementation（D69 Phase I-T2 / T104）を待つ。

---

## 7. A_max への証拠鎖

```text
A_max
  ↑
R0 → R1
  Acquire (Lifetime Budget reservation acquire)
    ↑
[production code: NONE — src/ に WorldRetirementReservation 0件]
  ↑
[future: WorldRetirementReservation::acquire(key = prevWorld identity)]
    ↑
[bounded resource: R (WorldRetirementReservation capacity)]
    ↑
[future: count CAS < R (gate invariant)]
  ↑
[I4_D48: acquire location = publish() 内・publishAndSwap 前]
  ↑
[I4_D53: INV-TOKEN-1 — acquire(token) ⇒ exactly one queued entry]
```

### 7.1 A_max に使用してはいけないもの（Step 2 継承 — 厳守）

| 禁止対象 | 理由 |
| --- | --- |
| `commitRuntimePublication()` count | publish pipeline facade — S0→S1 Acquire ではない（Step 2 §2.6） |
| `enqueueRuntimePublicationFireAndForget()` count | same — ownership transfer facade |
| `pendingIntentCount_` | Intent transport residency — Intent domain |
| `publicationIntentResidencyCount_` | Publish Intent residency — ISR domain |
| `PendingRecoveryAdmission::reservationOwned` | Recovery durable admission — Recovery domain |
| `OwnerChannel::kCapacity` (256) | in-flight Owner ownership slot — ownership capacity であり lifetime budget ではない |
| `PendingPublishRegistry::kPendingPublishCapacity` (64) | async enqueue→commit gap metadata ring |
| `worldId` generation count | World identity — `S2` 以降で existence、`S1` の reservation ではない |
| publication count (`WorldLifecycleAudit::publishedCount_`) | Diagnostic telemetry only |
| `WorldRetirementReferenceObserver::onAcquire` | measurement only / no authority |
| `WorldRetirementReservation` design contract | design contract は存在するが implementation NO-GO |
| Recovery durable admission tests | `testRecoveryDurableAdmission` 等 — Recovery domain |

---

## 8. Step 3 Verdict

```text
D101-8 Step 3
  = A_MAX_DERIVATION_DESIGN_DEFINED
    / PRODUCTION-CODE-EVIDENCE MISSING

A_max = [確定値ではなく将来の design value]
  ↑
R0 → R1 Acquire event
  ↑
[production code: NONE — WorldRetirementReservation は design contract only / Phase I NO-GO]
  ↑
[future: WorldRetirementReservation::acquire(key)]
  ↑
[bounded resource: R — I4_D53: "R の数値は Policy・T104"]
```

| 項目 | 状態 |
| --- | --- |
| `A = successful Lifetime Budget reservation acquire events` | **semantic definition OK / production code MISSING** |
| `R0→R1 Acquire event` | **CODE-EVIDENCE MISSING** — `src/` に 0件。I4 D48 で design location は定義されているが implementation は NO-GO |
| `A_max` | **DESIGN-DEFINED** — production code にて実装/計測される mechanism が存在しない |
| `A_max ≠ commitRuntimePublication count` | **PASS** (Step 2 §2.6 継承) |
| `A_max ≠ OwnerChannel 256` | **PASS** — ownership capacity であり lifetime budget ではない |
| `A_max ≠ PendingPublishRegistry 64` | **PASS** — metadata ring buffer である |
| `A_max ≠ worldId generation count` | **PASS** — World identity である |
| `A_max ≠ publication count` | **PASS** — WorldLifecycleAudit は Diagnostic only |
| `A_max ≠ WorldRetirementReservation design contract` | **PASS** — design contract は存在するが implementation NO-GO |
| `A_max ≠ pendingIntentCount_` | **PASS** — Intent transport residency |
| `A_max ≠ publicationIntentResidencyCount_` | **PASS** — Publish Intent residency |
| `A_max ≠ reservationOwned` (Recovery) | **PASS** — Recovery durable admission |
| I-W5 | **PARTIAL** (継承) — `M_world <= B_world` は semantic model で成立するが code evidence なし |
| O7 | **SEMANTIC MODEL ONLY / DESIGN-DEFINED / CODE-EVIDENCE MISSING** (継承) |

---

## 9. Step 4 への引継ぎ

Step 3 で確定した `A` の定義と証拠鎖を Step 4（P_max）へ継承する。特に、`A_max` が `commitRuntimePublication` / `OwnerChannel` / `PendingPublishRegistry` / `pendingIntentCount_` / `publicationIntentResidencyCount_` / `worldId` / publication count から **決して**導出してはならない境界をそのまま引き継ぐ。

`WorldRetirementReservation` design contract（D48-D53）は将来の `A_max` implementation candidate として記録するが、**実装されるまで A_max は DESIGN-DEFINED** である。

```text
D101-8 Step 3  A_MAX_DERIVATION_DESIGN_DEFINED
             / PRODUCTION-CODE-EVIDENCE MISSING

I-W5 = PARTIAL (継承)
O7   = SEMANTIC MODEL ONLY / DESIGN-DEFINED / CODE-EVIDENCE MISSING

A_max = future R (WorldRetirementReservation capacity) — T104 設計完了後
```
