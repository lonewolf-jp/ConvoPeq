# D101-8 Step 3 — A_max Derivation (reservation/admission event count)

| 項目 | 内容 |
| --- | --- |
| **ID** | D101-8 Step 3 |
| **日付** | 2026-08-22 |
| **対象ブランチ** | `main` |
| **前提** | D101-8 Step 2 verdict: `RESERVATION_SEMANTICS_DESIGN_REQUIRED — CODE-EVIDENCE MISSING` — Lifetime Budget reservation token は現行 production code に存在しない。`reserveRuntimePublicationIdentity()` は identity/counter issuance のみ。D101-8 Step 1 (B_world/M_world formal separation) + D101-7 (state machine contract R0→R8) + D101-7 §7.1 A_max definition |
| **制約** | **コード変更 0 / 契約定義のみ / 数値決定 0**。`A_max` を現行コードの任意の実測値に紐付けてはいけない。`reserveRuntimePublicationIdentity()` を `A_max` enforcement と再解釈してはならない。`B_world` と `A_count` の semantic distinction を維持する |
| **判定** | **A_MAX_DERIVATION_DESIGN_REQUIRED — PRODUCTION-CODE-EVIDENCE MISSING** |

---

## 1. Core Definition (Step 3-A)

**CRITICAL CORRECTION (2026-08-22)**: Step 3-A の最終定義（Step 2 のあとのセッションで確定）：

> **`A_max` は 1 interval に許容される successful Lifetime Budget reservation acquire events の最大数。**
> これは **Lifetime Budget for RuntimeWorld の reservation acquire** であり、Intent/Recovery admission や design contract の `WorldRetirementReservation` ではない。

```text
A_count(I) = |{ successful acquire events e : e is Lifetime Budget reservation acquire, timestamp(e) ∈ I }|
A_max(I) = max(A_count(I))

where:
  - acquire = S0 → R1 (Lifetime Budget Authority が token を mint)
  - A_count は rate/count (events per interval), NOT occupancy
  - A_max は finite design-time constant

K_world(t) = |{ token T : state(T, t) ∈ {R1, R2, R3, R4, R5, R6} }|
  - outstanding reservation token の数 (state/level)
  - A_max ≠ K_world: rate ≠ level, independent invariants
```

> **A_max = rate bound (events/interval) — controls flow IN**
> **K_world = occupancy bound (tokens at time t) — controls resident lifetime**

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

---

## 2.8 Step 3-B — `A_max` interval の定義 (design contract)

### 2.8.1 interval の formal definition

```text
Interval I[t] = [t_start, t_start + interval_duration)

where:
  - interval_duration: design-time fixed duration (NOT derived from observed max)
  - t_start: 定期的な epoch（future Budget Authority の定義）
  - A_count(I) = |{ e : e is acquire event, timestamp(e) ∈ I }|
```

**Production code evidence**: `ISRWorldRetirementTelemetry` は `samplingPeriodMs_`（`AudioEngine.Timer.cpp:575+`）による定期 sampling を行うが、これは **telemetry sampling interval** であり `A_max` enforcement interval ではない。`A_max` の interval は将来の Budget Authority が定義する。

### 2.8.2 Sliding window vs Fixed window

| Window type | A_count 定義 | Burst suppression | Current code | Decision |
| --- | --- | --- | --- | --- |
| **Sliding window** | `A_count(t) = count of acquire events in [t - interval_duration, t)` | Strong (continuous) | MISSING | DESIGN-DEFINED |
| **Fixed window** | `A_count(I_n) = count in [n*D, (n+1)*D)` | Weak (boundary burst) | MISSING | DESIGN-DEFINED |

**選択**: Sliding window が burst suppression に優れるが、fixed window が実装・証明が容易。将来の `BudgetAuthority` で選択可能。現行コードではどちらも存在しない。

### 2.8.3 Sustained rate vs Burst

```text
A_max = A_sustained + A_burst

A_sustained: steady-state rate bound（interval 平均）
            = long-interval での average acquire rate
A_burst: short-interval spike allowance（token bucket burst capacity）
```

**Production code evidence**: token bucket または類似 rate limiter は `src/` に存在しない → **CODE-EVIDENCE MISSING**。

### 2.8.4 A_count の counting pattern（acquire only, no rollback on build failure）

```text
A_countは次のイベントでのみ変化する:

  acquire success (R0→R1):  A_count += 1
  acquire failure (rejected): A_count unchanged
  build failure (R1→R0 rollback): A_count unchanged  ← CRITICAL
  publish failure: A_count unchanged
  drain/release (S7→R0): A_count unchanged  ← release は B_world に影響するが A_count には影響しない

∴ A_countは monotonically increasing または reset at interval boundary
    build failure による rollback は B_world (token) に影響するが A_count には影響しない
```

**Production code evidence**: `acquireObserved_`（`ISRWorldRetirementTelemetry.h:82`）は **publish success**（S3 enter、`AudioEngine.Commit.cpp:404`）で +1 される。これは `A_count`（S0→S1 acquire event）とは異なるタイミング。また D76.4 により **observational state ONLY** であり admission authority ではない。→ **A_count ≠ acquireObserved_**: CODE-EVIDENCE MISSING for A_count

---

## 2.9 Step 3-C — Enforcement point（future Budget Authority）

### 2.9.1 Enforcement architecture

```text
Admission decision (A_max check)
    ↓
Budget reservation acquire (S0→S1)  — A_count++, B_world++
    ↓
Build (S1→S2)
    ↓
Commit / Publish (S2→S3)
    ↓
... retire/terminal/release ...
    ↓
Release (S6→S7→S0)  — B_world--, A_count unchanged
```

### 2.9.2 Enforcement point の formal definition

```text
Enforcement point = BudgetAuthority::acquire(key) の呼び出し前

Precondition:
  - A_count(current_interval) < A_max  (rate limit)
  - B_world(current) < K_world        (occupancy limit)

Postcondition (on acquire success):
  - A_count(current_interval) += 1
  - B_world += 1
  - R0→R1 token minted

Postcondition (on acquire failure):
  - No token minted
  - A_count unchanged
  - B_world unchanged
  - backpressure / BLOCK to producer
```

### 2.9.3 RejectedPressure は A_max enforcement ではない（Step 2E 継承）

`RejectedPressure`（`PublicationAdmission.cpp:39-47`）は：

```cpp
const bool pressureActive = convo::consumeAtomic(
    engine.retirePressurePublicationThrottleActive_, std::memory_order_acquire);
if (pressureActive) {
    return Decision::RejectedPressure;
}
```

- **Reactive backpressure**: retire backlog depth >= hwm から boolean 派生（`ISRRetireRouter::enqueueWithRetry` chain）
- **Boolean gate**: `retirePressurePublicationThrottleActive_` は `std::atomic<bool>`（counter ではない）
- **Admission path と分離**: `admission_.evaluate()` は reservation を取得しない
- **A_count 非関連**: `RejectedPressure` は `A_count` も `B_world` も track しない
- **Direction 逆**: `RejectedPressure` は retire queue 満村を信号 source にする（flow OUT が詰まった時）。`A_max` は reservation acquire rate（flow IN）を制御する

```text
RejectedPressure ≠ A_max exhaustion

RejectedPressure:
  - trigger: retire backlog depth >= hwm（reactive, flow OUT）
  - type: boolean gate
  - monitors: retireQueueDepth, ISRHealthState
  - domain: reactive backpressure
  - code: PublicationAdmission.cpp:39-47, AudioEngine.Retire.cpp:205+

A_max exhaustion:
  - trigger: A_count(interval) >= A_max（proactive rate limit, flow IN）
  - type: counter-based
  - monitors: acquire events in interval
  - domain: reservation admission control
  - code: MISSING — will be in future BudgetAuthority::acquire()
```

**結論**: `RejectedPressure` は A_max enforcement ではない。A_max enforcement は将来の `BudgetAuthority::acquire()` で行われる。現行 code には A_max enforcement mechanism は存在しない。

---

## 2.10 Step 3-D — Acquire / Rollback formal contract

### 2.10.1 State machine (Step 2 R0→R8 を継承)

```text
R0 Available ──acquire()──→ R1 Reserved ──build──→ R2 WorldCreated
  (BudgetAuthority)         (A_count++, B_world++)   (builder)
                                    ↓ rollback
                                  build fail → R0
                                  (B_world--, A_count unchanged)
                                  ↓ transfer fail
                                  Intent enqueue fail → R0
                                  (B_world--, A_count unchanged)
                                  ↓ shutdown drain
                                  drainAll → R7 → R0
                                  (B_world--, A_count unchanged)
```

### 2.10.2 Acquire success

```text
Precondition:
  - A_count(current_interval) < A_max
  - B_world < K_world

Action:
  - BudgetAuthority::acquire(key) mints token (R0→R1)
  - A_count += 1 (interval 内)
  - B_world += 1 (global outstanding)

Postcondition:
  ∃! token T: state(T) = R1 Reserved
  B_world(t+) = B_world(t) + 1
  A_count(interval+) = A_count(interval) + 1
```

### 2.10.3 Acquire failure

```text
Precondition:
  - A_count(current_interval) >= A_max  OR  B_world >= K_world

Action:
  - No token minted
  - Admission rejected (backpressure / BLOCK)

Postcondition:
  no token exists
  A_count unchanged
  B_world unchanged
  producer receives backpressure signal
```

### 2.10.4 Build failure rollback (R1→R0)

```text
Event: RuntimeBuilder::buildRuntimePublishWorld() failure

Action:
  - Reservation token release (R1→R0)
  - B_world -= 1
  - A_count は rollback しない（acquire が成功したため count 済み）

Postcondition:
  token released (R0 Available)
  B_world(t+) = B_world(t) - 1
  A_count(interval) unchanged
```

> **CRITICAL distinction**: Build failure 時の `B_world` release は `A_count` に影響しない。`A_count` は **acquire event** の count であり、acquire が成功した時点で count される。build failure は acquire 後の失敗である。

### 2.10.5 Transfer failure rollback (R1/R2→R0)

```text
Event: OwnerChannel::enqueue() failure まいは Intent enqueue failure

Action:
  - Reservation token release (R1→R0 or R2→R0)
  - B_world -= 1
  - A_count は rollback しない

Postcondition:
  token released
  B_world は -1
  A_count は unchanged
```

### 2.10.6 Shutdown drain rollback (R3/R4/R5/R6→R0)

```text
Event: AudioEngine shutdown, drainAllNonRt() → ISRRetireRouter::drainAll()

Action:
  - All resident worlds reclaimed
  - All reservation tokens released (S3..S6 → S7 → S0)
  - B_world = 0

Postcondition:
  all tokens released exactly once
  B_world(session_end) = 0
  A_count(interval) は session boundary で reset
```

### 2.10.7 4 prevention contracts（design obligations）

| Contract | Statement | Evidence |
| --- | --- | --- |
| **Double acquire prevention** | `acquire(key)` は同一 key に対して at-most-once | **MISSING** — will be in Budget Authority |
| **Double release prevention** | `release(token)` は exactly-once | **MISSING** — will be in Budget Authority (Step 2 §5.2) |
| **Lost release prevention** | `acquired ⇒ (released ∨ token exists in R1..R7)` | **MISSING** — will be in Budget Authority |
| **Release without acquire prevention** | `release(token)` は事前に `acquire(token)` が成功していることを要求 | **MISSING** — will be in Budget Authority |

---

## 2.11 Step 3-E — A_max < ∞ の形式的十分条件

### 2.11.1 十分条件の列挙

`A_max < ∞` を証明するために必要な前提条件：

| # | Condition | Evidence classification | 根拠 |
| --- | --- | --- | --- |
| 1 | **interval I is finite** | DESIGN-DEFINED | A_max interval duration は design-time constant |
| 2 | **A_count is atomically incremented per accepted acquire** | CODE-EVIDENCE MISSING | Future BudgetAuthority::acquire() が atomic increment。現行 acquireObserved_ は observational only (D76.4) |
| 3 | **each accepted admission consumes exactly one A-count unit** | DESIGN-DEFINED | Step 2 §4.1: `Acquire success ⇒ exactly one reservation token exists` |
| 4 | **A-count is not reset by build failure** | DESIGN-DEFINED | Step 2 §4.1: build failure 時は A_count unchanged |
| 5 | **reservation occupancy B_world ≤ K_world** | CODE-EVIDENCE MISSING | K_world は Step 1 conditional proof。Budget Authority 実装待ち |
| 6 | **enforcement is performed before accepting the reservation** | CODE-EVIDENCE MISSING | A_max check は acquire() precondition。現行 evaluate() は A_max を check しない |
| 7 | **A_max < ∞ (finite constant)** | DESIGN-DEFINED | A_max は design-time constant。値は Step 3F で decision |

### 2.11.2 Proof structure

```text
A_max < ∞  の証明:

  Premise 1: A_max is finite design-time constant  (DESIGN-DEFINED)
  Premise 2: interval I is finite  (DESIGN-DEFINED)
  Premise 3: A_count is atomic increment per accepted acquire  (CODE-EVIDENCE MISSING → future BudgetAuthority)
  Premise 4: acquire failure returns backpressure (A_count unaffected)  (DESIGN-DEFINED)
  Premise 5: B_world ≤ K_world < ∞  (CODE-EVIDENCE MISSING — Step 1 conditional)

  ∴ A_count(I) ≤ A_max < ∞
    （Premise 3 + Premise 4: A_count is monotonic increment or backpressure stops acceptance）
    （Premise 5: B_world saturation also halts accepts via occupancy check）
```

---

## 2.12 Step 3-F — A_max と K_world の関係（semantic distinction）

### 2.12.1 定義の再確認

```text
A_max: rate bound (events per interval)
  - unit: acquire events / time interval
  - domain: admission / reservation event stream
  - direction: controls flow IN
  - reset: interval boundary で reset

K_world: occupancy bound (state count at time t)
  - unit: outstanding reservation tokens
  - domain: budget pool state
  - direction: controls flow IN/OUT balance
  - reset: never (cumulative, released → S7 → S0)
```

### 2.12.2 なぜ A_max から K_world を導出してはならないか

| 論拠 | 説明 |
| --- | --- |
| **Different temporal semantics** | `A_max` は interval 内の event 数（rate）。`K_world` は time t の state 数（level）。Rate が有限であっても、flow out（drain/release）が停止すれば level は無限になり得る。逆も然り。 |
| **A_max = ∞ でも K_world < ∞ は可能** | A_max が無制限だが drain が十分速ければ K_world は bounded に留まり得る。A_max が有限であっても drain が停止すれば K_world は発散する。 |
| **K_world は A_max から独立して証明可能** | K_world は M_world の各 component（M_current=1, M_retire≤4096, M_quarantine≤1024, M_terminal≤K, M_reader≤f(H_max)）の和として証明される。A_max は別の invariant。 |

### 2.12.3 A_max と K_world の協働

```text
A_max は K_world の「flow in を制御」する
K_world は A_max の「buffer occupancy を保証」する

Flow in (A_count, controlled by A_max)
  ↓
B_world(t) = Σ(acquire) - Σ(release)  ∈  [0, K_world]
  ↓
Flow out (release, controlled by drain/H_max/G_max)

A_max < ∞  かつ  drain rate > 0  ならば  B_world(t) ≤ K_world < ∞
```

**Critical**: `A_max < ∞` だけでは `K_world < ∞` は保証されない（drain 停止の可能性）。同様に `K_world < ∞` だけでは `A_max < ∞` は保証されない（rate 無制限）。**両者は独立した invariant である。**

---

## 2.14 Step 3-C — `WorldRetirementReservation` は「比較対象」として継承

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

### 2.15.1 `WorldRetirementReservation` と Recovery durable admission の分離

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

## 2.13 Step 3-D (supplement) — Token↔World invariant (Step 2 継承)

Step 2 で確定した R0→R8 state machine における token↔world の関係：

```text
Token (reservation budget unit)
  = R0→R1 で mint され、R1..R7 で存続、R7→R0 で release される
  = World W の Lifetime Budget を 1 unit 保持する
  = ownership-bearing ではない（token は World に紐付き ownership は持たない）

World (ownership object)
  = R1→R2 で生成（local owner）、R2 で OwnerChannel へ移る
  = Token は物理的に移動しない（token は World W の budget を維持）

∴ Token 数 ≠ World ownership 数
   （token = budget reservation、World = ownership object）
```

> **Note**: この節は 2.10 (Step 3-D) の supplementary 説明。Step 2 の R0→R8（2.10.1）と対比。

---

## 2.15 Step 3-E — Authority invariant (Step 2 継承)

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

## 2.16 Step 3-F — Design/Implementation boundary (Step 2 継承)

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

## 2.17 A_max への証拠鎖

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

### 3.1 Step 3 stopping conditions

| #  | Step 3 closure 条件 | Status | Evidence |
| --- | --- | --- | --- |
| 1  | `A_max` の semantic unit が reservation/admission event に固定 | ✅ | 2.1: A = count of successful acquire events |
| 2  | `A_max` の interval 定義が固定 | ✅ (DESIGN-DEFINED) | 2.8: fixed/sliding window, interval is finite |
| 3  | burst / sustained の扱いが固定 | ✅ (DESIGN-DEFINED) | 2.8.3: A_sustained + A_burst |
| 4  | enforcement point が固定 | ✅ (DESIGN-DEFINED) | 2.9: `BudgetAuthority::acquire()` 前. RejectedPressure ≠ A_max (2.9.3) |
| 5  | Acquire成功時の `A_count` / `B_token` の関係が固定 | ✅ | 2.10.2: both +1 atomically |
| 6  | build failure時の rollback semantics が固定 | ✅ | 2.10.4: B_world release, A_count unchanged |
| 7  | publish success時の token persistence が固定 | ✅ | 2.10.2-2.10.6: token persists R1..R7, A_count unchanged |
| 8  | double-acquire / double-release / leak / invalid-release を禁止 | ✅ (DESIGN-DEFINED) | 2.10.7: 4 prevention contracts. CODE-EVIDENCE MISSING |
| 9  | `A_max < ∞` の形式的十分条件を列挙 | ✅ | 2.11.1: 7 conditions |
| 10 | 各十分条件について code/design evidence を分類 | ✅ | 2.11.1: CODE-PROVEN / DESIGN-DEFINED / CODE-EVIDENCE MISSING |
| 11 | `A_max` と `K_world` の semantic distinction を維持 | ✅ | 2.12: rate vs level, independent invariants |
| 12 | 現行コードに Budget Authority が存在しないことを結論に反映 | ✅ | 2.2-2.7, 2.9.3, 2.11.1: all evidence → MISSING |

---

## 3. Step 3 Verdict

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

| 項目 | 状態 | Evidence |
| --- | --- | --- |
| `A = successful Lifetime Budget reservation acquire events` | **semantic definition OK / production code MISSING** | Step 2A confirmed: `reserveRuntimePublicationIdentity()` = identity issuance, not budget acquire |
| `R0→R1 Acquire event` | **CODE-EVIDENCE MISSING** — `src/` に 0件。I4 D48 で design location は定義されているが implementation は NO-GO | Step 2 §2.5-2.6, 3A-3B |
| `A_max` | **DESIGN-DEFINED** — production code にて実装/計測される mechanism が存在しない | 3E: `A_max < ∞` proof conditions classified |
| `A_count ≠ B_world` | **PASS** (3A-2) | rate (events/interval) ≠ level (outstanding tokens) |
| `A_count ≠ publish success` | **PASS** (3A-4) | `acquireObserved_` observational (D76.4), incremented at S3 enter not S0→R1 |
| `A_count ≠ worldId count` | **PASS** (3A-4) | `worldId` is S2 identity, not S0→S1 reservation |
| `A_count ≠ pendingIntentCount_` | **PASS** (3C) | Intent transport residency, not Lifetime Budget |
| `A_count ≠ publicationIntentResidencyCount_` | **PASS** (3C) | Publish Intent residency |
| `A_count ≠ reservationOwned` (Recovery) | **PASS** (3C) | Recovery durable admission |
| `A_max ≠ commitRuntimePublication count` | **PASS** (Step 2 §2.6 継承) | `commitRuntimePublication` is S1→S2 transfer facade |
| `A_max ≠ OwnerChannel 256` | **PASS** | ownership capacity, not lifetime budget |
| `A_max ≠ PendingPublishRegistry 64` | **PASS** | metadata ring buffer |
| `A_max ≠ WorldRetirementReservation design contract` | **PASS** | design contract exists but implementation NO-GO |
| `RejectedPressure ≠ A_max exhaustion` | **PASS** (3C-3, Step 2E) | reactive boolean backpressure, not proactive rate limit |
| Enforcement point | **DESIGN-DEFINED** | `BudgetAuthority::acquire()` before R0→R1 (3C-2) |
| Acquire/rollback contract | **DESIGN-DEFINED** | 2.10: Acquire/Release/Build fail/Shutdown/4 prevention contracts |
| `A_max < ∞` proof conditions | **CODE-EVIDENCE MISSING** | 2.11: 7 conditions, 5 DESIGN-DEFINED + 2 CODE-EVIDENCE MISSING |
| `A_max` and `K_world` distinction | **PASS** (3F) | rate (flow IN) ≠ level (IN/OUT balance) |
| I-W5 | **PARTIAL** (継承) | `M_world <= B_world` semantic model ok, code evidence missing |
| O7 | **SEMANTIC MODEL ONLY / DESIGN-DEFINED / CODE-EVIDENCE MISSING** | (継承) |

---

## 4. Step 4 への引継ぎ

Step 3 で確定した `A` の定義と証拠鎖を Step 4（P_max）へ継承する。特に、`A_max` が `commitRuntimePublication` / `OwnerChannel` / `PendingPublishRegistry` / `pendingIntentCount_` / `publicationIntentResidencyCount_` / `worldId` / publication count から **決して**導出してはならない境界をそのまま引き継ぐ。

`WorldRetirementReservation` design contract（D48-D53）は将来の `A_max` implementation candidate として記録するが、**実装されるまで A_max は DESIGN-DEFINED** である。

```text
D101-8 Step 3  A_MAX_DERIVATION_DESIGN_DEFINED
             / PRODUCTION-CODE-EVIDENCE MISSING

I-W5 = PARTIAL (継承)
O7   = SEMANTIC MODEL ONLY / DESIGN-DEFINED / CODE-EVIDENCE MISSING

A_max = future R (WorldRetirementReservation capacity) — T104 設計完了後
```
