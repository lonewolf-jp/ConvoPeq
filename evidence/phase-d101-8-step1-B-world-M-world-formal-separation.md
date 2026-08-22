# D101-8 Step 1 — B_world / M_world Formal Separation Audit (Evidence)

| 項目 | 内容 |
| --- | --- |
| **ID** | D101-8 Step 1 |
| **日付** | 2026-08-22 |
| **対象ブランチ** | `main` |
| **一次資料** | `ConvoPeq.md` (2026-08-22 latest source extract), `src/audioengine/RuntimeWorldAuthority.h`, `src/audioengine/OwnerChannel.h`, `src/audioengine/ISRRuntimeWorldAuthority.h`, `src/audioengine/RuntimeWorldAuthority.h`, `src/audioengine/DSPLifetimeManager.cpp`, `src/audioengine/ISRRetireRouter.h`, `src/audioengine/ISRRetire.h`, `src/audioengine/DeferredDeletionQueue.h`, `src/audioengine/RetireQuarantineStore.h`, `src/audioengine/ISRRuntimeSemanticSchema.h`, `src/core/RuntimeStore.h` |
| **前提** | D101-8 Step 0 verdict: `RECONCILED_WITH_OPEN_CONTRACT_ITEMS` — 6 contradictions resolved, O1-O7 open proof obligations |
| **目的** | `B_world(t) = S1 + S2 + S3 + S4 + S5 + S6` と `M_world(t) = S2 + S3 + S4 + S5 + S6` を形式的に分離し、`M_world(t) ≤ B_world(t)` と `B_world(t) ≤ K_world` をコード証拠に対応させる。S1-S6 の semantic object / enter event / leave event / authority をコードから cenus する。 |
| **制約** | **コード変更 0 / 契約変更 0 / 数値決定 0 / 仮定とコード証拠を明確分離**。`K_world` の数値導出は Step 6 まで保留。`K_world` は形式的上限として扱い、有限性は `A_max`/`K_terminal` 依存関係として後段で閉じる。 |
| **判定** | **CENSES COMPLETE / FORMAL BOUNDARIES CONFIRMED** — S1-S6 の event/authority/conservation 表をコード証拠で再抽出。Open: O4/O6 (container-entry ≠ RuntimeWorld cardinality, Step 6 局所問題)。`M_world ≤ B_world` は S1 token-only 差分により form 毎に成立。`B_world ≤ K_world` は K_world が有限上界の仮定の下で条件付き証明。 |

---

## 1. S1-S6 Lifecycle Census (ConvoPeq.md + コード照合)

### 1-1 World identity の生成点 (S1/S2 entry)

```text
RuntimeState::createForBuilder(BuilderToken token)  // AudioEngine.h:165-169
  → aligned_make_unique<RuntimeState>(token)        // builder token が唯一の生成権
  → worldId = builder stamp (RuntimeBuilder.cpp:113, AudioEngine.h:176)

BuilderToken は createForBuilder 以外存在しない（createForTest は test-only, AudioEngine.h:152-172）。
worldId (std::uint64) は authoritative identity、lifecycle 全体で不変。
```

**コード証拠**: `AudioEngine.h:149-176` (BuilderToken, createForBuilder, worldId field).
**Authority**: RuntimeBuilder (BuilderToken holder) → RuntimeWorldAuthority (enqueue recipient).

### 1-2 S1_Reserved → S2_Transferred: enter event

| State | Enter event | Code | Authority | World exists? | OwnerCount |
| --- | --- | --- | --- | --- | --- |
| S1_Reserved | reservation token stamped (B_world slot) | RuntimeBuilder.cpp:63 `reserveRuntimePublicationIdentity()`; AudioEngine.h:4508-4516 registerPublish before enqueue | Lifetime Budget Authority (admission gate) | NO (world not yet built) | 0 |
| S2_Transferred | `OwnerChannel::enqueue(key, std::move(world))` | AudioEngine.h:4519 | RuntimeWorldAuthority (ownerChannel_ value) | YES | 1 |

**Key**: S1 は reservation だけ（world なし）、S2 enter で `std::move(world)` が OwnerChannel に移る。
S1 で `B_world` に +1、`M_world` に +0（world が存在しないため）。

### 1-3 S2_Transferred → S3_Published: enter event

```text
S2: OwnerChannel resident (OwnerCount=1, RegistryCount=1)
  ↓ take(key) — OwnershipLocation 変更のみ、World count 不変 (I-W2)
S2: local owner (OwnerCount=1, RegistryCount=1)
  ↓ publish(std::move(owner), metadata) — authority.publish
S3: RuntimeStore::current (OwnerCount=1, RegistryCount=0 after unregister)
```

**コード証拠**:

- `OwnerChannel::take(key)` (OwnerChannel.h:90-107): single-transfer CAS-drain, returns `OwnerPtr(raw)` — ownership location change, world count invariant.
- `RuntimeWorldAuthority::publish()` (RuntimeWorldAuthority.h:220-255): `coordinator_.commit()` → `writeAccess_.publishAndSwap(next)` — sole physical swap (INV-X4-3).
- `registry().unregister(seqId)` (RuntimePublishExecutor.h:85) — RegistryCount → 0 **after** publish.

**Authority**: RuntimeWorldAuthority (WriteAccess owns RuntimeStore).

### 1-4 S3_Published → S4_Retiring: enter event (old world eviction)

`publishAndSwap` returns `oldWorld` (RuntimeWorldAuthority.h:249). The old S3 world becomes S4:

```cpp
auto* oldWorld = writeAccess_.publishAndSwap(next); // oldWorld → caller (retire path)
```

**Authority transition**: RuntimeWorldAuthority → ISRRetireRouter (via DSPLifetimeManager / retire).
**Code**: `RuntimeWorldAuthority.h:249` (publishAndSwap returns oldWorld); `DSPTransition.h:67` (`lifetime.retire(oldDSP)`); `ISRRetireRouter::enqueueRetire/enqueueWithRetry`.

### 1-5 S4_Retiring → S5_Quarantined: enter event

`DeferredDeletionQueue::enqueue(ptr, deleter, epoch)` (DeferredDeletionQueue.h:66-72) — 4096 slot bounded ring. Queue full → `quarantineRetire()` (ISRRetireRouter.h: overflow path, BUG-015/027).

**Code**: `DeferredDeletionQueue.h:66-89` (enqueue, kCapacity=4096); `ISRRetireRouter.h` `quarantineRetire` (退避ストア overflow).
**Authority**: ISRRetireRouter (owns DeferredDeletionQueue + RetireQuarantineStore).

### 1-6 S5_Quarantined → S6_Terminal: enter event

`quarantineRetire` → `RetireQuarantineStore` (Q 512 + E 512, RetireQuarantineStore.h:36-43). All stores exhausted (DeferredDeletionQueue + Quarantine + EmergencyQ) → `TerminalReclaimAuthority::enqueueWithRetry()` (ISRRetireRouter.h:62, GROWABLE `std::vector`).

**Code**: `RetireQuarantineStore.h:36-43` (QuarantinedEntry, 512+512); `ISRRetireRouter.h:62-77` (TerminalReclaimAuthority, growable).
**Authority**: ISRRetireRouter (TerminalReclaimAuthority as final owner).

> **Note on S5/S6 boundary**: The `QuarantinedEntry` (RetireQuarantineStore.h:37) and
> `TerminalReclaimAuthority::Entry` (ISRRetireRouter.h:68) carry `DeletionEntryType type`
> — `type == DeletionEntryType::World` distinguishes world entries from generic (DSP) entries.
> This is critical for O4/O6 (container entry ≠ RuntimeWorld cardinality).

### 1-7 S6_Terminal → S7_Released: enter event (reclaim)

`TerminalReclaimAuthority::drain()` / `drainAll()` (ISRRetireRouter.h:77-79) — executes deleter for
epoch-safe entries (`isOlder(entry.epoch, minReaderEpoch)`). `drainAll()` (shutdown) force-releases all.

**Code**: `ISRRetireRouter.h:77-79` (drain); AudioEngine.h:1494 `isFullyDrained()` / `waitForDrain(2000ms)`.
**Authority**: ISRRetireRouter (Non-RT context; audio thread joined for drainAll).

## 2. Failure / Shutdown Census (3D carry-forward + 3E resolution)

### 2-1 OwnerChannel drain on shutdown (S2 → terminal)

`OwnerChannel::drainAllNonRt(Fn&& reclaim)` (OwnerChannel.h:121-135): drains residual owners when
producer/consumer quiescent. Each raw `Owner*` handed to `reclaim` which MUST transfer to retire
authority (NOT delete). Caller contract: enqueue/take MUST be quiescent.

**Code**: `OwnerChannel.h:110-135` (drainAllNonRt, single-transfer pattern same as take).
**Authority**: AudioEngine.shutdown (orchestrates drainAllNonRt → ISRRetireRouter).

### 2-2 Registry orphan on shutdown-drain-without-executePublish (3D-8 gap)

If shutdown drains OwnerChannel via `drainAllNonRt` WITHOUT running `executePublish`:

- World pointer is reclaimed by `drainAllNonRt`'s `reclaim` → ISRRetireRouter (enters S4).
- BUT `PendingPublishRegistry` entry (registered at AudioEngine.h:4516) is NOT `unregister`'d —
  `unregister` is called in `executePublish` (RuntimePublishExecutor.h:85), which never runs.

**Result**: stale registry entry (seqId → dangling world pointer) remains until slot overwrite
(cursor modulo 64, ISRRuntimeWorldAuthority.h:46).

**Resolution for 3E closure**: this is safe because (a) seqId is monotonic + never reused in-session
(3B-2, 3C-B2-B), (b) the stale entry is `const void*` non-owning metadata that is never dereferenced
(registry lookup is fallback-only, RuntimePublishExecutor.h:60-63: only reached on `take(key)==null`),
(c) shutdown prevents new `executePublish` calls. The stale entry is overwritten on next seqId cycle
(cursor modulo). **No mis-binding occurs** — the registry cannot cause O→E confusion because it
doesn't participate in the binding (it is Layer B metadata, not the binding carrier).

## 3. S1-S6 Census Table (formal)

| State | Semantic object | Enter event | Exit event | Authoritative holder | World exists? (M_world) | Budget token (B_world) | Ownership |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S0_Available | free budget slot | session start / S7 reclamation | reservation | Lifetime Budget Authority | NO | 0 | None |
| S1_Reserved | budget reservation token | `reserveRuntimePublicationIdentity()` | build/start enqueue | Lifetime Budget Authority (admission) | NO | 1 | None (world not built) |
| S2_Transferred | RuntimeState (frozen) | `OwnerChannel::enqueue(key, std::move(world))` | `take(key)` → publish | RuntimeWorldAuthority (OwnerChannel) | YES | 1 | OwnerChannel (1) |
| S3_Published | RuntimeState (current) | `publishAndSwap(next)` | eviction (oldWorld returned) | RuntimeWorldAuthority (RuntimeStore) | YES | 1 | RuntimeStore (1) |
| S4_Retiring | RuntimeState (retired) | `publishAndSwap` returns oldWorld → retire | reclamation or quarantine | ISRRetireRouter (DeferredDeletionQueue) | YES | 1 | RetireChain (1) |
| S5_Quarantined | RuntimeState (quarantined) | DeferredDeletionQueue full → quarantineRetire | terminal overflow or drain | ISRRetireRouter (RetireQuarantineStore) | YES | 1 | Quarantine (1) |
| S6_Terminal | RuntimeState (terminal pending) | all stores full → TerminalReclaimAuthority | epoch-safe drain | ISRRetireRouter (TerminalReclaimAuthority) | YES | 1 | Terminal (1) |
| S7_Released | reclaimed | drain (epoch-safe deleter) | slot reuse (S0) | ISRRetireRouter (drainAll) | NO | 0 | None |

**Authority single-hat**:

- S1: Lifetime Budget Authority (admission/reservation)
- S2-S3: RuntimeWorldAuthority (sole physical store, INV-X4-3/INV-X4-5)
- S4-S7: ISRRetireRouter (TerminalReclaimAuthority as final owner)

## 4. M_world ≤ B_world — Formal Proof (Layer A)

### 4-1 Definition

```text
M_world(t) = |{ W : lifecycle(W, t) ∈ {S2, S3, S4, S5, S6} }|
           = count of distinct RuntimeWorld identity with live body

B_world(t) = |{ b : state(b, t) ∈ {S1, S2, S3, S4, S5, S6} }|
           = count of budget reservation tokens in Reserved..Terminal

b (budget token) と W (RuntimeWorld) の対応:
  S0: b=0, W=0
  S1: b=1, W=0  (token exists, world not yet built)
  S2-S6: b=1, W=1 (1:1 correspondence)
  S7: b=0, W=0 (both reclaimed)
```

### 4-2 Proof

**I-W1** (lifecycle uniqueness): `lifecycle(W, t)` は常に1状態 (I-W1, Step 1 formalization.md:49).
同一 World は複数 state に同時に属さない。

**I-W2** (OwnerCount ∈ {0,1}): OwnerChannel single-transfer (OwnerChannel.h: take-once CAS),
`publishAndSwap` は owner.release() で唯一の transfer (RuntimeWorldAuthority.h:248).
`drainAllNonRt` single-transfer pattern (OwnerChannel.h:126-128).

**I-W3-I-W4** (registry isolation): RegistryCount は M_world/B_world に加算しない
(PendingPublishRegistry = non-owning const void* metadata, ISRRuntimeWorldAuthority.h:25-32)。
`M_world ≠ OwnerChannel.size()`、`M_world ≠ Registry.size()`、二重計上しない。

**M_world ≤ B_world**: S1 で `b=1, W=0` となるため、`B_world` は `M_world` より常に ≥。
各 S2-S6 では `b=1, W=1` (1:1)。S0/S7 では `b=0, W=0`。
∴ `M_world(t) = Σ S2..S6(W=1) ≤ Σ S1..S6(b=1) = B_world(t)` ∎

**コード証拠**:

- S1 token-only: `reserveRuntimePublicationIdentity()` returns identity (RuntimeBuilder.cpp:63),
  world built AFTER reservation (RuntimeBuilder.cpp:113 stamp is on the world, not the reservation token).

- S2-S6 world+token 1:1: `createForBuilder` → enqueue (S2) → publishAndSwap (S3) → publishAndSwap
  returns oldWorld → retire (S4) → quarantine/terminal (S5/S6) → drain (S7). 1 world per 1 token.

## 5. B_world ≤ K_world — Conditional Proof (K_world as formal upper bound)

```text
K_world は B_world の admissible maximum。
K_world < ∞ かつ outstandingWorlds ≤ K_world を保証する。
```

**K_world の存在するが、その値は Step 5-3 (A_max / P_max) により導出される。**
Step 1 では K_world の具体値を決めない — 形式的上界の存在を仮定する。

### 5-1 Budget enforcement point (Authority)

**コード証拠**: AudioEngine.h コメント (ConvoPeq.md:2003)

```text
commitRuntimePublication facade → OwnerChannel → IntentQueue → CoordinatorLoop →
```

Budget reservation は admission gate (PublicationAdmission::evaluate) で行われ、`trySubmitImpl`
が `Accepted` を返す前に `reserveRuntimePublicationIdentity()` が呼ばれる (RuntimeBuilder.cpp:63,
RuntimePublicationOrchestrator.cpp:90-96)。`RejectedPressure` (PublicationAdmission.h:41) は
budget exhausted 時の admission rejection — これが `B_world ≤ K_world` の enforcement を保証する。

### 5-2 Conditional proof statement

```text
前提: K_world = admissible maximum of B_world (defined in Step 5-3, value deferred)
      budget enforcement at admission gate (PublicationAdmission::evaluate → RejectedPressure)

B_world(t) ≤ K_world
    iff admission gate rejects new reservations when B_world == K_world

この前提の下、B_world(t) ≤ K_world < ∞ は成立する。
K_world の有限性（K_world < ∞）は A_max / K_terminal 等の依存関係で Step 5-3 で証明する。
Step 1 では K_world := admissible maximum of B_world として仮定する。
```

**Verdict**: `B_world ≤ K_world` は **条件付き証明** — K_world が有限上界として仮定され、budget
enforcement (admission gate) が B_world を K_world にクリップする。K_world の有限性は Step 5+ で閉じる。

## 6. S1-S6 Conservation — state transition invariant

### 6-1 Ownership conservation across transitions

| Transition | OwnerCount before | OwnerCount after | RegistryCount before | RegistryCount after | M_world | B_world |
| --- | --- | --- | --- | --- | --- | --- |
| S1→S2 (enqueue) | 0 | 1 (OwnerChannel) | 0 | 1 | +1 | invariant |
| S2→S2 (take, OwnerChannel→local) | 1 | 1 (local) | 1 | 1 | 0 | 0 |
| S2→S3 (publish+swap) | 1 (OwnerChannel) | 1 (RuntimeStore) | 1 | 0 (unregister after) | 0 | 0 |
| S3→S4 (publishAndSwap oldWorld) | 1 (RuntimeStore) | 1 (RetireChain) | 0 | 0 | 0 | 0 |
| S4→S5 (quarantine overflow) | 1 (Deferred) | 1 (Quarantine) | 0 | 0 | 0 | 0 |
| S5→S6 (terminal overflow) | 1 (Quarantine) | 1 (Terminal) | 0 | 0 | 0 | 0 |
| S6→S7 (drain) | 1 (Terminal) | 0 | 0 | 0 | -1 | -1 |

**Conservation**: OwnerCount は 0↔1 の間で移動するだけ、2 にはならない (I-W2)。
M_world は S1 で +0 (world なし), S2-S6 で 0 (location change only), S6→S7 で -1 (reclaim)。
B_world は S0→S1 で +1 (token), S6→S7 で -1 (token release)。

### 6-2 take() is ownership-location change, NOT world-count change

`OwnerChannel::take(key)` (OwnerChannel.h:90-107): `consumeAtomic(s.owner, acquire)` → `publishAtomic(s.owner, nullptr, release)` → re-wrap `OwnerPtr(raw)`.
Same `raw` pointer re-wrapped — **world identity unchanged, OwnerCount unchanged** (still 1, just relocated).

This is the **critical invariant** for `M_world ≤ B_world`: `take` does not create or destroy worlds.

## 7. S1-S6 Open Proofs (O4/O6 carry-forward)

| 項目 | 判定 | 理由 |
| --- | --- | --- |
| O4: M_world_S4 ≤ D_entry (DeferredDeletionQueue resident) | OPEN | DeletionEntry は `type == World` か `Generic` かで分離。`D_entry` は総 entry 数。`M_world_S4` は world entry 数のみ。`M_world_S4 ≤ D_entry` は成立するが `M_world_S4 ≤ 4096` を証明するには world-only capacity が 4096 であることを別途証明必要 → Step 6 |
| O6: M_world_S5 ≤ N_Q + N_E (Quarantine) | OPEN | 同様に `QuarantinedEntry` は world/generic で分離。container cardinality ≠ RuntimeWorld cardinality → Step 1 で定義を固定、Step 6 で証明 |
| O5: M_world_S6 ≤ N_T (Terminal) | OPEN | TerminalReclaimAuthority は growable (std::vector)。`K_terminal < ∞` は D101-9 で仮定/証明 |

**3E closure position**: O4/O6 は Step 1 の M_world 定義そのものに影響するが、`M_world ≤ B_world` の proof には影響しない (B_world は token-based, container-entry-independent)。O4/O6 は `B_world ≤ K_world` (有限性) の証明に関わる Step 5-6 の局所問題。

## 8. Layer Separation (A/B/C) — confirmed against code

| Layer | Quantity | Code backing | Container |
| --- | --- | --- | --- |
| **A** (lifetime semantic) | B_world, M_world, M_world_S2..S6 | RuntimeState identity (worldId), lifecycle state | — (logical count) |
| **B** (physical/container) | OwnerChannel occupancy (256), Registry (64), D_entry (4096), Q_entry (512+512), T_entry (growable) | OwnerChannel::kCapacity, kPendingPublishCapacity, DeferredDeletionQueue::kCapacity, RetireQuarantineStore | Bounded FIFO / growable vector |
| **C** (rate-control) | A_count, P_count | RuntimePublicationState::publishedWorldCount_/rejectedCount_ (atomic counters) | Sliding-window accounting |

**Code confirmation**:

- OwnerChannel kCapacity=256: `OwnerChannel.h:42`
- Registry kPendingPublishCapacity=64: `RuntimeWorldAuthority.h:35`
- DeferredDeletionQueue kCapacity=4096: `DeferredDeletionQueue.h` (66 lines)
- RetireQuarantineStore: `RetireQuarantineStore.h:37` (QuarantinedEntry, 512+512)
- TerminalReclaimAuthority growable: `ISRRetireRouter.h:62` (std::vector, P-4: ALWAYS accepts)

## 9. Step 1 Verdict

| Item | Status |
| --- | --- |
| S1-S6 census (object/event/authority) | ✅ COMPLETE (code-backed, §1-2) |
| Ownership Location Table | ✅ CONFIRMED (3C-3 formalization matches code, §6-1) |
| Registry isolation (Layer B) | ✅ CONFIRMED (non-owning const void*, §4-2) |
| S2 transient ownership (take = location change, not count) | ✅ CONFIRMED (OwnerChannel.h:90-107) |
| distinct identity (worldId) | ✅ CONFIRMED (AudioEngine.h:176, unique per build) |
| M_world ≤ B_world | ✅ PROVEN (S1 token-only gap, §4-2) |
| B_world ≤ K_world | ⚠️ CONDITIONAL (K_world := admissible upper bound, §5-2) |
| M_world_S4 ≤ D_entry | OPEN (O4, Step 6) |
| M_world_S5 ≤ N_Q + N_E | OPEN (O6, Step 6) |
| M_world_S6 ≤ N_T | OPEN (O5/O6, D101-9 K_terminal < ∞) |
| Failure/shutdown conservation | ✅ CONFIRMED (3D carry-forward, §2) |
| Registry stale-entry on shutdown-drain | ✅ CONTAINED (seqId monotonic, fallback-only, §2-2) |

**Verdict**: **CENSES COMPLETE / FORMAL BOUNDARIES CONFIRMED** — `M_world ≤ B_world` form 毎に証明済み。
`B_world ≤ K_world` は K_world を有限上界として仮定すれば条件付き証明。O4/O6 は Step 6 (A_max/K_terminal
derivation) の局所問題として保留。Step 2 (Reservation Token Semantics) への準備完了。
