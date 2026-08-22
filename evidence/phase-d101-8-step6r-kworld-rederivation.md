# D101-8 Step 6R — K_world Re-Derivation

> **Status**: COMPLETE — K_world re-derived with **K_transferred = 256** correction (not 65).
> Step 5 5D theorem carry-forward: P contributes 0 additive worlds.
> **P_max = 4098 は K_world の項として使用しません**。
> **コード変更: なし** — audit only (re-derivation of existing evidence file).
> Date: 2026-08-22
> Verified tools: WSL rg/ast-grep/sed/awk/fdfind/fd/ag/fzf, serena MCP, AiDex MCP, graphify, semble, cocoindex/ccc (init required), context-mode MCP (ctx_batch_execute 8 parallel WSL census), headroom.

---

## 6A — World state taxonomy (production code, disjoint S0–S7)

```text
S0 Available    — free capacity (world不存在)
S1 Reserved     — reservation token acquired (design contract; code MISSING → A_max assumption)
S2 Transferred  — OwnerChannel (256 owning) + PendingPublishRegistry (64 non-owning metadata gap)
S3 Published    — RuntimeStore::current (1, atomic<RuntimeState*>, single-store publishAndSwap)
S4 Retiring     — DeferredDeletionQueue (4096, epoch-gated FIFO reclaim via isOlder)
S5 Quarantined  — RetireQuarantineStore (512) + EmergencyQuarantineStore (512); ownership-transfer overflow
S6 Terminal     — TerminalReclaimAuthority (growable std::vector, store() ALWAYS true)
S7 Released     — deleter executed (non-counting; budget returned)
```

**Disjoint ownership**: ∀ world, ∃! at most one of S1..S6 at any time; S7 = destroy (count 0).
Transitions: `S1→S0 (rollback)`, `S2→S3 (publishAndSwap)`, `S3→S4 (enqueueRetire)`, `S4→S5 (quarantine)`, `S5→S6 (emergencyQuarantine/terminalReclaim)`, `S6→S7 (drain/drainAll)`, `S2→S7 (enqueueFail→take reclaim)`, `S4→S2 (dequeue→publish)` はない (commit は S4 から戻らない)。

> ⚠️ **S2 ownership is OwnerChannel-only**: PendingPublishRegistry (64) は `const void* sealedWorld` (non-owning read-only metadata)。World ownership は OwnerChannel が 1 つ保持する。Step 5 5D theorem と一致。

---

## 6B — Ownership census (production code verified)

| Container | owning / non-owning | capacity | producer | consumer | transfer semantics | failure semantics |
|---|---|---|---|---|---|---|
| **OwnerChannel** | **owning** (aligned_unique_ptr<const RuntimeState>) | **256** (h:41) | `enqueue(key, std::move(world))` (AudioEngine.h:4559) | `take(key)` (RuntimePublishExecutor.h:23) | SPSC, single-transfer, key-isolation, take drains once (CAS→nullptr) | store false (full/duplicate) → caller retains (AudioEngine.h:4568 rollback) |
| **PendingPublishRegistry** | **non-owning** (seqId→`const void*` sealedSnapshot) | 64 (RuntimeWorldAuthority.h:34) | `registerPublish(seqId, sealedWorld)` (AudioEngine.h:4564) | `publishAndSwap` success → unregister; lookup fallback (h:28) | lock-free cursor, oldest-overwrite | full → overwrite oldest cursor (metadata only, NO world leak) |
| **RuntimeStore::current** | owning (atomic<RuntimeState*>) | **1** | `publishAndSwap(next)` via WriteAccess (RuntimeStore.h:58) | `observe()` (read-only borrow, RuntimeStore.h:79) | single acq_rel exchange; old=retire target, next=publish | Faulted → old destroyed in publish() (RuntimePublishExecutor.h:82-88) |
| **DeferredDeletionQueue** | owning (DeletionEntry) | 4096 (DeferredDeletionQueue.h:262, kQueueSize) | `enqueueRetire`/`enqueueWithRetry` (ISRRetireRouter.cpp:282) | `reclaim(minReaderEpoch)` (Def-145) / `drainAllUnsafe` (shutdown) | Vyukov MPMC ring, FIFO head-block reclaim via isOlder | full → retry+tryReclaim → Q |
| **RetireQuarantineStore (Q)** | owning (QuarantinedEntry) | 512 (RetireQuarantineStore.h:65, kMaxQuarantinedEntries) | `quarantine()` via enqueueWithRetry Stage 3 | `drain(minReaderEpoch, isOlderFn)` | array+size (std::array), mutex | full → EmergencyQ |
| **EmergencyQuarantineStore (E)** | owning (QuarantinedEntry) | 512 (same type) | `emergencyQuarantine()` Stage 4 | `drainEmergencyAndTerminal` | array+size, mutex | full → Terminal |
| **TerminalReclaimAuthority** | owning (Entry) | **growable** (`std::vector<Entry>` ISRRetireRouter.h:47) | `store()` Stage 5 | `drain()` / `drainAll()` (shutdown) | push_back | **no store-full** (growable) |
| **Reservation domain** | non-owning token (design) | A_max (design, code MISSING) | admission | rollback / publish-commit | token→world ownership transfer | rollback → token destroyed (0 worlds) |

> **PendingPublishRegistry non-owning — re-proof**: `registerPublish(PublicationSequenceId seqId, const void* sealedWorld)` は `const void*` を受け取る — **non-owning metadata**。World deleter は呼ばれない。unregister は metadata を消すのみ。 → Step 5 5H-1 Tier-4 resolution と一致。**64 は world count に加算しない**。

---

## 6C — OwnerChannel = 256 (Step 5 carry-forward, re-verified)

`OwnerChannel.h:41`:
```cpp
static constexpr std::size_t kCapacity = 256;              // >> max in-flight publishes
```

**Single-transfer**: `take()` は `publishAtomic(s.owner, nullptr, release)` で CAS 成功時のみ nullptr にする — single-transfer, key-isolation probe (h:60-108 re-read). SPSO: sole producer = `commitRuntimePublication` path (AudioEngine.h:4559); sole consumer = `CoordinatorLoop processIntent → executePublish → authority.ownerChannel().take(key)` (RuntimePublishExecutor.h:23).

Step 5 5D theorem carry-forward:
```text
P_queue(t) ≤ |worlds in OwnerChannel(t)| ≤ 256  ← S2 owning bound
Intent residency → non-owning pointer → OwnerChannel-owned World
```

**6C verdict**: `|S2(OwnerChannel)| ≤ 256` — CODE-PROVEN (owning, fixed, single-transfer).

---

## 6D — RuntimeStore / current contribution (K_current ≤ 1)

`RuntimeStore.h`: `std::atomic<T*> current` (single pointer, capacity=1). `publishAndSwap`:
```cpp
return exchangeAtomic(store_->current, next, std::memory_order_acq_rel);
```
Single `acq_rel exchange` — old = retire target (→S4), next = publish to S3. INV-X4-3 (RuntimeWorldAuthority.h:88): **publishAndSwap is RuntimeWorldAuthority-owned WriteAccess only**. No other write-capable RuntimeStore exists (INV-X4-5).

Ownership transfer chain S3→S4: `publishAndSwap` returns `oldWorld` → `willRetireRuntimeNonRt(oldWorld)` → `retirePublishedRuntimeWorldNonRt` → `enqueueDeferredDeleteNonRt(..., DeletionEntryType::World)` → `enqueueWithRetry` (D queue) — ownership handed to DeferredDeletionQueue. **current world は OwnerChannel の 256 と重複しない** (take 後 publish, single-transfer): S2 の world が `take` で consume → `publish` で S3 に進み、同時に OwnerChannel slot は nullptr になる。

**6D verdict**: `K_current ≤ 1` — CODE-PROVEN.

---

## 6E — DeferredDeletionQueue (S4, D↔Q↔E↔T conservation)

`DeferredDeletionQueue.h:262`: `kQueueSize = 4096` (fixed array, `alignas(64) std::array<DeletionEntry, kQueueSize>`). `DeferredDeletionQueue.h:279-310` re-read: reclaim は `isOlder(entry.epoch, minReaderEpoch)` が真の **先頭ブロックのみ** 回収 (FIFO head-block)。`drainAllUnsafe` (h:324) は shutdown専用 (epoch判定無視)。

### D→Q→E→T ownership transfer chain (ISRRetireRouter.cpp:282, re-verified)

```text
Stage 1: enqueueRetire(ptr) → D (DeferredDeletionQueue)
    if full / QueuePressure:
        retry: tryReclaim() → drainEmergencyAndTerminal() → retry enqueueRetire (kMaxRetry=2)
        if still full/QueuePressure:
Stage 3: m_retireQuarantine.quarantine(ptr) → Q
    if full (false return):
Stage 4: m_emergencyQuarantine.quarantine(ptr) → E
    if full:
Stage 5: terminalReclaim(ptr) → T (TerminalReclaimAuthority::store)
    store() ALWAYS true (growable)  ← ownership は必ず次authorityへ移転
```

**P-4 invariant** (ISRRetireRouter.cpp:40 commentary, re-read): "Ownership invariant: ptr を手放す前に、必ず次の authority に ownership が移る。assert(false) → return という経路は残さない。"

`store()` (ISRRetireRouter.cpp:40-49):
```cpp
bool TerminalReclaimAuthority::store(void* ptr, void (*deleter)(void*),
                                     uint64_t epoch, DeletionEntryType type,
                                     const char* reason) noexcept
{
    if (ptr == nullptr || deleter == nullptr)
        return true;  // no-op は成功扱い
    std::lock_guard<std::mutex> lock(mtx_);
    entries_.push_back(Entry{ptr, deleter, epoch, type, reason});
    residentAtomic_.fetch_add(1, std::memory_order_release);
    return true;  // ★ P-4: growable — ALWAYS accepts
}
```

**6E verdict**: D→Q→E→T は **ownership transfer** (non-owning copy なし)。各 stage は成功時のみ ownership を受領。`store() ALWAYS true` により Terminal は最後の fallback。**同一 world が D/Q/E/T に同時に存在することはない** — transfer は move-semantics。

### Reclaim / drain epoch-gating (conservation)

- D `reclaim`: `isOlder(entry.epoch, minReaderEpoch)` 真のみ deleter 実行 (DeferredDeletionQueue.h:148,199)
- Q `drain`: `isOlder(entry.epoch, minReaderEpoch)` 真のみ (RetireQuarantineStore.h:140,177)
- E `drainEmergencyAndTerminal`: 同じ predicate
- T `drain`: `isOlderFn(e.epoch, minReaderEpoch)` 真のみ (ISRRetireRouter.cpp:57)
- T `drainAll` (shutdown): epoch 判定無視 (all entries deleted)

**onRelease gating**: `DeletionEntryType::World` のみ `referenceObserver_->onRelease()` を通知 (DeferredDeletionQueue.h:148-153, RetireQuarantineStore.h:140-145, ISRRetireRouter.cpp:57-62)。Step 5 5H-1 Tier-4 verified.

---

## 6F — Quarantine / EmergencyQuarantine conservation (512 + 512)

`RetireQuarantineStore.h:65`:
```cpp
static constexpr std::size_t kMaxQuarantinedEntries = 512;
std::array<QuarantinedEntry, kMaxQuarantinedEntries> entries_;  // fixed, allocation-free
```

`quarantine()` (h:70-105): `false` (full) 時 caller は deleter を実行せず (UAF排除), 次 stage へ transfer。`overflowCount_` increment (health escalation)。E は同型 (512)。

**K_quarantine = 512 + 512 = 1024** — fixed, CODE-PROVEN.

### D→Q→E→T 同一 World 多重計上禁止 (conservation)

```text
World W が D から開始して T に到達する遷移:
D.enqueueRetire(W)  → W ∈ D   (D.count += 1)
  if full: D.dequeue(W) は起きない (FIFO head block) ─ retry は tryReclaim で D 内回収を試みる
  if Q へ transfer: W は D から手放され Q に move (D.count -= 1, Q.count += 1)  ← DからQへの ownership transfer
  if E へ transfer: W は Q から手放され E に move (Q.count -= 1, E.count += 1)  ← 同上
  if T へ transfer: W は E から手放され T に push_back (E.count -= 1, T.count += 1)  ← 同上
T.drain/drainAll(W) → deleter(W) 実行, T.count -= 1, S7 → counting 対象外
```

各 transfer は **move semantics** (ptr handoff): D.dequeue は CAS dequeuePos++ で slot を空にする (DeferredDeletionQueue.h:148)。Q.quarantine は `entries_[writeIndex]` へ move + writeIndex++。E 同じ。T.store は `entries_.push_back`。**いずれの stage でも W は前の container から除去される** — double counting しない。

> ⚠️ 既存 Step 6 6-E の "S2 capacity 64 で評価" は **stale**。Step 5 5D theorem: P は additive に 0。S2 owning bound = OwnerChannel(256), PendingPublishRegistry は non-owning。

---

## 6G — TerminalReclaimAuthority (K_terminal — growable)

`ISRRetireRouter.h:47`: `std::vector<Entry> entries_` + mutex. `store()` always true (re-verified §6E)。**K_terminal は growable = bounded でない** (heap growth depends on system memory)。

```text
K_terminal < ∞   は current production code から PROVEN できない
                 （growable store, no max cap）
```

D101-9 (bounded Terminal実装) 予定済み — Step 6R では **ASSUMPTION** と明記:
```text
A2: K_terminal < ∞   (D101-9 bounded implementation 予定)
```

> ✅ `store() ALWAYS true` により **world leak はない** (ownership は必ず T に移転)。K_world < ∞ が block されるのは **bounded** であるかという existence 証明であり、leak 証明ではない。

---

## 6H — A_max / Reservation token (S1, design-defined)

Step 3 carry-forward. Reservation token (`WorldRetirementReservation`) は **non-owning design contract** — `src/` 未実装。S1 Reserved は Lifetime Budget reservation (design)であり World ownership ではない。`A_max < ∞` は assumption。

**reservation token ≠ World ownership** を明確に分離:
```text
A_max = reservation-obligation bound (design, code MISSING)  → K_reserved(S1)
K_world は World lifetime (owning) のみを count
```
A_max は K_world の直接項ではない (reservation → world transfer で S1 が S2 に移行, S1 count は world count と同期しない design-defined obligation)。

**6H verdict**: `K_reserved(S1) = A_max < ∞` — CONDITIONAL (design assumption, D101-9 implementation pending)。

---

## 6I — K_world conservation equation (symbolic, disjoint)

Step 5 5D theorem + Tier-4 resolution を反映した **pairwise-disjoint** conservation:

```text
K_world(t) = |S1(t)| + |S2(t)| + |S3(t)| + |S4(t)| + |S5(t)| + |S6(t)|
           = K_reserved + K_ownerChannel + K_current + K_retire + K_quarantine + K_terminal
```

| Term | Domain | Symbolic bound | Status |
|---|---|---|---|
| K_reserved | S1 (reservation token, design) | A_max | CONDITIONAL (code MISSING) |
| K_ownerChannel | S2 (owning slot, single-transfer) | ≤ 256 | CODE-PROVEN (OwnerChannel.h:41) |
| K_current | S3 (RuntimeStore::current) | ≤ 1 | CODE-PROVEN (RuntimeStore.h, single atomic ptr) |
| K_retire | S4 (DeferredDeletionQueue) | ≤ 4096 | CODE-PROVEN (kQueueSize=4096) |
| K_quarantine | S5 (Q+E) | ≤ 1024 | CODE-PROVEN (512+512, fixed) |
| K_terminal | S6 (TerminalReclaimAuthority) | growable | ASSUMPTION (D101-9) |

### K_reader (stranding) — disjoint? YES

Step 5/6-F 6-G carry-forward: `K_reader` は **独立した container ではない** — S4/S5 内の epoch-safe でない entry の stranding を表す。**stranding は capacity を超過しない** (FIFO head-block: reader 1 つでも古い epoch を保持すると全後続 S4/S5 entry が block されるが、その数 ≤ S4+S5 capacity = 5120)。

tight bound interpretation: `K_reader ⊆ K_retire + K_quarantine` (stranded worlds はすべて S4/S5 内)。保守的に独立加算する場合 (conservative bound), `K_reader = reader_count(2) × E_max_message × 1` だが `E_max_message` は **current production code では fixed bound を導出できない** (Step 6-G: publish throughput は固定 bound を持たない)。

```text
E_max_audio ≤ 1        CONDITIONAL (topology-dependent, Step 5/6-G)
E_max_message          UNBOUNDED  (no fixed publish throttle, Step 6-G)
```

### Final conservation

```text
K_world ≤ A_max + 256 + 1 + 4096 + 1024 + K_terminal
      = A_max + 5377 + K_terminal    （K_reader は S4/S5 内に contain する tight bound）
```

Disjointness verified: S1(token), S2(OwnerChannel owning slot), S3(current ptr), S4(D ring), S5(Q+E array), S6(T vector) — **各 world は遷移時に Move され、いずれの container にも 2 重で乗ることはない** (FIFO reclaim, single-transfer take, ownership handoff D→Q→E→T, publishAndSwap single exchange)。

---

## 6J — Classification table (Step 5 correction reflected)

| Item | Value | Classification | Evidence |
|---|---|---|---|
| **OwnerChannel = 256** (owning, S2) | 256 | **CODE-PROVEN** | OwnerChannel.h:41 kCapacity=256; SPSC single-transfer take |
| **PendingPublishRegistry = 64** (non-owning, S2 metadata gap) | 64 non-owning | **CODE-PROVEN (non-owning)** | RuntimeWorldAuthority.h:34,34 `const void* sealedWorld` |
| **PendingPublishRegistry → World count** | 0 | **CODE-PROVEN (contribution 0)** | non-owning pointer; no deleter; Step 5 5H-1 |
| **P → K_world additive contribution** | 0 | **CODE-PROVEN (Step 5 5D)** | P_queue ⊆ OwnerChannel(256) ⊆ K_transferred; f(P)=0 |
| **P_max = 4098** as K_world term | N/A | **EXCLUDED** | 5D theorem: contribution = 0; P は intent-slot bound |
| RuntimeStore current = 1 | 1 | **CODE-PROVEN** | RuntimeStore.h: atomic single ptr; INV-X4-3 single write access |
| DeferredDeletionQueue | 4096 | **CODE-PROVEN (fixed)** | DeferredDeletionQueue.h:262 kQueueSize |
| RetireQuarantineStore | 512 | **CODE-PROVEN (fixed)** | RetireQuarantineStore.h:65 kMaxQuarantinedEntries |
| EmergencyQuarantineStore | 512 | **CODE-PROVEN (fixed)** | same type, 2nd instance |
| D→Q→E→T ownership conservation | transfer-only | **CODE-PROVEN** | ISRRetireRouter.cpp:282-360; P-4 invariant |
| A_max finite | A_max | **CONDITIONAL** (design, code MISSING) | Step 3 |
| K_terminal finite | growable | **ASSUMPTION (D101-9)** | ISRRetireRouter.h:47 std::vector; store() always true |
| E_max_audio ≤ 1 | 1 | **CONDITIONAL (topology)** | Step 6-G |
| E_max_message bounded | — | **UNBOUNDED (current code)** | Step 6-G |
| K_reader finite | stranding | **CONTAINED in K_retire+K_quarantine (tight) / CONSERVATIVE OPEN** | Step 5/6-G |
| **K_world < ∞** | A_max+5377+K_terminal | **CONDITIONAL GO (tight) / NO-GO (conservative)** | tight bound: K_terminal<∞ + A_max<∞ assumptions |

---

## K_world final bound (tight, Step 5 corrected)

```text
K_world ≤ K_reserved + K_ownerChannel + K_current + K_retire + K_quarantine + K_terminal
       ≤ A_max + 256 + 1 + 4096 + 1024 + K_terminal
       = A_max + 5377 + K_terminal    (K_reader ⊆ S4/S5, tight)

< ∞  IF  A_max < ∞  AND  K_terminal < ∞  (both: D101-9 assumptions)
```

| Interpretation | K_world < ∞? | Reason |
|---|---|---|
| **Tight** (K_reader contained in S4/S5) | ✅ **CONDITIONAL GO** | A_max<∞ + K_terminal<∞ assumptions (D101-9) |
| **Conservative** (K_reader independent, E_max_message bounded) | ❌ **NO-GO** | E_max_message unbounded in current code |

**Step 6R final verdict**: `K_world < ∞` is **CONDITIONAL GO** under tight-bound interpretation (Step 5 5D theorem + disjoint S-domain conservation + D101-9 assumptions A_max<∞/K_terminal<∞). The `K_transferred = 65 → 256` correction and the `P contributes 0 additive` theorem are verified against production code. The conservative interpretation remains NO-GO due to `E_max_message` unbounded.

---

## Stale-value correction log (Step 6 → Step 6R)

| Item | Existing Step 6 (stale) | Step 6R (corrected) | Basis |
|---|---|---|---|
| K_transferred | `OwnerChannel(1) + PendingPublishRegistry(64) = 65` | **`OwnerChannel(256 owning) + PendingPublishRegistry(64 non-owning, 0 world contribution)`** | OwnerChannel.h:41; Step 5 5D; RuntimeWorldAuthority.h:43 `const void*` |
| S2 world bound | 65 | **≤ 256** (OwnerChannel owning) | Step 5 5D theorem |
| P → K_world | "S2 capacity 64で評価" | **P additive contribution = 0** (f(P)=0) | Step 5 5D: P_queue ⊆ OwnerChannel ⊆ K_transferred |
| P_max=4098 in K_world | implicit in 6-E | **EXCLUDED** (P is intent-slot bound, not world budget) | Step 5 5D/5E |
| K_reader independent term | tight bound contains; conservative independent | **same** (stranding ≤ S4+S5 capacity; conservative: E_max_message unbounded) | DeferredDeletionQueue.h FIFO; EpochDomain.h reader model |

---

## Task Completion Checklist (Step 6R)

| Task | Content | Status |
|---|---|---|
| Task 1 | World state taxonomy (S0-S7) from production code | ✅ 6A |
| Task 2 | Disjoint ownership across containers | ✅ 6B, 6I |
| Task 3 | PendingPublishRegistry non-owning re-proof (const void*) | ✅ 6B, 6H |
| Task 4 | OwnerChannel = 256 re-verified (kCapacity, single-transfer) | ✅ 6C |
| Task 5 | RuntimeStore current = 1 (single atomic ptr, publishAndSwap) | ✅ 6D |
| Task 6 | DeferredDeletionQueue = 4096 (fixed, FIFO epoch-gated) | ✅ 6E |
| Task 7 | Q(512) + E(512) = 1024 (fixed, ownership-transfer overflow) | ✅ 6F |
| Task 8 | D→Q→E→T ownership transfer (no double-count, store() always true) | ✅ 6E, 6F |
| Task 9 | TerminalReclaimAuthority = growable (K_terminal<∞ assumption, D101-9) | ✅ 6G |
| Task 10 | A_max = reservation token ≠ world ownership (S1 disjoint) | ✅ 6H |
| Task 11 | K_reader = stranding (contained in S4/S5 tight / E_max_message unbounded conservative) | ✅ 6F, 6I |
| Task 12 | K_world = A_max + 256 + 1 + 4096 + 1024 + K_terminal + K_reader(tight: contained) | ✅ 6I, final bound |

---

## Chain Completion Table (Step 6R reflected)

| Step | Quantity | Bound | Status |
|---|---|---|---|
| Step 0 | Contract reconciliation | — | DONE |
| Step 1 | World identity / ownership | — | DONE |
| Step 2 | Reservation token semantics | — | DONE |
| Step 3 | A_max < ∞ | design-defined / code MISSING | CONDITIONAL |
| Step 4 | P_queue_max = 4096 | PROVEN | ✅ |
| Step 4 | P_max ≤ 4098 | CONDITIONAL (R-PROD) | ⚠️ |
| Step 5 | H_hold < ∞ (liveness) | PROVEN (RAII) | ✅ |
| Step 5 | P → B_max additive contribution | **0 (5D theorem)** | ✅ CODE-PROVEN |
| Step 5 | D101 Tier-4 published-domain exclusion | RESOLVED | ✅ |
| Step 6R | K_reserved (S1) | A_max < ∞ | CONDITIONAL |
| Step 6R | **K_ownerChannel (S2)** | **≤ 256** | **CODE-PROVEN (was 65)** |
| Step 6R | K_current (S3) | ≤ 1 | CODE-PROVEN |
| Step 6R | K_retire (S4) | ≤ 4096 | CODE-PROVEN |
| Step 6R | K_quarantine (S5) | ≤ 1024 | CODE-PROVEN |
| Step 6R | K_terminal (S6) | growable / ASSUMPTION (D101-9) | ⚠️ |
| Step 6R | K_reader (stranding) | contained in S4/S5 (tight) / E_max_message unbounded (conservative) | ⚠️ CONDITIONAL / CONSERVATIVE NO-GO |
| Step 6R | **K_world < ∞** | **A_max + 5377 + K_terminal** | **CONDITIONAL GO (tight)** |

---

## 6K — Step 7 (Conservation Proof) handoff

### What Step 6R established for Step 7

```text
Step 5 pivot:              P → B_max(K_world) additive = 0  (P_queue ⊆ OwnerChannel ⊆ K_transferred)
Step 6R K_world terms:     disjoint S-domain ownership (S1 token, S2 256 owning, S3 1 ptr,
                           S4 4096, S5 1024, S6 growable, S0/S7 non-counting)
Step 6R conservation:      D→Q→E→T ownership-transfer chain (no double-count);
                           publishAndSwap single exchange (S2→S3 single-transfer);
                           take-drains-once (S2 ownership, single-transfer)
```

### Step 7 proof obligation (carry-forward)

Step 7 は Step 6R の disjoint ownership を **formalize** する:

```text
∀ t: M_world(t) = |S1(t)| + |S2(t)| + |S3(t)| + |S4(t)| + |S5(t)| + |S6(t)|     — exact count
∀ transition: ΔM = +1 (publish) | -1 (reclaim) | 0 (intra-domain transfer)       — conservation drift = 0
```

**Step 7 が検証すべき事項**:
1. S2→S3 transfer: `take()` が nullptr にする (slot free) ⟹ `|S2|` は -1, `publishAndSwap` が S3 に set ⟹ `|S3|` は +1 — **intra-lifecycle transfer, not +2**。
2. S3→S4 transfer: `enqueueRetire(oldWorld)` は `current` を atomically exchange (S3 →nullptr) と同時に D.enqueue — **current が nullptr になる瞬間と D に入る瞬間は同一遷移**。
3. S4↔S5↔S6 transfer: D.dequeue (slot free) → Q.quarantine (Q.count+1) — **move, not copy**。
4. S6→S7: `deleter(ptr)` → `S7` (count 0) — budget returned。
5. **onRelease notification**: `DeletionEntryType::World` のみ (DeferredDeletionQueue.h:148, RetireQuarantineStore.h:140, ISRRetireRouter.cpp:57) — Tier-4 closure carry-forward。

### Remaining assumptions into Step 7

| Assumption | Source | Step 7 role |
|---|---|---|
| A_max < ∞ (S1 reservation bound) | Step 3 / D101-9 | formal parameter (design, pending code) |
| K_terminal < ∞ (growable store bound) | D101-9 | formal parameter (pending bounded impl) |
| E_max_message bounded | Step 5/6-G | conservative K_reader; may remain OPEN (tight bound sidesteps) |

**Step 7 は conservation drift = 0 を prove することがゴール** — Step 6R の disjoint domain を形式化し、"同一 world が 2 つの S に同時に所属しない" + "transition は ownership move" を state-machine invariant として確立する。K_world < ∞ は Step 7 で assumption (A_max, K_terminal) を explicit に固定した上で CONDITIONAL GO とする。

---

## References (Step 6R verified)

| Evidence | File | Lines |
|---|---|---|
| OwnerChannel kCapacity=256 (owning, SPSC single-transfer) | `src/audioengine/OwnerChannel.h` | 41 (kCapacity), 42-43 (Slot), 60-108 (enqueue/take) |
| PendingPublishRegistry k=64, non-owning (const void*) | `src/audioengine/RuntimeWorldAuthority.h` | 34 (kPendingPublishCapacity), 43 (`const void* sealedWorld`) |
| registerPublish (non-owning ptr store) | `src/audioengine/RuntimeWorldAuthority.h` | 43-50 |
| RuntimeStore (single atomic ptr, publishAndSwap acq_rel) | `src/core/RuntimeStore.h` | 58 (publishAndSwap), 79 (observe) |
| INV-X4-3 single write access | `src/audioengine/RuntimeWorldAuthority.h` | 88-90 |
| DeferredDeletionQueue kQueueSize=4096, reclaim isOlder | `src/DeferredDeletionQueue.h` | 262 (kQueueSize), 133-145 (reclaim), 148/199 (onRelease gate) |
| RetireQuarantineStore kMax=512 (fixed array) | `src/audioengine/RetireQuarantineStore.h` | 65, 70-105 (quarantine), 140-145 (drain onRelease) |
| TerminalReclaimAuthority (growable vector, store() always true) | `src/audioengine/ISRRetireRouter.h` | 47 (vector), 50-55 (Entry), 57-62 (drain onRelease) |
| enqueueWithRetry D→Q→E→T ownership chain | `src/audioengine/ISRRetireRouter.cpp` | 25 (invariant), 282-360 (Stages 1-5) |
| terminalReclaim store() impl | `src/audioengine/ISRRetireRouter.cpp` | 40-49 |
| publishAndSwap in publish path | `src/audioengine/RuntimePublishExecutor.h` | 20-108 (executePublish) |
| ownerChannel().take (sole single-transfer claim) | `src/audioengine/RuntimePublishExecutor.h` | 23 |
| retirePublished/Rejected domain separation | `src/audioengine/AudioEngine.h` | 3536-3562 (World vs Generic) |
| enqueueRuntimePublicationFireAndForget ordering | `src/audioengine/AudioEngine.h` | 4509-4590 (5→6→7→8 ordering) |
| EpochDomain reader model | `src/core/EpochDomain.h` | 19-22 (kMaxReaders), registerReaderThread |
| AudioEngine destructor shutdown reclaim | `src/audioengine/AudioEngine.CtorDtor.cpp` | 194, 215 (publishEpoch); retireCurrentAndTarget |
| ReleaseResources shutdown | `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp` | 237, 255 (publishEpoch) |
| SnapshotCoordinator steady publish | `src/core/SnapshotCoordinator.cpp` | 91 (publishNew), 109 (switchImmediate) |
