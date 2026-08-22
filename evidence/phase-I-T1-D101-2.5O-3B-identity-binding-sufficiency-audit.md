# D101-2.5O-3B — Identity / Binding Candidate Exhaustion (IDENTITY & KEY SUFFICIENCY AUDIT)

**Status**: CENSUS COMPLETE — O→E binding provisional.

## 0. Scope & Prohibitions (per user instruction)

O-3B is a **pure identity / binding-key sufficiency audit** — zero of the following:
コード変更 0 / evaluate変更 0 / evaluateDeferred変更 0 / 新 token 0 / 新 permit 0 /
新 ID 0 / 新 deadline 0 / 新 timeout 0 / 新 field 0 / 新 binding API 0 / 設計案の採用 0.

This is NOT "what binding key to introduce" — it is "does a genuine O→E semantic relation
exist in the **current** code?". The provisional finding from 2.5O-3A stands until 3B
overturns it by code evidence.

## 1. Identity Partitioning (O vs E)

O-3A grouped identity candidates into three semantic roles. 3B formally separates them:

| Symbol | Role | Scope | Stamps | Consumed at |
| --- | --- | --- | --- | --- |
| `correlationId` | **O telemetry correlation** | trySubmitImpl (orchestrator) | `nextCorrelationId()` line 567 | `stateOwner_.*`, `telemetryRecorder_.*` (lines 55,171,177,181,182,236,242,247,248,276,279,280,297,298) |
| `RuntimePublicationIdentity{generation, worldId, publicationSequence}` | **O build reservation** | RuntimeBuilder | `engine.reserveRuntimePublicationIdentity()` line 63/165 | stamped into `world->publication.*` |
| `OwnerChannelKey{sequenceId, epoch, mappedGeneration}` | **E transfer key** | producer→intent→take→registry | `commitRuntimePublication` (line 4510) from `newWorld->publication.*` | `ownerChannel().take()` in executePublish (RuntimePublishExecutor.h line 50) |

## 2. O-3B-1 — O Identity Re-audit


### 2-1 CorrelationId lifetime (the accepted obligation)
`correlationId` is generated once per `trySubmitImpl` call (`nextCorrelationId()` L567). Census of every reference shows it touches **only**:


- `stateOwner_.onSubmitted(correlationId.shortValue())` — L55
- `telemetryRecorder_.recordProgress(correlationId, ...)` — L62, L172, L182, L237, L248, L280, L298
- `stateOwner_.onBuilt/onExecutorFailed/onValidated/onPublished(correlationId.shortValue())` — L171,181,236,247,276,297
- `stateOwner_.setLastCorrelationId(cid)` — L570

**Critical**: `correlationId` NEVER reaches `commitRuntimePublication`, `enqueueRuntimePublicationFireAndForget`, the `Intent` struct, `OwnerChannelKey`, `PendingPublishRegistry`, or `executePublish`.

AudioEngine.Commit.cpp:338 `CorrelationId{engineInstanceId_, world.publication.sequenceId}` is **construction of a telemetry CorrelationId from seqId** — NOT the obligation identity flowing into E. It is telemetry-only.


### 2-2 CorrelationId ↔ sequenceId relation
`TelemetryRecorder::nextCorrelationId(engineInstanceId)` (TelemetryRecorder.cpp L44/L49) packs engineInstanceId + a monotonic counter. **seqId is NOT embedded** in correlationId. The only seqId-touching site is `stateOwner_.onSubmitted(correlationId.shortValue())` — a progress marker, not identity binding.

→ **correlationId is O telemetry correlation, NOT O obligation identity, and is structurally excluded from the E pipeline.**


### 2-3 Does correlationId reach E pipeline?
**NO.** The producer path (`commitRuntimePublication` → `enqueueRuntimePublicationFireAndForget`) derives the E key **solely** from `newWorld->publication.{sequenceId, epoch, mappedRuntimeGeneration}`. correlationId is absent from `Intent`, `OwnerChannelKey`, and `PublishMetadata`.

**Verdict 3B-1: O identity (correlationId) is PROVEN ABSENT from E pipeline.** O obligation "Accepted" has no deterministic relation into E via correlationId.

## 3. O-3B-2 — E Identity Lineage (sequenceId/epoch/mappedGeneration)

The single deterministic lineage for the E transfer key, traced source-to-sink:


### 3-1 sequenceId lineage

```text
engine.reserveRuntimePublicationIdentity()           // RuntimeBuilder.cpp:63
  → publicationIdentity.publicationSequence          // RuntimeBuilder.cpp:64
  → worldOwner->publication.sequenceId = bootstrapPublicationSequence  // RuntimeBuilder.cpp:113
  → newWorld->publication.sequenceId                 // builder stamps
  → seqId = newWorld->publication.sequenceId         // AudioEngine.h:4508 (commitRuntimePublication)
  → intent.sequenceId = seqId                        // AudioEngine.h:4531
  → intent.payload.publish.version = seqId           // AudioEngine.h:4534
  → OwnerChannelKey{intent.sequenceId, ...}          // RuntimePublishExecutor.h:50
  → registry().lookup(intent.sequenceId)             // RuntimePublishExecutor.h:63
  → authority.registry().unregister(intent.sequenceId)  // RuntimePublishExecutor.h:85
```

**Strict monotonicity**: `SequenceArithmetic.h` specifies `isAfter(seq, prev)` for commit monotonicity (ISRRuntimePublicationCoordinator.cpp commit path) — same counter domain across all sites. seqId is a single monotonic counter; `a < b` is semantically `isAfter(b,a)` (RFC 1982).

**Verdict (A) — seqId deterministic lineage: PASS.**


### 3-2 epoch equality

```text
worldOwner->publication.epoch = static_cast<PublicationEpoch>(bootstrapGeneration)  // RuntimeBuilder.cpp:114
  → epoch = static_cast<uint32_t>(newWorld->publication.epoch)                      // AudioEngine.h:4509
  → intent.payload.publish.epoch = newWorld->publication.epoch                      // AudioEngine.h:4533
  → OwnerChannelKey{..., static_cast<uint32_t>(intent.payload.publish.epoch), ...}  // RuntimePublishExecutor.h:51
```
epoch is `generation` (PublicationEpoch = same increment source as generation). The same value is read at 4 independent sites; no transformation between them. Producer epoch ≠ retire epoch (`retire.retireEpoch`, RuntimeBuilder.cpp:122) — correctly distinct domains.

**Verdict (B) — epoch equality across producer→intent→key: PASS.**


### 3-3 mappedGeneration propagation

```text
worldOwner->publication.mappedRuntimeGeneration = bootstrapGeneration          // RuntimeBuilder.cpp:115
  → mappedGen = static_cast<uint64_t>(newWorld->publication.mappedRuntimeGeneration)  // AudioEngine.h:4510
  → intent.payload.publish.mappedGeneration = mappedGen                        // AudioEngine.h:4535
  → OwnerChannelKey{..., intent.payload.publish.mappedGeneration}               // RuntimePublishExecutor.h:52
```
Propagation is verbatim copy (no arithmetic). `mappedRuntimeGeneration` is stamped from `generation` at builder bake time.

**Verdict (C) — mappedGeneration propagation: PASS.**

## 4. O-3B-3 — Composite Key Sufficiency


### 4-A: Multiple E per K (two executions, same key)
**PASS.** `OwnerChannel::take(key)` is **single-transfer**: once `owner` for `key` is drained (line 70-72: `publishAtomic(s.owner, nullptr, release)`), a second `take(key)` probes the same slots, finds `seen == nullptr` (drained), and returns `nullptr` — no second `RuntimeStateOwner` is produced. The `RuntimeWorldAuthority::publish()` (line owner.release) only accepts a non-null owner. Therefore **no key maps to two execution worlds**.


### 4-B: Different key, same E (seqId alias collision)
**PASS.** PendingPublishRegistry::lookup(seqId) matches one seqId → one `void* world`. OwnerChannel::enqueue(key, owner) **rejects key duplicates** (line 39: `if (s.key == key) return false`) — a second enqueue of the same key returns false (no overwrite). OwnerChannelKey contains epoch + mappedGeneration, so a retry/retry with a different generation produces a *different* key → distinct slot.


### 4-C: Same seqId, different epoch/generation (retry across generations)
**PASS.** Generation strictly increases per `reserveRuntimePublicationIdentity()` (RuntimeBuilder line 63). A retry after a generation bump produces `epoch ≠ epoch'` → key mismatch → no false take. OwnerChannel::enqueue rejects collisions on `s.key == key` (full composite match), so a re-enqueue with a mutated epoch/generation is a *new* key, correctly distinct.


### 4-D: Wraparound / retry / shutdown key reuse
**PASS.** seqId is `uint64_t` with modular arithmetic (SequenceArithmetic.h §RFC 1982). Practical distance `<< 2^63`. The full OwnerChannelKey (seq, epoch, gen) is the hash basis (line 42: Knuth mixing of all three) — wraparound in one component without the others changes the key. Drain (`drainAllNonRt`, OwnerChannel.h line 78-91) clears residual slots at shutdown before any reuse window. **No key is ever recycled within a live sequence number space.**


### 4-E: Registry seqId-only lookup — partial-key aliasing risk
**CAUTION (mitigated).** `PendingPublishRegistry::lookup(seqId)` (ISRRuntimeWorldAuthority.h line 53-62) matches on `seqId` alone. In theory, if two distinct OwnerChannelKeys shared a seqId but differed in epoch/mappedGeneration, the registry fallback could alias. However:
1. `seqId` is **monotonic and unique per publish** (stamped atomically by reserveRuntimePublicationIdentity). Two distinct publishes never share a seqId in a live session.
2. The registry is ONLY reached on the **fallback path** (`owner == nullptr` in executePublish, RuntimePublishExecutor.h line 58) — i.e. when OwnerChannel take returned null (normal async gap case). It is indexed by the same seqId the key carries.
3. `unregister(seqId)` is called post-commit (line 85) — entries are single-flight.

**Verdict: PASS (not PROOF) — single-flight seqId + single consumer take makes aliasing unreachable in the current single-CordinatorLoop execution model.** This is the one point where OwnerChannel key (composite) and registry key (seqId-only) differ structurally; it is neutralized by seqId's uniqueness guarantee.

## 5. O-3B-4 — OwnerChannel Take-Once Formalization

OwnerChannel.h lines 25-36 document the invariant:

- `enqueue(key, owner&&) -> false if key already queued or full`
- `take(key) -> nullptr if absent/mismatch; slot drained on hit (no 2nd take)`

**Single-Producer / Single-Consumer (SPSC)** is declared in the header comment (line 6) and structurally enforced: only `commitRuntimePublication` (Non-RT producer) calls `enqueue`; only `executePublish` on the ISR/audio thread calls `take`.

Collision taxonomy:

- **Key collision**: `enqueue` probes (line 45-48); if `s.key == key`, returns false (reject, no overwrite). If `s.key != key`, continues probing (`continue`). → **No false overwrite on key collision.**
- **Slot collision (hash collision, different key)**: resolved by linear probing; `take` checks `s.key != key` (line 58) and skips. → **No wrong-key drain.**
- **Full channel** (no free slot in 256): `enqueue` returns false (line 73). → **Caller retains owner** (AudioEngine.h:4525 `Failed, CallerDestroy`).
- **Double take**: `take` CAS-drains (`publishAtomic nullptr`) on match (line 70-72). A second `take(sameKey)` finds `nullptr` → returns `nullptr`. → **Exactly-once transfer.**


### Drain semantics (`drainAllNonRt`, line 78-91)
Called at Non-RT shutdown (producer/consumer quiesced). Reuses the same `consume→publish(nullptr,release)` single-transfer pattern as `take`. **Idempotent**: re-entrant drain is a no-op (slots already nullptr). `s.key` is NOT reset — key-matching irrelevant for full scan; emptiness is `owner==nullptr`.

**Verdict 3B-4: OwnerChannel take-once invariant is PROVEN structurally** (enqueue reject-on-key-match + take CAS-drain + SPSC + idempotent drain).

## 6. O-3B-5 — Registry / Payload Fallback Semantic Target Equivalence

executePublish (RuntimePublishExecutor.h lines 58-64):
```cpp
const auto* newWorld = owner ? owner.get()
                             : static_cast<const RuntimeState*>(authority.registry().lookup(intent.sequenceId));
if (newWorld == nullptr)
    newWorld = static_cast<const RuntimeState*>(p.newWorld);
```

Three candidate sources, all must denote the **same semantic world**:
1. **OwnerChannel** (primary): the `aligned_unique_ptr<const RuntimeState>` enqueued by producer, sealed (`sealRecursively()`, line 70) before `publish()`.
2. **Registry** (async gap fallback): `registry().lookup(seqId)` — the `const void*` registered at `registerPublish(seqId, newWorld)` (AudioEngine.h:4516) **before** the owner is enqueued (line 4520, register happens at L4516 before enqueue L4519).
3. **Intent payload** (legacy fallback): `p.newWorld` — set to `static_cast<const void*>(newWorld)` (AudioEngine.h:4532), the **same pointer** as the registry `newWorld`.

**Pointer identity**: registry stores `newWorld` (line 4516) = payload `.newWorld` (line 4532) = the sealed world. OwnerChannel holds the same world via the moved `world` unique_ptr (which wraps `newWorld`'s pointee). `sealRecursively()` is idempotent-safe (RuntimePublishExecutor.h:70, called only when `hasOwner`). All three are the **identical RuntimeState**.

**Verdict 3B-5: PASS.** All three fallback sources resolve to the same sealed RuntimeState — the semantic target is invariant across primary and fallback paths.

## 7. O-3B-6 — Receipt Timeout as State Continuity (NOT failure)

commitRuntimePublication (AudioEngine.h lines 4550-4565) documents (verbatim):
> `timeout ≠ publish failure。 所有権は enqueue 時点で Transferred（Allocated → Transferred → Committed → Completed）。timeout を failure と誆けして rollback すると double ownership / double publish を生む。`
> (`work88 (X2 §6.2 — 二十三次レビュー timeout semantics)`)

The 250ms `waitForPublishReceipt` (line 4563) is **informational** — the diag log at L4560 fires on timeout, but the `result` already carries `OwnershipDisposition::Transferred` from `enqueueRuntimePublicationFireAndForget`. No rollback/retake occurs on timeout.

State machine (AudioEngine.Commit.cpp `PublishCommitResult`):

- **Allocated** → (enqueue success) → **Transferred**
- (CoordinatorLoop executePublish commit + onPublishCommitted → notifyPublishReceipt) → **Completed**
- timeout: remains **Transferred** — the world is committed-or-will-be; ownership has exited the producer.

**Verdict 3B-6: PASS.** Receipt timeout preserves `Transferred` state. Timeout ≠ publish failure is **formally encoded in ownership disposition** — no rollback path exists on timeout.

## 8. O-3B-7 — Fire-and-Forget Transaction Separation

`enqueueRuntimePublicationFireAndForget` (AudioEngine.h line 4471) returns immediately with:

- `{ Success, OwnershipDisposition::Transferred }` (line 4547)
- ownership relinquished at `ownerChannel().enqueue(...)` (line 4519-4525)
- `rollbackHandle = DSPHandle::null()` (line 4545) — ScopeExit rollback is **disabled** post-enqueue-success

`commitRuntimePublication` (line 4471, the **synchronous wrapper**) calls the fire-and-forget core then `waitForPublishReceipt(seqId, 250ms)` (line 4562). The key derivation (seqId/epoch/mappedGen from `newWorld->publication`) is **identical** in both paths (same line 4508-4510 source).

**Transaction boundary separation**: key derivation and ownership transfer are **common** to both; only the wait differs. Fire-and-forget is not a *different identity*, it is the **same transaction without synchronous completion wait**.

**Verdict 3B-7: PASS.** Fire-and-forget is the same key-transfer transaction with the completion wait elided. O→E binding semantics are unaffected by wait/no-wait choice.

## 9. Scorecard

| O-3B sub-step | Question | Verdict | Evidence location |
| --- | --- | --- | --- |
| 3B-1 | Does correlationId identify the Accepted obligation into E? | PROVEN ABSENT (N/A) | `correlationId` refs census: RuntimePublicationOrchestrator.cpp:54-55,62,171-182,236-248,276-280,297-298,567-570 |
| 3B-1 | seqId deterministic relation from reserveIdentity→world→intent→Key | PASS | RuntimeBuilder.cpp:63-115; AudioEngine.h:4508-4535; RuntimePublishExecutor.h:50 |
| 3B-1 | correlationId reaches executePublish/Intent/OwnerChannel/registry? | NO (structurally excluded) | AudioEngine.h:4471-4565 (intent build omits correlationId) |
| 3B-2 (A) | seqId lineage deterministic, monotonic, single counter | PASS | SequenceArithmetic.h; RuntimeBuilder.cpp:113; AudioEngine.h:4508,4531,4534 |
| 3B-2 (B) | epoch equality across producer→intent→OwnerChannelKey | PASS | RuntimeBuilder.cpp:114; AudioEngine.h:4509,4533; RuntimePublishExecutor.h:51 |
| 3B-2 (C) | mappedGeneration propagation verbatim | PASS | RuntimeBuilder.cpp:115; AudioEngine.h:4510,4535; RuntimePublishExecutor.h:52 |
| 3B-3 (A) | Multiple E per Key (two worlds, one key)? | PASS | OwnerChannel.h:39 (reject), 70-72 (CAS drain single-transfer) |
| 3B-3 (B) | Different key, same E (seqId alias)? | PASS | OwnerChannel.h:39 (reject dup); registry single-flight |
| 3B-3 (C) | Same seqId, different epoch/gen retry | PASS | OwnerChannel.h key includes epoch+gen → distinct key |
| 3B-3 (D) | Wraparound/retry/shutdown key reuse | PASS | SequenceArithmetic.h RFC1982; OwnerChannel.h:78 drainAllNonRt |
| 3B-3 (E) | Registry seqId-only lookup partial-key aliasing | PASS (CAUTION mitigated) | PendingPublishRegistry lookup seqId-only but single-flight seqId guarantee |
| 3B-4 | OwnerChannel take-once (no double take) | PROVEN | OwnerChannel.h:14-36 (SPSC, reject, CAS drain), 78-91 (drain idempotent) |
| 3B-5 | Registry/payload fallback ≡ same semantic world | PASS | RuntimePublishExecutor.h:58-67 (3 sources: owner/registry/payload.newWorld, ptr-identical) |
| 3B-6 | Receipt timeout ≠ publish failure (state continuity) | PASS (formally encoded) | AudioEngine.h:4550-4565 (Transferred on timeout) |
| 3B-7 | Fire-and-forget transaction separation | PASS | AudioEngine.h:4471 (key derivation identical; wait only differs) |

## 10. Final Determination

**O→E binding key (provisional, pending 3C): `OwnerChannelKey{sequenceId, epoch, mappedGeneration}`**

The key is:
1. **Deterministically derived** from `newWorld->publication.*` (no new ID/field/token/permit introduced).
2. **Deterministic lineage preserved** seqId→intent→Key→registry→take→commit (3B-2 A/B/C PASS).
3. **No multi-E-per-key, no key-aliasing, no wraparound reuse** (3B-3 A/B/C/D PASS).
4. **Single-transfer enforced** structurally on OwnerChannel (3B-4 PROVEN).
5. **Fallback sources semantically identical** (3B-5 PASS).
6. **Timeout preserves Transferred** (3B-6 PASS).

**counterId / correlationId is NOT an O→E binding** — structurally excluded from E pipeline (3B-1).

**PendingPublishRegistry::lookup(seqId) seqId-only access** is the one structural asymmetry (3B-3 E). It is currently safe under the single-CordinatorLoop model (seqId unique per publish) but is the **only candidate that *would* require an E-side key field** (epoch/mappedGeneration on the registry Entry) to harden against future multi-producer relaxation. 3B does **not** propose this change (3B is audit-only); it is a deferred item for 3D (failure/shutdown) or 3E (closure) if multi-producer execution is ever introduced.

**→ 3B complete. Proceed to 3C (B1/B2/B3 existence, uniqueness, preservation judgments).**
