# Phase I-T1-D101-2.5O-3C — O→E Binding Existence / Uniqueness / Preservation Judgment

**Status**: COMPLETE (audit-only, 0 code changes).

## 0. Scope & Prohibitions (per user instruction)

Audit only — zero of:
コード変更 0 / evaluate変更 0 / evaluateDeferred変更 0 / 新 token 0 / 新 permit 0 /
新 ID 0 / 新 deadline 0 / 新 timeout 0 / 新 field 0 / 新 binding API 0 / 設計案の採用 0.

No "how to make O→E binding" design proposals (3C = does the *existing* code satisfy the
conditions). 3B findings are taken as facts; we do NOT derive A→D from A→B+B→C+C→D.

## 1. O Accepted Obligation — Fixed Definition (3C-0)

`O = Accepted` is fixed at `RuntimePublicationOrchestrator::trySubmitImpl`.

### Source of Accepted (the emission event)

`RuntimePublicationOrchestrator.cpp:90` (line numbers from 3B/3C read):

```cpp
auto decision = admission_.evaluate(req, engine_, pubCtx);
if (decision != PublicationAdmission::Decision::Accepted) { return decision; }
```

`evaluate()` is the sole gate (bypass-prohibited, confirmed at C1). `Accepted` is returned **after**
this comparison succeeds. Admission does not alter the world; it only classifies.

### State/event immediately after Accepted

- `correlationId = nextCorrelationId()` — telemetry correlation (see §7).

- `stateOwner_.onSubmitted(correlationId.shortValue())` — ledger mark (state owner).

- `telemetryRecorder_.recordProgress(correlationId, req.generation, 0, Submitted, nowUs)`.

- **No seqId/epoch/mappedGeneration is stamped yet** — those come from the Builder reservation.

### Information carried by Accepted obligation

Accepted carries:

1. `req` (PublishRequest) — the caller's intent (DSP handles, build input, `req.generation`).

2. `correlationId` — O-side telemetry correlation (NOT O obligation identity — see §7).

3. **Nothing else** — Accepted itself carries no seqId. The obligation **becomes** identified
   by the world the Builder produces from `req`.

### Accepted → publish request boundary

After Accepted:

```cpp
auto worldBuilder = convo::RuntimeBuilder(engine_);
auto frozen = buildRuntimePublishWorld(...spec);   // builder stamps world->publication.{seq,epoch,mappedGen}
auto result = executor_.publish(engine_, std::move(frozen), req.newDSP, oldHandle);  // L96
return PublicationAdmission::Decision::Accepted;   // L183

```

The **sole** bridge from O Accepted to E is: `executor_.publish(engine_, std::move(frozen), ...)`.
The `frozen` world (immutable, sealed) carries `publication.{sequenceId, epoch, mappedRuntimeGeneration}`
stamped by the Builder from `reserveRuntimePublicationIdentity()`.

## 2. Lineage Census (3C-1 → 3C-6)

### 3C-1 — Accepted → Build

- **Event**: `trySubmitImpl` L90 `Accepted` → `buildRuntimePublishWorld(spec)` (L93 area).

- **Preserved**: `req` (DSP handles, buildInput) → spec fields. `req.generation` is **NOT** copied to
  `world->publication.epoch` — the Builder stamps `epoch = bootstrapGeneration =
  reserveRuntimePublicationIdentity().generation` (RuntimeBuilder.cpp:114).

- **Loss**: `correlationId` is **dropped** at the boundary — it is not passed into `frozen` or into
  `executor_.publish`. This is **by design** (3B confirmed: correlationId ∈ telemetry/stateOwner only).

- **Verdict 3C-1**: O→Build preserves the *world* (via `req`) but **not** a correlation token.
  The binding is carried by the **world object itself**, not a token.

### 3C-2 — Build → RuntimePublicationIdentity

- **Event**: `reserveRuntimePublicationIdentity()` at RuntimeBuilder.cpp:63,165.

- `RuntimePublicationIdentity = {generation, worldId, publicationSequence}` is **O build reservation**
  (3B partition). It is **NOT** the O Accepted obligation identity — it is the reservation that
  *anoints* the build.

- **Stamp**: builder writes
  - `worldOwner->publication.sequenceId = bootstrapPublicationSequence` (L113)
  - `worldOwner->publication.epoch = static_cast<PublicationEpoch>(bootstrapGeneration)` (L114)
  - `worldOwner->publication.mappedRuntimeGeneration = bootstrapGeneration` (L115)

- **Verdict 3C-2**: Build identity lineage is `reserveRuntimePublicationIdentity() → world.publication.*`.
  This is **build reservation identity → world stamp**, not O obligation identity.

### 3C-3 — Publication → Intent

- **Event**: `commitRuntimePublication` / `enqueueRuntimePublicationFireAndForget` (AudioEngine.h:4471,4508-4537).

- Key derivation reads **only** from `newWorld->publication.*`:
  - `seqId = newWorld->publication.sequenceId` (L4508)
  - `epoch  = static_cast<uint32_t>(newWorld->publication.epoch)` (L4509)
  - `mappedGen = static_cast<uint64_t>(newWorld->publication.mappedRuntimeGeneration)` (L4510)

- Intent fill (L4530-4537):
  - `intent.sequenceId = seqId`
  - `intent.payload.publish.version = seqId`
  - `intent.payload.publish.epoch = newWorld->publication.epoch`
  - `intent.payload.publish.mappedGeneration = mappedGen`
  - `intent.payload.publish.newWorld = static_cast<const void*>(newWorld)`

- **Verdict 3C-3**: Publication → Intent is **verbatim projection** of the world's publication
  identity into the Intent. No transformation, no loss. The Intent is an **E-side artifact**; it
  does not carry any O- Accepted token.

### 3C-4 — Intent → OwnerChannelKey

- **Event**: `PublishExecutor::executePublish` (RuntimePublishExecutor.h:50):

  ```cpp
  auto owner = authority.ownerChannel().take(
      OwnerChannelKey{ intent.sequenceId,
                       static_cast<std::uint32_t>(intent.payload.publish.epoch),
                       intent.payload.publish.mappedGeneration });

```text

- The Key is **exactly** `{intent.sequenceId, intent.payload.publish.epoch, intent.payload.publish.mappedGeneration}` —
  the same triple that `commitRuntimePublication` stamped from `world->publication.*` (§3C-3).

- **Verdict 3C-4**: Intent → Key is identity (field-for-field copy, same origin values). The Key is
  **E transfer identity**.

### 3C-5 — Key → executePublish

- **Event**: `executePublish` takes the owner via `OwnerChannelKey`; if `hasOwner`, seals the world,
  calls `authority.publish(owner, PublishMetadata{..., intent.sequenceId, p.epoch, p.mappedGeneration})`
  (RuntimePublishExecutor.h:71-79).

- `RuntimeWorldAuthority::publish` (RuntimeWorldAuthority.h) commits metadata (bakes publication into
  the sealed world) then `writeAccess_.publishAndSwap(next)` (the sole physical store swap, INV-X4-3).

- **Verdict 3C-5**: Key → executePublish is the **take-once** transfer (3B-4 proven: enqueue rejects
  key-duplicates, take CAS-drains exactly once). The same `RuntimeState` object is sealed once and
  published once.

### 3C-6 — executePublish → Completion

- **Event**: `ctx.engine.runtimeOrchestrator_->onPublishCommitted(intent.sequenceId)` (RuntimePublishExecutor.h:88).

- `onPublishCommitted` (RuntimePublicationOrchestrator.cpp:293-295):

  ```cpp
  convo::publishAtomic(m_lastObservedSequence, seqId, release);
  convo::publishAtomic(m_lastProgressTimestampUs, nowUs, release);
```

- **Completion/receipt**: the Producer's `waitForPublishReceipt(seqId)` (commitRuntimePublication,
  AudioEngine.h:4562) completes on the seqId committed here. seqId is the **same** sequenceId that
  originated from `world->publication.sequenceId` (§3C-3).

- **Verdict 3C-6**: Completion is keyed on the **same seqId** that the Builder stamped from the
  reservation — i.e. the E execution's seqId.

## 3. The Semantic Bridge (the core of 3C)

The **only** thing that binds an O Accepted obligation to an E execution in the current code is:

> **The `frozen` RuntimeState object produced by the Builder on behalf of an Accepted obligation.**
> Its `publication.{sequenceId, epoch, mappedRuntimeGeneration}` is stamped once (from the reservation),
> then **verbatim** (no arithmetic) projected into the Intent → OwnerChannelKey → PublishMetadata,
> survives `take` (single-transfer) and `publishAndSwap`, and is re-attached to Completion via
> `onPublishCommitted(seqId)`.

The object identity (the `RuntimeState*` / `newWorld` pointer) is preserved end-to-end:

- `worldOwner.release()` → `frozen` holds the unique_ptr (RuntimePublicationOrchestrator.cpp:88).

- `frozen->releaseState()` → `stateOwner` (PublicationExecutor.cpp:24).

- `commitRuntimePublication(std::move(stateOwner), …)` → OwnerChannel `enqueue(key, owner)` (key derived from `stateOwner->publication.*`).

- `executePublish` → `take(key)` → same `RuntimeState*` deref → `sealRecursively` → `authority.publish`.

- `registry().lookup(seqId)` returns the **same** `newWorld` pointer (registerPublish stored it).

- `intent.payload.publish.newWorld` = same `newWorld` (AudioEngine.h:4532).

**Therefore the O Accepted obligation is semantically bound to its single E execution by the
frozen world object + its publication triple — not by a separate obligation identity token.**

## 4. B1 — Existence

> Does a semantic relation exist (in the current code) between an Accepted O obligation and the E
> execution that realizes it?

**Answer — PASS.**

The relation exists and is **singular per Accepted obligation**:

1. `trySubmitImpl` yields `Accepted` (L90) and immediately builds **one** `frozen` world (L93).

2. `executor_.publish(engine_, std::move(frozen), …)` (L96) passes the **sole** world to the E side.

3. `publishImpl` calls `commitRuntimePublication` (or fire-and-forget) — both derive the E key
   **from this one world's** `publication.{sequenceId, epoch, mappedRuntimeGeneration}` (AudioEngine.h:4508-4510).

4. `OwnershipDisposition::Transferred` is returned (AudioEngine.h:4547) — the world's ownership
   physically leaves the producer at enqueue (`ownerChannel().enqueue(...)` L4519-4525).

5. `executePublish` is the sole consumer (`take(key)`, RuntimePublishExecutor.h:50) — one take per key
   (3B-4: single-transfer, no second take).

The relation is **not** "seqId happens to appear on both sides" — it is the **frozen world object**
carrying the publication triple, transitively projected into every E-side artifact (Intent, OwnerChannelKey,
PublishMetadata, registry entry, completion seqId). There is exactly one such path per Accepted.

**Code evidence**:

- `trySubmitImpl` Accepted → single `executor_.publish` call: `RuntimePublicationOrchestrator.cpp:90-96`.

- PublicationExecutor::publishImpl → commitRuntimePublication with the world: `PublicationExecutor.cpp:46-52`.

- Key derivation from world publication fields: `AudioEngine.h:4508-4510`.

- Intent fill: `AudioEngine.h:4530-4537`.

- Single take: `RuntimePublishExecutor.h:50` + `OwnerChannel.h:50-76` (CAS drain).

- Completion re-attaches to same seqId: `RuntimePublicationOrchestrator.cpp:293` + `RuntimePublishExecutor.h:88`.

## 5. B2 — Uniqueness

> Does 1 Accepted O ever yield 2 E executions? Do 2 Accepted O merge into 1 E? Does any key alias
> occur?

### B2-A — 1 O → 2 E

**PASS (not possible).** `trySubmitImpl` calls `executor_.publish(...)` **exactly once** per Accepted
(RuntimePublicationOrchestrator.cpp:96). `publishImpl` calls `commitRuntimePublication` (or fire-and-forget)
**exactly once** (PublicationExecutor.cpp:46-52, the ternary picks one branch). Each call performs
exactly one `ownerChannel().enqueue(key, owner)` (AudioEngine.h:4519), and `enqueue` **rejects
key-duplicates** (OwnerChannel.h:39, returns false → `Failed, CallerDestroy`). `executePublish` performs
exactly one `take(key)` (RuntimePublishExecutor.h:50). So one world = one enqueue = one take = one
executePublish. No path doubles execution.

### B2-B — 2 O → 1 E

**PASS (not possible).** Each Accepted builds a distinct `frozen` world via a fresh
`reserveRuntimePublicationIdentity()` (RuntimeBuilder.cpp:63/165) — `publicationSequence` is strictly
monotonic (`isAfter` per SequenceArithmetic.h, confirmed in 3B-2). A distinct seqId ⇒ a distinct
`OwnerChannelKey` ⇒ a distinct `executePublish`. The completion receipt (`onPublishCommitted(seqId)`,
RuntimePublicationOrchestrator.cpp:293) confirms a per-seqId completion — two Accepted obligations
produce two completion seqIds. Merge across obligations is impossible: seqId is the dedup key and it
is per-obligation-unique.

### B2-C — O1 → Key, O2 → same Key (alias)

**PASS.** `OwnerChannelKey = {seqId, epoch, mappedGeneration}`. seqId is monotonic per publish
(§B2-B). Even if a hypothetical seqId collision occurred, `epoch` (= generation, also monotonic per
publish via reserveRuntimePublicationIdentity) and `mappedGeneration` (= generation) differ across
distinct obligations. `OwnerChannel::enqueue` rejects `s.key == key` (OwnerChannel.h:39). No two distinct
owners map to the same Key in a live session. (Wraparound safety was verified in 3B-3 D: Knuth-mixed
3-field key + idempotent drain.)

### B2-D — Registry asymmetry (seqId-only lookup)

**CAUTION → mitigated PASS.** `PendingPublishRegistry::lookup(seqId)` matches **seqId only**
(ISRRuntimeWorldAuthority.h:53-62), while `OwnerChannelKey` is the composite triple. This is the
**one structural asymmetry** in the system.

Timeline (producer = Non-RT `commitRuntimePublication`; consumer = ISR `executePublish`):

```text
t0: registerPublish(seqId, world)        // AudioEngine.h:4516  (enqueue 前 — gap 開始)
t1: ownerChannel().enqueue(key, owner)   // AudioEngine.h:4519  (ownership transfer)
t2: intent enqueue (seqId, ...)          // AudioEngine.h:4530  (dispatch)
   ─── async gap ───
t3: executePublish → ownerChannel().take(key)  // RuntimePublishExecutor.h:50
t3': executePublish → registry().lookup(seqId)  // RuntimePublishExecutor.h:63  (fallback ONLY if take returned null)
t4: authority.publish(owner, metadata)   // RuntimePublishExecutor.h:71  (commit + swap)
t5: registry().unregister(seqId)        // RuntimePublishExecutor.h:85
   ─── gap 終了 ───
t6: onPublishCommitted(seqId)            // RuntimePublishExecutor.h:88 → Orchestrator.cpp:293

```

Why this is safe **in the current model**:

- The registry is **fallback-only** (`newWorld = owner ? owner.get() : registry().lookup(seqId)`,
  RuntimePublishExecutor.h:60-63). It is reached **only when `take` returned null** — i.e. the owner
  has not yet been drained. The primary path (`owner.get()`) is the OwnerChannel composite key.

- `seqId` is **single-flight**: registered once (t0), unregistered once (t5). `unregister` zeroes the
  entry (ISRRuntimeWorldAuthority.h:62-66). A second register of the same seqId overwrites the same
  slot (cursor modulo), but this cannot happen in the single-CordinatorLoop model because seqId is
  unique per publish and `executePublish` is sole gateway.

- `take(key)` is single-transfer (3B-4): the owner is drained exactly once, so the registry entry
  (which mirrors the same world) is consumed exactly once via `unregister`.

**If** the system were ever relaxed to a multi-producer / out-of-order commit model, the seqId-only
registry **would** be an alias hazard (two distinct OwnerChannelKeys could share a seqId if seqId
were reused). It is **not** reachable today because seqId is never reused within a live session and
is monotonic per publish.

**Verdict B2-D**: PASS under the current single-CoordinatorLoop + seqId-uniqueness model. The asymmetry
is real but neutralized — this is the strongest CAUTION in 3C and the **only** item that would require
an E-side change (adding epoch/mappedGeneration to registry Entry) under a future model.

## 6. B3 — Preservation

> Is the O identity / binding information preserved (untransformed, non-lost, unswapped) at every
> boundary from Accepted through Completion?

| Boundary | O↔E binding carrier | Transformation? | Loss? | Verdict |
| --- | --- | --- | --- | --- |
| Accepted → Build (3C-1) | `req` → `frozen` world; `correlationId` dropped | world built; **correlationId dropped by design** | correlationId (telemetry-only, correct) | **PASS** (binding = world object, not token) |
| Build → PubIdentity (3C-2) | `reserveRuntimePublicationIdentity()` stamps `world->publication.*` | verbatim stamp from reservation | none | **PASS** |
| Publication → Intent (3C-3) | `world->publication.{seq,epoch,mappedGen}` → `intent.*` | verbatim projection | none | **PASS** |
| Intent → OwnerChannelKey (3C-4) | `intent.sequenceId, intent.payload.publish.epoch/mappedGeneration` → Key | field-for-field copy | none | **PASS** |
| Key → executePublish (3C-5) | Key → `take(key)` → same `RuntimeState*` → `publish` | single-transfer, no copy of state | none (seal is idempotent-safe) | **PASS** |
| executePublish → Completion (3C-6) | `onPublishCommitted(seqId)` → receipt | same seqId | none | **PASS** |

### 6-1 Object identity preservation (the strongest evidence)

The **same `RuntimeState*`** flows:
`worldOwner.release()` (Orchestrator.cpp:88, into `frozen`)
→ `frozen->releaseState()` (PublicationExecutor.cpp:23, → `stateOwner` unique_ptr)
→ OwnerChannel `enqueue(key, std::move(stateOwner))` (AudioEngine.h:4519) — same raw pointer.
→ `executePublish` `take(key)` re-wraps the **same raw pointer** (OwnerChannel.h:66, `OwnerPtr(raw, deleter)`).
→ `authority.publish(std::move(owner), …)` — `owner.get()` is the same `RuntimeState*`.
→ `registry().lookup(seqId)` / `p.newWorld` return the same `newWorld`.

**The world pointer is preserved end-to-end; the publication triple is preserved verbatim.** This is
semantic preservation — not just "the same number appeared."

### 6-2 What IS lost at the boundary (and why that is correct)

- `correlationId` is dropped at Accepted→Build. **This is correct**: 3B-7 established correlationId
  is **O telemetry correlation**, never intended to be an E binding carrier. The binding is carried
  by the world object + its publication triple, not by correlationId.

- `req.generation` is NOT used to stamp `world->publication.epoch` — the Builder stamps the
  reservation generation. **Correct**: req.generation is a caller request, not the build identity;
  the reservation is.

### 6-3 Receipt timeout / fire-and-forget preservation (3C-10/3C-7 carry-over)

- On timeout, `OwnershipDisposition::Transferred` (3B-6) — the world is already enqueued; `take`
  will still occur on the ISR thread. The seqId-based completion is not blocked by the producer-side
  wait abandoning. → **binding preserved across timeout**.

- Fire-and-forget (`waitForReceipt=false`) stamps the **same key** (PublicationExecutor.cpp:52 uses
  `enqueueRuntimePublicationFireAndForget`, same AudioEngine.h:4508-4510 derivation) — the E execution
  is identical; only the producer's wait is elided. → **no preservation gap**.

**Verdict B3 — Preservation: PASS.** The O Accepted obligation's binding information (carried by the
frozen world + publication triple) is preserved verbatim and by object-identity across every boundary
to Completion. The only dropped token (correlationId) is provably an O-telemetry concern, not an
O→E binding carrier.

## 7. Final Determination of correlationId

Per 3B and re-confirmed in 3C:

```text
correlationId = O telemetry correlation
              ≠ O obligation identity
              ≠ E binding identity

```

- Generated `nextCorrelationId()` (RuntimePublicationOrchestrator.cpp:567) = engineInstanceId + monotonic counter.

- Consumed **only** by `stateOwner_.*` and `telemetryRecorder_.*` (3B census).

- Never reaches `Intent`, `OwnerChannelKey`, `PendingPublishRegistry`, `executePublish`, or completion.

**Final**: correlationId is **O telemetry correlation only**. Its absence from E is **not a gap** in
the O→E binding — the binding is the world object + publication triple. 3C does not claim "no correlationId
⇒ no binding"; it establishes the binding is carried by a different, structurally-proven channel.

## 8. B1 / B2 / B3 Formal Judgment

```text
B1 — Existence
    PASS
    理由：Each Accepted obligation produces exactly one frozen world (trySubmitImpl →
          buildRuntimePublishWorld → executor_.publish), from which the E key
          (OwnerChannelKey{pub.sequenceId, pub.epoch, pub.mappedGeneration}) and the
          Intent are derived verbatim. The relation is the frozen world object + its
          publication triple, projected field-for-field into every E artifact and
          re-attached to completion via the same seqId.
    コード証拠：trySubmitImpl Accepted→single publish call (RuntimePublicationOrchestrator.cpp:90-96);
      key derivation from world publication (AudioEngine.h:4508-4510); intent fill (AudioEngine.h:4530-4537);
      single take (RuntimePublishExecutor.h:50 + OwnerChannel.h:50-76); completion via
      same seqId (RuntimePublicationOrchestrator.cpp:293, RuntimePublishExecutor.h:88).

B2 — Uniqueness
    PASS (CAUTION: registry seqId-only asymmetry, mitigated under current model)
    理由：1O→2E impossible (1 publish call, 1 enqueue, take-once reject-on-key-dup);
      2O→1E impossible (monotonic seqId ⇒ distinct OwnerChannelKey ⇒ distinct take/complete);
      O1/O2→same-Key impossible (3-field key + monotonic epoch/mappedGeneration +
      enqueue reject). Registry (seqId-only) is fallback-only and single-flight under
      single-CoordinatorLoop.
    コード証拠：trySubmitImpl single publish (L96); publishImpl single commit call
      (PublicationExecutor.cpp:46-52); OwnerChannel enqueue reject (OwnerChannel.h:39)
      + take CAS-drain (OwnerChannel.h:70-72); monotonic seqId
      (reserveRuntimePublicationIdentity, SequenceArithmetic.h); registry fallback-only +
      single-flight timeline (AudioEngine.h:4516,4548; RuntimePublishExecutor.h:60-66,85).

B3 — Preservation
    PASS
    理由：The frozen RuntimeState* is object-identical end-to-end (worldOwner.release →
      frozen.releaseState → OwnerChannel enqueue → take re-wrap → publish → registry.lookup
          / payload.newWorld all return the same pointer). The publication triple
      {seqId, epoch, mappedGeneration} is projected verbatim (no arithmetic, no loss).
      The only dropped token (correlationId) is provably O-telemetry, not a binding carrier.
      Timeout preserves Transferred (3B-6); fire-and-forget uses identical key derivation.
    コード証拠：world pointer identity (RuntimePublicationOrchestrator.cpp:88 releaseState,
      PublicationExecutor.cpp:23-24; OwnerChannel.h:66 re-wrap; RuntimePublishExecutor.h:60-67;
      AudioEngine.h:4532 same newWorld); verbatim projection (AudioEngine.h:4508-4510,4530-4537;
      RuntimePublishExecutor.h:50-52); timeout Transferred (AudioEngine.h:4547,4562-4565).

```

## 9. O→E Binding — Final Determination

```text
O→E Binding
    PROVISIONAL

```

### Rationale

The existing code **already establishes** an O→E semantic relation: each Accepted obligation
produces one frozen world whose publication triple is preserved verbatim into the E transfer key,
the intent, the registry (fallback), and the completion seqId. Object identity (the `RuntimeState*`)
is preserved end-to-end. No O Accepted obligation is lost, duplicated, or mis-bound to a different
execution.

The determination is **PROVISIONAL** (not PROVEN) for two reasons:

1. **Registry key asymmetry (B2-D)**: `PendingPublishRegistry::lookup(seqId)` is seqId-only while
   `OwnerChannelKey` is composite. This is structurally safe today only because (a) the registry is
   fallback-only and (b) seqId is single-flight under the single-CordinatorLoop model. It is the
   single deferred hardening item — would need epoch/mappedGeneration on registry Entry under a
   future multi-producer model. 3C does **not** propose this change (audit-only).

2. **correlationId is not the binding**: the O Accepted obligation has **no dedicated obligation
   identity token** on the E side — the binding is carried structurally by the frozen world object +
   its publication triple. This is sufficient for existence/uniqueness/preservation, but it means
   there is **no first-class O identity symbol** that transitively annotates the E execution. If a
   literal "O identifier appears in E" is required as a proof convention, 3C cannot produce it
   without code change (forbidden). The structural relation is proven; a symbolic annotation is not.

The binding holds in the **current single-CoordinatorLoop, SPSC OwnerChannel, monotonic seqId** model.
