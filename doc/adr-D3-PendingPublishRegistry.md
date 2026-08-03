# ADR-D3 — PendingPublishRegistry (Lifetime Gap Owner)

**Status:** Accepted (design-locked, pre-implementation)
**Context:** `releaseState()` (audio thread) → `commit()` (ISR thread) has NO owner in between (5-2 consumed `frozen` inline). 5-3 async creates a lifetime gap requiring a registry.

## Decision
- **Registry ≠ Authority.** `PendingPublishRegistry` nests under `RuntimeWorldAuthority` as a SEPARATE responsibility.
- **Owner:** `RuntimeWorldAuthority` (cross-thread anchor via `AudioEngine::worldAuthority()`, AudioEngine.h:4444).
- **Key:** `PublicationSequenceId`.

## Lifecycle
```
REGISTER(seqId, RuntimeState*)
        ↓
    PENDING
   ↙   |   ↘
COMMITTED  TIMEOUT  CANCELLED
```

## API responsibilities
| Op | Thread | Responsibility |
|----|--------|----------------|
| `register(seqId, statePtr)` | Audio (Phase 2 enqueue) | store non-owning ptr from `frozen.releaseState()` |
| `lookup(seqId)` | ISR (Phase 4 commit) | return ptr for `publishAtomic(currentWorld_, …)` |
| `unregister(seqId)` | ISR (post-commit-success only) | remove entry; do NOT destroy (committed ptr owned by currentWorld_) |
| timeout | ISR | `PENDING→CANCELLED`, destroy `RuntimeState*` (never committed) |
| cancel (reject/shutdown) | ISR | `PENDING→CANCELLED`, unregister; destroy only if NOT already handed to commit |

## Evidence
- `coordinator.cpp:110-112` stamps `PublicationSemantic` then `publishAtomic(currentWorld_, newWorld)` — owner is established only at commit.
- Duplicate rejection already enforced by monotonic seq/epoch/gen gate (`coordinator.cpp:96-100`).
- Move-only `frozen`/`ReleaseScope` stays audio-thread; non-owning `RuntimeState*` crosses via `Intent.payload.publish` (POD).

## Consequences
- prev-world retire path unchanged: `bridge.retireRuntimePublishWorldNonRt` (unseal→aligned_free).
- Gate grep targets: `releaseState`, `register`, `unregister`, bare `RuntimeState*` outside Registry = 0.
