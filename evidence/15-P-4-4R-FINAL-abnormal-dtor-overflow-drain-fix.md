# 15-P-4-4R-FINAL: Abnormal Destructor Overflow Drain Fix Verification

## A. Abnormal dtor invokes overflow drain — **FAIL**

`~AudioEngine()` は `drainOverflowRing()` を呼び出しません. `m_retireRouter->drainAll()` は D+Q+E+Terminal のみをドレインし, OverflowRing は対象外です.

## B. Overflow entries reach terminal disposition — **FAIL**

`drainOverflowRing()` → `emitRetireIntent()` は `LifetimeState` の MPSC キュー (`slots_`) に push するだけです. **`dequeuePendingRetireIntents()` が呼ばれない限り, Intent はキューから取り出されず, `DeferredDeletionQueue` への enqueue は行われません.**

`dequeuePendingRetireIntents()` は `AudioEngine.Commit.cpp:492` (コミットパス) でのみ呼び出されます. `~AudioEngine()` では呼び出されません.

**queue transfer ≠ terminal disposition**: `emitRetireIntent()` は**work item を別 queue に移すだけ**です. Intent は `LifetimeState` の MPSC キュー (`slots_`) に停滞し, `LifetimeState` のデストラクタ (trivially destructible) によって破棄されます.

## C. Destruction-order safety — **PASS**

`~AudioEngine()` のデストラクタボディ実行中は全メンバが生存します. `drainOverflowRing()` が `runtimePublicationBridge_` を呼び出す場合, そのメンバは生存しています. `emitRetireIntent()` が `LifetimeState` (RuntimeWorldAuthority 内) にアクセスする場合も, そのメンバは生存しています.

Member destruction order (REVERSE declaration):
1. `dspQuarantineManager_` (line 4780) ← destroyed FIRST
2. `ShutdownRuntime` (line 4787)
3. `runtimePublicationBridge_` (line 4748) ← destroyed later
4. `RuntimeWorldAuthority` (line 4752) ← contains LifetimeState, destroyed later
5. `m_epochDomain` (line 4676) ← contains DeferredDeletionQueue, destroyed after RuntimeWorldAuthority
6. `m_retireRouter` (line 4681)

**In dtor body (before member destruction)**: All members alive. `drainOverflowRing()` + `dequeuePendingRetireIntents()` + `reclaim()` は安全に呼び出せます.

## D. Normal→dtor idempotence — **FAIL**

`drainOverflowRing()` は idempotent です (ring が空の場合は no-op). しかし, `dequeuePendingRetireIntents()` + `reclaim()` は `reclaim()` が slot state を変更するため, 二重呼び出しにより副作用が発生する可能性があります.

**Deeper issue**: `releaseResources()` 自体が `dequeuePendingRetireIntents()` を呼び出しません. 通常パスでも, OverflowRing drain で `emitRetireIntent()` によって MPSC queue に入れられた Intent は, `dequeuePendingRetireIntents()` が呼ばれないため破棄されます.

## E. No silent RetireIntent loss — **FAIL**

`emitRetireIntent()` は `RetireIntent` を `LifetimeState` の MPSC キューに移すだけです. `dequeuePendingRetireIntents()` が呼ばれない限り, Intent はキューから取り出されず, `DeferredDeletionQueue` への enqueue は行われません. `LifetimeState` のデストラクタは trivially destructible であるため, キュー内の Intent は暗黙的に破棄されます.

### Critical finding: Normal shutdown path also fails

`dequeuePendingRetireIntents()` / `dequeueOne()` / `dequeueFallback()` are called **ONLY** in `AudioEngine.Commit.cpp:492` (RT publish commit path). Neither `releaseResources()` nor `~AudioEngine()` calls them.

`isFullyDrained()` (Threading.cpp:114) checks `worldAuthority_.lifetime().pendingIntentCount() == 0`. After `emitRetireIntent()` (ReleaseResources.cpp:230, 263) pushes to MPSC queue, `pendingIntentCount() > 0` permanently. `waitForDrain(2000, 2)` polls `isFullyDrained()` → **times out** → falls to `else` branch in ReleaseResources.cpp:486 which drains D only via `drainDeferredRetireQueues(true)`, **MPSC queue intents are silently discarded** when `LifetimeState` is destroyed.

This is an **existing latent bug in the normal shutdown path too**.

## F. Regression tests — **PASS**

`src/tests/RetireGraceSemanticsTests.cpp` に `testOverflowRingFifoOrder()` (line 193) が存在します. しかし, abnormal destructor path をテストする既存テストはありません.

## G. Normal shutdown path analysis — **FAIL**

```
emitRetireIntent()          ↓ enqueueTicket_++
    → LifetimeState MPSC queue (slots_[256])
      → dequeuePendingRetireIntents() (ONLY in AudioEngine.Commit.cpp:492, RT commit path)
        → dequeueOne() / dequeueFallback()
          → emitIntent + enqueueRetire + settleEpoch + reclaim
            → DeferredDeletionQueue.enqueue()
              → drainAllUnsafe() → deleter(ptr)
```

`releaseResources()` は:
- `emitRetireIntent()` (line 230, 263) → MPSC queue に push
- `drainOverflowRing()` → も `emitRetireIntent()` を呼び出す
- `dequeuePendingRetireIntents()` を呼び出さない

`waitForDrain()` は `pendingIntentCount() == 0` をポーリングするが, drain がないためタイムアウト. MPSC queue の Intent は `LifetimeState` のデストラクタで破棄される.

## H. Root cause reclassification

The root cause is NOT just "OverflowRing not drained in ~AudioEngine()". The deeper issue is:

**`LifetimeState::pendingIntentCount()` (MPSC queue) has NO drain path in either shutdown route.** The only drain (`dequeuePendingRetireIntents()`) is on the RT commit path, which is stopped before shutdown.

`emitRetireIntent()` → MPSC queue → `dequeuePendingRetireIntents()` → `reclaim()` → `DeferredDeletionQueue` → `drainAllUnsafe()`.

OverflowRing drain (`drainOverflowRing()`) only pushes to the MPSC queue via `emitRetireIntent()`. Without MPSC queue drain, the entire chain is broken.

## GAP-CROSS-3

```
GAP-CROSS-3: OPEN

Root cause: Two-layer ownership disposition gap:

Layer 1: OverflowRing not drained in ~AudioEngine()
  - drainOverflowRing() is NOT called in destructor
  - OverflowRing entries are silently abandoned

Layer 2 (deeper): LifetimeState MPSC queue has NO shutdown drain path
  - emitRetireIntent() pushes to MPSC queue (enqueueTicket_++)
  - dequeuePendingRetireIntents() is ONLY called in RT commit path
    (AudioEngine.Commit.cpp:492) — stops before shutdown
  - waitForDrain() hangs on pendingIntentCount() > 0, times out
  - Timed-out path drains D only, MPSC queue intents silently discarded
  - LifetimeState (trivially destructible) destroys queued intents

Both normal AND abnormal shutdown paths silently discard
RetireIntent entries queued in LifetimeState MPSC queue.

VERDICT: drainOverflowRing() alone is insufficient. Need:
  1. drainOverflowRing() in ~AudioEngine()
  2. dequeueOne()/dequeueFallback() loop in BOTH releaseResources() and ~AudioEngine()
     to drain LifetimeState MPSC/fallback queue → DeferredDeletionQueue before member destruction
```

```
15-P-4-4R-FINAL

A. Abnormal dtor invokes overflow drain       FAIL  (drainOverflowRing not called in ~AudioEngine())
B. Overflow entries reach terminal disposition FAIL  (emitRetireIntent queues to MPSC; dequeuePendingRetireIntents not in any shutdown path)
C. Destruction-order safety                    PASS  (dtor body — all members alive)
D. Normal→dtor idempotence                     FAIL  (normal path also fails to drain MPSC queue; timedOut path drops intents)
E. No silent RetireIntent loss                 FAIL  (pendingIntentCount() hangs in isFullyDrained; intents silently discarded in both paths)
F. Regression tests                             PASS  (existing OverflowRing tests; no abnormal dtor test)
GAP-CROSS-3:
    OPEN
```
