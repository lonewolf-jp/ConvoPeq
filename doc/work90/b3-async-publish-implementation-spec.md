# B3 — Publish 経路非同期化（OwnerChannel → IntentQueue → executePublish → Core Store publish）

**Status**: IMPLEMENTED as **B4** (2026-08-03). Release build green; 22/22 test exes PASS（21 単体 + AudioEngineHarness IntegrationTest）。
**Scope**: This doc is the (B3) design spec. B4 実装の差分・確定事項は本ドキュメント末尾の **「B4 実装ログ」** に記録する。
**Goal**: Rewrite the **sole publish pipeline** with a single-verify cycle: 一括実装 → Release build → 24 unit tests → Integration → Debug → Regression.

---

## B4 実装ログ (2026-08-03) — B3 spec に対する確定差分

### B4-a. 実装済み項目
- **async facade 化**: `AudioEngine::commitRuntimePublication(world, regCtx, oldHandle)` が唯一の publish 入口。
  register → `PendingPublishRegistry::registerPublish` → `OwnerChannel::enqueue`(key=seq/epoch/mappedGen) →
  ISR `enqueuePublicationIntent(Publish)` → `waitForPublishReceipt(seqId, 250ms)`。activate は呼ばない。
  enqueue 失敗時は take で所有権回収 + registry クリア + ScopeExit rollback → `CallerDestroy`。
  wait タイムアウト後も所有権は移譲済みのため `Transferred`（後続の executePublish が commit）。
- **executePublish 一本化**: `RuntimePublishExecutor.h` の Execution tail が activate 責務を一元化。
  take → `authority.commit` → `publishWorld`(唯一の store swap) → unregister → `onPublishCompleted`
  (DSPTransition + DSPLifetimeManager, null-safe) → `advanceRetireEpoch` → `onPublishCommitted(seq)` → receipt。
- **Bootstrap 同期例外 (B4-a3)**: `AudioEngine.Init.cpp` は `coordinator.publishWorld(std::move(bootstrapWorld))`
  を直接呼ぶ（CoordinatorLoop 起動前の enqueue+wait はデッドロックのため）。**これが publishWorld の唯一の非-ISR 呼び出し**。
- **trySubmit 二重実行削除**: Orchestrator trySubmit の末尾 `onPublishCompleted` + `advanceRetireEpoch` を削除
  （executePublish の tail が実行するため）。B3 Step 7 の「inline-completion 維持」から変更。
- **makePublishDecisionSnapshot**: `AudioEngine.Publication.cpp` に追加。newHandle/oldHandle + CrossfadeAuthority.evaluate
  + HealthState Critical 抑制。型は `RuntimePublicationCoordinator::PublishDecisionSnapshot`（ネスト型、完全修飾）。
- **全 Producer 移行**: PrepareToPlay(#2/#3), ReleaseResources(#4), Timer(#5), Transition.publishIdleWorldOnly(#6),
  Rebuild(#7)。idle publish は `DSPHandle::null()` 固定、Rebuild のみ current active DSP handle を渡す。

### B4-b. 検証
- Release build green（mkl.h の環境ノイズ以外 diagnostic なし）。
- build\Release\*Tests.exe 21 本すべて exit=0 PASS。
- **AudioEngineHarness IntegrationTest（実 Audio Thread、B4-c#1 完了）**: 4 ケースすべて PASS。
  - `src/tests/AudioEngineHarness/`（AudioEngineHarness.h/.cpp + PublishPipelineIntegrationTests.cpp）: GUI を起動せず
    実 AudioEngine を実体化する最小ハーネス。MKL setup + initialize + prepareToPlay 後に実 audio thread を spawn し、
    published world の投影値（sequenceId / generation / engine.current / dspProjection）を監視して publish を実測。
  - test1 **rebuild publish**: `AudioEngine.Init.cpp` の Structural rebuild intent → CoordinatorLoop → executePublish →
    store swap（seq 進捗）を確認。PASS。
  - test2 **idle publish (#2)**: `publishIdleWorldOnly` facade → waitForPublishReceipt → seq 一致 swap。PASS。
  - test3 **transition publish (#6)**: rebuild world の active DSP（`w->engine.current`）を渡し HardReset policy の
    publishIdleWorldOnly で seq 前進を確認。PASS。※`getActiveRuntimeDSPHandle()` は IR 未ロード時 null
    （activate は commitRuntimePublication 成功時のみのため）、world 投影値から直接解決する。
  - test4 **teardown publish (#4)**: `releaseResources()` が 15s 以内に復帰（idle publish → waitForPublishReceipt →
    shutdownCoordinatorLoop join まで同期実行、デッドロックなし）。PASS。
  - 実装詳細: AudioEngineHarness は `std::unique_ptr<AudioEngine>`（`sizeof(AudioEngine)=19,236,032B` のため
    スタック配置不可）。CMakeLists.txt で AVX2 フラグ（fastTanhV256 の `__AVX2__/__FMA__` ガード）と
    IPP::ippcore/IPP::ipps（ProductionFft::createPlan / NoiseShaperLearner の ippsMalloc_8u 等）をリンク。
- publishWorld 呼び出しは executePublish と Bootstrap の 2 箇所のみ（重複 swap なし）。
- activation null-safety: DSPLifetimeManager::activate(nullptr) は early-return（idle publish 安全）。

### B4-c. 残課題（本 spec の Section 5 相当、次サイクル）
- [x] IntegrationTest（audio thread）: idle publish の挙動実測（xrun/glitch なし、publish レイテンシ既存 bound 内）。
      → AudioEngineHarness 4 ケース PASS（2026-08-03）。実測ベースの xrun/glitch 計測は GUI 起動が必要なため未実施。
- [ ] Debug-config 実行: ASan/race レポートなし。
- [ ] 回帰: boot→idle, パラメータ変更→publish, IR swap, EQ toggle, prepare/release サイクル, shutdown,
      queue-full 強制（backpressure）。
- [ ] 旧 `PublishCommitResult` の sync 前提呼び出しが残っていないことの最終 grep。

---

## 0. Decision (Session precedent)

The user confirmed **stop at green baseline in this session** and produce review-only artifacts. Rationale:
`commitRuntimePublication()` の契約変更 / `executePublish()` の責務変更 / `PublicationExecutor` 同期→非同期変更 / bootstrap install 切替 / publish call site 一斉変更 は、個別修正ではなく **一つの Publish Pipeline 全体の書き換え**。コンパイル不能・テスト未実施の状態でリアルタイム経路を触ると、semantics が 1 つ外れるだけで「コンパイルは通るが意味論が壊れる」。したがって **Green を保持し、設計のみ固定**する。

---

## 1. Target pipeline (B3 v4)

```
Producer (Non-RT publish thread)
    │  build world, transfer const-ownership into OwnerChannel
    ▼
RuntimeWorldAuthority::ownerChannel<aligned_unique_ptr<const RuntimeState>>  (by value)
    │  key = (seq, epoch, mappedGen); OwnerPtr stays resident until take()
    ▼
RuntimePublicationCoordinator::enqueuePublicationIntent(Intent{Publish})   (payload-only)
    ▼  intentQueue_  (existing LockFreeRingBuffer)
ISR Coordinator Loop
    ▼  processIntent → PublishIntentHandler → PublicationExecutor
PublicationExecutor::executePublish  (Take → Commit → Core-store-swap → Completion)
    │  take(owner) from OwnerChannel (release carves ownership)
    │  coordinator.commit(...) → Store publish  →   Completion → onPublishCommitted(seq)
    ▼
activate / crossfade / retire  tail  (unchanged seams)
```

---

## 2. Verified additive edits (ALREADY APPLIED, GREEN — do not redo)

These two were confirmed compile-safe (additive, non-semantic). Keep them as the baseline.

| File | Edit | Config |
|------|------|--------|
| `src/audioengine/RuntimeWorldAuthority.h` | `OwnerChannel<aligned_unique_ptr<const RuntimeState>>` by value; `ownerChannel()` getter; `RuntimeOwner` / `OwnerChannelType` aliases; global `struct RuntimeState;` fwd-decl (in the `.h`'s forward-decl region) | ✓ present |
| `src/audioengine/ISRRuntimePublicationCoordinator.h` | Added `enqueuePublicationIntent(const Intent& intent) noexcept` → forces `type = IntentType::Publish`, `intentQueue_.push(prepared)`, returns `false` if full. | ✓ present (L.225-240) |

---

## 3. Edit order (mechanical, single cycle next session)

Apply in this exact order. Each step is the smallest self-contained diff; do **not** proceed to the next until the previous is confirmed by compile-her them.

### Step 1 — `RuntimeAuthority.h` (already done) — no-op.

### Step 2 — `ISRRuntimePublicationCoordinator.h` (already done) — no-op.

### Step 3 — `AudioEngine` unchanged **publish-completion seam** (add only)
- Add to coordinator the ISR post-commit notifier that already routes to `transition()`. Confirm `onPublishCommitted(seqId)` remains the single seam.

### Step 4 — `PublicationExecutor` (new file) → async contract
- Change `PublishResult` from sync-completion to async submission.
- `executePublish(...)`: Take fulfillment from owner channel (by key) → `commit(...)` (currently the old-instant) → **Core-store publish** → completion stage → `onPublishCommitted`.
- Return a **new result type** that distinguishes `Queued` (accepted, not yet committed) from `Committed` — do NOT reuse `isCommitted`.
- Sign off: `PublishResult::Queued` replaces the "distinuous/sync" conceptual path.

### Step 5 — `AudioEngine` facade (the seam between RT-NonRT and ISR)
- `commitRuntimePublication(...)`:
  - Become **async**: transfer world ownership into owner channel (keyed), `enqueuePublicationIntent(PublishIntent built from sealedSnapshot)`.
  - Return new `PublishSubmissionResult` enum `{ Queued, Full, Failed }` (NOT the old `PublishCommitResult`).
  - **Registration** (DSPHandle Activate) moves to the ISR publish executor tail (per-call read of regCtx), not the enqueue site.
  - Do **not** call `coordinator.publishWorld()` inline anymore — that Store-swap relocates to executePublish.
- Add `installBootstrapRuntime()` for the first publish — uses the dedicated simple/OneServer path (NOT the async pipeline).
- `registerDSPHandleForRuntime` / activeRuntimeDSPHandle activation: unchanged.

### Step 6 — Update 6 publish call sites
Each converts to the new return type and consumes `Queued` as acceptance (stop awaiting synchronous commit):

| Site | File:line | Current |
|------|-----------|---------|
| Bootstrap | `AudioEngine.Init.cpp:51` | `commitRuntimePublication(...)` → replace with `installBootstrapRuntime(...)` |
| PrepareToPlay (2 calls) | `AudioEngine.Processing.PrepareToPlay.cpp:154,275` | sync → async submission |
| ReleaseResources | `AudioEngine.Processing.ReleaseResources.cpp:173` | same |
| Timer | `AudioEngine.Timer.cpp:915` | same |
| ISR Executor | `PublicationExecutor.cpp:41` | same |

All must: (a) no longer assume Store-publish completed synchronously; (b) treat `Queued` as success head; (c) handle `Failed` by caller-side destroy (existing `OwnershipDisposition` semantics preserved for the post-commit rollback only).

### Step 7 — `RuntimePublicationOrchestrator` publish-completion tail
- `executePublish` calls `onPublishCommitted(seq)` (non-RT). Keep audio-thread `trySubmit` inline-completion path **unchanged** (its contract is not affected).

---

## 4. Invariants that must hold after full implementation

For each publish:
1. **Ownership transfer exactly once** — owner channel Key is carved by exactly one consumer (`Release` in executePublish). No double free, no leak on queue-full (ownerChannel remains resident; caller uses it as the fallback).
2. **Store publish atomic** — the `current` Store swap happens exactly once per publication, only at the ISR executor (`coordinator.commit`), never at the producer.
3. **Activate once** — DSP Handle Activate after commit, not at enqueue (no premature activate of a world you might roll back).
4. **Backpressure explicit** — queue-full ⇒ `Queued`-with-retry or explicit `Failed`; never silent drop.
5. **Bootstrap world present before first RT callback** — `installBootstrapRuntime()` completes synchronously at init; the async path must never run before it.
6. **Drain on shutdown** — outstanding owner-channel residency is reclaimed during teardown (no UAF on Destroy).

---

## 5. Regression / acceptance gate (run next session, single cycle)

- [ ] `build` (Release) — no diagnostics beyond the pre-existing `<mkl.h>` env-only noise.
- [ ] All 24 unit tests green.
- [ ] IntegrationTest (audio thread) — no new xrun/glitches; publish latency within existing bound.
- [ ] Debug-Config test runtime — no ASan/race reports.
- [ ] Regression: boot → idle, parameter change → publish, IR swap, EQ toggle, prepare/release cycle, shutdown, **queue-full forced** (backpressure) — each path hits each return-branch once.

---

## 6. Diff-review checklist (what to re-check at application time)

- `PublishCommitResult` (old) fully replaced by new async result? No stale `isCommitted` reader left.
- `RegistrationContext` activation moved to executor tail; old commit-side `rollbackHandle` scope resolved.
- Bootstrap does not enqueue; uses installBootstrapRuntime.
- The old-instant `commit(...)` Store swap appears in **exactly one** place (executePublish) and nowhere else.
- `RuntimeWorldAuthority` owner-channel addits survive (Step 1/2) unchanged.