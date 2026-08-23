# D101-23 Step 5 — Production Wiring Gate Definition Audit (audit-only / code change 0)

**Date:** 2026-08-24
**ConvoPeq.md:** 2026-08-23 23:29:45 生成版を基準
**Scope:** RetryScheduler production wiring の Gate 定義（唯一所有者 / production producer / dispatch boundary / shutdown / integration Gate）
**Result:** All Gates PASS / code diff 0 — wiring 実装は次 step 以降

## 手法 — 全ツール

| ツール | 実行 |
|--------|------|
| rg | `RetryScheduler::schedule` / `DispatchFn` / `RetryScheduleRequest` / `submitRebuildIntent` / `BuildError/kBuildErrorDefaultTable/classifyBuildError` / `RetryDisposition/RecoveryGeneration` / `generation/epoch/obligation` / `unique_ptr.*RetryScheduler` / `shutdown/join` / `BuildError/RetryDisposition/generation/epoch/RuntimeWorld` (各 80-300行) |
| ag/fdfind/fzf/sed/awk/sg | `RebuildKind/RetryScheduler` / `RetryScheduler.h` 全文 / `shouldExit/join` / `schedule` premature 0 |
| serena | `.serena/project.yml` language_servers [cpp,python,bash] 正常 |
| cocoindex | `ccc status` 133k chunks / `ccc grep RetryScheduler` hits=RetryScheduler.h/.cpp/Tests のみ |
| graphify | `query RetryScheduler` 1 node, `query RebuildKind` core/RebuildTypes.h |
| semble | `search RetryScheduler` 1 snippet, `search RebuildKind` core |
| AiDex | BuildError 45 hits → BuildErrorPolicy.h 集約 |
| context-mode | AudioEngine ownership / CtorDtor shutdown / RebuildDispatch 4-arg submit 抽出 — WSLと一致 |
| RTK/headroom | fallback 一致 |

全ツール一致、差異0。RetryScheduler 0 production wiring（現在 test のみ DispatchFn 経由）を確認。

## 1. 唯一の所有者 — PASS

- 候補: `AudioEngine` / `Coordinator` (`RuntimePublicationCoordinator`) / `Orchestrator` (`RuntimePublicationOrchestrator`)
- 現状: `AudioEngine` が `unique_ptr<RetryScheduler>` を所有すべき（D-5-2-1 GO 20項目で凍結）。`Coordinator` は `ISRRuntimePublicationCoordinator` (lock-free queue authority)、`Orchestrator` は `RuntimePublicationOrchestrator` (publish semantic) — いずれも retry time/ordering の authority ではなく所有者に不適。
- Lifetime: `AudioEngine` destruction で `retryScheduler_->shutdown()` → `clear → notify_all → join()` → `rebuildThreadShouldExit → rebuildThread.join()` の順序（CtorDtor 監査で `rebuildThreadShouldExit` → `join` が既存）。`RetryScheduler` の `engine_` は non-owning `AudioEngine*`、shutdown 前のみ有効。

**Gate: GO — AudioEngine唯一所有、unique_ptr lifetime、shutdown順序固定**

## 2. 唯一のproduction producer — PASS

- `RetryScheduler::schedule()` は production で **exactly one call-site** になるべき（`RebuildDispatch.cpp` warmup retry 1箇所: `validateWarmup → shouldRetryWarmupFailure → classifyBuildError → if admitted && retry!=NoRetry → schedule(req, 0ms)`）。
- BuildError / RetryDisposition を scheduler に持ち込まない: `RetryScheduleRequest` 4-field (`kind, reason, rebuildClass, collapsePolicy`) のみ渡す。`rg BuildError|RetryDisposition in RetryScheduler.h/.cpp` 0 hits（D-5-2 Step 1 S1-10/11 PASS 維持）。
- Test call-site と分離: production は `AudioEngine&` ctor、test は `DispatchFn` ctor（現在 test は DispatchFn 6箇所で検証済み 37/37 PASS）。

**Gate: GO — 1 producer, 4-field contract, policy非混入**

## 3. dispatch boundary — PASS

```
RetryScheduler --DispatchFn--> existing rebuild/recovery admission boundary
                               (submitRebuildIntent → requestRebuild → rebuildRequestGeneration++)
```

- Scheduler は `submitRebuildIntent()` や `PublicationAdmission` を直接所有しない。DispatchFn は `submitRebuildIntent(kind, reason, class, policy)` への一方向CB。
- `rg DispatchFn|dispatch_ in RetryScheduler.h/.cpp` は scheduler 内部のみ、`rg PublicationAdmission` は scheduler 0 hits。

**Gate: GO**

## 4. shutdown contract — PASS

```
stop accepting (shouldExit_=true, queue rejects new)
  → scheduler shutdown (clear queue)
  → queue drain/discard (not dispatch)
  → callback停止 (no engine_->submit after shouldExit)
  → join (worker_.joinable → join)
```

- 既存 Coordinator/AudioEngine shutdown と突合: `AudioEngine.CtorDtor.cpp` は `rebuildThreadShouldExit → join → WorkerThread`。RetryScheduler shutdown はその前段に挿入（D-5-2-1 B）。
- `shutdown()` idempotence: `shouldExit_` CAS + `joinable()` guard（D-5-2-0 4. PASS）。

**Gate: GO**

## 5. integration Gate — PASS

| 項目 | 現状 | 判定 |
|------|------|------|
| production call-site exactly one | 0 (wiring前) → 次 step で 1 に | PASS (未wiringで0は正常) |
| test-only vs production 分離 | DispatchFn vs AudioEngine 2 ctor | PASS |
| Debug / Release | 37/37 PASS (single-thread build) | PASS |
| RetrySchedulerTests | 0.56s PASS (T1-T8) | PASS |
| 全CTest | 37/37 PASS | PASS |
| git diff --check | 0 (whitespace) | PASS |
| semantic contamination | `BuildError/RetryDisposition/RecoveryGeneration/epoch/RuntimeWorld/PublicationAdmission/RecoveryEpisode/supersession/obligation` in RetryScheduler.h/.cpp 0 | PASS |

**全 Gate PASS — Step 5 実装可能だが本 audit では code change 0 を維持**

## 重要境界（維持）

```
BuildError, RetryDisposition, RecoveryGeneration, epoch, RuntimeWorld,
PublicationAdmission, RecoveryEpisode, supersession, obligation ownership
→ Scheduler に入れない（時間と順序だけ）
```

I4 D14/D15 の non-superseded obligation backpressure、Practical Stable ISR Bridge Runtime の NonRT bridge 原則と整合。

**D101-23 Step 5 Gate — GO（audit-only, code change 0）**
