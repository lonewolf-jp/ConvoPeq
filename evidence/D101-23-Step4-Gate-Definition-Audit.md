# D101-23 Step 4 — Gate Definition Audit (audit-only / code change 0)

**Date:** 2026-08-23
**Scope:** D-5-2 Step 4 実装可否の Gate 定義監査
**ConvoPeq.md:** 2026-08-23 22:02:11 生成版を基準

## 1. Retry disposition の意味

- `RetryDisposition` (`BuildErrorPolicy.h: NoRetry / RetryBackoff / RetryImmediate`) は `BuildError → FailureClassification` 分類の結果に対する **policy 分類**であり、「retryしてよい」という恒久的分類。
- 「retryをスケジュール済み」という一時的状態ではない。状態は `PendingRetry` の queue 内存在で表現。
- `BuildError` → `classifyBuildError()` → `BuildOutcome{ FailureClassification, RetryDisposition }` で決定。`RetryScheduleRequest` 生成は `RetryDisposition != NoRetry` の場合のみ `schedule()` へ委譲。
- `RetryScheduleRequest` 自体に判断ロジック (`if disposition==...`) を混入させない。4 enum の値コピーのみに限定。

**Gate: GO**

## 2. PendingRetry lifecycle

```
RetryScheduleRequest (caller stack, value)
  ↓ enqueue owner: RetryScheduler::schedule() (NonRT, mutex)
PendingRetry { request, deadline } in queue
  ↓ deadline owner: set at enqueue (steady_clock::now()+delay)
[waiting] in deque (ordered by deadline)
  ↓ due 判定 owner: RetryScheduler::run() thread (wait_until)
due
  ↓ dequeue owner: RetryScheduler::run() (pop_front under lock)
  ↓ dispatch owner: RetryScheduler::run() (unlock → submitRebuildIntent)
terminal disposition owner: submitRebuildIntent 内の既存 admission (not scheduler)
```

各遷移の唯一所有者を固定。PendingRetry は queue 内でのみ生存、dispatch 後は破棄。

**Gate: GO**

## 3. deadline の意味論

- **意味:** 「retry可能になる時刻」(earliest execution time)。期限(deadline after which discard)ではない。
- **clock source:** `std::chrono::steady_clock` 固定 (monotonic, system clock wraparound 影響なし)。
- **到達時:** `wait_until(deadline)` expiry → `submitRebuildIntent()` 呼出 (queue から pop 済み)。
- **過去値:** `deadline <= now` なら即時 dispatch (zero 含む)。
- **ゼロ値:** `delay==0ms` → `deadline==now` → 即時 dispatch (1ms 以内の wake)。
- **overflow/wraparound:** `steady_clock::time_point` は 64bit、overflow 実質なし。`delay` は `milliseconds` (int64) で最大年単位でも表現可能。

**Gate: GO**

## 4. queue/timer ownership Gate

- `RetryScheduler` が **queue と timer の双方を所有** (single thread owns both)。
- Timer callback は存在しない。`std::thread` + `std::condition_variable::wait_until()` で実装。Callback が直接 submission する設計は採用しない。
- Queue consumer と timer は同一 entity (`run()` loop) のため別 producer/consumer にならない。
- Cardinality: **MPSC** — Multiple NonRT producers (`schedule()` を MessageThread / RebuildThread から呼出) → Single consumer (`run()` thread) + `std::mutex`. 既存 recovery queue の「lock-free だから RT-safe」ではなく mutex で保護する architectural invariant。RT producer は 0 (Gate 10 で確認)。

**Gate: GO**

## 5. RetryScheduler と既存 Authority の境界

Gate として明文化 (RetryScheduler.h に持たせない):

- `generation` 生成・判定: `rebuildRequestGeneration` / `isObsolete()` に委譲
- `epoch` 生成・判定: `EpochDomain` に委譲
- `BuildError` 意味判断: `BuildErrorPolicy::classifyBuildError()` に委譲
- `RuntimeWorld` 参照・変更: なし
- `publication authority`: `RuntimePublicationCoordinator` に委譲
- `recovery obligation supersession`: `drainPendingRecoveryAdmission` に委譲 (semantic target containment 必要、domain 一致のみでは不十分)

`semble`/`graphify`/`rg`/`sg`/`cocoindex` 全ツールで `RetryScheduler.h` に上記 0 hits を確認。

**Gate: GO**

## 6. Retry の再試行回数・terminal condition

- **attempt count owner:** Admission 側 (`RebuildDispatch.cpp` の caller が `classifyBuildError` 後に `attempt` を管理)。RetryScheduler は保持しない (D-5-1.1 C2)。
- **max retry 判定:** Admission 側。`PendingRetry` に `attempt` フィールドなし。
- **RetryScheduler の役割:** 既に決定された request を時間的に保持するだけ (delay executor)。Retry policy を決めない。
- **terminal failure:** Obligation disappearance の理由にしない。D14/D15 に従い非 superseded obligation は容量不足でも silently discard せず backpressure。`PendingRetry` queue full は `reject-newest` + counter で backpressure。

**Gate: GO**

## 7. Shutdown Gate

```
accepting retry (schedule() with lock)
  ↓ stop scheduling: shutdown() sets shouldExit=true, queue.clear()
drain PendingRetry (queue.clear() under lock, not dispatch)
  ↓
shutdown disposition: join() scheduler thread → join rebuildThread
```

- Owner: `AudioEngine::~AudioEngine()` (first `retryScheduler_->shutdown()` then `rebuildThreadShouldExit`).
- Order: `shutdown()` → `clear` → `notify_all` → `join` → `rebuildThread.join()`.
- Stable ISR Runtime の `Stop → Drain → Reclaim → Verify` に準拠。PendingRetry は Drain で破棄、Reclaim 不要。

**Gate: GO**

## 8. Step 4 GO/NO-GO 判定

| Gate | 判定 |
|------|------|
| RetryDisposition semantic fixed | GO |
| RetryScheduleRequest ownership fixed | GO |
| PendingRetry ownership fixed | GO |
| deadline semantics fixed | GO |
| queue cardinality fixed | GO |
| timer ownership fixed | GO |
| RetryScheduler authority boundary fixed | GO |
| retry-count/terminal semantics fixed | GO |
| shutdown/drain semantics fixed | GO |
| forbidden field/authority contamination absent | GO |
| code diff remains 0 | GO |

**全 Gate GO — Step 4 implementation GO**

- 実装しないもの (Step 4 Gate Audit): `RetryScheduler class`, `queue member`, `timer member`, `enqueue/dequeue`, `retry callback`, `deadline calculation`, `retry dispatch`, `tests` は全て 0 のまま。
- 次は Step 4 実装 (queue + timer + worker) へ進入可能。`ConvoPeq.md` を基準に単一 queue/timer 所有を維持。

## 検証コマンド

- `rg -n 'RetryDisposition|RetryScheduleRequest|PendingRetry|deadline|queue|timer' src --type cpp --type h` (rg/ast-grep)
- `fdfind`/`ag`/`fzf`/`sed`/`awk` (WSL)
- `serena` (`aidex` 索引整備済み), `cocoindex` (`ccc status`), `graphify query 'RetryDisposition'`, `semble search 'PendingRetry'`, `AiDex` (BuildError 45 hits)

参照: `src/audioengine/RetryScheduler.h` (4-field request + 2-field PendingRetry), `src/audioengine/RetrySchedulerTypes.h` (3 enum), `src/core/RebuildTypes.h` (RebuildKind), `src/audioengine/BuildErrorPolicy.h` (RetryDisposition)
