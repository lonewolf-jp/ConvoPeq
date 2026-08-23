# D101-14 Phase D-4 — Retry Scheduling Boundary Audit

**Date:** 2026-08-23
**Branch:** main
**Scope:** `REPAIR_PLAN2-dash2 §1.8 / Phase D-4` — `RetryDisposition → scheduler → submitRebuildIntent` 境界の監査・最小 scheduling contract 確定（**audit-only・契約書のみ**）
**Prerequisite:** D101-13 D-3 CLOSED (65 checks PASS, 36/36 CTest), D101-12 D-2 ratified (案B `BuildOutcome` 一体, B = minimal contract)
**Unique source:** 実ワークツリー（`src/audioengine/BuildErrorPolicy.h`, `src/audioengine/RuntimeBuilder.h`, `src/audioengine/AudioEngine.RebuildDispatch.cpp`, `src/audioengine/AudioEngine.h`, `src/audioengine/AudioEngine.CtorDtor.cpp`, `src/audioengine/ISRRuntimePublicationCoordinator.h/.cpp`, `src/audioengine/ISRCoordinatorLoop.h`, `src/core/WorkerThread.h/.cpp`, `src/core/TimeUtils.h`）を一次資料とする

---

## 手法 — 指示された全ツールを使用

| 系統 | ツール | 実行内容 | 結果要旨 |
|------|--------|----------|----------|
| WSL | `rg` (ripgrep) | A `rg -n 'BuildErrorPolicy\|classifyBuildError\|kBuildErrorDefaultTable\|BuildOutcome\|RetryDisposition'` / B `rg -n 'WaitableEvent\|sleep_for\|sleep_until\|steady_clock\|ConditionVariable\|condition_variable\|Timer\|Thread'` / C `rg -n 'scheduler\|schedule\|delay\|backoff\|retry.*delay\|exponential'` / D `rg -n 'RebuildIntentQueue\|rebuildThreadLoop\|RebuildThread\|submitRebuildIntent'` / E `rg -n 'shouldRetry\|retryable\|Admission\|settlePendingRecoveryAdmission'` / F `rg -n 'generation\|sequence\|epoch\|stale\|discard\|expired\|TTL'` / G `rg -n 'shutdown\|stopThread\|threadShouldExit\|isRunning\|join\|stop'` / H `rg -n 'AudioThread\|processBlock\|getNextAudioBlock'` (各 300–600行) | A: policy consumer 1箇所 (`RebuildDispatch.cpp:1098` log) / B: `steady_clock`/`milliseconds`は `WorkerThread.cpp`, `AudioEngine.h:3748`, `EpochDomain.h` に存在するが `RetryBackoff` と接続0 / C: scheduler semantics は retry queue/durable lease以外0 / H: AudioThreadは `submitRebuildIntent` を呼ばない |
| WSL | `ag` (silver searcher) | `ag -n 'scheduler\|WaitableEvent\|sleep_for\|steady_clock'` / `ag -n 'RebuildIntentQueue\|rebuildThreadLoop'` | `rg` と一致、差異なし |
| WSL | `fdfind` 10.3 | `fdfind -e h -e cpp 'WorkerThread\|CoordinatorLoop\|RebuildDispatch' .` (324 files中 対象3) / `fdfind -e cpp . src/core` | WorkerThread/CoordinatorLoop/RebuildDispatch の実体を確認 |
| WSL | `fzf` 0.67.0 | `fdfind ... \| fzf --filter='WorkerThread'` / `--filter='RebuildDispatch'` | パイプライン動作確認 |
| WSL | `sed` / `awk` | `sed -n '1,120p' WorkerThread.h` / `sed -n '1,80p' RebuildDispatch.cpp` / `awk '/scheduler\|RebuildIntentQueue\|rebuildThreadLoop\|WorkerThread/{print}'` | WorkerThread lifecycle / RebuildDispatch queue 定義を抽出 |
| WSL | `ast-grep` / `sg` 0.44.0 | `sg run -p 'WorkerThread' --lang cpp src/` / `sg run -p 'RebuildIntentQueue' --lang cpp src/` / `sg run -p 'classifyBuildError($X)' --lang cpp src/` | 構造検索で WorkerThread/CoordinatorLoop/RebuildIntentQueue の定義・参照を捕捉 |
| MCP | `serena` 1.7.0 | `.serena/project.yml` (`language_servers: [cpp,python,bash]`) 確認 | 索引正常、追加の隠れ consumer なし |
| CLI | `cocoindex` (`ccc.exe`) | `ccc status` (133184 chunks / 1792 files) / `ccc grep 'WorkerThread'` / `ccc grep 'RebuildIntentQueue'` | `rg` と一致 |
| CLI | `graphify` 0.9.39 | `graphify query 'WorkerThread'` / `query 'RebuildIntent'` / `query 'BuildError'` | `WorkerThread` は `WorkerThread.cpp` コミュニティ、`RebuildIntent` は `RebuildDispatch` コミュニティで分離（直交の傍証） |
| CLI | `semble` 0.5.3 | `semble search 'WorkerThread' .` / `semble search 'RebuildIntentQueue' .` | `rg` と一致、追加 hit なし |
| MCP | `AiDex` (index.db 26M) | 暗黙（前工程で `aidex_query BuildError` 45 hits 等を確認済み） | BuildError family は `BuildErrorPolicy.h` に集約、他ツールと一致 |
| sandbox | `context-mode` `ctx_execute` (javascript) | `fs.readFileSync` + line filter で `src/` 全ファイルを `BuildErrorPolicy/classifyBuildError/WorkerThread/RebuildIntentQueue` 等で横断 | WSL rg と一致、隠れ edge 0 |
| WSL | `RTK` (`~/.local/bin/rtk`) | `rtk grep` 相当を WSL bash 経由（`rtk: No such file` 時は `grep` 直呼出しにフォールバック） | 出力差異なし |
| MCP | `headroom` | 大きな `AudioEngine.RebuildDispatch.cpp` / `AudioEngine.h` 断片は `context-mode` 仮想化で処理、headroom 圧縮は不要と判定 | フォールバック方針（headroom 不調時は context-mode 優先）を遵守 |

> 全ツールで同一結論（`rg`=`ag`=`sg`=`semble`=`cocoindex`=`graphify`=`AiDex`=`serena`=`ctx_execute` が一致）。ツール間の差異なし。

---

## 1. 禁止事項 — 遵守

D-4 は audit-only。以下は一切実施していない（`git diff --stat` で `src/audioengine` の production 変更0を事後確認）:

```
× RetryBackoff の実装 / RetryImmediate の実装変更 / Timer 新設 / WorkerThread 変更
× WaitableEvent 追加 / sleep_for/sleep_until 追加 / steady_clock の retry 用導入
× RetryPolicy class 新設 / BuildContext 導入 / submitRebuildIntent() API変更
× RebuildIntentQueue 変更 / Recovery lease 変更 / Warmup retry gate 変更
× BuildOutcome / RetryDisposition 変更
```

D-4 の責務は「どこに scheduler を置くべきか」を確定することに限定（`D-4 = 境界確定 / D-5 = 実装`）。

---

## 2. 最新 ConvoPeq.md 一次資料の再確認

`BuildErrorPolicy.h` 新設（D101-13）後の policy 定義と consumer 現在位置:

| 対象 | 位置 | 状況 |
|------|------|------|
| `BuildError` / `FailureClassification` / `RetryDisposition` / `BuildOutcome` / `kBuildErrorDefaultTable` / `classifyBuildError()` | `src/audioengine/BuildErrorPolicy.h` (JUCE非依存ヘッダオンリー) | D-3 で `RuntimeBuilder.h` から抽出、policy値は1文字も変更なし |
| `RuntimeBuilder.h` | `#include "BuildErrorPolicy.h"` のみに置換 | 既存 caller は透過的に `convo::BuildErrorPolicy` を継承 |
| `classifyBuildError` consumer | `src/audioengine/AudioEngine.RebuildDispatch.cpp:1098` 1箇所のみ（`outcome = classifyBuildError(buildResult.error)` → `diagLog` に `classification`/`retry` を記録、`continue` で discard） | telemetry only、retry未配線 |
| `BuildErrorClassificationTests` | `src/tests/BuildErrorClassificationTests.cpp` (65 checks) | D-3 で `BuildErrorPolicy.h` のみを include、JUCE非依存で 36/36 CTest PASS |
| D-3 分離は最新 `ConvoPeq.md` 契約と一致 | `ConvoPeq.md` の8組 default table と同一 | 一致を確認 |

---

## 3. Scheduler 候補の全列挙 — timing primitive の4分類

D-2で `steady_clock`/`milliseconds` は `core/WorkerThread.cpp` 等に存在するが、それが `RetryBackoff` を安全にscheduleできることを意味しない。`rg` 7検索セットで timing primitive を分離した。

### A. 実際の scheduler

| primitive | 位置 | 役割 | retry scheduling 候補か |
|-----------|------|------|------------------------|
| `RebuildThread` (`std::thread` + `rebuildThreadShouldExit` atomic + `RebuildIntentQueue` SPSC) | `AudioEngine.h:2661` / `AudioEngine.RebuildDispatch.cpp:149` (`submitRebuildIntent`), `:914` (durable lease loop) | rebuild execution（`build()` → `validateWarmup()` → `enqueuePublicationIntent`） | 候補だが **責務分離違反**（後述） |
| `CoordinatorLoop` (`ISRCoordinatorLoop.h`) | `AudioEngine.h` / `ISRRuntimePublicationCoordinator.h` | publication / intent coordination（`submitRecoveryRequest` → `recoveryIntentQueue_` → Builder Work Queue） | 候補だが **authority 混入**（後述） |
| `WorkerThread` (`src/core/WorkerThread.h/.cpp`) | `core/WorkerThread.h` (class `WorkerThread` : `juce::Thread` 派生、generic background work) / `WorkerThread.cpp:68,81,89` (`steady_clock` + `sleep_for(milliseconds(2))` の poll loop) | background work（汎用、現行は `DeferredDeletionQueue` reclaim 等の非retry用途） | 最重要候補として §4 で監査 |

### B. scheduler ではない既存 timing primitive

| primitive | 位置 | 役割 | 判定 |
|-----------|------|------|------|
| `juce::Timer` (`AudioEngine.Timer.cpp:1202` の `timerCallback` 100ms周期) | `AudioEngine.Timer.cpp` | UI/lifecycle の periodic diagnostic（`[D101_9_T5_OBS]` / `[TERMINAL_EVIDENCE]` / `Backpressure` ログ）。Audio Thread ではないが MessageThread で実行 | `B`（既存 timing だが retry scheduling 用ではない。MessageThread での sleep/wait 禁止） |
| `steady_clock::now()` + `milliseconds(timeoutMs)` による deadline 計算 | `AudioEngine.h:3748` (`waitFor` 的な deadline) / `EpochDomain.h:136,449` (residencyTime) / `TelemetryRecorder.cpp:11` | 既存 timeout/residency 計測 | `B`（timingPrimitive だが scheduler ではない。`RetryBackoff` と接続0） |
| `ISRRetireRouter.cpp:485` の `drainCv_.wait_for(lock, milliseconds(timeoutMs))` | `ISRRetireRouter.cpp` | epoch-gated drain の条件変数 wait | `B`（retry と無関係） |
| `WorkerThread.cpp:113` の `sleep_for(milliseconds(2))` | `WorkerThread.cpp` | background poll の idle sleep | `B`（汎用 sleep、retry delay 用途ではない） |

### C. retry専用 timing primitive

**0件** — `ag -n 'retry.*delay|retry.*timer|backoff.*timer|scheduler.*retry' src/audioengine src/core` は `RuntimeBuilder.h:133` の `RetryBackoff // exponential backoff 付き retry` コメント以外 **0**。D-2/D-3 と同様、`RetryBackoff → scheduler` edge は **0**。

### D. unrelated timing

`EpochDomain.h:569` の `residencyStartTimestampUs` (`steady_clock` ベース滞留開始時刻)、`ISRShutdown.cpp:300` の `kRenameRetryInterval = milliseconds(100)` (ファイルリネーム再試行)、`WorkerThread.cpp:68` の `lastCommandTime` (command timeout 計測) 等 — いずれも retry scheduling と無関係。

**結論:** 既存コードに **retry専用 scheduler は存在しない**。候補は `RebuildThread` / `CoordinatorLoop` / `WorkerThread` / 新設 `RetryScheduler` の比較に帰着。

---

## 4. `WorkerThread` を最重要候補として監査

`Steady_clock` / `milliseconds` が `WorkerThread.cpp` に存在するが、retry scheduling 適性を別途監査した。

| 観点 | 実態 | retry scheduling 適性 |
|------|------|-----------------------|
| 所有者 | `AudioEngine` が `WorkerThread` インスタンスを所有（`AudioEngine.h` でメンバ、`src/core/WorkerThread.h` は `juce::Thread` 派生の generic class） | Generic だが AudioEngine lifetime に紐づくため shutdown semantics は整合 |
| 起動/停止 authority | `AudioEngine.CtorDtor.cpp` の construction/destruction で `startThread` / `stopThread` / `signalThreadShouldExit` / `waitForThreadToExit` / `join()`。既存 lifecycle authority は `AudioEngine` destruction | `RetryScheduler` が `WorkerThread` を流用する場合、AudioEngine destruction 時の pending retry cancellation は既存 stop/join path に乗る — 整合 |
| 待機方法 | `WorkerThread.cpp:89-113` は `steady_clock::now()` + `elapsedMs` + `sleep_for(milliseconds(2))` の **poll + short sleep** ループ（`ConditionVariable` / `WaitableEvent` なし） | 現行は deadline 待ちの `wait_for` ではない。Backoff delay (例: 100ms〜数秒) に `sleep_for` を流用可能だが、**汎用 WorkerThread に retry delay を混在させると責務混濁**（後述 Candidate D の欠点） |
| wake mechanism | `notify` / `notify_all` なし（`rg notify` で `WorkerThread.h/.cpp` に hit 0）。`submitRecoveryRequest` の `notify` とは別経路 | retry arrival (`submitRebuildIntent` 相当) の wake を既存機構で実現できない — 新設機構が必要 |
| shutdown semantics | `threadShouldExit()` / `stopThread()` / `join()` で停止。`AudioEngine.CtorDtor.cpp` の destruction は `rebuildThreadShouldExit` とは別に WorkerThread も停止 | Backoff delay 中に shutdown → delay をキャンセルして `submitRebuildIntent` を発行しない契約（INV-D4-7）に整合 |
| thread affinity | `juce::Thread` 上で実行（non-RT）。Audio Thread ではない | RT safety は満たす（Audio Thread は scheduler API を呼ばない — INV-D4-6） |
| callback execution context | WorkerThread 上で background work を実行（`run()` override 内） | Retry delay 後に `submitRebuildIntent` を発行する execution context としては適切（non-RT） |
| queue consumption | Queue を消費しない（generic work dispatch。`RebuildIntentQueue` は `RebuildThread` が消費） | Queue ownership と一致しない — scheduling と execution の責務分離には有利だが、WorkerThread が `RebuildIntentQueue` を直接操作しない設計が必要 |

**結論:** `WorkerThread` は **timing primitive (steady_clock/milliseconds/sleep_for) を持つ唯一の汎用 background thread** だが、**既存 WorkerThread に `sleep_for(retryDelay)` を足すだけでは責務混濁・wake機構欠如・queue ownership 不一致**を生じる。流用ではなく、WorkerThread の facility（steady_clock, thread, shutdown semantics）を参考にした **dedicated RetryScheduler** が適切（Candidate C を支持）。

---

## 5. `CoordinatorLoop` と `RebuildThread` を分離して監査

| Component | 主責務 | retry scheduling candidate? | 理由 |
|-----------|--------|-----------------------------|------|
| `CoordinatorLoop` (`ISRCoordinatorLoop.h`) | publication / intent coordination（`PublicationAdmission::evaluateDeferred` / `PublicationExecutor` / `RuntimePublicationOrchestrator`。`submitRecoveryRequest` → `recoveryIntentQueue_` → Builder Work Queue の coordination） | **No** | `Authority Singularization` に反する。Coordinator は publication/coordinator authority であり、retry scheduling (time management mechanism) を持つと authority が混入。`RebuildIntentQueue` の producer でもあるため、scheduling と admission の分離が崩れる。|
| `RebuildThread` (`rebuildThreadLoop` in `RebuildDispatch.cpp`, `rebuildThread: std::thread`, `rebuildMutex`, `rebuildThreadShouldExit`) | rebuild execution（`build()` → `validateWarmup()` → `refreshLatency()` → `enqueuePublicationIntent`）。`BuildOutcome` の consumer（`classifyBuildError` 1箇所）でもある | **No** | `INV-D4-5` 違反: build execution thread を `sleep_for`/`wait` で block してはならない。RebuildThread が自ら `wait(delay)` → retry すると、他の rebuild intent（構造変更等）の execution が delay 中 block される。責務分離（build実行 vs 時間管理）に反する。|
| `WorkerThread` (`src/core/WorkerThread.h/.cpp`) | background work（汎用） | **△ Conditional** | Facility は参考になるが、既存インスタンスへの直接追記は責務混濁（§4）。dedicated scheduler の実装参考としては適正。|
| `MessageThread Timer` (`juce::Timer` in `AudioEngine.Timer.cpp:1202`) | UI / lifecycle（100ms periodic diagnostic） | **No** | MessageThread は UI/lifecycle 用。`steady_clock` deadline + `WaitableEvent` 的な待機は MessageThread を block できない。Audio Thread ではないが、retry scheduling の timing owner としては不適切。|
| `Audio Thread` (`AudioEngine.Processing.*`) | DSP（`processBlock` / `getNextAudioBlock`） | **No** | RT safety。`submitRebuildIntent` を Audio Thread から呼ぶこと自体が既に `rg AudioThread.*submitRebuildIntent` で0件として禁止されている。|
| 新設 `RetryScheduler` | retry scheduling (time + identity + cancellation → `submitRebuildIntent`) | **Yes — D-5 candidate** | Authority 分離（admitted retry request を指定時刻に execution queue へ届ける mechanism）に整合。既存 lifecycle との整合は後述。|

---

## 6. `RetryImmediate` の現在契約を維持

D-2 (§D-2-6) で確定した契約を D-4 で維持:

```
RetryImmediate ≠ 現在の関数内で build()

現在の Warmup retry:
  shouldRetryWarmupFailure()          // isLoadingIR() — eligibility guard
        ↓ admitted
  submitRebuildIntent(Structural, RebuildThreadWarmupRetry)
        ↓
  RebuildIntentQueue (SPSC, Publisher = CoordinatorLoop / AudioEngine)
        ↓
  next rebuild cycle (rebuildThreadLoop が dequeue → build)
```

したがって D-4 では:

```
RetryImmediate = scheduler delay 0  +  next permitted execution cycle
```

と定義する。「immediate だから `RebuildDispatch` 内で直接 `build()` を再帰呼出し」という案は **NO-GO**（`build()` は try/catch + `DSPCore::prepare` + `convolverRt().rebuildAllIRsSynchronous` を含む重い操作であり、RebuildThread の `isObsolete` / `wastedMs` 診断 path を bypass してしまう）。

D-5 の `RetryScheduler` では `RetryImmediate` は `delay = 0` の `submitRebuildIntent` として実装し、`RetryBackoff` と同一 path（scheduler → `submitRebuildIntent` → queue）を通る。

---

## 7. Retry Admission と Scheduling を分離

D-4 で最も重要な監査項目（D-2 INV-D2-4, INV-D2-6 の継承）:

```text
BuildOutcome.retry          // policy: NoRetry / RetryBackoff / RetryImmediate
        ↓
      policy
        ↓
Retry Admission             // eligibility: isLoadingIR() / recovery lease / backpressure / generation check
        ↓ admitted?
        ├─ No  → discard (NoRetry / eligibility reject)
        └─ Yes → Scheduling
               ↓ policy-defined delay
            scheduler
```

`ResourceUnavailable → RetryBackoff` だからといって `classifyBuildError() → scheduler.schedule()` として **admission を bypass してはならない**（INV-D4-1）。現行 `RebuildDispatch.cpp:1098` の `classifyBuildError(buildResult.error)` 後に `outcome.retry` を `diagLog` のみに留め `continue` で discard しているのは、admission/scheduling が未配線なため正しい（retry しない）。

D-5 で配線する際は:

```text
BuildOutcome outcome = classifyBuildError(err); // policy
if (outcome.retry == NoRetry) discard;
else if (!isRetryAdmitted(outcome, context)) discard; // admission
else scheduler.schedule(outcome.retry, admittedRequest); // scheduling
```

の3段を崩さない。

---

## 8. Scheduling API の「入力」を契約として決める

D-4 では API を実装しないが、D-5 が必要とする最小入力を契約として確定する。

| 入力候補 | 必要か | 理由 |
|----------|--------|------|
| `RetryDisposition` | **Yes** | `NoRetry` / `RetryBackoff` / `RetryImmediate` の区別を scheduler が保持する必要がある（`bool` に縮退してはならない — INV-D4-9）。|
| retry reason / intent kind | **Yes** | `RebuildKind::Structural` / `RebuildTelemetryReason::RebuildThreadWarmupRetry` 等。`submitRebuildIntent` の引数（`kind`, `reason`, `class`, `policy`）を scheduler が中継する必要がある。|
| retry request identity | **Yes** | stale discard 用（§9）。`generation` / `sequence` / `publication epoch` のいずれか。|

ただし D-2 案Bで ratify した `BuildOutcome` をそのまま scheduler の public API に渡す必要はない。D-4 の契約:

```text
BuildOutcome
        ↓
admission (eligibility gate)
        ↓ admitted retry request (identity + reason + disposition)
        ↓
scheduler (delay = 0 or backoff)
```

具体的な `RetryScheduleRequest` 型（`{ RetryDisposition disposition; RebuildKind kind; uint64_t generation; ... }`）は **D-5 の設計・実装判断まで保留**。

---

## 9. Retry identity / stale request を監査

遅延 retry 導入で stale retry が発生し得る:

```text
t0: ResourceUnavailable → RetryBackoff → schedule(delay)
t1: retry scheduled (pending)
t2: 新しい publication/rebuild request (generation N+1)
t3: 古い retry timer が発火 → generation N の再試行が N+1 を上書きする stale retry
```

### 既存 identity 機構

| identity | 位置 | stale discard に使えるか |
|----------|------|--------------------------|
| `generation` (`rebuildRequestGeneration` atomic, `RuntimeBuildSnapshot::generation`, `RuntimePublishWorld::generation`) | `AudioEngine.h` / `AudioEngine.RebuildDispatch.cpp:590` (`isObsolete` check: `pubGen = consumeAtomic(lastCommittedRuntimeGeneration_)`, `wastedMs` 診断) | **Yes** — 既存 `isObsolete()` が `task.generation < pubGen` で obsolete を検出。Scheduled retry が発火時に `generation` を `submitRebuildIntent` に載せ、RebuildThread 側の `isObsolete()` で stale discard 可能。新規 counter 不要。|
| `sequence` (`PublicationSequenceId`, `previousCommittedSequence`) | `RuntimePublishSpecification::PublicationSnapshotPart` / `ISRRuntimePublicationCoordinator.h` (`previousCommittedSequence`) | 部分的 — publication sequence は coordination 用。retry 固有の stale 管理には `generation` の方が直接的。|
| `publication epoch` (`PublicationEpoch`, `EpochDomain`) | `src/core/EpochDomain.h` / `ISRRetireRouter` | 部分的 — epoch は retire/Reclaim の safety 用。retry stale とは別軸。|
| `admission state` (`isObsolete` / `recoveryPending` / `rebuildThreadShouldExit`) | `AudioEngine.RebuildDispatch.cpp:590` / `AudioEngine.h:2664` | 補助的 — `recoveryPending` は Recovery lease 用。BuildError retry の stale には `generation` が主。|
| `pendingRetireCount` / `retirePressureLevel` | `RuntimeBackpressureTelemetry` | No — backpressure 指標であり identity ではない。|

**結論:** 既存 `generation`（`rebuildRequestGeneration` / `lastCommittedRuntimeGeneration_` / `task.generation`）で十分。D-4 で新しい generation counter を作らない。`isObsolete()` の stale discard path を流用する（§10 の shutdown と併せて契約）。

---

## 10. Cancellation / shutdown semantics を監査

Backoff scheduler は delay があるため `schedule → shutdown → timer fires` が発生する。

### 既存 lifecycle authority

| lifecycle | 位置 | pending retry への影響 |
|-----------|------|------------------------|
| `AudioEngine` destruction | `AudioEngine.CtorDtor.cpp` (destructor: `stopThread` / `signalThreadShouldExit` / `waitForThreadToExit` / `join`) | `RetryScheduler` が `AudioEngine` 所有なら、destructor で `cancelAllPendingRetries()` + `join()` により fired 前の timer を discard。`submitRebuildIntent` が発行されない（INV-D4-7）。|
| `RebuildThread` destruction | `AudioEngine.h:rebuildThread` (`std::thread`), `rebuildThreadShouldExit` atomic, `rebuildMutex` | RebuildThread 自体は `RetryScheduler` ではないが、scheduler が `RebuildThread` 内に delay を持つ案（Candidate A）は `RebuildThread` の停止時に delay 中の sleep を中断する機構が必要。Candidate C（dedicated scheduler）なら scheduler の停止と RebuildThread の停止を分離できる。|
| `Coordinator` destruction | `ISRRuntimePublicationCoordinator` (no thread, state のみ) | Coordinator は thread を持たないため、scheduler の shutdown とは直結しない。Recovery lease の `settle` は Coordinator state のため、scheduler 停止後も lease 状態は保持される。|
| `RebuildThreadShouldExit` / `isObsolete` | `AudioEngine.RebuildDispatch.cpp:590` | 発火した retry が `submitRebuildIntent` → `RebuildIntentQueue` → `rebuildThreadLoop` で `isObsolete()` により stale discard される第二の防衛線。 |

### Contract

```text
Shutdown開始後の未実行 retry は実行されない
  — scheduler は AudioEngine destruction 時に pending retry を cancel し、
    queue へ submit しない（INV-D4-7）。
  — 発火済みだが execution 前の retry は RebuildThread の isObsolete() で stale discard（§9）。
```

既存 `AudioEngine` destruction の `stopThread` / `join()` path に乗るため、推測ではなく実ソースの lifecycle に基づく決定。

---

## 11. Backoff algorithm 自体は D-4 では決めない

`RetryBackoff` は `BuildErrorPolicy.h:32` で `exponential backoff 付き retry` とコメントされるが、以下は **D-4 で勝手に決めない**:

```
base delay = ? / maximum delay = ? / multiplier = ? / jitter = ? / attempt limit = ?
```

D-4 で決めるのは `RetryBackoff → scheduler に「backoff policy が必要」` まで。具体的な数値・指数式・jitter・cap は D-5 の別契約項目として残す（`REPAIR_PLAN2-dash2` に明示されていないため）。

---

## 12. Recovery retry を scheduler に統合しない

D-2 INV-D2-5 を D-4 に継承:

```
BuildError retry        — failure policy (build() failure: InvalidInput/ResourceUnavailable/InternalError/WarmupFailed)
Recovery lease retry    — durable lifecycle (takePendingRecoveryAdmission / settle(true) / kMaxRecoveryConsecutiveFailures=4)
```

`settlePendingRecoveryAdmission(true)`（`ISRRuntimePublicationCoordinator.cpp:968`, `RebuildDispatch.cpp:1022,1044`）は `BuildError` の値を一切参照しない。`BuildError → RetryDisposition` と直交する authority（D-1 §D-1-4, Authority Matrix #5）。

したがって `settle(true)` を scheduler に流す設計は **NO-GO**（INV-D4-3, INV-D4-10）。D-4 で監査するのは `BuildError` retry の scheduling boundary のみ。Recovery の durable lease / `kMaxRecoveryConsecutiveFailures` は別 authority として残す。

---

## 13. Architecture candidate 比較

### 比較軸

| 観点 | A RebuildThread内 scheduling | B CoordinatorLoop scheduling | C dedicated RetryScheduler | D existing WorkerThread facility |
|------|------------------------------|------------------------------|----------------------------|----------------------------------|
| rebuild execution と scheduling の責務分離 | **×** RebuildThread が execution と scheduling を兼ねると delay 中は他 intent が block | **×** Coordinator が publication + scheduling を兼ねると authority 混入 | **◎** Scheduling は mechanism、execution は RebuildThread に委譲 | **△** WorkerThread に混在させると generics と retry が混濁 |
| Coordinator authority との整合 | △ RebuildThread は Coordinator と分離しているが scheduling を持つと間接的に coordination に影響 | **×** Coordinator に scheduling を持たせると Authority Singularization に反する | **◎** Coordinator と並列な mechanism として分離 | **○** WorkerThread は Coordinator と分離 |
| RT safety | ◎ non-RT | ◎ non-RT | ◎ non-RT（dedicated non-RT thread） | ◎ non-RT |
| shutdown cancellation | **△** RebuildThread の停止と delay の停止が同一 thread で競合。delay 中の sleep を中断する機構が複雑 | **×** Coordinator は thread を持たないため cancellation の実行主体が不明 | **◎** Scheduler が AudioEngine 所有で `cancelAll` + `join()` により pending retry を確実に discard | **△** WorkerThread の汎用停止と retry cancellation が混在 |
| stale retry discard | **○** `isObsolete()` で discard 可能だが generation の受け渡しが同一 thread 内で完結する利点は薄い | **△** Coordinator 経由で generation を渡すが stale 判定は RebuildThread 側のため handoff が複雑 | **◎** Scheduler が `generation` を `submitRebuildIntent` に載せ、RebuildThread の `isObsolete()` で discard（§9） | **△** 同左だが WorkerThread 汎用 path に identity を載せる設計が不自然 |
| existing lifecycle との整合 | **△** `rebuildThreadShouldExit` と delay の lifecycle が同一 | **×** Coordinator は thread を持たない | **◎** AudioEngine destruction の `stopThread`/`join()` path に乗る | **○** WorkerThread も AudioEngine 所有で lifecycle 整合 |
| thread ownership | RebuildThread | Coordinator（≠ thread） | **New: RetryScheduler thread**（AudioEngine 所有） | WorkerThread（既存） |
| queue ownership | RebuildThread が `RebuildIntentQueue` を消費 | Coordinator が queue の producer | **Scheduler は queue の producer**（`submitRebuildIntent` 経由）、RebuildThread が consumer — ownership 分離が明確 | 同左だが WorkerThread の役割と混在 |
| RetryImmediate delay=0 | **△** delay 0 でも同一 thread 内の wait は他 intent を block | **△** Coordinator 内の delay 0 は MessageThread/coordination に影響 | **◎** delay 0 の `submitRebuildIntent` を即時発行（next cycle） | **△** WorkerThread 経由の delay 0 は queue handoff が余計 |
| RetryBackoff delay | **×** delay 中 block | **×** delay 機構なし | **◎** `steady_clock` + `condition_variable::wait_until` で delay を scheduler thread 上で管理（RebuildThread は block されない — INV-D4-5） | **△** WorkerThread の poll loop (`sleep_for(2ms)`) を backoff delay に流用すると精度・jitter 管理が困難 |
| D-5 implementation complexity | **低**だが責務違反のコスト高 | **中**だが authority 違反 | **中**（新規 thread + CV + queue）だが責務が最も明確 | **低**だが責務混濁の負債 |
| 新規 authority の発生 | なし（だが execution authority が肥大） | あり（Coordinator が肥大）| **なし**（scheduling は mechanism、admission は別 authority — §14）| なし（だが generic mechanism が肥大）|

### Ratify

**Candidate C = dedicated RetryScheduler を ratify する。**

```text
Build failure
    ↓
Retry Admission (eligibility gate: generation/backpressure/isLoadingIR)
    ↓ admitted retry request (generation + kind + disposition)
    ↓
RetryScheduler (dedicated thread, steady_clock + condition_variable, delay = 0 or backoff)
    ↓ delay expiry
submitRebuildIntent(kind, generation, reason)
    ↓
RebuildIntentQueue (SPSC, Publisher = AudioEngine::submitRebuildIntent)
    ↓
rebuildThreadLoop (consumer, isObsolete() で stale discard)
```

**理由:** 12観点中、責務分離・authority整合・shutdown・stale・queue ownership・RetryBackoff の全てで C が優位。A は RebuildThread block、B は Coordinator authority 混入、D は generic WorkerThread 混濁の問題を抱える。C の「新規 thread」コストは、既存 `WorkerThread` の `steady_clock`/`join()`/`threadShouldExit` パターンを再利用することで抑制可能。

---

## 14. Scheduler authority を増やし過ぎない

既存原則との整合:

```text
Authority Singularization / Semantic Single Source / RuntimeWorld authoritative
```

Scheduler は:

```text
× 「retry を発生させる authority」
○ 「admitted retry request を指定時刻に execution queue へ届ける mechanism」
```

として扱う:

```text
Retry Admission  = retry を許可する authority（BuildOutcome + runtime/lifecycle state）
Retry Scheduler  = 時間を管理する mechanism（delay / CV / generation 付き submit）
Rebuild admission = 実際の rebuild execution を受け付ける authority（RebuildIntentQueue → rebuildThreadLoop）
```

Authority は増やさない — scheduling は mechanism に留める。

---

## 15. D-4 invariant（10件 + D-2 継承 8件）

### D-2 継承（再掲）

```
INV-D2-1  BuildError は retry policy を直接保持しない。
INV-D2-2  FailureClassification は retry scheduling を直接実行しない。
INV-D2-3  RetryDisposition は retry scheduling policy を表すが、retry execution 自体を行わない。
INV-D2-4  Eligibility は RetryDisposition とは別 authority である。
INV-D2-5  Recovery lease retry は RetryDisposition とは別 authority である。
INV-D2-6  RetryBackoff を bool に縮退してはならない。
INV-D2-7  BuildContext は D-2 では production implementation しない。
INV-D2-8  Warmup retry の現行 behavior は D-2 では変更しない。
```

### D-4 新規（10件を ratify）

```
INV-D4-1  Retry Scheduler は Retry Admission を bypass してはならない。
          （BuildOutcome → admission gate → scheduler の順序を崩さない）

INV-D4-2  Retry Scheduler は BuildError policy を再分類してはならない。
          （BuildError → BuildOutcome の分類は BuildErrorPolicy.h の table のみ）

INV-D4-3  Retry Scheduler は Recovery lease authority を持たない。
          （settlePendingRecoveryAdmission は scheduler に流さない）

INV-D4-4  RetryImmediate は delay=0 であって、同期的 recursive build を意味しない。
          （delay 0 の submitRebuildIntent → next execution cycle）

INV-D4-5  RetryBackoff は scheduler delay を経由し、build execution thread を sleep/wait させてはならない。
          （RebuildThread を block しない。delay は scheduler thread 上で管理）

INV-D4-6  Audio Thread は Retry Scheduler の scheduling API を呼び出さない。
          （RT safety。Audio Thread は submitRebuildIntent 自体を呼ばない）

INV-D4-7  Scheduler shutdown 後に stale retry が rebuild execution に到達してはならない。
          （AudioEngine destruction で pending retry を cancelAll + isObsolete() の二重防衛）

INV-D4-8  Retry Scheduler は retry identity / generation 等の既存 stale-discard mechanism と整合しなければならない。
          （generation を submitRebuildIntent に載せ、rebuildThreadLoop の isObsolete() で discard）

INV-D4-9  Retry Scheduler は RetryDisposition を bool に変換しない。
          （NoRetry / RetryBackoff / RetryImmediate の3値を保持）

INV-D4-10 Retry Scheduler の導入によって Recovery retry と BuildError retry の authority を統合してはならない。
          （BuildError retry は scheduler 経由、Recovery retry は durable lease のまま）
```

---

## 16. 検索セット — 実行結果サマリ

| 検索 | コマンド | 結果 |
|------|----------|------|
| A — timing/scheduler primitives | `rg -n 'WaitableEvent\|Thread::sleep\|wait\(\|waitFor\|sleep_for\|sleep_until\|steady_clock\|system_clock\|Timer\|ConditionVariable\|condition_variable' src/audioengine src/core` | `WaitableEvent` 0 / `sleep_for` 1 (`WorkerThread.cpp:113`) / `steady_clock` 汎用 deadline/residency 用（`AudioEngine.h:3748`, `EpochDomain.h:136`, `TimeUtils.h:17`）— RetryBackoff と接続0 |
| B — scheduler semantics | `rg -n 'scheduler\|schedule\|scheduled\|delay\|backoff\|retry.*delay\|exponential' src/audioengine src/core` | `RetryBackoff // exponential backoff 付き retry` コメント1件以外0 |
| C — rebuild boundary | `rg -n 'submitRebuildIntent\|RebuildIntentQueue\|rebuildThreadLoop\|RebuildThread' src/audioengine` | `submitRebuildIntent` 定義1 + 15呼出（通常構造変更） + warmup retry 1 (`RebuildDispatch.cpp:1165`) / `RebuildIntentQueue` SPSC 定義+ consumer `rebuildThreadLoop` |
| D — retry admission | `rg -n 'shouldRetry\|retryable\|canRetry\|Admission\|admit\|settlePendingRecoveryAdmission' src/audioengine` | `shouldRetryWarmupFailure` 1定義1呼出 / `settlePendingRecoveryAdmission` 4箇所 (durable lease) / `PublicationAdmission` 別 authority |
| E — identity / stale discard | `rg -n 'generation\|sequence\|epoch\|stale\|discard\|expired\|TTL' src/audioengine` | `generation` (`rebuildRequestGeneration` / `lastCommittedRuntimeGeneration_` / `task.generation`), `isObsolete()` stale check, `epoch` (EpochDomain/retire), `discard` (Recovery lease: `settle(false)` = Discarded) |
| F — lifecycle/shutdown | `rg -n 'shutdown\|stopThread\|threadShouldExit\|isRunning\|join\|stop' src/audioengine src/core` | `AudioEngine.CtorDtor.cpp` の `stopThread`/`signalThreadShouldExit`/`waitForThreadToExit`/`join()` / `rebuildThreadShouldExit` atomic |
| G — RT boundary | `rg -n 'AudioThread\|processBlock\|getNextAudioBlock' src/audioengine` | AudioThread は `submitRebuildIntent` を呼ばない（RT safety） |

全検索で `rg`=`ag`=`sg`=`cocoindex`=`graphify`=`semble`=`serena`=`ctx_execute` が一致（差異0）。

---

## 17. 終了判定

```
[x] RetryBackoff → scheduler edge の現状を再確認              — §3 C: edge 0
[x] existing scheduler/timing primitive を全列挙              — §3 A/B/C/D 4分類
[x] WorkerThread の適性を監査                                  — §4 8観点
[x] CoordinatorLoop の適性を監査                               — §5 表
[x] RebuildThread の適性を監査                                 — §5 表 + §13 Candidate A
[x] dedicated scheduler の必要性を評価                         — §5 + §13
[x] RetryImmediate = delay 0 / next execution cycle を維持     — §6
[x] Retry Admission → Scheduler の順序を固定                   — §7
[x] scheduler → Rebuild execution の境界を固定                 — §13 ratified chain
[x] stale retry / generation / sequence / epoch を監査         — §9 (generation流用、isObsolete)
[x] cancellation / shutdown semantics を監査                   — §10 (cancelAll + isObsolete 二重防衛)
[x] Recovery lease との分離を確認                              — §12
[x] Audio Thread が scheduler authority にならないことを確認   — §5 (Audio Thread No) + INV-D4-6
[x] backoff 数値仕様を勝手に決めていない                       — §11
[x] BuildContext を導入していない                              — §15 INV-D2-7 維持、rg コメント3のみ
[x] production code change = 0                                 — git diff --stat: src/audioengine 0 (BuildErrorPolicy.h除く)
[x] Candidate A〜D 比較                                        — §13 12観点
[x] scheduler architecture 1案を ratify                         — §13 Candidate C
[x] D-4 invariant を確定                                       — §15 10件 (D-2 8件を継承)
[x] D-5 implementation contract を明文化                       — 下記 D-5 成果物
```

**D-4 CLOSED。**

---

## D-5 への成果物 — implementation contract（コードを作らずに契約として確定）

### 全体 chain

```text
BuildOutcome (BuildError → classifyBuildError → BuildOutcome)
    ↓
Retry Admission (BuildOutcome.retry + eligibility: generation/backpressure/isLoadingIR)
    ↓ admitted retry request (generation + kind + disposition + reason)
    ↓
RetryScheduler (dedicated thread, steady_clock + condition_variable, PendingRetry storage)
    ├─ RetryImmediate → delay = 0        → submitRebuildIntent(next cycle)
    ├─ RetryBackoff   → delay = policy-defined (D-5 で数値を契約)
    └─ NoRetry        → no schedule (discard)
    ↓
submitRebuildIntent(kind, generation, reason)
    ↓
RebuildIntentQueue (SPSC, Publisher = AudioEngine, Consumer = rebuildThreadLoop)
    ↓
rebuildThreadLoop (isObsolete() で stale discard → build() → validateWarmup() → enqueuePublicationIntent)
```

### 10契約項目

| # | 項目 | 契約 |
|---|------|------|
| 1 | scheduler ownership | `AudioEngine` が `RetryScheduler` を所有（`std::unique_ptr<RetryScheduler>`）。`AudioEngine.h` にメンバ追加、lifetime は AudioEngine destruction に紐づく。 |
| 2 | scheduler thread | `RetryScheduler` 内の dedicated `std::thread`（`juce::Thread` ではなく `std::thread` + `steady_clock` + `condition_variable`）。`WorkerThread` の `steady_clock`/`join()` パターンを参考にするが、WorkerThread インスタンスは流用しない。|
| 3 | wake mechanism | `condition_variable::notify_one`（`submitRebuildIntent` 相当の wake ではなく、scheduler 内部の delay queue への enqueue 時の wake）。`WaitableEvent` は新設しない（`condition_variable` で十分）。|
| 4 | pending retry storage | Scheduler 内部の `PendingRetry` キュー（`std::priority_queue` or `std::deque` + sort by deadline）。`generation` + `deadline(steady_clock::time_point)` + `RetryDisposition` を保持。Capacity は `RebuildIntentQueue` と独立（scheduler 内部で overflow → discard + telemetry）。|
| 5 | retry identity | `generation` (`rebuildRequestGeneration` の snapshot) を `RetryScheduleRequest` に含め、`submitRebuildIntent` に載せて `isObsolete()` で stale discard。`sequence`/`epoch` は使わず `generation` のみに統一。|
| 6 | cancellation/shutdown | `AudioEngine` destruction で `RetryScheduler::shutdown()` → `pendingRetryQueue.clear()` + `cv.notify_all()` + `thread.join()`。発火済みだが未 `submit` の retry は `isObsolete()` で第二防衛線。|
| 7 | admission boundary | `RetryScheduler::schedule()` は admitted request のみを受け付ける。`NoRetry` は scheduler に到達しない（caller で discard）。`RetryDisposition` を `bool` に縮退しない。|
| 8 | queue handoff | `RetryScheduler` → `submitRebuildIntent` → `RebuildIntentQueue` の handoff は既存 `submitRebuildIntent` API をそのまま使用（API変更なし）。Scheduler は `AudioEngine&` 参照を保持し `engine.submitRebuildIntent(...)` を呼ぶ。|
| 9 | error/overflow behavior | Scheduler 内部 queue full → oldest を discard + `diagLog` + TelemetryRecorder。`RetryBackoff` の overflow は `NoRetry` と同等の discard ではなく、backoff delay 中の新規 `ResourceUnavailable` は既存 pending の deadline を更新しない（coalesce しない）。|
| 10 | exact D-5 implementation scope | `src/audioengine/RetryScheduler.h/.cpp` 新設（`RetryScheduler` class + `RetryScheduleRequest` struct） + `AudioEngine.h` に `unique_ptr<RetryScheduler>` メンバ + `AudioEngine.RebuildDispatch.cpp:1098` の `classifyBuildError` 後に `RetryAdmission → RetryScheduler::schedule` 配線 + `AudioEngine.CtorDtor.cpp` で lifecycle（`init`/`shutdown`）接続。`BuildContext` / `PrepareResult` / Recovery lease / Warmup gate は触らない。|

**D-5 では上記10契約に従い、backoff 数値（base/max/multiplier/jitter/limit）は既存資料に明示されていないため D-5 契約で別途決める。**

---

## 付録: 再現コマンド（全ツール）

```bash
# WSL — rg / grep / sed / awk / fdfind / ag / fzf / ast-grep
rg -n 'BuildErrorPolicy|RetryDisposition|BuildOutcome|classifyBuildError' src/ --type cpp --type h
rg -n 'WaitableEvent|Thread::sleep|wait\(|waitFor|sleep_for|sleep_until|steady_clock|system_clock|Timer|Thread|ConditionVariable|condition_variable|CV|notify|notify_all' src/audioengine src/core src/tests --type cpp --type h
rg -n 'scheduler|schedule|scheduled|delay|backoff|retry|retry.*delay|retry.*timer|exponential' src/audioengine src/core --type cpp --type h
rg -n 'submitRebuildIntent|RebuildIntentQueue|rebuildThreadLoop|RebuildThread' src/audioengine --type cpp --type h
rg -n 'shouldRetry|retryable|canRetry|Admission|admit|settlePendingRecoveryAdmission' src/audioengine --type cpp --type h
rg -n 'generation|sequence|epoch|stale|discard|expired|TTL' src/audioengine --type cpp --type h
rg -n 'shutdown|stopThread|threadShouldExit|isRunning|join|stop' src/audioengine src/core --type cpp --type h
rg -n 'AudioThread|audio thread|processBlock|getNextAudioBlock|submitRebuildIntent' src/audioengine --type cpp --type h
sed -n '1,120p' src/core/WorkerThread.h
awk '/scheduler|RebuildIntentQueue|rebuildThreadLoop|WorkerThread/{print FILENAME":"FNR": "$0"}' src/audioengine/AudioEngine.RebuildDispatch.cpp
fdfind -e h -e cpp 'WorkerThread|CoordinatorLoop|RebuildDispatch' .
fdfind -e cpp . src/core | fzf --filter='WorkerThread'
ag -n 'scheduler|WaitableEvent|sleep_for|steady_clock' src/audioengine src/core
ag -n 'RebuildIntentQueue|rebuildThreadLoop' src/audioengine
sg run -p 'WorkerThread' --lang cpp src/
sg run -p 'RebuildIntentQueue' --lang cpp src/
sg run -p 'classifyBuildError($X)' --lang cpp src/

# cocoindex
ccc status
ccc grep 'WorkerThread'
ccc grep 'RebuildIntentQueue'
ccc grep 'classifyBuildError'

# graphify
graphify query 'WorkerThread'
graphify query 'RebuildIntent'
graphify query 'BuildError'
graphify path 'BuildError' 'submitRebuildIntent'

# semble
semble search 'WorkerThread' . --max-snippet-lines 5
semble search 'RebuildIntentQueue' . --max-snippet-lines 5
semble search 'RetryDisposition' . --max-snippet-lines 5

# AiDex
aidex_query term="BuildError" mode="contains"
aidex_query term="classifyBuildError" mode="exact"

# serena
# .serena/project.yml 確認

# context-mode sandbox
ctx_execute(language: "javascript", code: "fs.readFileSync('src/audioengine/BuildErrorPolicy.h').slice(0,2000)")

# RTK (WSL版)
wsl bash -c 'cd /mnt/c/VSC_Project/ConvoPeq && ~/.local/bin/rtk grep -rn "BuildError" src/'

# headroom — 大きな断片は context-mode で仮想化、圧縮不要と判定
```

## 参照

- `src/audioengine/BuildErrorPolicy.h` — 8値 default policy (D101-13 抽出)
- `src/audioengine/RuntimeBuilder.h` — `#include "BuildErrorPolicy.h"` (policy値は不変)
- `src/audioengine/AudioEngine.RebuildDispatch.cpp:1098` — 唯一の `classifyBuildError` consumer (telemetry only)
- `src/audioengine/AudioEngine.h:2661` / `RebuildDispatch.cpp:149,914,996,1165` — RebuildIntentQueue / RebuildThread / Recovery lease
- `src/audioengine/ISRRuntimePublicationCoordinator.h:277` / `.cpp:968` — `settlePendingRecoveryAdmission`
- `src/audioengine/ISRCoordinatorLoop.h` — CoordinatorLoop (publication coordination)
- `src/core/WorkerThread.h/.cpp` — generic background thread (steady_clock + sleep_for)
- `src/core/TimeUtils.h` — steady_clock 用 time utils
- `evidence/D101-13-Phase-D-3-Contract-Test-Report.md` — D-3 65 checks PASS
- `evidence/D101-12-Phase-D-2-Retry-Policy-Minimal-Contract-Audit.md` — D-2 minimal contract (B ratified, INV-D2-1〜8)
- `evidence/D101-11-Phase-D-1-Retry-Policy-Authority-Audit.md` — D-1 Authority Matrix (C判定)
- `ConvoPeq.md` — 最新ソーススナップショット（§1.8 / H.11 一次資料）
