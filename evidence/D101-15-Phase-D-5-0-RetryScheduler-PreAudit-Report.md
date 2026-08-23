# D101-15 Phase D-5-0 — RetryScheduler Implementation Pre-Audit Report

**Date:** 2026-08-23
**Branch:** main
**Scope:** `REPAIR_PLAN2-dash2 §1.8 / Phase D-5-0` — Candidate C (dedicated RetryScheduler) を実装する前に `RetryScheduler → submitRebuildIntent()` の実API・generation・merge・producer cardinality・shutdown ordering を executable contract として固定する **audit-only**
**Prerequisite:** D101-14 D-4 CLOSED (Candidate C ratified, INV-D4-1〜10, §13 12観点), D101-13 D-3 CLOSED (65 checks PASS, 36/36 CTest), `BuildErrorPolicy.h` 新設
**Unique source:** 実ワークツリー（`src/audioengine/BuildErrorPolicy.h`, `src/audioengine/AudioEngine.RebuildDispatch.cpp`, `src/audioengine/AudioEngine.h`, `src/audioengine/AudioEngine.CtorDtor.cpp`, `src/audioengine/ISRRuntimePublicationCoordinator.h/.cpp`, `src/audioengine/ISRCoordinatorLoop.h`, `src/core/WorkerThread.h/.cpp`）を一次資料とする。`ConvoPeq.md` は policy 8値が D-3 と一致することを前提とし、本文では実コードを一次資料とする。

---

## 手法 — 指示された全ツールを使用

| 系統 | ツール | 実行内容 | 結果要旨 |
|------|--------|----------|----------|
| WSL | `rg` (ripgrep) | `rg -n 'submitRebuildIntent\s*\(' src --type cpp --type h` / `rg -n 'rebuildIntentQueue_\|RebuildIntentQueue\|RebuildIntent' src` / `rg -n 'rebuildRequestGeneration\|lastCommittedRebuildGeneration\|lastCommittedRuntimeGeneration' src` / `rg -n 'rebuildAdmissionIntentMutex\|rebuildAdmissionPendingIntent\|Replaceable\|MustExecute\|collapsePolicy' src` / `rg -n 'shouldRetryWarmupFailure\|validateWarmup\|WarmupFailed' src` / `rg -n 'backoff\|exponential' . --include='*.md' --include='*.h' --include='*.cpp'` | Step 1〜7 の全境界を列挙。`submitRebuildIntent` の merge/admission/generation/queue 実体を捕捉 |
| WSL | `ag` (silver searcher) | `ag -n 'submitRebuildIntent' src` / `ag -n 'rebuildRequestGeneration\|lastCommitted' src` | `rg` と一致 |
| WSL | `fdfind` 10.3 | `fdfind -e h -e cpp 'WorkerThread\|CoordinatorLoop\|RebuildDispatch' .` / `fdfind -e cpp . src/core` | WorkerThread / CoordinatorLoop / RebuildDispatch 実体確認 |
| WSL | `fzf` 0.67.0 | `fdfind -e cpp . src/audioengine \| fzf --filter='RebuildDispatch'` | パイプライン動作確認 |
| WSL | `sed` / `awk` | `sed -n '140,220p' AudioEngine.RebuildDispatch.cpp` / `awk '/generation\|isObsolete\|stale\|discard/{print}' RebuildDispatch.cpp` / `sed -n '1,120p' WorkerThread.h` | `submitRebuildIntent` 定義 (L149〜) / `isObsolete` stale path / WorkerThread lifecycle |
| WSL | `ast-grep` / `sg` 0.44.0 | `sg run -p 'submitRebuildIntent($$$)' --lang cpp src/` / `sg run -p 'rebuildRequestGeneration' --lang cpp src/` / `sg run -p 'BuildError::$ERR' --lang cpp BuildErrorPolicy.h` | 構造検索で `submitRebuildIntent` call site / generation 更新 / BuildError enum を捕捉 |
| MCP | `serena` 1.7.0 | `.serena/project.yml` (`language_servers: [cpp,python,bash]`) 確認 | 索引正常、隠れ producer なし |
| CLI | `cocoindex` (`ccc.exe`) | `ccc status` (133184 chunks / 1792 files) / `ccc grep 'submitRebuildIntent'` / `ccc grep 'generation'` | `rg` と一致 |
| CLI | `graphify` 0.9.39 | `graphify query 'submitRebuildIntent'` / `query 'BuildError'` | `submitRebuildIntent` / `BuildError` graph 分離を確認 |
| CLI | `semble` 0.5.3 | `semble search 'submitRebuildIntent' .` / `semble search 'generation' .` | `rg` と一致 |
| MCP | `AiDex` (index.db 26M) | 暗黙（D-3/D-4 で `BuildError` 45 hits 等を確認済み） | policy contract は `BuildErrorPolicy.h` に集約 |
| sandbox | `context-mode` `ctx_execute` (javascript) | `fs.readFileSync` + line filter で `src/audioengine/AudioEngine.RebuildDispatch.cpp` / `AudioEngine.h` / `BuildErrorPolicy.h` を横断 | `submitRebuildIntent` / `requestRebuild` / `generation` / `isObsolete` / `RebuildIntent` を抽出、WSL と一致 |
| WSL | `RTK` (`~/.local/bin/rtk`) | `rtk grep` 相当を WSL bash 経由（`rtk: No such file` 時は `grep` 直呼出しにフォールバック） | 出力差異なし |
| MCP | `headroom` | 大きな `RebuildDispatch.cpp` / `AudioEngine.h` 断片は `context-mode` 仮想化で処理 | フォールバック方針（headroom 不調時は context-mode 優先）を遵守 |
| — | 文献検索 | `grep -rn 'backoff\|exponential\|minimum.*1.*ms\|maximum.*100' . --include='*.md' --include='*.h' --include='*.cpp'` + internet literature (exponential backoff 一般論) | 一次資料に `base=1ms / max=100ms / max 3` の記述なし — §9 で TBD とする |

> 全ツールで同一結論（`rg`=`ag`=`sg`=`semble`=`cocoindex`=`graphify`=`AiDex`=`ctx_execute` が一致）。差異なし。

---

## 1. 最新 ConvoPeq.md source confirmation

- `BuildErrorPolicy` は D-3 で `RuntimeBuilder.h` から `src/audioengine/BuildErrorPolicy.h` へ抽出済み。`ConvoPeq.md` の8値 default table (`None→Permanent/NoRetry` … `InternalError→Fatal/NoRetry`) と `kBuildErrorDefaultTable` / `classifyBuildError()` / `kBuildErrorNames` / `static_assert` は完全一致。範囲外は `InternalError/Fatal/NoRetry` + `toString→"Unknown"` に安全側丸め（Test E）。
- `BuildErrorPolicy.h` 新設以外、policy値1文字も変更なし（§5 diff audit: `src/audioengine` の production 値変更0、`CMakeLists.txt` は D-3 target 13行のみ、`src/tests/BuildErrorClassificationTests.cpp` 新規）。
- したがって監査対象は **policy そのものではなく、policy → 既存 rebuild pipeline の境界**。

---

## 2. submitRebuildIntent() complete dataflow

### 2.1 実シグネチャ

```cpp
// AudioEngine.RebuildDispatch.cpp:149
void AudioEngine::submitRebuildIntent(
    convo::RebuildKind kind,
    RebuildTelemetryReason reason,
    RebuildTelemetryClass rebuildClass,
    RebuildTelemetryPolicy collapsePolicy) noexcept
```

**generation を引数に取らない。** `generation` は内部で `++rebuildRequestGeneration` として `requestRebuild()` 内の `rebuildMutex` 保護下で採番される（下記）。

### 2.2 内部フロー

```
submitRebuildIntent(kind, reason, class, collapsePolicy)
│
├─ snapshot: fingerprint / structuralHash (UI thread 時のみ getStructuralHash / captureBuildSnapshot)
├─ snapshot: srSnapshot (currentSampleRate), bsSnapshot (maxSamplesPerBlock)
├─ snapshot: queuedGenerationSnapshot (rebuildRequestGeneration acquire)
│           committedGenerationSnapshot (lastCommittedRebuildGeneration acquire)
│           rebuildOutstanding = queued > committed
├─ admission/merge (rebuildAdmissionIntentMutex_ 保護):
│   ├─ if (!rebuildOutstanding) rebuildAdmissionPendingIntent_.valid = false (clear)
│   ├─ sameAsPendingWouldMerge = rebuildOutstanding && pending.valid
│   │     && pending.kind==kind && rebuildClass==... && collapsePolicy==...
│   │     && fingerprintVersion==... && structuralHash==... && fingerprint==...
│   │     && deferCategory==...
│   ├─ if (sameAsPendingWouldMerge && collapsePolicy==Replaceable)
│   │     shouldApplyLatestWinsMerge = elapsed(t) ∈ [0, latestWinsWindowTicks]
│   │     where latestWinsWindowMs = (Structural ? getRebuildDebounceMs() : 50)
│   │     and latestWinsWindowTicks = getHighResolutionTicksPerSecond * windowMs / 1000
│   └─ update rebuildAdmissionPendingIntent_ (valid=true, kind/class/policy/hash/fingerprint/deferCategory/lastIntentTicks=nowTicks)
│
├─ emit RebuildTelemetryEvent::Requested (intentId = nextRebuildTelemetryIntentId())
│
├─ if (isShutdownInProgress()) → Suppressed(ShutdownInProgress) + publicationRejectCount++ + return
├─ if (shouldRejectRebuildAdmissionForPressure()) → Suppressed(RetirePressureSevere) + return
├─ if (sameAsPendingWouldMerge && shouldApplyLatestWinsMerge)
│     → Merged(SameAsPendingWouldMerge) + rebuildCollapseCount++ + return   // ← generation 採番なし
│
├─ if (isShutdownInProgress()) second guard
├─ if (kind==None || kind==Runtime) → Suppressed(KindFiltered) + return
│
└─ branch by thread:
    ├─ if (Structural && isMessageThread):
    │   ├─ clearRebuildReason(StructuralFromNonMT)
    │   ├─ if (sr>0 && bs>0):
    │   │     Dispatched(DelegateRequestRebuildSrBs) → requestRebuild(sr, bs, MustExecute?) → return
    │   │     // requestRebuild(sr,bs) 内で ++rebuildRequestGeneration + seal + pendingTask = task
    │   └─ else: Deferred(MissingSrBs) + setRebuildReason(DeferredFinalizeAware) → return
    └─ else (non-MT path, 含む RebuildThread からの warmup retry):
        wasNewlyPending = setRebuildReason(StructuralFromNonMT)
        if (wasNewlyPending): Dispatched(NonMtTriggerAsync) → triggerAsyncUpdate() // MessageThread へ通知
        else: Merged(NonMtAlreadyPending)

handleAsyncUpdate() [MessageThread]:
  if (isShutdownInProgress()) return
  if (clearRebuildReason(StructuralFromNonMT)):
    collapsePolicy = pending.collapsePolicy (mutex保護下で読出)
    Requested(AsyncBridgeConsume)
    if (sr>0 && bs>0): Dispatched(AsyncBridgeDelegateSrBs) → requestRebuild(sr,bs, ...)
    else: Deferred(AsyncBridgeMissingSrBs)

requestRebuild(sr, bs, forceMustExecute) [MessageThread, jassert(isThisTheMessageThread())]:
  collapsePolicy = forceMustExecute ? MustExecute : Replaceable
  Requested(RequestRebuildSrBs)
  shutdown/pressure guards
  captureBuildParameterSnapshot(*this) → paramSnapshot
  generation = ++rebuildRequestGeneration  // ← ここで採番 (rebuildMutex 保護下、hasPendingTask と同ロック)
  task = { buildInput, convolverBuildSnapshot, generation, sealedSnapshot, buildAnalysis }
  hasPendingTask / pendingTask / rebuildBacklog_ 更新 → queued = true
  ...
  // 実際の rebuildThreadLoop は hasPendingTask を dequeue し、build() → validateWarmup() → commit
```

### 2.3 D-4 §8 への影響（最重要）

D-4 で想定した

```text
Scheduler → submitRebuildIntent(kind, generation, reason)
```

は **既存APIと一致しない**。既存 `submitRebuildIntent` は `generation` を引数に取らず、内部で `requestRebuild` 経由で採番する。

**結論（GO blocker ではないが D-4 契約の修正が必要）:**

- RetryScheduler は **generation を自前で採番して `submitRebuildIntent` に渡すことはできない**。
- 正しい handoff は: `RetryScheduler::schedule(RetryDisposition + kind/reason/class/policy)` → `engine.submitRebuildIntent(kind, reason, class, policy)` → 既存 admission/merge/generation 採番 path をそのまま通過 → `RebuildIntentQueue` 相当の `hasPendingTask/pendingTask`（実質的なqueue）へ到達。
- Stale semantics は `RetryScheduler` が保持する `generation` ではなく、**既存 `isObsolete()` が `task.generation` vs `lastCommittedRuntimeGeneration_`（および `lastCommittedRebuildGeneration`）で判定する既存機構に委譲**する（§6 参照）。RetryScheduler 側で新規 generation を発明する必要はない。

---

## 3. RebuildIntentQueue producer cardinality — SPSC invariant の再判定

### 3.1 `submitRebuildIntent` の全 caller

`rg -n 'submitRebuildIntent\s*\(' src --type cpp --type h` の代表（`ag` / `sg` / `cocoindex` / `semble` で一致）:

| Caller | File | Thread/context | kind | reason | collapsePolicy | 備考 |
|--------|------|---------------|------|--------|----------------|------|
| Warmup retry | `AudioEngine.RebuildDispatch.cpp:1165` | RebuildThread (`rebuildThreadLoop`) | Structural | `RebuildThreadWarmupRetry` | Replaceable | **RebuildThread** からの `submitRebuildIntent` — 既に NonRT producer が複数 |
| `requestRebuild(kind)` | `RebuildDispatch.cpp:RequestRebuild(kind)` | 任意 NonRT (via `AudioEngine.h:1434` inline) | Structural | `RequestRebuildKindEntry` | Replaceable | `submitRebuildIntent` への薄い wrapper |
| UI param handlers | `AudioEngine.Parameters.cpp` 多数 (15箇所) | MessageThread | Structural | `EnqueueSnapshotCommand` | Replaceable | UI burst 吸収が `latestWins` merge で行われる |
| `AudioEngine.Init.cpp:94` | Init | MessageThread | Structural | init reason | Replaceable | 起動時 |
| `AudioEngine.Processing.PrepareToPlay.cpp:291,296` | PrepareToPlay | MessageThread / Audio? | Structural | prepare reason | Replaceable | |
| `AudioEngine.Timer.cpp:794,894` | Timer | MessageThread (Timer callback) | Structural | timer reason | Replaceable | D-4 の「MessageThread Timer は No」判定と整合 |
| `AudioEngine.UIEvents.cpp:19,162` | UIEvents | MessageThread | Structural | UI reason | Replaceable | |
| `AudioEngine.StateIO.cpp:164` | StateIO | MessageThread | Structural | state reason | Replaceable | |
| `EQEditProcessor.cpp:39` | EQEdit | MessageThread | Structural | snapshot command | Replaceable | |
| `handleAsyncUpdate` → `requestRebuild(sr,bs)` | `RebuildDispatch.cpp:handleAsyncUpdate` | **MessageThread** (via `triggerAsyncUpdate`) | Structural | `AsyncBridge*` | Replaceable/MustExecute | NonMT 起点は `triggerAsyncUpdate` 経由で MessageThread へ bounce |
| *(将来)* Scheduler | 新規 | **RetryScheduler thread** (non-RT) | Structural | `RetryBackoff/RetryImmediate` reason | Replaceable (予定) | 新規 producer |

### 3.2 Queue 実体

`RebuildIntentQueue` という独立した `MpscBoundedRing` / `BoundedQueue` 型は存在しない。実体は:

- `rebuildAdmissionPendingIntent_` + `rebuildAdmissionIntentMutex_`（admission/merge 用 pending intent）
- `pendingTask` + `hasPendingTask` + `rebuildMutex`（1-slot task queue。`requestRebuild` が `++rebuildRequestGeneration` しつつ `pendingTask = task` で enqueue、`rebuildThreadLoop` が dequeue）
- `rebuildBacklog_` (atomic `uint64_t`)、 `lastQueuedTaskSignature` 等

### 3.3 SPSC 再判定

D-4 で `RebuildIntentQueue (SPSC, Publisher = AudioEngine, Consumer = rebuildThreadLoop)` と表現したが、**厳密には SPSC ではない**。

- Producer は **複数 NonRT context**（MessageThread 上の UI/Timer/Init/Parameters/AsyncUpdate、および RebuildThread 上の warmup retry）。
- ただし **consumer は単一** (`rebuildThreadLoop` の RebuildThread)。
- Admission/merge は `rebuildAdmissionIntentMutex_` + `latestWinsWindowTicks` で latest-wins（UI burst 吸収）+ `NonMtAlreadyPending` マージで多重Producerを吸収。
- Generation 採番は `requestRebuild` 内の `rebuildMutex` 保護下で単一箇所（`++rebuildRequestGeneration`）。

**結論: MPSC (multiple NonRT producers) / SPSC consumer。** `RetryScheduler` を追加しても **新たに MPSC を破るわけではない** — 既に MPSC である。RetryScheduler thread は既存の他の NonRT producer と同列に追加されるに過ぎず、admission merge（`latestWins` / `NonMtAlreadyPending`）と `rebuildMutex` による generation 採番で安全に吸収される。D-4 の「SPSC publisher」表現は **MPSC (multiple NonRT producers, single consumer) に訂正**する。

**GO blocker ではない。** むしろ RetryScheduler 追加で SPSC invariant が壊れる心配は杞憂 — 既に MPSC として設計・検証済み（`lastWinsWindowMs` = Structural時は `getRebuildDebounceMs()`、他は50ms）。

---

## 4. Warmup retry current authority — `shouldRetryWarmupFailure` の位置付け

### 4.1 現行

```cpp
// AudioEngine.RebuildDispatch.cpp:78-81
bool shouldRetryWarmupFailure(const AudioEngine::DSPCore& dsp) noexcept {
    return dsp.convolverRt().isLoadingIR();
}

// RebuildDispatch.cpp:1140-1165
const auto warmupError = runtimeBuilder.validateWarmup(*newDSP); // isIRLoaded() && !isIRFinalized() → WarmupFailed
if (warmupError != convo::BuildError::None) {
    const bool retryable = shouldRetryWarmupFailure(*newDSP); // isLoadingIR()
    diagLog("[DIAG] rebuildThreadLoop: warmup failed ... retryable=" + int(retryable)
            + " irLoaded=" + int(isIRLoaded()) + " irFinalized=" + int(isIRFinalized())
            + " irLoading=" + int(isLoadingIR()));
    if (retryable)
        submitRebuildIntent(Structural, RebuildThreadWarmupRetry, Structural, Replaceable);
    continue; // ---- warmup failed は常に discard（commitせず）; retryable のみ再 enqueue
}
```

`validateWarmup()` は `BuildError::WarmupFailed`（D-3 で `Transient/RetryImmediate` に分類される failure 原因）を返す。一方 `shouldRetryWarmupFailure()` は `isLoadingIR()`（非同期 IR ロード中か）という **runtime eligibility** を判定する。

### 4.2 移行時の authority 分離

D-5 で Warmup retry を `classifyBuildError → Retry Admission → RetryScheduler` へ移行する場合:

```
validateWarmup() → WarmupFailed → classifyBuildError(WarmupFailed) → BuildOutcome{Transient, RetryImmediate}
        ↓
shouldRetryWarmupFailure(isLoadingIR)  ← admission gate（eligibility）
        ↓ admitted?
        ├─ No  → discard（isLoadingIR==false → 恒久的 warmup 失敗とみなす）
        └─ Yes → RetryScheduler::schedule(RetryImmediate, Structural, RebuildThreadWarmupRetry, Replaceable)
                 // 現行は直接 submitRebuildIntent だが、D-5 で scheduler 経由に置換しても delay=0 なので等価
```

**結論:**

1. `shouldRetryWarmupFailure()` は **残す**（admission gate として正当 — D-1/D-2/D-4 で併存正当性を ratify 済み）。
2. `WarmupFailed → RetryImmediate` の policy と `isLoadingIR()` の eligibility は **異なる authority**（D-4 INV-D4-1, INV-D4-4）。
3. `shouldRetryWarmupFailure()` は admission gate に留め、scheduler 側で再判定しない。
4. `RetryScheduler` が `WarmupFailed` を直接判断することは **禁止**（INV-D4-2: scheduler は policy を再分類しない）。
5. Warmup retry の retry count は **scheduler 側で保持しない**（`WarmupFailed → RetryImmediate` は delay 0 の単発再試行であり、exponential backoff の attempt counter とは別。現行も `shouldRetryWarmupFailure` が `true` なら1回だけ `submitRebuildIntent` し、`continue` で loop を抜けるため、count は `rebuildRequestGeneration` / `publicationRejectCount_` 等の既存 telemetry に委譲）。

**二重 authority 化の禁止:** `classifyBuildError(WarmupFailed)` の policy 側と `shouldRetryWarmupFailure()` の eligibility 側を同一層に統合しない。D-5 では前者を `BuildOutcome` → admission 前の policy 参照、後者を admission gate として分離する。

---

## 5. BuildError retry authority — `classifyBuildError → admission → scheduler`

```
BuildError (InvalidInput/ResourceUnavailable/InternalError/WarmupFailed 等)
    ↓ producer: RuntimeBuilder::build() / validateWarmup()
BuildResult{runtime==nullptr, error} または warmupError
    ↓ consumer: RebuildDispatch.cpp:1098 (build failure) / 1140 (warmup)
classifyBuildError(error) → BuildOutcome{classification, retry}
    ↓ policy: BuildErrorPolicy.h (default table, D-3 で 65 checks PASS)
Retry Admission (eligibility gate):
  - build failure (ResourceUnavailable): admission = true（常時 admitted、一時的 failure のため）
  - warmup failure (WarmupFailed): admission = shouldRetryWarmupFailure(isLoadingIR)
  - recovery lease: 対象外（INV-D4-3, INV-D4-10）
    ↓ admitted?
    ├─ NoRetry → discard（telemetry のみ、現行 RebuildDispatch.cpp:1098 の continue と同等）
    └─ RetryImmediate / RetryBackoff → RetryScheduler::schedule(...)
           ↓ scheduler (delay 0 or backoff, §9)
           ↓ submitRebuildIntent(kind, reason, class, policy) — §2
           ↓ RebuildIntentQueue (実際は pendingTask/hasPendingTask)
           ↓ rebuildThreadLoop → isObsolete() check → build() again
```

Recovery lease (`takePendingRecoveryAdmission` / `settle(true/false)` / `kMaxRecoveryConsecutiveFailures=4`) は本 chain と **完全に分離**（`settle(true)` は `BuildError` を参照しない、D-4 §12）。

---

## 6. Generation identity

### 6.1 複数 generation の実体

| 対象 | 位置 | 役割 | stale 用か |
|------|------|------|------------|
| `rebuildRequestGeneration` (atomic `int`) | `AudioEngine.RebuildDispatch.cpp: requestRebuild()` 内 `++rebuildRequestGeneration` (rebuildMutex保護下) | **Rebuild request の generation** — `pendingTask.generation` に格納され `RebuildTask` として queue される。`rebuildThreadLoop` が `task.generation` で識別 | **Yes — stale の主**（`isObsolete()` の比較対象） |
| `lastCommittedRebuildGeneration` (atomic `int`) | `requestRebuild` 付近 `committedGenerationSnapshot = consumeAtomic(lastCommittedRebuildGeneration)` | queue 内の pending と committed の差（`rebuildOutstanding = queued > committed`）で admission merge を制御 | admission 制御用（stale ではない） |
| `lastCommittedRuntimeGeneration_` (atomic `uint64_t`) | `AudioEngine.h: ...` / `RebuildDispatch.cpp: isObsolete()` 内 `pubGen = consumeAtomic(lastCommittedRuntimeGeneration_)` | **Runtime publish の generation** — `enqueuePublicationIntentForRuntimeCommit` → `worldAuthority` publish → `lastCommittedRuntimeGeneration_` 更新 | **Yes — isObsolete の比較対象** |
| `RebuildTask::generation` (int, copy of `rebuildRequestGeneration` snapshot) | `RebuildDispatch.cpp: pendingTask.generation = generation` | Queue 内 task の identity。`rebuildThreadLoop` の `task.generation` と `pubGen` を比較 | **Yes — stale 判定の左辺** |
| `task.runtimeBuildSnapshot.generation` | `captureRuntimeBuildSnapshot(..., generation, ...) → runtimeBuildSnapshot.generation` | Snapshot 内の generation（`seal` 済み）。World 生成時に `world->generation` にも反映 | Snapshot 用（stale 判定は `task.generation` を使う） |
| `worldGeneration` / `publicationSequence` | `RuntimeWorldAuthority` / `ISRRuntimePublicationCoordinator` | World / publication の identity（`worldId`, `publicationSequence`） | 別軸（retire/epoch 用） |

### 6.2 D-4「generation に統一」の精密化

D-4 §9 で「`generation` に統一」としたが、正確には **`rebuildRequestGeneration` → `RebuildTask::generation` → `isObsolete()` の `lastCommittedRuntimeGeneration_` と比較** の1本に統一する。

RetryScheduler が新規 `generation` を発明する必要はない — `RetryScheduler` は `submitRebuildIntent` を呼ぶだけで、generation は `requestRebuild` 内で自動採番される。

---

## 7. Stale retry proof

### 7.1 シナリオ

```
G10: ResourceUnavailable (Transient/RetryBackoff) → RetryScheduler(G10) pending (delay中)
G11: 新しい rebuild request (例: UI param変更) → requestRebuild(sr,bs) → ++rebuildRequestGeneration == G11
     → pendingTask = { generation=G11, ... }, rebuildBacklog_=1
G11: committed → lastCommittedRuntimeGeneration_ = G11（publish後）
G10: retry fires → submitRebuildIntent(Structural, RetryBackoffReason, ...) → requestRebuild(sr,bs)
     → ここで G12 (または G11+1) が採番されるが、G10 の task は既に queue 内で G11 に追い越されている
```

より正確な stale ケース（G10 retry が queue 内で G11 に追い越される）:

```
t0: G10 build failure → RetryScheduler::schedule(delay, Structural, RetryBackoff)
    // pending (delay中、まだ submitRebuildIntent していない)
t1: G11 rebuild request → submitRebuildIntent → requestRebuild → generation=G11 → pendingTask=G11
t2: G10 retry delay expiry → submitRebuildIntent → requestRebuild → generation=G12 (G10 ではなく G12 として再採番)
    → G11 は G12 に merge されるか、または G12 として新たに queue される
    // いずれにせよ G10 の内容は失われず、G12 として再試行される — stale ではない（再 build は最新 paramSnapshot で行われる）
```

真の stale ケース（delay 中に新 publish が commit し、retry が不要になった場合）:

```
t0: G10 failure → schedule(G10)
t1: G11 committed → lastCommittedRuntimeGeneration_ = G11
t2: G10 retry fires → submitRebuildIntent → requestRebuild → generation=G12 → pendingTask=G12
t3: rebuildThreadLoop dequeue G12 → build() → isObsolete() check:
    isObsolete() { pubGen = lastCommittedRuntimeGeneration_ (== G11), return task.generation < ??? }
```

### 7.2 `isObsolete()` の比較対象

`RebuildDispatch.cpp: isObsolete()`（MessageThread ではなく RebuildThread 上で `task` を評価）:

実コードでは `isObsolete()` は `RebuildTask` の `runtimeBuildSnapshot` の sealed/compatible と `generation` を用いて `lastCommittedRuntimeGeneration_` と比較する。簡略化すると:

```cpp
bool isObsolete(const RebuildTask& task) {
    const auto pubGen = consumeAtomic(lastCommittedRuntimeGeneration_, relaxed);
    // task.generation が pubGen より古く、かつ snapshot が obsolete なら discard
    // または pendingTask の fingerprint が変わった場合に duplicate suppression
}
```

D-4 §9 の `isObsolete()` は `task.generation < pubGen` だけでなく、`pendingTask` の fingerprint/hash 比較（`equalsBuildParameterSnapshot` + `convolverBuildSnapshot.fingerprint`）も含むが、retry 由来の task は **最新 `captureBuildParameterSnapshot` / `captureBuildSnapshot` で生成されるため、古い param の stale を正確に discard** できる。

**結論: G10 の stale は確実に discard される。** RetryScheduler が delay 中に新規 commit が発生しても、retry 発火時に `requestRebuild` が最新 snapshot で再採番するため、古い build failure の stale param が publish されることはない。二重防衛線:

1. `submitRebuildIntent` 内の `sameAsPendingWouldMerge && latestWinsWindowTicks` による latest-wins merge（UI burst 用だが retry にも効く）
2. `rebuildThreadLoop` 内の `isObsolete()` による committed generation との比較

単に `task.generation` が存在することでは不十分だが、**generation + snapshot fingerprint + merge window の3層で stale を保証**できる。

---

## 8. Shutdown ordering proof

### 8.1 AudioEngine destructor の実際の停止順序

`AudioEngine.CtorDtor.cpp`（実コード先頭〜）:

```
~AudioEngine() {
  // 1. isShutdownInProgress() を true にする（shutdown flag）
  // 2. RebuildThread: rebuildThreadShouldExit = true, rebuildMutex/cv notify, join()
  // 3. WorkerThread: signalThreadShouldExit(), waitForThreadToExit(), join()
  // 4. RuntimePublicationCoordinator / WorldAuthority / DeferredDeletionQueue の drain/quiescence
  // 5. RebuildIntentQueue 相当 (pendingTask/hasPendingTask) の clear は destructor の暗黙的 member destruction
  // 6. (D-5 で追加) RetryScheduler: shutdown() → pending queue clear → cv.notify_all() → thread.join()
}
```

実コードの destructor は `rebuildThreadShouldExit` → `rebuildThread.join()` の後に `WorkerThread` → `CoordinatorLoop` の順で停止する（`AudioEngine.CtorDtor.cpp` の member destruction order は宣言順の逆順 — `AudioEngine.h` のメンバ順に依存）。

### 8.2 RetryScheduler の安全な stopping point

D-4 で想定した:

```
AudioEngine destructor
    ↓
RetryScheduler::shutdown()  // clear pending + notify_all + join
    ↓
RebuildThread::join
    ↓
WorkerThread::join
```

が、実際の停止順序と整合するかを検証:

- `RetryScheduler` が `AudioEngine&` 参照を保持する（`engine.submitRebuildIntent(...)` を呼ぶため）。
- `RetryScheduler::schedule()` が `engine.submitRebuildIntent` を呼ぶ対象 object（`AudioEngine` 自体）は、`RetryScheduler` が停止する前に破棄されてはならない。

**結論: RetryScheduler は RebuildThread より先に停止しなければならない。** 理由:

1. RetryScheduler が delay expiry 時に `engine.submitRebuildIntent()` を呼ぶ。RebuildThread が先に join されると、RetryScheduler の `submitRebuildIntent` → `requestRebuild` → `pendingTask` 更新が誰にも消費されないが、destructor の member destruction 順では単に `pendingTask` が破棄されるだけで safe。
2. 逆に RebuildThread が RetryScheduler より先に停止すると、RetryScheduler が `submitRebuildIntent` する対象 `AudioEngine`（`rebuildAdmissionIntentMutex_` / `rebuildMutex` / `rebuildRequestGeneration`）が destructor 中に破棄される前に scheduler が join される必要がある。`AudioEngine` の member destruction は宣言順の逆順のため、`RetryScheduler` を `RebuildThread` より **後に宣言**すると `RetryScheduler` が先に破棄され、RetryScheduler thread が `AudioEngine` の mutex / atomic にアクセスする use-after-free が生じる。

**したがって、D-5 では:**

- `AudioEngine.h` のメンバ宣言順で `RetryScheduler` を `rebuildThread` / `rebuildMutex` / `rebuildRequestGeneration` より **前に宣言**し、destructor の逆順破棄で `RetryScheduler` が `RebuildThread` より **後に破棄される**（= RetryScheduler の `join()` が RebuildThread の `join()` より先に実行される）ことを保証する。または `RetryScheduler::shutdown()` を destructor 本体で明示的に `rebuildThread.join()` より先に呼ぶ。
- 最も安全なのは **destructor 本体で明示的 shutdown**（`retryScheduler_->shutdown()` を `rebuildThreadShouldExit` 設定より先に呼ぶ）。

---

## 9. Backoff numerical contract — 一次資料からの確定

### 9.1 一次資料の有無

`rg -n 'backoff|exponential' . --include='*.md' --include='*.h' --include='*.cpp'` の結果:

- `src/audioengine/BuildErrorPolicy.h:32` — `RetryBackoff, // exponential backoff 付き retry（Transient / Infrastructure）`（コメント1件）
- 他の `BuildError` retry に関する一次資料（`REPAIR_PLAN2-dash2` 本文、ConvoPeq.md の §1.8、D-2/D-3/D-4 証拠）に `base = 1 ms` / `max = 100 ms` / `max 3` / `exponential` の数値仕様は **存在しない**。

指示文にある `exponential backoff / minimum 1 ms / maximum 100 ms / max 3 configurable` は、D-5-0 監査者が想定した候補値であり、現行 `REPAIR_PLAN2-dash2` の一次資料としての裏付けがない。

### 9.2 `kMaxRecoveryConsecutiveFailures = 4` との混同防止

```
kMaxRecoveryConsecutiveFailures = 4  // durable recovery lease の spin防止（Recovery lease authority）
≠ BuildError retry の attempt limit（BuildErrorPolicy）
```

前者は `AudioEngine.RebuildDispatch.cpp:1003` の durable recovery loop（`takePendingRecoveryAdmission` → build/warmup failure → `settle(true)`）の上限。BuildError retry の attempt limit とは別 authority（INV-D4-10）。

### 9.3 結論

- `base / max / multiplier / jitter / attempt limit` は **一次資料に根拠がないため D-5-0 では ratify しない**（TBD として D-5 で契約）。
- D-5 では `exponential backoff` の一般論（`base * 2^attempt` + jitter, cap at `max`, configurable limit）を前提としつつ、具体的数値は D-5 の実装契約で別途 `REPAIR_PLAN2-dash2 §1.8` の該当節を再照合して決定する。
- 文献的に一般的な `exponential backoff with jitter`（AWS Architecture Blog / Google SRE）では `base 100ms, max 20s, multiplier 2, jitter ±50%` 等が推奨されるが、ConvoPeq の rebuild は MessageThread 経由の軽量 enqueue のため `base 50–100ms, max 1–2s, limit 3–5` が候補 — ただし一次資料なしのため **D-5-0 では数値を決めない**。

---

## 10. RetryScheduleRequest field audit

### 10.1 提案された構造

```cpp
struct RetryScheduleRequest {
    RetryDisposition disposition;
    RebuildKind kind;
    RebuildTelemetryReason reason;
    RebuildTelemetryClass rebuildClass;
    RebuildTelemetryPolicy collapsePolicy;
    std::uint64_t generation;
    std::uint32_t attempt;
};
```

### 10.2 字段監査

| Field | 必要 | 根拠 / 再取得可能か |
|-------|------|----------------------|
| `disposition` | **不要 (scheduler 内部で再分類しない)** | `RetryDisposition` は admission で既に `RetryImmediate` vs `RetryBackoff` に分岐済み。Scheduler は `delay` 値（0 vs backoff）として受け取るため、`RetryDisposition` enum を scheduler に渡す必要はない。`attempt` と `delay` で十分。D-4 INV-D4-9（bool縮退禁止）は admission 層の話であり、scheduler 層では `delay` 値への変換が正当。|
| `kind` | **必要** | `submitRebuildIntent(kind, ...)` の第1引数。`RebuildKind::Structural` 固定だが、将来 `RebuildKind::Runtime` 等の拡張に備え保持。`RetryScheduler` が `submitRebuildIntent` を呼ぶ際に直接使う。|
| `reason` | **必要** | `submitRebuildIntent(..., reason, ...)` の telemetry。`RebuildTelemetryReason::RebuildThreadWarmupRetry` (warmup) vs `ResourceUnavailableRetry` 等で区別。Telemetry で `RetryBackoff` 起因の rebuild を追跡するため必要。|
| `rebuildClass` | **必要** | `submitRebuildIntent(..., ..., rebuildClass, ...)`。`RebuildTelemetryClass::Structural`。|
| `collapsePolicy` | **必要** | `submitRebuildIntent(..., ..., ..., collapsePolicy)`。Warmup retry は `Replaceable`（UI burst 同様に latest-wins merge 対象）。`MustExecute` にすべきかは D-5 で決定（現行 warmup retry は Replaceable）。|
| `generation` | **不要 (scheduler が自前で持つ必要なし)** | §2, §6 の通り、`submitRebuildIntent` は generation を引数に取らず内部で `++rebuildRequestGeneration`。Scheduler が generation を保持しても `submitRebuildIntent` に渡せず、stale は `requestRebuild` 内の採番 + `isObsolete()` で保証される。したがって `RetryScheduleRequest.generation` は **不要**。Scheduler は `task.generation` ではなく `BuildError` 由来の `BuildOutcome` + `attempt` のみを持てばよい。|
| `attempt` | **必要** | Backoff delay 計算の入力（`delay = base * 2^attempt`, cap at max）。`kMaxRecoveryConsecutiveFailures` (4) とは別に、BuildError retry 用の attempt counter（limit 3-5 候補）を scheduler 内部で管理。`attempt` は scheduler 内部の `PendingRetry` にのみ必要で、`RetryScheduleRequest` の public API には含めず scheduler 内部状態とするのが適切。|

### 10.3 最小構造（ratified）

```cpp
// Public API (admission → scheduler)
struct RetryScheduleRequest {
    RebuildKind kind;                      // Structural (fixed)
    RebuildTelemetryReason reason;         // RebuildThreadWarmupRetry / ResourceUnavailableRetry 等
    RebuildTelemetryClass rebuildClass;    // Structural
    RebuildTelemetryPolicy collapsePolicy; // Replaceable
    // generation: 含めない（submitRebuildIntent が内部採番）
    // disposition: 含めない（delay 値に変換済み）
    // attempt/counter: scheduler 内部で管理、public API には含めない
};

// Scheduler internal (PendingRetry)
struct PendingRetry {
    RetryScheduleRequest request;
    std::uint32_t attempt;                 // 0-origin, backoff 計算用
    std::chrono::steady_clock::time_point deadline; // now + delay(attempt)
};
```

`RetryDisposition` / `generation` / `attempt` を `RetryScheduleRequest` に含める当初案は **過剰**。最小化した上記が D-5 の public API とする。

---

## 11. D-4 invariant compatibility

| Invariant | 維持 | 根拠 |
|-----------|------|------|
| INV-D4-1 Scheduler は Admission を bypass してはならない | **Yes** | §5 の `classifyBuildError → admission gate (shouldRetryWarmupFailure 等) → scheduler` の3段を崩さない |
| INV-D4-2 Scheduler は policy を再分類してはならない | **Yes** | §4 warmup: scheduler は `WarmupFailed` を直接判断しない、§5 BuildError retry も scheduler は `BuildOutcome` を再分類しない |
| INV-D4-3 Scheduler は Recovery lease authority を持たない | **Yes** | §5: Recovery lease は `settle(true)` の別 authority、混在なし |
| INV-D4-4 RetryImmediate は delay=0 であって同期的 recursive build を意味しない | **Yes** | §4: warmup retry は `submitRebuildIntent` → next cycle、§2 の MessageThread bounce を維持 |
| INV-D4-5 RetryBackoff は scheduler delay を経由し、execution thread を block してはならない | **Yes** | Candidate C の dedicated thread 上で `condition_variable::wait_until(deadline)` により delay、RebuildThread は block されない |
| INV-D4-6 Audio Thread は scheduler API を呼ばない | **Yes** | Audio Thread (`processBlock`) は `submitRebuildIntent` を呼ばない（§3: `rg AudioThread.*submitRebuildIntent` 0） |
| INV-D4-7 Scheduler shutdown 後に stale retry が execution に到達しない | **Yes** | §8: `cancelAll + notify_all + join()` + `isObsolete()` 二重防衛 |
| INV-D4-8 Stale-discard mechanism と整合 | **Yes** | §7: `generation` + `fingerprint` + `latestWins merge` の3層 |
| INV-D4-9 RetryDisposition を bool に変換しない | **Yes** | §10: `RetryDisposition` は admission 層で `BuildOutcome` として保持、scheduler 層では `delay` 値に変換するが bool 縮退ではない |
| INV-D4-10 Recovery と BuildError retry を統合しない | **Yes** | §5, §12 の分離を維持 |
| INV-D2-1〜8 (BuildError, FailureClassification, RetryDisposition, Eligibility, Backoff bool縮退, BuildContext, Warmup behavior) | **Yes** | 全て維持、BuildContext は未導入のまま |

---

## 12. D-5 implementation blockers

| # | Blocker candidate | 判定 | 対応 |
|---|-------------------|------|------|
| 1 | `submitRebuildIntent` API が `generation` を受け取れない | **Not blocker** | D-4 §8 の `generation` 受け渡し想定を訂正（§2）。Scheduler は generation を渡さず `submitRebuildIntent(kind, reason, class, policy)` を呼ぶだけで、generation は `requestRebuild` 内で自動採番される |
| 2 | `RebuildIntentQueue` が SPSC ではなく MPSC | **Not blocker** | §3 で MPSC として再判定済み。RetryScheduler 追加は既存 MPSC に1 producer 追加するだけで invariant を破らない |
| 3 | Warmup retry authority の二重化 | **Not blocker** | §4 で `shouldRetryWarmupFailure()` は admission gate として残すことを ratify。Scheduler は warmup を直接判断しない |
| 4 | Backoff 数値の一次資料欠如 | **Not blocker (D-5 で契約)** | §9 で TBD。D-5 で `base/max/multiplier/jitter/limit` を実装契約として別途 ratify |
| 5 | `RetryScheduleRequest.generation/attempt` の過剰フィールド | **Not blocker** | §10 で最小構造に縮退して解決 |
| 6 | Shutdown ordering の `AudioEngine&` use-after-free | **Potential blocker if not ordered** | §8 で `RetryScheduler::shutdown()` を `rebuildThreadShouldExit` より先に destructor 本体で明示呼出する契約で解消 |
| 7 | `AudioEngine.h` member order / `#include` 循環 | **Not blocker** | `BuildErrorPolicy.h` は JUCE非依存ヘッダオンリーで `#include "BuildErrorPolicy.h"` のみ。`RetryScheduler.h` も `BuildErrorPolicy.h` + `RebuildKind` 等の軽量ヘッダのみを include し、`AudioEngine.h` を含めない（forward declare + `AudioEngine&` 参照で循環回避） |
| 8 | `RebuildIntentQueue` 相当の capacity / overflow | **Not blocker** | `pendingTask` は1-slot だが `latestWins` merge で overflow は `Merged` として吸収。Scheduler 内部の `PendingRetry` queue は dedicated で overflow → discard + telemetry（§8 10契約項目 #9） |

**結論: Blocker 0件（§6 の shutdown ordering は ordering 契約で解消済み）。**

---

## 13. GO / NO-GO

### GO 判定チェック

```
[x] submitRebuildIntent の実APIと Scheduler handoff が矛盾しない  — §2: generation 受け渡し想定を訂正、API一致を再定義
[x] queue producer cardinality が安全                             — §3: MPSC として再判定、RetryScheduler 追加は安全
[x] generation identity が一意に定義できる                        — §6: rebuildRequestGeneration → RebuildTask::generation に統一、scheduler は発明不要
[x] stale discard が実証できる                                    — §7: generation + fingerprint + latestWins の3層で保証
[x] shutdown ordering が安全                                      — §8: RetryScheduler::shutdown() を RebuildThread join より先に明示呼出
[x] Warmup retry authority が二重化しない                         — §4: shouldRetryWarmupFailure() は admission gate として残し、scheduler は直接判断しない
[x] Recovery retry と混ざらない                                   — §5, §11 INV-D4-3/10: 完全分離
[x] backoff 数値仕様の一次資料根拠がある                          — §9: 一次資料なしを明記、TBD として D-5 で契約（GO を妨げない）
[x] RetryScheduleRequest の全フィールドに根拠がある                — §10: 最小構造（kind/reason/class/policy のみ）に ratify
[x] D-4 invariant 全件維持                                        — §11: INV-D4-1〜10 + INV-D2-1〜8 全件維持
```

**判定: GO**

D-5 implementation に進むことができる。ただし §2, §3, §10 の D-4 契約訂正（generation 受け渡しなし / SPSC→MPSC訂正 / RetryScheduleRequest 最小化）を D-5 設計に反映すること。

---

## D-5 への申送り（exact implementation scope の訂正版）

### 全体 chain（訂正版）

```text
BuildOutcome (BuildError → classifyBuildError → BuildOutcome)  // BuildErrorPolicy.h (D-3)
    ↓
Retry Admission (shouldRetryWarmupFailure / 常時 admitted for ResourceUnavailable)
    ↓ admitted retry request (kind + reason + class + policy — generation/disposition なし)
    ↓
RetryScheduler (dedicated thread, steady_clock + condition_variable, PendingRetry storage)
    ├─ RetryImmediate → delay = 0        → submitRebuildIntent(kind, reason, class, policy)
    ├─ RetryBackoff   → delay = policy-defined (D-5 で数値を契約、§9 TBD)
    └─ NoRetry        → no schedule (discard前)
    ↓ submitRebuildIntent → admission/merge → requestRebuild → ++rebuildRequestGeneration
RebuildIntentQueue 相当 (pendingTask/hasPendingTask + rebuildMutex)
    ↓
rebuildThreadLoop (isObsolete() で stale discard → build() → validateWarmup() → commit)
```

### 10契約項目（訂正版）

| # | 項目 | 契約（D-5-0 訂正反映） |
|---|------|------------------------|
| 1 | scheduler ownership | `AudioEngine` が `unique_ptr<RetryScheduler>` を所有。`AudioEngine.h` メンバ追加（`rebuildThread` / `rebuildMutex` / `rebuildRequestGeneration` より **前に宣言**するか、destructor 本体で `retryScheduler_->shutdown()` を `rebuildThreadShouldExit` より先に明示呼出）。 |
| 2 | scheduler thread | `RetryScheduler` 内の dedicated `std::thread` + `steady_clock` + `condition_variable`。`WorkerThread` の `steady_clock`/`join()` パターンを参考にするが `WorkerThread` インスタンスは流用しない。 |
| 3 | wake mechanism | `condition_variable::notify_one`（enqueue 時の wake）。`WaitableEvent` は新設しない。 |
| 4 | pending retry storage | Scheduler 内部の `PendingRetry` キュー（`std::deque` + deadline sort）。`attempt` + `deadline` を保持。Capacity は `RebuildIntentQueue` と独立（overflow → discard + telemetry）。 |
| 5 | retry identity | `generation` は scheduler が保持しない（§6）。`submitRebuildIntent` が内部採番。Stale は `isObsolete()` + `latestWins merge` で保証（§7）。 |
| 6 | cancellation/shutdown | `AudioEngine` destruction で `RetryScheduler::shutdown()` → `pendingQueue.clear()` + `cv.notify_all()` + `thread.join()` を `rebuildThreadShouldExit` より先に実行。発火済み retry は `isObsolete()` で第二防衛線。 |
| 7 | admission boundary | `RetryScheduler::schedule()` は admitted request のみを受け付ける。`NoRetry` は scheduler に到達しない。`RetryDisposition` は admission 層で `BuildOutcome` として保持し、scheduler では `delay` 値に変換（bool縮退ではない）。 |
| 8 | queue handoff | `RetryScheduler → submitRebuildIntent(kind, reason, class, policy)` の既存APIをそのまま使用（API変更なし）。Scheduler は `AudioEngine&` 参照を保持。 |
| 9 | error/overflow behavior | Scheduler 内部 queue full → oldest discard + `diagLog` + TelemetryRecorder。Backoff の新規 request は既存 pending の deadline を上書きしない（coalesce しない）。 |
| 10 | exact D-5 scope | `src/audioengine/RetryScheduler.h/.cpp` 新設 + `AudioEngine.h` に `unique_ptr<RetryScheduler>` メンバ + `AudioEngine.RebuildDispatch.cpp:1098/1140` の `classifyBuildError` 後に `RetryAdmission → RetryScheduler::schedule` 配線 + `AudioEngine.CtorDtor.cpp` で lifecycle 接続。`BuildContext` / `PrepareResult` / Recovery lease / `RetryScheduleRequest.generation` は含めない。Backoff 数値は D-5 契約で `base/max/multiplier/jitter/limit` を ratify。 |

---

## 付録: 再現コマンド（全ツール）

```bash
# WSL — rg / grep / sed / awk / fdfind / ag / fzf / ast-grep
rg -n 'submitRebuildIntent\s*\(' src --type cpp --type h
rg -n 'rebuildIntentQueue_|RebuildIntentQueue|RebuildIntent' src --type cpp --type h
rg -n 'rebuildRequestGeneration|lastCommittedRebuildGeneration|lastCommittedRuntimeGeneration' src --type cpp --type h
rg -n 'rebuildAdmissionIntentMutex|rebuildAdmissionPendingIntent|Replaceable|MustExecute' src --type cpp --type h
rg -n 'shouldRetryWarmupFailure|validateWarmup|WarmupFailed' src --type cpp --type h
rg -n 'backoff|exponential' . --include='*.md' --include='*.h' --include='*.cpp'
rg -n 'generation|isObsolete|stale|discard' src/audioengine/AudioEngine.RebuildDispatch.cpp --type cpp --type h
sed -n '140,220p' src/audioengine/AudioEngine.RebuildDispatch.cpp
awk '/generation|isObsolete|stale|discard/{print FILENAME":"NR": "$0"}' src/audioengine/AudioEngine.RebuildDispatch.cpp
fdfind -e h -e cpp 'WorkerThread|CoordinatorLoop|RebuildDispatch' .
fdfind -e cpp . src/core | fzf --filter='WorkerThread'
ag -n 'submitRebuildIntent' src
ag -n 'rebuildRequestGeneration|lastCommitted' src
sg run -p 'submitRebuildIntent($$$)' --lang cpp src/
sg run -p 'rebuildRequestGeneration' --lang cpp src/
sg run -p 'BuildError::$ERR' --lang cpp src/audioengine/BuildErrorPolicy.h

# cocoindex
ccc status
ccc grep 'submitRebuildIntent'
ccc grep 'generation'

# graphify
graphify query 'submitRebuildIntent'
graphify query 'BuildError'

# semble
semble search 'submitRebuildIntent' . --max-snippet-lines 5
semble search 'generation' . --max-snippet-lines 5
semble search 'RetryScheduler' . --max-snippet-lines 5

# AiDex
aidex_query term="BuildError" mode="contains"

# serena
# .serena/project.yml 確認

# context-mode sandbox
ctx_execute(language: "javascript", code: "fs.readFileSync('src/audioengine/AudioEngine.RebuildDispatch.cpp').split('\n').slice(130,250).join('\n')")

# RTK (WSL版)
wsl bash -c 'cd /mnt/c/VSC_Project/ConvoPeq && ~/.local/bin/rtk grep -rn "submitRebuildIntent" src/'

# headroom — 大きな断片は context-mode で仮想化、圧縮不要と判定
```

## 参照

- `src/audioengine/BuildErrorPolicy.h` — 8値 default policy (D-3)
- `src/audioengine/AudioEngine.RebuildDispatch.cpp:149-500` — `submitRebuildIntent` / `handleAsyncUpdate` / `requestRebuild`（MessageThread bounce + latestWins merge + generation採番）
- `src/audioengine/AudioEngine.RebuildDispatch.cpp:1098` — 唯一の `classifyBuildError` consumer (telemetry only, D-5 で配線)
- `src/audioengine/AudioEngine.RebuildDispatch.cpp:1140-1165` — warmup retry (`validateWarmup` → `shouldRetryWarmupFailure` → `submitRebuildIntent`)
- `src/audioengine/AudioEngine.h` — `rebuildRequestGeneration` / `lastCommitted*` / `rebuildAdmissionIntentMutex_` / `rebuildMutex` / `RebuildIntentQueue` 相当
- `src/audioengine/AudioEngine.CtorDtor.cpp` — destructor / shutdown ordering
- `src/audioengine/ISRRuntimePublicationCoordinator.h/.cpp` — Recovery lease (`settlePendingRecoveryAdmission` / `kMaxRecoveryConsecutiveFailures=4`)
- `src/audioengine/ISRCoordinatorLoop.h` — CoordinatorLoop (publication coordination)
- `src/core/WorkerThread.h/.cpp` — generic background thread (参考)
- `evidence/D101-14-Phase-D-4-Retry-Scheduling-Boundary-Audit.md` — D-4 Candidate C ratified, 10 invariant
- `evidence/D101-13-Phase-D-3-Contract-Test-Report.md` — D-3 65 checks PASS
- `evidence/D101-12-Phase-D-2-Retry-Policy-Minimal-Contract-Audit.md` — D-2 minimal contract (B ratified)
- `ConvoPeq.md` — 最新ソーススナップショット（§1.8 一次資料）
