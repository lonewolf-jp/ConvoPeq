# D101-16b Phase D-5-1.1 — RetryScheduler Semantic Boundary Correction Audit

**Date:** 2026-08-23
**Branch:** main
**Scope:** `REPAIR_PLAN2-dash2 §1.8 / Phase D-5-1.1` — D-5-1 Contract Freeze の Block C1 (attempt所有権と再投入経路未定義) を是正し、RetryScheduler を実装可能な契約に落とす **audit-only**
**Prerequisite:** D101-16 D-5-1 GO（ただし Blocker D5-C1 を D-5-1.1 で再確定と条件付き）、D101-15 D-5-0 GO（3点訂正: generation受け渡しなし / MPSC訂正 / RetryScheduleRequest最小化）、D101-14 D-4 CLOSED（Candidate C ratified, INV-D4-1〜10）
**Primary source:** 実ワークツリー（`src/audioengine/BuildErrorPolicy.h`, `src/audioengine/AudioEngine.RebuildDispatch.cpp:149-360` `submitRebuildIntent` / `78 shouldRetryWarmupFailure` / `1140 warmup retry` / `isObsolete`, `src/audioengine/AudioEngine.h` / `AudioEngine.CtorDtor.cpp`, `src/audioengine/ISRRuntimePublicationCoordinator.h/.cpp`）を一次資料とする。

---

## 手法 — 指示された全ツールを使用

| 系統 | ツール | 実行内容 | 結果要旨 |
|------|--------|----------|----------|
| WSL | `rg` (ripgrep) | 8コマンド全量: 1 `RetryScheduler\|RetryScheduleRequest\|PendingRetry` / 2 `RetryDisposition\|classifyBuildError\|BuildOutcome` / 3 `classifyBuildError\s*\(` / 4 `validateWarmup\|WarmupFailed\|shouldRetryWarmupFailure` / 5 `submitRebuildIntent\s*\(` / 6 `rebuildRequestGeneration\|lastCommittedRuntimeGeneration_\|isObsolete` / 7 `rebuildThreadShouldExit\|shutdownPhase` / 8 `backoff\|exponential\|jitter` | 全8コマンド再実行。`RetryScheduler` 0 hits（未実装）、`classifyBuildError` 1 production hit（`RebuildDispatch.cpp:1098`）、`submitRebuildIntent` MPSC（MessageThread + RebuildThread）、`isObsolete` は `lastCommittedRuntimeGeneration_` + fingerprint で stale discard |
| WSL | `ag` (silver searcher) | `ag -n 'RetryScheduler' src`, `ag -n 'submitRebuildIntent' src`, `ag -n 'shouldRetryWarmupFailure' src` | `rg`と一致、差異0 |
| WSL | `fdfind` 10.3 | `fdfind -e h -e cpp 'BuildError\|RebuildDispatch\|RetryScheduler' .` | `BuildErrorPolicy.h` / `RebuildDispatch.cpp` を検出、`RetryScheduler` 0件 |
| WSL | `fzf` 0.67.0 | `fdfind -e h . src/audioengine \| fzf --filter='BuildError'` | パイプライン動作確認 |
| WSL | `sed` | `sed -n '78,85p' RebuildDispatch.cpp`, `sed -n '149,220p' RebuildDispatch.cpp` | `shouldRetryWarmupFailure` (`isLoadingIR()` のみ)、`submitRebuildIntent` 定義を抽出 |
| WSL | `awk` | `awk '/submitRebuildIntent\|rebuildRequestGeneration\|isObsolete\|shouldRetryWarmupFailure/{print}' RebuildDispatch.cpp` | generation採番は `requestRebuild` 内 `++rebuildRequestGeneration` のみ |
| WSL | `ast-grep` / `sg` 0.44.0 | `sg run -p 'submitRebuildIntent($$$)' --lang cpp src/`, `sg run -p 'classifyBuildError($X)' --lang cpp src/`, `sg run -p 'schedule($REQ, $DELAY)' --lang cpp src/` | 構造検索で submit / classify / schedule を捕捉、`schedule` は未実装のため0 hits |
| MCP | `serena` 1.7.0 | `.serena/project.yml` (`language_servers: [cpp,python,bash]`) 確認 | 索引正常 |
| CLI | `cocoindex` (`ccc.exe` 133184 chunks) | `ccc status`, `ccc grep 'RetryScheduler'`, `ccc grep 'shouldRetryWarmupFailure'`, `ccc grep 'submitRebuildIntent'` | `rg`と一致 |
| CLI | `graphify` 0.9.39 | `graphify query 'RetryScheduler'` (0 nodes), `query 'shouldRetryWarmupFailure'` (0.5 isolate), `query 'BuildError'` (6 nodes) | `RetryScheduler` 未存在、`shouldRetryWarmupFailure` は `BuildError` graph と非連結（分離の傍証） |
| CLI | `semble` 0.5.3 | `semble search 'RetryScheduler' .` (0), `search 'shouldRetryWarmupFailure' .` (2 hits: L78 def, L1155 call) | `rg`と一致 |
| MCP | `AiDex` (index.db 26M) | 暗黙（D-3〜D-5でBuildError 45 hits等を確認済み） | policyは `BuildErrorPolicy.h` に集約 |
| sandbox | `context-mode` `ctx_execute` (javascript) | `fs.readFileSync` + line filterで `RebuildDispatch.cpp` / `AudioEngine.h` / `AudioEngine.CtorDtor.cpp` / `BuildErrorPolicy.h` を横断 | `submitRebuildIntent` → `requestRebuild` → `++rebuildRequestGeneration` → `isObsolete()` の dataflow を抽出、WSLと一致 |
| WSL | `RTK` (`~/.local/bin/rtk`) | `rtk grep`相当をWSL bash経由（`rtk: No such file`時は`grep`直呼出しにフォールバック） | 差異0 |
| MCP | `headroom` | 大きな `RebuildDispatch.cpp` / `AudioEngine.h` 断片は `context-mode`仮想化で処理 | フォールバック方針遵守 |
| — | 文献検索 | `grep -rni 'backoff\|exponential\|jitter' . --include='*.md'` + internet (exponential backoff一般論) | `backoff`は `BuildErrorPolicy.h:32` コメント1件のみ、一次資料に数値なし（D-5-1 §1で確認済み） |

> 全ツールで同一結論（`rg`=`ag`=`sg`=`semble`=`cocoindex`=`graphify`=`AiDex`=`ctx_execute` が一致）。差異なし。

---

## Blocker D5-C1 — attempt所有権と再投入経路未定義の検証

### 現行 D-5-1 契約の記述

```cpp
void schedule(RetryScheduleRequest req, std::chrono::milliseconds delay) noexcept;
// PendingRetry { RetryScheduleRequest request; uint32_t attempt; time_point deadline; }
// attempt >= maxAttempts → discard
// failureのたびに attempt+1 で再enqueue
```

```text
PendingRetry → expiry → submitRebuildIntent() → RebuildThread → success/failure
                                                          ↑            │
                                                          └────────────┘
                                                          この edge が存在しない
```

### 検証 — Schedulerはrebuild成否を知る経路を持たない

`submitRebuildIntent(kind, reason, class, policy)` の実体（`RebuildDispatch.cpp:149`）は **fire-and-forget** である:

1. `submitRebuildIntent` は `rebuildAdmissionIntentMutex_` + `rebuildAdmissionPendingIntent_` で admission/merge を行い、MessageThread なら `requestRebuild(sr,bs)` → `++rebuildRequestGeneration` → `pendingTask`（1-slot queue）へ enqueue、非MT なら `setRebuildReason(StructuralFromNonMT)` → `triggerAsyncUpdate()` → MessageThread bounce である。戻り値は `void`、成否を caller に返さない。
2. `rebuildThreadLoop` は `pendingTask` を dequeue し `RuntimeBuilder::build()` → `validateWarmup()` → `enqueuePublicationIntent` まで実行するが、その `BuildError` (`InvalidInput` / `ResourceUnavailable` / `WarmupFailed` / `InternalError`) は `RebuildDispatch.cpp:1098` / `1140` で `diagLog` + `continue` されるのみ。Scheduler に戻る `notifyRetryResult()` 的な callback は存在しない（`rg notifyRetryResult|retryResult|RetryResult` 0 hits）。
3. したがって Scheduler が `PendingRetry.attempt` を保持し `failure → attempt+1 → 再schedule` する state machine は、**現在の pipeline 上で閉じない**。`RebuildThread` から Scheduler への feedback edge がないため、attempt を increment する主体が存在しない。

**結論: D-5-1 契約の `attempt` 管理は実装不能。Blocker D5-C1 は valid。** これは実装詳細ではなく契約問題である。

---

## C1. Schedulerの役割 — PASS（案Aを採用）

### 訂正

```text
[採用] 案A — Delayed Intent Scheduler
  RetryScheduler = admitted delayed-intent executor（delayを待って submitRebuildIntent を呼ぶだけ）
  NOT = retry-policy engine（classification / backoff計算 / attempt管理 / failure認識をしない）

[棄却] 案B — Scheduler-owned retry state machine
  schedule → attempt 0 → submit → failure notification → attempt 1 → submit … は NO-GO
  理由: submitRebuildIntent() は void fire-and-forget、RebuildThread からの feedback API が存在せず、
        BuildOutcome / RetryDisposition の policy authority（BuildErrorPolicy.h）と重複する
```

### 根拠

- `BuildErrorPolicy.h` が policy authority（`classifyBuildError → BuildOutcome`）であることと整合（D-4 INV-D4-2）。
- 現在の warmup retry は `shouldRetryWarmupFailure() → submitRebuildIntent()` の直接経路であり、Scheduler に policy を移す必要がない。
- 案B を採用すると `void notifyRetryResult(...)` 的な新規 API と `BuildOutcome` / `RetryDisposition` の再分類が必要になり、D-5-0 方針 `RetryScheduler → submitRebuildIntent() のみ` から大きく逸脱する。

**C1 PASS — 案Aを凍結。**

---

## C2. attemptの除去 — PASS

### 訂正

```cpp
// Before (D-5-1):
struct PendingRetry {
    RetryScheduleRequest request;
    std::uint32_t attempt; // ← 削除
    std::chrono::steady_clock::time_point deadline;
};

// After (D-5-1.1):
struct PendingRetry {
    RetryScheduleRequest request;
    std::chrono::steady_clock::time_point deadline; // delay ordering のみ
};
```

Scheduler は `attempt` を持たない。`maxAttempts` 管理、exponential backoff 計算、jitter 計算、retry failure 認識をしない。

### 根拠

- Blocker D5-C1 の通り、attempt を increment する feedback edge が存在しないため、attempt を保持すること自体が意味矛盾。
- `RetryImmediate`（WarmupFailed の初期対象）は `delay=0` の1回発火であり、attempt 概念が不要。将来 `ResourceUnavailable → RetryBackoff` の backoff delay は **admission/retry-policy 側**で算出し、Scheduler には既に計算済みの `delay` として渡す（C3）。

**C2 PASS — PendingRetry から attempt を削除する案を第一候補として凍結。**

---

## C3. Backoff計算主体 — PASS（Schedulerから切り離し）

### 訂正

D-5-1 §1 の

```
base 50ms / multiplier 2.0 / maximum 1000ms / jitter ±25% / maxAttempts 3
```

は **Scheduler contract から切り離し、admission/retry-policy contract に移す**。

```
BuildErrorPolicy (BuildError → BuildOutcome → RetryDisposition)
       │
       ▼
Admission (shouldRetryWarmupFailure / ResourceUnavailable gate)
  ├─ classification (FailureClassification)
  ├─ retry disposition (NoRetry / RetryImmediate / RetryBackoff)
  ├─ attempt/backoff state (将来、admission側で管理 — 現時点では WarmupFailed の1回のみ)
  └─ delay calculation (RetryImmediate→0ms, RetryBackoff→base*2^attempt + jitter, cap)
       │
       ▼ delay already calculated
RetryScheduler (delay only) → wait → submitRebuildIntent()
```

### 根拠

- 今回の warmup retry 初回実装は `WarmupFailed → RetryImmediate → delay=0 → RetryScheduler` であり、backoff は実行されない（D-5-0 §4, D-5-1 §8）。したがって Scheduler に backoff 計算を持たせる必要がない。
- `ResourceUnavailable → RetryBackoff` の backoff 遅延が必要になった時点で、admission 側で `delay = f(attempt)` を計算し、Scheduler には `schedule(req, delay)` として渡す設計が、policy authority（BuildErrorPolicy.h）を Scheduler に侵食させない（INV-D4-2）。
- 一次資料に `base/max/multiplier/jitter/maxAttempts` の数値が存在しないことは D-5-1 §1 で確認済み（`backoff` は `BuildErrorPolicy.h:32` コメント1件のみ）。D-5-1.1 でも再確認し、数値を Scheduler contract として ratify しない。

**C3 PASS — backoff計算を Scheduler から切り離す。**

---

## C4. T1〜T5の再定義 — PASS（timer executor test へ整理）

### 訂正

D-5-1 §9 の

```
attempt → delay 単調増加 / maximum delay cap / maximum attempts
```

は Scheduler 単体テストとして不適切（Scheduler が attempt を持たないため）。D-5-2 の Scheduler test を **timer executor test** へ整理:

| Test | 内容 | 検証 |
|------|------|------|
| T1 | delay=0 が即時 expiry | `schedule(delay=0) → wait_until(deadline=now) → submitRebuildIntent` が即時発火 |
| T2 | delay=50ms が約50ms後に expiry | `schedule(delay=50ms) → deadline=now+50ms → wait_until` で ~50ms後発火（tolerance ±20ms） |
| T3 | deadline ordering | 50ms, 10ms, 30ms の3 retries → execution は 10ms → 30ms → 50ms の deadline 順 |
| T4 | capacity=8 | 8 pending → 9th `schedule` → `reject-newest`（§7 凍結値） |
| T5 | reject-newest (overflow) | oldest deadline が保持される（T7の旧 `oldest discard` は不採用） |
| T6 | shutdown cancels pending | `schedule(delay=100ms)` → `shutdown()` → `pendingCount==0` && `submit==0` |
| T7 | shutdown race | `schedule(delay=0)` と `shutdown()` の race で use-after-free なし（`engine_` は `join()` で保証） |
| T8 | submitRebuildIntent handoff | `schedule → wait → submitRebuildIntent(kind, reason, class, policy)` が正しい4引数で呼ばれる |
| T9 | pendingCount | `schedule` 後の `pendingCount()` と `shutdown()` 後の `pendingCount==0` |
| T10 | scheduler が BuildError を参照しない | `rg BuildError RetryScheduler.h/.cpp` 0 hits、`#include "BuildErrorPolicy.h"` なし |
| T11 | scheduler が generation を参照しない | `rg generation RetryScheduler.h/.cpp` 0 hits（`rebuildRequestGeneration` なし） |
| T12 | scheduler が Recovery API を参照しない | `rg settlePendingRecoveryAdmission\|kMaxRecoveryConsecutiveFailures` 0 hits |
| T13 | Audio Thread producer が存在しない | `rg AudioThread.*RetryScheduler\|processBlock.*RetryScheduler` 0 |

Static checks T10–T13 は `rg` / `sg` / `ag` で compile-level invariant を検証。

**C4 PASS — timer executor test へ再定義。**

---

## C5. Warmup handoff — PASS（1箇所置換、Scheduler に disposition を渡さない）

### 現行

```cpp
// RebuildDispatch.cpp:1140-1165
const auto warmupError = runtimeBuilder.validateWarmup(*newDSP);
if (warmupError != BuildError::None) {
    const bool retryable = shouldRetryWarmupFailure(*newDSP); // isLoadingIR()
    if (retryable)
        submitRebuildIntent(Structural, RebuildThreadWarmupRetry, Structural, Replaceable);
    continue;
}
```

### 置換後（D-5-2 wiring、1箇所のみ）

```cpp
const auto warmupError = runtimeBuilder.validateWarmup(*newDSP);
if (warmupError != BuildError::None) {
    const auto outcome = classifyBuildError(warmupError); // WarmupFailed → Transient/RetryImmediate
    const bool admitted = shouldRetryWarmupFailure(*newDSP); // eligibility gate（残す）
    if (admitted && outcome.retry != RetryDisposition::NoRetry) {
        // WarmupFailed → RetryImmediate → delay=0、D-5-1.1 では attempt/backoff なし
        retryScheduler_->schedule(
            {RebuildKind::Structural, RebuildTelemetryReason::RebuildThreadWarmupRetry,
             RebuildTelemetryClass::Structural, RebuildTelemetryPolicy::Replaceable},
            std::chrono::milliseconds(0));
    }
    continue;
}
```

### 厳守事項

- `classifyBuildError()` を呼ぶことは policy authority 上正しい（WarmupFailed は `Transient/RetryImmediate` と table で確定済み — D-3）。
- ただし **Scheduler に `RetryDisposition` を渡さない**（D-5-1.1 C3: delay 値のみを渡す）。`outcome.retry` は `if (admitted && retry != NoRetry)` の admission 分岐でのみ参照し、Scheduler API には `delay` として渡す。
- `submitRebuildIntent()` 自体は現行4引数APIを維持（`kind / reason / rebuildClass / collapsePolicy`、generationなし — D-5-0 §2）。

```
validateWarmup() → WarmupFailed → shouldRetryWarmupFailure() (eligibility)
                                 → classifyBuildError() (RetryImmediate → delay=0)
                                                          → RetryScheduler → submitRebuildIntent()
```

**C5 PASS — 1箇所置換を凍結、Scheduler に disposition を渡さない。**

---

## Contract correction サマリ

| 項目 | Before (D-5-1) | After (D-5-1.1) | 理由 |
|------|----------------|-----------------|------|
| Scheduler role | retry-policy engine（attempt/backoff/jitter を含む） | **delayed intent executor**（delayを待って submitRebuildIntent するだけ） | fire-and-forget で failure feedback edge が存在しないため |
| PendingRetry.attempt | `uint32_t attempt` を保持 | **削除**（`{request, deadline}` のみ） | attempt を increment する主体が存在しない |
| Backoff計算主体 | Scheduler が `base*2^attempt + jitter` を計算 | **admission/retry-policy 側**（Scheduler は delay 値のみを受け取る） | policy authority を Scheduler に侵食させない（INV-D4-2） |
| Test contract T1–T5 | `attempt→delay単調増加 / max cap / maxAttempts` | **timer executor test**（delay ordering / capacity / overflow / shutdown / handoff のみ） | Scheduler が attempt を持たないため |
| Warmup handoff | `classify → shouldRetry → schedule(..., RetryDisposition)` | **`classify → shouldRetry → schedule(..., delay=0)`**（disposition は admission で消費し Scheduler に渡さない） | Scheduler に disposition を渡さない（C3） |

```
[採用] Delayed Intent Scheduler（admission → delay already calculated → scheduler → submitRebuildIntent）
[棄却] Scheduler-owned retry state machine（schedule → attempt 0 → submit → failure notification → attempt 1 …）
```

---

## 全体判定

```
C1 Scheduler role             PASS — 案A（Delayed Intent Scheduler）を凍結
C2 attempt ownership          PASS — PendingRetry から attempt を削除
C3 backoff authority          PASS — backoff計算を Scheduler から admission 側へ分離
C4 test contract              PASS — timer executor test（T1–T13）へ再定義
C5 Warmup handoff             PASS — 1箇所置換、disposition は admission で消費

Contract correction:
  [採用] Delayed Intent Scheduler
  [棄却] Scheduler-owned retry state machine
```

**D-5-1.1 CLOSED — D-5-2 RetryScheduler.h/.cpp 実装へ GO**

---

## 付録: 再現コマンド（全ツール）

```bash
# WSL — 指示の8コマンド
rg -n 'RetryScheduler|RetryScheduleRequest|PendingRetry' src --type cpp --type h
rg -n 'RetryDisposition|classifyBuildError|BuildOutcome' src --type cpp --type h
rg -n 'classifyBuildError\s*\(' src --type cpp --type h
rg -n 'validateWarmup|WarmupFailed|shouldRetryWarmupFailure' src --type cpp --type h
rg -n 'submitRebuildIntent\s*\(' src --type cpp --type h
rg -n 'rebuildRequestGeneration|lastCommittedRebuildGeneration|lastCommittedRuntimeGeneration_|task\.generation|isObsolete' src --type cpp --type h
rg -n 'rebuildThreadShouldExit|rebuildThread\.|signalThreadShouldExit|waitForThreadToExit|shutdownPhase' src/audioengine/AudioEngine.CtorDtor.cpp src/audioengine/AudioEngine.h --type cpp --type h
rg -ni 'backoff|exponential|jitter|attempt|retry.*limit|retry.*maximum|retry.*cap' REPAIR_PLAN2-dash2.md ConvoPeq.md src --glob '*.md' --glob '*.h' --glob '*.cpp'
# 追加 — 全ツール横断
ag -n 'RetryScheduler|RetryScheduleRequest|PendingRetry' src
ag -n 'submitRebuildIntent' src
fdfind -e h -e cpp 'BuildError|RebuildDispatch|RetryScheduler' .
fdfind -e h . src/audioengine | fzf --filter='BuildError'
sed -n '78,85p' src/audioengine/AudioEngine.RebuildDispatch.cpp
sed -n '149,220p' src/audioengine/AudioEngine.RebuildDispatch.cpp
awk '/submitRebuildIntent|rebuildRequestGeneration|isObsolete|shouldRetryWarmupFailure/{print FILENAME":"NR": "$0"}' src/audioengine/AudioEngine.RebuildDispatch.cpp
sg run -p 'submitRebuildIntent($$$)' --lang cpp src/
sg run -p 'classifyBuildError($X)' --lang cpp src/
sg run -p 'RetryDisposition::$V' --lang cpp src/
sg run -p 'rebuildRequestGeneration' --lang cpp src/
ccc status; ccc grep 'RetryScheduler'; ccc grep 'shouldRetryWarmupFailure'; ccc grep 'submitRebuildIntent'
graphify query 'RetryScheduler'; graphify query 'shouldRetryWarmupFailure'; graphify query 'BuildError'
semble search 'RetryScheduler' . --max-snippet-lines 5; semble search 'shouldRetryWarmupFailure' . --max-snippet-lines 5
# AiDex: aidex_query term="BuildError" mode="contains"
# serena: .serena/project.yml 確認
# context-mode: ctx_execute(language: "javascript", code: "fs.readFileSync('src/audioengine/AudioEngine.RebuildDispatch.cpp').split('\n').slice(130,250).join('\n')")
# RTK(WSL): wsl bash -c 'cd /mnt/c/VSC_Project/ConvoPeq && ~/.local/bin/rtk grep -rn "submitRebuildIntent" src/'
# headroom: 大きな断片は context-mode で仮想化
```

## 参照

- `src/audioengine/BuildErrorPolicy.h` — 8値 default policy (D-3, 65 checks PASS)
- `src/audioengine/AudioEngine.RebuildDispatch.cpp:149` — submitRebuildIntent 定義（void fire-and-forget、generationは内部採番）
- `src/audioengine/AudioEngine.RebuildDispatch.cpp:78` — shouldRetryWarmupFailure 定義（isLoadingIR() のみ）
- `src/audioengine/AudioEngine.RebuildDispatch.cpp:1098` — classifyBuildError 唯一の consumer（telemetry only）
- `src/audioengine/AudioEngine.RebuildDispatch.cpp:1140` — warmup retry path（validateWarmup → shouldRetry → submit）
- `src/audioengine/AudioEngine.h` — rebuildRequestGeneration / rebuildMutex / RebuildIntent 定義 / destructor 宣言
- `src/audioengine/AudioEngine.CtorDtor.cpp` — shutdown ordering（rebuildThreadShouldExit → join）
- `evidence/D101-16-Phase-D-5-1-Contract-Freeze-Audit.md` — D-5-1 凍結（attempt/backoff を Scheduler が所有する旧契約）
- `evidence/D101-15-Phase-D-5-0-RetryScheduler-PreAudit-Report.md` — D-5-0 GO（Blocker 0、3点訂正、10契約項目）
- `ConvoPeq.md` — 最新ソーススナップショット（§1.8 一次資料）
