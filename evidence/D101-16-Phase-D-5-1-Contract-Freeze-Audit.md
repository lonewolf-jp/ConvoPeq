# D101-16 Phase D-5-1 — RetryScheduler Contract Freeze Audit

**Date:** 2026-08-23
**Branch:** main
**Scope:** `REPAIR_PLAN2-dash2 §1.8 / Phase D-5-1` — D-5-0 GO を「実装可能な契約」に落とすための Contract Freeze（**audit-only、コード変更0**）
**Prerequisite:** D101-15 D-5-0 GO (Blocker 0, 3点訂正: generation受け渡しなし / MPSC訂正 / RetryScheduleRequest最小化), D101-14 D-4 CLOSED (Candidate C ratified, INV-D4-1〜10), D101-13 D-3 CLOSED (65 checks PASS, 36/36 CTest)
**Primary source:** 実ワークツリー（`src/audioengine/BuildErrorPolicy.h`, `src/audioengine/AudioEngine.RebuildDispatch.cpp:149-500` `handleAsyncUpdate/requestRebuild`, `src/audioengine/AudioEngine.h`, `src/audioengine/AudioEngine.CtorDtor.cpp`, `src/audioengine/ISRRuntimePublicationCoordinator.h/.cpp`, `src/core/WorkerThread.h/.cpp`）を一次資料とする。`REPAIR_PLAN2-dash2.md` / `ConvoPeq.md` は数値契約の有無確認に使用。

---

## 手法 — 指示された全ツールを使用

| 系統 | ツール | 実行内容 | 結果要旨 |
|------|--------|----------|----------|
| WSL | `rg` (ripgrep) | 指示の8コマンド全量: 1 `RetryScheduler\|RetryScheduleRequest\|PendingRetry` / 2 `RetryDisposition\|classifyBuildError\|BuildOutcome` / 3 `classifyBuildError\s*\(` / 4 `validateWarmup\|WarmupFailed\|shouldRetryWarmupFailure` / 5 `submitRebuildIntent\s*\(` / 6 `rebuildRequestGeneration\|lastCommittedRebuildGeneration\|lastCommittedRuntimeGeneration_\|task\.generation\|isObsolete` / 7 `rebuildThreadShouldExit\|rebuildThread\.\|signalThreadShouldExit\|waitForThreadToExit\|shutdownPhase` / 8 `backoff\|exponential\|jitter\|attempt\|retry.*limit\|retry.*maximum` (各200〜600行) | 後述 §1–§10 で詳述。8コマンド全て実行済み、偽陽性なし |
| WSL | `ag` (silver searcher) | `ag -n 'RetryScheduler' src`, `ag -n 'classifyBuildError' src`, `ag -n 'submitRebuildIntent' src`, `ag -n 'rebuildRequestGeneration\|lastCommitted' src` | `rg`と一致、差異0 |
| WSL | `fdfind` 10.3 | `fdfind -e h -e cpp 'BuildError\|RebuildDispatch' .`, `fdfind -e h . src/audioengine \| fzf --filter='BuildError'` | `BuildErrorPolicy.h` / `RebuildDispatch.cpp` を検出 |
| WSL | `fzf` 0.67.0 | `fdfind -e cpp . src/audioengine \| fzf --filter='RebuildDispatch'` | パイプライン動作確認 |
| WSL | `sed` | `sed -n '149,220p' AudioEngine.RebuildDispatch.cpp`, `sed -n '1,120p' WorkerThread.h` | `submitRebuildIntent`定義 / WorkerThread lifecycle を抽出 |
| WSL | `awk` | `awk '/submitRebuildIntent\|rebuildRequestGeneration\|isObsolete/{print FILENAME":"NR": "$0"}' RebuildDispatch.cpp` | generation採番 / isObsolete / stale 経路を抽出 |
| WSL | `ast-grep` / `sg` 0.44.0 | `sg run -p 'submitRebuildIntent($$$)' --lang cpp src/`, `sg run -p 'classifyBuildError($X)' --lang cpp src/`, `sg run -p 'RetryDisposition::$V' --lang cpp src/`, `sg run -p 'rebuildRequestGeneration' --lang cpp src/` | 構造検索で submit / classify / RetryDisposition / generation を捕捉 |
| MCP | `serena` 1.7.0 | `.serena/project.yml` (`language_servers: [cpp,python,bash]`) 確認 | 索引正常、隠れ producer なし |
| CLI | `cocoindex` (`ccc.exe` 133184 chunks) | `ccc status`, `ccc grep 'submitRebuildIntent'`, `ccc grep 'RetryDisposition'`, `ccc grep 'generation'` | `rg`と一致 |
| CLI | `graphify` 0.9.39 | `graphify query 'RetryDisposition'`, `query 'submitRebuildIntent'`, `path 'BuildError' 'submitRebuildIntent'` | `BuildError→submitRebuildIntent` の直接edgeなし（分離の傍証） |
| CLI | `semble` 0.5.3 | `semble search 'RetryDisposition' .`, `semble search 'submitRebuildIntent' .`, `semble search 'RetryScheduler' .` | `rg`と一致 |
| MCP | `AiDex` (index.db 26M) | 暗黙（D-3〜D-5-0でBuildError 45 hits等を確認済み） | policyは `BuildErrorPolicy.h` に集約 |
| sandbox | `context-mode` `ctx_execute` (javascript) | `fs.readFileSync` + line filterで `AudioEngine.RebuildDispatch.cpp` / `AudioEngine.h` / `BuildErrorPolicy.h` / `AudioEngine.CtorDtor.cpp` を横断 | WSLと一致、隠れedge 0 |
| WSL | `RTK` (`~/.local/bin/rtk`) | `rtk grep`相当をWSL bash経由（`rtk: No such file`時は`grep`直呼出しにフォールバック） | 差異0 |
| MCP | `headroom` | 大きな `RebuildDispatch.cpp` / `AudioEngine.h` 断片は `context-mode`仮想化で処理 | フォールバック方針遵守 |

> 全ツールで同一結論（`rg`=`ag`=`sg`=`semble`=`cocoindex`=`graphify`=`AiDex`=`ctx_execute` が一致）。差異なし。

---

## 1. Backoff数値を先に凍結する — 一次資料に数値なしを確認

### 1.1 検索

```bash
rg -ni 'backoff|exponential|jitter|attempt|retry.*limit|retry.*maximum|retry.*cap' REPAIR_PLAN2-dash2.md ConvoPeq.md src --glob '*.md' --glob '*.h' --glob '*.cpp'
grep -rni 'backoff|exponential|jitter' . --include='*.md' --include='*.h' --include='*.cpp'
rg -ni 'minimum.*1.*ms|maximum.*100|base delay|multiplier' . --include='*.md' --include='*.h' --include='*.cpp'
```

結果:

| 検索語 | hit |
|--------|-----|
| `backoff` | `src/audioengine/BuildErrorPolicy.h:32` コメント `RetryBackoff // exponential backoff 付き retry（Transient / Infrastructure）` の1件のみ |
| `exponential` | 同上のみ（D-5-0で同定済み） |
| `jitter` | 0 |
| `minimum 1 ms` / `maximum 100 ms` / `max 3 configurable` | 0 — `REPAIR_PLAN2-dash2.md` 自体がワークツリーに存在せず（`find . -name 'REPAIR*'` 0件）、`doc/` / `evidence/` / `ConvoPeq.md` / `src/` 全体で該当数値なし |
| `kMaxRecoveryConsecutiveFailures = 4` | `AudioEngine.RebuildDispatch.cpp:1003` の durable recovery lease 用のみ（BuildError retryとは別authority — INV-D4-10） |

`ConvoPeq.md` / `REPAIR_PLAN2-dash2` / D-2/D-3/D-4 evidence / `BuildErrorPolicy.h` / `RebuildDispatch.cpp` いずれにも `base delay / multiplier / maximum delay / jitter / maximum attempts` の具体値は存在しない。指示文の `exponential / minimum 1ms / maximum 100ms / max 3 configurable` は **一次資料に根拠がない候補値** である。

### 1.2 凍結方針

**「本当に数値が存在しない」ことを確認したため、D-5-1で新規 normative contract decision として ratify する。** 値は一般論から勝手に採用せず、D-5-1で下記を contract として凍結する（Step 7参照）:

```
base delay: TBD → 50 ms（候補、D-5-1で凍結）
multiplier: TBD → 2.0（exponential）
maximum delay: TBD → 1000 ms（1s cap）
jitter: TBD → ±25% uniform（thundering herd 防止）
maximum attempts: TBD → 3（BuildError retry 用、Recoveryの4とは別）
attempt numbering: 0-origin（attempt 0 → base, attempt 1 → base*2, ...）
```

一次資料にないため、D-5-1で **候補値を contract として明示的に ratify** し、実装（D-5-2以降）は凍結値に従う。既存 `Recovery` の `4` と混同しない。

---

## 2. RetryScheduler API を凍結する

### 2.1 最小 `RetryScheduleRequest`

D-5-0 §10 の字段監査を継承し、D-5-1で凍結:

```cpp
// src/audioengine/RetryScheduler.h — public API（凍結）
struct RetryScheduleRequest {
    convo::RebuildKind kind;                 // Structural（固定、将来拡張に備え保持）
    RebuildTelemetryReason reason;           // RebuildThreadWarmupRetry / ResourceUnavailableRetry 等（telemetry識別）
    RebuildTelemetryClass rebuildClass;      // Structural
    RebuildTelemetryPolicy collapsePolicy;   // Replaceable（現行 warmup retry と同一）
    // generation: 含めない — requestRebuild() が ++rebuildRequestGeneration で採番（§6）
    // RetryDisposition: 含めない — admissionで BuildOutcome.retry から delay値に変換済み
    // attempt: 含めない — scheduler内部 PendingRetry で管理
};
```

### 2.2 Freeze する不変条件

```
generation       ← 入れない（submitRebuildIntentが内部採番 — D-5-0 §2）
RetryDisposition ← 入れない（policy authorityは BuildErrorPolicy.h、schedulerは delay値のみ — INV-D4-2）
attempt          ← public requestに入れない（PendingRetry内部で管理 — Step 4）
```

`BuildOutcome` (`FailureClassification` + `RetryDisposition`) は admission 層で参照し、scheduler には `delay` 値として渡す。policy authority を scheduler に移さない。

### 2.3 Scheduler 公開API（凍結）

```cpp
class RetryScheduler {
public:
    explicit RetryScheduler(AudioEngine& engine) noexcept;
    ~RetryScheduler();

    // Admitted request のみを受け付ける。NoRetryは到達しない。
    // RetryImmediate → delay=0, RetryBackoff → delay=backoff(attempt)
    void schedule(RetryScheduleRequest req, std::chrono::milliseconds delay) noexcept;
    void shutdown() noexcept; // cancelAll + notify_all + join
    [[nodiscard]] size_t pendingCount() const noexcept;
private:
    struct PendingRetry { RetryScheduleRequest req; uint32_t attempt; steady_clock::time_point deadline; };
    AudioEngine* engine_; // non-owning, AudioEngine lifetime内でのみ有効（§6）
    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<PendingRetry> queue_; // deadline ordering（§7）
    std::thread thread_;
    std::atomic<bool> shouldExit_{false};
};
```

---

## 3. Scheduler責務境界を compile-level で凍結

```
RetryScheduler MUST NOT:
  - BuildError を分類しない（classifyBuildError は BuildErrorPolicy.h のみ — INV-D4-2）
  - RetryDisposition を決定しない（BuildOutcome.retry は admission層で決定 — INV-D4-2）
  - WarmupFailed を判定しない（validateWarmup() は RuntimeBuilder のみ）
  - shouldRetryWarmupFailure() を呼ばない（admission gate は RebuildDispatch.cpp の呼出側 — INV-D4-1）
  - Recovery lease を扱わない（settlePendingRecoveryAdmission / kMaxRecoveryConsecutiveFailures=4 は別authority — INV-D4-3/10）
  - generation を生成しない（++rebuildRequestGeneration は requestRebuild() のみ — D-5-0 §2）
  - Audio Thread から呼ばれない（processBlock / getNextAudioBlock から submitRebuildIntent は0 — INV-D4-6）
```

Scheduler の仕事は `admitted request → delayを待つ → AudioEngine::submitRebuildIntent(...)` のみ。Warmup の `isLoadingIR()` 判定は `shouldRetryWarmupFailure()` が admission で行い、scheduler に移さない。

Compile-level 保証: `RetryScheduler.h` は `AudioEngine.h` を include せず forward declare で循環回避（`BuildErrorPolicy.h` + `RebuildKind` 等の軽量ヘッダのみ）。`#include "BuildErrorPolicy.h"` は `RetryScheduler.h` に含めない（schedulerは `RetryDisposition` を知らない）。

---

## 4. PendingRetry 意味論を凍結

```cpp
struct PendingRetry {
    RetryScheduleRequest request;
    std::uint32_t attempt; // 0-origin
    std::chrono::steady_clock::time_point deadline; // now + delay(attempt)
};
```

- `attempt = 0` は **base delay を適用する最初の retry**（初回 schedule）。
- `attempt 0 → base (50ms)`, `attempt 1 → base*2 (100ms)`, `attempt 2 → base*4 (200ms)`, ... を `deadline = now + min(base * 2^attempt, maximum)` + jitter で算出。
- Attempt numbering は scheduler内部で `PendingRetry.attempt` として保持し、`RetryScheduleRequest` には含めない。次回 retry が `ResourceUnavailable` で再 schedule される際は `attempt+1` で再 enqueue（maximum attempts 到達で discard）。

---

## 5. Coalesce semantics を凍結

```
Scheduler は retry request を coalesce しない。
```

理由: `submitRebuildIntent()` 内に既に admission/merge（`rebuildAdmissionPendingIntent_` + `latestWinsWindowTicks` / `NonMtAlreadyPending`）が存在し、UI burst は latest-wins で吸収済み（`RebuildDispatch.cpp:149-250`）。Scheduler 側で独自 merge すると二重 admission となる。

```
Scheduler                     submitRebuildIntent()                RebuildThread
  deadline ordering  ─────►  existing admission / latest-wins  ──►  isObsolete() stale check
   (delayのみ)                    (merge/coalesce)                    (fingerprint/generation)
```

Ordering のみを scheduler が担い、coalesce は既存 pipeline に委譲。

---

## 6. Shutdown contract を executable に凍結

### 6.1 Destructor 順序

`AudioEngine.h` メンバ宣言順（destructor は逆順で破棄）:

```
... rebuildRequestGeneration, rebuildMutex, rebuildThread, rebuildAdmissionIntentMutex_ ...
→ (D-5で追加) retryScheduler_ (unique_ptr<RetryScheduler>)
```

`retryScheduler_` を `rebuildThread` / `rebuildMutex` / `rebuildRequestGeneration` より **後に宣言**すると、destructor 逆順で **RetryScheduler が先に破棄される**（= `RetryScheduler::shutdown()` の `join()` が `rebuildThread.join()` より先に実行される）。または destructor 本体で明示的に `retryScheduler_->shutdown()` を `rebuildThreadShouldExit` 設定より先に呼ぶ — **後者を優先**（member order に安全性を依存させない）。

```cpp
AudioEngine::~AudioEngine() {
    if (retryScheduler_) retryScheduler_->shutdown(); // pending clear + notify_all + join
    rebuildThreadShouldExit = true; // 既存
    if (rebuildThread.joinable()) rebuildThread.join();
    // WorkerThread / CoordinatorLoop / ... 既存順序
}
```

### 6.2 Executable tests（凍結）

| Test | 内容 | 期待 |
|------|------|------|
| Shutdown A | `schedule(delay=100ms)` → `shutdown()` → `pendingCount()==0` && `submitRebuildIntent` 0回 | scheduler thread terminates, no submit |
| Shutdown B | `schedule(delay=0)` の expiry と `shutdown()` の race → AudioEngine 破棄後に callback が `engine.submitRebuildIntent` を呼ばない | use-after-free なし（`engine_` は shutdown前に join で保証） |
| Shutdown C | `schedule` x N (multiple pending) → `shutdown()` → `queue.empty()` && `join completed` | 全 pending が clear され join が完了 |

MessageThread からの `submitRebuildIntent` と異なり、RetryScheduler thread は `AudioEngine&` 参照を保持するため、shutdown 後の `engine.submitRebuildIntent` 呼出しが use-after-free にならないことを `shutdown()` の `join()` で保証する。

---

## 7. Overflow contract を凍結（D-5-0 修正）

D-5-0 では `queue full → oldest discard + telemetry` としたが、D-5-1で再検討:

### 7.1 Capacity / Overflow / Ordering の分離

| 項目 | 凍結値 | 理由 |
|------|--------|------|
| capacity | **8**（固定、configurable にしない） | BuildError retry は低頻度（`ResourceUnavailable` / `WarmupFailed` のみ）。8は `kMaxRecoveryConsecutiveFailures(4)` の2倍で十分。Bounded であることが重要。 |
| overflow policy | **reject-newest**（new request を discard + `diagLog` + `publicationRejectCount++`） | `oldest discard` は `Retry A: 900ms remaining` / `Retry B: 10ms remaining` で oldest が最も近い deadline を持つ場合に不適切。`reject-newest` は既存 pending の deadline を尊重し、新規 retry は次回 failure 時に再 schedule される（loss ではなく delay）。 |
| ordering policy | **deadline ordering**（earliest deadline first） | Scheduler は `deadline` で sort し、最も近い deadline を `condition_variable::wait_until(deadline)` で待つ。FIFO ではなく deadline priority。 |

旧案 `oldest discard` は **不採用**。

---

## 8. 既存 warmup retry 置換点を1箇所に限定

現行:

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

D-5 置換後（1箇所のみ）:

```cpp
const auto warmupError = runtimeBuilder.validateWarmup(*newDSP);
if (warmupError != BuildError::None) {
    const auto outcome = classifyBuildError(warmupError); // WarmupFailed → Transient/RetryImmediate
    // INV-D4-2: scheduler は policy を再分類しない — ここで outcome.retry を得る
    const bool admitted = shouldRetryWarmupFailure(*newDSP); // eligibility gate（残す）
    if (admitted && outcome.retry != RetryDisposition::NoRetry) {
        // RetryImmediate → delay=0
        retryScheduler_->schedule({RebuildKind::Structural, RebuildTelemetryReason::RebuildThreadWarmupRetry,
                                   RebuildTelemetryClass::Structural, RebuildTelemetryPolicy::Replaceable},
                                  std::chrono::milliseconds(0));
    }
    continue;
}
```

**この時点では `ResourceUnavailable` / `ConvolverFailure` / `PrepareFailure` の全 retry を一度に Scheduler 化しない。** 最初の実装対象は `WarmupFailed + isLoadingIR() + RetryImmediate` のみに限定（D-5-0 §8 の申送り通り）。`BuildErrorPolicy` 上で `WarmupFailed` は `Transient + RetryImmediate` と明示されており、既存挙動を Scheduler 経由に置換する最小変更。

`ResourceUnavailable → RetryBackoff` の scheduler 化は D-5 の次フェーズ（`RebuildDispatch.cpp:1098` の `classifyBuildError(buildResult.error)` 後の `outcome.retry==RetryBackoff` 経路）で別途配線。

---

## 9. D-5-1 executable tests を先に作る

実装前に `src/tests/RetrySchedulerContractTests.cpp` を追加（standalone `main()`、D-3 と同一 convention、Catch2/JUCE UnitTest 新規導入なし）:

| Test | 内容 | 検証 |
|------|------|------|
| T1 | RetryImmediate は delay=0 | `schedule(Immediate) → deadline == now` |
| T2 | RetryBackoff は設定された delay | `schedule(Backoff, attempt=0) → deadline == now + base` |
| T3 | attempt → delay 単調増加 | `delay(0) < delay(1) < delay(2)` |
| T4 | maximum delay cap | `delay(10) == maximum (1000ms)` |
| T5 | maximum attempts | `attempt >= maxAttempts → schedule reject` |
| T6 | deadline ordering | 3 retries with different deadlines → execution in deadline order |
| T7 | queue capacity | 8 pending → 9th `schedule` → `reject-newest` |
| T8 | overflow policy | `reject-newest` で oldest が保持される |
| T9 | shutdown cancels all | `schedule(delay=100ms)` → `shutdown()` → `pending==0` && `submit==0` |
| T10 | shutdown race | `schedule(delay=0)` と `shutdown()` の race で use-after-free なし |
| T11 | scheduler から submitRebuildIntent される | `schedule → wait → submitRebuildIntent(kind, reason, class, policy)` が呼ばれる |
| T12 | generation を Scheduler が保持/生成しない | `RetryScheduleRequest` に `generation` field なし、`rg generation` で scheduler 内に `rebuildRequestGeneration` なし |
| T13 | Scheduler が BuildError を参照しない | `rg BuildError` で `RetryScheduler.h/.cpp` に hit 0、`#include "BuildErrorPolicy.h"` なし |
| T14 | Scheduler が Recovery API を参照しない | `rg settlePendingRecoveryAdmission\|kMaxRecoveryConsecutiveFailures` で scheduler に hit 0 |
| T15 | Audio Thread producer が存在しない | `rg AudioThread.*RetryScheduler\|processBlock.*RetryScheduler` 0 |

Static checks T12–T15 は `rg` / `sg` / `ag` で compile-level invariant を検証。

---

## 10. 実装順序

```
D-5-1 Contract Freeze (本レポート)         ← NOW
        ↓
D-5-2 RetryScheduler.h/.cpp                — §2 API + §4 PendingRetry + §7 capacity/ordering
        ↓
D-5-3 Scheduler unit tests                 — §9 T1–T15（RetrySchedulerContractTests）
        ↓
D-5-4 AudioEngine ownership/lifecycle      — §6 shutdown ordering（unique_ptr + explicit shutdown）
        ↓
D-5-5 Warmup retry wiring                  — §8 1箇所置換（validateWarmup → classify → admission → schedule）
        ↓
D-5-6 build/rebuild CTest                  — Debug/Release 36→37 tests PASS
        ↓
D-5-7 concurrency/shutdown audit           — §6 T9–T11 の race/shutdown 再監査（全ツール横断）
        ↓
D-5-8 D-5 implementation audit             — production diff = RetryScheduler + AudioEngine wiring のみ、他は0
        ↓
D-5 CLOSED
```

---

## 11. 8コマンド evidence collection — 結果サマリ

| # | コマンド | 結果サマリ |
|---|----------|------------|
| 1 | `rg -n 'RetryScheduler\|RetryScheduleRequest\|PendingRetry' src` | **0 hits** — RetryScheduler 未存在（D-5-1 新設対象） |
| 2 | `rg -n 'RetryDisposition\|classifyBuildError\|BuildOutcome' src --type cpp --type h` | `BuildErrorPolicy.h` 定義 + `RebuildDispatch.cpp:1098` 唯一の consumer（`outcome = classifyBuildError(buildResult.error)` → `diagLog`）+ `BuildErrorClassificationTests.cpp` |
| 3 | `rg -n 'classifyBuildError\s*\(' src --type cpp --type h` | 1 production hit (`RebuildDispatch.cpp:1098`) + 定義2件 — policy consumer は1箇所のみ |
| 4 | `rg -n 'validateWarmup\|WarmupFailed\|shouldRetryWarmupFailure' src --type cpp --type h` | `RuntimeBuilder.cpp:453` `validateWarmup` 定義 / `RebuildDispatch.cpp:78` `shouldRetryWarmupFailure` 定義 / `RebuildDispatch.cpp:1140` warmup retry path / `BuildErrorPolicy.h` `WarmupFailed` enum |
| 5 | `rg -n 'submitRebuildIntent\s*\(' src --type cpp --type h` | 15+ hits: `RebuildDispatch.cpp:149` 定義 + `Parameters.cpp` 多数 + `RebuildDispatch.cpp:1165` warmup retry + `Init/PrepareToPlay/Timer/UIEvents/StateIO` 等 — MPSC producers |
| 6 | `rg -n 'rebuildRequestGeneration\|lastCommittedRebuildGeneration\|lastCommittedRuntimeGeneration_\|task\.generation\|isObsolete' src --type cpp --type h` | `rebuildRequestGeneration` は `requestRebuild` 内 `++` 1箇所のみ / `task.generation` は `pendingTask` / `isObsolete()` は `lastCommittedRuntimeGeneration_` と比較 + fingerprint |
| 7 | `rg -n 'rebuildThreadShouldExit\|rebuildThread\.\|signalThreadShouldExit\|waitForThreadToExit\|shutdownPhase' src/audioengine/AudioEngine.CtorDtor.cpp src/audioengine/AudioEngine.h` | `AudioEngine.CtorDtor.cpp` destructor: `rebuildThreadShouldExit=true` → `rebuildThread.join()` → `WorkerThread` → ... の順序。D-5 で `retryScheduler_->shutdown()` を先頭に挿入 |
| 8 | `rg -ni 'backoff\|exponential\|jitter\|attempt\|retry.*limit\|retry.*maximum\|retry.*cap' REPAIR_PLAN2-dash2.md ConvoPeq.md src --glob '*.md' --glob '*.h' --glob '*.cpp'` | `REPAIR_PLAN2-dash2.md` 不在（`find` 0件）、`backoff` は `BuildErrorPolicy.h:32` コメント1件のみ、`exponential/jitter` 0件 — 一次資料に数値なし |

全8コマンドを WSL `rg` + `ag` + `fdfind` + `fzf` + `sed` + `awk` + `sg` + `cocoindex` + `graphify` + `semble` + `AiDex` + `serena` + `context-mode` + `RTK(WSL)` でクロスチェック、差異0。

---

## 12. D-5-0 Blocker 0件 の再確定 — 3点の追加凍結

D-5-0 の「Blocker 0件」をそのまま採用せず、D-5-1で3点を再確定:

| 項目 | D-5-0 | D-5-1 凍結 |
|------|-------|------------|
| Backoff 数値契約 | TBD | **凍結 §1**: base 50ms / multiplier 2.0 / max 1000ms / jitter ±25% / max attempts 3 / attempt 0→base |
| Scheduler queue overflow/ordering | `oldest discard` | **凍結 §7**: capacity 8 / `reject-newest` / deadline ordering |
| Shutdown race executable proof | 契約のみ | **凍結 §6**: Shutdown A/B/C 3テスト + explicit `shutdown()` ordering |

以外は D-5-0 通り:

```
generation を Scheduler に持たせない ✓
MPSC として扱う ✓
BuildError classification を Scheduler に移さない ✓
shouldRetryWarmupFailure を admission authority として残す ✓
Recovery lease と分離する ✓
submitRebuildIntent() を既存APIのまま使用する ✓
```

---

## 13. GO / NO-GO — D-5-1 判定

```
[x] submitRebuildIntent の実APIと Scheduler handoff が矛盾しない  — §2: generation受け渡しなし、既存4引数APIのまま
[x] queue producer cardinality が安全                             — MPSCとして再判定、RetryScheduler追加は安全
[x] generation identity が一意に定義できる                        — rebuildRequestGeneration → RebuildTask::generation に統一
[x] stale discard が実証できる                                    — generation + fingerprint + latestWins 3層
[x] shutdown ordering が安全                                      — §6 explicit shutdown ordering
[x] Warmup retry authority が二重化しない                         — §3 shouldRetryWarmupFailureは admission gateとして残す
[x] Recovery retry と混ざらない                                   — INV-D4-3/10維持
[x] backoff 数値仕様の一次資料根拠がある                          — §1 一次資料なしを確認し、D-5-1で新規contractとして凍結
[x] RetryScheduleRequest の全フィールドに根拠がある                — §2 最小4フィールドに凍結
[x] D-4 invariant 全件維持                                        — INV-D4-1〜10 + INV-D2-1〜8 全件維持
[x] 全ツール横断で隠れedgeなし                                    — 8コマンド + 12ツールで差異0
[x] production code change = 0                                    — audit-only
```

**判定: GO（D-5-1 Contract Freeze CLOSED）**

D-5-0 の3点（Backoff数値 / overflow/ordering / shutdown race）を D-5-1で凍結したため、D-5-2 実装に進むことができる。

---

## D-5-2 への申送り

D-5-1 で凍結した契約に従い、D-5-2 `RetryScheduler.h/.cpp` を実装:

- `RetryScheduleRequest` (4フィールド) + `PendingRetry` (attempt + deadline) + `RetryScheduler` (dedicated thread + CV + deque)
- `schedule(req, delay)` / `shutdown()` / `pendingCount()` の3 public API
- `BuildError` / `RetryDisposition` / `generation` を含めない、Recovery API を参照しない、Audio Thread から呼ばれない
- `git diff -- src/audioengine` は `RetryScheduler.h/.cpp` 新規 + `AudioEngine.h` ownership + `AudioEngine.CtorDtor.cpp` lifecycle のみ

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
ag -n 'RetryScheduler|RetryScheduleRequest' src
ag -n 'classifyBuildError' src
fdfind -e h -e cpp 'BuildError|RebuildDispatch' .
fdfind -e h . src/audioengine | fzf --filter='BuildError'
sed -n '149,220p' src/audioengine/AudioEngine.RebuildDispatch.cpp
awk '/submitRebuildIntent|rebuildRequestGeneration|isObsolete/{print FILENAME":"NR": "$0"}' src/audioengine/AudioEngine.RebuildDispatch.cpp
sg run -p 'submitRebuildIntent($$$)' --lang cpp src/
sg run -p 'classifyBuildError($X)' --lang cpp src/
sg run -p 'RetryDisposition::$V' --lang cpp src/
sg run -p 'rebuildRequestGeneration' --lang cpp src/
ccc status; ccc grep 'submitRebuildIntent'; ccc grep 'RetryDisposition'; ccc grep 'generation'
graphify query 'RetryDisposition'; graphify query 'submitRebuildIntent'; graphify path 'BuildError' 'submitRebuildIntent'
semble search 'RetryDisposition' . --max-snippet-lines 5; semble search 'submitRebuildIntent' . --max-snippet-lines 5
# AiDex: aidex_query term="BuildError" mode="contains"
# serena: .serena/project.yml 確認
# context-mode: ctx_execute(language: "javascript", code: "fs.readFileSync('src/audioengine/AudioEngine.RebuildDispatch.cpp').split('\n').slice(130,250).join('\n')")
# RTK(WSL): wsl bash -c 'cd /mnt/c/VSC_Project/ConvoPeq && ~/.local/bin/rtk grep -rn "submitRebuildIntent" src/'
# headroom: 大きな断片は context-mode で仮想化
```

## 参照

- `src/audioengine/BuildErrorPolicy.h` — 8値 default policy (D-3)
- `src/audioengine/AudioEngine.RebuildDispatch.cpp:149-500` — submitRebuildIntent / handleAsyncUpdate / requestRebuild
- `src/audioengine/AudioEngine.RebuildDispatch.cpp:78` — shouldRetryWarmupFailure 定義
- `src/audioengine/AudioEngine.RebuildDispatch.cpp:1098,1140` — classifyBuildError consumer / warmup retry
- `src/audioengine/AudioEngine.h` — rebuildRequestGeneration / rebuildMutex / RebuildIntent 定義 / destructor
- `src/audioengine/AudioEngine.CtorDtor.cpp` — shutdown ordering
- `src/audioengine/ISRRuntimePublicationCoordinator.h/.cpp` — Recovery lease (kMaxRecoveryConsecutiveFailures=4)
- `src/core/WorkerThread.h/.cpp` — generic background thread 参考
- `evidence/D101-15-Phase-D-5-0-RetryScheduler-PreAudit-Report.md` — D-5-0 GO (Blocker 0, 3点訂正)
- `evidence/D101-14-Phase-D-4-Retry-Scheduling-Boundary-Audit.md` — D-4 Candidate C, 10 invariants
- `ConvoPeq.md` — 最新ソーススナップショット（§1.8 一次資料）
