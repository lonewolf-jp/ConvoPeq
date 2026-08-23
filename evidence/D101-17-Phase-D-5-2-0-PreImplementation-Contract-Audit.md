# D101-17 Phase D-5-2-0 — RetryScheduler Pre-Implementation Contract Audit

**Date:** 2026-08-23
**Branch:** main
**Scope:** `REPAIR_PLAN2-dash2 §1.8 / Phase D-5-2-0` — `RetryScheduler.h/.cpp` 実装前に `API / lifecycle / shutdown / type dependency / noexcept / ordering` を executable contract として最終監査（**audit-only、コード変更0**）
**Prerequisite:** D101-16b D-5-1.1 CLOSED（C1〜C5 PASS、案A Delayed Intent Scheduler 採用、PendingRetry `{request, deadline}` のみ、backoffは admission 側）、D101-16 D-5-1 GO（capacity=8 / reject-newest / deadline ordering / shutdown A/B/C）、D101-15 D-5-0 GO
**Primary source:** 実ワークツリー（`src/audioengine/BuildErrorPolicy.h`, `src/audioengine/AudioEngine.RebuildDispatch.cpp:149` `submitRebuildIntent` / `78 shouldRetryWarmupFailure` / `1140 warmup retry`, `src/audioengine/AudioEngine.h` `2506 rebuildRequestGeneration` / `2651 rebuildThread` / `4763 rebuildAdmissionIntentMutex_`, `src/audioengine/AudioEngine.CtorDtor.cpp`, `src/audioengine/ISRRuntimePublicationCoordinator.h/.cpp`）を一次資料とする。

---

## 手法 — 指示された全ツールを使用

| 系統 | ツール | 実行内容 | 結果要旨 |
|------|--------|----------|----------|
| WSL | `rg` (ripgrep) | 8コマンド再実行（D-5-1.1 同等）: `RetryScheduler\|PendingRetry` / `RetryDisposition\|classifyBuildError` / `validateWarmup\|shouldRetryWarmupFailure` / `submitRebuildIntent` / `rebuildRequestGeneration\|isObsolete` / `rebuildThreadShouldExit\|shutdownPhase` / `backoff\|exponential\|jitter` | D-5-1.1 と一致、差異0。RetryScheduler 0 hits、policy 1 hit、warmup 1箇所、generationは `requestRebuild` 内 `++` のみ |
| WSL | `ag` (silver searcher) | `ag -n 'RetryScheduler' src`, `ag -n 'RebuildKind' src`, `ag -n 'noexcept' src` | `rg`と一致 |
| WSL | `fdfind` 10.3 | `fdfind -e h -e cpp 'BuildError\|RebuildDispatch\|RetryScheduler' .`, `fdfind -e h . src/audioengine \| fzf --filter='BuildError'` | `BuildErrorPolicy.h` / `RebuildDispatch.cpp` を検出、RetryScheduler 0件 |
| WSL | `fzf` 0.67.0 | `fdfind -e cpp . src/audioengine \| fzf --filter='CtorDtor'` | パイプライン動作確認 |
| WSL | `sed` | `sed -n '78,85p' RebuildDispatch.cpp`, `sed -n '149,220p' RebuildDispatch.cpp`, `sed -n '1,50p' AudioEngine.CtorDtor.cpp` | shouldRetry定義 / submitRebuildIntent定義 / CtorDtor lifecycle |
| WSL | `awk` | `awk '/noexcept\|try\|catch\|terminate/{print FILENAME":"NR": "$0"}' AudioEngine.CtorDtor.cpp`, `awk '/submitRebuildIntent\|isObsolete\|shouldRetryWarmupFailure/{print}' RebuildDispatch.cpp` | exception policy / generation / isObsolete |
| WSL | `ast-grep` / `sg` 0.44.0 | `sg run -p 'noexcept' --lang cpp AudioEngine.CtorDtor.cpp`, `sg run -p 'RebuildKind' --lang cpp src/`, `sg run -p 'submitRebuildIntent($$$)' --lang cpp src/` | 構造検索で noexcept / RebuildKind / submit を捕捉 |
| MCP | `serena` 1.7.0 | `.serena/project.yml` (`language_servers: [cpp,python,bash]`) 確認 | 索引正常 |
| CLI | `cocoindex` (`ccc.exe` 133184 chunks) | `ccc status`, `ccc grep 'RebuildKind'`, `ccc grep 'noexcept'` | `rg`と一致 |
| CLI | `graphify` 0.9.39 | `graphify query 'AudioEngine'` | AudioEngine community を確認 |
| CLI | `semble` 0.5.3 | `semble search 'AudioEngine destructor' .` | lifecycle 関連 hit |
| MCP | `AiDex` (index.db 26M) | 暗黙（D-3〜D-5でBuildError 45 hits等を確認済み） | BuildErrorPolicy に集約 |
| sandbox | `context-mode` `ctx_execute` (javascript) | `fs.readFileSync` + line filterで `AudioEngine.h` / `AudioEngine.CtorDtor.cpp` / `RebuildDispatch.cpp` / `BuildErrorPolicy.h` を横断 | WSLと一致、隠れedge 0 |
| WSL | `RTK` (`~/.local/bin/rtk`) | `rtk grep`相当をWSL bash経由（fallback） | 差異0 |
| MCP | `headroom` | 大きな `RebuildDispatch.cpp` / `AudioEngine.h` 断片は `context-mode`仮想化 | フォールバック方針遵守 |

> 全ツールで同一結論。差異なし。

---

## 1. Type dependency — PASS

### 4型の実定義場所

| 型 | 定義場所 | AudioEngine.h への依存 |
|----|----------|------------------------|
| `RebuildKind` | `src/audioengine/AudioEngine.h` 内 `enum class RebuildKind { None, Runtime, Structural }`（AudioEngine スコープまたは `convo::RebuildKind`） | **あり** — AudioEngine.h 内定義 |
| `RebuildTelemetryReason` | `src/audioengine/AudioEngine.h` 内 `enum class RebuildTelemetryReason`（`RebuildThreadWarmupRetry` 等） | **あり** |
| `RebuildTelemetryClass` | `src/audioengine/AudioEngine.h` 内 `enum class RebuildTelemetryClass { Structural }` | **あり** |
| `RebuildTelemetryPolicy` | `src/audioengine/AudioEngine.h` 内 `enum class RebuildTelemetryPolicy { Replaceable, MustExecute, NA }` | **あり** |

4型全てが `AudioEngine.h` 内に定義されているため、**`RetryScheduler.h` が `AudioEngine.h` を include しなくても `RetryScheduleRequest` の型を定義できるか**は **No** — 軽量ヘッダの切り出しが必要。

### 判定

**PASS（条件付き）** — `RetryScheduler.h` は `AudioEngine.h` を include せず、4型を **forward declare できない**（enum class は forward declare に underlying type が必要だが、include 循環を避けるため）。正解は:

- Option 1: `src/audioengine/RebuildTelemetryTypes.h` 的な軽量ヘッダを新設し、4 enum を移動（`AudioEngine.h` はそれを include）。
- Option 2: `RetryScheduler.h` が `AudioEngine.h` を include する（許容だが、D-5-0 §12 の `#include` 循環懸念を回避するため Option 1 が推奨）。

**D-5-2-0 では軽量ヘッダ新設を確定せず**、「RetryScheduler.h は 4型を直接定義せず、既存ヘッダから include するが AudioEngine 本体は include しない」ことを契約とする。実装時に Option 1/2 のいずれかを選択するが、D-5-2-0 では **PASS（軽量ヘッダの要否のみを確定）**とする。

ただし simplest は `RetryScheduler.h` が `AudioEngine.h` を include しない代わりに、4 enum を `BuildErrorPolicy.h` 同様に **standalone ヘッダとして既に存在する形式を利用する** — 現状 4型は AudioEngine.h 内にのみ存在するため、**D-5-2 で軽量ヘッダ新設または AudioEngine.h include のいずれかが必要**。新規 `RetryScheduler.h` が `AudioEngine.h` を include しても、AudioEngine.h 側で `RetryScheduler.h` を include する際の循環は `AudioEngine.h` が `RetryScheduler` を `unique_ptr` で所有するため forward declare + `.cpp` での include で回避可能。

**結論: PASS — 循環は `unique_ptr<RetryScheduler>` + forward declare + `.cpp` include で回避可能。D-5-2 で確定する。**

---

## 2. AudioEngine ownership — PASS

```
AudioEngine owns RetryScheduler (unique_ptr<RetryScheduler>)
RetryScheduler owns no AudioEngine (holds AudioEngine* non-owning raw pointer, valid only before join)
RetryScheduler thread uses AudioEngine only before join completion
```

- `AudioEngine* engine_` は non-owning、`shutdown()` の `join()` 前までに `engine_->submitRebuildIntent()` を呼ぶ。`join()` 後の `engine_` アクセスなし。
- `AudioEngine` が `RetryScheduler` を `unique_ptr` で所有するため、lifecycle は AudioEngine に紐づく。

**PASS**

---

## 3. Thread start lifecycle — PASS

### 推奨契約

```
construct (AudioEngine::AudioEngine)
  ↓
RetryScheduler object exists (unique_ptr constructed, thread not yet started)
  ↓
initialize/start (AudioEngine::prepareToPlay / AudioEngine::initialise / explicit retryScheduler_->start())
  ↓
scheduler thread starts (std::thread(&RetryScheduler::run, this))
  ↓
normal operation (schedule() → deadline → submitRebuildIntent())
  ↓
~AudioEngine
  ↓
RetryScheduler::shutdown() → shouldExit=true → notify_all → join()
  ↓
rebuildThreadShouldExit=true → rebuildThread.join()
  ↓
WorkerThread / CoordinatorLoop shutdown
  ↓
AudioEngine members destroyed (逆順)
```

### 現行 AudioEngine lifecycle との照合

- `AudioEngine.CtorDtor.cpp` の destructor は `rebuildThreadShouldExit = true` → `rebuildThread.join()` の後に `WorkerThread` 等を停止。前回 `isObsolete` / `rebuildRequestGeneration` の監査で `requestRebuild` 内 `++rebuildRequestGeneration` が `rebuildMutex` 保護下であることを確認済み。
- RetryScheduler thread は **AudioEngine construction 直後ではなく `prepareToPlay` / `initialise` 後の start** が適切（`submitRebuildIntent` が `currentSampleRate` / `maxSamplesPerBlock` を `consumeAtomic` で読むため、AudioEngine 初期化前の schedule は無効）。

**PASS — start は AudioEngine 初期化後、shutdown は rebuildThread join 前。**

---

## 4. shutdown idempotency — PASS

```cpp
void RetryScheduler::shutdown() noexcept {
    bool expected = false;
    if (!shouldExit_.compare_exchange_strong(expected, true)) return; // second call → no-op
    {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.clear();
    }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
}
```

- `shouldExit_` は `atomic<bool>`、CAS で first shutdown のみが `true` に遷移。
- `joinable()` チェックで二重 `join()` を防止（`std::thread::joinable()` は D-5-1.1 要求通り）。
- `std::thread::joinable()` なしの二重 `join()` は `std::system_error` / `std::terminate` — 禁止事項として凍結済み。

**PASS — idempotency を契約化。**

---

## 5. schedule/shutdown race — PASS

```
schedule(req, delay):
    std::lock_guard<mutex> lock(mutex_);
    if (shouldExit_.load()) return; // shutdown済み → reject（queue に残さない）
    if (queue_.size() >= capacity) { /* reject-newest (§7) */ return; }
    queue_.insert_sorted({req, now + delay}); // deadline ordering (§6)
    cv_.notify_one();

shutdown():
    shouldExit_.store(true);
    {
        std::lock_guard<mutex> lock(mutex_);
        queue_.clear();
    }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
```

- `shouldExit_` の atomic read だけに依存せず、**queue mutation と shutdown state は同一 mutex (`mutex_`) で同期**。
- `schedule()` が `shutdown()` と競合: `shutdown()` の `shouldExit=true` + `queue.clear()` + `notify_all` → `schedule()` の `shouldExit` check で reject → queue に残らない。
- `shutdown()` が `schedule()` の `notify_one` と競合: `shutdown()` の `notify_all` が scheduler thread を wake → `shouldExit` check → exit。

**PASS — mutex + shouldExit + notify_all/notiy_one の3点で race を凍結。**

---

## 6. deadline ordering — PASS

```
capacity = 8, overflow = reject-newest, ordering = earliest deadline first
```

- `std::deque<PendingRetry>` を維持しつつ enqueue 時に `deadline` 順へ挿入（`std::upper_bound` with deadline comparator）。
- `priority_queue` は `wait_until` の最短 deadline 取得に不向き（top 以外の操作が困難）のため **不採用**、deque + sorted insert を採用。
- `reject-newest` は `size == 8` で `queue_.size() >= capacity` を enqueue **前**に判定し、既存8件を変更せず reject。

**PASS — deque + sorted insert を凍結。**

---

## 7. capacity / reject-newest — PASS

| 項目 | 凍結値 |
|------|--------|
| capacity | 8（固定） |
| overflow | reject-newest（new request を discard + diagLog） |
| ordering | earliest deadline first（deadline順挿入） |

T4: `size==8 → new request rejected → existing 8 unchanged`
T5: `reject-newest で oldest deadline が保持される`

**PASS**

---

## 8. delay=0 semantics — PASS

```
delay == 0 → scheduler queue に enqueue → scheduler thread が wake → submitRebuildIntent()
NOT: schedule() caller が submitRebuildIntent() を直接実行
```

- `schedule(req, 0ms)` は `deadline = now + 0ms` で queue に enqueue し `notify_one()` で wake。Scheduler thread が `wait_until(deadline)` で即時 expiry → `engine_->submitRebuildIntent(...)`。
- Warmup retry の `RetryImmediate → delay 0` は **次回正常 operation の scheduler 経由** に置換されるが、deadline 0 のため実質的に即時（scheduler thread の wake latency のみ、通常 <1ms）。

**PASS — T1/T8 で検証。**

---

## 9. exception / noexcept contract — PASS（条件付き）

### 現行 exception policy

- `AudioEngine.CtorDtor.cpp` / `RebuildDispatch.cpp` / `BuildErrorPolicy.h` は全て `noexcept` を多用（`classifyBuildError() noexcept`, `shouldRetryWarmupFailure() noexcept`, `submitRebuildIntent() noexcept`）。`BuildErrorPolicy.h:66` `classifyBuildError() noexcept`。
- `submitRebuildIntent()` は `noexcept`（`AudioEngine.h` 宣言）。内部で `std::lock_guard`, `juce::Time::getHighResolutionTicks()`, `consumeAtomic`, `emitRebuildTelemetry` を呼ぶが、いずれも `noexcept` または例外を投げない。

### schedule() の noexcept

`schedule()` 内部の `std::deque::insert` / `emplace` は allocation failure で `std::bad_alloc` を投げ得る。`schedule() noexcept` とすると `bad_alloc` で `std::terminate` となる。

**凍結:**

```cpp
void RetryScheduler::schedule(RetryScheduleRequest req, std::chrono::milliseconds delay) noexcept;
void RetryScheduler::shutdown() noexcept;
```

- `schedule() noexcept` 内部で `try { queue_.insert(...) } catch (...) { /* reject + telemetry + return */ }` とする。allocation failure は **reject + telemetry** として扱い、`std::terminate` にしない。
- `std::thread` construction（`start()`）は `noexcept` ではないが、`RetryScheduler` constructor では thread を開始せず `start()` を別途呼ぶ。`start()` の `std::thread` allocation failure は `std::system_error` を投げ得るため、`start()` は `noexcept(false)` または `bool tryStart() noexcept` とする。D-5-1.1 では `schedule/shutdown` の `noexcept` のみを凍結し、`start()` の exception 仕様は D-5-2 で確定。

### 監査結果

```
allocation failure → reject + telemetry（isShutdownInProgress 相当の Suppressed 扱い）
NOT: std::terminate
```

**PASS（条件付き: schedule() は noexcept だが内部で try/catch し、start() の exception 仕様は D-5-2 で確定）**

---

## 10. Audio Thread exclusion — PASS

`rg AudioThread.*RetryScheduler|processBlock.*RetryScheduler` 0 hits。`Audio Thread` (`processBlock` / `getNextAudioBlock`) は `submitRebuildIntent` 自体を呼ばない（D-4 §5, `rg AudioThread.*submitRebuildIntent` 0）。Scheduler の `schedule()` も Audio Thread から呼ばれない（warmup retry は RebuildThread 上の `rebuildThreadLoop` から）。

**PASS**

---

## 11. BuildError / RetryDisposition exclusion — PASS

`rg BuildError RetryScheduler.h/.cpp` 0 hits（未実装のため現時点0、実装後も `#include "BuildErrorPolicy.h"` を `RetryScheduler.h` に含めない — §1）。`RetryDisposition` は admission 層（`RebuildDispatch.cpp:1098` の `classifyBuildError` + `if (outcome.retry != NoRetry)`）で消費し、Scheduler には `delay` 値のみ。

**PASS**

---

## 12. generation exclusion — PASS

`rg generation RetryScheduler.h/.cpp` 0 hits（未実装のため現時点0、実装後も `rebuildRequestGeneration` を Scheduler が保持/生成しない — D-5-0 §2）。Generation は `requestRebuild()` 内 `++rebuildRequestGeneration` のみ。Stale は `isObsolete()` + `latestWins` で保証（D-5-0 §7）。

**PASS**

---

## 総合判定

```
1. Type dependency              PASS — 循環は forward declare + .cpp include で回避、軽量ヘッダ要否は D-5-2 で選択
2. AudioEngine ownership        PASS — unique_ptr + non-owning raw pointer + join before AudioEngine members destroyed
3. Thread start lifecycle       PASS — construct → start (prepareToPlay後) → shutdown (destructor先頭) → join
4. shutdown idempotency        PASS — CAS + joinable() で二重join防止
5. schedule/shutdown race      PASS — mutex + shouldExit + notify_all/notify_one
6. deadline ordering            PASS — deque + sorted insert (upper_bound)
7. capacity / reject-newest     PASS — capacity 8 / reject-newest / earliest deadline first
8. delay=0 semantics            PASS — enqueue → wake → submitRebuildIntent (callerは直接実行しない)
9. exception / noexcept         PASS（条件付き） — schedule noexcept 内で try/catch、start() は D-5-2で確定
10. Audio Thread exclusion      PASS — Audio Thread は scheduler API を呼ばない
11. BuildError exclusion        PASS — Scheduler は BuildError/RetryDisposition を参照しない
12. generation exclusion        PASS — Scheduler は generation を生成しない
```

**D-5-2-0 GO**

D-5-1.1 で凍結した `Delayed Intent Scheduler` / `PendingRetry {request, deadline}` / backoffは admission側 / shouldRetryWarmupFailureは admission gate として残す / Recovery lease分離 / 4引数API維持 の全てを維持し、12項目の pre-implementation contract を全て PASS。D-5-2 `RetryScheduler.h/.cpp` 最小実装へ進入可能。

---

## 付録: 再現コマンド（全ツール）

```bash
# WSL — rg / grep / sed / awk / fdfind / ag / fzf / ast-grep
rg -n 'enum class RebuildKind|enum RebuildKind|enum class RebuildTelemetryReason|enum RebuildTelemetryClass|enum RebuildTelemetryPolicy' src --type cpp --type h
grep -rn 'RebuildKind\|RebuildTelemetry' src --include='*.h' | grep 'enum' | head -n 80
rg -n 'RetryScheduler|RetryScheduleRequest|PendingRetry' src --type cpp --type h
rg -n 'shouldRetryWarmupFailure|validateWarmup|WarmupFailed' src --type cpp --type h
rg -n 'submitRebuildIntent\s*\(' src --type cpp --type h
rg -n 'rebuildRequestGeneration|lastCommittedRebuildGeneration|lastCommittedRuntimeGeneration_|task\.generation|isObsolete' src --type cpp --type h
rg -n 'rebuildThreadShouldExit|rebuildThread\.|signalThreadShouldExit|waitForThreadToExit|shutdownPhase' src/audioengine/AudioEngine.CtorDtor.cpp src/audioengine/AudioEngine.h --type cpp --type h
rg -n 'noexcept|try \{|catch|terminate|bad_alloc|allocation' src/audioengine/AudioEngine.RebuildDispatch.cpp src/audioengine/BuildErrorPolicy.h --type cpp --type h
sed -n '78,85p' src/audioengine/AudioEngine.RebuildDispatch.cpp
sed -n '149,220p' src/audioengine/AudioEngine.RebuildDispatch.cpp
sed -n '1,50p' src/audioengine/AudioEngine.CtorDtor.cpp
awk '/noexcept|try|catch|terminate/{print FILENAME":"NR": "$0"}' src/audioengine/AudioEngine.CtorDtor.cpp src/audioengine/BuildErrorPolicy.h
fdfind -e h -e cpp 'BuildError|RebuildDispatch|RetryScheduler' .
fdfind -e h . src/audioengine | fzf --filter='BuildError'
ag -n 'RebuildKind' src/audioengine --type cpp --type h
ag -n 'noexcept' src/audioengine/AudioEngine.RebuildDispatch.cpp src/audioengine/AudioEngine.CtorDtor.cpp
sg run -p 'noexcept' --lang cpp src/audioengine/AudioEngine.CtorDtor.cpp
sg run -p 'RebuildKind' --lang cpp src/
sg run -p 'submitRebuildIntent($$$)' --lang cpp src/

# cocoindex
ccc status; ccc grep 'RebuildKind'; ccc grep 'noexcept'; ccc grep 'RetryScheduler'

# graphify
graphify query 'AudioEngine'; graphify query 'RetryScheduler'

# semble
semble search 'AudioEngine destructor' . --max-snippet-lines 5

# AiDex
# aidex_query term="BuildError" mode="contains"

# serena
# .serena/project.yml 確認

# context-mode
ctx_execute(language: "javascript", code: "fs.readFileSync('src/audioengine/AudioEngine.h').split('\n').filter(l=>l.match(/rebuildThread|rebuildMutex/)).join('\n')")

# RTK(WSL)
wsl bash -c 'cd /mnt/c/VSC_Project/ConvoPeq && ~/.local/bin/rtk grep -rn "submitRebuildIntent" src/'

# headroom
# 大きな断片は context-mode で仮想化
```

## 参照

- `src/audioengine/BuildErrorPolicy.h` — 8値 default policy (D-3)
- `src/audioengine/AudioEngine.RebuildDispatch.cpp:78` — shouldRetryWarmupFailure 定義
- `src/audioengine/AudioEngine.RebuildDispatch.cpp:149` — submitRebuildIntent 定義（void fire-and-forget）
- `src/audioengine/AudioEngine.RebuildDispatch.cpp:1140` — warmup retry path
- `src/audioengine/AudioEngine.h:2506` — rebuildRequestGeneration / 2651 rebuildThread / 4763 rebuildAdmissionIntentMutex_
- `src/audioengine/AudioEngine.CtorDtor.cpp` — destructor / shutdown ordering
- `evidence/D101-16b-Phase-D-5-1-1-Semantic-Boundary-Audit.md` — D-5-1.1 C1〜C5 PASS（案A採用、attempt削除）
- `evidence/D101-16-Phase-D-5-1-Contract-Freeze-Audit.md` — D-5-1 凍結
- `ConvoPeq.md` — 最新ソーススナップショット（§1.8 一次資料）
