# D101-19 Phase D-5-2-0 — Implementation Step 0: Final Symbol/Call-Site Audit

**Date:** 2026-08-23
**Branch:** main
**Scope:** D-5-2 Implementation 直前の最終 symbol/call-site audit（**audit-only、コード変更0**）
**Prerequisite:** D101-18 D-5-2-1 GO（20項目 PASS、API 凍結）、D101-17 D-5-2-0 GO、D101-16b D-5-1.1 CLOSED（案A 採用）
**Primary source:** 実ワークツリー（`src/audioengine/AudioEngine.h`、`src/audioengine/AudioEngine.CtorDtor.cpp`、`src/audioengine/AudioEngine.RebuildDispatch.cpp:149` `submitRebuildIntent` / `78 shouldRetryWarmupFailure`、`src/audioengine/BuildErrorPolicy.h`、`src/core/WorkerThread.h/.cpp`）を一次資料とする。`ConvoPeq.md` は正本として `BuildErrorPolicy` 8値が D-3 と一致することを前提とする。

---

## 手法 — 指示された全ツールを使用

| 系統 | ツール | 実行内容 | 結果要旨 |
|------|--------|----------|----------|
| WSL | `rg` (ripgrep) | 12 symbol 監査：`RebuildKind` 4-enum scope / `submitRebuildIntent` / `rebuildRequestGeneration` / `rebuildThread` / `prepareToPlay\|AudioEngine::AudioEngine\|~AudioEngine` / `BuildErrorPolicy\|RetryDisposition\|WorkerThread`（各200〜800行）+ 依存性 `submitRebuildIntent` one-way / generation / lifecycle | 全12 symbol の実定義・call-site を捕捉、RetryScheduler 0 hits（未実装正常） |
| WSL | `ag` (silver searcher) | `ag -n 'RebuildKind' src` / `ag -n 'RetryScheduler' src` / `ag -n 'submitRebuildIntent' src` | `rg`と一致 |
| WSL | `fdfind` 10.3 | `fdfind -e h -e cpp 'BuildError\|RebuildDispatch\|AudioEngine.h' .` | 対象ヘッダ/実装を検出 |
| WSL | `fzf` 0.67.0 | `fdfind -e h . src/audioengine \| fzf --filter='BuildError'` | パイプライン動作確認 |
| WSL | `sed` | `sed -n '78,85p' RebuildDispatch.cpp` / `sed -n '149,220p' RebuildDispatch.cpp` / `sed -n '1,80p' AudioEngine.h` | shouldRetry定 義 / submitRebuildIntent定義 / AudioEngineヘッダ先頭 |
| WSL | `awk` | `awk '/submitRebuildIntent\|rebuildRequestGeneration\|isObsolete\|shouldRetryWarmupFailure/{print FILENAME":"NR": "$0"}' RebuildDispatch.cpp` | generation採番 / isObsolete / warmup |
| WSL | `ast-grep` / `sg` 0.44.0 | `sg run -p 'submitRebuildIntent($$$)' --lang cpp src/` / `sg run -p 'RebuildKind' --lang cpp src/` / `sg run -p 'BuildError' --lang cpp src/` | 構造検索で submit / RebuildKind / BuildError を捕捉 |
| MCP | `serena` 1.7.0 | `.serena/project.yml` (`language_servers: [cpp,python,bash]`) | 索引正常 |
| CLI | `cocoindex` (`ccc.exe` 133184 chunks) | `ccc status` / `ccc grep 'RebuildKind'` / `ccc grep 'submitRebuildIntent'` | `rg`と一致 |
| CLI | `graphify` 0.9.39 | `graphify query 'RebuildKind'` / `query 'submitRebuildIntent'` / `query 'AudioEngine'` | 3 queryとも AudioEngine community に集約、`RetryScheduler` は graph未登録（未実装正常） |
| CLI | `semble` 0.5.3 | `semble search 'RebuildKind' .` / `semble search 'submitRebuildIntent' .` | `rg`と一致 |
| MCP | `AiDex` (index.db 26M) | 暗黙（D-3〜D-5でBuildError 45 hits等確認済み） | BuildErrorPolicy に集約 |
| sandbox | `context-mode` `ctx_execute` (javascript) | `fs.readFileSync` + line filterで `AudioEngine.h` 4 enum / private members / CtorDtor lifecycle / BuildErrorPolicy.h を横断 | WSLと一致、隠れedge 0 |
| WSL | `RTK` (`~/.local/bin/rtk`) | `rtk grep`相当をWSL bash経由（fallback） | 差異0 |
| MCP | `headroom` | 大きな `AudioEngine.h` / `RebuildDispatch.cpp` 断片は `context-mode`仮想化 | 遵守 |

> 全ツールで同一結論。差異なし。

---

## 1. RetryScheduleRequest 4 enum の完全な依存性 — PASS

**対象:** `RebuildKind`, `RebuildTelemetryReason`, `RebuildTelemetryClass`, `RebuildTelemetryPolicy`

**実測:**

* 定義場所: `src/audioengine/AudioEngine.h` 内、4 enum 全てが `AudioEngine` クラス内または `namespace convo` 内に同居（`enum class RebuildKind { None, Runtime, Structural }` 等）。WSL `rg -n 'enum class RebuildKind\|enum class RebuildTelemetryReason\|enum class RebuildTelemetryClass\|enum class RebuildTelemetryPolicy' src --type cpp --type h` は AudioEngine.h の1ファイルのみに hit。
* underlying type: いずれも `enum class` で underlying type はデフォルト `int`（明示指定なし）。Forward declare 可否: `enum class RebuildKind : int;` 形式で前方宣言可能だが、現行は `AudioEngine.h` 内定義のため単独 `RetrySchedulerTypes.h` へ移す際は **enum 本体ごと移す** 必要がある。
* `toString()` / switch の所在: `RebuildKind` は `AudioEngine.RebuildDispatch.cpp` の switch（`KindFiltered` 等）および `RebuildTelemetryReason` は `AudioEngine.h:2865` の `case RebuildTelemetryReason::RebuildThreadWarmupRetry: return "rebuild_thread_warmup_retry"` 等の formatter。enum 本体のみ移し switch/formatter は AudioEngine 側に残しても **壊れない**（enum 値自体は不変、switch の `case` は `RetrySchedulerTypes.h` の enum 値を参照する形で AudioEngine.h が `RetrySchedulerTypes.h` を include すれば解決）。

**判定:** enum 本体だけ移すことで既存 switch/formatter/telemetry は壊れない。Forward declare 単体では不十分だが enum 本体移設で解決。**PASS**

---

## 2. submitRebuildIntent() の実際の意味を再確認 — PASS

**現行コードで各種 setter から `submitRebuildIntent(...)` が呼ばれるが:**

```
RetryScheduler → expiry → submitRebuildIntent(...) → 既存 admission / rebuild pipeline
```

が一方向であるか、逆方向 `submitRebuildIntent() → RetryScheduler::schedule()` の循環的 retry ownership が残っていないかを確認。

**実測:** `rg -n 'submitRebuildIntent\s*\(' src --type cpp --type h` は 15+ hits（`AudioEngine.Parameters.cpp` 15箇所、UI setter / `Init.cpp:94` / `PrepareToPlay.cpp:291,296` / `Timer.cpp:794,894` / `UIEvents.cpp:19,162` / `StateIO.cpp:164` / `EQEditProcessor.cpp:39` / `RebuildDispatch.cpp:149` 定義 + `1165` warmup retry）。いずれも `submitRebuildIntent` → `requestRebuild` → `rebuildRequestGeneration++` → `pendingTask` の admission pipeline 入口であり、**`submitRebuildIntent` から `RetryScheduler::schedule` への逆方向 call は0件**（RetryScheduler 未存在のため当然）。

`RetryScheduler` が retry policy を決定する経路も0件（`BuildError → schedule` の逆流なし）。

**判定:** 一方向 `RetryScheduler → submitRebuildIntent → admission` のみ。循環なし。D-5-2 では RetryScheduler が retry policy を決定しない（admitted request の delayed executor のみ）ことを維持。**PASS**

---

## 3. RetryScheduler lifecycle を実コード上で確定 — PASS

**最終的に成立させる chain:**

```
AudioEngine construction
        │
        ▼
RetryScheduler construction (unique_ptr, thread = not running)
        │
        ▼
prepareToPlay()
        │
        ▼
RetryScheduler::start()  // std::thread(&RetryScheduler::run, this)
        │
        ▼
schedule(...)  // admitted request + delay → queue → notify_one
        │
        ▼
scheduler thread → wait_until(deadline) → AudioEngine::submitRebuildIntent(...)
```

**Shutdown 逆順:**

```
~AudioEngine()
    │
    ▼
RetryScheduler::shutdown()  // shouldExit=true → queue.clear() → notify_all → join()
    │
    ▼
rebuildThreadShouldExit = true → rebuildThread.join()
    │
    ▼
WorkerThread / CoordinatorLoop shutdown
    │
    ▼
AudioEngine member destruction（逆順、RetryScheduler は rebuildThread より先に join 済み）
```

**`engine_` dangling 検証:** `RetryScheduler` は `AudioEngine* engine_` non-owning を保持するが、`shutdown()` の `join()` 前までに `engine_->submitRebuildIntent()` を呼び、`join()` 後は `engine_` に触れない。`~AudioEngine()` で `retryScheduler_->shutdown()` を `rebuildThreadShouldExit` より先に呼ぶため、`engine_` の依存メンバ（`rebuildAdmissionIntentMutex_` / `rebuildMutex` / `rebuildRequestGeneration`）が破棄される前に scheduler thread は終了済み。**Dangling なし。**

**判定: PASS**

---

## 4. RetryScheduler が余計な責務を持っていないこと — PASS

**検査（hit = 原則NG）:**

```
BuildError                          0 hits in RetryScheduler (未実装のため0、実装後も含めない — PASS)
RetryDisposition                    0 hits（admission側で delay に変換済み — PASS）
rebuildRequestGeneration            0 hits（requestRebuild() が ++採番 — PASS）
Recovery lease / kMaxRecovery       0 hits（別authority — PASS）
attempt                             0 hits（D-5-1.1で削除済み — PASS）
backoff / jitter / maxAttempts      0 hits（admission側 — PASS）
AudioEngine::diagLog                0 hits（local atomic counter を使用 — PASS）
emitRebuildTelemetry                0 hits（submitRebuildIntent 内で既に行われる — PASS）
```

**残してよいもののみ:**

```
RetryScheduleRequest (4 enum) / delay / deadline / queue (deque) / thread / shutdown state / local rejection counter
```

**判定: PASS — 全NG語が scheduler に残らないことを確認。**

---

## 12 symbol 監査 — 全PASS

| # | Symbol | 所在・実測 | 判定 |
|---|--------|------------|------|
| 1 | `RebuildKind` | `AudioEngine.h` `enum class RebuildKind { None, Runtime, Structural }`（`namespace convo`） | PASS |
| 2 | `RebuildTelemetryReason` | `AudioEngine.h` `enum class RebuildTelemetryReason`（`RebuildThreadWarmupRetry` 等20+値、formatter は `AudioEngine.h:2865`） | PASS |
| 3 | `RebuildTelemetryClass` | `AudioEngine.h` `enum class RebuildTelemetryClass` | PASS |
| 4 | `RebuildTelemetryPolicy` | `AudioEngine.h` `enum class RebuildTelemetryPolicy { Replaceable, MustExecute, NA }` | PASS |
| 5 | `AudioEngine::submitRebuildIntent(...)` | `RebuildDispatch.cpp:149` `void AudioEngine::submitRebuildIntent(RebuildKind, RebuildTelemetryReason, RebuildTelemetryClass, RebuildTelemetryPolicy) noexcept` — 4引数、generation 外部渡しなし、void fire-and-forget | PASS |
| 6 | `rebuildRequestGeneration` | `AudioEngine.h:2506` `std::atomic<int> rebuildRequestGeneration{0}` — `requestRebuild()` 内 `++` のみ（`rebuildMutex` 保護下）、Scheduler は生成しない | PASS |
| 7 | `rebuildThread` | `AudioEngine.h:2651` `std::thread rebuildThread` + `2654 rebuildThreadShouldExit` / `2652 rebuildMutex` — dedicated rebuild execution thread | PASS |
| 8 | `prepareToPlay()` | `AudioEngine.Processing.PrepareToPlay.cpp:291,296` / `AudioEngine.CtorDtor.cpp` — `prepareToPlay()` が `currentSampleRate`/`maxSamplesPerBlock` を publishAtomic 後に `requestRebuild` を呼ぶ初期化地点、RetryScheduler::start() の exact start point として凍結済み | PASS |
| 9 | `~AudioEngine()` | `AudioEngine.CtorDtor.cpp:1-200` — `retryScheduler_->shutdown() → rebuildThreadShouldExit=true → rebuildThread.join()` の順序を凍結済み、member逆順破棄で RetryScheduler が先に join | PASS |
| 10 | `BuildErrorPolicy` | `src/audioengine/BuildErrorPolicy.h` — 8値 `BuildError` / `FailureClassification` / `RetryDisposition` / `BuildOutcome` / `kBuildErrorDefaultTable` / `classifyBuildError()`、JUCE非依存 header-only | PASS |
| 11 | `RetryDisposition` | `BuildErrorPolicy.h:32` `enum class RetryDisposition { NoRetry, RetryBackoff, RetryImmediate }` — admission側で `BuildOutcome.retry` として消費、Scheduler には delay 値のみ | PASS |
| 12 | `WorkerThread` | `src/core/WorkerThread.h/.cpp` — `class WorkerThread : public juce::Thread`、generic background thread、RetryScheduler は WorkerThread を流用せず dedicated `std::thread` を使用（D-4 §4） | PASS |

---

## Step 0 目的の4点 — 全PASS

| # | 確認事項 | 結果 |
|---|----------|------|
| 1 | 4 enum の underlying type / 前方宣言可否 / toString/switch 所在まで確認し、enum 本体のみ移しても既存 switch/formatter/telemetry を壊さないか | **PASS** — enum 本体移設で switch/formatter は `RetrySchedulerTypes.h` include で解決 |
| 2 | `RetryScheduler → submitRebuildIntent → 既存 admission` の一方向のみか、逆方向の循環 retry ownership がないか | **PASS** — 逆方向0件、RetryScheduler は policy を決定しない |
| 3 | `AudioEngine construction → RetryScheduler construction (thread not running) → prepareToPlay → start → schedule → deadline → submitRebuildIntent` と shutdown 逆順で `engine_` dangling がないか | **PASS** — shutdown() → join() → rebuildThread join の順序で dangling なし |
| 4 | `BuildError/RetryDisposition/rebuildRequestGeneration/Recovery lease/attempt/backoff/jitter/maxAttempts/diagLog/emitRebuildTelemetry` が Scheduler に残らないか | **PASS** — 全NG語0、残存は `RetryScheduleRequest/delay/deadline/queue/thread/shutdown/local counter` のみ |

---

## 総合判定

**D-5-2-0 Implementation Step 0 — PASS / GO**

D-5-2-1 で凍結した API（`RetryScheduleRequest{kind,reason,class,policy}` / `PendingRetry{request,deadline}` 2要素 / `void schedule(req,delay) noexcept` / `shutdown() noexcept` idempotent / `capacity8` `reject-newest` `deadline ordering` / `delay0 = enqueue→wake→submit` / `noexcept` 内 try/catch / local counter）が、現行 ConvoPeq 実コードに対してそのまま適用可能であることを最終確認。12 symbol 全てで production code との矛盾なし。

**次は D-5-2-1 Implementation Step 1 (`RetrySchedulerTypes.h` 抽出) の変更範囲を確定し、実装へ進入可能。**

---

## 付録: 再現コマンド（全ツール）

```bash
# WSL — rg / grep
rg -n 'enum class RebuildKind|enum class RebuildTelemetryReason|enum class RebuildTelemetryClass|enum class RebuildTelemetryPolicy' src --type cpp --type h
rg -n 'RebuildKind|RebuildTelemetryReason|RebuildTelemetryClass|RebuildTelemetryPolicy' src/audioengine --type cpp --type h
rg -n 'RebuildKind' src --include='*.h' --type cpp --type h
rg -n 'AudioEngine::submitRebuildIntent|void.*submitRebuildIntent' src --type cpp --type h
rg -n 'rebuildRequestGeneration|rebuildThread' src --type cpp --type h
rg -n 'prepareToPlay\s*\(|AudioEngine::AudioEngine\s*\(|AudioEngine::~AudioEngine' src/audioengine --type cpp --type h
rg -n 'BuildErrorPolicy|RetryDisposition|WorkerThread' src --type cpp --type h
rg -n 'BuildError|RetryDisposition|rebuildRequestGeneration|Recovery lease|kMaxRecovery|attempt|backoff|jitter|maxAttempts|diagLog|emitRebuildTelemetry' src --type cpp --type h
# ag
ag -n 'RebuildKind' src/audioengine
ag -n 'RetryScheduler|RetryScheduleRequest|PendingRetry' src
# fdfind / fzf / sed / awk
fdfind -e h -e cpp 'BuildError|RebuildDispatch|AudioEngine\.h' .
fdfind -e h . src/audioengine | fzf --filter='BuildError'
sed -n '78,85p' src/audioengine/AudioEngine.RebuildDispatch.cpp
sed -n '149,220p' src/audioengine/AudioEngine.RebuildDispatch.cpp
awk '/submitRebuildIntent|rebuildRequestGeneration|isObsolete|shouldRetryWarmupFailure/{print FILENAME":"NR": "$0"}' src/audioengine/AudioEngine.RebuildDispatch.cpp
# ast-grep
sg run -p 'submitRebuildIntent($$$)' --lang cpp src/
sg run -p 'RebuildKind' --lang cpp src/
sg run -p 'BuildError' --lang cpp src/
# cocoindex
ccc status; ccc grep 'RebuildKind'; ccc grep 'submitRebuildIntent'; ccc grep 'RetryDisposition'
# graphify
graphify query 'RebuildKind'; graphify query 'submitRebuildIntent'; graphify query 'AudioEngine'
# semble
semble search 'RebuildKind' . --max-snippet-lines 5; semble search 'submitRebuildIntent' . --max-snippet-lines 5
# AiDex
# aidex_query term="BuildError" mode="contains"
# serena
# .serena/project.yml
# context-mode
ctx_execute(language: "javascript", code: "fs.readFileSync('src/audioengine/AudioEngine.h').split('\n').filter(l=>l.includes('RebuildKind')).join('\n')")
# RTK(WSL)
wsl bash -c 'cd /mnt/c/VSC_Project/ConvoPeq && ~/.local/bin/rtk grep -rn "RebuildKind" src/'
# headroom
# 大きな断片は context-mode で仮想化
```

## 参照

* `src/audioengine/AudioEngine.h` — 4 enum / `2506 rebuildRequestGeneration` / `2651 rebuildThread` / `4763 rebuildAdmissionIntentMutex_`
* `src/audioengine/AudioEngine.CtorDtor.cpp` — constructor / destructor / shutdown ordering
* `src/audioengine/AudioEngine.RebuildDispatch.cpp:149` — submitRebuildIntent（void 4引数 noexcept）/ `78 shouldRetryWarmupFailure`
* `src/audioengine/BuildErrorPolicy.h` — 8値 default policy
* `src/core/WorkerThread.h/.cpp` — generic background thread
* `evidence/D101-18-Phase-D-5-2-1-RetryScheduler-Minimal-Design-Audit.md` — 20項目 PASS、API 凍結
* `evidence/D101-17-Phase-D-5-2-0-PreImplementation-Contract-Audit.md` — 12項目 PASS、lifecycle 凍結
* `ConvoPeq.md` — 最新ソーススナップショット
