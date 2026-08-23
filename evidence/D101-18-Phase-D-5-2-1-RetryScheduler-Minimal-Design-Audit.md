# D101-18 Phase D-5-2-1 — RetryScheduler Minimal Design / Type & Lifecycle Freeze Audit

**Date:** 2026-08-23
**Branch:** main
**Scope:** D-5-2 実装前の最小設計確定（**audit-only、コード変更0**）
**Prerequisite:** D101-17 D-5-2-0 GO（12項目 PASS）、D101-16b D-5-1.1 CLOSED（案A 採用、attempt 削除）、D101-13 D-3 CLOSED
**Primary source:** 実ワークツリー（`src/audioengine/AudioEngine.h`、`src/audioengine/AudioEngine.CtorDtor.cpp`、`src/audioengine/AudioEngine.RebuildDispatch.cpp:149` `submitRebuildIntent`、`src/audioengine/BuildErrorPolicy.h`、`src/core/WorkerThread.h/.cpp`）

## 手法 — 全ツール

| 系統 | ツール | 実行内容 | 結果 |
|------|--------|----------|------|
| WSL | `rg` | 8コマンド再実行：4 enum scope / include graph / lifecycle start / thread start / schedule noexcept / generation / diagLog / AudioEngine.h enum block | D-5-1.1と一致 |
| WSL | `ag` | `ag -n 'RebuildKind' src` / `ag -n 'noexcept' src` | `rg`と一致 |
| WSL | `fdfind`/`fzf`/`sed`/`awk` | `fdfind -e h 'AudioEngine'` / `sed -n '1,80p' AudioEngine.h` / `awk '/noexcept\|try\|catch/'` | enum block / noexcept 抽出 |
| WSL | `sg` | `sg run -p 'enum class RebuildKind'` / `sg run -p 'noexcept'` | 捕捉 |
| MCP | `serena` | `.serena/project.yml` | 索引正常 |
| CLI | `cocoindex` `ccc status` / `ccc grep` | `RebuildKind` / `noexcept` | 一致 |
| CLI | `graphify query 'AudioEngine'` | AudioEngine community | 確認 |
| CLI | `semble search 'RebuildKind'` | hit あり | 一致 |
| MCP | `AiDex` | BuildError 45 hits | 集約確認 |
| sandbox | `context-mode ctx_execute` | `AudioEngine.h` 4 enum / private members / CtorDtor lifecycle | WSLと一致 |
| WSL | `RTK` | `rtk grep` fallback | 差異0 |
| MCP | `headroom` | context-mode仮想化 | 遵守 |

全ツールで一致。

## A. 型依存 — PASS（条件付き）

**実測:**

* 4 enum は `src/audioengine/AudioEngine.h` 内で定義：

  * `enum class RebuildKind`（`None`/`Runtime`/`Structural`）— `namespace convo` 内、AudioEngine スコープ外だが AudioEngine.h に同居
  * `enum class RebuildTelemetryReason`（`RebuildThreadWarmupRetry` 等 20+値）
  * `enum class RebuildTelemetryClass`（`Structural` / `Snapshot` 等）
  * `enum class RebuildTelemetryPolicy`（`Replaceable` / `MustExecute` / `NA`）
* 既存 standalone header は存在しない（`fdfind -e h RebuildTelemetry` 0、`grep -rn 'RebuildTelemetry' src --include='*.h' \| grep enum` は AudioEngine.h のみ）。
* `RetryScheduler.h` が `AudioEngine.h` を include すると、`AudioEngine.h → RetryScheduler.h → AudioEngine.h` の循環（`AudioEngine` が `unique_ptr<RetryScheduler>` を所有）が発生。

**凍結構造（推奨）:**

```
RetrySchedulerTypes.h          // 新設・軽量ヘッダ（4 enum のみ、JUCE非依存）
    └── 4 enum (RebuildKind / Reason / Class / Policy)

AudioEngine.h
    ├── RetrySchedulerTypes.h
    └── class RetryScheduler; // forward declare

RetryScheduler.h
    └── RetrySchedulerTypes.h  // 4 enum を得る、AudioEngine.h は含まない

RetryScheduler.cpp
    ├── RetryScheduler.h
    └── AudioEngine.h          // 完全定義（submitRebuildIntent 呼出しのため）
```

AudioEngine 本体を `RetryScheduler.h` が include しない方向を維持。既存構造より安全で、D-5-0 §12 の `#include` 循環懸念を解消。**ただし新規型ヘッダを増やさない選択も可** — その場合は `RetryScheduler.h` が `AudioEngine.h` を include し、`AudioEngine.h` 側は `class RetryScheduler;` forward declare + `.cpp` で完全定義を得る構造で同じ循環回避が可能。D-5-2-1 では **軽量ヘッダ新設を第一候補としつつ、AudioEngine.h 直includeでも循環は回避可能** として PASS。

**PASS**

## B. Lifecycle 開始地点 — PASS

**実測:**

* `AudioEngine::AudioEngine()` — メンバ初期化（`rebuildRequestGeneration{0}` / `rebuildThread` / `rebuildMutex` / `rebuildAdmissionIntentMutex_`）
* `AudioEngine::prepareToPlay(double sr, int blockSize)` — `currentSampleRate` / `maxSamplesPerBlock` を `publishAtomic`、rebuild 準備
* `AudioEngine::initialise` 系は `prepareToPlay` 経由
* `AudioEngine::~AudioEngine()` — `rebuildThreadShouldExit = true` → `rebuildThread.join()` → `WorkerThread` 停止（`AudioEngine.CtorDtor.cpp:1-200`）

**凍結:**

```
constructor
  ↓
RetryScheduler object construction (unique_ptr, thread not yet started)
  ↓
exact start point = AudioEngine::prepareToPlay 後（または明示的 retryScheduler_->start()）
  ↓ start() 内で std::thread(&RetryScheduler::run, this) を生成
normal schedule()
  ↓
~AudioEngine
  ↓
retryScheduler_->shutdown()  // explicit、rebuildThreadShouldExit より先
  ↓
rebuildThreadShouldExit = true → rebuildThread.join()
  ↓
WorkerThread / CoordinatorLoop shutdown
  ↓
members destroyed（逆順）
```

`start()` が不要なら作らない — しかし `std::thread` を constructor で開始すると `currentSampleRate` 等が未初期化のまま `submitRebuildIntent` が呼ばれる危険があるため、**`start()` を分離する**。`start()` は `prepareToPlay` から呼ぶか、`AudioEngine` 初期化完了後に1回だけ呼ぶ。

**PASS**

## C. schedule() 戻り値 — PASS（void 維持）

**候補2択:**

* `void schedule(...) noexcept` — 呼び出し側は成功/拒否を再利用しない
* `bool schedule(...) noexcept` — 呼び出し側が overflow/shutdown を知る

**凍結: `void` 維持**（推奨通り）

理由：scheduler は admission authority ではなく admitted delayed intent の executor。`schedule()` が `bool` を返し caller が再分類（例：`if (!schedule(...)) retry differently`）すると scheduler 境界が再び広がる。reject 時は内部 `diagLog` / `publicationRejectCount_` 相当の local counter で telemetry し、caller へ retry policy を返さない。

```
schedule() = enqueue attempt を実行
reject時 = 内部 telemetry/counter（caller へ policy を返さない）
```

**PASS**

## D. diagLog 扱い — PASS

**実測:** `RebuildDispatch.cpp` の overflow/shutdown は `diagLog("[DIAG] ...")` + `fetchAddAtomic(publicationRejectCount_)` + `emitRebuildTelemetry(Suppressed, ...)`。`diagLog` は `AudioEngine` の huge 実装依存（`DBG` + `Logger::writeToLog`）。

**凍結:**

* **第一候補：RetryScheduler 内部の local atomic counter**（`std::atomic<uint64_t> retrySchedulerRejectCount_{0}`）+ 必要なら `DBG` のみ。`AudioEngine::diagLog()` を直接呼ばない。
* 既存 `publicationRejectCount_` / `rebuildCollapseCount_` を流用しない（scheduler は AudioEngine.h を include しないため）。
* `emitRebuildTelemetry` も scheduler から呼ばない（telemetry は `submitRebuildIntent` 内で既に行われる）。

**避ける:** `RetryScheduler → AudioEngine::diagLog()`（scheduler が AudioEngine の巨大実装依存を持つ）

**PASS**

## E. 最終 D-5-2 API — 凍結（コードはまだ作らない）

```cpp
// RetrySchedulerTypes.h — 4 enum（AudioEngine.h から抽出または新設）
enum class RebuildKind;
enum class RebuildTelemetryReason;
enum class RebuildTelemetryClass;
enum class RebuildTelemetryPolicy;

// RetryScheduler.h
struct RetryScheduleRequest {
    convo::RebuildKind kind;
    RebuildTelemetryReason reason;
    RebuildTelemetryClass rebuildClass;
    RebuildTelemetryPolicy collapsePolicy;
    // generation なし / RetryDisposition なし / attempt なし
};

class RetryScheduler {
public:
    explicit RetryScheduler(AudioEngine& engine) noexcept;
    ~RetryScheduler(); // noexcept（shutdown() を呼ぶ）

    void start(); // not noexcept（std::thread may throw）— D-5-2 で確定
    void schedule(RetryScheduleRequest request, std::chrono::milliseconds delay) noexcept;
    void shutdown() noexcept; // idempotent: CAS + joinable()
    [[nodiscard]] std::size_t pendingCount() const noexcept;

private:
    struct PendingRetry {
        RetryScheduleRequest request;
        std::chrono::steady_clock::time_point deadline;
    };
    static constexpr std::size_t kCapacity = 8;
    void run(); // wait_until(deadline) loop

    AudioEngine* engine_ = nullptr; // non-owning, valid until join
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<PendingRetry> queue_; // deadline ordering（upper_bound insert）
    std::atomic<bool> shouldExit_{false};
    std::thread thread_;
};
```

**最終確認（9項目）:**

| 項目 | 判定 |
|------|------|
| 型依存 | RetrySchedulerTypes.h で AudioEngine.h 循環なし — PASS |
| constructor noexcept | `RetryScheduler(AudioEngine&) noexcept` — thread未開始のため `noexcept` 安全 — PASS |
| destructor noexcept | `~RetryScheduler()` は `noexcept`（`shutdown()` が `noexcept`）— PASS |
| thread start を constructor に含めるか | **含めない** — `start()` 分離（prepareToPlay後）— PASS |
| engine_ lifetime | non-owning raw ptr、`shutdown()->join()` 前のみ有効 — PASS |
| telemetry/error counter | local atomic counter（AudioEngine::diagLog 依存なし）— PASS |
| schedule() allocation failure | `try { insert } catch (...) { reject + counter; return; }` — PASS |
| shutdown/schedule 同時実行 | mutex + shouldExit + notify_all — PASS |
| pendingCount() shutdown race | `lock_guard` 下で `queue_.size()` — PASS |

## D-5-2-1 GO 条件（20項目）

```
[x] 4 enum の scope / include graph が確定              — AudioEngine.h 内、RetrySchedulerTypes.h 新設で循環回避
[x] RetryScheduler.h → AudioEngine.h 循環なし            — Types.h 経由、forward declare + .cpp include
[x] RetryScheduler.cpp → AudioEngine.h の依存が明確      — submitRebuildIntent 呼出しのため完全定義が必要
[x] scheduler thread の正確な start point が確定          — prepareToPlay後の explicit start()
[x] destructor の shutdown → join 順序が確定              — explicit retryScheduler_->shutdown() → rebuildThread join
[x] schedule/shutdown mutex protocol が確定               — mutex + shouldExit + notify_all/notify_one
[x] capacity=8 が固定                                    — 固定、configurableにしない
[x] reject-newest が固定                                 — size==8 で既存8件不変
[x] deadline ordering が固定                              — deque + upper_bound insert、priority_queue不採用
[x] PendingRetry に attempt が存在しない                  — {request, deadline} 2要素のみ
[x] scheduler が BuildError を参照しない                  — #include BuildErrorPolicy.h なし（T10）
[x] scheduler が RetryDisposition を参照しない            — delay値のみ（T10）
[x] scheduler が generation を参照しない                  — rebuildRequestGeneration なし（T11）
[x] scheduler が Recovery lease を参照しない              — settle/kMaxRecovery なし（T12）
[x] schedule() allocation failure の扱いが確定            — try/catch → reject + counter
[x] diagLog/telemetry の依存方法が確定                   — local atomic counter、AudioEngine::diagLog 依存なし
[x] Audio Thread producer = 0                             — processBlock から schedule 0（T13）
[x] submitRebuildIntent は scheduler thread の expiry 時だけ — run() 内 wait_until → submit
[x] schedule() caller が submitRebuildIntent を直接呼ばない — schedule() は enqueue → scheduler thread が submit
[x] D-5-2 production code change = まだ0                  — audit-only
```

**D-5-2-1 GO — 全20項目 PASS**

次は **D-5-2 RetryScheduler.h/.cpp 最小実装**へ進行可能。実装時は上記 API のまま `PendingRetry {request, deadline}` 2要素、attempt/backoff/jitter/maxAttempts は Scheduler に持たせない。

## 参照

* `src/audioengine/AudioEngine.h` — 4 enum / `2506 rebuildRequestGeneration` / `2651 rebuildThread` / `4763 rebuildAdmissionIntentMutex_`
* `src/audioengine/AudioEngine.CtorDtor.cpp` — constructor / destructor / shutdown ordering
* `src/audioengine/AudioEngine.RebuildDispatch.cpp:149` — submitRebuildIntent（void fire-and-forget、4引数、noexcept）
* `src/audioengine/AudioEngine.RebuildDispatch.cpp:78` — shouldRetryWarmupFailure（isLoadingIR のみ）
* `src/audioengine/BuildErrorPolicy.h` — 8値 default policy
* `evidence/D101-17-Phase-D-5-2-0-PreImplementation-Contract-Audit.md` — 12項目全PASS
* `evidence/D101-16b-Phase-D-5-1-1-Semantic-Boundary-Audit.md` — C1〜C5 PASS（案A採用）
* `ConvoPeq.md` — 最新ソーススナップショット
