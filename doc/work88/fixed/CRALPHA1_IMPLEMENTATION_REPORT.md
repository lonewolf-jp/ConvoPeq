# CR-α-1 — Site 3 Bounded Warmup Retry Implementation（Work Report）

```text
CR-α-1 — Site 3 Bounded Warmup Retry Implementation

Production source: 2 files changed（BuildErrorPolicy.h +81/−0・AudioEngine.RebuildDispatch.cpp +67/−17）
Test source: 1 file changed（BuildErrorClassificationTests.cpp +87/−2）
CMake: 0
Build: NOT RUN（CR-α-3 で実施）
CTest: NOT RUN（CR-α-4 で実施）
stress: 0

baseline:
ConvoPeq.md Generated 2026-09-01 19:31:33（実装時照合・FRESH）
→ 実装後 FRESH 2026-09-01 20:32:43（CR-α-2 用・NEWER_SRC_COUNT=0・収録 21 hits 確認）
```

## 総合判定

> ## **CR-α-1 完了 — Anchor Audit 変更 map どおりの 3 ファイル実装 / forbidden changes 0 / CR-α-2 へ進行可**

---

## Step 1: BuildErrorPolicy.h（+81 行・純追加）

namespace `convo` 閉じ直前に追加（**純 constexpr / noexcept / JUCE・Scheduler・Logger 非依存**）:

| 要素 | 内容 |
|---|---|
| `kMaxWarmupConsecutiveRetries = 3` | Site 3 専用 bound。**recovery K=4 とは別ドメイン・別値**（混同検出可能化の意図をコメント明記・ISRRuntimePublicationCoordinator.h:401 を参照先として記載） |
| `RetryBackoffPolicy { initialDelayMs=10, maxDelayMs=80, multiplier=2 }` | tuning parameter（H.11.27.6 — invariant は NonRT / non-blocking / bounded / caller-side counter と分離） |
| `kDefaultWarmupRetryBackoff {10, 80, 2}` | normative default（ND-07 §4 確定値） |
| `retryBackoffDelayMs(policy, attempt)` | attempt 0→0 / 1→10 / 2→20 / 3→40 / 4→80 / 5+→80（saturation: uint64 中間 + 上限早期 return で overflow 構造排除） |
| `WarmupRetryAction` | `Schedule / Exhausted / NoRetry` |
| `WarmupRetryDecision { action, delayMs }` | decision 結果 |
| `warmupRetryDecision(attempt, maxRetries, contextRetryable, obsolete, disposition, policy)` | 純関数。条件 = `contextRetryable AND !obsolete AND attempt <= maxRetries AND disposition != NoRetry`。**RetryImmediate は無条件 retry ではない**（delay 0 の bounded retry）。delay mapping: Immediate→0 / Backoff→表 / NoRetry→action NoRetry |

## Step 2-4: AudioEngine.RebuildDispatch.cpp（+67/−17）

1. **counter 宣言**（`while (true)` 直前・rebuildThreadLoop 関数スコープ）: `warmupRetryBoundGeneration = -1`（sentinel）・`warmupRetryCount = 0`（**warmup failure 回数**）・`warmupRetryExhausted = false`。RebuildThread 単一所有・mutex/atomic 不要。
2. **generation rebind**（`!task.runtimeBuildSnapshot.sealed continue` の直後）: `task.generation != warmupRetryBoundGeneration` → count=0 / exhausted=false / bound 更新。retry budget は generation ごとに独立（exhaustion は永続 mask ではない）。
3. **warmup failure block 置換**（旧 :1232-1265）: `++warmupRetryCount` → `classifyBuildError(warmupError)` → `warmupRetryDecision(count, kMax=3, retryable, isObsolete(), outcome.retry, kDefaultWarmupRetryBackoff)` → decision で制御:
   - **Schedule**: diagLog（generation/attempt/limit/delayMs/error）+ `schedule(req, ms(decision.delayMs))`（fallback: scheduler nullptr 時のみ submitRebuildIntent 直）
   - **Exhausted**: `warmupRetryExhausted` transition 時 1 回のみ terminal diagLog（attempts=3 + limit + error + no-further-retry 注記）— 以後同 generation では telemetry なし
   - **NoRetry**: nothing
4. counter 意味論（指示 Step 4 厳密化）: counter は **failure 回数**（schedule 前 ++）。counter 1..3 = retry #1..#3 / 4 回目 failure = Exhausted / 5 回目以後 = nothing。`schedule()` は void のため reject 時は attempt 消費の retry drop（巻き戻し処理は実装しない — ND-07 §7）。

## Step 6: BuildErrorClassificationTests.cpp（+87/−2）

- `runTestF()` 追加（**Scheduler 本体を呼ばない純関数テスト**）: T-CRα-1（delay table 0/10/20/40/80/80/100→80 + inverted policy cap）/ T-CRα-2（default {10,80,2} + max=3）/ T-CRα-3（context false → NoRetry・obsolete → NoRetry・attempt 4/5 → Exhausted）/ T-CRα-4（Immediate→0 / Backoff→10,20）。
- `main()` の and 結合に runTestF を接続（`a&&b&&c&&d&&e&&f`）。
- T-CRα-5/6（exhaustion 再発行なしの source inspection・diagLog 項目）は CR-α-2 の read-only verification で実施（本ファイルは純関数のみ — 設計目的どおり）。

## forbidden changes 確認（diff audit 実測）

| 項目 | 実測 |
|---|---|
| RetrySchedulerTypes.h / AudioEngine.h | **diff 0**（F-1 どおり telemetry enum 追加なし） |
| RetryScheduler.h / RetryScheduler.cpp | **diff 0**（infrastructure 無変更・delay 引数は既存） |
| RuntimeStore.h / Coordinator.h | **diff 0** |
| CMakeLists.txt（両） | **diff 0** |
| Recovery K=4 / Site 1/2 への接触 | **0**（新規 diff 内に `kMaxRecoveryConsecutiveFailures` / `settlePendingRecoveryAdmission` / `postRecoveryFailureSignal` の追加 0 件） |
| Site 3 build failure path（runtime == nullptr） | **無変更**（retry 化は将来拡張保留） |
| その他の変更 | RuntimeWorldAuthority.h（CW-8・ND-03/04 済分）+ ISRSemanticValidationTests.cpp（同）+ output_sourcecode_markdown.py（D160 済分）— 本ターンの新規変更は 3 ファイルのみ |

## snapshot

- 実装後再生成: **`Generated: 2026-09-01 20:32:43`・FRESH / NEWER_SRC_COUNT=0**（収録: `warmupRetryDecision` / `kMaxWarmupConsecutiveRetries` 21 hits）。

## CR-α-2 への引き継ぎ（read-only verification 項目）

1. counter owner = RebuildThread（関数スコープ・mutex/atomic なし）
2. generation reset = 正しい（rebind 位置・sentinel -1）
3. K=3 ≠ Site 2 K=4（別名別値）
4. NoRetry path = no schedule
5. Exhausted = one-shot（exhausted flag）
6. Schedule = decision.delayMs（0 = Immediate / policy = Backoff）
7. fallback = Schedule action のみ（scheduler nullptr 時）
8. RetrySchedulerTypes.h unchanged / AudioEngine.h unchanged
9. T-CRα-5: exhaustion 分岐内に submitRebuildIntent が無いこと（source inspection）
10. T-CRα-6: diagLog に generation/attempt/limit/exhausted/error が含まれること

→ 次工程: **CR-α-2 read-only implementation verification**（指示どおり build 前に実施）。
