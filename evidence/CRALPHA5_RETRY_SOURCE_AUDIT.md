# CR-α-5 Retry-Specific Source Audit — evidence（read-only 実測記録）

Date: 2026-09-01 / baseline: ConvoPeq.md Generated 2026-09-01 21:47:45（FRESH・NEWER_SRC_COUNT=0・CHECK_EXIT=0）
対象: src/audioengine/AudioEngine.RebuildDispatch.cpp（Site 3）+ src/audioengine/BuildErrorPolicy.h
変更: 0（production 0 / test 0 / CMake 0 / Build 0 / CTest 0 / stress 0）

## E-01 Counter ownership（V-α5-01）

- 宣言: RebuildDispatch.cpp:833-835 — `rebuildThreadLoop()`（:824 開始）の関数スコープ、
  `while (true)`（:846）**外側**・try ブロック（:848）外側。
  `int warmupRetryBoundGeneration = -1; / std::uint32_t warmupRetryCount = 0; / bool warmupRetryExhausted = false;`
- 全参照 14 箇所: :833-835（宣言）・:1177-1181（rebind）・:1272（++）・:1274-1275（decision）・
  :1289（Schedule log）・:1305（guard）・:1308（write）・:1311（Exhausted log）— **同ファイル同関数のみ**。
- src/ 全域 grep `warmupRetry`: RebuildDispatch.cpp 以外 = 0 件（BuildErrorPolicy.h は policy 定義のみ・
  BuildErrorClassificationTests.cpp は test のみ・AudioEngine.h / coordinator / scheduler = 0 件）。
- atomic 化 = 0 件・mutex 共有 = 0 件（grep 実測）。
- → **RebuildThread 単一所有がコード構造で証明**（rebuildThreadLoop 専用スレッドのスタック局所変数）。

## E-02 Generation rebind（V-α5-02）

- 順序: obsolete check（:1161-1169）→ sealed check（:1171-1172）→ **rebind（:1177-1182）**:
  ```cpp
  if (task.generation != warmupRetryBoundGeneration) {
      warmupRetryBoundGeneration = task.generation;
      warmupRetryCount = 0;
      warmupRetryExhausted = false;
  }
  ```
  → generation 変化で counter/exhausted とも reset・**exhaustion は次 generation に漏れない**。
- sentinel 衝突なし: `rebuildRequestGeneration` = `std::atomic<int>` 初期値 0（AudioEngine.h:2553）、
  更新は `++rebuildRequestGeneration`（RebuildDispatch.cpp:655）と
  `fetchAddAtomic(rebuildRequestGeneration, 1, ...)`（CtorDtor.cpp:143）のみ = **0 起点単調増加・非負**。
  `task.generation` は `int generation = 0`（AudioEngine.h:2740）。`-1` は実 generation 域と非交差。

## E-03 Failure #4 boundary（V-α5-03）

- `++warmupRetryCount`（:1272）は decision（:1274-1277）**の前**。
- 境界: BuildErrorPolicy.h:155-156 `if (attempt > maxRetries) return Exhausted` —
  attempt = ++ 済み failure 回数（header :144 コメント「この failure で ++ 済みの値」）。
- counter 意味論コメント: :834「1..3 = retry #1..#3 / 4 = exhausted」・header :129
  「counter 1..max → retry #1..#max / counter max+1 回目の failure → Exhausted」。

## E-04 Exhausted branch 全文（V-α5-04）

:1304-1315 — `else if (decision.action == Exhausted && !warmupRetryExhausted)` 本体 =
`warmupRetryExhausted = true;`（:1308）+ diagLog（:1309-1314）**のみ**。
- `submitRebuildIntent` 出現箇所（RebuildDispatch.cpp 全域 grep）: **:151（定義）・:459（requestRebuild
  入口経路 — Site 3 retry 無関係）・:1298（Schedule 分岐 scheduler-null fallback のみ）**。
  Exhausted branch 内 = **0**。
- `RetryScheduleRequest` 生成: ファイル内 :1281 の 1 箇所のみ（Schedule 分岐内）。
- schedule 呼び出し: :1294 の 1 箇所のみ。

## E-05 NoRetry path（V-α5-05）

- policy（BuildErrorPolicy.h:152-153）:
  `if (disposition == RetryDisposition::NoRetry || !contextRetryable || obsolete) return { NoRetry, 0 };`
  → NoRetry / !contextRetryable（shouldRetryWarmupFailure = isLoadingIR・:1254）/ obsolete の
  いずれも先頭で NoRetry 返却。
- caller: NoRetry は `if (Schedule)`（:1279）・`else if (Exhausted)`（:1304）の**いずれにも不該当** →
  :1316 コメント「NoRetry / exhausted 重複後は telemetry なし（nothing）」→ :1318 `continue`。
  schedule（:1293-1294）・fallback（:1295-1302）とも到達不能。

## E-06 Immediate boundedness（V-α5-06）

- Exhausted 判定（header :155-156）は delay 計算（:157-159）**より先行** →
  RetryImmediate でも attempt > 3 なら Exhausted（bound bypass 構造的に不存在）。
- delay: `disposition == RetryBackoff ? retryBackoffDelayMs(policy, attempt) : 0`（:157-159）→
  Immediate = delay 0 の **bounded** retry（header :127 コメント明示）。
- 即時性の遅延ゼロは schedule 時点のみで、K=3 exhaustion 打ち切りは disposition 非依存で適用。

## E-07 Scheduler reject 契約（V-α5-07）

- `void schedule(RetryScheduleRequest request, ...)`（RetryScheduler.h:40）= **void 返却**。
- reject（capacity 満杯 / shutdown）時も counter は ++ 済みのまま・rollback 経路なし
  （++ は :1272 の 1 箇所・減算/リセットは rebind :1180 のみ）。
- コメント: :1269-1270「schedule() は void: reject 時は attempt 消費の retry drop として扱う
  （巻き戻ししない — ND-07 §7 契約）」。

## E-08 Exhausted one-shot（V-α5-08）

- guard 条件: `decision.action == Exhausted && !warmupRetryExhausted`（:1304-1305）。
- `warmupRetryExhausted = true` の write: **:1308 の 1 箇所のみ**（grep 実測）。
  他 = 宣言 :835（false 初期化）・rebind :1181（false reset）・read :1305・log :1311 は count 側。
- failure #4 → log 発火 + flag set → failure #5+ は `!warmupRetryExhausted` 不成立 → telemetry なし。
- rebind（:1180-1181）で次 generation は exhausted=false → V-α5-02 と cross-check 一致。

## E-09 Exhausted telemetry 意味論（V-α5-09）

- log: :1309-1314 `generation / attempts / limit / error / "(no further retry until generation changes)"`。
- `attempts = warmupRetryCount - 1`（:1311）: failure #4 時点で warmupRetryCount=4 → **attempts=3**。
- 契約突合: ND-07 :147「diagLog（generation・error・**attempts=3**）」— **実装 = 契約どおり一致**。
  意味論: attempts = schedule 済み retry 消費数（#1..#3 全失敗 → 4 回目の failure で exhaustion 検出）
  = K=3 上限値。failure 回数（4）でも retry ordinal（#4）でもない契約値 3 を出力。**矛盾なし**。
- Schedule log の `attempt=warmupRetryCount`（:1289）= retry ordinal #1..#3 — header :154
  「1..maxRetries = retry #1..#maxRetries」定義と一致（同一変数の文脈別意味論は policy 契約に明記済み）。
- 注記（既決偏差・新規発見ではない）: ND-07 :147 の「enum 追加 1 値」は CR-α-1 anchor audit F-1 で
  取りやめ決定済み（telemetry enum は AudioEngine.h toString switch に依存 → **diagLog-only terminal
  telemetry** が CR-α-1 で ratify・AudioEngine.h diff 0 維持と整合）。

## E-10 K domain separation（V-α5-10）

```text
kMaxWarmupConsecutiveRetries       = 3  （BuildErrorPolicy.h:92・Site 3 専用コメント :87-91）
  使用: RebuildDispatch :1275 / :1290 / :1312（Site 3 のみ）
kMaxRecoveryConsecutiveFailures    = 4  （RebuildDispatch :1076 Builder-local constexpr・Site 1/2）
  使用: :1102 / :1126（recoveryConsecutiveFailures）
kMaxObligationConsecutiveFailures  = 4  （ISRRuntimePublicationCoordinator.h:401 uint8_t）
  使用: ISRRuntimePublicationCoordinator.cpp:1090 / :1130（obligation-level）
```
- 3 ドメイン: Site 3 warmup K=3 / Site 1/2 Builder recovery K=4 / obligation K=4 —
  別定数・別 counter・別 linearization。`kMaxWarmupConsecutiveRetries` を Site 1/2 が参照 = 0 件、
  逆も 0 件（header :87-91「意図的に別値（3 vs 4）にすることで混同を検出可能にする」）。
- coordinator h:397 コメントも Builder-local K=4 を「別 counter」と明記。

## E-11 Forbidden footprint（V-α5-11）

`git diff --name-only` 実測（CR-α 対象の全 forbidden ファイル不在）:
RetrySchedulerTypes.h / AudioEngine.h / RetryScheduler.h / RetryScheduler.cpp /
RuntimeStore.h / Coordinator.h / CMakeLists.txt = **すべて diff 0**。
CR-α 新規変更 = BuildErrorPolicy.h（+81/−0）・AudioEngine.RebuildDispatch.cpp（+67/−17）・
BuildErrorClassificationTests.cpp（+88/−2 = α-1 +87/−2 + α-3R +1/−0）の 3 ファイルのみ。
ND 残留（RuntimeWorldAuthority.h・ISRSemanticValidationTests.cpp・運用 3 ファイル）は CR-α 変更ではない。

## E-12 test 側の裏付け（CR-α-4 で実行済み）

T-CRα-1〜4（BuildErrorClassificationTests.cpp runTestF・checks=86 fails=0 × Debug/Release standalone）:
delay table 0/10/20/40/80/80・saturation・{10,80,2}・K=3・decision truth table（!context→NoRetry /
obsolete→NoRetry / NoRetry→NoRetry / #4→Exhausted / #5→Exhausted）・Immediate delay 0・Backoff 表 —
本 audit の E-03〜E-06 の構造証明と実行時証明が一致。
