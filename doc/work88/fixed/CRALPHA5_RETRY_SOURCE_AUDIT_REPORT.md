# CR-α-5 — Retry-Specific Source Audit（read-only Work Report）

```text
CR-α-5 — Retry-Specific Source Audit

Date: 2026-09-01
baseline: ConvoPeq.md Generated 2026-09-01 21:47:45（--check 実測 FRESH / NEWER_SRC_COUNT=0 / CHECK_EXIT=0）
Production source: 0（完全 read-only・source modification 禁止遵守）
Test source: 0
CMake: 0
Build: 0（実施不要どおり）
CTest: 0（実施不要どおり）
stress: 0
```

## 総合判定

> ## **CR-α-5 = PASS（V-α5-01〜12 全 12 項目 PASS）— Closure（最終閉鎖判定）へ進行可**
>
> 特に V-α5-03（failure #4 boundary）: counter は decision 前に ++ され、境界
> `attempt > kMaxWarmupConsecutiveRetries(=3)` により **failure #4 = Exhausted・retry なし**。
> 「K=3 だから 4 回目まで retry」の読み違いは構造的に成立しない（K=3 = schedule 可能な retry 数、
> counter 4 回目の failure で打ち切り）。V-α5-09 の `attempts = warmupRetryCount - 1` は
> failure #4 時点で **attempts=3** となり ND-07 :147 契約原文（attempts=3）と**一致** — 矛盾なし。
> BLOCK 条件（BLOCKER）は 1 件も発生せず。

---

## V-α5-01〜12 判定表

| ID | Verification | 判定 | 実測根拠（行番号は現行ソース・ConvoPeq.md 21:47:45 基準） |
|---|---|---|---|
| **V-α5-01** | Counter ownership = RebuildThread 単一 | **PASS** | 3 変数（:833-835）は `rebuildThreadLoop()`（:824）関数スコープ・`while (true)`（:846）**外側**・try（:848）外。全参照 14 箇所 = 同関数内のみ（src 全域 grep: RebuildDispatch.cpp 以外 0 件 — AudioEngine.h / coordinator / scheduler 0）。atomic 化 0・mutex 共有 0。**RebuildThread 専用スレッドのスタック局所変数であることがコード構造から証明** |
| **V-α5-02** | Generation rebind | **PASS** | obsolete check（:1161-1169）→ sealed check（:1171-1172）**後**に rebind（:1177-1182）: generation 比較 → count=0 / exhausted=false / bound 更新。exhaustion は次 generation に漏れない。sentinel `-1` 衝突なし: `rebuildRequestGeneration` は atomic\<int\> 初期値 0（AudioEngine.h:2553）で `++`（RebuildDispatch:655）/ `fetchAddAtomic(…,1,…)`（CtorDtor:143）のみ = **0 起点単調増加・非負**。task.generation も `int generation = 0`（AudioEngine.h:2740） |
| **V-α5-03** | Failure #4 boundary（最重要） | **PASS** | `++warmupRetryCount`（:1272）は decision（:1274-1277）**の前**。境界 = header :155-156 `attempt > maxRetries`（attempt = ++ 済み failure 回数 — header :144 明記）。truth table 下記 §V-α5-03。**K=3 = schedule 可能 retry 数であり「4 回目まで retry」ではない**: failure #1→retry#1 / #2→#2 / #3→#3 / **failure #4 → Exhausted・retry なし** |
| **V-α5-04** | Exhausted 再発行禁止 | **PASS** | Exhausted 分岐（:1304-1315）本体 = flag set（:1308）+ diagLog のみ。`submitRebuildIntent` 出現列挙（grep 実測）: **:151（定義）・:459（requestRebuild 入口 — Site 3 retry 無関係）・:1298（Schedule 分岐の scheduler-null fallback）** → Exhausted branch 内 = **0**。`RetryScheduleRequest` 生成 = :1281 の 1 箇所のみ（Schedule 分岐内）。schedule 呼び出し = :1294 の 1 箇所のみ。間接再発行経路なし |
| **V-α5-05** | NoRetry path | **PASS** | policy（header :152-153）: `NoRetry \|\| !contextRetryable \|\| obsolete` → 先頭で `{NoRetry, 0}`。caller 側: NoRetry は `if (Schedule)`（:1279）/ `else if (Exhausted && !flag)`（:1304）の**いずれにも不該当** → :1316 コメント（nothing）→ :1318 `continue`。**NoRetry → no schedule → no fallback submitRebuildIntent** の両側（policy + caller）で成立 |
| **V-α5-06** | Immediate boundedness | **PASS** | Exhausted 判定（header :155-156）が delay 計算（:157-159）**より先行** → RetryImmediate でも attempt > 3 なら Exhausted（bypass 構造的不存在）。`RetryBackoff ? retryBackoffDelayMs(...) : 0`（:157-159）→ **Immediate = zero-delay bounded retry**（header :127 明示）。Immediate ≠ unbounded retry をコードから証明 |
| **V-α5-07** | Scheduler reject = attempt 消費 | **PASS** | `void schedule(RetryScheduleRequest, …)`（RetryScheduler.h:40）= **void**。counter ++ は :1272 の 1 箇所・rollback 減算は存在しない（リセットは rebind :1180 のみ）→ reject 後も rollback なし = **ND-07 §7「schedule reject は attempt を消費した retry drop」契約どおり**（:1269-1270 コメント明記）。source modification は行わず記録のみ |
| **V-α5-08** | Exhausted one-shot | **PASS** | guard = `Exhausted && !warmupRetryExhausted`（:1304-1305）。`warmupRetryExhausted = true` の write = **:1308 唯一**（宣言 :835・rebind :1181 は false 代入）。failure #4 → log 1 回 / #5+ → guard 不成立 → telemetry なし。rebind で exhausted=false 復帰（V-α5-02 と cross-check 一致） |
| **V-α5-09** | Exhausted telemetry 追跡性 | **PASS** | log（:1309-1314）に generation / attempts / limit / error / "(no further retry until generation changes)" を全含む。**`attempts = warmupRetryCount - 1`（:1311）→ failure #4 で attempts=3**。ND-07 :147 契約原文「diagLog（generation・error・**attempts=3**）」と**一致** — attempts = schedule 済み retry 消費数（3 = K 上限）であり failure 回数（4）ではない。**意味論上の不一致なし → BLOCKER 該当なし**（注記: enum 追加の取りやめ = CR-α-1 anchor audit F-1 で既決の diagLog-only・新規発見ではない） |
| **V-α5-10** | K=3 / K=4 domain separation | **PASS** | grep 実測: `kMaxWarmupConsecutiveRetries=3`（BuildErrorPolicy.h:92・Site 3 専用 :1275/:1290/:1312）/ `kMaxRecoveryConsecutiveFailures=4`（RebuildDispatch :1076 Builder-local・:1102/:1126）/ `kMaxObligationConsecutiveFailures=4`（coordinator h:401・cpp:1090/:1130）。**3 定数 = 3 ドメイン**（Site 3 warmup / Site 1/2 Builder recovery / obligation-level）で相互参照 0 件。header :87-91「意図的に別値（3 vs 4）」設計明記・混線なし |
| **V-α5-11** | Forbidden footprint | **PASS** | `git diff --name-only` 実測: **RetrySchedulerTypes.h / AudioEngine.h / RetryScheduler.h / RetryScheduler.cpp / RuntimeStore.h / Coordinator.h / CMakeLists.txt = すべて diff 0（不在）**。CR-α 新規変更 = 期待 3 ファイルのみ（BuildErrorPolicy.h +81/−0・RebuildDispatch.cpp +67/−17・BuildErrorClassificationTests.cpp +88/−2 = α-1 +87/−2 + α-3R +1/−0）。ND 残留（RuntimeWorldAuthority.h / ISRSemanticValidationTests.cpp / .gitignore / ConvoPeq.md / output_sourcecode_markdown.py）は CR-α 変更として扱わず分離 |
| **V-α5-12** | Final semantic chain | **PASS** | §V-α5-12 のとおりコード引用付きで再構成（warmup failure → ++ → classify → decision → 3 分岐 → generation rebind で budget 再開） |

## §V-α5-03 — counter 意味論 truth table（コード実測）

`++warmupRetryCount`（RebuildDispatch.cpp:1272）→ `warmupRetryDecision(warmupRetryCount, 3, …)`（:1274-1277）→
header :155-156 `if (attempt > maxRetries) → Exhausted`:

| warmup failure | counter（++ 後） | decision（構造） | retry |
| --------------: | ---------------: | ---------------- | ----- |
| 1 | 1 | Schedule（1 ≤ 3） | #1（Backoff 10ms / Immediate 0ms） |
| 2 | 2 | Schedule（2 ≤ 3） | #2（20ms） |
| 3 | 3 | Schedule（3 ≤ 3） | #3（40ms） |
| **4** | **4** | **Exhausted（4 > 3）** | **なし**（one-shot terminal log のみ） |
| 5+ | 5+ | Exhausted | **なし**（telemetry もなし — flag guard） |

**読み違いの排除**: `kMaxWarmupConsecutiveRetries = 3` は「schedule できる retry の回数」であり
「4 回目の failure まで retry する」ではない。counter は failure 回数として schedule 前に ++ され、
4 回目の failure（counter=4）は `attempt > 3` で Exhausted に分類され schedule されない。
実行時証明（T-CRα-3: attempt 4 → Exhausted / CR-α-4 standalone checks=86 fails=0 × 2 config）と
構造証明が一致。

## §V-α5-12 — Final semantic chain（コード引用付き再構成）

```text
① warmup failure
   RebuildDispatch.cpp:1252  if (warmupError != convo::BuildError::None)
② counter increment（decision 前必須）
   :1272  ++warmupRetryCount;
③ classify（disposition source）
   :1273  const auto outcome = convo::classifyBuildError(warmupError);
④ decision（純関数 — BuildErrorPolicy.h:143-161、caller :1274-1277）
   :1274  const auto decision = convo::warmupRetryDecision(
              warmupRetryCount, kMaxWarmupConsecutiveRetries, retryable, isObsolete(), outcome.retry, …);
   header:152  NoRetry || !contextRetryable || obsolete → {NoRetry, 0}
   header:155  attempt > maxRetries                → {Exhausted, 0}
   header:157  Backoff ? retryBackoffDelayMs : 0    → {Schedule, delay}
⑤ 3 分岐（caller :1279-1316）
   ┌────────────────────┬──────────────────────┬────────────────────┐
   │ Schedule (:1279)   │ Exhausted (:1304)    │ NoRetry（fall-thru）│
   │  req 生成 (:1281)  │  flag=true (:1308)   │  nothing (:1316)    │
   │  diagLog (:1287)   │  diagLog 1 回 (:1309)│  no schedule        │
   │  schedule (:1294)  │  no retry            │  no fallback        │
   │  fallback (:1298)  │                      │                     │
   └────────────────────┴──────────────────────┴────────────────────┘
   → :1318 continue（retry 経路はすべて RebuildThread ループ再入に帰着）
⑥ generation change → budget 再開
   :1177  if (task.generation != warmupRetryBoundGeneration)
   :1180      warmupRetryCount = 0;
   :1181      warmupRetryExhausted = false;
   → counter reset → retry budget が新 generation で再び利用可能（exhaustion は永続 mask ではない）
```

## 禁止事項遵守（実測）

```text
Production source 変更: 0 / Test source 変更: 0 / CMake 変更: 0
Build: 0 / CTest: 0 / stress: 0（指示どおり）
policy tuning / RetryScheduler 変更: 0
本 audit の artifacts: evidence/CRALPHA5_RETRY_SOURCE_AUDIT.md のみ（source/test/CMake 以外）
```

## BLOCK 条件チェック（指示の 8 項目 — すべて非該当）

| BLOCK 条件 | 判定 |
|---|---|
| failure #4 でも retry が発行される | 非該当（4 > 3 → Exhausted） |
| Exhausted から submitRebuildIntent / scheduler 到達可能 | 非該当（E-04 実測 0 経路） |
| NoRetry から retry が発行される | 非該当（policy 先頭 return + caller 不該当） |
| generation を跨いで exhaustion が残る | 非該当（rebind :1181） |
| counter ownership が共有化されている | 非該当（関数スコープ・外部参照 0） |
| K=3 と K=4 が実装上混線 | 非該当（3 定数 3 ドメイン・相互参照 0） |
| telemetry の attempts が契約意味と矛盾 | 非該当（attempts=3 = ND-07 :147 契約どおり） |
| forbidden file に CR-α の変更が存在 | 非該当（全 7 ファイル diff 0） |

## Overall

```text
Overall:
CR-α-5 = PASS（V-α5-01〜12 全 PASS・BLOCK 条件 8/8 非該当）

試験通過（CR-α-4 実行時証明）とは独立に、ソース構造による証明を完了:
  retry bound = K=3 / failure #4 = Exhausted / Exhausted 再発行なし / NoRetry schedule なし /
  Immediate bounded / reject rollback なしが契約 / exhaustion one-shot / K=3≠K=4 別 domain /
  forbidden footprint なし
```

## Next

```text
CR-α-5 = PASS → Closure（最終閉鎖判定）へ進行可
```

## 遷移

```text
CR-α-1   Implementation             PASS
CR-α-2   Read-only verification      PASS
CR-α-3   Build                       FAIL（using 漏れ）
CR-α-3R  Minimal test repair         PASS
CR-α-4   CTest 40/40 × 2             PASS
CR-α-5   Retry-specific audit        ← NOW = PASS
Closure                              解禁（待機）
```
