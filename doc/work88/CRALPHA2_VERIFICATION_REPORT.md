# CR-α-2 — Read-only Implementation Verification（read-only）

```text
Date: 2026-09-01
Production source: 0 / Test source: 0 / CMake: 0 / Build: 0 / CTest: 0 / stress: 0（完全 read-only）
baseline: ConvoPeq.md Generated 2026-09-01 20:32:43（--check FRESH / NEWER_SRC_COUNT=0 実測 — CR-α-1 実装後の再生成 snapshot をコード基準に使用）
検証対象: CR-α-1 の 3 ファイル実装（BuildErrorPolicy.h / AudioEngine.RebuildDispatch.cpp / BuildErrorClassificationTests.cpp）
```

## 総合判定

> ## **CR-α-2 = PASS（V1〜V16 全 16 項目 PASS）— CR-α-3（Debug/Release build）へ進行可**
>
> **V5（最重要 blocker）の解消**: ND-07 §4 契約原文（ユーザー GO 済み）は
> 「attempt 1 = 10ms / attempt 2 = 20ms / attempt 3 = 40ms（80 は cap）」と明記しており、
> 実装 `{10,80,2}` → 0/10/20/40/80/80 は**契約どおり**。「attempt 2 → 80 / attempt 3 → 160」の
> 数値は CR-α-1 指示メッセージの prose にのみ存在し、ND-07 §4 契約・policy 定義
> `{10,80,2}`・saturation テスト項目と矛盾する**誤記と確定**（§V5 詳細）。

---

## V1〜V16 判定表

| ID | Verification | 判定 | 実測根拠 |
|---|---|---|---|
| **V1** | Counter owner = RebuildThread | **PASS** | 3 変数（:833-835 `warmupRetryBoundGeneration=-1 / warmupRetryCount=0 / warmupRetryExhausted=false`）は `rebuildThreadLoop()` :824 の**関数スコープ**・`while (true)` :845 の外。atomic/mutex 化 0 件。Coordinator / Scheduler / RT 側からの参照 0 件（`warmupRetry` grep: coordinator/scheduler/AudioEngine.h = 0 件） |
| **V2** | Generation rebind | **PASS** | sealed check（:1171-1172）の直後に rebind（:1177-1182）: generation 比較 → count=0 / exhausted=false / bound 更新。過去 generation の exhaustion は新 generation に**持ち越されない**。`task.generation` は `int generation = 0`（AudioEngine.h:2740）非負・sentinel `-1` と非衝突。rebuildRequestGeneration は `atomic<int>` 0 起点実測 |
| **V3** | K=3 と K=4 の分離 | **PASS** | `kMaxWarmupConsecutiveRetries`（=3）: BuildErrorPolicy.h:87 / RebuildDispatch :832・:1275・:1290・:1312 / tests。`kMaxRecoveryConsecutiveFailures`（=4）: RebuildDispatch :1076・:1102・:1126（Site 2 既存）+ coordinator h:397 + test。**新規 warmup コードは kMaxRecoveryConsecutiveFailures を参照せず（0 件）**、Site 2 側 3 箇所は diff で無変更 |
| **V4** | Retry decision truth table | **PASS** | 純関数分岐（BuildErrorPolicy.h）: ①`disposition==NoRetry \|\| !contextRetryable \|\| obsolete` → NoRetry ②`attempt > maxRetries` → Exhausted ③それ以外 → Schedule。truth table 全 7 行成立（context=false→NoRetry / obsolete→NoRetry / NoRetry→NoRetry / attempt 1,2,3→Schedule / attempt 4→Exhausted） |
| **V5** | Backoff `{10,80,2}`（最重要） | **PASS** | **§V5 詳細（下記）** — ND-07 §4 契約原文との突合で解消。実装 = `{10,80,2}` 標準 exponential + saturation |
| **V6** | `decision.delayMs` → scheduler | **PASS** | Schedule 分岐: `retryScheduler_->schedule(req, std::chrono::milliseconds(decision.delayMs))`（:1294）。旧 `milliseconds(0)` の残存 = **0 件** |
| **V7** | NoRetry → no schedule | **PASS** | NoRetry action は if/else-if のいずれにも該当せず fall-through → schedule / fallback ともに実行されない（「NoRetry / exhausted 重複後は telemetry なし」コメント付き） |
| **V8** | Immediate → 0 ms but bounded | **PASS** | 純関数内: `disposition == RetryImmediate` → delay 0。ただし Exhausted 判定は disposition とは独立に `attempt > maxRetries` で先に実行 → **bounded をバイパスしない** |
| **V9** | Fallback = Schedule only | **PASS** | `submitRebuildIntent` は Schedule 分岐内の `else`（scheduler nullptr 時）のみ。Exhausted 分岐・NoRetry からは到達しない（制御フロー精読・:1298 のみ 1 箇所） |
| **V10** | Exhausted one-shot | **PASS** | `else if (action == Exhausted && !warmupRetryExhausted)` → flag set + diagLog。failure #4 → log 1 回 / #5 以後 → flag により log なし（`= true` が1度のみ）。generation rebind で exhausted=false に復帰 |
| **V11** | Exhausted に `submitRebuildIntent` なし | **PASS** | Exhausted 分岐本体 = flag set + diagLog のみ（制御フロー精読）。`submitRebuildIntent` はファイル内 :1298（Schedule fallback）の 1 箇所のみ |
| **V12** | diagLog required fields | **PASS** | Schedule log: generation / attempt / limit / delayMs / error。Exhausted log: generation / attempts / limit / error / "no further retry until generation changes"（意味明示） |
| **V13** | RetrySchedulerTypes.h unchanged | **PASS** | diff = 0（実測） |
| **V14** | AudioEngine.h unchanged | **PASS** | diff = 0（実測。`toTelemetryReasonString` の既存 case も無変更 — `RebuildThreadWarmupRetry` 現役維持） |
| **V15** | Site 1/2 / build-failure path untouched | **PASS** | diff 内に settle/postSignal/kMaxRecovery への変更 0 件。Site 3 build failure block（:1186-1196）は現行のまま（classify ログ + continue・retry 化なし）。Site 2 spin guard（kMax=4）も diff 0 |
| **V16** | T-CRα-1〜4 source/test correspondence | **PASS** | runTestF に T-CRα-1〜4 対応 CHECK 30 件実装（delay table・saturation・inverted cap・default 値・max=3・decision truth table・mapping）+ main and 接続。T-CRα-5/6 は本 audit V11/V12 として実施済み |

## §V5 詳細 — backoff 記述の内部矛盾の解消（最重要 blocker）

**乖離の所在の確定:** 「attempt 1→10 / attempt 2→80 / attempt 3→160」の数値は **CR-α-1 指示メッセージの prose**（Step 1 の table + Step 6 の test 項目 5-7）に存在します。CR-α-1 Work Report および本実装は一貫して **10/20/40/80/80** です（Work Report に 10/80/160 の記述はありません）。

**契約原文との突合（ND-07 §4 — ユーザー GO 済み）:**

| ND-07 §4 原文（evidence/ND07_CRALPHA_SITE3_RETRY_CONTRACT.md） | 内容 |
|---|---|
| :16 | 「backoff normative default = **10→20→40→80ms**（`{10, 80, 2}`）を採用」 |
| :59 | 「normative default を確定: 設計記録の **10→20→40→80ms** を採用する」 |
| :88 | tuning parameter = `{initialDelayMs=10, maxDelayMs=80, multiplier=2}` |
| :89 | 「**attempt 1 = 10ms / attempt 2 = 20ms / attempt 3 = 40ms**（max 3 のため 80 は cap としてのみ存在）」 |
| :194（T-CRα-1 test spec） | 「attempt 1/2/3/4 → **10/20/40/80**・saturation（attempt 5+ → 80）」 |

**判定:** 契約が `{10,80,2}` を「initial / max / multiplier」と定義しているため、標準 exponential `min(10×mult^(attempt-1), 80)` = **10 / 20 / 40 / 80 / 80...** が契約どおり。実装はこれを正確に実装している（attempt 0→0 も含む）。「attempt 2 → 80 / attempt 3 → 160」は **{10,80,2} からはいかなる式でも導出できず**、同一指示メッセージ内の policy 定義・saturation 項目・ND-07 §4 契約の三方と矛盾する誤記と確定。

**V5 = PASS**（実装 = 契約）。参考: もし 10/80/160 が意図だった場合、`kDefaultWarmupRetryBackoff` を `{10, 160, 8}` にする 1 行変更で対応可能（試算: 10 / 80 / 160 / 160...）— 現時点では契約どおり {10,80,2} を維持する。

## 禁止事項遵守（diff 実測）

```text
Production source 変更: 0（本 audit は read-only）
Test source 変更: 0 / CMake 変更: 0 / Build: 0 / CTest: 0 / stress: 0
累積 diff（CR-α-1 分）: BuildErrorPolicy.h +81/−0・RebuildDispatch.cpp +67/−17・BuildErrorClassificationTests.cpp +87/−2
Snapshot: FRESH 20:32:43（NEWER_SRC_COUNT=0・--check 実測）
```

## 遷移

```text
CR-α-1 実装 → CR-α-2 PASS（本報告）
   ↓
CR-α-3: Debug / Release build（指示どおり）
   ↓
CR-α-4: CTest（40/40 現行基準 × Debug/Release）+ regression
   ↓
CR-α-5: retry-specific source audit → CR-α closure
```
