# D174 — OPEN Candidate Triage / BuildError Phase-II Preflight（evidence）

- 日付: 2026-09-08
- Type: **read-only preflight** — production source 0 / test source 0 / CMake 0 / build 0 / CTest 0
- Authority stamp: `ConvoPeq.md` — `Generated: 2026-09-08 07:15:27`（NEWER_SRC_COUNT=0 FRESH・D172-3 反映済み）
- 実装 commit 基準: **54ba7b40**（D172-3）

---

## D174-0 — Source Authority 確定 — **PASS**

| 確認項目 | 実測 |
|---|---|
| Generated timestamp | **2026-09-08 07:15:27**（D173-0 冪等再生成版） |
| NEWER_SRC_COUNT | **0（FRESH）** |
| D172-3 production diff 反映 | 済（MEM_SNAP resolver・R3 comment — D173-1 実測） |
| `BuildErrorPolicy.h` | enum BuildError ×8 / FailureClassification / RetryDisposition / §1.8.10.3 constexpr descriptor table / RetryBackoffPolicy {10,80,2} / warmupRetryDecision 純関数 |
| `RetryScheduler` | 専用 worker thread・kCapacity=8・deadline-sorted deque・DispatchFn = submitRebuildIntent（CtorDtor.cpp:98） |
| `PublishedWorldObservation` | RuntimeWorldAuthority.h（型 + factory + T-CW8 テスト） |
| `buildErrorCount_` | src 0 hits（RuntimeBuilder.h:124 コメントのみ） |
| Site 2 retry | classify+diagLog のみ（:1214）・schedule 呼び出しなし |
| D159 freeze anchors | D171-1 再実測のまま全現役 |

---

## D174-1 — BuildError Phase-II 再分類（A-G・actual call-chain 実測）— **STALE（CR-α 接続完了）/ DEFER（残部）**

### 実測 call-chain（production）

```text
classifyBuildError 呼び出し: 2 箇所のみ
  :1214  Site 2（RuntimeBuilder build failure）→ classify + diagLog → continue（retry 適用なし）
  :1290  Site 3（warmup failure）        → warmupRetryDecision（純関数）
                                              ↓ action == Schedule
retryScheduler_->schedule(req, delayMs):   :1311 — production call site はこれ 1 箇所のみ
                                              ↓ RetryScheduler 専用 worker thread
                                           DispatchFn → submitRebuildIntent（NonRT enqueue）
```

### A. exponential backoff は本当に全対象 Site に接続済みか — **接続済み（対象 = Site 3）**

- `schedule(req, decision.delayMs)` は production で **1 箇所のみ**（RebuildDispatch.cpp:1311）。`decision.delayMs` は `warmupRetryDecision` 純関数が返す: `RetryBackoff → retryBackoffDelayMs(policy, attempt)`（10/20/40/80/80 saturation）・`RetryImmediate → 0`
- CR-α（ND-07）契約の対象は **Site 3 warmup retry**。Site 2（build failure）は dash2 §1.8 Phase D の現行仕様「分類をログに記録」+ 将来拡張コメント（:1209-1213）— **未接続は仕様通り**（defect ではない）
- T-CRα-1（delay table 0/10/20/40/80/80）・T-CRα-4（disposition → delay mapping）テスト実在（BuildErrorClassificationTests.cpp:252-260）

### B. retry count はどこで保持・制限されるか — **3 層で分離**

| 層 | counter | 上限 | 契約 |
|---|---|---|---|
| Site 3 warmup | `warmupRetryCount`（generation rebind :1193-1198） | kMaxWarmupConsecutiveRetries=**3** | Exhausted で terminal telemetry 1 回・次の明示 request で再開 |
| Recovery spin 回避 | `recoveryConsecutiveFailures`（builder-local） | K=**4** | 超過で次サイクルへ委譲（break） |
| Obligation-level | `postRecoveryFailureSignal(obligationId)` | obligation 契約 | **D105-R18: Dual-LP — durable-slot sub-state と obligation counter は別フィールドの独立線形化** |

### C. `buildErrorCount_` は何を数えるのか — **仕様: aggregate build failure telemetry（未実装）**

RuntimeBuilder.h:118-124 の仕様記述: 「将来 convolver/prepare の実 failure が観測可能になり subsystem 別 retry 判定が必要になる設計確定時のみ、最小 wiring（status 伝播 + caller failure check 強化 + **buildErrorCount_ telemetry**）で対応」。現行は集約 counter の要求事象なし。

### D. telemetry は既存 invariant を満たすか — **満たす（既存観測で十分）**

- REBUILD_TELEMETRY（request/dispatch/merged/classification）+ `[DIAG] RuntimeBuilder build failed（error/classification/retry disposition ログ）`+ Site 3 Exhausted terminal telemetry — failure 分類と retry 動作は現行観測で追跡可能

### E. Site 3 と Recovery Site 1/2 の retry domain 混線 — **なし（明文分離）**

- BuildErrorPolicy.h:87「**kMaxWarmupConsecutiveRetries = 3 は Site 3 専用**」
- RebuildDispatch.cpp:849「recovery の K=4（Site 1/2）とは別ドメイン」
- Site 3 の schedule 呼び出しは 1 箇所のみ → Recovery K=4 の構造的誤用は発生経路なし

### F. `RetryImmediate` が仕様上意図されたものか — **意図された設計**

- descriptor table: `WarmupFailed → { Transient, RetryImmediate }`（latency-sensitive 向け zero-delay）
- CRALPHA5 V-α5-06 PASS: **Exhausted 判定（attempt > maxRetries）が delay 計算より先行** → RetryImmediate でも attempt > 3 なら Exhausted → **Immediate = zero-delay bounded retry（unbounded bypass の構造的不存在）**

### G. inventory の「未実装」は stale か — **stale**

CR-α は CR-α-1..6 CLOSED（commit 0aeb22ca・2026-09-01）・D163 で REJECT 判定・D171-1 で STALE 再確認。inventory 1-C-1 の「即時 retry（delay=0）＝ backoff 未接続」記述は現行 source と乖離。**残部で真に未実装なのは buildErrorCount_（DEFER）と Site 2 retry 適用（将来拡張コメント現役）のみ。**

---

## D174-2 — Site 2 retry の authority / semantics audit — **DEFER（必要性未証明・非 defect）**

| 判定軸 | 実測 | 判定 |
|---|---|---|
| Site 2 が kMaxWarmupConsecutiveRetries=3 を誤用 | Site 2 は **schedule 呼び出しなし**（classify+log+continue）→ 誤用経路が構造的に存在しない | 安全 |
| Recovery K=4 との混同 | Site 2 は counter を保持しない・Recovery は別変数 | 混線なし |
| RT path への retry 侵入 | schedule 呼び出し元 = RebuildThread（NonRT）・RetryScheduler は**専用 worker thread**（RetryScheduler.cpp:5）で dispatch・DispatchFn → submitRebuildIntent（rebuild thread enqueue） | RT 排除 ✔ |
| unbounded 化 | Site 2 = 単発（rebuild intent は次の明示 request 待ち）・Site 3 max=3・Recovery K=4 — 全 bounded | ✔ |
| authority 分散 | schedule() production call site = **1 箇所**（:1311）| 単一 ✔ |
| obligation lifetime / delivery への影響 | Recovery retry は DurablePending 戻し + obligation counter（D105-R18 独立線形化）・Site 2/3 は obligation を触れない | 影響なし ✔ |

**判定**: Site 2 の retry 適用は「現行で欠損している安全機構」ではない（分類+ログで十分・次の明示 request で自然に再開）。subsystem 別 retry 判定が必要になる設計確定（RuntimeBuilder.h:118-124 の条件）まで** DEFER**。実装する場合も D174-1 実測の既存 authority（BuildErrorPolicy 純関数 + RetryScheduler）を再利用し二重化しない契約が前提。

---

## D174-3 — `buildErrorCount_` trigger audit — **DEFER / monitoring 維持（trigger なし）**

```text
buildErrorCount_（未実装）
  ↓ 要求する観測ギャップは？
既存 telemetry: REBUILD_TELEMETRY + classifyBuildError ログ（error/classification/retry）+ Exhausted terminal
  ↓
failure / retry policy: 全 bounded（D174-1/2 実測）
```

- trigger 条件（RuntimeBuilder.h:118-124）: 「convolver/prepare の**実 failure が観測可能になり**、subsystem 別 retry 判定が必要になる設計確定時」— 現行では MKLFailure / ConvolverFailure / PrepareFailure は**生成経路なし**（enum+toString のみ・休眠分類）
- 現行 observability で Phase-II 実装を開始すべき具体的事象: **未観測**
- 判定: **trigger なし → freeze register / monitoring item として維持**（D163・CRBETA0 B-7 の二重記録どおり）

---

## D174-4 — CW-8 正式確認 — **STALE / ALREADY COVERED（実装禁止）**

| 要素 | fresh snapshot 実測 |
|---|---|
| 型 | RuntimeWorldAuthority.h:90-101 — `PublishedWorldObservation`（private ctor + friend 構造遮断・trivially copyable・独立構築不能） |
| factory | :240-248 `observePublishedObservation(const ReadToken&)` — **単一 acquire load から {world, &world->publication} 同時確定**・未 publish 時 {nullptr, nullptr} |
| テスト | `testCW8_PublishedWorldObservation`（ConvoPeq.md L93290）+ harness 登録 L94966-94968（T-CW8-1/2/3/4/6/7） |
| inventory 記述 | 「src 0 件」→ **stale**（19 hits 実測） |
| production caller | 0 件（保守的休止 — D163/CRBETA0 同結論） |

→ **CW-8 = ALREADY COVERED / STALE。D174 候補から除外・実装禁止。**

---

## D174-5 — Inventory relevance

| 候補 | D174 判定 |
|---|---|
| CW-8（1-C-2） | **STALE / ALREADY COVERED** — 実装禁止 |
| CR-α backoff（1-C-1 本体） | **STALE** — Site 3 に接続済み（call-chain 実測） |
| buildErrorCount_（1-C-1 残部） | **DEFER / monitoring** — trigger なし |
| Site 2 retry 適用 | **DEFER** — 非 defect・必要性未証明・将来拡張コメント現役 |
| D159 DEFER D1-D6 | **DEFER 維持**（全 anchor 現役） |

- inventory 内に MEM_SNAP / D172 起因の追加 stale 項目 = 0 件（D173-3 再確認）

## Final Decision

> ## **NO IMPLEMENTATION — D175 implementation contract は起票不要**
>
> - CW-8 = STALE / ALREADY COVERED（実装禁止）
> - CR-α backoff = STALE（Site 3 に wired・call-chain 実測で再証明）
> - buildErrorCount_ = DEFER / monitoring（trigger なし）
> - Site 2 retry 適用 = DEFER（非 defect・authority 再利用前提の将来拡張）
> - genuine OPEN implementation item = **0 件**
>
> 次の着手可能な作業は doc-only maintenance（inventory STALE 化反映 / buildErrorCount_ trigger 登録 / h:2265 コメント修正）のみ。

## 実測コマンド系譜（主要分）

```bash
rg -n "classifyBuildError\(" src/audioengine/*.cpp                  # 2 箇所（:1214 Site2 / :1290 Site3）
rg -n "schedule\(req" src/audioengine/*.cpp                          # 1 箇所（:1311 Site3 のみ）
sed -n '30,80p;97,165p' src/audioengine/BuildErrorPolicy.h           # descriptor table / policy / decision
sed -n '1,100p' src/audioengine/RetryScheduler.h / RetryScheduler.cpp # worker thread / kCapacity=8 / dispatch
sed -n '92,99p' src/audioengine/AudioEngine.CtorDtor.cpp             # DispatchFn → submitRebuildIntent
sed -n '1108,1120p' src/audioengine/AudioEngine.RebuildDispatch.cpp  # D105-R18 obligation counter
sed -n '115,128p' src/audioengine/RuntimeBuilder.h                   # buildErrorCount_ 仕様（trigger 条件）
rg -n "PublishedWorldObservation|testCW8" src/audioengine/RuntimeWorldAuthority.h ConvoPeq.md
rg -n "D105-R18" src/audioengine/                                    # obligation-level counter
# 交差検証: context-mode ctx_batch_execute（5 並列）＋ CRALPHA 報告書（V8/V-α5-06 実測記録）
```
