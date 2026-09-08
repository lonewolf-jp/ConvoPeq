# D178 — CR-α BuildError Retry Backoff / Count / Telemetry 実装前 read-only Scope Audit（Evidence）

- 日時: 2026-09-08（D177 PASS 直後）
- 性質: **read-only**。production / test / CMake / build 変更 0 件（実測: 作業中 `git diff HEAD -- src/` は D177 時点から既存の AudioEngine.h コメント 9+/3- のみ）
- authority: ConvoPeq.md `Generated: 2026-09-08 20:32:26`（src 更新 0 件 = FRESH 実測）

---

## 総合判定

> ## **D178 = REJECT（CR-α は既に実装済み・closure 完了）→ 実装タスクなし**
>
> ユーザー指示の前提「実際の scheduler 呼び出しは 0ms 固定」は **2026-09-01 時点の stale 情報**。
> 現行ソースは CR-α-1..6（ND-06 CONDITIONAL-GO → ND-07 GO 契約 → 実装 → build/CTest → retry audit 12/12 → closure、commit 0aeb22ca・2026-09-01）で **Site 3 warmup retry に K=3 bound + backoff {10,80,2} + delay 接続 + telemetry（diagLog）が実装済み**。
> D163 REJECT / D171-1 / D174 / D175 の inventory 更新と一致。genuine OPEN implementation item = 0 件。

---

## 0. 発見の系譜（今回 D178 で確定したこと）

1. `doc/work88/` に既存報告書群を確認:
   - `ND06_BUILDERROR_RETRY_AUDIT_REPORT.md` = **CONDITIONAL-GO**（Site 3 のみ乖離: bound 無し即時 retry loop）
   - `ND07_CRALPHA_RETRY_CONTRACT_REPORT.md` = **GO**（実装判断余地ゼロの Site 3 専用 retry contract §1-§14）
   - `CRALPHA1..6_*.md` = 実装 → 検証 → build（FAIL→3R 修復）→ CTest 40/40×2 → retry audit 12/12 → **closure CLOSED/ACCEPTED（2026-09-01）**
   - `CRBETA0_RESIDUAL_TRIAGE_REPORT.md` = CW-8 ALREADY COVERED / buildErrorCount_ DEFER
2. ユーザー指示の「0ms 固定」記述は inventory 1-C-1 の historical 記述（2026-09-01 時点実測）に基づく。D175-0 判定更新（2026-09-08）で既に STALE 化済み。
3. よって D178 は「実装前 scope audit」ではなく「**現行 HEAD での intact 再検証 + 未確定事項の棚卸し**」として実施した。

## 1. D178-1 — CR-α scope 確定（7 項目・現行ソース実測）

### (1) BuildError 発生地点の全列挙

production 生成経路は `src/audioengine/RuntimeBuilder.cpp` の 4 種のみ（実測 grep `BuildError::` = 5 行）:

| 行 | Error | 経路 |
|---|---|---|
| :417 | `InvalidInput` | build 入力不備 |
| :443 | `ResourceUnavailable` | bad_alloc 経路 |
| :448 | `InternalError` | catch(...) |
| :458 | `WarmupFailed` | `isIRLoaded && !isIRFinalized` |
| :460 | `None` | 成功 |

MKLFailure / ConvolverFailure / PrepareFailure は生成経路なし（table・test のみ・既知 V-5）。

### (2) classifyBuildError() 全 caller

production 2 箇所（RebuildDispatch.cpp 内のみ）:
- **:1214** build failure（Site 3）— **ログ記録のみ**、retry 制御フローに使わない
- **:1290** warmup failure（Site 3）— `warmupRetryDecision(..., outcome.retry, ...)` に接続 = **唯一の機能使用**

Site 1/2（recovery）は `runtime == nullptr` のみ検査・error 値は retry 判断に使わない（ND-06 §5 どおり現行維持）。

### (3) RetryDisposition 実適用状況 — **実装済み**

`BuildErrorPolicy.h`:
- descriptor table `kBuildErrorDefaultTable`（8 値・static_assert 網羅）
- `warmupRetryDecision()`（:143-161）: `NoRetry / !contextRetryable / obsolete → NoRetry`、`attempt > maxRetries → Exhausted`、`RetryBackoff → retryBackoffDelayMs(policy, attempt)` / `RetryImmediate → 0`
- `retryBackoffDelayMs()`（:108-123）: uint64 中間計算 + 上限早期 return の saturation（構造的 overflow 排除）

### (4) RetryScheduler::schedule() 全 caller と delay 決定元

production caller = **1 箇所のみ**（実測 rtk rg `\.schedule\(` src/）:
- `AudioEngine.RebuildDispatch.cpp:1311`: `retryScheduler_->schedule(req, std::chrono::milliseconds(decision.delayMs))` — **delay は `decision.delayMs`（backoff 表）・0ms 固定ではない**
- delay 決定元: `kDefaultWarmupRetryBackoff {10, 80, 2}` → attempt 1/2/3 = 10/20/40ms（K=3 のため 80 は cap のみ）

fallback: `retryScheduler_ == nullptr` の異常系（テスト/単体ビルド）のみ `submitRebuildIntent` 従来経路。

### (5) retry counter 既存同等値の全探索

実測（rtk rg `ConsecutiveFailures|warmupRetryCount|recoveryConsecutive`）:

| 定数/変数 | 場所 | 値 | ドメイン |
|---|---|---|---|
| `kMaxWarmupConsecutiveRetries` | BuildErrorPolicy.h:92 | **3** | Site 3 warmup（CR-α 専用） |
| `kMaxRecoveryConsecutiveFailures` | RebuildDispatch.cpp:1093 | 4 | Site 1/2 Builder-local spin guard |
| `kMaxObligationConsecutiveFailures` | ISRRuntimePublicationCoordinator.h:401 | 4 | obligation T3c CAS（Site 1/2） |
| `kMaxDeferredRetries` | RuntimePublicationOrchestrator.h:322 | 2 | D135 deferred（dormant・retention 不増加） |

**K=3 / K=4 の意図的別値によるドメイン分離がコード＋コメントで確立済み**（BuildErrorPolicy.h:87-91、RebuildDispatch.cpp:849）。

### (6) BuildError telemetry の既存観測可能性

- warmup retry scheduled/exhausted: diagLog（RebuildDispatch.cpp:1304-1310 / 1326-1332）— generation / attempt / limit / delayMs / error を記録
- telemetry enum 追加は CR-α-1 F-1 deviation で diagLog-only に確定変更（ND-07 §9 は「本質を diagLog で完全充足」として取りやめ。CRALPHA1_ANCHOR_AUDIT_REPORT.md F-1 参照）
- 既存 `RebuildTelemetryReason::RebuildThreadWarmupRetry` は schedule request の reason として現役（AudioEngine.h:2915 `toTelemetryReasonString` 網羅）
- D167-5（closure 後追加分・CR-α 領域外）: `AdmissionClosed` reason 追加のみ（RetrySchedulerTypes.h +4 行・switch case AudioEngine.h:2942）

### (7) Site 3 と Site 1/2 Recovery retry の境界再確認

- Site 1: `postRecoveryFailureSignal(oblId)` → obligation K=4（T3c CAS）— classify 不使用
- Site 2: `settle(true)` durable re-lease + signal — Builder-local 4（成功 reset: cpp:1160）+ obligation K=4 — classify 不使用
- Site 3: `shouldRetryWarmupFailure`（isLoadingIR）→ classify → decision 純関数 → schedule(delay) — **K=3 専用**
- 統合・混線の構造は存在しない。3 ドメイン（Site 3 = 3 / Site 1/2 = 4+4 / deferred = 2 dormant）が独立。

## 2. D178-2 — invariant impact 連鎖の現行証明

```text
BuildError（RuntimeBuilder.cpp 4 種生成）
   ↓ classifyBuildError（BuildErrorPolicy.h:66・table 参照純関数）
   ↓ RetryDisposition（kBuildErrorDefaultTable）
   ↓ warmupRetryDecision（純関数・AND gate: contextRetryable ∧ !obsolete ∧ attempt<=3 ∧ !=NoRetry）
   ↓ bounded NonRT scheduling（RebuildDispatch.cpp:1311・RebuildThread 上）
   ↓ RetryScheduler（capacity 8・deadline-ascending deque・専用 worker・wait_until）
   ↓ submitRebuildIntent（merge/Replaceable collapse）→ RebuildThread
```

| invariant | 判定 | 根拠（現行実測） |
|---|---|---|
| RT blocking なし | **PASS** | RT 系 6 ファイル（Threading/Timer/PrepareToPlay/ReleaseResources + ProcessBlock 実体 DSPCoreFloat/Double/BlockDouble/AudioBlock）の `classifyBuildError|RetryDisposition|warmupRetryDecision|BuildErrorPolicy|retryScheduler_` 参照 = **0 件**（per-file grep 実測）。BE-8 維持 |
| retry 無限化なし | **PASS** | K=3 hard bound（`attempt > maxRetries → Exhausted`・exhausted 後 schedule 経路 0・`continue` のみ。5+ は one-shot flag で telemetry も 0） |
| capacity 8 越え ownership 生成なし | **PASS** | `kCapacity=8`（RetryScheduler.h:62）+ reject 時 `rejectCount++`・pending 不変。schedule() は void → reject = attempt 消費の retry drop（ND-07 §7 契約・cpp:1279-1281 コメント）。T6 test 実在 |
| Recovery obligation と retry counter の混線なし | **PASS** | counter は rebuildThreadLoop 関数スコープ（cpp:849-852）・generation rebind（:1196-1201）・Recovery 系は obligation lifecycle word で独立。Site 3 のみ bound 対象 |
| Publish authority 増加なし | **PASS** | CR-α 領域に publish 経路追加なし。唯一の schedule caller は既存 pipeline への intent 発行のみ |
| Retire authority 到達なし | **PASS** | retry 経路は RuntimeStore/EBR/retire に触れない（CR-α-5 V-α5-11 forbidden footprint 0・closure 後 diff も +17 行は D167-5 AdmissionClosed telemetry のみ） |
| HealthMonitor を retry decision authority にしない | **PASS** | RuntimeHealthMonitor.cpp/.h の retry/Retry/schedule/decision 参照 = **0 件**（実測）。observation-only 維持（D177 契約） |

closure 後の CR-α 4 ファイル差分（0aeb22ca..HEAD 実測）:
- `BuildErrorPolicy.h` = 0 diff
- `BuildErrorClassificationTests.cpp` = 0 diff
- `RetrySchedulerTypes.h` = +4/-1（D167-5 AdmissionClosed reason 追加のみ・CR-α 意味論不変）
- `AudioEngine.RebuildDispatch.cpp` = +17（D167-5: tryAdmit 失敗時 Suppressed(AdmissionClosed) telemetry・warmup retry 領域は無変更）

→ **CR-α 実装は closure 時から intact**。

## 3. D178-3 — 実装境界（closure 固定版）の確認

CR-α-6 §8 の closure 後規約どおり、本監査は一切の変更を行わなかった。forbidden changes（Site 1/2 recovery・K=4・spin guard 4・RetryScheduler infrastructure・RuntimeStore/EBR/retire・MKLFailure 等の生成経路・prepare() status propagation・Site 3 build failure の retry 化・新 authority/queue/atomic・ReadToken/CW-8・ScheduleRequest field 追加）は現行 HEAD でも非該当を実測。

残存 DEFER（trigger 待ち・実装禁止継続）:
- Site 2 retry 適用（非 defect・dash2 §1.8 Phase D 仕様どおり）
- buildErrorCount_ telemetry（D175-1 補助 trigger 登録済み・monitoring）
- MKLFailure / ConvolverFailure / PrepareFailure 生成経路（V-5・将来拡張）

## 4. D178-4 — targeted test contract（T-α-1..7）と実在テストの対応

| 指示 AC | 実在テスト | 状態 |
|---|---|---|
| T-α-1 backoff delay sequence | T-CRα-1（BuildErrorClassificationTests.cpp runTestF: 0/10/20/40/80/80/100→80） | **実装済み** |
| T-α-2 maxDelay saturation | T-CRα-1 saturation + inverted {100,50,2} cap | **実装済み** |
| T-α-3 retry count upper bound | T-CRα-3（attempt 4→Exhausted / 5→Exhausted） | **実装済み** |
| T-α-4 NoRetry/Immediate/Backoff 分離 | T-CRα-3/4（context/obsolete/max + delay 0 vs 10/20） | **実装済み** |
| T-α-5 scheduler capacity pressure | RetrySchedulerTests T5/T6/T7/T8（queue full・rejectCount・shutdown・concurrent shutdown） | **実装済み** |
| T-α-6 existing Recovery retry regression | CTest 40/40 × Debug/Release（CR-α-4）+ D169-2-7 full regression | **実装済み** |
| T-α-7 RT path exclusion | RT 6 ファイル 0 件参照（本監査 per-file 実測）+ CR-α-5 §V-α5 | **実装済み（static 実測）** |

CR-α-4 standalone checks=86 fails=0 × 2 config（evidence/cra4_ctest_*.log）。

## 5. 未確定事項の棚卸し結果（本監査で確定したもの）

| 事項 | 確定結果 |
|---|---|
| 「scheduler 呼び出し 0ms 固定」説 | **stale** — 現行は `decision.delayMs`（10/20/40ms backoff・WarmupFailed は 0ms RetryImmediate が正当意味論） |
| genuine OPEN implementation item | **0 件**（D174 結論を現行 HEAD で再実証） |
| CR-β / CW-8 | ALREADY COVERED（ND-01..04 実装済み・production caller 0 は保守的休止）— 実装タスクなし |
| CR-α intact 性（closure 後の改変有無） | **intact**（4 ファイル diff 実測・warmup retry 領域 0 変更） |
| F-1 deviation | 正当（enum 追加 → AudioEngine.h 変更必然化 → diagLog-only に縮約。telemetry 本質は充足） |
| F-2（toTelemetryReasonString に static_assert 機構なし） | 将来リスクとして継続記録（新規 CR 起票不要・機能影響なし） |

## 6. 次の一手

```text
D178 = REJECT（実装済み closure の再実装要求）→ 実装タスクなし
   ↓
D159 通常開発サイクルへ復帰
   ↓
Phase-II 項目は D159 freeze register（D1-D6 + 補助 trigger）待ち
   ↓
RuntimeBuilder.h:118-124 trigger 条件成立（設計確定イベント）まで新規 track 起票禁止（D174 Final Decision）
```

CR-α の再オープンは禁止（CR-α-6 §8: backoff 値変更等は別 work item 起票）。

## 7. 使用ツール（token reduction 3 層パイプライン）

ctx_batch_execute（並列コマンド実行・全検索）+ rtk (WSL) rg + grep/sed + aidex_session（外部変更検出）+ Write（報告書作成のみ）。生ファイルは index 化され context に入れず、必要節のみ ctx_search で抽出。
