# ND-06 — CR-α BuildError Retry Contract Audit（read-only）

```text
Date: 2026-09-01
Production source: 0 / Test source: 0 / CMake: 0 / Build: 0 / CTest: 0 / stress: 0（完全 read-only）
baseline: ConvoPeq.md Generated 2026-09-01 19:31:33（ND-06-1 で --check FRESH / NEWER_SRC_COUNT=0 を再実測 —
ND-04 の stamp を引用せず開始時点で再確認）
```

## 総合判定（先出し）

> ## **ND-06 = CONDITIONAL-GO**
>
> Recovery 系 2 site（transient / durable）の retry 意味論は設計契約と**一致**（obligation-level K=4 bound + durable re-lease・実装済み）。
> **乖離は Site 3（main rebuild warmup）に集中**: classification は実装済みだが retry decision への適用は
> `schedule(req, delay=0)` の gate のみ・retry count / exponential backoff / retry telemetry は未実装 →
> 永続 warmup failure 時に **bound 無しの即時 retry loop**（Site 3 のみ）。
> CR-α は「Site 3 限定の bounded retry + backoff tuning parameter + telemetry」として
> **CONDITIONAL-GO**（§12）。

---

## 1. Current implementation topology

```text
RuntimeBuilder::build()                          ← 生成（4 種のみ）: InvalidInput / ResourceUnavailable / InternalError / WarmupFailed
    ↓
BuildResult { runtime, error }                   ← failure detection は二系統: runtime==nullptr（primary）と error!=None（warmup）
    ↓
rebuildThreadLoop（専用 NonRT スレッド・AudioEngine.RebuildDispatch.cpp:824-）
    ├─ Site 1: transport recovery（:998-1018）     → postRecoveryFailureSignal(obligationId) → continue
    ├─ Site 2: durable recovery（:1083-1113）      → settle(true) + postRecoveryFailureSignal + spin guard(4) 
    └─ Site 3: main rebuild（:1171-1265）          → classify（log / gate）→ schedule(req, 0ms) or fallback
    ↓
classifyBuildError()（BuildErrorPolicy.h — constexpr descriptor table）
    ↓
RetryScheduler（deadline-ascending deque・capacity 8・専用 NonRT worker thread）
    ↓
DispatchFn → submitRebuildIntent()（:151-386・producer-side merge/collapse）
```

- `BuildErrorPolicy.h` は契約完全体: `BuildError` 8 値 / `FailureClassification` 4 値（Permanent・Transient・Infrastructure・Fatal）/ `RetryDisposition` 3 値（NoRetry・**RetryBackoff**・**RetryImmediate**）/ `BuildOutcome` / constexpr `kBuildErrorDefaultTable`（全 8 値 static_assert 2 件）/ bounds-checked `classifyBuildError` / `classifyBuildErrorToString`。
- `MKLFailure` / `ConvolverFailure` / `PrepareFailure` は **production 生成経路なし**（table・test のみ。dash2 §1.8.12 item 3 の調査どおり — IPP/MKL は status code ベースで例外を投げない・`prepare()` は void）。`WarmupFailed` は `validateWarmup` の `isIRLoaded() && !isIRFinalized()` のみ（RuntimeBuilder.cpp:458）。

## 2. BuildError production sites

| Error | 生成箇所 | 実測 |
|---|---|---|
| InvalidInput | RuntimeBuilder.cpp:417 | 入力検証 |
| ResourceUnavailable | :443（`catch (const std::bad_alloc&)`） | 全 TU 常時有効（§1.8.12 item 6） |
| InternalError | :448（`catch (...)`） | 同上 |
| WarmupFailed | :458（`validateWarmup`） | IR loaded && !finalized |
| MKLFailure / ConvolverFailure / PrepareFailure | **生成経路なし** | §1.8.9 Phase-2 item 8-10 未実装（prepare() は void・init() bool は WarmupFailed に集約） |

## 3. Classification path

`classifyBuildError()` の production call site は **2 箇所のみ**（実測）:

| Call site | 用途 | 実際の使用 |
|---|---|---|
| RebuildDispatch.cpp:1178（Site 3 build failure） | **diagLog への記録のみ** — `outcome.classification` / `outcome.retry` は制御フローに使われない（`continue` 固定） | 分類は diagnostic |
| RebuildDispatch.cpp:1247（Site 3 warmup failure） | `if (outcome.retry != RetryDisposition::NoRetry)` — **schedule() 実行の gate** | 分類が retry decision を支配（Site 3 warmup のみ） |
| RuntimeBuilder.cpp:57 | `classifyBuildErrorToString` | 表示 |

`BuildOutcome` / `FailureClassification` の他の production 使用: **0 件**。`BuildResult.error` と `runtime == nullptr` の二重 failure semantics: Site 1/2 は `runtime == nullptr` のみ検査（error はログ）、Site 3 は build failure を `runtime == nullptr`、warmup を `error != None` で検査 — **二重というより役割分担**だが、Site 1/2 は error 値を retry 判断に使っていない（obligation-level counter が判断）。

## 4. RetryDisposition path

| Disposition | 実装状態 |
|---|---|
| NoRetry | 実質（site 3 warmup gate のみで参照・table 定義） |
| **RetryBackoff** | **未実装** — enum label のみ。delay 計算なし・attempt counter なし・`RetryBackoffPolicy` 構造体なし（src 0 件）・`retryCount` / `retryLatency` / `retryStormDetected` telemetry なし。**「enum が存在することは backoff 実装の証明ではない」がここで実証される** |
| **RetryImmediate** | **明示的実装なし** — WarmupFailed は実際には `schedule(req, milliseconds(0))`（delay 0 = 事実上の即時）で処理される。disposition の意味（latency-sensitive immediate）と実装（delay=0 ハードコード）が**値ではなく偶然一致** |

## 5. WarmupFailed context dependence（3 site 個別監査）

| Site | failure 検知 | retry 方針 | bound | classifyBuildError |
|---|---|---|---|---|
| **Site 1 — transport recovery**（:998-1018） | `recoveryResult.runtime == nullptr` / `validateWarmup` | `postRecoveryFailureSignal(oblId)` → delivery=None → redrive 候補 | **obligation-level K=4**（T3c full-word CAS・CL adjudicate） | **不使用**（error はログのみ） |
| **Site 2 — durable recovery**（:1083-1113） | 同上 | `settlePendingRecoveryAdmission(true)`（DurablePending へ再 lease = retry の構造保証）+ `postRecoveryFailureSignal` | **2 重 bound**: Builder-local spin guard `kMaxRecoveryConsecutiveFailures = 4`（:1067・loop scope・成功で reset・transient のみ加算）+ obligation K=4 | **不使用** |
| **Site 3 — main rebuild warmup**（:1232-1265） | `validateWarmup` | `shouldRetryWarmupFailure`（= `isLoadingIR()`・:80-83）→ classify gate → `retryScheduler_->schedule(req, ms(0))`（fallback: scheduler 無し時 `submitRebuildIntent` 直） | **bound 無し**（recoveryConsecutiveFailures 相当 = 0 件実測） | **使用**（唯一 gate として） |

**context が retryability を決める箇所の特定（ND-06 の重要成果）:**
- Site 1/2: retryability は **obligation identity**（obligationId 単位の failure counter・K=4 で枯渇 → ResolvedFailed）と **Builder-local spin guard** が決める。BuildError 値は使わない。
- Site 3: retryability は **IR loading の進行状態**（`isLoadingIR()`）が決め、BuildError 値は descriptor table の既定 disposition で gate するだけ。counter / backoff / telemetry は無い。
- 「BuildError ごとに retry policy を一意に決めればよい」仮定は**不成立**（WarmupFailed は call-site dependent — これは既存設計どおり。dash2 Amendment 1.8 の AC は Site 3 用に書かれたものであり Recovery site には適用されない）。

## 6. Backoff implementation status

**CR-α の核心 — 未実装を確定:**

| 要素 | 実測 |
|---|---|
| `RetryBackoffPolicy`（state / attempt counter / base delay / exponential calc / max delay / reset） | **src 0 件**（H.11.27.6 の設計のみ — tuning parameter 化設計は完成済み） |
| exponential backoff 計算 | なし（Amendment 1.8 の min 1ms / max 100ms 未実装） |
| retry count（Site 3） | なし（Amendment 1.8「max retry count configurable default 3」未実装） |
| retry telemetry（retryCount / retryLatency / retryStormDetected） | 0 件 |
| `buildErrorCount_` | コメント言及のみ（RuntimeBuilder.h:124） |
| 実際の遅延付き retry | `RetryScheduler::schedule(req, milliseconds(0))` — **delay は 0 固定**（唯一 call site :1256） |

`RetryBackoff → actual delayed retry` は**不成立**。RetryScheduler 自体は deadline-ascending deque + `wait_until` の delayed dispatch 機構として**完成済み**（delay を渡せば動く）— 不足は infrastructure ではなく **delay 計算と bound policy**。

## 7. Retry scheduling path

```text
schedule(req, delay) → deadline ascending deque（capacity 8・溢れ時 rejectCount++）→ notify
    → worker thread（専用 std::thread・wait_until(front.deadline)）→ dispatch_(request)
    → submitRebuildIntent（rebuildAdmissionPendingIntent_ merge / Replaceable latest-wins collapse）
    → rebuild thread 起床
```

- dispatch は **RetryScheduler 専用 worker thread 上**で実行（`unlock before dispatch` コメント実測）— NonRT ✓。
- shutdown: `AudioEngine::~AudioEngine` が rebuildThread 停止前に `retryScheduler_->shutdown()`（明示・idempotent）— shutdown 後 dispatch 防止 ✓。
- capacity 8 溢れは rejectCount++ の静かな破棄 — telemetry getter はあるが HealthEvent 昇格なし。

## 8. RT / NonRT reachability proof（BE-8 再証明）

| 経路 | 実測 |
|---|---|
| Audio callback（AudioBlock / BlockDouble / Snapshot / Retire.cpp）からの `classifyBuildError` / `BuildError` / `RetryDisposition` 参照 | **0 件** |
| RT callback → `submitRebuildIntent` 呼び出し | **0 件**（AudioBlock/BlockDouble 実測 0。RebuildKind 発火は Message thread / Timer / CL / prepareToPlay / scheduler worker 等の NonRT のみ） |
| `submitRebuildIntent` 内の MessageThread jassert | **存在しない**（:151-386 は jassert 無し。`isThisTheMessageThread()` の jassert は UI 系 `requestRebuild(double,int,bool)` :469- 内 — 別関数。RetryScheduler worker → submitRebuildIntent の Debug jassert 潜在問題は**なし**と実測で排除） |
| call graph | Audio Thread →（intent fetchAdd counter のみ）→ CoordinatorLoop / RebuildThread（NonRT）→ build → **BuildError classification → retry decision → schedule**。後半は全部 RebuildThread / scheduler worker 上 |

**BE-8 = 再確認 PASS**: classification / RetryDecision / backoff（未来のもの含む）/ retry scheduling は Audio Thread から到達不能。Audio Thread は intent + bounded accounting（`debugRebuildDispatchRequestCount` 等 fetchAdd counters）のみ。

## 9. Retry bound / liveness

| path | bound | counter owner | reset | obligation 分離 | 無限 loop |
|---|---|---|---|---|---|
| Site 1/2（recovery transport / durable） | **K=4**（`kMaxObligationConsecutiveFailures = 4`・h:401） | obligation lifecycle word（T3c tagged word・`pending`/`adjudicated` field） | terminal resolve で word zero（D149 STOP#5 構造排除）+ slot 再利用時 {N,0} 対原子 | obligationId 単位（CAS が id を保護・slot 再利用時旧 signal は inert） | なし — 枯渇 → `ResolvedFailed`（CL adjudicate・:1118-1147 実測） |
| Site 2（Builder-local spin） | 4（loop scope） | `recoveryConsecutiveFailures` ローカル | build 成功で 0（:1130 付近実測） | durable admission 単位 | なし — 超過で break → 次サイクル委譲（admission は DurablePending 保持） |
| **Site 3（main rebuild warmup）** | **なし** | なし（counter 自体が不在） | — | task generation 単位（Replaceable collapse のみ） | **永続 warmup failure 時に即時 retry loop が継続**（schedule(0) → dispatch → submitRebuildIntent → task → build → warmup fail → schedule(0)…） |

- wake / retry の関係: recovery 系は `adjudicateRecoveryFailureSignals()`（CL-only・Threading.cpp:272 — processIntent 後・redrive 前の配置で同 tick redrive 可能）+ `redriveDeferredRecoveryObligations()` + `consumeRedriveWake()`（D137 event-driven wake・zero-wake when nothing attached）。**durable admission は retry 中も lease として保持**（takePendingRecoveryAdmission = lease、settle(true) で DurablePending 復帰 — lost しない）。
- Site 3 の無限 loop は **CPU/telemetry に影響するが safety には非影響**（NonRT・capacity 8 で queue bound・RT には届かない）。

## 10. Ownership / lifetime impact

CW-8 closure により確定した現行 topology（RuntimeStore / RuntimeWorldAuthority / PublishExecutor / Retire-EBR / Recovery admission / Crossfade）に対する retry 実装の影響:

- **影響 0** — retry は RebuildThread / RetryScheduler worker 上の NonRT 制御フローであり、ownership・lifetime authority に触れない。将来 CR-α 実装時も「retry → new ownership / new queue / new authority」の導入は不要（ScheduleRequest に backoff state を持つか Authority 側 caller が持つかの設計判断のみ）。
- 変更禁止リスト（指示 §8）への接触: なし。

## 11. Contract violations, if any

| # | 事項 | 重度 |
|---|---|---|
| V-1 | **Site 3: retry bound 無し**（Amendment 1.8「max retry count default 3」未実装）→ 永続 warmup failure で無限即時 retry | 中（NonRT bounded-resource 内・safety 非影響だが liveness/telemetry 契約違反） |
| V-2 | **RetryBackoff 未実装**（exponential backoff 1-100ms 未実装・delay=0 固定）| 中（同上・Amendment 1.8 AC 違反） |
| V-3 | **retry telemetry 無し**（retryCount/retryLatency/retryStormDetected/buildErrorCount_） | 低（observability gap） |
| V-4 | **RetryImmediate の意味論未実装**（WarmupFailed が delay=0 経由で偶然即時になっているだけで disposition 駆動でない） | 低（挙動は一致・駆動が table と無関係） |
| V-5 | MKLFailure / ConvolverFailure / PrepareFailure の生成経路なし（table は死に値） | 低（§1.8.9 Phase-2 item 8-10 未実装・既知） |
| 非違反 | Site 1/2 の retry 意味論 | 設計契約（T3c K=4 + durable re-lease）と一致 |
| 非違反 | Site 3 の WarmupFailed context dependency | 既存設計どおり（1.8.5.2 ContextDependent） |

## 12. CR-α implementation recommendation

**Scope（Site 3 限定・最小）:**

1. **Retry bound**: Site 3 warmup retry に attempt counter を追加（consecutive・build 成功で reset・上限は Amendment 1.8 準拠の configurable default 3 または既存 spin guard 4 に揃える）。超過時は retry 打ち切り + telemetry。
2. **Backoff**: H.11.27.6 設計の `RetryBackoffPolicy { initialDelayMs, maxDelayMs, multiplier }` を **tuning parameter** として導入（値は invariant にしない）。既存 `RetryScheduler::schedule(req, delay)` の delay 引数に接続 — **scheduler の変更は不要**（infrastructure 完成済み）。
3. **Telemetry**: retryCount / retryExhausted（Site 3 用）を最小追加。`retryStormDetected` は bound で代替可。
4. **非変更**: RetryScheduler を BuildError-aware にしない（現在どおり classification→disposition→delay は caller 側・BE-8 維持）。MKLFailure/ConvolverFailure/PrepareFailure の生成経路追加は CR-α に含めない（§1.8.9 Phase-2 item 8-10 は別件 — prepare() void の全サブシステム戻り値変更を伴う）。
5. **disposition 駆動の是正**: `RetryImmediate` / `RetryBackoff` が実際に delay 計算へ接続される形にする（table 値を参照する delay 初期値選択）。

## 13. GO / CONDITIONAL-GO / NO-GO

> ## **CONDITIONAL-GO**
>
> 判定基準「実装された意味論と設計契約の一致」について:
> - Recovery 系（Site 1/2）: **一致**（K=4 bound・durable re-lease・obligation counter — 実装済み・回帰テスト済み）。
> - Site 3（main rebuild warmup）: **不一致**（classification はあるが bound/backoff/telemetry が未実装・V-1/V-2/V-4）。
>
| 条件 | 内容 |
|---|---|
| GO の前提 | §12 の scope（Site 3 限定・RetryScheduler 変更なし・ownership 変更なし）を承認すること |
| CONDITIONAL の理由 | Site 3 のみ契約乖離が残る。ただし NonRT bounded・safety 非影響のため緊急性は低い |
| NO-GO 該当 | なし（Site 3 の context dependency は既存設計どおりで、call-site 差異自体は NO-GO 理由にならない — 指示の判定規則どおり） |

実装着手はユーザーの承認後。ND-06 の時点では production/test/CMake/Build/CTest/stress すべて 0。
