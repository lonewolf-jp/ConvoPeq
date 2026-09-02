# ND-06 — CR-α BuildError Retry Contract Audit（Work Report）

```text
Production source: 0 / Test source: 0 / CMake: 0 / Build: 0 / CTest: 0 / stress: 0（完全 read-only）
baseline: ConvoPeq.md Generated 2026-09-01 19:31:33（ND-06-1 で --check FRESH / NEWER_SRC_COUNT=0 再実測）
詳細: evidence/ND06_BUILDERROR_RETRY_CONTRACT_AUDIT.md（13 節・実測アンカー付き全文）
```

## 総合判定

> ## **ND-06 = CONDITIONAL-GO**
>
> Recovery 系 2 site（transient / durable）の retry 意味論は設計契約と**一致**（obligation-level K=4 bound + durable re-lease — 実装済み・回帰済み）。
> **乖離は Site 3（main rebuild warmup）に集中**: classification は実装済みだが、retry decision への適用は
> `schedule(req, delay=0)` の gate のみ・retry count / exponential backoff / retry telemetry は未実装 →
> 永続 warmup failure 時に **bound 無しの即時 retry loop**（Site 3 のみ・NonRT bounded・safety 非影響）。

## 監査結果サマリ（13 節の要点）

**1-2. topology / production sites** — `BuildErrorPolicy.h` は契約完全体（8 error / 4 classification / 3 disposition / constexpr table + static_assert ×2）。生成は 4 種のみ: InvalidInput / ResourceUnavailable（bad_alloc）/ InternalError（catch...）/ WarmupFailed（`isIRLoaded && !isIRFinalized`）。**MKLFailure / ConvolverFailure / PrepareFailure は生成経路なし**（table・test のみ）。

**3. classification path** — `classifyBuildError` の production call site は **2 箇所のみ**: Site 3 build failure（:1178・**ログ記録のみ**で制御フローに使われない）と Site 3 warmup（:1247・`retry != NoRetry` が schedule gate として**唯一機能**）。Site 1/2 は `runtime == nullptr` のみ検査し error 値を retry 判断に使わない。

**4. RetryDisposition path** — **RetryBackoff は未実装**（enum label のみ・delay 計算なし・`RetryBackoffPolicy` 0 件・retry telemetry 0 件）。**RetryImmediate も明示実装なし**（WarmupFailed は `schedule(req, milliseconds(0))` の delay=0 で偶然即時になるだけ）。

**5. WarmupFailed context dependence（3 site 個別）** — context が retryability を決める箇所を特定:
| Site | retry 方針 | bound | classify 使用 |
|---|---|---|---|
| Site 1 transport recovery | `postRecoveryFailureSignal(oblId)` → redrive | **obligation K=4**（T3c CAS） | 不使用 |
| Site 2 durable recovery | `settle(true)`（DurablePending 再 lease）+ signal | **2 重 bound**: Builder-local 4（成功で reset）+ obligation K=4 | 不使用 |
| Site 3 main rebuild warmup | `shouldRetryWarmupFailure`（isLoadingIR）→ classify gate → schedule(0) | **なし**（counter 不在） | **使用（唯一）** |

「BuildError ごとに retry policy を一意に決める」仮定は不成立 — WarmupFailed の call-site dependency は既存設計（1.8.5.2 ContextDependent）どおり。

**6. Backoff 実体** — `RetryBackoff → actual delayed retry` は**不成立**。RetryScheduler 自体は deadline-ascending deque + `wait_until` の delayed dispatch 機構として完成済み（delay を渡せば動く）— 不足は infrastructure ではなく **delay 計算と bound policy**。

**7. Retry scheduling** — schedule → capacity 8 deque（溢れ rejectCount++）→ 専用 worker thread → dispatch → `submitRebuildIntent`（merge/Replaceable collapse）→ rebuild thread。shutdown は rebuildThread 停止前に明示 shutdown（idempotent）。

**8. RT/NonRT 到達可能性（BE-8 再証明 PASS）** — Audio callback 系 4 ファイルからの classifyBuildError / BuildError / RetryDisposition 参照 = **0 件**・RT → submitRebuildIntent = **0 件**。call graph: Audio Thread は intent + bounded accounting（fetchAdd counters）のみ。**追加実測**: `submitRebuildIntent`（:151-386）には MessageThread jassert が**存在しない**（jassert は UI 系 `requestRebuild` :469- 内の別関数）— RetryScheduler worker → submitRebuildIntent の Debug jassert 潜在問題は排除。

**9. bound / liveness** — Recovery 系: obligation lifecycle word（T3c tagged word）単位の K=4・CL adjudicate（Threading.cpp:272・processIntent 後 redrive 前の同 tick 配置）で枯渇 → `ResolvedFailed`・terminal で word zero（D149 STOP#5 構造排除）・durable admission は retry 中も lease 保持（lost しない）・redrive wake は D137 event-driven（zero-wake）。**Site 3 のみ bound 無し**。

**10. ownership / lifetime** — retry は NonRT 制御フローであり CW-8 で確定した topology への影響 0。CR-α 実装でも new ownership / new queue / new authority は不要。

**11. violations** — V-1 Site 3 retry bound 無し（中）/ V-2 RetryBackoff 未実装（中）/ V-3 retry telemetry 無し（低）/ V-4 RetryImmediate 意味論未接続（低）/ V-5 MKLFailure 等の生成経路なし（低・既知）。いずれも NonRT bounded で safety 非影響。

## 12. CR-α 実装推奨（Site 3 限定・最小 scope）

1. **Retry bound**: Site 3 warmup retry に consecutive counter（build 成功で reset・上限 configurable default 3 または spin guard 4 に揃える）。
2. **Backoff**: H.11.27.6 設計の `RetryBackoffPolicy { initialDelayMs, maxDelayMs, multiplier }` を tuning parameter として導入し、**既存 `RetryScheduler::schedule(req, delay)` の delay 引数に接続**（scheduler 変更不要）。
3. **Telemetry**: retryCount / retryExhausted（Site 3 用）最小追加。
4. **非変更**: RetryScheduler を BuildError-aware にしない（BE-8 維持）。MKLFailure 等の生成経路追加（prepare() void の全面変更）は CR-α に含めない。
5. **disposition 駆動の是正**: RetryImmediate / RetryBackoff が delay 計算に接続される形にする。

## 13. 判定

> **CONDITIONAL-GO** — Recovery 系は設計契約と一致。Site 3 のみ乖離（V-1/V-2/V-4）。NO-GO 該当なし（Site 3 の context dependency は既存設計どおりで call-site 差異自体は NO-GO 理由にならない）。GO の前提 = §12 scope（Site 3 限定・scheduler 変更なし・ownership 変更なし）の承認。

実装着手はユーザー承認後。本監査では production/test/CMake/Build/CTest/stress すべて 0。
