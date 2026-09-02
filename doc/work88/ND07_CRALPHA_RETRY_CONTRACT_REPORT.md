# ND-07 — CR-α Site 3 Retry Contract Finalization（Work Report）

```text
Production source: 0 / Test source: 0 / CMake: 0 / Build: 0 / CTest: 0 / stress: 0（完全 read-only）
baseline: ConvoPeq.md Generated 2026-09-01 19:31:33（--check FRESH / NEWER_SRC_COUNT=0 再実測）
前提: ND-06 = CONDITIONAL-GO（V-1〜V-4 のうち Site 3 が対象）
契約書全文: evidence/ND07_CRALPHA_SITE3_RETRY_CONTRACT.md（14 節）
```

## 総合判定

> ## **ND-07 = GO — CR-α 実装の判断余地をゼロにした Site 3 専用 retry contract を確定**

## 確定した契約の要点（14 節対応）

**① Site 3 state owner** — `rebuildThreadLoop` 関数スコープ変数（RebuildThread 単一所有・mutex/atomic 不要・Site 2 の `recoveryConsecutiveFailures` と同一パターン）。ScheduleRequest（4-field 固定）・RetryScheduler には持たせない。collapse（Replaceable latest-wins）は intent 送信側の burst 吸収で retry attempt 計数と構造的に独立。

**② attempt semantics** — consecutive failure = warmup failure かつ `shouldRetryWarmupFailure`（isLoadingIR・ContextDependent）かつ !obsolete。reset: publish 到達 / generation 変更。

**③ max bound = 3（決定）** — 根拠: dash2 Amendment 1.8 の normative default「configurable, default 3」＋ **recovery の K=4 と意図的に別値**（値の一致は grep/telemetry 読解時の混同を隠すため・定数名 `kMaxWarmupConsecutiveRetries` で Site 3 専用を明示）。Recovery の obligation K=4 とは別ドメイン・混同禁止を契約化。

**④ BackoffPolicy — normative default を確定: 設計記録の 10→20→40→80ms を採用**（`{initialDelayMs=10, maxDelayMs=80, multiplier=2}`・constexpr）。tuning/invariant 分離は H.11.27.6 設計どおり（invariant = NonRT / non-blocking / bounded / caller-side counter）。saturation は uint64 中間計算 + 上限早期 return で overflow を構造排除。attempt 1/2/3 = 10/20/40ms（max 3 のため 80 は cap）。**配置は `BuildErrorPolicy.h`**（JUCE 非依存・standalone contract test 可能 → CMake 変更不要）。

**⑤ disposition→delay mapping** — NoRetry=schedule しない / **RetryImmediate=delay 0**（WarmupFailed・latency-sensitive）/**RetryBackoff=policy delay**。**WarmupFailed は無条件 retry ではない**（context gate `isLoadingIR()` 必須・enum を無条件 retry と解釈する設計は禁止）。RetryBackoff は接続済みだが現行 dormant（Site 3 で retry 対象は WarmupFailed のみ・build failure retry は将来拡張として保留）。

**⑥ generation/obsolete** — counter は generation 紐付け（新 generation = retry state 再開・exhaustion は永続 mask ではない）。obsolete → schedule しない。exhaustion 後の同 generation 再発行は構造的に不可能（schedule しない = 新規 intent 発行なし → collapse による隠れた無限再発行も排除）。`!wokeByPendingTask → continue`（D132 防御・実測）と合致。

**⑦ scheduler 接続** — 既存 `schedule(req, delay)` の delay 引数のみ使用・**RetryScheduler infrastructure は 1 行も変更しない**。`schedule()` は void（rejection 非通知）のため reject 時は attempt 消費の retry drop とする（再 queue storm 避免の簡潔規則・契約明記）。

**⑧ exhaustion behavior** — max 超過 → schedule しない + `submitRebuildIntent` 再発行なし + terminal telemetry 1 回（同 generation 重複なし）+ diagLog（generation/error/attempt）。新規 request（新 generation）で counter 0 から再開。

**⑨ telemetry 最小化** — `RebuildTelemetryReason` に 2 値追加（WarmupRetryScheduled / WarmupRetryExhausted・additive・既存 static_assert 網羅パターン準拠）+ diagLog。**retryStormDetected は必須にしない判定**（bounded retry が構造的に代替）。

**⑩ BE-8 維持** — 追加コードは BuildErrorPolicy.h（純関数）+ RebuildDispatch.cpp（RebuildThread）のみ。Audio Thread との到達不能性は ND-06 §8 実測どおり維持。

**⑪ ownership** — 影響 0（counter は RebuildThread stack・policy constexpr・decision 純関数）。

**⑫ forbidden changes** — Site 1/2 recovery・obligation K=4・spin guard 4・RetryScheduler infrastructure・RuntimeStore/EBR/retire・MKLFailure 等の生成経路・prepare() status propagation・Site 3 build failure の retry 化・新 authority/queue/atomic・ReadToken/CW-8・RetryScheduleRequest field 追加 — すべて対象外として固定。

**⑬ test contract** — T-CRα-1〜6: backoff 純関数（10/20/40/80・saturation）・normative default 値・warmup retry decision・disposition→delay mapping・exhaustion で submit 再発行なし・telemetry enum 網羅。**追記先 `BuildErrorClassificationTests`（既存 target・CMake 変更なし）**。実装推奨: retry decision を**純関数**（BuildErrorPolicy.h 内）にして rebuildThreadLoop は実行のみ — T-CW8-7 と同じ「decision の unit test 可能化」パターン。

**⑭ 判定** — **GO**。判断余地の残存なし（§1-9 が実装コードに直接写像）。NO-GO 条件発火なし。

## 実装規模見込み（CR-α-1 前提）

| ファイル | 内容 | 見込み |
|---|---|---|
| `src/audioengine/BuildErrorPolicy.h` | RetryBackoffPolicy + retryBackoffDelayMs + warmup retry decision 純関数 | +40 行前後 |
| `src/audioengine/AudioEngine.RebuildDispatch.cpp` | Site 3 counter（generation 紐付け）+ 純関数呼び出し + schedule delay 接続 | +25 行前後 |
| `src/audioengine/RetrySchedulerTypes.h` | RebuildTelemetryReason 2 値追加 | +2 行 |
| `src/tests/BuildErrorClassificationTests.cpp` | T-CRα-1〜6 | +80 行前後 |
| CMake | **0** | — |

## 遷移

```text
ND-06 CONDITIONAL-GO
   ↓
ND-07 = GO（本契約書）   ← 実装判断余地ゼロ
   ↓
CR-α-1 最小実装 → CR-α-2 targeted tests → CR-α-3 Debug/Release build → CR-α-4 CTest + regression
   ↓
CR-α closure
```
