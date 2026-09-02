# CR-α-1 実装前 Source Anchor Audit（read-only）

```text
Date: 2026-09-01
Production source: 0 / Test source: 0 / CMake: 0 / Build: 0 / CTest: 0 / stress: 0（完全 read-only）
baseline: ConvoPeq.md Generated 2026-09-01 19:31:33（--check FRESH / NEWER_SRC_COUNT=0 実測）
対象: ND-07 契約（evidence/ND07_CRALPHA_SITE3_RETRY_CONTRACT.md §1〜§13）と実コードの 1 項目ずつ突合 + 変更予定行の確定
対象ファイル（指示どおり 4 件）: BuildErrorPolicy.h / AudioEngine.RebuildDispatch.cpp / RetrySchedulerTypes.h / BuildErrorClassificationTests.cpp
```

## 総合判定

> ## **Anchor Audit = PASS（1 件の footprint 調整を確定）**
>
> ND-07 §1〜§13 は全て現行コードに anchor を持つ。**1 件のみ契約 footprint の修正が必要**:
> `RebuildTelemetryReason` の enum→toString switch は **AudioEngine.h:2863-2892** に在り、
> ND-07 §9 の「enum 2 値追加」は AudioEngine.h への case 追加を必然化する（ND-07 の
> footprint 4 ファイルから漏れていた）。**推奨解 = telemetry を diagLog-only にして
> enum 追加を取りやめる**（footprint 4 ファイルを維持・詳細 §F-1）。CR-α-1 は本調整込みで GO。

---

## 1. 実装直前 exact anchor（4 ファイル実測）

### ① src/audioengine/BuildErrorPolicy.h（追加対象）

| anchor | 行 | 内容 |
|---|---|---|
| ファイル性格 | 1-9 | JUCE 非依存 standalone contract test 用ヘッダ（「no external framework」コメント実測）— RetryBackoffPolicy / decision 純関数の追加先として適合 |
| `BuildError` | 12-21 | 8 値（None/InvalidInput/ResourceUnavailable/MKLFailure/ConvolverFailure/PrepareFailure/WarmupFailed/InternalError） |
| `RetryDisposition` | 30-34 | NoRetry / **RetryBackoff** / **RetryImmediate** |
| `BuildOutcome` | 36-40 | error/classification/retry |
| `kBuildErrorDefaultTable` | 43-52 | 全 8 値 + static_assert ×2（:53-61） |
| `classifyBuildError` | 64-71 | bounds-checked table lookup |
| 追加位置 | ファイル末尾（`} // namespace convo` の前） | RetryBackoffPolicy / kDefaultWarmupRetryBackoff / retryBackoffDelayMs / WarmupRetryAction / WarmupRetryDecision / warmupRetryDecision |

### ② src/audioengine/AudioEngine.RebuildDispatch.cpp（置換対象）

| anchor | 行 | 内容 / 変更 |
|---|---|---|
| `shouldRetryWarmupFailure` | :80-83 | `return dsp.convolverRt().isLoadingIR();` — ContextDependent 判定の実体。**無変更**（decision 関数の context 入力として使用） |
| `rebuildThreadLoop` 開始 | :824 | 関数スコープ counter の挿入位置: `while (true)`（**:837**）の直前に `warmupRetryBoundGeneration(-1) / warmupRetryCount(0) / warmupRetryExhausted(false)` を宣言（RebuildThread 単一所有・mutex 不要） |
| generation rebind 挿入位置 | :1169-1170 の直後（`if (!task.runtimeBuildSnapshot.sealed) continue;` の次） | `task.generation != warmupRetryBoundGeneration → count=0 / exhausted=false / bound 更新` |
| **Site 3 build failure** | :1171-1188 | `runtime == nullptr` → classify（ログのみ）→ `continue`。**変更なし**（retry 化は将来拡張保留） |
| **Site 3 warmup failure** | :1232-1265 | **置換対象 block**: :1232 `validateWarmup` / :1235 `retryable` / :1247 `classifyBuildError` / :1248 gate / :1255 `schedule(req, milliseconds(0))` / :1256-1263 fallback / :1265 `continue` |
| 置換内容 | — | ①`++warmupRetryCount` ②`warmupRetryDecision(count, kMax=3, retryable, isObsolete(), outcome.retry, kDefaultWarmupRetryBackoff)` ③ Schedule → `schedule(req, ms(decision.delayMs))` ④ Exhausted → `warmupRetryExhausted` transition 時 1 回 diagLog ⑤ NoRetry → なし。fallback（scheduler nullptr）経路は decision Schedule 時のみ維持 |
| spin guard（Site 2） | :1067 `kMaxRecoveryConsecutiveFailures = 4` | **無変更**（recovery 専用・Site 3 とは別ドメイン） |

### ③ src/audioengine/RetrySchedulerTypes.h（**変更取りやめ** — §F-1）

| anchor | 行 | 内容 |
|---|---|---|
| `RebuildTelemetryReason` | :9-46 | 36 値・uint8_t。`RebuildThreadWarmupRetry` は :20 に**既存**（schedule request の reason として現役） |
| enum→toString | **AudioEngine.h:2863-2892** | `AudioEngine::toTelemetryReasonString` switch（`return "unknown_reason"` fallback・**static_assert 機構なし**） |

### ④ src/tests/BuildErrorClassificationTests.cpp（追記対象）

| anchor | 行 | 内容 |
|---|---|---|
| 構造 | 1-末尾 | standalone main()（JUCE 不使用）+ `check/CHECK` マクロ + runTestA〜E + fail カウンタ — T-CRα-1〜4 の純関数テスト追記先として完全適合 |
| include | :8 | `audioengine/BuildErrorPolicy.h` のみ — 追加純関数は同ヘッダから見える |
| main | 末尾 | `runTestA..E` の and 結合 → runTestF（T-CRα-1〜4）を 1 関数で追加・and に追加 |

## 2. ND-07 §1〜§13 との突合結果

| ND-07 節 | 現行コードとの整合 | 備考 |
|---|---|---|
| §1 state owner（rebuildThreadLoop 関数スコープ） | ✅ :824/:837 に anchor。RebuildThread 単一所有（Site 2 前例 :1067 同パターン） | 挿入行確定 |
| §2 attempt semantics（context + !obsolete = 1 attempt） | ✅ :1235 `shouldRetryWarmupFailure` / isObsolete ラムダ使用可 | decision 関数の入力 |
| §3 max = 3（K=4 と別値） | ✅ :1067 は Site 2 専用・名前で分離 | `kMaxWarmupConsecutiveRetries = 3` を BuildErrorPolicy.h に追加 |
| §4 policy {10,80,2} + saturation | ✅ 追加のみ・依存なし | BuildErrorPolicy.h 末尾 |
| §5 disposition→delay mapping | ✅ :1247-1248 gate を decision 関数に置換 | Immediate=0 / Backoff=policy / NoRetry=none |
| §6 generation 紐付け | ✅ :1169-1170 直後に rebind / task.generation は int | sentinel = -1（generation は非負） |
| §7 scheduler 接続（delay 引数のみ） | ✅ :1255 が唯一の schedule call site | `schedule()` は void → reject 時 attempt 消費（ND-07 §7 規則） |
| §8 exhaustion | ✅ :1232-1265 内に既存 counter/bound なし（0 件実測）→ 新設の余地あり | exhausted flag は関数スコープ |
| §9 telemetry | ⚠️ **footprint 調整**（§F-1） | diagLog-only を推奨 |
| §10 BE-8 | ✅ 追加コードは RebuildThread / 純関数のみ | ND-06 §8 実測の再確認 |
| §11 ownership | ✅ 影響 0 | — |
| §12 forbidden | ✅ 全 12 項目・現行 diff に接触なし | — |
| §13 test contract | ✅ BuildErrorClassificationTests 構造が適合（standalone main + CHECK） | runTestF 追加 |
| — | D132 防御（`!wokeByPendingTask → continue`）・`RebuildTask.generation`（AudioEngine.h:2740 int）・pendingTask 単一スロット転送 | exhaustion 永続 mask 非該当の根拠 |

## 3. Anchor audit findings

### F-1（footprint 調整）: telemetry enum 追加は AudioEngine.h 変更を必然化する

ND-07 §9 は「`RetrySchedulerTypes.h` に enum 2 値追加」を規定したが、実測の結果:

- `RebuildTelemetryReason` の enum→string は **`AudioEngine::toTelemetryReasonString`（AudioEngine.h:2863-2892）の switch** にあり、static_assert 機構は存在しない。
- enum 値を追加すると **AudioEngine.h の switch に 2 case を追加しない限り** `emitRebuildTelemetry` が "unknown_reason" を出力する（= footprint 5 ファイル化）。
- さらに `emitRebuildTelemetry` の `RebuildTelemetryEvent` には retry に対応する event 値が存在しない（Suppressed/Deferred/ForcedDispatch/Dispatched のみ）— 追加すると 2 つの enum 変更になる。

**推奨解（本 audit による確定）: telemetry は diagLog-only とし、enum 追加を取りやめる。**

- 根拠: (a) 対象 2 site には既存 diagLog が実在し拡張形式も同一、(b) AudioEngine.h 変更（ND-07 footprint 外）を回避、(c) ND-07 §9 の本質（attempt/exhausted/generation/error の追跡可能性）は diagLog で完全に充足、(d) RetrySchedulerTypes.h は **無変更**（既存 `RebuildThreadWarmupRetry` は schedule request の reason として現役維持 — 削除・置換しない）。
- T-CRα-6（telemetry 網羅）は「enum↔toString 網羅」から「diagLog 項目の source inspection（attempt/limit/generation/exhausted フィールドの実在）」に置換。

### F-2: `toTelemetryReasonString` に static_assert 機構がない

既存 switch は全数保証なし（36 値手書き）。CR-α では追加しない（ footprint 維持）。将来の enum 変更時のリスクとして記録。

### F-3: generation sentinel

`task.generation` は int（非負・rebuildRequestGeneration 由来）。bound 初期値は `-1`（実 generation と非一致の sentinel）。初回 task で必ず rebind が発火する。

### F-4: `schedule()` が void

capacity 満杯 / shutdown reject は非通知。契約（ND-07 §7）どおり「attempt 消費の retry drop」で処理（巻き戻しなし）。

## 4. 変更予定行リスト（CR-α-1 実装の確定 map）

| # | ファイル | 行 | 操作 |
|---|---|---|---|
| 1 | BuildErrorPolicy.h | 末尾（namespace 閉じ前） | 追加: `kMaxWarmupConsecutiveRetries = 3` / `RetryBackoffPolicy` / `kDefaultWarmupRetryBackoff{10,80,2}` / `retryBackoffDelayMs` / `WarmupRetryAction` enum / `WarmupRetryDecision` / `warmupRetryDecision` 純関数 |
| 2 | RebuildDispatch.cpp | :837 直前 | 追加: counter 3 変数宣言（boundGeneration=-1 / count=0 / exhausted=false） |
| 3 | RebuildDispatch.cpp | :1170 直後 | 追加: generation rebind（count=0 / exhausted=false / bound 更新） |
| 4 | RebuildDispatch.cpp | :1235-1263 | 置換: decision 関数呼び出し（`++warmupRetryCount` → decision → Schedule: schedule(req, delay) / Exhausted: 1 回 diagLog / NoRetry: nothing）。fallback は Schedule 時のみ維持 |
| 5 | BuildErrorClassificationTests.cpp | 末尾 | 追加: runTestF（T-CRα-1〜4 + saturation + default 値検証）+ main の and 結合 |
| — | RetrySchedulerTypes.h | **無変更**（F-1 推奨解） | — |
| — | AudioEngine.h | **無変更**（F-1 推奨解） | — |

## 5. 判定

> ## **Anchor Audit = PASS**
> ND-07 §1〜§13 は全て実コードに anchor を持ち、変更予定行が確定した。
> F-1（telemetry footprint 調整 — diagLog-only 推奨）を CR-α-1 の確定仕様に繰り込めば、
> 実装は 4 ファイル・純追加主体・既存 API 無変更で実施可能。
> **CR-α-1 コード変更への進行可（ユーザー承認済みの契約 + 本 audit の調整 1 件）。**
