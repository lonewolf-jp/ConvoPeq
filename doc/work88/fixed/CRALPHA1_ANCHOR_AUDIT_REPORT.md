# CR-α-1 実装前 Source Anchor Audit（Work Report）

```text
Production source: 0 / Test source: 0 / CMake: 0 / Build: 0 / CTest: 0 / stress: 0（完全 read-only）
baseline: ConvoPeq.md Generated 2026-09-01 19:31:33（--check FRESH / NEWER_SRC_COUNT=0 実測）
対象: 4 ファイル（BuildErrorPolicy.h / AudioEngine.RebuildDispatch.cpp / RetrySchedulerTypes.h / BuildErrorClassificationTests.cpp）
詳細: evidence/CRALPHA1_SOURCE_ANCHOR_AUDIT.md（exact anchor 表 + 変更予定行リスト）
```

## 総合判定

> ## **Anchor Audit = PASS — CR-α-1 コード変更への進行可**

ND-07 契約 §1〜§13 を実コードと 1 項目ずつ突合し、変更予定行を確定した。

## exact anchor 確定（主要点）

- **BuildErrorPolicy.h**: JUCE 非依存 standalone ヘッダ（「no external framework」コメント実測）— RetryBackoffPolicy / `retryBackoffDelayMs` / `warmupRetryDecision` 純関数の追加先として適合。追加位置は namespace 閉じ前。
- **AudioEngine.RebuildDispatch.cpp**: counter 挿入 = `while (true)`（:837）直前。generation rebind = `if (!task.runtimeBuildSnapshot.sealed) continue;` の直後。置換対象 = warmup block :1232-1265（`validateWarmup` → `retryable` → classify gate → `schedule(req, ms(0))` :1255 → fallback）。Site 3 build failure（:1171-1188）と Site 2 spin guard（:1067）は無変更。
- **BuildErrorClassificationTests.cpp**: standalone main + CHECK マクロ + runTestA〜E 構造 — runTestF 追加で完全適合。
- **既存 `RebuildThreadWarmupRetry`** は RetrySchedulerTypes.h:20 に現役（schedule request の reason）。削除・置換しない。

## 発見（footprint 調整 1 件 — F-1）

ND-07 §9 の「`RetrySchedulerTypes.h` に telemetry enum 2 値追加」は、実測では **enum→toString switch が `AudioEngine::toTelemetryReasonString`（AudioEngine.h:2863-2892）に在り**、enum 追加は AudioEngine.h 変更（footprint 外）を必然化する。さらに `RebuildTelemetryEvent` に retry 対応 event 値が存在しない（追加すると enum 変更 2 件になる）。

**推奨解（audit による確定）: telemetry は diagLog-only とし、enum 追加を取りやめる。** 対象 2 site には既存 diagLog が実在し、ND-07 §9 の本質（attempt/exhausted/generation/error の追跡可能性）は diagLog で完全充足。RetrySchedulerTypes.h と AudioEngine.h は **無変更**。T-CRα-6 は「diagLog 項目の source inspection」に置換。

補足 findings: F-2（toTelemetryReasonString に static_assert 機構なし — 将来リスクとして記録）/ F-3（generation sentinel = -1・task.generation は int 非負）/ F-4（schedule() は void → reject 時は attempt 消費の retry drop・ND-07 §7 規則どおり）。

## 変更予定行リスト（CR-α-1 実装 map）

| # | ファイル | 行 | 操作 |
|---|---|---|---|
| 1 | BuildErrorPolicy.h | 末尾 | 追加: kMax=3 / RetryBackoffPolicy / kDefault{10,80,2} / retryBackoffDelayMs / WarmupRetryAction / WarmupRetryDecision / warmupRetryDecision |
| 2 | RebuildDispatch.cpp | :837 直前 | 追加: counter 3 変数（boundGeneration=-1 / count=0 / exhausted=false） |
| 3 | RebuildDispatch.cpp | :1170 直後 | 追加: generation rebind |
| 4 | RebuildDispatch.cpp | :1235-1263 | 置換: decision 関数呼び出し（Schedule → schedule(req, delay) / Exhausted → 1 回 diagLog / NoRetry → nothing） |
| 5 | BuildErrorClassificationTests.cpp | 末尾 | 追加: runTestF（T-CRα-1〜4 + saturation + default 値） |
| — | RetrySchedulerTypes.h / AudioEngine.h | **無変更** | — |

## 判定

**PASS — CR-α-1 実装へ進行可。** 次ターンから本 audit の変更 map に従い CR-α-1（4 ファイル・純追加主体）→ CR-α-2（T-CRα-1〜4 + source inspection）→ CR-α-3（Debug/Release build）→ CR-α-4（CTest 40/40 現行基準）→ CR-α closure の順で進める。
