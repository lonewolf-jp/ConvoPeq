# D163 — OPEN CR Read-only Boundary / Priority Audit — 作業報告

```text
Date:      2026-09-06
Type:      read-only audit (D163-A CR-α + D163-B CR-β)
Changes:   production/test/CMake/build/build.bat/tools = 0
Baseline:  HEAD 9cacee1f + H5 CMakeLists (+10/-0)。ConvoPeq.md (09-05) は stale → source authority 不使用
判定:      **CR-α = REJECT (Case D — 実装済み・CLOSED済み) / CR-β = REJECT (ALREADY COVERED)。
           実装候補 0 件 → 次の実装タスクなし。インベントリ更新と運用検証への復帰が正しい後処理。**
```

## 要旨

インベントリ (PLAN_DOCS_UNIMPLEMENTED_INVENTORY_20260901.md) の OPEN 2 系統は、**現行 snapshot ではどちらも「未実装」ではない**ことが code-level で確定した:

- **CR-α (BuildError retry backoff/count/telemetry)**: 2026-09-01 に CR-α-1..6 (implementation → verification V1-V16 → CTest 40/40×2 → retry source audit 12/12 → closure 15/15) として **実装・CLOSED 済み** (doc/work88/CRALPHA6_CLOSURE_REPORT.md)。現行 source: `kMaxWarmupConsecutiveRetries=3`・`RetryBackoffPolicy{kDefaultWarmupRetryBackoff={10,80,2}}`・`warmupRetryDecision()`・`retryBackoffDelayMs()` 実装済み、唯一の production call site `schedule(req, decision.delayMs)` (RebuildDispatch.cpp:1294) に policy 接続済み。**「schedule(..., 0ms) = backoff 未接続」記述は stale** — delay=0 は RetryImmediate (WarmupFailed) の intentional bounded immediate retry。
- **CR-β (CW-8 PublishedWorldObservation)**: 型 (private ctor + friend 構造遮断) + factory `observePublishedObservation()` (単一 acquire load から {world, &world->publication} 同時確定) + T-CW8-1..7 test が **commit 0aeb22ca (ND-01..04) で導入済み**。production caller 0 = 保守的休止。CRBETA0 triage (2026-09-01) も ALREADY COVERED と同結論。

## A2 domain 分離 / A5 RT safety

- Site 3 warmup (K=3・{10,80,2}・generation rebind) と Recovery Site 1/2 (obligation K=4 + builder-local guard 4・durable state) は**意図的に別ドメイン** (BuildErrorPolicy.h:86-91 明記・ND-07 §3)。統合禁止が妥当。
- RT safety 責務境界 (RT は待機しない → NonRT RebuildThread → RetryScheduler 専用 worker で待機 → dispatch) は backoff 導入済みの現行でも維持。kCapacity=8 / rejectCount_ / shutdown 順序 (CtorDtor StopWorkers 先頭) まで確認。

## A6 taxonomy

MKLFailure / ConvolverFailure / PrepareFailure は production 生成 site 0 件 (enum+toString+table のみ・保険分類)。RuntimeBuilder.h:120-124 の既決監査 (E-NEXT-6 / Phase D2-0 NO-GO 2026-08-19) が trigger 条件 (subsystem 別 retry 判定の設計確定) まで明示 — D163 では生成実装しない。

## Final

| 項目 | 判定 |
| --- | --- |
| CR-α | **REJECT** (Case D: existing infrastructure satisfies contract) |
| CR-β | **REJECT** (ALREADY COVERED) |
| priority | 実装候補 0 件 — OPEN からの削除 (インベントリ更新) が正しい後処理 |
| next task | **なし** — インベントリ更新 (CR-α→CLOSED・CR-β→ALREADY COVERED 注記) + 運用検証 (D116 系) へ復帰 |

### 棚卸し

1. buildErrorCount_ 集約 counter が唯一の residual (observability-only・trigger 条件付き DEFER — 既決監査 + CRBETA0 B-7 に二重記録済み)。CRBETA0 の「freeze register 補助 trigger 登録 (次の編集 window)」は未実施 → 次回 doc-only 作業候補。
2. 教訓: インベントリ (09-01 16:20 棚卸し) の OPEN 2 系統は同日内の作業で陳腐化していた。以後、棚卸し時は「inventory 日付 < 対象 report 日付」の確認を必須とする。

## 詳細

full report: `evidence/D163/D163_OPEN_CR_BOUNDARY_PRIORITY_AUDIT.md`
evidence: `d163_a0_freeze.json`・`d163_a3_telemetry_census.txt`

## ツール使用記録

- symbol 探索・path 追跡: grep (WSL) + ctx-mode (ctx_batch_execute/ctx_execute) + Read (BuildErrorPolicy.h / RetryScheduler.{h,cpp} / RebuildDispatch.cpp:1040-1330 / RuntimeWorldAuthority.h)
- git -S による実装 commit 特定: 0aeb22ca (CR-α / CW-8 両方)
- serena: 初期指示読込。cppcheck / clang-tidy / Dr.Memory: C++ 変更 0 の read-only 監査のため適用なし。ccc / graphify / semble: path 追跡は grep と serena find_symbol で完結したため不使用。
