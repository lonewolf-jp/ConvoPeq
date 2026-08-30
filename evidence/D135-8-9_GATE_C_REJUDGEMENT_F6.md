# D135-8/9 Gate C 再判定 — F6 実装後の Debug/Release CTest

Date: 2026-08-30
前提: F1（root cause F1-B）→ F2（retry≠retention）→ F3（TTL churn 失効）→ F4（identity=(gen,oblId)）→ F5（GO）→ **F6 実装（必須5点のみ・test source 0変更）** → F6-B0（PASS 12/12）→ 本再判定。

## Verdict: **Gate C = PASS（Debug 40/40 + Release 40/40 + Release harness ×5 決定論 PASS）**

| 段階 | 結果 | 証拠 |
| --- | --- | --- |
| Debug build | PASS（`GATB_CMAKE_EXIT=0`） | `evidence/D135-8-9_GATE_B_DEBUG_BUILD_LOG.txt`（F6 後更新） |
| Debug CTest | **40/40 PASS**（`GATC_CTEST_EXIT=0`） | `evidence/D135-8-9_GATE_C_DEBUG_CTEST_LOG.txt` |
| Release build | PASS（`GATB_CMAKE_EXIT=0`） | `evidence/D135-8-9_GATE_B_RELEASE_BUILD_LOG.txt`（F6 後更新） |
| **Release CTest** | **40/40 PASS**（`GATC_CTEST_EXIT=0`、失敗 0） | `evidence/D135-8-9_GATE_C_RELEASE_CTEST_LOG.txt` |
| Release harness ×5 | **全 EXIT=0 / FAIL 0 / "all publish pipeline tests PASS" ×5** | 直接実行（決定論確認） |

## Gate C 失敗（F6 前）→ 解消の対比

| 指標 | F6 前 | F6 後 |
| --- | --- | --- |
| Release `AudioEngineHarness` (#40) | ***Failed 60.81 sec**（ctest exit 8） | **Passed 16.46 sec** |
| `testDeferredBacklogDrainsCompletely` | "cycle 1 did not defer"（3/3 決定論失敗） | PASS（2x deferred cycles drained） |
| Release harness 直接実行 | exit 1 ×3 | exit 0 ×5 |

## 解消機序（F6 実装内容との対応）

- **F6-1/2**: `enqueueDeferred` の identity を `(generation, recoveryObligationId)` に拡張し、
  `DeferredFadingActive` の再 enqueue（retention）では `deferredRetryCount_` を増加させない。
  → cycle 1 の re-defer churn（1ms coordinator 再駆動）で count が増えず、
  **`RetryExhaustedDiscard` が発火しなくなった**（F1-B の直接原因を除去）。
- **F6-3**: `metadata.enqueueTimestampUs = deferredObligationCreatedAtUs`（re-drive で維持）。
  → TTL が obligation dwell を正しく測る（F3 の churn 失効バグを修正）。
- **F6-4**: fade-complete wake（Timer.cpp fadeCompleted 末尾、既存 handoff 同型・recoveryRetryReady 非汚染）。
  → fading 解消を即時に RebuildThread へ通知（正常経路）。
- **F6-5**: coordinator poll を watchdog 化（`kDeferredWakeWatchdogTicks`）。
  → 毎 tick churn が消え、lost-wake 自己修復のみ残る（churn bound）。
- **F6-7**: terminal（Accepted / StaleDiscard / Expired / ShutdownDiscard / recovery reset）で
  obligation metadata を無効化。

## 制約遵守

- production source 変更: F6 必須5点 + F6-7/8（契約文書）のみ。`kMaxDeferredRetries` 削除なし（dormant 保持）。
- **test source 変更: 0**（`DeferredFlowIntegrationTests.cpp` 無編集 — 既存 2-cycle テストがそのまま PASS）。
- 禁止事項（Expired 活性化 / Rejected* 再設計 / telemetry schema / CV predicate / recovery provenance /
  recovery obligation table / DSP retire pipeline）: いずれも未変更（F6-B0 §契約遵守チェック参照）。

## 残存・次フェーズ

- **Gate D（retry state-machine test）**: F6 で retention は budget 外になったため、Gate D の検証対象は
  「ordinary wake で count が増えない / recovery wake で reset / 将来 Type A 導入時のみ count++」に再定義が必要。
  F5 §F4-6/F5-9 の最終状態表がそのまま受け皿になる。
- 別パッチ候補（F5 §D #6-8）: Expired enum 活性化 / Rejected*（non-recovery）の handle 回収 / telemetry dwell 化。
- 未確定（低優先）: D135-1 文書の「Release 40/40 PASS」当時と F6 前の決定論失敗の差分帰属
  （D135-8 codegen レース反転 or 当時 kMax 実効値ドリフト）。F6 で race 自体が契約上無害化済みのため
  実務上の残課題ではない。

## 成果物一覧（F6 一式）

- `evidence/D135-8-9_GATE_C_F6_B0_DIFF_AUDIT.md`（read-only diff audit 12/12）
- 本再判定書
- 更新済み CTest ログ（Debug/Release）
- 実装: Orch.cpp/.h、AudioEngine.h、Timer.cpp、Threading.cpp、State.h、D135-1 文書
