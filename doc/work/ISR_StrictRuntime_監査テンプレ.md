# ISR 厳密Closed 監査テンプレ（コピー運用用）

- 対象: `workflow_dispatch(requireRuntimeEvidence=true)` 実行時
- 目的: runtime証跡のみで `R18 / R16 / R25` を判定する

---

## 0) 実行条件（固定）

- Workflow: `ISR Verification`
- Trigger: `workflow_dispatch`
- Input: `requireRuntimeEvidence=true`
- 期待ログ:
  - `[INFO] Strict runtime evidence mode: runtimeRunId=gh-<run_id>-<run_attempt>`
  - `[INFO] Strict runtime evidence mode enabled: generating runtime evidence in CI.`

---

## 1) R18 CI Verification Pipeline（必須）

- [ ] V2〜V10 がすべて PASS
- [ ] `isr-verify-evidence-provenance.ps1` が runtime mode で PASS
  - 期待ログ: `[PASS] evidence provenance gate (runtime mode)`
- [ ] artifact upload が成功（`if-no-files-found: error` を通過）
- 判定メモ:
  - Fail 例: `Strict evidence mode: seeded evidence manifest detected`

---

## 2) R16 HB Failure Spec + Reorder Simulation（必須）

- [ ] `isr-verify-v5.ps1` PASS
- [ ] `hb_graph_trace.json`:
  - [ ] `schema=hb_trace_v1`
  - [ ] `eventCount > 0`
  - [ ] `events.Count == eventCount`
  - [ ] timestamp 単調非減少
- [ ] `hb_violation_report.json`:
  - [ ] `status=ok`
  - [ ] `scenarioResults` が 4件（`forced_reorder`, `epoch_lag`, `retire_delay`, `observe_race`）
  - [ ] 全 scenario が `pass`

---

## 3) R25 DebugRuntime CI限定化（必須）

- [ ] `isr-verify-proof-scope.ps1` PASS
  - 期待: Release=off / Debug=partial / CI=full
- [ ] `isr-verify-runtime-reduction-gate.ps1` PASS
  - 期待ログ: `[PASS] RuntimeReductionGate`
- [ ] strict時に seed 生成を実行していない
  - [ ] 期待ログ: `generating runtime evidence in CI`
  - [ ] 非期待ログ: `isr-seed-evidence.ps1` 実行ログ

---

## 4) 最終判定ラベル

- `R18/R16/R25` が上記すべて PASS -> **Closed（厳密）**
- いずれか欠落/Fail -> **部分適合**（不足項目を明記）

---

## 監査記録（貼り付け欄）

- 実行日時:
- 実行者:
- Workflow Run URL:
- 判定:
- 不足項目（あれば）:
