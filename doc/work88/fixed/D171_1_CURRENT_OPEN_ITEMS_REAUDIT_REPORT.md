# D171-1 — Current Open Items Re-audit Report

- Date: 2026-09-07
- Task: D169-2 CLOSED 後の次工程選定 read-only preflight（OPEN Inventory Reconciliation）
- Scope: `PLAN_DOCS_UNIMPLEMENTED_INVENTORY_20260901.md` × 最新 `ConvoPeq.md`（Generated 2026-09-07 21:38:42）照合 / D159 freeze register 再分類 / MEM_SNAP hazard 独立 audit
- Type: read-only（source 変更 0 / テスト実行 0 / inventory 改変 0）
- Evidence: `evidence/D171/D171_1_CURRENT_OPEN_ITEMS_REAUDIT.md`

## 判定

> ## **D171-1 PASS — 着手可能な OPEN implementation 項目は 0 件。9/1 inventory の 1-C 2 件は両方 STALE。唯一の新規調査対象は MEM_SNAP hazard（D172-1 audit 候補）。**

| Candidate | Inventory 9/1 | Current source | Freeze | Current verdict |
| --- | --- | --- | --- | --- |
| CW-8 PublishedWorldObservation | UNIMPLEMENTED | **実装済み**（RuntimeWorldAuthority.h 型+factory `observePublishedObservation()` + T-CW8-1..7 テスト、commit 0aeb22ca） | 未登録 | **STALE** |
| BuildError retry/backoff（CR-α 本体） | UNIMPLEMENTED | **実装済み**（`RetryBackoffPolicy{kDefaultWarmupRetryBackoff={10,80,2}}`・`warmupRetryDecision()`・唯一 production call site RebuildDispatch.cpp:1294 接続・T-CRα-1..4、CR-α-1..6 CLOSED） | 未登録 | **STALE** |
| BuildError telemetry（buildErrorCount_） | UNIMPLEMENTED | **未実装**（RuntimeBuilder.h:124 コメントのみ・src 0 hits） | DEFER（D163 §39 / CRBETA0 B-7 trigger 条件付き・二重記録済み） | **DEFER（observability-only）** |
| Site 2 build failure retry 適用 | UNIMPLEMENTED | 分類 diagLog のみ（:1212-1213 将来拡張コメント・dash2 §1.8 Phase D 通り） | 未登録 | **DEFER（設計通り）** |
| MEM_SNAP dangling hazard | inventory 外 / 新規 | **経路現存**（Timer.cpp:1078-1088） | — | **AUDIT REQUIRED → D172-1** |
| D1 R1 MPSC / D2 Supersession / D3 C3C4 gates / D4 sparse / D5 X2 / D6 static bound | DEFER | 各 trigger 未発生（D1: 第2 producer 0 件・D2: isSemanticSuperset 0 hits・D3: dormant 現役・D4: completedOutOfOrder 0 hits・D5: INV-X2-6 維持・D6: doc-level） | D1〜D6 | **DEFER 維持** |

## 根拠（要約）

1. **CW-8 は実装・テスト済み** — 型（private ctor + nothrow copy）+ factory `observePublishedObservation()`（単一 acquire load から `{world, &world->publication}` 同時確定）+ `testCW8_PublishedWorldObservation`（T-CW8-1..7 static_asserts）が現行 source に実在。commit 0aeb22ca（ND-01..04）。D163 は CR-β = REJECT (ALREADY COVERED) 判定済み。**新規 CW-8 track は起票しない**。
2. **BuildError backoff も実装済み** — inventory 1-C-1 の「現行は即時 retry（delay=0）」記述は陳腐化。現行は `kDefaultWarmupRetryBackoff={10,80,2}` が唯一の call site `schedule(req, decision.delayMs)` に接続され、T-CRα-1（delay table 0/10/20/40/80/80 飽和）〜 T-CRα-4 で検証済み。CR-α-1..6 が 2026-09-01 に CLOSED。D163 は CR-α = REJECT (Case D)。ユーザー指摘のとおり既存 bounded 構造（`kMaxWarmupConsecutiveRetries` / Recovery 系 K=4 別ドメイン / RetryScheduler）が安全に機能しており、**実装への着手は不適**。
3. **唯一の residual は `buildErrorCount_` telemetry** — observability-only・trigger 条件付き DEFER として D163 §39 と CRBETA0 B-7 に二重記録済み。D171-2 preflight 対象外。CRBETA0 の「freeze register 補助 trigger 登録」は未実施のまま → 次回 doc-only 作業候補（D171-1 では inventory 改変しないため実施せず記録）。
4. **D159 freeze register 全件妥当** — D1〜D6 + 補助 trigger（RecoveryEpisodeId / pendingRecoveryAdmission_ MPSC 化 / stale コメント清掃）とも trigger 発生なしを source-wide grep で再実測。
5. **MEM_SNAP hazard（P4）** — `activeRuntimeDSPSlot` は非所有 topology mirror（placeholder 専用・D169-1R RC-D169-1-2「ownership authority に昇格させない」）。writer 4 箇所（CtorDtor:153 / PrepareToPlay:287,318 / ReleaseResources:180）の完全列挙で **destroy 経路は slot を null 化しない**ことを実証。placeholder が rebuild 置換 → retire → EBR destroy された後、slot は dangling pointer を保持し、MEM_SNAP（Timer.cpp:1078、`#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` 内）が観測専用契約を超えて dereference（`collectTrackedMemoryStatistics()` は `ASSERT_NON_RT_THREAD()` 通過の上メンバ静的読み取り）。**実害は未観測・production は flag OFF で compile-out（CMakeLists.txt:129）・diagnostic build での timing 依存 flaky AV potential**。D169-2-5 の crash はテスト契約違反（logger race）が原因であり MEM_SNAP UAF ではない点に注意。cppcheck 指摘 0（構造的 UAF は静的解析射程外）。lifetime / ownership が未証明のため STOP 条件該当 → **修復せず D172-1 lifetime/source audit へ**。

## D171-1 後の進行判断

```text
D169-2 CLOSED → D171-1 PASS
      │
      ├─ OPEN implementation 項目: 0 件 → freeze / maintenance 側へ
      │
      ├─ BuildError Phase-2 → D171-2 preflight: 不発（STALE 実証済み・residual telemetry は DEFER 済み）
      │
      └─ MEM_SNAP hazard → D172-1 lifetime/source audit（推奨次工程・別 track）
```

- 次工程の推奨: **D172-1（MEM_SNAP dangling hazard の lifetime/source audit）**。audit 問いは (1) reader 4 箇所の dereference 実態、(2) 修復方向（world 経由読み替え / destroy 側 mirror 清掃 / slot 廃止）の authority 契約適合性、(3) diagnostic build での動的実証。
- doc-only 残課題: inventory 1-C-1/1-C-2 の STALE 化反映 + buildErrorCount_ の freeze register 補助 trigger 登録（次編集 window にて）。
