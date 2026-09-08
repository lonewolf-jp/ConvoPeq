# D178 — CR-α Selection & Scope Audit（Work Report）

- 日付: 2026-09-08
- 性質: read-only（production / test / CMake / build 変更 0 件）
- 前段: D176 invariant audit PASS / D177 health-overflow-fault audit PASS
- evidence: `evidence/D178/D178_CRALPHA_SCOPE_AUDIT.md`

---

## 総合判定

> ## **D178 = REJECT — CR-α は既に実装済み・closure 完了（2026-09-01, commit 0aeb22ca）。実装タスクなし**
>
> 指示の第 1 候補「CR-α（BuildError retry backoff / retry count / retry telemetry）」は、
> **ND-06（CONDITIONAL-GO）→ ND-07（GO 契約）→ CR-α-1..6（実装 → 検証 → build/CTest 40/40×2 →
> retry audit 12/12 → closure CLOSED/ACCEPTED）** として既に完了している。
> 「scheduler 呼び出しは 0ms 固定」という指示前提は 2026-09-01 時点の stale inventory 記述であり、
> D175-0 判定更新（2026-09-08）で既に STALE 化済みだった。
> 現行 HEAD での intact 再検証を実施し、**K=3 bound + backoff {10,80,2} + delay 接続 + diagLog telemetry
> の全てが実装済み・closure 後に改変なし**を確認した。genuine OPEN implementation item = 0 件（D174 再確定）。

## 背景（指示 → 判定の乖離の所在）

指示では「CR-α が未実装の第 1 候補・RetryScheduler 呼び出しは 0ms 固定」と想定されていた。実測では:

1. `doc/work88/ND06_BUILDERROR_RETRY_AUDIT_REPORT.md` — CR-α 実装前 audit（CONDITIONAL-GO・Site 3 のみ乖離）
2. `doc/work88/ND07_CRALPHA_RETRY_CONTRACT_REPORT.md` — 実装判断余地ゼロの契約確定（GO・§1-14）
3. `doc/work88/CRALPHA1..6_*.md` — 実装・検証・build（FAIL→3R 修復）・CTest 40/40×2・retry audit 12/12・**closure**
4. `doc/work88/PLAN_DOCS_UNIMPLEMENTED_INVENTORY_20260901.md` — D175-0 判定更新で「CR-α 本体 = 実装済み CLOSED」明記

つまり D178 の本来の問い（CR-α の scope/invariant/boundary）は ND-06/ND-07 + CR-α-2/-5 で既に確定済み。
本監査は、(a) closure 後の改変有無（intact 性）、(b) 指示の 7 項目・invariant 連鎖を現行 HEAD で再実証、(c) 残存 DEFER の棚卸し、を行った。

## 検証結果（現行 HEAD 実測 — 詳細は evidence 参照）

### D178-1 scope 確定（7 項目）

1. **BuildError 発生地点**: RuntimeBuilder.cpp の 4 種のみ（InvalidInput :417 / ResourceUnavailable :443 / InternalError :448 / WarmupFailed :458）。MKL/Convolver/Prepare は生成経路なし（既知 V-5）
2. **classifyBuildError caller**: production 2 箇所（build failure :1214 = ログのみ / warmup :1290 = 唯一の機能使用）。Site 1/2 は classify 不使用
3. **RetryDisposition 実適用**: 実装済み。`warmupRetryDecision()` 純関数 + `retryBackoffDelayMs()` saturation（BuildErrorPolicy.h）
4. **schedule caller**: production 1 箇所のみ = `RebuildDispatch.cpp:1311 schedule(req, milliseconds(decision.delayMs))` — **0ms 固定ではない**（backoff 表 10/20/40ms・WarmupFailed の RetryImmediate=0 は正当意味論）
5. **retry counter 全探索**: K=3（Site 3 専用・BuildErrorPolicy.h:92）/ K=4×2（Site 1/2・別ドメイン）/ kMaxDeferredRetries=2（D135 dormant）— 3 ドメイン分離確立
6. **telemetry**: diagLog（attempt/limit/delayMs/error）実装済み。enum 追加は F-1 deviation で diagLog-only に確定（ND-07 §9 本質充足）
7. **Site 3 vs Recovery 境界**: 混線構造なし。K=3/K=4 の意図的別値で読み違いを検出可能にする契約がコード＋コメントに実装済み

### D178-2 invariant impact — 7 項目全 PASS

| invariant | 判定 |
|---|---|
| RT blocking なし | PASS（RT 系ファイルの classify/RetryDisposition/retryScheduler_ 参照 0 件 per-file 実測） |
| retry 無限化なし | PASS（K=3 hard bound・exhausted 後 schedule 経路 0） |
| capacity 8 越え ownership 生成なし | PASS（kCapacity=8 + reject=attempt 消費 drop・T6 test 実在） |
| Recovery obligation と混線なし | PASS（counter は loop スコープ + generation rebind） |
| Publish authority 増加なし | PASS |
| Retire authority 到達なし | PASS（forbidden footprint 0） |
| HealthMonitor を decision authority にしない | PASS（RuntimeHealthMonitor に retry/schedule/decision 0 件・D177 契約維持） |

closure 後の CR-α 4 ファイル差分（0aeb22ca..HEAD）: BuildErrorPolicy.h=0 / tests=0 / RetrySchedulerTypes.h=+4（D167-5 AdmissionClosed・CR-α 意味論不変）/ RebuildDispatch.cpp=+17（D167-5 telemetry・warmup retry 領域無変更）→ **intact**。

### D178-3 実装境界 — closure 固定版を確認・変更なし

forbidden changes は現行 HEAD でも全て非該当。残存 DEFER（実装禁止継続・trigger 待ち）:
- Site 2 retry 適用（非 defect・dash2 §1.8 Phase D 仕様どおり）
- buildErrorCount_ telemetry（D175-1 補助 trigger 登録済み・monitoring）
- MKL/Convolver/PrepareFailure 生成経路（V-5・将来拡張）

### D178-4 test contract — T-α-1..7 全て実在テストで充足

T-CRα-1..4（純関数）+ RetrySchedulerTests T1-T8（capacity/shutdown/concurrent）+ CTest 40/40×2（CR-α-4・checks=86 fails=0）+ RT path 排除の静的実測。**追加テスト不要**。

## 判断

```text
D178 REJECT（CR-α 再実装要求のため）
   ↓ 実装タスクなし → D179/D180（implementation/targeted verification）も不要
   ↓
D159 通常開発サイクルへ復帰
   ↓
Phase-II は D159 freeze register 待ち（D174 Final Decision 維持）
新規 track は RuntimeBuilder.h:118-124 trigger 成立まで起票禁止
```

CR-β（第 2 候補）への切り替えも不要（ALREADY COVERED / production caller 0 は保守的休止・D163/D174 確定）。

## 規約

- CR-α = CLOSED/ACCEPTED のため再オープン禁止（CR-α-6 §8）。backoff 値変更等は別 work item 起票。
- 本報告は指示された D178-1〜D178-4 の全項目を実施完了（ただし判定は「実装前 audit」ではなく「実装済み確認・stale 前提訂正」）。

## 成果物

- `evidence/D178/D178_CRALPHA_SCOPE_AUDIT.md`（7 項目実測・invariant 証明・intact 差分・test 対応表）
- `doc/work88/D178_CRALPHA_SCOPE_AUDIT_REPORT.md`（本報告）

## 遷移

```text
D176 PASS → D177 PASS → D178 REJECT（CR-α 実装済み closure 確認）
   ↓
D159 通常開発サイクル復帰（Phase-II trigger 待ちがデフォルト状態）
```
