# D174 — OPEN Candidate Triage / BuildError Phase-II Preflight Report

- Date: 2026-09-08
- Task: D172 CLOSED 後の OPEN 候補 read-only triage（BuildError Phase-II / Site 2 / buildErrorCount_ / CW-8）
- Type: read-only（production source 0 / test source 0 / CMake 0 / build 0 / CTest 0）
- Evidence: `evidence/D174/D174_OPEN_CANDIDATE_TRIAGE.md`

## 判定

> ## **D174 PASS — Final Decision: NO IMPLEMENTATION。genuine OPEN implementation item = 0 件。D175 implementation contract は起票不要。**

```text
                 D174（triage 完了）
                   │
       ┌───────────┼───────────────┐
       ▼           ▼               ▼
  CW-8: STALE   CR-α: STALE    buildErrorCount_: DEFER    Site 2 retry: DEFER
  (ALREADY      (Site 3 wired  (trigger なし・            (非 defect・
   COVERED)      実測)          monitoring 維持)           将来拡張コメント現役)
       └───────────┴───────────────┴────────────────────────────┘
                                  ▼
                    Implementation contract audit: 起票不要
                    残作業 = doc-only maintenance のみ
```

## D174-0 Source Authority — PASS

`ConvoPeq.md` Generated 2026-09-08 07:15:27（NEWER_SRC_COUNT=0 FRESH・D172-3 反映済み・実装 commit 54ba7b40 基準）。

## D174-1 BuildError Phase-II — STALE（CR-α 接続完了）/ DEFER（残部）

**actual call-chain 実測**（policy → scheduler → retry call-site の接続追跡）:

```text
classifyBuildError: 2 箇所のみ
  :1214 Site 2（build failure）→ classify + diagLog → continue（retry 適用なし＝dash2 §1.8 Phase D 仕様通り）
  :1290 Site 3（warmup failure）→ warmupRetryDecision（純関数）
                                     ↓ Schedule
schedule(req, decision.delayMs): :1311 — production call site 1 箇所のみ
                                     ↓ RetryScheduler 専用 worker thread
                                   DispatchFn → submitRebuildIntent（NonRT enqueue）
```

- **A**: backoff は Site 3 に接続済み（RetryBackoff → 10/20/40/80/80 saturation・T-CRα-1 テスト）。Site 2 未接続は仕様通り（defect ではない）
- **B**: retry count は 3 層分離 — Site 3 `warmupRetryCount`（max=3・generation rebind）/ Recovery `recoveryConsecutiveFailures`（K=4）/ obligation counter（`postRecoveryFailureSignal`・**D105-R18 Dual-LP 独立線形化**）
- **C**: buildErrorCount_ は「subsystem 別 retry 判定が必要になる設計確定時」の最小 wiring という仕様（RuntimeBuilder.h:118-124）
- **D**: 既存 telemetry（REBUILD_TELEMETRY + classify ログ + Exhausted terminal）で観測十分
- **E**: Site 分離は明文（BuildErrorPolicy.h:87「=3 は Site 3 専用」/ RebuildDispatch.cpp:849「K=4 とは別ドメイン」）— schedule 1 箇所のみで混線構造なし
- **F**: RetryImmediate は意図された zero-delay bounded retry（Exhausted 判定が delay 計算に先行 — CRALPHA5 V-α5-06 PASS）
- **G**: inventory「未実装」記述は **stale**（CR-α CLOSED・0aeb22ca）

## D174-2 Site 2 Retry — DEFER（非 defect）

| 軸 | 結果 |
| --- | --- |
| kMax=3 誤用 | 構造的に不可能（Site 2 は schedule 呼び出しなし） |
| Recovery K=4 混同 | なし（Site 2 は counter 保持せず） |
| RT path 侵入 | なし（schedule 元 = RebuildThread・scheduler は専用 worker・dispatch は submitRebuildIntent） |
| unbounded | なし（全 bounded: max=3 / K=4 / Site 2 単発） |
| authority 分散 | なし（schedule production call site 1 箇所） |
| obligation への影響 | なし（Recovery は D105-R18 独立線形化・Site 2/3 は obligation 不接触） |

Site 2 の retry 適用は欠損した安全機構ではない（classify+log で十分）。将来実装する場合も既存 authority（BuildErrorPolicy 純関数 + RetryScheduler）の再利用が前提 — 二重化禁止。

## D174-3 buildErrorCount_ — DEFER / monitoring（trigger なし）

RuntimeBuilder.h:118-124 の trigger 条件（「convolver/prepare の実 failure が観測可能になり subsystem 別 retry 判定が必要になる設計確定時」）は未発生（MKLFailure / ConvolverFailure / PrepareFailure は生成経路なし・休眠分類）。現行 observability で実装を開始すべき具体的事象は未観測 → **freeze register / monitoring item として維持**（D163・CRBETA0 B-7 二重記録どおり）。

## D174-4 CW-8 — STALE / ALREADY COVERED（実装禁止）

型（private ctor + friend 構造遮断）+ factory `observePublishedObservation()`（単一 acquire load から {world, &world->publication} 同時確定）+ T-CW8-1..7 テスト + harness 登録が fresh snapshot に実在（19 hits）。inventory「src 0 件」は stale。production caller 0 = 保守的休止（D163/CRBETA0 同結論）。**D174 候補から除外。**

## D174-5 Inventory relevance

| 候補 | 判定 |
| --- | --- |
| CW-8 | STALE / ALREADY COVERED |
| CR-α backoff | STALE（Site 3 wired） |
| buildErrorCount_ | DEFER / monitoring |
| Site 2 retry 適用 | DEFER（将来拡張） |
| D159 D1-D6 | DEFER 維持 |

D172 起因の追加 stale 項目 = 0 件（D173-3 再確認）。

## Final Decision

**NO IMPLEMENTATION — D175 implementation contract 起票不要。**

次の着手可能作業は doc-only maintenance のみ:
1. inventory 1-C-1/1-C-2 の STALE 化反映（CR-α CLOSED / CR-β ALREADY COVERED）
2. buildErrorCount_ の freeze register 補助 trigger 登録（CRBETA0「次編集 window」約束分）
3. h:2265 stale コメント修正（「通常動作では null」の W2 発動時挙動に関する不正確さ）

実装系の新規 track（BuildError Phase-II / CW-8 / Site 2 retry）は、RuntimeBuilder.h:118-124 の trigger 条件が成立する設計確定イベントが発生するまで起票禁止。
