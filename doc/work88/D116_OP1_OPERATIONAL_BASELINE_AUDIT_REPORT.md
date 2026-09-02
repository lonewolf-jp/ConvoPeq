# D116-OP-1 — Normal Development / Operational Validation Baseline Audit（read-only Work Report）

```text
D116-OP-1 — Normal Development / Operational Validation Baseline Audit

Type: read-only / operational validation preparation
Date: 2026-09-01
Baseline: ConvoPeq.md Generated 2026-09-01 21:47:45（--check 実測 FRESH / NEWER_SRC_COUNT=0 / CHECK_EXIT=0）
Production source: 0 / Test source: 0 / CMake: 0 / Build: 0 / CTest: 0 / stress: 0
CR-α reopen: 0（禁止遵守 — {10,80,2} / K=3 は契約前提としてのみ参照）
D159 CLOSED 領域の再監査: 0 / Episode・RecoveryEpisodeId・E×O≤32 再導入: 0
行番号基準: 現行ソース（21:47:45 snapshot 相当）のみ — 旧 report 行番号は使用しない
```

## 総合判定

> ## **D116-OP-1 = 完了。TRIGGERED（実 failure 起票対象）は 0 件。**
> 5 系統 × 10 観測対象のうち **PROVEN 7 / OPERATIONAL-VALIDATION 4（うち重複あり実質 3 工程）/
> DEFERRED 2 / STALE 1（注記扱い）**。新規実装・新規 CR は不要。
> **唯一の重要な証拠空白 = D117/D118 漏出修正（HW-1 retirePublishedDSP 配線）後の長時間運用検証が
> 未実施**（D116-6 の HOLD は修正前の 2026-08-29 実施）→ 最優先の OPERATIONAL-VALIDATION 工程
> （D116-OP-2 候補: 実機 soak + failure injection + restart cycle を現行 HEAD で再実施）。
> 本 audit は「何を直すか」を確定しない — 「何を測ればよいか」のみを確定する。

---

## Classification legend

```text
PROVEN                 = 既存 test / structural audit で十分証明済み
OPERATIONAL-VALIDATION = コード変更なしで実機・実運用観測する価値がある
TRIGGERED              = 実際の failure / requirement が発生し新規 work item が必要
DEFERRED               = D159 freeze register の trigger 未発生
STALE                  = 旧資料の記述で現行 source に対して authority を持たない
```

## 系統 1 — 通常 rebuild → publish → retire lifecycle

| 対象 | 現行 source anchor | 既存証明 | 実運用で不足する証拠 | 観測方法 | 期待される正常状態 | failure trigger | Disposition |
|---|---|---|---|---|---|---|---|
| O-1 rebuild→publish 鎖 | `RebuildDispatch.cpp:824` rebuildThreadLoop（prepare :1186 / IR :1239 / warmup :1251）→ commit → publishAndSwap（`RuntimeWorldAuthority.h` X4-B-4 publish/commit 境界 :251+） | CTest 40/40（#36 HeadlessAudioPathVerification・#34 PartialPublicationReject・#20 CoordinatorRejects）+ D116-3 burst ×12 PASS（0異常） | 修正後 HEAD での連続 publication 下のメモリ・レイテンシ傾向（D116-3 は修正前） | `--cli-intent-burst-count/interval-ms` + `BUILD_PHASE` log（:1375-1386 memBuild/memIR） | publication 反復で 0 異常・e2e 安定・メモリ plateau | Event 1000 / 0xC0000005 / 異常テレメトリ | **PROVEN**（機能）+ **OPERATIONAL-VALIDATION**（長時間傾向） |
| O-2 retire→reclaim | `AudioEngine.Retire.cpp:65` requestReclaim(minReaderEpoch)・:68 保留 reclaim 再試行・INV-EPOCH-1/2（grace 内 reclaim 禁止） | CTest #24 RetireGraceSemantics・#27 NormalRetireDSPHandleCompare・#10 DeferredDeletionQueueReclaim + Gate E/F lifecycle audit（ownership conservation liveCount ±1 単一 site） | 実 RT reader が audio callback 中に居る実機条件下での reclaim タイミング | soak 中の retire/reclaim diagLog + minReaderEpoch 挙動 | grace 満了後に reclaim 完了・UAF なし・liveCount 収支一致 | reclaim 滞留 / UAF / liveCount 不整合 | **PROVEN**（構造）+ **OPERATIONAL-VALIDATION**（実 RT 交錯） |
| O-3 漏出修正の実運用確認（最重要） | `AudioEngine.Timer.cpp:1897-1975` retirePublishedDSP（HW-1: publication epoch 伝搬 retire パス）・:954 endCrossfade 順序修正 | Gate E lifecycle/ownership audit + F6 retireDSPHandleForRuntime 閉域（cpp:940-958）+ CTest 40/40 | **修正後の長時間メモリ検証が未実施** — D116-6 HOLD（612MB→9.4GB 線形増加）は修正前の 2026-08-29 実測 | `D116_memory_sampler_poll.ps1`（CSV パス必ず明示指定）+ `--cli-run` 長時間 + intent burst | メモリが publication 反復でも plateau（線形増加なし） | publish 反復あたりの線形メモリ増加（漏出再燃） | **OPERATIONAL-VALIDATION（最優先）** |

## 系統 2 — Recovery / deferred recovery の実運用挙動

| 対象 | 現行 source anchor | 既存証明 | 実運用で不足する証拠 | 観測方法 | 期待される正常状態 | failure trigger | Disposition |
|---|---|---|---|---|---|---|---|
| O-4 recovery admission / coalesce / resolution | coordinator `ISRRuntimePublicationCoordinator.h:560` submitRecoveryRequest・coalesce（同一 slots_[i].state CAS・G-4.3-R 監査 PASS）・terminal resolve | CTest #21 ISRSemanticValidationRejects（T1/T2/T7/T8/T10: coalesce ΔL=0・identity {handle,target}・terminal→NEW） | 実 failure（壊れた IR 連続投入等）で recovery 鎖が実機で完走するか | `D116_ir_swapper.py` で異常 IR を注入 → recovery diagLog（coalesced / capacity / shutdown discard counters） | recovery が bounded queue で成立・coalesce ΔL=0・duplicate delivery なし | recovery 無限 loop / obligation 滞留 | **PROVEN**（契約・test）+ **OPERATIONAL-VALIDATION**（failure injection 実機） |
| O-5 deferred redrive / wake / P3 repair | `RuntimePublicationOrchestrator.cpp:700` processDeferredAdmission(wasRecoveryWake)・`AudioEngine.Timer.cpp:1662/1829/1849` requestDeferredClear・budget kMax=2（dormant）・F6 drain（`testDeferredBacklogDrainsCompletely`） | D135-8/9 Gate A〜G + F6 12/12 + CTest 40/40 ×2（2026-09-01 実測） | 実機・実デバイスでの deferred redrive（markTransientFailure → delivery=None → redrive）の通過実績 | soak 中の recovery/deferred diagLog + recoveryRetryDeferredCount_ 等の counters | backlog 完全 drain・同一 obligation 重複消費なし・exhaustion 時 terminal log 1 回 | backlog 残存 / 重複消費 / 無限 redrive | **PROVEN**（構造）+ **OPERATIONAL-VALIDATION**（実機通過） |
| O-6 MPSC 化 / supersession | D159 freeze register D1（第 2 producer 出現）/ D2（Supersession 要件化） | — | —（trigger 非発生・Timer/Processor からの recovery API 呼び出し 0 件） | —（監視のみ） | — | 第 2 producer 出現 / supersession 実要件 | **DEFERRED** |

## 系統 3 — Site 3 warmup retry（CR-α 契約を前提条件として扱う・再監査禁止）

| 対象 | 現行 source anchor | 既存証明 | 実運用で不足する証拠 | 観測方法 | 期待される正常状態 | failure trigger | Disposition |
|---|---|---|---|---|---|---|---|
| O-7 CR-α 契約（観測点のみ） | RebuildDispatch.cpp:1272-1318（CR-α 契約実装・CR-α-5 12/12 audit PASS） | CR-α-4: CTest 40/40 ×2 + TestF checks=86 ×2 / CR-α-5: 構造証明 12/12 — **再実装・再監査しない** | 実 warmup failure 発生時の観測可能性の確認（発生していない） | soak log 中の "warmup retry scheduled generation=… attempt=… limit=3 delayMs=… error=…" / "warmup retry exhausted … attempts=3" | 正常運用では warmup retry log 0 行。注入 failure 時は 3 retry 後 exhausted log **1 回** | retry 無限化 / exhausted 重複 log（契約逸脱の兆候 → TRIGGERED） | **PROVEN**（契約前提・閉鎖）+ 観測付帯（soak 内・変更なし） |

## 系統 4 — RuntimeWorld / ISR read-side（CW-8 再監査禁止）

| 対象 | 現行 source anchor | 既存証明 | 実運用で不足する証拠 | 観測方法 | 期待される正常状態 | failure trigger | Disposition |
|---|---|---|---|---|---|---|---|
| O-8 read-side 単一 source | `core/RuntimeStore.h:77-82` observe（1 回 acquire load）・`RuntimeWorldAuthority.h:219-249` observePublishedWorld / PublishedWorldObservation factory・INV-X4-6/7/A | CTest #33 RuntimeWorldAuthorityProjectionContract + T-CW8-1〜7（ND-04 検証）+ CR-β-0 ALREADY COVERED 確定 | 実 RT audio callback と publish/retire の交錯下での reader 観測（構造 test は決定論的単独実行） | soak（通常運用が RT reader を常時供給）中の crash 無・テレメトリ異常無の negative 観測 | stale pair / stale identity read なし・crash なし | 0xC0000005 / 不整合 identity テレメトリ | **PROVEN**（構造）+ **OPERATIONAL-VALIDATION**（RT 交錯・soak 付帯） |
| 注記: CW-8 未実装の旧記述 | PLAN_DOCS_UNIMPLEMENTED_INVENTORY_20260901.md（16:20 版） | — | — | — | — | — | **STALE**（ND-01〜04 実装前に作成 — authority なし・CR-β-0 判定を優先） |

## 系統 5 — 既存 diagnostics / telemetry（failure chain 追跡可能性）

| 対象 | 現行 source anchor | 既存証明 | 実運用で不足する証拠 | 観測方法 | 期待される正常状態 | failure trigger | Disposition |
|---|---|---|---|---|---|---|---|
| O-9 log chain 追跡性 | `RebuildDispatch.cpp:20` diagLog（DBG + juce::Logger → ODS/DbgView64）・:1198-1204 / :1255-1261 build・warmup failure 記録・:1279-1314 retry/exhausted・:1375-1386 BUILD_PHASE MEM・`TelemetryRecorder.h:228`・`RetryScheduler.h:46/58` rejectCount_/pendingCount() | CR-α-5 V-α5-09（項目網羅）・ND-04/D116 で DbgView64 捕捉実績 | **実 failure 1 件を log だけ端末から追跡できることの実証**（構造は揃っているが実 failure 鎖での通し検証が未実施） | soak 中に failure injection → 単一 log file で build failure → classify → retry #1..#3 → exhausted → recovery → publish まで reconstruct | 1 log file 内で failure chain を時系列 reconstruct 可能 | chain の断絶（log 欠落で原因追跡不能になった時点で観測性 TRIGGERED 判定） | **PROVEN**（構造）+ **OPERATIONAL-VALIDATION**（実鎖通し） |
| O-10 buildErrorCount_ 等の counter | src 0 件（RuntimeBuilder.h:124 コメントのみ） | CR-β-0: DEFER / NO TRIGGER 確定（機能欠陥ではない・CR-α retry contract と分離） | 長時間統計の実需要 | —（導入時は新規 CR） | — | subsystem 別 retry 判定の設計確定 / 運用統計の実需要 | **DEFERRED**（D159 補助 trigger 登録推奨） |

## Disposition 集計と分岐

```text
PROVEN（既存証明で充足・変更なし）        : O-1機能/O-2構造/O-7契約/O-8構造/O-9構造
OPERATIONAL-VALIDATION（測定工程が必要）   : O-3（最優先: 修正後長時間メモリ）/ O-4（failure injection 実機）
                                           / O-5（実機 deferred 通過）/ O-8・O-9（soak 付帯観測）
TRIGGERED（実 failure → 新規 CR）          : 0 件（現行 HEAD で実 failure は未発生 — 起票不要）
DEFERRED（D159 freeze 維持）               : O-6（D1 MPSC / D2 supersession）・O-10（telemetry counter）
STALE（authority なし）                    : CW-8「未実装」の旧棚卸し記述（CR-β-0 判定を優先）

分岐 = 「OPERATIONAL-VALIDATION あり」→ 実機・長時間・diagnostic validation（D116-OP-2 候補）へ。
        新規実装・新規 CR は起票しない。
```

## 実運用観測に使用する既存資産（コード変更不要）

```text
CLI 自動化: --cli-run / --cli-log-file <path> / --cli-exit-ms / --cli-ir <wav> /
  --cli-ir-reload-count|interval-ms / --cli-intent-burst-count|interval-ms /
  --cli-device-type|sample-rate-hz|buffer-samples（起動は cmd 直起動 — PowerShell
  Start-Process は引数が渡らない実績あり）
メモリ: evidence/D116_memory_sampler_poll.ps1（CSV パス第 1 引数で明示指定 — 上書き教訓）
IR 注入: evidence/D116_ir_swapper.py + D116_ir{A,B,C}.wav（異 fingerprint 切替・
  同一 IR reload は fingerprint 一致で intent 不発＝設計どおり、burst で強制）
log 捕捉: --cli-log-file + DbgView64（JUCE logger → ODS フォールバック）
注意: CLI telemetry モードでは convolverParamsChanged 通知抑制（reload 単独で rebuild 不発）
```

## 禁止事項遵守（実測）

```text
Production source 変更: 0 / Test source 変更: 0 / CMake 変更: 0
Build: 0 / CTest: 0 / stress: 0
CR-α reopen: 0（{10,80,2} / K=3 は前提として引用のみ・再監査なし）
D159 CLOSED 領域の再監査: 0（T3c lifecycle / RecoveryLifecycleWord / 16B CAS / DSPHandle /
  RT affinity / A2 Permit・Proof / isFullyDrained / Phase-I coalesce に触れない）
Episode / RecoveryEpisodeId / E×O≤32: 0（Phase-II 凍結維持・本報告でも未導入）
本報告の artifacts: 本ファイルのみ
```

## Next

```text
D116-OP-1 完了 → 分岐「OPERATIONAL-VALIDATION」:
  次工程候補 = D116-OP-2（実機 operational validation 再実施）
  内容: 現行 HEAD で ①長時間 soak（メモリ plateau 確認 — O-3 最優先）
        ②failure injection（異常 IR → recovery/deferred/Site 3 観測 — O-4/O-5/O-7/O-9）
        ③restart cycle（O-2/O-8 付帯）
  実測で failure が観測された場合のみ TRIGGERED work item を起票する
  （本 audit は測定項目の確定まで — 修正対象の確定は行わない）
```
