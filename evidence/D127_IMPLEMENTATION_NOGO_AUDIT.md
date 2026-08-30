# D127 — Unified Lifecycle Repair Implementation → **NO-GO（Gate 4 不達・実装取消・診断収獲大）**

**Date:** 2026-08-29
**Frozen HEAD:** `a65ace1` + 観測専用トレース群（D117/D125/D126/D127 — 全て macro-gated または log-timing のみ・production semantics 変更 0 に復帰済み）
**実装:** M1（activate 公開）+ M3（stale task 再build ガード）+ M5（Observe 撤去）+ M6（shutdown 診断）。**M2（crossfade completion）はユーザー指示により意図的に範囲外。**
**結果:** Gate 1-3 PASS、**Gate 4 NO-GO → 実装全取消** → CTest 40/40 で安定状態復帰。

---

## Gate 結果

| Gate | 結果 | 詳細 |
| --- | --- | --- |
| Gate 1 Compile（DIAG ON/OFF） | **PASS**（初回 C2065 スコープバグ修正後・error 0） |
| Gate 2 CTest | **PASS** — test21 ×2 PASS、full **40/40**（diag） |
| Gate 3 3-burst | **PASS（M3 効果実証）** — **TASK_WAKE 15,373 回のうち 15,371 回が wokeByRetryReady=1**（1ms tick の deferred publish 再通知 ≈440Hz）。M3 前は全 wakeup が stale task 再 build だった（D123 storm の正体）。M3 後: pendingTask wake 2 → build 2 → **DC live 3 で bounded**、CONV_REBUILD/REQUESTED = 1:1 |
| Gate 4 6-publish | **NO-GO** — CONV_REBUILD≈REQUESTED ✓ だが **RETIRE=0 / DESTROY=0 / DC live が publish 数に比例増加** |
| Gate 5-6 | 実施するも同傾向（RETIRE 0） |

規約「1 failure でも D127 修正を止める」→ **即時中止・production 変更全取消**（M1/M3/M5 を git checkout）。診断専用変更（D117/D125/D126/D127 trace、shutdown marker、M6 logger-timing）は保持。CTest 40/40 復帰確認。

## Gate 4 NO-GO の診断的解明（D127-NOGO probe・最大の収穫）

`[D127_TAIL]` / `[D127_TRANS]` トレース（macro-gated）により、retire 不通の構造を確定:

### 発見 1 — tail（onPublishCompleted）は 6 intent 中 **2 回しか実行されない**

```text
[D127_TAIL] seq=6 oldHandleNull=0 oldResolvedValid=1 needsCrossfade=1
[D127_TRANS] new=...4294F080 old=...4948A080 oldHandleNull=0 needsCrossfade=1
[D127_TAIL] seq=7 oldHandleNull=1 oldResolvedValid=0 needsCrossfade=0
[D127_TRANS] new=...4294F080 old=null oldHandleNull=1 needsCrossfade=0
```

6 burst intent のうち tail に到達したのは 2 つのみ。**Replaceable/Phase5-KEEP による intent 統合で、置き換えられた intent の DSP transition は永遠に実行されない**（統合された intent の build は実行されるが DSP transition は最後の 1 つにしか配されない）。

### 発見 2 — crossfade 分岐は到達するが completion がないため claim DSP が滞留

seq=6: needsCrossfade=1 → claim → storeReceipt → crossfadeRuntime_.start → **M2 未実装のため完了せず** → claim した旧 DSPCore（bootstrap 側）が滞留（D121-C の未完成状態機械の実証）。

### 発見 3 — seq=7 で oldHandle が再び null になる

seq=6 の `activate(newHandle)` で `activeRuntimeDSPHandle_` が公開されたはずだが、seq=7 の trySubmit では null。**activate が書いた値が消える（または読まれない）経路が存在** — 候補: (a) endCrossfade/他 writer による上書きタイミング、(b) sr_bs intent の decision が activate 前に作成、(c) getActiveRuntimeDSPHandle の読み先と書き先の不一致。**D128 で特定必須**。

### 発見 4 — M3 は storm 防止として完全に機能

15,371 回の RetryReady wake が発生しても stale build は 0 回（CONV_REBUILD = REBUILD_REQUESTED = 1:1）。D123 の 10×/73× 再実行は完全遮断。

## 現状の漏出構造（D126-A 経路表の更新）

| 経路 | 状態 |
| --- | --- |
| 経路 1: published DSPCore の retire 到達不能 | **未修復** — M1 だけでは不十分。(i) tail 実行自体が intent 統合で間引かれ、(ii) crossfade 分岐に入ると completion 不在で滞留、(iii) seq=7 型の oldHandle 消失経路が存在 |
| 経路 2: crossfade 分岐到達時の obsolete DSPCore 滞留 | D123 実測どおり（M2 未実装が前提条件） |
| storm（同一世代再実行） | **M3 により解消済み**（D126-A M3 の正当性実証） |

## 次段階（D128）への引き継ぎ事項

1. **tail 実行間引けの解明**: intent 統合（Replaceable/Phase5-KEEP）時に、置き換えられた intent の build DSPCore が DSPGuard 破棄される一方、**publish された世界列の DSP transition が最後以外実行されない**設計の是正（全 commit に対して transition を実行するか、統合時に旧 DSP を直接 retire するか）
2. **activeRuntimeDSPHandle_ 消失経路の特定**（発見 3）
3. **M2（crossfade completion 駆動）の実装** — seq=6 型滞留の解消に必須（D126-A で設計済み: 方式 A）
4. shutdown 間欠 crash（D119-BLOCKER-2）: 本監査の logger 保持により次回から crash phase が特定可能

## 生成物

- `evidence/D127_diag_build.log` / `D127_prod_build.log` / `D127_ctest_diag.log` / `D127_ctest_reverted.log`
- `evidence/D127_g3_3burst.log`（M3 効果: 15,371 RetryReady wake → stale build 0）
- `evidence/D127_g4_6pub.log` / `D127_g4_10pub.log`（baseline 漏出継続の証跡）
- `evidence/D127_nogo_probe.log`（D127_TAIL/TRANS 診断 — tail 間引き + oldHandle 消失の実測）
- `evidence/D127_revert_*.log`（復帰確認・CTest 40/40）
- 本ファイル
